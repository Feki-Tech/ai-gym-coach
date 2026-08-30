"""coach_knowledge.py — retrieval (RAG) for the LLM coach.

The coach used to know only what fit in its system prompt: a dozen fault
notes and the nine trackable exercises. This module gives it a local,
searchable knowledge base and feeds the relevant part into each reply:

* coaching notes   — data/knowledge/*.md (form faults, programming,
                     recovery, nutrition, technique, alternatives), one
                     chunk per "##" heading
* exercise catalogue — data/exercises.json, an open exercise database
                     (name, muscles, equipment, mechanic, level,
                     instructions) fetched once with --fetch

Retrieval is dependency-free BM25 (deterministic, so the behaviour evals
stay reproducible). When an Ollama embedding model is available
(COACH_EMBED_MODEL, e.g. nomic-embed-text) the ranking becomes hybrid:
BM25 and cosine similarity fused by reciprocal rank. Everything stays on
the machine.

    python coach_knowledge.py --fetch                 # download the catalogue
    python coach_knowledge.py --search "knees cave in on squats"
    python coach_knowledge.py --exercise "romanian deadlift"
    python coach_knowledge.py --selftest
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import urllib.request
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.environ.get("COACH_KNOWLEDGE_DIR",
                               os.path.join(HERE, "data", "knowledge"))
CATALOGUE_FILE = os.environ.get("COACH_EXERCISES",
                                os.path.join(HERE, "data", "exercises.json"))
# free-exercise-db: 870+ exercises, public domain (Unlicense)
CATALOGUE_URL = ("https://raw.githubusercontent.com/yuhonas/free-exercise-db/"
                 "main/dist/exercises.json")
CATALOGUE_IMAGE_BASE = ("https://raw.githubusercontent.com/yuhonas/"
                        "free-exercise-db/main/exercises/")
EMBED_MODEL = os.environ.get("COACH_EMBED_MODEL", "")
EMBED_BASE = os.environ.get("COACH_LLM_BASE_URL", "http://localhost:11434/v1")

# the app's nine trackable exercises → catalogue names (exact, lower-case)
APP_TO_CATALOGUE = {
    "squat": ["barbell full squat", "bodyweight squat", "barbell squat",
              "goblet squat"],
    "pushup": ["pushups", "push-up", "close-hands push-up", "incline push-up"],
    "bench": ["barbell bench press - medium grip", "dumbbell bench press"],
    "deadlift": ["barbell deadlift", "romanian deadlift", "sumo deadlift"],
    "lunge": ["bodyweight walking lunge", "dumbbell lunges", "barbell lunge"],
    "shoulder_press": ["standing military press", "dumbbell shoulder press",
                       "seated dumbbell press"],
    "curl": ["barbell curl", "dumbbell bicep curl", "hammer curls"],
    "pullup": ["pullups", "chin-up", "wide-grip pull-up"],
    "plank": ["plank", "side bridge"],
}
APP_MUSCLES = {   # fallback when the catalogue is absent
    "squat": ["quadriceps", "glutes", "hamstrings"],
    "pushup": ["chest", "triceps", "shoulders"],
    "bench": ["chest", "triceps", "shoulders"],
    "deadlift": ["hamstrings", "glutes", "lower back"],
    "lunge": ["quadriceps", "glutes", "hamstrings"],
    "shoulder_press": ["shoulders", "triceps"],
    "curl": ["biceps", "forearms"],
    "pullup": ["lats", "biceps", "middle back"],
    "plank": ["abdominals", "lower back"],
}

_STOP = set("""a an the and or of to in on for with is are be this that it its
your you my i me we our do does how what why when which can should could would
at as by from into than then there their them about up down out off over under
""".split())
_WORD = re.compile(r"[a-z0-9][a-z0-9'-]*")


def tokenize(text: str) -> list[str]:
    toks = []
    for w in _WORD.findall(text.lower()):
        if w in _STOP or len(w) < 2:
            continue
        # crude stemming keeps "squats/squat", "presses/press" together
        for suf in ("ing", "es", "s", "ed"):
            if len(w) > 4 and w.endswith(suf):
                w = w[: -len(suf)]
                break
        toks.append(w)
    return toks


# ------------------------------------------------------------------ BM25
class BM25Index:
    """Okapi BM25 over pre-tokenized documents (k1=1.5, b=0.75)."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = (sum(self.len) / len(self.len)) if docs else 1.0
        df: Counter = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for tf, ln in zip(self.tf, self.len):
            s = 0.0
            for q in query:
                f = tf.get(q)
                if not f:
                    continue
                idf = self.idf.get(q, 0.0)
                s += idf * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * ln / self.avg))
            out.append(s)
        return out


# --------------------------------------------------------------- loading
def load_notes(path: str = KNOWLEDGE_DIR) -> list[dict]:
    """Every '## heading' section of every markdown file is one chunk."""
    docs = []
    if not os.path.isdir(path):
        return docs
    for fn in sorted(os.listdir(path)):
        if not fn.endswith(".md"):
            continue
        with open(os.path.join(path, fn), encoding="utf-8") as fh:
            text = fh.read()
        topic = fn[:-3].replace("_", " ")
        m = re.match(r"#\s*(.+)", text)
        if m:
            topic = m.group(1).strip()
        parts = re.split(r"^##\s+", text, flags=re.M)
        for part in parts[1:]:
            title, _, body = part.partition("\n")
            body = " ".join(body.split())
            if body:
                docs.append({"kind": "note", "id": f"{fn}:{title.strip()}",
                             "title": title.strip(), "topic": topic,
                             "text": body})
    return docs


def compact_catalogue(raw: list[dict]) -> list[dict]:
    """Keep the fields the coach uses; drop image blobs (names stay)."""
    out = []
    for e in raw:
        name = (e.get("name") or "").strip()
        if not name:
            continue
        out.append({
            "id": e.get("id") or re.sub(r"[^a-z0-9]+", "_", name.lower()),
            "name": name,
            "level": e.get("level"), "force": e.get("force"),
            "mechanic": e.get("mechanic"), "equipment": e.get("equipment"),
            "category": e.get("category"),
            "primaryMuscles": e.get("primaryMuscles") or [],
            "secondaryMuscles": e.get("secondaryMuscles") or [],
            "instructions": [" ".join(str(i).split())
                             for i in (e.get("instructions") or [])],
            "images": e.get("images") or [],
        })
    return out


def fetch_catalogue(url: str = CATALOGUE_URL, out: str = CATALOGUE_FILE,
                    timeout: float = 60.0) -> int:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        raw = json.loads(r.read().decode("utf-8"))
    rows = compact_catalogue(raw)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"source": url, "license": "Unlicense (public domain)",
                   "count": len(rows), "exercises": rows}, fh,
                  ensure_ascii=False, separators=(",", ":"))
    return len(rows)


WGER_URL = "https://wger.de/api/v2/exerciseinfo/?limit=200&format=json"


def _strip_html(text: str) -> str:
    text = re.sub(r"</(p|li|div|br)>", "\n", text or "", flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    import html as _html
    return _html.unescape(text)


def compact_wger(raw: list[dict]) -> list[dict]:
    """Normalize wger's exerciseinfo rows to the compact schema (English
    translation, CC-BY-SA per entry)."""
    out = []
    for e in raw:
        tr = next((t for t in e.get("translations") or []
                   if t.get("language") == 2), None)          # 2 = English
        if not tr or not (tr.get("name") or "").strip():
            continue
        steps = [" ".join(p.split()) for p in
                 _strip_html(tr.get("description", "")).split("\n")]
        out.append({
            "id": f"wger-{e.get('id')}", "name": tr["name"].strip(),
            "level": None, "force": None, "mechanic": None,
            "equipment": ", ".join(q["name"] for q in e.get("equipment") or [])
            or None,
            "category": (e.get("category") or {}).get("name"),
            "primaryMuscles": [m.get("name_en") or m.get("name")
                               for m in e.get("muscles") or []],
            "secondaryMuscles": [m.get("name_en") or m.get("name")
                                 for m in e.get("muscles_secondary") or []],
            "instructions": [x for x in steps if x],
            "images": [i.get("image") for i in e.get("images") or [] if i.get("image")],
            "license": (e.get("license") or {}).get("short_name"),
            "license_author": e.get("license_author"),
        })
    return out


def fetch_wger(out: str = CATALOGUE_FILE, timeout: float = 60.0) -> int:
    """Alternative source: wger.de (CC-BY-SA per exercise, no auth)."""
    rows, url = [], WGER_URL
    while url:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            d = json.loads(r.read().decode("utf-8"))
        rows += d.get("results", [])
        url = d.get("next")
    compact = compact_wger(rows)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"source": "https://wger.de/api/v2/exerciseinfo/",
                   "license": "CC-BY-SA 4.0/3.0 per entry (see license field)",
                   "count": len(compact), "exercises": compact}, fh,
                  ensure_ascii=False, separators=(",", ":"))
    return len(compact)


def load_catalogue(path: str = CATALOGUE_FILE) -> list[dict]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return []
    rows = data.get("exercises") if isinstance(data, dict) else data
    return rows if isinstance(rows, list) else []


def exercise_text(e: dict) -> str:
    """Flat text of one catalogue entry (what gets indexed and shown)."""
    bits = [e["name"]]
    for k in ("level", "force", "mechanic", "equipment", "category"):
        if e.get(k):
            bits.append(f"{k}: {e[k]}")
    if e.get("primaryMuscles"):
        bits.append("primary muscles: " + ", ".join(e["primaryMuscles"]))
    if e.get("secondaryMuscles"):
        bits.append("secondary: " + ", ".join(e["secondaryMuscles"]))
    if e.get("instructions"):
        bits.append("how: " + " ".join(e["instructions"]))
    return ". ".join(bits)


# ---------------------------------------------------------- embeddings
class Embedder:
    """Optional Ollama embeddings (nomic-embed-text, mxbai-embed-large…).
    Silent no-op when the model/env is absent so retrieval stays offline."""

    def __init__(self, model: str = EMBED_MODEL, base: str = EMBED_BASE):
        self.model = model
        self.url = base.rstrip("/").removesuffix("/v1") + "/api/embed"
        self.ok = bool(model)

    def embed(self, texts: list[str]) -> list[list[float]] | None:
        if not self.ok or not texts:
            return None
        try:
            import coach_ops
            coach_ops.local_no_proxy()
        except Exception:
            pass
        try:
            req = urllib.request.Request(
                self.url, data=json.dumps({"model": self.model,
                                           "input": texts}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())["embeddings"]
        except Exception:
            self.ok = False
            return None


def _cos(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


# -------------------------------------------------------------- the KB
class KnowledgeBase:
    def __init__(self, notes_dir: str = KNOWLEDGE_DIR,
                 catalogue: str = CATALOGUE_FILE, embedder: Embedder | None = None):
        self.docs: list[dict] = load_notes(notes_dir)
        self.exercises = load_catalogue(catalogue)
        self._by_name = {e["name"].lower(): e for e in self.exercises}
        for e in self.exercises:
            self.docs.append({"kind": "exercise", "id": e["id"],
                              "title": e["name"], "topic": "exercise catalogue",
                              "text": exercise_text(e), "entry": e})
        # titles count triple: "Protein" must beat a body that merely
        # mentions protein
        self.index = BM25Index([tokenize(d["title"]) * 3 + tokenize(d["text"])
                                for d in self.docs])
        self.embedder = embedder if embedder is not None else Embedder()
        self._vecs: list[list[float]] | None = None

    # ---- ranking
    def _vectors(self):
        if self._vecs is None and self.embedder.ok:
            vecs = []
            batch = 64
            for i in range(0, len(self.docs), batch):
                got = self.embedder.embed(
                    [d["title"] + ". " + d["text"][:1500]
                     for d in self.docs[i:i + batch]])
                if got is None:
                    vecs = None
                    break
                vecs.extend(got)
            self._vecs = vecs
        return self._vecs

    def search(self, query: str, k: int = 5, kinds: tuple[str, ...] = ("note", "exercise"),
               boost: dict[str, float] | None = None) -> list[dict]:
        """Top-k chunks for a question. `boost` adds score to docs whose
        title contains a key (e.g. the live exercise)."""
        q = tokenize(query)
        if not q or not self.docs:
            return []
        bm = self.index.scores(q)
        order_bm = sorted(range(len(bm)), key=lambda i: -bm[i])
        rank = {i: 1.0 / (60 + r) for r, i in enumerate(order_bm) if bm[i] > 0}
        vecs = self._vectors()
        if vecs:
            qv = self.embedder.embed([query])
            if qv:
                sims = [_cos(qv[0], v) for v in vecs]
                for r, i in enumerate(sorted(range(len(sims)), key=lambda i: -sims[i])):
                    rank[i] = rank.get(i, 0.0) + 1.0 / (60 + r)
        for i, d in enumerate(self.docs):
            if boost:
                for key, w in boost.items():
                    if key and key.lower() in d["title"].lower():
                        rank[i] = rank.get(i, 0.0) + w
        hits = [(s, i) for i, s in rank.items() if self.docs[i]["kind"] in kinds]
        hits.sort(key=lambda t: -t[0])
        out = []
        for s, i in hits[:k]:
            d = dict(self.docs[i])
            d["score"] = round(s, 4)
            out.append(d)
        return out

    # ---- catalogue access
    def exercise(self, name: str) -> dict | None:
        """Exact (case-insensitive) name, an app exercise name, or the best
        catalogue match for a free-text name."""
        if not name:
            return None
        key = name.strip().lower()
        if key in self._by_name:
            return self._by_name[key]
        for cand in APP_TO_CATALOGUE.get(key.replace(" ", "_"), []):
            if cand in self._by_name:
                return self._by_name[cand]
        hits = self.search(name, k=1, kinds=("exercise",))
        return hits[0]["entry"] if hits else None

    def muscles_for(self, app_exercise: str) -> list[str]:
        e = self.exercise(app_exercise)
        if e and e.get("primaryMuscles"):
            return list(e["primaryMuscles"]) + list(e.get("secondaryMuscles") or [])
        return list(APP_MUSCLES.get(app_exercise, []))

    def find_exercises(self, query: str = "", muscle: str = "",
                       equipment: str = "", limit: int = 8) -> list[dict]:
        """Filter + rank the catalogue (for the coach's lookup action and
        the MCP tool)."""
        rows = self.exercises
        if muscle:
            m = muscle.lower()
            rows = [e for e in rows if any(m in x.lower() for x in
                                           e["primaryMuscles"] + e["secondaryMuscles"])]
        if equipment:
            q = equipment.lower()
            rows = [e for e in rows if q in (e.get("equipment") or "").lower()
                    or (q in ("none", "bodyweight", "body only")
                        and (e.get("equipment") or "body only") in ("body only", None))]
        if query:
            ids = {e["id"] for e in rows}
            ranked = [h["entry"] for h in self.search(query, k=limit * 4,
                                                      kinds=("exercise",))
                      if h["entry"]["id"] in ids]
            rows = ranked
        return rows[:limit]

    # ---- prompt building
    def context_block(self, question: str, live_exercise: str | None = None,
                      k: int = 4, max_chars: int = 1800) -> str:
        """The RELEVANT KNOWLEDGE block for the system prompt: top notes for
        the question (boosted toward the exercise on screen) plus, when the
        question names one, a catalogue entry."""
        boost = {}
        if live_exercise:
            boost[live_exercise.replace("_", " ")] = 0.01
            boost[live_exercise] = 0.01
        hits = self.search(question, k=k, kinds=("note",), boost=boost)
        ex_hits = self.search(question, k=1, kinds=("exercise",))
        if ex_hits and any(t in tokenize(ex_hits[0]["title"]) for t in tokenize(question)):
            hits.append(ex_hits[0])
        lines, used = [], 0
        for h in hits:
            body = h["text"] if h["kind"] == "note" else \
                describe_exercise(h["entry"], steps=3)
            line = f"- [{h['topic']}: {h['title']}] {body}"
            if used + len(line) > max_chars:
                line = line[: max(0, max_chars - used - 1)] + "…"
            lines.append(line)
            used += len(line)
            if used >= max_chars:
                break
        return "\n".join(lines)


def describe_exercise(e: dict, steps: int = 6) -> str:
    """Compact, speakable description of a catalogue entry."""
    bits = [e["name"]]
    meta = ", ".join(x for x in (e.get("level"), e.get("mechanic"),
                                 e.get("equipment") and f"equipment: {e['equipment']}")
                     if x)
    if meta:
        bits.append(f"({meta})")
    if e.get("primaryMuscles"):
        bits.append("works " + ", ".join(e["primaryMuscles"])
                    + (" (+ " + ", ".join(e["secondaryMuscles"]) + ")"
                       if e.get("secondaryMuscles") else ""))
    if e.get("instructions"):
        bits.append("Steps: " + " ".join(
            f"{i + 1}. {s}" for i, s in enumerate(e["instructions"][:steps])))
    return " ".join(bits)


def exercise_lookup(kb: KnowledgeBase, action: dict) -> tuple[str, str]:
    """ACTION {"do": "exercise_lookup", "query": ..., "muscle": ...,
    "equipment": ...} → (spoken ack, [APP DATA] feedback for the model)."""
    query = str(action.get("query") or action.get("exercise") or "").strip()
    muscle = str(action.get("muscle") or "").strip()
    equipment = str(action.get("equipment") or "").strip()
    if not kb.exercises:
        return ("", "EXERCISE LOOKUP ERROR: the exercise catalogue is not "
                    "installed (python coach_knowledge.py --fetch).")
    exact = kb.exercise(query) if query and not (muscle or equipment) else None
    if exact and query.lower() in (exact["name"].lower(),) or \
            (exact and len(kb.find_exercises(query, limit=2)) == 1):
        return ("", "EXERCISE: " + describe_exercise(exact))
    rows = kb.find_exercises(query, muscle, equipment, limit=6)
    if not rows:
        return ("", f"EXERCISE LOOKUP: nothing matched '{query or muscle or equipment}'.")
    lines = [f"EXERCISE OPTIONS ({len(rows)}):"]
    for e in rows:
        lines.append("- " + describe_exercise(e, steps=2))
    return ("", "\n".join(lines))


def plate_calculator(target_kg: float, bar_kg: float = 20.0,
                     plates=(20.0, 15.0, 10.0, 5.0, 2.5, 1.25)) -> str:
    """Plates per side to load `target_kg` on a `bar_kg` bar (greedy)."""
    if target_kg <= 0 or bar_kg < 0:
        return "PLATE CALC ERROR: weights must be positive."
    if target_kg < bar_kg:
        return (f"PLATES: {target_kg:g} kg is less than the {bar_kg:g} kg bar "
                "— use a lighter bar or dumbbells.")
    side = (target_kg - bar_kg) / 2
    used, rest = [], side
    for p in plates:
        n = int(rest // p + 1e-9)
        if n:
            used.append((p, n))
            rest = round(rest - n * p, 3)
    loaded = bar_kg + 2 * sum(p * n for p, n in used)
    desc = " + ".join(f"{n}×{p:g}" for p, n in used) or "nothing"
    line = f"PLATES for {target_kg:g} kg on a {bar_kg:g} kg bar: per side {desc}"
    if abs(loaded - target_kg) > 1e-6:
        line += f" = {loaded:g} kg (closest with standard plates)"
    return line + "."


def handle_command(kb: "KnowledgeBase | None", text: str) -> str | None:
    """/exercise <name|muscle> and /plates <kg> [bar]. None = not ours."""
    parts = text.strip().split()
    if not parts or parts[0].lower() not in ("/exercise", "/exercises", "/plates"):
        return None
    if parts[0].lower() == "/plates":
        try:
            kg = float(parts[1])
            bar = float(parts[2]) if len(parts) > 2 else 20.0
        except (IndexError, ValueError):
            return "usage: /plates <target kg> [bar kg, default 20]"
        return plate_calculator(kg, bar)
    kb = kb or default_kb()
    if not kb.exercises:
        return "(exercise catalogue not installed — python coach_knowledge.py --fetch)"
    q = " ".join(parts[1:])
    if not q:
        return f"usage: /exercise <name or muscle> — {len(kb.exercises)} exercises"
    e = kb.exercise(q)
    hits = kb.find_exercises(q, limit=6)
    if e and (e["name"].lower() == q.lower() or len(hits) <= 1):
        return describe_exercise(e)
    return "\n".join("- " + describe_exercise(x, steps=0) for x in hits) \
        or "no match"


_KB: KnowledgeBase | None = None


def default_kb() -> KnowledgeBase:
    global _KB
    if _KB is None:
        _KB = KnowledgeBase()
    return _KB


# --------------------------------------------------------------- selftest
def _sample_catalogue() -> list[dict]:
    return compact_catalogue([
        {"id": "Barbell_Full_Squat", "name": "Barbell Full Squat",
         "level": "intermediate", "force": "push", "mechanic": "compound",
         "equipment": "barbell", "category": "strength",
         "primaryMuscles": ["quadriceps"],
         "secondaryMuscles": ["glutes", "hamstrings", "calves"],
         "instructions": ["Set the bar on a rack just below shoulder level.",
                          "Step under, brace, unrack.",
                          "Squat until the hips pass the knees.",
                          "Drive up through the mid-foot."],
         "images": ["Barbell_Full_Squat/0.jpg"]},
        {"id": "Romanian_Deadlift", "name": "Romanian Deadlift",
         "level": "intermediate", "mechanic": "compound", "equipment": "barbell",
         "category": "strength", "primaryMuscles": ["hamstrings"],
         "secondaryMuscles": ["glutes", "lower back"],
         "instructions": ["Hold the bar at hip height.", "Hinge at the hips "
                          "keeping the back flat.", "Return by driving the hips."]},
        {"id": "Pushups", "name": "Pushups", "level": "beginner",
         "mechanic": "compound", "equipment": "body only", "category": "strength",
         "primaryMuscles": ["chest"], "secondaryMuscles": ["shoulders", "triceps"],
         "instructions": ["Lie prone, hands under the shoulders.",
                          "Push up to full extension."]},
        {"id": "Glute_Bridge", "name": "Glute Bridge", "level": "beginner",
         "mechanic": "isolation", "equipment": "body only", "category": "strength",
         "primaryMuscles": ["glutes"], "secondaryMuscles": ["hamstrings"],
         "instructions": ["Lie on your back, knees bent.", "Drive the hips up."]},
        {"name": ""},                                # dropped
    ])


def selftest():
    import tempfile
    print("== coach_knowledge selftests ==")

    print("1) notes load as one chunk per heading:", end=" ")
    notes = load_notes()
    assert len(notes) >= 30, len(notes)
    titles = {n["title"] for n in notes}
    for t in ("knees_cave — knee valgus (squat, lunge)", "Progressive overload",
              "Soreness vs. pain", "Protein"):
        assert t in titles, t
    assert all(n["text"] and n["kind"] == "note" for n in notes)
    print(f"OK ({len(notes)} chunks)")

    print("2) BM25 finds the right note (deterministic, no model):", end=" ")
    with tempfile.TemporaryDirectory() as td:
        cat = os.path.join(td, "ex.json")
        with open(cat, "w", encoding="utf-8") as fh:
            json.dump({"exercises": _sample_catalogue()}, fh)
        kb = KnowledgeBase(catalogue=cat, embedder=Embedder(model=""))
        assert len(kb.exercises) == 4
        top = kb.search("my knees cave in when I squat", k=3, kinds=("note",))
        assert top and "knees_cave" in top[0]["title"], [t["title"] for t in top]
        top = kb.search("how much protein should I eat", k=2, kinds=("note",))
        assert "Protein" in [t["title"] for t in top], [t["title"] for t in top]
        top = kb.search("protein per kg bodyweight", k=1, kinds=("note",))
        assert top[0]["title"] == "Protein", top[0]["title"]
        top = kb.search("elbows flare on push ups", k=2, kinds=("note",))
        assert "elbow_flare" in top[0]["title"], top[0]["title"]
        assert kb.search("", k=3) == []
        # same query, same order — evals depend on it
        a = [h["id"] for h in kb.search("rest between sets heart rate", k=4)]
        b = [h["id"] for h in kb.search("rest between sets heart rate", k=4)]
        assert a == b
        print("OK")

        print("3) catalogue: exact, app-name and fuzzy lookup, filters:", end=" ")
        assert kb.exercise("romanian deadlift")["id"] == "Romanian_Deadlift"
        assert kb.exercise("squat")["name"] == "Barbell Full Squat"      # app name
        assert kb.exercise("pushup")["name"] == "Pushups"
        assert kb.exercise("rdl hamstring hinge")["id"] == "Romanian_Deadlift"
        assert kb.exercise("") is None
        assert kb.muscles_for("squat")[0] == "quadriceps"
        assert kb.muscles_for("plank") == APP_MUSCLES["plank"]           # fallback
        glutes = kb.find_exercises(muscle="glute")
        assert {e["id"] for e in glutes} == {"Barbell_Full_Squat", "Romanian_Deadlift",
                                             "Glute_Bridge"}
        bw = kb.find_exercises(equipment="body only")
        assert {e["id"] for e in bw} == {"Pushups", "Glute_Bridge"}
        assert kb.find_exercises("hamstrings", equipment="barbell")[0]["id"] == \
            "Romanian_Deadlift"
        print("OK")

        print("4) context block is compact and question-driven:", end=" ")
        blk = kb.context_block("why do my knees cave?", live_exercise="squat")
        assert "knees_cave" in blk and len(blk) <= 1800, blk[:200]
        assert blk.count("\n") <= 5
        blk2 = kb.context_block("how do I do a romanian deadlift", live_exercise="squat")
        assert "Romanian Deadlift" in blk2 and "Steps: 1." in blk2, blk2
        assert kb.context_block("!!!", live_exercise=None) == ""
        print("OK")

        print("5) exercise_lookup action → APP DATA the model can use:", end=" ")
        _, fb = exercise_lookup(kb, {"do": "exercise_lookup", "query": "romanian deadlift"})
        assert fb.startswith("EXERCISE: Romanian Deadlift"), fb
        _, fb = exercise_lookup(kb, {"do": "exercise_lookup", "muscle": "glutes",
                                     "equipment": "body only"})
        assert fb.startswith("EXERCISE OPTIONS (1)") and "Glute Bridge" in fb, fb
        _, fb = exercise_lookup(kb, {"do": "exercise_lookup", "query": "zzzz-nothing"})
        assert "nothing matched" in fb or "EXERCISE" in fb
        empty = KnowledgeBase(catalogue=os.path.join(td, "none.json"),
                              embedder=Embedder(model=""))
        _, fb = exercise_lookup(empty, {"do": "exercise_lookup", "query": "squat"})
        assert "not installed" in fb
        print("OK")

    print("6) compact_catalogue keeps the used fields, drops blanks:", end=" ")
    rows = _sample_catalogue()
    assert len(rows) == 4 and rows[0]["images"] == ["Barbell_Full_Squat/0.jpg"]
    assert "instructions" in rows[0] and isinstance(rows[0]["primaryMuscles"], list)
    d = describe_exercise(rows[0], steps=2)
    assert d.startswith("Barbell Full Squat (intermediate, compound, equipment: barbell)")
    assert "Steps: 1." in d and "3." not in d
    print("OK")

    print("7) plate calculator + slash commands:", end=" ")
    assert plate_calculator(100) == "PLATES for 100 kg on a 20 kg bar: per side 2×20."
    assert "1×15 + 1×2.5" in plate_calculator(55)
    assert "closest" in plate_calculator(21)          # 0.5 kg per side impossible
    assert "lighter bar" in plate_calculator(10)
    assert "ERROR" in plate_calculator(-5)
    assert handle_command(None, "/plates 60").startswith("PLATES for 60 kg")
    assert handle_command(None, "/plates 60 15") == \
        "PLATES for 60 kg on a 15 kg bar: per side 1×20 + 1×2.5."
    assert handle_command(None, "/plates x").startswith("usage")
    assert handle_command(None, "hello") is None
    with tempfile.TemporaryDirectory() as td:
        cat = os.path.join(td, "ex.json")
        with open(cat, "w", encoding="utf-8") as fh:
            json.dump({"exercises": _sample_catalogue()}, fh)
        kb = KnowledgeBase(catalogue=cat, embedder=Embedder(model=""))
        assert handle_command(kb, "/exercise romanian deadlift").startswith(
            "Romanian Deadlift")
        assert handle_command(kb, "/exercise glutes").count("- ") >= 2
    print("OK")

    print("8) wger rows normalize to the same compact schema:", end=" ")
    w = compact_wger([{"id": 7, "category": {"name": "Legs"},
                       "muscles": [{"name": "Quadriceps femoris",
                                    "name_en": "Quads"}],
                       "muscles_secondary": [], "equipment": [{"name": "Barbell"}],
                       "images": [], "license": {"short_name": "CC-BY-SA 4"},
                       "license_author": "wger.de",
                       "translations": [
                           {"language": 1, "name": "Kniebeuge", "description": "x"},
                           {"language": 2, "name": "Squats",
                            "description": "<p>Stand tall.</p><p>Sit &amp; drive.</p>"}]},
                      {"id": 8, "translations": [{"language": 1, "name": "nur DE"}]}])
    assert len(w) == 1 and w[0]["name"] == "Squats" and w[0]["id"] == "wger-7"
    assert w[0]["instructions"] == ["Stand tall.", "Sit & drive."], w[0]["instructions"]
    assert w[0]["primaryMuscles"] == ["Quads"] and w[0]["equipment"] == "Barbell"
    assert describe_exercise(w[0]).startswith("Squats (equipment: Barbell) works Quads")
    print("OK")

    print("9) embeddings are optional and never break retrieval:", end=" ")
    emb = Embedder(model="nomic-embed-text", base="http://127.0.0.1:9/v1")
    assert emb.embed(["x"]) is None and emb.ok is False
    assert Embedder(model="").embed(["x"]) is None
    print("OK")

    print("\nAll coach_knowledge selftests passed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fetch", action="store_true",
                    help="download the open exercise catalogue (free-exercise-db, "
                         f"public domain) to {CATALOGUE_FILE}")
    ap.add_argument("--fetch-wger", action="store_true",
                    help="alternative source: wger.de (CC-BY-SA per entry, "
                         "20+ languages) to the same file")
    ap.add_argument("--plates", type=float, metavar="KG",
                    help="plates per side for KG on a 20 kg bar")
    ap.add_argument("--search", metavar="QUERY")
    ap.add_argument("--exercise", metavar="NAME")
    ap.add_argument("--muscle", default="")
    ap.add_argument("--equipment", default="")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    elif args.fetch:
        n = fetch_catalogue()
        print(f"{n} exercises -> {CATALOGUE_FILE}")
    elif args.fetch_wger:
        n = fetch_wger()
        print(f"{n} exercises (wger, CC-BY-SA) -> {CATALOGUE_FILE}")
    elif args.plates:
        print(plate_calculator(args.plates))
    elif args.exercise or args.muscle or args.equipment:
        kb = default_kb()
        if args.exercise and not (args.muscle or args.equipment):
            e = kb.exercise(args.exercise)
            print(describe_exercise(e) if e else "no match")
        else:
            for e in kb.find_exercises(args.exercise or "", args.muscle,
                                       args.equipment, limit=15):
                print("-", describe_exercise(e, steps=0))
    elif args.search:
        kb = default_kb()
        print(f"[{len(kb.docs)} chunks, backend: "
              f"{'hybrid' if kb.embedder.ok else 'bm25'}]")
        for h in kb.search(args.search, k=6):
            print(f"{h['score']:.4f}  [{h['kind']}] {h['title']}\n    {h['text'][:160]}…")
    else:
        ap.print_help()
        sys.exit(1)
