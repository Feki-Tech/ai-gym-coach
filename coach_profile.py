"""Local athlete profile — the coach remembers you.

A tiny SQLite store (stdlib only) of durable facts about the athlete:
age, body stats, goals, injuries, equipment, schedule, preferences.
Facts are extracted automatically from coach conversations by the LLM,
injected into the coach's system prompt on every question, and editable
with /profile, /remember and /forget in any chat.

Privacy: everything stays in one local file (coach_profile.db,
git-ignored). Nothing is uploaded anywhere — with the default Ollama
backend even fact extraction runs on your own machine.

Run `python coach_profile.py --selftest` for the offline test suite.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from contextlib import closing
from datetime import datetime

DEFAULT_DB = os.environ.get("COACH_PROFILE_DB", "coach_profile.db")

CATEGORIES = ("identity", "body", "goals", "injuries", "equipment",
              "schedule", "nutrition", "preferences")

_EXTRACT_PROMPT = (
    "You maintain a long-term athlete profile for a gym coach. From the "
    "conversation snippet, extract durable personal facts the ATHLETE "
    "stated about themselves that are worth remembering for future "
    "sessions: age, height, weight, goals, injuries or pain, available "
    "equipment, training schedule, diet, preferences. Ignore questions, "
    "chit-chat, the coach's advice, and anything about the current set. "
    "Reply with ONLY a JSON array, no prose. Example:\n"
    '[{"category": "body", "key": "weight", "value": "82 kg"}]\n'
    f"category must be one of: {', '.join(CATEGORIES)}. "
    "key is short snake_case; value is short and keeps units. "
    "Return [] if there is nothing durable."
)


class ProfileStore:
    """Thread-safe upsert store: one row per (category, key)."""

    def __init__(self, path: str = DEFAULT_DB):
        self.path = path
        self._lock = threading.Lock()
        with closing(self._conn()) as c, c:
            c.execute("""CREATE TABLE IF NOT EXISTS facts (
                             category   TEXT NOT NULL,
                             key        TEXT NOT NULL,
                             value      TEXT NOT NULL,
                             updated_at TEXT NOT NULL,
                             PRIMARY KEY (category, key))""")

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=5)

    def remember(self, category: str, key: str, value: str) -> None:
        category = (category or "preferences").strip().lower()
        if category not in CATEGORIES:
            category = "preferences"
        key = re.sub(r"\s+", "_", key.strip().lower())[:64]
        value = str(value).strip()[:200]
        if not key or not value:
            return
        now = datetime.now().isoformat(timespec="seconds")
        with self._lock, closing(self._conn()) as c, c:
            c.execute("INSERT INTO facts VALUES (?,?,?,?) "
                      "ON CONFLICT(category, key) DO UPDATE SET "
                      "value=excluded.value, updated_at=excluded.updated_at",
                      (category, key, value, now))

    def forget(self, key: str) -> int:
        """Delete by key (any category); '*' or 'all' wipes everything."""
        with self._lock, closing(self._conn()) as c, c:
            if key.strip().lower() in ("*", "all", "everything"):
                return c.execute("DELETE FROM facts").rowcount
            key = re.sub(r"\s+", "_", key.strip().lower())
            return c.execute("DELETE FROM facts WHERE key = ?",
                             (key,)).rowcount

    def facts(self) -> list[tuple[str, str, str, str]]:
        with self._lock, closing(self._conn()) as c:
            return c.execute("SELECT category, key, value, updated_at "
                             "FROM facts ORDER BY category, key").fetchall()

    def as_prompt(self) -> str:
        rows = self.facts()
        if not rows:
            return ""
        lines = [f"- [{cat}] {key.replace('_', ' ')}: {val}"
                 for cat, key, val, _ in rows]
        return ("ABOUT THE ATHLETE (long-term profile, remembered across "
                "sessions — personalise your coaching with it):\n"
                + "\n".join(lines))

    def pretty(self) -> str:
        rows = self.facts()
        if not rows:
            return ("(profile is empty — it fills up as you chat, or use "
                    "/remember <key> <value>)")
        w = max(len(k) for _, k, _, _ in rows)
        return "\n".join(f"  [{cat:^11}] {key:<{w}}  {val}   ({upd[:10]})"
                         for cat, key, val, upd in rows)


# ---------------------------------------------------------- LLM extraction
def parse_facts(raw: str) -> list[dict]:
    """Robustly pull a JSON array of fact dicts out of model output."""
    start, end = raw.find("["), raw.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        data = json.loads(raw[start:end + 1])
    except ValueError:
        return []
    out = []
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        cat = str(item.get("category", "")).strip().lower()
        key = str(item.get("key", "")).strip()
        val = str(item.get("value", "")).strip()
        if key and val:
            out.append({"category": cat, "key": key, "value": val})
    return out


def extract_facts(client, user_text: str, reply: str = "") -> list[dict]:
    """Ask the LLM for durable facts in the latest exchange."""
    snippet = f"Athlete said: {user_text[:500]}"
    if reply:
        snippet += f"\nCoach replied: {reply[:300]}"
    raw = client.chat([{"role": "system", "content": _EXTRACT_PROMPT},
                       {"role": "user", "content": snippet}])
    return parse_facts(raw)


# ------------------------------------------------------------ chat commands
HELP = ("/profile — show what the coach remembers   "
        "/remember [category] <key> <value…> — save a fact   "
        "/forget <key|all> — erase")


def handle_command(store: ProfileStore | None, text: str) -> str | None:
    """Execute /profile, /remember, /forget. None = not a profile command."""
    parts = text.strip().split()
    if not parts or parts[0].lower() not in ("/profile", "/remember",
                                             "/forget"):
        return None
    if store is None:
        return "(profile is disabled)"
    cmd = parts[0].lower()
    if cmd == "/profile":
        return "The coach remembers:\n" + store.pretty()
    if cmd == "/remember":
        args = parts[1:]
        if len(args) >= 3 and args[0].lower() in CATEGORIES:
            store.remember(args[0], args[1], " ".join(args[2:]))
            return f"(remembered [{args[0].lower()}] {args[1]})"
        if len(args) >= 2:
            store.remember("preferences", args[0], " ".join(args[1:]))
            return f"(remembered {args[0]})"
        return "usage: /remember [category] <key> <value…>"
    n = store.forget(" ".join(parts[1:])) if len(parts) > 1 else 0
    if len(parts) < 2:
        return "usage: /forget <key|all>"
    return f"(forgot {n} fact{'s' if n != 1 else ''})"


# ----------------------------------------------------------------- selftest
def selftest():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "p.db")

        print("1) store roundtrip (remember/facts/forget/wipe):", end=" ")
        s = ProfileStore(db)
        s.remember("body", "Weight", "82 kg")
        s.remember("injuries", "left knee", "meniscus strain 2023")
        s.remember("bogus_category", "coffee", "two cups pre-workout")
        rows = s.facts()
        assert len(rows) == 3, rows
        assert ("body", "weight", "82 kg") == rows[0][:3]
        assert rows[1][0] == "injuries" and rows[1][1] == "left_knee"
        assert rows[2][0] == "preferences", rows[2]      # bogus -> fallback
        s.remember("body", "weight", "80 kg")            # upsert
        assert [r for r in s.facts() if r[1] == "weight"][0][2] == "80 kg"
        assert s.forget("coffee") == 1 and len(s.facts()) == 2
        assert s.forget("all") == 2 and s.facts() == []
        print("ok")

        print("2) as_prompt/pretty include facts:", end=" ")
        s.remember("goals", "target", "first pull-up")
        p = s.as_prompt()
        assert "ABOUT THE ATHLETE" in p and "first pull-up" in p, p
        assert "first pull-up" in s.pretty()
        empty = ProfileStore(os.path.join(tmp, "e.db"))
        assert empty.as_prompt() == "" and "empty" in empty.pretty()
        print("ok")

        print("3) parse_facts handles fences/garbage:", end=" ")
        good = parse_facts('Here you go:\n```json\n[{"category":"body",'
                           '"key":"height","value":"178 cm"},'
                           '{"key":"","value":"x"},"noise"]\n```')
        assert good == [{"category": "body", "key": "height",
                         "value": "178 cm"}], good
        assert parse_facts("no json here") == []
        assert parse_facts("[not valid") == []
        print("ok")

        print("4) extract_facts calls LLM and stores clean facts:", end=" ")

        class FakeClient:
            def __init__(self): self.msgs = None
            def chat(self, messages):
                self.msgs = messages
                return ('[{"category":"schedule","key":"days_per_week",'
                        '"value":"3"}]')
        fc = FakeClient()
        facts = extract_facts(fc, "I train 3 days a week", "Great plan!")
        assert facts == [{"category": "schedule", "key": "days_per_week",
                          "value": "3"}], facts
        assert "3 days a week" in fc.msgs[1]["content"]
        assert "JSON array" in fc.msgs[0]["content"]
        print("ok")

        print("5) chat commands (/profile /remember /forget):", end=" ")
        s2 = ProfileStore(os.path.join(tmp, "c.db"))
        assert handle_command(s2, "hello coach") is None
        assert handle_command(None, "/profile") == "(profile is disabled)"
        out = handle_command(s2, "/remember injuries shoulder impingement "
                                 "right side")
        assert "injuries" in out and s2.facts()[0][1] == "shoulder"
        assert "remembered" in handle_command(s2, "/remember mood focused")
        assert "empty" not in handle_command(s2, "/profile")
        assert "forgot 1" in handle_command(s2, "/forget mood")
        assert "usage" in handle_command(s2, "/forget")
        assert "forgot 1" in handle_command(s2, "/forget all")
        print("ok")

        print("6) concurrent writers don't corrupt the store:", end=" ")
        s3 = ProfileStore(os.path.join(tmp, "t.db"))
        threads = [threading.Thread(
            target=lambda i=i: s3.remember("preferences", f"k{i}", f"v{i}"))
            for i in range(8)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert len(s3.facts()) == 8
        print("ok")

    print("\nAll coach_profile selftests passed.")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--show", action="store_true",
                    help="print the stored profile")
    ap.add_argument("--db", default=DEFAULT_DB)
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.show:
        print(ProfileStore(a.db).pretty())
    else:
        ap.print_help()
