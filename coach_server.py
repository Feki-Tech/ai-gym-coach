"""coach_server.py — the coach as a paired API for the phone (and anything else).

The iPhone app can count reps and check form on its own, but the LLM coach
— the persona, the retrieval over the knowledge base and the exercise
catalogue, the safety guardrails, the history/profile tools, the behaviour
evals — lives in Python. This server puts that coach on the local network
so the phone (or a browser, a watch, a second PC) can talk to it: the same
`ChatCoach` the desktop uses, streamed sentence by sentence, with the
athlete's live session sent along with every message.

    python coach_server.py                 # http://0.0.0.0:7799, prints the pairing code
    python coach_server.py --port 7799 --token mysecret
    python coach_server.py --selftest

Pairing: every request needs `Authorization: Bearer <code>`; the code is
printed at start (or given with --token / COACH_PAIR_TOKEN). Type the URL
and the code into the app's Coach settings once. No TLS by itself — this is
for your own Wi‑Fi; put it behind a TLS proxy or a VPN for anything else.

Endpoints (JSON; POST bodies are JSON):
    GET  /health                         llm reachable?, model, prompt version, knowledge size
    POST /chat    {text, state}          SSE stream: {delta}, {action}, {done, reply}
    POST /event   {event, payload, state} SSE stream (set_done, session_start, session_done)
    POST /log     {session record}       append a finished set to the workout log
    GET  /history?exercise=&days=        history_stats
    GET  /knowledge?q=                   coaching notes + catalogue hits
    GET  /exercises?q=&muscle=&equipment= catalogue search
    GET  /profile · POST /profile {category,key,value} · DELETE /profile?key=
    GET  /brief?exercise=                last-session brief for a greeting
Actions the model emits for the app (set_exercise, set_rep_goal, rest_timer,
set_tempo, cues, set_load, start_program, stop_program) are streamed to the
client to apply; history_query / exercise_lookup / plate_calc / calendar_*
run here through the same tool loop as the desktop chat.
"""
from __future__ import annotations

import argparse
import hmac
import json
import os
import secrets
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qsl, urlparse

import coach_chat
import coach_knowledge
import coach_ops
import coach_profile

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOG = os.environ.get("COACH_LOG", os.path.join(HERE, "workout_log.json"))
DEFAULT_PORT = 7799
APP_ACTIONS = {"set_exercise", "set_rep_goal", "rest_timer", "set_tempo", "cues",
               "set_load", "start_program", "stop_program"}


def _ack(action: dict) -> str:
    """Spoken acknowledgement for an app action (the phone applies it)."""
    import pose_coach
    return pose_coach.apply_chat_action(action, {"switch_to": None, "rep_goal": None,
                                                 "rest_until": 0.0, "tempo_ecc_target": None,
                                                 "cues_on": True, "program": None,
                                                 "program_new": None, "program_stop": False,
                                                 "load_kg": None})


class CoachService:
    """One athlete, one conversation; thread-safe enough for a phone."""

    def __init__(self, log_path: str = DEFAULT_LOG, profile_db: str | None = "",
                 client=None, kb: coach_knowledge.KnowledgeBase | None = None):
        self.log_path = log_path
        self.state: dict = {}
        self.profile = None
        if profile_db is not None:
            try:
                self.profile = coach_profile.ProfileStore(
                    profile_db or coach_profile.DEFAULT_DB)
            except Exception:
                self.profile = None
        self.kb = kb or coach_knowledge.default_kb()
        self.coach = coach_chat.ChatCoach(
            client=client, log_path=log_path,
            state_provider=lambda: dict(self.state), profile=self.profile,
            actions=True, calendar=None, kb=self.kb)
        self._lock = threading.Lock()

    # ---- streaming answer with the tool loop
    def answer(self, text: str, state: dict | None = None, is_event: bool = False):
        """Yields dicts: {"delta": str} | {"action": dict} | {"done": True, "reply": str}."""
        if state:
            self.state = state
        with self._lock:
            rounds, feedback, full = 0, [], []
            pending = text
            while pending and rounds < 3:
                rounds += 1
                buf, reply = "", []
                for chunk in self.coach.ask_stream(pending):
                    reply.append(chunk)
                    buf += chunk
                    sents, buf = coach_chat.split_sentences(buf)
                    for s in sents:
                        yield from self._route(s, feedback)
                if buf.strip():
                    yield from self._route(buf.strip(), feedback)
                full.append(coach_chat.parse_actions("".join(reply))[0])
                pending = None
                if feedback:
                    pending = self.coach.app_message(
                        "APP DATA", "automatic message from the app, not the athlete:\n"
                        + "\n".join(feedback) + "\nNow answer the athlete's request using "
                        "this data.")
                    feedback = []
            yield {"done": True, "reply": " ".join(x for x in full if x).strip()}

    def _route(self, sentence: str, feedback: list[str]):
        clean, acts = coach_chat.parse_actions(sentence)
        for a in acts:
            do = str(a.get("do", ""))
            if do == "history_query":
                _, fb = coach_chat.execute_history_action(self.log_path, a)
                if fb:
                    feedback.append(fb)
            elif do in ("exercise_lookup", "plate_calc"):
                _, fb = coach_chat.execute_knowledge_action(self.coach, a)
                if fb:
                    feedback.append(fb)
            elif do in APP_ACTIONS:
                ack = _ack(a)
                if ack and not ack.startswith(("I couldn't", "I don't know")):
                    yield {"action": a, "ack": ack}
                    yield {"delta": ack + " "}
                elif ack:
                    feedback.append(f"APP ERROR: {ack} Fix the problem and send a "
                                    "corrected ACTION line.")
            coach_ops.trace("action", do=do, ok=True, ack="", via="server")
        if clean:
            yield {"delta": clean + " "}

    def event(self, event: str, payload: dict | None, state: dict | None = None):
        body = dict(payload or {})
        body["event"] = event
        text = self.coach.app_tag("APP EVENT") + " " + json.dumps(body, ensure_ascii=False)
        yield from self.answer(text, state, is_event=True)

    # ---- plain JSON helpers
    def health(self) -> dict:
        ok, why = True, ""
        try:
            self.coach.client.warm_up()
        except Exception as e:                 # warm_up is best effort
            ok, why = False, str(e)[:120]
        return {"ok": True, "llm": {"model": getattr(self.coach.client, "model", "?"),
                                    "base_url": getattr(self.coach.client, "base_url", "?"),
                                    "reachable": ok, "error": why},
                "prompt_version": coach_ops.PROMPT_VERSION,
                "knowledge": {"chunks": len(self.kb.docs), "exercises": len(self.kb.exercises)},
                "profile": self.profile is not None}

    def append_log(self, record: dict) -> dict:
        if not isinstance(record, dict) or "exercise" not in record:
            return {"error": "record must be a session object"}
        history = []
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, encoding="utf-8") as fh:
                    history = json.load(fh)
            except (OSError, json.JSONDecodeError):
                history = []
        record.setdefault("source", "ios")
        history.append(record)
        with open(self.log_path, "w", encoding="utf-8") as fh:
            json.dump(history, fh, indent=1)
        return {"ok": True, "sessions": len(history)}


# ------------------------------------------------------------ HTTP layer
class Handler(BaseHTTPRequestHandler):
    service: CoachService = None
    token: str = ""

    # ---- plumbing
    def _authorized(self) -> bool:
        h = self.headers.get("Authorization", "")
        got = h[7:] if h.startswith("Bearer ") else ""
        return bool(self.token) and hmac.compare_digest(got, self.token)

    def _json(self, obj, code: int = 200):
        data = json.dumps(obj, ensure_ascii=False, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return {}
        try:
            d = json.loads(self.rfile.read(n).decode("utf-8"))
            return d if isinstance(d, dict) else {}
        except (ValueError, UnicodeDecodeError):
            return {}

    def _sse(self, gen):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            for item in gen:
                self.wfile.write(("data: " + json.dumps(item, ensure_ascii=False) + "\n\n")
                                 .encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except coach_chat.CoachOffline as e:
            try:
                self.wfile.write(("data: " + json.dumps({"error": str(e)[:300], "done": True})
                                 + "\n\n").encode())
            except OSError:
                pass

    def log_message(self, *a):
        pass

    # ---- routes
    def do_GET(self):  # noqa: N802
        if not self._authorized():
            return self._json({"error": "pair first: Authorization: Bearer <code>"}, 401)
        u = urlparse(self.path)
        q = dict(parse_qsl(u.query))
        s = self.service
        if u.path == "/health":
            return self._json(s.health())
        if u.path == "/history":
            days = max(1, min(int(q.get("days") or 90), 365))
            ex = q.get("exercise") or None
            return self._json(coach_chat.history_stats(s.log_path, exercise=ex, days=days))
        if u.path == "/brief":
            return self._json(coach_chat.last_session_brief(
                s.log_path, exercise=q.get("exercise") or None) or {})
        if u.path == "/knowledge":
            hits = s.kb.search(q.get("q", ""), k=int(q.get("k") or 5))
            return self._json({"hits": [{k: h[k] for k in ("kind", "title", "topic", "text")}
                                        for h in hits]})
        if u.path == "/exercises":
            rows = s.kb.find_exercises(q.get("q", ""), q.get("muscle", ""),
                                       q.get("equipment", ""), limit=int(q.get("limit") or 8))
            return self._json({"exercises": [
                {k: e.get(k) for k in ("id", "name", "level", "mechanic", "equipment",
                                       "primaryMuscles", "secondaryMuscles", "instructions")}
                for e in rows]})
        if u.path == "/profile":
            if s.profile is None:
                return self._json({"facts": []})
            return self._json({"facts": [{"category": c, "key": k, "value": v}
                                         for c, k, v, _ in s.profile.facts()]})
        self._json({"error": "not found"}, 404)

    def do_POST(self):  # noqa: N802
        if not self._authorized():
            return self._json({"error": "pair first: Authorization: Bearer <code>"}, 401)
        u = urlparse(self.path)
        body = self._body()
        s = self.service
        if u.path == "/chat":
            text = coach_ops.sanitize_athlete_text(str(body.get("text", "")).strip())
            if not text:
                return self._json({"error": "text required"}, 400)
            return self._sse(s.answer(text, body.get("state")))
        if u.path == "/event":
            ev = str(body.get("event", ""))
            if ev not in ("set_done", "session_start", "session_done"):
                return self._json({"error": "unknown event"}, 400)
            return self._sse(s.event(ev, body.get("payload"), body.get("state")))
        if u.path == "/log":
            return self._json(s.append_log(body.get("record") or body))
        if u.path == "/profile":
            if s.profile is None:
                return self._json({"error": "profile disabled"}, 400)
            cat, key, val = (str(body.get("category", "")), str(body.get("key", "")),
                             str(body.get("value", "")))
            if cat not in coach_profile.CATEGORIES or not key or not val:
                return self._json({"error": "category/key/value required"}, 400)
            s.profile.remember(cat, key, val)
            return self._json({"ok": True})
        self._json({"error": "not found"}, 404)

    def do_DELETE(self):  # noqa: N802
        if not self._authorized():
            return self._json({"error": "pair first"}, 401)
        u = urlparse(self.path)
        q = dict(parse_qsl(u.query))
        if u.path == "/profile" and self.service.profile is not None:
            return self._json({"ok": True, "forgot": self.service.profile.forget(
                q.get("key", ""))})
        self._json({"error": "not found"}, 404)


def local_ip() -> str:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("10.255.255.255", 1))
        ip = sock.getsockname()[0]
        sock.close()
        return ip
    except OSError:
        return "127.0.0.1"


def serve(host: str, port: int, token: str, service: CoachService) -> None:
    handler = type("H", (Handler,), {"service": service, "token": token})
    srv = ThreadingHTTPServer((host, port), handler)
    p = srv.server_address[1]
    h = service.health()
    print(f"Coach server on http://{local_ip()}:{p}  (also http://127.0.0.1:{p})")
    print(f"Pairing code: {token}   ← enter URL + code in the app: Coach settings")
    print(f"LLM: {h['llm']['model']} @ {h['llm']['base_url']} "
          f"({'reachable' if h['llm']['reachable'] else 'NOT reachable: ' + h['llm']['error']})")
    print(f"Knowledge: {h['knowledge']['chunks']} chunks, {h['knowledge']['exercises']} "
          f"exercises · prompt {h['prompt_version']} · Ctrl+C to stop")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        srv.server_close()


# --------------------------------------------------------------- selftest
class _FakeLLM:
    model, base_url, prompt_fp = "fake", "http://fake/v1", ""

    def __init__(self):
        self.calls: list[list[dict]] = []

    def warm_up(self):
        pass

    def chat_stream(self, messages):
        self.calls.append(messages)
        user = messages[-1]["content"]
        if "[APP DATA" in user:
            yield "Glute bridges and split squats — three sets of twelve. "
            return
        if "[APP EVENT" in user:
            yield "Nice set — knees out next time."
            return
        if "glutes" in user:
            yield "Let me look that up.\n"
            yield 'ACTION: {"do": "exercise_lookup", "muscle": "glutes", "equipment": "body only"}'
            return
        if "12 reps" in user:
            yield 'Twelve it is. ACTION: {"do": "set_rep_goal", "reps": 12}'
            return
        yield "Chest up, knees out. "
        yield "Own the bottom position."

    def chat(self, messages):
        return "".join(self.chat_stream(messages))


def selftest():
    import io
    import tempfile
    import urllib.error
    import urllib.request
    print("== coach_server selftests ==")
    with tempfile.TemporaryDirectory() as td:
        log = os.path.join(td, "log.json")
        cat = os.path.join(td, "ex.json")
        with open(cat, "w", encoding="utf-8") as fh:
            json.dump({"exercises": coach_knowledge._sample_catalogue()}, fh)
        kb = coach_knowledge.KnowledgeBase(catalogue=cat,
                                           embedder=coach_knowledge.Embedder(model=""))
        svc = CoachService(log_path=log, profile_db=os.path.join(td, "p.db"),
                           client=_FakeLLM(), kb=kb)

        print("1) streamed answer carries the live state and splits sentences:", end=" ")
        out = list(svc.answer("why do my knees cave?", {"exercise": "squat", "reps": 4}))
        deltas = [o["delta"] for o in out if "delta" in o]
        assert len(deltas) >= 2 and out[-1]["done"] and "knees out" in out[-1]["reply"]
        sysm = svc.coach._last_system
        assert '"reps": 4' in sysm and "RELEVANT KNOWLEDGE" in sysm and "knees_cave" in sysm
        print("OK")

        print("2) app actions are streamed to the phone with a spoken ack:", end=" ")
        out = list(svc.answer("let's do 12 reps this set"))
        acts = [o for o in out if "action" in o]
        assert acts and acts[0]["action"]["do"] == "set_rep_goal" and acts[0]["action"]["reps"] == 12
        assert acts[0]["ack"].startswith("Rep goal set")
        assert "ACTION" not in out[-1]["reply"]
        print("OK")

        print("3) knowledge tools run server-side through the tool loop:", end=" ")
        out = list(svc.answer("what can I do for glutes with no equipment?"))
        assert "Glute bridges" in out[-1]["reply"], out[-1]
        assert any("[APP DATA" in m[-1]["content"] for m in svc.coach.client.calls[-1:]) or \
            "Glute Bridge" in svc.coach.client.calls[-1][-1]["content"]
        print("OK")

        print("4) events (set_done) get a short proactive note:", end=" ")
        out = list(svc.event("set_done", {"reps": 8, "avg_score": 82}, {"exercise": "squat"}))
        assert "Nice set" in out[-1]["reply"]
        print("OK")

        print("5) HTTP: pairing token, SSE framing, log append, history, profile:", end=" ")
        token = "pair-123"
        handler = type("H", (Handler,), {"service": svc, "token": token})
        srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{srv.server_address[1]}"

        def call(method, path, body=None, auth=True):
            req = urllib.request.Request(base + path, method=method,
                                         data=json.dumps(body).encode() if body is not None else None,
                                         headers={"Content-Type": "application/json",
                                                  **({"Authorization": f"Bearer {token}"} if auth else {})})
            try:
                with urllib.request.urlopen(req, timeout=10) as r:
                    return r.status, r.read().decode()
            except urllib.error.HTTPError as e:
                return e.code, e.read().decode()
        try:
            code, _ = call("GET", "/health", auth=False)
            assert code == 401
            req = urllib.request.Request(base + "/health", headers={"Authorization": "Bearer nope"})
            try:
                urllib.request.urlopen(req, timeout=5)
                raise AssertionError("wrong token accepted")
            except urllib.error.HTTPError as e:
                assert e.code == 401
            code, body = call("GET", "/health")
            h = json.loads(body)
            assert code == 200 and h["knowledge"]["exercises"] == 4 and h["prompt_version"]
            code, body = call("POST", "/chat", {"text": "why do my knees cave?",
                                                "state": {"exercise": "squat"}})
            events = [json.loads(ln[6:]) for ln in body.splitlines() if ln.startswith("data: ")]
            assert code == 200 and events[-1]["done"] and any("delta" in e for e in events)
            code, body = call("POST", "/chat", {"text": ""})
            assert code == 400
            rec = {"started": "2026-08-30 18:00:00", "exercise": "squat", "duration_s": 60,
                   "reps": [{"n": 1, "score": 90, "faults": [], "eccentric_s": 1.2,
                             "concentric_s": 1.0, "min_angle": 85, "velocity": 30,
                             "similarity": None}],
                   "plank": None,
                   "summary": {"reps": 1, "avg_score": 90.0, "avg_concentric_s": 1.0,
                               "avg_similarity": None, "fault_counts": {},
                               "velocity_loss_pct": None}}
            code, body = call("POST", "/log", {"record": rec})
            assert json.loads(body)["sessions"] == 1
            code, body = call("GET", "/history?exercise=squat&days=30")
            assert code == 200 and "squat" in body
            code, body = call("GET", "/brief?exercise=squat")
            assert code == 200
            code, body = call("POST", "/profile", {"category": "injuries", "key": "knee",
                                                   "value": "left meniscus"})
            assert json.loads(body)["ok"]
            code, body = call("GET", "/profile")
            assert "meniscus" in body
            code, body = call("DELETE", "/profile?key=knee")
            assert json.loads(body)["forgot"] == 1
            code, body = call("GET", "/exercises?muscle=glutes&equipment=body%20only")
            assert json.loads(body)["exercises"][0]["name"] == "Glute Bridge"
            code, body = call("GET", "/knowledge?q=protein")
            assert "Protein" in body
            code, body = call("POST", "/event", {"event": "bogus"})
            assert code == 400
            code, body = call("GET", "/nope")
            assert code == 404
        finally:
            srv.shutdown()
            srv.server_close()
        print("OK")
    print("\nAll coach_server selftests passed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.environ.get("COACH_SERVER_PORT", DEFAULT_PORT)))
    ap.add_argument("--token", default=os.environ.get("COACH_PAIR_TOKEN", ""),
                    help="pairing code (default: random, printed at start)")
    ap.add_argument("--log", default=DEFAULT_LOG)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        sys.exit(0)
    coach_ops.local_no_proxy()
    token = args.token or secrets.token_hex(3).upper()
    svc = CoachService(log_path=args.log)
    coach_chat.warn_remote_backend(svc.coach.client.base_url)
    serve(args.host, args.port, token, svc)
