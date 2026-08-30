"""coach_mcp.py — the athlete's data as an MCP server.

Model Context Protocol lets any MCP client (Claude Desktop, Claude Code,
Cursor, an agent framework) call tools on the user's machine. This server
exposes what the built-in coach knows — the workout log, the athlete
profile, the exercise catalogue + coaching notes, the live session — so a
bigger model can coach with the same grounded data, and it can write the
same things back (remember a fact, queue an app command).

Standard library only, stdio transport, JSON-RPC 2.0, MCP 2025-06-18
(also answers 2024-11-05 clients). Nothing leaves the machine unless the
client you connect is remote — that is your choice, not the app's.

    python coach_mcp.py                       # serve on stdio (for clients)
    python coach_mcp.py --list                # print the tools
    python coach_mcp.py --call get_last_session
    python coach_mcp.py --call search_exercises '{"muscle": "glutes"}'
    python coach_mcp.py --selftest

Register with Claude Code:
    claude mcp add gym-coach -- uv run --directory /path/to/ai-gym-coach \\
        python coach_mcp.py
Claude Desktop (claude_desktop_config.json):
    {"mcpServers": {"gym-coach": {"command": "uv", "args": ["run",
      "--directory", "/path/to/ai-gym-coach", "python", "coach_mcp.py"]}}}

Tools: get_training_overview, get_training_history, get_last_session,
get_profile, remember_fact, forget_fact, search_exercises, get_exercise,
get_coaching_notes, plate_calculator, get_live_state, queue_app_command.
Resources: coach://profile, coach://history, coach://live
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import coach_knowledge

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOG = os.environ.get("COACH_LOG", os.path.join(HERE, "workout_log.json"))
LIVE_FILE = os.environ.get("COACH_LIVE_FILE",
                           os.path.join(HERE, "data", "live_state.json"))
COMMANDS_FILE = os.environ.get("COACH_COMMANDS_FILE",
                               os.path.join(HERE, "data", "app_commands.jsonl"))
PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
SERVER_INFO = {"name": "ai-gym-coach", "version": "0.2.0"}

APP_EXERCISES = ("squat", "pushup", "bench", "deadlift", "lunge",
                 "shoulder_press", "curl", "pullup", "plank")
ALLOWED_COMMANDS = {"set_exercise", "set_rep_goal", "rest_timer", "set_tempo",
                    "cues", "start_program", "stop_program", "set_load"}


# ------------------------------------------------------------- tool impls
class Tools:
    def __init__(self, log_path: str = DEFAULT_LOG, profile_db: str | None = "",
                 kb: coach_knowledge.KnowledgeBase | None = None,
                 live_file: str = LIVE_FILE, commands_file: str = COMMANDS_FILE):
        self.log_path = log_path
        self.live_file = live_file
        self.commands_file = commands_file
        self.kb = kb or coach_knowledge.default_kb()
        self.profile = None
        if profile_db is not None:
            try:
                import coach_profile
                self.profile = coach_profile.ProfileStore(
                    profile_db or coach_profile.DEFAULT_DB)
            except Exception:
                self.profile = None

    # ---- history
    def get_training_overview(self) -> dict:
        import coach_chat
        return {"recent_sessions": coach_chat.progress_summary(self.log_path),
                "per_exercise": coach_chat.history_overview(self.log_path)}

    def get_training_history(self, exercise: str | None = None,
                             days: int = 90) -> dict:
        import coach_chat
        days = max(1, min(int(days or 90), 365))
        if exercise and exercise not in APP_EXERCISES:
            return {"error": f"unknown exercise '{exercise}'; app exercises: "
                             f"{', '.join(APP_EXERCISES)}"}
        return coach_chat.history_stats(self.log_path, exercise=exercise or None,
                                        days=days)

    def get_last_session(self) -> dict:
        import coach_dashboard
        hist = coach_dashboard.load_history(self.log_path)
        if not hist:
            return {"error": "no sessions logged yet"}
        return coach_dashboard.session_detail(hist[-1])

    # ---- profile
    def get_profile(self) -> dict:
        if self.profile is None:
            return {"facts": [], "note": "profile memory disabled"}
        return {"facts": [{"category": c, "key": k, "value": v, "updated": u}
                          for c, k, v, u in self.profile.facts()]}

    def remember_fact(self, category: str, key: str, value: str) -> dict:
        if self.profile is None:
            return {"error": "profile memory disabled"}
        import coach_profile
        if category not in coach_profile.CATEGORIES:
            return {"error": f"category must be one of "
                             f"{', '.join(coach_profile.CATEGORIES)}"}
        self.profile.remember(category, key, value)
        return {"ok": True, "remembered": {"category": category, "key": key,
                                          "value": value}}

    def forget_fact(self, key: str) -> dict:
        if self.profile is None:
            return {"error": "profile memory disabled"}
        return {"ok": True, "forgot": self.profile.forget(key)}

    # ---- knowledge
    def search_exercises(self, query: str = "", muscle: str = "",
                         equipment: str = "", limit: int = 8) -> dict:
        if not self.kb.exercises:
            return {"error": "catalogue not installed: python coach_knowledge.py --fetch"}
        rows = self.kb.find_exercises(query, muscle, equipment,
                                      limit=max(1, min(int(limit or 8), 30)))
        return {"count": len(rows), "exercises": [
            {k: e.get(k) for k in ("id", "name", "level", "mechanic", "equipment",
                                   "category", "primaryMuscles",
                                   "secondaryMuscles")} for e in rows]}

    def get_exercise(self, name: str) -> dict:
        e = self.kb.exercise(name)
        if not e:
            return {"error": f"no exercise matches '{name}'"}
        out = dict(e)
        out["images"] = [coach_knowledge.CATALOGUE_IMAGE_BASE + i
                         if not str(i).startswith("http") else i
                         for i in e.get("images") or []]
        out["trackable_by_camera"] = any(
            e["name"].lower() in names
            for names in coach_knowledge.APP_TO_CATALOGUE.values())
        return out

    def get_coaching_notes(self, query: str, limit: int = 4) -> dict:
        hits = self.kb.search(query, k=max(1, min(int(limit or 4), 10)),
                              kinds=("note",))
        return {"notes": [{"topic": h["topic"], "title": h["title"],
                           "text": h["text"]} for h in hits]}

    def plate_calculator(self, kg: float, bar_kg: float = 20.0) -> dict:
        return {"result": coach_knowledge.plate_calculator(float(kg), float(bar_kg))}

    # ---- live app
    def get_live_state(self) -> dict:
        try:
            with open(self.live_file, encoding="utf-8") as fh:
                state = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return {"live": False, "note": "no session running (pose_coach.py "
                                           "--coach writes data/live_state.json)"}
        age = time.time() - float(state.get("_written", 0))
        state["live"] = age < 5.0
        state["age_s"] = round(age, 1)
        return state

    def queue_app_command(self, do: str, **args) -> dict:
        if do not in ALLOWED_COMMANDS:
            return {"error": f"unknown command '{do}'; allowed: "
                             f"{', '.join(sorted(ALLOWED_COMMANDS))}"}
        cmd = {"do": do, **args, "_queued": time.time(), "_by": "mcp"}
        os.makedirs(os.path.dirname(self.commands_file) or ".", exist_ok=True)
        with open(self.commands_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(cmd) + "\n")
        return {"ok": True, "queued": cmd,
                "note": "applied by a running pose_coach.py --coach session "
                        "within a second; ignored otherwise"}


TOOL_SPECS = [
    ("get_training_overview", "Recent sessions and per-exercise trends from the "
     "athlete's workout log (what the built-in coach sees).", {}),
    ("get_training_history", "Sessions and totals for a period, optionally one "
     "exercise. Use before stating any number.",
     {"exercise": {"type": "string", "enum": list(APP_EXERCISES)},
      "days": {"type": "integer", "minimum": 1, "maximum": 365, "default": 90}}),
    ("get_last_session", "The last session rep by rep: scores, faults, tempo, "
     "golden-rep similarity, fatigue, heart rate.", {}),
    ("get_profile", "Long-term facts the coach remembers: goals, injuries, "
     "equipment, preferences.", {}),
    ("remember_fact", "Store a fact about the athlete (categories: goals, "
     "injuries, equipment, preferences, stats).",
     {"category": {"type": "string"}, "key": {"type": "string"},
      "value": {"type": "string"}}, ("category", "key", "value")),
    ("forget_fact", "Delete a remembered fact by key ('all' wipes the profile).",
     {"key": {"type": "string"}}, ("key",)),
    ("search_exercises", "Search the open exercise catalogue (870+ exercises) by "
     "text, muscle and/or equipment.",
     {"query": {"type": "string"}, "muscle": {"type": "string"},
      "equipment": {"type": "string"}, "limit": {"type": "integer", "default": 8}}),
    ("get_exercise", "Full catalogue entry for one exercise: muscles, "
     "equipment, level, step-by-step instructions, image URLs.",
     {"name": {"type": "string"}}, ("name",)),
    ("get_coaching_notes", "Evidence-based coaching notes (form faults, "
     "programming, recovery, nutrition, alternatives) relevant to a question.",
     {"query": {"type": "string"}, "limit": {"type": "integer", "default": 4}},
     ("query",)),
    ("plate_calculator", "Plates per side to load a target weight on a bar.",
     {"kg": {"type": "number"}, "bar_kg": {"type": "number", "default": 20}},
     ("kg",)),
    ("get_live_state", "What the camera coach sees right now (exercise, phase, "
     "reps, last rep, faults, environment) if a session is running.", {}),
    ("queue_app_command", "Drive the running app: set_exercise, set_rep_goal, "
     "rest_timer, set_tempo, cues, start_program, stop_program, set_load.",
     {"do": {"type": "string", "enum": sorted(ALLOWED_COMMANDS)},
      "exercise": {"type": "string"}, "reps": {"type": "integer"},
      "seconds": {"type": "number"}, "eccentric_s": {"type": "number"},
      "enabled": {"type": "boolean"}, "plan": {"type": "string"},
      "kg": {"type": "number"}}, ("do",)),
]


def tool_list() -> list[dict]:
    out = []
    for spec in TOOL_SPECS:
        name, desc, props = spec[0], spec[1], spec[2]
        required = list(spec[3]) if len(spec) > 3 else []
        out.append({"name": name, "description": desc,
                    "inputSchema": {"type": "object", "properties": props,
                                    "required": required}})
    return out


RESOURCES = [
    {"uri": "coach://profile", "name": "Athlete profile",
     "mimeType": "application/json", "description": "Remembered facts"},
    {"uri": "coach://history", "name": "Training overview",
     "mimeType": "application/json", "description": "Recent sessions + trends"},
    {"uri": "coach://live", "name": "Live session",
     "mimeType": "application/json", "description": "Camera coach state now"},
]


# ------------------------------------------------------------- JSON-RPC
class Server:
    def __init__(self, tools: Tools):
        self.tools = tools
        self.initialized = False

    def handle(self, msg: dict) -> dict | None:
        mid = msg.get("id")
        method = msg.get("method", "")
        params = msg.get("params") or {}
        try:
            if method == "initialize":
                want = params.get("protocolVersion", PROTOCOL_VERSIONS[0])
                ver = want if want in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
                result = {"protocolVersion": ver,
                          "capabilities": {"tools": {"listChanged": False},
                                           "resources": {"listChanged": False}},
                          "serverInfo": SERVER_INFO,
                          "instructions": (
                              "Local AI Gym Coach data. Call get_training_history "
                              "or get_last_session before quoting numbers; "
                              "search_exercises/get_exercise for movements; "
                              "remember_fact to store what the athlete tells you. "
                              "Not medical advice: sharp pain, numbness, dizziness "
                              "or chest pain → stop and see a professional.")}
            elif method == "notifications/initialized":
                self.initialized = True
                return None
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": tool_list()}
            elif method == "tools/call":
                result = self._call(params.get("name", ""),
                                    params.get("arguments") or {})
            elif method == "resources/list":
                result = {"resources": RESOURCES}
            elif method == "resources/read":
                result = self._read(params.get("uri", ""))
            elif method.startswith("notifications/"):
                return None
            else:
                return self._error(mid, -32601, f"method not found: {method}")
        except Exception as e:                       # never kill the server
            return self._error(mid, -32603, f"{type(e).__name__}: {e}")
        if mid is None:
            return None
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    def _call(self, name: str, args: dict) -> dict:
        names = {t["name"] for t in tool_list()}
        if name not in names:
            return {"content": [{"type": "text", "text": f"unknown tool {name}"}],
                    "isError": True}
        fn = getattr(self.tools, name)
        try:
            data = fn(**args)
        except TypeError as e:
            return {"content": [{"type": "text", "text": f"bad arguments: {e}"}],
                    "isError": True}
        text = json.dumps(data, ensure_ascii=False, indent=1, default=str)
        return {"content": [{"type": "text", "text": text}],
                "structuredContent": data if isinstance(data, dict) else {"result": data},
                "isError": bool(isinstance(data, dict) and data.get("error"))}

    def _read(self, uri: str) -> dict:
        if uri == "coach://profile":
            data = self.tools.get_profile()
        elif uri == "coach://history":
            data = self.tools.get_training_overview()
        elif uri == "coach://live":
            data = self.tools.get_live_state()
        else:
            raise ValueError(f"unknown resource {uri}")
        return {"contents": [{"uri": uri, "mimeType": "application/json",
                              "text": json.dumps(data, ensure_ascii=False,
                                                 default=str)}]}

    @staticmethod
    def _error(mid, code: int, message: str) -> dict:
        return {"jsonrpc": "2.0", "id": mid,
                "error": {"code": code, "message": message}}


def serve_stdio(tools: Tools | None = None, inp=None, out=None) -> None:
    """Newline-delimited JSON-RPC over stdio (the MCP stdio transport)."""
    inp = inp or sys.stdin
    out = out or sys.stdout
    srv = Server(tools or Tools())
    for line in inp:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            resp = Server._error(None, -32700, "parse error")
        else:
            resp = srv.handle(msg)
        if resp is not None:
            out.write(json.dumps(resp, ensure_ascii=False) + "\n")
            out.flush()


# --------------------------------------------------------------- selftest
def selftest():
    import io
    import tempfile
    print("== coach_mcp selftests ==")
    with tempfile.TemporaryDirectory() as td:
        log = os.path.join(td, "log.json")
        import coach_dashboard
        with open(log, "w", encoding="utf-8") as fh:
            json.dump(coach_dashboard._fake_history(), fh)
        cat = os.path.join(td, "ex.json")
        with open(cat, "w", encoding="utf-8") as fh:
            json.dump({"exercises": coach_knowledge._sample_catalogue()}, fh)
        kb = coach_knowledge.KnowledgeBase(catalogue=cat,
                                           embedder=coach_knowledge.Embedder(model=""))
        tools = Tools(log_path=log, profile_db=os.path.join(td, "p.db"), kb=kb,
                      live_file=os.path.join(td, "live.json"),
                      commands_file=os.path.join(td, "cmds.jsonl"))
        srv = Server(tools)

        print("1) initialize / tools list / ping:", end=" ")
        r = srv.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                        "params": {"protocolVersion": "2024-11-05"}})
        assert r["result"]["protocolVersion"] == "2024-11-05"
        assert r["result"]["serverInfo"]["name"] == "ai-gym-coach"
        assert srv.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
        r = srv.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        names = [t["name"] for t in r["result"]["tools"]]
        assert "get_last_session" in names and "queue_app_command" in names
        assert all("inputSchema" in t for t in r["result"]["tools"])
        assert srv.handle({"jsonrpc": "2.0", "id": 3, "method": "ping"})["result"] == {}
        print(f"OK ({len(names)} tools)")

        print("2) history tools ground the numbers:", end=" ")
        r = srv.handle({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                        "params": {"name": "get_last_session"}})
        d = r["result"]["structuredContent"]
        assert d["exercise"] == "squat" and d["count"] == 3, d
        assert not r["result"]["isError"]
        r = srv.handle({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                        "params": {"name": "get_training_history",
                                   "arguments": {"exercise": "squat", "days": 365}}})
        assert "squat" in json.dumps(r["result"]["structuredContent"])
        r = srv.handle({"jsonrpc": "2.0", "id": 6, "method": "tools/call",
                        "params": {"name": "get_training_history",
                                   "arguments": {"exercise": "yoga"}}})
        assert r["result"]["isError"]
        print("OK")

        print("3) profile round-trip via tools and resource:", end=" ")
        r = srv.handle({"jsonrpc": "2.0", "id": 7, "method": "tools/call",
                        "params": {"name": "remember_fact",
                                   "arguments": {"category": "injuries",
                                                 "key": "left_knee",
                                                 "value": "meniscus, no deep squats"}}})
        assert r["result"]["structuredContent"]["ok"]
        r = srv.handle({"jsonrpc": "2.0", "id": 8, "method": "resources/read",
                        "params": {"uri": "coach://profile"}})
        assert "meniscus" in r["result"]["contents"][0]["text"]
        r = srv.handle({"jsonrpc": "2.0", "id": 9, "method": "tools/call",
                        "params": {"name": "remember_fact",
                                   "arguments": {"category": "nope", "key": "k",
                                                 "value": "v"}}})
        assert r["result"]["isError"]
        print("OK")

        print("4) catalogue + notes + plates:", end=" ")
        r = srv.handle({"jsonrpc": "2.0", "id": 10, "method": "tools/call",
                        "params": {"name": "search_exercises",
                                   "arguments": {"muscle": "glutes",
                                                 "equipment": "body only"}}})
        d = r["result"]["structuredContent"]
        assert d["count"] == 1 and d["exercises"][0]["name"] == "Glute Bridge", d
        r = srv.handle({"jsonrpc": "2.0", "id": 11, "method": "tools/call",
                        "params": {"name": "get_exercise", "arguments": {"name": "squat"}}})
        d = r["result"]["structuredContent"]
        assert d["name"] == "Barbell Full Squat" and d["trackable_by_camera"]
        assert d["images"][0].startswith("https://")
        r = srv.handle({"jsonrpc": "2.0", "id": 12, "method": "tools/call",
                        "params": {"name": "get_coaching_notes",
                                   "arguments": {"query": "knees cave squat"}}})
        assert "knees_cave" in r["result"]["structuredContent"]["notes"][0]["title"]
        r = srv.handle({"jsonrpc": "2.0", "id": 13, "method": "tools/call",
                        "params": {"name": "plate_calculator", "arguments": {"kg": 60}}})
        assert "1×20" in r["result"]["structuredContent"]["result"]
        print("OK")

        print("5) live state + queued commands (file hand-off to the app):", end=" ")
        r = srv.handle({"jsonrpc": "2.0", "id": 14, "method": "tools/call",
                        "params": {"name": "get_live_state"}})
        assert r["result"]["structuredContent"]["live"] is False
        with open(tools.live_file, "w", encoding="utf-8") as fh:
            json.dump({"exercise": "squat", "reps": 4, "_written": time.time()}, fh)
        r = srv.handle({"jsonrpc": "2.0", "id": 15, "method": "tools/call",
                        "params": {"name": "get_live_state"}})
        assert r["result"]["structuredContent"]["live"] is True
        r = srv.handle({"jsonrpc": "2.0", "id": 16, "method": "tools/call",
                        "params": {"name": "queue_app_command",
                                   "arguments": {"do": "set_rep_goal", "reps": 8}}})
        assert r["result"]["structuredContent"]["ok"]
        with open(tools.commands_file, encoding="utf-8") as fh:
            q = [json.loads(x) for x in fh]
        assert q[0]["do"] == "set_rep_goal" and q[0]["reps"] == 8
        r = srv.handle({"jsonrpc": "2.0", "id": 17, "method": "tools/call",
                        "params": {"name": "queue_app_command",
                                   "arguments": {"do": "rm -rf"}}})
        assert r["result"]["isError"]
        print("OK")

        print("6) transport: bad JSON, unknown method, unknown tool never crash:", end=" ")
        inp = io.StringIO("not json\n"
                          '{"jsonrpc":"2.0","id":1,"method":"nope"}\n'
                          '{"jsonrpc":"2.0","id":2,"method":"tools/call",'
                          '"params":{"name":"zzz"}}\n'
                          '{"jsonrpc":"2.0","id":3,"method":"tools/call",'
                          '"params":{"name":"get_exercise","arguments":{"bogus":1}}}\n'
                          '{"jsonrpc":"2.0","id":4,"method":"resources/list"}\n')
        out = io.StringIO()
        serve_stdio(tools, inp, out)
        lines = [json.loads(x) for x in out.getvalue().splitlines()]
        assert lines[0]["error"]["code"] == -32700
        assert lines[1]["error"]["code"] == -32601
        assert lines[2]["result"]["isError"]
        assert "bad arguments" in lines[3]["result"]["content"][0]["text"]
        assert len(lines[4]["result"]["resources"]) == 3
        print("OK")
    print("\nAll coach_mcp selftests passed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--log", default=DEFAULT_LOG)
    ap.add_argument("--list", action="store_true", help="print the tools and exit")
    ap.add_argument("--call", nargs="+", metavar=("TOOL", "JSON"),
                    help="call one tool locally and print the result")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    elif args.list:
        for t in tool_list():
            print(f"{t['name']:24s} {t['description']}")
    elif args.call:
        tools = Tools(log_path=args.log)
        payload = json.loads(args.call[1]) if len(args.call) > 1 else {}
        r = Server(tools).handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                  "params": {"name": args.call[0],
                                             "arguments": payload}})
        print(json.dumps(r, ensure_ascii=False, indent=1, default=str))
    else:
        serve_stdio(Tools(log_path=args.log))
