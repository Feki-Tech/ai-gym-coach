"""Progress dashboard for the AI Gym Coach — a local web page with charts.

Reads workout_log.json (written by pose_coach.py after every session) and
serves a self-contained HTML page: training volume per week, per-exercise
form-score and rep trends, personal records, fault breakdowns and a recent
session table. Charts are server-side SVG — no JavaScript frameworks, no
CDN, no internet. Standard library only.

Usage:
    python coach_dashboard.py                      # serve http://localhost:7788
    python coach_dashboard.py --port 9000
    python coach_dashboard.py --export report.html # write a static file
    python coach_dashboard.py --selftest

The page re-reads the log on every refresh, so you can keep it open while
you train (it auto-refreshes every 60 s).
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOG = os.environ.get("COACH_LOG",
                             os.path.join(HERE, "workout_log.json"))
DEFAULT_PORT = 7788

ACCENT = "#4ade80"      # green
ACCENT2 = "#60a5fa"     # blue
WARN = "#f87171"        # red


# ---------------------------------------------------------------- loading
def load_history(path: str) -> list[dict]:
    """Read the workout log; tolerate a missing or corrupt file."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _session_date(s: dict) -> dt.date | None:
    try:
        return dt.datetime.strptime(s.get("started", ""),
                                    "%Y-%m-%d %H:%M:%S").date()
    except ValueError:
        return None


# ------------------------------------------------------------ aggregation
def _streaks(days: list[dt.date]) -> tuple[int, int]:
    """(current, longest) streak of consecutive training days.

    `current` is the run ending at the most recent training day.
    """
    if not days:
        return 0, 0
    uniq = sorted(set(days))
    longest = cur = 1
    for a, b in zip(uniq, uniq[1:]):
        cur = cur + 1 if (b - a).days == 1 else 1
        longest = max(longest, cur)
    return cur, longest


def aggregate(history: list[dict]) -> dict:
    """Distill the raw session log into everything the page shows."""
    totals = {"sessions": len(history), "reps": 0, "duration_s": 0.0,
              "best_score": None, "hold_s": 0.0}
    weekly: dict[str, dict] = {}          # "2026-W28" -> {sessions, reps}
    exercises: dict[str, dict] = {}
    days: list[dt.date] = []
    recent: list[dict] = []

    for s in history:
        summ = s.get("summary") or {}
        ex = s.get("exercise") or "?"
        date = _session_date(s)
        reps = summ.get("reps") or 0
        score = summ.get("avg_score")
        plank = s.get("plank") or None
        hold = plank.get("total_hold_s", 0.0) if plank else 0.0

        totals["reps"] += reps
        totals["duration_s"] += s.get("duration_s") or 0.0
        totals["hold_s"] += hold
        best_rep = max((r.get("score", 0) for r in s.get("reps", [])),
                       default=None)
        if best_rep is not None:
            if totals["best_score"] is None or best_rep > totals["best_score"]:
                totals["best_score"] = best_rep

        if date:
            days.append(date)
            iso = date.isocalendar()
            wk = f"{iso[0]}-W{iso[1]:02d}"
            w = weekly.setdefault(wk, {"sessions": 0, "reps": 0, "hold_s": 0.0})
            w["sessions"] += 1
            w["reps"] += reps
            w["hold_s"] += hold

        e = exercises.setdefault(ex, {
            "sessions": 0, "total_reps": 0, "scores": [], "reps_series": [],
            "holds": [], "faults": {}, "prs": {}})
        e["sessions"] += 1
        e["total_reps"] += reps
        label = date.isoformat() if date else "?"
        if score is not None:
            e["scores"].append({"label": label, "value": score})
        if reps:
            e["reps_series"].append({"label": label, "value": reps})
        if plank:
            e["holds"].append({"label": label,
                               "value": plank.get("total_hold_s", 0.0)})
        for k, v in (summ.get("fault_counts") or {}).items():
            e["faults"][k] = e["faults"].get(k, 0) + v

        top_fault = max((summ.get("fault_counts") or {}).items(),
                        key=lambda kv: kv[1], default=(None, 0))[0]
        recent.append({"date": s.get("started", "?"), "exercise": ex,
                       "reps": reps, "score": score,
                       "hold_s": round(hold, 1) if hold else None,
                       "top_fault": top_fault})

    for ex, e in exercises.items():
        prs = e["prs"]
        if e["reps_series"]:
            prs["max_reps_session"] = max(p["value"] for p in e["reps_series"])
        if e["scores"]:
            prs["best_avg_score"] = max(p["value"] for p in e["scores"])
        if e["holds"]:
            prs["longest_hold_s"] = max(p["value"] for p in e["holds"])

    current, longest = _streaks(days)
    totals["streak"] = current
    totals["longest_streak"] = longest
    totals["last_day"] = max(days).isoformat() if days else None
    totals["active_days"] = len(set(days))

    weeks = sorted(weekly)[-12:]
    return {
        "totals": totals,
        "weekly": [{"label": w, **weekly[w]} for w in weeks],
        "exercises": exercises,
        "recent": list(reversed(recent))[:10],
    }


# ------------------------------------------------------------- SVG charts
def _scale(values: list[float], lo: float, hi: float,
           out_lo: float, out_hi: float) -> list[float]:
    span = (hi - lo) or 1.0
    return [out_lo + (v - lo) / span * (out_hi - out_lo) for v in values]


def svg_line(points: list[dict], width=560, height=130, color=ACCENT) -> str:
    """Line chart with dots + native tooltips from [{label, value}]."""
    if not points:
        return ""
    vals = [p["value"] for p in points]
    lo, hi = min(vals), max(vals)
    pad = 10
    if len(points) == 1:
        xs = [width / 2]
    else:
        xs = _scale(list(range(len(vals))), 0, len(vals) - 1,
                    pad, width - pad)
    ys = _scale(vals, lo, hi, height - pad, pad)
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    dots = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}">'
        f'<title>{html.escape(str(p["label"]))}: {p["value"]}</title></circle>'
        for x, y, p in zip(xs, ys, points))
    line = (f'<polyline points="{pts}" fill="none" stroke="{color}" '
            f'stroke-width="2"/>' if len(points) > 1 else "")
    return (f'<svg viewBox="0 0 {width} {height}" class="chart">'
            f'{line}{dots}'
            f'<text x="4" y="12" class="axis">{hi:g}</text>'
            f'<text x="4" y="{height - 4}" class="axis">{lo:g}</text></svg>')


def svg_bars(points: list[dict], width=560, height=130, color=ACCENT2) -> str:
    """Bar chart with native tooltips from [{label, value}]."""
    if not points:
        return ""
    vals = [p["value"] for p in points]
    hi = max(vals) or 1
    pad = 10
    n = len(points)
    slot = (width - 2 * pad) / n
    bw = max(3.0, slot * 0.65)
    bars = []
    for i, p in enumerate(points):
        h = (p["value"] / hi) * (height - 2 * pad)
        x = pad + i * slot + (slot - bw) / 2
        y = height - pad - h
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" '
            f'height="{max(h, 1):.1f}" rx="2" fill="{color}">'
            f'<title>{html.escape(str(p["label"]))}: {p["value"]}</title>'
            f'</rect>')
    return (f'<svg viewBox="0 0 {width} {height}" class="chart">'
            f'{"".join(bars)}'
            f'<text x="4" y="12" class="axis">{hi:g}</text></svg>')


# ---------------------------------------------------------------- page
_CSS = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin: 0; padding: 24px; background: #0f1115; color: #e5e7eb;
       font: 15px/1.5 system-ui, "Segoe UI", sans-serif; }
h1 { margin: 0 0 4px; font-size: 26px; }
h2 { margin: 32px 0 12px; font-size: 19px; color: #cbd5e1; }
.sub { color: #94a3b8; margin-bottom: 24px; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
         gap: 12px; }
.card { background: #171a21; border: 1px solid #262b36; border-radius: 12px;
        padding: 14px 16px; }
.card .num { font-size: 26px; font-weight: 700; color: #4ade80; }
.card .lbl { font-size: 12.5px; color: #94a3b8; }
.exgrid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
          gap: 16px; }
.ex { background: #171a21; border: 1px solid #262b36; border-radius: 12px;
      padding: 16px; }
.ex h3 { margin: 0 0 8px; font-size: 16.5px; text-transform: capitalize; }
.chart { width: 100%; height: auto; background: #12141a; border-radius: 8px;
         margin: 6px 0; }
.axis { fill: #64748b; font-size: 11px; }
.chip { display: inline-block; background: #1e2530; border-radius: 999px;
        padding: 3px 10px; margin: 2px 4px 2px 0; font-size: 12.5px;
        color: #cbd5e1; }
.chip b { color: #4ade80; }
.fault { color: #f87171; }
.small { font-size: 12.5px; color: #94a3b8; }
table { width: 100%; border-collapse: collapse; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid #262b36;
         font-size: 13.5px; }
th { color: #94a3b8; font-weight: 600; }
.empty { text-align: center; padding: 80px 20px; color: #94a3b8; }
.empty .big { font-size: 44px; }
"""


def _fmt_dur(seconds: float) -> str:
    m = int(seconds // 60)
    return f"{m // 60}h {m % 60}m" if m >= 60 else f"{m}m"


def render_html(agg: dict, log_path: str = "", refresh: bool = True) -> str:
    t = agg["totals"]
    meta = '<meta http-equiv="refresh" content="60">' if refresh else ""
    head = (f'<!doctype html><html><head><meta charset="utf-8">{meta}'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>AI Gym Coach - Progress</title>'
            f'<style>{_CSS}</style></head><body>')
    if not t["sessions"]:
        return (head + '<div class="empty"><div class="big">&#127947;</div>'
                "<h1>No workouts logged yet</h1>"
                "<p>Finish a session with <code>pose_coach.py</code> and "
                "refresh this page.</p></div></body></html>")

    cards = [
        (t["sessions"], "sessions"),
        (t["reps"], "total reps"),
        (_fmt_dur(t["duration_s"]), "training time"),
        (t["active_days"], "active days"),
        (f'{t["streak"]}d', "current streak"),
        (f'{t["longest_streak"]}d', "longest streak"),
    ]
    if t["best_score"] is not None:
        cards.append((t["best_score"], "best rep score"))
    if t["hold_s"]:
        cards.append((_fmt_dur(t["hold_s"]), "plank time"))
    cards_html = "".join(
        f'<div class="card"><div class="num">{v}</div>'
        f'<div class="lbl">{k}</div></div>' for v, k in cards)

    weekly = agg["weekly"]
    week_pts = [{"label": f'{w["label"]} ({w["sessions"]} session(s))',
                 "value": w["reps"]} for w in weekly]
    weekly_html = ""
    if any(p["value"] for p in week_pts):
        weekly_html = ("<h2>Weekly volume (reps)</h2>"
                       + svg_bars(week_pts, height=150)
                       + f'<div class="small">last {len(weekly)} training '
                         f'week(s) &mdash; hover a bar for details</div>')

    ex_cards = []
    for ex in sorted(agg["exercises"]):
        e = agg["exercises"][ex]
        prs = e["prs"]
        chips = [f'<span class="chip">{e["sessions"]} session(s)</span>']
        if e["total_reps"]:
            chips.append(f'<span class="chip"><b>{e["total_reps"]}</b> reps</span>')
        if "max_reps_session" in prs:
            chips.append(f'<span class="chip">PR <b>{prs["max_reps_session"]}'
                         f'</b> reps/session</span>')
        if "best_avg_score" in prs:
            chips.append(f'<span class="chip">PR <b>{prs["best_avg_score"]}'
                         f'</b> avg score</span>')
        if "longest_hold_s" in prs:
            chips.append(f'<span class="chip">PR <b>{prs["longest_hold_s"]:g}s'
                         f'</b> hold</span>')
        body = ""
        if len(e["scores"]) > 1:
            body += ('<div class="small">avg form score per session</div>'
                     + svg_line(e["scores"]))
        if len(e["reps_series"]) > 1:
            body += ('<div class="small">reps per session</div>'
                     + svg_bars(e["reps_series"]))
        if len(e["holds"]) > 1:
            body += ('<div class="small">hold time per session (s)</div>'
                     + svg_line(e["holds"], color=ACCENT2))
        if e["faults"]:
            top = sorted(e["faults"].items(), key=lambda kv: -kv[1])[:6]
            body += ('<div class="small">most common faults</div><div>'
                     + "".join(f'<span class="chip fault">{html.escape(k)} '
                               f'&times;{v}</span>' for k, v in top)
                     + "</div>")
        ex_cards.append(f'<div class="ex"><h3>{html.escape(ex)}</h3>'
                        f'<div>{"".join(chips)}</div>{body}</div>')

    def _reps_cell(r: dict) -> str:
        return f'{r["hold_s"]}s hold' if r["hold_s"] else str(r["reps"])

    rows = "".join(
        f'<tr><td>{html.escape(str(r["date"]))}</td>'
        f'<td>{html.escape(r["exercise"])}</td>'
        f'<td>{_reps_cell(r)}</td>'
        f'<td>{r["score"] if r["score"] is not None else "&mdash;"}</td>'
        f'<td class="fault">{html.escape(r["top_fault"] or "")}</td></tr>'
        for r in agg["recent"])

    footer = (f'<p class="small">generated '
              f'{dt.datetime.now().strftime("%Y-%m-%d %H:%M")} from '
              f'{html.escape(log_path or "workout_log.json")} &mdash; '
              f'page refreshes automatically</p>')
    return (head
            + "<h1>&#127947; AI Gym Coach &mdash; Progress</h1>"
            + f'<div class="sub">last workout: {t["last_day"] or "?"}</div>'
            + f'<div class="cards">{cards_html}</div>'
            + weekly_html
            + "<h2>Exercises</h2>"
            + f'<div class="exgrid">{"".join(ex_cards)}</div>'
            + "<h2>Recent sessions</h2>"
            + f'<table><tr><th>started</th><th>exercise</th><th>reps</th>'
              f'<th>avg score</th><th>top fault</th></tr>{rows}</table>'
            + footer + "</body></html>")


# ---------------------------------------------------------------- server
class _Handler(BaseHTTPRequestHandler):
    log_path = DEFAULT_LOG

    def do_GET(self):  # noqa: N802 (stdlib API name)
        if self.path.split("?")[0] == "/data.json":
            payload = json.dumps(aggregate(load_history(self.log_path)),
                                 default=str).encode()
            ctype = "application/json"
        elif self.path.split("?")[0] == "/":
            payload = render_html(aggregate(load_history(self.log_path)),
                                  self.log_path).encode()
            ctype = "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):  # keep the console quiet
        pass


def serve(log_path: str, host: str = "127.0.0.1", port: int = DEFAULT_PORT,
          open_browser: bool = True) -> None:
    handler = type("Handler", (_Handler,), {"log_path": log_path})
    srv = ThreadingHTTPServer((host, port), handler)
    url = f"http://{'localhost' if host in ('0.0.0.0', '127.0.0.1') else host}:{srv.server_address[1]}/"
    print(f"Dashboard on {url}  (log: {log_path}) — Ctrl+C to stop")
    if open_browser:
        threading.Timer(0.4, webbrowser.open, (url,)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        srv.server_close()


# --------------------------------------------------------------- selftest
def _fake_history() -> list[dict]:
    def sess(day, ex, scores, faults=None, plank=None):
        reps = [{"n": i + 1, "score": sc, "eccentric_s": 1.2,
                 "concentric_s": 1.0, "min_angle": 80.0, "velocity": 30.0,
                 "similarity": None, "faults": []} for i, sc in enumerate(scores)]
        s = {"started": f"{day} 18:00:00", "exercise": ex, "reps": reps,
             "plank": plank, "duration_s": 300.0,
             "summary": {"reps": len(reps),
                         "avg_score": round(sum(scores) / len(scores), 1)
                         if scores else None,
                         "avg_concentric_s": 1.0, "avg_similarity": None,
                         "fault_counts": faults or {},
                         "velocity_loss_pct": None}}
        return s

    return [
        sess("2026-07-06", "squat", [70, 75, 80], {"shallow": 2}),
        sess("2026-07-07", "squat", [80, 85, 90], {"shallow": 1, "knees_in": 1}),
        sess("2026-07-08", "pushup", [88, 92]),
        sess("2026-07-08", "plank", [], plank={"total_hold_s": 45.0,
                                               "best_streak_s": 30.0}),
        sess("2026-07-10", "squat", [85, 90, 95]),
    ]


def selftest() -> None:
    print("== coach_dashboard selftests ==")

    # 1 — aggregation totals, PRs, faults, weekly buckets, streaks
    agg = aggregate(_fake_history())
    t = agg["totals"]
    assert t["sessions"] == 5 and t["reps"] == 11, t
    assert t["best_score"] == 95 and t["hold_s"] == 45.0, t
    assert t["active_days"] == 4 and t["streak"] == 1, t   # gap before 07-10
    assert t["longest_streak"] == 3, t                      # 06,07,08
    sq = agg["exercises"]["squat"]
    assert sq["prs"]["max_reps_session"] == 3
    assert sq["prs"]["best_avg_score"] == 90.0
    assert sq["faults"] == {"shallow": 3, "knees_in": 1}
    assert agg["exercises"]["plank"]["prs"]["longest_hold_s"] == 45.0
    assert len(agg["weekly"]) == 1 and agg["weekly"][0]["reps"] == 11
    assert agg["recent"][0]["exercise"] == "squat"          # newest first
    print("ok 1 — aggregation (totals, PRs, faults, weekly, streaks)")

    # 2 — HTML rendering: charts + names present; empty log friendly
    page = render_html(agg, "workout_log.json")
    for marker in ("squat", "pushup", "plank", "<svg", "polyline", "<rect",
                   "current streak", "Recent sessions"):
        assert marker in page, marker
    empty = render_html(aggregate([]))
    assert "No workouts logged yet" in empty
    print("ok 2 — HTML rendering (charts, cards, empty state)")

    # 3 — malformed rows tolerated
    weird = [{"started": "not a date", "exercise": "", "reps": []},
             {"exercise": "squat"}, {}]
    agg2 = aggregate(weird)
    assert agg2["totals"]["sessions"] == 3
    assert agg2["totals"]["streak"] == 0
    assert render_html(agg2)                                # doesn't crash
    assert load_history(os.path.join(HERE, "_no_such_file_.json")) == []
    print("ok 3 — malformed/missing data tolerated")

    # 4 — HTTP server serves the page and the JSON API
    import tempfile
    import urllib.request
    with tempfile.TemporaryDirectory() as td:
        log = os.path.join(td, "log.json")
        with open(log, "w", encoding="utf-8") as fh:
            json.dump(_fake_history(), fh)
        handler = type("H", (_Handler,), {"log_path": log})
        srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        try:
            with urllib.request.urlopen(base + "/", timeout=5) as r:
                assert r.status == 200 and b"Progress" in r.read()
            with urllib.request.urlopen(base + "/data.json", timeout=5) as r:
                data = json.loads(r.read())
                assert data["totals"]["sessions"] == 5
            try:
                urllib.request.urlopen(base + "/nope", timeout=5)
                raise AssertionError("expected 404")
            except urllib.error.HTTPError as e:
                assert e.code == 404
        finally:
            srv.shutdown()
            srv.server_close()
    print("ok 4 — HTTP server (page, /data.json, 404)")
    print("All dashboard selftests passed.")


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--log", default=DEFAULT_LOG,
                    help="workout log file (default: %(default)s)")
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (0.0.0.0 for Docker)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--export", metavar="FILE",
                    help="write a static HTML report and exit")
    ap.add_argument("--no-browser", action="store_true",
                    help="don't open the browser automatically")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if args.export:
        page = render_html(aggregate(load_history(args.log)), args.log,
                           refresh=False)
        with open(args.export, "w", encoding="utf-8") as fh:
            fh.write(page)
        print(f"Wrote {args.export}")
        return
    serve(args.log, args.host, args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
