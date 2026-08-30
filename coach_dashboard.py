"""Progress dashboard for the AI Gym Coach — a local web page with charts.

Reads workout_log.json (written by pose_coach.py after every session) and
serves a self-contained HTML page: a spotlight on your last session (every
rep's score and faults, tempo, golden-rep similarity, fatigue, heart rate),
training volume per week, a training-days heatmap, per-exercise form-score
and rep trends, personal records, fault breakdowns and a session table.
Charts are server-side SVG — no JavaScript frameworks, no CDN, no internet.
Standard library only. Follows the OS light/dark theme.

Usage:
    python coach_dashboard.py                      # serve http://localhost:7788
    python coach_dashboard.py --port 9000
    python coach_dashboard.py --export report.html # write a static file
    python coach_dashboard.py --demo               # preview with sample data
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

# A corporate HTTP_PROXY would otherwise capture the selftest's own
# localhost requests (403 from the proxy). Same rule as coach_ops.local_no_proxy.
for _key in ("no_proxy", "NO_PROXY"):
    _cur = os.environ.get(_key, "")
    _have = {h.strip() for h in _cur.split(",") if h.strip()}
    _add = [h for h in ("localhost", "127.0.0.1", "::1")
            if h not in _have and "*" not in _have]
    if _add:
        os.environ[_key] = ",".join([_cur] + _add if _cur else _add)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOG = os.environ.get("COACH_LOG",
                             os.path.join(HERE, "workout_log.json"))
DEFAULT_PORT = 7788

ACCENT = "#4ade80"      # green
ACCENT2 = "#60a5fa"     # blue
WARN = "#f87171"        # red
AMBER = "#fbbf24"

# human names for the log's fault keys (kept in sync with coach_hud.py;
# duplicated so the dashboard stays standard-library only)
FAULT_NAMES = {
    "back_lean": "back leaning", "back_round": "rounded back",
    "body_sag": "hips sagging", "knees_cave": "knees caving in",
    "knees_in": "knees caving in", "shallow": "not deep enough",
    "elbow_swing": "elbows swinging", "elbow_flare": "elbows flaring",
    "swing": "elbows swinging", "torso_lean": "torso leaning",
    "lean_back": "leaning back", "uneven": "uneven sides",
    "chin": "chin under bar", "shrug_neck": "neck shrugged",
    "too_fast": "too fast",
}
EXERCISE_NAMES = {"pushup": "push-up", "pullup": "pull-up",
                  "shoulder_press": "shoulder press", "bench": "bench press",
                  "curl": "bicep curl"}


def fault_name(key: str) -> str:
    return FAULT_NAMES.get(key, key.replace("_", " "))


def exercise_name(key: str) -> str:
    return EXERCISE_NAMES.get(key, key.replace("_", " "))


def score_color(score) -> str:
    if score is None:
        return "#94a3b8"
    return ACCENT if score >= 85 else AMBER if score >= 65 else WARN


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


def _mean(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 1) if xs else None


def session_detail(s: dict) -> dict:
    """One session, ready to display: per-rep rows + what the set taught."""
    summ = s.get("summary") or {}
    reps = s.get("reps") or []
    plank = s.get("plank") or None
    date = s.get("started", "?")
    return {
        "date": date, "exercise": s.get("exercise") or "?",
        "duration_s": s.get("duration_s") or 0.0,
        "reps": [{"n": r.get("n", i + 1), "score": r.get("score"),
                  "faults": [fault_name(f) for f in r.get("faults") or []],
                  "eccentric_s": r.get("eccentric_s"),
                  "concentric_s": r.get("concentric_s"),
                  "velocity": r.get("velocity"),
                  "similarity": r.get("similarity")}
                 for i, r in enumerate(reps)],
        "count": summ.get("reps") or len(reps),
        "avg_score": summ.get("avg_score"),
        "best_score": max((r.get("score") for r in reps
                           if r.get("score") is not None), default=None),
        "avg_ecc_s": _mean(r.get("eccentric_s") for r in reps),
        "avg_con_s": summ.get("avg_concentric_s"),
        "avg_similarity": summ.get("avg_similarity"),
        "velocity_loss_pct": summ.get("velocity_loss_pct"),
        "avg_hr": summ.get("avg_hr"), "peak_hr": summ.get("peak_hr"),
        "fault_counts": {fault_name(k): v for k, v in
                         (summ.get("fault_counts") or {}).items()},
        "plank": plank,
    }


def aggregate(history: list[dict]) -> dict:
    """Distill the raw session log into everything the page shows."""
    totals = {"sessions": len(history), "reps": 0, "duration_s": 0.0,
              "best_score": None, "hold_s": 0.0}
    weekly: dict[str, dict] = {}          # "2026-W28" -> {sessions, reps}
    exercises: dict[str, dict] = {}
    days: list[dt.date] = []
    day_reps: dict[str, int] = {}
    recent: list[dict] = []
    scores_by_session: list[float] = []
    sims: list[float] = []
    hrs: list[float] = []

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
        if score is not None:
            scores_by_session.append(score)
        if summ.get("avg_similarity") is not None:
            sims.append(summ["avg_similarity"])
        if summ.get("avg_hr"):
            hrs.append(summ["avg_hr"])

        if date:
            days.append(date)
            iso = date.isocalendar()
            wk = f"{iso[0]}-W{iso[1]:02d}"
            w = weekly.setdefault(wk, {"sessions": 0, "reps": 0, "hold_s": 0.0})
            w["sessions"] += 1
            w["reps"] += reps
            w["hold_s"] += hold
            key = date.isoformat()
            day_reps[key] = day_reps.get(key, 0) + reps + (1 if hold else 0)

        e = exercises.setdefault(ex, {
            "sessions": 0, "total_reps": 0, "scores": [], "reps_series": [],
            "holds": [], "faults": {}, "prs": {}, "similarity": [],
            "tempo": []})
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
        if summ.get("avg_similarity") is not None:
            e["similarity"].append({"label": label,
                                    "value": summ["avg_similarity"]})
        if summ.get("avg_concentric_s") is not None:
            e["tempo"].append({"label": label,
                               "value": summ["avg_concentric_s"]})
        for k, v in (summ.get("fault_counts") or {}).items():
            e["faults"][k] = e["faults"].get(k, 0) + v

        top_fault = max((summ.get("fault_counts") or {}).items(),
                        key=lambda kv: kv[1], default=(None, 0))[0]
        recent.append({"date": s.get("started", "?"), "exercise": ex,
                       "reps": reps, "score": score,
                       "hold_s": round(hold, 1) if hold else None,
                       "top_fault": top_fault,
                       "tempo_s": summ.get("avg_concentric_s"),
                       "similarity": summ.get("avg_similarity"),
                       "velocity_loss_pct": summ.get("velocity_loss_pct"),
                       "avg_hr": summ.get("avg_hr")})

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
    # form trend: last 5 scored sessions vs the 5 before
    last5, prev5 = scores_by_session[-5:], scores_by_session[-10:-5]
    totals["recent_avg_score"] = _mean(last5)
    totals["score_delta"] = (round(_mean(last5) - _mean(prev5), 1)
                             if last5 and prev5 else None)
    totals["avg_similarity"] = _mean(sims)
    totals["avg_hr"] = round(_mean(hrs)) if hrs else None

    weeks = sorted(weekly)[-12:]
    return {
        "totals": totals,
        "weekly": [{"label": w, **weekly[w]} for w in weeks],
        "exercises": exercises,
        "recent": list(reversed(recent))[:10],
        "day_reps": day_reps,
        "last_session": session_detail(history[-1]) if history else None,
    }


# ------------------------------------------------------------- SVG charts
def _scale(values: list[float], lo: float, hi: float,
           out_lo: float, out_hi: float) -> list[float]:
    span = (hi - lo) or 1.0
    return [out_lo + (v - lo) / span * (out_hi - out_lo) for v in values]


def svg_line(points: list[dict], width=560, height=130, color=ACCENT,
             lo: float | None = None, hi: float | None = None,
             unit: str = "") -> str:
    """Line chart with dots + native tooltips from [{label, value}].

    A fixed lo/hi (e.g. 0-100 for scores) keeps day-to-day noise from
    looking like a cliff."""
    if not points:
        return ""
    vals = [p["value"] for p in points]
    lo = min(vals) if lo is None else lo
    hi = max(vals) if hi is None else hi
    pad = 12
    if len(points) == 1:
        xs = [width / 2]
    else:
        xs = _scale(list(range(len(vals))), 0, len(vals) - 1,
                    pad + 24, width - pad)
    ys = _scale(vals, lo, hi, height - pad, pad)
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    dots = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}">'
        f'<title>{html.escape(str(p["label"]))}: {p["value"]:g}{unit}</title>'
        f'</circle>' for x, y, p in zip(xs, ys, points))
    line = (f'<polyline points="{pts}" fill="none" stroke="{color}" '
            f'stroke-width="2"/>' if len(points) > 1 else "")
    grid = "".join(
        f'<line x1="{pad + 24}" y1="{y:.1f}" x2="{width - pad}" y2="{y:.1f}" '
        f'class="grid"/>' for y in _scale([lo, (lo + hi) / 2, hi], lo, hi,
                                         height - pad, pad))
    return (f'<svg viewBox="0 0 {width} {height}" class="chart">'
            f'{grid}{line}{dots}'
            f'<text x="4" y="{pad + 4}" class="axis">{hi:g}</text>'
            f'<text x="4" y="{height - pad + 4}" class="axis">{lo:g}</text></svg>')


def svg_bars(points: list[dict], width=560, height=130, color=ACCENT2,
             colors: list[str] | None = None, unit: str = "",
             hi: float | None = None) -> str:
    """Bar chart with native tooltips from [{label, value}]."""
    if not points:
        return ""
    vals = [p["value"] for p in points]
    hi = (max(vals) or 1) if hi is None else hi
    pad = 10
    n = len(points)
    slot = (width - 2 * pad - 24) / n
    bw = max(3.0, min(slot * 0.65, 40))
    bars = []
    for i, p in enumerate(points):
        h = (p["value"] / hi) * (height - 2 * pad)
        x = pad + 24 + i * slot + (slot - bw) / 2
        y = height - pad - h
        col = colors[i] if colors else color
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" '
            f'height="{max(h, 1):.1f}" rx="2" fill="{col}">'
            f'<title>{html.escape(str(p["label"]))}: {p["value"]:g}{unit}</title>'
            f'</rect>')
    return (f'<svg viewBox="0 0 {width} {height}" class="chart">'
            f'{"".join(bars)}'
            f'<text x="4" y="{pad + 4}" class="axis">{hi:g}</text></svg>')


def svg_heatmap(day_reps: dict[str, int], weeks: int = 16,
                today: dt.date | None = None) -> str:
    """GitHub-style training-days grid for the last `weeks` weeks."""
    today = today or dt.date.today()
    start = today - dt.timedelta(days=today.weekday() + 7 * (weeks - 1))
    cell, gap = 13, 3
    hi = max(day_reps.values(), default=1) or 1
    cells = []
    for w in range(weeks):
        for d in range(7):
            day = start + dt.timedelta(days=w * 7 + d)
            if day > today:
                continue
            v = day_reps.get(day.isoformat(), 0)
            op = 0.15 if not v else 0.35 + 0.65 * min(1.0, v / hi)
            cells.append(
                f'<rect x="{w * (cell + gap)}" y="{d * (cell + gap)}" width="{cell}" '
                f'height="{cell}" rx="3" fill="{ACCENT}" fill-opacity="{op:.2f}">'
                f'<title>{day.isoformat()}: {v} reps</title></rect>')
    width = weeks * (cell + gap)
    height = 7 * (cell + gap)
    return (f'<svg viewBox="0 0 {width} {height}" class="heat" '
            f'style="max-width:{width * 1.6}px">{"".join(cells)}</svg>')


# ---------------------------------------------------------------- page
_CSS = """
:root { color-scheme: light dark;
  --bg: #f6f7f9; --card: #ffffff; --line: #e3e6ea; --text: #17202a;
  --muted: #5b6b7c; --dim: #8a97a5; --chart: #f1f3f6; --grid: #e3e6ea; }
@media (prefers-color-scheme: dark) { :root {
  --bg: #0f1115; --card: #171a21; --line: #262b36; --text: #e5e7eb;
  --muted: #94a3b8; --dim: #64748b; --chart: #12141a; --grid: #232833; } }
* { box-sizing: border-box; }
body { margin: 0; padding: 24px; background: var(--bg); color: var(--text);
       font: 15px/1.5 system-ui, "Segoe UI", sans-serif; max-width: 1240px;
       margin-inline: auto; }
h1 { margin: 0 0 4px; font-size: 26px; }
h2 { margin: 32px 0 12px; font-size: 19px; color: var(--text); }
h2 small { font-weight: 400; color: var(--muted); font-size: 13px; margin-left: 8px; }
.sub { color: var(--muted); margin-bottom: 24px; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
         gap: 12px; }
.card, .ex, .spot { background: var(--card); border: 1px solid var(--line);
        border-radius: 12px; padding: 14px 16px; }
.card .num { font-size: 26px; font-weight: 700; color: #4ade80; }
.card .num.blue { color: #60a5fa; } .card .num.amber { color: #fbbf24; }
.card .num.red { color: #f87171; }
.card .lbl { font-size: 12.5px; color: var(--muted); }
.card .delta { font-size: 12px; margin-left: 6px; }
.up { color: #4ade80; } .down { color: #f87171; }
.exgrid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
          gap: 16px; }
.ex h3 { margin: 0 0 8px; font-size: 16.5px; text-transform: capitalize; }
.chart { width: 100%; height: auto; background: var(--chart); border-radius: 8px;
         margin: 6px 0; }
.heat { height: auto; margin: 6px 0; }
.axis { fill: var(--dim); font-size: 11px; }
.grid { stroke: var(--grid); stroke-width: 1; }
.chip { display: inline-block; background: var(--chart); border: 1px solid var(--line);
        border-radius: 999px; padding: 3px 10px; margin: 2px 4px 2px 0;
        font-size: 12.5px; color: var(--text); }
.chip b { color: #4ade80; }
.fault { color: #f87171; }
.small { font-size: 12.5px; color: var(--muted); }
.spot { padding: 18px 20px; }
.spot .head { display: flex; flex-wrap: wrap; gap: 8px 24px; align-items: baseline; }
.spot .head .ex-name { font-size: 20px; font-weight: 700; text-transform: capitalize; }
.stats { display: flex; flex-wrap: wrap; gap: 8px 28px; margin: 12px 0 4px; }
.stat .v { font-size: 22px; font-weight: 700; }
.stat .l { font-size: 12px; color: var(--muted); }
.legend { font-size: 12px; color: var(--muted); }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px;
            margin: 0 4px 0 10px; vertical-align: middle; }
table { width: 100%; border-collapse: collapse; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--line);
         font-size: 13.5px; white-space: nowrap; }
th { color: var(--muted); font-weight: 600; }
.tablewrap { overflow-x: auto; }
.empty { text-align: center; padding: 60px 20px 30px; color: var(--muted); }
.empty .big { font-size: 44px; }
.howto { max-width: 560px; margin: 0 auto; text-align: left; }
code { background: var(--chart); border: 1px solid var(--line); border-radius: 5px;
       padding: 1px 6px; font-size: 13px; }
footer { margin-top: 32px; }
"""


def _fmt_dur(seconds: float) -> str:
    m = int(seconds // 60)
    return f"{m // 60}h {m % 60}m" if m >= 60 else f"{m}m"


_HOWTO = """
<div class="howto">
<p><b>Start a set</b> — <code>python pose_coach.py --exercise squat</code>
(or <code>--exercise auto</code> to let it recognise the movement). Press
<code>h</code> in the video window for the keys.</p>
<p><b>Then</b> — <code>--coach</code> to talk to the coach,
<code>--program "squat 3x10 rest 90, plank 2x40s"</code> for a guided workout,
<code>--record-reference</code> to save a golden rep,
<code>--sensors ble</code> for a heart-rate strap.</p>
<p>Every finished set lands here. <code>python coach_dashboard.py --demo</code>
previews the page with sample data.</p>
</div>"""


def _spotlight(ls: dict) -> str:
    """The last session, rep by rep — the features made visible."""
    ex = html.escape(exercise_name(ls["exercise"]))
    head = (f'<div class="head"><span class="ex-name">{ex}</span>'
            f'<span class="small">{html.escape(str(ls["date"]))} · '
            f'{_fmt_dur(ls["duration_s"])}</span></div>')
    stats = []
    if ls["plank"]:
        p = ls["plank"]
        stats.append((f'{p.get("total_hold_s", 0):g}s', "held", ACCENT))
        stats.append((f'{p.get("best_streak_s", 0):g}s', "best unbroken", ACCENT2))
    else:
        stats.append((str(ls["count"]), "reps", "var(--text)"))
        sc = ls["avg_score"]
        stats.append(("—" if sc is None else f"{sc:g}", "avg score", score_color(sc)))
        if ls["best_score"] is not None:
            stats.append((str(ls["best_score"]), "best rep", ACCENT))
        if ls["avg_ecc_s"] is not None and ls["avg_con_s"] is not None:
            stats.append((f'↓{ls["avg_ecc_s"]:.1f}s ↑{ls["avg_con_s"]:.1f}s',
                          "tempo (down / up)", ACCENT2))
        if ls["avg_similarity"] is not None:
            stats.append((f'{ls["avg_similarity"]:g}%', "vs golden rep", AMBER))
        vl = ls["velocity_loss_pct"]
        if vl is not None:
            stats.append((f"-{vl:g}%", "speed loss (fatigue)" if vl > 20
                          else "speed loss", WARN if vl > 20 else ACCENT))
    if ls["avg_hr"]:
        stats.append((f'{ls["avg_hr"]}', f'avg bpm · peak {ls["peak_hr"]}', WARN))
    stats_html = "".join(
        f'<div class="stat"><div class="v" style="color:{c}">{html.escape(v)}</div>'
        f'<div class="l">{html.escape(l)}</div></div>' for v, l, c in stats)
    body = ""
    reps = [r for r in ls["reps"] if r["score"] is not None]
    if reps:
        pts = [{"label": f'rep {r["n"]}' + (" — " + ", ".join(r["faults"])
                                            if r["faults"] else " — clean")
                        + (f' · ↓{r["eccentric_s"]:.1f}s ↑{r["concentric_s"]:.1f}s'
                           if r["eccentric_s"] is not None else ""),
                "value": r["score"]} for r in reps]
        body += ('<div class="small">score per rep — hover a bar for its faults '
                 'and tempo</div>'
                 + svg_bars(pts, height=120, colors=[score_color(r["score"])
                                                     for r in reps], hi=100)
                 + '<div class="legend">'
                 f'<i style="background:{ACCENT}"></i>85+ clean'
                 f'<i style="background:{AMBER}"></i>65–84 one fault'
                 f'<i style="background:{WARN}"></i>below 65</div>')
    if ls["fault_counts"]:
        top = sorted(ls["fault_counts"].items(), key=lambda kv: -kv[1])
        body += ('<div class="small" style="margin-top:8px">what to work on</div><div>'
                 + "".join(f'<span class="chip fault">{html.escape(k)} ×{v}</span>'
                           for k, v in top) + "</div>")
    elif not ls["plank"] and ls["count"]:
        body += '<div class="small" style="margin-top:8px">no form faults — clean set</div>'
    if not ls["count"] and not ls["plank"]:
        body += ('<div class="small" style="margin-top:8px">No reps were counted '
                 'in this session — stand side-on with your whole body in view; '
                 'the gauge on the left of the video window shows the live '
                 'angle against the rep thresholds.</div>')
    return f'<div class="spot">{head}<div class="stats">{stats_html}</div>{body}</div>'


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
                "refresh this page.</p></div>" + _HOWTO + "</body></html>")

    cards = [
        (t["sessions"], "sessions", ""),
        (t["reps"], "total reps", ""),
        (_fmt_dur(t["duration_s"]), "training time", "blue"),
        (t["active_days"], "active days", "blue"),
        (f'{t["streak"]}d', "current streak", "amber"),
        (f'{t["longest_streak"]}d', "longest streak", "amber"),
    ]
    if t["recent_avg_score"] is not None:
        d = t["score_delta"]
        delta = ""
        if d is not None:
            cls = "up" if d >= 0 else "down"
            delta = f'<span class="delta {cls}">{"+" if d >= 0 else ""}{d:g}</span>'
        cards.append((f'{t["recent_avg_score"]:g}{delta}', "form score, last 5 sessions",
                      ""))
    if t["best_score"] is not None:
        cards.append((t["best_score"], "best rep score", ""))
    if t["avg_similarity"] is not None:
        cards.append((f'{t["avg_similarity"]:g}%', "avg vs golden rep", "amber"))
    if t["avg_hr"]:
        cards.append((t["avg_hr"], "avg heart rate", "red"))
    if t["hold_s"]:
        cards.append((_fmt_dur(t["hold_s"]), "plank time", "blue"))
    cards_html = "".join(
        f'<div class="card"><div class="num {cls}">{v}</div>'
        f'<div class="lbl">{k}</div></div>' for v, k, cls in cards)

    spot = _spotlight(agg["last_session"]) if agg.get("last_session") else ""

    weekly = agg["weekly"]
    week_pts = [{"label": f'{w["label"]} ({w["sessions"]} session(s))',
                 "value": w["reps"]} for w in weekly]
    weekly_html = ""
    if any(p["value"] for p in week_pts):
        weekly_html = ("<h2>Weekly volume <small>reps per week</small></h2>"
                       + svg_bars(week_pts, height=150)
                       + f'<div class="small">last {len(weekly)} training '
                         f'week(s) &mdash; hover a bar for details</div>')
    heat_html = ""
    if agg.get("day_reps"):
        heat_html = ("<h2>Training days <small>last 16 weeks, darker = more reps"
                     "</small></h2>" + svg_heatmap(agg["day_reps"]))

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
            chips.append(f'<span class="chip">PR <b>{prs["best_avg_score"]:g}'
                         f'</b> avg score</span>')
        if "longest_hold_s" in prs:
            chips.append(f'<span class="chip">PR <b>{prs["longest_hold_s"]:g}s'
                         f'</b> hold</span>')
        body = ""
        if len(e["scores"]) > 1:
            body += ('<div class="small">avg form score per session (0–100)</div>'
                     + svg_line(e["scores"], lo=0, hi=100))
        if len(e["reps_series"]) > 1:
            body += ('<div class="small">reps per session</div>'
                     + svg_bars(e["reps_series"]))
        if len(e["similarity"]) > 1:
            body += ('<div class="small">similarity to your golden rep (%)</div>'
                     + svg_line(e["similarity"], color=AMBER, lo=0, hi=100, unit="%"))
        if len(e["tempo"]) > 1:
            body += ('<div class="small">avg lift time per session (s)</div>'
                     + svg_line(e["tempo"], color=ACCENT2, lo=0, unit="s"))
        if len(e["holds"]) > 1:
            body += ('<div class="small">hold time per session (s)</div>'
                     + svg_line(e["holds"], color=ACCENT2, lo=0, unit="s"))
        if e["faults"]:
            top = sorted(e["faults"].items(), key=lambda kv: -kv[1])[:6]
            body += ('<div class="small">most common faults</div><div>'
                     + "".join(f'<span class="chip fault">{html.escape(fault_name(k))} '
                               f'&times;{v}</span>' for k, v in top)
                     + "</div>")
        ex_cards.append(f'<div class="ex"><h3>{html.escape(exercise_name(ex))}</h3>'
                        f'<div>{"".join(chips)}</div>{body}</div>')

    def _reps_cell(r: dict) -> str:
        return f'{r["hold_s"]:g}s hold' if r["hold_s"] else str(r["reps"])

    def _num(v, fmt="{:g}", suffix=""):
        return "&mdash;" if v is None else fmt.format(v) + suffix

    rows = "".join(
        f'<tr><td>{html.escape(str(r["date"]))}</td>'
        f'<td>{html.escape(exercise_name(r["exercise"]))}</td>'
        f'<td>{_reps_cell(r)}</td>'
        f'<td style="color:{score_color(r["score"])}">{_num(r["score"])}</td>'
        f'<td>{_num(r["tempo_s"], "{:.1f}", "s")}</td>'
        f'<td>{_num(r["similarity"], "{:g}", "%")}</td>'
        f'<td class="{"fault" if (r["velocity_loss_pct"] or 0) > 20 else ""}">'
        f'{_num(r["velocity_loss_pct"], "-{:g}", "%")}</td>'
        f'<td>{_num(r["avg_hr"])}</td>'
        f'<td class="fault">{html.escape(fault_name(r["top_fault"]) if r["top_fault"] else "")}'
        f'</td></tr>'
        for r in agg["recent"])

    footer = (f'<footer class="small">generated '
              f'{dt.datetime.now().strftime("%Y-%m-%d %H:%M")} from '
              f'{html.escape(log_path or "workout_log.json")} &mdash; '
              f'page refreshes automatically · everything stays on this machine'
              f'</footer>')
    return (head
            + "<h1>&#127947; AI Gym Coach &mdash; Progress</h1>"
            + f'<div class="sub">last workout: {t["last_day"] or "?"}</div>'
            + f'<div class="cards">{cards_html}</div>'
            + ("<h2>Last session <small>rep by rep</small></h2>" + spot if spot else "")
            + weekly_html
            + heat_html
            + "<h2>Exercises</h2>"
            + f'<div class="exgrid">{"".join(ex_cards)}</div>'
            + "<h2>Recent sessions</h2>"
            + f'<div class="tablewrap"><table><tr><th>started</th><th>exercise</th>'
              f'<th>reps</th><th>avg score</th><th>lift time</th><th>golden rep</th>'
              f'<th>speed loss</th><th>avg HR</th><th>top fault</th></tr>{rows}'
              f'</table></div>'
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


# -------------------------------------------------------------- demo data
def demo_history(weeks: int = 8, seed: int = 42) -> list[dict]:
    """Plausible synthetic training history for --demo: several weeks of
    sessions with improving scores, thinning fault counts and a weekly plank.
    Powers the hosted demo dashboard, which must never show real user data
    (docs/INFRA.md §2) — and lets anyone preview the page before training."""
    import random

    rng = random.Random(seed)
    plan = (("squat", ("shallow", "knees_cave")), ("pushup", ("elbow_flare",)),
            ("curl", ("elbow_swing",)), ("deadlift", ("back_round",)),
            ("shoulder_press", ("uneven",)))
    today = dt.date.today()
    history: list[dict] = []
    for day_back in range(weeks * 7, -1, -1):
        date = today - dt.timedelta(days=day_back)
        if rng.random() > 0.55:                      # ~4 training days a week
            continue
        progress = 1 - day_back / (weeks * 7)        # 0 -> 1 over the period
        for ex, faults in rng.sample(plan, k=rng.randint(1, 2)):
            base = 62 + 28 * progress                # scores drift 62 -> 90
            n = rng.randint(6, 12)
            reps = []
            fc: dict[str, int] = {}
            for i in range(n):
                rep_faults = [f for f in faults
                              if rng.random() < 0.35 * (1 - progress) + 0.05]
                if i >= n - 2 and rng.random() < 0.4:
                    rep_faults.append("too_fast")
                penalty = {"shallow": 20, "knees_cave": 25, "elbow_flare": 15,
                           "elbow_swing": 20, "back_round": 30, "uneven": 15,
                           "too_fast": 10}
                sc = max(0, min(100, round(base + 10 + rng.gauss(0, 4)
                                           - sum(penalty[f] for f in rep_faults))))
                for f in rep_faults:
                    fc[f] = fc.get(f, 0) + 1
                reps.append({"n": i + 1, "score": sc,
                             "eccentric_s": round(rng.uniform(1.0, 2.2), 2),
                             "concentric_s": round(rng.uniform(0.8, 1.6), 2),
                             "min_angle": round(rng.uniform(70, 95), 1),
                             "velocity": round(40 - 12 * i / n + rng.uniform(-3, 3), 1),
                             "similarity": round(55 + 35 * progress + rng.uniform(-8, 8))
                             if ex == "squat" else None,
                             "faults": sorted(rep_faults)})
            scores = [r["score"] for r in reps]
            sims = [r["similarity"] for r in reps if r["similarity"] is not None]
            vels = [r["velocity"] for r in reps]
            vloss = round(max(0.0, 1 - (sum(vels[-2:]) / 2) / max(vels[:3])) * 100, 1)
            history.append({
                "started": f"{date.isoformat()} {rng.randint(7, 20):02d}:30:00",
                "exercise": ex, "reps": reps, "plank": None,
                "duration_s": round(rng.uniform(240, 900), 1),
                "summary": {"reps": len(reps),
                            "avg_score": round(sum(scores) / len(scores), 1),
                            "avg_concentric_s": round(
                                sum(r["concentric_s"] for r in reps) / n, 2),
                            "avg_similarity": round(sum(sims) / len(sims), 1)
                            if sims else None,
                            "fault_counts": fc, "velocity_loss_pct": vloss,
                            **({"avg_hr": rng.randint(118, 150),
                                "peak_hr": rng.randint(155, 178)}
                               if progress > 0.5 else {})}})
        if date.isoweekday() == 6:                   # Saturday plank habit
            hold = round(30 + 60 * progress + rng.uniform(-5, 5), 1)
            history.append({
                "started": f"{date.isoformat()} 10:00:00", "exercise": "plank",
                "reps": [], "plank": {"total_hold_s": hold,
                                      "best_streak_s": round(hold * 0.7, 1)},
                "duration_s": hold + 30,
                "summary": {"reps": 0, "avg_score": None,
                            "avg_concentric_s": None, "avg_similarity": None,
                            "fault_counts": {}, "velocity_loss_pct": None}})
    return history


# --------------------------------------------------------------- selftest
def _fake_history() -> list[dict]:
    def sess(day, ex, scores, faults=None, plank=None, extra=None):
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
                         "velocity_loss_pct": None, **(extra or {})}}
        return s

    return [
        sess("2026-07-06", "squat", [70, 75, 80], {"shallow": 2}),
        sess("2026-07-07", "squat", [80, 85, 90], {"shallow": 1, "knees_cave": 1}),
        sess("2026-07-08", "pushup", [88, 92]),
        sess("2026-07-08", "plank", [], plank={"total_hold_s": 45.0,
                                               "best_streak_s": 30.0}),
        sess("2026-07-10", "squat", [85, 90, 95], {"too_fast": 1},
             extra={"avg_similarity": 71.0, "velocity_loss_pct": 24.0,
                    "avg_hr": 140, "peak_hr": 165}),
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
    assert sq["faults"] == {"shallow": 3, "knees_cave": 1, "too_fast": 1}
    assert agg["exercises"]["plank"]["prs"]["longest_hold_s"] == 45.0
    assert len(agg["weekly"]) == 1 and agg["weekly"][0]["reps"] == 11
    assert agg["recent"][0]["exercise"] == "squat"          # newest first
    assert t["avg_similarity"] == 71.0 and t["avg_hr"] == 140, t
    assert t["recent_avg_score"] == 85.0 and t["score_delta"] is None, t
    print("ok 1 — aggregation (totals, PRs, faults, weekly, streaks, trend)")

    # 2 — last-session spotlight carries the per-rep features
    ls = agg["last_session"]
    assert ls["exercise"] == "squat" and ls["count"] == 3 and ls["best_score"] == 95
    assert ls["avg_similarity"] == 71.0 and ls["velocity_loss_pct"] == 24.0
    assert ls["avg_hr"] == 140 and ls["fault_counts"] == {"too fast": 1}, ls
    assert [r["score"] for r in ls["reps"]] == [85, 90, 95]
    assert agg["day_reps"]["2026-07-08"] == 3      # 2 push-up reps + a plank
    print("ok 2 — last-session detail (reps, similarity, fatigue, HR, faults)")

    # 3 — HTML rendering: charts + names present; empty log friendly
    page = render_html(agg, "workout_log.json")
    for marker in ("squat", "push-up", "plank", "<svg", "polyline", "<rect",
                   "current streak", "Recent sessions", "Last session",
                   "score per rep", "vs golden rep", "speed loss (fatigue)",
                   "avg bpm", "Training days", "not deep enough",
                   "prefers-color-scheme: dark"):
        assert marker in page, marker
    assert "knees_cave" not in page and "too_fast" not in page  # human names
    empty = render_html(aggregate([]))
    assert "No workouts logged yet" in empty and "--exercise squat" in empty
    print("ok 3 — HTML rendering (spotlight, charts, cards, empty state)")

    # 4 — malformed rows tolerated; a 0-rep session explains itself
    weird = [{"started": "not a date", "exercise": "", "reps": []},
             {"exercise": "squat"}, {}]
    agg2 = aggregate(weird)
    assert agg2["totals"]["sessions"] == 3
    assert agg2["totals"]["streak"] == 0
    assert "No reps were counted" in render_html(agg2)
    assert load_history(os.path.join(HERE, "_no_such_file_.json")) == []
    print("ok 4 — malformed/missing data tolerated, 0-rep guidance")

    # 5 — HTTP server serves the page and the JSON API
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
                assert data["last_session"]["exercise"] == "squat"
            try:
                urllib.request.urlopen(base + "/nope", timeout=5)
                raise AssertionError("expected 404")
            except urllib.error.HTTPError as e:
                assert e.code == 404
        finally:
            srv.shutdown()
            srv.server_close()
    print("ok 5 — HTTP server (page, /data.json, 404)")

    # 6 — demo history: valid schema, aggregates cleanly, shows progress
    demo = demo_history(weeks=8, seed=42)
    assert len(demo) > 20, f"only {len(demo)} demo sessions"
    agg6 = aggregate(demo)
    assert agg6["totals"]["reps"] > 100
    assert "plank" in agg6["exercises"] and len(agg6["weekly"]) >= 6
    sq6 = agg6["exercises"]["squat"]["scores"]
    first, last = sq6[0]["value"], sq6[-1]["value"]
    assert last > first, f"demo scores should trend up ({first} -> {last})"
    assert agg6["exercises"]["squat"]["similarity"], "demo has golden-rep data"
    assert any(r["faults"] for s in demo for r in s["reps"]), "per-rep faults"
    assert render_html(agg6, "demo")                    # renders
    assert demo_history(weeks=8, seed=42) == demo       # deterministic
    print("ok 6 — demo history (schema, trends, deterministic)")
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
    ap.add_argument("--demo", action="store_true",
                    help="serve synthetic demo history instead of a real log "
                         "(what the hosted demo runs — real data never leaves "
                         "your machine)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if args.demo:
        import tempfile
        path = os.path.join(tempfile.gettempdir(), "demo_workout_log.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(demo_history(), fh)
        args.log = path
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
