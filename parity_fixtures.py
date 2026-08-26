"""Cross-platform parity fixtures — one truth, three engines.

The repo claims "thresholds identical to the desktop prototype" in three
places: this Python engine, iOS CoachCore (Swift) and the Android core
(Kotlin) all implement the same rep FSM, plank tracker and rule-based
auto-detector. Nothing used to *test* that they behave identically — the
most likely silent bug in this codebase is the three engines drifting
apart one threshold at a time.

This module is the single source of truth: it generates deterministic
input streams (no RNG — parametric cosine waves), runs them through the
Python engine, and writes inputs + expected outputs to
data/parity_fixtures.json. The committed file is then replayed by all
three platforms in CI:

    python parity_fixtures.py --selftest     # Python re-derives + compares
    swift test  (ParityTests.swift)          # PARITY_FIXTURES env -> replay
    ./gradlew test  (ParityTest.kt)          # PARITY_FIXTURES env -> replay

Regenerate after an intentional engine change (and say so in the commit):

    python parity_fixtures.py --generate

Angles are rounded to 4 decimals BEFORE the Python engine computes the
expectations, so every platform consumes bit-identical doubles from JSON;
timestamps are i/fps, identically computed everywhere. Comparisons use a
1e-4 tolerance — generous against IEEE noise, tiny against any real
threshold drift.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

from pose_coach import SPECS, AutoDetector, PlankTracker, RepCounter

FIXTURE_FILE = os.path.join("data", "parity_fixtures.json")
SCHEMA = 1
TOL = 1e-4


def _wave(lo: float, hi: float, period_s: float, seconds: float,
          fps: int = 30, phase: float = 0.0) -> list[float]:
    """hi -> lo -> hi cosine, starting at the top (lockout)."""
    mid, amp = (lo + hi) / 2.0, (hi - lo) / 2.0
    n = int(seconds * fps)
    return [round(mid + amp * math.cos(2 * math.pi * t / (period_s * fps)
                                       + phase), 4)
            for t in range(n)]


def _const(v: float, seconds: float, fps: int = 30) -> list[float]:
    return [round(v, 4)] * int(seconds * fps)


# --------------------------------------------------------------- FSM cases
def _fsm_inputs() -> list[dict]:
    """name, exercise, fps, angles — inputs only; expectations are derived."""
    cases = [
        # 3 clean full-depth squats, 3 s cadence
        {"name": "squat_3_clean", "exercise": "squat", "fps": 30,
         "angles": _wave(85, 172, 3.0, 9.4)},
        # dips to ~120: past start (150), above bottom (100) -> early
        # turnaround path, rep counts but full_depth must be False
        {"name": "squat_shallow", "exercise": "squat", "fps": 30,
         "angles": _wave(120, 172, 2.5, 5.2)},
        # sub-min_rep_s blip must NOT count; the clean rep after it must
        {"name": "squat_blip_then_clean", "exercise": "squat", "fps": 30,
         "angles": (_const(170, 0.5) + _wave(90, 170, 0.5, 0.5)
                    + _const(170, 0.5) + _wave(88, 170, 3.0, 3.2)
                    + _const(170, 0.5))},
        # curl: the ANGLE-descent is the lift -> ecc/con must swap
        {"name": "curl_2_reps", "exercise": "curl", "fps": 30,
         "angles": _wave(60, 165, 3.0, 6.4)},
        # pull-up: same swap, different thresholds
        {"name": "pullup_2_reps", "exercise": "pullup", "fps": 30,
         "angles": _wave(70, 165, 2.8, 6.0)},
        # push-up at a fast 1.4 s cadence
        {"name": "pushup_fast_4", "exercise": "pushup", "fps": 30,
         "angles": _wave(88, 168, 1.4, 6.0)},
        # deadlift drives the hip signal
        {"name": "deadlift_2", "exercise": "deadlift", "fps": 30,
         "angles": _wave(92, 172, 3.2, 7.0)},
        # lunge: knee to 105 (bottom_below 110 -> full depth True)
        {"name": "lunge_2", "exercise": "lunge", "fps": 30,
         "angles": _wave(105, 170, 3.0, 6.4)},
        # racked press: stream starts BELOW start_below (mid-rep state
        # entry) — pins the documented "first eccentric is rack time" path
        {"name": "press_racked_start", "exercise": "shoulder_press",
         "fps": 30,
         "angles": _const(100, 1.0) + _wave(100, 170, 3.0, 3.2,
                                            phase=math.pi)},
    ]
    return cases


def _run_fsm(exercise: str, angles: list[float], fps: int) -> dict:
    c = RepCounter(SPECS[exercise])
    reps = []
    for i, a in enumerate(angles):
        ev = c.update(a, i / fps)
        if ev is not None:
            reps.append({"n": ev.count,
                         "duration_s": round(ev.duration, 6),
                         "ecc_s": round(ev.eccentric_s, 6),
                         "con_s": round(ev.concentric_s, 6),
                         "min_angle": round(ev.min_angle, 6),
                         "full_depth": ev.full_depth})
    return {"count": c.count, "final_state": c.state, "reps": reps}


# ------------------------------------------------------------- plank cases
def _plank_inputs() -> list[dict]:
    return [
        # hold 4 s, sag 2.5 s (cue after the 1 s grace), recover 3 s
        {"name": "plank_hold_sag_recover", "fps": 30,
         "body_line": _const(171, 4.0) + _const(140, 2.5) + _const(173, 3.0)},
        # flicker below grace never fires a cue or resets the streak
        {"name": "plank_brief_flicker", "fps": 30,
         "body_line": _const(170, 2.0) + _const(150, 0.5) + _const(170, 2.0)},
    ]


def _run_plank(body_line: list[float], fps: int) -> dict:
    p = PlankTracker()
    cues = 0
    for i, v in enumerate(body_line):
        if p.update(v, i / fps):
            cues += 1
    return {"total_s": round(p.total, 6), "best_s": round(p.best, 6),
            "cues": cues}


# ------------------------------------------------------------ detect cases
def _feat(trunk=10.0, knee=170.0, elbow=170.0, hip=170.0, sho_y=0.3,
          wri_y=0.5, torso=0.25, overhead=False, knee_split=0.1) -> dict:
    return {"trunk": round(trunk, 4), "knee": round(knee, 4),
            "elbow": round(elbow, 4), "hip": round(hip, 4),
            "sho_y": round(sho_y, 4), "wri_y": round(wri_y, 4),
            "torso": round(torso, 4), "overhead": bool(overhead),
            "knee_split": round(knee_split, 4)}


def _detect_inputs() -> list[dict]:
    fps = 30

    def stream(seconds, fn):
        return [fn(i / fps) for i in range(int(seconds * fps))]

    knee_wave = lambda t: 130 + 40 * math.cos(2 * math.pi * t / 3.0)
    elbow_wave = lambda t: 125 + 35 * math.cos(2 * math.pi * t / 2.5)
    return [
        {"name": "detect_squat", "fps": fps,
         "frames": stream(5.0, lambda t: _feat(
             trunk=20.0, knee=knee_wave(t), hip=knee_wave(t)))},
        {"name": "detect_pushup", "fps": fps,
         "frames": stream(5.0, lambda t: _feat(
             trunk=75.0, elbow=elbow_wave(t)))},
        {"name": "detect_plank", "fps": fps,
         "frames": stream(5.0, lambda t: _feat(trunk=75.0, elbow=168.0))},
        {"name": "detect_curl", "fps": fps,
         "frames": stream(5.0, lambda t: _feat(
             elbow=105 + 55 * math.cos(2 * math.pi * t / 2.5)))},
        {"name": "detect_idle_none", "fps": fps,
         "frames": stream(5.0, lambda t: _feat())},
    ]


def _run_detect(frames: list[dict], fps: int) -> dict:
    d = AutoDetector()
    detected, at_frame = None, None
    for i, f in enumerate(frames):
        got = d.update(f, i / fps)
        if got is not None and detected is None:
            detected, at_frame = got, i
    return {"exercise": detected, "at_frame": at_frame}


# ----------------------------------------------------------------- build
def build() -> dict:
    return {
        "schema": SCHEMA,
        "generator": "parity_fixtures.py --generate",
        "tolerance": TOL,
        "fsm_cases": [{**c, "expected": _run_fsm(c["exercise"], c["angles"],
                                                 c["fps"])}
                      for c in _fsm_inputs()],
        "plank_cases": [{**c, "expected": _run_plank(c["body_line"],
                                                     c["fps"])}
                        for c in _plank_inputs()],
        "detect_cases": [{**c, "expected": _run_detect(c["frames"],
                                                       c["fps"])}
                         for c in _detect_inputs()],
    }


def generate(path: str = FIXTURE_FILE):
    data = build()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1, sort_keys=True)
        fh.write("\n")
    n = (len(data["fsm_cases"]) + len(data["plank_cases"])
         + len(data["detect_cases"]))
    print(f"Wrote {n} parity cases -> {path}")


def verify(path: str = FIXTURE_FILE) -> int:
    """Re-derive every expectation from the committed inputs and compare.
    Guards the PYTHON engine against drift; Swift/Kotlin replay the same
    file in their own test suites."""
    if not os.path.exists(path):
        print(f"missing fixtures: {path} — run --generate and commit")
        return 1
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if data.get("schema") != SCHEMA:
        print(f"schema {data.get('schema')} != {SCHEMA} — regenerate")
        return 1
    bad = 0
    for c in data["fsm_cases"]:
        got = _run_fsm(c["exercise"], c["angles"], c["fps"])
        if got != c["expected"]:
            print(f"FSM drift in {c['name']}:\n  stored {c['expected']}\n"
                  f"  now    {got}")
            bad += 1
    for c in data["plank_cases"]:
        got = _run_plank(c["body_line"], c["fps"])
        if got != c["expected"]:
            print(f"plank drift in {c['name']}: {c['expected']} -> {got}")
            bad += 1
    for c in data["detect_cases"]:
        got = _run_detect(c["frames"], c["fps"])
        if got != c["expected"]:
            print(f"detect drift in {c['name']}: {c['expected']} -> {got}")
            bad += 1
    if bad:
        print(f"{bad} case(s) drifted. If the change is intentional, "
              "regenerate the fixtures and update Swift/Kotlin in the "
              "same commit.")
        return 1
    n = (len(data["fsm_cases"]) + len(data["plank_cases"])
         + len(data["detect_cases"]))
    print(f"All {n} parity cases match the committed expectations.")
    return 0


def selftest():
    """Structural sanity + verify — run by CI on every push."""
    data = build()
    counts = {c["name"]: c["expected"]["count"] for c in data["fsm_cases"]}
    assert counts["squat_3_clean"] == 3, counts
    assert counts["curl_2_reps"] == 2 and counts["pullup_2_reps"] == 2
    assert counts["pushup_fast_4"] == 4, counts
    shallow = next(c for c in data["fsm_cases"]
                   if c["name"] == "squat_shallow")
    assert shallow["expected"]["count"] >= 1
    assert not shallow["expected"]["reps"][0]["full_depth"]
    blip = next(c for c in data["fsm_cases"]
                if c["name"] == "squat_blip_then_clean")
    assert blip["expected"]["count"] == 1, blip["expected"]
    curl = next(c for c in data["fsm_cases"] if c["name"] == "curl_2_reps")
    r = curl["expected"]["reps"][1]          # rep 2: settled tempo
    assert r["con_s"] < r["ecc_s"], r        # concentric = angle-descent
    racked = next(c for c in data["fsm_cases"]
                  if c["name"] == "press_racked_start")
    assert racked["expected"]["count"] == 1
    assert not racked["expected"]["reps"][0]["full_depth"]
    plank = next(c for c in data["plank_cases"]
                 if c["name"] == "plank_hold_sag_recover")
    assert plank["expected"]["cues"] == 1
    assert 6.5 < plank["expected"]["total_s"] < 7.5, plank["expected"]
    flick = next(c for c in data["plank_cases"]
                 if c["name"] == "plank_brief_flicker")
    assert flick["expected"]["cues"] == 0
    det = {c["name"]: c["expected"]["exercise"]
           for c in data["detect_cases"]}
    assert det == {"detect_squat": "squat", "detect_pushup": "pushup",
                   "detect_plank": "plank", "detect_curl": "curl",
                   "detect_idle_none": None}, det
    print("parity fixture generation invariants OK")
    return verify()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--generate", action="store_true",
                    help=f"write {FIXTURE_FILE} from the Python engine")
    ap.add_argument("--selftest", action="store_true",
                    help="derive + verify against the committed file")
    args = ap.parse_args()
    if args.generate:
        generate()
    elif args.selftest:
        sys.exit(selftest())
    else:
        ap.print_help()
