"""
pose_coach.py — AI Gym Coach prototype v2 (desktop, Python).

Real-time pose estimation (MediaPipe Tasks Pose Landmarker) + One Euro
smoothing + joint angles + FSM rep counting/tempo + per-exercise form rules
+ voice coaching (pyttsx3) + JSON workout logging.

Usage:
    pip install mediapipe opencv-python numpy pyttsx3
    python pose_coach.py --exercise squat                 # webcam
    python pose_coach.py --exercise deadlift --video a.mp4
    python pose_coach.py --exercise plank --no-voice
    python pose_coach.py --selftest                       # no camera needed

Exercises: squat, pushup, bench, deadlift, lunge, shoulder_press, curl,
           pullup, plank (timed hold)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import sys
import threading
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")
MODEL_PATH = os.path.join(HERE, "pose_landmarker_lite.task")
DEFAULT_LOG = os.path.join(HERE, "workout_log.json")

# BlazePose 33-landmark indices
NOSE, L_EAR, R_EAR = 0, 7, 8
L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI = 11, 12, 13, 14, 15, 16
L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK = 23, 24, 25, 26, 27, 28
L_HEE, R_HEE, L_TOE, R_TOE = 29, 30, 31, 32

VIS_MIN = 0.5  # ignore keypoints below this visibility


# ---------------------------------------------------------------- smoothing
class OneEuroFilter:
    """Adaptive low-pass filter: smooth when slow, responsive when fast."""

    def __init__(self, min_cutoff=1.0, beta=0.02, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x_prev = self.dx_prev = self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.t_prev is None:
            self.x_prev, self.dx_prev, self.t_prev = x, 0.0, t
            return x
        dt = max(t - self.t_prev, 1e-6)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


class SkeletonSmoother:
    """One Euro per coordinate per landmark, with visibility gating."""

    def __init__(self, n_landmarks=33):
        self.filters = [[OneEuroFilter() for _ in range(3)] for _ in range(n_landmarks)]
        self.last = np.zeros((n_landmarks, 4), dtype=np.float32)  # x,y,z,vis

    def update(self, pts: np.ndarray, t: float) -> np.ndarray:
        out = pts.copy()
        for i in range(pts.shape[0]):
            if pts[i, 3] < VIS_MIN and self.last[i, 3] >= VIS_MIN:
                out[i] = self.last[i]          # hold last good value
                continue
            for c in range(3):
                out[i, c] = self.filters[i][c](float(pts[i, c]), t)
        self.last = out
        return out


# ------------------------------------------------------------------- angles
def joint_angle(a, b, c) -> float:
    """Angle ABC in degrees at vertex b (2D or 3D points)."""
    ba, bc = np.asarray(a) - np.asarray(b), np.asarray(c) - np.asarray(b)
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-9:
        return 180.0
    cosang = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return math.degrees(math.acos(cosang))


def segment_vs_vertical(p_top, p_bottom) -> float:
    """Angle of segment vs the vertical axis (image coords, degrees)."""
    v = np.asarray(p_top)[:2] - np.asarray(p_bottom)[:2]
    n = np.linalg.norm(v)
    if n < 1e-9:
        return 0.0
    return math.degrees(math.acos(np.clip(-v[1] / n, -1.0, 1.0)))  # -y is "up"


def pick_side(pts) -> str:
    left = pts[[L_SHO, L_ELB, L_HIP, L_KNE, L_ANK], 3].mean()
    right = pts[[R_SHO, R_ELB, R_HIP, R_KNE, R_ANK], 3].mean()
    return "L" if left >= right else "R"


def body_angles(pts) -> dict:
    """All per-frame features used by the FSM and the form rules."""
    s = pick_side(pts)
    ear = L_EAR if s == "L" else R_EAR
    sho, elb, wri = (L_SHO, L_ELB, L_WRI) if s == "L" else (R_SHO, R_ELB, R_WRI)
    hip, kne, ank = (L_HIP, L_KNE, L_ANK) if s == "L" else (R_HIP, R_KNE, R_ANK)
    p = pts[:, :3]
    ang = {
        "side": s,
        "knee": joint_angle(p[hip], p[kne], p[ank]),
        "hip": joint_angle(p[sho], p[hip], p[kne]),
        "elbow": joint_angle(p[sho], p[elb], p[wri]),
        "trunk_lean": segment_vs_vertical(p[sho], p[hip]),
        "upper_arm_swing": segment_vs_vertical(p[sho], p[elb]),
        "body_line": joint_angle(p[sho], p[hip], p[ank]),   # 180 = straight
        "elbow_flare": joint_angle(p[hip], p[sho], p[elb]),
        "neck": (joint_angle(p[ear], p[sho], p[hip])
                 if pts[ear, 3] > VIS_MIN else 180.0),
    }
    # bilateral features (only when both sides are visible)
    if min(pts[L_KNE, 3], pts[R_KNE, 3], pts[L_ANK, 3], pts[R_ANK, 3]) > VIS_MIN:
        knee_w = abs(p[L_KNE][0] - p[R_KNE][0])
        ankle_w = max(abs(p[L_ANK][0] - p[R_ANK][0]), 1e-4)
        ang["valgus_ratio"] = knee_w / ankle_w        # < 1 => knees caving in
    else:
        ang["valgus_ratio"] = 1.0
    if min(pts[L_WRI, 3], pts[R_WRI, 3]) > VIS_MIN:
        ang["wrist_y_diff"] = abs(p[L_WRI][1] - p[R_WRI][1])
        ang["nose_above_wrists"] = (p[L_WRI][1] + p[R_WRI][1]) / 2 - p[NOSE][1]
    else:
        ang["wrist_y_diff"], ang["nose_above_wrists"] = 0.0, 1.0
    return ang


# ------------------------------------------------------- exercise definitions
@dataclass
class ExerciseSpec:
    name: str
    signal: str             # angle key driving the FSM (goes down, then up)
    start_below: float = 0  # signal below this => rep started (leave lockout)
    bottom_below: float = 0 # deep enough to count as full ROM
    lockout_above: float = 0  # back above this => rep complete
    concentric: str = "ascent"  # FSM phase that is the lift ("ascent"|"descent")
    min_rep_s: float = 0.8
    min_concentric_s: float = 0.6   # faster => "slow down"
    mode: str = "reps"      # "reps" | "hold"
    camera_hint: str = "side view"


SPECS = {
    "squat": ExerciseSpec("squat", "knee", 150, 100, 165,
                          camera_hint="side or 45° front"),
    "pushup": ExerciseSpec("pushup", "elbow", 140, 95, 155,
                           min_concentric_s=0.4),
    "bench": ExerciseSpec("bench", "elbow", 140, 90, 160,
                          camera_hint="side, camera at head height"),
    "deadlift": ExerciseSpec("deadlift", "hip", 150, 100, 165),
    "lunge": ExerciseSpec("lunge", "knee", 150, 110, 165,
                          camera_hint="side (45° front for knee tracking)"),
    "shoulder_press": ExerciseSpec("shoulder_press", "elbow", 150, 100, 160,
                                   camera_hint="front view"),
    "curl": ExerciseSpec("curl", "elbow", 140, 70, 155,
                         concentric="descent", min_concentric_s=0.5),
    "pullup": ExerciseSpec("pullup", "elbow", 140, 80, 160,
                           concentric="descent", camera_hint="front view"),
    "plank": ExerciseSpec("plank", "body_line", mode="hold"),
}


# ----------------------------------------------------------- rep counter FSM
@dataclass
class RepEvent:
    count: int
    duration: float
    eccentric_s: float
    concentric_s: float
    min_angle: float
    full_depth: bool
    faults: list = field(default_factory=list)
    score: int = 100


class RepCounter:
    """IDLE -> DESCENT -> BOTTOM -> ASCENT -> (rep++) on the signal angle.

    "Descent/ascent" refer to the *angle*: for curls and pull-ups the angle
    descends during the lift, so spec.concentric maps phases to tempo names.
    A press started from the rack enters DESCENT immediately; its first
    "eccentric" time is just time at the rack and settles from rep 2 on.
    """

    def __init__(self, spec: ExerciseSpec):
        self.spec = spec
        self.state = "IDLE"
        self.count = 0
        self.t_start = self.t_bottom = 0.0
        self.min_angle = 180.0
        self.rep_faults: set[str] = set()

    def note_fault(self, fault: str):
        if self.state != "IDLE":
            self.rep_faults.add(fault)

    def update(self, angle: float, t: float) -> RepEvent | None:
        sp = self.spec
        if self.state == "IDLE":
            if angle < sp.start_below:
                self.state, self.t_start, self.min_angle = "DESCENT", t, angle
                self.rep_faults = set()
        elif self.state == "DESCENT":
            self.min_angle = min(self.min_angle, angle)
            if angle < sp.bottom_below:
                self.state, self.t_bottom = "BOTTOM", t
            elif angle > self.min_angle + 15:          # turned around early
                self.state, self.t_bottom = "ASCENT", t
        elif self.state == "BOTTOM":
            self.min_angle = min(self.min_angle, angle)
            if angle > self.min_angle + 10:
                self.state = "ASCENT"
        elif self.state == "ASCENT":
            if angle > sp.lockout_above:
                dur = t - self.t_start
                self.state = "IDLE"
                if dur < sp.min_rep_s:                 # noise blip, not a rep
                    return None
                self.count += 1
                down_s, up_s = self.t_bottom - self.t_start, t - self.t_bottom
                ecc, con = (down_s, up_s) if sp.concentric == "ascent" else (up_s, down_s)
                return RepEvent(
                    count=self.count, duration=dur,
                    eccentric_s=ecc, concentric_s=con,
                    min_angle=self.min_angle,
                    full_depth=self.min_angle < sp.bottom_below,
                    faults=sorted(self.rep_faults),
                )
        return None


class PlankTracker:
    """Timed hold: accumulate time while the body line stays straight."""

    def __init__(self, good_above=160.0, grace_s=1.0):
        self.good_above, self.grace_s = good_above, grace_s
        self.total = self.streak = self.best = 0.0
        self.bad_for = 0.0
        self.t_prev: float | None = None

    def update(self, body_line: float, t: float) -> bool:
        """Returns True when a 'fix your line' cue should fire."""
        dt = 0.0 if self.t_prev is None else max(t - self.t_prev, 0.0)
        self.t_prev = t
        if body_line >= self.good_above:
            self.total += dt
            self.streak += dt
            self.best = max(self.best, self.streak)
            self.bad_for = 0.0
            return False
        was_ok = self.bad_for <= self.grace_s
        self.bad_for += dt
        if self.bad_for > self.grace_s:
            self.streak = 0.0
            return was_ok            # fire cue once when grace expires
        return False


# ------------------------------------------------------ auto exercise detect
def frame_features(ang: dict, pts: np.ndarray) -> dict:
    """Per-frame features consumed by AutoDetector (kept minimal so tests
    can synthesize them without full skeletons)."""
    p = pts[:, :3]
    sho_y = (p[L_SHO][1] + p[R_SHO][1]) / 2
    hip_y = (p[L_HIP][1] + p[R_HIP][1]) / 2
    wri_y = (p[L_WRI][1] + p[R_WRI][1]) / 2
    torso = max(abs(hip_y - sho_y), 1e-3)
    return {
        "trunk": ang["trunk_lean"], "knee": ang["knee"],
        "elbow": ang["elbow"], "hip": ang["hip"],
        "sho_y": sho_y, "wri_y": wri_y, "torso": torso,
        "overhead": wri_y < sho_y - 0.03,          # image y grows downward
        "knee_split": abs(p[L_KNE][1] - p[R_KNE][1]) / torso,
    }


class AutoDetector:
    """Rule-based exercise classifier over a sliding window of skeleton
    features (design doc §4.3 stage-2 MVP). Locks after 3 agreeing votes.

    Not detectable from skeleton alone: bench press (indistinguishable
    from push-up without bench context) — select it manually.
    """

    WINDOW_S, VOTE_EVERY_S, NEED_AGREE = 2.0, 0.5, 3

    def __init__(self):
        self.buf: deque = deque()
        self.votes: deque = deque(maxlen=self.NEED_AGREE)
        self.next_vote_t = self.WINDOW_S

    def update(self, feat: dict, t: float) -> str | None:
        self.buf.append((t, feat))
        while self.buf and t - self.buf[0][0] > self.WINDOW_S:
            self.buf.popleft()
        if t < self.next_vote_t or len(self.buf) < 20:
            return None
        self.next_vote_t = t + self.VOTE_EVERY_S
        vote = self._classify()
        self.votes.append(vote)
        if (len(self.votes) == self.NEED_AGREE and vote
                and all(v == vote for v in self.votes)):
            return vote
        return None

    def _classify(self) -> str | None:
        f = [x for _, x in self.buf]
        get = lambda k: [x[k] for x in f]
        rom = lambda k: max(get(k)) - min(get(k))
        torso = sum(get("torso")) / len(f)
        trunk_mean, trunk_max = sum(get("trunk")) / len(f), max(get("trunk"))
        rom_knee, rom_elbow, rom_hip = rom("knee"), rom("elbow"), rom("hip")
        overhead = sum(x["overhead"] for x in f) / len(f)
        disp_sho, disp_wri = rom("sho_y") / torso, rom("wri_y") / torso
        knee_split = max(get("knee_split"))

        if trunk_mean > 55:                        # body horizontal
            return "pushup" if rom_elbow > 25 else "plank"
        if overhead > 0.7 and rom_elbow > 30:      # hands overhead
            return "pullup" if disp_sho > 1.3 * disp_wri else "shoulder_press"
        if rom_knee > 35:                          # legs driving
            if trunk_max > 55:
                return "deadlift"
            if knee_split > 0.35:
                return "lunge"
            return "squat"
        if trunk_max > 55 and rom_hip > 30:        # hip hinge, stiff knees
            return "deadlift"
        if rom_elbow > 40 and overhead < 0.3:      # arms only, below head
            return "curl"
        return None


# ------------------------------------------------------------------ fatigue
FATIGUE_MSG = "You're slowing down — keep form tight or end the set."


class FatigueMonitor:
    """Velocity-based fatigue: warn when concentric speed drops >20%
    against the best of the first three reps."""

    def __init__(self, threshold=0.20):
        self.threshold = threshold
        self.vels: list[float] = []
        self.warned = False
        self.loss = 0.0

    def add(self, velocity: float) -> bool:
        """Feed one rep's concentric velocity; True => fire fatigue cue."""
        self.vels.append(velocity)
        if len(self.vels) < 4:
            return False
        base = max(self.vels[:3])
        cur = sum(self.vels[-2:]) / 2
        self.loss = max(0.0, 1 - cur / base) if base > 0 else 0.0
        if self.loss > self.threshold and not self.warned:
            self.warned = True
            return True
        return False


# -------------------------------------------------------------- form rules
FAULT_MSGS = {  # fault -> (priority: lower = more urgent, message, penalty)
    "back_lean": (0, "Straighten your back — chest up!", 25),
    "back_round": (0, "Keep your back flat.", 30),
    "body_sag": (0, "Keep your body in a straight line.", 25),
    "knees_cave": (0, "Push your knees out — don't let them cave in.", 25),
    "shallow": (1, "Go deeper — full range of motion.", 20),
    "elbow_swing": (1, "Keep your elbows pinned to your sides.", 20),
    "elbow_flare": (1, "Tuck your elbows closer to your body.", 15),
    "torso_lean": (1, "Keep your torso upright.", 15),
    "lean_back": (1, "Don't lean back — brace your core.", 15),
    "uneven": (1, "Even it out — both sides together.", 15),
    "chin": (1, "Pull higher — chin over the bar.", 15),
    "shrug_neck": (1, "Keep your neck neutral.", 10),
    "too_fast": (2, "Slow down — control the movement.", 10),
}

MOVING = ("DESCENT", "BOTTOM", "ASCENT")

# fault -> predicate(ang, state); evaluated every frame, phase-gated
LIVE_RULES: dict[str, list] = {
    "squat": [
        ("back_lean", lambda a, s: s in MOVING and a["trunk_lean"] > 50),
        ("knees_cave", lambda a, s: s in ("BOTTOM", "ASCENT") and a["valgus_ratio"] < 0.7),
    ],
    "pushup": [
        ("body_sag", lambda a, s: s in MOVING and a["body_line"] < 155),
        ("elbow_flare", lambda a, s: s == "BOTTOM" and a["elbow_flare"] > 100),
    ],
    "bench": [
        ("uneven", lambda a, s: s in MOVING and a["wrist_y_diff"] > 0.08),
    ],
    "deadlift": [
        ("back_round", lambda a, s: s in MOVING and a["neck"] < 150),
    ],
    "lunge": [
        ("torso_lean", lambda a, s: s in MOVING and a["trunk_lean"] > 30),
    ],
    "shoulder_press": [
        ("lean_back", lambda a, s: s in MOVING and a["trunk_lean"] > 20),
        ("uneven", lambda a, s: s in MOVING and a["wrist_y_diff"] > 0.08),
    ],
    "curl": [
        ("elbow_swing", lambda a, s: s in MOVING and a["upper_arm_swing"] > 25),
        ("torso_lean", lambda a, s: s in MOVING and a["trunk_lean"] > 20),
    ],
    "pullup": [
        ("chin", lambda a, s: s == "BOTTOM" and a["nose_above_wrists"] < 0),
        ("uneven", lambda a, s: s in MOVING and a["wrist_y_diff"] > 0.10),
    ],
    "plank": [
        ("shrug_neck", lambda a, s: a["neck"] < 140),
    ],
}


def live_faults(exercise: str, ang: dict, state: str) -> list[str]:
    return [f for f, pred in LIVE_RULES.get(exercise, []) if pred(ang, state)]


def rep_faults(spec: ExerciseSpec, ev: RepEvent) -> list[str]:
    """Faults judged once per completed rep."""
    f = []
    if not ev.full_depth:
        f.append("shallow")
    if ev.concentric_s < spec.min_concentric_s:
        f.append("too_fast")
    return f


def score_rep(ev: RepEvent) -> int:
    return max(0, 100 - sum(FAULT_MSGS[f][2] for f in ev.faults))


class FeedbackEngine:
    """Rate-limited, priority-ordered coaching cues."""

    def __init__(self, cooldown=3.0):
        self.cooldown = cooldown
        self.last_said: dict[str, float] = {}
        self.current = ""

    def push(self, faults: list[str], t: float) -> str | None:
        """Returns the message if a new cue fired (for the voice channel)."""
        for fault in sorted(faults, key=lambda x: FAULT_MSGS[x][0]):
            if t - self.last_said.get(fault, -1e9) >= self.cooldown:
                self.last_said[fault] = t
                self.current = FAULT_MSGS[fault][1]
                return self.current
        if not faults:
            self.current = ""
        return None

    def praise(self):
        self.current = "Great form!"
        return self.current


# ------------------------------------------------------------------- voice
class Voice:
    """Background TTS thread (SAPI/NSSpeech/espeak via pyttsx3).

    The engine is created inside the worker thread because Windows SAPI COM
    objects must be used from the thread that initialized them.
    """

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.q: queue.Queue[str | None] = queue.Queue()
        if not enabled:
            return
        try:
            import pyttsx3  # noqa: F401
            self._t = threading.Thread(target=self._worker, daemon=True)
            self._t.start()
        except Exception:
            self.enabled = False
            print("(voice disabled: pyttsx3 unavailable)")

    def _worker(self):
        import pyttsx3
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 175)
        except Exception:
            self.enabled = False
            return
        while True:
            msg = self.q.get()
            if msg is None:
                return
            try:
                engine.say(msg)
                engine.runAndWait()
            except Exception:
                pass

    def say(self, msg: str):
        if self.enabled and self.q.qsize() < 2:   # drop cues if backlogged
            self.q.put(msg)

    def stop(self):
        if self.enabled:
            self.q.put(None)


# ------------------------------------------------------------- workout log
class WorkoutLog:
    """Appends one session record per run to a JSON file."""

    def __init__(self, path: str):
        self.path = path
        self.session = {
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "exercise": "",
            "reps": [],
            "plank": None,
        }

    def add_rep(self, ev: RepEvent, velocity: float | None = None):
        self.session["reps"].append({
            "n": ev.count, "score": ev.score,
            "eccentric_s": round(ev.eccentric_s, 2),
            "concentric_s": round(ev.concentric_s, 2),
            "min_angle": round(ev.min_angle, 1),
            "velocity": round(velocity, 1) if velocity is not None else None,
            "faults": ev.faults,
        })

    def finish(self, exercise: str, duration_s: float,
               plank: PlankTracker | None = None) -> dict:
        s = self.session
        s["exercise"], s["duration_s"] = exercise, round(duration_s, 1)
        if plank:
            s["plank"] = {"total_hold_s": round(plank.total, 1),
                          "best_streak_s": round(plank.best, 1)}
        reps = s["reps"]
        s["summary"] = {
            "reps": len(reps),
            "avg_score": round(sum(r["score"] for r in reps) / len(reps), 1) if reps else None,
            "avg_concentric_s": round(sum(r["concentric_s"] for r in reps) / len(reps), 2) if reps else None,
            "fault_counts": self._fault_counts(reps),
            "velocity_loss_pct": self._velocity_loss(reps),
        }
        history = []
        if os.path.exists(self.path):
            try:
                with open(self.path, encoding="utf-8") as fh:
                    history = json.load(fh)
            except (json.JSONDecodeError, OSError):
                history = []
        history.append(s)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(history, fh, indent=1)
        return s

    @staticmethod
    def _fault_counts(reps) -> dict:
        counts: dict[str, int] = {}
        for r in reps:
            for f in r["faults"]:
                counts[f] = counts.get(f, 0) + 1
        return counts

    @staticmethod
    def _velocity_loss(reps) -> float | None:
        vels = [r.get("velocity") for r in reps if r.get("velocity")]
        if len(vels) < 4:
            return None
        base = max(vels[:3])
        cur = sum(vels[-2:]) / 2
        return round(max(0.0, 1 - cur / base) * 100, 1) if base > 0 else None


def print_summary(s: dict):
    print("\n=== Session summary ===")
    print(f"Exercise: {s['exercise']}   duration: {s['duration_s']}s")
    if s.get("plank"):
        print(f"Plank hold: {s['plank']['total_hold_s']}s "
              f"(best unbroken {s['plank']['best_streak_s']}s)")
    sm = s["summary"]
    if sm["reps"]:
        print(f"Reps: {sm['reps']}   avg score: {sm['avg_score']}/100   "
              f"avg concentric: {sm['avg_concentric_s']}s")
        if sm.get("velocity_loss_pct") is not None:
            print(f"Velocity loss across set: {sm['velocity_loss_pct']}%"
                  + ("  (fatigue!)" if sm["velocity_loss_pct"] > 20 else ""))
        if sm["fault_counts"]:
            print("Faults:", ", ".join(f"{k}×{v}" for k, v in
                                       sorted(sm["fault_counts"].items())))
        else:
            print("Faults: none — great set!")
    print(f"Logged to workout log.")


# ---------------------------------------------------------- stats dashboard
def sparkline(values) -> str:
    bars = "▁▂▃▄▅▆▇█"
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return bars[4] * len(values)
    return "".join(bars[int((v - lo) / (hi - lo) * (len(bars) - 1))] for v in values)


def print_stats(log_path: str):
    """Progress dashboard aggregated from the workout log."""
    if not os.path.exists(log_path):
        print(f"No workout log at {log_path} yet — go train!")
        return
    with open(log_path, encoding="utf-8") as fh:
        history = json.load(fh)
    by_ex: dict[str, list] = {}
    for s in history:
        by_ex.setdefault(s.get("exercise", "?"), []).append(s)

    print(f"=== Progress ({len(history)} sessions) ===")
    for ex, sessions in sorted(by_ex.items()):
        print(f"\n{ex.upper()}  —  {len(sessions)} session(s)")
        holds = [s["plank"]["total_hold_s"] for s in sessions if s.get("plank")]
        if holds:
            print(f"  hold time per session: {sparkline(holds)}  "
                  f"last {holds[-1]}s, best {max(holds)}s")
        scores = [s["summary"]["avg_score"] for s in sessions
                  if s.get("summary", {}).get("avg_score") is not None]
        total_reps = sum(s.get("summary", {}).get("reps", 0) for s in sessions)
        if scores:
            trend = ("↑" if len(scores) > 1 and scores[-1] > scores[0] else
                     "↓" if len(scores) > 1 and scores[-1] < scores[0] else "→")
            print(f"  total reps: {total_reps}   score trend {trend}: "
                  f"{sparkline(scores)}  last {scores[-1]}, best {max(scores)}")
        faults: dict[str, int] = {}
        for s in sessions:
            for k, v in s.get("summary", {}).get("fault_counts", {}).items():
                faults[k] = faults.get(k, 0) + v
        if faults:
            top = sorted(faults.items(), key=lambda kv: -kv[1])[:3]
            print("  top faults: " + ", ".join(f"{k}×{v}" for k, v in top))


# ------------------------------------------------------------ pose backend
def ensure_model() -> str:
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading pose model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def make_landmarker():
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    opts = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=ensure_model()),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp, vision.PoseLandmarker.create_from_options(opts)


def landmarks_to_array(result) -> np.ndarray | None:
    if not result.pose_landmarks:
        return None
    lm = result.pose_landmarks[0]
    return np.array([[p.x, p.y, p.z, p.visibility] for p in lm], dtype=np.float32)


EDGES = [(L_SHO, R_SHO), (L_SHO, L_ELB), (L_ELB, L_WRI), (R_SHO, R_ELB),
         (R_ELB, R_WRI), (L_SHO, L_HIP), (R_SHO, R_HIP), (L_HIP, R_HIP),
         (L_HIP, L_KNE), (L_KNE, L_ANK), (R_HIP, R_KNE), (R_KNE, R_ANK),
         (L_ANK, L_HEE), (L_HEE, L_TOE), (R_ANK, R_HEE), (R_HEE, R_TOE)]


# --------------------------------------------------------------- main loop
def run(exercise: str, video: str | None, use_voice: bool, log_path: str,
        headless: bool = False, output: str | None = None):
    import cv2
    auto = exercise == "auto"
    spec = None if auto else SPECS[exercise]
    detector = AutoDetector() if auto else None
    mp, landmarker = make_landmarker()
    smoother, feedback = SkeletonSmoother(), FeedbackEngine()
    counter = RepCounter(spec) if spec else None
    plank = PlankTracker() if spec and spec.mode == "hold" else None
    fatigue = FatigueMonitor()
    voice, log = Voice(use_voice), WorkoutLog(log_path)

    cap = cv2.VideoCapture(video if video else 0)
    if not cap.isOpened():
        sys.exit("Could not open camera/video.")
    # video files use frame timestamps so processing speed doesn't skew
    # tempo/rep timing (e.g. faster-than-realtime headless runs in Docker)
    fps = cap.get(cv2.CAP_PROP_FPS) if video else 0.0
    fps = fps if fps and fps > 1 else 30.0
    writer = None
    quit_hint = "Ctrl+C" if headless else "q"
    if spec:
        print(f"{exercise}: camera hint — {spec.camera_hint}. "
              f"Press {quit_hint} to finish.")
        voice.say(f"Ready for {exercise.replace('_', ' ')}. Let's go!")
    else:
        print(f"Auto-detect mode: start exercising. Press {quit_hint} to finish.")
        voice.say("Start exercising, I'll recognize the movement.")
    t0, ts_ms, frame_idx, last_score = time.time(), 0, 0, None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = frame_idx / fps if video else time.time() - t0
            frame_idx += 1
            ts_ms = max(ts_ms + 1, int(t * 1000))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)

            pts = landmarks_to_array(result)
            if pts is not None:
                pts = smoother.update(pts, t)
                ang = body_angles(pts)

                if spec is None:                            # ---- auto-detect
                    det = detector.update(frame_features(ang, pts), t)
                    hud1 = "AUTO  detecting exercise..."
                    hud2 = (f"knee: {ang['knee']:5.1f}   "
                            f"elbow: {ang['elbow']:5.1f}   "
                            f"trunk: {ang['trunk_lean']:4.1f}")
                    if det:
                        exercise, spec = det, SPECS[det]
                        counter = RepCounter(spec)
                        plank = PlankTracker() if spec.mode == "hold" else None
                        print(f"Auto-detected exercise: {det} "
                              f"(camera hint — {spec.camera_hint})")
                        voice.say(f"{det.replace('_', ' ')} detected. Let's go!")
                elif plank:                                 # ---- timed hold
                    faults_now = live_faults(exercise, ang, counter.state)
                    if plank.update(ang["body_line"], t):
                        faults_now.append("body_sag")
                    msg = feedback.push(faults_now, t)
                    if msg:
                        voice.say(msg)
                    hud1 = (f"PLANK  hold: {plank.total:5.1f}s   "
                            f"best: {plank.best:5.1f}s")
                    hud2 = f"body line: {ang['body_line']:5.1f}"
                else:                                       # ---- rep exercise
                    faults_now = live_faults(exercise, ang, counter.state)
                    for fault in faults_now:
                        counter.note_fault(fault)
                    ev = counter.update(ang[spec.signal], t)
                    msg = feedback.push(faults_now, t)
                    if msg:
                        voice.say(msg)
                    if ev:
                        ev.faults = sorted(set(ev.faults) | set(rep_faults(spec, ev)))
                        ev.score = score_rep(ev)
                        last_score = ev.score
                        # concentric velocity proxy: ROM (deg) / lift time (s)
                        vel = (max(spec.lockout_above - ev.min_angle, 1.0)
                               / max(ev.concentric_s, 0.05))
                        log.add_rep(ev, velocity=vel)
                        if fatigue.add(vel):
                            feedback.current = FATIGUE_MSG
                            voice.say(FATIGUE_MSG)
                            print(f"Fatigue warning: velocity down "
                                  f"{fatigue.loss * 100:.0f}% from baseline")
                        elif ev.faults:
                            cue = feedback.push(ev.faults, t)
                            voice.say(f"{ev.count}. {cue or ''}")
                        else:
                            voice.say(f"{ev.count}. {feedback.praise()}")
                        print(f"Rep {ev.count}: score {ev.score}  "
                              f"ecc {ev.eccentric_s:.1f}s / con {ev.concentric_s:.1f}s  "
                              f"vel {vel:.0f} deg/s  "
                              f"min {spec.signal} {ev.min_angle:.0f}  "
                              f"faults {ev.faults or 'none'}")
                    hud1 = (f"{exercise.upper()}  reps: {counter.count}   "
                            f"phase: {counter.state}"
                            + (f"   last score: {last_score}" if last_score is not None else ""))
                    hud2 = (f"{spec.signal}: {ang[spec.signal]:5.1f}   "
                            f"trunk: {ang['trunk_lean']:4.1f}")

                h, w = frame.shape[:2]
                for i, j in EDGES:
                    if pts[i, 3] > VIS_MIN and pts[j, 3] > VIS_MIN:
                        cv2.line(frame, (int(pts[i, 0] * w), int(pts[i, 1] * h)),
                                 (int(pts[j, 0] * w), int(pts[j, 1] * h)), (0, 255, 120), 2)
                for k, line in enumerate((hud1, hud2)):
                    cv2.putText(frame, line, (10, 30 + 28 * k),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                if feedback.current:
                    cv2.putText(frame, feedback.current, (10, frame.shape[0] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 80, 255), 2)

            if output:
                if writer is None:
                    writer = cv2.VideoWriter(
                        output, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                        (frame.shape[1], frame.shape[0]))
                writer.write(frame)
            if not headless:
                cv2.imshow("AI Gym Coach", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    except KeyboardInterrupt:
        print("\nInterrupted — finishing session...")

    cap.release()
    landmarker.close()
    if writer is not None:
        writer.release()
        print(f"Annotated video written to {output}")
    if not headless:
        cv2.destroyAllWindows()
    summary = log.finish(exercise, time.time() - t0, plank)
    print_summary(summary)
    if plank:
        voice.say(f"Done. You held {int(plank.total)} seconds.")
    elif summary["summary"]["reps"]:
        voice.say(f"Set done. {summary['summary']['reps']} reps, "
                  f"average score {int(summary['summary']['avg_score'])}.")
    time.sleep(1.5)   # let the last voice line play
    voice.stop()


# ----------------------------------------------------------------- selftest
def selftest():
    print("1) joint_angle sanity:", end=" ")
    assert abs(joint_angle((0, 1), (0, 0), (1, 0)) - 90) < 1e-6
    assert abs(joint_angle((0, 1, 0), (0, 0, 0), (0, 2, 0)) - 0) < 1e-6
    print("OK")

    print("2) One Euro filter reduces jitter:", end=" ")
    rng = np.random.default_rng(0)
    ts = np.arange(0, 5, 1 / 30)
    noisy_static = 0.7 + rng.normal(0, 0.05, ts.size)
    f = OneEuroFilter(min_cutoff=1.0, beta=0.02)
    sm_static = np.array([f(x, t) for x, t in zip(noisy_static, ts)])
    err_n = np.abs(noisy_static - 0.7)[10:].mean()
    err_s = np.abs(sm_static - 0.7)[10:].mean()
    assert err_s < err_n, (err_s, err_n)
    noisy_move = np.sin(ts) + rng.normal(0, 0.05, ts.size)
    f2 = OneEuroFilter(min_cutoff=1.0, beta=0.02)
    sm_move = np.array([f2(x, t) for x, t in zip(noisy_move, ts)])
    jit_n, jit_s = np.abs(np.diff(noisy_move)).mean(), np.abs(np.diff(sm_move)).mean()
    assert jit_s < jit_n, (jit_s, jit_n)
    print(f"OK (static err {err_n:.3f}->{err_s:.3f}, jitter {jit_n:.3f}->{jit_s:.3f})")

    print("3) FSM counts 5 synthetic squat reps:", end=" ")
    counter, n = RepCounter(SPECS["squat"]), 0
    for t in np.arange(0, 15, 1 / 30):
        if counter.update(130 + 45 * math.cos(2 * math.pi * t / 3), float(t)):
            n += 1
    assert n == 5, n
    print("OK")

    print("4) shallow rep flagged:", end=" ")
    counter, ev = RepCounter(SPECS["squat"]), None
    for t in np.arange(0, 3, 1 / 30):
        ev = counter.update(140 + 32 * math.cos(2 * math.pi * t / 3), float(t)) or ev
    assert ev and not ev.full_depth and "shallow" in rep_faults(SPECS["squat"], ev)
    print("OK")

    print("5) concentric-first FSM (curl) maps tempo correctly:", end=" ")
    counter, ev = RepCounter(SPECS["curl"]), None
    for t in np.arange(0, 4, 1 / 30):
        # elbow 170 -> 50 in 1s (curl up), hold, back up 50 -> 170 in 2s
        if t < 1:
            a = 170 - 120 * t
        elif t < 1.5:
            a = 50
        else:
            a = min(170, 50 + 120 * (t - 1.5) / 2 * 2)
        ev = counter.update(a, float(t)) or ev
    assert ev is not None and ev.count == 1
    assert ev.concentric_s < ev.eccentric_s, (ev.concentric_s, ev.eccentric_s)
    print(f"OK (con {ev.concentric_s:.1f}s < ecc {ev.eccentric_s:.1f}s)")

    print("6) plank tracker accumulates hold + fires cue:", end=" ")
    pl, cued = PlankTracker(), 0
    for t in np.arange(0, 12, 1 / 30):
        line = 175.0 if (t < 5 or t > 8) else 145.0    # 3 s sag in the middle
        if pl.update(line, float(t)):
            cued += 1
    assert 8.5 < pl.total < 9.5, pl.total
    assert cued == 1, cued
    assert 4.5 < pl.best < 5.5, pl.best
    print(f"OK (hold {pl.total:.1f}s, best {pl.best:.1f}s, cues {cued})")

    print("7) all rep specs have ordered thresholds:", end=" ")
    for sp in SPECS.values():
        if sp.mode == "reps":
            assert sp.bottom_below < sp.start_below < sp.lockout_above + 1, sp.name
            assert sp.concentric in ("ascent", "descent")
    print("OK")

    print("8) workout log roundtrip:", end=" ")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "log.json")
        for _ in range(2):                    # two sessions append
            wl = WorkoutLog(path)
            wl.add_rep(RepEvent(1, 3.0, 2.0, 1.0, 85.0, True,
                                faults=["too_fast"], score=90))
            s = wl.finish("squat", 30.0)
        with open(path, encoding="utf-8") as fh:
            hist = json.load(fh)
        assert len(hist) == 2
        sm = hist[-1]["summary"]
        assert sm["reps"] == 1 and sm["avg_score"] == 90.0
        assert sm["avg_concentric_s"] == 1.0
        assert sm["fault_counts"] == {"too_fast": 1}
        assert sm["velocity_loss_pct"] is None     # needs >=4 tracked reps
        assert s["exercise"] == "squat"
        wl = WorkoutLog(path)                      # velocity-loss summary
        for i, v in enumerate([100, 100, 100, 70, 60]):
            wl.add_rep(RepEvent(i + 1, 3.0, 2.0, 1.0, 85.0, True, [], 90),
                       velocity=v)
        s = wl.finish("squat", 60.0)
        assert s["summary"]["velocity_loss_pct"] == 35.0
    print("OK")

    print("9) voice engine:", end=" ")
    v = Voice(enabled=False)
    v.say("silent")                            # must be a no-op
    try:
        import pyttsx3
        eng = pyttsx3.init()                   # driver init only, no audio
        assert eng is not None
        print("OK (pyttsx3 driver initialized)")
    except Exception as e:
        print(f"SKIPPED ({type(e).__name__})")

    print("10) MediaPipe landmarker init + blank frame:", end=" ")
    try:
        mp, lm = make_landmarker()
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=blank), 1)
        assert res.pose_landmarks is not None
        lm.close()   # avoid noisy teardown at interpreter shutdown
        print("OK (no pose in blank frame, as expected)")
    except ImportError:
        print("SKIPPED (mediapipe not installed)")

    print("11) auto exercise detection:", end=" ")

    def synth_stream(expected, **kw):
        ad = AutoDetector()
        osc = lambda lo, hi, t: (lo + hi) / 2 + (hi - lo) / 2 * math.cos(
            2 * math.pi * t / 2.0)
        for i in range(150):
            t = i / 30.0
            feat = {"trunk": 10.0, "knee": 170.0, "elbow": 170.0, "hip": 170.0,
                    "sho_y": 0.3, "wri_y": 0.5, "torso": 0.25,
                    "overhead": False, "knee_split": 0.1}
            for k, v in kw.items():
                feat[k] = osc(*v, t) if isinstance(v, tuple) else v
            det = ad.update(feat, t)
            if det:
                assert det == expected, f"{expected} misread as {det}"
                return
        raise AssertionError(f"{expected} never detected")

    synth_stream("squat", knee=(80, 170), hip=(90, 170), trunk=(5, 35))
    synth_stream("pushup", trunk=75.0, elbow=(90, 160))
    synth_stream("plank", trunk=75.0)
    synth_stream("pullup", overhead=True, elbow=(60, 160),
                 sho_y=(0.3, 0.6), wri_y=0.1)
    synth_stream("shoulder_press", overhead=True, elbow=(90, 170),
                 wri_y=(0.05, 0.25))
    synth_stream("deadlift", knee=(120, 170), hip=(90, 170), trunk=(10, 70))
    synth_stream("lunge", knee=(90, 170), knee_split=(0.1, 0.5))
    synth_stream("curl", elbow=(60, 160))
    print("OK (8 movements classified)")

    print("12) fatigue monitor:", end=" ")
    fm = FatigueMonitor()
    fired = [fm.add(v) for v in [10, 10, 10, 9, 8, 7, 5]]
    assert fired == [False, False, False, False, False, True, False], fired
    assert fm.loss > 0.2
    print("OK (warns once at >20% velocity loss)")

    print("13) stats dashboard:", end=" ")
    assert len(sparkline([1, 2, 3])) == 3 and len(set(sparkline([5, 5]))) == 1
    import io
    from contextlib import redirect_stdout
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "log.json")
        for score in (70, 85):
            wl = WorkoutLog(path)
            wl.add_rep(RepEvent(1, 3.0, 2.0, 1.0, 85.0, True,
                                ["knee_valgus"], score))
            wl.finish("squat", 30.0)
        wl = WorkoutLog(path)
        wl.finish("plank", 40.0, plank=PlankTracker())
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_stats(path)
        out = buf.getvalue()
        assert "SQUAT" in out and "PLANK" in out
        assert "total reps: 2" in out and "knee_valgus" in out
    print("OK")

    print("\nAll selftests passed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="AI Gym Coach prototype v3")
    ap.add_argument("--exercise", choices=sorted(SPECS) + ["auto"], default="squat",
                    help="exercise to coach, or 'auto' to detect from movement")
    ap.add_argument("--video", help="video file instead of webcam")
    ap.add_argument("--no-voice", action="store_true", help="disable TTS voice")
    ap.add_argument("--headless", action="store_true",
                    help="no GUI window (Docker/servers); requires --video "
                         "or Ctrl+C to stop a webcam session")
    ap.add_argument("--output", help="write annotated video to this file (mp4)")
    ap.add_argument("--log-file", default=DEFAULT_LOG,
                    help=f"workout log path (default {DEFAULT_LOG})")
    ap.add_argument("--stats", action="store_true",
                    help="print progress dashboard from the workout log and exit")
    ap.add_argument("--selftest", action="store_true", help="run without camera")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    elif args.stats:
        print_stats(args.log_file)
    else:
        run(args.exercise, args.video, not args.no_voice, args.log_file,
            headless=args.headless, output=args.output)
