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

    def add_rep(self, ev: RepEvent):
        self.session["reps"].append({
            "n": ev.count, "score": ev.score,
            "eccentric_s": round(ev.eccentric_s, 2),
            "concentric_s": round(ev.concentric_s, 2),
            "min_angle": round(ev.min_angle, 1),
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
        if sm["fault_counts"]:
            print("Faults:", ", ".join(f"{k}×{v}" for k, v in
                                       sorted(sm["fault_counts"].items())))
        else:
            print("Faults: none — great set!")
    print(f"Logged to workout log.")


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
def run(exercise: str, video: str | None, use_voice: bool, log_path: str):
    import cv2
    spec = SPECS[exercise]
    mp, landmarker = make_landmarker()
    smoother, feedback = SkeletonSmoother(), FeedbackEngine()
    counter = RepCounter(spec)
    plank = PlankTracker() if spec.mode == "hold" else None
    voice, log = Voice(use_voice), WorkoutLog(log_path)

    cap = cv2.VideoCapture(video if video else 0)
    if not cap.isOpened():
        sys.exit("Could not open camera/video.")
    print(f"{exercise}: camera hint — {spec.camera_hint}. Press q to finish.")
    voice.say(f"Ready for {exercise.replace('_', ' ')}. Let's go!")
    t0, ts_ms, last_score = time.time(), 0, None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = time.time() - t0
        ts_ms = max(ts_ms + 1, int(t * 1000))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)

        pts = landmarks_to_array(result)
        if pts is not None:
            pts = smoother.update(pts, t)
            ang = body_angles(pts)
            faults_now = live_faults(exercise, ang, counter.state)

            if plank:                                   # ---- timed hold
                if plank.update(ang["body_line"], t):
                    faults_now.append("body_sag")
                msg = feedback.push(faults_now, t)
                if msg:
                    voice.say(msg)
                hud1 = (f"PLANK  hold: {plank.total:5.1f}s   "
                        f"best: {plank.best:5.1f}s")
                hud2 = f"body line: {ang['body_line']:5.1f}"
            else:                                       # ---- rep exercise
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
                    log.add_rep(ev)
                    if ev.faults:
                        cue = feedback.push(ev.faults, t)
                        voice.say(f"{ev.count}. {cue or ''}")
                    else:
                        voice.say(f"{ev.count}. {feedback.praise()}")
                    print(f"Rep {ev.count}: score {ev.score}  "
                          f"ecc {ev.eccentric_s:.1f}s / con {ev.concentric_s:.1f}s  "
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
                cv2.putText(frame, feedback.current, (10, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 80, 255), 2)

        cv2.imshow("AI Gym Coach", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break

    cap.release()
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
        assert hist[-1]["summary"] == {"reps": 1, "avg_score": 90.0,
                                       "avg_concentric_s": 1.0,
                                       "fault_counts": {"too_fast": 1}}
        assert s["exercise"] == "squat"
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
        print("OK (no pose in blank frame, as expected)")
    except ImportError:
        print("SKIPPED (mediapipe not installed)")

    print("\nAll selftests passed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="AI Gym Coach prototype v2")
    ap.add_argument("--exercise", choices=sorted(SPECS), default="squat")
    ap.add_argument("--video", help="video file instead of webcam")
    ap.add_argument("--no-voice", action="store_true", help="disable TTS voice")
    ap.add_argument("--log-file", default=DEFAULT_LOG,
                    help=f"workout log path (default {DEFAULT_LOG})")
    ap.add_argument("--selftest", action="store_true", help="run without camera")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    else:
        run(args.exercise, args.video, not args.no_voice, args.log_file)
