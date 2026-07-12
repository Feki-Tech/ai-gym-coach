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
    python pose_coach.py --train-classifier               # ML auto-detect
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


# ------------------------------------------------- ML exercise classifier
ML_CLASSES = ("curl", "deadlift", "lunge", "plank", "pullup", "pushup",
              "shoulder_press", "squat")   # bench: manual-only (see AutoDetector)
FEAT_KEYS = ("trunk", "knee", "elbow", "hip", "sho_y", "wri_y",
             "torso", "overhead", "knee_split")
NDIM = 4 * len(FEAT_KEYS) + 2
MODEL_FILE = "classifier.npz"


def window_features(frames: list[dict]) -> np.ndarray:
    """Fixed-size vector from a window of frame_features dicts: per-channel
    mean/std/min/max plus torso-normalized shoulder & wrist travel (the same
    cues the rule-based detector keys on) -> NDIM dims."""
    a = np.array([[float(f[k]) for k in FEAT_KEYS] for f in frames])
    torso = max(float(a[:, FEAT_KEYS.index("torso")].mean()), 1e-3)
    rom = lambda k: float(np.ptp(a[:, FEAT_KEYS.index(k)]))
    return np.concatenate([a.mean(0), a.std(0), a.min(0), a.max(0),
                           [rom("sho_y") / torso, rom("wri_y") / torso]])


def synth_frames(exercise: str, rng: np.random.Generator,
                 seconds: float = 2.0, fps: int = 30) -> list[dict]:
    """Randomized synthetic feature stream for one exercise — amplitude,
    tempo, phase and sensor noise all jittered. Bootstraps classifier
    training without a labeled video dataset; blend in real recordings
    with --collect."""
    n = int(seconds * fps)
    period, phase = rng.uniform(1.5, 4.0), rng.uniform(0, 2 * math.pi)
    U = rng.uniform

    def wave(lo, hi):
        mid, amp = (lo + hi) / 2, (hi - lo) / 2
        return [mid + amp * math.cos(2 * math.pi * t / (period * fps) + phase)
                for t in range(n)]

    const = lambda v: [v] * n
    ch = dict(trunk=const(U(3, 15)), knee=const(U(165, 175)),
              elbow=const(U(160, 175)), hip=const(U(165, 175)),
              sho_y=const(U(0.25, 0.35)), wri_y=const(U(0.45, 0.6)),
              torso=const(U(0.2, 0.3)), overhead=const(0.0),
              knee_split=const(U(0.05, 0.15)))
    if exercise == "squat":
        ch.update(knee=wave(U(65, 95), U(160, 175)),
                  hip=wave(U(80, 100), U(160, 175)),
                  trunk=wave(U(3, 8), U(30, 50)))
    elif exercise == "pushup":
        ch.update(trunk=const(U(60, 85)),
                  elbow=wave(U(80, 100), U(150, 170)))
    elif exercise == "plank":
        ch.update(trunk=const(U(60, 85)),
                  elbow=const(U(70, 100) if rng.random() < 0.5
                              else U(150, 175)))       # forearm or straight-arm
    elif exercise == "pullup":
        lo = U(0.22, 0.32)
        ch.update(overhead=const(1.0), elbow=wave(U(50, 75), U(150, 170)),
                  sho_y=wave(lo, lo + U(0.18, 0.32)), wri_y=const(U(0.06, 0.14)))
    elif exercise == "shoulder_press":
        ch.update(overhead=const(1.0), elbow=wave(U(85, 105), U(155, 175)),
                  wri_y=wave(U(0.04, 0.1), U(0.24, 0.36)))
    elif exercise == "deadlift":
        ch.update(knee=wave(U(110, 130), U(160, 175)),
                  hip=wave(U(85, 105), U(160, 175)),
                  trunk=wave(U(8, 15), U(55, 75)))
    elif exercise == "lunge":
        ch.update(knee=wave(U(80, 105), U(160, 175)),
                  knee_split=wave(U(0.05, 0.1), U(0.4, 0.6)),
                  trunk=wave(U(3, 8), U(10, 25)))
    elif exercise == "curl":
        ch.update(elbow=wave(U(45, 70), U(145, 165)),
                  wri_y=wave(U(0.3, 0.38), U(0.5, 0.6)))
    else:
        raise ValueError(f"no synthetic model for {exercise}")
    frames = []
    for i in range(n):
        f = {k: v[i] for k, v in ch.items()}
        for k, sd in (("trunk", 2), ("knee", 2), ("elbow", 2), ("hip", 2),
                      ("sho_y", .008), ("wri_y", .008), ("torso", .004)):
            f[k] += rng.normal(0, sd)
        f["knee_split"] = abs(f["knee_split"] + rng.normal(0, .02))
        frames.append(f)
    return frames


class TinyMLP:
    """Two-layer numpy MLP (NDIM features -> ReLU hidden -> softmax classes).

    ~1.5k parameters: trains in <1 s on CPU and ships without any deep
    learning framework — same "small + local" philosophy as the rest of
    the prototype.
    """

    def __init__(self, n_in: int = NDIM, n_hidden: int = 32,
                 classes=ML_CLASSES, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.classes = [str(c) for c in classes]
        self.W1 = rng.normal(0, math.sqrt(2 / n_in), (n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, math.sqrt(2 / n_hidden),
                             (n_hidden, len(self.classes)))
        self.b2 = np.zeros(len(self.classes))
        self.mu, self.sd = np.zeros(n_in), np.ones(n_in)

    def _forward(self, Xn):
        h = np.maximum(0.0, Xn @ self.W1 + self.b1)
        z = h @ self.W2 + self.b2
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        return h, p / p.sum(axis=1, keepdims=True)

    def predict_proba(self, x) -> np.ndarray:
        Xn = (np.atleast_2d(np.asarray(x, dtype=float)) - self.mu) / self.sd
        return self._forward(Xn)[1]

    def fit(self, X, y, epochs: int = 300, lr: float = 0.05,
            momentum: float = 0.9):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Xn = (X - self.mu) / self.sd
        Y = np.eye(len(self.classes))[y]
        params = (self.W1, self.b1, self.W2, self.b2)
        vel = [np.zeros_like(p) for p in params]
        for _ in range(epochs):
            h, p = self._forward(Xn)
            g = (p - Y) / len(Xn)                      # softmax-CE gradient
            gh = g @ self.W2.T
            gh[h <= 0] = 0.0
            grads = (Xn.T @ gh, gh.sum(0), h.T @ g, g.sum(0))
            for v, prm, grd in zip(vel, params, grads):
                v *= momentum
                v -= lr * grd
                prm += v
        return self

    def save(self, path: str):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                 mu=self.mu, sd=self.sd, classes=np.array(self.classes))

    @classmethod
    def load(cls, path: str) -> "TinyMLP":
        d = np.load(path, allow_pickle=False)
        m = cls(d["W1"].shape[0], d["W1"].shape[1],
                [str(c) for c in d["classes"]])
        m.W1, m.b1, m.W2, m.b2 = d["W1"], d["b1"], d["W2"], d["b2"]
        m.mu, m.sd = d["mu"], d["sd"]
        return m


def build_dataset(samples_per_class: int = 120, seed: int = 0,
                  collected: str | None = None):
    """Synthetic windows for every class + optional real labeled windows
    appended by --collect (JSONL rows {"label": ..., "x": [...]})"""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for ci, ex in enumerate(ML_CLASSES):
        for _ in range(samples_per_class):
            X.append(window_features(synth_frames(ex, rng)))
            y.append(ci)
    n_real = 0
    if collected and os.path.exists(collected):
        with open(collected, encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("label") in ML_CLASSES and len(row.get("x", [])) == NDIM:
                    X.append(np.asarray(row["x"], dtype=float))
                    y.append(ML_CLASSES.index(row["label"]))
                    n_real += 1
    return np.array(X), np.array(y), n_real


def train_classifier(model_path: str = MODEL_FILE,
                     collected: str | None = None,
                     samples_per_class: int = 120, epochs: int = 300,
                     seed: int = 0) -> float:
    """Train the exercise classifier and save weights; returns val accuracy."""
    X, y, n_real = build_dataset(samples_per_class, seed, collected)
    idx = np.random.default_rng(seed).permutation(len(X))
    X, y = X[idx], y[idx]
    n_val = max(1, len(X) // 5)
    model = TinyMLP(seed=seed).fit(X[n_val:], y[n_val:], epochs=epochs)
    pred = model.predict_proba(X[:n_val]).argmax(1)
    acc = float((pred == y[:n_val]).mean())
    print(f"Trained on {len(X) - n_val} windows ({samples_per_class}/class "
          f"synthetic + {n_real} collected), validation accuracy {acc:.1%}")
    for ci, ex in enumerate(ML_CLASSES):
        m = y[:n_val] == ci
        if m.any():
            print(f"  {ex:15s} {float((pred[m] == ci).mean()):6.1%} "
                  f"({int(m.sum())} val windows)")
    model.save(model_path)
    print(f"Model saved -> {model_path}")
    return acc


class MLDetector(AutoDetector):
    """AutoDetector with the rule-based vote swapped for the trained MLP —
    same sliding window, vote cadence, and 3-agreeing-votes lock-in."""

    MIN_PROBA = 0.75

    def __init__(self, model: TinyMLP):
        super().__init__()
        self.model = model

    def _classify(self) -> str | None:
        p = self.model.predict_proba(
            window_features([x for _, x in self.buf]))[0]
        ci = int(p.argmax())
        return self.model.classes[ci] if p[ci] >= self.MIN_PROBA else None


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


# ------------------------------------------------- reference rep comparison
REFERENCE_FILE = "references.json"
REF_SAMPLES = 50


def resample(values, n: int = REF_SAMPLES) -> np.ndarray:
    """Linearly resample a 1-D sequence to n points."""
    v = np.asarray(list(values), dtype=float)
    if v.size == 0:
        return np.zeros(n)
    if v.size == 1:
        return np.full(n, v[0])
    return np.interp(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, v.size), v)


def dtw_distance(a, b) -> float:
    """Dynamic-time-warping distance normalized by path-length bound (n+m).

    Classic O(n*m) DP with |a_i − b_j| local cost — tolerant to tempo
    differences between two reps of the same movement, sensitive to shape
    (depth, lockout, asymmetry of descent vs ascent).
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n, m = a.size, b.size
    if n == 0 or m == 0:
        return float("inf")
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        cost = np.abs(a[i - 1] - b)
        for j in range(1, m + 1):
            D[i, j] = cost[j - 1] + min(D[i - 1, j], D[i, j - 1],
                                        D[i - 1, j - 1])
    return float(D[n, m]) / (n + m)


def similarity(user_traj, ref_traj, tol_deg: float = 25.0) -> int:
    """0-100: how closely a rep's angle trajectory matches the reference.

    100 = same shape (tempo-normalized); 0 = mean DTW deviation ≥ tol_deg°.
    """
    d = dtw_distance(resample(user_traj), resample(ref_traj))
    return int(round(max(0.0, 1.0 - d / tol_deg) * 100))


def load_references(path: str = REFERENCE_FILE) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            refs = json.load(fh)
        return refs if isinstance(refs, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_reference(exercise: str, traj, score: int,
                   path: str = REFERENCE_FILE):
    refs = load_references(path)
    refs[exercise] = {
        "recorded": time.strftime("%Y-%m-%d %H:%M:%S"),
        "score": score,
        "trajectory": [round(float(x), 2) for x in resample(traj)],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(refs, fh, indent=1)


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
        self._engine = None
        self._speaking = False
        self._interrupted = threading.Event()
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
        engine = None
        while True:
            msg = self.q.get()
            if msg is None:
                return
            if engine is None:      # fresh engine after start or interrupt
                try:
                    engine = pyttsx3.init()
                    engine.setProperty("rate", 175)
                except Exception:
                    self.enabled = False
                    return
                self._engine = engine
            self._speaking = True
            try:
                engine.say(msg)
                engine.runAndWait()
            except Exception:
                engine = self._engine = None
            finally:
                self._speaking = False
            if self._interrupted.is_set():
                # Windows SAPI can go permanently mute if an engine is
                # reused after stop() — dispose it and start clean.
                self._interrupted.clear()
                engine = self._engine = None

    def say(self, msg: str):
        if self.enabled and self.q.qsize() < 2:   # drop cues if backlogged
            self.q.put(msg)

    def say_chat(self, msg: str):
        """Chat sentences are never dropped (unlike backlogged form cues)."""
        if self.enabled:
            self.q.put(msg)

    def is_speaking(self) -> bool:
        """True while talking or with queued speech (gates the open mic)."""
        return self.enabled and (self._speaking or not self.q.empty())

    def interrupt(self):
        """Barge-in: drop queued speech and cut the current utterance."""
        if not self.enabled:
            return
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass
        if self._speaking:
            self._interrupted.set()
            eng = self._engine
            if eng is not None:
                try:
                    eng.stop()
                except Exception:
                    pass

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

    def add_rep(self, ev: RepEvent, velocity: float | None = None,
                similarity: int | None = None):
        self.session["reps"].append({
            "n": ev.count, "score": ev.score,
            "eccentric_s": round(ev.eccentric_s, 2),
            "concentric_s": round(ev.concentric_s, 2),
            "min_angle": round(ev.min_angle, 1),
            "velocity": round(velocity, 1) if velocity is not None else None,
            "similarity": similarity,
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
        sims = [r["similarity"] for r in reps if r.get("similarity") is not None]
        s["summary"] = {
            "reps": len(reps),
            "avg_score": round(sum(r["score"] for r in reps) / len(reps), 1) if reps else None,
            "avg_concentric_s": round(sum(r["concentric_s"] for r in reps) / len(reps), 2) if reps else None,
            "avg_similarity": round(sum(sims) / len(sims), 1) if sims else None,
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
        if sm.get("avg_similarity") is not None:
            print(f"Reference similarity: {sm['avg_similarity']}/100 "
                  f"(vs your recorded golden rep)")
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
        headless: bool = False, output: str | None = None,
        coach: bool = False, record_reference: bool = False,
        reference_file: str = REFERENCE_FILE, detector_kind: str = "auto",
        model_file: str = MODEL_FILE, collect: str | None = None):
    import cv2
    auto = exercise == "auto"
    spec = None if auto else SPECS[exercise]
    detector = None
    if auto:
        use_ml = detector_kind == "ml" or (detector_kind == "auto"
                                           and os.path.exists(model_file))
        if use_ml:
            if not os.path.exists(model_file):
                sys.exit(f"No trained model at {model_file} — run "
                         "'python pose_coach.py --train-classifier' first.")
            detector = MLDetector(TinyMLP.load(model_file))
            print(f"Auto-detect backend: ML classifier ({model_file})")
        else:
            detector = AutoDetector()
            print("Auto-detect backend: rules "
                  "(run --train-classifier once to upgrade to the ML model)")

    cap = cv2.VideoCapture(video if video else 0)
    if not cap.isOpened():
        sys.exit(f"Could not open {'video: ' + video if video else 'webcam 0'}"
                 + ("" if video else " (camera busy or access blocked?)"))
    mp, landmarker = make_landmarker()
    smoother, feedback = SkeletonSmoother(), FeedbackEngine()
    counter = RepCounter(spec) if spec else None
    plank = PlankTracker() if spec and spec.mode == "hold" else None
    fatigue = FatigueMonitor()
    voice, log = Voice(use_voice), WorkoutLog(log_path)

    references = load_references(reference_file)
    ref_traj = None
    if not record_reference and spec:
        ref_traj = (references.get(exercise) or {}).get("trajectory")
        if ref_traj:
            print(f"Scoring each rep against your reference rep "
                  f"(recorded {references[exercise]['recorded']}).")
    if record_reference:
        print("Recording mode: the best rep of this set becomes the "
              f"golden reference for future sessions ({reference_file}).")
    rep_traj: list[float] = []
    best_ref: tuple[int, list[float]] | None = None
    last_sim = None

    collect_rows: list[dict] | None = None
    if collect:
        if auto:
            print("--collect needs a fixed --exercise as the label; ignoring.")
        else:
            collect_rows = []
            print(f"Collecting labeled training windows -> {collect}")
    collect_buf: deque = deque()
    next_collect_t = 2.0

    chat = None
    if coach:
        import coach_chat
        live_state = {"exercise": None if auto else exercise, "phase": "IDLE",
                      "reps": 0, "last_score": None, "fault_counts": {},
                      "velocity_loss_pct": None, "plank_hold_s": None}
        chat = coach_chat.start_background_chat(
            state_provider=lambda: dict(live_state),
            speak=voice.say_chat, stop_speaking=voice.interrupt,
            tts_active=voice.is_speaking, hands_free=not headless,
            log_path=log_path)
        if not headless:
            print("Press 'c' in the video window to interrupt the coach "
                  "and ask right away.")

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

                if collect_rows is not None:
                    collect_buf.append((t, frame_features(ang, pts)))
                    while collect_buf and t - collect_buf[0][0] > 2.0:
                        collect_buf.popleft()
                    if len(collect_buf) >= 20 and t >= next_collect_t:
                        next_collect_t = t + 1.0
                        collect_rows.append({"label": exercise, "x": [
                            round(float(v), 5) for v in window_features(
                                [f for _, f in collect_buf])]})

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
                        if not record_reference:
                            ref_traj = (references.get(det) or {}).get("trajectory")
                            if ref_traj:
                                print("Reference rep found — scoring similarity.")
                        if chat:
                            live_state["exercise"] = det
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
                    if chat:
                        live_state["plank_hold_s"] = round(plank.total, 1)
                else:                                       # ---- rep exercise
                    faults_now = live_faults(exercise, ang, counter.state)
                    for fault in faults_now:
                        counter.note_fault(fault)
                    ev = counter.update(ang[spec.signal], t)
                    if counter.state != "IDLE" or ev:
                        rep_traj.append(ang[spec.signal])   # in-rep trajectory
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
                        sim = None
                        if ref_traj:
                            sim = similarity(rep_traj, ref_traj)
                            last_sim = sim
                        if record_reference and (best_ref is None
                                                 or ev.score >= best_ref[0]):
                            best_ref = (ev.score, list(rep_traj))
                        rep_traj = []
                        log.add_rep(ev, velocity=vel, similarity=sim)
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
                              + (f"ref-sim {sim}  " if sim is not None else "")
                              + f"ecc {ev.eccentric_s:.1f}s / con {ev.concentric_s:.1f}s  "
                              f"vel {vel:.0f} deg/s  "
                              f"min {spec.signal} {ev.min_angle:.0f}  "
                              f"faults {ev.faults or 'none'}")
                        if chat:
                            live_state.update(
                                reps=ev.count, last_score=ev.score,
                                last_similarity=sim,
                                fault_counts=WorkoutLog._fault_counts(
                                    log.session["reps"]),
                                velocity_loss_pct=round(fatigue.loss * 100, 1)
                                if fatigue.loss else None)
                    elif counter.state == "IDLE":
                        rep_traj = []           # discarded blip / idle frames
                    if chat:
                        live_state["phase"] = counter.state
                    hud1 = (f"{exercise.upper()}  reps: {counter.count}   "
                            f"phase: {counter.state}"
                            + (f"   last score: {last_score}" if last_score is not None else "")
                            + (f"   ref-sim: {last_sim}" if last_sim is not None else ""))
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
                if chat:
                    st = chat.status
                    col = ((0, 220, 255) if "hearing" in st else
                           (80, 200, 80) if st == "listening" else
                           (200, 200, 200))
                    cv2.putText(frame, f"mic: {st}", (10, 30 + 28 * 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
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
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("c") and chat:
                    chat.push_to_talk()
    except KeyboardInterrupt:
        print("\nInterrupted — finishing session...")

    cap.release()
    landmarker.close()
    if writer is not None:
        writer.release()
        print(f"Annotated video written to {output}")
    if not headless:
        cv2.destroyAllWindows()
    if record_reference:
        if best_ref:
            save_reference(exercise, best_ref[1], best_ref[0], reference_file)
            print(f"Reference rep saved for {exercise} (score {best_ref[0]}) "
                  f"→ {reference_file}")
            voice.say("Reference rep saved.")
        else:
            print("No completed rep — reference not saved.")
    if collect_rows:
        with open(collect, "a", encoding="utf-8") as fh:
            for row in collect_rows:
                fh.write(json.dumps(row) + "\n")
        print(f"Collected {len(collect_rows)} labeled windows -> {collect}  "
              f"(retrain: python pose_coach.py --train-classifier "
              f"--collect {collect})")
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

    print("14) DTW similarity:", end=" ")
    t50 = [130 + 45 * math.cos(2 * math.pi * i / 50) for i in range(50)]
    assert similarity(t50, t50) == 100                       # identical
    t30 = [130 + 45 * math.cos(2 * math.pi * i / 30) for i in range(30)]
    assert similarity(t30, t50) >= 95                        # tempo-invariant
    shallow = [130 + 20 * math.cos(2 * math.pi * i / 50) for i in range(50)]
    s_shallow = similarity(shallow, t50)
    assert s_shallow < 80, s_shallow                         # half-depth penalized
    flat = [170.0] * 50
    assert similarity(flat, t50) < s_shallow                 # no movement worst
    assert dtw_distance([], [1.0]) == float("inf")
    assert len(resample([1, 2, 3], 50)) == 50 and resample([], 50).sum() == 0
    print("OK (identity=100, tempo-proof, depth-sensitive)")

    print("15) reference store:", end=" ")
    with tempfile.TemporaryDirectory() as td:
        rp = os.path.join(td, "refs.json")
        assert load_references(rp) == {}
        save_reference("squat", t50, 92, rp)
        save_reference("curl", t30, 88, rp)
        refs = load_references(rp)
        assert set(refs) == {"squat", "curl"}
        assert len(refs["squat"]["trajectory"]) == REF_SAMPLES
        assert refs["squat"]["score"] == 92
        noisy = [v + 3 * math.sin(7.3 * i) for i, v in enumerate(t50)]
        assert similarity(noisy, refs["squat"]["trajectory"]) > 80
    print("OK (save/load roundtrip, noisy rep still >80)")

    print("16) similarity in workout log:", end=" ")
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "log.json")
        wl = WorkoutLog(path)
        wl.add_rep(RepEvent(1, 3.0, 2.0, 1.0, 85.0, True, [], 90),
                   velocity=40.0, similarity=93)
        wl.add_rep(RepEvent(2, 3.0, 2.0, 1.0, 88.0, True, [], 85),
                   velocity=38.0, similarity=81)
        s = wl.finish("squat", 20.0)
        assert s["summary"]["avg_similarity"] == 87.0
        assert s["reps"][0]["similarity"] == 93
        wl2 = WorkoutLog(path)
        wl2.add_rep(RepEvent(1, 3.0, 2.0, 1.0, 85.0, True, [], 90))
        s2 = wl2.finish("squat", 20.0)
        assert s2["summary"]["avg_similarity"] is None       # no reference used
    print("OK (avg_similarity aggregated; None without reference)")

    print("17) ML feature windows:", end=" ")
    rng = np.random.default_rng(1)
    fr = synth_frames("squat", rng)
    assert len(fr) == 60 and set(fr[0]) == set(FEAT_KEYS)
    x = window_features(fr)
    assert x.shape == (NDIM,) and np.isfinite(x).all()
    sq = window_features(synth_frames("squat", np.random.default_rng(2)))
    pu = window_features(synth_frames("pushup", np.random.default_rng(2)))
    assert abs(sq[FEAT_KEYS.index("knee")] - pu[FEAT_KEYS.index("knee")]) > 20
    print(f"OK ({NDIM}-dim, classes separable)")

    print("18) classifier training:", end=" ")
    with tempfile.TemporaryDirectory() as td:
        mp_ = os.path.join(td, "clf.npz")
        buf = io.StringIO()
        with redirect_stdout(buf):
            acc = train_classifier(mp_, samples_per_class=40, epochs=200)
        assert acc >= 0.9, f"val accuracy only {acc:.1%}"
        m = TinyMLP.load(mp_)
        X, y, _ = build_dataset(samples_per_class=10, seed=7)
        assert (m.predict_proba(X).argmax(1) == y).mean() >= 0.9
    print(f"OK (val accuracy {acc:.1%}, save/load roundtrip)")

    print("19) ML auto-detection:", end=" ")
    with tempfile.TemporaryDirectory() as td:
        mp_ = os.path.join(td, "clf.npz")
        with redirect_stdout(io.StringIO()):
            train_classifier(mp_, samples_per_class=40, epochs=200)
        model = TinyMLP.load(mp_)
        rng = np.random.default_rng(3)
        for expected in ("squat", "pushup", "curl", "shoulder_press"):
            det = MLDetector(model)
            frames = synth_frames(expected, rng, seconds=6.0)
            got = None
            for i, f in enumerate(frames):
                got = det.update(f, i / 30.0) or got
            assert got == expected, f"{expected} misread as {got}"
    print("OK (4 movements classified by the MLP)")

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
    ap.add_argument("--coach", action="store_true",
                    help="conversational LLM coach: hands-free mic (just "
                         "speak), typed questions, or 'c' to interrupt "
                         "(see docs/COACH.md)")
    ap.add_argument("--record-reference", action="store_true",
                    help="save this set's best rep as the golden reference; "
                         "future sessions score every rep against it (DTW)")
    ap.add_argument("--reference-file", default=REFERENCE_FILE,
                    help=f"reference reps file (default {REFERENCE_FILE})")
    ap.add_argument("--train-classifier", action="store_true",
                    help="train the ML exercise classifier on synthetic "
                         "motion data (+ any --collect recordings), save "
                         "weights, and report validation accuracy")
    ap.add_argument("--detector", choices=("auto", "rules", "ml"),
                    default="auto",
                    help="auto-detect backend: ml = trained classifier, "
                         "rules = heuristics, auto = ml when weights exist")
    ap.add_argument("--model-file", default=MODEL_FILE,
                    help=f"classifier weights file (default {MODEL_FILE})")
    ap.add_argument("--collect", metavar="JSONL",
                    help="with --exercise: append labeled feature windows "
                         "from this session; with --train-classifier: also "
                         "train on that file")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    elif args.stats:
        print_stats(args.log_file)
    elif args.train_classifier:
        train_classifier(args.model_file, collected=args.collect)
    else:
        run(args.exercise, args.video, not args.no_voice, args.log_file,
            headless=args.headless, output=args.output, coach=args.coach,
            record_reference=args.record_reference,
            reference_file=args.reference_file, detector_kind=args.detector,
            model_file=args.model_file, collect=args.collect)
