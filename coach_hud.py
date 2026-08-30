"""coach_hud.py — the on-screen experience of the live coach window.

Everything pose_coach.py knows about a set — phase, reps vs goal, which
body part is at fault, range of motion against the exercise's own
thresholds, tempo, fatigue, golden-rep similarity, heart-rate zone, the
guided program, the rest timer, what the LLM coach is saying and whether
the mic hears you — is drawn here as one consistent layer over the camera
frame. It also covers the moments around a set: framing guidance before a
person is tracked, a help overlay, and an end-of-set summary card.

Text is rendered with Pillow (TrueType, anti-aliased, full Unicode —
Pillow already ships with mediapipe) and falls back to OpenCV's Hershey
fonts when Pillow is missing. No new dependencies.

    python coach_hud.py --selftest          # renders every state offscreen
    python coach_hud.py --preview DIR       # writes PNGs of every state
"""
from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np

# BlazePose landmark indices (same numbering as pose_coach.py)
NOSE, L_EAR, R_EAR = 0, 7, 8
L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI = 11, 12, 13, 14, 15, 16
L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK = 23, 24, 25, 26, 27, 28
L_HEE, R_HEE, L_TOE, R_TOE = 29, 30, 31, 32
VIS_MIN = 0.5

EDGES = [(L_SHO, R_SHO), (L_SHO, L_ELB), (L_ELB, L_WRI), (R_SHO, R_ELB),
         (R_ELB, R_WRI), (L_SHO, L_HIP), (R_SHO, R_HIP), (L_HIP, R_HIP),
         (L_HIP, L_KNE), (L_KNE, L_ANK), (R_HIP, R_KNE), (R_KNE, R_ANK),
         (L_ANK, L_HEE), (L_HEE, L_TOE), (R_ANK, R_HEE), (R_HEE, R_TOE)]

# body regions a form fault lights up on the skeleton
_REGION_EDGES = {
    "trunk": [(L_SHO, L_HIP), (R_SHO, R_HIP), (L_SHO, R_SHO), (L_HIP, R_HIP)],
    "neck": [(L_EAR, L_SHO), (R_EAR, R_SHO)],
    "legs": [(L_HIP, L_KNE), (L_KNE, L_ANK), (R_HIP, R_KNE), (R_KNE, R_ANK)],
    "knees": [(L_HIP, L_KNE), (L_KNE, L_ANK), (R_HIP, R_KNE), (R_KNE, R_ANK)],
    "arms": [(L_SHO, L_ELB), (L_ELB, L_WRI), (R_SHO, R_ELB), (R_ELB, R_WRI)],
    "wrists": [(L_ELB, L_WRI), (R_ELB, R_WRI)],
    "head": [(L_EAR, L_SHO), (R_EAR, R_SHO)],
}
FAULT_REGIONS = {
    "back_lean": ("trunk",), "back_round": ("trunk", "neck"),
    "torso_lean": ("trunk",), "lean_back": ("trunk",),
    "body_sag": ("trunk", "legs"), "knees_cave": ("knees",),
    "shallow": ("driver",), "too_fast": ("driver",),
    "elbow_swing": ("arms",), "elbow_flare": ("arms",),
    "uneven": ("wrists",), "chin": ("head", "arms"), "shrug_neck": ("neck",),
}
_SIGNAL_REGION = {"knee": "legs", "elbow": "arms", "hip": "trunk",
                  "body_line": "trunk"}
# the three landmarks whose angle is the FSM signal, per side
_SIGNAL_TRIPLET = {
    "knee": ((L_HIP, L_KNE, L_ANK), (R_HIP, R_KNE, R_ANK)),
    "hip": ((L_SHO, L_HIP, L_KNE), (R_SHO, R_HIP, R_KNE)),
    "elbow": ((L_SHO, L_ELB, L_WRI), (R_SHO, R_ELB, R_WRI)),
    "body_line": ((L_SHO, L_HIP, L_ANK), (R_SHO, R_HIP, R_ANK)),
}

# human names for the log's fault keys (the HUD never shows snake_case)
FAULT_NAMES = {
    "back_lean": "back leaning", "back_round": "rounded back",
    "body_sag": "hips sagging", "knees_cave": "knees caving in",
    "shallow": "not deep enough", "elbow_swing": "elbows swinging",
    "elbow_flare": "elbows flaring", "torso_lean": "torso leaning",
    "lean_back": "leaning back", "uneven": "uneven sides",
    "chin": "chin under bar", "shrug_neck": "neck shrugged",
    "too_fast": "too fast",
}
# what the FSM phase means for the athlete (depends on which direction is
# the lift: presses/squats lift on the angle's way up, curls/pull-ups on
# the way down)
PHASE_LABELS = {
    "ascent": {"IDLE": "READY", "DESCENT": "LOWERING", "BOTTOM": "BOTTOM",
               "ASCENT": "LIFTING"},
    "descent": {"IDLE": "READY", "DESCENT": "LIFTING", "BOTTOM": "TOP",
                "ASCENT": "LOWERING"},
}
EXERCISE_KEYS = ("squat", "pushup", "bench", "deadlift", "lunge",
                 "shoulder_press", "curl", "pullup", "plank")   # keys 1-9
KEY_HELP = [
    ("1-9", "switch exercise: 1 squat · 2 push-up · 3 bench · 4 deadlift · "
            "5 lunge · 6 shoulder press · 7 curl · 8 pull-up · 9 plank "
            "(the current number starts a fresh set)"),
    ("a", "auto-detect the exercise from your movement"),
    ("r", "start / cancel a 60 s rest"),
    ("v", "mute / unmute the voice"),
    ("m", "mirror the camera on / off"),
    ("c", "talk to the coach now (--coach)"),
    ("h", "this help"),
    ("q / Esc", "finish the set and see the summary"),
]
CLI_HELP = [
    ("--coach", "talk to a local LLM coach by voice or text"),
    ("--program \"…\"", "guided workout (\"squat 3x10 rest 90, plank 2x40s\"): "
                      "sets, rests and exercise switches run for you"),
    ("--record-reference", "save a golden rep; later reps get a similarity score"),
    ("--sensors ble", "heart-rate strap: zones + recovery-based rest"),
    ("--exercise auto", "recognise the exercise automatically"),
    ("coach_dashboard.py", "progress charts in the browser"),
]

# palette (BGR)
C_TEXT = (245, 245, 245)
C_MUTED = (190, 178, 165)
C_DIM = (110, 105, 98)
C_PANEL = (28, 24, 20)
C_GREEN = (128, 222, 74)
C_BLUE = (250, 165, 96)
C_AMBER = (36, 191, 251)
C_RED = (113, 113, 248)
C_PURPLE = (252, 132, 192)
C_CYAN = (235, 220, 34)
C_SKEL = (110, 235, 90)
C_SKEL_DIM = (120, 120, 120)
C_HR_ZONE = {1: C_BLUE, 2: C_GREEN, 3: C_AMBER, 4: (30, 140, 250), 5: C_RED}


def score_color(score: float | None):
    if score is None:
        return C_MUTED
    return C_GREEN if score >= 85 else C_AMBER if score >= 65 else C_RED


# ------------------------------------------------------------------- text
_FONT_PAIRS = [
    (os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "segoeui.ttf"),
     os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "segoeuib.ttf")),
    (os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "arial.ttf"),
     os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "arialbd.ttf")),
    ("/System/Library/Fonts/Supplemental/Arial.ttf",
     "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("/usr/share/fonts/dejavu/DejaVuSans.ttf",
     "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"),
    ("/usr/share/fonts/TTF/DejaVuSans.ttf", "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"),
    ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
     "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
    ("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
     "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"),
    ("/usr/share/fonts/truetype/freefont/FreeSans.ttf",
     "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"),
]

_ASCII_MAP = str.maketrans({"—": "-", "–": "-", "°": " deg", "↓": "v", "↑": "^",
                            "·": "-", "…": "...", "✓": "ok", "×": "x",
                            "’": "'", "“": '"', "”": '"', "→": "->"})


def font_files() -> tuple[str | None, str | None]:
    """(regular, bold) TrueType paths for this OS; COACH_FONT overrides."""
    env = os.environ.get("COACH_FONT")
    if env and os.path.exists(env):
        return env, env
    for reg, bold in _FONT_PAIRS:
        if os.path.exists(reg):
            return reg, bold if os.path.exists(bold) else reg
    return None, None


class TextLayer:
    """Batched text: shapes go straight onto the frame with OpenCV, text is
    queued and drawn in ONE Pillow pass per frame (one BGR<->RGB round
    trip). Without Pillow, Hershey fonts draw immediately."""

    def __init__(self, use_pil: bool | None = None):
        self._fonts: dict = {}
        self.ops: list[tuple] = []
        self.pil = None
        if use_pil is not False:
            try:
                from PIL import Image, ImageDraw, ImageFont
                self.pil = (Image, ImageDraw, ImageFont)
            except ImportError:
                self.pil = None
        self.reg, self.bold = font_files() if self.pil else (None, None)
        self.backend = ("truetype" if self.reg else "pil-default") if self.pil \
            else "hershey"

    def _font(self, size: int, bold: bool):
        key = (size, bold)
        f = self._fonts.get(key)
        if f is None:
            ImageFont = self.pil[2]
            path = self.bold if bold else self.reg
            try:
                f = ImageFont.truetype(path, size) if path else \
                    ImageFont.load_default(size=size)
            except Exception:
                f = ImageFont.load_default()
            self._fonts[key] = f
        return f

    def measure(self, s: str, size: int, bold: bool = False) -> tuple[int, int]:
        if self.pil:
            f = self._font(size, bold)
            try:
                l, t, r, b = f.getbbox(s)
                return int(r - l), int(size * 1.2)
            except Exception:
                return int(len(s) * size * 0.55), int(size * 1.2)
        import cv2
        (w, h), _ = cv2.getTextSize(s.translate(_ASCII_MAP), cv2.FONT_HERSHEY_SIMPLEX,
                                    size / 30.0, 2 if bold else 1)
        return w, int(size * 1.2)

    def add(self, frame, x: int, y: int, s: str, size: int, color,
            bold: bool = False, align: str = "l"):
        """Queue text with its TOP at y; align l/m/r around x."""
        if not s:
            return
        if self.pil:
            self.ops.append((x, y, s, size, color, bold, align))
            return
        import cv2
        s = s.translate(_ASCII_MAP).encode("ascii", "ignore").decode()
        scale = size / 30.0
        th = 2 if bold else 1
        (w, h), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
        if align == "m":
            x -= w // 2
        elif align == "r":
            x -= w
        cv2.putText(frame, s, (int(x), int(y + h)), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 0, 0), th + 2, cv2.LINE_AA)
        cv2.putText(frame, s, (int(x), int(y + h)), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, th, cv2.LINE_AA)

    def flush(self, frame):
        """Draw all queued text; returns the (possibly new) frame."""
        if not self.ops:
            return frame
        Image, ImageDraw, _ = self.pil
        img = Image.fromarray(frame[:, :, ::-1])
        draw = ImageDraw.Draw(img)
        for x, y, s, size, color, bold, align in self.ops:
            f = self._font(size, bold)
            anchor = {"l": "la", "m": "ma", "r": "ra"}[align]
            rgb = (int(color[2]), int(color[1]), int(color[0]))
            try:
                draw.text((x, y), s, font=f, fill=rgb, anchor=anchor)
            except Exception:              # bitmap fallback font: no anchors
                w = self.measure(s, size, bold)[0]
                dx = -w // 2 if align == "m" else -w if align == "r" else 0
                draw.text((x + dx, y), s, font=f, fill=rgb)
        self.ops.clear()
        return np.ascontiguousarray(np.asarray(img)[:, :, ::-1])

    def wrap(self, s: str, size: int, bold: bool, max_w: int) -> list[str]:
        words, lines, cur = s.split(), [], ""
        for w in words:
            cand = f"{cur} {w}".strip()
            if cur and self.measure(cand, size, bold)[0] > max_w:
                lines.append(cur)
                cur = w
            else:
                cur = cand
        if cur:
            lines.append(cur)
        return lines or [""]


# ----------------------------------------------------------------- shapes
def rounded_rect(img, x0, y0, x1, y1, color, r=12, thickness=-1):
    import cv2
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    r = int(max(0, min(r, (x1 - x0) // 2, (y1 - y0) // 2)))
    if r == 0:
        cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness, cv2.LINE_AA)
        return
    if thickness < 0:
        cv2.rectangle(img, (x0 + r, y0), (x1 - r, y1), color, -1)
        cv2.rectangle(img, (x0, y0 + r), (x1, y1 - r), color, -1)
        for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r),
                       (x0 + r, y1 - r), (x1 - r, y1 - r)):
            cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
    else:
        cv2.line(img, (x0 + r, y0), (x1 - r, y0), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x0 + r, y1), (x1 - r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x0, y0 + r), (x0, y1 - r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y0 + r), (x1, y1 - r), color, thickness, cv2.LINE_AA)
        for (cx, cy), a in (((x0 + r, y0 + r), 180), ((x1 - r, y0 + r), 270),
                            ((x0 + r, y1 - r), 90), ((x1 - r, y1 - r), 0)):
            cv2.ellipse(img, (cx, cy), (r, r), a, 0, 90, color, thickness,
                        cv2.LINE_AA)


def panel(img, x0, y0, x1, y1, color=C_PANEL, alpha=0.66, r=14, border=None):
    """Translucent rounded card (alpha-blended on the region it covers)."""
    import cv2
    h, w = img.shape[:2]
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(w, int(x1)), min(h, int(y1))
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    over = roi.copy()
    rounded_rect(over, 0, 0, x1 - x0 - 1, y1 - y0 - 1, color, r)
    cv2.addWeighted(over, alpha, roi, 1 - alpha, 0, roi)
    if border:
        rounded_rect(img, x0, y0, x1 - 1, y1 - 1, border, r, 1)


def dim(img, alpha=0.5):
    import cv2
    cv2.addWeighted(np.zeros_like(img), alpha, img, 1 - alpha, 0, img)


def ring(img, cx, cy, radius, frac, color, thickness, track=(60, 56, 50)):
    import cv2
    c = (int(cx), int(cy))
    cv2.ellipse(img, c, (int(radius), int(radius)), 0, 0, 360, track,
                thickness, cv2.LINE_AA)
    if frac > 0:
        cv2.ellipse(img, c, (int(radius), int(radius)), -90, 0,
                    int(360 * min(1.0, frac)), color, thickness, cv2.LINE_AA)


def hbar(img, x0, y0, w, h, frac, color, track=(60, 56, 50)):
    rounded_rect(img, x0, y0, x0 + w, y0 + h, track, h // 2)
    fw = int(w * max(0.0, min(1.0, frac)))
    if fw > 0:
        rounded_rect(img, x0, y0, x0 + max(fw, h), y0 + h, color, h // 2)


# ------------------------------------------------------------------ state
@dataclass
class HudState:
    """Everything the HUD shows for one frame — filled by pose_coach.run()."""
    exercise: str | None = None           # None => auto-detecting
    mode: str = "reps"                    # reps | hold | auto
    concentric: str = "ascent"
    phase: str = "IDLE"
    reps: int = 0
    rep_goal: int | None = None
    rep_scores: list = field(default_factory=list)   # this set
    last_score: int | None = None
    similarity: int | None = None
    has_reference: bool = False
    recording_reference: bool = False
    tempo: tuple | None = None            # (eccentric_s, concentric_s)
    tempo_target: float | None = None
    velocity_ratio: float | None = None   # current / fresh baseline
    fatigue_warned: bool = False
    signal: str | None = None             # angle key driving the FSM
    signal_value: float | None = None
    thresholds: tuple | None = None       # (start_below, bottom_below, lockout_above)
    side: str = "L"
    hold_total: float = 0.0
    hold_streak: float = 0.0
    hold_best: float = 0.0
    hold_good_above: float = 160.0
    faults_now: list = field(default_factory=list)
    cue: str = ""
    cue_kind: str = ""                    # fault | praise | fatigue | info
    rest_left: float = 0.0
    rest_next: str | None = None
    program: dict | None = None           # WorkoutProgram.status()
    program_overview: str | None = None
    hr: int | None = None
    hr_zone: int | None = None
    chat: dict | None = None              # status, level, user, reply, hint
    voice_on: bool = True
    mirrored: bool = False
    fps: float = 0.0
    source: str = ""
    tracking: str = "none"                # ok | partial | none
    missing: list = field(default_factory=list)
    brightness: float = 128.0
    camera_hint: str = ""
    detector_label: str = ""
    show_help: bool = False
    elapsed_s: float = 0.0
    cues_on: bool = True
    angles: dict | None = None            # live joint angles (auto mode)
    load_kg: float | None = None          # external load per rep
    best_reps: int = 0                    # session rep record to beat
    muscles: list = field(default_factory=list)   # from the exercise catalogue


# -------------------------------------------------------------------- HUD
class Hud:
    """Stateful renderer: keeps small animation/hysteresis timers so the
    caller only reports facts."""

    def __init__(self, min_width: int = 960, use_pil: bool | None = None):
        self.min_width = min_width
        self.text = TextLayer(use_pil)
        self._last_reps = 0
        self._rep_pop_t = -10.0
        self._lost_since: float | None = None
        self._rest_total = 0.0
        self._cue_since = 0.0
        self._cue_last = ""

    # ---- helpers
    def _prep(self, frame, now):
        import cv2
        h, w = frame.shape[:2]
        if w < self.min_width:
            s = self.min_width / w
            frame = cv2.resize(frame, (self.min_width, int(h * s)),
                               interpolation=cv2.INTER_LINEAR)
        else:
            frame = frame.copy()
        h, w = frame.shape[:2]
        u = max(0.6, min(w / 1280.0, h / 720.0))
        return frame, w, h, u

    def _px(self, pts, w, h, i):
        return int(pts[i, 0] * w), int(pts[i, 1] * h)

    # ---- skeleton with fault highlighting + signal angle arc
    def draw_skeleton(self, frame, pts, st: HudState, w, h, u):
        import cv2
        if pts is None:
            return
        hot = set()
        for f in st.faults_now:
            for region in FAULT_REGIONS.get(f, ()):
                if region == "driver":
                    region = _SIGNAL_REGION.get(st.signal or "", "trunk")
                hot.update(_REGION_EDGES.get(region, []))
        thick = max(2, int(3 * u))
        for i, j in EDGES:
            vi, vj = pts[i, 3], pts[j, 3]
            if vi < 0.2 or vj < 0.2:
                continue
            a, b = self._px(pts, w, h, i), self._px(pts, w, h, j)
            if (i, j) in hot or (j, i) in hot:
                cv2.line(frame, a, b, (40, 40, 160), thick * 3, cv2.LINE_AA)
                cv2.line(frame, a, b, C_RED, thick, cv2.LINE_AA)
            elif vi > VIS_MIN and vj > VIS_MIN:
                cv2.line(frame, a, b, (30, 60, 30), thick + 2, cv2.LINE_AA)
                cv2.line(frame, a, b, C_SKEL, thick, cv2.LINE_AA)
            else:
                cv2.line(frame, a, b, C_SKEL_DIM, 1, cv2.LINE_AA)
        for i in {k for e in EDGES for k in e}:
            if pts[i, 3] > VIS_MIN:
                p = self._px(pts, w, h, i)
                cv2.circle(frame, p, max(3, int(4 * u)), (20, 30, 20), -1, cv2.LINE_AA)
                cv2.circle(frame, p, max(2, int(2.5 * u)), C_TEXT, -1, cv2.LINE_AA)
        # angle arc at the joint that drives the rep counter
        trip = _SIGNAL_TRIPLET.get(st.signal or "")
        if trip and st.signal_value is not None and not st.show_help:
            a_i, b_i, c_i = trip[0] if st.side == "L" else trip[1]
            if min(pts[a_i, 3], pts[b_i, 3], pts[c_i, 3]) > VIS_MIN:
                a, b, c = (self._px(pts, w, h, k) for k in (a_i, b_i, c_i))
                a1 = math.degrees(math.atan2(a[1] - b[1], a[0] - b[0]))
                a2 = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]))
                diff = (a2 - a1) % 360
                if diff > 180:
                    a1, diff = a2, 360 - diff
                r = int(22 * u)
                col = self._rom_color(st)
                cv2.ellipse(frame, b, (r, r), 0, a1, a1 + diff, col,
                            max(2, int(2 * u)), cv2.LINE_AA)
                lab = f"{st.signal_value:.0f}°"
                tw, th = self.text.measure(lab, int(15 * u), True)
                lx = b[0] + r + int(6 * u)
                ly = b[1] + int(r * 0.6)
                panel(frame, lx - 4 * u, ly - 2 * u, lx + tw + 4 * u,
                      ly + th, alpha=0.55, r=int(5 * u))
                self.text.add(frame, lx, ly, lab, int(15 * u), col, True)

    def _rom_color(self, st: HudState):
        if st.mode == "hold":
            return C_GREEN if (st.signal_value or 0) >= st.hold_good_above else C_RED
        if not st.thresholds or st.signal_value is None:
            return C_MUTED
        start, bottom, lock = st.thresholds
        v = st.signal_value
        if v < bottom:
            return C_GREEN
        if v < start:
            return C_AMBER
        return C_BLUE

    # ---- top-left: exercise, phase, rep counter, program
    def draw_exercise_card(self, frame, st: HudState, w, h, u, now):
        x0, y0 = int(16 * u), int(16 * u)
        cw = int(330 * u)
        name = (st.exercise or "auto").replace("_", " ").upper()
        if st.exercise is None:
            name = "DETECTING…"
        y1 = y0 + int(128 * u)
        prog = st.program
        if prog:
            y1 += int(38 * u)
        panel(frame, x0, y0, x0 + cw, y1)
        pad = int(14 * u)
        self.text.add(frame, x0 + pad, y0 + pad - 2 * u, name, int(24 * u),
                      C_TEXT, True)
        # detector / reference / recording pills on the name row
        px = x0 + cw - pad
        for label, col in self._name_pills(st):
            tw, th = self.text.measure(label, int(11 * u), True)
            rounded_rect(frame, px - tw - 10 * u, y0 + pad + 2 * u,
                         px, y0 + pad + th + 2 * u, col, int(6 * u))
            self.text.add(frame, px - tw - 5 * u, y0 + pad + 3 * u, label,
                          int(11 * u), (15, 15, 15), True)
            px -= tw + 16 * u
        # phase pill
        py = y0 + pad + int(34 * u)
        if st.mode == "auto":
            plabel, pcol = "MOVE TO START", C_MUTED
        elif st.mode == "hold":
            good = (st.signal_value or 0) >= st.hold_good_above
            plabel, pcol = ("HOLDING", C_GREEN) if good and st.tracking == "ok" \
                else ("FIX YOUR LINE" if st.tracking == "ok" else "READY", C_AMBER)
        else:
            plabel = PHASE_LABELS[st.concentric].get(st.phase, st.phase)
            pcol = {"READY": C_MUTED, "LOWERING": C_BLUE, "LIFTING": C_GREEN,
                    "BOTTOM": C_AMBER, "TOP": C_AMBER}.get(plabel, C_MUTED)
        tw, th = self.text.measure(plabel, int(12 * u), True)
        rounded_rect(frame, x0 + pad, py, x0 + pad + tw + 16 * u,
                     py + th + 4 * u, pcol, int(7 * u))
        self.text.add(frame, x0 + pad + 8 * u, py + 2 * u, plabel, int(12 * u),
                      (15, 15, 15), True)
        if st.rest_left <= 0 and st.mode != "auto" and st.tracking == "ok" \
                and st.phase == "IDLE" and st.reps == 0 and st.mode == "reps":
            self.text.add(frame, x0 + pad, py + th + 10 * u,
                          "start your first rep", int(12 * u), C_MUTED)
        elif st.muscles and st.mode != "auto":
            self.text.add(frame, x0 + pad, py + th + 10 * u,
                          " · ".join(st.muscles[:3]), int(12 * u), C_DIM)
        elif st.mode == "auto" and st.tracking == "ok":
            self.text.add(frame, x0 + pad, py + th + 10 * u,
                          "I'll recognise the exercise", int(12 * u), C_MUTED)
            if st.angles:
                live = "   ".join(f"{k.replace('_', ' ')} {v:.0f}°"
                                  for k, v in st.angles.items())
                self.text.add(frame, x0 + pad, py + th + 30 * u, live,
                              int(12 * u), C_DIM)
        # rep counter / hold timer (right side of the card)
        if st.mode == "hold":
            big = f"{st.hold_total:.0f}"
            unit = "s"
            sub = f"best unbroken {st.hold_best:.0f}s"
        elif st.mode == "auto":
            big, unit, sub = "", "", ""
        else:
            big = str(st.reps)
            unit = f"/{st.rep_goal}" if st.rep_goal else ""
            sub = "REPS"
        if st.reps != self._last_reps:
            self._last_reps = st.reps
            self._rep_pop_t = now
        pop = max(0.0, 1 - (now - self._rep_pop_t) / 0.35)
        size = int((46 + 14 * pop) * u)
        bw = self.text.measure(big, size, True)[0]
        uw = self.text.measure(unit, int(20 * u), True)[0] if unit else 0
        bx = x0 + cw - pad - uw
        by = y0 + pad + int(22 * u) - int(8 * pop * u)
        if big:
            self.text.add(frame, bx, by, big, size, C_TEXT, True, "r")
            if unit:
                self.text.add(frame, bx + 2 * u, by + int(24 * u), unit,
                              int(20 * u), C_MUTED, True)
            extra = ""
            if st.mode == "reps" and st.load_kg:
                extra = f"{st.load_kg:g} kg · "
            if st.mode == "reps" and st.best_reps and st.reps < st.best_reps:
                extra += f"record {st.best_reps} · "
            self.text.add(frame, x0 + cw - pad, by + int(58 * u), extra + sub,
                          int(11 * u), C_MUTED, False, "r")
        # goal ring / rep-score dots
        if st.mode == "reps" and st.rep_goal:
            frac = st.reps / max(st.rep_goal, 1)
            ring(frame, bx - bw - int(26 * u), by + int(30 * u), int(16 * u),
                 frac, C_GREEN if frac >= 1 else C_BLUE, max(3, int(4 * u)))
        if st.mode == "reps" and st.rep_scores:
            dx = x0 + pad
            dy = y1 - int(16 * u) - (int(38 * u) if prog else 0)
            for sc in st.rep_scores[-14:]:
                import cv2
                cv2.circle(frame, (int(dx + 4 * u), int(dy)), max(3, int(4 * u)),
                           score_color(sc), -1, cv2.LINE_AA)
                dx += int(12 * u)
        if prog:
            self._program_strip(frame, st, x0, y1 - int(38 * u), cw, u)

    def _name_pills(self, st: HudState):
        pills = []
        if st.recording_reference:
            pills.append(("● REC GOLDEN REP", C_RED))
        elif st.has_reference:
            pills.append(("vs GOLDEN REP", C_AMBER))
        if st.detector_label and st.exercise is None:
            pills.append((st.detector_label.upper(), C_BLUE))
        if not st.cues_on:
            pills.append(("CUES OFF", C_MUTED))
        return pills

    def _program_strip(self, frame, st, x0, y, cw, u):
        import cv2
        p = st.program
        pad = int(14 * u)
        cv2.line(frame, (x0 + pad, y), (x0 + cw - pad, y), (70, 64, 58), 1)
        label = (f"PROGRAM  block {p['block']}/{p['blocks']} · "
                 f"{p['exercise'].replace('_', ' ')} · set {p['set']}/{p['sets']}"
                 f" · {p['target']}")
        self.text.add(frame, x0 + pad, y + int(7 * u), label, int(12 * u), C_AMBER,
                      True)
        dx = x0 + pad
        dy = y + int(29 * u)
        for b in range(1, p["blocks"] + 1):
            col = C_GREEN if b < p["block"] else C_AMBER if b == p["block"] else C_DIM
            rounded_rect(frame, dx, dy, dx + int(22 * u), dy + int(4 * u), col,
                         int(2 * u))
            dx += int(26 * u)

    # ---- top-right: last rep quality, tempo, fatigue, heart rate
    def draw_metrics_card(self, frame, st: HudState, w, h, u):
        rows = []
        if st.mode == "reps" and st.last_score is not None:
            rows.append("score")
            if st.similarity is not None or st.has_reference:
                rows.append("sim")
            if st.tempo or st.tempo_target:
                rows.append("tempo")
            if st.velocity_ratio is not None or st.fatigue_warned:
                rows.append("speed")
        if st.hr is not None:
            rows.append("hr")
        if not rows:
            return
        cw = int(290 * u)
        x0, y0 = w - cw - int(16 * u), int(16 * u)
        pad = int(14 * u)
        rh = int(30 * u)
        vx = x0 + int(cw * 0.60)          # values right-align here
        bx = x0 + int(cw * 0.64)          # bars start here
        bw = x0 + cw - pad - bx
        y1 = y0 + pad * 2 + int(56 * u) * ("score" in rows) \
            + rh * (len(rows) - ("score" in rows)) \
            + int(14 * u) * bool(st.tempo_target and "tempo" in rows)
        panel(frame, x0, y0, x0 + cw, y1)
        y = y0 + pad
        if "score" in rows:
            self.text.add(frame, x0 + pad, y, "LAST REP", int(11 * u), C_MUTED)
            sc = st.last_score
            self.text.add(frame, x0 + pad, y + int(12 * u), str(sc), int(36 * u),
                          score_color(sc), True)
            sw = self.text.measure(str(sc), int(36 * u), True)[0]
            self.text.add(frame, x0 + pad + sw + int(4 * u), y + int(30 * u),
                          "/100", int(13 * u), C_MUTED)
            hbar(frame, bx, y + int(26 * u), bw, int(8 * u), sc / 100,
                 score_color(sc))
            y += int(56 * u)
        if "sim" in rows:
            self.text.add(frame, x0 + pad, y, "GOLDEN REP", int(11 * u), C_MUTED)
            sim = st.similarity
            self.text.add(frame, vx, y - int(2 * u), "—" if sim is None else f"{sim}%",
                          int(16 * u), C_AMBER, True, "r")
            hbar(frame, bx, y + int(6 * u), bw, int(8 * u), (sim or 0) / 100, C_AMBER)
            y += rh
        if "tempo" in rows:
            self.text.add(frame, x0 + pad, y, "TEMPO", int(11 * u), C_MUTED)
            if st.tempo:
                e, c = st.tempo
                lab = f"↓ {e:.1f}s   ↑ {c:.1f}s"
            else:
                lab = "—"
            self.text.add(frame, x0 + cw - pad, y - int(2 * u), lab, int(15 * u),
                          C_TEXT, True, "r")
            if st.tempo_target:
                self.text.add(frame, x0 + cw - pad, y + int(16 * u),
                              f"target ↓ {st.tempo_target:g}s down", int(10 * u),
                              C_MUTED, False, "r")
                y += int(14 * u)
            y += rh
        if "speed" in rows:
            r = st.velocity_ratio if st.velocity_ratio is not None else 1.0
            col = C_RED if st.fatigue_warned or r < 0.8 else \
                C_AMBER if r < 0.9 else C_GREEN
            self.text.add(frame, x0 + pad, y, "FATIGUE" if st.fatigue_warned
                          else "SPEED", int(11 * u), col if st.fatigue_warned
                          else C_MUTED, st.fatigue_warned)
            self.text.add(frame, vx, y - int(2 * u), f"{r * 100:.0f}%", int(16 * u),
                          col, True, "r")
            hbar(frame, bx, y + int(6 * u), bw, int(8 * u), r, col)
            y += rh
        if "hr" in rows:
            self.text.add(frame, x0 + pad, y, "HEART RATE", int(11 * u), C_MUTED)
            z = st.hr_zone
            col = C_HR_ZONE.get(z or 0, C_MUTED)
            self.text.add(frame, vx, y - int(2 * u), f"{st.hr}", int(16 * u), col,
                          True, "r")
            self.text.add(frame, vx + int(4 * u), y + int(3 * u), "bpm", int(10 * u),
                          C_MUTED)
            if z:
                lab = f"ZONE {z}"
                tw, th = self.text.measure(lab, int(11 * u), True)
                rounded_rect(frame, x0 + cw - pad - tw - 10 * u, y,
                             x0 + cw - pad, y + th + 2 * u, col, int(6 * u))
                self.text.add(frame, x0 + cw - pad - 5 * u, y + 1 * u, lab,
                              int(11 * u), (15, 15, 15), True, "r")
            y += rh

    # ---- left: range-of-motion gauge against the exercise thresholds
    def draw_gauge(self, frame, st: HudState, w, h, u):
        if st.signal is None or st.signal_value is None or st.mode == "auto":
            return
        gx = int(30 * u)
        gy0, gy1 = int(h * 0.30), int(h * 0.76)
        gw = int(14 * u)
        if st.mode == "hold":
            top, bottom = 180.0, 120.0
            marks = [(st.hold_good_above, "straight")]
        else:
            start, deep, lock = st.thresholds
            top, bottom = 180.0, max(0.0, deep - 30)
            marks = [(lock, "lockout"), (start, "rep starts"), (deep, "full depth")]
        span = top - bottom

        def ypos(v):
            return int(gy1 - (max(bottom, min(top, v)) - bottom) / span * (gy1 - gy0))

        label = f"{(st.signal or '').replace('_', ' ').upper()}"
        panel(frame, gx - int(14 * u), gy0 - int(46 * u), gx + int(150 * u),
              gy1 + int(22 * u), alpha=0.5, r=int(10 * u))
        self.text.add(frame, gx - int(4 * u), gy0 - int(42 * u), label,
                      int(11 * u), C_MUTED)
        col = self._rom_color(st)
        self.text.add(frame, gx - int(4 * u), gy0 - int(30 * u),
                      f"{st.signal_value:.0f}°", int(18 * u), col, True)
        rounded_rect(frame, gx, gy0, gx + gw, gy1, (60, 56, 50), gw // 2)
        yv = ypos(st.signal_value)
        if st.mode == "hold":
            rounded_rect(frame, gx, yv, gx + gw, gy1, col, gw // 2)
        else:
            rounded_rect(frame, gx, gy0, gx + gw, max(yv, gy0 + gw), col, gw // 2)
        import cv2
        for v, name in marks:
            y = ypos(v)
            cv2.line(frame, (gx - int(5 * u), y), (gx + gw + int(5 * u), y),
                     C_TEXT, max(1, int(2 * u)), cv2.LINE_AA)
            self.text.add(frame, gx + gw + int(10 * u), y - int(7 * u),
                          f"{name} {v:.0f}°", int(11 * u), C_MUTED)
        cv2.circle(frame, (gx + gw // 2, yv), int(gw * 0.7), C_TEXT, -1, cv2.LINE_AA)
        cv2.circle(frame, (gx + gw // 2, yv), int(gw * 0.7), (20, 20, 20), 1,
                   cv2.LINE_AA)

    # ---- bottom: coaching cue banner
    def draw_cue(self, frame, st: HudState, w, h, u, now):
        if not st.cue:
            self._cue_last = ""
            return
        if st.cue != self._cue_last:
            self._cue_last, self._cue_since = st.cue, now
        col = {"praise": C_GREEN, "fatigue": C_RED, "info": C_BLUE}.get(
            st.cue_kind, C_AMBER)
        size = int(24 * u)
        # keep clear of the coach panel (bottom-right) and the gauge (left)
        left = int(170 * u) if st.signal and st.mode != "auto" else int(16 * u)
        right = w - (int(380 * u) + int(32 * u) if st.chat else int(16 * u))
        cx = (left + right) // 2
        lines = self.text.wrap(st.cue, size, True, int((right - left) * 0.9))
        lh = int(size * 1.25)
        tw = max(self.text.measure(ln, size, True)[0] for ln in lines)
        pad = int(16 * u)
        bh = lh * len(lines) + pad * 2 - int(4 * u)
        by1 = h - int(44 * u)
        by0 = by1 - bh
        bx0 = cx - tw // 2 - pad - int(10 * u)
        bx1 = cx + tw // 2 + pad
        panel(frame, bx0, by0, bx1, by1, alpha=0.72, r=int(12 * u))
        rounded_rect(frame, bx0 + int(6 * u), by0 + int(8 * u), bx0 + int(11 * u),
                     by1 - int(8 * u), col, int(2 * u))
        y = by0 + pad - int(2 * u)
        for ln in lines:
            self.text.add(frame, cx + int(5 * u), y, ln, size, col, True, "m")
            y += lh

    # ---- rest overlay
    def draw_rest(self, frame, st: HudState, w, h, u):
        if st.rest_left <= 0:
            self._rest_total = 0.0
            return
        if st.rest_left > self._rest_total:
            self._rest_total = st.rest_left
        dim(frame, 0.45)
        cx, cy = w // 2, int(h * 0.46)
        r = int(86 * u)
        ring(frame, cx, cy, r, st.rest_left / max(self._rest_total, 1e-6), C_CYAN,
             max(6, int(9 * u)))
        self.text.add(frame, cx, cy - int(46 * u), "REST", int(16 * u), C_MUTED,
                      True, "m")
        self.text.add(frame, cx, cy - int(28 * u), f"{int(st.rest_left) + 1}",
                      int(60 * u), C_TEXT, True, "m")
        self.text.add(frame, cx, cy + int(40 * u), "seconds", int(13 * u), C_MUTED,
                      False, "m")
        nxt = st.rest_next or "then back to work"
        self.text.add(frame, cx, cy + r + int(22 * u), nxt, int(18 * u), C_TEXT,
                      True, "m")
        self.text.add(frame, cx, cy + r + int(50 * u),
                      "r cancels the rest · breathe, shake it out", int(12 * u),
                      C_MUTED, False, "m")

    # ---- bottom-right: the LLM coach
    def draw_coach(self, frame, st: HudState, w, h, u):
        c = st.chat
        if not c:
            return
        cw = int(380 * u)
        pad = int(12 * u)
        size = int(14 * u)
        lh = int(size * 1.3)
        lines = []
        if c.get("user"):
            lines += [("you", ln) for ln in self.text.wrap(
                "You: " + c["user"], size, False, cw - 2 * pad)][:2]
        if c.get("reply"):
            rl = self.text.wrap(c["reply"], size, False, cw - 2 * pad)
            lines += [("coach", ln) for ln in rl[-5:]]
        if not lines:
            lines = [("hint", ln) for ln in self.text.wrap(
                c.get("hint") or "Ask me anything — just speak, or type in "
                "the terminal.", size, False, cw - 2 * pad)]
        ch = pad * 2 + int(26 * u) + lh * len(lines)
        x0 = w - cw - int(16 * u)
        y1 = h - int(44 * u)
        y0 = y1 - ch
        panel(frame, x0, y0, x0 + cw, y1, alpha=0.7)
        self.text.add(frame, x0 + pad, y0 + pad - 2 * u, "COACH", int(12 * u),
                      C_PURPLE, True)
        status = c.get("status", "")
        scol = (C_CYAN if "hearing" in status else C_GREEN if status == "listening"
                else C_PURPLE if "answer" in status else C_MUTED)
        slabel = {"listening": "● listening", "answering...": "● answering"}.get(
            status, "● " + status if status else "")
        if "hearing" in status:
            slabel = "● hearing you…"
        sx = x0 + cw - pad
        # mic level meter: 8 segments, lit up to the live level
        level = float(c.get("level") or 0.0)
        segs = 8
        seg_w = int(5 * u)
        mx = sx - segs * (seg_w + int(2 * u))
        for i in range(segs):
            lit = level * 3 > i
            rounded_rect(frame, mx + i * (seg_w + int(2 * u)), y0 + pad + int(2 * u),
                         mx + i * (seg_w + int(2 * u)) + seg_w, y0 + pad + int(12 * u),
                         scol if lit else (70, 64, 58), int(2 * u))
        self.text.add(frame, mx - int(8 * u), y0 + pad - 2 * u, slabel, int(11 * u),
                      scol, True, "r")
        y = y0 + pad + int(24 * u)
        for kind, ln in lines:
            col = {"you": C_MUTED, "coach": C_TEXT, "hint": C_DIM}[kind]
            self.text.add(frame, x0 + pad, y, ln, size, col)
            y += lh

    # ---- footer: keys + status
    def draw_footer(self, frame, st: HudState, w, h, u):
        y0 = h - int(30 * u)
        panel(frame, 0, y0, w, h, alpha=0.55, r=0)
        keys = "1-9 exercise · a auto · r rest · v voice · m mirror · h help · q finish"
        if st.chat:
            keys = "c talk · " + keys
        self.text.add(frame, int(16 * u), y0 + int(8 * u), keys, int(12 * u), C_MUTED)
        right = []
        if st.elapsed_s:
            m, s = divmod(int(st.elapsed_s), 60)
            right.append(f"{m:02d}:{s:02d}")
        right.append("voice on" if st.voice_on else "voice muted")
        if st.mirrored:
            right.append("mirror")
        if st.fps:
            right.append(f"{st.fps:.0f} fps")
        if st.source:
            right.append(st.source)
        self.text.add(frame, w - int(16 * u), y0 + int(8 * u), " · ".join(right),
                      int(12 * u), C_MUTED, False, "r")

    # ---- framing guidance when nobody / not enough body is tracked
    def draw_setup(self, frame, st: HudState, w, h, u, now):
        if st.tracking == "ok":
            self._lost_since = None
            return
        if self._lost_since is None:
            self._lost_since = now
        if now - self._lost_since < 0.6:
            return
        cw, ch = int(440 * u), int(178 * u)
        x0, y0 = w // 2 - cw // 2, int(h * 0.30)
        panel(frame, x0, y0, x0 + cw, y0 + ch, alpha=0.78, border=(90, 84, 76))
        pad = int(18 * u)
        if st.tracking == "none":
            title = "Step into the frame"
            sub = "I can't see anyone yet"
        else:
            miss = ", ".join(st.missing[:3]) or "your whole body"
            title = "Step back a little"
            sub = f"I can't see your {miss}"
        self.text.add(frame, x0 + pad, y0 + pad - 2 * u, title, int(22 * u), C_TEXT,
                      True)
        self.text.add(frame, x0 + pad, y0 + pad + int(28 * u), sub, int(14 * u),
                      C_AMBER)
        checks = [
            ("Whole body in view", st.tracking == "ok"),
            ("Enough light", st.brightness >= 60),
            (f"Camera: {st.camera_hint or 'side view'}, ~2–3 m away", None),
        ]
        y = y0 + pad + int(58 * u)
        import cv2
        for label, ok in checks:
            col = C_DIM if ok is None else C_GREEN if ok else C_RED
            cv2.circle(frame, (x0 + pad + int(6 * u), y + int(9 * u)), int(5 * u),
                       col, -1 if ok else 1, cv2.LINE_AA)
            self.text.add(frame, x0 + pad + int(20 * u), y, label, int(14 * u),
                          C_TEXT if ok is not False else C_AMBER)
            y += int(24 * u)
        if st.brightness < 60:
            self.text.add(frame, x0 + pad, y0 + ch - int(22 * u),
                          "Too dark — turn on a light or face a window",
                          int(12 * u), C_RED)

    # ---- help overlay
    def draw_help(self, frame, st: HudState, w, h, u):
        if not st.show_help:
            return
        dim(frame, 0.6)
        cw = min(int(760 * u), w - int(32 * u))
        pad = int(20 * u)
        kcol = int(190 * u)
        dsize, lh = int(13 * u), int(17 * u)
        wrapped = []
        for row in KEY_HELP + [None] + CLI_HELP:
            if row is None:
                wrapped.append(None)
            else:
                wrapped.append((row[0], self.text.wrap(
                    row[1], dsize, False, cw - pad * 2 - kcol)))
        ch = int(70 * u) + sum(int(26 * u) if r is None else
                               lh * len(r[1]) + int(9 * u) for r in wrapped)
        x0, y0 = w // 2 - cw // 2, max(int(16 * u), h // 2 - ch // 2)
        panel(frame, x0, y0, x0 + cw, y0 + ch, alpha=0.9, border=(90, 84, 76))
        self.text.add(frame, x0 + pad, y0 + pad - 4 * u, "Keys in this window",
                      int(20 * u), C_TEXT, True)
        self.text.add(frame, x0 + cw - pad, y0 + pad, "h closes", int(12 * u),
                      C_MUTED, False, "r")
        y = y0 + pad + int(34 * u)
        for row in wrapped:
            if row is None:
                self.text.add(frame, x0 + pad, y + int(6 * u),
                              "More on the command line", int(13 * u), C_MUTED, True)
                y += int(26 * u)
                continue
            key, lines = row
            self.text.add(frame, x0 + pad, y, key, int(14 * u), C_AMBER, True)
            for i, ln in enumerate(lines):
                self.text.add(frame, x0 + pad + kcol, y + i * lh, ln, dsize, C_TEXT)
            y += lh * len(lines) + int(9 * u)

    # ---- one frame
    def render(self, frame, pts, st: HudState, now: float | None = None):
        """Draw the full HUD over `frame` (BGR); returns the display frame,
        upscaled to at least `min_width` so text stays legible."""
        now = time.time() if now is None else now
        frame, w, h, u = self._prep(frame, now)
        self.draw_skeleton(frame, pts, st, w, h, u)
        self.draw_gauge(frame, st, w, h, u)
        self.draw_exercise_card(frame, st, w, h, u, now)
        self.draw_metrics_card(frame, st, w, h, u)
        self.draw_cue(frame, st, w, h, u, now)
        self.draw_coach(frame, st, w, h, u)
        self.draw_setup(frame, st, w, h, u, now)
        self.draw_footer(frame, st, w, h, u)
        self.draw_rest(frame, st, w, h, u)
        self.draw_help(frame, st, w, h, u)
        return self.text.flush(frame)

    # ---- end of set
    def render_summary(self, frame, session: dict, deepest: float | None = None,
                       thresholds: tuple | None = None, signal: str | None = None,
                       log_path: str = "workout_log.json"):
        """Summary card over the last frame: what happened, what to fix."""
        frame, w, h, u = self._prep(frame, time.time())
        dim(frame, 0.62)
        sm = session.get("summary") or {}
        plank = session.get("plank")
        reps = sm.get("reps") or 0
        cw = min(int(620 * u), w - int(32 * u))
        pad = int(22 * u)
        tips = self._summary_tips(sm, plank, deepest, thresholds, signal)
        tip_lines = [self.text.wrap(t, int(14 * u), False,
                                    cw - pad * 2 - int(20 * u)) for _, t in tips]
        ch = int(150 * u) + int(48 * u) + sum(
            int(17 * u) * len(ls) + int(11 * u) for ls in tip_lines)
        x0, y0 = w // 2 - cw // 2, max(int(16 * u), h // 2 - ch // 2)
        panel(frame, x0, y0, x0 + cw, y0 + ch, alpha=0.92, border=(90, 84, 76))
        ex = (session.get("exercise") or "").replace("_", " ").upper()
        title = "SET COMPLETE" if (reps or plank) else "NO REPS COUNTED"
        self.text.add(frame, x0 + pad, y0 + pad - 4 * u, title, int(13 * u),
                      C_GREEN if (reps or plank) else C_AMBER, True)
        self.text.add(frame, x0 + pad, y0 + pad + int(14 * u), ex, int(26 * u),
                      C_TEXT, True)
        dur = session.get("duration_s") or 0
        m, s = divmod(int(dur), 60)
        self.text.add(frame, x0 + cw - pad, y0 + pad, f"{m:02d}:{s:02d}",
                      int(16 * u), C_MUTED, True, "r")
        # stat tiles
        tiles = []
        if plank:
            tiles.append((f"{plank.get('total_hold_s', 0):.0f}s", "held", C_GREEN))
            tiles.append((f"{plank.get('best_streak_s', 0):.0f}s", "best unbroken",
                          C_BLUE))
        else:
            tiles.append((str(reps), "reps", C_TEXT))
            sc = sm.get("avg_score")
            if sc is not None:
                tiles.append((f"{sc:.0f}", "avg score", score_color(sc)))
            if sm.get("avg_concentric_s") is not None:
                tiles.append((f"{sm['avg_concentric_s']:.1f}s", "avg lift time",
                              C_BLUE))
            if sm.get("avg_similarity") is not None:
                tiles.append((f"{sm['avg_similarity']:.0f}%", "vs golden rep", C_AMBER))
            vl = sm.get("velocity_loss_pct")
            if vl is not None:
                tiles.append((f"-{vl:.0f}%", "speed loss",
                              C_RED if vl > 20 else C_GREEN))
        if sm.get("volume_kg"):
            tiles.append((f"{sm['volume_kg']:g}", "kg volume", C_BLUE))
            tiles.append((f"{sm['e1rm_kg']:g}", "kg est. 1RM", C_GREEN))
        if sm.get("avg_hr"):
            tiles.append((f"{sm['avg_hr']}", f"avg bpm · peak {sm.get('peak_hr')}",
                          C_RED))
        tx = x0 + pad
        ty = y0 + pad + int(56 * u)
        tiles = tiles[:6]
        tw = (cw - pad * 2) // max(len(tiles), 1)
        for val, lab, col in tiles:
            self.text.add(frame, tx, ty, val, int(28 * u), col, True)
            self.text.add(frame, tx, ty + int(34 * u), lab, int(11 * u), C_MUTED)
            tx += tw
        y = ty + int(64 * u)
        import cv2
        cv2.line(frame, (x0 + pad, y), (x0 + cw - pad, y), (70, 64, 58), 1)
        y += int(12 * u)
        for (kind, _), lines in zip(tips, tip_lines):
            col = {"fault": C_RED, "tip": C_AMBER, "good": C_GREEN}.get(kind, C_TEXT)
            cv2.circle(frame, (x0 + pad + int(5 * u), y + int(9 * u)), int(4 * u),
                       col, -1, cv2.LINE_AA)
            for i, ln in enumerate(lines):
                self.text.add(frame, x0 + pad + int(18 * u), y + i * int(17 * u), ln,
                              int(14 * u), C_TEXT)
            y += int(17 * u) * len(lines) + int(11 * u)
        foot = (f"saved to {os.path.basename(log_path)} · charts: python "
                "coach_dashboard.py · any key closes")
        self.text.add(frame, x0 + pad, y0 + ch - int(30 * u), foot, int(12 * u),
                      C_MUTED)
        return self.text.flush(frame)

    @staticmethod
    def _summary_tips(sm, plank, deepest, thresholds, signal) -> list[tuple[str, str]]:
        tips = []
        reps = sm.get("reps") or 0
        fc = sm.get("fault_counts") or {}
        for pr in sm.get("prs") or []:
            tips.append(("good", "Personal record — " + pr))
        if plank:
            if plank.get("total_hold_s", 0) >= 30:
                tips.append(("good", "Solid hold — try a longer unbroken streak next time."))
            for k, v in sorted(fc.items(), key=lambda kv: -kv[1])[:2]:
                tips.append(("fault", f"{FAULT_NAMES.get(k, k)} ×{v}"))
            return tips
        if not reps:
            if deepest is not None and thresholds and signal:
                start, deep, lock = thresholds
                name = signal.replace("_", " ")
                if deepest >= start:
                    tips.append(("tip", f"Your {name} angle only reached {deepest:.0f}° — "
                                        f"a rep starts below {start:.0f}° and counts "
                                        f"once you pass {deep:.0f}° and stand back up "
                                        f"above {lock:.0f}°."))
                else:
                    tips.append(("tip", f"You got down to {deepest:.0f}° but a rep only "
                                        f"counts after locking out above {lock:.0f}° "
                                        f"— finish each rep fully."))
            tips.append(("tip", "Stand side-on to the camera with your whole body "
                                "in view; the gauge on the left shows the live angle."))
            tips.append(("tip", "Press h in the window for keys — 1-9 switch "
                                "exercise, a auto-detects."))
            return tips
        top = sorted(fc.items(), key=lambda kv: -kv[1])[:3]
        for k, v in top:
            tips.append(("fault", f"{FAULT_NAMES.get(k, k)} — {v} of {reps} reps"))
        if not top:
            tips.append(("good", "No form faults — great set!"))
        vl = sm.get("velocity_loss_pct")
        if vl is not None and vl > 20:
            tips.append(("tip", f"Bar speed dropped {vl:.0f}% by the end — that's "
                                "real fatigue; rest longer or stop the set earlier."))
        if sm.get("avg_similarity") is not None and sm["avg_similarity"] < 60:
            tips.append(("tip", "Reps drifted from your golden rep — slow down and "
                                "match its shape."))
        return tips


# ------------------------------------------------------------- previews
def demo_states() -> dict[str, tuple[np.ndarray | None, HudState]]:
    """Representative frames for previews and the selftest."""
    def figure(depth: float) -> np.ndarray:
        off = 0.24 * depth
        pts = np.zeros((33, 4), dtype=np.float32)

        def put(i, x, y, v=1.0):
            pts[i] = (x, y, 0.0, v)
        put(NOSE, 0.50, 0.14)
        put(L_EAR, 0.48, 0.14); put(R_EAR, 0.52, 0.14)
        put(L_SHO, 0.45, 0.28); put(R_SHO, 0.55, 0.28)
        put(L_ELB, 0.42, 0.40); put(R_ELB, 0.58, 0.40)
        put(L_WRI, 0.41, 0.50); put(R_WRI, 0.59, 0.50)
        put(L_HIP, 0.46, 0.52); put(R_HIP, 0.54, 0.52)
        put(L_KNE, 0.46 - off, 0.70); put(R_KNE, 0.54 + off, 0.70)
        put(L_ANK, 0.46, 0.88); put(R_ANK, 0.54, 0.88)
        put(L_HEE, 0.45, 0.90); put(R_HEE, 0.55, 0.90)
        put(L_TOE, 0.48, 0.91); put(R_TOE, 0.52, 0.91)
        return pts

    base = dict(exercise="squat", mode="reps", signal="knee",
                thresholds=(150, 100, 165), camera_hint="side or 45° front",
                tracking="ok", fps=28.0, source="cam 0", mirrored=True,
                elapsed_s=95.0)
    out = {}
    out["tracking"] = (figure(0.6), HudState(
        **base, phase="BOTTOM", reps=3, rep_goal=10, rep_scores=[90, 75, 85],
        load_kg=60.0, best_reps=8, muscles=["quadriceps", "glutes", "hamstrings"],
        last_score=85, similarity=72, has_reference=True, tempo=(1.4, 0.9),
        velocity_ratio=0.93, signal_value=112.0, faults_now=["knees_cave"],
        cue="Push your knees out — don't let them cave in.", cue_kind="fault",
        chat={"status": "listening", "level": 1.2,
              "user": "why do my knees hurt at the bottom?",
              "reply": "Your knees are caving in on the way up — that loads the "
                       "inside of the joint. Screw your feet into the floor and "
                       "push the knees out over your toes."},
        hr=142, hr_zone=3,
        program={"exercise": "squat", "set": 2, "sets": 3, "block": 1,
                 "blocks": 3, "target": "10 reps"}))
    out["setup"] = (None, HudState(**{**base, "tracking": "none"},
                                   brightness=40, voice_on=False))
    out["rest"] = (figure(0.0), HudState(
        **base, phase="IDLE", reps=10, rep_goal=10, rep_scores=[90] * 10,
        last_score=95, signal_value=172.0, rest_left=42.0,
        rest_next="next: push-up · set 1/2 · 15 reps",
        program={"exercise": "pushup", "set": 1, "sets": 2, "block": 2,
                 "blocks": 3, "target": "15 reps"}))
    out["plank"] = (figure(0.0), HudState(
        exercise="plank", mode="hold", signal="body_line", signal_value=171.0,
        hold_total=34.2, hold_streak=12.0, hold_best=22.0, tracking="ok",
        camera_hint="side view", fps=30.0, source="cam 0", elapsed_s=40.0,
        cue="Great form!", cue_kind="praise"))
    out["auto"] = (figure(0.2), HudState(
        exercise=None, mode="auto", signal=None, tracking="ok",
        detector_label="ML", camera_hint="auto-detecting", fps=25.0,
        source="cam 1", angles={"knee": 168.0, "elbow": 171.0, "trunk_lean": 4.0}))
    out["help"] = (figure(0.0), HudState(**base, show_help=True, signal_value=170.0))
    out["fatigue"] = (figure(0.5), HudState(
        **base, phase="ASCENT", reps=9, rep_scores=[90, 88, 80, 75, 70, 70, 65, 60, 55],
        last_score=55, tempo=(1.0, 0.6), velocity_ratio=0.72, fatigue_warned=True,
        signal_value=130.0, faults_now=["back_lean"],
        cue="You're slowing down — keep form tight or end the set.",
        cue_kind="fatigue", recording_reference=True, tempo_target=2.0,
        chat={"status": "answering...", "level": 0.0, "user": None,
              "reply": "Set 2 done: scores fell from 90 to 55 and your back "
                       "started leaning — rest a full 90 seconds."}))
    return out


def demo_summary() -> dict:
    return {"exercise": "squat", "duration_s": 131.5,
            "reps": [{"score": s} for s in (80, 75, 90, 85, 70, 60)],
            "plank": None,
            "summary": {"reps": 6, "avg_score": 76.7, "avg_concentric_s": 1.1,
                        "avg_similarity": 68.0,
                        "fault_counts": {"shallow": 3, "knees_cave": 1},
                        "velocity_loss_pct": 24.0, "avg_hr": 138, "peak_hr": 161}}


def write_previews(out_dir: str, size=(640, 480)) -> list[str]:
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    hud = Hud()
    paths = []
    for name, (pts, st) in demo_states().items():
        frame = _backdrop(size)
        hud.render(frame, pts, st, now=1000.0)
        img = hud.render(frame, pts, st, now=1001.0)   # past the 0.6 s guide delay
        p = os.path.join(out_dir, f"hud_{name}.png")
        cv2.imwrite(p, img)
        paths.append(p)
    img = hud.render_summary(_backdrop(size), demo_summary(), deepest=112.0,
                             thresholds=(150, 100, 165), signal="knee")
    p = os.path.join(out_dir, "hud_summary.png")
    cv2.imwrite(p, img)
    paths.append(p)
    empty = {"exercise": "squat", "duration_s": 48.1, "reps": [], "plank": None,
             "summary": {"reps": 0, "avg_score": None, "fault_counts": {}}}
    img = hud.render_summary(_backdrop(size), empty, deepest=156.0,
                             thresholds=(150, 100, 165), signal="knee")
    p = os.path.join(out_dir, "hud_summary_empty.png")
    cv2.imwrite(p, img)
    paths.append(p)
    return paths


def _backdrop(size) -> np.ndarray:
    """A gym-ish gradient so previews show panels blending over 'video'."""
    w, h = size
    ys = np.linspace(0, 1, h)[:, None]
    xs = np.linspace(0, 1, w)[None, :]
    b = (70 + 60 * ys + 20 * xs).astype(np.uint8)
    g = (75 + 50 * ys).astype(np.uint8)
    r = (85 + 40 * ys - 20 * xs).astype(np.uint8)
    return np.dstack([np.broadcast_to(b, (h, w)), np.broadcast_to(g, (h, w)),
                      np.broadcast_to(r, (h, w))]).copy()


# --------------------------------------------------------------- selftest
def selftest():
    print("== coach_hud selftests ==")
    import cv2

    print("1) every HUD state renders, upscales, and actually draws:", end=" ")
    hud = Hud()
    for name, (pts, st) in demo_states().items():
        frame = _backdrop((640, 480))
        img = hud.render(frame, pts, st, now=1000.0)
        assert img.shape == (720, 960, 3), (name, img.shape)
        assert img.dtype == np.uint8 and img.flags["C_CONTIGUOUS"]
        ref = cv2.resize(frame, (960, 720))
        assert np.abs(img.astype(int) - ref.astype(int)).mean() > 2, name
    big = hud.render(_backdrop((1920, 1080)), *demo_states()["tracking"], now=1000.0)
    assert big.shape == (1080, 1920, 3)
    print(f"OK (backend: {hud.text.backend})")

    print("2) Hershey fallback draws the same states without Pillow:", end=" ")
    hud2 = Hud(use_pil=False)
    assert hud2.text.backend == "hershey"
    for name, (pts, st) in demo_states().items():
        img = hud2.render(_backdrop((640, 480)), pts, st, now=1000.0)
        assert img.shape == (720, 960, 3), name
    print("OK")

    print("3) summary card: tips explain 0 reps in terms of the thresholds:", end=" ")
    tips = Hud._summary_tips({"reps": 0, "fault_counts": {}}, None, 156.0,
                             (150, 100, 165), "knee")
    assert any("156°" in t and "150°" in t for _, t in tips), tips
    tips = Hud._summary_tips({"reps": 0, "fault_counts": {}}, None, 120.0,
                             (150, 100, 165), "knee")
    assert any("locking out" in t for _, t in tips), tips
    tips = Hud._summary_tips(demo_summary()["summary"], None, None, None, None)
    kinds = [k for k, _ in tips]
    assert kinds.count("fault") == 2 and "tip" in kinds, tips
    assert any("not deep enough — 3 of 6" in t for _, t in tips), tips
    img = hud.render_summary(_backdrop((640, 480)), demo_summary(), 112.0,
                             (150, 100, 165), "knee")
    assert img.shape == (720, 960, 3)
    print("OK")

    print("4) framing guidance appears after a short loss, not on a blip:", end=" ")
    hud3 = Hud()
    st = HudState(exercise="squat", signal="knee", thresholds=(150, 100, 165),
                  tracking="none")
    a = hud3.render(_backdrop((640, 480)), None, st, now=10.0)
    b = hud3.render(_backdrop((640, 480)), None, st, now=10.3)
    c = hud3.render(_backdrop((640, 480)), None, st, now=11.0)
    assert np.array_equal(a, b), "0.3 s loss must not flash the guide"
    assert not np.array_equal(b, c), "guide must show after 0.6 s"
    st.tracking = "ok"
    hud3.render(_backdrop((640, 480)), None, st, now=11.5)
    assert hud3._lost_since is None
    print("OK")

    print("5) phase labels follow the lift direction; fault names are human:", end=" ")
    assert PHASE_LABELS["ascent"]["ASCENT"] == "LIFTING"
    assert PHASE_LABELS["descent"]["DESCENT"] == "LIFTING"
    assert PHASE_LABELS["descent"]["BOTTOM"] == "TOP"
    for k in ("back_lean", "knees_cave", "shallow", "too_fast", "chin"):
        assert k in FAULT_NAMES and "_" not in FAULT_NAMES[k]
        assert k in FAULT_REGIONS
    assert len(EXERCISE_KEYS) == 9 and len(set(EXERCISE_KEYS)) == 9
    print("OK")

    print("6) text layer wraps by measured width and survives non-ASCII:", end=" ")
    tl = TextLayer()
    lines = tl.wrap("Straighten your back — chest up! Keep going, deeper.", 20,
                    True, 200)
    assert len(lines) >= 2 and all(tl.measure(ln, 20, True)[0] <= 260 for ln in lines)
    f = _backdrop((320, 240))
    tl.add(f, 10, 10, "élan — 45° ↓↑ ✓", 18, C_TEXT, True)
    out = tl.flush(f)
    assert out.shape == f.shape and not tl.ops
    tl2 = TextLayer(use_pil=False)
    tl2.add(f, 10, 10, "élan — 45° ↓↑ ✓", 18, C_TEXT, True)   # must not throw
    print("OK")

    print("\nAll coach_hud selftests passed.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--preview", metavar="DIR",
                    help="write PNGs of every HUD state into DIR and exit")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    elif args.preview:
        for p in write_previews(args.preview):
            print(p)
    else:
        ap.print_help()
        sys.exit(1)
