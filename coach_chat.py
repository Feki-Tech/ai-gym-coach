"""Talk to your gym coach — conversational LLM layer for AI Gym Coach.

Works with any OpenAI-compatible chat API. The default backend is a local
Ollama server — private, free, no API key:

    docker compose up -d ollama
    docker compose exec ollama ollama pull llama3.2:3b

    python coach_chat.py                  # text chat on the host
    python coach_chat.py --voice          # + spoken replies (TTS)
    python coach_chat.py --listen         # + push-to-talk mic input
    docker compose run --rm coach         # text chat fully inside Docker

    python pose_coach.py --exercise auto --coach   # chat DURING a workout

The coach sees your training history (workout_log.json) and, when running
inside pose_coach.py, the live session (exercise, reps, scores, faults,
fatigue) — so you can ask "why are my squat scores dropping?" mid-set.
It answers in the language you speak to it. Replies stream in live,
are spoken sentence-by-sentence, and you can interrupt at any time:
type a new question mid-answer (workout mode) or press Ctrl+C (chat mode).

The coach also keeps a local *athlete profile* (coach_profile.db, SQLite):
durable facts you mention in chat (goals, injuries, equipment, schedule…)
are extracted automatically and remembered across sessions. Commands:
/profile shows it, /remember and /forget edit it, --no-profile disables.

Config (env vars):
    COACH_LLM_BASE_URL   default http://localhost:11434/v1   (Ollama)
    COACH_LLM_MODEL      default llama3.2:3b
    COACH_LLM_API_KEY    default "ollama" (set a real key for OpenAI etc.)
    COACH_LOG            default workout_log.json
    COACH_PROFILE_DB     default coach_profile.db

Voice input needs optional extras (host only):  pip install -r requirements-voice.txt
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import urllib.error
import urllib.request
from collections import deque
from datetime import datetime

import coach_calendar
import coach_profile

DEFAULT_BASE = os.environ.get("COACH_LLM_BASE_URL", "http://localhost:11434/v1")
DEFAULT_MODEL = os.environ.get("COACH_LLM_MODEL", "llama3.2:3b")
DEFAULT_KEY = os.environ.get("COACH_LLM_API_KEY", "ollama")
DEFAULT_LOG = os.environ.get("COACH_LOG", "workout_log.json")

MAX_TURNS = 16          # user/assistant messages kept in context
LISTEN_SECONDS = 6      # push-to-talk recording length
# Spoken replies stay snappy on small local models; raise for long answers.
MAX_REPLY_TOKENS = int(os.environ.get("COACH_MAX_TOKENS", "300"))

PERSONA = """\
You are "Coach", the friendly personal trainer inside the AI Gym Coach app.
Style: warm, encouraging and practical — celebrate effort, never shame.
Replies are read aloud: default to 2-4 short sentences (under 70 words);
go longer only when the user asks for detail. LANGUAGE RULE: write the
entire reply in the language of the user's last message and never mix in
words from any other language (English question = 100% English answer).

SAFETY (non-negotiable): you are not a doctor. Sharp, stabbing or radiating
pain, numbness, dizziness or chest pain → tell the user to stop the set NOW
and see a medical professional. Never diagnose conditions or prescribe
medication. Dull muscle burn during a set and next-day soreness are normal.

COACHING KNOWLEDGE you rely on (evidence-based, matches the app's fault codes):
- knees_cave (knee valgus): usually weak glutes/hip abductors, not the
  kneecap — cue "push your knees out over your toes"; build with banded
  squats, side-lying hip abductions, glute bridges.
- back_round / back_lean: brace the core, chest up, neutral spine; lighten
  the load and hinge from the hips, not the lower back.
- body_sag (plank/push-up): squeeze glutes and abs — one straight line.
- elbow_flare: tuck elbows to ~45° from the torso to protect the shoulders.
- elbow_swing (curls): pin elbows to your ribs; no momentum.
- shallow: full range of motion beats heavy-and-short — reduce the weight
  and own the bottom position.
- too_fast: a 2-3 s lowering phase improves control and muscle growth.
- Programming: progressive overload (small weekly rep/weight increases),
  ~48 h rest per muscle group, stop 1-3 reps short of failure on most sets.
- The app's fatigue warning fires at >20% rep-velocity loss — ending the
  set there protects form quality.
- Nutrition basics only: ~1.6-2.2 g protein per kg bodyweight daily,
  hydrate; no diet prescriptions.

APP FACTS: exercises = squat, pushup, bench, deadlift, lunge,
shoulder_press, curl, pullup, plank; reps are scored 0-100; "ref-sim" is
similarity to the user's recorded golden rep. The LIVE SESSION block gives
you real physics and environment data: joint_angles_deg (current joint
angles), last_rep (score, ecc_s/con_s tempo, rom_deg range of motion,
vel_deg_s speed, faults), environment (brightness, pose visibility, how
much of the body is in frame, fps, camera_hint) and coach_config (rep
goal, rest timer, tempo target, cues on/off). Use these to give specific,
personal advice — e.g. poor visibility/brightness → ask them to fix
framing or lighting; never invent numbers that are not in the blocks."""

ACTIONS_PROMPT = """\
APP CONTROL — you can drive the app. When (and only when) the user asks
for it or clearly agrees, end your reply with action lines, each on its
own line, exactly in this form:
ACTION: {"do": "set_exercise", "exercise": "squat"}
ACTION: {"do": "set_rep_goal", "reps": 10}
ACTION: {"do": "rest_timer", "seconds": 60}
ACTION: {"do": "set_tempo", "eccentric_s": 3}
ACTION: {"do": "cues", "enabled": false}
ACTION: {"do": "start_program", "plan": "squat 3x10 rest 90, \
pushup 2x15 rest 45, plank 2x40s rest 30"}
ACTION: {"do": "stop_program"}
set_exercise accepts squat, pushup, bench, deadlift, lunge,
shoulder_press, curl, pullup, plank or "auto" (re-detect). set_rep_goal
sets the target reps for this set; rest_timer starts a rest countdown;
set_tempo sets the lowering-phase seconds to enforce; cues mutes/unmutes
the spoken form corrections. start_program runs a whole guided workout:
the app counts every set, starts each rest and switches exercises by
itself. The plan is comma-separated blocks "exercise SETSxREPS rest
SECONDS"; use e.g. 40s for timed holds (plank 2x40s). The plan may ONLY
use these exact exercise names: squat, pushup, bench, deadlift, lunge,
shoulder_press, curl, pullup, plank — never invent variations (no
"squat-left-leg", no equipment names). When the athlete asks you to
plan or program a workout, design it from their profile, history and
today's shape, say it in one short sentence, then emit ONE
start_program action with the full plan. Confirm in one short sentence
what you set. Never invent other action names; without a clear user
request, no ACTION lines at all."""

CALENDAR_PROMPT = """\
CALENDAR — the athlete's Google Calendar is connected. You can use it
ONLY through ACTION lines; the app executes them and reports back.
STRICT RULES:
1. You have ZERO knowledge of their schedule. NEVER state, guess or
   invent events or free times from memory. To know anything, emit:
ACTION: {"do": "calendar_check", "days": 7}
   The app then sends you [APP DATA] with the real agenda — only after
   that may you talk about their schedule or propose slots.
2. An event exists ONLY when you emit (after the athlete agreed to an
   exact date and time):
ACTION: {"do": "calendar_book", "title": "Leg day", \
"start": "2026-07-14T18:00", "minutes": 60}
   The app announces the booking itself. NEVER claim something is
   booked without emitting that line — that would be lying.
3. "start" is local time YYYY-MM-DDTHH:MM; work out real dates from
   the NOW line (e.g. "tomorrow 18:00" = NOW's date + 1 day).
Example:
  Athlete: when can I train this week?
  You: Let me check your calendar.
  ACTION: {"do": "calendar_check", "days": 7}
  [APP DATA gives the agenda] → You: Tuesday evening is free — 18:00?"""


class CoachOffline(RuntimeError):
    """LLM backend unreachable — carries setup instructions."""


# ------------------------------------------------------------- LLM client
class LLMClient:
    """Minimal OpenAI-compatible /chat/completions client (stdlib only)."""

    def __init__(self, base_url: str = DEFAULT_BASE, model: str = DEFAULT_MODEL,
                 api_key: str = DEFAULT_KEY, timeout: float = 180.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def _open(self, payload: dict):
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        try:
            return urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode()[:300]
            except Exception:
                pass
            if e.code == 404 and "11434" in self.base_url:
                raise CoachOffline(
                    f"Model '{self.model}' not found on Ollama. Pull it with:\n"
                    f"  docker compose exec ollama ollama pull {self.model}"
                ) from e
            raise CoachOffline(f"LLM backend error {e.code}: {detail}") from e
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            raise CoachOffline(
                f"Cannot reach the LLM backend at {self.base_url}.\n"
                "Start the local one with:\n"
                "  docker compose up -d ollama\n"
                "  docker compose exec ollama ollama pull " + self.model + "\n"
                "or point COACH_LLM_BASE_URL / COACH_LLM_MODEL / "
                "COACH_LLM_API_KEY at another OpenAI-compatible API."
            ) from e

    def chat(self, messages: list[dict]) -> str:
        with self._open({"model": self.model, "messages": messages,
                         "max_tokens": MAX_REPLY_TOKENS,
                         "stream": False}) as resp:
            data = json.loads(resp.read().decode())
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as e:
            raise CoachOffline(f"Unexpected LLM response: {str(data)[:300]}") from e

    def warm_up(self):
        """Pull the model into memory so the first real answer is instant.

        Ollama loads a model on first use (seconds of cold start); a 1-token
        background ping at startup pays that cost before the user speaks.
        Best-effort: failures stay silent, the first question reports them.
        """
        def _ping():
            try:
                with self._open({"model": self.model, "max_tokens": 1,
                                 "messages": [{"role": "user",
                                               "content": "hi"}],
                                 "stream": False}) as resp:
                    resp.read()
            except Exception:
                pass
        threading.Thread(target=_ping, daemon=True).start()

    def chat_stream(self, messages: list[dict]):
        """Yield reply text deltas as the model produces them (SSE)."""
        resp = self._open({"model": self.model, "messages": messages,
                           "max_tokens": MAX_REPLY_TOKENS,
                           "stream": True})
        try:
            with resp:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    try:
                        delta = data and json.loads(data)["choices"][0]["delta"]
                    except (json.JSONDecodeError, KeyError, IndexError,
                            TypeError):
                        continue
                    if delta and delta.get("content"):
                        yield delta["content"]
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            raise CoachOffline("Connection to the LLM was lost "
                               "mid-reply — is the backend still up?") from e


# ---------------------------------------------------------- text streaming
_SENT_END = re.compile(r"(.*?(?:[.!?…](?=\s|$)|[؟。！？]))\s*", re.S)


def split_sentences(buf: str) -> tuple[list[str], str]:
    """Split complete sentences off the front of a streaming text buffer.

    Returns (sentences, remainder). Handles ., !, ?, … plus Arabic ؟ and
    CJK 。！？ so every app language can be spoken sentence-by-sentence.
    """
    out, pos = [], 0
    for m in _SENT_END.finditer(buf):
        s = m.group(1).strip()
        if s:
            out.append(s)
        pos = m.end()
    return out, buf[pos:]


# ------------------------------------------------------------ app actions
_ACTION_INLINE = re.compile(r"(?:[-*•]\s*)?ACTION\s*:\s*(\{[^{}]*\})[.。]?",
                            re.I)
_ACTION_LINE = re.compile(r"^\s*(?:[-*•]\s*)?ACTION\s*:", re.I)


def parse_actions(text: str) -> tuple[str, list[dict]]:
    """Split 'ACTION: {json}' commands out of reply text.

    Returns (clean_text_to_speak, actions). Valid actions are extracted
    wherever they appear; any leftover line that still starts with
    'ACTION:' (malformed JSON, unclosed brace) is dropped so raw JSON is
    never read aloud to the user.
    """
    actions: list[dict] = []

    def _grab(m: re.Match) -> str:
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            return m.group(0)          # leave it; the line filter drops it
        if isinstance(obj, dict) and obj.get("do"):
            actions.append(obj)
        return ""
    clean = _ACTION_INLINE.sub(_grab, text)
    clean = "\n".join(line for line in clean.splitlines()
                      if line.strip() and not _ACTION_LINE.match(line))
    return clean.strip(), actions


def execute_calendar_action(calendar, action: dict) -> tuple[str, str | None]:
    """Run a calendar_* action. Returns (spoken ack, feedback for the LLM).

    Feedback (agenda data / error text) is sent back to the model as an
    [APP DATA] message so it can finish answering with real facts.
    """
    do = action.get("do")
    try:
        if do == "calendar_check":
            try:
                days = min(max(int(action.get("days", 7) or 7), 1), 31)
            except (TypeError, ValueError):
                days = 7
            return ("Let me check your calendar.",
                    f"CALENDAR — next {days} days:\n" + calendar.agenda(days))
        if do == "calendar_book":
            title = str(action.get("title") or "Training with Coach")[:80]
            try:
                minutes = min(max(int(action.get("minutes", 60) or 60), 10),
                              240)
            except (TypeError, ValueError):
                minutes = 60
            when = calendar.book(title, str(action.get("start", "")), minutes)
            return (f"Booked {title}: {when}.", None)
    except coach_calendar.CalendarError as e:
        return ("", f"CALENDAR ERROR: {e}")     # model explains / retries
    except Exception as e:
        return ("", f"CALENDAR ERROR: {e}")
    return ("", None)


# ------------------------------------------------------- workout context
def progress_summary(log_path: str, limit: int = 6) -> str:
    """Compact text summary of the last sessions in workout_log.json."""
    if not os.path.exists(log_path):
        return "No workouts logged yet."
    try:
        with open(log_path, encoding="utf-8") as fh:
            history = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return "No workouts logged yet."
    if not isinstance(history, list) or not history:
        return "No workouts logged yet."
    lines = []
    for s in history[-limit:]:
        try:
            ex, started = s.get("exercise", "?"), s.get("started", "?")
            plank, summ = s.get("plank"), s.get("summary", {})
            if plank:
                lines.append(f"- {started} {ex}: hold {plank.get('total_hold_s')}s"
                             f" (best streak {plank.get('best_streak_s')}s)")
            else:
                part = (f"- {started} {ex}: {summ.get('reps')} reps"
                        f", avg score {summ.get('avg_score')}")
                if summ.get("velocity_loss_pct"):
                    part += f", velocity loss {summ['velocity_loss_pct']}%"
                faults = summ.get("fault_counts") or {}
                if faults:
                    top = sorted(faults.items(), key=lambda kv: -kv[1])[:3]
                    part += ", faults: " + ", ".join(f"{k}×{v}" for k, v in top)
                lines.append(part)
        except Exception:
            continue
    return "\n".join(lines) if lines else "No workouts logged yet."


class ChatCoach:
    """Conversation state: persona + history/live-session context + memory."""

    def __init__(self, client: LLMClient | None = None,
                 log_path: str = DEFAULT_LOG, state_provider=None,
                 profile: "coach_profile.ProfileStore | None" = None,
                 actions: bool = False,
                 calendar: "coach_calendar.CalendarClient | None" = None):
        self.client = client or LLMClient()
        self.log_path = log_path
        self.state_provider = state_provider   # () -> dict with live session
        self.profile = profile                 # long-term athlete facts
        self.actions = actions                 # app-control protocol enabled
        self.calendar = calendar               # Google Calendar, if connected
        self.history: list[dict] = []          # user/assistant turns only

    def _system(self) -> str:
        parts = [PERSONA, "", "TRAINING HISTORY (most recent last):",
                 progress_summary(self.log_path)]
        if self.actions:
            parts += ["", ACTIONS_PROMPT]
        if self.calendar is not None:
            parts += ["", CALENDAR_PROMPT]
        parts += ["", "NOW: " + datetime.now().astimezone().strftime(
            "%Y-%m-%d %H:%M, %A (UTC%z)")]
        if self.profile is not None:
            try:
                block = self.profile.as_prompt()
                if block:
                    parts += ["", block]
            except Exception:
                pass
        if self.state_provider:
            try:
                live = self.state_provider()
                parts += ["", "LIVE SESSION RIGHT NOW:",
                          json.dumps(live, ensure_ascii=False)]
            except Exception:
                pass
        return "\n".join(parts)

    def ask(self, text: str) -> str:
        self.history.append({"role": "user", "content": text})
        self.history = self.history[-MAX_TURNS:]
        messages = [{"role": "system", "content": self._system()}] + self.history
        reply = self.client.chat(messages)
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def ask_stream(self, text: str, cancel: threading.Event | None = None):
        """Yield the reply in chunks as the model writes it.

        If cancel is set mid-stream the answer stops early; whatever was
        already said is kept in history (marked "…") so a follow-up
        question stays coherent."""
        self.history.append({"role": "user", "content": text})
        self.history = self.history[-MAX_TURNS:]
        messages = [{"role": "system", "content": self._system()}] + self.history
        parts: list[str] = []
        try:
            for delta in self.client.chat_stream(messages):
                if cancel is not None and cancel.is_set():
                    parts.append(" …")
                    break
                parts.append(delta)
                yield delta
        finally:
            reply = "".join(parts).strip()
            if reply:
                self.history.append({"role": "assistant", "content": reply})

    def learn_async(self):
        """Mine the last exchange for durable athlete facts (background)."""
        if self.profile is None or not self.history:
            return
        if self.history[-1]["role"] != "assistant":
            return
        reply = self.history[-1]["content"]
        user = next((m["content"] for m in reversed(self.history)
                     if m["role"] == "user"), "")
        if not user:
            return

        def _bg():
            try:
                for f in coach_profile.extract_facts(self.client, user,
                                                     reply):
                    self.profile.remember(f["category"], f["key"],
                                          f["value"])
            except Exception:
                pass                      # memory is best-effort, never fatal
        threading.Thread(target=_bg, daemon=True).start()


# ------------------------------------------------------------ voice I/O
def voice_input_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        import sounddevice     # noqa: F401
        return True
    except Exception:
        return False


_whisper_model = None
_whisper_lock = threading.Lock()

# Whisper hallucinates these on noise-only audio — never treat as a question.
_JUNK = {"you", "you.", "uh", "um", "bye.", "thank you.", "thanks.",
         "thank you very much.", "thanks for watching!",
         "thank you for watching!", "subtitles by the amara.org community"}


def looks_like_speech(text: str) -> str:
    """Filter Whisper hallucinations; returns cleaned text or ''."""
    t = text.strip()
    if len(t) < 2 or not re.search(r"\w", t):
        return ""
    if t.lower() in _JUNK:
        return ""
    words = t.lower().split()
    # noise loops like "Music Music Music" / the same phrase over and over
    if len(words) >= 4 and len(set(words)) / len(words) <= 0.5:
        return ""
    return t


def _mic_hint() -> str:
    """Actionable hint for the most common Windows mic blocker."""
    import sys
    if sys.platform == "win32" and "WindowsApps" in sys.executable:
        return ("(hint: Microsoft Store Python cannot open the microphone "
                "on many Windows setups — install Python from python.org "
                "or `winget install Python.Python.3.12` and run the app "
                "with `py -3.12`)")
    return ("(hint: check the OS microphone permission for this app, and "
            "that an input device is plugged in and set as default)")


def _mic_volume_warning() -> str:
    """Windows: warn when the OS input volume is so low the coach is deaf.

    Found in the field: a mic at 8 % input volume opens fine and streams
    near-silence — everything 'works' except nothing is ever heard."""
    if sys.platform != "win32":
        return ""
    try:
        import comtypes
        try:
            comtypes.CoInitialize()          # we're called from a thread
        except Exception:
            pass
        from comtypes import CLSCTX_ALL
        from pycaw.constants import EDataFlow, ERole
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        dev = AudioUtilities.GetDeviceEnumerator().GetDefaultAudioEndpoint(
            EDataFlow.eCapture.value, ERole.eConsole.value)
        vol = dev.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL,
                           None).QueryInterface(IAudioEndpointVolume)
        pct = round(vol.GetMasterVolumeLevelScalar() * 100)
        if vol.GetMute():
            return ("WARNING: your microphone is MUTED in Windows — "
                    "unmute it in Settings > Sound > Input")
        if pct < 30:
            return (f"WARNING: Windows mic input volume is only {pct}% — "
                    "the coach probably can't hear you. Raise it in "
                    "Settings > Sound > Input > Volume (or: Sound Control "
                    "Panel > Recording > Microphone > Levels)")
    except Exception:
        pass
    return ""


def _load_whisper():
    global _whisper_model
    with _whisper_lock:
        if _whisper_model is None:
            from faster_whisper import WhisperModel
            # COACH_WHISPER_MODEL=tiny halves transcription time if needed
            _whisper_model = WhisperModel(
                os.environ.get("COACH_WHISPER_MODEL", "base"),
                device="cpu", compute_type="int8")
    return _whisper_model


def _normalize(audio, target: float = 0.5, max_gain: float = 30.0):
    """Peak-normalize quiet audio so Whisper hears low-gain mics."""
    import numpy as np
    peak = float(np.abs(audio).max())
    if 0 < peak < target:
        audio = audio * min(max_gain, target / peak)
    return audio


def _transcribe_audio(audio) -> str:
    """1-D float32 mono @ 16 kHz -> text ('' for silence/junk)."""
    # greedy decode (beam 1) is 2-3x faster than the default beam of 5 and
    # just as good on short gym questions; no cross-utterance conditioning
    segments, _info = _load_whisper().transcribe(
        _normalize(audio), beam_size=1, condition_on_previous_text=False)
    parts = [s.text.strip() for s in segments
             if getattr(s, "no_speech_prob", 0.0) < 0.6
             and getattr(s, "avg_logprob", 0.0) > -1.35]
    return looks_like_speech(" ".join(p for p in parts if p))


def record_and_transcribe(seconds: float = LISTEN_SECONDS) -> str:
    """Record from the default mic and transcribe locally (any language)."""
    import numpy as np
    import sounddevice as sd

    rate = 16000
    print(f"🎤 listening for {seconds:.0f}s — speak now...")
    audio = sd.rec(int(seconds * rate), samplerate=rate, channels=1,
                   dtype="float32")
    sd.wait()
    if _whisper_model is None:
        print("(loading speech model — first time downloads ~150 MB)")
    return _transcribe_audio(np.squeeze(audio))


class VadSegmenter:
    """Tiny energy-based voice-activity detector (pure logic, testable).

    Feed one RMS value per audio block; returns "start" when an utterance
    begins, "end" when it finishes, else None. The noise floor adapts to
    the room. gated=True (coach currently talking) hard-resets detection
    so the coach never triggers on its own voice.
    """

    def __init__(self, block_s: float = 0.03, start_blocks: int = 4,
                 end_silence_s: float = 0.75, max_utt_s: float = 15.0,
                 min_floor: float = 0.0008):
        self.block_s = block_s
        self.start_blocks = start_blocks
        self.end_blocks = max(1, int(end_silence_s / block_s))
        self.max_blocks = int(max_utt_s / block_s)
        self.min_floor = min_floor
        self.floor = 0.01
        self.in_speech = False
        self._above = 0
        self._silence = 0
        self._utt = 0

    @property
    def threshold(self) -> float:
        return max(3.5 * self.floor, 3 * self.min_floor)

    def feed(self, rms: float, gated: bool = False) -> str | None:
        if gated:
            self.in_speech = False
            self._above = self._silence = self._utt = 0
            return None
        if not self.in_speech:
            if rms < self.threshold:
                self.floor = 0.95 * self.floor + 0.05 * max(rms, self.min_floor)
                self._above = 0
                return None
            self._above += 1
            if self._above >= self.start_blocks:
                self.in_speech = True
                self._above = self._silence = self._utt = 0
                return "start"
            return None
        self._utt += 1
        self._silence = self._silence + 1 if rms < self.threshold else 0
        if self._silence >= self.end_blocks or self._utt >= self.max_blocks:
            self.in_speech = False
            return "end"
        return None


class HandsFreeListener:
    """Open-mic loop: VAD segments your speech -> local Whisper -> chat.

    There is no acoustic echo cancellation, so the mic is *gated* while
    the coach's TTS is talking (plus a short hangover) — it cannot hear
    you over its own voice. Interrupt by typing, or press 'c' to silence
    the coach and reopen the mic instantly.
    """

    RATE = 16000
    BLOCK = 480                    # 30 ms
    PRE_ROLL_BLOCKS = 15           # 450 ms of context kept before speech
    MIN_SPEECH_S = 0.35
    TTS_HANGOVER_S = 0.6

    def __init__(self, on_text, tts_active=None):
        self.on_text = on_text
        self.tts_active = tts_active or (lambda: False)
        self.state = "starting"
        self.level = 0.0               # rms/threshold ratio for HUD meters
        self._stop = threading.Event()
        self._gate_until = 0.0
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._stop.set()

    def open_gate_now(self):
        """Skip the post-TTS hangover (push-to-talk barge-in)."""
        self._gate_until = 0.0

    def _loop(self):
        import time as _t

        import numpy as np
        try:
            import sounddevice as sd
            stream = sd.InputStream(samplerate=self.RATE, channels=1,
                                    dtype="float32", blocksize=self.BLOCK)
            stream.start()
        except Exception as e:
            print(f"(hands-free mic unavailable: {e})")
            print(_mic_hint())
            self.state = "off"
            return
        vad = VadSegmenter()
        pre: deque = deque(maxlen=self.PRE_ROLL_BLOCKS)
        utt: list = []
        warn = _mic_volume_warning()
        if warn:
            print(warn)
        self.state = "listening"
        with stream:
            while not self._stop.is_set():
                try:
                    data, _ = stream.read(self.BLOCK)
                except Exception as e:
                    print(f"(mic stream lost: {e})")
                    self.state = "off"
                    return
                mono = np.squeeze(data)
                now = _t.monotonic()
                if self.tts_active():
                    self._gate_until = now + self.TTS_HANGOVER_S
                rms = float(np.sqrt(np.mean(mono ** 2)))
                self.level = rms / vad.threshold
                ev = vad.feed(rms, gated=now < self._gate_until)
                if ev == "start":
                    utt = list(pre)
                    pre.clear()
                    self.state = "hearing you..."
                if vad.in_speech:
                    utt.append(mono)
                else:
                    pre.append(mono)
                if ev != "end":
                    continue
                audio = np.concatenate(utt) if utt else np.zeros(1, "float32")
                utt = []
                speech_s = (len(audio) / self.RATE
                            - vad.end_blocks * vad.block_s)
                if speech_s < self.MIN_SPEECH_S:   # clank/cough blip
                    self.state = "listening"
                    continue
                self.state = "thinking..."
                text = ""
                try:
                    text = _transcribe_audio(audio)
                except Exception as e:
                    print(f"(transcription failed: {e})")
                if text:
                    print(f"\nYou (voice): {text}")
                    self.on_text(text)
                self.state = "listening"


class _Speaker:
    """Tiny background TTS (pyttsx3) so replies can be spoken.

    The engine is disposed and re-created after every interrupt: Windows
    SAPI often goes permanently silent if reused after engine.stop().
    """

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._engine = None
        self._speaking = False
        self._interrupted = threading.Event()
        if not enabled:
            return
        import queue
        self.q: "queue.Queue[str | None]" = queue.Queue()
        try:
            import pyttsx3  # noqa: F401
            threading.Thread(target=self._worker, daemon=True).start()
        except Exception:
            self.enabled = False
            print("(voice replies disabled: pyttsx3 unavailable)")

    def _worker(self):
        import pyttsx3
        engine = None
        while True:
            msg = self.q.get()
            if msg is None:
                return
            if engine is None:
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
                self._interrupted.clear()
                engine = self._engine = None    # never reuse after stop()

    def say(self, msg: str):
        if self.enabled:
            self.q.put(msg)

    def is_speaking(self) -> bool:
        return self.enabled and (self._speaking or not self.q.empty())

    def stop(self):
        """Barge-in: drop queued sentences and cut the current one."""
        if not self.enabled:
            return
        try:
            while True:
                self.q.get_nowait()
        except Exception:
            pass
        if self._speaking:
            self._interrupted.set()
            eng = self._engine
            if eng is not None:
                try:
                    eng.stop()
                except Exception:
                    pass


# ------------------------------------------- background chat for pose_coach
class BackgroundChat:
    """Terminal + push-to-talk chat running beside the workout loop.

    Answers stream in live and are spoken sentence-by-sentence. Barge-in:
    a new question typed (or spoken) while the coach is mid-answer cancels
    the rest of the reply — text, speech and LLM stream — and is answered
    next, so changing topic mid-sentence feels natural.
    """

    def __init__(self, coach: ChatCoach, speak=None, stop_speaking=None,
                 tts_active=None, hands_free: bool = False, on_action=None):
        import queue
        self.coach = coach
        self.speak = speak or (lambda _msg: None)
        self.stop_speaking = stop_speaking or (lambda: None)
        self.tts_active = tts_active or (lambda: False)
        self.on_action = on_action     # callable(dict) -> ack str, or None
        self.calendar = getattr(coach, "calendar", None)
        self._cancel = threading.Event()
        self._busy = False
        self._ptt = threading.Lock()   # one push-to-talk recording at a time
        self._q: "queue.Queue[str]" = queue.Queue()
        self.listener: HandsFreeListener | None = None
        if hands_free:
            if voice_input_available():
                self.listener = HandsFreeListener(
                    on_text=self.submit,
                    tts_active=lambda: self.tts_active() or self._busy)
                # preload Whisper now so the first utterance answers fast
                threading.Thread(target=_load_whisper, daemon=True).start()
                print("🎤 Hands-free mic is ON — just speak; the coach "
                      "answers. It can't hear you while it is talking "
                      "(press 'c' to cut it off and ask right away).")
            else:
                print("(hands-free mic needs: pip install -r "
                      "requirements-voice.txt — falling back to 'c' key)")
        threading.Thread(target=self._worker, daemon=True).start()
        threading.Thread(target=self._stdin_loop, daemon=True).start()

    @property
    def status(self) -> str:
        """One-word state for HUDs: listening / hearing you... / ..."""
        if self._busy:
            return "answering..."
        if self.listener is not None:
            return self.listener.state
        return "press c to talk"

    @property
    def mic_level(self) -> float:
        """Live input level as a multiple of the VAD threshold (HUD meter)."""
        return self.listener.level if self.listener is not None else 0.0

    def _stdin_loop(self):
        try:
            for line in sys.stdin:
                text = line.strip()
                if text:
                    self.submit(text)
        except Exception:
            pass

    def submit(self, text: str):
        """Queue a question; interrupts any answer in progress (barge-in)."""
        if text.strip().lower() == "/calendar":
            if self.calendar is None:
                print("\n(calendar not connected — see docs/COACH.md §5, "
                      "then: python coach_calendar.py --connect)")
            else:
                try:
                    print("\n" + self.calendar.agenda(7))
                except Exception as e:
                    print(f"\n(calendar error: {e})")
            return
        cmd_out = coach_profile.handle_command(
            getattr(self.coach, "profile", None), text)
        if cmd_out is not None:
            print("\n" + cmd_out)
            return
        if self._busy:
            self._cancel.set()
            self.stop_speaking()
            print("\n(interrupted — switching to your new question)")
        self._q.put(text)

    def _say_or_act(self, text: str, feedback: list[str]):
        """Route a reply chunk: ACTION lines drive the app or the calendar
        (results collected into `feedback` for a second LLM pass), plain
        text is spoken."""
        clean, acts = parse_actions(text)
        for a in acts:
            ack = None
            if str(a.get("do", "")).startswith("calendar_"):
                if self.calendar is not None:
                    ack, fb = execute_calendar_action(self.calendar, a)
                    if fb:
                        feedback.append(fb)
            elif self.on_action is not None:
                try:
                    ack = self.on_action(a)
                except Exception as e:
                    ack = f"(action failed: {e})"
                if ack and ack.startswith(("I couldn't", "I don't know")):
                    feedback.append(
                        f"APP ERROR: {ack} Fix the problem and send a "
                        "corrected ACTION line.")
            if ack:
                print(f"\n⚙️  {ack}")
                self.speak(ack)
        if clean:
            self.speak(clean)

    def _answer_once(self, text: str) -> str | None:
        """Stream one reply; returns [APP DATA] follow-up text when an
        action produced data the model still needs (tool loop)."""
        buf = ""
        feedback: list[str] = []
        print("\n🏋️  Coach: ", end="", flush=True)
        for chunk in self.coach.ask_stream(text, cancel=self._cancel):
            print(chunk, end="", flush=True)
            buf += chunk
            sents, buf = split_sentences(buf)
            for s in sents:
                self._say_or_act(s, feedback)
        if buf.strip() and not self._cancel.is_set():
            self._say_or_act(buf.strip(), feedback)
        print("\n")
        if feedback and not self._cancel.is_set():
            return ("[APP DATA — automatic message from the app, not the "
                    "athlete]\n" + "\n".join(feedback)
                    + "\nNow answer the athlete's request using this data.")
        return None

    def _worker(self):
        while True:
            text = self._q.get()
            self._busy = True
            self._cancel.clear()
            try:
                followup = self._answer_once(text)
                for _ in range(2):             # tool loop, hard-capped
                    if not followup or self._cancel.is_set():
                        break
                    followup = self._answer_once(followup)
                learn = getattr(self.coach, "learn_async", None)
                if learn:
                    learn()
            except CoachOffline as e:
                print(f"\n(coach offline) {e}\n")
            finally:
                self._busy = False

    def ask_async(self, text: str):
        self.submit(text)

    def push_to_talk(self):
        """Voice question via the 'c' key.

        Hands-free mode: silences the coach and reopens the mic instantly
        (barge-in). Otherwise: one ~6 s push-to-talk recording. Either way
        the coach is muted first so the mic never hears its own voice."""
        if not voice_input_available():
            print("(voice input needs: pip install -r requirements-voice.txt)")
            return
        if self.listener is not None:
            self._cancel.set()
            self.stop_speaking()
            self.listener.open_gate_now()
            print("\n(coach muted — mic open, go ahead)")
            return

        def _worker():
            if not self._ptt.acquire(blocking=False):
                return                 # already recording — ignore key spam
            try:
                if self._busy:
                    self._cancel.set()
                    self.stop_speaking()
                try:
                    text = record_and_transcribe()
                except Exception as e:
                    print(f"(mic/transcription failed: {e})")
                    print(_mic_hint())
                    return
                if not text:
                    print("(heard nothing)")
                    return
                print(f"You (voice): {text}")
                self.submit(text)
            finally:
                self._ptt.release()

        threading.Thread(target=_worker, daemon=True).start()


def start_background_chat(state_provider=None, speak=None, stop_speaking=None,
                          tts_active=None, hands_free: bool = False,
                          log_path: str = DEFAULT_LOG,
                          profile_db: str | None = "",
                          on_action=None) -> BackgroundChat:
    """profile_db: "" = default file, None = memory disabled."""
    profile = None
    if profile_db is not None:
        try:
            profile = coach_profile.ProfileStore(
                profile_db or coach_profile.DEFAULT_DB)
        except Exception:
            pass
    coach = ChatCoach(log_path=log_path, state_provider=state_provider,
                      profile=profile, actions=on_action is not None,
                      calendar=coach_calendar.connect_if_configured())
    coach.client.warm_up()      # load the LLM now, not on the first question
    print(f"Coach chat ready (LLM: {coach.client.model} @ "
          f"{coach.client.base_url}) — type a question anytime; asking "
          "again mid-answer interrupts the coach.")
    if profile is not None:
        print("The coach remembers you between sessions (local file "
              f"{profile.path}). Commands: /profile /remember /forget")
    if on_action is not None:
        print("The coach can drive the app: ask it to switch exercise, set "
              "a rep goal, start a rest timer, set tempo or mute cues.")
    if coach.calendar is not None:
        print("📅 Google Calendar connected — ask the coach to check your "
              "week or book a training session. /calendar shows the agenda.")
    return BackgroundChat(coach, speak=speak, stop_speaking=stop_speaking,
                          tts_active=tts_active, hands_free=hands_free,
                          on_action=on_action)


# ------------------------------------------------------------ interactive
def _speak_or_calendar(sentence: str, coach: ChatCoach, speaker,
                       feedback: list[str]):
    """interactive(): speak plain text, run calendar ACTION lines."""
    clean, acts = parse_actions(sentence)
    for a in acts:
        if (str(a.get("do", "")).startswith("calendar_")
                and coach.calendar is not None):
            ack, fb = execute_calendar_action(coach.calendar, a)
            if fb:
                feedback.append(fb)
            if ack:
                print(f"\n⚙️  {ack}")
                speaker.say(ack)
    if clean:
        speaker.say(clean)


def interactive(args):
    profile = None
    if not getattr(args, "no_profile", False):
        try:
            profile = coach_profile.ProfileStore(args.profile_file)
        except Exception as e:
            print(f"(profile store unavailable: {e})")
    coach = ChatCoach(LLMClient(args.base_url, args.model),
                      log_path=args.log_file, profile=profile,
                      calendar=coach_calendar.connect_if_configured())
    coach.client.warm_up()
    speaker = _Speaker(args.voice)
    if getattr(args, "hands_free", False):
        if not voice_input_available():
            print("--hands-free needs extras:  "
                  "pip install -r requirements-voice.txt")
        else:
            import time
            print(f"AI Gym Coach chat — model {coach.client.model} @ "
                  f"{coach.client.base_url}")
            BackgroundChat(coach, speak=speaker.say,
                           stop_speaking=speaker.stop,
                           tts_active=speaker.is_speaking, hands_free=True)
            print("Speak anytime — typing works too. Ctrl+C quits.")
            try:
                while True:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                print()
            return
    listen = args.listen
    if listen and not voice_input_available():
        print("--listen needs extras:  pip install -r requirements-voice.txt")
        listen = False
    print(f"AI Gym Coach chat — model {coach.client.model} @ "
          f"{coach.client.base_url}")
    print("Ask about your workouts, form, programming, nutrition basics.")
    if coach.calendar is not None:
        print("📅 Google Calendar connected — the coach can check your week "
              "and book sessions. /calendar shows the agenda.")
    print("Ctrl+C interrupts an answer. Commands: /quit"
          + (" /profile /remember /forget" if profile is not None else "")
          + (" /calendar" if coach.calendar is not None else "")
          + ("   (empty line = talk with the mic)" if listen else ""))
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.lower() in ("/quit", "/exit", "quit", "exit"):
            break
        if text.lower() == "/calendar" and coach.calendar is not None:
            try:
                print(coach.calendar.agenda(7))
            except Exception as e:
                print(f"(calendar error: {e})")
            continue
        cmd_out = coach_profile.handle_command(profile, text)
        if cmd_out is not None:
            print(cmd_out)
            continue
        if not text:
            if not listen:
                continue
            try:
                text = record_and_transcribe()
            except Exception as e:
                print(f"(mic/transcription failed: {e})")
                continue
            if not text:
                print("(heard nothing)")
                continue
            print(f"you (voice)> {text}")
        pending = text
        try:
            for _round in range(3):            # question + up to 2 data passes
                buf = ""
                feedback: list[str] = []
                print("\ncoach> ", end="", flush=True)
                for chunk in coach.ask_stream(pending):
                    print(chunk, end="", flush=True)
                    buf += chunk
                    sents, buf = split_sentences(buf)
                    for s in sents:
                        _speak_or_calendar(s, coach, speaker, feedback)
                if buf.strip():
                    _speak_or_calendar(buf.strip(), coach, speaker, feedback)
                print("\n")
                if not feedback:
                    break
                pending = ("[APP DATA — automatic message from the app, not "
                           "the athlete]\n" + "\n".join(feedback)
                           + "\nNow answer the athlete's request using this "
                           "data.")
            coach.learn_async()
        except KeyboardInterrupt:
            speaker.stop()
            print("\n(interrupted)\n")
        except CoachOffline as e:
            print(f"\n(coach offline) {e}")


# --------------------------------------------------------------- selftest
def selftest():
    import tempfile
    import unittest.mock as mock

    print("1) LLM client parses a chat completion:", end=" ")
    body = json.dumps({"choices": [{"message": {"content": " Push hard! "}}]})

    class FakeResp:
        def __init__(self, data): self.data = data.encode()
        def read(self): return self.data
        def __enter__(self): return self
        def __exit__(self, *a): return False

    with mock.patch.object(urllib.request, "urlopen",
                           return_value=FakeResp(body)) as m:
        out = LLMClient("http://x/v1", "m").chat([{"role": "user",
                                                   "content": "hi"}])
        assert out == "Push hard!", out
        sent = json.loads(m.call_args[0][0].data.decode())
        assert sent["model"] == "m" and sent["messages"][0]["content"] == "hi"
    print("ok")

    print("2) offline error is friendly:", end=" ")
    with mock.patch.object(urllib.request, "urlopen",
                           side_effect=urllib.error.URLError("refused")):
        try:
            LLMClient("http://localhost:11434/v1").chat([])
            raise AssertionError("expected CoachOffline")
        except CoachOffline as e:
            assert "docker compose up -d ollama" in str(e)
    print("ok")

    print("3) progress summary from workout log:", end=" ")
    log = [{"started": "2026-07-12 10:00:00", "exercise": "squat",
            "summary": {"reps": 10, "avg_score": 84.5,
                        "fault_counts": {"knees_cave": 3},
                        "velocity_loss_pct": 12.0}},
           {"started": "2026-07-12 11:00:00", "exercise": "plank",
            "plank": {"total_hold_s": 45.0, "best_streak_s": 30.0},
            "summary": {"reps": 0, "avg_score": None,
                        "fault_counts": {}, "velocity_loss_pct": None}}]
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "log.json")
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(log, fh)
        s = progress_summary(p)
        assert "squat: 10 reps" in s and "avg score 84.5" in s, s
        assert "knees_cave×3" in s and "hold 45.0s" in s, s
        assert progress_summary(os.path.join(td, "nope.json")) \
            == "No workouts logged yet."
    print("ok")

    print("4) live state lands in the system prompt:", end=" ")
    coach = ChatCoach(client=LLMClient("http://x/v1"), log_path="missing.json",
                      state_provider=lambda: {"exercise": "curl", "reps": 7})
    sysmsg = coach._system()
    assert "LIVE SESSION" in sysmsg and '"reps": 7' in sysmsg, sysmsg
    assert "TRAINING HISTORY" in sysmsg
    print("ok")

    print("5) history trimming + reply stored:", end=" ")
    with mock.patch.object(ChatCoach, "_system", return_value="sys"), \
         mock.patch.object(LLMClient, "chat", return_value="ok!") as m:
        c = ChatCoach(client=LLMClient("http://x/v1"))
        for i in range(30):
            c.ask(f"q{i}")
        assert len(c.history) <= MAX_TURNS + 1
        assert c.history[-1] == {"role": "assistant", "content": "ok!"}
        sent = m.call_args[0][0]
        assert sent[0]["role"] == "system"
        assert len(sent) <= MAX_TURNS + 1
    print("ok")

    print("6) streaming SSE parse:", end=" ")

    class FakeStream:
        def __init__(self, lines): self.lines = lines
        def __iter__(self): return iter(self.lines)
        def __enter__(self): return self
        def __exit__(self, *a): return False

    lines = [b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n',
             b'\n',
             b': keepalive\n',
             b'data: {"choices":[{"delta":{"content":"Push "}}]}\n',
             b'data: {"choices":[{"delta":{"content":"hard!"}}]}\n',
             b'data: [DONE]\n',
             b'data: {"choices":[{"delta":{"content":"IGNORED"}}]}\n']
    with mock.patch.object(urllib.request, "urlopen",
                           return_value=FakeStream(lines)):
        chunks = list(LLMClient("http://x/v1", "m").chat_stream(
            [{"role": "user", "content": "hi"}]))
    assert chunks == ["Push ", "hard!"], chunks
    print("ok")

    print("7) sentence splitting (6 languages):", end=" ")
    s, rest = split_sentences("Go deeper. Keep your chest up! And then")
    assert s == ["Go deeper.", "Keep your chest up!"] and rest == "And then", (s, rest)
    s, rest = split_sentences("Weight is 62.5 kg. Nice!")
    assert s == ["Weight is 62.5 kg.", "Nice!"] and rest == "", (s, rest)
    s, rest = split_sentences("هل تشعر بألم؟ توقف فوراً.")
    assert len(s) == 2 and rest == "", (s, rest)
    s, rest = split_sentences("加油！保持呼吸。继续")
    assert len(s) == 2 and rest == "继续", (s, rest)
    print("ok")

    print("8) barge-in interrupts an answer:", end=" ")
    import io
    import time as _time
    from contextlib import redirect_stdout

    cancel = threading.Event()

    def slow_stream(_msgs):
        for i in range(50):
            yield f"s{i}. "
            _time.sleep(0.005)

    with mock.patch.object(ChatCoach, "_system", return_value="sys"), \
         mock.patch.object(LLMClient, "chat_stream",
                           side_effect=lambda m: slow_stream(m)):
        c = ChatCoach(client=LLMClient("http://x/v1"))
        got = []
        for chunk in c.ask_stream("q", cancel=cancel):
            got.append(chunk)
            if len(got) == 3:
                cancel.set()
        assert len(got) == 3, got
        assert c.history[-1]["role"] == "assistant"
        assert c.history[-1]["content"].endswith("…")   # partial reply kept

    class FakeCoach:
        def __init__(self): self.calls = []

        def ask_stream(self, text, cancel=None):
            self.calls.append(text)
            for _ in range(100):
                if cancel is not None and cancel.is_set():
                    return
                yield "x. "
                _time.sleep(0.003)

    stops: list[int] = []
    fc = FakeCoach()
    with redirect_stdout(io.StringIO()):
        bc = BackgroundChat(fc, speak=lambda _s: None,
                            stop_speaking=lambda: stops.append(1))
        bc.submit("first")
        _time.sleep(0.08)
        bc.submit("second")                      # barge-in mid-answer
        deadline = _time.time() + 8
        while (len(fc.calls) < 2 or bc._busy) and _time.time() < deadline:
            _time.sleep(0.02)
    assert fc.calls == ["first", "second"], fc.calls
    assert stops, "stop_speaking was not called on barge-in"
    print("ok")

    print("9) VAD segments speech, gates during TTS, junk filter:", end=" ")
    v = VadSegmenter()
    for _ in range(100):
        assert v.feed(0.002) is None          # adapting to room noise
    seq = [v.feed(0.05) for _ in range(10)]   # someone talks
    assert "start" in seq and v.in_speech
    seq = [v.feed(0.001) for _ in range(40)]  # goes quiet
    assert "end" in seq and not v.in_speech
    v2 = VadSegmenter()
    for _ in range(50):
        v2.feed(0.002)
    assert all(v2.feed(0.08, gated=True) is None for _ in range(30))
    assert not v2.in_speech                   # coach's own voice ignored
    assert looks_like_speech(" Thanks for watching! ") == ""
    assert looks_like_speech("you") == ""
    assert looks_like_speech("...") == ""
    assert looks_like_speech("Music Music Music Music") == ""
    q = "How deep should I squat?"
    assert looks_like_speech(f"  {q} ") == q
    import numpy as _np
    quiet = _np.full(1600, 0.02, dtype="float32")      # low-gain mic
    boosted = _normalize(quiet)
    assert 0.45 <= float(_np.abs(boosted).max()) <= 0.55, boosted.max()
    tiny = _np.full(1600, 0.004, dtype="float32")      # gain capped at 30x
    assert abs(float(_np.abs(_normalize(tiny)).max()) - 0.12) < 0.01
    loud = _np.full(1600, 0.8, dtype="float32")
    assert _normalize(loud) is loud                    # untouched
    print("ok")

    print("10) athlete profile: prompt injection + auto-learning:", end=" ")
    with tempfile.TemporaryDirectory() as tmp:
        store = coach_profile.ProfileStore(os.path.join(tmp, "p.db"))
        store.remember("injuries", "left_knee", "meniscus strain")

        class ProfClient:
            model, base_url = "m", "http://x/v1"
            def __init__(self): self.extracted = threading.Event()
            def chat(self, messages):
                self.extracted.set()
                return ('[{"category":"goals","key":"target",'
                        '"value":"first pull-up"}]')
            def chat_stream(self, messages):
                yield "Nice. "

        pc = ProfClient()
        coach = ChatCoach(pc, log_path=os.path.join(tmp, "none.json"),
                          profile=store)
        assert "meniscus strain" in coach._system()
        list(coach.ask_stream("My goal is a pull-up"))
        coach.learn_async()
        assert pc.extracted.wait(5), "fact extraction never ran"
        deadline = _time.time() + 5
        while _time.time() < deadline:
            if any(k == "target" for _, k, _, _ in store.facts()):
                break
            _time.sleep(0.02)
        assert ("goals", "target", "first pull-up") in [
            r[:3] for r in store.facts()], store.facts()
        out = coach_profile.handle_command(store, "/profile")
        assert "first pull-up" in out
        assert coach_profile.handle_command(store, "not a command") is None
    print("ok")

    print("11) app-control actions: parse + routed, never spoken:", end=" ")
    clean, acts = parse_actions(
        'On it — squats next.\n'
        'ACTION: {"do": "set_exercise", "exercise": "squat"}.\n'
        'ACTION: {"do": "set_rep_goal", "reps": 10}')
    assert [a["do"] for a in acts] == ["set_exercise", "set_rep_goal"]
    assert clean == "On it — squats next." and "ACTION" not in clean
    assert parse_actions("Plain advice only.") == ("Plain advice only.", [])
    assert parse_actions('ACTION: {broken json') == ("", [])   # never spoken
    assert parse_actions('action: {"do":"cues","enabled":false}')[1][0][
        "enabled"] is False                                    # case + bool

    class ActingCoach:
        def ask_stream(self, text, cancel=None):
            yield 'Rest time. '
            yield 'Enjoy!\nACTION: {"do": "rest_timer", "seconds": 60}'

    spoken: list[str] = []
    fired: list[dict] = []
    with redirect_stdout(io.StringIO()):
        bc2 = BackgroundChat(ActingCoach(), speak=spoken.append,
                             on_action=lambda a: fired.append(a)
                             or f"Rest timer: {a['seconds']} seconds.")
        bc2.submit("give me a minute")
        deadline = _time.time() + 5
        while _time.time() < deadline and not fired:
            _time.sleep(0.02)
        _time.sleep(0.1)               # let the ack reach the speaker
    assert fired == [{"do": "rest_timer", "seconds": 60}], fired
    assert "Rest time." in spoken and any("Enjoy!" in s for s in spoken)
    assert any("Rest timer" in s for s in spoken), spoken
    assert not any("{" in s or s.upper().startswith("ACTION") for s in spoken)
    a_coach = ChatCoach(client=LLMClient("http://x/v1"),
                        log_path="missing.json", actions=True)
    assert "APP CONTROL" in a_coach._system()
    no_a = ChatCoach(client=LLMClient("http://x/v1"), log_path="missing.json")
    assert "APP CONTROL" not in no_a._system()
    print("ok")

    print("12) calendar: prompt gating, check/data/answer loop, booking:",
          end=" ")

    class FakeCal:
        def __init__(self):
            self.booked = []
        def agenda(self, days=7):
            return "- Mon 13 Jul 09:00 to 09:30: Standup"
        def book(self, title, start, minutes=60, description=""):
            self.booked.append((title, start, minutes))
            return "Tuesday 14 Jul 18:00–19:00"

    fake_cal = FakeCal()
    cal_coach = ChatCoach(client=LLMClient("http://x/v1"),
                          log_path="missing.json", calendar=fake_cal)
    assert "calendar_check" in cal_coach._system()
    assert "NOW:" in cal_coach._system()
    assert "calendar_check" not in no_a._system()
    ack, fb = execute_calendar_action(
        fake_cal, {"do": "calendar_check", "days": 3})
    assert "Standup" in fb and ack
    ack, fb = execute_calendar_action(
        fake_cal, {"do": "calendar_book", "title": "Leg day",
                   "start": "2026-07-14T18:00", "minutes": 45})
    assert fb is None and "Booked Leg day" in ack and "18:00" in ack
    assert fake_cal.booked == [("Leg day", "2026-07-14T18:00", 45)]

    class CalCoach:                      # check → [APP DATA] → real answer
        calendar = fake_cal
        def __init__(self): self.asked = []
        def ask_stream(self, text, cancel=None):
            self.asked.append(text)
            if text.startswith("[APP DATA"):
                yield "Tuesday 18:00 is free — book it?"
            else:
                yield 'Let me look. ACTION: {"do": "calendar_check", "days": 7}'

    cc = CalCoach()
    spoken2: list[str] = []
    with redirect_stdout(io.StringIO()):
        bc3 = BackgroundChat(cc, speak=spoken2.append)
        bc3.submit("when can I train this week?")
        deadline = _time.time() + 5
        while _time.time() < deadline and len(cc.asked) < 2:
            _time.sleep(0.02)
        _time.sleep(0.1)
    assert len(cc.asked) == 2, cc.asked
    assert cc.asked[1].startswith("[APP DATA") and "Standup" in cc.asked[1]
    assert any("book it" in s for s in spoken2), spoken2
    assert not any("{" in s for s in spoken2)
    print("ok")

    print("\nAll coach_chat selftests passed.")


# -------------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Chat with your AI gym coach (LLM)")
    ap.add_argument("--base-url", default=DEFAULT_BASE,
                    help=f"OpenAI-compatible API base (default {DEFAULT_BASE})")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"model name (default {DEFAULT_MODEL})")
    ap.add_argument("--log-file", default=DEFAULT_LOG,
                    help="workout log used for context")
    ap.add_argument("--voice", action="store_true",
                    help="speak replies aloud (TTS)")
    ap.add_argument("--listen", action="store_true",
                    help="empty input records the mic (needs voice extras)")
    ap.add_argument("--hands-free", action="store_true",
                    help="open-mic conversation: just speak, no keys "
                         "(needs voice extras)")
    ap.add_argument("--once", metavar="QUESTION",
                    help="ask one question, print the answer, exit")
    ap.add_argument("--profile-file", default=coach_profile.DEFAULT_DB,
                    help="athlete profile DB the coach remembers you with "
                         f"(default {coach_profile.DEFAULT_DB})")
    ap.add_argument("--no-profile", action="store_true",
                    help="don't read or store any athlete profile")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    elif args.once:
        try:
            prof = (None if args.no_profile
                    else coach_profile.ProfileStore(args.profile_file))
            print(ChatCoach(LLMClient(args.base_url, args.model),
                            log_path=args.log_file, profile=prof)
                  .ask(args.once))
        except CoachOffline as e:
            sys.exit(str(e))
    else:
        interactive(args)
