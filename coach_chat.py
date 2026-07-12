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
It answers in the language you speak to it.

Config (env vars):
    COACH_LLM_BASE_URL   default http://localhost:11434/v1   (Ollama)
    COACH_LLM_MODEL      default llama3.2:3b
    COACH_LLM_API_KEY    default "ollama" (set a real key for OpenAI etc.)
    COACH_LOG            default workout_log.json

Voice input needs optional extras (host only):  pip install -r requirements-voice.txt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import urllib.error
import urllib.request

DEFAULT_BASE = os.environ.get("COACH_LLM_BASE_URL", "http://localhost:11434/v1")
DEFAULT_MODEL = os.environ.get("COACH_LLM_MODEL", "llama3.2:3b")
DEFAULT_KEY = os.environ.get("COACH_LLM_API_KEY", "ollama")
DEFAULT_LOG = os.environ.get("COACH_LOG", "workout_log.json")

MAX_TURNS = 16          # user/assistant messages kept in context
LISTEN_SECONDS = 6      # push-to-talk recording length

PERSONA = """\
You are "Coach", the friendly personal trainer inside the AI Gym Coach app.
Style: encouraging, practical and concise — replies are read aloud, so keep
them to 2-4 short sentences (under 70 words) unless the user asks for detail.
Safety first: if the user mentions pain or injury, tell them to stop and see
a professional. Always reply in the language of the user's last message.
App facts you may rely on: supported exercises are squat, pushup, bench,
deadlift, lunge, shoulder_press, curl, pullup and plank; the app counts reps,
scores each rep 0-100, detects form faults, and estimates fatigue from
rep-velocity loss. Use the data blocks below to give specific, personal
advice; never invent numbers that are not in them."""


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

    def chat(self, messages: list[dict]) -> str:
        payload = json.dumps({"model": self.model, "messages": messages,
                              "stream": False}).encode()
        req = urllib.request.Request(
            self.base_url + "/chat/completions", data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode())
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
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as e:
            raise CoachOffline(f"Unexpected LLM response: {str(data)[:300]}") from e


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
                 log_path: str = DEFAULT_LOG, state_provider=None):
        self.client = client or LLMClient()
        self.log_path = log_path
        self.state_provider = state_provider   # () -> dict with live session
        self.history: list[dict] = []          # user/assistant turns only

    def _system(self) -> str:
        parts = [PERSONA, "", "TRAINING HISTORY (most recent last):",
                 progress_summary(self.log_path)]
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


# ------------------------------------------------------------ voice I/O
def voice_input_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        import sounddevice     # noqa: F401
        return True
    except Exception:
        return False


_whisper_model = None


def record_and_transcribe(seconds: float = LISTEN_SECONDS) -> str:
    """Record from the default mic and transcribe locally (any language)."""
    global _whisper_model
    import numpy as np
    import sounddevice as sd
    from faster_whisper import WhisperModel

    rate = 16000
    print(f"🎤 listening for {seconds:.0f}s — speak now...")
    audio = sd.rec(int(seconds * rate), samplerate=rate, channels=1,
                   dtype="float32")
    sd.wait()
    if _whisper_model is None:
        print("(loading speech model — first time downloads ~150 MB)")
        _whisper_model = WhisperModel("base", device="cpu",
                                      compute_type="int8")
    segments, _info = _whisper_model.transcribe(np.squeeze(audio))
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return text


class _Speaker:
    """Tiny background TTS (pyttsx3) so replies can be spoken."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
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
        if self.enabled:
            self.q.put(msg)


# ------------------------------------------- background chat for pose_coach
class BackgroundChat:
    """Terminal + push-to-talk chat running beside the workout loop."""

    def __init__(self, coach: ChatCoach, speak=None):
        self.coach = coach
        self.speak = speak or (lambda _msg: None)
        self._busy = False
        threading.Thread(target=self._stdin_loop, daemon=True).start()

    def _stdin_loop(self):
        try:
            for line in sys.stdin:
                text = line.strip()
                if text:
                    self._ask(text)
        except Exception:
            pass

    def _ask(self, text: str):
        if self._busy:
            print("(coach is still answering the previous question)")
            return
        self._busy = True
        try:
            reply = self.coach.ask(text)
            print(f"\n🏋️  Coach: {reply}\n")
            self.speak(reply)
        except CoachOffline as e:
            print(f"\n(coach offline) {e}\n")
        finally:
            self._busy = False

    def ask_async(self, text: str):
        threading.Thread(target=self._ask, args=(text,), daemon=True).start()

    def push_to_talk(self):
        """Record from the mic and send the transcript (needs voice extras)."""
        if self._busy:
            return
        if not voice_input_available():
            print("(voice input needs: pip install -r requirements-voice.txt)")
            return

        def _worker():
            try:
                text = record_and_transcribe()
            except Exception as e:
                print(f"(mic/transcription failed: {e})")
                return
            if not text:
                print("(heard nothing)")
                return
            print(f"You (voice): {text}")
            self._ask(text)

        threading.Thread(target=_worker, daemon=True).start()


def start_background_chat(state_provider=None, speak=None,
                          log_path: str = DEFAULT_LOG) -> BackgroundChat:
    coach = ChatCoach(log_path=log_path, state_provider=state_provider)
    print(f"Coach chat ready (LLM: {coach.client.model} @ "
          f"{coach.client.base_url}) — type a question in this terminal.")
    return BackgroundChat(coach, speak=speak)


# ------------------------------------------------------------ interactive
def interactive(args):
    coach = ChatCoach(LLMClient(args.base_url, args.model),
                      log_path=args.log_file)
    speaker = _Speaker(args.voice)
    listen = args.listen
    if listen and not voice_input_available():
        print("--listen needs extras:  pip install -r requirements-voice.txt")
        listen = False
    print(f"AI Gym Coach chat — model {coach.client.model} @ "
          f"{coach.client.base_url}")
    print("Ask about your workouts, form, programming, nutrition basics.")
    print("Commands: /quit" + ("   (empty line = talk with the mic)"
                               if listen else ""))
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.lower() in ("/quit", "/exit", "quit", "exit"):
            break
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
        try:
            reply = coach.ask(text)
        except CoachOffline as e:
            print(f"(coach offline) {e}")
            continue
        print(f"\ncoach> {reply}\n")
        speaker.say(reply)


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
    ap.add_argument("--once", metavar="QUESTION",
                    help="ask one question, print the answer, exit")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    elif args.once:
        try:
            print(ChatCoach(LLMClient(args.base_url, args.model),
                            log_path=args.log_file).ask(args.once))
        except CoachOffline as e:
            sys.exit(str(e))
    else:
        interactive(args)
