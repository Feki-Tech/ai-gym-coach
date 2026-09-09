#!/usr/bin/env python3
"""
Voice turn-taking for the coach — one honest state machine for "can it hear
me right now?", earcons that say so out loud, per-turn latency metrics, and
the deterministic harnesses that test all of it without a microphone.

States (what the athlete needs to know; the HUD panel and the phone pill
render exactly these):

    off        mic unavailable — type instead
    listening  mic open — just talk
    hearing    your sentence is being captured
    thinking   captured; transcribing / the LLM is working, no audio yet
    speaking   the coach has the floor (answer audio playing or streaming)

The mic gate follows the state: open in `listening` / `hearing`, closed in
every other state — unless *full duplex* (`--headset`: earbuds, or echo
cancellation on the phone), where it stays open in every state but `off`
so the athlete can interrupt by voice. Earcons: a rising two-note when the
mic (re)opens, a tick when a sentence was captured, a soft pulse every few
seconds while the answer is still being computed, a falling two-note when
what was heard was not usable. Every transition, earcon and turn metric
comes out of `TurnTaking`, which takes a clock, so the whole behaviour can
be replayed with a fake clock (data/voice_scenarios.jsonl) and audio can be
injected through coach_chat's `_TEST_MIC_FACTORY` seam (WAV fixtures in
data/voice_fixtures/, synthesized with espeak-ng).

    python coach_voice.py --selftest              # machine, scenarios, earcons, echo filter, WER
    python coach_voice.py --scenarios [FILE]      # replay the scenario timelines, print a table
    python coach_voice.py --make-fixtures [DIR]   # synthesize test audio (espeak-ng + numpy)
    python coach_voice.py --bench [DIR]           # WAVs -> real VAD -> real Whisper: WER + timings
    python coach_voice.py --echo-check            # real speakers + mic: is --headset safe here?
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

import coach_ops

OFF, LISTENING, HEARING, THINKING, SPEAKING = (
    "off", "listening", "hearing", "thinking", "speaking")
STATES = (OFF, LISTENING, HEARING, THINKING, SPEAKING)

# earcons the host plays (see Earcons)
OPEN, CAPTURED, THINK, REJECT = "open", "captured", "thinking", "reject"
EARCONS = (OPEN, CAPTURED, THINK, REJECT)

LABELS = {OFF: "mic off", LISTENING: "listening", HEARING: "hearing you…",
          THINKING: "thinking…", SPEAKING: "coach speaking"}

DEFAULT_SCENARIOS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "data", "voice_scenarios.jsonl")
DEFAULT_FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "data", "voice_fixtures")
RATE = 16000


# ---------------------------------------------------------------- turns
@dataclass
class Turn:
    """One athlete question, from the first sound to the end of the answer."""
    source: str                        # voice | typed
    t0: float                          # speech start (voice) or submit (typed)
    full_duplex: bool = False
    speech_end_t: float | None = None
    transcript_t: float | None = None
    words: int = 0
    answer_t: float | None = None      # the LLM started working
    first_audio_t: float | None = None
    end_t: float | None = None
    interrupted: bool = False
    pulses: int = 0                    # "thinking" earcons played

    @property
    def heard_t(self) -> float:
        return self.speech_end_t if self.speech_end_t is not None else self.t0

    @property
    def answering(self) -> bool:
        """The coach started on it (LLM working or audio out)."""
        return self.answer_t is not None or self.first_audio_t is not None

    @property
    def substantial(self) -> bool:
        """Worth a metrics row: something was transcribed or answered."""
        return (self.transcript_t is not None or self.answer_t is not None
                or self.first_audio_t is not None)

    def metrics(self) -> dict:
        def ms(a, b):
            if a is None or b is None:
                return None
            return int(round((b - a) * 1000))
        return {
            "source": self.source, "words": self.words,
            "stt_ms": ms(self.speech_end_t, self.transcript_t),
            "ttfa_ms": ms(self.heard_t, self.first_audio_t),  # heard → first word
            "llm_ms": ms(self.transcript_t, self.first_audio_t),
            "speak_ms": ms(self.first_audio_t, self.end_t),
            "total_ms": ms(self.t0, self.end_t),
            "interrupted": self.interrupted,
            "full_duplex": self.full_duplex,
        }


class TurnTaking:
    """The turn-taking state machine. Pure: takes a clock, emits effects.

    Feed it observations (`observe(tts=, busy=)` from whoever polls — the
    listener every audio block, the HUD every frame) and events
    (`speech_start/end`, `blip`, `transcript`, `typed`, `barge_in`,
    `mic_ready/lost`). Read `state`, `gate_open`, `label()`, `hint()`.
    `on_effect(name)` fires for earcons, `on_turn(metrics)` when a turn
    ends (default: a `turn` line in the coach_ops trace).
    """

    def __init__(self, clock=None, full_duplex: bool = False,
                 hangover_s: float = 0.5, barge_s: float = 0.3,
                 think_pulse_s: float = 5.0, long_close_s: float = 2.0,
                 answer_wait_s: float = 8.0, grace_s: float = 0.6,
                 on_effect=None, on_turn=None):
        self.clock = clock or time.monotonic
        self.full_duplex = bool(full_duplex)
        self.hangover_s = hangover_s      # echo tail after the coach's voice
        self.barge_s = barge_s            # key press → mic open
        self.think_pulse_s = think_pulse_s
        self.long_close_s = long_close_s  # unprompted closure that earns an "open"
        self.answer_wait_s = answer_wait_s
        self.grace_s = grace_s            # ignore a dying TTS after an interrupt
        self.on_effect = on_effect
        self.on_turn = on_turn if on_turn is not None else (
            lambda m: coach_ops.trace("turn", **m))
        self.state = OFF
        self.off_hint = "type your question in the terminal"
        self.turn: Turn | None = None
        self.turns: list[dict] = []                    # finished turn metrics
        self.transitions: list[tuple[float, str]] = []
        self.effects: list[tuple[float, str]] = []
        self.tts = False
        self.busy = False
        self._mic = False
        self._lock = threading.RLock()
        self._closed_t: float | None = None            # gate closed since
        self._reopen_at: float | None = None
        self._reopen_forced = False
        self._ignore_floor_until = 0.0
        self._pending: Turn | None = None              # coach's turn while the athlete talks over it

    # ---- derived
    @property
    def gate_open(self) -> bool:
        if not self._mic or self.state == OFF:
            return False
        if self.full_duplex:
            return True
        return self.state in (LISTENING, HEARING)

    def label(self, now: float | None = None) -> str:
        s = self.state
        if s == THINKING and self.turn is not None:
            waited = self._now(now) - self.turn.heard_t
            if waited >= 3:
                return f"thinking… {int(waited)} s"
        return LABELS.get(s, s)

    def hint(self) -> str:
        s = self.state
        if s == LISTENING:
            return "just talk"
        if s == HEARING:
            return "finish your sentence"
        if s == THINKING:
            return "hang on"
        if s == SPEAKING:
            return ("just talk to interrupt" if self.full_duplex
                    else "press c to interrupt")
        return self.off_hint

    # ---- internals
    def _now(self, now):
        return self.clock() if now is None else now

    def _set(self, state: str, now: float):
        if state == self.state:
            return
        was_open = self.gate_open
        self.state = state
        self.transitions.append((now, state))
        if was_open and not self.gate_open:
            self._closed_t = now

    def _fx(self, name: str, now: float):
        self.effects.append((now, name))
        if self.on_effect is not None:
            try:
                self.on_effect(name)
            except Exception:
                pass

    def _closed_long(self, now: float) -> bool:
        return (self._closed_t is not None
                and now - self._closed_t >= self.long_close_s)

    def _finish_turn(self, now: float, interrupted: bool = False):
        t, self.turn = self.turn, None
        if t is None or not t.substantial:
            return
        t.end_t = now
        t.interrupted = t.interrupted or interrupted
        m = t.metrics()
        self.turns.append(m)
        if self.on_turn is not None:
            try:
                self.on_turn(m)
            except Exception:
                pass

    def _open(self, now: float):
        self._reopen_at = None
        self._finish_turn(now)
        self._set(LISTENING if self._mic else OFF, now)
        self._closed_t = None

    def _schedule_open(self, now: float, delay: float, forced: bool,
                       signal: bool):
        if signal and self._mic:
            self._fx(OPEN, now)        # the tone plays before the gate opens
        if delay <= 0:
            self._open(now)
            return
        self._reopen_at = now + delay
        self._reopen_forced = forced

    # ---- observations
    def observe(self, tts: bool | None = None, busy: bool | None = None,
                now: float | None = None) -> str:
        """Tell the machine whether the coach's voice / the LLM is active."""
        with self._lock:
            now = self._now(now)
            if tts is not None:
                self.tts = bool(tts)
            if busy is not None:
                self.busy = bool(busy)
            if not self.tts and not self.busy:
                self._ignore_floor_until = 0.0      # the old answer is gone
            grace = now < self._ignore_floor_until
            tts_on = self.tts and not grace
            busy_on = self.busy and not grace
            floor = tts_on or busy_on
            t = self.turn
            if self._reopen_at is not None:
                if floor and not self._reopen_forced:
                    self._reopen_at = None          # the coach resumed
                elif now >= self._reopen_at:
                    self._open(now)
                    return self.state
                else:
                    return self.state               # gate opens shortly
            s = self.state
            mine = t is not None and t.transcript_t is not None  # submitted
            if s in (LISTENING, HEARING, OFF):
                if tts_on and not (s == HEARING and self.full_duplex):
                    self._set(SPEAKING, now)
                    if mine and t.first_audio_t is None:
                        t.first_audio_t = now
                elif busy_on and s != HEARING:
                    self._set(THINKING, now)
                    if mine and t.answer_t is None:
                        t.answer_t = now
            elif s == THINKING and t is not None and not mine:
                pass            # transcribing: the floor is not about this
            elif s == THINKING:
                if tts_on:
                    self._set(SPEAKING, now)
                    if mine and t.first_audio_t is None:
                        t.first_audio_t = now
                elif busy_on:
                    if mine and t.answer_t is None:
                        t.answer_t = now
                elif t is None:
                    self._open(now)                  # unprompted, no audio
                elif t.answer_t is not None:
                    # answered without audio (offline, muted, action only)
                    self._schedule_open(now, 0.0, True, True)
                elif (t.transcript_t is not None
                      and now - t.transcript_t >= self.answer_wait_s):
                    self._schedule_open(now, 0.0, True, True)   # give up
                if (self.state == THINKING and t is not None
                        and t.first_audio_t is None):
                    due = t.heard_t + self.think_pulse_s * (t.pulses + 1)
                    if now >= due:
                        t.pulses += 1
                        self._fx(THINK, now)
            elif s == SPEAKING and not floor:
                signal = (t is not None) or self._closed_long(now)
                delay = (0.0 if self.full_duplex or not self._mic
                         else self.hangover_s)
                self._schedule_open(now, delay, False,
                                    signal and not self.full_duplex)
            return self.state

    # ---- events
    def mic_ready(self, now: float | None = None):
        with self._lock:
            now = self._now(now)
            self._mic = True
            if self.state == OFF:
                self._set(LISTENING, now)
                self._fx(OPEN, now)

    def mic_lost(self, now: float | None = None, hint: str | None = None):
        with self._lock:
            now = self._now(now)
            self._mic = False
            self._reopen_at = None
            self._finish_turn(now, interrupted=self.state == SPEAKING)
            if hint:
                self.off_hint = hint
            self._set(OFF, now)

    def speech_start(self, now: float | None = None):
        with self._lock:
            now = self._now(now)
            if not self._mic:
                return
            if self.state == LISTENING:
                self.turn = Turn("voice", now, self.full_duplex)
                self._set(HEARING, now)
            elif self.full_duplex and self.state in (THINKING, SPEAKING):
                # talking over the coach: capture it, the transcript decides
                t = self.turn
                if t is None:
                    self.turn = Turn("voice", now, self.full_duplex)
                elif t.transcript_t is not None or t.answer_t is not None:
                    self._pending = t              # the coach's running turn
                    self.turn = Turn("voice", now, self.full_duplex)
                # else: a voice turn still waiting for its transcript — keep it
                self._set(HEARING, now)

    def blip(self, now: float | None = None):
        """Too short to be speech (a clank) or the coach's own voice."""
        with self._lock:
            now = self._now(now)
            if self.state not in (HEARING, THINKING):
                return
            t = self.turn
            if t is not None and not t.substantial:
                self.turn, self._pending = self._pending, None
            if self.turn is not None and self.turn.answering:
                # back to the coach's answer in progress (full duplex)
                self._set(SPEAKING if self.tts else THINKING, now)
            else:
                self._set(LISTENING if self._mic else OFF, now)
                self._closed_t = None

    def speech_end(self, now: float | None = None):
        with self._lock:
            now = self._now(now)
            if self.state != HEARING:
                return
            t = self.turn
            if t is not None and t.transcript_t is None:
                t.speech_end_t = now
            self._fx(CAPTURED, now)
            self._set(THINKING, now)

    def _interrupt_current(self, now: float):
        t = self.turn
        if t is None:
            return
        self._finish_turn(now, interrupted=t.answering)

    def _start_answer_wait(self, now: float):
        if self.tts or self.busy:
            self._ignore_floor_until = now + self.grace_s   # old answer dying
        self._reopen_at = None
        self._set(THINKING, now)

    def transcript(self, text_or_words, now: float | None = None):
        """What the transcriber made of the captured audio ('' = nothing)."""
        words = (len(text_or_words.split()) if isinstance(text_or_words, str)
                 else int(text_or_words or 0))
        with self._lock:
            now = self._now(now)
            if words <= 0:
                t = self.turn
                if t is not None and not t.substantial:
                    self.turn, self._pending = self._pending, None
                self._fx(REJECT, now)
                if self.turn is not None and self.turn.answering:
                    self._set(SPEAKING if self.tts else THINKING, now)
                else:
                    self._schedule_open(now, 0.0, True, False)
                return
            pending, self._pending = self._pending, None
            t = self.turn
            if t is None or t.transcript_t is not None or t.answer_t is not None:
                self._interrupt_current(now)
                t = Turn("voice", now, self.full_duplex)
            elif pending is not None:
                keep, self.turn = self.turn, pending
                self._interrupt_current(now)
                t = keep
            t.transcript_t = now
            t.words = words
            self.turn = t
            self._start_answer_wait(now)

    def typed(self, words: int = 3, now: float | None = None):
        """A question typed in the terminal (or sent from a phone)."""
        with self._lock:
            now = self._now(now)
            self._interrupt_current(now)
            self.turn = Turn("typed", now, self.full_duplex,
                             transcript_t=now, words=int(words))
            self._start_answer_wait(now)

    def answer_started(self, now: float | None = None):
        """The host began answering the current question (the LLM call
        started): ends the interrupt grace so its voice counts."""
        with self._lock:
            now = self._now(now)
            self._ignore_floor_until = 0.0
            t = self.turn
            if t is not None and t.transcript_t is not None:
                if t.answer_t is None:
                    t.answer_t = now
                if self.state in (LISTENING, OFF):
                    self._set(THINKING, now)

    def barge_in(self, now: float | None = None):
        """The host cut the coach off (the 'c' key): give the mic back."""
        with self._lock:
            now = self._now(now)
            if self.turn is not None:
                self.turn.interrupted = self.turn.answering
            self._ignore_floor_until = now + self.grace_s
            if not self._mic:
                self._finish_turn(now)
                self._set(OFF, now)
                return
            self._schedule_open(now, 0.0 if self.full_duplex else self.barge_s,
                                True, True)


# --------------------------------------------------------------- earcons
def _tone(freq: float, ms: float, gain: float, rate: int = RATE):
    import numpy as np
    n = max(1, int(rate * ms / 1000))
    t = np.arange(n) / rate
    w = np.sin(2 * math.pi * freq * t) * gain
    fade = max(1, min(n // 3, int(rate * 0.006)))
    env = np.ones(n)
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    return (w * env).astype("float32")


_EARCON_SPECS = {
    OPEN: [(660, 60, 0.28), (880, 100, 0.28)],       # rising: your turn
    CAPTURED: [(1046, 50, 0.25)],                   # tick: got it
    THINK: [(440, 70, 0.10)],                       # soft pulse: still on it
    REJECT: [(330, 110, 0.22), (262, 130, 0.22)],   # falling: didn't catch that
}


def earcon_wave(name: str, rate: int = RATE):
    """float32 mono samples for an earcon (pure; no audio device needed)."""
    import numpy as np
    return np.concatenate([_tone(f, ms, g, rate) for f, ms, g in
                           _EARCON_SPECS[name]])


class Earcons:
    """Plays the earcons through the default output (sounddevice), or
    through an injected `player(wave, rate)`. COACH_EARCONS=0 silences."""

    def __init__(self, enabled: bool | None = None, player=None,
                 rate: int = RATE):
        self.enabled = ((os.environ.get("COACH_EARCONS", "1") != "0")
                        if enabled is None else bool(enabled))
        self.player = player
        self.rate = rate
        self.played: list[str] = []

    def duration_s(self, name: str) -> float:
        return sum(ms for _f, ms, _g in _EARCON_SPECS[name]) / 1000

    def play(self, name: str) -> bool:
        if not self.enabled or name not in _EARCON_SPECS:
            return False
        self.played.append(name)
        try:
            wave = earcon_wave(name, self.rate)
            if self.player is not None:
                self.player(wave, self.rate)
            else:
                import sounddevice as sd
                sd.play(wave, self.rate)
        except Exception:
            return False
        return True


# ------------------------------------------------------------ echo filter
_WORD_RE = re.compile(r"[\w']+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "")]


def looks_like_echo(text: str, spoken, min_overlap: float = 0.6) -> bool:
    """Is a transcript the coach's own recent sentence leaking into the mic?

    Full duplex has no gate, so a bit of the coach's voice can come back
    through the microphone. A transcript that mostly consists of words from
    one recent spoken sentence (or two consecutive ones) is an echo, never a
    question. Short transcripts must be a contiguous piece of a sentence.
    """
    toks = _tokens(text)
    sents = [_tokens(s) for s in (spoken or []) if s]
    sents = [s for s in sents if s]
    if not toks or not sents:
        return False
    cands = list(sents) + [a + b for a, b in zip(sents, sents[1:])]
    if len(toks) < 3:
        piece = " ".join(toks)
        return any(f" {piece} " in f" {' '.join(c)} " for c in cands)
    best = max(sum(1 for w in toks if w in set(c)) / len(toks) for c in cands)
    return best >= min_overlap


# --------------------------------------------------- scenario replay (L1)
class FakeClock:
    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float):
        self.t += dt


def load_scenarios(path: str = DEFAULT_SCENARIOS) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(json.loads(line))
    return out


def run_scenario(spec: dict, step_s: float = 0.03) -> dict:
    """Replay a timeline against TurnTaking with a fake clock.

    steps: [{"at": s, "ev": name, ...}] with ev in mic_ready, mic_lost,
    speech_start, speech_end, blip, transcript{text|words}, typed{words},
    barge_in, tts{on}, busy{on}. Returns the state trace, effects, turns,
    invariant violations and the expectation check.
    """
    clock = FakeClock()
    tt = TurnTaking(clock=clock, full_duplex=spec.get("full_duplex", False),
                    on_turn=lambda m: None, **spec.get("params", {}))
    steps = sorted(spec.get("steps", []), key=lambda s: s["at"])
    until = spec.get("until", (steps[-1]["at"] if steps else 0) + 3.0)
    tts = busy = False
    samples: list[tuple] = []
    violations: list[str] = []
    i = 0
    while clock.t <= until + 1e-9:
        while i < len(steps) and steps[i]["at"] <= clock.t + 1e-9:
            ev = steps[i]
            i += 1
            k = ev["ev"]
            if k == "tts":
                tts = bool(ev.get("on", True))
            elif k == "busy":
                busy = bool(ev.get("on", True))
            elif k == "mic_ready":
                tt.mic_ready()
            elif k == "mic_lost":
                tt.mic_lost()
            elif k == "speech_start":
                tt.speech_start()
            elif k == "speech_end":
                tt.speech_end()
            elif k == "blip":
                tt.blip()
            elif k == "transcript":
                tt.transcript(ev.get("text", ev.get("words", 0)))
            elif k == "typed":
                tt.typed(int(ev.get("words", 3)))
            elif k == "barge_in":
                tt.barge_in()
            else:
                raise ValueError(f"unknown scenario event {k!r}")
        tt.observe(tts=tts, busy=busy)
        t = round(clock.t, 3)
        samples.append((t, tt.state, tt.gate_open, tts, busy, tt.label()))
        grace = clock.t < tt._ignore_floor_until
        if tt.state == LISTENING and tt._mic and not tt.gate_open:
            violations.append(f"t={t}: 'listening' shown with the mic closed")
        if tt.gate_open and not tt.full_duplex and tt.state not in (
                LISTENING, HEARING):
            violations.append(f"t={t}: mic open while {tt.state}")
        if tts and not tt.full_duplex and tt.gate_open and not grace:
            violations.append(f"t={t}: mic open while the coach talks")
        if tt.state == OFF and tt.gate_open:
            violations.append(f"t={t}: mic open while off")
        clock.advance(step_s)
    states = [s for _t, s in tt.transitions]
    effects = [e for _t, e in tt.effects]
    res = {"id": spec.get("id", "?"), "final_state": tt.state,
           "states": states, "effects": effects, "turns": tt.turns,
           "violations": violations, "samples": samples,
           "transitions": tt.transitions, "effect_times": tt.effects}
    res["problems"] = violations + check_expectations(spec.get("expect", {}),
                                                     res)
    res["ok"] = not res["problems"]
    return res


def _sample_at(samples, t: float):
    best = None
    for s in samples:
        if s[0] <= t + 1e-9:
            best = s
        else:
            break
    return best


def check_expectations(exp: dict, res: dict) -> list[str]:
    bad = []
    if "final_state" in exp and res["final_state"] != exp["final_state"]:
        bad.append(f"final state {res['final_state']} != {exp['final_state']}")
    if "states" in exp and res["states"] != exp["states"]:
        bad.append(f"states {res['states']} != {exp['states']}")
    if "effects" in exp and res["effects"] != exp["effects"]:
        bad.append(f"effects {res['effects']} != {exp['effects']}")
    if "turns" in exp and len(res["turns"]) != exp["turns"]:
        bad.append(f"{len(res['turns'])} turns != {exp['turns']}")
    for spec in exp.get("turn", []):
        idx = spec.get("index", -1)
        if not res["turns"]:
            bad.append("no turn recorded")
            break
        m = res["turns"][idx]
        for k, want in spec.items():
            if k == "index":
                continue
            got = m.get(k)
            if isinstance(want, list) and len(want) == 2 and all(
                    isinstance(x, (int, float)) for x in want):
                if got is None or not want[0] <= got <= want[1]:
                    bad.append(f"turn[{idx}].{k}={got} not in {want}")
            elif got != want:
                bad.append(f"turn[{idx}].{k}={got} != {want}")
    for t, state in exp.get("state_at", []):
        s = _sample_at(res["samples"], t)
        if s is None or s[1] != state:
            bad.append(f"state at {t}s is {s and s[1]} not {state}")
    for t, want in exp.get("gate_at", []):
        s = _sample_at(res["samples"], t)
        if s is None or s[2] != bool(want):
            bad.append(f"mic at {t}s is {'open' if s and s[2] else 'closed'}"
                       f", expected {'open' if want else 'closed'}")
    for t, sub in exp.get("label_at", []):
        s = _sample_at(res["samples"], t)
        lab = s[5] if s else ""
        if sub not in lab:
            bad.append(f"label at {t}s is {lab!r}, expected {sub!r} in it")
    if "gate_always_open_after" in exp:
        t0 = exp["gate_always_open_after"]
        closed = [s[0] for s in res["samples"] if s[0] >= t0 and not s[2]]
        if closed:
            bad.append(f"mic closed at {closed[:3]} (full duplex)")
    if exp.get("gate_never_open") and any(s[2] for s in res["samples"]):
        bad.append("mic opened")
    return bad


def format_scenarios(results: list[dict]) -> str:
    lines = [f"{'scenario':32} {'ok':3} {'final':10} effects / states"]
    for r in results:
        lines.append(f"{r['id']:32} {'ok' if r['ok'] else 'FAIL':3} "
                     f"{r['final_state']:10} {','.join(r['effects']) or '-'} / "
                     f"{'>'.join(r['states'])}")
        for p in r["problems"]:
            lines.append(f"    ! {p}")
    return "\n".join(lines)


# ------------------------------------------------- audio fixtures (L2)
def read_wav(path: str, rate: int = RATE):
    """WAV (any rate, 8/16/32-bit int or float, mono/stereo) → float32 mono."""
    import wave

    import numpy as np
    with wave.open(path, "rb") as w:
        ch, width, sr, n = (w.getnchannels(), w.getsampwidth(),
                            w.getframerate(), w.getnframes())
        raw = w.readframes(n)
    if width == 2:
        a = np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0
    elif width == 4:
        a = np.frombuffer(raw, dtype="<i4").astype("float32") / 2147483648.0
    elif width == 1:
        a = (np.frombuffer(raw, dtype="u1").astype("float32") - 128) / 128.0
    else:
        raise ValueError(f"unsupported sample width {width}")
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    if sr != rate and len(a):
        n_out = int(round(len(a) * rate / sr))
        a = np.interp(np.linspace(0, len(a) - 1, n_out),
                      np.arange(len(a)), a).astype("float32")
    return a.astype("float32")


def write_wav(path: str, audio, rate: int = RATE):
    import wave

    import numpy as np
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


class WavMic:
    """A microphone that plays WAV data into HandsFreeListener.

    Same surface as sounddevice.InputStream (start/read/with). Every block
    advances the FakeClock by the block's duration, so VAD timing, hangovers
    and turn metrics are in audio time — identical on every machine.
    """

    def __init__(self, parts, clock: FakeClock | None = None,
                 tail_s: float = 1.5, lead_s: float = 0.6, rate: int = RATE):
        import numpy as np
        chunks = [np.zeros(int(lead_s * rate), "float32")]
        for p in parts:
            chunks.append(read_wav(p, rate) if isinstance(p, str)
                          else np.asarray(p, dtype="float32"))
        chunks.append(np.zeros(int(tail_s * rate), "float32"))
        self.samples = np.concatenate(chunks)
        self.rate = rate
        self.clock = clock
        self.pos = 0

    def start(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n: int):
        import numpy as np
        if self.pos >= len(self.samples):
            return None, False
        block = self.samples[self.pos:self.pos + n]
        self.pos += n
        if len(block) < n:
            block = np.concatenate([block, np.zeros(n - len(block), "float32")])
        if self.clock is not None:
            self.clock.advance(n / self.rate)
        return block.reshape(-1, 1), False


_NUM_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_NUM_WORDS.update({"twenty": "20", "thirty": "30", "forty": "40",
                   "fifty": "50", "sixty": "60", "seventy": "70",
                   "eighty": "80", "ninety": "90", "hundred": "100"})


def _wer_tokens(text: str) -> list[str]:
    """Tokens for scoring: number words count as their digits ("ninety"
    == "90"), so a transcriber's spelling choice is not an error."""
    return [_NUM_WORDS.get(w, w) for w in _tokens(text)]


def wer(ref: str, hyp: str) -> float:
    """Word error rate on normalized tokens (0 = perfect, ≥1 = useless)."""
    r, h = _wer_tokens(ref), _wer_tokens(hyp)
    if not r:
        return 0.0 if not h else 1.0
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1,
                       prev + (0 if r[i - 1] == h[j - 1] else 1))
            prev = cur
    return d[len(h)] / len(r)


# Synthesized with espeak-ng: `voice`/`wpm` per entry because Whisper's
# tolerance of the robotic voice is sentence-specific (tuned on Whisper
# base int8); `say` is what espeak reads when it differs from the expected
# `text` (Whisper writes numbers as digits). Real recordings beat all of
# this — drop 16 kHz WAVs next to the manifest and add an entry;
# --make-fixtures keeps entries it did not generate.
FIXTURES = [
    # kind question: must transcribe within max_wer; soft = report only
    {"file": "q_en_depth.wav", "kind": "question", "lang": "en",
     "voice": "en-us", "wpm": 130,
     "text": "How deep should I squat?", "max_wer": 0.34},
    {"file": "q_en_knees.wav", "kind": "question", "lang": "en",
     "voice": "en", "wpm": 150,
     "text": "Why do my knees cave in on the way up?", "max_wer": 0.34},
    {"file": "q_en_rest.wav", "kind": "question", "lang": "en",
     "voice": "en-us", "wpm": 130,
     "text": "Give me two minutes of rest.", "max_wer": 0.34},
    {"file": "q_fr_dos.wav", "kind": "question", "lang": "fr",
     "voice": "fr+f3", "wpm": 120,
     "text": "Pourquoi mon dos se penche en avant pendant les squats?",
     "max_wer": 0.5, "soft": True},
    {"file": "q_ar_rest.wav", "kind": "question", "lang": "ar",
     "voice": "ar", "wpm": 140,
     "text": "كم ثانية أرتاح بين المجموعات", "max_wer": 0.6, "soft": True},
    # kind noise: the listener must not hand any text to the coach
    {"file": "noise_gym.wav", "kind": "noise"},
    {"file": "noise_plate.wav", "kind": "noise"},
    {"file": "silence.wav", "kind": "noise"},
    # kind echo: the coach's own sentence through the mic → dropped as echo
    {"file": "echo_coach.wav", "kind": "echo", "lang": "en",
     "voice": "en", "wpm": 150,
     "text": "Push your knees out over your toes and keep your chest up."},
]


def _espeak(text: str, lang: str, out_path: str, wpm: int = 150,
            rate: int = RATE, level: float = 0.5):
    import tempfile

    import numpy as np
    with tempfile.TemporaryDirectory() as td:
        raw = os.path.join(td, "raw.wav")
        subprocess.run(["espeak-ng", "-v", lang, "-s", str(wpm), "-w", raw,
                        text], check=True, capture_output=True)
        a = read_wav(raw, rate)
    peak = float(np.abs(a).max()) or 1.0
    write_wav(out_path, a * (level / peak), rate)


def make_fixtures(out_dir: str = DEFAULT_FIXTURES, rate: int = RATE) -> list[str]:
    """Synthesize the WAV fixtures (espeak-ng voices + numpy noise)."""
    import numpy as np
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(7)
    paths = []
    for f in FIXTURES:
        p = os.path.join(out_dir, f["file"])
        if f["kind"] in ("question", "echo"):
            _espeak(f.get("say", f["text"]), f.get("voice", f["lang"]), p,
                    wpm=int(f.get("wpm", 150)), rate=rate,
                    level=0.5 if f["kind"] == "question" else 0.35)
        elif f["file"] == "noise_gym.wav":
            # room rumble (low-passed white noise) + three short clanks
            n = 4 * rate
            white = rng.normal(0, 1, n)
            rumble = np.convolve(white, np.ones(64) / 64, mode="same") * 0.03
            for at in (1.0, 2.2, 3.1):
                i = int(at * rate)
                rumble[i:i + int(0.04 * rate)] += rng.normal(0, 0.35,
                                                             int(0.04 * rate))
            write_wav(p, rumble, rate)
        elif f["file"] == "noise_plate.wav":
            # a plate rolling on the floor: 0.7 s of broadband noise — long
            # enough to trigger the VAD, so Whisper's junk filter is exercised
            n = 3 * rate
            a = rng.normal(0, 0.004, n)
            i = int(1.0 * rate)
            burst = rng.normal(0, 0.3, int(0.7 * rate))
            burst *= np.linspace(1, 0.2, len(burst))
            a[i:i + len(burst)] += burst
            write_wav(p, a, rate)
        elif f["file"] == "silence.wav":
            write_wav(p, rng.normal(0, 0.0015, 3 * rate), rate)
        paths.append(p)
    # keep entries someone added by hand (their own recordings)
    man_path = os.path.join(out_dir, "manifest.json")
    fixtures = list(FIXTURES)
    if os.path.exists(man_path):
        try:
            with open(man_path, encoding="utf-8") as fh:
                old = json.load(fh).get("fixtures", [])
            mine = {f["file"] for f in FIXTURES}
            fixtures += [f for f in old if f.get("file") not in mine]
        except Exception:
            pass
    with open(man_path, "w", encoding="utf-8") as fh:
        json.dump({"rate": rate, "fixtures": fixtures}, fh,
                  ensure_ascii=False, indent=1)
    return paths


def bench_fixture(fix: dict, audio, rate: int = RATE,
                  transcribe=None) -> dict:
    """Run one fixture through the real listener (VAD → transcriber →
    echo filter → turn machine) on a fake clock. `transcribe` overrides
    coach_chat._transcribe_audio (tests); default = real Whisper."""
    from unittest import mock

    import coach_chat
    clock = FakeClock()
    heard: list[str] = []
    effects: list[str] = []
    calls: list[dict] = []
    real = coach_chat._transcribe_audio if transcribe is None else transcribe

    def timed(a):
        t0 = time.perf_counter()
        text = real(a)
        calls.append({"text": text, "wall_ms": int((time.perf_counter() - t0)
                                                   * 1000),
                      "audio_s": round(len(a) / rate, 2)})
        return text

    spoken = [fix["text"]] if fix["kind"] == "echo" else []
    tt = TurnTaking(clock=clock, on_effect=effects.append,
                    on_turn=lambda m: None)
    with mock.patch.object(coach_chat, "_TEST_MIC_FACTORY",
                           lambda: WavMic([audio], clock, rate=rate)), \
            mock.patch.object(coach_chat, "_transcribe_audio", timed):
        lst = coach_chat.HandsFreeListener(
            on_text=heard.append, turns=tt, clock=clock,
            is_echo=lambda t: looks_like_echo(t, spoken))
        lst.done.wait(300)
    text = calls[-1]["text"] if calls else ""
    res = {"file": fix["file"], "kind": fix["kind"], "lang": fix.get("lang"),
           "expected": fix.get("text", ""), "transcript": text,
           "heard": heard, "utterances": sum(1 for _t, s in tt.transitions
                                             if s == HEARING),
           "transcriptions": len(calls),
           "stt_wall_ms": max((c["wall_ms"] for c in calls), default=None),
           "effects": effects, "soft": bool(fix.get("soft"))}
    if fix["kind"] == "question":
        res["wer"] = round(wer(fix["text"], text), 2) if text else 1.0
        res["ok"] = (heard == [text] and text != ""
                     and res["wer"] <= fix.get("max_wer", 0.34))
        if not res["ok"] and not heard:
            res["why"] = "nothing handed to the coach"
        elif not res["ok"]:
            res["why"] = f"WER {res['wer']} > {fix.get('max_wer', 0.34)}"
    elif fix["kind"] == "noise":
        res["ok"] = not heard
        if not res["ok"]:
            res["why"] = f"noise reached the coach as {heard!r}"
    elif fix["kind"] == "echo":
        res["ok"] = not heard and bool(text)
        if not text:
            res["ok"] = False
            res["why"] = "the coach's voice was not even transcribed"
        elif heard:
            res["why"] = "the echo filter let the coach's own sentence through"
    return res


def run_bench(fix_dir: str = DEFAULT_FIXTURES, only=None,
              transcribe=None) -> list[dict]:
    man = os.path.join(fix_dir, "manifest.json")
    if os.path.exists(man):
        with open(man, encoding="utf-8") as fh:
            m = json.load(fh)
        fixtures, rate = m["fixtures"], m.get("rate", RATE)
    else:
        fixtures, rate = FIXTURES, RATE
    out = []
    for fix in fixtures:
        if only and fix["file"] not in only and fix["kind"] not in only:
            continue
        p = os.path.join(fix_dir, fix["file"])
        if not os.path.exists(p):
            out.append({"file": fix["file"], "kind": fix["kind"], "ok": False,
                        "soft": bool(fix.get("soft")),
                        "why": "missing — run --make-fixtures"})
            continue
        out.append(bench_fixture(fix, read_wav(p, rate), rate, transcribe))
    return out


def format_bench(results: list[dict]) -> str:
    lines = [f"{'fixture':20} {'kind':9} {'ok':4} {'WER':5} {'stt ms':7} "
             f"transcript / note"]
    for r in results:
        ok = "ok" if r.get("ok") else ("soft" if r.get("soft") else "FAIL")
        w = "" if r.get("wer") is None else f"{r['wer']:.2f}"
        stt = "" if r.get("stt_wall_ms") is None else str(r["stt_wall_ms"])
        note = r.get("transcript") or ""
        if r.get("why"):
            note = f"{note!r} — {r['why']}" if note else r["why"]
        lines.append(f"{r['file']:20} {r['kind']:9} {ok:4} {w:5} {stt:7} {note}")
    hard = [r for r in results if not r.get("ok") and not r.get("soft")]
    soft = [r for r in results if not r.get("ok") and r.get("soft")]
    lines.append(f"{len(results) - len(hard) - len(soft)} ok, "
                 f"{len(soft)} soft failures, {len(hard)} failures")
    return "\n".join(lines)


def bench_markdown(results: list[dict]) -> str:
    rows = ["# Voice bench", "", "| fixture | kind | result | WER | "
            "transcription ms | transcript |", "|---|---|---|---|---|---|"]
    for r in results:
        ok = "✅" if r.get("ok") else ("⚠️ soft" if r.get("soft") else "❌")
        rows.append(f"| {r['file']} | {r['kind']} | {ok} | "
                    f"{'' if r.get('wer') is None else r['wer']} | "
                    f"{r.get('stt_wall_ms') or ''} | "
                    f"{(r.get('transcript') or r.get('why') or '').replace('|', '/')} |")
    return "\n".join(rows)


# ---------------------------------------------------- echo check (L3)
def echo_verdict(floor_rms: float, playback_rms: float) -> dict:
    """How much of the speaker output comes back through the mic.

    ratio < 2: the mic barely hears the speakers (headset, or far apart) →
    full duplex (--headset) is safe. Otherwise keep the gated default."""
    ratio = playback_rms / max(floor_rms, 1e-6)
    return {"floor_rms": round(floor_rms, 5), "playback_rms": round(playback_rms, 5),
            "ratio": round(ratio, 1),
            "mode": "headset" if ratio < 2.0 else "gated",
            "advice": ("the mic barely hears the speakers — --headset "
                       "(full duplex, interrupt by voice) is safe here"
                       if ratio < 2.0 else
                       "the mic hears the speakers clearly — keep the "
                       "default gated mode (or wear earbuds and use --headset)")}


def echo_check(seconds: float = 2.0, rate: int = RATE) -> dict:
    """Play a test sound while recording: is --headset safe on this setup?"""
    import numpy as np
    import sounddevice as sd

    import coach_chat
    dev = coach_chat.MIC_DEVICE
    print("measuring the room… (stay quiet 1 s)")
    floor = sd.rec(int(1.0 * rate), samplerate=rate, channels=1,
                   dtype="float32", device=dev)
    sd.wait()
    print("playing a test sound through the speakers…")
    t = np.arange(int(seconds * rate)) / rate
    sweep = (np.sin(2 * math.pi * (300 + 900 * t / seconds) * t) * 0.5
             ).astype("float32")
    rec = sd.playrec(sweep, samplerate=rate, channels=1, dtype="float32",
                     input_mapping=None if dev is None else None)
    sd.wait()
    rms = lambda a: float(np.sqrt(np.mean(np.squeeze(a) ** 2)))  # noqa: E731
    v = echo_verdict(rms(floor), rms(rec))
    print(f"room floor {v['floor_rms']}  during playback {v['playback_rms']}  "
          f"ratio {v['ratio']}× → {v['advice']}")
    return v


# --------------------------------------------------------------- selftest
def selftest():
    print("== coach_voice selftests ==")
    import numpy as np

    print("1) gate follows the state, honestly:", end=" ")
    c = FakeClock()
    fx = []
    tt = TurnTaking(clock=c, on_effect=fx.append, on_turn=lambda m: None)
    assert tt.state == OFF and not tt.gate_open
    tt.mic_ready()
    assert tt.state == LISTENING and tt.gate_open and fx == [OPEN]
    tt.speech_start()
    assert tt.state == HEARING and tt.gate_open
    c.advance(1.5)
    tt.speech_end()
    assert tt.state == THINKING and not tt.gate_open and fx[-1] == CAPTURED
    c.advance(0.4)
    tt.transcript("how deep should I squat")
    tt.observe(busy=True)
    assert tt.state == THINKING and tt.turn.answer_t == c.t
    c.advance(2.0)
    tt.observe(tts=True, busy=True)
    assert tt.state == SPEAKING and not tt.gate_open, tt.state
    assert tt.label() == "coach speaking" and "press c" in tt.hint()
    tt.observe(tts=True, busy=False)
    assert tt.state == SPEAKING, "between sentences the coach keeps the floor"
    c.advance(3.0)
    tt.observe(tts=False, busy=False)
    assert tt.state == SPEAKING and not tt.gate_open, "hangover is honest"
    assert fx[-1] == OPEN, "the open tone plays before the mic opens"
    c.advance(0.49)
    tt.observe()
    assert tt.state == SPEAKING
    c.advance(0.02)
    tt.observe()
    assert tt.state == LISTENING and tt.gate_open
    m = tt.turns[-1]
    assert m["stt_ms"] == 400 and m["ttfa_ms"] == 2400, m
    assert m["speak_ms"] == 3510 and not m["interrupted"], m
    print("ok")

    print("2) barge-in, typed interrupt, blips, rejects, full duplex:", end=" ")
    c = FakeClock()
    fx = []
    tt = TurnTaking(clock=c, on_effect=fx.append, on_turn=lambda m: None)
    tt.mic_ready()
    tt.typed(4)
    tt.observe(busy=True)
    tt.observe(tts=True)
    assert tt.state == SPEAKING
    tt.barge_in()                             # the 'c' key
    assert fx[-1] == OPEN and tt.state == SPEAKING and not tt.gate_open
    tt.observe(tts=True, busy=True)           # the dying TTS is ignored
    c.advance(0.31)
    tt.observe(tts=False, busy=False)
    assert tt.state == LISTENING and tt.turns[-1]["interrupted"]
    c.advance(0.5)                            # past the interrupt grace
    tt.typed(3)
    tt.observe(busy=True)
    tt.observe(tts=True)
    tt.typed(5)                               # new question mid-answer
    assert tt.state == THINKING and tt.turns[-1]["interrupted"]
    assert len(tt.turns) == 2
    tt.observe(tts=True, busy=True)           # still the old answer (grace)
    assert tt.state == THINKING
    c.advance(0.7)
    tt.observe(tts=False, busy=True)
    tt.observe(tts=True, busy=True)
    assert tt.state == SPEAKING and tt.turn.first_audio_t == c.t
    tt.observe(tts=False, busy=False)
    c.advance(0.6)
    tt.observe()
    assert tt.state == LISTENING and len(tt.turns) == 3
    n_fx = len(fx)
    tt.speech_start()
    tt.blip()
    assert tt.state == LISTENING and len(fx) == n_fx and len(tt.turns) == 3
    tt.speech_start()
    tt.speech_end()
    tt.transcript("")
    assert fx[-2:] == [CAPTURED, REJECT] and tt.state == LISTENING
    assert len(tt.turns) == 3, "a rejected utterance is not a turn"
    # a short cue closes the mic without an earcon; a long one earns "open"
    n_fx = len(fx)
    tt.observe(tts=True)
    c.advance(0.8)
    tt.observe(tts=False)
    c.advance(0.6)
    tt.observe()
    assert tt.state == LISTENING and len(fx) == n_fx
    tt.observe(tts=True)
    c.advance(3.0)
    tt.observe(tts=False)
    assert fx[-1] == OPEN
    # full duplex: mic open while the coach talks; voice interrupt
    c = FakeClock()
    fx = []
    fd = TurnTaking(clock=c, full_duplex=True, on_effect=fx.append,
                    on_turn=lambda m: None)
    fd.mic_ready()
    fd.typed(3)
    fd.observe(busy=True, tts=True)
    assert fd.state == SPEAKING and fd.gate_open and "just talk" in fd.hint()
    fd.speech_start()
    assert fd.state == HEARING and fd.gate_open
    fd.observe(busy=True, tts=True)
    assert fd.state == HEARING, "the athlete keeps the floor"
    c.advance(1.0)
    fd.speech_end()
    fd.blip()                                  # it was the coach's echo
    assert fd.state == SPEAKING and len(fd.turns) == 0
    fd.speech_start()
    c.advance(1.0)
    fd.speech_end()
    fd.transcript("wait what about my knees")
    assert fd.state == THINKING and fd.turns[-1]["interrupted"]
    c.advance(0.7)
    fd.observe(busy=True, tts=False)
    fd.observe(busy=True, tts=True)
    fd.observe(busy=False, tts=False)
    assert fd.state == LISTENING and len(fd.turns) == 2
    assert OPEN not in fx[1:], "no open tone when the mic never closed"
    # answered without audio (offline / muted) and the no-answer timeout
    c = FakeClock()
    tt = TurnTaking(clock=c, on_turn=lambda m: None)
    tt.mic_ready()
    tt.typed(2)
    tt.observe(busy=True)
    tt.observe(busy=False)
    assert tt.state == LISTENING and tt.turns[-1]["ttfa_ms"] is None
    tt.typed(2)
    c.advance(8.1)
    tt.observe()
    assert tt.state == LISTENING and len(tt.turns) == 2
    # no mic at all: typed questions still drive thinking/speaking → off
    tt = TurnTaking(clock=FakeClock(), on_turn=lambda m: None)
    tt.typed(2)
    tt.observe(busy=True)
    tt.observe(tts=True)
    assert tt.state == SPEAKING and not tt.gate_open
    tt.observe(tts=False, busy=False)
    assert tt.state == OFF and not tt.gate_open
    print("ok")

    print("3) thinking pulses + label while the LLM is slow:", end=" ")
    c = FakeClock()
    fx = []
    tt = TurnTaking(clock=c, on_effect=fx.append, on_turn=lambda m: None)
    tt.mic_ready()
    tt.speech_start()
    c.advance(1.0)
    tt.speech_end()
    c.advance(0.5)
    tt.transcript("plan my next leg day please")
    tt.observe(busy=True)
    for _ in range(400):                      # 12 s of waiting
        c.advance(0.03)
        tt.observe(busy=True)
    assert fx.count(THINK) == 2, fx
    assert tt.label().startswith("thinking… 1"), tt.label()
    tt.observe(tts=True)
    assert tt.state == SPEAKING and tt.turn.pulses == 2
    print("ok")

    print("4) scenario timelines replay deterministically:", end=" ")
    scen = load_scenarios()
    assert len(scen) >= 10, len(scen)
    results = [run_scenario(s) for s in scen]
    bad = [r for r in results if not r["ok"]]
    assert not bad, "\n" + format_scenarios(bad)
    assert run_scenario(scen[0]) == run_scenario(scen[0]), "not deterministic"
    print(f"ok ({len(results)} scenarios)")

    print("5) earcons: pure waves, player seam, off switch:", end=" ")
    for name in EARCONS:
        w = earcon_wave(name)
        assert w.dtype == np.float32 and 0.05 < float(np.abs(w).max()) <= 0.3
        assert 0.04 <= len(w) / RATE <= 0.3, (name, len(w))
    played = []
    e = Earcons(enabled=True, player=lambda w, r: played.append((len(w), r)))
    assert e.play(OPEN) and played[0][1] == RATE
    assert not e.play("bogus")
    assert not Earcons(enabled=False, player=lambda w, r: played.append(1)
                       ).play(OPEN) and len(played) == 1
    os.environ["COACH_EARCONS"] = "0"
    assert not Earcons().enabled
    del os.environ["COACH_EARCONS"]
    assert Earcons(player=lambda w, r: None).enabled
    print("ok")

    print("6) echo filter: the coach's own sentence vs a real question:",
          end=" ")
    said = ["Push your knees out over your toes.",
            "Keep your chest up and brace before you go down."]
    assert looks_like_echo("push your knees out over your toes", said)
    assert looks_like_echo("Knees out over your toes and keep your chest", said)
    assert looks_like_echo("your toes", said)
    assert not looks_like_echo("why do my knees cave in", said)
    assert not looks_like_echo("give me ninety seconds", said)
    assert not looks_like_echo("more", said)
    assert not looks_like_echo("", said) and not looks_like_echo("hi", [])
    print("ok")

    print("7) WAV mic seam + WER + a full listener pass without Whisper:",
          end=" ")
    import tempfile
    assert wer("How deep should I squat", "how deep should i squat") == 0.0
    assert wer("ninety seconds of rest", "90 seconds of rest") == 0.0
    assert abs(wer("a b c d", "a b x d") - 0.25) < 1e-9
    assert wer("a b", "") == 1.0
    with tempfile.TemporaryDirectory() as td:
        rng = np.random.default_rng(1)
        speech = np.sin(2 * math.pi * 220 * np.arange(int(1.2 * RATE)) / RATE)
        speech = (speech * 0.3 * (1 + 0.3 * rng.normal(0, 1, len(speech)))
                  ).astype("float32")
        p = os.path.join(td, "s.wav")
        write_wav(p, speech)
        back = read_wav(p)
        assert back.shape == speech.shape and np.abs(back - speech).max() < 1e-3
        p22 = os.path.join(td, "s22.wav")
        write_wav(p22, speech, 22050)          # resampled on the way back in
        assert abs(len(read_wav(p22)) - round(len(speech) * 16000 / 22050)) <= 1
        fix = {"file": "s.wav", "kind": "question", "lang": "en",
               "text": "How deep should I squat?", "max_wer": 0.34}
        r = bench_fixture(fix, back, transcribe=lambda a: "How deep should I squat")
        assert r["ok"] and r["wer"] == 0.0 and r["utterances"] == 1, r
        assert r["heard"] == ["How deep should I squat"]
        assert r["effects"][:2] == [OPEN, CAPTURED], r["effects"]
        r = bench_fixture({**fix, "kind": "noise"}, back,
                          transcribe=lambda a: "")
        assert r["ok"] and REJECT in r["effects"], r
        echo = {"file": "e.wav", "kind": "echo", "lang": "en",
                "text": "Push your knees out over your toes."}
        r = bench_fixture(echo, back,
                          transcribe=lambda a: "push your knees out over your toes")
        assert r["ok"] and not r["heard"], r
        quiet = np.zeros(int(2.0 * RATE), "float32")
        r = bench_fixture({**fix, "file": "q.wav"}, quiet,
                          transcribe=lambda a: "never called")
        assert not r["ok"] and r["transcriptions"] == 0
        fmt = format_bench([r])
        assert "nothing handed" in fmt and "FAIL" in fmt
        assert "| q.wav |" in bench_markdown([r])
    print("ok")

    print("8) echo verdict thresholds:", end=" ")
    assert echo_verdict(0.001, 0.0015)["mode"] == "headset"
    assert echo_verdict(0.001, 0.02)["mode"] == "gated"
    assert echo_verdict(0.0, 0.0)["ratio"] == 0.0
    print("ok")

    print("\nAll coach_voice selftests passed.")


# ------------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Voice turn-taking: state machine, earcons, tests")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--scenarios", nargs="?", const=DEFAULT_SCENARIOS,
                    metavar="FILE", help="replay the scenario timelines")
    ap.add_argument("--make-fixtures", nargs="?", const=DEFAULT_FIXTURES,
                    metavar="DIR", help="synthesize the WAV fixtures")
    ap.add_argument("--bench", nargs="?", const=DEFAULT_FIXTURES,
                    metavar="DIR", help="fixtures → VAD → Whisper: WER + ms")
    ap.add_argument("--only", help="bench: comma-separated files or kinds")
    ap.add_argument("--md", metavar="FILE", help="bench: write a markdown report")
    ap.add_argument("--json", metavar="FILE", help="bench: write the raw results")
    ap.add_argument("--echo-check", action="store_true",
                    help="play a sound and measure how much the mic hears")
    ap.add_argument("--mic", metavar="INDEX|NAME",
                    default=os.environ.get("COACH_MIC", ""),
                    help="microphone for --echo-check")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    elif args.scenarios:
        res = [run_scenario(s) for s in load_scenarios(args.scenarios)]
        print(format_scenarios(res))
        sys.exit(0 if all(r["ok"] for r in res) else 1)
    elif args.make_fixtures:
        for p in make_fixtures(args.make_fixtures):
            print(p)
    elif args.bench:
        only = [s.strip() for s in args.only.split(",")] if args.only else None
        res = run_bench(args.bench, only=only)
        print(format_bench(res))
        if args.md:
            with open(args.md, "w", encoding="utf-8") as fh:
                fh.write(bench_markdown(res) + "\n")
        if args.json:
            with open(args.json, "w", encoding="utf-8") as fh:
                json.dump(res, fh, ensure_ascii=False, indent=1)
        sys.exit(0 if all(r.get("ok") or r.get("soft") for r in res) else 1)
    elif args.echo_check:
        if args.mic:
            import coach_chat
            coach_chat.set_mic(args.mic)
        echo_check()
    else:
        ap.print_help()
