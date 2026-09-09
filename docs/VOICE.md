# Talking to the coach — turn-taking, and how it is tested

The voice side of the coach is a conversation, and a conversation has
rules about whose turn it is. This page is the contract for those rules
(`coach_voice.py`), what the athlete sees and hears, the two ways to talk
over the coach, and the three levels of tests that keep it honest — none
of which needs a microphone.

## 1. The five states

Everything the athlete needs to know is one of five states, shown as a
coloured pill on the desktop HUD (bottom-right coach panel) and on the
phone (HUD pill and chat header), always with a one-line hint:

| state | colour | meaning | hint |
|---|---|---|---|
| `listening` | green | the mic is open — talk | *just talk* |
| `hearing` | cyan | your sentence is being captured (level meter moves) | *finish your sentence* |
| `thinking` | amber | captured; transcribing / the LLM is working, no audio yet — the label counts seconds after 3 s | *hang on* |
| `speaking` | purple | the coach has the floor: answer audio playing or streaming | *press c to interrupt* / *just talk to interrupt* |
| `off` | grey | no usable microphone | *type your question…* |

The mic gate follows the state: **open in `listening` and `hearing`,
closed otherwise.** On speakers there is no acoustic echo cancellation,
so the mic must be closed while the coach talks or it would answer
itself. The HUD never says "listening" with the mic closed — that was
the bug this design replaces — and the level meter is dark while the
gate is closed, so it never shows the coach's own voice.

### Earcons — what you hear

Nobody looks at the screen mid-set, so the state is audible too
(`COACH_EARCONS=0` turns them off; on the phone: *Mic sounds*):

| sound | when |
|---|---|
| rising two-note | the mic is yours: at start, after an answer, after a long unprompted cue, right after you press `c` |
| short tick | your sentence was captured — the coach is on it |
| soft pulse every 5 s | still computing the answer (slow CPU / big model) |
| falling two-note | what was heard was not usable (no words) — say it again |

Short form cues ("go deeper") close the mic for under two seconds and play
no tone: that would be noise.

## 2. Talking over the coach

**Speakers (default, gated).** The mic is closed while the coach talks.
`c` cuts it off and gives you the mic (rising tone, 0.3 s). To keep the
closed window short, the coach reads **two sentences per answer** and
then says *"Say more for the rest."* — say **"more"** (or *encore*, *mehr*,
*المزيد*, *继续* …) and the next two are read without another LLM call.
The full text is always on screen. `--spoken-sentences N` /
`COACH_SPOKEN_SENTENCES` changes the cap (0 = read everything).

**Headset / earbuds (`--headset`, full duplex).** No echo, so the mic
stays open in every state: talk over the coach and it stops and answers
the new question (the trace logs the old turn as interrupted). Whatever
still leaks back from the coach's own voice is dropped by the **echo
filter**: a transcript made mostly of words from a sentence the coach
just said is never a question. `python coach_voice.py --echo-check`
plays a sweep while recording and tells you whether your speakers/mic
setup is quiet enough for `--headset` (the phone uses Apple's echo
cancellation instead — *Interrupt by voice* in Coach settings, on by
default).

## 3. What gets measured

Every athlete turn writes one `turn` line to the local trace
(`COACH_TRACE=…`, see [LLMOPS.md](LLMOPS.md)) — numbers only, no text:

| field | meaning |
|---|---|
| `stt_ms` | end of your speech → transcript |
| `ttfa_ms` | end of your speech → the coach's first word (what you actually wait) |
| `llm_ms` | transcript → first word |
| `speak_ms` | how long the answer audio lasted |
| `interrupted` | cut off by `c` or by a new question |
| `source` | `voice` / `typed`, `full_duplex` |

`python coach_ops.py --report TRACE` prints p50/p95 of heard→first word,
transcription and answer lengths, and the interrupted count.

## 4. How it is tested — three levels, no microphone needed

**Level 1 — the state machine on a fake clock** (`coach_voice.py
--selftest`, every push in CI). `TurnTaking` takes a clock and emits
effects, so `data/voice_scenarios.jsonl` replays whole conversations in
30 ms ticks: a spoken question on speakers, a 12 s LLM with pulses, a
clank, junk audio, the `c` key mid-answer, a typed question mid-answer, a
short cue, a long unprompted brief, headset interruption by voice, the
coach's echo being dropped, an answer without audio, a question that never
gets answered, no mic at all, a cue during transcription, the mic dying
mid-answer. Each scenario asserts the exact state sequence, the earcons,
turn metrics to ±100 ms, and the state at chosen instants; invariants are
checked at every tick (never "listening" with the mic closed; never mic
open while the coach talks on speakers; never open while off).
`python coach_voice.py --scenarios` prints the table. Add a scenario =
add a line.

**Level 2 — real audio through the real listener** (`coach_voice.py
--bench`, nightly in `voice-bench.yml`). `_TEST_MIC_FACTORY` in
`coach_chat.py` mirrors the camera's test seam: `WavMic` plays WAV files
block by block into the unchanged `HandsFreeListener` (VAD → Whisper →
echo filter → turn machine), advancing the fake clock by audio time, so
utterance boundaries and hangovers are identical on every machine while
transcription wall time is measured separately. `--make-fixtures`
synthesizes the fixtures with `espeak-ng` (spoken questions in English,
French and Arabic; gym rumble with clanks; a rolling plate long enough to
trigger the VAD; near-silence; the coach's own sentence) — nothing is
committed, `data/voice_fixtures/manifest.json` lists them with expected
text and a word-error-rate threshold. Questions must transcribe within
their WER, noise must never reach the coach, the coach's voice must be
recognised as echo. The synthetic French and Arabic voices are marked
*soft* (reported, not gating): Whisper does not understand espeak in
those languages, and real recordings are the upgrade path — put your own
16 kHz WAVs next to the manifest, add an entry, run `--bench DIR`.

**Level 3 — the real machine** (Windows box with a mic and speakers).
Run the app with `COACH_TRACE=coach_trace.jsonl`, talk, then
`python coach_ops.py --report coach_trace.jsonl` for the numbers above;
`python coach_voice.py --echo-check` calibrates speakers vs. headset;
`python coach_hud.py --preview DIR` renders every mic state as a PNG to
check the panel without a camera.

**The coordinator** (`coach_chat.py --selftest` 23) drives a real
`BackgroundChat` with a fake LLM and a fake speaker: the status is polled
while the coach talks and must never read "listening"; the spoken cap
holds back sentences and "more" reads them without a second LLM call; a
typed question mid-answer stops the speech and logs the turn as
interrupted; every turn lands in the trace with its timings and no text.

## 5. Design notes

- One source of truth. The HUD, the earcons, the mic gate and the metrics
  are all derived from `TurnTaking`; the listener and the coordinator
  only feed it events (`speech_start/end`, `blip`, `transcript`, `typed`,
  `answer_started`, `barge_in`, `mic_ready/lost`) and observations
  (`observe(tts=, busy=)`).
- The open tone plays *before* the gate opens (0.5 s echo hangover after
  the coach's voice, 0.3 s after `c`), and the listener ignores the mic
  for the length of any earcon, so the coach never hears its own beeps.
- A turn is only recorded once something was transcribed or answered;
  clanks, echoes and captures cut short by a cue leave no trace rows.
- After an interrupt the old answer's dying voice is ignored until it is
  seen down or the new answer starts (`answer_started`), so a new question
  never flips the state back to "speaking" for the wrong reason.
- Not done yet: a neural VAD (the energy VAD adapts to room noise and
  is exercised by the fixtures; Silero would be the next step for loud
  gyms), and streaming partial transcripts on the desktop (Whisper is
  utterance-based; the phone already shows partials).
