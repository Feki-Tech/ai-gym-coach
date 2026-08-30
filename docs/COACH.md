# Talk to Your Coach — LLM Chat Guide

The app includes a conversational AI coach (`coach_chat.py`) you can talk to
— by typing or with your **microphone** — while it watches your workout
through the camera and answers through your **speakers**.

It knows your training history (`workout_log.json`) and, during a live
session, the current exercise, rep count, scores, faults and fatigue — so
you can ask things like:

- *"Why are my squat scores dropping?"*
- *"What should I train tomorrow?"*
- *"My lower back hurts on deadlifts — what am I doing wrong?"*
- *"كيف أحسن تمرين العقلة؟"* — it answers in whatever language you use.

The LLM runs **locally in Docker** ([Ollama](https://ollama.com)) — private,
free, no API key. Any OpenAI-compatible API works too (see [Config](#config)).

---

## 1. Start the LLM (Docker, any OS)

```bash
docker compose up -d ollama
docker compose exec ollama ollama pull llama3.2:3b   # once, ~2 GB
```

The server persists models in a Docker volume and listens on
`localhost:11434` — both containers *and* natively-running apps can use it.

> **Model quality ladder** — pick per your RAM/patience:
> `qwen2.5:0.5b` (smoke-test only, weak knowledge) → `llama3.2:3b` /
> `qwen2.5:3b` (default; good coaching + multilingual) → `llama3.1:8b` /
> `qwen2.5:7b` (strongest exercise-science answers, needs ~8 GB RAM).
> Set `COACH_MODEL=…` (compose) or `--model …` (CLI). The app injects an
> evidence-based coaching knowledge base into every model, so even small
> models explain faults (e.g. knee valgus → weak hip abductors) correctly.

## 2. Chat with the coach

### Fully in Docker (any OS — text chat)

```bash
docker compose run --rm coach
```

Terminal chat with your workout history mounted from `./data/`. Containers
can't reach your camera/mic/speakers on Windows/macOS, which is why this
service is text-only — for the full experience, see the next section.

### On your machine (camera + mic + speakers) — recommended

The pose app runs natively (full hardware access) and talks to the LLM in
the Docker container:

```bash
pip install -r requirements.txt

# live workout + chat: type questions in the terminal while training
python pose_coach.py --exercise auto --coach

# hands-free voice chat during the workout: install the voice extras once
# and just TALK — no key needed (the HUD shows "mic: listening")
pip install -r requirements-voice.txt        # once (local speech-to-text)

# or a standalone chat session (no camera):
python coach_chat.py --voice                 # spoken replies
python coach_chat.py --voice --hands-free    # open-mic conversation
python coach_chat.py --voice --listen        # empty line = talk with mic
python coach_chat.py --once "Plan my next workout"
```

Voice input uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
locally (first use downloads a ~150 MB model) and auto-detects the language
you speak. Spoken replies use the same TTS as the workout cues.

### Everything in Docker (Linux only)

Linux can pass the webcam, ALSA audio (mic + speakers) and X11 into the
container:

```bash
xhost +local:docker
EXERCISE=auto docker compose run --rm coach-live
```

## 3. During a workout

With `--coach` active:

| Action | How |
|---|---|
| Ask by voice | **just speak** — the open mic segments your speech automatically (hands-free) |
| Ask by typing | type in the terminal where you launched the app, Enter |
| Hear answers | replies stream in as they're generated and are spoken sentence-by-sentence via TTS |
| Interrupt | press **`c`** in the video window — the coach shuts up and the mic opens instantly; or just type |

Answers **stream**: text appears word-by-word and the voice starts with the
first sentence instead of waiting for the full reply. Asking something new
mid-answer cancels the old reply (the partial answer stays in the coach's
memory, so follow-ups remain coherent).

### The coach speaks up on its own

You don't have to ask. Three moments trigger a short spoken note (max two
sentences) without any question:

- **Session start** — the coach connects to last time: *"Last squat session
  averaged 84 but your knees kept caving — let's fix that today."* Skipped
  silently on your very first session.
- **Set done** (rep goal reached, or every set of a guided `--program`) —
  the trend *inside* the set (scores fading? tempo rushing?) or its dominant
  fault, plus one cue for the next set. A clean set gets celebrated.
- **Session done** — a one-sentence wrap-up with the day's headline number.

These notes are **disposable by design**: if you're mid-conversation with
the coach (or a question is queued), the event is dropped rather than
interrupting you. They never feed the athlete profile — only things *you*
say do.

### The coach can look up your history

The prompt always carries your recent sessions plus a per-exercise overview
of the *whole* log (sessions, total reps, best/recent average with an
improving/steady/declining tag, top fault). For anything more specific the
coach fetches real data instead of guessing: asking *"how did my squats go
last month?"* makes it emit `ACTION: {"do": "history_query", "exercise":
"squat", "days": 31}`; the app answers with the matching sessions and
totals, and only then does the coach state numbers. Works in workout mode
and in plain `python coach_chat.py` alike.

### Hands-free listening

The coach panel (bottom-right of the window) shows the mic state and a
live level meter next to your last question and the streaming answer:

| status | Meaning |
|---|---|
| `listening` | open mic — just talk |
| `hearing you...` | speech detected, recording your sentence |
| `thinking...` | transcribing locally (Whisper) |
| `answering...` | the coach is replying — **it cannot hear you now** |
| `press c to talk` | voice extras not installed → push-to-talk only |
| `off` | mic unavailable (see Troubleshooting) |

Which microphone: the OS default, unless you pass `--mic 3` or
`--mic "Camo"` (index or part of the name — `--list-devices` shows them; the
`COACH_MIC` env var sets a permanent default). Works for both push-to-talk
and hands-free, in `pose_coach.py --coach` and `coach_chat.py`.

There is no echo cancellation: while the coach talks through your speakers
the mic is gated so it never hears its own voice. To barge in, press `c`
(mutes the coach + reopens the mic immediately) or type your question. A
built-in voice-activity detector adapts to room noise, ignores coughs and
clanking plates, and a filter drops non-speech transcriptions.

The coach sees live session data — current exercise, phase, rep count, the
last rep's physics (score, tempo, range of motion, speed, faults), your
**live joint angles**, the **environment** (image brightness, how much of
your body is visible/in frame, camera view, processing fps) and its own
config (rep goal, rest timer, tempo target) — plus your history, and
tailors its answers ("your knees caved in 3 times this set — try a wider
stance…", "the image is dark and only 60% of your body is visible — step
back and add light").

### The coach can drive the app

During a workout (`--coach`) the coach doesn't just talk — ask it and it
**acts**, by emitting validated commands the app executes:

| Say something like | What happens |
|---|---|
| "switch me to squats" | app changes exercise (counter, form rules, reference rep reset) |
| "let's do 10 reps" | rep goal set — the rep counter shows `3 /10` with a goal ring, coach announces when you hit it |
| "give me 90 seconds rest" | full-screen REST countdown with what's next, "back to work" when it ends |
| "make me lower in 3 seconds" | tempo target enforced — too-fast reps get a voice cue |
| "stop correcting me" / "cues on" | mutes/unmutes the spoken form corrections |
| "re-detect my exercise" | back to auto-detect mode |
| "plan me a 15-minute leg workout and start it" | coach designs a **guided program** from your profile/history and runs it |
| "stop the program" | back to free training |

Under the hood the model appends machine-readable `ACTION: {json}` lines
to its reply; the app validates them (unknown actions and out-of-range
values are ignored), applies them to the live session and speaks a short
confirmation. Action lines are never read aloud.

### Guided workout programs

A program is a list of blocks — `squat 3x10 rest 90, pushup 2x15 rest
45, plank 2x40s rest 30` (`40s` = timed hold). Once one is running the
app becomes the trainer: it counts every set, announces "Set 1 of 3
done — rest 90 seconds", runs the countdown, switches exercises between
blocks and celebrates when the session is complete. The exercise card
shows `PROGRAM block 1/3 · squat · set 2/3 · 10 reps` with a progress bar
per block, the rest screen says what's next, and the coach always knows
where you are if you ask.

You don't need the LLM for this — start one directly:

```bash
python pose_coach.py --exercise squat --program "squat 3x10 rest 90, pushup 2x15 rest 45, plank 2x40s"
```

### Making it feel fast

Startup **warms the LLM up** in the background, so the first answer skips
Ollama's cold model load (which can take 30+ s on CPU). Speech is
transcribed with a greedy decode and answers stream sentence-by-sentence
into the TTS. Tuning knobs:

| Variable | Default | Effect |
|---|---|---|
| `COACH_WHISPER_MODEL` | `base` | `tiny` ≈ 2× faster transcription, slightly less accurate |
| `COACH_MAX_TOKENS` | `300` | hard cap on reply length — lower = snappier answers |

A smaller/faster LLM also helps: `COACH_LLM_MODEL=llama3.2:1b`. Keep the
Ollama container running between sessions so the model stays warm — the
compose file now tells Ollama to hold the model in RAM for 2 h after the
last request (`OLLAMA_KEEP_ALIVE`), so a long rest no longer pays the cold
load again. `python coach_ops.py --doctor` tells you whether the model is
warm right now.

### Safety guardrail (deterministic)

The persona tells the model to say *stop and see a professional* for
sharp/radiating pain, numbness, dizziness, chest pain or breathing
trouble. Small models don't always listen, so the app enforces it: when
your message matches one of those symptoms (in English, French, Spanish,
Arabic, German or Chinese) a `[SAFETY NOTE]` is attached to the turn before
the model answers, and the reply is checked afterwards. This is a
guardrail, not medical advice — the coach is still not a doctor.

Two more deterministic helpers work the same way: when you ask for a
plan or program and your profile lists injuries, they are restated on
that turn so the plan respects them; and the live-session block is
prefixed with plain-language hints ("image is DARK", "only 60 % of the
body in frame", "rep velocity down 25 %") so a small model doesn't have to
decode the raw numbers.

### Is the coach still behaving? (evals + trace)

`python coach_eval.py` runs 34 scripted scenarios — safety, language, prompt injection,
number grounding, app actions, calendar discipline, proactive events, style
— against your local model and prints a per-category pass rate;
`COACH_TRACE=coach_trace.jsonl` records latency and guardrail flags for a
real session. Both are local files. Details in [LLMOPS.md](LLMOPS.md).

## 4. The coach remembers you (athlete profile)

Durable facts you mention in chat — age, weight, goals, injuries,
equipment, schedule, diet, preferences — are **extracted automatically**
after each exchange and saved to a local SQLite file
(`coach_profile.db`, git-ignored). Next session, the coach already knows:

> *"Given your left-knee history, let's keep squats above parallel today."*

| Command | Effect |
|---|---|
| `/profile` | show everything the coach remembers |
| `/remember <key> <value…>` | save a fact by hand (`/remember weight 82 kg`) |
| `/remember <category> <key> <value…>` | with category: `identity body goals injuries equipment schedule nutrition preferences` |
| `/forget <key>` | erase one fact |
| `/forget all` | wipe the whole profile |

Commands work in the workout terminal (`--coach`) and standalone chat.
`--no-profile` disables memory entirely; `--profile-file PATH` (or env
`COACH_PROFILE_DB`) relocates it — handy for multiple athletes sharing
a machine: one file each.

**Privacy**: the profile never leaves your machine. With the default
Ollama backend, even the fact-extraction step runs locally. Inspect it
anytime (`python coach_profile.py --show`) — it's a plain SQLite file
you can delete whenever you like.

## 5. Google Calendar — the coach plans training with you

Connect your Google account once and the coach can **read your week and
book training sessions** right from the conversation:

> **You:** "When can I train this week?"
> **Coach:** *(checks your calendar)* "Tuesday evening and Friday morning
> are free. Tuesday 18:00 for legs?"
> **You:** "Yes, book it."
> **Coach:** ⚙️ Booked Leg day: Tuesday 14 Jul 18:00–19:00.

### One-time setup (~3 minutes, free)

1. Go to [console.cloud.google.com](https://console.cloud.google.com) →
   create a project → **APIs & Services → Library** → enable the
   **Google Calendar API**.
2. **OAuth consent screen** → External → fill the two required fields →
   add your own Gmail address as a **test user**.
3. **Credentials → Create credentials → OAuth client ID → Desktop app**
   → **Download JSON** → save it as `google_credentials.json` next to
   `pose_coach.py`.
4. Connect (opens your browser once, sign in, allow):

   ```bash
   python coach_calendar.py --connect
   ```

Done. Every later start of `--coach` or `coach_chat.py` prints
"📅 Google Calendar connected" automatically.

| You can say | What happens |
|---|---|
| "what's my week look like?" | coach reads the next days and summarizes |
| "find me a slot for legs" | coach proposes free times that fit your schedule |
| "book Tuesday 18:00, one hour" | event lands in your Google Calendar |
| `/calendar` | prints your 7-day agenda directly |

Also: `python coach_calendar.py --agenda 7` prints the agenda without
starting the app.

**Privacy & safety**: the only permission requested is *calendar events*
(scope `calendar.events`) — the coach cannot read mail, contacts or
files. Tokens stay in `google_token.json` on your machine (git-ignored).
With the default Ollama backend your agenda is only ever shown to the
local model. The coach books events only after you agree to a specific
time; delete `google_token.json` to disconnect at any moment.

## Config

Environment variables (or CLI flags on `coach_chat.py`):

| Variable | Default | Meaning |
|---|---|---|
| `COACH_LLM_BASE_URL` | `http://localhost:11434/v1` | any OpenAI-compatible endpoint |
| `COACH_LLM_MODEL` | `llama3.2:3b` | model name |
| `COACH_LLM_API_KEY` | `ollama` | API key (only needed for hosted APIs) |
| `COACH_LOG` | `workout_log.json` | history used for context |
| `COACH_PROFILE_DB` | `coach_profile.db` | athlete profile the coach remembers you with |
| `COACH_WHISPER_MODEL` | `base` | speech-recognition model size (`tiny`/`base`/`small`) |
| `COACH_MIC` | OS default | microphone for voice input: sounddevice index or part of its name (`--mic`; `--list-devices` shows them) |
| `COACH_CAMERA` | `0` | camera for `pose_coach.py`: index, `/dev/videoN` or stream URL (`--camera`) |
| `COACH_MAX_TOKENS` | `300` | max reply length in tokens |
| `COACH_TEMPERATURE` | `0.5` | sampling temperature — lower = steadier, more literal (Ollama's own default is 0.8) |
| `COACH_SEED` | unset | integer seed for repeatable replies (the eval harness sets one) |
| `COACH_TRACE` | unset | path of a local JSONL trace of every LLM call: latency, guardrail flags, actions — no message text unless `COACH_TRACE_TEXT=1`. `python coach_ops.py --report FILE` summarizes it. See [LLMOPS.md](LLMOPS.md) |
| `OLLAMA_KEEP_ALIVE` | `2h` (compose) | how long the Ollama container keeps the model in RAM after the last request; the stock default of 5 min meant a long rest paid the cold load again |
| `GOOGLE_CREDENTIALS_FILE` | `google_credentials.json` | OAuth client JSON from Google Cloud (§5) |
| `GOOGLE_TOKEN_FILE` | `google_token.json` | where the calendar tokens are stored |

Examples: point it at **OpenAI** (`COACH_LLM_BASE_URL=https://api.openai.com/v1`,
`COACH_LLM_MODEL=gpt-4o-mini`, `COACH_LLM_API_KEY=sk-…`) or any other
compatible server (LM Studio, llama.cpp, vLLM, …). Note that hosted APIs
receive your questions and workout summaries — the Ollama default keeps
everything on your machine.

## Troubleshooting

| Problem | Fix |
|---|---|
| "Cannot reach the LLM backend" | `docker compose up -d ollama`, then check `docker compose logs ollama` |
| "Model not found" | `docker compose exec ollama ollama pull llama3.2:3b` |
| First answer is slow | the model loads into RAM on first request; subsequent replies are fast |
| Push-to-talk says extras missing | `pip install -r requirements-voice.txt` (host Python, not Docker) |
| Mic not picked up | check the OS default input device; `python -m sounddevice` lists devices |
| `mic: off` / `PaErrorCode -9999` on Windows | **Microsoft Store Python is blocked from the microphone** on many machines. Install standard Python (`winget install Python.Python.3.12` or python.org), `py -3.12 -m pip install -r requirements.txt -r requirements-voice.txt`, run with `py -3.12 pose_coach.py …`. Also check Settings → Privacy & security → Microphone |
| Mic is "listening" but never hears you | your OS **input volume** is probably very low (we've seen 8%!) — the app now warns at startup. Raise it: Settings → Sound → Input → Volume. The HUD `mic:` line shows a live level meter: bars should jump when you speak |
| Coach hears itself | it shouldn't — the mic is gated during TTS. If your speakers are very loud and the room echoes, lower the volume slightly |
| Coach replies not spoken | TTS uses pyttsx3 — see voice notes in [WEBCAM.md](WEBCAM.md) |
