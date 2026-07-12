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

### Hands-free listening

The HUD's third line shows the mic state:

| `mic:` | Meaning |
|---|---|
| `listening` | open mic — just talk |
| `hearing you...` | speech detected, recording your sentence |
| `thinking...` | transcribing locally (Whisper) |
| `answering...` | the coach is replying — **it cannot hear you now** |
| `press c to talk` | voice extras not installed → push-to-talk only |
| `off` | mic unavailable (see Troubleshooting) |

There is no echo cancellation: while the coach talks through your speakers
the mic is gated so it never hears its own voice. To barge in, press `c`
(mutes the coach + reopens the mic immediately) or type your question. A
built-in voice-activity detector adapts to room noise, ignores coughs and
clanking plates, and a filter drops non-speech transcriptions.

The coach sees live session data — current exercise, phase, rep count, last
score, fault counts, velocity loss — plus your history, and tailors its
answers ("your knees caved in 3 times this set — try a wider stance…").

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

## Config

Environment variables (or CLI flags on `coach_chat.py`):

| Variable | Default | Meaning |
|---|---|---|
| `COACH_LLM_BASE_URL` | `http://localhost:11434/v1` | any OpenAI-compatible endpoint |
| `COACH_LLM_MODEL` | `llama3.2:3b` | model name |
| `COACH_LLM_API_KEY` | `ollama` | API key (only needed for hosted APIs) |
| `COACH_LOG` | `workout_log.json` | history used for context |
| `COACH_PROFILE_DB` | `coach_profile.db` | athlete profile the coach remembers you with |

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
| Coach hears itself | it shouldn't — the mic is gated during TTS. If your speakers are very loud and the room echoes, lower the volume slightly |
| Coach replies not spoken | TTS uses pyttsx3 — see voice notes in [WEBCAM.md](WEBCAM.md) |
