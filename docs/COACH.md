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

> Any Ollama model works: `qwen2.5:3b` (great multilingual), `llama3.1:8b`
> (smarter, needs ~8 GB RAM), `gemma2:2b` (fastest). Set
> `COACH_MODEL=qwen2.5:3b` (compose) or `--model qwen2.5:3b` (CLI).

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

# push-to-talk with your MIC during the workout: press 'c' in the video
# window, speak, and the coach answers through your speakers
pip install -r requirements-voice.txt        # once (local speech-to-text)

# or a standalone chat session (no camera):
python coach_chat.py --voice                 # spoken replies
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
| Ask by typing | type in the terminal where you launched the app, Enter |
| Ask by voice | press **`c`** in the video window, speak (~6 s) |
| Hear answers | replies are spoken via TTS and printed in the terminal |

The coach sees live session data — current exercise, phase, rep count, last
score, fault counts, velocity loss — plus your history, and tailors its
answers ("your knees caved in 3 times this set — try a wider stance…").

## Config

Environment variables (or CLI flags on `coach_chat.py`):

| Variable | Default | Meaning |
|---|---|---|
| `COACH_LLM_BASE_URL` | `http://localhost:11434/v1` | any OpenAI-compatible endpoint |
| `COACH_LLM_MODEL` | `llama3.2:3b` | model name |
| `COACH_LLM_API_KEY` | `ollama` | API key (only needed for hosted APIs) |
| `COACH_LOG` | `workout_log.json` | history used for context |

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
| Coach replies not spoken | TTS uses pyttsx3 — see voice notes in [WEBCAM.md](WEBCAM.md) |
