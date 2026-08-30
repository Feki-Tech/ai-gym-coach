# AI Gym Coach 🏋️

[![CI](https://github.com/Feki-Tech/ai-gym-coach/actions/workflows/ci.yml/badge.svg)](https://github.com/Feki-Tech/ai-gym-coach/actions/workflows/ci.yml)

Real-time AI fitness coach using computer vision: it watches your exercise
through a webcam, tracks your skeleton, counts reps, measures tempo, checks
your form, and coaches you with on-screen + voice feedback.

## Features

- **Pose estimation** — MediaPipe Pose Landmarker (BlazePose, 33 keypoints), real-time on CPU
- **9 exercises** — squat, push-up, bench press, deadlift, lunge, shoulder press, bicep curl, pull-up, plank (timed hold)
- **Rep counting & phases** — finite-state machine on joint-angle signals (descent → bottom → ascent → lockout), tempo per phase
- **Form evaluation** — biomechanical rules per exercise: back rounding, knee valgus, insufficient depth, elbow flare/swing, uneven pressing, chin-over-bar, body sag…
- **Smoothing** — One Euro filter per keypoint with visibility gating
- **A HUD that shows what the coach sees** — the video window is the app: the skeleton lights up the body part at fault, a live **range-of-motion gauge** shows your joint angle against the exercise's own *rep starts / full depth / lockout* thresholds (so you see *why* a rep did or didn't count), rep counter with goal ring, phase (lowering / bottom / lifting), last-rep score, tempo, golden-rep similarity, fatigue, heart-rate zone, program progress, rest countdown, the coach's answers and mic state — plus a **framing guide** before anyone is tracked, an end-of-set **summary card** with what to fix, and **keys** to switch exercise (1-9), auto-detect (a), rest (r), mute (v), mirror (m), help (h). Webcams are mirrored like a gym mirror (`--no-mirror` to disable). Real TrueType text on every OS; no new dependencies
- **Pick your devices** — `--list-devices` shows every camera (with resolution), microphone and speaker the machine has (`python coach_devices.py --ble` also scans for heart-rate straps); `--camera 1` / `--camera /dev/video2` / `--camera rtsp://…` chooses the video source and `--mic 3` or `--mic "Camo"` the microphone the coach listens on — also via `COACH_CAMERA` / `COACH_MIC` env vars and `CAMERA_DEV=/dev/video2` for the compose services
- **Voice coaching** — prioritized, rate-limited cues via TTS ("Straighten your back", "Slow down", "Great form!")
- **Auto exercise detection** — `--exercise auto` recognizes the movement from the skeleton (8 of 9 exercises; bench press needs manual selection)
- **ML exercise classifier** — `--train-classifier` trains a small neural network (numpy MLP on windowed skeleton features, no extra deps) that replaces the rule-based detector; bootstrapped from synthetic motion data, improvable with your own recordings via `--collect`. Every model ships with a version manifest, and retrains pass a **champion/challenger gate** on a fixed eval harness — a worse model never silently replaces the one you have (`--no-gate` opts out). `--export-model` writes the promoted model as portable JSON, and the **same classifier runs on iOS and Android** (`MLDetector` in CoachCore and the Kotlin core, inference parity pinned by the cross-engine fixtures). The harness also judges on a committed set of real windows (`data/eval_windows.jsonl`, grown via `--export-eval`) — recording protocol in [docs/DATA_COLLECTION.md](docs/DATA_COLLECTION.md)
- **Fatigue monitor** — warns once when concentric rep velocity drops >20% vs your first reps
- **Golden-rep comparison** — record your best rep once (`--record-reference`), then every future rep gets a 0-100 DTW similarity score against it (tempo-independent shape match)
- **Workout log & progress** — per-rep scores, tempo, velocity, and fault statistics in `workout_log.json`; `--stats` prints a progress dashboard with score trends
- **Web progress dashboard** — `python coach_dashboard.py` opens a local page with charts: weekly volume, form-score and rep trends per exercise, PRs, fault breakdowns, streaks (offline, no dependencies); `--demo` previews it with synthetic sample data
- **Talk to your coach** — a local LLM (Ollama in Docker) answers questions by text or **voice** during the workout, with your live session + history as context; replies **stream** in real time, you can **interrupt** anytime (barge-in), and with the voice extras it's fully **hands-free**: just speak, a VAD segments your sentence, Whisper transcribes it locally. The coach also **speaks up on its own** — greets you with last session's key point, debriefs every finished set (score trend, dominant fault, one cue) and wraps up the session — and can **query your full training history** on demand instead of guessing numbers — see [docs/COACH.md](docs/COACH.md)
- **The coach knows exercises, not just the nine it can see** — a local **RAG** layer (`coach_knowledge.py`, dependency-free BM25, optional Ollama embeddings for hybrid ranking) retrieves the relevant coaching notes (`data/knowledge/*.md`: form faults, programming, recovery, nutrition, technique, alternatives) and entries from an **open exercise catalogue** (`data/exercises.json`: 876 exercises with muscles, equipment, level, mechanic and step-by-step instructions, from [free-exercise-db](https://github.com/yuhonas/free-exercise-db), public domain; `--fetch-wger` swaps in [wger](https://wger.de)'s CC-BY-SA catalogue with 20+ languages) into every reply. The coach looks things up instead of guessing (`exercise_lookup`, `plate_calc` actions; `/exercise <name|muscle>`, `/plates 100` commands) — the behaviour contract is in [docs/MODEL_REQUIREMENTS.md](docs/MODEL_REQUIREMENTS.md)
- **MCP server** — `python coach_mcp.py` exposes the athlete's data to any MCP client (Claude Desktop, Claude Code, Cursor…): training history, last session rep by rep, profile read/write, exercise catalogue + coaching notes, plate calculator, the **live session** and a command queue that drives the running app — so a bigger model can coach with the same grounded numbers, all local (`claude mcp add gym-coach -- uv run --directory … python coach_mcp.py`)
- **Load, volume, e1RM and PRs** — `--load 60` (or tell the coach "I'm on 60 kilos") logs the weight per rep; sessions get volume and an Epley **estimated 1RM**, the app announces **personal records** live (most reps ever) and at the end (e1RM, hold), the dashboard charts e1RM/volume per exercise and a **muscle recovery** view (sets per muscle group this week, recovering / ready / detraining — muscles come from the catalogue), and `coach_dashboard.py --export-csv` / `--import-csv` speak the Strong/Hevy CSV format so history moves in and out
- **Sign in with Google or Microsoft** (optional) — `python coach_auth.py --login google|microsoft`, or `/login` in the chat: the verified name/e-mail personalise the coach, and `coach_dashboard.py --auth` puts the progress page behind a login with an e-mail allow-list for Docker or the Azure demo. OAuth 2.0 for native apps (RFC 8252 loopback), PKCE, OpenID Connect with full ID-token verification — standard library only; the iOS app adds Sign in with Apple — see [docs/AUTH.md](docs/AUTH.md)
- **The coach remembers you** — a local athlete profile (SQLite, never uploaded) auto-learns your goals, injuries, equipment and preferences from conversation and personalises future coaching; `/profile` `/remember` `/forget` to inspect or edit — see [docs/COACH.md](docs/COACH.md)
- **The coach drives the app** — ask it to switch exercise, set a rep goal, start a rest timer, enforce tempo or mute cues, and it happens live; it also sees your joint angles and environment (lighting, framing, visibility) for smarter advice — see [docs/COACH.md](docs/COACH.md)
- **Google Calendar** — connect once and the coach checks your week and books training sessions with you ("when can I train?" → "Tuesday 18:00 is free — book it?"); only calendar-events access, tokens stay local — see [docs/COACH.md §5](docs/COACH.md)
- **Guided workout programs** — `--program "squat 3x10 rest 90, pushup 2x15 rest 45, plank 2x40s"` (or just tell the coach *"plan me a leg workout and start it"*): the app counts sets, runs the rest countdowns, switches exercises and announces every step
- **Apple Health & Fitness (iOS app)** — every finished set is saved to Health as a Strength Training workout (visible in the Fitness app, with links straight into Health and Fitness from the summary), live heart rate from Apple Watch or a strap shows as a zone pill on the HUD and lands in the log as `avg_hr`/`peak_hr`, and the coach reads resting HR, HRV, VO₂ max, weight, height, age, sleep, steps, active energy, exercise minutes and this week's workouts — on-device only — see [docs/IOS.md §6](docs/IOS.md)
- **Sensor fusion (PoC)** — `--sensors ble` pairs any standard BLE heart-rate strap (or `sim`/`udp:PORT`/`replay:FILE` without hardware): HR zones on the HUD, rest advice based on your actual heart-rate recovery instead of a countdown, per-session `avg_hr`/`peak_hr` in the log, and live physiology in the LLM coach's context — architecture and roadmap (IMU velocity, occlusion-proof reps) in [docs/SENSORS.md](docs/SENSORS.md)
- **Coach LLMOps** — the coach is versioned, traced and tested like the classifier: a deterministic safety guardrail for red-flag symptoms (six languages), an opt-in local trace of latency + guardrail flags (`COACH_TRACE=…`, no text by default), `python coach_eval.py` running 34 behaviour scenarios (safety, language, number grounding, app actions, calendar, events, prompt injection) against your local model with a baseline gate, and `python coach_ops.py --doctor` — all local files, nothing uploaded — see [docs/LLMOPS.md](docs/LLMOPS.md). Threat model + hardening of the model→app boundary (nonce-tagged app messages, neutralized tool data, a spoken **yes** before any calendar booking, remote-LLM notice) in [docs/SECURITY.md](docs/SECURITY.md)

## Quick start

```bash
pip install -r requirements.txt        # or, reproducible from the lockfile:
                                       #   uv sync   (https://docs.astral.sh/uv/)

python pose_coach.py --list-devices              # which cameras / mics / speakers do I have?
python pose_coach.py --exercise squat            # webcam + voice (h in the window = keys)
python pose_coach.py --exercise squat --camera 1 # a different webcam (index, /dev/videoN or rtsp:// URL)
python pose_coach.py --exercise auto             # detect the exercise for me
python pose_coach.py --exercise plank --no-voice
python pose_coach.py --exercise deadlift --video set1.mp4
python pose_coach.py --exercise squat --record-reference   # save your best rep as the golden rep
python pose_coach.py --exercise squat            # future reps get a ref-sim 0-100 score
python pose_coach.py --train-classifier          # train the ML detector (~2 s, then auto uses it)
python pose_coach.py --exercise squat --load 60  # log 60 kg per rep → volume, est. 1RM, PRs
python pose_coach.py --stats                     # progress dashboard from the log
python coach_dashboard.py                        # same, as a web page with charts
python coach_dashboard.py --import-csv hevy.csv  # bring history from Strong / Hevy exports
python pose_coach.py --selftest                  # verify install, no camera needed
```

The pose model (~5 MB) downloads automatically on first run. Press `q` to end
a set: a summary card shows what happened and what to fix, and the set is
logged.

### In the window

| Key | Does |
|---|---|
| `1`–`9` | switch exercise (1 squat · 2 push-up · 3 bench · 4 deadlift · 5 lunge · 6 shoulder press · 7 curl · 8 pull-up · 9 plank) — the current number starts a fresh set |
| `a` | auto-detect the exercise from your movement |
| `r` | start / cancel a 60 s rest |
| `v` | mute / unmute the voice |
| `m` | mirror the camera on / off |
| `c` | talk to the coach now (`--coach`) |
| `h` | help overlay |
| `q` / `Esc` | finish the set → summary card |

Before anyone is tracked the window tells you what it needs (whole body in
view, light, camera placement); once you are, the gauge on the left shows
the live joint angle against the *rep starts / full depth / lockout* lines,
so a rep that doesn't count is never a mystery.

**Webcam setup, camera placement, and Docker webcam options:
[docs/WEBCAM.md](docs/WEBCAM.md).**

## Talk to your coach 🗣️

```bash
docker compose up -d ollama                          # local LLM in Docker
docker compose exec ollama ollama pull llama3.2:3b   # once

python pose_coach.py --exercise auto --coach   # chat while you train
                                               # (hands-free: just speak;
                                               #  'c' = interrupt the coach)
python coach_chat.py --voice --hands-free      # standalone voice chat
python coach_chat.py --voice --hands-free --mic "Camo"   # ...on a specific microphone
docker compose run --rm coach                  # text chat fully in Docker
python coach_knowledge.py --search "knees cave"   # what the coach retrieves for a question
python coach_mcp.py --list                     # the same data as MCP tools for Claude & co
```

The coach sees your live session (reps, scores, faults, fatigue) and your
training history, answers in your language, and speaks through your
speakers. Replies stream word-by-word, TTS starts at the first sentence,
and asking a new question instantly interrupts the old answer. Grounded in
an evidence-based coaching knowledge base (form faults, tempo, recovery,
nutrition basics) with strict medical-safety guardrails. Private by
default — the LLM runs on your machine. Full guide:
**[docs/COACH.md](docs/COACH.md)**.

## Docker

The image runs headless: video-file analysis, annotated output, and workout
logging (webcam/GUI from a container works on Linux hosts only).

```bash
docker build -t ai-gym-coach .
docker run --rm ai-gym-coach                      # selftest (default cmd)

# analyze a video: put it in ./data, get annotated.mp4 + workout_log.json back
docker run --rm -v ./data:/data ai-gym-coach \
    --exercise squat --video /data/squats.mp4 \
    --headless --no-voice --output /data/annotated.mp4 \
    --log-file /data/workout_log.json
```

Or with compose:

```bash
docker compose run --rm selftest
VIDEO=squats.mp4 EXERCISE=squat docker compose run --rm analyze
EXERCISE=squat docker compose run --rm webcam     # Linux host only
docker compose up dashboard                       # progress charts on :7788
```

Prebuilt image (published by CI from `main`):
`ghcr.io/feki-tech/ai-gym-coach:latest`.

Live webcam **inside** a container works on Linux hosts (device + X11
passthrough); on Windows/macOS record a video and analyze it, or run natively
— full guide in [docs/WEBCAM.md](docs/WEBCAM.md).

## CI

GitHub Actions runs the selftest suite on Ubuntu + Windows (Python 3.11/3.12),
builds the Docker image, re-runs the selftests inside the container, pushes
the image to GHCR, and builds the iOS app (CoachCore unit tests + simulator
build on macOS) on every push to `main`.

Beyond example tests, three heavier layers run in CI: **cross-platform
parity fixtures** (`parity_fixtures.py` generates deterministic angle
streams + expected FSM/plank/detector outputs from the Python engine into
`data/parity_fixtures.json`; the Swift and Kotlin test suites replay the
same file, so the three engines cannot drift apart silently), a
**property-based harness** (`prop_tests.py`, dependency-free seeded fuzz:
parsers never throw on garbage, the FSM can't count more reps than lockout
crossings, partitions lose nothing — it found a real crash on its second
generated case), and an **end-to-end golden session** (selftest 25 drives
the complete `run()` pipeline — smoothing, FSM, faults, log — through
injected synthetic skeletons and requires exactly 3 counted reps). The LLM
coach additionally gets a **nightly behaviour eval** against a real local
model (`coach-eval.yml`, docs/LLMOPS.md) so prompt drift shows up in the
morning, not in a workout.

### Camera placement

| Exercise | View |
|---|---|
| Squat, push-up, bench, deadlift, lunge, plank | Side view |
| Shoulder press, curl, pull-up | Front view |

## How it works

Camera → pose estimation → One Euro smoothing → joint angles (e.g. knee =
hip–knee–ankle) → per-exercise FSM for phases/reps/tempo → rule engine for
faults → prioritized feedback (screen + voice) → JSON workout log.

Full system design (model comparison, architecture, datasets, mobile
deployment, roadmap): **[docs/DESIGN.md](docs/DESIGN.md)**.

## iPhone app 📱

A native SwiftUI app lives in [`ios/`](ios/): Apple Vision body-pose on the
Neural Engine, live skeleton overlay, rep counting, voice coaching, and a
progress dashboard — same engine, same thresholds, same `workout_log.json`
schema as the desktop app. Fully localized (UI **and** spoken coaching cues)
in 6 languages: **English, 中文, हिन्दी, Español, Français, العربية**.
Build & App Store submission guide:
**[docs/IOS.md](docs/IOS.md)** — including getting it onto your iPhone via
TestFlight **without a Mac** (CI does the signing and uploading).

## Roadmap

- [x] Rule-based exercise auto-detection (`--exercise auto`)
- [x] Fatigue estimation from velocity loss
- [x] Progress dashboard (`--stats`)
- [x] Web progress dashboard with charts (`coach_dashboard.py`)
- [x] iOS app (SwiftUI + Apple Vision) — see [docs/IOS.md](docs/IOS.md)
- [x] iOS localization: en · zh-Hans · hi · es · fr · ar (UI + voice coaching)
- [x] Conversational LLM coach — talk to it by text/mic during workouts (Ollama in Docker), see [docs/COACH.md](docs/COACH.md)
- [x] DTW comparison against expert reference reps (`--record-reference`)
- [x] ML exercise auto-classification — numpy MLP on windowed skeleton features (`--train-classifier`, `--collect`)
- [x] Guided workout programs — the app (or the LLM coach) runs whole sessions: sets, rests, exercise switches (`--program`)
- [x] Android app (MediaPipe Tasks, Kotlin) — sideloadable APK from CI, see [android/README.md](android/README.md)
- [x] Sensor fusion PoC — BLE heart rate + IMU sources, recovery-based rest, HR in the coach's context; design + phased plan in [docs/SENSORS.md](docs/SENSORS.md)
- [x] Coach LLMOps — prompt versioning, local trace, deterministic guardrails, 34-scenario behaviour eval with baseline gate, `coach-eval` workflow — see [docs/LLMOPS.md](docs/LLMOPS.md); threat model (STRIDE + LLM-agent threats) and TB1 hardening in [docs/SECURITY.md](docs/SECURITY.md)
- [ ] Sport coach expansion — running module, smart garments, HRV readiness, physio companion, condition-aware & adaptive coaching (CGM context, gentle/seated modes, clinician-guarded zones); evidence review in [docs/RESEARCH.md](docs/RESEARCH.md)
- [ ] Infrastructure: locked dependencies (uv), classifier versioning + promotion gate, optional Azure demo deploy — phased plan in [docs/INFRA.md](docs/INFRA.md)

## Disclaimer

Not medical advice. Consult a professional trainer for heavy lifts.
