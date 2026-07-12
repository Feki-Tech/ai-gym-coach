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
- **Voice coaching** — prioritized, rate-limited cues via TTS ("Straighten your back", "Slow down", "Great form!")
- **Auto exercise detection** — `--exercise auto` recognizes the movement from the skeleton (8 of 9 exercises; bench press needs manual selection)
- **Fatigue monitor** — warns once when concentric rep velocity drops >20% vs your first reps
- **Golden-rep comparison** — record your best rep once (`--record-reference`), then every future rep gets a 0-100 DTW similarity score against it (tempo-independent shape match)
- **Workout log & progress** — per-rep scores, tempo, velocity, and fault statistics in `workout_log.json`; `--stats` prints a progress dashboard with score trends
- **Talk to your coach** — a local LLM (Ollama in Docker) answers questions by text or **voice** during the workout, with your live session + history as context — see [docs/COACH.md](docs/COACH.md)

## Quick start

```bash
pip install -r requirements.txt

python pose_coach.py --exercise squat            # webcam + voice
python pose_coach.py --exercise auto             # detect the exercise for me
python pose_coach.py --exercise plank --no-voice
python pose_coach.py --exercise deadlift --video set1.mp4
python pose_coach.py --exercise squat --record-reference   # save your best rep as the golden rep
python pose_coach.py --exercise squat            # future reps get a ref-sim 0-100 score
python pose_coach.py --stats                     # progress dashboard from the log
python pose_coach.py --selftest                  # verify install, no camera needed
```

The pose model (~5 MB) downloads automatically on first run. Press `q` to end
a set and print the session summary.

**Webcam setup, camera placement, and Docker webcam options:
[docs/WEBCAM.md](docs/WEBCAM.md).**

## Talk to your coach 🗣️

```bash
docker compose up -d ollama                          # local LLM in Docker
docker compose exec ollama ollama pull llama3.2:3b   # once

python pose_coach.py --exercise auto --coach   # chat while you train
                                               # ('c' in the window = ask by mic)
python coach_chat.py --voice --listen          # standalone voice chat
docker compose run --rm coach                  # text chat fully in Docker
```

The coach sees your live session (reps, scores, faults, fatigue) and your
training history, answers in your language, and speaks through your
speakers. Private by default — the LLM runs on your machine. Full guide:
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
- [x] iOS app (SwiftUI + Apple Vision) — see [docs/IOS.md](docs/IOS.md)
- [x] iOS localization: en · zh-Hans · hi · es · fr · ar (UI + voice coaching)
- [x] Conversational LLM coach — talk to it by text/mic during workouts (Ollama in Docker), see [docs/COACH.md](docs/COACH.md)
- [x] DTW comparison against expert reference reps (`--record-reference`)
- [ ] ML exercise auto-classification (GRU/ST-GCN on normalized keypoints)
- [ ] Android app (MediaPipe Tasks, Kotlin)

## Disclaimer

Not medical advice. Consult a professional trainer for heavy lifts.
