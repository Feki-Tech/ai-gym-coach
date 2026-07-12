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
- **Workout log** — per-rep scores, tempo, and fault statistics appended to `workout_log.json`

## Quick start

```bash
pip install -r requirements.txt

python pose_coach.py --exercise squat            # webcam + voice
python pose_coach.py --exercise plank --no-voice
python pose_coach.py --exercise deadlift --video set1.mp4
python pose_coach.py --selftest                  # verify install, no camera needed
```

The pose model (~5 MB) downloads automatically on first run. Press `q` to end
a set and print the session summary.

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

## CI

GitHub Actions runs the selftest suite on Ubuntu + Windows (Python 3.11/3.12),
builds the Docker image, re-runs the selftests inside the container, and
pushes the image to GHCR on every push to `main`.

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

## Roadmap

- [ ] ML exercise auto-classification (GRU/ST-GCN on normalized keypoints)
- [ ] DTW comparison against expert reference reps
- [ ] Fatigue estimation from velocity loss
- [ ] Android app (MediaPipe Tasks, Kotlin)

## Disclaimer

Not medical advice. Consult a professional trainer for heavy lifts.
