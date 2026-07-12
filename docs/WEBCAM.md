# Trying the coach with your webcam

Everything runs locally — no video ever leaves your machine.

There are two ways to use your webcam:

| Setup | Live GUI + voice | Webcam access | Best for |
|---|---|---|---|
| **Native Python** | ✅ | ✅ all OSes | Windows, macOS, everyday use |
| **Docker (Linux host)** | ✅ (X11) | ✅ `/dev/video0` | Linux boxes, reproducible setup |
| **Docker (Windows/macOS)** | ❌ no camera passthrough | record → analyze | CI-like analysis of recorded sets |

---

## 1. Native (recommended on Windows/macOS)

```bash
pip install -r requirements.txt

python pose_coach.py --exercise auto        # let it recognize the movement
python pose_coach.py --exercise squat       # or pick the exercise yourself
python pose_coach.py --exercise plank --no-voice
python pose_coach.py --stats                # progress dashboard afterwards
```

- A window opens with your skeleton, rep counter, phase, and live cues.
- Voice coaching speaks the cues and rep counts (`--no-voice` to mute).
- Press **q** or **Esc** in the video window to end the set (click the window
  first so it has keyboard focus). You'll get a session summary and the set is
  appended to `workout_log.json`.

### Camera placement

| Exercise | Camera view | Distance |
|---|---|---|
| Squat, push-up, bench, deadlift, lunge, plank | **Side** (90° profile) | 2.5–4 m, whole body in frame |
| Shoulder press, bicep curl, pull-up | **Front** | 2–3 m, head to hips minimum |

Tips for clean tracking:

- Whole body (or the working joints) visible the entire rep — cropped ankles
  ruin squat depth detection.
- Even lighting from the front; avoid a bright window behind you.
- Plain background and fitted clothing help; baggy hoodies hide elbows.
- Put the camera on a tripod/shelf at hip height, not on a wobbling surface.
- `--exercise auto` needs ~2–3 s of movement before it locks on; it announces
  the detected exercise on screen and by voice.

---

## 2. Docker on Linux — live webcam in the container

The image is GUI-capable; you pass the camera device and the X11 socket:

```bash
docker build -t ai-gym-coach .

xhost +local:docker          # allow the container to open a window (once per login)

docker run --rm \
  --device /dev/video0:/dev/video0 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  ai-gym-coach --exercise auto --no-voice
```

Or the ready-made compose service (same flags, plus `./data` mounted):

```bash
EXERCISE=squat docker compose run --rm webcam
```

Notes:

- Your user must be able to read `/dev/video0` (usually the `video` group).
- Wayland sessions: XWayland makes the X11 socket above work on most distros.
- Keep `--no-voice` in containers — there is no audio device inside by
  default. (If you really want voice, additionally mount the PulseAudio
  socket: `-e PULSE_SERVER=unix:/run/user/1000/pulse/native
  -v /run/user/1000/pulse/native:/run/user/1000/pulse/native`.)
- **Headless webcam** (no GUI, e.g. over SSH): add `--headless` plus a volume
  for the results, stop with **Ctrl+C**:

  ```bash
  docker run --rm --device /dev/video0:/dev/video0 -v ./data:/data \
    ai-gym-coach --exercise squat --headless --no-voice \
    --output /data/annotated.mp4 --log-file /data/workout_log.json
  ```

---

## 3. Docker on Windows / macOS — record, then analyze

Docker Desktop runs containers in a VM (WSL2/HyperKit) that **cannot see the
host webcam**, so live camera-in-container is not possible there. Two options:

**a) Run natively for live coaching** (section 1) — the Docker image is still
useful for CI and video analysis.

**b) Record a set, analyze it in the container** — works identically on every
OS:

1. Record yourself with the Windows Camera app, your phone, or OBS
   (side view for squats/deadlifts — see the placement table above).
2. Save it as `data\squats.mp4` in the repo folder.
3. Analyze:

   ```powershell
   docker run --rm -v ${PWD}\data:/data ai-gym-coach `
     --exercise auto --video /data/squats.mp4 `
     --headless --no-voice `
     --output /data/annotated.mp4 --log-file /data/workout_log.json
   ```

   or with compose:

   ```powershell
   $env:VIDEO="squats.mp4"; $env:EXERCISE="auto"; docker compose run --rm analyze
   ```

4. Results land back in `.\data\`: `annotated.mp4` (skeleton + HUD overlay)
   and `workout_log.json`. Show the progress dashboard from the same log:

   ```powershell
   docker run --rm -v ${PWD}\data:/data ai-gym-coach --stats --log-file /data/workout_log.json
   ```

> Advanced (unsupported): `usbipd-win` can attach a USB webcam to WSL2, but
> the stock WSL2 kernel ships without the `uvcvideo` driver, so it requires
> building a custom kernel. Recording + analyzing is the pragmatic path.

---

## 4. Troubleshooting

| Symptom | Fix |
|---|---|
| `Could not open camera/video.` | Camera busy (close Teams/Zoom/OBS), or blocked: Windows *Settings → Privacy → Camera*, macOS *System Settings → Privacy → Camera*, Linux check `/dev/video0` permissions. |
| Window opens but no skeleton | Step back until your whole body is in frame; improve lighting; avoid strong backlight. |
| Reps not counted / counted late | Wrong camera angle (use the placement table); make full-range reps — half reps below the FSM thresholds don't latch. |
| `q` does nothing | Click the video window first (keyboard focus), or use Ctrl+C in the terminal with `--headless`. |
| Jittery skeleton | More light, higher camera, plain background. The One Euro filter handles small noise; darkness causes big noise. |
| Linux Docker: `cannot open display` | Run `xhost +local:docker` in your desktop session; make sure `DISPLAY` is set. |
| Auto-detect locks the wrong exercise | Restart and select it explicitly, e.g. `--exercise lunge`. Bench press always needs manual selection. |
