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

- A window opens. Until it sees you, a **framing guide** says what it needs
  (whole body in view, enough light, camera placement for the exercise).
- Once tracked: your skeleton (the body part at fault turns red), a
  **range-of-motion gauge** on the left with the live joint angle against the
  exercise's *rep starts / full depth / lockout* lines, the rep counter with
  its phase (lowering / bottom / lifting) and goal ring, the last rep's score,
  tempo, golden-rep similarity and fatigue on the right, coaching cues at the
  bottom, the rest countdown, program progress and — with `--coach` — the
  coach's answers and the mic meter.
- The webcam is mirrored like a gym mirror (`m` toggles, `--no-mirror`
  disables). Video files and streams keep their orientation.
- Voice coaching speaks the cues and rep counts (`--no-voice`, or `v` in the
  window to mute).
- Keys (also under `h`): `1`–`9` switch exercise (the current number starts a
  fresh set), `a` auto-detect, `r` rest 60 s, `v` voice, `m` mirror, `c` talk
  to the coach, `q`/`Esc` finish.
- Press **q** or **Esc** to end the set (click the window first so it has
  keyboard focus). A **summary card** shows reps, scores, tempo, faults and
  what to fix — and if no rep was counted, *why* (how deep you got versus the
  thresholds). The set is appended to `workout_log.json`.

### Choosing a camera (and a microphone)

```bash
python pose_coach.py --list-devices      # cameras with resolution, mics, speakers
python coach_devices.py --ble            # same, plus a 5 s scan for BLE heart-rate straps

python pose_coach.py --exercise squat --camera 1               # second webcam
python pose_coach.py --exercise squat --camera /dev/video2     # Linux device path
python pose_coach.py --exercise squat --camera rtsp://phone:8554/live   # phone/IP camera stream
python pose_coach.py --exercise auto --coach --mic "Camo"      # mic by (part of) name, or index
```

- Default is camera `0` and the OS default microphone. `COACH_CAMERA` and
  `COACH_MIC` environment variables set the defaults; the flags override.
- Any OpenCV-openable source works as `--camera`: an index, a device path,
  or an `rtsp://` / `http://…/mjpg` URL (phone apps like Camo, DroidCam or
  IP Webcam expose one).
- Windows: `--list-devices` shows the PnP names next to the indices; the
  order usually matches OpenCV's, so if `--camera 1` opens the wrong one,
  try the neighbour.
- `--mic` takes a sounddevice index or a case-insensitive part of the name
  (`"camo"`, `"intel"`); an unknown name exits with the list of inputs.
- Docker (Linux): `CAMERA_DEV=/dev/video2 docker compose run --rm webcam`
  maps that host camera to `/dev/video0` inside the container.

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

> Advanced (unsupported): `usbipd-win` can attach a USB webcam to WSL2.
> Older stock WSL2 kernels shipped without the `uvcvideo` driver (custom
> kernel needed); recent ones (6.6+, check `modinfo uvcvideo` inside WSL)
> ship it as a module — `usbipd attach --wsl --busid <id>` on Windows, then
> `sudo modprobe uvcvideo` and `python pose_coach.py --list-devices` in WSL.
> Built-in laptop cameras are often not attachable this way. Recording +
> analyzing, or running natively on Windows, remains the pragmatic path.

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
| Windows/macOS: `error gathering device information ... "/dev/video0"` | Expected — Docker Desktop has no webcam passthrough. Use section 3 (record → analyze) or run natively. |
| Auto-detect locks the wrong exercise | Restart and select it explicitly, e.g. `--exercise lunge`. Bench press always needs manual selection. |
