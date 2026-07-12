# AI Personal Gym Coach — Computer Vision System Design

*Version 1.0 — July 2026*

A complete blueprint for a real-time, camera-based fitness coach: pose estimation, exercise recognition, form evaluation, rep counting, tempo analysis, and live feedback.

---

## 1. Recommended stack (TL;DR)

| Layer | Recommendation | Why |
|---|---|---|
| Pose estimation (mobile + desktop) | **MediaPipe Pose Landmarker (BlazePose)** via Tasks API | 33 keypoints incl. feet, **3D world landmarks**, z-depth, runs 30+ FPS on phones, one codebase for Android/iOS/Web/Python |
| Pose estimation (high accuracy / server or GPU desktop) | **RTMPose** (MMPose) | Near-SOTA accuracy at 70–90+ FPS on CPU, ONNX/TFLite export, multi-person |
| Exercise classification | Rule/heuristic gate → **lightweight temporal model (ST-GCN or GRU)** on normalized keypoints | Robust, tiny (<1 MB), trainable on public datasets |
| Form evaluation | **Hybrid: biomechanical rules first, ML anomaly scoring second** | Explainable feedback ("straighten your back") requires rules; ML catches subtle deviations |
| Rep counting / phases | **Finite-state machine on filtered joint-angle signals** | Deterministic, debuggable, no training data needed |
| Smoothing | **One Euro filter** per keypoint (+ visibility gating) | Best lag/jitter trade-off for interactive use |
| Feedback | Prioritized message queue → on-screen + TTS (voice) | One cue at a time, rate-limited |

Start with MediaPipe everywhere; swap in RTMPose where accuracy matters and hardware allows.

---

## 2. Pose estimation model comparison

| Model | Keypoints | Accuracy (COCO AP*) | Speed | Mobile | GPU needed | Deploy ease | Real-time | Multi-person | 3D |
|---|---|---|---|---|---|---|---|---|---|
| **MediaPipe BlazePose** (Pose Landmarker) | 33 (+world 3D) | Good (~high 60s eq.) | 25–75 FPS phone | ★★★★★ | No (CPU/NNAPI/CoreML) | ★★★★★ (Tasks API, all platforms) | Yes | Limited (top-K) | **Yes** |
| **MoveNet** Lightning / Thunder | 17 | Good / very good | 34–51 FPS phone | ★★★★★ | No | ★★★★ (TFLite/TF.js) | Yes | Single (MultiPose variant exists) | No |
| **YOLO11-pose / YOLOv8-pose** | 17 | High (~68–71) | 30+ FPS GPU; heavier on phone CPU | ★★★ | Recommended | ★★★★ (Ultralytics API, ONNX/TFLite/CoreML export) | Yes | **Yes, native** | No |
| **OpenPose** | 18/25 + hands/face | Good | Slow (needs desktop GPU) | ★ | **Yes (CUDA)** | ★★ (C++/Caffe, licensing restrictions for commercial use) | GPU only | Yes (bottom-up) | No |
| **MMPose** (framework: HRNet, ViTPose…) | 17–133 | **SOTA (ViTPose ~78+)** | Varies; HRNet/ViT slow | ★★ | Usually | ★★★ (research-grade, ONNX export) | With GPU | Yes | Some models |
| **RTMPose** (in MMPose) | 17/26/133 | **Very high (~75)** | **70–90 FPS on CPU**, 400+ GPU | ★★★★ | No | ★★★★ (ONNX/TensorRT/ncnn/TFLite via MMDeploy) | Yes | Yes (top-down w/ detector) | RTMPose3D variant |
| **BlazePose** = the model inside MediaPipe Pose | — | — | — | — | — | — | — | — | — |

\* Rough COCO val AP equivalence; exact numbers vary by input size/variant.

### Verdicts
- **Phone app, fastest path to product** → MediaPipe Pose Landmarker (heavy variant for accuracy, lite for low-end devices). Its **world landmarks (metric 3D)** solve the camera-angle problem for joint angles — no other option gives you this for free.
- **Best accuracy/speed ratio 2026** → RTMPose. Ideal for desktop app, smart-mirror, or server-side video analysis.
- **Gym with multiple people in frame** → YOLO11-pose (detector+pose in one) or RTMPose + person detector.
- **Avoid OpenPose** for new projects: GPU-hungry, unmaintained, non-commercial license.
- **MoveNet** is fine but 17 keypoints (no feet) hurts squat/deadlift heel tracking; BlazePose's 33 points include heels and toes.

---

## 3. System architecture

```
┌────────────┐   frames    ┌──────────────────┐  33 kpts+conf  ┌───────────────┐
│  Camera    │───30 FPS───▶│  Pose Estimator   │───────────────▶│  Tracker +    │
│ (webcam /  │             │ (MediaPipe /      │                │  Smoother     │
│  phone)    │             │  RTMPose)         │                │ (One Euro,    │
└────────────┘             └──────────────────┘                │  gap filling) │
                                                                └───────┬───────┘
                                                                        │ clean skeleton stream
                          ┌─────────────────────────────────────────────┼─────────────────┐
                          ▼                                             ▼                 ▼
                 ┌────────────────┐                            ┌────────────────┐ ┌──────────────┐
                 │ Feature Layer  │                            │   Exercise     │ │  Calibration │
                 │ joint angles,  │                            │  Classifier    │ │ (body ratios,│
                 │ velocities,    │◀───────normalized pose─────│ (rules + GRU/  │ │  baseline    │
                 │ distances,     │                            │   ST-GCN)      │ │  posture)    │
                 │ alignment      │                            └───────┬────────┘ └──────┬───────┘
                 └───────┬────────┘                                    │ "squat"          │
                         │ angle signals                               ▼                  │
                         │                                    ┌────────────────┐          │
                         ├───────────────────────────────────▶│ Phase FSM +    │          │
                         │                                    │ Rep Counter +  │          │
                         │                                    │ Tempo Timer    │          │
                         │                                    └───────┬────────┘          │
                         ▼                                            │ phase, rep events │
                 ┌────────────────┐    per-rep metrics                ▼                   │
                 │ Form Evaluator │◀──────────────────────── ┌────────────────┐          │
                 │ rules engine + │                          │  Rep Segmenter │◀─────────┘
                 │ ML anomaly     │                          └────────────────┘
                 └───────┬────────┘
                         │ faults + severity + rep score
                         ▼
                 ┌────────────────┐    ┌─────────────────────┐   ┌───────────────────┐
                 │ Feedback Engine│───▶│ UI/Dashboard        │   │ Storage/Analytics │
                 │ prioritize,    │    │ skeleton overlay,   │   │ workout log, rep  │
                 │ rate-limit,    │    │ rep count, score,   │   │ history, progress │
                 │ TTS voice      │    │ tempo bar, cues     │   │ charts, summaries │
                 └────────────────┘    └─────────────────────┘   └───────────────────┘
```

**Threading model (critical for real-time):** camera capture, pose inference, and analytics/UI run on separate threads/queues. Pose inference is the bottleneck — never block it on drawing. MediaPipe's `LIVE_STREAM` mode does this via callback; drop stale frames rather than queueing.

---

## 4. Technical deep dives

### 4.1 Joint angles from keypoints
Angle at joint B formed by segments BA and BC (e.g., knee angle = hip–knee–ankle):

```
cosθ = (BA · BC) / (|BA| |BC|)      θ = arccos(clamp(cosθ, -1, 1))
```

In 2D you can also use `atan2` per segment and subtract — more robust near 0/180°:
`θ = |atan2(Cy−By, Cx−Bx) − atan2(Ay−By, Ax−Bx)|` (wrap to ≤180°).

**Key rules:**
- Prefer **3D world landmarks** (BlazePose) — 2D angles distort badly when the camera isn't perpendicular to the movement plane. With 2D-only models, instruct the user to film from the side for squats/deadlifts, front for curls/press.
- Trunk lean, shin angle etc. are measured **against gravity**: angle between the segment and the vertical image axis (or gravity vector from phone IMU).
- Ignore/hold angles when any of the three keypoints has visibility/confidence < ~0.5.

Core angles per exercise: knee (hip-knee-ankle), hip (shoulder-hip-knee), elbow (shoulder-elbow-wrist), shoulder (hip-shoulder-elbow), trunk-vs-vertical, shin-vs-vertical, neck (ear-shoulder-hip).

### 4.2 Smoothing noisy keypoints
Raw keypoints jitter by several pixels and occasionally jump on occlusion.

1. **One Euro filter (recommended default)** — adaptive low-pass: heavy smoothing when slow (kills jitter), light when fast (low lag). Two params (`min_cutoff≈1.0`, `beta≈0.01–0.05` for normalized coords). Apply per coordinate per keypoint.
2. **EMA** — simplest, fixed lag; fine for slow lifts, laggy on fast reps.
3. **Kalman filter** (constant-velocity) — good for occlusion gap-filling and outlier rejection via innovation gating; more tuning.
4. **Savitzky–Golay** — excellent for *offline* rep analysis (preserves peaks); non-causal, so not for live overlay.
5. **Confidence gating** — if visibility < threshold, hold last value / interpolate; never feed garbage into the FSM.
6. Smooth **derived angle signals** again with a short median filter (window 5) before the rep FSM — kills single-frame spikes that cause double-counting.

### 4.3 Exercise classification from skeleton data
Two-stage approach:

**Stage 1 — normalization (essential):** translate pelvis to origin, scale by torso length (mid-shoulder↔mid-hip), optionally rotate so shoulders are horizontal. Removes camera distance, position, and body size. Use these normalized coords + angles as features.

**Stage 2 — classify:**
- *MVP (rule-based):* a decision tree over posture statistics works surprisingly well for 9 distinct exercises: body horizontal + elbows flexing → push-up/plank (elbow ROM distinguishes them); standing + deep knee flexion → squat vs lunge (stance asymmetry); standing + only elbow flexion → curl; wrists above head cycling → shoulder press vs pull-up (body suspended, wrists fixed, body moves).
- *Production (ML):* sliding window of 2–3 s (60–90 frames) of normalized keypoints → **GRU/LSTM (~0.5 MB)** or **ST-GCN** (graph convolution over the skeleton — best accuracy for action recognition). Train on Fit3D/mm-Fit/Countix-Fit + your own recordings. Expect >95% on 9 classes.
- Add a "no exercise / rest" class and require N consecutive agreeing windows before switching state.

In practice most apps let the user **select the exercise** and use the classifier only for auto-detection convenience/validation.

### 4.4 Comparing user movement vs ideal form
- **Reference templates:** record experts, extract normalized angle trajectories per rep, time-normalize each rep to 0–100% phase, average → mean trajectory ± tolerance band per angle.
- **Alignment:** **Dynamic Time Warping (DTW)** aligns user rep to the template despite speed differences; the DTW distance itself is a form-quality feature. For live use, align by **phase** (from the FSM) instead — compare user's knee/hip/trunk angles at the same phase percentage, which is O(1) per frame.
- **Per-phase checkpoints** beat whole-trajectory distance for feedback: e.g. "at bottom of squat: hip angle should be < 90°, trunk lean < 45°."

### 4.5 Rules vs ML for form evaluation → **Hybrid**
- **Rules (biomechanical thresholds) are the backbone:** explainable ("knee 15° past toe"), zero training data, tunable per user, deterministic → safe coaching. Weakness: hand-tuning, camera-angle sensitivity (mitigated by 3D landmarks).
- **ML (anomaly detection / fault classifiers):** train per-exercise fault classifiers (good/fault-X) or an autoencoder on good reps → reconstruction error = "something's off" score. Catches subtle compound faults rules miss. Weakness: needs labeled fault data, can't explain itself well.
- **Ship rules first.** Add ML scoring in v2 as a secondary signal and to auto-tune rule thresholds from data. Every user-facing cue should trace to a rule.

### 4.6 Exercise phase detection (start → descent → bottom → ascent → finish)
Use a **finite-state machine on the primary angle signal** (squat: knee angle; curl: elbow angle; push-up: elbow angle):

```
IDLE ──angle < θ_start_moving──▶ DESCENT ──velocity ≈ 0 & angle < θ_deep──▶ BOTTOM
  ▲                                                                          │
  │                                                                    angle rising
 rep++ ◀── angle > θ_lockout ─── ASCENT ◀────────────────────────────────────┘
```

- Thresholds with **hysteresis** (e.g., descent starts < 160°, lockout > 165°) prevent chattering.
- Velocity = filtered dθ/dt; zero-crossings mark turning points (bottom/top).
- **Rep counted only on full cycle** DESCENT→BOTTOM→ASCENT→lockout; enforce minimum ROM (else "incomplete rep — go deeper") and minimum duration (else it's noise).
- **Tempo** = timestamps between state transitions → eccentric time, pause, concentric time (e.g., "3-1-1"). Cue "slow down" if concentric < user's target.
- **Rep quality:** evaluate form rules per phase during the rep, aggregate at lockout → per-rep score.

### 4.7 Scoring
Per rep: `score = 100 − Σ (fault_severity × weight)`, faults gated by phase (depth fault only judged at BOTTOM). Set score: mean rep score, penalized for tempo collapse and ROM decline (fatigue signal). Report per-fault breakdown, not just a number.

---

## 5. Per-exercise fault rules (initial 9)

Camera: S = side view, F = front view. Angles from 3D world landmarks where possible.

| Exercise | Primary signal | Key checks |
|---|---|---|
| **Squat** (S+F) | knee angle | Depth: hip below knee (hip angle <~90°) at BOTTOM; trunk lean vs vertical < ~45°; knee-over-toe: knee.x beyond toe.x by margin (side); knees cave in: F view knee distance / ankle distance < 0.8 (valgus); heels down (heel keypoint stable); back not rounding (shoulder-hip-knee collinearity stable) |
| **Push-up** (S) | elbow angle | Body line: shoulder-hip-ankle collinear within ~10° (no sag/pike); depth: elbow < ~90° at BOTTOM; full lockout > ~160°; elbows tucked: shoulder-elbow abduction < ~60° from torso; neck neutral (ear-shoulder-hip) |
| **Bench press** (S, camera at 45° head-side) | elbow angle | Bar path proxy: wrist vertical trajectory straightness; touch depth; lockout; elbow flare angle; wrists stacked over elbows |
| **Deadlift** (S) | hip angle | Back rounding: shoulder-hip vs hip-knee angle change during pull (spine proxy: ear-shoulder-hip); bar (wrist) close to shins; hips not shooting up first (hip vs knee extension rate ratio); lockout: full hip extension, no hyperextension lean-back |
| **Lunge** (S+F) | front knee angle | Front knee ~90° and over ankle (not past toes); torso upright; rear knee approaches floor; F: hip drop / lateral wobble (pelvis tilt), stance alignment |
| **Shoulder press** (F) | elbow angle | Full lockout overhead, wrists over shoulders; no excessive lumbar arch (S view trunk lean); symmetric: L/R wrist height diff < threshold; elbow path vertical |
| **Bicep curl** (S/F) | elbow angle | Elbow pinned: upper-arm (shoulder-elbow) angle vs vertical stays < ~15° (no swinging); full ROM top and bottom; no shoulder shrug (shoulder-ear distance); trunk stays vertical (no momentum lean) |
| **Pull-up** (F) | elbow angle + wrist-fixed | Chin over bar (nose.y above wrist.y) at TOP; full hang at bottom (elbow > ~160°); no excessive kipping (hip.x oscillation amplitude); symmetric pull (L/R elbow angle diff) |
| **Plank** (S, static) | none — posture hold | Continuous check: shoulder-hip-ankle line within band (sag = hip below line, pike = above); neck neutral; timer instead of reps; alert on drift with grace period |

Uneven alignment (all bilateral moves): compare L vs R joint angles each phase; flag if sustained diff > ~10–15°.

---

## 6. Advanced features

- **Personalized coaching:** calibration flow (stand in T-pose + a few bodyweight reps) → limb-length ratios (femur/torso long → naturally more trunk lean in squat: widen the lean threshold). Store per-user threshold offsets; adapt from their pain-free ROM.
- **Injury-risk detection:** flag high-risk patterns with severity levels — lumbar flexion under load (deadlift rounding), knee valgus at depth, cervical hyperextension. Persistent risk → suggest regression exercise + reduce intensity. *Always ship with a "not medical advice" disclaimer.*
- **Fatigue estimation:** within-set trends of concentric velocity (velocity-based training: >20% velocity loss ≈ meaningful fatigue), ROM decline, form-score decline, inter-rep rest creep → "2 reps left in the tank" style cues.
- **Workout summaries & progress tracking:** per-set score/tempo/ROM stored → weekly charts, PRs on ROM/velocity, fault frequency trend ("knee valgus down 40% this month").
- **Voice feedback:** TTS with priority queue — safety cues (back rounding) preempt everything; max ~1 cue per 3–4 s; positive reinforcement after clean reps; count reps aloud.
- **Automatic logging:** classifier + rep counter → auto-log exercise, sets, reps, rest times; export to Google Fit / Apple Health.
- **Multi-camera:** two calibrated views → triangulated true 3D, eliminates single-view ambiguity. Sync via timestamps; needs one-time checkerboard/AprilTag calibration. Great for smart-mirror/studio product, overkill for phone MVP.
- **3D pose:** BlazePose world landmarks (MVP) → RTMPose3D / MotionBERT lifting (better) → multi-camera triangulation (best).
- **Wearable/IMU fusion:** smartwatch IMU gives wrist acceleration when hands leave frame, bar-speed proxy, and heart rate for effort context; fuse via Kalman with vision (vision = position ground truth, IMU = high-rate motion).

---

## 7. Datasets

| Dataset | What | Use |
|---|---|---|
| **COCO Keypoints / COCO-WholeBody** | 200K+ images, 17/133 kpts | Pretraining/benchmarking pose models (already baked into all listed models) |
| **MPII Human Pose** | 25K images, activity labels incl. gym | Pose benchmarking |
| **Fit3D** | 3M+ frames, 47 exercises, 3D SMPL-X, rep boundaries | Exercise classification, rep segmentation, 3D reference form |
| **mm-Fit** | Multimodal workout: video-derived poses + IMU | Classifier training, IMU fusion |
| **Countix / RepCount** | Repetitive-action videos with rep counts | Training/validating rep counting |
| **InfiniteRep** (synthetic) | 1K videos, avatars doing reps, perfect labels | Augmenting classifier/counter training |
| **Waseda GYM / Fitness-AQA** | Exercise videos with **fault annotations** (e.g., knees-in) | Form-fault classifier training |
| **NTU RGB+D 120** | 114K skeleton action clips | Pretraining ST-GCN action backbone |
| **Your own capture** (essential) | 20–50 people, phone cameras, labeled faults | The single highest-value dataset for form evaluation; budget for it |

## 8. Open-source implementations to study/reuse

- `google-ai-edge/mediapipe` + `mediapipe-samples` — Pose Landmarker for Python/Android/iOS/Web.
- `open-mmlab/mmpose` — RTMPose, ViTPose, RTMPose3D; `mmdeploy` for ONNX/TensorRT/ncnn export.
- `Tencent/ncnn` + RTMPose ncnn ports — phone-CPU deployment of RTMPose.
- `ultralytics/ultralytics` — YOLO11-pose, simplest multi-person API.
- **Reference gym apps:** `Musclesinaction`, "AI-Fitness-Trainer" style repos (search: *squat form checker mediapipe*), `KNN-based exercise classification` demos; `moon-hotel/BlazePose-tf2`; Kalidokit (smoothing patterns for live avatars).
- `jaantollander/OneEuroFilter` — clean One Euro implementation.
- TTS: `pyttsx3` (offline desktop), platform TTS on mobile.

## 9. Deployment

| Target | Stack | Notes |
|---|---|---|
| **Android** | MediaPipe Tasks (Kotlin) `PoseLandmarker` LIVE_STREAM + GPU delegate/NNAPI; analytics in Kotlin; or RTMPose via ncnn/TFLite | 30+ FPS on mid-range; CameraX for capture |
| **iPhone** | MediaPipe Tasks (Swift) w/ CoreML delegate, or convert MoveNet/RTMPose to CoreML; alternative: Apple Vision `VNDetectHumanBodyPose3DRequest` (native, 3D, zero deps) | Metal GPU delegate; consider Vision framework to cut binary size |
| **Desktop (dev + smart-mirror)** | Python: `mediapipe` pip + OpenCV UI (prototype); production: Electron/Tauri + web build, or Python + Qt; RTMPose ONNX-runtime for accuracy | This repo's `pose_coach.py` is the working prototype |
| **Web (zero-install)** | MediaPipe Tasks JS (WASM+WebGPU) or TF.js MoveNet | Great for demos and reach; analytics in TS reused from a shared core |

**Architecture tip:** write the analytics core (angles, filters, FSM, rules) once in a portable form — either TypeScript (web+RN) or Kotlin Multiplatform, or keep it pure-math and port; it's ~1K lines and has no ML dependencies. All heavy ML stays in the vendor runtimes.

**Privacy:** all inference on-device; store only derived metrics (angles/scores), never video, unless the user opts in.

## 10. Development roadmap

| Phase | Duration | Scope | Exit criteria |
|---|---|---|---|
| **0 — Spike** | 1–2 wk | Python + MediaPipe webcam demo: skeleton overlay, angles, One Euro filter (`pose_coach.py`) | 25+ FPS laptop; stable angles |
| **1 — Vertical slice** | 3–4 wk | Squat + curl: FSM rep counter, tempo, 4–5 form rules, on-screen + voice cues, session summary | Rep count ≥95% accurate on 20 test videos; cues correct on seeded faults |
| **2 — Exercise coverage** | 4–6 wk | All 9 exercises, rule library, per-exercise camera-placement guides, calibration flow, local workout log | Full library demoable; per-rep scores stable |
| **3 — ML upgrades** | 4–8 wk | Exercise auto-classifier (GRU/ST-GCN on Fit3D + own data), DTW template scoring, fault classifiers on Fitness-AQA + own labeled captures | Classifier >95%; ML score correlates with coach ratings (collect them!) |
| **4 — Mobile app** | 6–10 wk | Android first (MediaPipe Tasks), shared analytics core, UI/dashboard, TTS, auto-logging | 30 FPS mid-range phone; field test with 20 users |
| **5 — Production** | ongoing | iOS, progress analytics, fatigue/injury-risk, health-platform export, A/B threshold tuning, optional cloud sync | Retention + coaching-accuracy metrics |

**Risks to manage early:** camera placement variance (mitigate: placement wizard + 3D landmarks), baggy-clothing keypoint noise (test early), false-positive cues eroding trust (bias thresholds conservative — silence beats wrong advice), plank/floor exercises with poor viewing angles (dedicated camera guide).

## 11. Reference implementation

See **`pose_coach.py`** (repo root) — runnable prototype v3:
- MediaPipe Tasks Pose Landmarker (webcam or video file), One Euro filtering, joint angles
- **All 9 exercises**: FSM rep counting/tempo for squat, push-up, bench, deadlift, lunge, shoulder press, curl, pull-up (concentric-first handling for curl/pull-up) + timed **plank** hold with body-line monitoring
- Per-exercise live form rules (back lean/rounding, knee valgus, elbow flare/swing, unevenness, chin-over-bar, body sag) + per-rep depth/tempo checks and scores
- **Auto exercise detection** (`--exercise auto`): rule-based classifier over a 2 s sliding window of skeleton features (trunk angle, joint ROMs, wrist-overhead ratio, shoulder vs wrist displacement, knee split), majority-vote lock after 3 agreeing votes — the §4.3 stage-2 MVP. Detects 8 of 9 (bench press reads as push-up from skeleton alone → manual)
- **Fatigue estimation** (§6): concentric velocity proxy = primary-angle ROM / concentric time; warns once when the average of the last 2 reps drops >20 % below the best of the first 3; `velocity_loss_pct` stored per session
- **Voice coaching** via pyttsx3 (background TTS thread, prioritized + rate-limited cues, spoken rep counts)
- **Workout logging + progress**: per-rep metrics (incl. velocity), fault counts, session summaries in `workout_log.json`; `--stats` renders a per-exercise dashboard (sessions, total reps, score-trend sparklines, top faults, plank hold bests)
- Headless/Docker mode (`--headless`, `--output annotated.mp4`), CI-tested (13 selftests)

```
pip install mediapipe opencv-python numpy pyttsx3
python pose_coach.py --exercise squat            # webcam + voice
python pose_coach.py --exercise auto             # recognize the movement
python pose_coach.py --exercise deadlift --video set1.mp4
python pose_coach.py --stats                     # progress dashboard
python pose_coach.py --selftest                  # verify without a camera
```
