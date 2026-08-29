# Gym Coach — Android

Kotlin port of the coaching engine, running on **MediaPipe Tasks** pose
landmarks (the last unchecked product roadmap item). Same architecture as
iOS: a pure, JVM-testable core (`core/` — geometry, One Euro smoothing,
rep-counting FSM, plank tracker, rule-based auto-detection, form rules,
fatigue monitor, session engine) with **thresholds identical to the desktop
prototype**, plus a thin app layer (CameraX front camera → PoseLandmarker
LIVE_STREAM → 33→15-joint mapping → overlay/HUD/TTS).

Local-first, like everything else here: sessions are logged to an
app-private `workout_log.json` (same record shape as the desktop); nothing
leaves the phone. No accounts, no network use at runtime — the app's only
download is the pose model, and that happens at *build* time.

## Install on your phone (no Play Store)

Every CI run on `main` builds a sideloadable APK:

1. GitHub → Actions → **Android** → latest run → download the
   `gymcoach-debug-apk` artifact (needs a GitHub login), **or** push a
   `android-v*` tag — the workflow then attaches the APK to a public GitHub
   Release you can open directly in the phone's browser.
2. On the phone: open the APK → Android asks to allow installs from that
   browser/file manager ("install unknown apps") → allow → install.
   Samsung shows an extra "unsafe app blocked" dialog: More details →
   Install anyway (it is the debug-signed build, not a store build).

iPhones cannot sideload an APK, and this Kotlin app doesn't run on iOS.
The store-free path on iPhone is the existing SwiftUI app via **TestFlight**
(`.github/workflows/testflight.yml`, docs/IOS.md) — install the TestFlight
app, open the beta invite link, done. That is Apple-hosted but not the App
Store.

## Build locally

```bash
cd android
./gradlew assembleDebug     # downloads the pose model into assets on first run
./gradlew test              # JVM unit tests for the core port (no device)
```

Then `adb install app/build/outputs/apk/debug/app-debug.apk`.

## What works / what's next

Working: live pose overlay, 8-exercise auto-detect (bench stays manual —
indistinguishable from a push-up in skeleton view), rep counting with
depth/tempo checks, per-exercise form faults with spoken cues, plank hold
timer, velocity-loss fatigue warning, per-session JSON log.

**The trained classifier runs here too.** Auto-detect uses the same gated,
versioned ~1.5k-parameter MLP the desktop trains — export it once and drop
it into the app:

```bash
python pose_coach.py --train-classifier                 # desktop, gated
python pose_coach.py --export-model classifier.json     # portable weights
adb push classifier.json /data/data/com.fekitech.gymcoach/files/
```

On the next launch auto-detect switches from the rule tier to the MLP
(`MlDetector`, same sliding window and 3-agreeing-votes lock-in); without
the file the rules keep working, exactly like the desktop before
`--train-classifier`. The pushed file is treated as untrusted input: a
malformed or shape-inconsistent model is refused at load (`TinyMlp.checked`
— see docs/SECURITY.md S14) and the rules keep working. Inference parity
with Python is pinned by the `window_feature`/`mlp` sections of
`data/parity_fixtures.json`.

Next: localization (the core keeps all strings in `FormRules.kt` for that
reason), history UI, guided programs, phone-IMU sensor fusion
(docs/SENSORS.md).
