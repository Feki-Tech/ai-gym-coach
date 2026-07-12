# iOS App — Build & App Store Guide

The `ios/` folder contains a native iPhone app of the AI Gym Coach:

| Part | What it is |
|---|---|
| `ios/CoachCore/` | Swift package with the entire coaching engine — geometry, One Euro smoothing, rep-counting FSM, form rules, auto-detection, fatigue monitor, workout log. Pure Swift, no UI, fully unit-tested (mirrors the Python `--selftest` suite). |
| `ios/GymCoach/` | SwiftUI app: camera capture, **Apple Vision** body-pose detection (`VNDetectHumanBodyPoseRequest`, runs on the Neural Engine), skeleton overlay, live HUD, voice coaching, history with score-trend charts. |

Pose estimation uses Apple's built-in Vision framework instead of MediaPipe:
zero external dependencies, no model file to bundle, hardware-accelerated on
every iPhone since ~2018. All video is processed on-device; nothing is
uploaded — which also makes the App Store privacy questionnaire trivial.

---

## 1. Requirements

- A Mac with **Xcode 15+** (iOS 17 SDK; the app targets iOS 16+)
- [XcodeGen](https://github.com/yonaskolb/XcodeGen) — `brew install xcodegen`
- For device runs & App Store: an [Apple Developer](https://developer.apple.com/programs/) account ($99/year)

> No Mac? GitHub Actions builds the app on every push (see `.github/workflows/ci.yml`,
> job `ios`) — it runs the CoachCore tests and compiles the app for the iOS
> Simulator, so the code is always verified even when developing from
> Windows/Linux. Signing and uploading still require a Mac (or a CI signing
> setup with fastlane, see §5).

## 2. Build & run

```bash
git clone https://github.com/Feki-Tech/ai-gym-coach
cd ai-gym-coach

# run the engine unit tests (works on any Mac, no Xcode project needed)
swift test --package-path ios/CoachCore

# generate the Xcode project (the .xcodeproj is not committed)
cd ios/GymCoach
xcodegen generate
open GymCoach.xcodeproj
```

In Xcode:

1. Select the **GymCoach** scheme.
2. **Signing & Capabilities** → choose your Team (bundle id `tech.fekitech.gymcoach`
   — change it to your own reverse-DNS id if you fork).
3. Plug in an iPhone (or pick a simulator — note the simulator has no camera,
   so the workout screen stays black there; History/UI still work).
4. ⌘R.

Phone placement is the same as the desktop app: ~2–3 m away, whole body in
frame, exercise-specific angles listed on each card on the home screen.

## 3. App architecture

```
CameraService (AVCaptureSession 720p)
      │ CVPixelBuffer, portrait
      ▼
VNDetectHumanBodyPoseRequest        ← Apple Vision, Neural Engine
      │ 15 joints, bottom-left origin
      ▼ y → 1 − y                   ← convert to top-left like the Python app
CoachCore.SessionEngine
      ├─ SkeletonSmoother (One Euro + visibility hold)
      ├─ AutoDetector ("Auto" mode) / RepCounter FSM / PlankTracker
      ├─ live + per-rep form rules → scores, faults
      ├─ FatigueMonitor (velocity loss)
      └─ SessionBuilder → workout_log.json (Documents/)
      ▼
SwiftUI: skeleton overlay · HUD · cue banner   +   AVSpeechSynthesizer voice
```

`workout_log.json` uses the **same schema** as the desktop prototype, so you
can copy it off the phone (Files app → GymCoach) and run
`python pose_coach.py --stats` on it.

## 4. TestFlight & App Store submission

1. **App Store Connect** → *My Apps* → **＋ New App*** — name “AI Gym Coach”
   (or your own), bundle id `tech.fekitech.gymcoach`, SKU anything.
2. In Xcode: **Product → Archive** (destination *Any iOS Device*), then
   **Distribute App → App Store Connect → Upload**.
3. Fill the listing: description, keywords, screenshots
   (6.7″ and 6.5″ iPhone sizes are mandatory; run on a simulator and ⌘S).
4. **App Privacy** questionnaire → **Data Not Collected**
   (all processing is on-device; the workout log never leaves the phone —
   `PrivacyInfo.xcprivacy` in the repo declares the same).
5. Export compliance: already answered by `ITSAppUsesNonExemptEncryption = false`
   in the Info.plist — no yearly encryption paperwork.
6. Camera permission text is preset
   (“analyzes your exercise form … never stored or uploaded”). Apple reviewers
   check that the app remains usable in its core flow after denying optional
   permissions — camera is core here, so a denial simply shows a black
   preview; that is acceptable for a camera-centric fitness app.
7. Add **Review Notes**: “Point the camera at a person doing squats; the app
   counts reps and gives form feedback. No account needed.” Reviewers love
   apps they can test in 30 seconds.
8. Submit → typical review time is 24–48 h. Use **TestFlight** (internal
   testers, no review) to dogfood first.

## 5. Suggested next steps

- **fastlane** (`fastlane gym` + `fastlane pilot`) to archive/upload from CI —
  needs an App Store Connect API key stored as a GitHub secret.
- App icon variants, localized listings, and a landscape iPad layout.
- ARKit 3D body tracking (`ARBodyTrackingConfiguration`) for depth-aware
  joint angles on LiDAR devices.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `xcodegen: command not found` | `brew install xcodegen` |
| “Signing for GymCoach requires a development team” | Xcode → target → Signing & Capabilities → pick your Apple ID team |
| Black camera screen on simulator | Expected — simulators have no camera; use a device |
| No skeleton overlay | Ensure the whole body is visible and well-lit; Vision needs ~full-body framing |
| Voice cues silent | Check the mute switch; the app ducks (not stops) background music |
