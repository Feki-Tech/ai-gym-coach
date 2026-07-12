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

**Languages:** the app — including the spoken coaching cues — ships in the
5 most-spoken languages in the world: English (`en`), Simplified Chinese
(`zh-Hans`), Hindi (`hi`), Spanish (`es`) and French (`fr`). It follows the
iPhone's system language automatically; the voice coach picks a matching
`AVSpeechSynthesisVoice`. Engine strings live in
`ios/CoachCore/Sources/CoachCore/Resources/<lang>.lproj/Localizable.strings`,
app strings in `ios/GymCoach/Resources/<lang>.lproj/` (plus
`InfoPlist.strings` for the localized app name and camera-permission text).
To add a language: copy the two `en.lproj` folders, translate, and add the
code to `CFBundleLocalizations` in `ios/GymCoach/project.yml`.

---

## 1. Requirements

- A Mac with **Xcode 15+** (iOS 17 SDK; the app targets iOS 16+)
- [XcodeGen](https://github.com/yonaskolb/XcodeGen) — `brew install xcodegen`
- For device runs & App Store: an [Apple Developer](https://developer.apple.com/programs/) account ($99/year)

> No Mac? GitHub Actions builds the app on every push (see `.github/workflows/ci.yml`,
> job `ios`) — it runs the CoachCore tests and compiles the app for the iOS
> Simulator, so the code is always verified even when developing from
> Windows/Linux. And the **TestFlight workflow** signs and uploads the app to
> your iPhone entirely from CI — see §5. A Mac is never required.

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

## 5. TestFlight from CI — no Mac needed

The repo ships a manual workflow (`.github/workflows/testflight.yml`) that
builds, signs (Apple cloud-managed signing) and uploads the app to TestFlight
entirely on GitHub's macOS runners. One-time setup:

1. **Enroll** in the [Apple Developer Program](https://developer.apple.com/programs/enroll/)
   ($99/year; approval is usually instant–48 h). You can do this from any
   browser or from the iPhone itself.
2. **Create the app record**: [App Store Connect](https://appstoreconnect.apple.com)
   → *My Apps* → **＋ New App** — platform iOS, name e.g. “AI Gym Coach”,
   bundle ID **`tech.fekitech.gymcoach`** (register it when prompted; must
   match `project.yml`), SKU anything.
3. **Create an API key**: App Store Connect → *Users and Access* →
   [*Integrations → App Store Connect API*](https://appstoreconnect.apple.com/access/integrations/api)
   → **＋** — role **App Manager**. Download the `.p8` file (one chance!),
   note the **Key ID** and **Issuer ID** shown on that page.
4. **Find your Team ID**: [developer.apple.com/account](https://developer.apple.com/account)
   → Membership details → 10-character Team ID.
5. **Add 4 repository secrets** (GitHub → repo → Settings → Secrets and
   variables → Actions → New repository secret):

   | Secret | Value |
   |---|---|
   | `APPLE_TEAM_ID` | 10-char Team ID, e.g. `AB12CD34EF` |
   | `ASC_KEY_ID` | API Key ID, e.g. `2X9R4HXF34` |
   | `ASC_ISSUER_ID` | Issuer ID (UUID) |
   | `ASC_PRIVATE_KEY` | full text of the `.p8` file, including the BEGIN/END lines |

6. **Run it**: repo → Actions → **TestFlight** → *Run workflow* (~10 min).
7. **Install on the iPhone**: App Store Connect → your app → TestFlight →
   add yourself under *Internal Testing* (once) → install the
   [TestFlight app](https://apps.apple.com/app/testflight/id899247664) on the
   phone → the build appears there after ~5–15 min of processing.

Every later run uploads a new build (build number = CI run number) and
TestFlight notifies your phone. Internal-tester builds need **no App Review**.

## 6. Suggested next steps

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
