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

**Languages:** the app — including the spoken coaching cues — ships in
6 languages: English (`en`), Simplified Chinese (`zh-Hans`), Hindi (`hi`),
Spanish (`es`), French (`fr`) and Arabic (`ar`). It follows the iPhone's
system language automatically; the voice coach picks a matching
`AVSpeechSynthesisVoice`, and SwiftUI mirrors the layout right-to-left for
Arabic. Engine strings live in
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
      │    └─ MLDetector: the desktop's gated TinyMLP via
      │       `TinyMLP.load(url:)` + `SessionEngine(exercise:model:)` —
      │       export with `pose_coach.py --export-model classifier.json`,
      │       ship the file in the app's Documents; rules stay the fallback
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
   (all processing is on-device; the workout log and the Apple Health data
   the app reads never leave the phone — `PrivacyInfo.xcprivacy` in the
   repo declares the same). Because the app uses HealthKit, App Review
   additionally requires a **privacy policy URL** in the listing (a short
   page stating that health data is read on-device only and never
   transmitted) and checks that the Health permission texts match what the
   app does — they are preset in `project.yml` and localized in
   `InfoPlist.strings`.
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

## 6. Apple Health & the Fitness app

The app integrates with Apple Health (HealthKit) — `Sources/HealthService.swift`,
settings screen `Sources/HealthView.swift` (♥ button on the home screen).
Nothing leaves the phone: the app has no server, and HealthKit data is never
written anywhere except the on-device log.

**What it writes** — every finished set is saved as an `HKWorkout` of type
*Traditional Strength Training* (indoor), with the coach's numbers in the
workout metadata (`tech.fekitech.gymcoach.exercise`, `.reps`, `.avg_score`,
`.hold_s`, `.faults`). It shows up in **Fitness → Summary** (and counts toward
the Move/Exercise rings when heart-rate or energy data from a Watch accompany
it) and in **Health → Activity → Workouts**. The summary sheet links straight
into both apps via Apple's URL schemes (`fitnessapp://`, `x-apple-health://`).

**What it reads, and what the coach does with it:**

| Health data (HealthKit type) | Why the coach wants it |
|---|---|
| Heart rate (`heartRate`) — live during a set | zone pill on the HUD (Z1–Z5, same bands as the desktop `EffortModel`), `avg_hr` / `peak_hr` in the session log — the keys the desktop sensor fusion writes, so the web dashboard reads both |
| Resting heart rate, HRV SDNN (`restingHeartRate`, `heartRateVariabilitySDNN`) | readiness / recovery: elevated resting HR or depressed HRV vs. baseline → lighter session, longer rests |
| VO₂ max (`vo2Max`) | conditioning level; conditioning-appropriate rest advice |
| Body mass, height (`bodyMass`, `height`) | protein target (1.6–2.2 g/kg), relative-strength context for loads and e1RM |
| Date of birth, biological sex (characteristics) | estimated max HR (220 − age) for the zones; the coach says it is an estimate |
| Sleep (`sleepAnalysis`, asleep stages, last 24 h) | recovery: short sleep → expect lower scores, don't chase PRs |
| Steps, active energy, exercise minutes (today) | how much the athlete has already done today before this set |
| Workouts (last 7 days, any app) | training frequency and muscle rest across apps — the same muscle-recovery logic as the dashboard |

Live heart rate comes from Apple Watch (continuously while a Watch workout is
running, every few minutes otherwise) or from any chest strap / app that
writes to Health; without a source the HUD simply shows no pill.

Enable it in the app (♥ → *Connect Apple Health*); iOS shows the permission
sheet with the texts from `NSHealthShareUsageDescription` /
`NSHealthUpdateUsageDescription`. HealthKit never reveals *read* denials, so
a denied item just reads as "—". Permissions can be changed later in
**Health → Sharing → Apps → AI Gym Coach**. The XcodeGen spec adds the
`com.apple.developer.healthkit` entitlement; with automatic signing Xcode
turns the capability on for your team.

Simulator note: HealthKit works in the simulator but has no data — add
samples in the simulator's Health app or run on a device.

## 7. Sign in — Apple, Google, Microsoft

`Sources/AuthService.swift` + `Sources/AccountView.swift` (person icon on the
home screen). Same conventions as the desktop (`coach_auth.py`, docs/AUTH.md):
Authorization Code + PKCE S256, `state` + `nonce`, provider discovery, ID-token
claim checks, and the **system browser** through `ASWebAuthenticationSession`
— never an embedded web view. Identity (provider, subject, name, e-mail) lives
in the Keychain (`AfterFirstUnlockThisDeviceOnly`); no tokens are kept.

- **Sign in with Apple** is always offered and listed first — App Store
  Review Guideline 4.8 requires it whenever a third-party login is present.
  Entitlement `com.apple.developer.applesignin` is in the XcodeGen spec; enable
  the capability for your App ID in the developer portal.
- **Google**: create an *iOS* OAuth client (bundle id `tech.fekitech.gymcoach`);
  the redirect is the reversed client id (`com.googleusercontent.apps.<id>:/oauth2redirect`),
  derived at runtime. Pass the id at build time — it is not in the repo:
  `xcodebuild … COACH_GOOGLE_IOS_CLIENT_ID=<id>.apps.googleusercontent.com`
  (a `$(…)` build setting feeds `GoogleClientID` in Info.plist; in CI use a
  repository secret).
- **Microsoft**: app registration with an *iOS/macOS* platform, bundle id
  `tech.fekitech.gymcoach` → redirect `msauth.tech.fekitech.gymcoach://auth`
  (registered as a URL type). Pass `COACH_MICROSOFT_CLIENT_ID` (and
  `COACH_MICROSOFT_TENANT`, default `common`) the same way.

Empty ids simply hide those buttons, so the CI simulator build needs no
secrets.

## 8. The interactive coach on the phone

The LLM coach cannot run on the phone, and it should not lose anything by
moving: the persona, the retrieval over the knowledge base and the exercise
catalogue, the safety guardrails, the history/profile tools and the
behaviour evals all live in Python. So the phone **talks to the desktop
coach over your Wi‑Fi**:

```bash
# on the PC, in the ai-gym-coach folder (Ollama running, as for --coach)
python coach_server.py
#   Coach server on http://192.168.1.20:7799
#   Pairing code: 7F3A2C   ← enter both once in the app: home → 💬 → Coach settings
```

`scripts\start-coach.bat` does all of that with one double-click on
Windows (starts Ollama, pulls the model on first run, launches the server).

`coach_server.py` (standard library, selftested) exposes the same
`ChatCoach` the desktop uses: `/chat` and `/event` stream the answer
sentence by sentence (SSE) with the phone's **live session** attached to
every message — exercise, phase, reps, last rep's score/tempo/faults, joint
angle, heart rate, rest/goal/load/program state — plus `/log` (the finished
set lands in the desktop `workout_log.json`, so the web dashboard and the
desktop history include phone sessions), `/history`, `/brief`,
`/knowledge`, `/exercises`, `/profile`. Every request carries the pairing
code as a bearer token; it is meant for your own network (TLS proxy or VPN
beyond that). `docker compose up coach-server` runs it next to Ollama.

On the phone, once paired:

- **Talk** button (or the coach panel) opens the chat: type, or **hold the
  mic** and speak — on-device speech recognition, only the text leaves the
  phone. Answers stream in and are spoken sentence by sentence; a new
  question interrupts the old answer (barge-in).
- The coach **speaks up on its own**: greets you with last session's key
  point, debriefs every finished set (score trend, dominant fault, one cue)
  and wraps up the session.
- The coach **drives the workout** through the same ACTION protocol as the
  desktop: "switch me to push-ups", "let's do 12 reps", "give me 90
  seconds", "lower for 3 seconds", "stop correcting me", "I'm on 60 kilos",
  "plan me a leg workout and start it" → exercise switch, rep goal ring,
  rest overlay with countdown, tempo cue, cues muted, load logged (volume,
  estimated 1RM, records), guided program (sets counted, rests run,
  exercises switched, announcements spoken).
- The HUD grew to match the desktop: range-of-motion gauge against the
  exercise's *rep starts / full depth / lockout* lines, the faulty body part
  turns red on the skeleton, phase in athlete terms (LOWERING / BOTTOM /
  LIFTING), rep goal ring, load and record-to-beat, program strip, rest
  overlay, heart-rate zone pill, personal-record cues; buttons for rest,
  load and the coach.

- **Hands-free by default**: when a coach server is paired, the mic starts
  listening as the set starts (after the one-time microphone + speech
  permissions) — just speak, like on the desktop. A pill on the HUD shows
  *listening / hearing you… / coach talking*; the mic is gated while the
  coach thinks or talks so it never hears its own voice. Hold-to-talk in
  the chat sheet remains as a fallback, and everything can still be typed.
- **Pick your devices**: the 🔄 button in the workout screen switches
  between the phone's cameras (back, front — mirrored like a gym mirror —
  ultra-wide, telephoto) and, on iOS 17+, **external USB-C cameras**, so a
  webcam on a tripod can film the set while the phone stays in your hands;
  the choice persists. The microphone (built-in, AirPods / Bluetooth
  headset, wired or USB-C) is picked in Coach settings or from the chat
  sheet's mic menu.

Without a paired server the app still counts, scores and speaks cues on
its own — the Talk button just points you to the settings.

## 9. Testing on your own iPhone — no Mac, no paid account

Two free routes. Both use a **free Apple ID** (no $99 program): the app is
signed with a personal certificate, runs for **7 days**, then must be
refreshed (AltStore does that automatically over Wi‑Fi). Free accounts are
limited to 3 sideloaded apps and 10 app ids per week, and cannot use
paid-only capabilities — that is why the sideload build hides **Sign in
with Apple** (Google/Microsoft sign-in still work) and the **HealthKit**
capability (personal certificates are usually refused it, and a sideloader
fails the whole install when it can't provision an entitlement); the Health
screen then reports the missing entitlement and everything else works. The
IPA is ad-hoc signed so AltStore/Sideloadly can re-sign it.

### Route A — Windows PC (or Linux/macOS) + AltStore or Sideloadly

1. **Get the IPA.** Every push to `main`/`feat/**` touching `ios/` (or a
   manual *Run workflow*) runs `.github/workflows/ios-ipa.yml`, which builds
   an unsigned `GymCoach.ipa` on GitHub's macOS runner. Open the repo →
   **Actions → ios-ipa → latest run → Artifacts → `GymCoach-unsigned-ipa-…`**
   and unzip it to get `GymCoach.ipa`. (Client ids for Google/Microsoft
   sign-in come from the repository secrets `COACH_GOOGLE_IOS_CLIENT_ID` /
   `COACH_MICROSOFT_CLIENT_ID` — optional.)
2. **Prepare the iPhone** (iOS 16+): Settings → Privacy & Security →
   **Developer Mode** → on (appears after the first install attempt on some
   versions; the phone restarts). Connect the phone by USB and tap *Trust*.
3. **Install a sideloader on the PC:**
   - **AltStore** (altstore.io, "Classic"): install *iTunes* and *iCloud*
     **from Apple's website** (not the Microsoft Store versions), then
     AltServer. From the AltServer tray icon → *Install AltStore* → pick your
     iPhone → sign in with your Apple ID (an app-specific password works). On
     the phone: Settings → General → VPN & Device Management → trust your
     Apple ID's developer app. Then open **AltStore → My Apps → + → choose
     GymCoach.ipa**. AltServer refreshes it every week while the PC is on the
     same Wi‑Fi.
   - **Sideloadly** (sideloadly.io) is the one-shot alternative: drag the
     IPA in, enter your Apple ID, *Start*. Its *Advanced options* let you keep
     the bundle id and, on some versions, enable HealthKit.
4. **Open the app.** Camera permission prompt → point the phone at a
   person doing squats. Voice cues, history and the Health screen are all
   there.

Bundle id note: sideloaders usually append your team id to the bundle id
(`tech.fekitech.gymcoach.XXXXXXXXXX`); if you configure Microsoft sign-in,
register that id / redirect `msauth.<bundle id>://auth` in Entra.

### Route B — a Mac with Xcode (free personal team)

Xcode → Settings → Accounts → add your Apple ID → in the project's *Signing
& Capabilities* pick the **Personal Team**, change the bundle id to something
unique, remove the *Sign in with Apple* capability, plug the iPhone in and
press Run. Same 7-day limit; the app is refreshed by running it from Xcode
again.

### When you do get the $99 membership

Route A/B become unnecessary: `.github/workflows/testflight.yml` signs and
uploads to TestFlight from CI (§5), with all capabilities including Sign in
with Apple, 90-day builds and up to 10,000 testers.

## 10. Suggested next steps

- Read the Health snapshot into the LLM coach's context on desktop via the
  MCP bridge (export it from the phone, or sync the workout log).
- HKWorkoutSession on watchOS for continuous heart rate without the Watch
  Workout app.
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
