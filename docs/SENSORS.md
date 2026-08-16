# Sensor fusion — beyond the camera

The camera is a remarkable single sensor: pose, reps, tempo, form faults.
It is also blind in exactly the ways that matter to a coach. It cannot see
**effort** (a 10-rep set at RPE 6 and one at RPE 9.5 look identical in
joint angles), **recovery** (rest timers are dumb countdowns today),
**load** (bodyweight squat vs. 100 kg on the bar), and it fails under
**occlusion** (rack uprights, bad framing, a plate blocking the hip). Every
one of those gaps is a sensor somebody already owns.

This doc is the design for closing them: which sensors, what each one adds,
the architecture that fuses them, and the shipped PoC (`coach_sensors.py`).
Where this goes beyond the gym — running, smart garments, readiness, and
the medical-assistance boundary — is the evidence review in
[docs/RESEARCH.md](RESEARCH.md).
The local-first stance (docs/INFRA.md §2) is non-negotiable throughout:
sensor streams are processed in memory on the machine, summarized into the
local workout log, and never uploaded.

## 1. Sensor survey

### Committed (this design targets them)

| Sensor | Adds | Transport | Hardware, cost |
|---|---|---|---|
| **Heart rate** | effort zones, per-set strain, HR-recovery-driven rest, session readiness | BLE GATT Heart Rate Service (0x180D/0x2A37) — an open standard | any chest strap or sports watch in broadcast mode (Polar, Garmin, Wahoo, …), ~€30–80, most athletes already own one |
| **IMU** (wrist, phone, or bar-mounted) | concentric velocity that vision can't match (velocity-based training), rep detection when the camera can't see, bar-path later | phone's own sensors (Android app!), UDP stream from sensor apps, or BLE pucks (Movella DOT, WitMotion WT901) | €0 (phone) to ~€60 (puck) |

Heart rate is first: the standard is genuinely universal, the fusion is
simple, and it upgrades an existing weak feature (the rest timer) into a
physiological one. IMU is second: highest ceiling (VBT is *the*
strength-training metric vision approximates worst), but placement,
calibration and per-exercise signatures make it a longer road.

### Considered and parked (and why)

- **Force insoles / plates** — ground-truth load + balance, but plates are
  €500+ lab gear and insole APIs are closed. Revisit if a maker option with
  an open protocol appears.
- **EMG sleeves** — muscle activation would be gold for "is this actually
  hitting your glutes", but consumer EMG is niche, noisy and dead (Myo).
- **Microphone** — breathing cadence and bar clank are real signals, and the
  mic already exists for the voice coach; parked because the voice pipeline
  owns the mic and sharing it adds complexity before the simpler sensors pay
  off.
- **UWB / radar presence** — nothing pose doesn't already give here.
- **Barbell collar "smart" sensors** — proprietary apps, no open streams;
  a bar-strapped IMU puck does the same job openly.

## 2. Architecture

One rule shapes everything: **every sensor is optional, and absence must
degrade to exactly today's behaviour.** No sensor, no change. That forces
the layering below — fusion consumers read from the hub through interfaces
that answer "unknown" gracefully.

```
 sources (one thread each)          hub                    fusion               consumers
┌──────────────────────┐   ┌──────────────────┐   ┌──────────────────────┐   ┌─────────────────┐
│ BleHeartRate (bleak) │   │                  │   │ EffortModel          │   │ HUD (hr, zone)  │
│ UdpJsonSource (phone │──▶│  SensorHub       │──▶│  zones, per-set peak │──▶│ live_state →    │
│   IMU/HR apps)       │   │  per-kind ring   │   │ RestAdvisor          │   │   LLM coach     │
│ ReplaySource (JSONL) │   │  buffers, time-  │   │  HR-recovery rest    │   │ RepCounter      │
│ SimulatedSession     │   │  stamped Samples │   │ VelocityFuser        │   │   (occlusion    │
│   (deterministic)    │   │  window()/latest │   │  vision ⊕ IMU        │   │    fallback)    │
└──────────────────────┘   └──────────────────┘   └──────────────────────┘   │ workout log     │
                                                                             └─────────────────┘
```

**Samples and time.** Everything is a `Sample(t, kind, value)` with `t`
from the host's monotonic clock, stamped on arrival. At our rates (HR ~1 Hz,
IMU ≤100 Hz, pose 30 Hz) arrival-time alignment is sufficient; per-source
clock offset estimation is deliberately out of scope until a fuser
demonstrably needs it (YAGNI — the rest advisor tolerates seconds of skew,
velocity fusion works on per-rep windows, not per-frame sync).

**The hub is dumb on purpose.** Ring buffers per kind, time-windowed reads,
thread-safe, no interpretation. All intelligence lives in fusers, all
fusers are independent, and each answers `None`/no-op when its inputs are
missing — that's the degradation rule enforced structurally.

**Fusers (per concern, not per sensor):**

- `EffortModel` — HR → zone (1–5 of HRmax), rolling peak, per-set strain.
  Config: `hr_max`/`hr_rest` from the athlete profile when known, else
  age-formula default with an explicit "estimate" tag the coach can mention.
- `RestAdvisor` — replaces the dumb countdown: after a set, "ready" when HR
  drops below `hr_rest + 0.35 × (set_peak − hr_rest)` (heart-rate-reserve
  recovery), with a hard cap so a noisy strap never blocks training.
- `VelocityFuser` — per rep: vision gives ROM °/s, an IMU window gives peak
  acceleration/velocity. Fusion is complementary, not averaging: IMU wins on
  speed precision, vision wins on segmentation and ROM; disagreement is
  itself a signal (loose strap, occlusion) and is surfaced, not hidden.
  When pose visibility collapses mid-set, IMU rep ticks keep the count.

**Consumers.** The HUD shows hr/zone; `live_state` carries them to the LLM
coach (which already relays live physics — now it can say "your heart rate
is still in zone 4, give it another 30 seconds"); the workout log summary
gains `avg_hr`/`peak_hr` so history and the dashboard can show strain; the
rep FSM later accepts IMU ticks as a second vote.

**Cross-platform mapping.** The same layering ports 1:1: Android —
`SensorManager` (built-in IMU!) + `BluetoothLeScanner` feeding the Kotlin
`core/` engine; iOS — CoreMotion + CoreBluetooth/HealthKit (Apple Watch HR)
feeding CoachCore. The fusers are pure logic, exactly like the rep FSM, so
they port the way CoachCore did.

## 3. The PoC (`coach_sensors.py`, shipped)

Hardware-free by design, same philosophy as `synth_frames`: the pipeline is
proven end-to-end on deterministic synthetic streams, and real hardware is
just another source.

- `SimulatedSession` — seeded HR + IMU generator (rest → set climb →
  exponential recovery; acceleration bursts per rep). Drives demo + tests.
- `ReplaySource` — JSONL replay (`{"t":…,"kind":"hr","value":…}`), the
  `--collect` idea applied to sensors.
- `UdpJsonSource` — stdlib UDP listener; any phone sensor-streaming app
  (HyperIMU, Sensor Stream, …) or a 5-line script becomes a live source.
- `BleHeartRate` — real straps via the GATT Heart Rate Service. Needs the
  optional extra (`pip install -r requirements-sensors.txt` → bleak); the
  0x2A37 payload parser is pure stdlib and unit-tested against spec
  vectors.
- `EffortModel`, `RestAdvisor`, `VelocityFuser` as above.

Try it:

```bash
python coach_sensors.py --demo              # simulated set + recovery, printed
python coach_sensors.py --selftest          # deterministic, CI-run
python pose_coach.py --exercise squat --sensors sim         # full app, fake HR
python pose_coach.py --exercise squat --sensors udp:9999    # phone streams JSON
python pose_coach.py --exercise squat --sensors ble         # real chest strap
```

With `--sensors` active: HR + zone on the HUD, `heart_rate`/`hr_zone`/
`hr_peak` in the coach's live context, `avg_hr`/`peak_hr` in the session
log, and after a rep-goal set the rest advisor speaks when your heart rate
has actually recovered instead of when a countdown guesses.

## 4. Phases

| Phase | Scope | Exit criterion |
|---|---|---|
| **PoC (this PR)** | hub + sources + HR fusers, sim/UDP/replay/BLE, log + HUD + coach context, selftests in CI | a set with `--sensors sim` shows zones, logs HR, advises rest |
| v1 | profile-driven HRmax/HRrest, dashboard strain charts, per-set strain in set debriefs | coach debriefs mention effort, dashboard plots HR |
| v2 | IMU velocity fusion + occlusion-proof rep counting; Android uses the phone's own IMU | reps counted with lens covered mid-set; VBT number per rep |
| v3 | bar-mounted puck bar-path, watch companion | bar-path overlay |

## 5. Testing

Deterministic simulations (seeded), spec-vector tests for the GATT parser
(uint8/uint16 flag variants), fusion tests on synthetic recovery curves
(`RestAdvisor` fires between the right samples), threading-free fuser cores
so everything runs in CI like tests 1–15 of the coach — no hardware, no
Bluetooth stack, no sleep-dependent flakiness.
