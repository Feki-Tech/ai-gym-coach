# From gym coach to sport coach — evidence review & extension design

The fusion architecture (docs/SENSORS.md) was built for one room: sources →
hub → per-concern fusers → coach. This doc widens the lens twice — from the
gym to **sport in general**, and from performance to the edge of **medical
assistance** — and grounds every proposed feature in current research. The
rule for inclusion: a feature ships only if (a) the evidence supports it,
(b) it stays on the right side of the wellness/medical-device line, and
(c) it maps onto the existing architecture without a rewrite.

Convention: statements about evidence cite the literature inline; each
section ends with what it means for *this* system.

## 1. What the athlete wears — beyond watch and phone

**Chest straps stay the reference.** Validation work keeps finding that
wrist PPG degrades exactly when a coach needs it most — high and variable
intensities — and that accuracy varies with anatomical placement and even
[skin tone](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0318724);
arm- and chest-worn sensors track better across intensities
([JMIR Cardio 2025](https://cardio.jmir.org/2025/1/e67110/),
[Sensors 2026](https://www.mdpi.com/1424-8220/26/1/176)). Nocturnal resting
HR/HRV from consumer wearables, by contrast, validates well
([Physiological Reports 2025](https://physoc.onlinelibrary.wiley.com/doi/10.14814/phy2.70527)),
and athlete-focused device comparisons exist
([Frontiers in Physiology 2025](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2025.1707318/full)).

**Smart garments are real now.** Textile-electrode shirts stream ECG-grade
HR, respiration and activity continuously; Hexoskin positions its shirts as
[clinically validated research instruments](https://hexoskin.com/pages/health-research),
and the category has reached
[FDA clearance for real-time cardiopulmonary monitoring](https://cardiovascularbusiness.com/topics/cardiac-imaging/electrocardiography-ecg/fda-clears-new-smart-shirt-real-time-cardiopulmonary-monitoring).
The e-textiles literature is equally clear about the catch: washing
durability, electrode drift and per-garment calibration remain open
problems ([Sensors 2024 review](https://www.mdpi.com/1424-8220/24/4/1058)).

**Foot is a first-class sensor location.** For running, foot/lace-mounted
IMUs with refined gait-event detection outperform generic placements
([PMC 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12835337/)), sensor
count/placement can be optimized systematically
([Frontiers in Bioengineering 2026](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2026.1762919/full)),
and smartwatch-derived gait metrics are **not interchangeable** with
lace-mounted IMUs ([PubMed 2025](https://pubmed.ncbi.nlm.nih.gov/40942989/)).

**Design consequence — sensor tiers.** The hub gains a per-source
`quality` tag (reference / validated-wearable / consumer-estimate). Fusers
already answer "unknown" when data is absent; they additionally weight by
tier, and the coach says which tier a number came from when it matters
("wrist HR during intervals is rough — a strap would make this reliable").
That is a one-field extension of `Sample`/`SensorSource`, not a redesign.

## 2. Beyond the gym: running first

Running is the obvious second sport: biggest population, best-researched
metrics, and our exact stack — one camera plus IMUs — is already validated
there: markerless running gait assessment from a **single smartphone
camera** ([Sensors 2023](https://doi.org/10.3390/s23020696)), spatiotemporal
gait parameters from a phone **in the pocket**
([Sensors 2025](https://www.mdpi.com/1424-8220/25/14/4395)).

A `RunningSession` module mirrors the gym pipeline: pose/IMU → cadence,
ground-contact time, vertical oscillation → form cues from the same
rate-limited FeedbackEngine ("cadence 158 — try quicker, shorter steps"),
HR zones from the same EffortModel, session records into the same log. The
best-evidenced intervention is also the simplest: modest cadence increases
reduce joint loading — a spoken metronome cue, no new hardware. Cycling
(power meters speak BLE too) and jump-based sports follow the same
template later; swimming is explicitly out (camera and BLE both break at
the pool).

## 3. Training intelligence the evidence actually supports

- **HRV-guided training** — meta-analyses find HRV-guided prescription
  performs as well as or better than predefined plans for aerobic
  adaptations ([Vesterinen et al., meta-analysis](https://pubmed.ncbi.nlm.nih.gov/34639599/),
  [Appl. Sci. meta-analysis](https://www.mdpi.com/2076-3417/10/23/8532),
  [JSAMS review](https://www.sciencedirect.com/science/article/pii/S1440244021001080)),
  including in [professional runners](https://www.sciencedirect.com/science/article/abs/pii/S0031938421003413)
  and [technology-guided sedentary adults](https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2025.1578478/full).
  → `ReadinessModel` fuser: morning/nocturnal HRV baseline (validated
  wearable import or strap measurement) → a daily readiness state the LLM
  coach uses to *modulate* the day's plan ("HRV well below your baseline —
  today becomes the easy session").
- **Velocity-based training** — meta-analyses show VBT matches or beats
  %1RM-based loading for strength/power, typically at lower accumulated
  fatigue ([BMC 2025](https://link.springer.com/article/10.1186/s13102-025-01504-9),
  [J Sports Sci 2022](https://www.tandfonline.com/doi/full/10.1080/02640414.2022.2059320),
  [PLOS One](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0259790));
  autoregulated loading in general is supported by a
  [2025 network meta-analysis](https://www.sciencedirect.com/science/article/pii/S1728869X25000590).
  → confirms SENSORS.md phase v2 (IMU bar velocity) as the highest-value
  sensor investment; the app already measures velocity loss — VBT closes
  the loop by *prescribing* from it.
- **Load management, honestly** — the acute:chronic workload ratio is
  popular and deeply contested: conceptual-pitfall critiques
  ([Impellizzeri et al.](https://www.researchgate.net/publication/341936245_AcuteChronic_Workload_Ratio_Conceptual_Issues_and_Fundamental_Pitfalls))
  and a [2025 meta-analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC12487117/)
  find little reliable injury-predictive power, in line with broader
  wearable-injury reviews that report promise for *monitoring* but not
  *prediction* ([MDPI scoping review](https://www.mdpi.com/1424-8220/22/9/3225)).
  → `LoadTracker` shows volume, monotony and strain trends and flags
  *spikes* as "worth a conversation" — the coach never claims to predict
  injuries, because the evidence says it can't.

## 4. Medical assistance — and the line we don't cross

The regulatory boundary is well defined and recently re-clarified: software
making healthy-lifestyle / general-fitness claims falls under the FDA's
**general wellness** policy (updated guidance,
[Jan 2026](https://www.ropesgray.com/en/insights/alerts/2026/01/fda-adapts-with-the-times-on-digital-health-updated-guidances-on-general-wellness-products),
[analysis](https://www.faegredrinker.com/en/insights/publications/2026/1/key-updates-in-fdas-2026-general-wellness-and-clinical-decision-support-software-guidance),
[overview](https://wirelesslifesciences.org/2026/01/general-wellness-vs-medical-device-in-2026-fda-rules-for-wearables-apps-and-claims/));
the same logic governs intended purpose under the
[EU MDR](https://meddeviceguide.com/blog/mobile-medical-applications-regulatory-guide).
Diagnosis, treatment or disease-management claims make an app a medical
device. Therefore, structurally:

**Inside the line (we build):**

- *Education and referral.* The persona's safety rules already stop the
  set on red-flag symptoms and refer out; that stays the ceiling for
  symptom handling. An expanded, sourced red-flag catalog (chest pain,
  radiating pain, dizziness, numbness…) makes referral *better*, not more
  diagnostic.
- *Recovery & readiness tracking* (HRV, resting HR trends, load) with
  wellness framing — "your recovery looks poor, train easy today", never
  "you may have X".
- *Sensor-artifact honesty.* "That HR reading looks like a sensor artifact"
  is fine; naming an arrhythmia is a diagnosis and is out — permanently,
  regardless of how good the signal gets.
- *Exercise-as-prevention content* for healthy users, which the wellness
  policy explicitly covers.

**On the line (needs review before any release):** a **physio companion
mode**. The evidence base is genuinely strong — telerehabilitation and
app-delivered exercise therapy show effectiveness for musculoskeletal
conditions ([JOSPT network meta-analysis 2025](https://www.jospt.org/doi/10.2519/jospt.2025.13366),
[telehealth exercise meta-analysis](https://pubmed.ncbi.nlm.nih.gov/35715175/),
[low-back-pain app meta-analysis](https://www.sciencedirect.com/science/article/pii/S0003999325009074),
[safety review](https://rehab.jmir.org/2025/1/e68681)), and MediaPipe-class
pose estimation has been assessed specifically for physiotherapy exercises
([accuracy study](https://www.sciencedirect.com/science/article/pii/S1877050924033660),
[depth-camera systematic review](https://pmc.ncbi.nlm.nih.gov/articles/PMC11902703/),
[CV-assessment review](https://pmc.ncbi.nlm.nih.gov/articles/PMC12158133/)).
The only defensible shape: the app **executes a plan a clinician
prescribed** (reps, ROM limits, adherence log the patient can share), the
clinician remains the medical authority, and claims stay at adherence
support. Anything beyond that shape is a medical device and out of scope
for this repo.

**Outside the line (we don't build):** diagnosis of any kind, arrhythmia
or disease detection, treatment recommendations, pain-condition management
claims, medication anything.

## 5. Mapping onto the architecture

Everything above lands as fusers and sources — the SENSORS.md layering
holds:

```
new sources                 new fusers                   consumers (existing)
smart shirt (BLE HRS+resp)  ReadinessModel (HRV baseline) → coach modulates plan
foot/lace IMU               GaitAnalyzer (cadence, GCT)   → FeedbackEngine cues
nightly HRV import (file)   LoadTracker (volume/monotony) → dashboard, debriefs
power meter (BLE CPS)       tier-weighted fusion          → honest uncertainty
```

Plus two cross-cutting changes: the `quality` tier tag on `Sample`
(section 1), and an activity abstraction so a session can be sets-and-reps
*or* intervals/steady-state — the workout log grows a `type` field, the
dashboard grows per-type views.

## 6. Phasing (extends SENSORS.md §4)

| Phase | Scope | Evidence gate |
|---|---|---|
| S1 | sensor tiers + nightly HRV import + `ReadinessModel`; coach modulates the day | HRV-guided training meta-analyses (§3) |
| S2 | `RunningSession`: phone-camera + pocket-IMU gait, cadence cues, HR zones | smartphone gait validation (§2) |
| S3 | VBT prescription from bar-IMU velocity (SENSORS.md v2 hardware) | VBT meta-analyses (§3) |
| S4 | `LoadTracker` trends (explicitly non-predictive) | §3 load-management honesty |
| S5 | physio companion mode — only with clinical/regulatory review | §4 telerehab evidence + wellness boundary |

Research backlog to re-check before each phase: e-textile drift/washing
progress, wrist-PPG accuracy at intensity, ACWR successor metrics, updated
FDA/MDR guidance.

## 7. Condition-aware and adaptive coaching

A coach that only works for healthy, standing, able-bodied users is half a
coach. Three feature classes extend the system to people with
Einschränkungen — each with its own evidence base, and each judged against
the §4 boundary. The pattern is always the same: a **condition profile**
(athlete profile, SQLite, local) changes *exercise selection, cues and
context* — never diagnosis, never treatment.

### 7.1 Metabolic context: diabetes and CGM

Exercise moves glucose in direction-dependent ways — sustained aerobic work
tends to drop it, short anaerobic/high-intensity work can raise it, and
hypoglycemia risk extends hours after training (including overnight) — all
codified in the
[Lancet consensus on exercise management in type 1 diabetes](https://www.thelancet.com/journals/landia/article/PIIS2213-8587(17)30014-1/abstract)
and updated through
[technology-focused reviews](https://link.springer.com/article/10.1007/s00125-024-06229-x)
and [CGM-and-exercise literature](https://pmc.ncbi.nlm.nih.gov/articles/PMC6165159/);
[interpretation targets for CGM data are internationally standardized](https://diabetesjournals.org/care/article/42/8/1593/36184/Clinical-Targets-for-Continuous-Glucose-Monitoring)
(time-in-range, ADA
[Standards of Care 2025](https://diabetesjournals.org/care/article/48/Supplement_1/S128/157561/6-Glycemic-Goals-and-Hypoglycemia-Standards-of)).
[Exercise-advisor apps for T1D are an active research area](https://pmc.ncbi.nlm.nih.gov/articles/PMC9158247/).

**Data path, local-first:** vendor CGM APIs are cloud-bound, but the
open-source diabetes ecosystem (xDrip+, Nightscout) exposes readings
locally — a natural `glucose` kind for the SensorHub (UDP/replay first,
tier: consumer-estimate; the 5–15 min interstitial lag is documented and
must be surfaced, not hidden).

**Inside the line:** show the athlete's own CGM value + trend arrow on the
HUD and in the coach's live context; per-session glucose journaling ("your
glucose fell 40 mg/dL during that aerobic block — a pattern worth
discussing with your care team"); consensus-sourced *education* framed as
questions for their clinician. **On the line, review first:** anything
*reactive* — "stop, glucose is falling" — is acute-care territory; hypo
*alarms* stay the regulated CGM app's job, permanently. **Out:** insulin or
carb dosing advice of any kind.

### 7.2 Condition-adapted programming (same engine, different rules)

- **Hypertension** — the counterintuitive, well-evidenced result: isometric
  training (wall sits, planks — exercises the app already coaches) is among
  the *most* effective modes for lowering resting BP
  ([large network meta-analysis](https://pubmed.ncbi.nlm.nih.gov/37491419/),
  [IRT meta-review 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12357484/),
  [RCT meta-analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC12989405/)).
  A BP-aware mode reweights the catalog toward evidenced holds, adds
  breathing cues (no breath-holding/Valsalva during holds), and journals
  home BP readings as context.
- **Cardiac maintenance** — home-based cardiac telerehab with wearable
  ECG/HR is effective
  ([Lancet Digital Health 2025 meta-analysis](https://www.thelancet.com/journals/landig/article/PIIS2589-7500(25)00012-3/fulltext),
  [wearable-HR telerehab meta-analysis](https://pubmed.ncbi.nlm.nih.gov/38838406/),
  [HF telerehab review](https://pmc.ncbi.nlm.nih.gov/articles/PMC12842665/)).
  Shape: like the physio companion — **clinician-set** HR ceilings become
  hard cue thresholds the EffortModel enforces ("above your prescribed
  zone — ease off"). Clinician-prescribed only; never self-serve.
- **Older adults / fall prevention** — strength-and-balance work is the
  canonical intervention, and single-camera assessment of exactly our kind
  is validated:
  [chair-based exercise quality from joint angles](https://doi.org/10.3390/s25133907),
  [vision-based sit-to-stand balance assessment](https://www.sciencedirect.com/science/article/abs/pii/S0966636225000013),
  [CV gait features for fall-risk](https://www.mdpi.com/2076-3417/14/9/3867),
  [postural-control screening (JMIR Aging 2025)](https://aging.jmir.org/2025/1/e73290).
  A "gentle mode": chair-based catalog, the 30-second sit-to-stand as a
  tracked functional metric, slower cue cadence, voice-first interaction.

### 7.3 Body-adapted perception: wheelchair and seated athletes

The uncomfortable research finding: mainstream pose estimators underperform
on wheelchair users — enough that dedicated work exists on
[synthetic-data techniques to fix the bias (WheelPose)](https://www.researchgate.net/publication/380523240_WheelPose_Data_Synthesis_Techniques_to_Improve_Pose_Estimation_Performance_on_Wheelchair_Users)
and on [sparse-IMU alternatives (WheelPoser)](https://arxiv.org/html/2409.08494v1).
Two consequences for us: a **seated exercise catalog** (curls, presses,
seated raises — the ExerciseSpec/FSM machinery works unchanged on
upper-body signals; established catalogs exist, e.g.
[NCHPAD](https://www.nchpad.org/resources/upper-body-workout-for-wheelchair-users-seated-strength-training/)),
with detection and form rules that never assume visible hips/knees; and a
second argument for the SENSORS.md IMU path — where camera models are
biased, worn sensors aren't. The synthetic-data fix mirrors this repo's own
`synth_frames` approach; generating seated variants is the same trick.

Accessibility of the *interaction* rides along for free: the coach is
already voice-first, hands-free and localized (iOS) — eyes-free and
low-vision use is a supported path, not an afterthought.

### Phasing addendum

| Phase | Scope | Evidence gate / boundary |
|---|---|---|
| S6 | condition profile + CGM display & glucose journal (xDrip-style local source) | §7.1 — display + education only; anything reactive needs review |
| S7 | adaptive catalogs: gentle mode (chair + sit-to-stand metric), seated/wheelchair mode, BP-aware holds with breathing cues | §7.2/§7.3 validation studies |
| S8 | clinician-prescribed guarded modes (cardiac HR ceilings, physio plans) | §4 + §7.2 — regulatory review before any release |
