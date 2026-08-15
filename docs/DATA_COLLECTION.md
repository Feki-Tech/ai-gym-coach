# Real training data: recording protocol

The classifier bootstraps from synthetic motion (`synth_frames`), which is
what lets it exist without a dataset — and also its biggest accuracy lie:
until real windows arrive, the harness only proves the model can classify
*the generator*, not people. This doc is the working protocol for fixing that
with volume, and for growing the **committed real eval set** that makes the
harness judge real movement everywhere — including the `model-gate` CI run,
which has no local recordings.

Everything here uses existing plumbing: `--collect`, the positional
train/eval split (`_split_collected`), `--export-eval`, `--collect-report`.

## How collection works (code facts)

- `--collect` needs a fixed `--exercise` as the label; it appends roughly one
  window per second of active movement (2 s sliding window, after a 2 s
  warm-up) to a JSONL file. Name it `*.windows.jsonl` — that pattern is
  gitignored, so raw recordings never end up in git by accident.
- A row is `{"label": ..., "x": [38 floats]}` — aggregate statistics of joint
  angles and torso-normalized travel over the window. No images, no
  landmarks, no timestamps, no identity. This is why a small labeled set is
  safe to commit.
- Every 5th row (positions 0, 5, 10, …) is **reserved for the harness and
  never trains**. The split is positional: only ever APPEND to a collect
  file. Deleting or reordering rows reshuffles the split and leaks former
  eval rows into training.
- `--collect data/eval_windows.jsonl` is refused by the CLI and by
  `build_dataset` — the committed set judges models, it never trains.

## Recording setup

Whole body in frame, camera roughly hip height, 2–4 m away, stable (lean the
phone against something — no handheld). Live webcam works, but the highest
volume path is filming with a phone and running the video through afterwards:

```bash
# live (one class per run):
python pose_coach.py --exercise squat --collect me.windows.jsonl

# offline, from phone clips — collect an entire session in one sitting:
python pose_coach.py --exercise squat --video squats.mp4 \
    --headless --no-voice --collect me.windows.jsonl
```

Vary what the synthetic generator also varies — tempo, amplitude, standing
position/angle to the camera — plus what it can't: clothing, lighting,
room, your actual form on tired sets. For `lunge` and `curl` record both
left- and right-leading; for `plank` record both forearm and straight-arm
variants (the generator emits both). `bench` stays manual-only
(see `AutoDetector`) — don't record it.

## Volume plan

~1 window/second means 60–90 s of active movement per class per session
yields 60–90 windows. Three sessions on different days (different clothes,
lighting, tempo) per class:

| Milestone | Windows/class | What it unlocks |
|---|---|---|
| 1 session | ~60–90 | first honest `real_overall` numbers |
| 3 sessions | ~200 | training share meaningfully non-synthetic; commit 10–20/class to the eval set |
| sustained | 40+ *committed*/class | flip real bars on in the promotion gate (`promote_classifier` has the marker comment) |

Track progress with:

```bash
python pose_coach.py --collect-report --collect me.windows.jsonl
```

## The cycle

```bash
# 1) record (live or from clips), any number of sessions, appending:
python pose_coach.py --exercise curl --video curls.mp4 --headless \
    --no-voice --collect me.windows.jsonl

# 2) retrain through the gate — real windows join training, the harness
#    reports synthetic and real accuracy separately:
python pose_coach.py --train-classifier --collect me.windows.jsonl

# 3) grow the committed eval set from the RESERVED slice (deduped,
#    balanced, ≤N new per class), then review the diff and commit:
python pose_coach.py --export-eval --collect me.windows.jsonl --eval-per-class 10
git add data/eval_windows.jsonl && git diff --cached --stat
```

`real_overall` in the manifest (`classifier.manifest.json`) is the number to
watch. Expect it to start well below the synthetic accuracy — that gap *is*
the reason this protocol exists; it should close as collected volume grows.
It is deliberately not gated yet: with a tiny real set the gate would be
noise. The bar flips on once the committed set holds ~40+ windows per class.

## Why committing eval windows is OK (and logs are not)

The committed rows are the same 38 derived statistics the model consumes —
they cannot be inverted to video, poses, or a person, and they carry no
timing or session metadata. Workout logs, profiles, references and raw
collect files stay local (gitignored) per the local-first stance
(docs/INFRA.md §2). Only the curated eval sample, taken exclusively from the
already-reserved harness slice, is versioned — so the model gate everywhere
judges on real movement without anyone's training data leaving their machine.
