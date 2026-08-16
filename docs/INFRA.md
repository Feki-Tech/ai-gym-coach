# Infrastructure roadmap

This document transfers the infrastructure patterns proven in
[edgesense-ai](https://github.com/Feki-Tech/edgesense-ai) — dependency
locking, model lifecycle management, and a phased Azure deployment with
OIDC-authenticated CD — to this repo, adapted to what this app actually is.
Everything below current state is a **proposal**; nothing is implemented until
its phase lands, and each phase ships as its own PR.

Convention carried over from edgesense: statements about current behaviour
cite the code; everything else is marked *(proposed)*.

## 1. Current state (verified)

| Surface | State | Where |
|---|---|---|
| Dependencies | loose `pip` ranges, no lock (`mediapipe>=0.10`, …); optional voice extras in a second file | `requirements.txt`, `requirements-voice.txt` |
| Docker | single image, pip-installs from `requirements.txt`, bakes the pose model for offline use | `Dockerfile` |
| Compose | selftest / analyze / webcam / ollama / coach / coach-live / dashboard | `docker-compose.yml` |
| CI | matrix selftests (ubuntu+windows × py3.11/3.12), then containerized selftests; image pushed to **GHCR** on `main` | `.github/workflows/ci.yml` |
| CI gating | ✅ the docker job `needs: selftest` — deploy artifacts are already test-gated (edgesense's deploy workflow didn't have this; keep it) | `ci.yml` |
| iOS delivery | manual-dispatch TestFlight upload, secrets-based | `.github/workflows/testflight.yml` |
| ML model | `TinyMLP` exercise classifier: synthetic windows + optional collected real windows → bare `classifier.npz` | `pose_coach.py` (`train_classifier`, `TinyMLP.save`) |
| Model lifecycle | **none** — no version, no seed/data record in the artifact, validation accuracy printed but nothing gates; `train_classifier` unconditionally overwrites the model | `pose_coach.py:541-562` |
| Cloud | none (by design — see §2) | — |
| Observability | none beyond stdout; the dashboard is a local page | `coach_dashboard.py` |
| LLM coach lifecycle | ✅ prompt version + fingerprint, opt-in local JSONL trace (metrics only by default), deterministic reply graders, safety guardrail, 31-scenario eval set with a baseline gate (exit 0/1/2), manual `coach-eval.yml` running a real model on the runner — see [LLMOPS.md](LLMOPS.md) | `coach_ops.py`, `coach_eval.py`, `data/coach_evals.jsonl` |

## 2. What deliberately does NOT transfer

edgesense is a fleet product whose readings *belong* in a cloud pipeline. This
app is **local-first as a feature**: the athlete profile is "SQLite, never
uploaded" (README), video never leaves the machine, and the LLM coach runs on
a local Ollama. That stance constrains the transfer:

- **No user telemetry.** edgesense's Prometheus/Grafana loop watches machine
  data; here the equivalent would be watching *people*. Out.
- **No hosted LLM coach.** Ollama-in-the-cloud means paying for standing
  LLM compute; the coach stays local.
- **No cloud path for user data.** Anything deployed serves *bundled demo
  data only*. A hosted dashboard must never accept a real `workout_log.json`
  upload without an explicit, separate decision.
- **The iOS app keeps TestFlight.** Its delivery pipeline is already right
  for its platform; nothing Azure touches it.

## 3. Phase 1 — uv + lock-driven builds *(proposed)*

The edgesense move (its PR #16): `pyproject.toml` + committed `uv.lock` as the
single source of truth, extras per optional feature, images and CI installing
from the lock.

- Base deps = `requirements.txt` today; `voice` extra = `requirements-voice.txt`
  (incl. the `pycaw` win32 marker); `dev` group for anything CI-only.
- `Dockerfile` installs via `uv sync --frozen --no-dev` (or `uv export` →
  `pip install`) so the image is reproducible byte-for-byte from the lock.
- CI switches to `astral-sh/setup-uv` with cache — the matrix (both OSes, both
  Pythons) stays; the lock resolves platform markers per-OS.
- `requirements*.txt` kept briefly as generated exports for pip users, or
  removed with a README note — decided in the phase PR.

**Cost: none. Risk: low.** Watch out for: `mediapipe` wheel availability per
Python version is the usual sticking point — the lock pins what CI proved.

## 4. Phase 2 — classifier MLOps *(proposed)*

The classifier has the exact failure mode edgesense's phase 1 closed: a
retrain (`--train-classifier`, optionally blending user recordings from
`--collect`) can silently make auto-detection *worse*, and nothing records
what a given `classifier.npz` was trained on.

Transfer, scaled to a ~500-line numpy model — no MLflow, no registry, files
only:

1. **Manifest** — embed alongside the weights (same `.npz`, extra keys, and a
   `classifier.manifest.json` sidecar): schema version, model version
   (`{YYYYMMDD.HHMMSS}+{git7}` like edgesense), seed, epochs,
   samples-per-class, count + hash of collected real windows, class list,
   and the eval snapshot. `TinyMLP.load` stays backward compatible with
   legacy bare `.npz` files (version reports `unknown`).
2. **Fixed eval harness** — today's "validation accuracy" is a random split
   of the *same* synthetic generation that trained the model
   (`train_classifier`, `pose_coach.py:546-552`), so it can't detect
   distribution overfit. Add a held-out harness: windows generated from a
   **different fixed seed**, plus (when present) a reserved slice of
   collected real windows that training never sees. Per-class accuracy and
   confusion reported.
3. **Promotion gate** — challenger trains to a candidate path, both champion
   (existing `classifier.npz`) and challenger run the harness, and the
   challenger replaces the champion only if it clears an absolute bar AND
   doesn't regress any class beyond a tolerance. Refusal prints the diff
   table and leaves the champion untouched — same contract as edgesense's
   `ml/promote.py` (exit 0 promoted / 1 refused / 2 error).
4. **CI** — a manual-dispatch `model-gate` workflow runs the gate on CPU and
   uploads the candidate + manifest + report as artifacts; models are never
   auto-committed.

**Cost: none.** This is the highest-value phase: it protects a user-facing
behaviour (auto-detect) that users can degrade themselves via `--collect`.

## 5. Phase 3 — Azure demo deployment + OIDC CD *(proposed, cost-gated)*

The edgesense-azure pattern (Terraform, Container Apps, ACR, managed-identity
pulls, GitHub Actions OIDC), minus everything this app doesn't need. **What to
host:** the progress dashboard (`coach_dashboard.py`) with a bundled synthetic
`workout_log.json` — it's stdlib-only, tiny, already a compose service, and
per §2 it demos charts without touching anyone's data.

- `azure/infra/` Terraform: resource group, ACR (Basic), Container Apps env +
  Log Analytics, one user-assigned identity with AcrPull, one container app
  (external ingress, **scale-to-zero**, 0.25 vCPU / 0.5 Gi). No Key Vault, no
  AML, no Grafana — there is nothing here that warrants them yet.
- `azure-deploy.yml`: OIDC federated credential (no stored secret), build on
  the runner and push to ACR, `az containerapp update` to roll. **Reuse the
  edgesense finding directly: ACR Tasks is disabled on this subscription
  (`TasksOperationsNotAllowed`) — don't waste a day rediscovering it.**
  Unlike edgesense's pipeline, make the deploy job `needs:` the selftest job —
  this repo already has that discipline in `ci.yml`; keep it in CD.
- Image note: the GHCR image CI already pushes could serve as the deploy
  source instead of ACR; the phase PR decides (ACR keeps parity with the
  edgesense playbook and managed-identity pulls; GHCR is one registry fewer).

**Cost model (the gate that matters on a shared trial credit):** scale-to-zero
dashboard ≈ €0 idle + ACR Basic ≈ €4/mo + Log Analytics cents. No standing
compute. Tear-down (`terraform destroy`) between demo periods is the default
posture, same as edgesense. This subscription already hosts edgesense-rg —
**check remaining credit before applying anything.**

## 6. Phase 4 — outlook, only if a backend ever exists *(proposed)*

Recorded so the option is visible, gated on a product decision that hasn't
happened:

- If the iOS app ever gets a sync/backend service, that's when edgesense's
  Key Vault + managed-identity secret pattern and `/metrics` + scraper
  observability transfer. Remember the edgesense lesson: **an alert rule
  without a data source holding the metric silently never fires** — its
  drift trigger sat dead for weeks because no Prometheus existed to scrape
  the gauge.
- If the classifier ever trains in the cloud on pooled opt-in data, that's
  when the MLflow registry + champion-alias-in-tags pattern applies (Azure
  ML's MLflow registry has no alias API — tags are the pointer).

## 7. Hard-won lessons imported wholesale

1. **Test against the real thing when feasible.** Real MLflow (not mocks)
   exposed a 4-minute retry hang in edgesense; real Azure state (not tfvars)
   exposed that its drift trigger had never fired. Fixtures verify logic;
   only the real dependency verifies integration — and docs must say which
   one was done.
2. **Windows consoles are cp1252.** Any report/CLI printing `σ`, `→`, emoji:
   `sys.stdout.reconfigure(encoding="utf-8")` first. (Crashed edgesense's
   benchmark on this machine; this repo's CI matrix includes Windows, which
   is the right defence — keep it.)
3. **Delete merged branches immediately.** Stale refs misled work twice in
   one edgesense session (a "conflicting PR" that was long merged; a "9
   commits ahead" that was 0). `--delete-branch` on merge.
4. **Trial-subscription quirks** (same subscription as edgesense): ACR Tasks
   disabled; several VM SKUs at 0 quota (`Standard_E*sv3` — its retrain job
   had to move to `D4s_v3`, then a scale-to-zero cluster); the consumption
   API returns every cost field as `null`, so credit burn is only visible in
   the portal.
5. **Docs that describe generated files get overwritten.** Acquisition/setup
   notes live in a stable doc (this file), never in a `--out` target.

## 8. Sequencing

| Phase | Ships | Cost | Depends on |
|---|---|---|---|
| 1. uv + lock | pyproject/uv.lock, lock-driven Docker, uv CI | none | — |
| 2. classifier MLOps | manifest, eval harness, promotion gate, model-gate workflow | none | 1 (dev group) |
| 3. Azure demo + CD | Terraform scaffold, OIDC deploy of the demo dashboard | ~€4/mo standing, €0 idle compute | 1 (lock-driven image) |
| 2b. coach LLMOps | ✅ shipped — prompt registry, local trace, graders, guardrail, eval set + gate ([LLMOPS.md](LLMOPS.md)) | none | 1 |
| 4. backend outlook | not scheduled | — | a product decision |

Phases 1–2 improve the repo regardless of any cloud ambition. Phase 3 is the
"Azure + Terraform + CD" line for this repo and is reversible with one
`terraform destroy`.
