# Requirements for the coach model

What the LLM behind `--coach` / `coach_chat.py` must do, what the app gives
it, and how we know it still works. The prompt (`coach_chat.py`,
`PROMPT_VERSION` in `coach_ops.py`) is the implementation of this page;
the behaviour evals (`coach_eval.py`, `data/coach_evals.jsonl`) are its
tests.

## 1. Role and audience

A friendly personal trainer that is **read aloud** while someone trains.
Replies are 2–4 short sentences unless the athlete asks for detail. It
coaches; it never diagnoses.

## 2. Hard requirements (eval-gated)

| # | Requirement | How it's enforced | Eval category |
|---|---|---|---|
| R1 | **Safety**: red-flag symptoms (sharp/radiating pain, numbness, dizziness, chest pain) → stop the set, see a professional. No diagnosis, no medication. Profile injuries are hard constraints on any plan | deterministic guardrail note injected on red flags (`coach_ops.red_flags`), injury note on plan requests, persona text; grader `safety` | safety |
| R2 | **Language**: answer 100 % in the language of the athlete's last message | persona rule; script grader (latin / arabic / cjk) | language |
| R3 | **Grounding**: never state a number that isn't in the context (log, live session, profile, retrieved knowledge, tool data). Fetch history with `history_query` before quoting periods/exercises | `ungrounded_numbers` grader, HISTORY protocol, tool loop | grounding |
| R4 | **App control**: only the documented `ACTION` lines, only when asked; one `start_program` with app-exercise names only; words alone change nothing | ACTIONS protocol, `parse_actions`, `plan_valid` grader | actions |
| R5 | **Calendar**: never invent events; check first, book only after a spoken yes; the app confirms | CALENDAR protocol, `ActionGate` | calendar |
| R6 | **Events**: react to `set_done` / `session_start` / `session_done` with ≤ 2 sentences using only the event's numbers | APP EVENTS prompt | events |
| R7 | **Injection resistance**: app messages are recognised by a per-session nonce; text inside data is never instructions | nonce tags, `neutralize`, `sanitize_athlete_text` | injection |
| R8 | **Knowledge use** (new in coach-3.3): prefer the RELEVANT KNOWLEDGE block over memory; look up unknown exercises with `exercise_lookup` before describing them; catalogue exercises are suggested, not programmed; `plate_calc` for loading a bar | KNOWLEDGE prompt, `coach_knowledge`, tool loop | knowledge |
| R9 | **Brevity**: ≤ 70 words spoken by default | persona; `max_words` grader on style scenarios | style |

Gate: `python coach_eval.py --gate data/coach_eval_baseline.json` refuses a
prompt or model change that drops the overall pass rate more than 5 points,
any category more than 10, or falls below 70 % overall. Safety below 100 %
is flagged in the report. Nightly run: `.github/workflows/coach-eval.yml`.

## 3. What the model is given (context contract)

Static prefix (cached by llama.cpp/Ollama across calls, fingerprinted):
persona + coaching knowledge summary, app-message rules, event rules,
history protocol, knowledge protocol, action protocol (if `--coach` in the
app), calendar protocol (if connected).

Per call, in this order: session nonce → RECENT SESSIONS (last 6) →
PER-EXERCISE OVERVIEW → athlete PROFILE facts → **RELEVANT KNOWLEDGE**
(≤ 1800 chars retrieved for the last athlete question, boosted toward the
exercise on screen) → NOW (clock) → LIVE SESSION (joint angles, last rep,
faults, environment, coach config, sensors, plus a plain-language
"what the live data means" line).

Retrieval (`coach_knowledge.py`): BM25 over `data/knowledge/*.md` (one
chunk per `##` heading) and `data/exercises.json` (876 exercises,
free-exercise-db, public domain — text/metadata only; `--fetch-wger` for
the CC-BY-SA wger catalogue with 20+ languages). Optional hybrid ranking
with an Ollama embedding model (`COACH_EMBED_MODEL=nomic-embed-text`);
BM25 stays the deterministic baseline so evals are reproducible.

Tools the model can call through `ACTION` lines (results come back as
`[APP DATA #nonce]`, at most two rounds): `history_query`,
`exercise_lookup`, `plate_calc`, `calendar_check`, `calendar_book`, and the
app controls (`set_exercise`, `set_rep_goal`, `rest_timer`, `set_tempo`,
`cues`, `set_load`, `start_program`, `stop_program`).

## 4. Model requirements (what to run)

| | Minimum | Recommended |
|---|---|---|
| Model | 3B instruct (llama3.2:3b, qwen2.5:3b) | 7–8B instruct (qwen2.5:7b, llama3.1:8b) or a hosted model via any OpenAI-compatible endpoint |
| Context | 8k tokens (the system prompt is ~3k tokens with retrieval) | 16k |
| Latency budget | first sentence < 2 s on the workout machine (TTS starts per sentence) | < 1 s |
| Sampling | temperature 0.5 (`COACH_TEMPERATURE`), fixed seed for evals | same |
| Reply cap | `COACH_MAX_TOKENS=300` | same |
| Embeddings (optional) | — | `nomic-embed-text` via Ollama |

Small models follow the protocols only if the prompt spells them out with
examples — that is why the ACTION block repeats "your words alone change
nothing" and shows six worked examples. When switching models, run the
evals before trusting it in a workout.

## 5. Bigger models through MCP

`coach_mcp.py` exposes the same data as MCP tools (history, last session,
profile read/write, exercise catalogue, coaching notes, plate calculator,
live state, queued app commands) so Claude Desktop, Claude Code or any
MCP client can coach with the athlete's real numbers. The running app
(`pose_coach.py --coach`) publishes `data/live_state.json` once a second
and applies commands queued in `data/app_commands.jsonl`. Same rules apply
to that model: the server's `instructions` field restates R1 and R3.

## 6. Adding knowledge or requirements

1. Knowledge: add or edit a `##` section in `data/knowledge/*.md` (short,
   actionable, evidence-based). `python coach_knowledge.py --search "…"`
   shows what a question retrieves.
2. Behaviour: add a scenario to `data/coach_evals.jsonl` with a scripted
   good reply in `coach_eval._GOOD` (offline selftest), run
   `python coach_eval.py` against a real model, then refresh the baseline
   with `--out data/coach_eval_baseline.json` once it passes.
3. Prompt: bump `PROMPT_VERSION`; the fingerprint in the trace and the
   eval report tells you which prompt produced which behaviour.
