# trace_synth — Agent Trace Synthesis Engine

Synthesize `agent_trace.v2` execution traces for SWE-bench-style tasks. Given a
task instance (a repo at a base commit plus failing tests), the engine drives an
LLM through a **plan → attempt → augment** loop inside a real Docker container,
using **pytest as the sole ground-truth judge**, and emits a structured JSON
trace of the whole session — plans, patches, tool calls, and real test results.

Every `FAIL_TO_PASS` / `PASS_TO_PASS` test in the output was actually executed in
a container and filtered by real results; nothing is model-asserted.

- The trace is the only exchange format. The executor reads a generic trace JSON
  and never depends on task-specific scripts.
- Two generation modes: **from-scratch** (the model produces the plan) and
  **fill-skeleton** (replay a recovered skeleton whose `user_query` /
  `acceptance_criteria` are fixed and cannot be regenerated).

---

## How it works — the plan/attempt/augment loop

A task is decomposed into a hierarchy: **BigTask** (one full multi-turn task) →
**Subtask** (one step, in conversation order) → **Attempt** (one patch try at a
subtask). The engine runs a closed loop per BigTask:

```
PLAN      Generate a BigTask blueprint (reason = initial | repair | escalate).
  │       A rubric scores it for realism; only a passing plan is committed.
  │
QUERY     For the current subtask, generate its user_query + acceptance_criteria.
  │       (Retried in isolation — a bad query never re-rolls the whole plan.)
  │
ATTEMPT   Produce patch + test_patch + declared own_f2p, then verify in-container:
  │         pre-check (F2P must FAIL before the fix)
  │         → apply patch → smoke import → prior-regression check
  │         → post-test (F2P must PASS after the fix)
  │
AUGMENT   After the attempt passes, optionally add own_p2p (pass-to-pass) tests.
  │       Augment never adds F2P — only P2P hardening.
  │
CLOSE     BigTask fully passed → backfill PLAN.result.ok = true.
          Stuck → repair / escalate / give up per the retry budget.
```

**Core invariants** (do not break these — they are what makes a trace trustworthy):

- **pytest is the only pass/fail authority.** What the model claims does not count.
- The container's working tree accumulates forward and is **never proactively
  reset**; a failed attempt only rolls back to the snapshot taken at that
  attempt's start.
- `prior_regression` (the set of previously-passing tests to protect) grows
  **monotonically within a BigTask** and does not carry across BigTasks.
- Every `unified_diff` must apply cleanly with `git apply` — no fuzzy fallback.

---

## The `agent_trace.v2` schema

Top level:

```jsonc
{
  "schema": "agent_trace.v2",
  "instance": { /* subset of the source SWE-bench instance JSON */ },
  "config":   { "image": "...", "pytest_bin": "...", "cwd": "...", "smoke_import": null },
  "state":    { "current_phase": "DONE", "terminal": "..." },
  "events":   [ /* PLAN / ATTEMPT / AUGMENT ... */ ]
}
```

| Event `kind` | What it carries | Who writes `result` |
|---|---|---|
| `PLAN` | BigTask blueprint: `big_task_id`, `reason`, `round`, `natural_language_goal`, `acceptance_criteria`, `subtasks_blueprint[]` (no patch, no tests) | main flow, after all subtasks pass → `ok=true` |
| `ATTEMPT` | Subtask body: `subtask_id`, `user_query`, `patch` (flat SubPatch list), `own_f2p`, `own_p2p`, `test_patch`, `rationale` | the executor |
| `AUGMENT` | Optional hardening of an attempt — adds `own_p2p` only | the executor |

- **Completion signal:** `state.current_phase == "DONE"`. In fill mode also
  `config.fill_mode == True`.
- The first event has `subtask_id == "SEED_TASK"` (`big_task_id == "SEED"`) and
  carries the ground-truth F2P/P2P, with zero overlap with generated results.
- `enabled == false` marks an abandoned path; it is kept for audit and ignored by
  the executor and all derived-state computation.
- **A SubPatch** is either `{"format": "unified_diff", "diff": "..."}` (the git
  path) or `{"format": "env_setup", "commands": [...]}` (non-interactive shell in
  the container). `patch` is a flat list, applied in order.
- **Derived state** (never persisted — replayed from `events`): effective
  F2P/P2P (union over `ok=true` attempts), `prior_regression`, the active
  BigTask, and per-subtask pass status. The authoritative queries live in
  `agent_trace.py`.

Common `result.error_kind` values: `PATCH_APPLY_FAIL`, `PATCH_SMOKE_FAIL`,
`TEST_PATCH_APPLY_FAIL`, `COLLECT_MISS`, `PYTEST_FAIL`, `TIMEOUT`,
`F2P_PRE_NOT_FAILING`, `P2P_PRE_FAIL`, `augment_failed`.

---

## Architecture — three-client cascade

Three LLM clients cooperate, configured in `config.yaml`:

| Client | Role | Default backend |
|--------|------|-----------------|
| **main** | PLAN, verify/augment, and late ATTEMPT/AUGMENT rounds | Anthropic official Messages API |
| **first_call** | early ATTEMPT/AUGMENT rounds | local OpenAI-compatible server |
| **repair** | JSON repair of malformed model output | OpenAI official Chat Completions |

The cascade sends the **cheap local model** the first few attempt rounds and
**falls back to the main model** once those are exhausted (default: local for
rounds 1–4, main from round 5). PLAN uses the main model from its first round and
does not participate in the cascade. This keeps the common case cheap while
guaranteeing a strong model gets the hard cases. All three clients speak standard
vendor protocols — there is no proxy layer.

When the model emits malformed JSON, PLAN/ATTEMPT/AUGMENT first route the
`GeneratedJSONParseError` through the repair client before re-validating.

Work runs inside per-instance Docker containers (invoked through the `docker`
CLI); the `swebench` package supplies the test specs / parsers / directives.

### Docker: Verified vs Pro

The two SWE-bench variants are handled differently end-to-end:

| Source | Image family | `cwd` | login shell | Test judge |
|---|---|---|---|---|
| **Verified** | `swebench/sweb.eval.x86_64.*` | `/testbed` | `True` (`-lc`, sources conda) | `swebench`'s `MAP_REPO_VERSION_TO_SPECS` test command + official parser |
| **Pro** | `jefzda/sweap-images:*` | `/app` | `False` (`-c`, preserves image `ENV PATH`) | per-instance official `run_script.sh` + `parser.py` (supports Go / jest / mocha) |

- All docker/git/pytest calls funnel through `swebench_dockerlib.py`; upstream
  code depends only on its stable API.
- `DockerSession` starts one long-lived `sleep infinity` container per instance
  and drives everything via `docker exec`. Snapshots/rollbacks use tar, not
  `git stash`.
- Images are **not pulled online** in the intended deployment: pre-`docker save`
  the image tars and set `paths.docker_pro_dir` / `paths.docker_verified_dir`;
  the engine derives the image name from the instance and idempotently
  `docker load`s it. (With those paths empty it falls back to whatever image is
  available locally.)

---

## Layout

```
trace_synth/
├── main.py                     # single-instance entrypoint (synthesize / --fill-skeleton / --resume)
├── run_all_instances.py        # batch driver (from-scratch), resumable
├── run_fill_all.py             # batch driver (fill recovered skeletons), resumable
├── progress_store.py           # cross-process progress file (fcntl-locked)
├── config.py / config.yaml     # runtime configuration (endpoints / models / paths)
├── api_credentials.py.example  # optional file-based credential template
├── agent_trace.py              # agent_trace.v2 data structures + derived-state queries
├── synthesize_flows.py         # PLAN/ATTEMPT/AUGMENT flow control (BaseFlow retry skeleton)
├── synthesize_gateway.py       # LLM gateway: protocol translation + cost accounting + cache estimation
├── synthesize_*.py             # event history/schema, workspace, augment utils, flow utils
├── local_test_executor.py      # in-container test execution (bootstrap / replay / pre / post / attempt loop)
├── swebench_dockerlib.py       # DockerSession + patch/pytest primitives
├── swe_bench_verified_runner.py / swe_bench_pro_runner.py   # Verified / Pro test runners
├── skills_*.py                 # optional cross-instance "skills" memory
├── prompts/                    # LLM prompt templates (English)
└── swe_bench_pro_official/     # minimal SWE-bench Pro run_scripts (see note)
```

---

## Setup

```bash
pip install -r requirements.txt          # requests, swebench, unidiff, datasets, pyyaml
# Docker must be installed and runnable, plus the task images available locally.
```

### 1. Configure endpoints & models

Edit `config.yaml` (ships with public defaults). Nothing in it is secret — real
API keys are read from the environment variables it names.

```yaml
clients:
  main:       { backend: anthropic, base_url: https://api.anthropic.com/v1/messages, model: <model>, api_key_env: ANTHROPIC_API_KEY }
  repair:     { backend: openai,    base_url: https://api.openai.com/v1/chat/completions, model: <model>, api_key_env: OPENAI_API_KEY }
  first_call: { backend: local,     base_url: http://127.0.0.1:8000/v1/chat/completions, model: <model>, api_key_env: LOCAL_API_KEY }
```

`backend: local` is any self-hosted OpenAI-compatible server (vLLM, SGLang, …);
if it needs no auth, just leave its `api_key_env` variable unset.

### 2. Set credentials (environment)

```bash
export ANTHROPIC_API_KEY=sk-ant-...      # required (main client)
export OPENAI_API_KEY=sk-...             # required if the repair client is used
export LOCAL_API_KEY=...                 # only if your local server needs auth
```

(Alternatively copy `api_credentials.py.example` → `api_credentials.py`; it is
git-ignored. Environment variables take precedence.)

### 3. Point at your dataset

Set `paths.dataset` / `paths.full_dataset` / `paths.subset_jsonl` in
`config.yaml`, or pass them per-run via `--dataset` / `--subset`. Datasets are
HuggingFace `load_from_disk` directories with an `instance_id` column.

---

## Usage

**Single instance** (an input source is required — `--dataset/--instance-id`,
`--input`, `--resume`, or `--resume-from`):

```bash
python main.py --dataset /path/to/dataset --instance-id <id> --output out.json
```

**Batch (from scratch), resumable, parallel:**

```bash
python run_all_instances.py --output-dir traces/ --workers 4
```

**Batch (fill recovered skeletons):**

```bash
python run_fill_all.py --traces-dir traces/ --workers 4
```

### Resume & failure recovery

Both batch drivers track progress in a JSON file (`progress_store.py`, an
`fcntl`-locked file safe across processes and machines) with `completed` /
`failed` / `running` states, and skip completed instances — so they can be
interrupted and re-run safely. A trace is complete when
`state.current_phase == "DONE"`.

Useful flags:

- `--only <id,...>` — restrict to specific instances (isolated debugging).
- `--retry-failed` — re-run instances that failed last time (skipped by default).
- `--reclaim-stale` — take over `running` records whose worker process is gone
  (conservatively skipped by default; use only when no leftover process exists).
- `--reverse` — process the subset back-to-front (can run alongside a forward
  driver; they share the output dir and rely on DONE-detection to avoid dupes,
  but each should use its own `--progress-file`).

Two-level checkpointing: at the **instance** level, DONE traces are skipped; at
the **file** level, an existing-but-unfinished output is continued via `--resume`
(the driver passes it through automatically). Because `run.log` is the sole
source for per-call cost accounting, a resume **appends** to the log rather than
overwriting it.

If an instance ends with `state.terminal == "BIG_TASK_REPAIR_EXHAUSTED"`, note
that this state is baked into the trace JSON (the BigTask's PLAN + ATTEMPT events
are disabled), so clearing the progress file alone will not re-run it — the trace
JSON's internal state must be reset first.

---

## Configuration reference

| Section | Key | Meaning |
|---------|-----|---------|
| `clients.<name>` | `backend` | `anthropic` \| `openai` \| `local` |
| | `base_url` | full endpoint URL (`${ENV}` expansion supported) |
| | `model` | model name sent in the request |
| | `api_key_env` | env var holding the API key (empty ⇒ no auth) |
| | `anthropic_version` | (main only) `anthropic-version` header |
| `paths` | `dataset` / `full_dataset` / `subset_jsonl` | dataset locations |
| | `docker_pro_dir` / `docker_verified_dir` | optional local image-tar caches |
| | `hf_home` | optional `HF_HOME` override |
| `runtime` | `python` | interpreter for batch subprocesses (empty ⇒ current) |
| | `proxy` | outbound proxy for test containers (empty ⇒ none) |
| | `usage` | usage tag passed through to the gateway |

Environment overrides: `SWEJOURNEY_SYNTH_CONFIG` (config path), `PYTHON`,
`DOCKER_HTTP_PROXY`, `HF_HOME`, plus the `api_key_env` variables.

---

## SWE-bench Pro data

`swe_bench_pro_official/run_scripts/<instance_id>/` ships the two files the Pro
runner actually reads — `run_script.sh` and `parser.py` — for each instance.
This is a minimal subset of the official
[scaleapi/SWE-bench_Pro-os](https://github.com/scaleapi/SWE-bench_Pro-os)
run_scripts (the large per-instance `instance_info.txt` and auxiliary files are
omitted). See `swe_bench_pro_official/README.md` for upstream attribution.
SWE-bench Verified needs no vendored data — its specs come from the `swebench`
package.
