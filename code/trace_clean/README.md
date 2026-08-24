# trace_clean — Trace Cleaning & Final f2p/p2p Calibration

Turns the raw `agent_trace.v2` traces synthesized by `trace_synth` into **clean,
publishable benchmark JSON** (one file per task), and computes each task's
**final f2p / p2p test lists** calibrated against the gold side.

All paths resolve relative to this directory (see `config.py`), so the whole
`SWEJourney/` tree can be relocated as a unit.

## Directory layout

```
SWEJourney/
├── code/
│   ├── trace_synth/           # synthesis engine (this package reuses its 5 low-level modules; no copy vendored)
│   └── trace_clean/           # ← this package
│       ├── config.py          # relative-path configuration (every script takes paths from here)
│       ├── probe_base_seed.py # p2p "starting-state" probe (base + SEED gold only)
│       ├── probe_allgold.py   # "bad-node" probe (apply all gold, final state)
│       ├── build_benchmark.py # clean + assemble final_f2p/final_p2p, emit benchmark
│       └── README.md
├── data/
│   ├── trace/                 # [input]  raw traces (trace_synth's output)
│   ├── _base_seed_probe/      # intermediate: probe_base_seed output
│   └── _allgold_probe/        # intermediate: probe_allgold output
└── benchmark/                 # [output] final cleaned clean JSON (one file per task)
```

## Pipeline (three steps)

```bash
cd SWEJourney/code/trace_clean

# 1. Two gold-side calibration probes (need Docker + local task images; read-only,
#    each spins up a temporary container and cleans up after the run).
python probe_allgold.py       # → ../../data/_allgold_probe/*.json
python probe_base_seed.py     # → ../../data/_base_seed_probe/*.json

# 2. Clean + assemble, emit the final benchmark (no Docker; purely consumes step 1's results).
python build_benchmark.py --force   # → ../../benchmark/<id>.json
```

Every script's default directories come from `config.py`; override with
`--traces-dir` / `--out-dir` (probes) or the environment variables
(`SWEJOURNEY_TRACE_IN` / `SWEJOURNEY_DATA_DIR` / `SWEJOURNEY_BENCHMARK_DIR` /
`SWEJOURNEY_SYNTH_DIR`).

## How the final f2p / p2p lists are calibrated

Each subtask declares its `own_f2p` / `own_p2p` (fail-to-pass / pass-to-pass test
nodes) in its ATTEMPT event. The final lists are the `own_f2p ∪ own_p2p` over all
deduplicated subtasks, then run through two gold-side calibration steps
(independent of any model under test):

1. **Bad-node removal** (`probe_allgold`): a node that still comes up `FAILED` /
   `NOT_FOUND` even after applying **all** gold (final state) is a data/environment
   problem and is dropped from **both** f2p and p2p.
2. **p2p starting-state split** (`probe_base_seed`): an `own_p2p` node that is
   `!= PASSED` at the "base + SEED gold only" starting state does not pass from the
   start — it is not a stable regression, so it is **moved from p2p into f2p**.
   (SEED's own p2p is the official PASS_TO_PASS regression set — inherited by
   nature, not probed.)

So:

```
final_f2p = (own_f2p − bad nodes) ∪ (own_p2p nodes that fail at the starting state and are not bad nodes)
final_p2p = (own_p2p − starting-state failures − bad nodes)
```

The output is flat-deduplicated per task; the two lists have zero intersection, so
they union directly with no conflict resolution.

> p2p has two states to keep distinct: the **starting state** (does it pass at
> base+SEED — decided here by `probe_base_seed`, and it determines whether the node
> goes to f2p or p2p) versus the **final state** (does it pass after some model
> finishes — a per-model evaluation result, which does not enter this
> model-independent benchmark).

## Cleaning rules (`build_benchmark.py`)

- **Top level**: drop `schema` / `state`; flatten `instance` metadata into
  `config`; append `final_f2p` / `final_p2p` at the end.
- **events**: keep every `PLAN` plus the `ok=true` attempt of each
  `(big_task_id, subtask_id)` (deduplicated); discard failed/retried attempts.
  - `PLAN`: drop uninformative fields, hoist `kind`/`id` to the top, place
    `uplift_proof` after `subtasks_blueprint`.
  - `ATTEMPT`: drop `result`/`rationale` etc.; rename `patch` → `gold_patch` and
    place it before `test_patch`.
- The raw trace is never modified; output goes only to `../../benchmark/`. Existing
  files are skipped by default; `--force` overwrites/updates them.

## Dependencies

Reuses `../trace_synth`'s `swebench_dockerlib` / `local_test_executor` /
`agent_trace` / `swe_bench_{verified,pro}_runner` (injected onto `sys.path` by
`config.ensure_synth_on_path()`), plus the `swebench` / `unidiff` / `datasets`
packages from `trace_synth/requirements.txt`. The probe steps require Docker.

## `prompts/` — task-quality & interaction rubrics

`prompts/` holds the LLM scoring/classification prompts used to profile the
quality and shape of a task (or a user-simulation conversation). They are
reference rubrics — each is a self-contained prompt that takes one sample and
emits a small JSON verdict — and are not invoked by the cleaning scripts above;
keep them here as the documented, reproducible definition of every dimension the
project scores on. Each file has the same layout: a `Sample to annotate:`
placeholder, the dimension definition + focus + examples, a 1–5 rubric (a few
also allow `0` = N/A), and a strict JSON output block whose key is the dimension
name.

| Group | Files | What it scores |
|-------|-------|----------------|
| `task_authenticity/` | 6 prompts (1–5): `problem_statement_clarity`, `problem_statement_no_leakage`, `cross_file_collaboration`, `skill_compositionality`, `economic_professional_value`, `repo_responsibility_alignment` | how *authentic* a task is — the same six dimensions the PLAN rubric (`trace_synth/prompts/verify_plan.txt`) scores |
| `instance_classification/` | 3 prompts: `context_scope`, `change_scope` (single-label), `requirement_depth` (multi-label) | the *scope / kind* of a task — how much context it needs, how far its change reaches, what depth of requirement it represents |
| `interaction_authenticity/` | 6 prompts (D2, 1–5; `d2_05` allows 0=N/A): `d2_01_language_naturalness`, `d2_02_non_oracle_input`, `d2_03_disclosure_pacing`, `d2_04_context_coherence`, `d2_05_clarification_closure`, `d2_08_feedback_driven` | how *real* a user↔assistant conversation looks — natural language, no oracle leakage, on-demand disclosure, coherence, clarification closure, feedback-driven turns |
