<div align="center">

# SWEJourney

**A benchmark of multi-step, execution-verified agent trajectories for real-world software engineering tasks.**

30 tasks · 16 repositories · Python / Go / JavaScript / TypeScript · every test verified in a real container

</div>

---

## What is this?

**SWEJourney** packages complete *agent trajectories* for solving real GitHub
issues — not just the final patch, but the whole plan → attempt → verify journey
— together with **execution-verified** fail-to-pass / pass-to-pass test lists for
each task.

Each task is derived from a real repository at a fixed commit. An LLM agent
decomposes it into subtasks and produces, for each, a working code patch plus the
tests that prove it. **Every declared test was actually executed inside the task's
Docker image** — nothing is model-asserted. The result is a dataset you can use to
train or evaluate coding agents on *how* to get to a solution, with a trustworthy
oracle for *whether* a solution is correct.

Tasks are drawn from **[SWE-bench Verified](https://www.swebench.com/) (8)** and
**[SWE-bench Pro](https://github.com/scaleapi/SWE-bench_Pro-os) (22)**, spanning 16
repositories across four languages.

---

## The benchmark

The published artifact lives in **[`benchmark/`](benchmark/)** — one self-contained
JSON file per task (30 files, ~12 MB total). Everything you need to consume the
benchmark is in that directory; the rest of this repo is the tooling that produced
it.

### Task file schema

```jsonc
{
  "config": {
    "repo": "astropy/astropy",
    "instance_id": "astropy__astropy-13398",
    "base_commit": "...",             // commit the task starts from
    "image": "...",                    // Docker image the tests run in
    "test_runner_kind": "...",         // pytest (Verified) | pro_official (Pro)
    "repo_language": "python",
    "source": "SWE-bench Verified",
    "difficulty": "15 min - 1 hour",
    ...                                 // more provenance metadata
  },

  "events": [
    { "kind": "PLAN",    ... },         // a BigTask blueprint
    { "kind": "ATTEMPT",                // one solved subtask
      "big_task_id": "...", "subtask_id": "...",
      "user_query": { ... },            // the request the agent answered
      "gold_patch": [ ... ],            // the verified code change (ordered patches)
      "test_patch": [ ... ],            // tests added/modified to check it
      "own_f2p": [ ... ],               // this subtask's fail-to-pass nodes
      "own_p2p": [ ... ]                // this subtask's pass-to-pass nodes
    }
  ],

  "final_f2p": [ "path/to/test.py::test_name", ... ],   // task-level, calibrated
  "final_p2p": [ ... ]
}
```

- **`config`** — provenance + everything needed to reconstruct the environment
  (repo, base commit, Docker image, test runner).
- **`events`** — the trajectory: `PLAN` blueprints interleaved with the `ok=true`
  `ATTEMPT` for each subtask (failed/retried attempts are dropped). Each ATTEMPT
  carries the gold code change and the tests that gate it.
- **`final_f2p` / `final_p2p`** — the task-level oracle: which tests must go from
  **fail → pass** (proving the fix) and which must **stay passing** (proving no
  regression). These are flat, de-duplicated, and **gold-calibrated** — nodes that
  are broken by data/environment issues are removed, and any "pass-to-pass" node
  that does not actually pass from the starting state is moved into fail-to-pass.
  The two lists never overlap.

### Using it

Load a task and read its oracle:

```python
import json

task = json.load(open("benchmark/astropy__astropy-13398.json"))

cfg   = task["config"]
image = cfg["image"]                  # docker image to run tests in
f2p   = task["final_f2p"]             # must go fail -> pass
p2p   = task["final_p2p"]             # must stay passing

# The gold solution, subtask by subtask:
for e in task["events"]:
    if e["kind"] == "ATTEMPT":
        print(e["subtask_id"], "->", len(e["gold_patch"]), "patch part(s)")
```

To **evaluate an agent**: start the repo at `config.base_commit` inside
`config.image`, have the agent produce a patch, apply the task's `test_patch`, and
run the tests — the run passes iff every `final_f2p` node goes from failing to
passing and every `final_p2p` node stays passing. To **use it as training data**:
the `events` stream is a ready-made plan → patch → test trajectory.

> Running the tests requires Docker and the task's image. Verified images come
> from the public `swebench/*` family; Pro images from `jefzda/sweap-images:*`.

---

## Repository layout

```
SWEJourney/
├── benchmark/                  # ← THE DATASET: 30 clean, self-contained task JSON files
├── run_generate_benchmark.sh   # one-file driver: synthesize → probe → clean → benchmark/
├── run_model_eval.sh           # one-file driver: user-sim closed loop → score table
├── code/
│   ├── trace_synth/            # Trace synthesis engine (agent plan/attempt/augment loop)
│   ├── trace_clean/            # Cleaning + gold-side f2p/p2p calibration → benchmark/
│   ├── user_simulation/        # User-simulation closed-loop model evaluation harness
│   ├── evaluation/             # Scoring: reads run outputs + benchmark → score tables
│   └── mini_swe_agent/         # Vendored mini-swe-agent (assistant runtime; third-party)
└── data/                       # Regenerated intermediates (git-ignored; see below)
```

| Directory | What it is | Docs |
|-----------|-----------|------|
| [`benchmark/`](benchmark/) | The published dataset — start here to *use* SWEJourney. | this file |
| [`code/trace_synth/`](code/trace_synth/) | Drives an LLM agent through a plan → attempt → augment loop in a container, using pytest as the sole judge, and emits raw `agent_trace.v2` traces. | [README](code/trace_synth/README.md) |
| [`code/trace_clean/`](code/trace_clean/) | Consumes raw traces, runs two gold-side probes to calibrate the final test lists, and assembles `benchmark/`. Its [`prompts/`](code/trace_clean/prompts/) also collect the LLM rubrics that score task authenticity, task scope/classification, and user↔assistant interaction authenticity. | [README](code/trace_clean/README.md) |
| [`code/user_simulation/`](code/user_simulation/) | Runs a simulated-user closed loop against an assistant model (4 personas), then executes the final f2p/p2p per subtask. | [README](code/user_simulation/README.md) |
| [`code/evaluation/`](code/evaluation/) | Reads the user-sim run outputs + the benchmark and prints the score / cost tables. | [README](code/evaluation/README.md) |
| [`code/mini_swe_agent/`](code/mini_swe_agent/) | Vendored [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent); the assistant runtime the harness drives (third-party, MIT). | [README](code/mini_swe_agent/README.md) |

---

## Reproducing the benchmark

The quickest path is the top-level driver script, which wires the whole
synthesize → probe → clean pipeline (it prints each step and is meant to be read
and unblocked step by step — configure `code/trace_synth/config.yaml` first):

```bash
./run_generate_benchmark.sh          # → benchmark/
```

Or run the two stages by hand (each stage's own README has the full details):

```bash
# 1. Synthesize raw agent traces (needs LLM API access + Docker task images).
cd code/trace_synth
python run_all_instances.py --output-dir ../../data/trace --workers 4

# 2. Calibrate and clean into the published benchmark (needs Docker).
cd ../trace_clean
python probe_allgold.py            # apply all gold, find bad nodes
python probe_base_seed.py          # apply SEED only, split p2p baseline
python build_benchmark.py --force  # → ../../benchmark/<id>.json
```

## Evaluating a model

To score a coding model on the benchmark with the simulated-user closed loop,
use the second driver script (configure the assistant endpoints in
`code/user_simulation/config.yaml` first):

```bash
MODELS="your-model" ./run_model_eval.sh   # → data/usersim_final/… + a score table
```

It runs `code/user_simulation` for each model × instance × persona, then
`code/evaluation` prints the per-dimension score table.


### About `data/`

`data/` holds the large, regenerated intermediates and is **not tracked in git**
(the raw traces alone are multiple GB):

- `data/trace/` — raw `agent_trace.v2` traces emitted by `trace_synth` (stage 1).
- `data/_allgold_probe/`, `data/_base_seed_probe/` — the two gold-side calibration
  probe outputs consumed by `build_benchmark.py` (stage 2).

You do **not** need `data/` to *use* the benchmark — it is fully reconstructible by
re-running the pipeline above, and the finished product is already in
`benchmark/`.

---

## Citation

If you use SWEJourney in your research, please cite this repository.

```bibtex
@misc{swejourney,
  title  = {SWEJourney: Execution-Verified Agent Trajectories for Software Engineering},
  author = {SWEJourney contributors},
  year   = {2026},
  howpublished = {\url{https://github.com/<your-org>/SWEJourney}}
}
```

Underlying task instances originate from
[SWE-bench Verified](https://www.swebench.com/) and
[SWE-bench Pro](https://github.com/scaleapi/SWE-bench_Pro-os); please cite those
works as well.

---

## License

SWEJourney is released under the [MIT License](LICENSE).

`code/mini_swe_agent/` is a vendored copy of the third-party
[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) project and remains
under its own MIT License (see
[`code/mini_swe_agent/LICENSE`](code/mini_swe_agent/LICENSE) and its
[README](code/mini_swe_agent/README.md) for the list of local modifications).
