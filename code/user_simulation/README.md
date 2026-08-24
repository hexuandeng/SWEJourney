# user_simulation — Simulated-User Closed-Loop Model Evaluation

Evaluates a coding-assistant model the way a *person* would: a **simulated user**
(one of four personas) holds a multi-turn conversation with the assistant (the
vendored [mini-swe-agent](../mini_swe_agent/), running in a real Docker
container), steering it through a task's subtasks. When the conversation ends,
the harness silently runs each subtask's **fail-to-pass / pass-to-pass** tests to
score whether the work is actually correct — the user persona never sees those
tests, and the assistant never sees them either.

The simulated user and the assistant run as **two separate processes** (the
assistant is an HTTP server subprocess holding one live container for the whole
run), so neither can peek at the other's internals. All persona behaviour lives
in prompts, not code.

---

## End-to-end flow

```
data/trace/<id>.json            a SWEJourney task trace (input; see ../trace_synth)
        │
usersim_main.py                 parse trace + CLI, wire everything
        │
usersim_scenario.py             build per-subtask specs + the user scenario from the trace
        │
agent/usersim_assistant_server.py   spawn the assistant server subprocess + open its container
        │
usersim_driver.py               run the closed loop, persona-by-turn:
   ┌────────────────────────────────────────────────────────────────┐
   │  user turn  → assistant reply → (silently) per-big-task eval     │
   │  repeat until: all subtasks resolved / confidence exhausted /    │
   │                max turns / assistant exits                       │
   └────────────────────────────────────────────────────────────────┘
        │
bigtask_eval.run_final_eval     re-run every subtask's f2p/p2p once at the end
        │
out/  →  <id>.traj.json  ·  assistant/<id>.assistant.traj.json
         bigtask_eval.json  ·  final_eval.json
```

The batch driver `../../run_model_eval.sh` calls this once per
`<model> × <instance> × <persona>`, writing to
`data/usersim_final/<model>/<instance>/<persona>/`. `code/evaluation/` then reads
those outputs and prints the score tables.

---

## Entry point

```bash
cd code/user_simulation
python usersim_main.py \
    --trace       ../../data/trace/<id>.json \
    --sim-config  prompts/sim_<persona>.yaml \
    -c swebench.yaml -c swejourney.yaml \
    --assistant-model <alias|name@base_url> \
    --out         ../../data/usersim_final/<model>/<id>/<persona> \
    --log-name    <model>/<id>/<persona>
```

| Flag | Meaning |
|------|---------|
| `--trace` (required) | the task trace to drive; its enabled `ATTEMPT` events become the subtasks |
| `--sim-config` (required) | a persona YAML (`prompts/sim_*.yaml`) — selects who the simulated user is |
| `-c <file\|key=value>` (repeatable) | assistant config files / deep-merge overrides (see below) |
| `--out` (required) | output directory for this run's trajectory + eval JSON |
| `--assistant-model <alias\|name@base_url>` | switch the assistant model **and** its endpoint together (alias from `config.yaml`, or freeform `name@base_url`) |
| `--model <name>` | override only the assistant `model_name` |
| `--assistant-url <url>` | talk to an already-running assistant server instead of spawning one |
| `--host` / `--port` | bind address for the spawned server (`127.0.0.1` / `0` = random) |
| `--log-name <name>` | conversation-log subdirectory name (defaults to the output dir name) |
| `--resume` | continue an interrupted run: restore the trimmed conversation, re-attach the old container, no git reset |

`--assistant-model` is parsed into two `-c` overrides for the server
(`model.model_name=…` and `model.model_kwargs.api_base=…`), so the persona
(user-sim) side — fixed by `--sim-config` — is never affected.

---

## The four personas

Each persona is a self-contained `--sim-config` file under `prompts/`; the
natural-language behaviour is rendered from `prompts/user_sim_call.txt` +
`prompts/persona_dimensions.txt`.

| Persona file | Who they are |
|--------------|--------------|
| `sim_non_programming_user.yaml` | a non-technical stakeholder — cannot read code or run tests |
| `sim_product_stakeholder.yaml` | a product/business owner — outcome-driven, does not verify each step |
| `sim_repo_newcomer_engineer.yaml` | a junior engineer new to the repo — learns as they go, asks questions |
| `sim_repository_steward.yaml` | a codebase architect — structured, evidence-driven, meticulous |

A persona YAML sets the user-sim `model_name` / `endpoint_url`, a confidence
budget (`confidence_init` / `confidence_floor` / …), the five persona
`dimensions:` (rendered into the prompt), the tools the persona may **not** use
(`tools_disabled:`), and an optional self-check `rubric:`.

---

## Configuration

Two-tier, matching the rest of the repo (env var **>** YAML **>** built-in default):

**`config.yaml`** — endpoint registry (single source of truth):

```yaml
assistant_models:            # aliases for --assistant-model
  <alias>: { model: "openai/<model>", api_base: "http://…/v1" }
gateway:                     # user-sim LLM gateway (OpenAI-compatible)
  url: "…"
  provider: "openai"
docker_proxy: ""             # optional outbound proxy for containers
```

**`swejourney.yaml`** — the assistant-agent config passed via `-c` (model,
docker `run_args`, agent limits). Fill in your own OpenAI-compatible endpoint.

**Environment overrides:**

| Variable | Overrides |
|----------|-----------|
| `SWEJOURNEY_USERSIM_CONFIG` | path to `config.yaml` |
| `SWEJOURNEY_SYNTH_DIR` | the `../trace_synth` dependency directory |
| `SWEJOURNEY_MINISWE_DIR` | the `../mini_swe_agent` directory |
| `SWEJOURNEY_DATA_DIR` | the data root (trace input + `usersim_final` output) |

---

## Files

| File | Role |
|------|------|
| `usersim_main.py` | entry point; wires trace → scenario → server → loop → final eval → outputs |
| `config.py` | path/endpoint config; puts `trace_synth` + `mini_swe_agent` + this dir on `sys.path`; loads `config.yaml` |
| `usersim_scenario.py` | builds subtask specs + the user scenario from a trace's events |
| `usersim_driver.py` | the closed-loop driver: generates each user turn, runs the persona's tools, tracks confidence + subtask status |
| `user_simulator.py` | the user-sim LLM client + message/tool-call/scenario data types |
| `assistant_client.py` | thin HTTP client for the assistant server (`/health`, `/message`, `/reset`, `/shutdown`) |
| `live_subtask_test_tool.py` | the in-conversation tools (`run_subtask_test` / `mark_subtask` / `read_status`) that test against the live container with snapshot/restore |
| `bigtask_eval.py` | runs a big-task's f2p/p2p when it finishes, and `run_final_eval` for all subtasks at the end |
| `infra_failure_classify.py` | classifies infrastructure (build/dependency/runtime) failures in test output |
| `agent/usersim_assistant_server.py` | the assistant-side HTTP server (spawns the mini-swe-agent + its container) |
| `agent/usersim_agent.py` | the assistant agent subclass handling the `reply_to_user` turn boundary |
| `agent/usersim_model.py` | the assistant model wrapper (bash + `reply_to_user` tools) |
| `agent/reply_tool.py` | the `reply_to_user` tool schema + parser |

---

## Requirements

Docker (the assistant holds one live container per run) plus the top-level
`requirements.txt` (litellm / pydantic / typer / jinja2 for the assistant runtime;
PyYAML; the `swebench` package). Depends on `../trace_synth`'s low-level modules
(`agent_trace`, `local_test_executor`, `swebench_dockerlib`, `swe_bench_*_runner`)
and the vendored `../mini_swe_agent` (`minisweagent.*`) — both are put on
`sys.path` automatically by `config.setup_paths()`.
