# mini_swe_agent (vendored, third-party)

This directory is a **vendored copy** of [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)
(version **2.3.0**), the assistant runtime that the SWEJourney harness drives. It
is third-party software, redistributed here under its original **MIT License**
(see [`LICENSE`](LICENSE); copyright © 2024 Kilian Lieret and Carlos Jimenez).

We include it in-tree (rather than as a pip dependency) so that a few
SWEJourney-specific integration points can live alongside the rest of the code
and stay pinned to a known-good version. Everything under `minisweagent/` is
upstream code **except** the additions and edits listed below.

## SWEJourney-specific additions

New files added for this project (they extend upstream by
composition/inheritance and do not modify the stock modules):

| File | What it adds |
|------|--------------|
| `minisweagent/run/trace_runner_v2.py` | replays a SWEJourney `agent_trace.v2` file through mini-swe-agent, adding per-message history truncation on top of the base `trace_runner.py`; wires `../trace_synth` + `../evaluation` + the SWE-bench source onto `sys.path` |
| `minisweagent/models/utils/history_truncate.py` | the shared per-message truncation implementation used by both `trace_runner_v2` and `truncating_proxy` |
| `minisweagent/models/utils/truncating_proxy.py` | a model proxy applying the same two-layer input truncation for benchmark runs, independent of trace metadata |

## SWEJourney-specific edits to upstream files

| File | Edit |
|------|------|
| `minisweagent/environments/docker.py` | optional on-demand image preloading via the `SWEJOURNEY_ENSURE_IMAGE` env var — points at a loader script that `docker load`s a local image tar; **degrades to a no-op when unset**, so upstream behaviour is unchanged by default |
| `minisweagent/run/benchmarks/swebench.py` | applies `trace_runner_v2`'s two-layer input truncation on the benchmark path |
| `minisweagent/config/benchmarks/swebench.yaml` | `run_args` tuned + commented for the SWEJourney container lifecycle (platform alignment; `--entrypoint sleep` container survival) |

## Environment variables read by the integration

All optional; each degrades gracefully when unset:

- `SWEJOURNEY_ENSURE_IMAGE` — path to the on-demand image-loader script (`docker.py`)
- `SWEJOURNEY_SYNTH_DIR` — the `trace_synth` directory (`trace_runner_v2.py`; defaults to `../trace_synth`)
- `SWEJOURNEY_EVAL_DIR` — the `evaluation` directory (`trace_runner_v2.py`; defaults to `../evaluation`)
- `SWEJOURNEY_SWEBENCH_SRC` — an optional local SWE-bench source checkout (`trace_runner_v2.py`)

## Upstream

For usage, documentation, and the canonical source of the unmodified code, see
the upstream project: <https://github.com/SWE-agent/mini-swe-agent>. Refer there
for anything not in the tables above.
