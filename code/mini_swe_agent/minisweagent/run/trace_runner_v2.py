#!/usr/bin/env python3
"""trace_runner_v2 — adds per-message truncation on top of trace_runner.

Does not modify trace_runner.py; extends it purely via inheritance/composition.

Added truncation rules (applied after trace_runner's existing "per-turn compression"):

  role="tool"       keep the earliest PER_MSG_MAX_CHARS chars (head kept)
  role="assistant"  keep only the latest PER_MSG_MAX_CHARS chars of the content field (tail kept)
                    the tool_calls field is unaffected
  role="user"       not truncated

The truncated position gets the minimal elision marker ``... ...`` (no char count / explanatory text,
to avoid the model learning it as a text pattern and echoing it back into its own output).

Usage
----
python -m minisweagent.run.trace_runner_v2 \\
    /path/to/trace.json \\
    -o runs/trace_out_v2 \\
    -c swebench.yaml \\
    -c runs/v4flash_hxsz.yaml
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import traceback
from pathlib import Path

import typer

from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.models import get_model
from minisweagent.models.utils.history_truncate import PER_MSG_MAX_CHARS, truncate_msg
from minisweagent.run.benchmarks.swebench import get_sb_environment
from minisweagent.run.trace_runner import (
    DEFAULT_CONFIG_FILE,
    TraceRunnerAgent,
    _CompressingModelProxy,
    extract_tasks,
)
from minisweagent.utils.log import add_file_handler
from minisweagent.utils.serialize import recursive_merge

logger = logging.getLogger(__name__)

# The per-message content cap PER_MSG_MAX_CHARS and the per-message truncation truncate_msg are both provided
# by the shared module minisweagent.models.utils.history_truncate, reusing the same implementation as truncating_proxy.

# Dependency directories, resolved relative to the SWEJourney/code/ root
# (../../../ from this file: run -> minisweagent -> mini_swe_agent -> code),
# each overridable via an environment variable.
_CODE_ROOT = Path(__file__).resolve().parents[3]
# trace_synth/ execution library (gold-prefix mode reuses bootstrap_session /
# replay_model_patches).
_EXTEND_DIR = os.environ.get("SWEJOURNEY_SYNTH_DIR", str(_CODE_ROOT / "trace_synth"))
# evaluation/ modules added by the user-simulation branch (user_simulator /
# live_subtask_test_tool). This branch only adds files; it imports the existing
# trace_synth modules directly rather than copying them.
_EVALUATION_DIR = os.environ.get("SWEJOURNEY_EVAL_DIR", str(_CODE_ROOT / "evaluation"))
# Optional local editable SWE-bench source. Empty by default: swebench is
# expected to be pip-installed (see requirements.txt). Set this env var only if
# you want to import a local checkout instead.
_SWEBENCH_SRC = os.environ.get("SWEJOURNEY_SWEBENCH_SRC", "")


def _ensure_extend_path() -> None:
    """Add trace_synth/ + evaluation/ and the local SWE-bench source to sys.path (idempotent)."""
    import sys
    for p in (_SWEBENCH_SRC, _EXTEND_DIR, _EVALUATION_DIR):
        if p and p not in sys.path:
            sys.path.insert(0, p)


# ── Extension proxy: do per-message truncation after per-turn compression ───────────────────

class _TruncatingProxy(_CompressingModelProxy):
    """On top of the parent's per-turn compression, additionally truncate the content of each message.

    Truncation rules are in the shared module ``history_truncate.truncate_msg`` (tool keeps the head,
    assistant.content keeps the tail, user is untouched). This class only chains this layer after the parent's _compress.
    """

    def _compress(self, messages: list[dict], *, force: bool = False) -> list[dict]:
        # Step 1: the parent's per-turn compression (delete / halve old big_task's tool results).
        # force is passed through to the parent: after hitting context-400, force compression to the low watermark (see _CompressingModelProxy._compress).
        compressed = super()._compress(messages, force=force)
        # Step 2: per-message content truncation (shared implementation)
        return [truncate_msg(m) for m in compressed]


# ── Extended Agent ───────────────────────────────────────────────────────────────

class TraceRunnerAgentV2(TraceRunnerAgent):
    """Adds per-message truncation on top of TraceRunnerAgent.

    The parent __init__ replaces self.model with _CompressingModelProxy;
    this class then swaps it for _TruncatingProxy, keeping the same underlying real model instance.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # The parent has already set self.model to _CompressingModelProxy(real_model, self)
        # Take out the real model and re-wrap it in the extension proxy
        real_model = self.model._model
        self.model = _TruncatingProxy(real_model, self)


# ── gold-prefix mode: each subtask starts from the "preceding gold" baseline ────────────────────
#
# Default (backward-compatible with old behavior): all subtasks share one container, and model edits accumulate
# continuously — subtask N continues on top of "the model having done 1..N-1 itself (possibly wrong)".
#
# gold-prefix mode: before each subtask begins, reset the container repo to
#   base_commit + all preceding enabled&ok ATTEMPT gold model patches
# and commit (making HEAD = the preceding gold state, so the model's `git diff` naturally outputs only this
# step's increment), while also resetting the conversation (agent.run gets a fresh system+user). Container
# preparation fully reuses the executor's bootstrap_session / replay_model_patches, without reinventing the git-apply logic.


def _enabled_ok_attempts(trace: dict, *, skip_seed: bool = False) -> list[dict]:
    """Return, in trace order, all enabled & ok=True ATTEMPT events.

    These events correspond 1:1 to the tasks produced by extract_tasks() (each with n_uq=1).
    When ``skip_seed=True``, skip events with ``subtask_id == "SEED_TASK"`` — gold-prefix mode does not run the
    seed precheck (user_query is just the placeholder "seed precheck", and having the model solve it is
    pointless). The extract_tasks side must filter is_seed in sync to keep the indices aligned.
    """
    _ensure_extend_path()
    import agent_trace as at  # noqa: E402

    out: list[dict] = []
    for ev in trace.get("events") or []:
        if ev.get("kind") != "ATTEMPT":
            continue
        if not at.is_enabled(ev):
            continue
        if (ev.get("result") or {}).get("ok") is not True:
            continue
        if skip_seed and ev.get("subtask_id") == "SEED_TASK":
            continue
        out.append(ev)
    return out


def _gold_prefix_tasks(trace: dict) -> tuple[list[dict], list[dict]]:
    """gold-prefix-mode-specific: return (tasks, gold_events), both already skipping SEED_TASK.

    extract_tasks does not look at result.ok, _enabled_ok_attempts does; after each filters SEED separately,
    they remain 1:1 aligned (each enabled&ok non-SEED ATTEMPT has exactly 1 user_query).
    """
    tasks = [t for t in extract_tasks(trace) if not t.get("is_seed")]
    gold_events = _enabled_ok_attempts(trace, skip_seed=True)
    return tasks, gold_events


def _make_executor_session(env, image: str, login_shell: bool):
    """Construct an executor DockerSession pointing at the mini env's already-started container (does not rebuild the container).

    All of DockerSession's primitives (exec/write/read/fs_snapshot) locate the container only via
    container_name and are stateless `docker exec`. So we bypass start() here and point container_name at
    env.container_id, which 100% reuses the executor logic.

    Key: the container's lifecycle is owned by the mini env; this session merely "borrows" it to run docker exec.
    Therefore `_started` MUST be False, otherwise DockerSession.stop()/__del__ would run
    `docker rm -f <container>` when this object is GC'd, deleting the container the agent is using (this once
    caused the agent's first command to be "No such container" and spin the whole time).
    `_started=False` makes stop()/__del__ early-return, while exec/fs_snapshot and other primitives do not
    depend on `_started`, so their functionality is unaffected.
    """
    _ensure_extend_path()
    from swebench_dockerlib import DockerSession  # noqa: E402

    sess = DockerSession.__new__(DockerSession)
    sess.image = image
    sess.container_name = env.container_id  # docker exec accepts an ID or a name
    sess.keep_on_fail = True   # double insurance: even if misjudged as failed, never rm the borrowed container
    sess.login_shell = login_shell
    sess._started = False      # does not own the container → stop()/__del__ will not docker rm
    sess._failed = False
    return sess


def _prepare_gold_prefix(sess, trace: dict, current_event: dict, cwd: str) -> None:
    """Reset the container repo to base_commit + all preceding gold, then commit it as the new baseline.

    Fully reuses the executor's bootstrap_session (before_repo_set_cmd → base_commit) and replay_model_patches
    (replay the preceding enabled&ok gold model patches, resetting each one's test_patch files afterward),
    without applying the current event itself (stop_before_event=current_event).
    Finally git add -A && commit, so the model's `git diff HEAD` shows only its own increment.
    """
    _ensure_extend_path()
    from local_test_executor import bootstrap_session, replay_model_patches  # noqa: E402

    # 1) Return to a clean base (including before_repo_set_cmd: reset/clean/checkout the test files)
    bootstrap_session(sess, trace)
    # 2) Replay all preceding enabled&ok gold model patches (not including the current event)
    replay_model_patches(sess, trace, stop_before_event=current_event, verbose=False)
    # 3) Commit as the new baseline so the subsequent model's git diff outputs only the increment
    q = shlex.quote(cwd)
    sess.exec(
        f"cd {q} && git config --global --add safe.directory {q} && "
        f"git config user.email trace@runner && git config user.name trace-runner && "
        f"git add -A && git commit -q -m gold_prefix_baseline --allow-empty"
    )


def _extract_model_patch(sess, cwd: str) -> str:
    """Extract the incremental patch the model produced in this subtask.

    _prepare_gold_prefix has already committed "base + all preceding gold" as the baseline, so at this moment
    `git diff HEAD` (plus add -A to include new files) is the model's own clean increment, not containing the
    preceding gold. Used at grading time as the model patch applied onto the same baseline to run own_f2p/own_p2p.

    Returns an empty string on failure (the grading side treats an empty patch as no output, handling it as a fail); never raises in a way that affects the traj write-out.
    """
    try:
        import shlex as _shlex
        q = _shlex.quote(cwd)
        # After including new files, take the full increment relative to the baseline; --no-color prevents ANSI; binary guarantees replayability
        res = sess.exec(
            f"cd {q} && git add -A 2>/dev/null; git diff --no-color --binary HEAD"
        )
        return res.stdout or ""
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to extract model patch (ignored, storing empty string): %s", exc)
        return ""


def _run_trace_gold_prefix(
    trace: dict,
    instance: dict,
    instance_id: str,
    tasks: list[dict],
    config: dict,
    env,
    agent: "TraceRunnerAgentV2",
    traj_path: Path,
) -> dict:
    """gold-prefix execution path: each subtask is independent (reset the container to the preceding gold + reset the conversation).

    Each subtask produces one exit; finally the 14 conversation segments are merged and written back to
    traj_path, keeping the interface that Phase2 reads a list of role=="exit" unchanged (1 subtask = 1 exit).
    """
    cfg = trace.get("config", {})
    cwd = cfg.get("cwd", "/app")
    login_shell = cfg.get("test_runner_kind") != "pro_official"
    sess = _make_executor_session(env, image=cfg.get("image", ""), login_shell=login_shell)

    # gold-prefix mode skips SEED_TASK (gold is still applied in the preceding replay)
    tasks, gold_events = _gold_prefix_tasks(trace)
    if len(gold_events) != len(tasks):
        logger.warning(
            "gold-prefix: number of enabled&ok ATTEMPTs (%d) does not match number of tasks (%d), aligning by min",
            len(gold_events), len(tasks),
        )
    n = min(len(gold_events), len(tasks))

    merged_messages: list[dict] = []
    result: dict = {}
    for idx in range(n):
        task = tasks[idx]
        ev = gold_events[idx]
        logger.info(
            "[gold-prefix %d/%d] subtask_id=%s big_task_id=%s",
            idx, n - 1, task["subtask_id"], task["big_task_id"],
        )
        # 1) Reset the container to base + preceding gold (not including the current event)
        try:
            _prepare_gold_prefix(sess, trace, ev, cwd)
        except Exception as exc:
            logger.error(
                "[gold-prefix %d] container preparation failed: %s\n%s",
                idx, exc, traceback.format_exc(),
            )
            continue
        # 2) Reset the conversation + run the current subtask (agent.run clears messages and rebuilds system+user)
        try:
            agent.n_calls = 0
            result = agent.run(task["content"], big_task_id=task["big_task_id"])
            logger.info("[gold-prefix %d] exit_status=%s", idx, result.get("exit_status"))
        except Exception as exc:
            logger.error(
                "[gold-prefix %d] (%s) crashed: %s\n%s",
                idx, task["subtask_id"], exc, traceback.format_exc(),
            )
        # 3) Collect this subtask's conversation (agent.messages is now the full turn)
        merged_messages.extend(list(agent.messages))

    # Merge and write back the traj: messages is the concatenation of 14 segments, containing 14 exits
    data = {
        "info": {
            "instance_id": instance_id,
            "gold_prefix": True,
            "n_subtasks": n,
        },
        "messages": merged_messages,
        "trajectory_format": "mini-swe-agent-1.1",
    }
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    traj_path.write_text(json.dumps(data, indent=2))
    logger.info("gold-prefix all done. Trajectory (%d exits) saved to %s",
                sum(1 for m in merged_messages if m.get("role") == "exit"), traj_path)
    return result


def _run_single_subtask(
    trace: dict,
    instance: dict,
    instance_id: str,
    tasks: list[dict],
    config: dict,
    env,
    agent: "TraceRunnerAgentV2",
    traj_path: Path,
    subtask_index: int,
) -> dict:
    """Independent-process / independent-container mode: this call runs only the ``subtask_index``-th subtask.

    This process exclusively owns the one container the env started; after preparing "base + preceding 0..N-1
    gold", it runs this one subtask with a fresh agent conversation and writes a traj containing only 1 exit.
    32 samples × 14 subtasks = 448 such independent calls, scheduled by the orchestrator.
    """
    cfg = trace.get("config", {})
    cwd = cfg.get("cwd", "/app")
    login_shell = cfg.get("test_runner_kind") != "pro_official"
    sess = _make_executor_session(env, image=cfg.get("image", ""), login_shell=login_shell)

    # gold-prefix mode skips SEED_TASK (gold is still applied in the preceding replay)
    tasks, gold_events = _gold_prefix_tasks(trace)
    n = min(len(gold_events), len(tasks))
    if subtask_index < 0 or subtask_index >= n:
        logger.error("subtask_index=%d out of range (valid 0..%d)", subtask_index, n - 1)
        return {}

    task = tasks[subtask_index]
    ev = gold_events[subtask_index]
    logger.info(
        "[single-subtask %d/%d] subtask_id=%s big_task_id=%s",
        subtask_index, n - 1, task["subtask_id"], task["big_task_id"],
    )

    # 1) Reset the container to base + preceding 0..N-1 gold (not including the current event)
    _prepare_gold_prefix(sess, trace, ev, cwd)

    # 2) Fresh conversation running the current subtask (agent.run clears messages and rebuilds system+user)
    result: dict = {}
    try:
        agent.n_calls = 0
        result = agent.run(task["content"], big_task_id=task["big_task_id"])
        logger.info("[single-subtask %d] exit_status=%s", subtask_index, result.get("exit_status"))
    except Exception as exc:
        logger.error(
            "[single-subtask %d] (%s) crashed: %s\n%s",
            subtask_index, task["subtask_id"], exc, traceback.format_exc(),
        )

    # 2.5) Extract the model's incremental patch (before the container is destroyed, the git diff relative to the gold-prefix baseline)
    model_patch = _extract_model_patch(sess, cwd)

    # 3) Write out this subtask's independent traj (1 exit)
    data = {
        "info": {
            "instance_id": instance_id,
            "gold_prefix": True,
            "subtask_index": subtask_index,
            "subtask_id": task["subtask_id"],
            "big_task_id": task["big_task_id"],
            "model_patch": model_patch,
        },
        "messages": list(agent.messages),
        "trajectory_format": "mini-swe-agent-1.1",
    }
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    traj_path.write_text(json.dumps(data, indent=2))
    logger.info("[single-subtask %d] trajectory saved to %s", subtask_index, traj_path)
    return result


# ── Main flow ────────────────────────────────────────────────────────────────────

def run_trace_v2(
    trace_path: Path,
    output_dir: Path,
    config_spec: list[str],
    model_name: str | None = None,
    gold_prefix: bool = False,
    subtask_index: int | None = None,
) -> dict:
    """Replay all tasks in the trace file using TraceRunnerAgentV2.

    Same logic as trace_runner.run_trace(); the only difference is using TraceRunnerAgentV2.

    When gold_prefix=True, enable the "each step continues from the preceding gold baseline" mode (see above).
    When subtask_index is specified (must go with gold_prefix): this call runs only this one subtask, this
    process exclusively owns one container, and outputs a traj containing only 1 exit (one cell of the 448 full matrix).

    User-sim mode has been moved out of this file: it is now driven by evaluation/usersim_main.py, and the
    coding assistant goes through minisweagent.run.assistant_server (HTTP).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    add_file_handler(output_dir / "trace_runner_v2.log")

    trace = json.loads(Path(trace_path).read_text())
    instance: dict = trace.get("instance", {})
    instance_id: str = instance.get("instance_id", Path(trace_path).stem)

    tasks = extract_tasks(trace)
    if not tasks:
        logger.warning("No enabled ATTEMPT tasks found in %s", trace_path)
        return {}

    logger.info("Instance '%s': extracted %d tasks (from %s)", instance_id, len(tasks), trace_path)

    # ── Build merged config ──────────────────────────────────────────────────────────
    configs = [get_config_from_spec(spec) for spec in config_spec]
    if model_name:
        configs.append({"model": {"model_name": model_name}})
    config = recursive_merge(*configs)

    # ── Create model / environment (shared across the whole session) ───────────────────────────────
    model = get_model(config=config.get("model", {}))
    env = get_sb_environment(config, instance)

    # ── Create Agent ────────────────────────────────────────────────────────────
    traj_path = output_dir / f"{instance_id}.traj.json"
    agent_kwargs = dict(config.get("agent", {}))
    agent_kwargs["output_path"] = traj_path

    agent = TraceRunnerAgentV2(model, env, **agent_kwargs)

    # ── gold-prefix mode: each subtask is independent (reset the container to the preceding gold + reset the conversation) ──
    # Wrap all execution paths in try/finally to ensure the container created by this process's
    # get_sb_environment is explicitly cleaned up on every exit path (normal return / exception), eliminating
    # container leaks. Note: what's cleaned up here is the env container this process "created itself"; the
    # DockerSession borrowed by _make_executor_session still uses _started=False and is unaffected.
    try:
        if gold_prefix:
            if subtask_index is not None:
                # Independent single task: this process runs only the subtask_index-th subtask
                return _run_single_subtask(
                    trace, instance, instance_id, tasks, config, env, agent,
                    traj_path, subtask_index,
                )
            return _run_trace_gold_prefix(
                trace, instance, instance_id, tasks, config, env, agent, traj_path,
            )

        # ── Execute the task sequence (default: continuous accumulation, scheme B full-history) ─────────────────────
        seed = tasks[0]
        if not seed["is_seed"]:
            logger.warning(
                "The first task is not SEED_TASK (subtask_id=%s), running it as the seed task anyway",
                seed["subtask_id"],
            )

        logger.info("[0/%d] SEED  subtask_id=%s", len(tasks) - 1, seed["subtask_id"])
        result: dict = {}
        try:
            agent.n_calls = 0  # each subtask counts steps independently, avoiding cross-task accumulation triggering step_limit
            result = agent.run(seed["content"], big_task_id=seed["big_task_id"])
            logger.info("[0/%d] exit_status=%s", len(tasks) - 1, result.get("exit_status"))
        except Exception as exc:
            logger.error("SEED task crashed: %s\n%s", exc, traceback.format_exc())

        for idx, task in enumerate(tasks[1:], start=1):
            logger.info(
                "[%d/%d] subtask_id=%-20s  big_task_id=%s",
                idx, len(tasks) - 1,
                task["subtask_id"],
                task["big_task_id"],
            )
            try:
                agent.n_calls = 0  # each subtask counts steps independently, avoiding cross-task accumulation triggering step_limit
                result = agent.continue_with(
                    task["content"],
                    big_task_id=task["big_task_id"],
                )
                logger.info("[%d/%d] exit_status=%s", idx, len(tasks) - 1, result.get("exit_status"))
            except Exception as exc:
                logger.error(
                    "Task %d (%s) crashed: %s\n%s",
                    idx, task["subtask_id"], exc, traceback.format_exc(),
                )

        logger.info("All done. Trajectory saved to %s", traj_path)
        return result
    finally:
        try:
            env.cleanup()
            logger.info("Cleaned up this process's env container (container_id=%s)", getattr(env, "container_id", None))
        except Exception as exc:
            logger.warning("env.cleanup() failed (ignored): %s", exc)


# ── CLI ───────────────────────────────────────────────────────────────────────

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

_CONFIG_SPEC_HELP = (
    "Path to a YAML config file, a filename, or key=value pairs. "
    "Multiple -c are recursively merged in order (later overrides earlier). "
    "Example: -c swebench.yaml -c runs/v4flash_hxsz.yaml"
)


@app.command()
def main(
    trace_path: Path = typer.Argument(..., help="Path to the agent_trace.v2 JSON file"),
    output: str = typer.Option("", "-o", "--output", help="Output directory"),
    config_spec: list[str] = typer.Option(
        [str(DEFAULT_CONFIG_FILE)], "-c", "--config", help=_CONFIG_SPEC_HELP
    ),
    model: str | None = typer.Option(None, "-m", "--model", help="Model name (overrides the config file)"),
    gold_prefix: bool = typer.Option(
        False, "--gold-prefix/--no-gold-prefix",
        help="Each subtask starts from the 'base+all preceding gold' baseline (reset container + reset conversation), "
             "so the model's git diff outputs only this step's increment. Off by default (continuous accumulation, old behavior).",
    ),
    subtask_index: int = typer.Option(
        -1, "--subtask-index",
        help="Run only the N-th enabled&ok subtask (goes with --gold-prefix). "
             "This process exclusively owns one container and outputs a single-subtask traj (1 exit). -1=run the whole trace.",
    ),
) -> None:
    """Replay an agent trace with the enhanced Agent that has content truncation.

    User-sim mode has been moved out: see user_simulation/usersim_main.py + minisweagent.run.assistant_server.
    """
    output_dir = Path(output) if output else Path("runs") / Path(trace_path).stem

    run_trace_v2(
        trace_path=trace_path,
        output_dir=output_dir,
        config_spec=config_spec,
        model_name=model,
        gold_prefix=gold_prefix,
        subtask_index=(subtask_index if subtask_index >= 0 else None),
    )


if __name__ == "__main__":
    app()
