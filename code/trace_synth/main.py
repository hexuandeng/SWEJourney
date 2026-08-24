#!/usr/bin/env python3
"""Main entry point for v2 agent trace synthesis.

This file only handles the outermost orchestration:
- Assemble shared dependencies such as model clients, gateway, and workspace.
- Drive the PLAN -> user_query -> ATTEMPT -> verify -> augment main flow.
- Write the trace back to disk at every persistence boundary.

Prompt assembly, schema normalization, history slicing, model transport details,
and the augment docker-filtering logic all live in dedicated modules to keep this
main file from growing further.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import os
import sys
from pathlib import Path
from typing import Any

import agent_trace as at
from local_test_executor import ensure_seed_attempt, run_pending_attempts, run_seed_precheck

from synthesize_event_history import EventHistory
from synthesize_event_schema import EventSchema
from synthesize_flows import (
    AttemptFlow,
    AttemptUserQueryFlow,
    AugmentOrchestrator,
    PlanFlow,
    SchemaOps,
    ToolLoopConfig,
)
from synthesize_flow_utils import one_line, printable_text
from skills_compactor import compact_skills_after_instance
from synthesize_gateway import LLMGateway
from synthesize_trace_workspace import TraceWorkspace
import config as _config

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROMPT_DIR = HERE / "prompts"

# ── Runtime configuration (endpoints / models / paths) sourced from config.yaml.
# Nothing here is hard-coded to a private endpoint; edit config.yaml or set the
# relevant environment variables for your deployment. ──
_CFG = _config.load_config()
_MAIN_CFG = _config.client_config("main", _CFG)          # Anthropic official Messages API
_REPAIR_CFG = _config.client_config("repair", _CFG)      # OpenAI official Chat Completions
_LOCAL_CFG = _config.client_config("first_call", _CFG)   # local OpenAI-compatible server
_PATHS = _config.path_config(_CFG)
_RUNTIME = _config.runtime_config(_CFG)

# Propagate an optional outbound proxy for test containers (swebench_dockerlib
# reads DOCKER_HTTP_PROXY). Empty → containers get no proxy.
if _RUNTIME.get("proxy") and not os.environ.get("DOCKER_HTTP_PROXY"):
    os.environ["DOCKER_HTTP_PROXY"] = str(_RUNTIME["proxy"])

# Docker image tar directories (optional local cache; empty → skip tar loading).
_DOCKER_PRO_DIR = Path(_PATHS["docker_pro_dir"]) if _PATHS.get("docker_pro_dir") else None
_DOCKER_VER_DIR = Path(_PATHS["docker_verified_dir"]) if _PATHS.get("docker_verified_dir") else None

DEFAULT_USAGE = str(_RUNTIME.get("usage") or "1")
DEFAULT_LLM_BACKEND = "anthropic"

# Main client (Anthropic official Messages API).
MAIN_MESSAGES_URL = _MAIN_CFG["base_url"]
MAIN_MODEL = _MAIN_CFG["model"]
MAIN_ANTHROPIC_VERSION = _MAIN_CFG.get("anthropic_version") or "2023-06-01"

# JSON-repair client (OpenAI official Chat Completions API).
JSON_REPAIR_MODEL = _REPAIR_CFG["model"]
JSON_REPAIR_BACKEND = _REPAIR_CFG.get("backend") or "openai"
OPENAI_CHAT_COMPLETIONS_URL = _REPAIR_CFG["base_url"]

# First-call (cheap) client: a local / self-hosted OpenAI-compatible server,
# used for ATTEMPT/AUGMENT rounds 1-3; round 4+ falls back to the main client.
LOCAL_BACKEND = _LOCAL_CFG.get("backend") or "local"
LOCAL_DEFAULT_MODEL = _LOCAL_CFG["model"]
LOCAL_ENDPOINT_URL = _LOCAL_CFG["base_url"]

PROMPT_USAGE_TYPES = ("PLAN", "VERIFY_PLAN", "ATTEMPT", "VERIFY_ATTEMPT", "AUGMENT_TESTS")
DEFAULT_PYTEST_BIN = "/opt/miniconda3/envs/testbed/bin/pytest"
DEFAULT_CWD = "/testbed"
DEFAULT_PRO_CWD = "/app"
PLAN_MAX_TOOL_CALLS = 50
PLAN_REMIND_EVERY_TOOL_CALLS = 30
ATTEMPT_MAX_TOOL_CALLS = 60
ATTEMPT_REMIND_EVERY_TOOL_CALLS = 40
AUGMENT_MAX_TOOL_CALLS = 60
AUGMENT_REMIND_EVERY_TOOL_CALLS = 40
# For the first round (attempt_no==1, sglang) only: raise the cap to 120; on the
# 120th call emit a first "consider entering the wrap-up phase" reminder (without
# exposing remaining count/cap), then remind again every 60 calls. Applies only to
# the first round of ATTEMPT / AUGMENT.
FIRST_ROUND_MAX_TOOL_CALLS = 120
FIRST_ROUND_REMIND_START_TOOL_CALLS = 120
FIRST_ROUND_REMIND_EVERY_TOOL_CALLS = 60

# Inner repair cap in ATTEMPT generate that does "not consume a retry slot": JSON
# parse failures (openai repair without tools) and infra errors (deepseek
# 400/rate-limit/network etc. -> rerun with tool-enabled generation) are each
# absorbed at most this many times within a single attempt; once exhausted, it
# degrades to one normal attempt (attempt_no+1, a tool-enabled functional retry).
# The inner counter resets to 0 each time generate is entered (i.e. resets after
# an attempt retry).
ATTEMPT_INNER_REPAIR_MAX = 3

WORKSPACE = TraceWorkspace(prompt_dir=PROMPT_DIR, schema_version=at.SCHEMA_VERSION)


def _load_default_api_credentials() -> tuple[str, str | None]:
    """Resolve default API credentials.

    Precedence: ANTHROPIC_API_KEY / LLM_APP_ID env vars first; otherwise an
    optional local ``api_credentials.py`` (git-ignored; copy the provided
    ``api_credentials.py.example``). Returns ("", None) when nothing is set —
    the per-client config keys in _build_clients are the primary source.
    """
    sys.path.insert(0, str(HERE))
    try:
        from api_credentials import DEFAULT_APP_ID, DEFAULT_APP_KEY  # type: ignore
        return os.environ.get("ANTHROPIC_API_KEY", DEFAULT_APP_KEY), DEFAULT_APP_ID
    except Exception:
        return os.environ.get("ANTHROPIC_API_KEY", ""), os.environ.get("LLM_APP_ID") or None


def _configure_safe_stdio() -> None:
    """Avoid corrupting stdout/stderr when the gateway returns malformed characters."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(errors="backslashreplace")


_configure_safe_stdio()


def _trim_tool_output(text: str, *, max_chars: int = 12000) -> str:
    """Trim overly long tool output so observations don't blow up the context."""
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars // 2] + "\n...[tool output truncated]...\n" + text[-max_chars // 2:]


def _log_synthesize_step(step: str, *, ok: bool = True, detail: str = "", key: str = "") -> None:
    """Print synthesis-phase logs uniformly so different steps share one format."""
    suffix = f"  {printable_text(detail)}" if detail else ""
    if key:
        suffix += f"  key={one_line(key)}"
    print(f"[synthesize] step {printable_text(step)}: {'ok' if ok else 'FAIL'}{suffix}", flush=True)


def _attach_llm_usage(trace: dict, client: Any) -> None:
    """Attach the current cumulative LLM usage summary onto trace.state."""
    if hasattr(client, "usage_summary"):
        trace.setdefault("state", {})["llm_usage"] = client.usage_summary()


def _log_llm_usage(client: Any) -> None:
    """Print usage stats by prompt type and in total, for offline cost analysis."""
    if not hasattr(client, "usage_summary"):
        return
    usage = client.usage_summary()
    by_prompt_type = usage.get("by_prompt_type") or {}
    for prompt_type in PROMPT_USAGE_TYPES:
        item = by_prompt_type.get(prompt_type) or {}
        print(
            "[synthesize] llm usage by prompt: "
            f"type={prompt_type} cost={item.get('cost', 0)} requests={item.get('requests', 0)} "
            f"input={item.get('input_tokens', 0)} output={item.get('output_tokens', 0)} "
            f"cache_write={item.get('cache_creation_input_tokens', 0)} "
            f"cache_read={item.get('cache_read_input_tokens', 0)} total={item.get('total_tokens', 0)}",
            flush=True,
        )
    print(
        "[synthesize] llm usage total: "
        f"cost={usage.get('cost', 0)} requests={usage.get('requests', 0)} "
        f"input={usage.get('input_tokens', 0)} output={usage.get('output_tokens', 0)} "
        f"cache_write={usage.get('cache_creation_input_tokens', 0)} "
        f"cache_read={usage.get('cache_read_input_tokens', 0)} total={usage.get('total_tokens', 0)}",
        flush=True,
    )


def _decide_plan_branch(trace: dict, instance: dict) -> dict:
    """The host decides which branch this PLAN round belongs to; the model must not guess.

    Routing signal: whether to escalate depends on "whether all subtasks of this
    PLAN blueprint are complete", NOT on the PLAN's own result.ok. The PLAN's
    result.ok is written by the plan rubric quality check (how well the plan is
    written), which is a separate matter from "whether all subtasks under the plan
    are done and correct". Judging escalate by result.ok would misjudge a BigTask
    whose subtasks all succeeded as failed and exhaust repair when result=None or
    when the rubric is only recorded rather than enforced.

    - No PLAN -> initial
    - The latest enabled PLAN whose blueprint subtasks all have an enabled+ok=True
      ATTEMPT -> escalate
    - The latest PLAN with enabled=False (disabled by disable_all_big_task_events)
      -> repair
    - Otherwise (some subtask still open) -> repair

    The scan includes disabled events so that after disable_all_big_task_events
    disables a PLAN we can still find the correct latest plan and route to repair,
    instead of wrongly returning initial just because no enabled PLAN is visible.
    """
    plans = [ev for ev in (trace.get("events") or []) if ev.get("kind") == "PLAN"]
    if not plans:
        return {
            "reason": "initial",
            "big_task_id": "BIG_1",
            "round": 1,
            "previous_big_task_id": None,
        }
    latest_plan = plans[-1]
    latest_big = latest_plan.get("big_task_id") or "BIG_1"
    # escalate criterion: all subtasks of this enabled PLAN blueprint already have
    # an enabled+ok=True ATTEMPT (blueprint non-empty and fully done). Only then is
    # the BigTask truly closed out and a new BigTask can start.
    if at.is_enabled(latest_plan):
        subtask_ids = {
            str(s.get("id") or "")
            for s in (latest_plan.get("subtasks_blueprint") or [])
            if s.get("id")
        }
        done = EventHistory.successful_subtask_ids_before(
            trace,
            big_task_id=str(latest_big),
            before_index=len(trace.get("events") or []),
        )
        if subtask_ids and subtask_ids.issubset(done):
            try:
                n = int(str(latest_big).split("_")[-1])
            except Exception:
                n = 1
            return {
                "reason": "escalate",
                "big_task_id": f"BIG_{n + 1}",
                "round": 1,
                "previous_big_task_id": latest_big,
            }
    # enabled=False (already disabled) or some subtask still open -> repair
    return {
        "reason": "repair",
        "big_task_id": latest_big,
        "round": int(latest_plan.get("round") or 1) + 1,
        "previous_big_task_id": None,
    }


def _big_task_index(big_task_id: Any) -> int | None:
    """Parse a big_task_id of the form `BIG_k` into the integer k."""
    text = str(big_task_id or "").strip()
    if not text.startswith("BIG_"):
        return None
    try:
        return int(text.split("_", 1)[1])
    except Exception:
        return None


def _plan_cycles_used(trace: dict, big_task_id: Any) -> int:
    """Count the plan cycles already consumed by the current BigTask (initial + repair)."""
    bid = str(big_task_id or "")
    return 1 + sum(
        1
        for ev in (trace.get("events") or [])
        if ev.get("kind") == "PLAN"
        and ev.get("reason") == "repair"
        and ev.get("big_task_id") == bid
    )


def _new_client(
    *,
    app_id: str | None,
    api_key: str,
    provider: str,
    model: str,
    usage: str | None,
    backend: str,
    endpoint_url: str | None,
    usage_owner=None,
    anthropic_version: str = MAIN_ANTHROPIC_VERSION,
):
    """Create a model gateway already wired to the workspace.

    This way the top-level orchestration only needs to decide "which model this
    phase uses", without repeatedly worrying about transport details like the
    underlying chat interface, tool-call entry point, or usage-tracking hookup.
    """
    client = LLMGateway(
        app_id=app_id,
        api_key=api_key,
        provider=provider,
        model=model,
        usage=usage,
        backend=backend,
        endpoint_url=endpoint_url,
        anthropic_url=MAIN_MESSAGES_URL,
        openai_url=OPENAI_CHAT_COMPLETIONS_URL,
        prompt_usage_types=PROMPT_USAGE_TYPES,
        log_step=_log_synthesize_step,
        trim_tool_output=_trim_tool_output,
        usage_owner=usage_owner,
        anthropic_version=anthropic_version,
    )
    client.workspace = WORKSPACE
    return client




async def _repair_json_via_model(client, raw_text: str, error: Exception) -> Any:
    """Run one JSON repair with GPT-5.4 and return the parsed object directly.

    Note this does NOT hand the repaired text back to the outer layer for a second
    round of repair. If GPT-5.4 itself still returns invalid JSON,
    `GeneratedJSONParseError` is raised as-is and BaseFlow terminates this round
    directly, avoiding repeated repair.
    """
    from synthesize_flow_utils import json_from_text

    messages = [
        {
            "role": "system",
            "content": (
                "You are a JSON repair tool. You are given a model output that should have been JSON, along with a parse error."
                "Your task is to repair it into a valid JSON value without changing the original meaning."
                "Output only the JSON itself; do not add explanations or code blocks."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Parse error: {type(error).__name__}: {error}\n\n"
                "The raw output is as follows:\n"
                f"{raw_text}"
            ),
        },
    ]
    body = await client.chat_once(messages, usage_stage="JSON_REPAIR")
    repaired_text = client.client_response_text(body)
    return json_from_text(repaired_text)


async def run_plan_flow(*, workspace: TraceWorkspace, plan_client, repair_client, trace: dict, plan_decision: dict, enable_docker_cli: bool, output_path=None, write_json=None, refine_plan: bool = False, verify_client=None, verify_max_attempts: int = 2, max_rubric_failures: int = 2, first_call_client=None):
    """Run the PLAN flow directly, avoiding another FlowRunner wrapper."""
    _log_synthesize_step(
        "plan-flow-start",
        detail=f"big_task_id={plan_decision.get('big_task_id')} reason={plan_decision.get('reason')}",
    )
    _flow = PlanFlow(
        client=plan_client,
        first_call_client=first_call_client,
        repair_client=repair_client,
        prompt=workspace.build_plan_prompt(
            trace=trace,
            decision=plan_decision,
            prompt_name="generate_plan.txt",
        ),
        trace=trace,
        plan_decision=plan_decision,
        schema_ops=SchemaOps(
            coerce_event=EventSchema.coerce_plan_event,
            validate_event=EventSchema.validate_plan_event,
        ),
        tool_loop=ToolLoopConfig(
            enable_docker_cli=enable_docker_cli,
            max_tool_calls=PLAN_MAX_TOOL_CALLS,
            remind_every_tool_calls=PLAN_REMIND_EVERY_TOOL_CALLS,
            first_round_max_tool_calls=FIRST_ROUND_MAX_TOOL_CALLS,
            first_round_remind_start_tool_calls=FIRST_ROUND_REMIND_START_TOOL_CALLS,
            first_round_remind_every_tool_calls=FIRST_ROUND_REMIND_EVERY_TOOL_CALLS,
        ),
        json_repair=(None if repair_client is None else (lambda raw_text, error: _repair_json_via_model(repair_client, raw_text, error))),
        output_path=output_path,
        write_json=write_json,
        refine_plan=refine_plan,
        verify_client=verify_client,
        verify_max_attempts=verify_max_attempts,
        max_rubric_failures=max_rubric_failures,
        workspace=workspace,
    )
    result, error = await _flow.run()
    _log_synthesize_step(
        "plan-flow-end",
        ok=not error,
        detail=(
            f"big_task_id={plan_decision.get('big_task_id')}"
            + (f" error={str(error)[:200]}" if error else "")
        ),
    )
    return result, error


async def run_attempt_user_query_flow(*, workspace: TraceWorkspace, client, trace: dict, big_task_id: str, subtask: dict, output_path=None, write_json=None):
    """Run the ATTEMPT user_query generation directly."""
    sid = str(subtask.get("id") or "")
    _log_synthesize_step(
        "user-query-flow-start",
        detail=f"big_task_id={big_task_id} subtask_id={sid}",
    )
    result, error = await AttemptUserQueryFlow(
        client=client,
        prompt=workspace.build_attempt_user_query_prompt(
            trace=trace,
            target={"big_task_id": big_task_id, "subtask_id": subtask["id"]},
            prompt_name="generate_attempt_user_query.txt",
        ),
        subtask=subtask,
        json_repair=(lambda raw_text, error: _repair_json_via_model(client, raw_text, error)),
        trace=trace,
        big_task_id=big_task_id,
        output_path=output_path,
        write_json=write_json,
    ).run()
    _log_synthesize_step(
        "user-query-flow-end",
        ok=not error,
        detail=(
            f"big_task_id={big_task_id} subtask_id={sid}"
            + (f" error={str(error)[:200]}" if error else "")
        ),
    )
    return result, error


async def run_attempt_impl_flow(
    *,
    workspace: TraceWorkspace,
    client,
    repair_client,
    verify_client,
    trace: dict,
    big_task_id: str,
    subtask: dict,
    user_query: dict,
    acceptance_criteria: str,
    enable_docker_cli: bool,
    enable_execute: bool,
    output_path: Path,
    max_attempts: int,
    max_rubric_failures: int = 2,
    first_call_client=None,
):
    """Run ATTEMPT patch+f2p synthesis; `quality_validate` performs all verification.

    `AttemptFlow.quality_validate` is the single verification entry point: format ->
    code execution -> rubric; code execution happens inside `quality_validate` via
    the injected `run_pending_attempts`, and persistence and rubric write-back also
    happen inside the flow. Here we only assemble dependencies and call the parent
    `BaseFlow.run()`.
    """
    prompt_name = "generate_f2p.txt"
    sid = str(subtask.get("id") or "")
    _log_synthesize_step(
        "attempt-impl-flow-start",
        detail=f"big_task_id={big_task_id} subtask_id={sid} max_attempts={max_attempts}",
    )
    _flow = AttemptFlow(
        client=client,
        first_call_client=first_call_client,
        repair_client=repair_client,
        verify_client=verify_client,
        prompt=workspace.build_attempt_impl_prompt(
            trace=trace,
            target={"big_task_id": big_task_id, "subtask_id": subtask["id"]},
            user_query=user_query,
            acceptance_criteria=acceptance_criteria,
            prompt_name=prompt_name,
        ),
        trace=trace,
        big_task_id=big_task_id,
        subtask=subtask,
        user_query=user_query,
        acceptance_criteria=acceptance_criteria,
        schema_ops=SchemaOps(
            coerce_event=EventSchema.coerce_attempt_event,
            validate_event=EventSchema.validate_attempt_event,
            is_request_instruction_subtask=EventSchema.is_request_instruction_subtask,
        ),
        prompt_name=prompt_name,
        tool_loop=ToolLoopConfig(
            enable_docker_cli=enable_docker_cli,
            max_tool_calls=ATTEMPT_MAX_TOOL_CALLS,
            remind_every_tool_calls=ATTEMPT_REMIND_EVERY_TOOL_CALLS,
            reminder_context=str(subtask.get("text") or ""),
            first_round_max_tool_calls=FIRST_ROUND_MAX_TOOL_CALLS,
            first_round_remind_start_tool_calls=FIRST_ROUND_REMIND_START_TOOL_CALLS,
            first_round_remind_every_tool_calls=FIRST_ROUND_REMIND_EVERY_TOOL_CALLS,
        ),
        max_attempts=max_attempts,
        max_rubric_failures=max_rubric_failures,
        inner_repair_max=ATTEMPT_INNER_REPAIR_MAX,
        output_path=output_path,
        write_json=workspace.write_json,
        load_trace=at.load_trace,
        executor=run_pending_attempts,
        enable_execute=enable_execute,
        workspace=workspace,
        json_repair=(None if repair_client is None else (lambda raw_text, error: _repair_json_via_model(repair_client, raw_text, error))),
    )
    result, error = await _flow.run()
    _log_synthesize_step(
        "attempt-impl-flow-end",
        ok=not error,
        detail=(
            f"big_task_id={big_task_id} subtask_id={sid}"
            + (f" error={str(error)[:200]}" if error else "")
        ),
    )
    return result, error


async def run_attempt_augment_flow(*, workspace: TraceWorkspace, client, trace: dict, target_event: dict, output_path: Path, max_attempts: int = 3, verbose: bool = True, first_call_client=None, repair_client=None):
    """Run the augmentation directly after a successful ATTEMPT."""
    bid = str(target_event.get("big_task_id") or "")
    sid = str(target_event.get("subtask_id") or "")
    _log_synthesize_step(
        "augment-flow-start",
        detail=f"big_task_id={bid} subtask_id={sid}",
    )
    result = await AugmentOrchestrator.run(
        client=client,
        first_call_client=first_call_client,
        trace=trace,
        target_event=target_event,
        scope="ATTEMPT",
        output_path=output_path,
        verbose=verbose,
        write_json=workspace.write_json,
        max_tool_calls=AUGMENT_MAX_TOOL_CALLS,
        remind_every_tool_calls=AUGMENT_REMIND_EVERY_TOOL_CALLS,
        first_round_max_tool_calls=FIRST_ROUND_MAX_TOOL_CALLS,
        first_round_remind_start_tool_calls=FIRST_ROUND_REMIND_START_TOOL_CALLS,
        first_round_remind_every_tool_calls=FIRST_ROUND_REMIND_EVERY_TOOL_CALLS,
        max_attempts=max_attempts,
        json_repair=(None if repair_client is None else (lambda raw_text, error: _repair_json_via_model(repair_client, raw_text, error))),
    )
    skipped = isinstance(result, dict) and result.get("skipped")
    _log_synthesize_step(
        "augment-flow-end",
        ok=True,
        detail=(
            f"big_task_id={bid} subtask_id={sid}"
            + (" skipped=true" if skipped else "")
        ),
    )
    return result



def _attempt_prior_regression_ok(ev: dict) -> bool:
    """Whether this ATTEMPT's prior-regression (regression of earlier subtasks in the same big_task) all passed.

    After the executor (the prior-regression loop in local_test_executor.execute_attempt)
    finishes, it writes each prior regression node's status into ``result.per_node_status``
    keyed by ``prior-regression::<node_id>``. The criterion must be *exactly* consistent
    with the executor's ``failing = [n for n in nodes if status.get(n) != "PASSED"]``:
    only an explicit ``PASSED`` counts as passing; everything else (FAILED / XFAIL /
    ERROR / NOT_FOUND / TIMEOUT / empty) counts as a regression failure.

    No ``prior-regression::`` key at all (e.g. the first subtask of this big_task, with
    no prior subtask to regress) is treated as passing.
    """
    per_node = (ev.get("result") or {}).get("per_node_status") or {}
    for key, status in per_node.items():
        if str(key).startswith("prior-regression::"):
            if str(status or "").upper() != "PASSED":
                return False
    return True


def _find_last_attempt_with_genuine_f2p(
    trace: dict,
    *,
    big_task_id: str,
    subtask_id: str,
) -> tuple[int | None, list[str]]:
    """Scan for the *last* ATTEMPT event of this (big_task_id, subtask_id) in the trace.

    Returns (event_index, genuinely_passing_f2p_nodes):
      - No ATTEMPT at all -> (None, []);
      - Last ATTEMPT exists -> (index, [...]), the list may be empty (meaning the
        last one cannot be kept as a usable artifact).

    "Usable f2p" = the genuinely passing nodes among the declared own_f2p whose
    post-patch per_node_status == "PASSED", not the length of the declared list.
    The criterion matches PARTIAL-KEEP (_passing_f2p) in synthesize_flows.py and
    genuine_f2p_count in local_test_executor.py exactly, avoiding "N declared, 0
    genuinely passing" being mistakenly kept as a usable artifact.

    **Extra tightening (to prevent conflicting data from slipping through)**: even
    if own_f2p has genuinely passing nodes, if any of this attempt's
    prior-regression did not pass (it broke an earlier subtask's own tests in the
    same big_task), it also **cannot be rescued**, and we return (index, []).
    Otherwise an attempt that "passes its own f2p but breaks a prior subtask on
    regression" would be forcibly let through as ok=True/enabled=True, carrying
    conflicting data into the trace all the way to DONE (the gold precheck would
    faithfully catch it). The check uses ``_attempt_prior_regression_ok``, the same
    criterion as the executor's regression gate.
    """
    events = trace.get("events") or []
    for _i in range(len(events) - 1, -1, -1):
        _ev = events[_i]
        if (
            _ev.get("kind") == "ATTEMPT"
            and _ev.get("big_task_id") == big_task_id
            and str(_ev.get("subtask_id") or "") == subtask_id
        ):
            _own_f2p = [str(n) for n in (_ev.get("own_f2p") or [])]
            _per_node = (_ev.get("result") or {}).get("per_node_status") or {}
            _passing = [
                n for n in _own_f2p
                if str(_per_node.get(n) or "").upper() == "PASSED"
            ]
            # Tighten: prior-regression broke an earlier subtask -> cannot rescue (clear passing).
            if _passing and not _attempt_prior_regression_ok(_ev):
                _passing = []
            return _i, _passing
    return None, []


def _promote_rescued_attempt(
    trace: dict,
    *,
    event_index: int,
    passing_f2p: list[str],
) -> dict:
    """Promote the given ATTEMPT event to a usable artifact: narrow own_f2p to
    genuinely passing nodes, set result.ok=True, enabled=True, and mark
    result.rescued=True (the fill and normal paths share this marker).
    Returns the event dict (modified in place).
    """
    _ev = trace["events"][event_index]
    _ev["own_f2p"] = passing_f2p
    _res = dict(_ev.get("result") or {})
    _res["ok"] = True
    _res["rescued"] = True
    _ev["result"] = _res
    _ev["enabled"] = True
    return _ev


async def synthesize_trace(
    *,
    instance_path: Path,
    output_path: Path,
    client,
    repair_client=None,
    augment_client=None,
    user_query_client=None,
    verify_client=None,
    first_call_client=None,
    image: str,
    pytest_bin: str,
    cwd: str,
    smoke_import: str | None,
    execute: bool,
    max_attempt_retries: int = 4,
    max_augment_retries: int = 4,
    max_big_task_retries: int = 1,
    max_big_tasks: int = 3,
    max_rubric_failures: int = 2,
    attempt_max_rubric_failures: int = 2,
    resume_trace_path: Path | None = None,
    refine_plan: bool = False,
    skills_dir: str | None = None,
) -> dict:
    """Run the complete v2 synthesis main loop.

    Several key phase boundaries:
    - PLAN sees the full enabled history.
    - ATTEMPT's user_query / impl / rubric only see the history slice of the
      current BigTask.
    - In v2, PLAN no longer does test augmentation or a separate test flow; it just
      marks completion directly once all enabled subtasks succeed.
    - Code verification always precedes rubric; if code verification fails, rubric
      is not run.
    - Both code failure and rubric failure only roll back to patch+f2p synthesis,
      never back to user_query.
    """
    raw_input = WORKSPACE.read_json(resume_trace_path or instance_path)
    resume_mode = WORKSPACE.is_trace_payload(raw_input)
    if resume_mode:
        trace = WORKSPACE.prepare_resume_trace(
            raw_input,
            image=image,
            pytest_bin=pytest_bin,
            cwd=cwd,
            smoke_import=smoke_import,
        )
        instance = trace.get("instance") or {}
        if hasattr(client, "load_usage_summary"):
            client.load_usage_summary((trace.get("state") or {}).get("llm_usage"))
    else:
        instance = raw_input[0] if isinstance(raw_input, list) else raw_input
        if not isinstance(instance, dict):
            raise ValueError("input must be a SWE-bench instance JSON or an agent trace JSON")
        trace = WORKSPACE.init_trace(
            instance,
            image=image,
            pytest_bin=pytest_bin,
            cwd=cwd,
            smoke_import=smoke_import,
        )

    if hasattr(client, "set_usage_checkpoint"):
        client.set_usage_checkpoint(output_path)
    bench_id = instance.get("instance_id") or instance_path.stem

    if resume_mode:
        WORKSPACE.write_json(output_path, trace)
        _log_synthesize_step(
            "resume-trace",
            detail=f"instance={bench_id} events={len(trace.get('events') or [])} source={resume_trace_path or instance_path}",
        )
    else:
        _log_synthesize_step("init-trace", detail=f"instance={bench_id} image={image}")
        ensure_seed_attempt(trace)
        WORKSPACE.write_json(output_path, trace)
        _log_synthesize_step("seed-event", detail=f"events={len(trace.get('events') or [])}", key="SEED materialized")
        if execute:
            trace = run_seed_precheck(str(output_path), stop_on_fail=True, verbose=True)
            if trace.get("state", {}).get("terminal") == "SKIP_SEED_PRECHECK_FAILED":
                _attach_llm_usage(trace, client)
                WORKSPACE.write_json(output_path, trace)
                return trace
            seed_res = ((trace.get("events") or [{}])[0].get("result") or {})
            _log_synthesize_step(
                "seed-precheck",
                ok=seed_res.get("ok") is True,
                detail=f"err={seed_res.get('error_kind')} tests={len(seed_res.get('per_node_status') or {})}",
                key=next((d.get("key_line") or "" for d in (seed_res.get("test_failure_details") or []) if d.get("key_line")), ""),
            )
        else:
            trace = at.load_trace(output_path)
            _log_synthesize_step("seed-precheck", detail="skipped execute=False")

    while True:
        active_plan = EventHistory.latest_active_plan(trace)
        if active_plan is not None:
            plan_index, plan_obj = active_plan
        else:
            plan_decision = _decide_plan_branch(trace, instance)
            decision_big_idx = _big_task_index(plan_decision.get("big_task_id"))
            if plan_decision.get("reason") == "escalate" and decision_big_idx is not None and decision_big_idx > max_big_tasks:
                state = trace.setdefault("state", {})
                state["current_phase"] = "DONE"
                state["terminal"] = None
                state["stop_reason"] = "MAX_BIG_TASK_REACHED"
                state.pop("last_error", None)
                _attach_llm_usage(trace, client)
                WORKSPACE.write_json(output_path, trace)
                return trace
            if plan_decision.get("reason") == "repair":
                # Add first, then judge: max_big_task_retries=N means at most N plan
                # cycles including the initial one, i.e. at most N-1 repairs. Same
                # "no +1" semantics as max_attempt_retries.
                repair_attempts_used = _plan_cycles_used(
                    trace, plan_decision.get("big_task_id")
                )
                if repair_attempts_used >= max_big_task_retries:
                    # At this point all events of this BigTask were already disabled
                    # by disable_all_big_task_events (reason="subtask_attempt_exhausted")
                    # when the subtask was exhausted, so there is no need to call
                    # disable_big_task_suffix (it would be a no-op: all events already
                    # enabled=False). Just write the terminal state and exit.
                    trace.setdefault("state", {})["current_phase"] = "PLAN"
                    trace.setdefault("state", {})["terminal"] = "BIG_TASK_REPAIR_EXHAUSTED"
                    trace.setdefault("state", {})["last_error"] = (
                        f"{plan_decision.get('big_task_id')} failed after {repair_attempts_used} repair attempt(s)"
                    )
                    WORKSPACE.write_json(output_path, trace)
                    break

            plan_obj, plan_error = await run_plan_flow(
                workspace=WORKSPACE,
                plan_client=client,
                first_call_client=first_call_client,
                repair_client=repair_client,
                trace=trace,
                plan_decision=plan_decision,
                enable_docker_cli=execute,
                output_path=output_path,
                write_json=WORKSPACE.write_json,
                refine_plan=refine_plan and plan_decision.get("reason") != "repair",
                verify_client=verify_client or client,
                max_rubric_failures=max_rubric_failures,
            )
            if plan_obj is None:
                trace.setdefault("state", {})["current_phase"] = "PLAN"
                trace.setdefault("state", {})["terminal"] = "PLAN_VALIDATION_FAILED"
                trace.setdefault("state", {})["last_error"] = plan_error or "PLAN validation failed"
                if plan_decision.get("reason") == "repair":
                    EventHistory.disable_big_task_suffix(trace, str(plan_decision.get("big_task_id") or ""), reason="repair_plan_validation_failed")
                WORKSPACE.write_json(output_path, trace)
                _attach_llm_usage(trace, client)
                WORKSPACE.write_json(output_path, trace)
                return trace
            plan_index = len(trace["events"])
            trace["events"].append(plan_obj)
            WORKSPACE.write_json(output_path, trace)

        subtasks = list(plan_obj["subtasks_blueprint"])
        # skip_subtask_ids = the set of subtasks that were previously enabled and
        # ok=True; skip them this round rather than redo.
        # **Key semantics**: when the repair branch (plan_obj.reason == "repair")
        # reaches here, the previous round's BigTask has already had all its ATTEMPT
        # enabled set to False by disable_all_big_task_events(reason="subtask_attempt_exhausted"),
        # so successful_subtask_ids_before must return an *empty set* -- i.e. repair
        # is equivalent to "redo all subtasks of this BigTask" (including those that
        # previously succeeded). This is intentional: repair = re-plan + re-execute,
        # not reusing old successful artifacts. Only the active_plan (resume) branch
        # gets a non-empty set, skipping already-successful subtasks and continuing
        # the unfinished ones.
        skip_subtask_ids = EventHistory.successful_subtask_ids_before(
            trace,
            big_task_id=plan_obj["big_task_id"],
            before_index=(len(trace.get("events") or []) if active_plan is not None else plan_index),
        ) if (active_plan is not None or plan_obj.get("reason") == "repair") else set()
        plan_needs_repair = False

        for subtask in subtasks:
            sid = str(subtask.get("id") or "")
            if sid in skip_subtask_ids:
                # This subtask already had an enabled ok=true ATTEMPT. But "success"
                # does not mean the artifact is complete: on resume, classify by
                # completeness --
                #   - own_f2p or test_patch is empty -> this ATTEMPT is unusable,
                #     redo from ATTEMPT;
                #   - else own_p2p is empty -> the ATTEMPT body is usable, only
                #     regression tests are missing, just do augment;
                #   - else -> complete, skip.
                _done_idx = EventHistory.latest_successful_attempt_index(
                    trace,
                    big_task_id=plan_obj["big_task_id"],
                    subtask_id=sid,
                    before_index=(len(trace.get("events") or []) if active_plan is not None else plan_index),
                )
                if _done_idx is not None:
                    _done_ev = trace["events"][_done_idx]
                    _has_f2p = bool(_done_ev.get("own_f2p"))
                    _has_test_patch = bool(_done_ev.get("test_patch"))
                    _has_p2p = bool(_done_ev.get("own_p2p"))
                    if _has_f2p and _has_test_patch:
                        if _has_p2p:
                            # Complete, skip.
                            continue
                        # Only own_p2p missing: reuse the successful ATTEMPT, just do augment.
                        _log_synthesize_step(
                            "subtask-resume-augment",
                            detail=f"{sid} own_f2p={len(_done_ev.get('own_f2p') or [])} own_p2p=0",
                        )
                        try:
                            await run_attempt_augment_flow(
                                workspace=WORKSPACE,
                                client=augment_client or client,
                                first_call_client=first_call_client,
                                repair_client=repair_client,
                                trace=trace,
                                target_event=trace["events"][_done_idx],
                                output_path=output_path,
                                max_attempts=max_augment_retries,
                            )
                            trace = at.load_trace(output_path)
                            _aug_ev = trace["events"][_done_idx]
                            _log_synthesize_step(
                                "attempt-augment",
                                detail=f"{sid} own_f2p={len(_aug_ev.get('own_f2p') or [])} own_p2p={len(_aug_ev.get('own_p2p') or [])}",
                            )
                        except Exception as e:
                            print(f"[synthesize][warn] ATTEMPT augmentation (resume) failed for {sid}: {printable_text(e)}; continuing", flush=True)
                        continue
                    # own_f2p or test_patch empty: old ATTEMPT unusable, disable it and redo from ATTEMPT.
                    _done_ev["enabled"] = False
                    _done_ev.setdefault("disabled_reason", "incomplete_attempt_redo")
                    WORKSPACE.write_json(output_path, trace)
                    _log_synthesize_step(
                        "subtask-resume-redo",
                        detail=f"{sid} own_f2p={len(_done_ev.get('own_f2p') or [])} test_patch={len(_done_ev.get('test_patch') or [])}",
                    )
                    # Do not continue; fall through to the normal ATTEMPT generation flow below.
                else:
                    continue
            _log_synthesize_step("subtask-start", detail=f"{sid} category={subtask.get('category')}", key=subtask.get("text") or "")

            # If the previous run's process crashed after append but before
            # execution, the trace may hold a leftover unexecuted (result=None)
            # ATTEMPT. In the new design append+execution completes atomically inside
            # quality_validate, so mark such orphan pending events directly as an
            # abandoned path (enabled=false) so this round regenerates from a clean state.
            pending_attempt_index = EventHistory.latest_pending_attempt_index(
                trace,
                big_task_id=plan_obj["big_task_id"],
                subtask_id=sid,
            )
            if pending_attempt_index is not None:
                trace["events"][pending_attempt_index]["enabled"] = False
                WORKSPACE.write_json(output_path, trace)

            _history_events, _failure_events = WORKSPACE.split_attempt_history_failure(
                trace, big_task_id=plan_obj["big_task_id"], subtask_id=sid,
            )
            _last_failed = _failure_events[-1] if _failure_events else None
            _raw_uq = (_last_failed or {}).get("user_query")
            _raw_ac = (_last_failed or {}).get("acceptance_criteria")
            if isinstance(_raw_uq, dict) and _raw_uq and isinstance(_raw_ac, str) and _raw_ac:
                user_query = EventSchema.coerce_user_query(_raw_uq)
                acceptance_criteria = _raw_ac
            else:
                user_query = None
                acceptance_criteria = None
            if user_query is None:
                uq_result, user_query_error = await run_attempt_user_query_flow(
                    workspace=WORKSPACE,
                    client=user_query_client or client,
                    trace=trace,
                    big_task_id=plan_obj["big_task_id"],
                    subtask=subtask,
                    output_path=output_path,
                    write_json=WORKSPACE.write_json,
                )
                if uq_result is None:
                    trace.setdefault("state", {})["current_phase"] = "ATTEMPT"
                    trace.setdefault("state", {})["terminal"] = "ATTEMPT_USER_QUERY_VALIDATION_FAILED"
                    trace.setdefault("state", {})["last_error"] = user_query_error or "ATTEMPT user query validation failed"
                    EventHistory.disable_big_task_suffix(trace, plan_obj["big_task_id"], reason="attempt_user_query_validation_failed")
                    WORKSPACE.write_json(output_path, trace)
                    _attach_llm_usage(trace, client)
                    WORKSPACE.write_json(output_path, trace)
                    return trace
                user_query = uq_result["user_query"]
                acceptance_criteria = uq_result["acceptance_criteria"]

            # ATTEMPT's three-layer verification (format -> code execution -> rubric)
            # is all folded into AttemptFlow.quality_validate, with BaseFlow.run()
            # exclusively owning all functional retries (total attempts =
            # max_attempt_retries). Here we only do generation scheduling and the
            # post-success close-out (disable earlier failures + augment).
            attempt_obj, attempt_error = await run_attempt_impl_flow(
                workspace=WORKSPACE,
                client=client,
                first_call_client=first_call_client,
                repair_client=repair_client,
                verify_client=verify_client or client,
                trace=trace,
                big_task_id=plan_obj["big_task_id"],
                subtask=subtask,
                user_query=user_query,
                acceptance_criteria=acceptance_criteria,
                enable_docker_cli=execute,
                enable_execute=execute,
                output_path=output_path,
                max_attempts=max_attempt_retries,
                max_rubric_failures=attempt_max_rubric_failures,
            )
            trace = at.load_trace(output_path)

            if attempt_error is not None:
                # ATTEMPT retries exhausted: first try to rescue (same helper as the
                # fill path) -- find the last ATTEMPT of this subtask; if it still
                # carries "genuinely passing own_f2p", promote it to a usable
                # artifact to keep, continue augment + the next subtask, and do *not*
                # trigger plan repair; only when there is no usable own_f2p at all do
                # we fall to plan_needs_repair (arbitrated by Phase E).
                _rescue_idx, _rescue_passing = _find_last_attempt_with_genuine_f2p(
                    trace, big_task_id=plan_obj["big_task_id"], subtask_id=sid,
                )
                if _rescue_idx is None or not _rescue_passing:
                    _log_synthesize_step("attempt-exhausted", ok=False, detail=f"{sid} no usable f2p (0 genuinely passing); {attempt_error}")
                    plan_needs_repair = True
                    break
                _promote_rescued_attempt(
                    trace, event_index=_rescue_idx, passing_f2p=_rescue_passing,
                )
                WORKSPACE.write_json(output_path, trace)
                _log_synthesize_step(
                    "attempt-rescued", ok=True,
                    detail=f"{sid} kept last attempt (genuine own_f2p={len(_rescue_passing)}); {attempt_error}",
                )

            if not execute:
                # dry-run: quality_validate only does schema, neither persisting nor
                # executing; append once here.
                trace["events"].append(attempt_obj)
                WORKSPACE.write_json(output_path, trace)
                continue

            # Success: locate the ATTEMPT that just passed (quality_validate appends
            # internally, which is exactly the most recent matching event).
            attempt_index = max(
                i for i, ev in enumerate(trace["events"])
                if ev.get("kind") == "ATTEMPT" and ev.get("big_task_id") == plan_obj["big_task_id"]
                and str(ev.get("subtask_id") or "") == sid and (ev.get("result") or {}).get("ok") is True
            )
            EventHistory.disable_previous_failures_for_subtask(
                trace,
                attempt_index=attempt_index,
                big_task_id=plan_obj["big_task_id"],
                subtask_id=sid,
            )
            WORKSPACE.write_json(output_path, trace)
            _log_synthesize_step("attempt-verify", ok=True, detail=f"{sid} passed code+rubric")

            try:
                await run_attempt_augment_flow(
                    workspace=WORKSPACE,
                    client=augment_client or client,
                    first_call_client=first_call_client,
                    repair_client=repair_client,
                    trace=trace,
                    target_event=trace["events"][attempt_index],
                    output_path=output_path,
                    max_attempts=max_augment_retries,
                )
                trace = at.load_trace(output_path)
                augmented = trace["events"][attempt_index]
                _log_synthesize_step(
                    "attempt-augment",
                    detail=f"{sid} own_f2p={len(augmented.get('own_f2p') or [])} own_p2p={len(augmented.get('own_p2p') or [])}",
                )
            except Exception as e:
                print(f"[synthesize][warn] ATTEMPT augmentation failed for {sid}: {printable_text(e)}; continuing", flush=True)
            continue

        if plan_needs_repair:
            # A subtask of the current PLAN failed to pass within the BaseFlow.run
            # budget. We must close out this PLAN as result.ok=False, otherwise the
            # next round's latest_active_plan would treat it as a still-executing
            # active plan and reuse it forever, never reaching the repair-budget
            # decision in _decide_plan_branch.
            current_plan = trace["events"][plan_index]
            current_plan["result"] = {
                "ok": False,
                "error_kind": "SUBTASK_ATTEMPT_EXHAUSTED",
                "per_node_status": {},
                "per_node_details": {},
                "expectation_errors": {},
                "failing_node_ids": [],
                "test_failure_details": [{
                    "node_id": "<plan>",
                    "phase": "SUBTASK_ATTEMPT_EXHAUSTED",
                    "status": "failed",
                    "command": "",
                    "expectation_error": "PLAN marked failed so the next loop iteration goes through repair-budget arbitration instead of reusing this plan.",
                    "key_line": "PLAN incomplete; a subtask exhausted its ATTEMPT retry budget",
                    "output_excerpt": "",
                }],
            }
            # Disable all PLAN and ATTEMPT events of the current BigTask; only change enabled, not result.ok.
            EventHistory.disable_all_big_task_events(
                trace, current_plan["big_task_id"], reason="subtask_attempt_exhausted"
            )
            WORKSPACE.write_json(output_path, trace)
            continue

        # All subtasks of the BigTask completed successfully: *no longer write any
        # close-out marker to PLAN.result*. Whether a PLAN is "done" is now decided
        # by latest_active_plan based on subtask completeness (all subtasks having an
        # enabled+ok ATTEMPT counts as done). Here we just continue and let the top
        # loop re-evaluate: latest_active_plan now returns None ->
        # _decide_plan_branch decides escalate / stop.
        WORKSPACE.write_json(output_path, trace)
        continue

    state = trace.setdefault("state", {})
    if not state.get("terminal"):
        state["current_phase"] = "DONE"
    _attach_llm_usage(trace, client)
    WORKSPACE.write_json(output_path, trace)

    # After the instance finishes, compact + dedupe the skills library to avoid
    # unbounded growth. Uses gpt-5.4 (the same client as rubric), acquiring all
    # three phase locks then fully rewriting. Any exception is swallowed and never
    # affects the already-persisted main artifact.
    if skills_dir and repair_client is not None:
        try:
            await compact_skills_after_instance(
                client=repair_client,
                skills_dir=skills_dir,
                prompt_dir=WORKSPACE.prompt_dir,
                label=bench_id,
            )
        except Exception as e:
            print(f"[synthesize][warn] skills compaction failed: {printable_text(e)}; continuing", flush=True)

    return trace


# ══════════════════════════════════════════════════════════════════════════
# FILL-SKELETON mode: linear skeleton replay
# ══════════════════════════════════════════════════════════════════════════
#
# Background: `traces_30/*.json` are *skeletons* recovered from the usersim system
# prompt checklist -- each has only SEED (complete gold) + PLAN (goal/acceptance/
# subtasks_blueprint) + each subtask's user_query/acceptance_criteria, while the
# attempt body (patch/test_patch/own_f2p/own_p2p/result) is *entirely empty*. The
# existing --resume mechanism assumes the trace was truly run and merely broke off
# midway; applied to a skeleton it misjudges (latest_active_plan only recognizes the
# last PLAN, empty-shell ATTEMPTs pollute history, is_trace_done skips directly), so
# this path is handled separately.
#
# Approach (strictly linear, only ever giving prior information): treat the skeleton
# as a read-only source, start from a blank working trace, and incrementally rebuild
# in recovery order -- SEED (run precheck) -> append PLAN BIG_1 -> run S1..Sn in
# order -> append PLAN BIG_2 -> .... Each subtask reuses the existing
# run_attempt_impl_flow + run_attempt_augment_flow; the only difference is that it
# does *not* generate PLAN or user_query (both taken from the recovered skeleton).
# Because PLAN is appended incrementally, when processing BIG_2 the PLAN of BIG_3 is
# not yet in the trace, so split_attempt_history_failure naturally only feeds "prior"
# information and never leaks later content.
#
# Failure handling (per the user's decision): if any subtask's attempt exhausts its
# retries -> the whole data terminates immediately.


def _skeleton_structure(skeleton: dict) -> list[dict]:
    """Parse the skeleton trace into an ordered [{plan, subtasks:[{id,text,category,
    user_query,acceptance_criteria}]}], used for linear replay. PLANs are in
    appearance order, subtasks in blueprint order; each subtask's uq/ac is taken
    from the ATTEMPT event of the same (big_task_id, subtask_id) in the skeleton."""
    events = skeleton.get("events") or []
    # Build (bid, sid) -> ATTEMPT event (exactly one per subtask in the skeleton)
    att_by_key: dict[tuple, dict] = {}
    for ev in events:
        if ev.get("kind") == "ATTEMPT" and ev.get("big_task_id") != "SEED":
            key = (str(ev.get("big_task_id") or ""), str(ev.get("subtask_id") or ""))
            att_by_key.setdefault(key, ev)
    out: list[dict] = []
    for ev in events:
        if ev.get("kind") != "PLAN":
            continue
        bid = str(ev.get("big_task_id") or "")
        subs = []
        for s in (ev.get("subtasks_blueprint") or []):
            sid = str(s.get("id") or "")
            att = att_by_key.get((bid, sid)) or {}
            subs.append({
                "id": sid,
                "text": s.get("text"),
                "category": s.get("category"),
                "user_query": att.get("user_query"),
                "acceptance_criteria": att.get("acceptance_criteria"),
            })
        out.append({"plan": ev, "subtasks": subs})
    return out


def _plan_event_from_skeleton(plan: dict) -> dict:
    """Copy a persistable PLAN event out of the skeleton PLAN (result cleared so linear replay follows the blueprint)."""
    return {
        "kind": "PLAN",
        "big_task_id": plan.get("big_task_id"),
        "id": plan.get("id"),
        "enabled": True,
        "reason": plan.get("reason") or "initial",
        "round": plan.get("round") or 1,
        "natural_language_goal": plan.get("natural_language_goal"),
        "acceptance_criteria": plan.get("acceptance_criteria"),
        "subtasks_blueprint": copy.deepcopy(plan.get("subtasks_blueprint") or []),
        "dim_tags": plan.get("dim_tags"),
        "uplift_proof": plan.get("uplift_proof"),
        "own_f2p": [],
        "own_p2p": [],
        "test_patch": [],
        "result": None,
        "_recovered_from": plan.get("_recovered_from"),
    }


async def fill_skeleton_trace(
    *,
    skeleton_path: Path,
    output_path: Path,
    client,
    repair_client=None,
    augment_client=None,
    verify_client=None,
    first_call_client=None,
    image: str,
    pytest_bin: str,
    cwd: str,
    smoke_import: str | None,
    execute: bool = True,
    max_attempt_retries: int = 4,
    max_augment_retries: int = 4,
    attempt_max_rubric_failures: int = 2,
    skills_dir: str | None = None,
    dataset_instance: dict | None = None,
) -> dict:
    """Linear skeleton replay: fill in attempt/augment for a recovered skeleton. See the block comment above.

    `dataset_instance` (if provided) is the *complete* instance fetched precisely
    from the full dataset, used for init_trace / SEED materialization, replacing the
    skeleton's instance whose fields are incomplete (skeleton recovery lacks the
    Pro-specific fields before_repo_set_cmd / dockerhub_tag / source).
    """
    skeleton = WORKSPACE.read_json(skeleton_path)
    if not WORKSPACE.is_trace_payload(skeleton):
        raise ValueError(f"--fill-skeleton input is not a trace payload: {skeleton_path}")
    instance = dataset_instance if dataset_instance else (skeleton.get("instance") or {})
    bench_id = instance.get("instance_id") or (skeleton.get("instance") or {}).get("instance_id") or skeleton_path.stem
    structure = _skeleton_structure(skeleton)

    # Pre-check: every subtask must have uq/ac, otherwise replay is impossible (recovery should have guaranteed this).
    _missing = [
        f"{blk['plan'].get('big_task_id')}/{s['id']}"
        for blk in structure for s in blk["subtasks"]
        if not (isinstance(s.get("user_query"), dict) and s.get("user_query")
                and isinstance(s.get("acceptance_criteria"), str) and s.get("acceptance_criteria"))
    ]
    if _missing:
        raise ValueError(f"Skeleton missing user_query/acceptance_criteria, cannot replay: {_missing}")

    # ── Working trace: start fresh or resume from partial output ──────────────────────────────
    resuming = False
    if output_path.exists():
        try:
            _existing = WORKSPACE.read_json(output_path)
        except Exception:
            _existing = None
        if WORKSPACE.is_trace_payload(_existing) and _existing.get("config", {}).get("fill_mode"):
            trace = _existing
            resuming = True
    if not resuming:
        trace = WORKSPACE.init_trace(
            instance, image=image, pytest_bin=pytest_bin, cwd=cwd, smoke_import=smoke_import,
        )
        trace.setdefault("config", {})["fill_mode"] = True

    if hasattr(client, "set_usage_checkpoint"):
        client.set_usage_checkpoint(output_path)
    if resuming and hasattr(client, "load_usage_summary"):
        client.load_usage_summary((trace.get("state") or {}).get("llm_usage"))

    # ── SEED: ensure it exists and has run precheck ──────────────────────────────────────
    _seed_ok = any(
        ev.get("kind") == "ATTEMPT" and ev.get("big_task_id") == "SEED"
        and (ev.get("result") or {}).get("ok") is True
        for ev in (trace.get("events") or [])
    )
    if not _seed_ok:
        # Clear any leftover failed SEED, re-materialize + run precheck.
        trace["events"] = [
            ev for ev in (trace.get("events") or []) if ev.get("big_task_id") != "SEED"
        ]
        ensure_seed_attempt(trace)
        WORKSPACE.write_json(output_path, trace)
        _log_synthesize_step("fill-seed", detail=f"instance={bench_id} materialized")
        if execute:
            trace = run_seed_precheck(str(output_path), stop_on_fail=True, verbose=True)
            if trace.get("state", {}).get("terminal") == "SKIP_SEED_PRECHECK_FAILED":
                _attach_llm_usage(trace, client)
                WORKSPACE.write_json(output_path, trace)
                return trace
        else:
            trace = at.load_trace(output_path)
    else:
        _log_synthesize_step("fill-resume", detail=f"instance={bench_id} events={len(trace.get('events') or [])}")

    trace.setdefault("state", {})["current_phase"] = "ATTEMPT"
    WORKSPACE.write_json(output_path, trace)

    # ── Linear replay of each BigTask ──────────────────────────────────────────────────
    for blk in structure:
        plan = blk["plan"]
        bid = str(plan.get("big_task_id") or "")

        # PLAN: append incrementally only when the working trace has no enabled PLAN
        # for this bid yet (guaranteeing "only give prior information").
        plan_present = any(
            ev.get("kind") == "PLAN" and at.is_enabled(ev) and ev.get("big_task_id") == bid
            for ev in (trace.get("events") or [])
        )
        if not plan_present:
            trace["events"].append(_plan_event_from_skeleton(plan))
            WORKSPACE.write_json(output_path, trace)
            _log_synthesize_step("fill-plan", detail=f"{bid} nsub={len(blk['subtasks'])}")

        for sub in blk["subtasks"]:
            sid = sub["id"]

            # Resume decision (aligned with the normal path L715-770): find the most
            # recent enabled+ok=True ATTEMPT and handle it via three branches by
            # artifact completeness --
            #   - patch + own_f2p + test_patch + own_p2p all present -> complete, skip;
            #   - first three present but own_p2p missing -> ATTEMPT body usable, just do augment;
            #   - any of the first three missing -> ATTEMPT body unusable, disable then clear leftovers and regenerate.
            _done_idx = EventHistory.latest_successful_attempt_index(
                trace, big_task_id=bid, subtask_id=sid,
                before_index=len(trace.get("events") or []),
            )
            if _done_idx is not None:
                _done_ev = trace["events"][_done_idx]
                _has_patch = bool(_done_ev.get("patch"))
                _has_f2p = bool(_done_ev.get("own_f2p"))
                _has_test_patch = bool(_done_ev.get("test_patch"))
                _has_p2p = bool(_done_ev.get("own_p2p"))

                if _has_patch and _has_f2p and _has_test_patch and _has_p2p:
                    _log_synthesize_step("fill-skip", detail=f"{bid}/{sid} already filled")
                    continue

                if _has_patch and _has_f2p and _has_test_patch and not _has_p2p:
                    # ATTEMPT complete but augment missing -> just do augment, do not delete the ATTEMPT.
                    _log_synthesize_step(
                        "fill-resume-augment",
                        detail=f"{bid}/{sid} own_f2p={len(_done_ev.get('own_f2p') or [])} own_p2p=0",
                    )
                    try:
                        await run_attempt_augment_flow(
                            workspace=WORKSPACE,
                            client=augment_client or client,
                            first_call_client=first_call_client,
                            repair_client=repair_client,
                            trace=trace,
                            target_event=trace["events"][_done_idx],
                            output_path=output_path,
                            max_attempts=max_augment_retries,
                        )
                        trace = at.load_trace(output_path)
                        _aug = trace["events"][_done_idx]
                        _log_synthesize_step(
                            "fill-augment",
                            detail=f"{bid}/{sid} own_f2p={len(_aug.get('own_f2p') or [])} own_p2p={len(_aug.get('own_p2p') or [])}",
                        )
                    except Exception as e:
                        print(f"[synthesize][warn] fill augment (resume) failed for {bid}/{sid}: {printable_text(e)}; continuing", flush=True)
                    continue

                # ATTEMPT body incomplete (missing patch/f2p/test_patch) -> disable then clear leftovers and regenerate.
                _done_ev["enabled"] = False
                _done_ev.setdefault("disabled_reason", "incomplete_attempt_redo")
                WORKSPACE.write_json(output_path, trace)
                _log_synthesize_step(
                    "fill-resume-redo",
                    detail=f"{bid}/{sid} ATTEMPT incomplete (patch={_has_patch} f2p={_has_f2p} test_patch={_has_test_patch}), redoing",
                )

            # Clear this subtask's leftover unfinished/failed ATTEMPT (crash debris or last failure), regenerate from a clean state.
            trace["events"] = [
                ev for ev in (trace.get("events") or [])
                if not (
                    ev.get("kind") == "ATTEMPT"
                    and ev.get("big_task_id") == bid
                    and str(ev.get("subtask_id") or "") == sid
                )
            ]
            WORKSPACE.write_json(output_path, trace)

            user_query = EventSchema.coerce_user_query(sub["user_query"])
            acceptance_criteria = sub["acceptance_criteria"]
            subtask = {"id": sid, "text": sub.get("text"), "category": sub.get("category")}
            _log_synthesize_step("fill-subtask", detail=f"{bid}/{sid} category={sub.get('category')}", key=sub.get("text") or "")

            attempt_obj, attempt_error = await run_attempt_impl_flow(
                workspace=WORKSPACE,
                client=client,
                first_call_client=first_call_client,
                repair_client=repair_client,
                verify_client=verify_client or client,
                trace=trace,
                big_task_id=bid,
                subtask=subtask,
                user_query=user_query,
                acceptance_criteria=acceptance_criteria,
                enable_docker_cli=execute,
                enable_execute=execute,
                output_path=output_path,
                max_attempts=max_attempt_retries,
                max_rubric_failures=attempt_max_rubric_failures,
            )
            trace = at.load_trace(output_path)

            if attempt_error is not None:
                # ATTEMPT retries exhausted: first try to rescue (find the last ATTEMPT
                # of this subtask; if it still carries "genuinely passing own_f2p",
                # promote it to a usable artifact to keep, continue augment + the next
                # subtask, and do *not* terminate the whole data); only terminate when
                # there is no usable own_f2p at all. The criterion matches the helper's
                # comment (genuine f2p = per_node_status == "PASSED").
                _rescue_idx, _rescue_passing = _find_last_attempt_with_genuine_f2p(
                    trace, big_task_id=bid, subtask_id=sid,
                )
                if _rescue_idx is None or not _rescue_passing:
                    _log_synthesize_step("fill-exhausted", ok=False, detail=f"{bid}/{sid} no usable f2p (0 genuinely passing); {attempt_error}")
                    state = trace.setdefault("state", {})
                    state["current_phase"] = "ATTEMPT"
                    state["terminal"] = "FILL_SUBTASK_ATTEMPT_EXHAUSTED"
                    state["last_error"] = f"{bid}/{sid} attempt exhausted (no genuinely-passing f2p): {attempt_error}"
                    _attach_llm_usage(trace, client)
                    WORKSPACE.write_json(output_path, trace)
                    return trace
                attempt_obj = _promote_rescued_attempt(
                    trace, event_index=_rescue_idx, passing_f2p=_rescue_passing,
                )
                WORKSPACE.write_json(output_path, trace)
                _log_synthesize_step(
                    "attempt-rescued", ok=True,
                    detail=f"{bid}/{sid} kept last attempt (genuine own_f2p={len(_rescue_passing)}); {attempt_error}",
                )

            if not execute:
                trace["events"].append(attempt_obj)
                WORKSPACE.write_json(output_path, trace)
                continue

            # Success (or promoted to keep): locate this ATTEMPT -> disable earlier failures -> augment to fill own_p2p.
            attempt_index = max(
                i for i, ev in enumerate(trace["events"])
                if ev.get("kind") == "ATTEMPT" and ev.get("big_task_id") == bid
                and str(ev.get("subtask_id") or "") == sid and (ev.get("result") or {}).get("ok") is True
            )
            EventHistory.disable_previous_failures_for_subtask(
                trace, attempt_index=attempt_index, big_task_id=bid, subtask_id=sid,
            )
            WORKSPACE.write_json(output_path, trace)
            _log_synthesize_step("fill-attempt", ok=True, detail=f"{bid}/{sid} passed code+rubric")

            try:
                await run_attempt_augment_flow(
                    workspace=WORKSPACE,
                    client=augment_client or client,
                    first_call_client=first_call_client,
                    repair_client=repair_client,
                    trace=trace,
                    target_event=trace["events"][attempt_index],
                    output_path=output_path,
                    max_attempts=max_augment_retries,
                )
                trace = at.load_trace(output_path)
                _aug = trace["events"][attempt_index]
                _log_synthesize_step(
                    "fill-augment",
                    detail=f"{bid}/{sid} own_f2p={len(_aug.get('own_f2p') or [])} own_p2p={len(_aug.get('own_p2p') or [])}",
                )
            except Exception as e:
                print(f"[synthesize][warn] fill augment failed for {bid}/{sid}: {printable_text(e)}; continuing", flush=True)

    # All subtasks filled in.
    state = trace.setdefault("state", {})
    state["current_phase"] = "DONE"
    state["terminal"] = None
    state.pop("last_error", None)
    _attach_llm_usage(trace, client)
    WORKSPACE.write_json(output_path, trace)

    if skills_dir and repair_client is not None:
        try:
            await compact_skills_after_instance(
                client=repair_client,
                skills_dir=skills_dir,
                prompt_dir=WORKSPACE.prompt_dir,
                label=bench_id,
            )
        except Exception as e:
            print(f"[synthesize][warn] skills compaction failed: {printable_text(e)}; continuing", flush=True)

    return trace


# Default single-instance dataset dir (HuggingFace load_from_disk). Sourced
# from config.paths.dataset; empty → must be supplied via --dataset.
_DATASET_PATH = Path(_PATHS["dataset"]) if _PATHS.get("dataset") else None


def _apply_hf_env() -> None:
    """Honour an optional HF_HOME from config (else leave the environment as-is)
    and enable HuggingFace offline mode for local `load_from_disk` datasets."""
    hf_home = _PATHS.get("hf_home")
    if hf_home:
        os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def _load_instance_from_dataset(dataset_path: str | Path, instance_id: str) -> dict:
    """Load one instance by instance_id from a HuggingFace Dataset (load_from_disk)."""
    import json as _json
    _apply_hf_env()
    from datasets import load_from_disk
    ds = load_from_disk(str(dataset_path))
    matches = [dict(r) for r in ds if r["instance_id"] == instance_id]
    if not matches:
        available = [r["instance_id"] for r in ds]
        raise SystemExit(
            f"instance_id '{instance_id}' not found in dataset.\n"
            f"Available IDs ({len(available)}):\n" + "\n".join(f"  {x}" for x in available)
        )
    row = matches[0]
    # fail_to_pass / pass_to_pass may be lists in the dataset; normalize to JSON strings
    for k in ("fail_to_pass", "pass_to_pass"):
        if isinstance(row.get(k), list):
            row[k] = _json.dumps(row[k])
    return row


# Authoritative source of the instance in fill mode: the full dataset (including the
# Pro-specific fields before_repo_set_cmd / dockerhub_tag / source). The skeleton
# instance has incomplete fields (recovery only rebuilt the SEED gold subset), so
# fill must go back to the dataset to fetch the complete instance precisely, exactly
# like normal synthesize.
# Sourced from config.paths.full_dataset; empty → must be supplied via --dataset.
_FILL_DATASET_PATH = Path(_PATHS["full_dataset"]) if _PATHS.get("full_dataset") else None


def _load_instance_for_skeleton(dataset_path: str | Path, skeleton_instance: dict) -> dict:
    """Fetch the *complete* instance precisely from the full dataset by the skeleton instance (including Pro-specific fields).

    Double validation, never guess:
      1. instance_id match (as-is, or the Pro naming with `instance_` prefix in the dataset);
      2. then cross-confirm with (repo, base_commit); the final candidate must be unique.
    Hitting 0 or >1 candidates raises and stops directly, to avoid mismatching multiple instances in the same repo.
    """
    import json as _json
    _apply_hf_env()
    from datasets import load_from_disk

    sk_iid = str(skeleton_instance.get("instance_id") or "").strip()
    sk_repo = skeleton_instance.get("repo")
    sk_commit = skeleton_instance.get("base_commit")
    if not sk_iid:
        raise SystemExit("Skeleton instance is missing instance_id, cannot match against the dataset")

    ds = load_from_disk(str(dataset_path))
    id_keys = {sk_iid, f"instance_{sk_iid}"}
    id_hits = [dict(r) for r in ds if r.get("instance_id") in id_keys]
    if not id_hits:
        raise SystemExit(
            f"instance_id '{sk_iid}' not found in dataset (also tried prefix 'instance_{sk_iid}')"
            f"; dataset={dataset_path}"
        )

    # (repo, base_commit) cross-validation
    commit_hits = [
        r for r in id_hits
        if r.get("repo") == sk_repo and r.get("base_commit") == sk_commit
    ]
    if len(commit_hits) == 1:
        row = commit_hits[0]
    elif len(id_hits) == 1:
        # id uniquely matched but repo/commit metadata missing -> fall back (still unique, no ambiguity)
        row = id_hits[0]
        if commit_hits != id_hits:
            print(
                f"[synthesize][fill][warn] {sk_iid}: (repo,base_commit) did not fully match the dataset"
                f" (skeleton repo={sk_repo} bc={str(sk_commit)[:12]}; dataset repo={row.get('repo')}"
                f" bc={str(row.get('base_commit'))[:12]}), but instance_id uniquely matched, using it.",
                flush=True,
            )
    else:
        detail = [
            f"instance_id={r.get('instance_id')} repo={r.get('repo')} bc={str(r.get('base_commit'))[:12]}"
            for r in id_hits
        ]
        raise SystemExit(
            f"{sk_iid}: dataset match is not unique (id matched {len(id_hits)}, "
            f"(repo,base_commit) matched {len(commit_hits)}), refusing to guess. Candidates:\n  "
            + "\n  ".join(detail)
        )

    for k in ("fail_to_pass", "pass_to_pass"):
        if isinstance(row.get(k), list):
            row[k] = _json.dumps(row[k])
    return row


def _derive_image(row: dict) -> str | None:
    """Derive the Docker image from instance data.
    - Pro: use jefzda/sweap-images:{dockerhub_tag} (loaded locally from a CFS tar)
    - Verified: swebench/sweb.eval.x86_64.{iid with __ → _1776_}:latest
      e.g. astropy__astropy-13579 → swebench/sweb.eval.x86_64.astropy_1776_astropy-13579:latest
    """
    tag = (row.get("dockerhub_tag") or "").strip()
    if tag:
        return f"jefzda/sweap-images:{tag}"
    # SWE-bench Verified standard naming: __ in the instance_id is replaced with _1776_
    iid = row.get("instance_id", "")
    if iid and "__" in iid:
        slug = iid.replace("__", "_1776_")
        return f"swebench/sweb.eval.x86_64.{slug}:latest"
    return None


def _is_pro_image(image: str | None) -> bool:
    image = (image or "").strip()
    return image.startswith("jefzda/sweap-images:") or image.startswith("docker.io/jefzda/sweap-images:")


def _is_pro_instance(row: dict | None, image: str | None = None) -> bool:
    row = row or {}
    return bool((row.get("dockerhub_tag") or "").strip()) or _is_pro_image(image)


def _derive_cwd(row: dict | None = None, image: str | None = None, *, existing_cwd: str | None = None) -> str:
    """Infer the repository working directory inside the Docker image.

    SWE-bench Verified images keep the repo under /testbed. SWE-bench Pro
    jefzda/sweap-images put the repo under /app. Existing traces may already
    contain the old /testbed default for Pro, so Pro inference intentionally
    wins over existing_cwd unless the CLI passed --cwd explicitly before this
    helper is called.
    """
    if _is_pro_instance(row, image):
        return DEFAULT_PRO_CWD
    return (existing_cwd or "").strip() or DEFAULT_CWD


def _derive_tar_path(image: str) -> "Path | None":
    """Derive the local tar file path from the Docker image name (an optional offline image-loading source).

    - Pro:      jefzda/sweap-images:{tag}
                → ScaleAI__SWE-bench_Pro/docker.io_jefzda_sweap-images_{tag}.tar
    - Verified: swebench/sweb.eval.x86_64.{slug}:latest
                → SumanthRH__SWE-bench_Verified/docker.io_swebench_sweb.eval.x86_64.{slug}_latest.tar

    Returns None when the corresponding tar directory is not configured
    (paths.docker_pro_dir / paths.docker_verified_dir), i.e. offline tar loading
    is disabled and the image is expected to be pullable/present already.
    """
    if image.startswith("jefzda/sweap-images:") or image.startswith("docker.io/jefzda/sweap-images:"):
        if _DOCKER_PRO_DIR is None:
            return None
        tag = image.split(":", 1)[1]
        return _DOCKER_PRO_DIR / f"docker.io_jefzda_sweap-images_{tag}.tar"
    if image.startswith("swebench/sweb.eval.x86_64.") or image.startswith("docker.io/swebench/sweb.eval.x86_64."):
        if _DOCKER_VER_DIR is None:
            return None
        prefix = "docker.io/swebench/sweb.eval.x86_64." if image.startswith("docker.io/") else "swebench/sweb.eval.x86_64."
        rest = image[len(prefix):]
        slug = rest[:rest.index(":")] if ":" in rest else rest
        return _DOCKER_VER_DIR / f"docker.io_swebench_sweb.eval.x86_64.{slug}_latest.tar"
    return None


def _ensure_docker_image(image: str) -> None:
    """Ensure the Docker image is loaded; if it does not exist locally, load it on demand from a CFS tar."""
    import subprocess as _sp

    check = _sp.run(["docker", "images", "-q", image], capture_output=True, text=True)
    if check.returncode == 0 and check.stdout.strip():
        print(f"[docker] image already loaded: {image}", flush=True)
        return

    tar_path = _derive_tar_path(image)
    if tar_path is None:
        raise SystemExit(f"[docker] cannot derive CFS tar path for image '{image}'")
    if not tar_path.exists():
        raise SystemExit(f"[docker] CFS tar not found for image '{image}': {tar_path}")

    print(f"[docker] loading image from tar: {tar_path}", flush=True)
    result = _sp.run(["docker", "load", "-i", str(tar_path)], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"[docker] OK: {image}", flush=True)
    else:
        raise SystemExit(f"[docker] load failed for {image} (rc={result.returncode}): {result.stderr.strip()}")


def _build_clients(args, *, append_detail_log: bool, detail_log_out: Path):
    """Construct the three clients (main opus / gpt repair / first_call sglang) and attach the skills dir.
    Shared by both the synthesize and fill-skeleton paths to avoid wiring drift. Returns
    (client, gpt_client, first_call_client, skills_dir).

    Cascade: ATTEMPT/AUGMENT rounds 1-3 use first_call_client (sglang), round 4+ falls back to the main client (opus)."""
    import tool_detail_log
    detail_log_path = detail_log_out.with_name(detail_log_out.stem + ".tooldetail.log")
    tool_detail_log.configure(detail_log_path, append=append_detail_log)
    print(f"[synthesize] tool detail log → {detail_log_path}", flush=True)

    # Per-client credentials come from config.yaml (api key resolved from the
    # environment variable named by each client's api_key_env). The main
    # Anthropic client requires a key; the local server may need none.
    main_key = _MAIN_CFG.get("api_key") or args.api_key or ""
    if not main_key:
        raise SystemExit(
            "Main-client API key missing; set the environment variable named by "
            f"clients.main.api_key_env ({_MAIN_CFG.get('api_key_env')!r}) or pass --api-key."
        )

    client = _new_client(
        app_id=args.app_id, api_key=main_key,
        provider="anthropic", model=MAIN_MODEL,
        usage=args.usage, backend="anthropic", endpoint_url=MAIN_MESSAGES_URL,
        anthropic_version=MAIN_ANTHROPIC_VERSION,
    )
    gpt_client = _new_client(
        app_id=args.app_id, api_key=_REPAIR_CFG.get("api_key") or "",
        provider="openai", model=JSON_REPAIR_MODEL,
        usage=args.usage, backend=JSON_REPAIR_BACKEND,
        endpoint_url=OPENAI_CHAT_COMPLETIONS_URL, usage_owner=client,
    )

    first_call_client = _new_client(
        app_id=args.app_id, api_key=_LOCAL_CFG.get("api_key") or "",
        provider="local", model=LOCAL_DEFAULT_MODEL,
        usage=args.usage, backend=LOCAL_BACKEND,
        endpoint_url=LOCAL_ENDPOINT_URL, usage_owner=client,
    )
    print(
        "[synthesize] cascade: ATTEMPT/AUGMENT rounds 1-3 = local (first_call), "
        "round 4+ = main client",
        flush=True,
    )

    skills_dir = str(args.skills_dir or "").strip()
    if skills_dir and hasattr(client, "set_skills_dirs"):
        skills_history_dir = Path(skills_dir).parent / "skills_history"
        client.set_skills_dirs(skills_dir, skills_history_dir)
        if first_call_client is not None and first_call_client is not client and hasattr(first_call_client, "set_skills_dirs"):
            first_call_client.set_skills_dirs(skills_dir, skills_history_dir)
        print(f"[synthesize] skills library: dir={skills_dir} history={skills_history_dir}", flush=True)

    return client, gpt_client, first_call_client, (skills_dir or None)


async def _run_fill_skeleton(args) -> int:
    """--fill-skeleton entry point: parse the skeleton -> construct clients -> linear replay."""
    skeleton_path = Path(args.fill_skeleton)
    if not skeleton_path.exists():
        raise SystemExit(f"--fill-skeleton file does not exist: {skeleton_path}")
    skeleton_payload = WORKSPACE.read_json(skeleton_path)
    if not WORKSPACE.is_trace_payload(skeleton_payload):
        raise SystemExit(f"--fill-skeleton input is not a trace payload: {skeleton_path}")
    skel_cfg = skeleton_payload.get("config") or {}
    skel_inst = skeleton_payload.get("instance") or {}

    # ── Authoritative instance source: fetch the complete instance precisely from the full dataset (including Pro-specific fields) ──
    # Exactly like normal synthesize: the instance comes from the dataset, not the field-incomplete skeleton.
    dataset_path = args.dataset or _FILL_DATASET_PATH
    if not dataset_path:
        raise SystemExit(
            "No dataset available for fill mode; pass --dataset or set "
            "paths.full_dataset in config.yaml."
        )
    full_inst = _load_instance_for_skeleton(dataset_path, skel_inst)
    print(
        f"[synthesize][fill] instance from dataset: {full_inst.get('instance_id')} "
        f"(source={full_inst.get('source')} before_repo_set_cmd={'yes' if str(full_inst.get('before_repo_set_cmd') or '').strip() else 'no'} "
        f"dockerhub_tag={'yes' if str(full_inst.get('dockerhub_tag') or '').strip() else 'no'})",
        flush=True,
    )

    out = Path(args.output)
    # image/cwd derived from the complete instance (Pro with dockerhub_tag uses sweap-images)
    if args.image is None:
        args.image = _derive_image(full_inst) or skel_cfg.get("image")
    if args.cwd is None:
        args.cwd = _derive_cwd(full_inst, args.image, existing_cwd=skel_cfg.get("cwd"))
    pytest_bin = args.pytest_bin or skel_cfg.get("pytest_bin") or DEFAULT_PYTEST_BIN
    smoke_import = args.smoke_import if args.smoke_import is not None else skel_cfg.get("smoke_import")

    if not args.image:
        raise SystemExit("Cannot derive --image, please specify it manually")
    print(
        f"[synthesize][fill] image={args.image} cwd={args.cwd} smoke_import={smoke_import or '(disabled)'} "
        f"skeleton={skeleton_path} → output={out}",
        flush=True,
    )
    _ensure_docker_image(args.image)

    client, gpt_client, first_call_client, skills_dir = _build_clients(
        args, append_detail_log=out.exists(), detail_log_out=out,
    )

    trace = await fill_skeleton_trace(
        skeleton_path=skeleton_path,
        output_path=out,
        client=client,
        repair_client=gpt_client,
        augment_client=client,
        verify_client=gpt_client,
        first_call_client=first_call_client,
        image=args.image,
        pytest_bin=pytest_bin,
        cwd=args.cwd,
        smoke_import=smoke_import,
        execute=args.execute,
        max_attempt_retries=args.max_attempt_retries,
        max_augment_retries=args.max_augment_retries,
        attempt_max_rubric_failures=args.max_attempt_rubric_failures,
        skills_dir=skills_dir,
        dataset_instance=full_inst,
    )
    _attach_llm_usage(trace, client)
    WORKSPACE.write_json(out, trace)
    _log_llm_usage(client)
    term = (trace.get("state") or {}).get("terminal")
    return 1 if term else 0


async def async_main(argv: list[str] | None = None) -> int:
    """Command-line entry point: parse args, create clients, launch the synthesis flow."""
    default_api_key, default_app_id = _load_default_api_credentials()
    default_output = HERE / "trace_output.json"

    ap = argparse.ArgumentParser(description="Synthesize a flat agent_trace.v2 JSON via LLM API.")
    # ── Input source (pick one) ────────────────────────────────────────────────────
    input_grp = ap.add_mutually_exclusive_group()
    input_grp.add_argument("--input", default=None,
                           help="Path to a single instance JSON file (mutually exclusive with --dataset/--instance-id)")
    input_grp.add_argument("--dataset", default=None,
                           help="HuggingFace Dataset directory (load_from_disk), used with --instance-id; "
                                f"default path: {_DATASET_PATH}")
    ap.add_argument("--instance-id", default=None,
                    help="instance_id to select from --dataset")
    # ── Other arguments ─────────────────────────────────────────────────────────────
    ap.add_argument("--output", default=str(default_output))
    ap.add_argument("--fill-skeleton", default=None,
                    help="Linear skeleton replay mode: take a recovered skeleton trace (with SEED + PLAN + each subtask's "
                         "user_query/acceptance_criteria, attempt body entirely empty), incrementally rebuild in recovery order, "
                         "run attempt+augment normally for each subtask, giving only prior information. The instance is fetched "
                         "precisely from the full dataset (--dataset, default full_dataset), same as normal synthesize. Mutually "
                         "exclusive with --resume/--resume-from/--input. Results are written to --output (may share the skeleton "
                         "source path, i.e. in-place write-back).")
    ap.add_argument("--resume-from", default=None)
    ap.add_argument("--resume", action="store_true",
                    help="Resume from the trace file pointed to by the current --output; other args like --dataset/--instance-id still apply")
    ap.add_argument("--image", default=None,
                    help="Docker image; auto-derivable when using --dataset, also overridable manually")
    ap.add_argument("--pytest-bin", default=DEFAULT_PYTEST_BIN)
    ap.add_argument("--cwd", default=None,
                    help=f"Repo working directory inside the container; auto-derived by dataset/image "
                         f"({DEFAULT_CWD} for Verified, {DEFAULT_PRO_CWD} for Pro)")
    ap.add_argument("--smoke-import", default=None,
                    help="Optional Python smoke-import package name; by default no smoke import is run")
    ap.add_argument("--api-key", default=os.environ.get("STAGE3_API_KEY", default_api_key))
    ap.add_argument("--app-id", default=default_app_id)
    ap.add_argument("--usage", default=os.environ.get("LLM_USAGE", DEFAULT_USAGE))
    # Client wiring (endpoints/models) is fixed by config.yaml and cannot be
    # switched from the command line. ATTEMPT/AUGMENT cascade: rounds 1-3 use the
    # local (first_call) client, round 4+ falls back to the main client.
    # PLAN/verify always use the main client.
    ap.add_argument("--max-attempt-retries", type=int, default=4)
    ap.add_argument("--max-augment-retries", type=int, default=4,
                    help="Maximum attempts for the augment flow; independent of attempt.")
    ap.add_argument("--max-big-task-retries", type=int, default=1,
                    help="At most N plan cycles including the initial one; default 1 means no retry (0 repairs), failure exits this BigTask directly.")
    ap.add_argument("--max-big-tasks", type=int, default=3)
    ap.add_argument("--max-rubric-failures", type=int, default=2,
                    help="Max consecutive rubric failures allowed per BigTask in PLAN; retries terminate early once exceeded")
    ap.add_argument("--max-attempt-rubric-failures", type=int, default=3,
                    help="Max consecutive rubric failures allowed per subtask in ATTEMPT; once exceeded, accept the last rubric-failure event")
    ap.add_argument("--execute", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--refine-plan", action=argparse.BooleanOptionalAction, default=False,
                    help="After the first PLAN passes schema, trigger one difficulty-strengthening retry (does not affect repair). Off by default.")
    ap.add_argument("--skills-dir", default=str(HERE / "traces_30" / "skills"),
                    help="Host-side skills experience library directory (permanently retained); the three phases "
                         "PLAN/ATTEMPT/AUGMENT expose read_skills/patch_skills. History snapshots are written to the "
                         "sibling skills_history/. Pass an empty string to disable the skills tools.")
    args = ap.parse_args(argv)
    if args.fill_skeleton:
        if args.resume or args.resume_from or args.input:
            raise SystemExit("--fill-skeleton is mutually exclusive with --resume/--resume-from/--input")
        return await _run_fill_skeleton(args)
    if args.resume:
        if args.resume_from:
            raise SystemExit("--resume and --resume-from can only pick one")
        args.resume_from = args.output

    # ── Parse the input source ──────────────────────────────────────────────────
    using_dataset = args.dataset is not None or (args.input is None and args.instance_id is not None)
    if using_dataset:
        dataset_path = args.dataset or (str(_DATASET_PATH) if _DATASET_PATH else None)
        if not dataset_path:
            raise SystemExit(
                "No dataset available; pass --dataset or set paths.dataset in config.yaml."
            )
        if not args.instance_id:
            raise SystemExit("--instance-id must be used together with --dataset")
        instance_row = _load_instance_from_dataset(dataset_path, args.instance_id)

        # Auto-derive image / cwd (overridable from the command line). smoke import
        # is disabled by default; Pro covers multilingual repos, and auto-importing
        # by repo name would produce many false positives.
        if args.image is None:
            args.image = _derive_image(instance_row)
        if args.cwd is None:
            args.cwd = _derive_cwd(instance_row, args.image)

        # Write the instance to a temporary JSON file and pass it into the existing flow
        import json as _json
        import tempfile
        _tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix=f"{args.instance_id}_",
            delete=False, encoding="utf-8",
        )
        _json.dump(instance_row, _tmp, ensure_ascii=False)
        _tmp.close()
        args.input = _tmp.name
        print(f"[synthesize] dataset instance written → {args.input}", flush=True)
    else:
        if args.input is None and args.resume_from is None:
            raise SystemExit(
                "No input source given. Pass one of: --input <instance.json>, "
                "--dataset <dir> --instance-id <id>, --resume, or --resume-from <trace.json>."
            )

    source_path = args.resume_from or args.input
    if source_path is None:
        raise SystemExit("Need --input, --dataset/--instance-id, or --resume-from")
    source_payload = WORKSPACE.read_json(source_path)
    source_is_trace = WORKSPACE.is_trace_payload(source_payload)
    source_cfg = source_payload.get("config") if source_is_trace else {}
    if source_is_trace:
        source_instance = source_payload.get("instance") or {}
    else:
        source_instance = source_payload[0] if isinstance(source_payload, list) else source_payload
    if not isinstance(source_instance, dict):
        source_instance = {}

    if args.image is None:
        args.image = (source_cfg or {}).get("image") or _derive_image(source_instance)
    if args.cwd is None:
        args.cwd = _derive_cwd(
            source_instance,
            args.image,
            existing_cwd=(source_cfg or {}).get("cwd"),
        )

    # Load the Docker image on demand (skip if already present locally)
    if not args.image:
        raise SystemExit("Cannot derive --image, please specify it manually")
    print(
        f"[synthesize] docker config: image={args.image} cwd={args.cwd} "
        f"smoke_import={args.smoke_import or '(disabled)'}",
        flush=True,
    )
    _ensure_docker_image(args.image)
    out = Path(args.resume_from) if args.resume_from and args.output == str(default_output) else Path(args.output)

    client, gpt_client, first_call_client, skills_dir = _build_clients(
        args, append_detail_log=bool(args.resume or args.resume_from), detail_log_out=out,
    )

    trace = await synthesize_trace(
        instance_path=Path(args.input or args.resume_from),
        output_path=out,
        client=client,
        repair_client=gpt_client,
        augment_client=client,
        user_query_client=gpt_client,
        verify_client=gpt_client,
        first_call_client=first_call_client,
        image=args.image,
        pytest_bin=args.pytest_bin,
        cwd=args.cwd,
        smoke_import=args.smoke_import,
        execute=args.execute,
        max_attempt_retries=args.max_attempt_retries,
        max_augment_retries=args.max_augment_retries,
        max_big_task_retries=args.max_big_task_retries,
        max_big_tasks=args.max_big_tasks,
        max_rubric_failures=args.max_rubric_failures,
        attempt_max_rubric_failures=args.max_attempt_rubric_failures,
        resume_trace_path=Path(args.resume_from) if args.resume_from else None,
        refine_plan=args.refine_plan,
        skills_dir=(skills_dir or None),
    )
    _attach_llm_usage(trace, client)
    WORKSPACE.write_json(out, trace)
    _log_llm_usage(client)
    return 0


if __name__ == "__main__":
    asyncio.run(async_main())
