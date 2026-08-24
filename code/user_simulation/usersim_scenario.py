"""
usersim_scenario.py
===================

Build the hidden user-sim scenario from a trace — moved out of mini-swe-agent
(trace_runner_v2._build_subtask_specs / _build_user_scenario / helpers) so that
ALL user-sim logic lives in user_simulation/. Depends only on:
  - agent_trace (trace_synth/, for is_enabled)
  - user_simulator (UserScenario / StructuredCodingUserInstructions)

Public API:
  build_subtask_specs(trace) -> list[spec]
  build_user_scenario(specs, persona) -> UserScenario
"""

from __future__ import annotations

import sys
import textwrap
from collections import OrderedDict
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
# Relative wiring: ../trace_synth provides agent_trace and other low-level modules; user_simulator lives in this package.
import config as _cfg  # noqa: E402
_cfg.setup_paths()

import agent_trace as at  # noqa: E402
from user_simulator import UserScenario  # noqa: E402


def _enabled_ok_attempts(trace: dict) -> list[dict]:
    """All ENABLED ATTEMPT events, in trace order.

    Selection is by ``enabled`` ONLY — identical to ``trace_runner.extract_tasks``
    (the task list the assistant agent actually receives). We deliberately do NOT
    additionally require ``result.ok is True`` here: the user-sim's subtask set,
    the gold precheck set, and the agent's task set must be the SAME set, so that
    "gold precheck passed" means exactly the subtasks the agent is asked to do.

    The host-injected SEED/SEED_TASK attempt IS included: it is a normal,
    testable subtask (carries test_patch / own_f2p like any other). Only its
    reference `patch` is sensitive, and spec assembly never copies `patch`, so
    nothing implementation-revealing leaks into the scenario.
    """
    out: list[dict] = []
    for ev in trace.get("events") or []:
        if ev.get("kind") != "ATTEMPT":
            continue
        if not at.is_enabled(ev):
            continue
        out.append(ev)
    return out


def _big_task_goals(trace: dict) -> dict:
    """big_task_id → {goal, acceptance} from each big task's latest enabled PLAN."""
    out: dict = {}
    for ev in trace.get("events") or []:
        if ev.get("kind") != "PLAN":
            continue
        if ev.get("big_task_id") == "SEED":
            continue
        if not at.is_enabled(ev):
            continue
        bid = ev.get("big_task_id", "UNKNOWN")
        out[bid] = {
            "goal": (ev.get("natural_language_goal") or "").strip(),
            "acceptance": (ev.get("acceptance_criteria") or "").strip(),
        }
    return out


def build_subtask_specs(trace: dict) -> list[dict]:
    """Each enabled&ok ATTEMPT → one testable subtask spec (trace order)."""
    big_goals = _big_task_goals(trace)
    specs: list[dict] = []
    for ev in _enabled_ok_attempts(trace):
        bid = ev.get("big_task_id", "UNKNOWN")
        sid = ev.get("subtask_id", "")
        uq = ev.get("user_query")
        goal = (uq.get("content") if isinstance(uq, dict) else "") or ""
        bg = big_goals.get(bid, {})
        specs.append({
            "id": f"{bid}/{sid}",
            "big_task_id": bid,
            "subtask_id": sid,
            "label": f"{bid}/{sid}",
            "big_goal": bg.get("goal", ""),
            "big_acceptance": bg.get("acceptance", ""),
            "goal": goal.strip(),
            "acceptance": (ev.get("acceptance_criteria") or "").strip(),
            "test_patch": list(ev.get("test_patch") or []),
            "own_f2p": list(ev.get("own_f2p") or []),
            "own_p2p": list(ev.get("own_p2p") or []),
        })
    return specs


def build_user_scenario(specs: list[dict], persona: str) -> UserScenario:
    """Pure data assembly: compute the user_sim_call.txt placeholder values.

    No prompt prose lives here. ALL user-sim wording (reason-for-call, known/
    unknown info, task instructions, dimension switching) is authored in
    prompts/user_sim_call*.txt and rendered by usersim_driver via {{VAR}} +
    {{#if_D<n> ...}} conditionals. This function only builds the {{...}} values
    that the txt template consumes: NUM_BIG_TASKS, NUM_SUBTASKS, HIDDEN_CHECKLIST.
    (There is no {{PERSONA}} slot anymore — persona behaviour is fully expressed
    through the D1–D5 dimension blocks.)

    The SEED/SEED_TASK subtask (original SWE-bench problem statement) flows
    through `specs` like any other testable subtask, so it appears as the first
    checklist group in trace order and is fully runnable via the subtask tools.
    """
    groups: "OrderedDict[str, list]" = OrderedDict()
    for s in specs:
        groups.setdefault(s["big_task_id"], []).append(s)

    block_lines = []
    for gi, (bid, items) in enumerate(groups.items()):
        big_goal = items[0].get("big_goal") or ""
        big_acc = items[0].get("big_acceptance") or ""
        if big_goal:
            block_lines.append(f"Task {gi + 1}: overall goal: {big_goal}")
        if big_acc:
            block_lines.append(f"  overall acceptance: {big_acc}")
        block_lines.append("  subtasks:")
        for si, s in enumerate(items):
            block_lines.append(
                f"  [{gi + 1}.{si + 1}] goal: {s['goal']}\n"
                f"        acceptance: {s['acceptance'] or '(none stated)'}"
            )
        block_lines.append("")
    checklist = "\n".join(block_lines).rstrip()

    template_vars = {
        "NUM_BIG_TASKS": str(len(groups)),
        "NUM_SUBTASKS": str(len(specs)),
        "HIDDEN_CHECKLIST": checklist,
    }
    return UserScenario(instructions="", persona=persona, template_vars=template_vars)
