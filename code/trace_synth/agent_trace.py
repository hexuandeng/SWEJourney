#!/usr/bin/env python3
"""
agent_trace.py  (v2: PLAN + ATTEMPT only)
=========================================

Single-file, append-only, human-readable JSON trace for the agent loop.

A trace is the *only* on-disk artifact between turns. Producers (an LLM, the
human, or a stub) read it to learn the current state, then append new events
(PLAN and ATTEMPT). The executor reads it, runs pending ATTEMPT events, and
writes results back in-place. The main flow finalizes PLAN results.

Design decisions (locked):
  - Only two event kinds: PLAN and ATTEMPT.
  - PLAN has a `result` field that the main flow fills after all enabled
    subtasks complete successfully.
  - ATTEMPT stores its executable body as flat top-level fields plus result.
  - Multiple BigTasks chain via big_task_id = "BIG_1", "BIG_2", ...
    Escalation appends a new PLAN with a fresh big_task_id; repair appends a
    new PLAN reusing the current big_task_id (round += 1).
  - BigTask <-> multi-turn conversation: one BigTask corresponds to ONE
    complete multi-turn user request; each planned conversation unit projects
    to ONE subtask and later ONE ATTEMPT. An
    erroneous_request_then_retract subtask is still a single blueprint item;
    its ATTEMPT simply contains a two-turn wrong-then-rollback sequence.
    Number of subtasks is therefore >= 1, no upper
    bound; ordering follows conversation time (each subtask depends on the
    accumulated worktree state of its predecessors).
  - cumulative_regression is NOT stored in state; it is recomputed from
    events on demand (compute_cumulative_regression). It is *partitioned*
    by big_task_id: the regression set of BIG_k does NOT bleed into
    BIG_{k+1} (see AGENT_LOOP_DESIGN.md section 0.2 rule R-B).
  - No COMMIT, no ROLLBACK events.

Top-level structure
-------------------

{
  "schema":  "agent_trace.v2",
  "instance": { ...subset of the original SWE-bench instance JSON... },
  "config":  {
     "image": "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest",
     "pytest_bin": "/opt/miniconda3/envs/testbed/bin/pytest",
     "cwd": "/testbed",
     "smoke_import": "astropy"
  },
  "state": {"current_phase": "INIT|PLAN|ATTEMPT|DONE|GIVE_UP", "terminal": null},
  "events": [PLAN/ATTEMPT events in append order]
}

Event schemas (latest flat form)
--------------------------------

PLAN:
  kind="PLAN", big_task_id, enabled, reason, round,
  natural_language_goal, test_patch: list[str], acceptance_criteria,
  own_f2p, own_p2p, dim_tags, uplift_proof,
  subtasks_blueprint: [{id, text, category}, ...],
  result: null | ExecuteReport dict

ATTEMPT:
  kind="ATTEMPT", big_task_id, subtask_id, enabled,
  natural_language_goal, test_patch: list[str], acceptance_criteria,
  own_f2p, own_p2p, user_query: {role, content},
  patch: list of SubPatch, rationale,
  result: null | ExecuteReport dict


This module is intentionally small and dependency-free; it only knows how to
read/write the trace file. The actual docker / pytest work happens in
local_test_executor.py.
"""

from __future__ import annotations

import datetime as _dt
import ast
import json
import os
from pathlib import Path
from typing import Any, Optional

SCHEMA_VERSION = "agent_trace.v2"


# =============================================================================
# 1. low-level read / write
# =============================================================================
def _now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _atomic_write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(obj, indent=2, ensure_ascii=False)
    tmp.write_text(text, encoding="utf-8", errors="backslashreplace")
    os.replace(tmp, path)


def load_trace(path: str | os.PathLike) -> dict:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def save_trace(path: str | os.PathLike, trace: dict) -> None:
    _atomic_write_json(Path(path), trace)


# =============================================================================
# 2. trace bootstrapping from a raw SWE-bench instance JSON
# =============================================================================
_INSTANCE_KEEP_KEYS = (
    "repo", "instance_id", "base_commit",
    "problem_statement", "hints_text",
    "version", "FAIL_TO_PASS", "PASS_TO_PASS",
    "environment_setup_commit", "difficulty",
    # we keep gold patches too; producers must NOT read them, but
    # auditors / replay tooling may want them.
    "patch", "test_patch",
)


def _parse_swebench_test_list(raw: Any) -> list[str]:
    """SWE-bench encodes FAIL_TO_PASS / PASS_TO_PASS as a JSON-encoded string."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return list(raw)
    if isinstance(raw, str):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(raw)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except Exception:
                pass
        return [raw]
    return []


def init_trace_from_swebench(
    swebench_json_path: str | os.PathLike,
    output_trace_path: str | os.PathLike,
    *,
    image: str,
    pytest_bin: str = "/opt/miniconda3/envs/testbed/bin/pytest",
    cwd: str = "/testbed",
    smoke_import: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    """Create a fresh trace file from a single-instance SWE-bench JSON.

    The trace file is the *single source of truth* from this point on.
    """
    out = Path(output_trace_path)
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} already exists; pass overwrite=True")

    raw = json.loads(Path(swebench_json_path).read_text(encoding="utf-8"))
    inst = {k: raw[k] for k in _INSTANCE_KEEP_KEYS if k in raw}
    inst["FAIL_TO_PASS_parsed"] = _parse_swebench_test_list(raw.get("FAIL_TO_PASS"))
    inst["PASS_TO_PASS_parsed"] = _parse_swebench_test_list(raw.get("PASS_TO_PASS"))

    # Best-effort default smoke target: first python package name from FAIL_TO_PASS.
    if smoke_import is None:
        ftp = inst["FAIL_TO_PASS_parsed"]
        if ftp and "/" in ftp[0]:
            smoke_import = ftp[0].split("/", 1)[0]
        else:
            smoke_import = (raw.get("repo") or "").split("/")[-1]

    trace = {
        "schema": SCHEMA_VERSION,
        "instance": inst,
        "config": {
            "image": image,
            "pytest_bin": pytest_bin,
            "cwd": cwd,
            "smoke_import": smoke_import,
            "test_runner_kind": (
                "pro_official"
                if str(image or "").startswith(
                    ("jefzda/sweap-images:", "docker.io/jefzda/sweap-images:")
                )
                or bool((raw.get("dockerhub_tag") or "").strip())
                else "pytest"
            ),
            "created_at": _now_iso(),
        },
        "state": {
            "current_phase": "INIT",
            "terminal": None,
        },
        "events": [],
    }
    save_trace(out, trace)
    return out


# -----------------------------------------------------------------------------
# PLAN
# -----------------------------------------------------------------------------
_VALID_REASONS = {"initial", "repair", "escalate"}


# =============================================================================
# 4. queries (derived state)
# =============================================================================
def is_enabled(ev: dict) -> bool:
    """Events are enabled by default; enabled=false marks abandoned paths."""
    return ev.get("enabled") is not False


def current_plan(trace: dict) -> Optional[dict]:
    """The most recently appended PLAN event (any big_task_id)."""
    for ev in reversed(trace["events"]):
        if ev.get("kind") == "PLAN" and ev.get("reason") in _VALID_REASONS and is_enabled(ev):
            return ev
    return None


def attempt_own_tests(ev: dict) -> tuple[list[str], list[str]]:
    """Return (own_f2p, own_p2p) in the flat ATTEMPT schema."""
    return list(ev.get("own_f2p") or []), list(ev.get("own_p2p") or [])


def plan_own_tests(ev: dict) -> tuple[list[str], list[str]]:
    """Return (own_f2p, own_p2p) in the flat PLAN schema."""
    return list(ev.get("own_f2p") or []), list(ev.get("own_p2p") or [])


def event_index(trace: dict, target: dict) -> int:
    for i, ev in enumerate(trace.get("events", [])):
        if ev is target:
            return i
    return -1


def find_pending_attempts(trace: dict) -> list[dict]:
    """ATTEMPT events with result is None."""
    return [ev for ev in trace["events"]
            if ev.get("kind") == "ATTEMPT" and is_enabled(ev) and ev.get("result") is None]


# -----------------------------------------------------------------------------
# Cumulative regression (recomputed from events; not stored in state)
# -----------------------------------------------------------------------------
def compute_cumulative_regression(
    trace: dict, *, big_task_id: Optional[str] = None,
) -> list[dict]:
    """Return per-event prior-regression entries for each successful ATTEMPT
    in the BigTask that owns tests. Each entry::

        {"test_patch": list[str], "own_f2p": list[str], "own_p2p": list[str]}

    preserves the event's own test_patch so the executor can apply/run/reset
    each event's tests independently under its own test_patch while keeping
    F2P/P2P semantics intact.

    When ``big_task_id`` is given, ONLY events belonging to that BigTask are
    considered; regression sets do not bleed across BigTask boundaries.
    """
    out: list[dict] = []

    for ev in trace.get("events", []):
        if not is_enabled(ev):
            continue
        if big_task_id is not None and ev.get("big_task_id") != big_task_id:
            continue
        res = ev.get("result") or {}
        if not res.get("ok"):
            continue
        if ev.get("kind") != "ATTEMPT":
            continue
        f2p, p2p = attempt_own_tests(ev)
        seen_f2p: set[str] = set()
        own_f2p: list[str] = []
        for n in list(f2p):
            if isinstance(n, str) and n and n not in seen_f2p:
                seen_f2p.add(n)
                own_f2p.append(n)
        seen_p2p: set[str] = set(own_f2p)
        own_p2p: list[str] = []
        for n in list(p2p):
            if isinstance(n, str) and n and n not in seen_p2p:
                seen_p2p.add(n)
                own_p2p.append(n)
        if own_f2p or own_p2p:
            out.append({
                "test_patch": ev.get("test_patch") or [],
                "own_f2p": own_f2p,
                "own_p2p": own_p2p,
            })
    return out


# =============================================================================
# 5. CLI smoke
# =============================================================================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", help="path to a single-instance SWE-bench JSON")
    ap.add_argument("--out",  help="path to write the trace JSON")
    ap.add_argument("--image", required=False,
                    default="swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest")
    ap.add_argument("--show", help="print summary of an existing trace")
    args = ap.parse_args()
    if args.init and args.out:
        p = init_trace_from_swebench(args.init, args.out,
                                     image=args.image, overwrite=True)
        print(f"[init] -> {p}")
    elif args.show:
        t = load_trace(args.show)
        print(f"schema={t['schema']}  phase={t['state']['current_phase']}  "
              f"events={len(t['events'])}  "
              f"regression={sum(len(e.get('own_f2p', [])) + len(e.get('own_p2p', [])) for e in compute_cumulative_regression(t))}")
        start = max(0, len(t["events"]) - 12)
        for i, ev in enumerate(t["events"][-12:], start=start):
            res = ev.get("result")
            ok = "" if res is None else ("ok" if res.get("ok") else "fail")
            tag = (ev.get("big_task_id", "-") +
                   "/" + (ev.get("subtask_id") or ev.get("reason") or "-"))
            print(f"  @{i:>3} {ev['kind']:<7}  {tag:<22}  {ok}")
    else:
        ap.print_help()
