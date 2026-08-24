"""
live_subtask_test_tool.py
=========================

A user-side tool (``UserTool``) that lets the *simulated user* run a single
gold subtask's hidden tests against the coding agent's **live** docker container,
on its CURRENT worktree state.

Design (see runs/USER_SIM_PLAN.md §3):

- name = ``run_subtask_tests``, one argument: ``subtask_id``.
- Reuses ``local_test_executor.run_tests_under_test_patch`` — the exact same
  chokepoint the synthesize/ATTEMPT path uses.
- DIFFERENCE vs. the synthesize path: we do NOT reset to base + replay gold.
  We test the agent's actual current worktree. To guarantee the test run does
  not pollute the agent's workspace we snapshot the docker worktree BEFORE the
  run and restore it AFTER (in addition to the snapshot/restore that lives
  inside ``run_tests_under_test_patch`` itself — this outer guard is explicit
  defense-in-depth requested by the plan).
- Maintains a shared three-state status dict:
    status[subtask_id] in {"pending"(not done), "to_test"(awaiting test), "done"(done)}
  pass  -> status[subtask_id] = "done"
  fail  -> status[subtask_id] = "pending", report failing node ids + error tail.

The tool returns a JSON string that the user simulator grounds its next message
in (couples with D2-08 execution-feedback-driven dialogue).
"""

from __future__ import annotations

import json
from typing import Any, Optional

from user_simulator import UserTool

# local_test_executor / swebench_dockerlib are importable because the caller
# (trace_runner_v2) puts trace_synth/ + SWE-bench on sys.path via _ensure_extend_path().
from local_test_executor import run_tests_under_test_patch
from infra_failure_classify import classify_infra_failure
from swebench_dockerlib import PatchApplyError

FAILURE_TAIL_CHARS = 10000


def _project_report_failure_details(report, f2p: list, p2p: list) -> list[dict]:
    """Project a TestPatchRunReport into synthesize-style ``test_failure_details``.

    Mirrors the per-node entry shape that ``execute_attempt`` /
    ``TraceWorkspace.project_result`` emit for the synthesize retry prompt:
    ``{node_id, phase, status, expectation_error, output_excerpt}``.

    Why this matters: ``run_tests_under_test_patch`` returns rich
    ``per_node_details`` + ``expectation_errors``, but the old tool return threw
    them away and only surfaced ``f2p/p2p`` statuses + a truncated ``output_tail``.
    That collapsed a package-level ``[setup failed]`` / ``[build failed]`` (e.g.
    missing go module, DNS ConnectionError, ImportError during collection) into a
    bare ``NOT_FOUND`` with no root cause. Projecting per-node details — and, for
    NOT_FOUND nodes whose own detail is empty, falling back to the shared output
    excerpt — makes the real setup failure visible, exactly as synthesize shows it.
    """
    details = report.per_node_details or {}
    exp_errs = report.expectation_errors or {}
    per_status = report.per_node_status or {}
    failing = set(str(n) for n in (report.failing_node_ids or []))
    out: list[dict] = []
    for node in list(f2p) + list(p2p):
        node = str(node)
        nd = details.get(node) if isinstance(details.get(node), dict) else {}
        status = nd.get("status") or per_status.get(node) or "NO_STATUS"
        if status == "PASSED":
            continue  # only report the failing/missing nodes, like synthesize
        out_exc = nd.get("output_excerpt") or nd.get("output") or ""
        exp_err = exp_errs.get(node) or ""
        # NOT_FOUND / COLLECT_MISS often carry no per-node detail but the shared
        # pytest/run_script output holds the real root cause (ImportError, go
        # `[setup failed]`, ConnectionError). Surface it so the cause isn't lost.
        if not out_exc and not exp_err:
            out_exc = (report.pytest_output or "")[-FAILURE_TAIL_CHARS:]
        if not exp_err and out_exc:
            exp_err = out_exc
        out.append({
            "node_id": node,
            "phase": "live_subtask_test",
            "status": status,
            "expectation_error": exp_err,
            "output_excerpt": out_exc,
        })
    return out



class LiveSubtaskTestTool(UserTool):
    """Run one gold subtask's own_f2p+own_p2p against the live container."""

    name = "run_subtask_tests"
    description = (
        "Run the hidden tests for ONE subtask against the current code state and "
        "return per-node PASSED/FAILED results. Call this to verify whether a "
        "subtask the agent claims to have finished actually works. Pass exactly "
        "one subtask_id at a time."
    )
    parameters = {
        "type": "object",
        "properties": {
            "subtask_id": {
                "type": "string",
                "description": "The id of the subtask to verify (one of the known subtask ids).",
            }
        },
        "required": ["subtask_id"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        session,
        subtask_specs: dict[str, dict],
        status: dict[str, str],
        instance: dict,
        base_commit: str,
        cwd: str,
        pytest_bin: str,
        timeout_sec: int = 1800,
    ):
        """
        Parameters
        ----------
        session : DockerSession
            Points at the coding agent's live container (see
            trace_runner_v2._make_executor_session).
        subtask_specs : dict[subtask_id -> {"test_patch", "own_f2p", "own_p2p", "label"}]
            Per-subtask hidden test spec, from build_gold_subtasks().
        status : dict[subtask_id -> str]
            Shared three-state dict. Mutated in place: pass->done, fail->pending.
        instance : dict
            Carries _config_image / _test_runner_kind so the runner routes to the
            Pro / Verified official harness when applicable.
        base_commit, cwd, pytest_bin, timeout_sec :
            Forwarded verbatim to run_tests_under_test_patch.
        """
        self.session = session
        self.subtask_specs = subtask_specs
        self.status = status
        self.instance = instance
        self.base_commit = base_commit
        self.cwd = cwd
        self.pytest_bin = pytest_bin
        self.timeout_sec = timeout_sec

    def run(self, arguments: dict[str, Any]) -> str:
        subtask_id = str(arguments.get("subtask_id", "")).strip()
        spec = self.subtask_specs.get(subtask_id)
        if spec is None:
            return json.dumps(
                {
                    "ok": False,
                    "error": f"unknown subtask_id: {subtask_id!r}",
                    "available_subtask_ids": sorted(self.subtask_specs.keys()),
                },
                ensure_ascii=False,
            )

        f2p = list(spec.get("own_f2p") or [])
        p2p = list(spec.get("own_p2p") or [])
        if not f2p and not p2p:
            self.status[subtask_id] = "done"
            return json.dumps(
                {
                    "ok": True,
                    "subtask_id": subtask_id,
                    "note": "subtask declares no testable nodes; treated as done.",
                    "status": self.status[subtask_id],
                },
                ensure_ascii=False,
            )

        # Explicit outer snapshot/restore so the test run cannot pollute the
        # agent's live worktree (plan §3). run_tests_under_test_patch also
        # snapshots/restores internally; this is intentional defense-in-depth.
        outer_snap = self.session.fs_snapshot(self.cwd)
        try:
            report = run_tests_under_test_patch(
                self.session,
                test_patch=spec.get("test_patch") or "",
                f2p_tests=f2p,
                p2p_tests=p2p,
                base_commit=self.base_commit or "HEAD",
                pytest_bin=self.pytest_bin,
                cwd=self.cwd,
                timeout_sec=self.timeout_sec,
                instance=self.instance,
                collect_node_details=True,
            )
        except PatchApplyError as e:
            self.status[subtask_id] = "pending"
            return json.dumps(
                {
                    "ok": False,
                    "subtask_id": subtask_id,
                    "error_kind": "TEST_PATCH_APPLY_FAIL",
                    "error": str(e)[:FAILURE_TAIL_CHARS],
                    "status": self.status[subtask_id],
                },
                ensure_ascii=False,
            )
        except Exception as e:  # keep simulator alive; surface as grounded evidence
            self.status[subtask_id] = "pending"
            return json.dumps(
                {
                    "ok": False,
                    "subtask_id": subtask_id,
                    "error_kind": "RUNNER_ERROR",
                    "error": repr(e)[:FAILURE_TAIL_CHARS],
                    "status": self.status[subtask_id],
                },
                ensure_ascii=False,
            )
        finally:
            try:
                self.session.fs_restore(self.cwd, outer_snap)
            finally:
                self.session.fs_drop_snapshot(outer_snap)

        status_map = report.per_node_status or {}
        f2p_status = {n: status_map.get(n, "NO_STATUS") for n in f2p}
        p2p_status = {n: status_map.get(n, "NO_STATUS") for n in p2p}
        failing = list(report.failing_node_ids or [])
        ok = bool(report.ok)

        self.status[subtask_id] = "done" if ok else "pending"

        return json.dumps(
            {
                "ok": ok,
                "subtask_id": subtask_id,
                "label": spec.get("label", subtask_id),
                "status": self.status[subtask_id],
                "f2p": f2p_status,
                "p2p": p2p_status,
                "failing_node_ids": failing,
                "output_tail": (report.pytest_output or "")[-FAILURE_TAIL_CHARS:] if not ok else "",
            },
            ensure_ascii=False,
        )


# =============================================================================
# Standalone-runner three-tool suite (pending/done/failed; no "to_test" state)
# =============================================================================
#
# The standalone usersim_runner replaces the single legacy run_subtask_tests with
# three explicit tools sharing one host-side status dict:
#
#   1. RunSubtaskTestTool  (run_subtask_test) — run hidden tests, REPORT only,
#      never mutates status.
#   2. MarkSubtaskTool     (mark_subtask)     — explicitly set status=done|failed.
#   3. ReadStatusTool      (read_status)      — read the full pending/done/failed
#      snapshot.
#
# States: "pending"(not done) / "done"(done) / "failed"(failed). All start "pending".
# Termination (handled by the runner, not here): no subtask left "pending", OR
# confidence hits the floor, OR max_user_turns.

_VALID_MARK_STATES = ("done", "failed")


class RunSubtaskTestTool(UserTool):
    """Run ONE subtask's hidden tests against the live container. REPORT only.

    Unlike the legacy LiveSubtaskTestTool, this NEVER mutates the shared status
    dict — the model decides what to do with the result and must call
    mark_subtask explicitly. Reuses the exact same snapshot/restore + executor
    chokepoint as LiveSubtaskTestTool for faithful, non-polluting test runs.
    """

    name = "run_subtask_test"
    description = (
        "Run the hidden tests for ONE subtask against the agent's CURRENT code and "
        "read the real PASSED/FAILED result. It REPORTS the result only; it does NOT "
        "change any subtask's status. Pass exactly one subtask_id at a time."
    )
    parameters = {
        "type": "object",
        "properties": {
            "subtask_id": {
                "type": "string",
                "description": "The id of the subtask to test (one of the known subtask ids).",
            }
        },
        "required": ["subtask_id"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        session,
        subtask_specs: dict[str, dict],
        status: dict[str, str],
        instance: dict,
        base_commit: str,
        cwd: str,
        pytest_bin: str,
        timeout_sec: int = 1800,
    ):
        self.session = session
        self.subtask_specs = subtask_specs
        self.status = status  # shared, but NOT mutated by this tool
        self.instance = instance
        self.base_commit = base_commit
        self.cwd = cwd
        self.pytest_bin = pytest_bin
        self.timeout_sec = timeout_sec

    def run(self, arguments: dict[str, Any]) -> str:
        subtask_id = str(arguments.get("subtask_id", "")).strip()
        spec = self.subtask_specs.get(subtask_id)
        if spec is None:
            return json.dumps(
                {
                    "ok": False,
                    "error": f"unknown subtask_id: {subtask_id!r}",
                    "available_subtask_ids": sorted(self.subtask_specs.keys()),
                },
                ensure_ascii=False,
            )

        f2p = list(spec.get("own_f2p") or [])
        p2p = list(spec.get("own_p2p") or [])
        if not f2p and not p2p:
            # No testable nodes: report PASS-by-default, but do NOT mutate status.
            return json.dumps(
                {
                    "ok": True,
                    "subtask_id": subtask_id,
                    "note": "subtask declares no testable nodes; tests vacuously pass. "
                            "Call mark_subtask to set its status.",
                    "current_status": self.status.get(subtask_id, "pending"),
                },
                ensure_ascii=False,
            )

        outer_snap = self.session.fs_snapshot(self.cwd)
        try:
            report = run_tests_under_test_patch(
                self.session,
                test_patch=spec.get("test_patch") or "",
                f2p_tests=f2p,
                p2p_tests=p2p,
                base_commit=self.base_commit or "HEAD",
                pytest_bin=self.pytest_bin,
                cwd=self.cwd,
                timeout_sec=self.timeout_sec,
                instance=self.instance,
                collect_node_details=True,
            )
        except PatchApplyError as e:
            return json.dumps(
                {
                    "ok": False,
                    "subtask_id": subtask_id,
                    "error_kind": "TEST_PATCH_APPLY_FAIL",
                    "error": str(e)[:FAILURE_TAIL_CHARS],
                    "current_status": self.status.get(subtask_id, "pending"),
                },
                ensure_ascii=False,
            )
        except Exception as e:
            return json.dumps(
                {
                    "ok": False,
                    "subtask_id": subtask_id,
                    "error_kind": "RUNNER_ERROR",
                    "error": repr(e)[:FAILURE_TAIL_CHARS],
                    "current_status": self.status.get(subtask_id, "pending"),
                },
                ensure_ascii=False,
            )
        finally:
            try:
                self.session.fs_restore(self.cwd, outer_snap)
            finally:
                self.session.fs_drop_snapshot(outer_snap)

        status_map = report.per_node_status or {}
        failing = list(report.failing_node_ids or [])
        ok = bool(report.ok)

        # Classify infra (non-test) failures — missing dependency, compile
        # error, or suite crash — across Go/Python/JS/TS, so the user-sim can
        # ground its feedback in the real root cause instead of a bare NOT_FOUND
        # that reads as "selector did not match". `is_test_layer` tells the
        # user-sim whether the detail is safe to relay concretely (False:
        # build/dep/runtime info the agent can also see) or must be paraphrased
        # (True: cites a hidden test-patch symbol the agent cannot see).
        infra = None if ok else classify_infra_failure(report.pytest_output or "")

        # When the failure is infra-level, a bare per-node NOT_FOUND is
        # MISLEADING: the node never ran because the suite failed to build/crash,
        # not because the selector missed. Re-label those NOT_FOUND nodes to
        # ERROR so the f2p/p2p map is consistent with infra_failure (the agent is
        # never told a node "wasn't found" when the real cause is a build/crash).
        # Display-only; does NOT touch the synthesize/executor path in trace_synth/.
        if infra is not None:
            status_map = {
                n: ("ERROR" if s == "NOT_FOUND" else s)
                for n, s in status_map.items()
            }
        f2p_status = {n: status_map.get(n, "NO_STATUS") for n in f2p}
        p2p_status = {n: status_map.get(n, "NO_STATUS") for n in p2p}

        if ok:
            hint = "tests passed — call mark_subtask(status=done)"
        elif infra and not infra.get("is_test_layer"):
            causes = "; ".join(infra.get("causes") or [])
            hint = (
                f"infra failure ({infra['kind']}, lang={infra.get('language') or '?'}): "
                f"this is NOT a test-selector miss — the suite never produced a "
                f"result for these nodes because the code under test failed to "
                f"build/run. Likely cause: {causes or 'see evidence'}. This is "
                f"build/runtime info the agent can see in its OWN container, so "
                f"you MAY relay it concretely in natural language. Suggested fix "
                f"to convey: {infra.get('actionable') or 'see evidence'}."
            )
        elif infra:  # is_test_layer
            hint = (
                f"infra failure ({infra['kind']}): the failure cites a symbol that "
                f"lives ONLY in the hidden tests and the agent CANNOT see it. "
                f"Paraphrase the symptom in plain user language (e.g. 'a helper "
                f"seems to be missing'); never paste the raw error, file name, or "
                f"symbol."
            )
        else:
            hint = "tests failed — fix via the agent, or mark_subtask(status=failed) to give up"

        # Align the failure payload to synthesize's project_result style so the
        # user-sim sees the SAME structured per-node diagnostics the synthesize
        # retry prompt sees (root cause of [setup failed]/[build failed] kept).
        return json.dumps(
            {
                "ok": ok,
                "subtask_id": subtask_id,
                "label": spec.get("label", subtask_id),
                "current_status": self.status.get(subtask_id, "pending"),
                "error_kind": None if ok else "PYTEST_FAIL",
                "infra_failure": infra,
                "f2p": f2p_status,
                "p2p": p2p_status,
                "failing_node_ids": failing,
                "test_failure_details": [] if ok
                    else _project_report_failure_details(report, f2p, p2p),
                "hint": hint,
            },
            ensure_ascii=False,
        )


class MarkSubtaskTool(UserTool):
    """Explicitly set a subtask's status to done or failed in the shared dict."""

    name = "mark_subtask"
    description = (
        "Explicitly set ONE subtask's status to 'done' or 'failed' based on your own "
        "judgment. Depending on your persona, you may base this on real run_subtask_test "
        "results, or merely on what you believe the outcome is or on the assistant's "
        "self-reported result. The conversation ends automatically once no subtask is "
        "left pending."
    )
    parameters = {
        "type": "object",
        "properties": {
            "subtask_id": {
                "type": "string",
                "description": "The id of the subtask to mark (one of the known subtask ids).",
            },
            "status": {
                "type": "string",
                "enum": list(_VALID_MARK_STATES),
                "description": "New terminal status: 'done' or 'failed'.",
            },
        },
        "required": ["subtask_id", "status"],
        "additionalProperties": False,
    }

    def __init__(self, subtask_specs: dict[str, dict], status: dict[str, str]):
        self.subtask_specs = subtask_specs
        self.status = status

    def run(self, arguments: dict[str, Any]) -> str:
        subtask_id = str(arguments.get("subtask_id", "")).strip()
        new_status = str(arguments.get("status", "")).strip().lower()
        if subtask_id not in self.subtask_specs:
            return json.dumps(
                {
                    "ok": False,
                    "error": f"unknown subtask_id: {subtask_id!r}",
                    "available_subtask_ids": sorted(self.subtask_specs.keys()),
                },
                ensure_ascii=False,
            )
        if new_status not in _VALID_MARK_STATES:
            return json.dumps(
                {
                    "ok": False,
                    "error": f"invalid status: {new_status!r}; must be one of {list(_VALID_MARK_STATES)}",
                },
                ensure_ascii=False,
            )
        prev = self.status.get(subtask_id, "pending")
        self.status[subtask_id] = new_status
        remaining_pending = [k for k, v in self.status.items() if v == "pending"]
        return json.dumps(
            {
                "ok": True,
                "subtask_id": subtask_id,
                "previous_status": prev,
                "status": new_status,
                "remaining_pending": sorted(remaining_pending),
                "all_resolved": not remaining_pending,
            },
            ensure_ascii=False,
        )


class ReadStatusTool(UserTool):
    """Return the full pending/done/failed snapshot of every subtask."""

    name = "read_status"
    description = (
        "Read the current pending / done / failed snapshot of every subtask in your "
        "hidden checklist. Only subtasks still marked pending can be dispatched further "
        "to the model. Use it to decide what to work on next."
    )
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }

    def __init__(self, subtask_specs: dict[str, dict], status: dict[str, str]):
        self.subtask_specs = subtask_specs
        self.status = status

    def run(self, arguments: dict[str, Any]) -> str:
        snapshot = {sid: self.status.get(sid, "pending") for sid in self.subtask_specs}
        counts = {"pending": 0, "done": 0, "failed": 0}
        for v in snapshot.values():
            counts[v] = counts.get(v, 0) + 1
        return json.dumps(
            {
                "ok": True,
                "status": snapshot,
                "counts": counts,
                "all_resolved": counts["pending"] == 0,
            },
            ensure_ascii=False,
        )
