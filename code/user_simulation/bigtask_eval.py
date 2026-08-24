"""
bigtask_eval.py
===============

Private (hidden) per-big-task correctness scoring for the standalone user-sim
closed loop. This is NOT part of the in-conversation flow: its results are
written to a host-side file and are NEVER shown to the user-sim or the coding
agent — they exist only to compute "did the agent's code pass this big task's
f2p/p2p tests at the moment the big task finished".

Trigger (decided by the caller, see usersim_main): a big task is considered
finished when EVERY subtask under it has reached a terminal state in the shared
status dict (each "done" or "failed", none "pending"). When that first happens,
we run — once — the hidden f2p/p2p tests for each subtask of that big task
against the agent's CURRENT live worktree, and record the pass/fail breakdown.

We reuse the existing ``RunSubtaskTestTool`` instance the runner already built:
it wraps ``run_tests_under_test_patch`` with an explicit fs snapshot/restore so
the scoring run cannot pollute the agent's workspace, and it REPORTS only (never
mutates status). We just parse its JSON output and tally f2p/p2p per node.
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from pathlib import Path

logger = logging.getLogger("usersim_main")


class BigTaskEvaluator:
    """Runs each big task's hidden f2p/p2p tests once, when the big task finishes.

    Parameters
    ----------
    specs : list[dict]
        The subtask specs (trace order), each with ``id`` / ``big_task_id`` /
        ``own_f2p`` / ``own_p2p`` — same list the runner builds.
    status : dict[str, str]
        The shared pending/done/failed status dict (read-only here).
    run_test_tool : RunSubtaskTestTool
        Already pointed at the agent's live container; REPORTS only.
    out_path : Path
        Where to dump the private result JSON (e.g. out_dir/bigtask_eval.json).
    instance_id : str
        For labeling the dumped report.
    """

    def __init__(self, specs, status, run_test_tool, out_path: Path, instance_id: str):
        self.status = status
        self.run_test_tool = run_test_tool
        self.out_path = Path(out_path)
        self.instance_id = instance_id

        # Ordered big_task_id -> [subtask_id, ...] (trace order preserved).
        self._groups: "OrderedDict[str, list[str]]" = OrderedDict()
        for s in specs:
            self._groups.setdefault(s["big_task_id"], []).append(s["id"])

        # big_task_id -> result record (also marks "already evaluated").
        self._results: "OrderedDict[str, dict]" = OrderedDict()

    # ---- helpers ------------------------------------------------------------

    def _big_task_terminal(self, bid: str) -> bool:
        """True iff every subtask of this big task is done/failed (none pending)."""
        return all(self.status.get(sid) in ("done", "failed")
                   for sid in self._groups[bid])

    def _run_one_subtask(self, sid: str) -> dict:
        """Invoke the live test tool for one subtask and parse its JSON result."""
        try:
            raw = self.run_test_tool.run({"subtask_id": sid})
            data = json.loads(raw)
        except Exception as exc:  # never let scoring crash the closed loop
            logger.warning("[bigtask-eval] subtask %s test run raised (recorded as error): %r", sid, exc)
            return {"subtask_id": sid, "ok": False, "error": repr(exc),
                    "f2p": {}, "p2p": {}, "failing_node_ids": []}
        f2p = data.get("f2p") or {}
        p2p = data.get("p2p") or {}
        return {
            "subtask_id": sid,
            "ok": bool(data.get("ok")),
            "marked_status": self.status.get(sid),
            "error_kind": data.get("error_kind"),
            "f2p": f2p,
            "p2p": p2p,
            "failing_node_ids": data.get("failing_node_ids") or [],
        }

    @staticmethod
    def _count_pass(node_map: dict) -> tuple[int, int]:
        """(#passed, #total) over a {node_id: STATUS} map; PASSED counts as pass."""
        total = len(node_map)
        passed = sum(1 for v in node_map.values() if str(v).upper() == "PASSED")
        return passed, total

    # ---- public API ---------------------------------------------------------

    def check_and_evaluate(self, turn: int) -> list[str]:
        """Evaluate every big task that JUST became terminal and isn't scored yet.

        Returns the list of big_task_ids evaluated this call (usually 0 or 1).
        Persists the cumulative report to ``out_path`` after each evaluation.
        """
        newly: list[str] = []
        for bid in self._groups:
            if bid in self._results:
                continue  # already scored once — never re-run
            if not self._big_task_terminal(bid):
                continue
            logger.info("[bigtask-eval] big task %s: all subtasks reached terminal state -> running f2p/p2p accounting (turn=%d)",
                        bid, turn)
            sub_results = [self._run_one_subtask(sid) for sid in self._groups[bid]]

            f2p_pass = f2p_total = p2p_pass = p2p_total = 0
            for r in sub_results:
                fp, ft = self._count_pass(r["f2p"])
                pp, pt = self._count_pass(r["p2p"])
                f2p_pass += fp; f2p_total += ft
                p2p_pass += pp; p2p_total += pt
            all_pass = all(r["ok"] for r in sub_results) if sub_results else False

            self._results[bid] = {
                "big_task_id": bid,
                "evaluated_at_turn": turn,
                "subtask_status_at_eval": {sid: self.status.get(sid)
                                           for sid in self._groups[bid]},
                "subtasks": sub_results,
                "n_f2p_pass": f2p_pass, "n_f2p_total": f2p_total,
                "n_p2p_pass": p2p_pass, "n_p2p_total": p2p_total,
                "all_pass": all_pass,
            }
            newly.append(bid)
            logger.info("[bigtask-eval] big task %s accounting done: f2p %d/%d, p2p %d/%d, all_pass=%s",
                        bid, f2p_pass, f2p_total, p2p_pass, p2p_total, all_pass)
            self.dump()
        return newly

    def dump(self) -> None:
        """Write the cumulative private report. Safe to call repeatedly."""
        n_bt = len(self._groups)
        n_scored = len(self._results)
        bt_pass = sum(1 for r in self._results.values() if r["all_pass"])
        report = {
            "instance_id": self.instance_id,
            "n_big_tasks": n_bt,
            "n_big_tasks_evaluated": n_scored,
            "n_big_tasks_all_pass": bt_pass,
            "big_tasks": list(self._results.values()),
        }
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))


def run_final_eval(specs, status, run_test_tool, out_path: Path, instance_id: str) -> None:
    """Run tests for ALL subtasks at the very end of a run, regardless of their status.

    Results are written to out_path (e.g. final_eval.json) and never touch bigtask_eval.json.
    Errors per subtask are recorded but do not raise.
    """
    out_path = Path(out_path)
    subtask_results = []
    f2p_pass = f2p_total = p2p_pass = p2p_total = 0

    for s in specs:
        sid = s["id"]
        try:
            raw = run_test_tool.run({"subtask_id": sid})
            data = json.loads(raw)
        except Exception as exc:
            logger.warning("[final-eval] subtask %s test run raised (recorded as error): %r", sid, exc)
            subtask_results.append({"subtask_id": sid, "ok": False, "error": repr(exc),
                                    "marked_status": status.get(sid),
                                    "f2p": {}, "p2p": {}, "failing_node_ids": []})
            continue

        f2p = data.get("f2p") or {}
        p2p = data.get("p2p") or {}
        fp = sum(1 for v in f2p.values() if str(v).upper() == "PASSED")
        ft = len(f2p)
        pp = sum(1 for v in p2p.values() if str(v).upper() == "PASSED")
        pt = len(p2p)
        f2p_pass += fp; f2p_total += ft
        p2p_pass += pp; p2p_total += pt

        subtask_results.append({
            "subtask_id": sid,
            "ok": bool(data.get("ok")),
            "marked_status": status.get(sid),
            "error_kind": data.get("error_kind"),
            "f2p": f2p, "p2p": p2p,
            "failing_node_ids": data.get("failing_node_ids") or [],
        })
        logger.info("[final-eval] subtask %s: ok=%s f2p %d/%d p2p %d/%d",
                    sid, data.get("ok"), fp, ft, pp, pt)

    all_pass = all(r["ok"] for r in subtask_results) if subtask_results else False
    report = {
        "instance_id": instance_id,
        "n_subtasks": len(specs),
        "n_f2p_pass": f2p_pass, "n_f2p_total": f2p_total,
        "n_p2p_pass": p2p_pass, "n_p2p_total": p2p_total,
        "all_pass": all_pass,
        "subtasks": subtask_results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    logger.info("[final-eval] done: f2p %d/%d, p2p %d/%d, all_pass=%s -> %s",
                f2p_pass, f2p_total, p2p_pass, p2p_total, all_pass, out_path)
