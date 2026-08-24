from __future__ import annotations

import agent_trace as at


class EventHistory:
    """Trace navigation helpers used by the v2 top-level scheduling loop.

    These functions are all small, but they collect the many repeated history-scanning
    logics from the main flow. They do not build prompts, only answer scheduling questions
    such as:
    - Which PLAN is currently active
    - Which subtasks have already succeeded before a given position
    - Whether a pending ATTEMPT was already appended before an interruption
    - How to disable a failed suffix while preserving audit information
    """

    @staticmethod
    def disable_big_task_suffix(trace: dict, big_task_id: str, *, reason: str) -> None:
        """Disable a BigTask's currently-failing trailing suffix in place.

        Disabled events remain in the trace for auditing and traceback; but the executor
        and downstream prompt projections all filter them out via `at.is_enabled(...)`.
        """
        seen_big = False
        for ev in reversed(trace.get("events") or []):
            if ev.get("big_task_id") != big_task_id:
                if seen_big:
                    break
                continue
            seen_big = True
            if not at.is_enabled(ev):
                continue
            ev["enabled"] = False
            ev.setdefault("disabled_reason", reason)

    @staticmethod
    def disable_previous_failures_for_subtask(
        trace: dict,
        *,
        attempt_index: int,
        big_task_id: str,
        subtask_id: str,
    ) -> None:
        """After a later retry of the same subtask succeeds, disable earlier failed ATTEMPTs."""
        events = trace.get("events") or []
        for idx, ev in enumerate(events):
            if idx >= attempt_index:
                break
            if (
                ev.get("kind") == "ATTEMPT"
                and ev.get("big_task_id") == big_task_id
                and ev.get("subtask_id") == subtask_id
                and at.is_enabled(ev)
                and (ev.get("result") or {}).get("ok") is False
            ):
                ev["enabled"] = False
                ev.setdefault("disabled_reason", "superseded_by_successful_retry")

    @staticmethod
    def successful_subtask_ids_before(trace: dict, *, big_task_id: str, before_index: int) -> set[str]:
        """Return the subtask ids that already have an enabled, ok=true ATTEMPT before a given position."""
        out: set[str] = set()
        events = trace.get("events") or []
        for ev in events[:before_index]:
            if (
                ev.get("kind") == "ATTEMPT"
                and at.is_enabled(ev)
                and ev.get("big_task_id") == big_task_id
                and (ev.get("result") or {}).get("ok") is True
            ):
                sid = str(ev.get("subtask_id") or "")
                if sid:
                    out.add(sid)
        return out

    @staticmethod
    def latest_successful_attempt_index(
        trace: dict, *, big_task_id: str, subtask_id: str, before_index: int
    ) -> int | None:
        """Return the index of the last enabled, ok=true ATTEMPT for a subtask before before_index."""
        events = trace.get("events") or []
        found: int | None = None
        for idx, ev in enumerate(events[:before_index]):
            if (
                ev.get("kind") == "ATTEMPT"
                and at.is_enabled(ev)
                and ev.get("big_task_id") == big_task_id
                and str(ev.get("subtask_id") or "") == subtask_id
                and (ev.get("result") or {}).get("ok") is True
            ):
                found = idx
        return found

    @staticmethod
    def latest_active_plan(trace: dict) -> tuple[int, dict] | None:
        """Return the last enabled PLAN still in progress (with subtasks not yet complete) and its index.

        Active status is judged by the completion of the blueprint subtasks: as long as this PLAN
        still has a subtask without an enabled+ok ATTEMPT, it is considered still in progress
        (return it, so the top-level loop resumes, skipping completed subtasks and continuing with
        the remaining ones). The PLAN rubric unconditionally writes a verdict into plan.result at
        generation time (for record only); result is not the active-status signal. When all
        subtasks are complete, it is no longer active: return None, letting _decide_plan_branch
        decide whether the next round is escalate / repair / stop.

        Note: only the "latest enabled PLAN" is considered -- it is the current BigTask's plan.
        Earlier PLANs belong to earlier BigTasks that have already escalated, and should not be
        treated as the active plan.
        """
        events = trace.get("events") or []
        for idx in range(len(events) - 1, -1, -1):
            ev = events[idx]
            if ev.get("kind") != "PLAN" or not at.is_enabled(ev):
                continue
            bid = str(ev.get("big_task_id") or "")
            done = EventHistory.successful_subtask_ids_before(
                trace, big_task_id=bid, before_index=len(events)
            )
            subtask_ids = [
                str(s.get("id") or "")
                for s in (ev.get("subtasks_blueprint") or [])
                if s.get("id")
            ]
            subtask_id_set = set(subtask_ids)
            # Integrity backfill only targets the blueprint's **last subtask**: it is the
            # BigTask's closing subtask, and only when its artifacts are incomplete do we keep
            # the PLAN active to go back and fill it. A middle subtask is not revisited even if
            # its p2p is empty (to avoid back-and-forth redo). Rules:
            #   - last subtask missing own_f2p or test_patch -> that ATTEMPT is unusable, redo
            #     the ATTEMPT (resume-redo branch at line 741+);
            #   - otherwise last subtask only missing own_p2p and augment has not finished
            #     (augment_status still None) -> only regression tests are missing, fill augment
            #     (resume-augment branch at line 716+).
            # After augment truly finishes, augment_status is persisted (ok/partial_ok/
            # failed_empty/partial_ok_exhausted); at that point, even if p2p is still empty we
            # do not re-enter, avoiding an infinite loop.
            if subtask_ids:
                _last_sid = subtask_ids[-1]
                if _last_sid in done:
                    _aidx = EventHistory.latest_successful_attempt_index(
                        trace, big_task_id=bid, subtask_id=_last_sid, before_index=len(events)
                    )
                    if _aidx is not None:
                        _aev = events[_aidx]
                        _has_f2p = bool(_aev.get("own_f2p"))
                        _has_tp = bool(_aev.get("test_patch"))
                        _has_p2p = bool(_aev.get("own_p2p"))
                        _aug_status = (_aev.get("result") or {}).get("augment_status")
                        if (not _has_f2p) or (not _has_tp):
                            # last subtask missing f2p/test_patch -> redo the ATTEMPT.
                            done = done - {_last_sid}
                        elif (not _has_p2p) and _aug_status is None:
                            # last subtask only missing p2p and augment not run -> fill augment.
                            done = done - {_last_sid}
            if not subtask_id_set or subtask_id_set.issubset(done):
                # All subtasks done (or blueprint empty) -> this PLAN is closed, no longer active.
                return None
            # Still has unfinished subtasks (including the last one only missing augment / needing
            # redo) -> current active plan, resume to continue.
            return idx, ev
        return None

    @staticmethod
    def latest_pending_attempt_index(trace: dict, *, big_task_id: str, subtask_id: str) -> int | None:
        """Find a pending ATTEMPT that was already appended before an interruption but whose `result` is still null."""
        events = trace.get("events") or []
        for idx in range(len(events) - 1, -1, -1):
            ev = events[idx]
            if ev.get("kind") == "PLAN":
                break
            if (
                ev.get("kind") == "ATTEMPT"
                and at.is_enabled(ev)
                and ev.get("big_task_id") == big_task_id
                and ev.get("subtask_id") == subtask_id
                and ev.get("result") is None
            ):
                return idx
        return None

    @staticmethod
    def disable_all_big_task_events(trace: dict, big_task_id: str, *, reason: str) -> None:
        """Disable all PLAN and ATTEMPT events of a given BigTask in place.

        Unlike disable_big_task_suffix, this function scans the whole trace and disables all
        events belonging to that big_task_id, not just a trailing contiguous segment.
        Note: only modifies the enabled field, not result.ok.
        """
        for ev in (trace.get("events") or []):
            if ev.get("big_task_id") == big_task_id and at.is_enabled(ev):
                ev["enabled"] = False
                ev.setdefault("disabled_reason", reason)
