from __future__ import annotations

import re
import shlex
from typing import Any

import agent_trace as at
from local_test_executor import (
    bootstrap_session,
    replay_model_patches,
    run_tests_expect_failure_under_test_patch,
    run_tests_under_test_patch,
)
from swe_bench_pro_runner import is_pro_instance
from swebench_dockerlib import DockerSession

from synthesize_flow_utils import one_line, printable_text


class AugmentSupport:
    """Pure helper logic and execution-side helpers for PLAN/ATTEMPT augmentation.

    This module consolidates the augment logic into three layers:
    1. prompt assembly helpers
       - extract a compact summary of the patch diff
       - assemble the compressed `patch` / `task_summary` inputs the augment prompt needs
       - infer which augment branch an ATTEMPT target should take
    2. structural normalization helpers
       - normalize / dedup test patch fragments
       - stably dedup the node-id list
       - filter out candidates that overlap with existing own tests
    3. docker-backed semantic validation
       - only temporarily apply the candidate test patch
       - first protect the old tests in the input
       - then validate the new candidates by branch semantics
       - clearly distinguish "the new test_patch broke old tests" from "the new candidate itself does not hold"

    The goal is to keep the orchestration code focused while preserving the v2 augment retry semantics.
    """

    @staticmethod
    def summarize_unified_diff(diff: str) -> tuple[list[str], list[str]]:
        """Extract a best-effort file/function summary from a unified diff.

        The augment prompt does not need AST-level precision; it only needs a
        compact hint about which files the current diff changed and which hunk
        contexts appear, so it can keep inferring which tests are not yet covered
        without rescanning the whole diff every time.
        """
        files: list[str] = []
        funcs: list[str] = []
        for line in (diff or "").splitlines():
            if line.startswith("+++ b/"):
                files.append(line[len("+++ b/"):].strip())
            elif line.startswith("--- a/") and not files:
                files.append(line[len("--- a/"):].strip())
            elif line.startswith("@@"):
                tail = line.split("@@", 2)[-1].strip()
                if tail and not tail.startswith(("+", "-")):
                    funcs.append(tail)
        files = list(dict.fromkeys([f for f in files if f and f != "/dev/null"]))
        funcs = list(dict.fromkeys([f for f in funcs if f]))
        return files, funcs

    @classmethod
    def summarize_attempt_patch(cls, patch: Any) -> dict:
        """Compress a flat ATTEMPT.patch into a summary for PLAN augment."""
        kinds: list[str] = []
        files: list[str] = []
        funcs: list[str] = []
        if not isinstance(patch, list):
            return {"patch_kinds": [], "changed_files": [], "changed_functions": []}
        for subpatch in patch:
            if not isinstance(subpatch, dict):
                continue
            fmt = str(subpatch.get("format") or "unified_diff")
            if fmt == "unified_diff":
                sub_files, sub_funcs = cls.summarize_unified_diff(
                    str(subpatch.get("diff") or "")
                )
                files.extend(sub_files)
                funcs.extend(sub_funcs)
            kinds.append(fmt)
        return {
            "patch_kinds": list(dict.fromkeys(kinds)),
            "changed_files": list(dict.fromkeys(files)),
            "changed_functions": list(dict.fromkeys(funcs)),
        }

    @staticmethod
    def all_declared_own(trace: dict) -> list[str]:
        """Collect all own tests already declared in the enabled history."""
        seen: set[str] = set()
        out: list[str] = []
        for ev in trace.get("events") or []:
            if not at.is_enabled(ev):
                continue
            if ev.get("kind") == "PLAN":
                f2p, p2p = at.plan_own_tests(ev)
            elif ev.get("kind") == "ATTEMPT":
                f2p, p2p = at.attempt_own_tests(ev)
            else:
                continue
            for node in list(f2p) + list(p2p):
                if isinstance(node, str) and node and node not in seen:
                    seen.add(node)
                    out.append(node)
        return out

    @staticmethod
    def attempt_blueprint_category(trace: dict, ev: dict) -> str:
        """Recover an ATTEMPT's category from the most recent enabled PLAN blueprint.

        An ATTEMPT itself only stores `subtask_id` and `user_query`, but the true
        source of which semantic branch augment should take is the category in the
        PLAN blueprint. If this mapping cannot be obtained, return `unknown` and let
        the caller skip, rather than silently mishandling it as the ordinary branch.
        """
        if ev.get("kind") != "ATTEMPT":
            return "unknown"
        bid = ev.get("big_task_id")
        sid = ev.get("subtask_id")
        latest_plan = next(
            (
                plan for plan in reversed(trace.get("events") or [])
                if plan.get("kind") == "PLAN"
                and plan.get("big_task_id") == bid
                and at.is_enabled(plan)
            ),
            {},
        )
        subtask = next(
            (
                st for st in (latest_plan.get("subtasks_blueprint") or [])
                if isinstance(st, dict) and st.get("id") == sid
            ),
            {},
        )
        category = subtask.get("category")
        if isinstance(category, list):
            return str(category[0]) if category else "unknown"
        category = str(category or "").strip()
        return category or "unknown"

    @staticmethod
    def augment_task_summary(
        target_event: dict,
        *,
        scope: str,
        recent_patches_summary: list[dict] | None = None,
    ) -> str:
        """Build the natural-language task summary for PLAN or ATTEMPT augment."""
        if scope == "PLAN":
            lines = [
                f"Task level: PLAN, test augmentation for the entire BigTask {target_event.get('big_task_id') or ''}.",
                f"Overall goal: {str(target_event.get('natural_language_goal') or '').strip() or 'none'}",
                f"Acceptance criteria: {str(target_event.get('acceptance_criteria') or '').strip() or 'none'}",
            ]
            subtasks = [
                f"{st.get('id') or '?'} | category={st.get('category') or 'unknown'} | content={st.get('text') or ''}"
                for st in (target_event.get("subtasks_blueprint") or [])
                if isinstance(st, dict)
            ]
            if subtasks:
                lines.append("Current subtask blueprint:\n" + "\n".join(subtasks))
            summaries = [
                f"Completed implementation {idx}: changed_files={item.get('changed_files') or []}; changed_functions={item.get('changed_functions') or []}"
                for idx, item in enumerate(recent_patches_summary or [], 1)
                if isinstance(item, dict)
            ]
            if summaries:
                lines.append("Current cumulative implementation summary:\n" + "\n".join(summaries))
        else:
            lines = [
                f"Task level: ATTEMPT, test augmentation for subtask {target_event.get('subtask_id') or ''}.",
                f"Subtask goal: {str(target_event.get('natural_language_goal') or '').strip() or 'none'}",
                f"Acceptance criteria: {str(target_event.get('acceptance_criteria') or '').strip() or 'none'}",
            ]
            user_query = target_event.get("user_query") or {}
            if isinstance(user_query, dict) and user_query.get("content"):
                rendered = (
                    f"{user_query.get('role') or 'user'}: {str(user_query.get('content') or '').strip()}"
                )
                lines.append("Current conversation context:\n" + rendered)
        return "\n\n".join(line for line in lines if line.strip())

    @staticmethod
    def augment_patch_payload(
        trace: dict, target_event: dict, *, scope: str,
    ) -> list[Any]:
        """ATTEMPT forwards its own patch; PLAN aggregates successful patches of the current BigTask."""
        if scope == "ATTEMPT":
            patch = target_event.get("patch")
            return patch if isinstance(patch, list) else []
        bid = target_event.get("big_task_id")
        patches: list[Any] = []
        for ev in trace.get("events") or []:
            if (
                ev.get("kind") == "ATTEMPT"
                and at.is_enabled(ev)
                and ev.get("big_task_id") == bid
                and (ev.get("result") or {}).get("ok") is True
            ):
                patches.extend(
                    item for item in (ev.get("patch") or [])
                    if isinstance(item, dict)
                )
        return patches

    @staticmethod
    def normalize_test_patch_list(test_patch: Any) -> list[str]:
        """Normalize the stored `test_patch` into a uniform `list[str]`."""
        if isinstance(test_patch, list):
            return [str(x) for x in test_patch if isinstance(x, str) and x.strip()]
        if isinstance(test_patch, str) and test_patch.strip():
            return [test_patch]
        return []

    @staticmethod
    def combine_test_patch_fragments(parts: list[str]) -> str:
        """Concatenate multiple test patch fragments into one temporary test patch."""
        return "\n".join(part for part in parts if isinstance(part, str) and part.strip())

    @staticmethod
    def dedup_patch_fragments(parts: list[str]) -> list[str]:
        """Exact-dedup test patch fragments by full string before merging.

        This is a host-side guard against the model verbatim-repeating a diff that
        already exists. Here we only remove fully identical duplicate fragments;
        near-duplicates or potentially conflicting patches are still left for the
        docker validation stage to handle.
        """
        seen: set[str] = set()
        out: list[str] = []
        for part in parts:
            if not isinstance(part, str):
                continue
            normalized = part.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            out.append(part)
        return out

    @staticmethod
    def dedup_keep_order(xs: list[str]) -> list[str]:
        """Dedup strings while preserving first-occurrence order."""
        seen: set[str] = set()
        out: list[str] = []
        for item in xs or []:
            if not isinstance(item, str):
                continue
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
        return out

    @classmethod
    def test_nodes_defined_by_patch(cls, parts: list[str]) -> list[str]:
        """Best-effort extraction of node-ids for newly defined tests from the added `test_patch`.

        This step enforces the rule: if `test_patch` adds test functions but in the
        end no survivor node-id lands on those tests, they should not be left behind
        in `test_patch`.
        """
        nodes: list[str] = []
        current_file = ""
        current_class = ""
        for raw in cls.combine_test_patch_fragments(parts).splitlines():
            if raw.startswith("+++ b/"):
                current_file = raw[len("+++ b/"):].strip()
                current_class = ""
                continue
            if not current_file or not raw.startswith("+") or raw.startswith("+++"):
                continue
            line = raw[1:]
            class_match = re.match(r"^class\s+(Test\w*)\b", line)
            if class_match:
                current_class = class_match.group(1)
                continue
            def_match = re.match(r"^(\s*)def\s+(test_\w+)\b", line)
            if not def_match:
                continue
            indent, name = def_match.groups()
            if current_class and indent:
                nodes.append(f"{current_file}::{current_class}::{name}")
            elif indent:
                continue
            else:
                current_class = ""
                nodes.append(f"{current_file}::{name}")
        return cls.dedup_keep_order(nodes)

    @classmethod
    def filter_extras_against_known(
        cls,
        trace: dict,
        target_event: dict,
        instance: dict,
        *,
        candidates_f2p: list[str],
        candidates_p2p: list[str],
    ) -> tuple[list[str], list[str], dict[str, str]]:
        """Filter out already-covered node-ids before entering docker validation.

        Only exact node-id matching is done (not name-similarity judgment), ensuring
        the filtering is based on whether the test node already exists in the trace
        history, not on names that look similar.

        Returns (f2p_clean, p2p_clean, rejected_reasons):
        - f2p_clean / p2p_clean: candidates that pass the structural filter.
        - rejected_reasons: {node_id: reason_str}, the specific reason for each
          filtered candidate, including which existing test node it exactly duplicates.
        """
        bid = target_event.get("big_task_id")
        # Exact node-id-level declared set, recording each node's first appearance (to point out which one is duplicated in the error)
        forbidden: dict[str, str] = {}
        for node in cls.all_declared_own(trace):
            if node not in forbidden:
                forbidden[node] = node
        for entry in at.compute_cumulative_regression(trace, big_task_id=bid):
            for node in entry.get("own_f2p", []) + entry.get("own_p2p", []):
                if node not in forbidden:
                    forbidden[node] = node

        rejected: dict[str, str] = {}

        def clean(nodes: list[str], list_name: str) -> list[str]:
            seen: set[str] = set()
            out: list[str] = []
            for node in nodes or []:
                if not isinstance(node, str) or not node.strip():
                    continue
                if node in forbidden:
                    rejected[node] = (
                        f"exactly duplicates the existing test node {forbidden[node]!r} "
                        f"(identical node-id; please generate a new test covering a different code path or boundary condition, not just a rename)"
                    )
                    continue
                if node in seen:
                    rejected[node] = f"{list_name} candidate list contains a self-duplicate"
                    continue
                seen.add(node)
                out.append(node)
            return out

        f2p = clean(candidates_f2p, "own_f2p")
        p2p = clean(candidates_p2p, "own_p2p")
        f2p_set = set(f2p)
        p2p_only: list[str] = []
        for node in p2p:
            if node in f2p_set:
                rejected[node] = "already present in own_f2p candidates; own_p2p should not overlap with own_f2p"
            else:
                p2p_only.append(node)
        return f2p, p2p_only, rejected

    @staticmethod
    def latest_ok_attempt_for_big_task(trace: dict, big_task_id: str) -> dict | None:
        """Return the last enabled and successful ATTEMPT under the given BigTask."""
        for ev in reversed(trace.get("events") or []):
            if (
                ev.get("kind") == "ATTEMPT"
                and at.is_enabled(ev)
                and ev.get("big_task_id") == big_task_id
                and (ev.get("result") or {}).get("ok") is True
            ):
                return ev
        return None

    @classmethod
    def docker_filter_extras(
        cls,
        trace: dict,
        target_event: dict,
        *,
        candidates_f2p: list[str],
        candidates_p2p: list[str],
        candidate_test_patch: list[str],
        include_prior_regression: bool,
        verbose: bool,
    ) -> tuple[list[str], list[str], dict]:
        """Run the full augment-side semantic validation inside docker.

        Validation order (old/new tests are merged and run together):
        1. bootstrap + replay up to just before the patch event.
        2. pre-check: merged_f2p must all fail before the fix, merged_p2p must all pass before the fix.
           - old tests (existing own_f2p/own_p2p) not meeting expectation -> the new test_patch is broken, hard stop.
           - only candidates not meeting expectation -> drop the offending candidates and continue (collect as many as possible).
        3. apply the current patch (replay patch_event).
        4. smoke import (catch obvious syntax/import errors).
        5. post-test: merged_f2p + merged_p2p must all pass.
           - old tests failing after the fix -> the new test_patch is broken, hard stop.
           - only candidates failing -> drop them, return the rest as survivors.
        6. unkept defined tests check.

        Error attribution principle:
        - existing own_f2p or own_p2p erroring -> the fault lies with the newly added test_patch.
        - only new candidate p2p erroring -> the fault lies with the candidate itself.
        """
        cfg = trace["config"]
        inst = dict(trace.get("instance", {}) or {})
        inst["_config_image"] = cfg.get("image")
        inst["_test_runner_kind"] = cfg.get("test_runner_kind")
        candidates = cls.dedup_keep_order(list(candidates_f2p) + list(candidates_p2p))
        if not candidates:
            return [], [], {"collect_miss": [], "run_failed": [], "per_candidate_errors": {}, "skipped": "no candidates"}

        patch_event = (
            target_event
            if target_event.get("kind") == "ATTEMPT"
            else cls.latest_ok_attempt_for_big_task(
                trace, str(target_event.get("big_task_id") or "")
            )
        )
        if target_event.get("kind") == "PLAN":
            current_f2p, current_p2p = at.plan_own_tests(target_event)
        else:
            current_f2p, current_p2p = at.attempt_own_tests(target_event)
        current_f2p = cls.dedup_keep_order(list(current_f2p))
        current_p2p = [n for n in cls.dedup_keep_order(list(current_p2p)) if n not in set(current_f2p)]
        existing_set = set(current_f2p) | set(current_p2p)

        # Merge old tests with new candidates: validate together
        merged_f2p = cls.dedup_keep_order(current_f2p + list(candidates_f2p))
        merged_p2p = cls.dedup_keep_order(
            current_p2p + [n for n in candidates_p2p if n not in set(merged_f2p)]
        )

        debug: dict[str, Any] = {
            "collect_miss": [],
            "run_failed": [],
            "pre_not_failing": [],
            "pre_failed": [],
            "guard_failed": [],
            "unkept_defined_tests": [],
            "per_candidate_errors": {},
            "skipped": "",
        }

        session = DockerSession(
            image=cfg["image"],
            keep_on_fail=False,
            login_shell=not is_pro_instance(
                trace.get("instance"),
                image=cfg.get("image"),
                runner_kind=cfg.get("test_runner_kind"),
            ),
        )
        session.start()
        try:
            bootstrap_session(session, trace)
            try:
                replay_model_patches(session, trace, stop_before_event=patch_event, verbose=False)
            except Exception as exc:
                return [], [], {**debug, "skipped": f"replay_fail:{printable_text(exc)}"}

            # -- Step 1: pre-check --------------------------------------------
            # merged_f2p must all fail before the fix, merged_p2p must all pass before the fix.
            # old tests not meeting expectation -> hard stop; new candidates not meeting expectation -> record error and keep going.
            per_candidate_errors: dict[str, str] = {}
            if merged_f2p or merged_p2p:
                try:
                    pre_result = run_tests_expect_failure_under_test_patch(
                        session,
                        test_patch=candidate_test_patch,
                        tests=merged_f2p,
                        pass_tests=merged_p2p,
                        base_commit=inst.get("base_commit", ""),
                        pytest_bin=cfg["pytest_bin"],
                        cwd=cfg["cwd"],
                        timeout_sec=1200,
                        instance=inst,
                        require_all_fail=True,
                    )
                except Exception as exc:
                    return [], [], {**debug, "per_candidate_errors": {}, "skipped": f"pre_test_patch_fail:{printable_text(exc)}"}
                _ok, pre_status, pre_unexpected, _out = pre_result
                if not pre_result.ok:
                    existing_unexpected = [n for n in pre_unexpected if n in existing_set]
                    if existing_unexpected:
                        # old tests not meeting expectation -> fault lies with the new test_patch, hard stop
                        return [], [], {
                            **debug,
                            "guard_failed": existing_unexpected,
                            "per_candidate_errors": {},
                            "skipped": "existing_own_pre_expectation_failed",
                        }
                    # new candidates not meeting expectation -> collect reasons, keep executing to gather more info
                    pre_not_failing = [n for n in candidates_f2p if pre_status.get(n) not in {"FAILED", "ERROR"}]
                    pre_failed_p2p = [n for n in candidates_p2p if pre_status.get(n) != "PASSED"]
                    for n in pre_not_failing:
                        per_candidate_errors[n] = (
                            f"the F2P candidate should fail before the patch is applied, but its actual status is {pre_status.get(n)!r}"
                            " -- this test may be unrelated to the bug being fixed, or the test itself has the wrong condition"
                        )
                    for n in pre_failed_p2p:
                        per_candidate_errors[n] = (
                            f"the P2P candidate should pass before the patch is applied, but its actual status is {pre_status.get(n)!r}"
                            " -- the test may have a syntax error, wrong file path, or missing dependency"
                        )
                    debug["pre_not_failing"] = pre_not_failing
                    debug["pre_failed"] = pre_failed_p2p
                    # do not return early; keep running post-test to collect complete errors

            # -- Step 2: apply patch ------------------------------------------
            try:
                replay_model_patches(
                    session, trace,
                    only_event=patch_event if patch_event is not None else target_event,
                )
            except Exception as exc:
                return [], [], {**debug, "skipped": f"replay_apply_fail:{printable_text(exc)}"}

            # -- Step 3: smoke import -----------------------------------------
            smoke_import = cfg.get("smoke_import")
            if smoke_import:
                smoke_python = cfg.get("smoke_python") or (
                    cfg["pytest_bin"].rsplit("/", 1)[0] + "/python"
                )
                sp = session.exec(
                    f"cd {shlex.quote(cfg['cwd'])} && "
                    f"{shlex.quote(smoke_python)} -c "
                    f"{shlex.quote('import ' + smoke_import)}",
                    timeout=180,
                )
                if sp.returncode != 0:
                    return [], [], {**debug, "skipped": "smoke_fail"}

            # -- Step 4: post-test --------------------------------------------
            # after merging all must pass; old-test failures attribute to test_patch, new-candidate failures collected into per_candidate_errors.
            post_status: dict = {}
            if merged_f2p or merged_p2p:
                try:
                    post_result = run_tests_under_test_patch(
                        session,
                        test_patch=candidate_test_patch,
                        f2p_tests=merged_f2p,
                        p2p_tests=merged_p2p,
                        base_commit=inst.get("base_commit", ""),
                        pytest_bin=cfg["pytest_bin"],
                        cwd=cfg["cwd"],
                        timeout_sec=1200,
                        instance=inst,
                    )
                except Exception as exc:
                    return [], [], {**debug, "per_candidate_errors": per_candidate_errors, "skipped": f"post_test_patch_fail:{printable_text(exc)}"}
                _ok, post_status, post_failing, _out = post_result
                if not post_result.ok:
                    existing_failing = [n for n in post_failing if n in existing_set]
                    if existing_failing:
                        # old tests failing after the fix -> fault lies with the new test_patch, hard stop
                        return [], [], {
                            **debug,
                            "collect_miss": [n for n in existing_failing if post_status.get(n) == "NOT_FOUND"],
                            "run_failed": [n for n in existing_failing if post_status.get(n) not in {"NOT_FOUND", "PASSED"}],
                            "guard_failed": existing_failing,
                            "per_candidate_errors": per_candidate_errors,
                            "skipped": "existing_own_post_expectation_failed",
                        }
                    # new candidates failing -> merge into per_candidate_errors (do not overwrite entries already recorded by pre-check)
                    all_candidates_local = cls.dedup_keep_order(list(candidates_f2p) + list(candidates_p2p))
                    collect_miss_local = [n for n in all_candidates_local if post_status.get(n) == "NOT_FOUND"]
                    run_failed_local = [n for n in all_candidates_local if n not in collect_miss_local and post_status.get(n) != "PASSED"]
                    for n in collect_miss_local:
                        if n not in per_candidate_errors:
                            per_candidate_errors[n] = "pytest cannot find this test node (the function name, file path, or class name may be wrong)"
                    for n in run_failed_local:
                        if n not in per_candidate_errors:
                            per_candidate_errors[n] = (
                                f"the test did not pass after the patch was applied (actual status: {post_status.get(n)!r})"
                                " -- the test assertion may be wrong, or the test_patch itself has a syntax/logic error"
                            )
                    debug["collect_miss"] = sorted(collect_miss_local)
                    debug["run_failed"] = run_failed_local

            # -- Step 5: collect survivors (partial candidate-level retention) --
            # candidate-level failures are no longer all-or-nothing: candidates that pass post-test are returned as survivors,
            # the reasons for failed candidates are recorded in per_candidate_errors / debug, and the upper-layer threshold
            # (passing count >= 2 and >= half of total candidates) decides whether to keep overall or retry.
            # Note: hard errors whose fault lies with test_patch -- breaking existing own tests, smoke_fail, replay_fail, etc.
            # -- have already returned via hard stop above and will not reach here.
            passed_set = {n for n, v in post_status.items() if v == "PASSED"}
            survivors_f2p = [n for n in candidates_f2p if n in passed_set or not post_status]
            survivors_p2p = [n for n in candidates_p2p if n in passed_set or not post_status]
            survivors_p2p = [n for n in survivors_p2p if n not in set(survivors_f2p)]

            # -- Step 6: unkept defined tests ---------------------------------
            # no longer hard stop; unkept is an independent issue and should not discard already-passing candidates
            defined_nodes = cls.test_nodes_defined_by_patch(
                cls.normalize_test_patch_list(candidate_test_patch)
            )
            unkept_defined_tests = []
            if defined_nodes:
                kept = current_f2p + current_p2p + survivors_f2p + survivors_p2p
                for defined_node in defined_nodes:
                    if not any(str(node) == defined_node or str(node).startswith(f"{defined_node}[") for node in kept):
                        unkept_defined_tests.append(defined_node)

            unkept_errors: dict[str, str] = {
                n: (
                    "test_patch defines this test function but does not add it to the own_p2p candidate list"
                    " -- please add it to own_p2p, or remove the definition from test_patch"
                )
                for n in unkept_defined_tests
            }

            # merge candidate-failure reasons with unkept reasons (candidate-failure reasons take priority, not overwritten by unkept)
            combined_errors: dict[str, str] = dict(per_candidate_errors)
            for n, msg in unkept_errors.items():
                combined_errors.setdefault(n, msg)

            if per_candidate_errors:
                skipped = "candidates_failed"
            elif unkept_defined_tests:
                skipped = "new_test_patch_has_unkept_tests"
            else:
                skipped = ""

            return survivors_f2p, survivors_p2p, {
                **debug,
                "unkept_defined_tests": unkept_defined_tests,
                "per_candidate_errors": combined_errors,
                "skipped": skipped,
            }
        finally:
            session.stop()
