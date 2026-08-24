#!/usr/bin/env python3
"""
local_test_executor.py  (D-1, v2 trace schema)
==============================================

The "heart" of the agent loop: apply trace model patches and run declared
(f2p, p2p) tests in a docker container. Failure modes auto-rollback the
worktree to the pre-attempt snapshot.

Public APIs:
    replay_model_patches(...)       — apply model patches from trace start to
                                      an optional event boundary, or apply one
                                      explicit event patch
    run_tests_under_test_patch(...) — apply exactly one test_patch, run f2p/p2p,
                                      return per-node diagnostics, then reset
                                      that test_patch
    run_tests_expect_failure_under_test_patch(...)
                                    — apply exactly one test_patch, run tests
                                      expected to fail in the current worktree,
                                      return per-node diagnostics, then reset
                                      that test_patch

Built on top of swebench_dockerlib (D-0).

What this file does NOT do (compared to the old v1 version):
  - No COMMIT events are appended.
  - No state.cumulative_regression / rolled_back_tests bookkeeping; the
    regression set is recomputed from events on demand.
  - No ROLLBACK events; failed attempts only roll back the *container
    worktree*, never the trace.
"""

from __future__ import annotations

import dataclasses
import math
import re
import subprocess
import shlex
from dataclasses import dataclass, field
from typing import Optional

from swebench_dockerlib import (
    DockerSession,
    PatchApplyError,
    apply_unified_diff,
    run_pytest,
    _snapshot_tree_digest,
    _strip_pytest_banner,
)
from swe_bench_pro_runner import is_pro_instance, run_pro_official_tests
from swe_bench_verified_runner import is_verified_instance, run_verified_official_tests
import agent_trace as at
from swebench.harness.utils import get_modified_files, get_new_files
from unidiff.errors import UnidiffParseError


FAILURE_LOG_TAIL_LINES = 240


# =============================================================================
# Result type
# =============================================================================
@dataclass
class ExecuteReport:
    ok: bool
    error_kind: Optional[str]   # PATCH_APPLY_FAIL / PATCH_SMOKE_FAIL /
                                # COLLECT_MISS / PYTEST_FAIL / TIMEOUT /
                                # ROLLBACK_NOT_NET_ZERO / None
    per_node_status: dict = field(default_factory=dict)
    per_node_details: dict = field(default_factory=dict)
    expectation_errors: dict = field(default_factory=dict)
    failing_node_ids: list = field(default_factory=list)
    test_failure_details: list = field(default_factory=list)
    pre_snapshot_id: str = ""
    post_snapshot_id: Optional[str] = None
    applied_files: list = field(default_factory=list)
    demoted_to_p2p: list = field(default_factory=list)
    deleted_from_f2p: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class TestPatchRunReport:
    """Detailed result for one temporary test_patch run.

    The object is intentionally tuple-compatible with the previous
    ``(ok, status, failing, output)`` return shape, so existing callers can keep
    unpacking it while newer callers can inspect per-node diagnostics.
    """
    ok: bool
    per_node_status: dict = field(default_factory=dict)
    failing_node_ids: list = field(default_factory=list)
    pytest_output: str = ""
    per_node_details: dict = field(default_factory=dict)
    expectation_errors: dict = field(default_factory=dict)

    def __iter__(self):
        yield self.ok
        yield self.per_node_status
        yield self.failing_node_ids
        yield self.pytest_output

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _dedup_keep_order(xs):
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _key_output_line(text: str, *, fallback: str = "") -> str:
    """Return the most useful short line from pytest/git output for logs."""
    for raw in reversed((text or "").splitlines()):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("=") or line.startswith("-"):
            continue
        return line[:220]
    return fallback


def _looks_like_pro_global_build_failure(output: str) -> bool:
    """Detect Pro compile/build failures that happen before test result lines."""
    text = output or ""
    lower = text.lower()
    if not lower.strip():
        return False
    go_markers = (
        "[build failed]",
        "has no field or method",
        "undefined:",
        "undefined (type ",
        "cannot use ",
        "not enough arguments",
        "too many arguments",
        "no required module provides package",
    )
    generic_markers = (
        "compilation error",
        "compilation failed",
        "compilation failure",
        "cannot find symbol",
        "build failed",
        "type error",
        "syntaxerror:",
    )
    return (
        any(marker in lower for marker in go_markers)
        or any(marker in lower for marker in generic_markers)
        or re.search(r"\bTS\d{4}:", text) is not None
    )


def _short_nodes(nodes: list, *, limit: int = 3) -> str:
    nodes = [str(n) for n in nodes or []]
    if not nodes:
        return "none"
    shown = ", ".join(nodes[:limit])
    if len(nodes) > limit:
        shown += f", ... (+{len(nodes) - limit})"
    return shown


def _tail_lines(text: str, *, max_lines: int = FAILURE_LOG_TAIL_LINES) -> str:
    return "\n".join((text or "").splitlines()[-max_lines:])


def _merge_prefixed_details(target: dict, prefix: str, details: dict) -> None:
    for node, detail in (details or {}).items():
        target[f"{prefix}{node}"] = detail


def _merge_prefixed_status(target: dict, prefix: str, statuses: dict) -> None:
    for node, status in (statuses or {}).items():
        target[f"{prefix}{node}"] = status


def _merge_prefixed_expectations(target: dict, prefix: str, errors: dict) -> None:
    for node, error in (errors or {}).items():
        target[f"{prefix}{node}"] = error


def _log_executor_step(verbose: bool, step: str, *, ok: bool,
                       detail: str = "", output: str = "") -> None:
    if not verbose:
        return
    key = _key_output_line(output)
    suffix = f"  {detail}" if detail else ""
    if key:
        suffix += f"  key={key}"
    print(f"[executor]   step {step}: {'ok' if ok else 'FAIL'}{suffix}", flush=True)


def _test_patch_parts(test_patch) -> list[str]:
    if isinstance(test_patch, list):
        return [
            part for part in test_patch
            if isinstance(part, str) and part.strip()
        ]
    if isinstance(test_patch, str) and test_patch.strip():
        return [test_patch]
    return []


def _modified_test_files(test_patch) -> list[str]:
    files: list[str] = []
    for part in _test_patch_parts(test_patch):
        try:
            files.extend(get_modified_files(part))
        except UnidiffParseError as exc:
            raise PatchApplyError(f"[test-patch-parse-fail] {exc}") from exc
    return _dedup_keep_order(files)


def _new_test_files(test_patch) -> list[str]:
    files: list[str] = []
    for part in _test_patch_parts(test_patch):
        try:
            files.extend(get_new_files(part))
        except UnidiffParseError as exc:
            raise PatchApplyError(f"[test-patch-parse-fail] {exc}") from exc
    return _dedup_keep_order(files)


def reset_eval_test_files(
    session: DockerSession, *, test_patch, base_commit: str, cwd: str,
) -> list[str]:
    """Official SWE-bench eval semantics: reset files touched by test_patch
    back to base_commit before applying tests, and again after the test run.
    File extraction is delegated to swebench.harness.utils.
    """
    modified_files = _modified_test_files(test_patch)
    new_files = _new_test_files(test_patch)
    new_file_set = set(new_files)
    modified_files = [path for path in modified_files if path not in new_file_set]
    if not modified_files and not new_files:
        return []
    if modified_files:
        quoted_files = " ".join(shlex.quote(f) for f in modified_files)
        ref = shlex.quote(base_commit or "HEAD")
        p = session.exec(
            f"cd {shlex.quote(cwd)} && "
            f"git checkout {ref} -- {quoted_files}"
        )
        if p.returncode != 0:
            raise PatchApplyError(
                f"[test-reset-fail]\n{(p.stderr or p.stdout).strip()}"
            )
    if new_files:
        quoted_new_files = " ".join(shlex.quote(f) for f in new_files)
        p = session.exec(f"cd {shlex.quote(cwd)} && rm -f -- {quoted_new_files}")
        if p.returncode != 0:
            raise PatchApplyError(
                f"[test-new-file-clean-fail]\n{(p.stderr or p.stdout).strip()}"
            )
    return modified_files + new_files


def apply_eval_test_patch(
    session: DockerSession, *, test_patch, base_commit: str, cwd: str,
) -> list[str]:
    parts = _test_patch_parts(test_patch)
    if not parts:
        return []
    files = reset_eval_test_files(
        session, test_patch=parts, base_commit=base_commit, cwd=cwd,
    )
    for part in parts:
        apply_unified_diff(session, part, cwd=cwd)
    return files


def _classify_pytest_node(returncode: int, output: str) -> str:
    combo = (output or "").lower()
    if returncode == 0:
        return "PASSED"
    if "no tests ran" in combo or "not found" in combo:
        return "NOT_FOUND"
    if " error" in combo and " failed" not in combo:
        return "ERROR"
    return "FAILED"


def _run_pytest_node_detail(
    session: DockerSession, node_id: str, *, pytest_bin: str, cwd: str,
    timeout_sec: int,
) -> tuple[str, dict]:
    """Run exactly one pytest node and return status plus full diagnostics."""
    cmd = (
        f"cd {shlex.quote(cwd)} && {pytest_bin} -p no:cacheprovider "
        f"--no-header --tb=short -q {shlex.quote(node_id)}"
    )
    try:
        p = session.exec(cmd, timeout=timeout_sec)
        output = _strip_pytest_banner(p.stdout + "\n" + p.stderr)
        status = _classify_pytest_node(p.returncode, output)
        if status == "NOT_FOUND":
            expanded = _run_pytest_prefix_detail(
                session, node_id, pytest_bin=pytest_bin, cwd=cwd,
                timeout_sec=timeout_sec,
            )
            if expanded is not None:
                return expanded
        return status, {
            "status": status,
            "returncode": p.returncode,
            "output_excerpt": output,
        }
    except subprocess.TimeoutExpired as e:
        output = ""
        if e.output:
            output += e.output.decode() if isinstance(e.output, bytes) else str(e.output)
        if e.stderr:
            output += "\n" + (
                e.stderr.decode() if isinstance(e.stderr, bytes) else str(e.stderr)
            )
        return "TIMEOUT", {
            "status": "TIMEOUT",
            "returncode": None,
            "output_excerpt": output or "[timeout]",
        }


def _run_pytest_prefix_detail(
    session: DockerSession, node_id: str, *, pytest_bin: str, cwd: str,
    timeout_sec: int,
) -> tuple[str, dict] | None:
    """Resolve truncated parameterized pytest ids by prefix.

    Some datasets carry selectors ending inside a parameter id, e.g.
    ``test_x.py::test_case[foo,``. Pytest cannot run that exact selector, but
    ``--collect-only`` exposes concrete node ids with the same prefix. Treat the
    prefix as a small selector group and aggregate the concrete results.
    """
    text = str(node_id or "").strip()
    if "::" not in text or "[" not in text or "]" in text.rsplit("[", 1)[-1]:
        return None
    file_part = text.split("::", 1)[0]
    p = session.exec(
        f"cd {shlex.quote(cwd)} && {pytest_bin} -p no:cacheprovider "
        f"--collect-only -q {shlex.quote(file_part)}"
    )
    collect_output = _strip_pytest_banner(p.stdout + "\n" + p.stderr)
    candidates = [
        line.strip()
        for line in collect_output.splitlines()
        if line.strip().startswith(text)
    ]
    if not candidates:
        return None
    statuses: list[str] = []
    outputs: list[str] = [f"expanded prefix selector {text!r} -> {candidates!r}"]
    for candidate in candidates:
        cmd = (
            f"cd {shlex.quote(cwd)} && {pytest_bin} -p no:cacheprovider "
            f"--no-header --tb=short -q {shlex.quote(candidate)}"
        )
        try:
            sub = session.exec(cmd, timeout=timeout_sec)
            output = _strip_pytest_banner(sub.stdout + "\n" + sub.stderr)
            status = _classify_pytest_node(sub.returncode, output)
        except subprocess.TimeoutExpired as e:
            output = ""
            if e.output:
                output += e.output.decode() if isinstance(e.output, bytes) else str(e.output)
            if e.stderr:
                output += "\n" + (
                    e.stderr.decode() if isinstance(e.stderr, bytes) else str(e.stderr)
                )
            status = "TIMEOUT"
        statuses.append(status)
        outputs.append(f"===== {candidate} [{status}] =====\n{output}")
    if any(status == "TIMEOUT" for status in statuses):
        aggregate = "TIMEOUT"
    elif any(status in {"FAILED", "ERROR"} for status in statuses):
        aggregate = "FAILED"
    elif all(status == "PASSED" for status in statuses):
        aggregate = "PASSED"
    else:
        aggregate = "NOT_FOUND"
    return aggregate, {
        "status": aggregate,
        "returncode": 0 if aggregate == "PASSED" else None,
        "expanded_node_ids": candidates,
        "output_excerpt": "\n".join(outputs),
    }


def _pytest_collect_detail(
    session: DockerSession, node_ids: list[str], *, pytest_bin: str, cwd: str,
) -> tuple[set[str], set[str], str]:
    if not node_ids:
        return set(), set(), ""
    quoted = " ".join(shlex.quote(n) for n in node_ids)
    p = session.exec(
        f"cd {shlex.quote(cwd)} && {pytest_bin} -p no:cacheprovider "
        f"--collect-only -q {quoted}"
    )
    out = p.stdout + "\n" + p.stderr
    collected = {n for n in node_ids if n in out}
    return collected, set(node_ids) - collected, out


def _run_env_setup(session: DockerSession, commands: list, *, cwd: str) -> list[str]:
    for cmd in commands or []:
        if not isinstance(cmd, str) or not cmd.strip():
            continue
        sp = session.exec(f"cd {shlex.quote(cwd)} && {cmd}", timeout=1200)
        if sp.returncode != 0:
            tail = _tail_lines(sp.stdout + "\n" + sp.stderr)
            raise PatchApplyError(f"env_setup command failed: {cmd}\n{tail}")
    return []


def _apply_leg(session: DockerSession, leg: dict, *, cwd: str) -> list[str]:
    """Apply one leg patch. Raises PatchApplyError on failure."""
    fmt = leg.get("format", "unified_diff")
    if fmt == "unified_diff":
        diff = leg.get("diff") or ""
        return apply_unified_diff(session, diff, cwd=cwd) if diff.strip() else []
    if fmt == "env_setup":
        return _run_env_setup(session, leg.get("commands") or [], cwd=cwd)
    raise PatchApplyError(f"unknown leg patch format {fmt!r}")



def _normalize_patch(patch) -> dict:
    """Normalize a flat ``list[SubPatch]`` into a single executor patch object.

    ATTEMPT.patch is a flat list of SubPatch objects:
    - empty list -> empty diff (treated as no changes)
    - single SubPatch -> return that object directly
    - multiple SubPatch -> combine into a composite, applied sequentially in order
    """
    if isinstance(patch, dict):
        return patch
    if not isinstance(patch, list):
        raise PatchApplyError(f"patch must be dict or list[SubPatch], got {type(patch).__name__}")
    if len(patch) == 0:
        return {"format": "unified_diff", "diff": ""}
    if len(patch) == 1:
        return dict(patch[0])
    return {"format": "composite", "legs": [dict(x) for x in patch]}


def run_tests_under_test_patch(session: DockerSession, **kwargs) -> TestPatchRunReport:
    """Worktree-neutral wrapper around the test-visibility entry point.

    This is the single chokepoint through which *every* test stage runs
    (augment filter pre-check/post-test, seed precheck, main ATTEMPT path,
    ``run_tests_expect_failure_under_test_patch``).  We snapshot the worktree
    on entry and restore it on exit so that ANY runner side effect is undone
    regardless of which branch / reset path runs below:

    - created package directories (e.g. ``internal/storage/cache/``),
    - partially-cleaned files from file-list-based reset
      (``reset_eval_test_files`` / ``_pro_reset_test_files``),
    - fuse-overlayfs dentry / delete-cache incoherence under concurrent
      container IO that leaves *phantom* files which ``git apply`` later sees
      as "already exists in working directory".

    Making the guard live here (rather than at individual call sites) means the
    correctness of subsequent ``git apply`` / ``replay_model_patches`` no longer
    depends on the fragile file-list-based reset — that reset is now pure
    defense-in-depth.  The brittle path was the root cause of the augment
    "git-apply conflict" class of failures and of intermittent seed-precheck
    breakage.

    All callers pass arguments by keyword (``session`` positional, everything
    else keyword), so we forward ``**kwargs`` verbatim to the implementation.
    """
    cwd = kwargs["cwd"]
    _guard_snap = session.fs_snapshot(cwd)
    try:
        return _run_tests_under_test_patch_impl(session, **kwargs)
    finally:
        # The post-stage restore is *cleanup only*: it undoes runner side
        # effects so the NEXT git apply starts from a clean worktree. By the
        # time we reach here the current attempt's f2p/p2p evaluation has
        # already completed (try-body returned or raised its own real error).
        #
        # A restore failure (e.g. BusyBox/GNU tar tripping over a transiently
        # renamed `node_modules/.../.bin` symlink dir under concurrent IO:
        # "tar: Directory renamed before its status could be extracted") is an
        # INFRA error, not an attempt failure. We must NOT let it propagate:
        # an exception raised in `finally` would override the try-body's
        # successful return value (or mask the try-body's genuine error),
        # turning an already-passing ATTEMPT into a spurious failure and
        # burning a retry. So we swallow-and-log it here. The next attempt
        # re-snapshots from scratch, and the impl's file-list reset is
        # defense-in-depth for any residual dirty state.
        try:
            session.fs_restore(cwd, _guard_snap)
        except RuntimeError as _restore_err:
            print(
                f"[restore-fail-ignored] post-stage worktree restore failed "
                f"(infra-only, does NOT affect this attempt's result): "
                f"{_restore_err}",
                flush=True,
            )
        finally:
            session.fs_drop_snapshot(_guard_snap)


def _run_tests_under_test_patch_impl(
    session: DockerSession,
    *,
    test_patch,
    f2p_tests: list,
    p2p_tests: list,
    base_commit: str,
    pytest_bin: str,
    cwd: str,
    timeout_sec: int,
    instance: Optional[dict] = None,
    collect_node_details: bool = True,
) -> TestPatchRunReport:
    """Run one test set under exactly one temporary test_patch.

    Applies the provided test_patch, collect/run ``f2p_tests + p2p_tests``,
    then reset only that test_patch before returning. By default this runs
    every collectable node individually and records full per-node output, so
    callers can distinguish guard failures from candidate failures without
    losing diagnostics.

    NOTE: This is the *unguarded* implementation. Always call it through
    :func:`run_tests_under_test_patch`, which wraps it in a worktree
    snapshot/restore so runner side effects never leak into later patch
    application.
    """
    nodes = _dedup_keep_order(list(f2p_tests) + list(p2p_tests))
    if is_pro_instance(
        instance,
        image=(instance or {}).get("_config_image"),
        runner_kind=(instance or {}).get("_test_runner_kind"),
    ):
        status, details, out = run_pro_official_tests(
            session,
            instance=instance or {},
            test_patch=test_patch or "",
            requested_nodes=nodes,
            base_commit=base_commit or "HEAD",
            cwd=cwd,
            timeout_sec=timeout_sec,
            apply_unified_diff=apply_unified_diff,
            reset_eval_test_files=reset_eval_test_files,
        )
        failing = [n for n in nodes if status.get(n) != "PASSED"]
        return TestPatchRunReport(
            ok=not failing,
            per_node_status=status,
            failing_node_ids=failing,
            pytest_output=out,
            per_node_details=details,
        )
    if is_verified_instance(
        instance,
        image=(instance or {}).get("_config_image"),
        runner_kind=(instance or {}).get("_test_runner_kind"),
    ):
        status, details, out = run_verified_official_tests(
            session,
            instance=instance or {},
            test_patch=test_patch or "",
            requested_nodes=nodes,
            base_commit=base_commit or "HEAD",
            cwd=cwd,
            timeout_sec=timeout_sec,
            apply_unified_diff=apply_unified_diff,
            reset_eval_test_files=reset_eval_test_files,
        )
        failing = [n for n in nodes if status.get(n) != "PASSED"]
        return TestPatchRunReport(
            ok=not failing,
            per_node_status=status,
            failing_node_ids=failing,
            pytest_output=out,
            per_node_details=details,
        )

    apply_eval_test_patch(
        session, test_patch=test_patch or "",
        base_commit=base_commit or "HEAD", cwd=cwd,
    )
    try:
        if not nodes:
            return TestPatchRunReport(
                ok=True,
                per_node_status={},
                failing_node_ids=[],
                pytest_output="(no tests requested)",
            )
        outputs: list[str] = []
        if collect_node_details:
            status = {}
            details = {}
            for node in nodes:
                node_status, node_detail = _run_pytest_node_detail(
                    session, node, pytest_bin=pytest_bin, cwd=cwd,
                    timeout_sec=timeout_sec,
                )
                status[node] = node_status
                details[node] = node_detail
                outputs.append(
                    f"===== {node} [{node_status}] =====\n"
                    f"{node_detail.get('output_excerpt') or node_detail.get('output') or ''}"
                )
        else:
            collected_set, missing_set, collect_out = _pytest_collect_detail(
                session, nodes, pytest_bin=pytest_bin, cwd=cwd,
            )
            missing = sorted(missing_set)
            collected = [n for n in nodes if n in collected_set]
            status = {
                n: "NOT_FOUND" for n in missing
            }
            details = {
                n: {
                    "status": "NOT_FOUND",
                    "returncode": None,
                    "output_excerpt": collect_out or f"declared but uncollectable: {n}",
                }
                for n in missing
            }
            if not collected:
                out = f"declared but uncollectable: {missing}"
                return TestPatchRunReport(
                    ok=False,
                    per_node_status=status,
                    failing_node_ids=missing,
                    pytest_output=out,
                    per_node_details=details,
                )
            ok_batch, batch_status, batch_out = run_pytest(
                session, collected, pytest_bin=pytest_bin, cwd=cwd,
                timeout_sec=timeout_sec,
            )
            status.update(batch_status)
            outputs.append(batch_out)
            for node, node_status in batch_status.items():
                details[node] = {
                    "status": node_status,
                    "returncode": 0 if node_status == "PASSED" else None,
                    "output_excerpt": batch_out,
                }
            if missing:
                outputs.insert(0, f"declared but uncollectable: {missing}")
        out = "\n".join(outputs)
        failing = [n for n in nodes if status.get(n) != "PASSED"]
        ok = not failing
        return TestPatchRunReport(
            ok=ok,
            per_node_status=status,
            failing_node_ids=failing,
            pytest_output=out,
            per_node_details=details,
        )
    finally:
        reset_eval_test_files(
            session, test_patch=test_patch or "",
            base_commit=base_commit or "HEAD", cwd=cwd,
        )


def run_tests_expect_failure_under_test_patch(
    session: DockerSession,
    *,
    test_patch,
    tests: list,
    pass_tests: Optional[list] = None,
    base_commit: str,
    pytest_bin: str,
    cwd: str,
    timeout_sec: int,
    instance: Optional[dict] = None,
    require_all_fail: bool = True,
    min_fail_fraction: float = 1.0,
    collect_node_details: bool = True,
) -> TestPatchRunReport:
    """Run pre-patch expectations in the current worktree.

    This is the negative/baseline counterpart to
    ``run_tests_under_test_patch``. It applies exactly one temporary
    test_patch, runs ``tests`` that are expected to fail and ``pass_tests`` that
    are expected to pass, then resets that same test_patch.

    The expected-failure side passes when tests are collectable and:

    - ``require_all_fail=True`` (default): every requested test fails/errors.
    - ``require_all_fail=False``: at least ``ceil(len(tests) * min_fail_fraction)``
      requested tests fail/error.  ``min_fail_fraction`` defaults to 1.0 (all),
      pass 0.5 to require at least half, 0.0 to require at least one.

    The expected-pass side passes only when every ``pass_tests`` node passes.

    Both Pro and Verified paths use the same membership semantics: a F2P node
    that is not explicitly ``PASSED`` (including ``NOT_FOUND``, ``FAILED``,
    ``ERROR``, ``TIMEOUT``, or ``None``) is counted as failed-as-expected.
    Only an explicit ``PASSED`` result is treated as an unexpected pass.
    """
    fail_nodes = _dedup_keep_order(list(tests or []))
    pass_nodes = [
        n for n in _dedup_keep_order(list(pass_tests or []))
        if n not in set(fail_nodes)
    ]
    raw_report = run_tests_under_test_patch(
        session,
        test_patch=test_patch or "",
        f2p_tests=fail_nodes,
        p2p_tests=pass_nodes,
        base_commit=base_commit or "HEAD",
        pytest_bin=pytest_bin,
        cwd=cwd,
        timeout_sec=timeout_sec,
        instance=instance,
        collect_node_details=collect_node_details,
    )
    _ok, status, _failing, out = raw_report
    is_pro = is_pro_instance(
        instance,
        image=(instance or {}).get("_config_image"),
        runner_kind=(instance or {}).get("_test_runner_kind"),
    )
    if is_pro and fail_nodes and _looks_like_pro_global_build_failure(out):
        details = getattr(raw_report, "per_node_details", {}) or {}
        for n in fail_nodes:
            if status.get(n) == "NOT_FOUND":
                status[n] = "ERROR"
                detail = details.get(n)
                if isinstance(detail, dict):
                    detail["status"] = "ERROR"
                    detail["pro_global_build_failure"] = True
    if not fail_nodes and not pass_nodes:
        return TestPatchRunReport(
            ok=False,
            per_node_status=status,
            failing_node_ids=[],
            pytest_output="(no pre-patch tests requested)",
            per_node_details=getattr(raw_report, "per_node_details", {}),
        )

    # Unified semantics for both Pro and Verified: any status that is not
    # explicitly PASSED (NOT_FOUND, FAILED, ERROR, TIMEOUT, None) counts as
    # failed-as-expected.  Only an explicit PASSED is an unexpected pass.
    failed_as_expected = [
        n for n in fail_nodes if status.get(n) != "PASSED"
    ]
    invalid = []
    passed_unexpectedly = [n for n in fail_nodes if status.get(n) == "PASSED"]
    pass_failed = [
        n for n in pass_nodes if status.get(n) != "PASSED"
    ]

    if fail_nodes and require_all_fail:
        fail_expectation_ok = (
            len(failed_as_expected) == len(fail_nodes)
            and not invalid
            and not passed_unexpectedly
        )
        unexpected = invalid + passed_unexpectedly
    elif fail_nodes:
        min_required = max(1, math.ceil(len(fail_nodes) * min_fail_fraction))
        fail_expectation_ok = (
            len(failed_as_expected) >= min_required and not invalid
        )
        unexpected = invalid if fail_expectation_ok else invalid + passed_unexpectedly
    else:
        fail_expectation_ok = True
        unexpected = []

    pass_expectation_ok = not pass_failed
    unexpected = unexpected + pass_failed

    expectation_errors: dict[str, str] = {}
    for n in invalid:
        expectation_errors[n] = (
            f"expected failure evidence, got {status.get(n) or 'NO_STATUS'}"
        )
    for n in passed_unexpectedly:
        expectation_errors[n] = "expected this F2P node to fail before patch, but it passed"
    for n in pass_failed:
        expectation_errors[n] = (
            f"expected this P2P node to pass before patch, got {status.get(n) or 'NO_STATUS'}"
        )

    return TestPatchRunReport(
        ok=fail_expectation_ok and pass_expectation_ok,
        per_node_status=status,
        failing_node_ids=unexpected,
        pytest_output=out,
        per_node_details=getattr(raw_report, "per_node_details", {}),
        expectation_errors=expectation_errors,
    )


def _apply_patch_to_worktree(
    session: DockerSession, patch, *, cwd: str, check_net_zero: bool = False,
) -> list[str]:
    """Apply model patch only. Does not apply test_patch or run tests."""
    before_digest = _snapshot_tree_digest(session, cwd) if check_net_zero else ""
    patch = _normalize_patch(patch)
    fmt = patch.get("format", "unified_diff")
    if fmt == "unified_diff":
        diff = patch.get("diff") or ""
        applied = apply_unified_diff(session, diff, cwd=cwd) if diff.strip() else []
    elif fmt == "env_setup":
        applied = _run_env_setup(session, patch.get("commands") or [], cwd=cwd)
    elif fmt == "composite":
        legs = patch.get("legs") or []
        if not legs:
            raise PatchApplyError("composite patch requires non-empty 'legs' list")
        applied = []
        for leg in legs:
            applied.extend(_apply_leg(session, leg, cwd=cwd))
        applied = list(dict.fromkeys(applied))
    else:
        raise PatchApplyError(f"unknown patch format {fmt!r}")
    if check_net_zero:
        after_digest = _snapshot_tree_digest(session, cwd)
        if after_digest != before_digest:
            raise PatchApplyError(
                "[net-zero-fail] rollback ATTEMPT did not restore the worktree "
                f"to its pre-attempt state: before={before_digest} after={after_digest}"
            )
    return applied


# =============================================================================
# Container startup protocol  (AGENT_LOOP_DESIGN §9.0)
# =============================================================================
def bootstrap_session(session: DockerSession, trace: dict) -> None:
    """Prepare a fresh SWE-bench container at base_commit.

    Do **not** apply test_patch here. Official SWE-bench evaluation applies the
    model patch first, then resets files touched by test_patch to base_commit,
    applies test_patch for the test run, and resets those test files again.
    """
    cfg = trace.get("config", {})
    cwd = cfg.get("cwd", "/testbed")
    session.exec(f"cd {shlex.quote(cwd)} && git config --global --add safe.directory {shlex.quote(cwd)}")
    # Pro instances may ship with a dirty worktree (e.g. go.work.sum modified
    # during image build). before_repo_set_cmd resets it to the clean base_commit
    # state so that the seed patch can apply without conflict.
    before_cmd = (trace.get("instance") or {}).get("before_repo_set_cmd", "")
    if before_cmd and before_cmd.strip():
        session.exec(f"cd {shlex.quote(cwd)} && {before_cmd}")


def replay_model_patches(
    session: DockerSession, trace: dict, *, until_event: Optional[dict] = None,
    stop_before_event: Optional[dict] = None,
    only_event: Optional[dict] = None,
    check_net_zero: bool = False,
    verbose: bool = False,
) -> list[str]:
    """Apply model patches from the beginning of trace history.

    Replays enabled ok=true ATTEMPT patches in trace order. If ``until_event``
    is provided, replay stops after that event is applied. If
    ``stop_before_event`` is provided, replay stops immediately before that event
    would be applied. If ``only_event`` is provided, apply just that event's
    model patch to the current worktree. test_patch is never applied here; use
    ``run_tests_under_test_patch`` for test visibility.
    """
    cfg = trace.get("config", {})
    cwd = cfg.get("cwd", "/testbed")
    applied_files: list[str] = []
    if only_event is not None:
        if verbose:
            print(
                f"[executor] replay explicit ATTEMPT "
                f"{only_event.get('big_task_id')}/{only_event.get('subtask_id')}"
            )
        patch = only_event.get("patch") or []
        return _apply_patch_to_worktree(
            session, patch, cwd=cwd, check_net_zero=check_net_zero,
        )

    replayed = 0
    for ev in trace.get("events") or []:
        if ev.get("kind") != "ATTEMPT" or not at.is_enabled(ev):
            continue
        if (ev.get("result") or {}).get("ok") is not True:
            continue
        if stop_before_event is not None and ev is stop_before_event:
            break
        if stop_before_event is not None and (
            ev.get("kind") == stop_before_event.get("kind")
            and ev.get("big_task_id") == stop_before_event.get("big_task_id")
            and ev.get("subtask_id") == stop_before_event.get("subtask_id")
        ):
            break
        if verbose:
            print(f"[executor] replay ok ATTEMPT {ev.get('big_task_id')}/{ev.get('subtask_id')}")
        try:
            applied_files.extend(
                _apply_patch_to_worktree(
                    session, ev.get("patch") or [], cwd=cwd,
                )
            )
            # execute_attempt() cleans eval-only test_patch files after a
            # successful run. Reproduce the same post-success physical state
            # during replay; otherwise ok history patches that touched test
            # files would leak into later worktrees.
            reset_eval_test_files(
                session,
                test_patch=ev.get("test_patch") or "",
                base_commit=(trace.get("instance") or {}).get("base_commit", ""),
                cwd=cwd,
            )
            replayed += 1
        except PatchApplyError as e:
            raise RuntimeError(
                f"[replay-fatal] failed to replay ok ATTEMPT "
                f"{ev.get('big_task_id')}/{ev.get('subtask_id')}: {e}"
            ) from e
        if until_event is not None and ev is until_event:
            break
        if until_event is not None and (
            ev.get("kind") == until_event.get("kind")
            and ev.get("big_task_id") == until_event.get("big_task_id")
            and ev.get("subtask_id") == until_event.get("subtask_id")
        ):
            break
    if verbose:
        print(f"[executor]   step replay-history: ok  attempts={replayed}", flush=True)
    return applied_files


# =============================================================================
# Pure entry: apply patch + run tests + auto-rollback on failure
# =============================================================================
def execute_attempt(
    session: DockerSession,
    *,
    patch: dict,
    trace: Optional[dict] = None,
    patch_event: Optional[dict] = None,
    f2p_tests: list,
    p2p_tests: list,
    prior_regression: list,
    test_patch=None,
    base_commit: str = "",
    pytest_bin: str = "/opt/miniconda3/envs/testbed/bin/pytest",
    cwd: str = "/testbed",
    smoke_import: Optional[str] = "astropy",
    smoke_python: Optional[str] = None,
    timeout_sec: int = 1200,
    include_prior_regression: bool = True,
    verbose: bool = False,
) -> ExecuteReport:
    """
    Apply model `patch`, then evaluate under official SWE-bench test_patch
    semantics.

    Prior-regression loop (in BigTask chronological order):
      For each prior ok ATTEMPT/PLAN entry in ``prior_regression``
      (each entry is ``{"test_patch": list[str], "own_f2p": list[str],
      "own_p2p": list[str]}``):
        apply that entry's test_patch
        run that entry's own_f2p and own_p2p node IDs
        reset that entry's test_patch files
        → if any node fails, rollback and return failure immediately

    Current-ATTEMPT evaluation:
      before applying a normal model patch, verify current own_f2p really fails
      reset files touched by test_patch to base_commit
      apply test_patch
      collect-only validate declared node-ids
      run current f2p + p2p tests
      reset test_patch files to base_commit

    On any failure the container worktree is rolled back to the pre-attempt
    snapshot. On success, the model patch remains in the worktree, while all
    test_patch files are cleaned back out.
    """
    if smoke_python is None:
        smoke_python = pytest_bin.rsplit("/", 1)[0] + "/python"

    # 1) snapshot worktree before the attempt
    pre_snap = session.fs_snapshot(cwd)
    _log_executor_step(
        verbose, "snapshot-pre", ok=True, detail=f"id={pre_snap}",
    )

    def _rollback_and(err: str, **kw) -> ExecuteReport:
        # NB: the rollback happens on the path where "this attempt has already
        # definitively failed and we are about to return an ExecuteReport".
        # If the container died mid-round (e.g. pro_official's
        # [pro-inplace-checkout-fail] container ... is not running), then
        # fs_restore's `test -s <snap>` fails inside the dead container and
        # raises RuntimeError([restore-fail] missing-or-empty ...).
        # This restore is only an "infrastructure action" that returns the
        # worktree to a clean state; whether it succeeds does not change the
        # conclusion that "this attempt already failed"; if we let it fatally
        # raise, it would propagate through BaseFlow.run()'s generic except and
        # burn all 5 attempts into the same infra error (the container is still
        # dead and hits the same spot every round). This is consistent with the
        # [restore-fail-ignored] handling at the run_tests_under_test_patch
        # chokepoint: swallow the restore failure, log it, and let the original
        # error_kind (e.g. TEST_PATCH_APPLY_FAIL) be returned faithfully, so we
        # go through the normal attempt-retry / rescue logic rather than fatal
        # exhaust.
        try:
            session.fs_restore(cwd, pre_snap)
        except RuntimeError as _restore_err:
            print(
                f"[restore-fail-ignored] pre-attempt worktree rollback failed "
                f"(infra-only, does NOT change this attempt's failed result): "
                f"{_restore_err}",
                flush=True,
            )
        finally:
            session.fs_drop_snapshot(pre_snap)
        return ExecuteReport(
            ok=False, error_kind=err, pre_snapshot_id=pre_snap, **kw,
        )

    all_status: dict[str, str] = {}
    all_details: dict[str, dict] = {}
    all_expectation_errors: dict[str, str] = {}
    validation_failures: list[dict] = []
    patch_trace = trace or {
        "config": {"cwd": cwd},
        "instance": {"base_commit": base_commit},
        "events": [],
    }
    instance = dict(patch_trace.get("instance") or {"base_commit": base_commit})
    trace_config = patch_trace.get("config") or {}
    instance["_config_image"] = trace_config.get("image")
    instance["_test_runner_kind"] = trace_config.get("test_runner_kind")
    pro_mode = is_pro_instance(
        instance,
        image=trace_config.get("image"),
        runner_kind=trace_config.get("test_runner_kind"),
    )

    def _record_validation_failure(
        kind: str,
        *,
        stage: str,
        statuses: Optional[dict] = None,
        details: Optional[dict] = None,
        expectation_errors: Optional[dict] = None,
        failing_node_ids: Optional[list] = None,
        output_text: str = "",
        hint_text: str = "",
    ) -> None:
        prefix = f"{stage}::"
        _merge_prefixed_status(all_status, prefix, statuses or {})
        _merge_prefixed_details(all_details, prefix, details or {})
        _merge_prefixed_expectations(all_expectation_errors, prefix, expectation_errors or {})
        _failing = [str(n) for n in (failing_node_ids or []) if str(n)]
        _dets = details or {}
        _stats = statuses or {}
        _exps = expectation_errors or {}
        node_tfd: list = []
        for node in _failing:
            matching_keys = [k for k in _dets if k == node or str(k).endswith(f"::{node}")]
            if not matching_keys:
                matching_keys = [node]
            for key in matching_keys:
                nd = _dets.get(key)
                nd = nd if isinstance(nd, dict) else {}
                exp_err = _exps.get(key) or _exps.get(node) or ""
                out_exc = nd.get("output_excerpt") or ""
                # COLLECT_MISS / NOT_FOUND: expectation_error is usually empty,
                # but output_excerpt carries the pytest collect output (including
                # the raw ImportError / SyntaxError text). Pass it through to
                # expectation_error so the retry prompt can see the root cause
                # directly rather than just "NOT_FOUND".
                if not exp_err and out_exc:
                    exp_err = out_exc
                node_tfd.append({
                    "node_id": node,
                    "phase": stage,
                    "status": nd.get("status") or _stats.get(key) or _stats.get(node) or "failed",
                    "expectation_error": exp_err,
                    "output_excerpt": out_exc,
                })
        # Summary entry: combines raw output tail and the [DATA BUG] hint
        combined = (
            (output_text.strip() + "\n\n" + hint_text.strip()).strip()
            if output_text.strip() and hint_text.strip()
            else (output_text.strip() or hint_text.strip())
        )
        if combined or not node_tfd:
            node_tfd.append({
                "node_id": f"<{stage}>",
                "phase": kind,
                "status": "failed",
                "expectation_error": combined,
                "output_excerpt": "",
            })
        validation_failures.append({
            "kind": kind,
            "stage": stage,
            "failing_node_ids": _failing,
            "test_failure_details": node_tfd,
        })

    def _finalize_validation_failure(default_kind: str) -> ExecuteReport:
        kinds = [str(item.get("kind") or "") for item in validation_failures if item.get("kind")]
        unique_kinds = list(dict.fromkeys(kinds))
        error_kind = unique_kinds[0] if len(unique_kinds) == 1 else "MULTI_CODE_VALIDATION_FAIL"
        if not unique_kinds:
            error_kind = default_kind
        all_failing = _dedup_keep_order([
            str(n)
            for item in validation_failures
            for n in (item.get("failing_node_ids") or [])
            if str(n)
        ])
        all_tfd: list = []
        for item in validation_failures:
            all_tfd.extend(item.get("test_failure_details") or [])
        if error_kind == "MULTI_CODE_VALIDATION_FAIL":
            stages_summary = ", ".join(
                f"{item.get('stage', '?')}({item.get('kind', '?')})"
                for item in validation_failures
            )
            all_tfd.append({
                "node_id": "<multi-validation>",
                "phase": "MULTI_CODE_VALIDATION_FAIL",
                "status": "failed",
                "expectation_error": (
                    f"[DATA BUG] Multiple validation phases failed simultaneously (MULTI_CODE_VALIDATION_FAIL): {stages_summary}."
                    " Check each phase's test_failure_details entry above; each contains specific [DATA BUG] fix guidance."
                    " Fix method: prioritize fixing the earliest failing phase; typically patch application issues (PATCH_APPLY_FAIL) and"
                    " test_patch issues (TEST_PATCH_APPLY_FAIL) appear before PYTEST_FAIL and should be resolved first."
                ),
                "output_excerpt": "",
            })
        return _rollback_and(
            error_kind,
            per_node_status=all_status,
            per_node_details=all_details,
            expectation_errors=all_expectation_errors,
            failing_node_ids=all_failing,
            applied_files=applied,
            test_failure_details=all_tfd,
        )

    def _apply_event_patch(event_patch, *, check_net_zero: bool) -> list[str]:
        event = dict(patch_event or {
            "kind": "ATTEMPT",
            "big_task_id": "<current>",
            "subtask_id": "<current>",
        })
        event["patch"] = event_patch
        return replay_model_patches(
            session, patch_trace, only_event=event,
            check_net_zero=check_net_zero, verbose=False,
        )

    def _requires_net_zero() -> bool:
        if not patch_event:
            return False
        bid = patch_event.get("big_task_id")
        sid = patch_event.get("subtask_id")
        for plan in reversed(patch_trace.get("events") or []):
            if plan.get("kind") != "PLAN" or plan.get("big_task_id") != bid:
                continue
            for subtask in plan.get("subtasks_blueprint") or []:
                if (
                    isinstance(subtask, dict)
                    and subtask.get("id") == sid
                    and subtask.get("category") == "erroneous_request_then_retract"
                ):
                    return True
            return False
        return False

    def _run_test_stage_with_restore(fn):
        """Run a test-only stage.

        Worktree restoration is now handled *inside* run_tests_under_test_patch
        (and therefore run_tests_expect_failure_under_test_patch), which every
        ``fn`` passed here ultimately calls. Keeping a second snapshot/restore
        layer would just double the per-stage tar cost (and worsen timeouts) for
        no correctness gain, so this is a thin passthrough retained only so the
        call sites read clearly.
        """
        return fn()

    # 2) apply model patch (skip cleanly if the unified diff is empty)
    applied: list = []
    pre_f2p_nodes = _dedup_keep_order(list(f2p_tests))
    pre_p2p_nodes = _dedup_keep_order(list(p2p_tests))
    suspect_nodes: set = set()
    original_f2p_count = len(pre_f2p_nodes)
    if pre_f2p_nodes or pre_p2p_nodes:
        try:
            pre_f2p_report = _run_test_stage_with_restore(
                lambda: run_tests_expect_failure_under_test_patch(
                    session,
                    test_patch=test_patch or "",
                    tests=pre_f2p_nodes,
                    pass_tests=pre_p2p_nodes,
                    base_commit=base_commit or "HEAD",
                    pytest_bin=pytest_bin,
                    cwd=cwd,
                    timeout_sec=timeout_sec,
                    instance=instance,
                    require_all_fail=True,
                )
            )
            pre_f2p_failed, status_pre_f2p, unexpected_pre_f2p, out_pre_f2p = pre_f2p_report
            suspect_nodes = {n for n in pre_f2p_nodes if status_pre_f2p.get(n) == "PASSED"}
            # Round-1 only: treat TIMEOUT as "failing".  If no f2p test PASSED
            # (all are FAILED / ERROR / TIMEOUT) the pre-f2p gate is satisfied.
            # Round-2 (post-patch) TIMEOUT handling is unchanged.
            if not pre_f2p_failed and not suspect_nodes:
                if any(status_pre_f2p.get(n) == "TIMEOUT" for n in pre_f2p_nodes):
                    pre_f2p_failed = True
        except PatchApplyError as e:
            _log_executor_step(
                verbose, "pre-f2p/p2p", ok=False,
                detail="test_patch apply failed", output=str(e),
            )
            return _rollback_and(
                "TEST_PATCH_APPLY_FAIL", applied_files=applied, test_failure_details=[{
                    "node_id": "<pre-f2p-p2p>",
                    "phase": "TEST_PATCH_APPLY_FAIL",
                    "status": "failed",
                    "expectation_error": str(e) + "\n\n[DATA BUG] test_patch cannot be applied by git apply in the pre-f2p/p2p phase (baseline pre-check before model patch is applied) (TEST_PATCH_APPLY_FAIL). Possible causes: (1) file paths in test_patch do not match the repository's actual directory structure; (2) diff hunk context lines do not match baseline file content; (3) unified diff format error (missing required header or malformed @@ line). Fix method: after reset_worktree (clean baseline with only previously successful patches), run git apply --check <test_patch> with docker_cli to confirm the patch applies cleanly; verify file path spelling and hunk context match the baseline file content line by line.",
                    "output_excerpt": "",
                }],
            )
        all_status.update({
            f"pre_f2p::{k}": ("EXPECTED_FAIL" if v != "PASSED" else v)
            for k, v in status_pre_f2p.items()
        })
        _merge_prefixed_details(
            all_details, "pre_f2p::", pre_f2p_report.per_node_details,
        )
        _merge_prefixed_expectations(
            all_expectation_errors,
            "pre_f2p::",
            pre_f2p_report.expectation_errors,
        )
        _log_executor_step(
            verbose, "pre-f2p/p2p", ok=pre_f2p_failed,
            detail=(
                f"expect_fail={len(pre_f2p_nodes)} "
                f"expect_pass={len(pre_p2p_nodes)} unexpected={len(unexpected_pre_f2p)}"
            ),
            output=out_pre_f2p,
        )
        if not pre_f2p_failed:
            tail_pre_f2p = _tail_lines(out_pre_f2p)
            kind_pre_f2p = (
                "TIMEOUT" if any(v == "TIMEOUT" for v in status_pre_f2p.values())
                else "COLLECT_MISS" if any(v == "NOT_FOUND" for v in status_pre_f2p.values())
                else "P2P_PRE_FAIL" if any(
                    status_pre_f2p.get(n) != "PASSED" for n in pre_p2p_nodes
                )
                else "F2P_PRE_NOT_FAILING"
            )
            if kind_pre_f2p == "F2P_PRE_NOT_FAILING":
                # suspect_nodes already captured; these will be demoted/deleted
                # after post-check — do NOT record a validation failure here.
                pass
            else:
                if kind_pre_f2p == "P2P_PRE_FAIL":
                    if pro_mode:
                        _hint = (
                            "[DATA BUG] own_p2p contains a selector that fails the Pro official test script on a clean baseline"
                            " (P2P_PRE_FAIL). P2P tests must pass both before and after applying the model patch."
                            " Fix method: after reset_worktree, use the instance's official run_script.sh and parser.py"
                            " to reproduce selected_test_files_to_run/selector and confirm its baseline status is PASSED;"
                            " selectors that fail on the baseline should not be placed in own_p2p."
                        )
                    else:
                        _hint = (
                            "[DATA BUG] own_p2p contains test nodes that already fail on the clean baseline (P2P_PRE_FAIL)."
                            " P2P = Pass-to-Pass: tests must pass both before and after applying the patch."
                            " If a test fails on the baseline, it is F2P not P2P and should not be written to own_p2p."
                            " Fix method: only write existing behavior assertions that already pass stably on the baseline to own_p2p;"
                            " run pytest immediately after reset_worktree with docker_cli to confirm all these nodes PASSED."
                        )
                elif kind_pre_f2p == "COLLECT_MISS":
                    if pro_mode:
                        _hint = (
                            "[DATA BUG] The Pro official parser did not return the declared test selector (COLLECT_MISS/NOT_FOUND)."
                            " Possible causes: the selector format is inconsistent with the test files/test names accepted by run_script.sh,"
                            " test_patch did not create the corresponding test, or parser.py output names do not match own_f2p/own_p2p."
                            " Fix method: after applying test_patch, run the instance's run_script.sh directly,"
                            " then check tests[].name in the output.json generated by parser.py,"
                            " and ensure the declared selector matches that name and the selected_test_files_to_run rules."
                        )
                    else:
                        _hint = (
                            "[DATA BUG] pytest cannot collect the declared test node (COLLECT_MISS)."
                            " Possible causes: malformed node-id format, incorrect test file path, function name spelling error,"
                            " or test_patch failed to correctly add the test."
                            " Fix method: after applying test_patch with docker_cli, run pytest --collect-only"
                            " to verify the node-id can be collected; ensure file paths in test_patch match the repository's actual paths."
                        )
                elif kind_pre_f2p == "TIMEOUT":
                    _hint = (
                        "[DATA BUG] Test run timed out (TIMEOUT)."
                        " Possible causes: test entered an infinite loop, depends on network or external services, or takes far longer than expected."
                        " Fix method: run the timed-out node individually with docker_cli to locate the root cause;"
                        " consider replacing slow I/O with mocks, or choose a lighter-weight test node."
                    )
                else:
                    _hint = ""
                _record_validation_failure(
                    kind_pre_f2p,
                    stage="pre-f2p-p2p",
                    statuses=status_pre_f2p,
                    details=pre_f2p_report.per_node_details,
                    expectation_errors=pre_f2p_report.expectation_errors,
                    failing_node_ids=unexpected_pre_f2p,
                    output_text=(tail_pre_f2p or ""),
                    hint_text=_hint,
                )
    try:
        applied = _apply_event_patch(patch, check_net_zero=_requires_net_zero())
    except PatchApplyError as e:
        msg = str(e)
        err = "ROLLBACK_NOT_NET_ZERO" if "net-zero-fail" in msg else "PATCH_APPLY_FAIL"
        _log_executor_step(
            verbose, "apply-model-patch", ok=False,
            detail=f"err={err}", output=msg,
        )
        _patch_hint = (
            "[DATA BUG] The model patch cannot be applied by git apply (PATCH_APPLY_FAIL)."
            " Possible causes: (1) the patch's diff hunk context lines do not match the current worktree file content"
            " (other previously successful patches have already modified those lines);"
            " (2) unified diff format error (wrong @@ line numbers, insufficient context lines, missing required headers);"
            " (3) the patch depends on a file that does not yet exist, or modifies a file already deleted by a previous patch."
            " Fix method: after reset_worktree (worktree = base_commit + all previously successful patches),"
            " run git apply --check with docker_cli to verify the patch applies;"
            " confirm hunk context matches the file content after reset_worktree line by line."
        ) if err == "PATCH_APPLY_FAIL" else (
            "[DATA BUG] After the rollback ATTEMPT executed, the workspace was not restored to its pre-execution state"
            " (ROLLBACK_NOT_NET_ZERO)."
            " This means the modifications produced by the erroneous request were not fully undone by the subsequent rollback steps."
            " Fix method: inspect each leg of the composite patch and ensure the final file content is byte-for-byte identical to"
            " the state before the ATTEMPT executed, with no leftover new files, configuration changes, or source code modifications."
        )
        return _rollback_and(err, applied_files=applied, test_failure_details=[{
            "node_id": "<apply-model-patch>",
            "phase": err,
            "status": "failed",
            "expectation_error": msg + "\n\n" + _patch_hint,
            "output_excerpt": "",
        }])
    _log_executor_step(
        verbose, "apply-model-patch", ok=True,
        detail=f"files={len(applied)} {_short_nodes(applied)}",
    )

    # 3) smoke import (catch SyntaxError / ImportError before pytest)
    if smoke_import:
        sp = session.exec(
            f"cd {shlex.quote(cwd)} && "
            f"{shlex.quote(smoke_python)} -c "
            f"{shlex.quote('import ' + smoke_import)}",
            timeout=180,
        )
        _log_executor_step(
            verbose, "smoke-import", ok=(sp.returncode == 0),
            detail=f"module={smoke_import}", output=sp.stdout + "\n" + sp.stderr,
        )
        if sp.returncode != 0:
            tail = _tail_lines(sp.stdout + "\n" + sp.stderr)
            _record_validation_failure(
                "PATCH_SMOKE_FAIL",
                stage="smoke-import",
                output_text=(tail or ""),
                hint_text=(
                    "[DATA BUG] After applying the model patch, the Python import smoke check failed (PATCH_SMOKE_FAIL)."
                    " This typically means the patch introduced a syntax error (e.g., indentation error, mismatched brackets, illegal expression,"
                    " Chinese quotes replacing ASCII quotes), or the patch broke a module's import chain"
                    " (removed a public name depended on by other modules, or modified __all__ causing import failure)."
                    " Fix method: (1) after applying the patch with docker_cli, run"
                    " python -c 'import <module>' to reproduce and view the full error traceback;"
                    " (2) check that indentation in the patch is consistent (mixing tab/space causes IndentationError);"
                    " (3) check that brackets, quotes, and colons are properly paired;"
                    " (4) ensure the patch does not remove public symbols that other modules are currently importing."
                ),
            )

    # 4) prior-regression loop: test each prior entry through the single
    # test_patch runner. Each entry owns exactly one test_patch plus its own
    # F2P/P2P split, which must stay intact.
    if include_prior_regression:
        for entry in prior_regression or []:
            prior_tp = entry.get("test_patch") or ""
            prior_f2p = entry.get("own_f2p") or []
            prior_p2p = entry.get("own_p2p") or []
            if not prior_f2p and not prior_p2p:
                continue
            try:
                prior_report = _run_test_stage_with_restore(
                    lambda: run_tests_under_test_patch(
                        session,
                        test_patch=prior_tp,
                        f2p_tests=prior_f2p,
                        p2p_tests=prior_p2p,
                        base_commit=base_commit or "HEAD",
                        pytest_bin=pytest_bin,
                        cwd=cwd,
                        timeout_sec=timeout_sec,
                        instance=instance,
                    )
                )
                ok_r, status_r, failing_r, out_r = prior_report
            except PatchApplyError as e:
                _log_executor_step(
                    verbose, "prior-regression", ok=False,
                    detail="test_patch apply failed", output=str(e),
                )
                return _rollback_and(
                    "TEST_PATCH_APPLY_FAIL", applied_files=applied, test_failure_details=[{
                        "node_id": "<prior-regression>",
                        "phase": "TEST_PATCH_APPLY_FAIL",
                        "status": "failed",
                        "expectation_error": str(e) + "\n\n[DATA BUG] The prior subtask's test_patch cannot be applied by git apply in the current worktree state (after the model patch has been applied) (TEST_PATCH_APPLY_FAIL). Possible causes: the new model patch modified files covered by the prior test_patch, causing hunk context to no longer match; or the file referenced by the prior test_patch was renamed or deleted by the new patch. Fix method: (1) check whether the new model patch accidentally modified the path or content of prior test files; (2) if the new patch must modify these files, update the prior test_patch's hunk context to be compatible with the new state; (3) after applying the new model patch with docker_cli, run git apply --check <prior_test_patch> to locate the specific conflicting lines.",
                        "output_excerpt": "",
                    }],
                )
            all_status.update(status_r)
            _merge_prefixed_details(all_details, "prior::", prior_report.per_node_details)
            _log_executor_step(
                verbose, "prior-regression", ok=ok_r,
                detail=(
                    f"tests={len(prior_f2p) + len(prior_p2p)} "
                    f"failing={len(failing_r)}"
                ),
                output=out_r,
            )
            if not ok_r:
                tail_r = _tail_lines(out_r)
                kind_r = (
                    "TIMEOUT" if any(v == "TIMEOUT" for v in status_r.values())
                    else "COLLECT_MISS" if any(v == "NOT_FOUND" for v in status_r.values())
                    else "PYTEST_FAIL"
                )
                if kind_r == "PYTEST_FAIL":
                    runner_name = "Pro official test script" if pro_mode else "pytest"
                    _hint_r = (
                        f"[DATA BUG] The current model patch causes previously passing historical tests to regress and fail under {runner_name}"
                        " (prior-regression PYTEST_FAIL)."
                        " This means the new patch broke behavior that a previous subtask had already verified."
                        " Fix: (1) locate the specific regressed tests from failing_node_ids and test_failure_details;"
                        " (2) check whether the new patch accidentally modified the code paths those tests cover;"
                        " (3) preserve the observable behavior of completed subtasks;"
                        " (4) for Pro instances, check both the official parser.py status and stdout/stderr."
                    )
                elif kind_r == "COLLECT_MISS":
                    if pro_mode:
                        _hint_r = (
                            "[DATA BUG] The Pro official parser did not return the historical test selector"
                            " (prior-regression COLLECT_MISS/NOT_FOUND)."
                            " Possible causes: the new patch changed the test file/name, run_script.sh did not run that selector,"
                            " or parser.py's output name no longer matches."
                            " Fix: check the official output.json, stdout.log and stderr.log;"
                            " keep the historical selector runnable and tests[].name consistent with own_f2p/own_p2p."
                        )
                    else:
                        _hint_r = (
                            "[DATA BUG] The historical test node cannot be collected by pytest in the current worktree"
                            " (prior-regression COLLECT_MISS)."
                            " Possible causes: the new patch renamed or deleted the file/class/function the historical test lives in,"
                            " invalidating the previously declared node-id."
                            " Fix: check whether the new patch accidentally changed the path or structure of the historical test file;"
                            " if a rename is needed, update the node-id in own_f2p / own_p2p accordingly."
                        )
                elif kind_r == "TIMEOUT":
                    _hint_r = (
                        "[DATA BUG] The historical test timed out in the current worktree"
                        " (prior-regression TIMEOUT)."
                        " Possible causes: the new patch introduced an infinite loop or greatly reduced the execution efficiency of the tested function."
                        " Fix: use docker_cli to run the timed-out node in isolation to locate the root cause;"
                        " check whether the new patch introduced an infinite loop or inefficient recursion."
                    )
                else:
                    _hint_r = ""
                _record_validation_failure(
                    kind_r,
                    stage="prior-regression",
                    statuses=status_r,
                    details=prior_report.per_node_details,
                    expectation_errors=prior_report.expectation_errors,
                    failing_node_ids=failing_r,
                    output_text=(tail_r or ""),
                    hint_text=_hint_r,
                )

    # 5) current ATTEMPT evaluation through the single test_patch runner.
    declared = _dedup_keep_order(list(f2p_tests) + list(p2p_tests))
    try:
        current_report = _run_test_stage_with_restore(
            lambda: run_tests_under_test_patch(
                session,
                test_patch=test_patch or "",
                f2p_tests=f2p_tests,
                p2p_tests=p2p_tests,
                base_commit=base_commit or "HEAD",
                pytest_bin=pytest_bin,
                cwd=cwd,
                timeout_sec=timeout_sec,
                instance=instance,
            )
        )
        ok, status, failing, out = current_report
    except PatchApplyError as e:
        _log_executor_step(
            verbose, "current-tests", ok=False,
            detail="test_patch apply failed", output=str(e),
        )
        return _rollback_and(
            "TEST_PATCH_APPLY_FAIL", applied_files=applied, test_failure_details=[{
                "node_id": "<current-tests>",
                "phase": "TEST_PATCH_APPLY_FAIL",
                "status": "failed",
                "expectation_error": str(e) + "\n\n[DATA BUG] The current ATTEMPT's test_patch cannot be applied by git apply (TEST_PATCH_APPLY_FAIL). At this stage the model patch has been applied, and applying test_patch on top of that state failed. Possible causes: (1) file paths in test_patch do not match the repository's actual directory structure; (2) the diff hunk context lines do not match the file content after the model patch was applied; (3) malformed unified diff (missing diff --git header, --- / +++ lines, or wrong @@ line format); (4) test_patch depends on code that the model patch deleted or renamed. Fix: use docker_cli to run git apply --check <test_patch> after applying the model patch to confirm it applies; check whether the test_patch hunk context matches the file state after the model patch line by line.",
                "output_excerpt": "",
            }],
        )
    if not declared:
        if validation_failures:
            return _finalize_validation_failure("MULTI_CODE_VALIDATION_FAIL")
        _log_executor_step(
            verbose, "current-tests", ok=True,
            detail="tests=0",
        )
        session.fs_drop_snapshot(pre_snap)
        return ExecuteReport(
            ok=True, error_kind=None,
            per_node_status=all_status,
            per_node_details=all_details,
            expectation_errors=all_expectation_errors,
            failing_node_ids=[],
            test_failure_details=[],
            applied_files=applied,
        )

    all_status.update(status)
    _merge_prefixed_details(all_details, "current::", current_report.per_node_details)
    tail = _tail_lines(out)

    # --- F2P demotion: classify suspect nodes based on post-patch status ---
    demoted_to_p2p: list = []
    deleted_from_f2p: list = []
    if suspect_nodes:
        # Round-1 (pre-patch) false F2P: a node declared as own_f2p that already
        # PASSED on the clean baseline before the model patch is applied. These
        # are NOT demoted to own_p2p anymore — a test that passes on the baseline
        # tells us nothing about the current subtask's behavioral change and only
        # adds noise to the regression set. We discard them from own_f2p outright,
        # regardless of their post-patch status.
        for n in suspect_nodes:
            deleted_from_f2p.append(n)
        _log_executor_step(
            verbose, "f2p-demotion", ok=True,
            detail=(
                f"suspects={len(suspect_nodes)} "
                f"demoted={len(demoted_to_p2p)} deleted={len(deleted_from_f2p)} "
                f"(round-1 false F2P discarded, not demoted to p2p)"
            ),
        )

    # --- 50% threshold: genuine F2P must be >= 50% of original own_f2p ---
    if original_f2p_count > 0:
        genuine_f2p_count = sum(
            1 for n in pre_f2p_nodes
            if n not in suspect_nodes and status.get(n) == "PASSED"
        )
        if genuine_f2p_count < original_f2p_count * 0.5:
            _hint_threshold = (
                f"[DATA BUG] The number of genuine F2P ({genuine_f2p_count}) is below 50% of the original declaration"
                f" (original {original_f2p_count})."
                " F2P_INSUFFICIENT: some tests already passed before the patch was applied (fake F2P) and also pass or fail post-patch,"
                " or the number of correct F2P is too small. Demoted and removed nodes do not count toward genuine F2P."
                " Fix: ensure own_f2p contains only test nodes that genuinely verify functionality newly added by the current patch;"
                " use docker_cli to confirm the node is FAILED after reset_worktree and PASSED after applying the patch."
            )
            _record_validation_failure(
                "F2P_INSUFFICIENT",
                stage="f2p-threshold",
                statuses=status,
                details=current_report.per_node_details,
                expectation_errors=current_report.expectation_errors,
                failing_node_ids=list(suspect_nodes),
                output_text=tail or "",
                hint_text=_hint_threshold,
            )

    _log_executor_step(
        verbose, "current-tests", ok=ok,
        detail=f"tests={len(declared)} failing={len(failing)}",
        output=out,
    )

    if not ok:
        kind = (
            "TIMEOUT" if any(v == "TIMEOUT" for v in status.values())
            else "COLLECT_MISS" if any(v == "NOT_FOUND" for v in status.values())
            else "PYTEST_FAIL"
        )
        if kind == "PYTEST_FAIL":
            runner_name = "Pro official run_script.sh/parser.py" if pro_mode else "pytest"
            _hint_cur = (
                f"[DATA BUG] The current ATTEMPT's tests were judged failing by {runner_name} after applying the model patch"
                " (current-tests PYTEST_FAIL)."
                " Possible causes: (1) the model patch's behavior does not match the test expectations;"
                " (2) F2P still fails after the patch; (3) P2P was broken by the model patch;"
                " (4) the Pro parser parsed the script output as FAILED/ERROR."
                " Fix: (1) locate the failing tests from test_failure_details / failing_node_ids;"
                " (2) for Pro instances check stdout.log, stderr.log and output.json, for normal instances check the pytest output;"
                " (3) confirm F2P verifies the new behavior and P2P preserves existing behavior."
            )
        elif kind == "COLLECT_MISS":
            if pro_mode:
                _hint_cur = (
                    "[DATA BUG] The Pro official parser did not return the declared test selector in output.json"
                    " (current-tests COLLECT_MISS/NOT_FOUND)."
                    " Possible causes: (1) own_f2p/own_p2p used pytest node-ids, but this instance's script accepts files or"
                    " other selectors; (2) test_patch did not create the corresponding tests;"
                    " (3) parser.py's tests[].name does not match the declared value."
                    " Fix: check run_script.sh's argument handling and the output.json generated by parser.py;"
                    " use the selector this instance actually supports, and ensure the declared value matches tests[].name."
                )
            else:
                _hint_cur = (
                    "[DATA BUG] pytest cannot collect the declared test node (current-tests COLLECT_MISS)."
                    " Possible causes: (1) the node-id in own_f2p / own_p2p does not match the actual"
                    " test path/class name/function name written by test_patch;"
                    " (2) test_patch has a syntax error that prevents pytest from parsing the file;"
                    " (3) the test_patch path points to a file that does not exist in the repository."
                    " Fix: use docker_cli to run pytest --collect-only after applying test_patch"
                    " to verify each node-id can be collected; ensure the file paths in test_patch match the repository's actual structure."
                )
        elif kind == "TIMEOUT":
            _hint_cur = (
                "[DATA BUG] The current ATTEMPT's test run timed out (current-tests TIMEOUT)."
                " Possible causes: the test depends on the network/external services, entered an infinite loop, or a single test case takes too long to run."
                " Fix: use docker_cli to run the timed-out node in isolation to locate the root cause;"
                " consider replacing slow I/O with mocks; or replace the timed-out node with a lighter equivalent test."
            )
        else:
            _hint_cur = ""
        _record_validation_failure(
            kind,
            stage="current-tests",
            statuses=status,
            details=current_report.per_node_details,
            expectation_errors=current_report.expectation_errors,
            failing_node_ids=failing,
            output_text=(tail or ""),
            hint_text=_hint_cur,
        )

    if validation_failures:
        return _finalize_validation_failure("MULTI_CODE_VALIDATION_FAIL")

    # 6) success: the model patch stays in the worktree, while the temporary
    # rollback snapshot is no longer needed.
    session.fs_drop_snapshot(pre_snap)
    return ExecuteReport(
        ok=True, error_kind=None,
        per_node_status=all_status,
        per_node_details=all_details,
        expectation_errors=all_expectation_errors,
        failing_node_ids=[],
        test_failure_details=[],
        applied_files=applied,
        demoted_to_p2p=demoted_to_p2p,
        deleted_from_f2p=deleted_from_f2p,
    )


# =============================================================================
# Seed task precheck
# =============================================================================
SEED_BIG_TASK_ID = "SEED"
SEED_SUBTASK_ID = "SEED_TASK"


def _instance_node_list(inst: dict, parsed_key: str, raw_key: str) -> list[str]:
    parsed = inst.get(parsed_key)
    if isinstance(parsed, list):
        return [str(x) for x in parsed]
    return [str(x) for x in at._parse_swebench_test_list(inst.get(raw_key))]


def is_seed_attempt(ev: dict) -> bool:
    return (
        ev.get("kind") == "ATTEMPT"
        and ev.get("big_task_id") == SEED_BIG_TASK_ID
        and ev.get("subtask_id") == SEED_SUBTASK_ID
    )


def build_seed_attempt(instance: dict) -> dict:
    """Represent the SWE-bench instance gold patch as the trace's first
    ATTEMPT. This is a normal ATTEMPT event, but it belongs to the isolated
    SEED big_task_id, so its own_f2p/own_p2p do not bleed into BIG_1 unless a
    caller explicitly chooses to do so.

    The goal text is intentionally a neutral, content-free placeholder: the
    SEED attempt must never leak the raw problem statement / reference-patch
    framing into any downstream consumer (e.g. the user-sim scenario).
    """
    goal = "seed precheck"
    return {
        "kind": "ATTEMPT",
        "big_task_id": SEED_BIG_TASK_ID,
        "subtask_id": SEED_SUBTASK_ID,
        "enabled": True,
        "natural_language_goal": goal,
        "test_patch": (
            [instance.get("test_patch") or ""]
            if str(instance.get("test_patch") or "").strip()
            else []
        ),
        "acceptance_criteria": "",
        "own_f2p": _instance_node_list(instance, "FAIL_TO_PASS_parsed", "FAIL_TO_PASS"),
        "own_p2p": _instance_node_list(instance, "PASS_TO_PASS_parsed", "PASS_TO_PASS"),
        "user_query": {"role": "user", "content": goal},
        "patch": [{"format": "unified_diff", "diff": instance.get("patch") or ""}],
        "rationale": (
            f"seed task precheck; repo={instance.get('repo') or ''}; "
            f"base_commit={instance.get('base_commit') or ''}"
        ),
        "result": None,
    }


def ensure_seed_attempt(trace: dict) -> dict:
    """Ensure trace.events[0] is the SEED/SEED_TASK ATTEMPT and return it."""
    events = trace.setdefault("events", [])
    if events and is_seed_attempt(events[0]):
        return events[0]
    for i, ev in enumerate(events):
        if is_seed_attempt(ev):
            seed = events.pop(i)
            events.insert(0, seed)
            return seed
    seed = build_seed_attempt(trace.get("instance", {}) or {})
    events.insert(0, seed)
    return seed


def run_seed_precheck(
    trace_path: str,
    *,
    session: Optional[DockerSession] = None,
    stop_on_fail: bool = True,
    verbose: bool = True,
) -> dict:
    """Append/move the SEED ATTEMPT to events[0], execute it if pending,
    write back result, and mark the trace skipped when it fails.

    This function intentionally keeps SEED isolated from later BIG_* prior
    regression. It also does not run PLAN final verification.
    """
    trace = at.load_trace(trace_path)
    seed = ensure_seed_attempt(trace)
    at.save_trace(trace_path, trace)

    own_session = session is None
    if seed.get("result") is None:
        cfg = trace["config"]
        if own_session:
            session = DockerSession(image=cfg["image"], keep_on_fail=False,
                                    login_shell=(cfg.get("test_runner_kind") != "pro_official"))
            session.start()
            if verbose:
                print(f"[executor]   step docker-start: ok  image={cfg['image']}", flush=True)
            bootstrap_session(session, trace)
            if verbose:
                print(f"[executor]   step bootstrap: ok  cwd={cfg.get('cwd')}", flush=True)
        try:
            _run_one_attempt(trace, seed, session, verbose=verbose)
            at.save_trace(trace_path, trace)
        finally:
            if own_session:
                session.stop()
                if verbose:
                    print("[executor]   step docker-stop: ok", flush=True)

    res = seed.get("result") or {}
    if res.get("ok") is not True and stop_on_fail:
        state = trace.setdefault("state", {})
        state["current_phase"] = "GIVE_UP"
        state["terminal"] = "SKIP_SEED_PRECHECK_FAILED"
        at.save_trace(trace_path, trace)
        if verbose:
            print(
                f"[executor] seed precheck failed; skip instance "
                f"err={res.get('error_kind')}",
                flush=True,
            )
    return trace


# =============================================================================
# Trace-driven driver
# =============================================================================
def _run_one_attempt(trace: dict, ev: dict, session: DockerSession,
                     *, verbose: bool) -> ExecuteReport:
    """Execute a single pending ATTEMPT event and write its result back."""
    cfg = trace["config"]
    inst = trace.get("instance", {}) or {}
    bid = ev.get("big_task_id")
    patch = ev.get("patch")
    test_patch = ev.get("test_patch", "")
    f2p, p2p = at.attempt_own_tests(ev)
    if verbose:
        idx = at.event_index(trace, ev)
        print(f"\n[executor] ATTEMPT  @{idx}  {bid}/{ev.get('subtask_id')}")
    rep = execute_attempt(
        session,
        patch=patch,
        trace=trace,
        patch_event=ev,
        f2p_tests=f2p,
        p2p_tests=p2p,
        prior_regression=at.compute_cumulative_regression(trace, big_task_id=bid),
        test_patch=test_patch or "",
        base_commit=inst.get("base_commit", ""),
        pytest_bin=cfg["pytest_bin"],
        cwd=cfg["cwd"],
        smoke_import=cfg.get("smoke_import"),
        verbose=verbose,
    )
    ev["result"] = rep.to_dict()
    # Write back F2P demotions/deletions to the event fields
    if rep.demoted_to_p2p or rep.deleted_from_f2p:
        demoted_set = set(rep.demoted_to_p2p)
        deleted_set = set(rep.deleted_from_f2p)
        current_f2p = list(ev.get("own_f2p") or [])
        current_p2p = list(ev.get("own_p2p") or [])
        ev["own_f2p"] = [n for n in current_f2p if n not in demoted_set and n not in deleted_set]
        existing_p2p_set = set(current_p2p)
        ev["own_p2p"] = current_p2p + [n for n in rep.demoted_to_p2p if n not in existing_p2p_set]
        if verbose:
            if rep.demoted_to_p2p:
                print(
                    f"[executor]   f2p-demotion: demoted to own_p2p {rep.demoted_to_p2p}",
                    flush=True,
                )
            if rep.deleted_from_f2p:
                print(
                    f"[executor]   f2p-demotion: deleted from own_f2p {rep.deleted_from_f2p}",
                    flush=True,
                )
    if verbose:
        key = _key_output_line(next((d.get("expectation_error") or "" for d in rep.test_failure_details if d.get("expectation_error")), ""))
        key_part = f"  key={key}" if key else ""
        print(
            f"[executor]   result ok={rep.ok}  err={rep.error_kind}  "
            f"#tests={len(rep.per_node_status)}  "
            f"failing={_short_nodes(rep.failing_node_ids)}{key_part}",
            flush=True,
        )
    return rep


def run_pending_attempts(
    trace_path: str,
    *,
    session: Optional[DockerSession] = None,
    stop_on_fail: bool = True,
    verbose: bool = True,
) -> dict:
    """Execute enabled pending ATTEMPT events in array order."""
    trace = at.load_trace(trace_path)
    cfg = trace["config"]

    own_session = session is None
    if own_session:
        session = DockerSession(image=cfg["image"], keep_on_fail=False,
                                login_shell=(cfg.get("test_runner_kind") != "pro_official"))
        session.start()
        if verbose:
            print(f"[executor]   step docker-start: ok  image={cfg['image']}", flush=True)
        # When we own the lifecycle, bootstrap to base_commit and replay every
        # historical ok=true ATTEMPT model patch so the physical worktree
        # matches trace history. External producers that manage their own
        # session must do the same before calling this driver.
        bootstrap_session(session, trace)
        if verbose:
            print(f"[executor]   step bootstrap: ok  cwd={cfg.get('cwd')}", flush=True)
        replay_model_patches(session, trace, verbose=verbose)
    try:
        # ---- pass 1: run pending ATTEMPTs -----------------------------------
        pending = at.find_pending_attempts(trace)
        if verbose:
            print(f"[executor] {len(pending)} pending ATTEMPT(s) in "
                  f"{trace_path}")
        attempts_run = 0
        for ev in pending:
            rep = _run_one_attempt(trace, ev, session, verbose=verbose)
            attempts_run += 1
            at.save_trace(trace_path, trace)
            if not rep.ok and stop_on_fail:
                if verbose:
                    print("[executor] stop_on_fail=True; "
                          "halting before remaining attempts.")
                break
        if verbose:
            print(
                f"[executor]   step pending-attempts: ok  ran={attempts_run}/{len(pending)}",
                flush=True,
            )

    finally:
        if own_session:
            session.stop()
            if verbose:
                print("[executor]   step docker-stop: ok", flush=True)

    return trace["state"]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Execute pending ATTEMPT work from a trace JSON file."
    )
    ap.add_argument("trace_path", help="path to agent_trace.v2 JSON")
    ap.add_argument("--no-stop-on-fail", action="store_true",
                    help="continue after a failed ATTEMPT")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    state = run_pending_attempts(
        args.trace_path,
        stop_on_fail=not args.no_stop_on_fail,
        verbose=not args.quiet,
    )
    if not args.quiet:
        print(f"[executor] final state: {state}")
