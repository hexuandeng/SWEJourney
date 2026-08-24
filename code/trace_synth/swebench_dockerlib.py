#!/usr/bin/env python3
"""
swebench_dockerlib.py  (D-0)
============================

Reusable infrastructure for the agent-loop pipeline:
- DockerSession:    long-lived container lifecycle + exec/read/write/snapshot
- apply_unified_diff: strict git-apply patch application
- run_pytest: deterministic in-container test execution

This file is the *only* place we call docker / git / pytest from. Everything
upstream (executor, agent loop, demos) depends on it through stable APIs so
we can swap backends (e.g. podman, k8s exec) later without a rewrite.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
import uuid
from typing import Optional


# =============================================================================
# 1. low-level shell
# =============================================================================
def sh(cmd: str, *, check: bool = False, timeout: Optional[int] = None
       ) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=timeout,
        check=check,
    )


# =============================================================================
# 2. errors
# =============================================================================
class PatchApplyError(Exception):
    """`git apply` (and the 3-way fallback) failed."""


# =============================================================================
# 3. DockerSession
# =============================================================================
class DockerSession:
    """A long-lived docker container handle (used as ContextManager).

    By default exec calls go through ``bash -lc`` (login shell) so that conda
    profiles are sourced, which is required for SWE-bench Verified images.
    Pass ``login_shell=False`` for images that rely on Dockerfile ``ENV PATH``
    entries (e.g. SWE-bench Pro / sweap-images with Go/Node runtimes): the
    login shell rewrites PATH and drops those entries.
    """

    # Top-level directory basenames that the worktree snapshot/restore guard
    # deliberately leaves UNTOUCHED (neither tar'd on snapshot nor wiped on
    # restore).
    #
    # These hold language dependency caches that are git-ignored and are NEVER
    # modified by a test_patch / model patch, so a snapshot does not need to
    # capture them and a restore must not delete them. They are also the bulk
    # of repo size on JS/TS images (e.g. tutanota /app is 871M, of which the
    # top-level node_modules alone is ~570M). Tarring + re-extracting them on
    # *every* test stage cost ~40s snapshot + ~80s restore per call and was the
    # dominant slowdown after the snapshot guard was generalised to all test
    # stages — pure overhead with zero correctness value.
    #
    # We match ONLY at the worktree top level (depth 1). The restore wipe runs
    # at ``-maxdepth 1``; excluding a nested dir from the tar while its parent
    # still gets wiped+restored would DELETE the nested dir, so nested caches
    # (small — a few MB) intentionally flow through the snapshot normally.
    #
    # NOTE: ``.git`` is intentionally NOT excluded. The whole point of the
    # snapshot is to preserve git worktree/index state so later ``git apply``
    # is coherent; dropping it would defeat the guard.
    SNAPSHOT_SKIP_DIRS: tuple[str, ...] = (
        "node_modules",  # npm/yarn/pnpm (JS/TS): git-ignored dependency cache
        ".venv",          # python virtualenv checked into worktree
        ".pytest_cache",
        ".mypy_cache",
        ".tox",
    )

    def __init__(self, image: str, container_name: Optional[str] = None,
                 keep_on_fail: bool = True, login_shell: bool = True):
        self.image = image
        self.container_name = container_name or self._auto_name()
        self.keep_on_fail = keep_on_fail
        self.login_shell = login_shell
        self._started = False
        self._failed = False

    @staticmethod
    def _auto_name() -> str:
        return f"agentloop_{int(time.time())}_{uuid.uuid4().hex[:6]}"

    # ---- lifecycle -----------------------------------------------------------
    def start(self) -> "DockerSession":
        # cleanup stale container with the same name (best-effort)
        if sh(f"docker ps -aq -f name=^{self.container_name}$").stdout.strip():
            sh(f"docker rm -f {self.container_name}")
        # Optional outbound proxy for the container (e.g. if the test env needs
        # to reach the internet from behind a firewall). Set DOCKER_HTTP_PROXY to
        # enable; unset (default) → no proxy env is injected.
        _proxy = os.environ.get("DOCKER_HTTP_PROXY", "").strip()
        _proxy_env = (
            f"-e http_proxy={_proxy} -e https_proxy={_proxy} -e ftp_proxy={_proxy} "
            f"-e HTTP_PROXY={_proxy} -e HTTPS_PROXY={_proxy} "
            if _proxy else ""
        )
        p = sh(
            f"docker run -d --platform linux/amd64 --network host "
            f"--name {self.container_name} "
            f"{_proxy_env}"
            f"--entrypoint sleep {shlex.quote(self.image)} infinity"
        )
        if p.returncode != 0:
            raise RuntimeError(f"[docker-run-fail] {p.stderr or p.stdout}")
        self._started = True
        # Configure an anonymous git identity inside the container so the
        # pipeline's internal `git commit -m baseline` (used only to compute
        # incremental diffs) does not stamp commits with root@<hostname>.
        # Best-effort: never fail container start.
        try:
            sh(
                f"docker exec {self.container_name} bash -lc "
                + shlex.quote(
                    'git config --global user.email "anonymous@example.com" && '
                    'git config --global user.name "anonymous"'
                )
            )
        except Exception:
            pass
        return self

    def stop(self, timeout: Optional[int] = None) -> None:
        if not self._started:
            return
        if self._failed and self.keep_on_fail:
            sys.stderr.write(
                f"[docker-keep] container {self.container_name} kept "
                f"for inspection\n"
            )
            return
        sh(f"docker rm -f {self.container_name}", timeout=timeout)
        self._started = False

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        if exc is not None:
            self._failed = True
        self.stop()
        return False

    def __del__(self):
        # Best-effort cleanup if stop() was never called (e.g. SIGTERM killed
        # the process before the finally block ran). Swallow all errors so GC
        # is not disrupted.
        try:
            if self._started:
                sh(f"docker rm -f {self.container_name}")
        except Exception:
            pass

    # ---- primitives ----------------------------------------------------------
    def exec(self, script: str, *, check: bool = False,
             timeout: Optional[int] = None) -> subprocess.CompletedProcess:
        shell_flag = "-lc" if self.login_shell else "-c"
        return sh(
            f"docker exec {self.container_name} bash {shell_flag} {shlex.quote(script)}",
            check=check, timeout=timeout,
        )

    def read(self, path: str) -> str:
        # NB: cat dumps bytes; we decode here. Caller must accept text. For a
        # tiny number of binary files this would be wrong, but our patches
        # only touch text source.
        p = self.exec(f"cat {shlex.quote(path)}")
        if p.returncode != 0:
            raise FileNotFoundError(
                f"[docker-read-fail] {path}: {p.stderr.strip()}"
            )
        return p.stdout

    def write(self, path: str, content: str) -> None:
        proc = subprocess.run(
            ["docker", "exec", "-i", self.container_name, "python3", "-c",
             "import sys,os;"
             f"p={path!r};"
             "d=sys.stdin.buffer.read();"
             "os.makedirs(os.path.dirname(p), exist_ok=True);"
             "open(p,'wb').write(d)"],
            input=content.encode("utf-8"),
            capture_output=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"[docker-write-fail] {path}: "
                f"{proc.stderr.decode('utf-8','ignore')}"
            )

    # ---- snapshots (filesystem-level via tar; survives any git state) -------
    def fs_snapshot(self, cwd: str) -> str:
        """
        Tar-snapshot the entire `cwd` to a unique path under /tmp inside the
        container. Returns the snapshot id (the tar path).

        We deliberately AVOID `git stash` because patched files may end up
        untracked, mid-merge, or in a state that confuses `git stash apply`.
        A tar of the worktree is the simplest, most reliable approach.
        """
        snap_id = f"/tmp/agentsnap_{uuid.uuid4().hex[:10]}.tar"
        # `--one-file-system` to avoid pulling in /proc-style mounts; safe
        # since /testbed is a real directory.
        # Exclude .git/objects to keep tar small? Keep it for correctness:
        # patches may rely on git state.
        # NOTE: do NOT pass GNU-only long options such as
        # `--warning=no-file-changed`. Many Pro images ship BusyBox tar
        # (`tar: unrecognized option: warning=no-file-changed`), which makes
        # the whole snapshot fail and kills otherwise-valid instances. The
        # only purpose of that flag was to silence GNU tar's harmless
        # "file changed as we read it" warning (exit 1); the trailing
        # `|| true` plus the non-empty (`test -s`) check below already
        # tolerate that, so portable plain `tar -cf` is sufficient for both
        # GNU and BusyBox tar.
        #
        # We skip top-level dependency-cache dirs (see SNAPSHOT_SKIP_DIRS):
        # they are git-ignored, never touched by a patch, and dominate repo
        # size on JS/TS images. ``--exclude=./node_modules`` is understood by
        # both GNU and BusyBox tar.
        excludes = " ".join(
            f"--exclude={shlex.quote('./' + d)}" for d in self.SNAPSHOT_SKIP_DIRS
        )
        p = self.exec(
            f"cd {shlex.quote(cwd)} && "
            f"tar {excludes} -cf {shlex.quote(snap_id)} . "
            f"|| true"
        )
        # tar may exit 1 on harmless 'file changed as we read it' which is
        # extremely rare for read-only worktrees; we accept it.
        if "Cannot open" in (p.stderr or ""):
            raise RuntimeError(f"[snapshot-fail] {p.stderr}")
        check = self.exec(f"test -s {shlex.quote(snap_id)}")
        if check.returncode != 0:
            raise RuntimeError(
                f"[snapshot-fail] missing-or-empty {snap_id}: "
                f"{(p.stderr or p.stdout).strip()}"
            )
        return snap_id

    def fs_restore(self, cwd: str, snap_id: str) -> None:
        """Wipe `cwd` and extract `snap_id` back into it."""
        check = self.exec(f"test -s {shlex.quote(snap_id)}")
        if check.returncode != 0:
            raise RuntimeError(f"[restore-fail] missing-or-empty {snap_id}")
        # Move current contents out of the worktree before extracting the
        # snapshot. This is more robust than `rm -rf` in Docker overlayfs: on
        # busy trees it can transiently report "Directory not empty" while
        # walking nested package directories.
        #
        # We must NOT move the dependency-cache dirs that fs_snapshot skipped
        # (SNAPSHOT_SKIP_DIRS): they are absent from the tar, so moving them to
        # trash would delete them permanently and break the next test run that
        # needs node_modules. Leaving them in place is correct precisely because
        # no patch ever modifies them.
        skip_preds = "".join(
            f" ! -name {shlex.quote(d)}" for d in self.SNAPSHOT_SKIP_DIRS
        )
        # The trash dir MUST live *inside* `cwd` (same filesystem), not under
        # /tmp. /tmp and /testbed are different filesystems in our images, so a
        # cross-fs `mv` silently degrades to copy+unlink; the unlink half then
        # races git's many small objects and intermittently fails with
        # "Directory not empty" on './.git', killing an otherwise-valid restore.
        # A same-filesystem move is a pure atomic rename — no copy, no unlink
        # race. We give it a fixed hidden name and exclude it from the find so
        # the move never tries to relocate the trash into itself.
        trash_name = f".agentrestore_{uuid.uuid4().hex[:10]}"
        trash = f"{cwd.rstrip('/')}/{trash_name}"
        p = self.exec(
            f"cd {shlex.quote(cwd)} && "
            f"mkdir -p {shlex.quote(trash_name)} && "
            f"find . -mindepth 1 -maxdepth 1{skip_preds} "
            f"! -name {shlex.quote(trash_name)} "
            f"-exec mv -f -t {shlex.quote(trash)} -- {{}} + && "
            f"tar -xf {shlex.quote(snap_id)} -C {shlex.quote(cwd)}; "
            f"status=$?; rm -rf {shlex.quote(trash)} >/dev/null 2>&1 || true; "
            f"exit $status"
        )
        if p.returncode != 0:
            raise RuntimeError(
                f"[restore-fail] {snap_id}: {p.stderr.strip()}"
            )

    def fs_drop_snapshot(self, snap_id: str) -> None:
        self.exec(f"rm -f {shlex.quote(snap_id)}")


# =============================================================================
# 4. patch application
# =============================================================================
def apply_unified_diff(session: DockerSession, diff_text: str, *, cwd: str
                       ) -> list[str]:
    """Apply a unified diff via *strict* `git apply`.

    Gold-answer hard quality requirement (AGENT_LOOP_DESIGN §3 Phase A #7,
    §9.3 invariant #3): the diff must carry correct ``@@ -a,b +c,d @@``
    hunk headers and enough surrounding context to apply *directly*. We
    deliberately do **not** fall back to ``--3way`` or any fuzzy strategy:
    a producer that gives an unapplyable diff is producing a low-quality
    gold answer and the executor must surface that as PATCH_APPLY_FAIL.
    """
    tmp = f"/tmp/agentpatch_{uuid.uuid4().hex[:8]}.diff"
    session.write(tmp, diff_text)
    p = session.exec(
        f"cd {shlex.quote(cwd)} && "
        f"git apply --whitespace=nowarn {shlex.quote(tmp)}"
    )
    if p.returncode != 0:
        # capture both stdout and stderr to surface git's reason
        raise PatchApplyError(
            f"[git-apply-fail]\n{(p.stderr or p.stdout).strip()}"
        )
    # list files actually changed against HEAD
    pf = session.exec(f"cd {shlex.quote(cwd)} && git diff --name-only HEAD")
    return [ln for ln in pf.stdout.splitlines() if ln.strip()]


# =============================================================================
# 4b. snapshot bytes-equal comparison (for rollback_sequence net-zero check)
# =============================================================================
def _snapshot_tree_digest(session: DockerSession, cwd: str) -> str:
    """Return a deterministic digest of every file under cwd (excluding .git).

    Used to verify a rollback_sequence subtask leaves the worktree byte-for-byte
    identical to the pre-attempt snapshot. We compute it inside the container
    via ``find … -print0 | sort -z | xargs -0 sha256sum | sha256sum`` so the
    result is a single fingerprint independent of file ordering.
    """
    script = (
        f"cd {shlex.quote(cwd)} && "
        # NB: exclude VCS and runtime caches so commit/index churn or pytest
        # bytecode artifacts never trip the net-zero source-state check.
        "find . "
        "\\( -path ./.git -o -path '*/__pycache__' -o -path './.pytest_cache' \\) -prune "
        "-o -name '*.pyc' -prune "
        "-o -type f -print0 "
        "| LC_ALL=C sort -z "
        "| xargs -0 sha256sum 2>/dev/null "
        "| sha256sum | awk '{print $1}'"
    )
    p = session.exec(script)
    if p.returncode != 0:
        raise RuntimeError(
            f"[tree-digest-fail] {(p.stderr or p.stdout).strip()}"
        )
    return p.stdout.strip()


# =============================================================================
# 5. pytest helpers
# =============================================================================
def _strip_pytest_banner(text: str) -> str:
    """Remove the leading 'Internet access disabled' banner printed by some SWE-bench images."""
    prefix = "Internet access disabled\n"
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def run_pytest(session: DockerSession, node_ids: list[str], *,
               pytest_bin: str, cwd: str, timeout_sec: int
               ) -> tuple[bool, dict[str, str], str]:
    """
    Run all node_ids in one go; if non-zero, re-run each individually to
    classify PASSED / FAILED / ERROR / NOT_FOUND / TIMEOUT.
    """
    if not node_ids:
        return True, {}, ""
    quoted = " ".join(shlex.quote(n) for n in node_ids)
    cmd = (f"cd {shlex.quote(cwd)} && {pytest_bin} -p no:cacheprovider "
           f"--no-header --tb=short -q {quoted}")
    try:
        p = session.exec(cmd, timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        return False, {n: "TIMEOUT" for n in node_ids}, "[batch-timeout]"

    out = _strip_pytest_banner(p.stdout + "\n" + p.stderr)
    if p.returncode == 0:
        return True, {n: "PASSED" for n in node_ids}, out

    status: dict[str, str] = {}
    for n in node_ids:
        try:
            sub = session.exec(
                f"cd {shlex.quote(cwd)} && {pytest_bin} -p no:cacheprovider "
                f"--no-header --tb=line -q {shlex.quote(n)}",
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            status[n] = "TIMEOUT"
            continue
        combo = _strip_pytest_banner(sub.stdout + "\n" + sub.stderr).lower()
        if sub.returncode == 0:
            status[n] = "PASSED"
        elif "no tests ran" in combo or "not found" in combo:
            status[n] = "NOT_FOUND"
        elif " error" in combo and " failed" not in combo:
            status[n] = "ERROR"
        else:
            status[n] = "FAILED"
    return False, status, out
