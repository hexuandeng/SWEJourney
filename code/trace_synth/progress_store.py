"""Cross-process-safe progress store: lets multiple run_all_instances processes
share a single run_progress.json in real time.

Each read-modify-write is wrapped in an fcntl.flock and re-merged against the
latest on-disk content, so concurrent writers never lose each other's updates.
A "running" record {pid, host, started_at, run_log} tracks in-flight instances;
liveness is checked with os.kill(pid, 0).

Progress file schema:
  {
    "completed": {iid: {"output","finished_at","elapsed_s",...}},
    "failed":    {iid: {"error","failed_at","run_log"}},
    "running":   {iid: {"pid","host","started_at","run_log"}},
  }

The lock is advisory: it only helps when every process goes through this module
(all callers in this project do).
"""
from __future__ import annotations

import contextlib
import json
import os
import socket
from datetime import datetime
from pathlib import Path

try:
    import fcntl  # POSIX only
except ImportError:  # pragma: no cover - degrade on non-POSIX platforms
    fcntl = None  # type: ignore[assignment]

_SKELETON_KEYS = ("completed", "failed", "running")


def _now() -> str:
    return datetime.now().isoformat()


def _host() -> str:
    return socket.gethostname()


@contextlib.contextmanager
def _locked(path: Path):
    """Take an exclusive fcntl lock on <path>.lock; release on exit.

    The lock file is separate from the data file to avoid racing the atomic
    write's rename/truncate. Degrades to a no-op when fcntl is unavailable
    (only safe within a single process).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fcntl is None:
        yield
        return
    lock_file = path.with_name(path.name + ".lock")
    fp = open(lock_file, "a+")
    try:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        with contextlib.suppress(Exception):
            fcntl.flock(fp.fileno(), fcntl.LOCK_UN)
        with contextlib.suppress(Exception):
            fp.close()


def _read_unlocked(path: Path) -> dict:
    """Read JSON and backfill the skeleton keys. Returns an empty skeleton if
    the file is missing or corrupt."""
    path = Path(path)
    data: dict = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            print(f"  [WARN] progress file corrupt, treating as empty: {path}", flush=True)
            data = {}
    if not isinstance(data, dict):
        data = {}
    for k in _SKELETON_KEYS:
        if not isinstance(data.get(k), dict):
            data[k] = {}
    return data


def _atomic_write_unlocked(path: Path, data: dict) -> None:
    """Write to a temp file then os.replace it into place, so a crash mid-write
    cannot corrupt the file."""
    path = Path(path)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _pid_alive(pid) -> bool:
    """os.kill(pid, 0): True if the process is alive, False if it does not
    exist; a permission error counts as alive."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # process exists but belongs to another user
    except OSError:
        return False
    return True


def _running_is_stale(info: dict) -> bool:
    """Whether a running record is stale (its worker process is gone).

    On the same host (host matches) this is decided precisely via
    os.kill(pid, 0); across hosts pid cannot be checked, so conservatively
    treat it as not stale (do not reclaim).
    """
    if not isinstance(info, dict):
        return True
    host = info.get("host")
    if host and host != _host():
        return False  # cannot decide across hosts; conservatively keep
    return not _pid_alive(info.get("pid"))


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def read(path: Path) -> dict:
    """Read-only snapshot (one locked read), for startup / summary stats."""
    with _locked(path):
        return _read_unlocked(path)


def claim(
    path: Path,
    iid: str,
    *,
    retry_failed: bool = False,
    reclaim_stale: bool = False,
    run_log: str = "",
) -> tuple[str, dict]:
    """Under the lock, decide in one shot whether to process an instance and
    atomically write the running record, eliminating TOCTOU.

    Returns (decision, info):
      - "run"          : running written; caller should execute
      - "reclaimed"    : reclaimed a stale running (rewritten to this process)
      - "skip_done"    : already completed in progress
      - "skip_failed"  : failed last time and retry_failed is off
      - "skip_running" : another live process owns it, or stale but reclaim off
    info carries relevant context (e.g. pid/host/stale for skip_running).
    """
    with _locked(path):
        data = _read_unlocked(path)

        if iid in data["completed"]:
            return "skip_done", dict(data["completed"][iid])

        if iid in data["failed"] and not retry_failed:
            return "skip_failed", dict(data["failed"][iid])

        cur = data["running"].get(iid)
        if cur is not None:
            stale = _running_is_stale(cur)
            if not stale:
                return "skip_running", {**dict(cur), "stale": False}
            if not reclaim_stale:
                return "skip_running", {**dict(cur), "stale": True}
            decision = "reclaimed"
        else:
            decision = "run"

        data["running"][iid] = {
            "pid": os.getpid(),
            "host": _host(),
            "started_at": _now(),
            "run_log": run_log,
        }
        _atomic_write_unlocked(path, data)
        return decision, dict(data["running"][iid])


def drop_completed(path: Path, iid: str) -> bool:
    """Remove a completed record (used to regenerate when progress says done
    but the trace file is not DONE)."""
    with _locked(path):
        data = _read_unlocked(path)
        if iid in data["completed"]:
            del data["completed"][iid]
            _atomic_write_unlocked(path, data)
            return True
        return False


def mark_completed(path: Path, iid: str, info: dict) -> None:
    """Mark completed: re-read latest -> drop from running/failed -> write
    completed -> atomic flush."""
    with _locked(path):
        data = _read_unlocked(path)
        data["running"].pop(iid, None)
        data["failed"].pop(iid, None)
        data["completed"][iid] = dict(info)
        _atomic_write_unlocked(path, data)


def mark_failed(path: Path, iid: str, info: dict) -> None:
    """Mark failed: re-read latest -> drop from running -> write failed ->
    atomic flush."""
    with _locked(path):
        data = _read_unlocked(path)
        data["running"].pop(iid, None)
        data["failed"][iid] = dict(info)
        _atomic_write_unlocked(path, data)


def release_running(path: Path, iid: str, pid: int | None = None) -> bool:
    """Best-effort cleanup of this process's own running record.

    Only pops when running[iid].pid == pid (defaults to this process), never
    removing another process's ownership. Usually a no-op since success/failure
    branches already mark_*; matters only on an abnormal exit.
    """
    if pid is None:
        pid = os.getpid()
    with _locked(path):
        data = _read_unlocked(path)
        cur = data["running"].get(iid)
        if cur is not None and str(cur.get("pid")) == str(pid):
            del data["running"][iid]
            _atomic_write_unlocked(path, data)
            return True
        return False


def prune_stale_running(path: Path, *, dry_run: bool = False) -> list[dict]:
    """Remove all stale running records (worker process gone; see
    `_running_is_stale`).

    When a previous batch crashed or was killed, its running records are not
    reclaimed automatically and would make a new batch skip those instances as
    skip_running. This deletes all stale running under the lock and returns the
    removed records (each with iid + original pid/host/started_at). Cross-host
    running cannot be decided by pid and is conservatively kept, as is any live
    running.

    With dry_run=True it only probes without writing, returning what would be
    removed.
    """
    pruned: list[dict] = []
    with _locked(path):
        data = _read_unlocked(path)
        for iid, info in list(data["running"].items()):
            if _running_is_stale(info):
                pruned.append({"iid": iid, **(info if isinstance(info, dict) else {})})
                if not dry_run:
                    del data["running"][iid]
        if pruned and not dry_run:
            _atomic_write_unlocked(path, data)
    return pruned



# ─────────────────────────────────────────────────────────────────────────────
# CLI: clean up stale running records
#   python progress_store.py prune-stale <progress.json> [--dry-run]
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="progress_store maintenance tool")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_prune = sub.add_parser("prune-stale", help="delete all stale running records (dead workers)")
    p_prune.add_argument("progress_file", help="progress JSON path")
    p_prune.add_argument("--dry-run", action="store_true", help="probe only, do not write")

    p_show = sub.add_parser("show", help="print the three-state counts")
    p_show.add_argument("progress_file", help="progress JSON path")

    a = ap.parse_args()
    pf = Path(a.progress_file)

    if a.cmd == "show":
        d = read(pf)
        print(f"completed: {len(d['completed'])}  failed: {len(d['failed'])}  running: {len(d['running'])}")
        for iid, info in d["running"].items():
            stale = _running_is_stale(info)
            print(f"  running{'[STALE]' if stale else ''}: {iid} (pid={info.get('pid')} host={info.get('host')})")
    elif a.cmd == "prune-stale":
        pruned = prune_stale_running(pf, dry_run=a.dry_run)
        verb = "would delete" if a.dry_run else "deleted"
        print(f"{verb} {len(pruned)} stale running:")
        for r in pruned:
            print(f"  - {r['iid']} (pid={r.get('pid')} host={r.get('host')} started={r.get('started_at')})")
        if not pruned:
            print("  (no stale running)")
