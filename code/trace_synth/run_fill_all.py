#!/usr/bin/env python3
"""
Batch-fill attempt/augment for 17 recovery skeletons (linear skeleton replay / fill-skeleton mode).

Modeled on run_all_instances.py, but targeting the special "recovery skeleton" input:
  - The input is not --dataset/--instance-id, but the already-recovered skeleton trace in traces_30/
    (contains SEED + PLAN + each subtask's user_query/acceptance_criteria, with the attempt body all empty).
  - Each call is main.py --fill-skeleton <skeleton> --output <target>,
    strictly linearly replayed by fill_skeleton_trace: SEED precheck → incrementally append PLAN per BigTask →
    run attempt+augment normally per subtask (only prior info is given; PLAN/user_query are not generated).

Key design (must understand):
  * user_query / acceptance_criteria only exist in the skeleton and are **non-regenerable** recovery artifacts.
    Per the user's requirement the fill result is **written back in place to traces_30/<iid>.json** (overwrites the skeleton). So before
    any fill, this script first idempotently snapshots traces_30/*.json to SKELETON_DIR (traces_30_skeleton/),
    as a read-only source of truth; fill reads from the snapshot and writes back to traces_30/. If a snapshot exists it is **never overwritten** (protects original uq/ac).
  * The skeleton's own state.current_phase is also "DONE", so it cannot be used to decide "already filled". A trace whose fill is complete
    will carry config.fill_mode==True; use it to distinguish skeleton-done from filled-done.
  * Checkpoint: fill_skeleton_trace itself supports resuming from partial output (skips already-filled subtasks); at this
    script's level, only filled-done is skipped.

Usage:
  python run_fill_all.py [--dry-run] [--workers N] [--only iid,...] [extra synthesize args ...]

Extra args are forwarded verbatim to main.py.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import progress_store as PS

# ─────────────────────────────────────────────────────────────────────────────
# Path configuration
# ─────────────────────────────────────────────────────────────────────────────

HERE = Path(__file__).resolve().parent

import config as _config

_RUNTIME = _config.runtime_config()

DEFAULT_TRACES_DIR = HERE / "traces_30"            # skeleton source + fill write-back target (in place)
DEFAULT_SKELETON_DIR = HERE / "traces_30_skeleton" # read-only skeleton snapshot (uq/ac source of truth)
PROGRESS_FILENAME = "fill_progress.json"

SYNTHESIZE_SCRIPT = str(HERE / "main.py")
# Python interpreter for spawned subprocesses: config.runtime.python, else the
# PYTHON env var, else the interpreter running this script.
PYTHON = os.environ.get("PYTHON") or (_RUNTIME.get("python") or sys.executable)

_progress_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Skeleton identification / completion decision
# ─────────────────────────────────────────────────────────────────────────────

def _read_trace(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            data = data[0]
        if isinstance(data, dict) and isinstance(data.get("events"), list):
            return data
    except Exception:
        pass
    return None


def is_skeleton(path: Path) -> bool:
    """Whether this is a recovery skeleton: a trace payload whose attempt body is essentially empty (no real filled ATTEMPT).

    Criterion: there exist non-SEED ATTEMPTs and they all have empty patch (skeleton characteristic). An already-filled trace will have
    config.fill_mode or a non-empty patch, returning False.
    """
    t = _read_trace(path)
    if t is None:
        return False
    if (t.get("config") or {}).get("fill_mode"):
        return False
    subtask_atts = [
        e for e in (t.get("events") or [])
        if e.get("kind") == "ATTEMPT" and e.get("big_task_id") != "SEED"
    ]
    if not subtask_atts:
        return False
    return all(not e.get("patch") for e in subtask_atts)


def is_fill_done(path: Path) -> bool:
    """Whether fill is complete: a trace payload with config.fill_mode==True and state.current_phase=='DONE'."""
    t = _read_trace(path)
    if t is None:
        return False
    if not (t.get("config") or {}).get("fill_mode"):
        return False
    return (t.get("state") or {}).get("current_phase") == "DONE"


# ─────────────────────────────────────────────────────────────────────────────
# Skeleton snapshot (idempotent, never overwrites an existing snapshot)
# ─────────────────────────────────────────────────────────────────────────────

def snapshot_skeletons(traces_dir: Path, skeleton_dir: Path, *, dry_run: bool) -> list[str]:
    """Idempotently snapshot the skeleton *.json files in traces_dir to skeleton_dir; return the list of fillable iids (file stems).

    - If skeleton_dir/<name> already exists: treat it as an already-snapshotted original skeleton, **do not overwrite** (protects non-regenerable uq/ac).
    - Only snapshot files that "look like a skeleton"; already-filled / non-trace files are not snapshotted, but if their snapshot already exists they still count into the list.
    """
    skeleton_dir.mkdir(parents=True, exist_ok=True)
    iids: list[str] = []
    for src in sorted(traces_dir.glob("*.json")):
        if src.name == PROGRESS_FILENAME or src.name.endswith("run_progress.json"):
            continue
        dst = skeleton_dir / src.name
        stem = src.stem
        if dst.exists():
            # Snapshot already exists → this instance is fillable (the snapshot is the source of truth).
            iids.append(stem)
            continue
        if is_skeleton(src):
            if dry_run:
                print(f"[snapshot] (dry-run) will snapshot {src.name} → {dst}")
            else:
                shutil.copy2(src, dst)
                print(f"[snapshot] {src.name} → {dst}")
            iids.append(stem)
        else:
            # In traces_dir it is neither a skeleton nor has an existing snapshot: cannot safely fill (no uq/ac source of truth).
            print(f"[snapshot][skip] {src.name} is not a skeleton and has no existing snapshot, skip (no uq/ac source)")
    return iids


# ─────────────────────────────────────────────────────────────────────────────
# Single-instance fill
# ─────────────────────────────────────────────────────────────────────────────

def run_fill(
    iid: str,
    skeleton_path: Path,
    output_path: Path,
    extra_args: list[str],
    log_path: Path,
    prefix: str = "",
    dry_run: bool = False,
) -> tuple[bool, str]:
    """Call main.py --fill-skeleton to fill a single trace.

    fill_skeleton_trace itself can resume from output_path's partial fill result, so there is no need to distinguish new/resume here.
    The subprocess's stdout/stderr are all written to log_path. Returns (success, error_message).
    """
    cmd = [
        PYTHON, SYNTHESIZE_SCRIPT,
        "--fill-skeleton", str(skeleton_path),
        "--output", str(output_path),
    ] + extra_args

    resuming = output_path.exists() and not is_skeleton(output_path)
    mode = "resume" if resuming else "fill"
    print(f"{prefix} [{mode}] subprocess output → {log_path}")

    if dry_run:
        print(f"{prefix} [{mode}] CMD: {' '.join(cmd)}")
        return True, ""

    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # Same rationale as run_all_instances: run.log is the sole data source for per-LLM-call cost accounting; a resume appends.
        open_mode = "a" if resuming else "w"
        with log_path.open(open_mode, encoding="utf-8") as logf:
            if open_mode == "a":
                logf.write(f"\n\n# ===== RESUME @ {datetime.now().isoformat()} =====\n")
            logf.write(f"# CMD: {' '.join(cmd)}\n")
            logf.write(f"# started_at: {datetime.now().isoformat()}\n\n")
            logf.flush()
            result = subprocess.run(
                cmd, check=False, text=True, stdout=logf, stderr=subprocess.STDOUT
            )
        if result.returncode == 0:
            return True, ""
        return False, f"returncode={result.returncode}"
    except Exception as e:
        return False, str(e)


def process_one(
    *,
    idx: int,
    total: int,
    iid: str,
    traces_dir: Path,
    skeleton_dir: Path,
    extra_args: list[str],
    progress_file: Path,
    dry_run: bool,
    retry_failed: bool,
    reclaim_stale: bool = False,
) -> str:
    """Process a single instance fill; returns "success" / "fail" / "skip". Cross-process safety same as run_all_instances."""
    prefix = f"[{idx:2d}/{total}][{iid}]"
    skeleton_path = skeleton_dir / f"{iid}.json"
    output_path = traces_dir / f"{iid}.json"
    run_log_path = traces_dir / f"{iid}.fill.log"

    print(f"{prefix} start")

    if not skeleton_path.exists():
        if dry_run:
            # dry-run has not actually snapshotted; use the source file as a placeholder to print the command.
            src = traces_dir / f"{iid}.json"
            run_fill(iid, src, output_path, extra_args,
                     log_path=run_log_path, prefix=prefix, dry_run=True)
            print(f"{prefix} → dry-run, only print the command (the snapshot will be created on the real run)")
            return "skip"
        print(f"{prefix} → [ERROR] skeleton snapshot missing: {skeleton_path}, skip")
        return "skip"

    # ── Already filled-done → skip directly (idempotently backfill completed) ──────────────────────
    if is_fill_done(output_path):
        if not dry_run:
            with _progress_lock:
                PS.mark_completed(progress_file, iid, {
                    "output": str(output_path),
                    "finished_at": datetime.now().isoformat(),
                    "elapsed_s": 0,
                    "note": "detected-fill-done-on-startup",
                })
        print(f"{prefix} → already filled (fill DONE), skip")
        return "skip"

    if dry_run:
        run_fill(iid, skeleton_path, output_path, extra_args,
                 log_path=run_log_path, prefix=prefix, dry_run=True)
        print(f"{prefix} → dry-run, only print the command, do not update progress")
        return "skip"

    # ── progress recorded as completed but file not filled-done → drop completed so it refills ─────────
    with _progress_lock:
        if PS.drop_completed(progress_file, iid):
            print(f"{prefix} [WARN] progress recorded as completed but the trace is not in filled-DONE, refilling")

    # ── claim ────────────────────────────────────────────────────────────────
    with _progress_lock:
        decision, info = PS.claim(
            progress_file, iid,
            retry_failed=retry_failed,
            reclaim_stale=reclaim_stale,
            run_log=str(run_log_path),
        )

    if decision == "skip_failed":
        print(f"{prefix} → failed last time, skip  (error: {info.get('error', '?')})")
        return "skip"
    if decision == "skip_done":
        print(f"{prefix} → progress already recorded as completed, skip")
        return "skip"
    if decision == "skip_running":
        who = f"pid={info.get('pid')} host={info.get('host')}"
        if info.get("stale"):
            print(f"{prefix} → stale running record ({who}, process is gone); conservatively skip. Add --reclaim-stale to take over")
        else:
            print(f"{prefix} → another process is running ({who}), skip")
        return "skip"
    if decision == "reclaimed":
        print(f"{prefix} [WARN] took over stale running record, rerun")

    if output_path.exists() and not is_skeleton(output_path):
        print(f"{prefix} → already partially filled, resume")
    else:
        print(f"{prefix} → fill fresh from skeleton")

    t0 = time.time()
    try:
        success, err = run_fill(
            iid, skeleton_path, output_path, extra_args,
            log_path=run_log_path, prefix=prefix, dry_run=False,
        )
        elapsed = time.time() - t0

        # returncode=0 does not mean filled-done (may have terminated mid-run: e.g. subtask exhausted → terminal non-empty)
        if success and not is_fill_done(output_path):
            success = False
            err = "process exited 0 but trace is not filled-DONE (see fill.log / state.terminal)"

        if success:
            print(f"{prefix} → done  ({elapsed:.1f}s)  → {output_path}")
            with _progress_lock:
                PS.mark_completed(progress_file, iid, {
                    "output": str(output_path),
                    "finished_at": datetime.now().isoformat(),
                    "elapsed_s": round(elapsed, 1),
                })
            return "success"
        else:
            print(f"{prefix} → failed  ({elapsed:.1f}s)  error: {err}  (subprocess log: {run_log_path})")
            with _progress_lock:
                PS.mark_failed(progress_file, iid, {
                    "error": err,
                    "failed_at": datetime.now().isoformat(),
                    "run_log": str(run_log_path),
                })
            return "fail"
    finally:
        with _progress_lock:
            PS.release_running(progress_file, iid, os.getpid())


# ─────────────────────────────────────────────────────────────────────────────
# Main flow
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Batch-fill attempt/augment for recovery skeletons (fill-skeleton linear replay), with interruption recovery.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--traces-dir", default=str(DEFAULT_TRACES_DIR),
                    help=f"skeleton source + fill write-back target directory (in place, default: {DEFAULT_TRACES_DIR})")
    ap.add_argument("--skeleton-dir", default=str(DEFAULT_SKELETON_DIR),
                    help=f"read-only skeleton snapshot directory (uq/ac source of truth, default: {DEFAULT_SKELETON_DIR})")
    ap.add_argument("--progress-file", default=None,
                    help="progress file path (default: <traces-dir>/fill_progress.json)")
    ap.add_argument("--dry-run", action="store_true", help="only print the command + snapshot plan, do not actually execute")
    ap.add_argument("--retry-failed", action="store_true", help="rerun instances that failed last time (failed ones are skipped by default)")
    ap.add_argument("--reclaim-stale", action="store_true",
                    help="take over stale running records and rerun (conservatively skips by default)")
    ap.add_argument("--only", default=None, help="only process the specified instance_id/file stem (comma-separated)")
    ap.add_argument("--limit", type=int, default=None, help="process at most the first N")
    ap.add_argument("--reverse", action="store_true", help="process in reverse order (back to front)")
    ap.add_argument("--workers", type=int, default=4,
                    help="number of parallel workers (default 4: each spawns an independent synthesize subprocess + Docker container; 1 means sequential)")

    args, extra_args = ap.parse_known_args(argv)

    traces_dir = Path(args.traces_dir)
    skeleton_dir = Path(args.skeleton_dir)
    if not traces_dir.is_dir():
        print(f"[ERROR] traces-dir does not exist: {traces_dir}", file=sys.stderr)
        sys.exit(1)

    progress_file = Path(args.progress_file) if args.progress_file else traces_dir / PROGRESS_FILENAME

    # ── Step 0: snapshot skeletons first (protect non-regenerable uq/ac), then decide the instance list ─────────────
    print("=" * 70)
    print(f"  Skeleton snapshot  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Source: {traces_dir}")
    print(f"  Snapshot: {skeleton_dir}  (existing snapshots are not overwritten)")
    instance_ids = snapshot_skeletons(traces_dir, skeleton_dir, dry_run=args.dry_run)
    if not instance_ids:
        print("[ERROR] no fillable skeleton instances (neither skeleton nor existing snapshot)", file=sys.stderr)
        sys.exit(1)

    instance_ids = sorted(instance_ids)
    if args.reverse:
        instance_ids = list(reversed(instance_ids))
    if args.only:
        only_set = {s.strip() for s in args.only.split(",") if s.strip()}
        instance_ids = [iid for iid in instance_ids if iid in only_set]
        if not instance_ids:
            print("[ERROR] none of the instance_id specified by --only were found", file=sys.stderr)
            sys.exit(1)
    if args.limit is not None:
        instance_ids = instance_ids[:args.limit]

    progress = PS.read(progress_file)
    total = len(instance_ids)

    print("=" * 70)
    print(f"  fill batch script  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Total instances : {total}")
    print(f"  Completed       : {len(progress['completed'])}")
    print(f"  Failed last time: {len(progress['failed'])}  (--retry-failed={'yes' if args.retry_failed else 'no, skip'})")
    already_running = len(progress.get("running", {}))
    if already_running:
        print(f"  Running         : {already_running}  (--reclaim-stale={'yes' if args.reclaim_stale else 'no, conservatively skip if stale'})")
        for rid, rinfo in progress["running"].items():
            print(f"    - {rid}  (pid={rinfo.get('pid')} host={rinfo.get('host')} started_at={rinfo.get('started_at')})")
    print(f"  Skeleton dir    : {skeleton_dir}")
    print(f"  Write-back dir  : {traces_dir}")
    print(f"  Progress file   : {progress_file}")
    if extra_args:
        print(f"  Forwarded args  : {extra_args}")
    workers = max(1, args.workers)
    print(f"  Parallel workers: {workers}")
    if args.dry_run:
        print("  [DRY-RUN mode: not actually executing]")
    print("=" * 70)

    n_success = n_fail = n_skip = 0

    def submit(idx: int, iid: str) -> str:
        return process_one(
            idx=idx, total=total, iid=iid,
            traces_dir=traces_dir, skeleton_dir=skeleton_dir,
            extra_args=extra_args, progress_file=progress_file,
            dry_run=args.dry_run, retry_failed=args.retry_failed,
            reclaim_stale=args.reclaim_stale,
        )

    if workers == 1:
        for idx, iid in enumerate(instance_ids, 1):
            status = submit(idx, iid)
            n_success += status == "success"
            n_fail += status == "fail"
            n_skip += status == "skip"
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(submit, idx, iid): iid for idx, iid in enumerate(instance_ids, 1)}
            for fut in as_completed(futures):
                iid = futures[fut]
                try:
                    status = fut.result()
                except Exception as exc:
                    print(f"[{iid}] → worker exception: {exc}")
                    with _progress_lock:
                        PS.mark_failed(progress_file, iid, {
                            "error": f"worker exception: {exc}",
                            "failed_at": datetime.now().isoformat(),
                        })
                        PS.release_running(progress_file, iid, os.getpid())
                    n_fail += 1
                    continue
                n_success += status == "success"
                n_fail += status == "fail"
                n_skip += status == "skip"

    final = PS.read(progress_file)
    print("\n" + "=" * 70)
    print(f"  This run's result")
    print(f"  Success: {n_success}  Failed: {n_fail}  Skipped: {n_skip}")
    print(f"  Cumulative completed: {len(final['completed'])}/{total}")
    if final.get("running"):
        print(f"  Still running ({len(final['running'])}):")
        for rid, rinfo in final["running"].items():
            print(f"    - {rid}  (pid={rinfo.get('pid')} host={rinfo.get('host')})")
    if final["failed"]:
        print(f"  Still failed ({len(final['failed'])}):")
        for fid, info in final["failed"].items():
            print(f"    - {fid}  ({info.get('error', '')})")
    print(f"  Progress file → {progress_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
