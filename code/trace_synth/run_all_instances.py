#!/usr/bin/env python3
"""
Batch-generate agent traces for 30 instances.

- Read the fixed order from select_coverage_subset_newlabels.jsonl
- Track progress with progress.json (completed / failed)
- Two-level checkpoint recovery:
    1. Instance level: already-completed instances (state.current_phase == "DONE") are skipped directly
    2. File level: when the output file exists but is unfinished, pass --resume to continue from the target output file

Usage:
  python run_all_instances.py [--output-dir OUTPUT_DIR] [--dry-run] [extra synthesize args ...]

Extra args are forwarded verbatim to main.py.
The main model is fixed to google (GCP Vertex) opus-4-8; --model/--provider/--backend/--first-call-backend are no longer accepted.
"""

from __future__ import annotations

import argparse
import json
import os
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
ROOT = HERE.parent

import config as _config

_PATHS = _config.path_config()
_RUNTIME = _config.runtime_config()

# Batch input: the instance-id subset list and the dataset it indexes into.
# Sourced from config.paths; both overridable via CLI. Empty config → the
# caller must pass --subset / --dataset explicitly.
SUBSET_PATH = Path(_PATHS["subset_jsonl"]) if _PATHS.get("subset_jsonl") else None
FULL_DATASET_PATH = Path(_PATHS["full_dataset"]) if _PATHS.get("full_dataset") else None
DATASET_PATH = FULL_DATASET_PATH

DEFAULT_OUTPUT_DIR = HERE / "traces_30"
PROGRESS_FILENAME = "run_progress.json"

SYNTHESIZE_SCRIPT = str(HERE / "main.py")
# Python interpreter for spawned subprocesses: config.runtime.python, else the
# PYTHON env var, else the interpreter running this script.
PYTHON = os.environ.get("PYTHON") or (_RUNTIME.get("python") or sys.executable)

# Lightweight in-process guard for multiple threads in this process (on top of the
# PS file lock; avoids repeated-claim churn within the same process).
# The cross-process source of truth + concurrency safety are all guaranteed by
# progress_store (fcntl file lock).
_progress_lock = threading.Lock()

# ─────────────────────────────────────────────────────────────────────────────
# Progress file structure (see progress_store.py for details)
# {
#   "completed": {"instance_id": {"output", "finished_at", "elapsed_s", ...}},
#   "failed":    {"instance_id": {"error", "failed_at", "run_log"}},
#   "running":   {"instance_id": {"pid", "host", "started_at", "run_log"}},
# }
# All read/modify/write goes through progress_store.*, always based on the latest
# on-disk content, synchronized across processes in real time.
# ─────────────────────────────────────────────────────────────────────────────


def is_trace_done(path: Path) -> bool:
    """Check whether the trace file is finished (state.current_phase == "DONE")."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        # main writes a single JSON object (not a list)
        if isinstance(data, list):
            data = data[0]
        return (data.get("state") or {}).get("current_phase") == "DONE"
    except Exception:
        return False


def load_progress(path: Path) -> dict:
    """Read-only snapshot, only for the startup summary / final aggregate stats. Decisions and writes always go through progress_store.*."""
    return PS.read(path)


def load_instance_ids(subset_path: Path) -> list[str]:
    """Read the instance_id list in order.

    - subset_path is a jsonl file: parse line by line, take each line's instance_id (fall back to sample_id).
    - subset_path is a directory (HuggingFace Dataset, load_from_disk): read the dataset's
      instance_id column directly. Makes it easy to feed the full_dataset/ directory produced by
      build_selected_dataset.py --full directly as input.
    """
    if subset_path.is_dir():
        from datasets import load_from_disk

        ds = load_from_disk(str(subset_path))
        ids = []
        for r in ds:
            iid = r.get("instance_id") or r.get("sample_id")
            if iid:
                ids.append(str(iid))
        return ids

    ids = []
    with subset_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                iid = obj.get("instance_id") or obj.get("sample_id")
                if iid:
                    ids.append(str(iid))
            except json.JSONDecodeError:
                pass
    return ids


def run_instance(
    instance_id: str,
    output_path: Path,
    dataset_path: str,
    extra_args: list[str],
    log_path: Path,
    prefix: str = "",
    dry_run: bool = False,
) -> tuple[bool, str]:
    """
    Call main.py to generate (or resume) a single trace.

    - Output file does not exist: fresh generation, pass --dataset/--instance-id/--output
    - Output file exists but unfinished: checkpoint resume, pass --resume (write back to the same file)

    The subprocess's stdout/stderr (including docker_cli / tool-call detailed output) are all written to
    log_path, no longer printed to the main log; the path is only surfaced by the caller on failure for debugging.
    Returns (success, error_message).
    """
    if output_path.exists():
        # Checkpoint resume: recover from the current target file pointed to by --output, while keeping
        # --dataset/--instance-id so image/cwd/smoke-import are still derived per instance.
        cmd = [
            PYTHON, SYNTHESIZE_SCRIPT,
            "--dataset", dataset_path,
            "--instance-id", instance_id,
            "--output", str(output_path),
            "--resume",
        ] + extra_args
        mode = "resume"
    else:
        # Fresh generation
        cmd = [
            PYTHON, SYNTHESIZE_SCRIPT,
            "--dataset", dataset_path,
            "--instance-id", instance_id,
            "--output", str(output_path),
        ] + extra_args
        mode = "new"

    print(f"{prefix} [{mode}] subprocess output → {log_path}")

    if dry_run:
        print(f"{prefix} [{mode}] CMD: {' '.join(cmd)}")
        return True, ""

    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # A resume must append ("a") rather than overwrite ("w"): run.log is the sole data source
        # for per-LLM-call cost accounting (recost_from_logs.py splits opus/deepseek cost by backend
        # from it); overwriting would lose the opus fallback call records of earlier rounds → price cannot be fully computed.
        open_mode = "a" if mode == "resume" else "w"
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
        else:
            return False, f"returncode={result.returncode}"
    except Exception as e:
        return False, str(e)


def process_one(
    *,
    idx: int,
    total: int,
    iid: str,
    output_dir: Path,
    dataset_path: str,
    extra_args: list[str],
    progress_file: Path,
    dry_run: bool,
    retry_failed: bool,
    reclaim_stale: bool = False,
) -> str:
    """Process a single instance; returns "success" / "fail" / "skip".

    Cross-process safe: all "should this be processed" decisions + running/completed/failed writes go through
    progress_store (fcntl file lock, always based on the latest on-disk content), synchronized across processes in
    real time with no data loss. Threads within this process additionally use _progress_lock for a lightweight inner guard.

    All logs carry the `[idx/total][iid]` prefix, to tell sources apart when running in parallel.
    """
    prefix = f"[{idx:2d}/{total}][{iid}]"
    output_path = output_dir / f"{iid}.json"
    run_log_path = output_dir / f"{iid}.run.log"

    print(f"{prefix} start")

    # ── Filesystem truth first: already DONE → skip directly (idempotently backfill completed) ─────
    if output_path.exists() and is_trace_done(output_path):
        if not dry_run:
            with _progress_lock:
                PS.mark_completed(progress_file, iid, {
                    "output": str(output_path),
                    "finished_at": datetime.now().isoformat(),
                    "elapsed_s": 0,
                    "note": "detected-done-on-startup",
                })
        print(f"{prefix} → already done (DONE), skip")
        return "skip"

    if dry_run:
        # dry-run does not claim, does not write running, only prints the command
        if output_path.exists():
            print(f"{prefix} → [dry-run] file exists but unfinished, will checkpoint-resume")
        else:
            print(f"{prefix} → [dry-run] will generate fresh")
        run_instance(
            iid, output_path, dataset_path, extra_args,
            log_path=run_log_path, prefix=prefix, dry_run=True,
        )
        print(f"{prefix} → dry-run, only print the command, do not update progress")
        return "skip"

    # ── progress recorded as completed but file not DONE → drop completed so it regenerates ──────
    with _progress_lock:
        if PS.drop_completed(progress_file, iid):
            print(f"{prefix} [WARN] progress recorded as completed but the trace file is not in DONE state, regenerating")

    # ── claim: decide + write running while holding the lock, eliminating multi-process TOCTOU ─────
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
            print(f"{prefix} → stale running record ({who}, process is gone); conservatively skip. "
                  f"Add --reclaim-stale to take over and rerun")
        else:
            print(f"{prefix} → another process is running ({who}), skip")
        return "skip"
    if decision == "reclaimed":
        print(f"{prefix} [WARN] took over stale running record, rerun")

    # ── run (new or resume); finally cleans up this process's leftover running as a fallback ────────
    if output_path.exists():
        print(f"{prefix} → file exists but unfinished, checkpoint-resume")
    else:
        print(f"{prefix} → generate fresh")

    t0 = time.time()
    try:
        success, err = run_instance(
            iid, output_path, dataset_path, extra_args,
            log_path=run_log_path, prefix=prefix, dry_run=False,
        )
        elapsed = time.time() - t0

        # After a resume finishes, check file state once more (returncode=0 does not mean DONE; may have been interrupted)
        if success and not is_trace_done(output_path):
            success = False
            err = "process exited 0 but trace state is not DONE"

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
        # The success/failure branches already mark_* (running has been removed); this is usually a no-op;
        # only clears this process's leftover running occupancy on a mid-run abnormal exit (does not delete others').
        with _progress_lock:
            PS.release_running(progress_file, iid, os.getpid())


# ─────────────────────────────────────────────────────────────────────────────
# Main flow
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Batch-generate agent traces for 30 instances in fixed order, with interruption recovery.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT_DIR),
        help=f"trace output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    ap.add_argument(
        "--progress-file", default=None,
        help="progress file path (default: <output-dir>/run_progress.json)",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="only print the command, do not actually execute",
    )
    ap.add_argument(
        "--retry-failed", action="store_true",
        help="rerun instances that failed last time (failed ones are skipped by default)",
    )
    ap.add_argument(
        "--reclaim-stale", action="store_true",
        help="take over stale running records (marked running but the worker process is gone) and rerun. "
             "Conservatively skips by default and prompts for manual cleanup; use only when confirmed there is no leftover process.",
    )
    ap.add_argument(
        "--only", default=None,
        help="only process the specified instance_id (comma-separated), for isolated debugging",
    )
    ap.add_argument(
        "--limit", type=int, default=None,
        help="process at most the first N instances (in subset order); processes all if unspecified",
    )
    ap.add_argument(
        "--subset", default=(str(SUBSET_PATH) if SUBSET_PATH else None),
        help=f"instance source: jsonl file or HF Dataset directory (default: {SUBSET_PATH})",
    )
    ap.add_argument(
        "--dataset", default=(str(DATASET_PATH) if DATASET_PATH else None),
        help=f"HuggingFace Dataset directory (default: {DATASET_PATH})",
    )
    ap.add_argument(
        "--reverse", action="store_true",
        help="process the subset in reverse order (back to front). Can run alongside a forward process: the two share output-dir and "
             "the phase-lock directory, relying on DONE detection to avoid duplication; but each should use its own --progress-file and --skills-dir.",
    )
    ap.add_argument(
        "--workers", type=int, default=4,
        help="number of parallel workers (default 4 parallel: each worker spawns an independent synthesize subprocess "
             "+ Docker container, mind machine resources; set to 1 for sequential execution, one finishes before the next, "
             "sharing the same skills experience library and accumulating one by one).",
    )

    # All unknown args are forwarded to main.py
    args, extra_args = ap.parse_known_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    progress_file = Path(args.progress_file) if args.progress_file else output_dir / PROGRESS_FILENAME

    # Load the instance order
    if not args.subset:
        print(
            "[ERROR] no --subset given and paths.subset_jsonl is empty in config.yaml",
            file=sys.stderr,
        )
        sys.exit(1)
    subset_path = Path(args.subset)
    if not subset_path.exists():
        print(f"[ERROR] subset file does not exist: {subset_path}", file=sys.stderr)
        sys.exit(1)

    instance_ids = load_instance_ids(subset_path)
    if not instance_ids:
        print("[ERROR] no instance_id read from the subset file", file=sys.stderr)
        sys.exit(1)

    # Reverse: synthesize from back to front
    if args.reverse:
        instance_ids = list(reversed(instance_ids))

    # If --only specifies a subset
    if args.only:
        only_set = {s.strip() for s in args.only.split(",") if s.strip()}
        instance_ids = [iid for iid in instance_ids if iid in only_set]
        if not instance_ids:
            print("[ERROR] none of the instance_id specified by --only were found in the subset", file=sys.stderr)
            sys.exit(1)

    # If --limit specifies a maximum count
    if args.limit is not None:
        instance_ids = instance_ids[:args.limit]

    # Load progress
    progress = load_progress(progress_file)

    total = len(instance_ids)
    already_done = len(progress["completed"])
    already_failed = len(progress["failed"])
    already_running = len(progress.get("running", {}))

    print("=" * 70)
    print(f"  Batch generation script  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Total instances : {total}")
    print(f"  Completed       : {already_done}")
    print(f"  Failed last time: {already_failed}  (--retry-failed={'yes' if args.retry_failed else 'no, skip'})")
    if already_running:
        print(f"  Running         : {already_running}  (--reclaim-stale={'yes' if args.reclaim_stale else 'no, conservatively skip if stale'})")
        for rid, rinfo in progress["running"].items():
            print(f"    - {rid}  (pid={rinfo.get('pid')} host={rinfo.get('host')} started_at={rinfo.get('started_at')})")
    print(f"  Output dir      : {output_dir}")
    print(f"  Progress file   : {progress_file}")
    if extra_args:
        print(f"  Forwarded args  : {extra_args}")
    if args.limit is not None:
        print(f"  Max count       : {args.limit}")
    workers = max(1, args.workers)
    print(f"  Parallel workers: {workers}")
    if args.dry_run:
        print("  [DRY-RUN mode: not actually executing]")
    print("=" * 70)

    n_success = 0
    n_fail = 0
    n_skip = 0

    def submit(idx: int, iid: str) -> str:
        return process_one(
            idx=idx,
            total=total,
            iid=iid,
            output_dir=output_dir,
            dataset_path=args.dataset,
            extra_args=extra_args,
            progress_file=progress_file,
            dry_run=args.dry_run,
            retry_failed=args.retry_failed,
            reclaim_stale=args.reclaim_stale,
        )

    if workers == 1:
        # Sequential execution
        for idx, iid in enumerate(instance_ids, 1):
            status = submit(idx, iid)
            if status == "success":
                n_success += 1
            elif status == "fail":
                n_fail += 1
            else:
                n_skip += 1
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(submit, idx, iid): iid
                for idx, iid in enumerate(instance_ids, 1)
            }
            for fut in as_completed(futures):
                iid = futures[fut]
                try:
                    status = fut.result()
                except Exception as exc:  # fallback: an uncaught exception in a worker should not swallow progress
                    print(f"[{iid}] → worker exception: {exc}")
                    with _progress_lock:
                        PS.mark_failed(progress_file, iid, {
                            "error": f"worker exception: {exc}",
                            "failed_at": datetime.now().isoformat(),
                        })
                        PS.release_running(progress_file, iid, os.getpid())
                    n_fail += 1
                    continue
                if status == "success":
                    n_success += 1
                elif status == "fail":
                    n_fail += 1
                else:
                    n_skip += 1

    # Final aggregation: reread the latest from disk (in-memory progress is only a startup snapshot; writes all go through PS.*)
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
