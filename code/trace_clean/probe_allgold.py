#!/usr/bin/env python3
"""
probe_allgold.py — offline read-only measurement of "which tests actually pass after applying all gold patches".

Purpose (find "bad nodes": nodes that fail even with all gold applied -> data/environment problem, ultimately dropped from both f2p and p2p):
  base -> replay_model_patches(no boundary) = successively apply SEED and every enabled+ok subtask delta
  -> the worktree becomes the final state of the whole trace; then, for each enabled+ok subtask (including SEED_TASK),
  run own_f2p/own_p2p under its own test_patch, recording PASSED/FAILED/... for each node.
  This directly exposes cross-big-task regressions such as "does a later big task's gold break an earlier big task's f2p".
  Downstream build_benchmark.py drops FAILED/NOT_FOUND nodes from both final_f2p and final_p2p.

Relationship to probe_base_seed.py (isomorphic mirror):
  - This script      : replay [all] gold (final state), exposing cross-big-task regressions.
  - probe_base_seed  : replay [only up to SEED] (baseline), judging whether p2p should pass at the starting state.

Safety: read-only / only imports existing functions from trace_synth. Each trace spins up its own short-named
ephemeral container (keep_on_fail=False, auto-cleaned by container ID after testing) and does not pull images.

Usage (all paths go through config.py, resolved relative to this package by default):
  python probe_allgold.py                 # run everything in ../../data/trace, output to ../../data/_allgold_probe
  python probe_allgold.py --only <iid>    # single-instance smoke test
  Use --traces-dir / --out-dir to override the default directories.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---- Dependencies: trace_synth low-level modules (relative path, injected into sys.path via config) ----
import config as cfg_paths
cfg_paths.ensure_synth_on_path()

from swebench_dockerlib import DockerSession           # noqa: E402
from local_test_executor import (                      # noqa: E402
    bootstrap_session,
    replay_model_patches,
    run_tests_under_test_patch,
)
import agent_trace as at                               # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("probe_allgold")

# Known container crash / infrastructure flakiness signatures. A hit triggers a trace-level retry.
_INFRA_SIGNS = (
    "is not running", "[restore-fail]", "setns", "procReady",
    "docker-write-fail", "exit 125", "no such container",
    "OCI runtime", "cannot exec", "connection reset",
)


def _looks_infra(msg: str) -> bool:
    m = (msg or "").lower()
    return any(s.lower() in m for s in _INFRA_SIGNS)


def _enabled_ok_events(trace: dict) -> list[dict]:
    """Return, in trace order, every ATTEMPT event that is enabled and has result.ok is True.

    This matches the selection rule of replay_model_patches verbatim (including SEED_TASK), so for each
    subtask we test, its gold has indeed already been applied into the current all-gold worktree. Among the
    multiple retries of each (big, sub), only the last one is enabled+ok, so each subtask appears exactly once.
    """
    out = []
    for ev in trace.get("events") or []:
        if ev.get("kind") != "ATTEMPT":
            continue
        if not at.is_enabled(ev):
            continue
        if (ev.get("result") or {}).get("ok") is not True:
            continue
        out.append(ev)
    return out


def _probe_one_trace(path: str, out_dir: str, timeout_sec: int,
                     max_trace_retries: int = 2) -> dict:
    """For one trace: apply all gold, then test own_f2p/own_p2p for each enabled+ok subtask.

    Returns a summary dict and writes per-node details to out_dir/<instance_id>.json.
    If the trace level (bootstrap/replay or a whole round) fails on an infrastructure signature, retry up to max_trace_retries.
    """
    trace = json.load(open(path))
    cfg = trace.get("config", {})
    instance = trace.get("instance", {}) or {}
    inst_id = instance.get("instance_id") or os.path.splitext(os.path.basename(path))[0]

    cwd = cfg.get("cwd", "/testbed")
    pytest_bin = cfg.get("pytest_bin", "/opt/miniconda3/envs/testbed/bin/pytest")
    image = cfg.get("image")
    runner_kind = cfg.get("test_runner_kind")
    base_commit = instance.get("base_commit", "") or "HEAD"
    login_shell = runner_kind != "pro_official"

    # The two keys needed for Pro/Verified routing
    sim_instance = dict(instance)
    sim_instance["_config_image"] = image
    sim_instance["_test_runner_kind"] = runner_kind

    events = _enabled_ok_events(trace)

    result: dict = {
        "instance_id": inst_id,
        "image": image,
        "runner_kind": runner_kind,
        "cwd": cwd,
        "base_commit": base_commit,
        "n_events_enabled_ok": len(events),
        "replay_ok": False,
        "replay_error": None,
        "attempts_used": 0,
        "subtasks": [],
    }

    last_err = None
    for attempt in range(1, max_trace_retries + 1):
        result["attempts_used"] = attempt
        sess = DockerSession(image=image, keep_on_fail=False, login_shell=login_shell)
        try:
            sess.start()
            cname = getattr(sess, "container_name", "?")
            logger.info("[%s] start container=%s (try %d/%d) -> bootstrap + replay ALL gold ...",
                        inst_id, cname, attempt, max_trace_retries)
            bootstrap_session(sess, trace)
            replay_model_patches(sess, trace, verbose=False)  # no boundary = apply all gold
            result["replay_ok"] = True
            result["replay_error"] = None

            subtasks = []
            for ev in events:
                bid = ev.get("big_task_id", "UNKNOWN")
                sid = ev.get("subtask_id", "")
                f2p, p2p = at.attempt_own_tests(ev)
                f2p = list(f2p or [])
                p2p = list(p2p or [])
                st = {
                    "big_task_id": bid, "subtask_id": sid,
                    "n_f2p": len(f2p), "n_p2p": len(p2p),
                    "f2p": {}, "p2p": {},
                    "f2p_pass": 0, "p2p_pass": 0,
                    "ok": True, "error": None,
                }
                if not f2p and not p2p:
                    # Subtask that declares no tests: record 0/0, treat as ok (nothing to test)
                    subtasks.append(st)
                    continue
                try:
                    rep = run_tests_under_test_patch(
                        sess,
                        test_patch=ev.get("test_patch") or "",
                        f2p_tests=f2p,
                        p2p_tests=p2p,
                        base_commit=base_commit,
                        pytest_bin=pytest_bin,
                        cwd=cwd,
                        timeout_sec=timeout_sec,
                        instance=sim_instance,
                        collect_node_details=True,
                    )
                    status = dict(rep.per_node_status or {})
                    st["f2p"] = {n: status.get(n, "NOT_FOUND") for n in f2p}
                    st["p2p"] = {n: status.get(n, "NOT_FOUND") for n in p2p}
                    st["f2p_pass"] = sum(1 for n in f2p if status.get(n) == "PASSED")
                    st["p2p_pass"] = sum(1 for n in p2p if status.get(n) == "PASSED")
                    st["ok"] = bool(rep.ok)
                except Exception as e:  # noqa: BLE001
                    st["error"] = repr(e)[:2000]
                    st["ok"] = False
                    if _looks_infra(st["error"]):
                        # A container crash would contaminate all subsequent subtasks -> raise to trigger a trace-level retry
                        raise RuntimeError(
                            f"[infra-abort during subtask {bid}/{sid}] {st['error']}")
                # Report each subtask in real time
                logger.info("[%s]   %-10s/%-12s f2p=%d/%d p2p=%d/%d %s",
                            inst_id, bid, sid, st["f2p_pass"], st["n_f2p"],
                            st["p2p_pass"], st["n_p2p"],
                            "OK" if st["ok"] else "FAIL"
                            + (f" err={st['error'][:80]}" if st["error"] else ""))
                subtasks.append(st)

            result["subtasks"] = subtasks
            last_err = None
            break  # this round succeeded
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            result["replay_error"] = last_err
            infra = _looks_infra(last_err)
            logger.warning("[%s] try %d failed (%s): %s",
                           inst_id, attempt, "infra" if infra else "non-infra",
                           last_err[:300])
            if not infra:
                break  # non-infrastructure error: a real data problem, do not retry
        finally:
            try:
                sess.stop()
            except Exception:
                pass

    # Compute trace-level summary (grouped by big_task)
    per_big: dict = {}
    all_f2p_pass = all_f2p_tot = all_p2p_pass = all_p2p_tot = 0
    for st in result["subtasks"]:
        b = st["big_task_id"]
        d = per_big.setdefault(b, {"f2p_pass": 0, "f2p_tot": 0,
                                   "p2p_pass": 0, "p2p_tot": 0,
                                   "subtasks_ok": 0, "subtasks_tot": 0})
        d["f2p_pass"] += st["f2p_pass"]; d["f2p_tot"] += st["n_f2p"]
        d["p2p_pass"] += st["p2p_pass"]; d["p2p_tot"] += st["n_p2p"]
        d["subtasks_tot"] += 1
        d["subtasks_ok"] += 1 if st["ok"] else 0
        all_f2p_pass += st["f2p_pass"]; all_f2p_tot += st["n_f2p"]
        all_p2p_pass += st["p2p_pass"]; all_p2p_tot += st["n_p2p"]
    result["per_big_task"] = per_big
    result["totals"] = {
        "f2p_pass": all_f2p_pass, "f2p_tot": all_f2p_tot,
        "p2p_pass": all_p2p_pass, "p2p_tot": all_p2p_tot,
        "all_f2p_pass": all_f2p_tot > 0 and all_f2p_pass == all_f2p_tot,
        "all_p2p_pass": all_p2p_tot > 0 and all_p2p_pass == all_p2p_tot,
    }

    # Write per-trace details to disk (written as soon as done, so progress can be inspected anytime)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{inst_id}.json"), "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    tag = "OK" if result["replay_ok"] and not last_err else "TRACE-FAIL"
    logger.info("[%s] DONE %s  f2p=%d/%d p2p=%d/%d (subtasks=%d)%s",
                inst_id, tag, all_f2p_pass, all_f2p_tot,
                all_p2p_pass, all_p2p_tot, len(result["subtasks"]),
                f"  replay_error={last_err[:120]}" if last_err else "")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces-dir", default=str(cfg_paths.TRACE_IN_DIR),
                    help="raw trace input directory (default ../../data/trace)")
    ap.add_argument("--out-dir", default=str(cfg_paths.ALLGOLD_DIR),
                    help="probe result output directory (default ../../data/_allgold_probe)")
    ap.add_argument("--pool", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--only", default="", help="comma-separated instance_id subset, for debugging")
    args = ap.parse_args()

    files = sorted(f for f in glob.glob(os.path.join(args.traces_dir, "*.json"))
                   if os.path.basename(f) != "run_progress.json")
    if args.only:
        want = set(x.strip() for x in args.only.split(",") if x.strip())
        files = [f for f in files
                 if os.path.splitext(os.path.basename(f))[0] in want
                 or (json.load(open(f)).get("instance", {}) or {}).get("instance_id") in want]
    if not files:
        logger.error("no trace files to run: %s", args.traces_dir)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    logger.info("%d traces total, concurrency pool=%d, timeout=%ds -> out=%s",
                len(files), args.pool, args.timeout, args.out_dir)

    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.pool)) as pool:
        futs = {pool.submit(_probe_one_trace, f, args.out_dir, args.timeout): f
                for f in files}
        done = 0
        for fut in as_completed(futs):
            done += 1
            f = futs[fut]
            try:
                results.append(fut.result())
            except Exception as e:  # noqa: BLE001
                logger.error("[%s] uncaught exception: %s\n%s",
                             os.path.basename(f), e, traceback.format_exc())
            logger.info("=== progress %d/%d ===", done, len(files))

    # Summary table
    summary = {"n_traces": len(results), "traces": []}
    for r in sorted(results, key=lambda x: x["instance_id"]):
        summary["traces"].append({
            "instance_id": r["instance_id"],
            "runner_kind": r["runner_kind"],
            "replay_ok": r["replay_ok"],
            "replay_error": r["replay_error"],
            "n_subtasks": len(r["subtasks"]),
            "totals": r["totals"],
            "per_big_task": r["per_big_task"],
        })
    with open(os.path.join(args.out_dir, "_SUMMARY.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print a compact overall table to the console
    print("\n" + "=" * 100)
    print("ALL-GOLD PROBE SUMMARY  (per-subtask own_f2p/own_p2p pass status after applying all gold)")
    print("=" * 100)
    print(f"{'instance_id':45s} {'runner':13s} {'rp':3s} {'f2p':>10s} {'p2p':>10s}")
    n_all_f2p = n_repl_fail = 0
    for t in summary["traces"]:
        tot = t["totals"]
        f2p = f"{tot['f2p_pass']}/{tot['f2p_tot']}"
        p2p = f"{tot['p2p_pass']}/{tot['p2p_tot']}"
        rp = "OK" if t["replay_ok"] else "ERR"
        if not t["replay_ok"]:
            n_repl_fail += 1
        if tot["all_f2p_pass"]:
            n_all_f2p += 1
        print(f"{t['instance_id']:45s} {str(t['runner_kind']):13s} {rp:3s} {f2p:>10s} {p2p:>10s}")
    print("-" * 100)
    print(f"traces={len(summary['traces'])}  all-f2p-pass={n_all_f2p}  replay-failed={n_repl_fail}  "
          f"elapsed={time.time()-t0:.0f}s")
    print(f"detail: {args.out_dir}/<instance_id>.json    summary: {args.out_dir}/_SUMMARY.json")


if __name__ == "__main__":
    main()
