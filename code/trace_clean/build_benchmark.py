#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_benchmark.py — turn raw agent_trace.v2 traces into clean benchmark JSON (one file per instance).

Inputs / outputs (all relative paths, see config.py):
  input   ../../data/trace/*.json              raw synthesized traces (produced by trace_synth)
  probes  ../../data/_base_seed_probe/*.json    p2p baseline probe (produced by probe_base_seed.py)
          ../../data/_allgold_probe/*.json      all-gold final-state bad-node probe (produced by probe_allgold.py)
  output  ../../benchmark/<id>.json             cleaned JSON (one file per instance)

Cleaning rules:
- Top level: drop schema / state; flatten instance metadata into config; append final_f2p / final_p2p at the end.
- events: keep every PLAN plus the ok=true attempt for each (big_task_id, subtask_id) (deduped), discard failures/retries.
  - PLAN: drop uninformative fields, move kind/id to the front, place uplift_proof after subtasks_blueprint.
  - ATTEMPT: drop result/rationale etc.; rename patch to gold_patch and place it before test_patch.
- final_f2p / final_p2p: object lists [{"sid": "<big>/<subtask>", "node": "<test node>"}] for the whole
  instance (sid kept so the evaluation loaders can rebuild the (cell_id, sid, node) key). Node set = union of
  own_f2p and own_p2p across the deduped subtasks, refined by two gold-side calibration steps: bad allgold
  nodes (FAILED/NOT_FOUND) are dropped from both (not-in-list = excluded); own_p2p nodes that fail at the
  baseline are reclassified into f2p. Every surviving (sid, node) is kept verbatim (no node-level dedup).

Usage:
  python build_benchmark.py                     # full run (skip existing)
  python build_benchmark.py --force             # full run, overwrite/update
  python build_benchmark.py --only <filename>    # run a single file (dry run; add --force if target exists)
Never modifies the raw traces; only writes to ../../benchmark/.
"""
import argparse
import glob
import json
import os
import sys

import config as cfg_paths

SRC = str(cfg_paths.TRACE_IN_DIR)          # ../../data/trace
DST = str(cfg_paths.BENCH_OUT_DIR)         # ../../benchmark
ALLGOLD_DIR = str(cfg_paths.ALLGOLD_DIR)   # ../../data/_allgold_probe
SEEDPROBE_DIR = str(cfg_paths.SEEDPROBE_DIR)  # ../../data/_base_seed_probe

# instance fields to drop: all duplicate the SEED event (same content, different format).
# problem_statement is dropped (keep only the cleaned seed.natural_language_goal); the str+list
# double representations of gold patch/test_patch/F2P/P2P are all dropped (seed has own_f2p/own_p2p/patch/test_patch).
# Kept: repo/instance_id/base_commit/version/difficulty/dockerhub_tag/repo_language/
#       source/hints_text/before_repo_set_cmd/selected_test_files_to_run and other provenance metadata.
INSTANCE_DROP = {
    "problem_statement",
    "patch", "test_patch",
    "FAIL_TO_PASS", "FAIL_TO_PASS_parsed",
    "PASS_TO_PASS", "PASS_TO_PASS_parsed",
}

# uninformative config fields to drop (all empty / constant); keep image/cwd/test_runner_kind/pytest_bin.
CONFIG_DROP = {"smoke_import", "fill_mode"}

# uninformative PLAN fields to drop:
#   own_f2p/own_p2p/test_patch -- always empty for PLAN (these are ATTEMPT fields);
#   enabled/round/_recovered_from -- constant values;
#   disabled_reason/acceptance_criteria_note -- almost always empty; result/dim_tags have no release value.
PLAN_DROP = {
    "own_f2p", "own_p2p", "test_patch",
    "enabled", "round", "_recovered_from",
    "disabled_reason", "acceptance_criteria_note",
    "result", "dim_tags",
}

# PLAN kept-field output order: kind / id at the front, uplift_proof after subtasks_blueprint.
PLAN_ORDER = [
    "kind", "id", "big_task_id", "reason",
    "natural_language_goal", "acceptance_criteria",
    "subtasks_blueprint", "uplift_proof",
]

# uninformative ATTEMPT fields to drop: enabled (all kept attempts are ok=true -> always true), disabled_reason (empty),
# result (only used to filter ok=true / dedupe, not written out), rationale.
ATTEMPT_DROP = {"enabled", "disabled_reason", "result", "rationale"}

# ATTEMPT kept-field output order: patch renamed to gold_patch and placed before test_patch.
ATTEMPT_ORDER = [
    "kind", "big_task_id", "subtask_id",
    "natural_language_goal", "acceptance_criteria", "user_query",
    "gold_patch", "test_patch",
    "own_f2p", "own_p2p",
]

# source abbreviation -> full dataset name.
SOURCE_FULLNAME = {
    "verified": "SWE-bench Verified",
    "pro": "SWE-bench Pro",
}


def _cell_id(stem):
    """probe / trace filename stem -> cell_id: strip the instance_ prefix (align both sides)."""
    return stem[len("instance_"):] if stem.startswith("instance_") else stem


def load_allgold_exclude():
    """Read _allgold_probe/*.json and return the set of bad nodes that still fail with the full gold applied
    {(cell_id, sid, node)} (both f2p/p2p collect FAILED/NOT_FOUND)."""
    exclude = set()
    for path in glob.glob(os.path.join(ALLGOLD_DIR, "*.json")):
        stem = os.path.basename(path)[:-len(".json")]
        if stem == "_SUMMARY":
            continue
        cid = _cell_id(stem)
        try:
            ag = json.load(open(path))
        except Exception as e:
            print(f"  [warn] allgold read failed {stem}: {e}", file=sys.stderr); continue
        for st in ag.get("subtasks", []):
            big, sub = st.get("big_task_id"), st.get("subtask_id")
            if not big or not sub:
                continue
            sid = f"{big}/{sub}"
            for kind in ("f2p", "p2p"):
                for node, status in (st.get(kind) or {}).items():
                    if status in ("FAILED", "NOT_FOUND"):
                        exclude.add((cid, sid, node))
    return exclude


def load_seedprobe_reclass():
    """Read _base_seed_probe/*.json and return the set of non-SEED p2p nodes that are != PASSED at the
    base+SEED baseline {(cell_id, sid, node)} (these fail at the baseline and are not stable regressions
    -> reclassify into f2p)."""
    reclass = set()
    for path in glob.glob(os.path.join(SEEDPROBE_DIR, "*.json")):
        stem = os.path.basename(path)[:-len(".json")]
        if stem == "_SUMMARY":
            continue
        cid = _cell_id(stem)
        try:
            pr = json.load(open(path))
        except Exception as e:
            print(f"  [warn] seedprobe read failed {stem}: {e}", file=sys.stderr); continue
        for st in pr.get("subtasks", []):
            big, sub = st.get("big_task_id"), st.get("subtask_id")
            if not big or not sub:
                continue
            sid = f"{big}/{sub}"
            for node, status in (st.get("p2p") or {}).items():
                if status != "PASSED":
                    reclass.add((cid, sid, node))
    return reclass


def compute_final_lists(d, cell_id, allgold_exclude, seed_reclass):
    """Compute the whole-instance final_f2p / final_p2p object lists from one trace.

    Each entry is {"sid": "<big>/<subtask>", "node": "<test node>"} -- sid is kept so the evaluation
    loaders can rebuild the (cell_id, sid, node) key. Node set = own_f2p + own_p2p across the deduped
    enabled&ok subtasks (dedup is by (big_task_id, subtask_id) at the events layer, NOT by node); then
    apply two gold-side calibration steps:
      - bad allgold nodes -> dropped from both f2p and p2p (they simply do not appear: not-in-list = excluded);
      - own_p2p nodes that fail at the baseline (seed_reclass) -> moved from p2p into f2p.
    Every surviving (sid, node) is kept verbatim -- no node-level de-duplication -- so the lists are a
    lossless view of the calibrated own_f2p/own_p2p.
    """
    seen = set()
    f2p_list, p2p_list = [], []
    for e in d.get("events", []):
        if e.get("kind") != "ATTEMPT":
            continue
        r = e.get("result") if isinstance(e.get("result"), dict) else {}
        if r.get("ok") is not True:
            continue
        key = (e.get("big_task_id"), e.get("subtask_id"))
        if key in seen:
            continue
        seen.add(key)
        sid = f"{e.get('big_task_id')}/{e.get('subtask_id')}"
        for node in (e.get("own_f2p") or []):
            if (cell_id, sid, node) in allgold_exclude:
                continue
            f2p_list.append({"sid": sid, "node": node})
        for node in (e.get("own_p2p") or []):
            if (cell_id, sid, node) in allgold_exclude:
                continue
            if (cell_id, sid, node) in seed_reclass:      # fails at baseline -> reclassify as f2p
                f2p_list.append({"sid": sid, "node": node})
            else:
                p2p_list.append({"sid": sid, "node": node})
    return f2p_list, p2p_list


def build_one(path, allgold_exclude, seed_reclass):
    d = json.load(open(path))
    out = {}
    # Top-level schema / state are not written to the benchmark.
    # config = environment config + instance metadata (flattened into one layer, no field-name conflicts).
    #   instance drops fields that duplicate SEED; config drops uninformative fields; source is expanded to the full dataset name.
    inst = {k: v for k, v in d["instance"].items() if k not in INSTANCE_DROP}
    if "source" in inst:
        inst["source"] = SOURCE_FULLNAME.get(inst["source"], inst["source"])
    conf = {k: v for k, v in d.get("config", {}).items() if k not in CONFIG_DROP}
    conf.update(inst)                        # flatten instance fields into config
    out["config"] = conf

    events = d.get("events", [])
    new_events = []
    seen_subtasks = set()                    # dedupe by (big_task_id, subtask_id)
    for e in events:
        kind = e.get("kind")
        if kind == "PLAN":
            src = {k: v for k, v in e.items() if k not in PLAN_DROP}
            ev_out = {k: src[k] for k in PLAN_ORDER if k in src}
            for k, v in src.items():
                if k not in ev_out:
                    ev_out[k] = v
            new_events.append(ev_out)
            continue
        if kind == "ATTEMPT":
            r = e.get("result") if isinstance(e.get("result"), dict) else {}
            if r.get("ok") is not True:
                continue                     # keep only successful attempts (filter on the raw result, then drop that field)
            key = (e.get("big_task_id"), e.get("subtask_id"))
            if key in seen_subtasks:
                continue                     # exactly one per subtask
            seen_subtasks.add(key)
            src = {k: v for k, v in e.items() if k not in ATTEMPT_DROP}
            if "patch" in src:
                src["gold_patch"] = src.pop("patch")
            ev_out = {k: src[k] for k in ATTEMPT_ORDER if k in src}
            for k, v in src.items():
                if k not in ev_out:
                    ev_out[k] = v
            new_events.append(ev_out)
            continue
        # other kinds (AUGMENT/VERIFY, fallback: drop result)
        new_events.append({k: v for k, v in e.items() if k != "result"})
    out["events"] = new_events

    # naming: strip the instance_ prefix from instance_id
    iid = d["instance"]["instance_id"]
    name = iid[len("instance_"):] if iid.startswith("instance_") else iid

    # append the final f2p/p2p (whole-instance flat-deduped lists) at the end.
    f2p_list, p2p_list = compute_final_lists(d, name, allgold_exclude, seed_reclass)
    out["final_f2p"] = f2p_list
    out["final_p2p"] = p2p_list

    return out, name, len(events), len(new_events)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="process only this filename (basename)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing target file (for updates; default skips existing)")
    args = ap.parse_args()

    os.makedirs(DST, exist_ok=True)
    # load both gold-side calibration probes once, for the final_f2p/final_p2p computation.
    allgold_exclude = load_allgold_exclude()
    seed_reclass = load_seedprobe_reclass()
    print(f"probes loaded: allgold bad nodes {len(allgold_exclude)} | seed reclassified {len(seed_reclass)} nodes",
          file=sys.stderr)
    if not allgold_exclude and not seed_reclass:
        print(f"  [warn] no probe data read ({ALLGOLD_DIR} / {SEEDPROBE_DIR} empty?) "
              f"final_f2p/final_p2p will fall back to uncalibrated own_f2p/own_p2p. "
              f"Please run probe_allgold.py + probe_base_seed.py first.", file=sys.stderr)

    files = [f for f in glob.glob(os.path.join(SRC, "*.json"))
             if ".bak" not in f and "run_progress" not in f]
    if args.only:
        files = [f for f in files if os.path.basename(f) == args.only]
        if not files:
            print(f"not found: {args.only}", file=sys.stderr); sys.exit(1)
    if not files:
        print(f"no trace files to process: {SRC}", file=sys.stderr); sys.exit(1)

    for f in sorted(files):
        out, name, n_in, n_out = build_one(f, allgold_exclude, seed_reclass)
        dst = os.path.join(DST, name + ".json")
        if os.path.exists(dst) and not args.force:
            print(f"⚠️  skip (target exists, not overwriting; add --force to update): {name}.json", file=sys.stderr); continue
        with open(dst, "w") as w:
            json.dump(out, w, ensure_ascii=False, indent=2)
        src_mb = os.path.getsize(f) / 1024 / 1024
        dst_mb = os.path.getsize(dst) / 1024 / 1024
        print(f"✅ {os.path.basename(f)}")
        print(f"     → {name}.json")
        print(f"     events: {n_in} → {n_out} | final_f2p {len(out['final_f2p'])} / final_p2p {len(out['final_p2p'])}"
              f" | size: {src_mb:.1f}MB → {dst_mb:.2f}MB (saved {100*(1-dst_mb/src_mb):.1f}%)" if src_mb else "")


if __name__ == "__main__":
    main()
