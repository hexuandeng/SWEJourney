#!/usr/bin/env bash
# =============================================================================
# run_generate_benchmark.sh — synthesize traces -> probe -> clean, producing benchmark/
#
# Pipeline (4 steps, all paths relative to this repo):
#   (1) trace_synth     synthesize raw agent_trace.v2 traces        -> data/trace/
#   (2) probe_allgold   all-gold final-state bad-node probe (Docker)-> data/_allgold_probe/
#   (3) probe_base_seed base+SEED starting-state p2p probe (Docker) -> data/_base_seed_probe/
#   (4) build_benchmark clean + assemble final_f2p/final_p2p        -> benchmark/
#
# This script does NOT auto-run anything — it wires the commands in order. First
# fill in your endpoints / dataset in code/trace_synth/config.yaml and confirm
# Docker is available, then unblock and run the steps one at a time. Steps (2)/(3)
# spawn many short-lived containers (one per task, self-cleaned after testing);
# confirm your machine resources before running.
# =============================================================================
set -euo pipefail

# -- Repo root (this script lives in SWEJourney/) -----------------------------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$ROOT/code"
PY="${PYTHON:-python}"          # override the interpreter with PYTHON=/path/to/python

# -- To fill in / confirm -----------------------------------------------------
#  - Synthesis endpoints/models: edit code/trace_synth/config.yaml (clients.* base_url/model + API-key env vars)
#  - Dataset/subset: config.yaml paths.subset_jsonl / paths.full_dataset, or override via the CLI below
SUBSET="${SUBSET:-}"            # e.g. SUBSET=/path/to/subset.jsonl (empty -> use config.yaml)
DATASET="${DATASET:-}"         # e.g. DATASET=/path/to/hf_dataset_dir (empty -> use config.yaml)
WORKERS="${WORKERS:-4}"        # synthesis concurrency (one synthesize subprocess + one container per worker)
PROBE_POOL="${PROBE_POOL:-16}" # probe concurrency (max simultaneous containers)
PROBE_TIMEOUT="${PROBE_TIMEOUT:-1800}"

echo "============================================================"
echo " SWEJourney - benchmark generation pipeline"
echo "   ROOT=$ROOT"
echo "   PY=$PY  WORKERS=$WORKERS  PROBE_POOL=$PROBE_POOL"
echo "============================================================"

# -- (1) synthesize raw traces -> data/trace/ ---------------------------------
echo
echo "===== (1) trace_synth: synthesize raw traces -> data/trace/ ====="
SUBSET_ARG=(); [ -n "$SUBSET" ]  && SUBSET_ARG=(--subset "$SUBSET")
DATASET_ARG=(); [ -n "$DATASET" ] && DATASET_ARG=(--dataset "$DATASET")
(
  cd "$CODE/trace_synth"
  "$PY" run_all_instances.py \
    --output-dir "$ROOT/data/trace" \
    --workers "$WORKERS" \
    "${SUBSET_ARG[@]}" "${DATASET_ARG[@]}"
)

# -- (2) all-gold final-state probe (Docker) -> data/_allgold_probe/ ----------
echo
echo "===== (2) probe_allgold: all-gold final-state bad-node probe (Docker) ====="
(
  cd "$CODE/trace_clean"
  "$PY" probe_allgold.py --pool "$PROBE_POOL" --timeout "$PROBE_TIMEOUT"
)

# -- (3) base+SEED starting-state probe (Docker) -> data/_base_seed_probe/ ----
echo
echo "===== (3) probe_base_seed: base+SEED starting-state p2p probe (Docker) ====="
(
  cd "$CODE/trace_clean"
  "$PY" probe_base_seed.py --pool "$PROBE_POOL" --timeout "$PROBE_TIMEOUT"
)

# -- (4) clean + assemble -> benchmark/ ---------------------------------------
echo
echo "===== (4) build_benchmark: clean + final_f2p/final_p2p -> benchmark/ ====="
(
  cd "$CODE/trace_clean"
  "$PY" build_benchmark.py --force
)

echo
echo "============================================================"
echo " Done: benchmark written to $ROOT/benchmark/"
echo "============================================================"
