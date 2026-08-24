#!/usr/bin/env bash
# =============================================================================
# run_model_eval.sh — evaluate an existing model with the user-simulation
#                      closed loop -> print f2p/p2p scores
#
# Two steps (all paths relative to this repo):
#   (1) user_simulation  run one closed-loop conversation per <model> x <instance>
#       x <persona>; at the end, actually run the final f2p/p2p for every subtask
#       -> data/usersim_final/<model>/<iid>/<persona>/
#       (needs Docker: the assistant holds one live container shared across the run)
#   (2) evaluation       stat_all_personas_final.py reads the run outputs + the
#       benchmark and prints the three-column score table
#
# This script does NOT auto-run. First:
#   - Point the assistant endpoints at your own: edit code/user_simulation/config.yaml
#     (the assistant_models registry) or pass --assistant-model name@base_url freeform;
#   - Make sure data/trace/ has raw traces (run run_generate_benchmark.sh step (1), or supply your own);
#   - Make sure Docker is available and the images are present locally.
# Then unblock and run the steps one at a time.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$ROOT/code"
PY="${PYTHON:-python}"

US="$CODE/user_simulation"
TRACE_DIR="$ROOT/data/trace"
OUT_ROOT="$ROOT/data/usersim_final"

# The assistant takes two -c configs: swebench.yaml is resolved from the vendored
# minisweagent's built-in config dir; swejourney.yaml is this repo's copy under
# user_simulation/ (absolute or relative both work).
CFG=(-c swebench.yaml -c "$US/swejourney.yaml")

# -- To fill in: assistant model aliases, the 4 personas, the instance list ---
#  - MODELS: aliases from config.yaml's assistant_models (e.g. flash/pro/glm), or
#    freeform name@base_url. The alias -> output-dir name (TAG) mapping is yours;
#    here we use the alias itself as the TAG.
#  - PERSONAS: the 4 personas map to prompts/sim_*.yaml (without the .yaml suffix).
MODELS=("${MODELS:-flash}")                    # e.g. MODELS="flash pro glm" (space-separated, quoted)
PERSONAS=(${PERSONAS:-non_programming_user product_stakeholder repo_newcomer_engineer repository_steward})

echo "============================================================"
echo " SWEJourney - user-simulation evaluation pipeline"
echo "   ROOT=$ROOT  PY=$PY"
echo "   MODELS=${MODELS[*]}"
echo "   PERSONAS=${PERSONAS[*]}"
echo "============================================================"

# -- (1) run the closed loop per model x instance x persona -------------------
echo
echo "===== (1) user_simulation: closed-loop conversation + final f2p/p2p (Docker) ====="
for MODEL in "${MODELS[@]}"; do
  # alias (freeform may contain @) -> a filesystem-safe directory TAG
  TAG="${MODEL%%@*}"; TAG="${TAG//\//_}"
  for TRACE in "$TRACE_DIR"/*.json; do
    IID="$(basename "$TRACE" .json)"
    for PERSONA in "${PERSONAS[@]}"; do
      OUT="$OUT_ROOT/$TAG/$IID/$PERSONA"
      mkdir -p "$OUT"
      echo "[run] $TAG / $IID / $PERSONA -> $OUT"
      ( cd "$US" && "$PY" usersim_main.py \
          --trace "$TRACE" \
          --sim-config "$US/prompts/sim_${PERSONA}.yaml" \
          "${CFG[@]}" \
          --assistant-model "$MODEL" \
          --out "$OUT" \
          --log-name "$TAG/$IID/$PERSONA" )
    done
  done
done

# -- (2) score: read run outputs + benchmark, print the three-column table ----
#    (D1 process f2p / D3 final f2p / D4 final p2p)
echo
echo "===== (2) evaluation: stat_all_personas_final prints the score table ====="
( cd "$CODE/evaluation" && "$PY" stat_all_personas_final.py )

echo
echo "============================================================"
echo " Done: run outputs in $OUT_ROOT/; the score table is printed above"
echo "============================================================"
