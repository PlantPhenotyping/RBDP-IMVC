#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${RBDP_PYTHON:-python}"
TRAINING_ROOT="${1:-${PROJECT_ROOT}/outputs}"
EVALUATION_ROOT="${2:-${PROJECT_ROOT}/outputs/gate2_adaptive}"

experiments=(
  "pilot_caltech_shift"
  "pilot_scene_shift"
  "pilot_landuse_shift"
)
datasets=(
  "Caltech101-20"
  "Scene-15"
  "LandUse-21"
)
seeds=(10 11 12 13 14)
collector_roots=()

cd "${PROJECT_ROOT}"
for index in "${!experiments[@]}"; do
  experiment="${experiments[$index]}"
  dataset="${datasets[$index]}"
  method_root="${TRAINING_ROOT}/${experiment}__a4_geometry_balanced__gate1_paired/${dataset}/view_skew/r0.9"
  for seed in "${seeds[@]}"; do
    parent="${method_root}/seed${seed}"
    if [[ ! -f "${parent}/checkpoint.pt" ]]; then
      echo "missing parent checkpoint: ${parent}/checkpoint.pt" >&2
      exit 2
    fi
    "${PYTHON_BIN}" -B tools/evaluate_checkpoint.py \
      --run-dir "${parent}" \
      --evaluation-ablation configs/ablations/a5_adaptive_reliability.json \
      --output-root "${EVALUATION_ROOT}" \
      --device cpu \
      --run-tag gate2_adaptive
  done
  collector_roots+=("${method_root}")
done

"${PYTHON_BIN}" -B tools/collect_results.py \
  "${collector_roots[@]}" "${EVALUATION_ROOT}" \
  --output-dir "${EVALUATION_ROOT}/gate2_pilot_summary" \
  --baseline a4_geometry_balanced \
  --expected-grid configs/grids/gate2_pilot.json \
  --strict
