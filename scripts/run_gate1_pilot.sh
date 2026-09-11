#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${RBDP_PYTHON:-python}"
OUTPUT_ROOT="${1:-${PROJECT_ROOT}/outputs}"

experiments=(
  "configs/experiments/pilot_caltech_shift.json"
  "configs/experiments/pilot_scene_shift.json"
  "configs/experiments/pilot_landuse_shift.json"
)
ablations=(
  "configs/ablations/a1_masked_dcp.json"
  "configs/ablations/a4_geometry_balanced.json"
)
seeds=(10 11 12 13 14)

cd "${PROJECT_ROOT}"
for experiment in "${experiments[@]}"; do
  for ablation in "${ablations[@]}"; do
    for seed in "${seeds[@]}"; do
      "${PYTHON_BIN}" -B tools/run_rbdp.py \
        --config "${experiment}" \
        --ablation "${ablation}" \
        --output-root "${OUTPUT_ROOT}" \
        --epochs 50 \
        --training-seed "${seed}" \
        --run-tag gate1_paired
    done
  done
done

"${PYTHON_BIN}" -B tools/collect_results.py "${OUTPUT_ROOT}" \
  --output-dir "${OUTPUT_ROOT}/gate1_pilot_summary" \
  --baseline a1_masked_dcp \
  --expected-grid configs/grids/gate1_pilot.json \
  --strict
