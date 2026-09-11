#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${RBDP_PYTHON:-python}"
OUTPUT_ROOT="${1:-${PROJECT_ROOT}/outputs/conflict_screen}"

experiments=(
  "configs/experiments/pilot_caltech_shift.json"
  "configs/experiments/pilot_scene_shift.json"
  "configs/experiments/pilot_landuse_shift.json"
)
ablations=(
  "configs/ablations/a1_masked_dcp.json"
  "configs/ablations/a6_conflict_safe.json"
)

cd "${PROJECT_ROOT}"
for experiment in "${experiments[@]}"; do
  for ablation in "${ablations[@]}"; do
    "${PYTHON_BIN}" -B tools/run_rbdp.py \
      --config "${experiment}" \
      --ablation "${ablation}" \
      --output-root "${OUTPUT_ROOT}" \
      --epochs 50 \
      --training-seed 10 \
      --device cpu \
      --run-tag conflict_screen
  done
done

"${PYTHON_BIN}" -B tools/collect_results.py "${OUTPUT_ROOT}" \
  --output-dir "${OUTPUT_ROOT}/summary" \
  --baseline a1_masked_dcp \
  --expected-grid configs/grids/conflict_screen.json \
  --strict
