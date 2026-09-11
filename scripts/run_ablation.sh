#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${RBDP_PYTHON:-python}"
OUTPUT_DIR="${RBDP_OUTPUT_ROOT:-${PROJECT_DIR}/outputs}"

cd "${PROJECT_DIR}"
ABLATIONS=(
  configs/ablations/a0_dcp.json
  configs/ablations/a1_masked_dcp.json
  configs/ablations/a2_mean_risk.json
  configs/ablations/a3_risk_balanced.json
  configs/ablations/a4_geometry_balanced.json
)
for ABLATION_FILE in "${ABLATIONS[@]}"
do
  "${PYTHON_BIN}" -B tools/run_rbdp.py \
    --config configs/experiments/pilot_caltech_shift.json \
    --ablation "${ABLATION_FILE}" \
    --output-root "${OUTPUT_DIR}" \
    "$@"
done
