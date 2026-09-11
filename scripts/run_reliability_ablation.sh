#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${RBDP_PYTHON:-python}"
OUTPUT_DIR="${RBDP_OUTPUT_ROOT:-${PROJECT_DIR}/outputs}"

cd "${PROJECT_DIR}"
for ABLATION_FILE in configs/ablations/reliability/a4_*.json
do
  "${PYTHON_BIN}" -B tools/run_rbdp.py \
    --config configs/experiments/pilot_caltech_shift.json \
    --ablation "${ABLATION_FILE}" \
    --output-root "${OUTPUT_DIR}" \
    "$@"
done
