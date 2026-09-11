#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${RBDP_PYTHON:-python}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 2
fi

cd "${PROJECT_DIR}"
"${PYTHON_BIN}" -B -m unittest discover -s tests -v
"${PYTHON_BIN}" -B tools/gate0_audit.py "$@"
