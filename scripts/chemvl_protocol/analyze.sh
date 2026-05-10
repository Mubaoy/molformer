#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT="${CHEMVL_DATA_ROOT:-${REPO_ROOT}/chemvl-data}"
LOG_DIR_BASE="${LOG_DIR_BASE:-${DATA_ROOT}/results/moleculenet}"
EXP_NAME="${EXP_NAME:-molformer_under_chemvl}"
RESULT_ROOT="${RESULT_ROOT:-${LOG_DIR_BASE}/${EXP_NAME}}"

PYTHON="${PYTHON:-python}"

"$PYTHON" scripts/chemvl_protocol/summarize_results.py \
  --root "$RESULT_ROOT" \
  --out-stem molformer_under_chemvl \
  "$@"
