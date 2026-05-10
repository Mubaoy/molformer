#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export CHEMVL_DATA_ROOT="${CHEMVL_DATA_ROOT:-${REPO_ROOT}/chemvl-data}"
export MOLFORMER_REPO="${MOLFORMER_REPO:-${REPO_ROOT}}"
export PYTHON="${PYTHON:-python}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
EXP_NAME="${EXP_NAME:-molformer_under_chemvl}"
SIDER_BATCH_SIZE="${SIDER_BATCH_SIZE:-16}"

date
echo "Resume after SIDER OOM"
echo "CHEMVL_DATA_ROOT=${CHEMVL_DATA_ROOT}"
echo "MOLFORMER_REPO=${MOLFORMER_REPO}"
echo "PYTHON=${PYTHON}"
echo "SIDER_BATCH_SIZE=${SIDER_BATCH_SIZE}"

"$PYTHON" scripts/chemvl_protocol/batch_run.py \
  --base-config configs/chemvl_protocol/molformer_moleculenet_classification_scaffold.external.json \
  --datasets sider \
  --runseed-start 1 \
  --runseed-end 3 \
  --exp-name "$EXP_NAME" \
  --batch-size "$SIDER_BATCH_SIZE"

"$PYTHON" scripts/chemvl_protocol/batch_run.py \
  --base-config configs/chemvl_protocol/molformer_moleculenet_classification_random-scaffold.external.json \
  --datasets sider \
  --runseed-start 1 \
  --runseed-end 3 \
  --exp-name "$EXP_NAME" \
  --batch-size "$SIDER_BATCH_SIZE"

bash scripts/chemvl_protocol/run_moleculenet_scaffold.sh
bash scripts/chemvl_protocol/run_moleculenet_random_scaffold.sh
bash scripts/chemvl_protocol/analyze.sh || echo "Final analyze step failed; training outputs remain under results."

date
