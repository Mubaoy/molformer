#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
EXP_NAME="${EXP_NAME:-molformer_under_chemvl}"
RUNSEED_START="${RUNSEED_START:-1}"
RUNSEED_END="${RUNSEED_END:-3}"
EXTRA_ARGS=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--dry-run)
fi
if [[ "${NO_SKIP:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--no-skip-existing)
fi

"$PYTHON" scripts/chemvl_protocol/batch_run.py \
  --base-config configs/chemvl_protocol/molformer_moleculenet_classification_random-scaffold.external.json \
  --dataset-list configs/chemvl_protocol/dataset_list_moleculenet_cls6.example.txt \
  --runseed-start "$RUNSEED_START" \
  --runseed-end "$RUNSEED_END" \
  --exp-name "$EXP_NAME" \
  "${EXTRA_ARGS[@]}"

"$PYTHON" scripts/chemvl_protocol/batch_run.py \
  --base-config configs/chemvl_protocol/molformer_moleculenet_regression_random-scaffold.external.json \
  --dataset-list configs/chemvl_protocol/dataset_list_moleculenet_reg4.example.txt \
  --runseed-start "$RUNSEED_START" \
  --runseed-end "$RUNSEED_END" \
  --exp-name "$EXP_NAME" \
  "${EXTRA_ARGS[@]}"
