#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export CHEMVL_DATA_ROOT="${CHEMVL_DATA_ROOT:-${REPO_ROOT}/chemvl-data}"
export MOLFORMER_REPO="${MOLFORMER_REPO:-${REPO_ROOT}}"
export PYTHON="${PYTHON:-python}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

date
echo "CHEMVL_DATA_ROOT=${CHEMVL_DATA_ROOT}"
echo "MOLFORMER_REPO=${MOLFORMER_REPO}"
echo "PYTHON=${PYTHON}"

bash scripts/chemvl_protocol/run_moleculenet_scaffold.sh
bash scripts/chemvl_protocol/run_moleculenet_random_scaffold.sh
bash scripts/chemvl_protocol/analyze.sh || echo "Final analyze step failed; training outputs remain under results."

date
