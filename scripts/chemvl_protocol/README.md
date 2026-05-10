# MoLFormer under ChemVL splits

This directory runs MoLFormer as an external baseline while keeping ChemVL's
MoleculeNet split and metric protocol. It does not import ChemVL's full
training stack; the runner embeds the same `scaffold` and `random_scaffold`
split logic used by ChemVL.

## Scope

- Table A: MoleculeNet 10 datasets, `dataset.split == "scaffold"`.
- Table B: MoleculeNet 10 datasets, `dataset.split == "random_scaffold"`.
- Each `(method, task, setting)` uses `runseed` 1, 2, and 3 by default.
- Metrics are ChemVL-compatible: classification `ROCAUC`; regression `MAE` for
  `qm7`, `RMSE` for `esol`, `freesolv`, and `lipo`.

MoleculeACE (Table C) is intentionally not implemented here yet; it needs a
separate loader for the 30 MoleculeACE tasks and the MolMCL split protocol.

## Required layout

Install the protocol-specific dependencies in the Python environment used for
these scripts:

```bash
pip install -r scripts/chemvl_protocol/requirements.txt
```

The RDKit pin is intentional. ChemVL's MoleculeNet scaffold split depends on
RDKit Murcko scaffold generation, so final reproduction runs should use
`rdkit-pypi==2022.9.5`.

Set the two roots before running:

```bash
export CHEMVL_DATA_ROOT=/path/to/chemvl-data
export MOLFORMER_REPO=/path/to/molformer
```

For this MoLFormer adapter, ChemVL data must at least contain the processed
CSV files:

```text
${CHEMVL_DATA_ROOT}/finetuning_datasets/MPP/classification/<task>/processed/<task>_processed_ac.csv
${CHEMVL_DATA_ROOT}/finetuning_datasets/MPP/regression/<task>/processed/<task>_processed_ac.csv
```

ChemVL's own image-based runs also require the corresponding
`processed/224/<index>.png` files. The Lipo dataset may be stored as
`lipophilicity`; the runner accepts the short dataset id `lipo`.

The MoLFormer checkpoint must exist at:

```text
${MOLFORMER_REPO}/data/Pretrained MoLFormer/checkpoints/N-Step-Checkpoint_3_30000.ckpt
```

## Run

Dry-run commands:

```bash
DRY_RUN=1 bash scripts/chemvl_protocol/run_moleculenet_scaffold.sh
DRY_RUN=1 bash scripts/chemvl_protocol/run_moleculenet_random_scaffold.sh
```

Actual runs:

```bash
bash scripts/chemvl_protocol/run_moleculenet_scaffold.sh
bash scripts/chemvl_protocol/run_moleculenet_random_scaffold.sh
```

Override seeds or Python executable:

```bash
PYTHON=/path/to/python RUNSEED_START=1 RUNSEED_END=3 \
  bash scripts/chemvl_protocol/run_moleculenet_scaffold.sh
```

## Aggregate

```bash
bash scripts/chemvl_protocol/analyze.sh
```

Outputs are written under:

```text
${CHEMVL_DATA_ROOT}/results/moleculenet/molformer_under_chemvl/
```

Each run writes ChemVL-style `config.json`, `train_val_test_history.csv`, and
`result.json`, so `scripts/chemvl_protocol/summarize_results.py` can produce
the per-dataset mean/std CSVs used for the deliverable tables.
