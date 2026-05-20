# MoLFormer under ChemVL Protocol

This repository is a research fork of IBM MoLFormer for running **MoLFormer as an external baseline under the ChemVL MoleculeNet protocol**.

The goal is not to change the original MoLFormer architecture. The goal is to fine-tune the MoLFormer pretrained checkpoint on the same datasets, splits, seeds, metrics, and output layout used by ChemVL, so MoLFormer can be compared fairly against ChemVL-style molecular representation methods.

## What This Fork Adds

- ChemVL-compatible MoleculeNet fine-tuning runner.
- `scaffold` and `random_scaffold` split logic matching ChemVL's protocol.
- Three-run protocol with `runseed = 1, 2, 3`.
- ChemVL-style outputs: `config.json`, `train_val_test_history.csv`, and `result.json`.
- Batch scripts for the two requested MoleculeNet tables.
- Result aggregation into mean +/- std CSV files.
- SIDER small-batch resume script for GPU memory constrained runs.

The protocol code lives in:

```text
scripts/chemvl_protocol/
configs/chemvl_protocol/
```

## Experiment Scope

This fork currently supports the MoleculeNet part of the comparison.

| Table | Split | Datasets |
|---|---|---|
| A | `scaffold` | BACE, BBBP, ClinTox, HIV, SIDER, Tox21, ESOL, FreeSolv, Lipo, QM7 |
| B | `random_scaffold` | BACE, BBBP, ClinTox, HIV, SIDER, Tox21, ESOL, FreeSolv, Lipo, QM7 |

MoleculeACE Table C is not implemented here yet. It needs a separate loader for the 30 MoleculeACE tasks and the MolMCL split protocol.

## Metrics

The metric convention follows the ChemVL main workflow:

| Task type | Dataset | Metric |
|---|---|---|
| Classification | BACE, BBBP, ClinTox, HIV, SIDER, Tox21 | ROC-AUC |
| Regression | QM7 | MAE |
| Regression | ESOL, FreeSolv, Lipo | RMSE |

For every `(method, task, split)` setting, the batch scripts run:

```text
runseed = 1, 2, 3
```

The data split seed is fixed to:

```text
seed = 1
```

This means the split is fixed across the three independent runs; `runseed` controls training randomness, not the data split.

## Environment

Install the ChemVL protocol dependencies in the Python environment used for these scripts:

```bash
pip install -r scripts/chemvl_protocol/requirements.txt
```

The RDKit version is important. ChemVL's `scaffold` and `random_scaffold` splits depend on RDKit Murcko scaffold generation, so final reproduction runs should use:

```text
rdkit-pypi==2022.9.5
```

The MoLFormer encoder additionally requires the original MoLFormer stack, including PyTorch and `pytorch-fast-transformers`.

## Required Data Layout

Set the two roots before running:

```bash
export CHEMVL_DATA_ROOT=/path/to/chemvl-data
export MOLFORMER_REPO=/path/to/this/molformer/repo
```

`CHEMVL_DATA_ROOT` must contain the processed ChemVL MoleculeNet CSV files:

```text
${CHEMVL_DATA_ROOT}/finetuning_datasets/MPP/classification/<task>/processed/<task>_processed_ac.csv
${CHEMVL_DATA_ROOT}/finetuning_datasets/MPP/regression/<task>/processed/<task>_processed_ac.csv
```

The Lipo dataset may be stored as `lipophilicity`; the runner accepts the short dataset id `lipo`.

`MOLFORMER_REPO` must contain the MoLFormer pretrained checkpoint:

```text
${MOLFORMER_REPO}/data/Pretrained MoLFormer/checkpoints/N-Step-Checkpoint_3_30000.ckpt
```

If `MOLFORMER_REPO` is not set, the scripts default to the current repository root.

## Run

Dry-run the commands first:

```bash
DRY_RUN=1 bash scripts/chemvl_protocol/run_moleculenet_scaffold.sh
DRY_RUN=1 bash scripts/chemvl_protocol/run_moleculenet_random_scaffold.sh
```

Run Table A:

```bash
bash scripts/chemvl_protocol/run_moleculenet_scaffold.sh
```

Run Table B:

```bash
bash scripts/chemvl_protocol/run_moleculenet_random_scaffold.sh
```

Run both sequentially in the background:

```bash
setsid -f bash scripts/chemvl_protocol/run_ab_background.sh \
  > "${CHEMVL_DATA_ROOT}/results/moleculenet/molformer_under_chemvl/run_ab.log" \
  2>&1 < /dev/null
```

Override the Python executable or runseed range if needed:

```bash
PYTHON=/path/to/python RUNSEED_START=1 RUNSEED_END=3 \
  bash scripts/chemvl_protocol/run_moleculenet_scaffold.sh
```

## SIDER OOM Recovery

SIDER has many classification targets and can exceed GPU memory with the default batch size on some machines. A small-batch recovery script is provided:

```bash
SIDER_BATCH_SIZE=16 bash scripts/chemvl_protocol/resume_after_sider_oom.sh
```

This script reruns SIDER with a smaller batch size and then resumes the normal batch workflow, skipping completed `result.json` runs.

## Aggregate Results

After runs finish:

```bash
bash scripts/chemvl_protocol/analyze.sh
```

Outputs are written under:

```text
${CHEMVL_DATA_ROOT}/results/moleculenet/molformer_under_chemvl/
```

Important files:

```text
molformer_under_chemvl_summary_by_dataset.csv
molformer_under_chemvl_summary_macro.csv
molformer_under_chemvl_summary.png
```

Each individual run is stored as:

```text
<result_root>/<version>/<dataset>/<timestamp>/
|-- config.json
|-- result.json
`-- train_val_test_history.csv
```

## Code Map

| File | Purpose |
|---|---|
| `scripts/chemvl_protocol/finetune_molformer.py` | Single-run fine-tuning entry point |
| `scripts/chemvl_protocol/batch_run.py` | Expands dataset x runseed jobs |
| `scripts/chemvl_protocol/summarize_results.py` | Aggregates result JSON files into mean +/- std tables |
| `scripts/chemvl_protocol/run_moleculenet_scaffold.sh` | Table A runner |
| `scripts/chemvl_protocol/run_moleculenet_random_scaffold.sh` | Table B runner |
| `scripts/chemvl_protocol/resume_after_sider_oom.sh` | Small-batch SIDER recovery |
| `configs/chemvl_protocol/*.json` | Base configs for classification/regression and split settings |

## Reproducibility Notes

- The split protocol is designed to match ChemVL's MoleculeNet `scaffold`/`random_scaffold` behavior.
- RDKit should be pinned to `rdkit-pypi==2022.9.5` for strict split reproducibility.
- `seed = 1` controls the split.
- `runseed = 1, 2, 3` controls training randomness.
- Classification labels use ChemVL's multitask convention with missing labels ignored.
- Regression uses ChemVL-compatible metric selection: QM7 uses MAE; ESOL, FreeSolv, and Lipo use RMSE.

## Original MoLFormer

This repository is based on IBM's MoLFormer code for:

> Large-scale chemical language representations capture molecular structure and properties

Original paper links:

- Nature Machine Intelligence: <https://rdcu.be/c12D0>
- arXiv: <https://arxiv.org/abs/2106.09553>

Original MoLFormer data and pretrained checkpoints are available from:

```text
https://ibm.box.com/v/MoLFormer-data
```

This fork keeps the original MoLFormer encoder/checkpoint interface and adds a ChemVL-compatible downstream evaluation protocol.

## Citation

Please cite the original MoLFormer work when using the pretrained MoLFormer model or architecture:

```bibtex
@article{10.1038/s42256-022-00580-7,
  year = {2022},
  title = {{Large-scale chemical language representations capture molecular structure and properties}},
  author = {Ross, Jerret and Belgodere, Brian and Chenthamarakshan, Vijil and Padhi, Inkit and Mroueh, Youssef and Das, Payel},
  journal = {Nature Machine Intelligence},
  doi = {10.1038/s42256-022-00580-7},
  pages = {1256--1264},
  number = {12},
  volume = {4}
}
```

```bibtex
@misc{https://doi.org/10.48550/arxiv.2106.09553,
  doi = {10.48550/ARXIV.2106.09553},
  url = {https://arxiv.org/abs/2106.09553},
  author = {Ross, Jerret and Belgodere, Brian and Chenthamarakshan, Vijil and Padhi, Inkit and Mroueh, Youssef and Das, Payel},
  title = {Large-Scale Chemical Language Representations Capture Molecular Structure and Properties},
  publisher = {arXiv},
  year = {2021}
}
```
