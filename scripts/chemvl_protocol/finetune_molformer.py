#!/usr/bin/env python3
"""Fine-tune MoLFormer under ChemVL MoleculeNet split protocols.

This entry keeps the ChemVL experimental shell:

    - ChemVL-equivalent ``scaffold``/``random_scaffold`` split logic.
    - ChemVL metric keys and ``result.json`` layout.
    - MoLFormer encoder/checkpoint for the model under comparison.

The script intentionally does not depend on ChemVL's training stack or on
MoLFormer's Lightning finetune entrypoints, so it can run as a MoLFormer
baseline while preserving ChemVL's split protocol.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


CLASSIFICATION_COLUMNS = {
    "bbbp": ("smiles", ["p_np"]),
    "bace": ("mol", ["Class"]),
    "clintox": ("smiles", ["FDA_APPROVED", "CT_TOX"]),
    "hiv": ("smiles", ["HIV_active"]),
    "tox21": (
        "smiles",
        [
            "NR-AR",
            "NR-AR-LBD",
            "NR-AhR",
            "NR-Aromatase",
            "NR-ER",
            "NR-ER-LBD",
            "NR-PPAR-gamma",
            "SR-ARE",
            "SR-ATAD5",
            "SR-HSE",
            "SR-MMP",
            "SR-p53",
        ],
    ),
}


def load_config(config_files: Sequence[str]) -> Dict[str, Any]:
    def update_config(base_config: Dict[str, Any], new_config: Dict[str, Any]) -> None:
        for key, value in new_config.items():
            if isinstance(value, dict) and isinstance(base_config.get(key), dict):
                update_config(base_config[key], value)
            else:
                base_config[key] = value

    config: Dict[str, Any] = {}
    for config_file in config_files:
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file {config_file} does not exist.")
        with open(config_file, encoding="utf-8") as f:
            update_config(config, json.load(f))
    return config


def moleculenet_search_roots(dataroot: str) -> List[str]:
    root = os.path.abspath(os.path.expanduser(dataroot.strip()))
    bases = [root]
    if os.path.basename(root.rstrip(os.sep)).lower() == "moleculenet":
        parent = os.path.dirname(root.rstrip(os.sep))
        if parent:
            bases.append(parent)
    seen: set = set()
    out: List[str] = []
    for base in bases:
        if base not in seen:
            seen.add(base)
            out.append(base)
    return out


def _mpp_subdir(cfg: Dict[str, Any]) -> str:
    return str((cfg.get("dataset") or {}).get("moleculenet_mpp_subdir") or "MPP").strip().strip("/") or "MPP"


def _split_search_order(cfg: Dict[str, Any]) -> Tuple[str, str]:
    task_type = str((cfg.get("dataset") or {}).get("task_type", "classification")).lower()
    if task_type == "regression":
        return "regression", "classification"
    return "classification", "regression"


def _resolve_under_repo(repo: Path, path: str) -> str:
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return expanded
    return str((repo / expanded).resolve())


def _processed_ac_path(base: str, mpp: str, split: str, stem: str) -> str:
    return os.path.join(base, mpp, split, stem, "processed", f"{stem}_processed_ac.csv")


def resolve_moleculenet_csv(cfg: Dict[str, Any], repo: Path) -> str:
    ds = cfg.get("dataset") or {}
    name = str(ds.get("dataset", "")).strip().lower()
    if not name:
        raise ValueError("dataset.dataset is empty.")
    root = _resolve_under_repo(repo, str(ds.get("dataroot", "")).strip())
    mpp = _mpp_subdir(cfg)
    first, second = _split_search_order(cfg)

    stems = [name]
    if name == "lipo":
        stems.append("lipophilicity")
    elif name == "lipophilicity":
        stems.append("lipo")

    tried: List[str] = []
    for stem in stems:
        for base in moleculenet_search_roots(root):
            for split in (first, second):
                candidate = _processed_ac_path(base, mpp, split, stem)
                tried.append(candidate)
                if os.path.isfile(candidate):
                    return candidate

        for base in moleculenet_search_roots(root):
            for candidate in (
                os.path.join(base, stem, f"{stem}.csv"),
                os.path.join(base, f"{stem}.csv"),
                os.path.join(base, stem.upper(), f"{stem}.csv"),
                os.path.join(base, stem.upper(), f"{stem.upper()}.csv"),
            ):
                tried.append(candidate)
                if os.path.isfile(candidate):
                    return candidate
    raise FileNotFoundError(f"MoleculeNet data CSV not found for {name!r}. Tried (first ~12): {tried[:12]!r}")


def generate_scaffold(smiles: str, include_chirality: bool = True) -> str:
    from rdkit.Chem.Scaffolds import MurckoScaffold

    return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=include_chirality)


def scaffold_split_train_val_test(
    index: Sequence[int],
    smiles_list: Sequence[str],
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    frac_test: float = 0.1,
    include_chirality: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    index_arr = np.array(index)
    all_scaffolds: Dict[str, List[int]] = {}
    for i, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles, include_chirality=include_chirality)
        all_scaffolds.setdefault(scaffold, []).append(i)

    all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
    all_scaffold_sets = [
        scaffold_set
        for (_scaffold, scaffold_set) in sorted(
            all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True
        )
    ]

    train_cutoff = frac_train * len(smiles_list)
    valid_cutoff = (frac_train + frac_valid) * len(smiles_list)
    train_idx: List[int] = []
    valid_idx: List[int] = []
    test_idx: List[int] = []
    for scaffold_set in all_scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    if set(train_idx).intersection(set(valid_idx)) or set(test_idx).intersection(set(valid_idx)):
        raise RuntimeError("Overlapping scaffold split indices.")
    return index_arr[train_idx], index_arr[valid_idx], index_arr[test_idx]


def random_scaffold_split_train_val_test(
    index: Sequence[int],
    smiles_list: Sequence[str],
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    frac_test: float = 0.1,
    seed: int = 42,
    include_chirality: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from collections import defaultdict

    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    index_arr = np.array(index)
    rng = np.random.RandomState(seed)
    scaffolds: Dict[str, List[int]] = defaultdict(list)
    for ind, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles, include_chirality=include_chirality)
        scaffolds[scaffold].append(ind)

    scaffold_sets = rng.permutation(np.array(list(scaffolds.values()), dtype=object))
    n_total_valid = int(np.floor(frac_valid * len(index_arr)))
    n_total_test = int(np.floor(frac_test * len(index_arr)))
    train_idx: List[int] = []
    valid_idx: List[int] = []
    test_idx: List[int] = []
    for scaffold_set in scaffold_sets:
        if len(valid_idx) + len(scaffold_set) <= n_total_valid:
            valid_idx.extend(scaffold_set)
        elif len(test_idx) + len(scaffold_set) <= n_total_test:
            test_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    if set(train_idx).intersection(set(valid_idx)) or set(test_idx).intersection(set(valid_idx)):
        raise RuntimeError("Overlapping random scaffold split indices.")
    return index_arr[train_idx], index_arr[valid_idx], index_arr[test_idx]


def get_split(cfg: Dict[str, Any], names: Sequence[str], labels: np.ndarray, smiles: Sequence[str]):
    split = cfg["dataset"]["split"]
    seed = int(cfg["training"].get("seed", cfg["training"].get("runseed", 1)))
    chirality = bool(cfg["dataset"].get("chirality", True))
    indices = list(range(0, len(names)))
    if split == "scaffold":
        return scaffold_split_train_val_test(indices, smiles, include_chirality=chirality)
    if split == "random_scaffold":
        return random_scaffold_split_train_val_test(indices, smiles, seed=seed, include_chirality=chirality)
    raise ValueError(f"This MoLFormer baseline expects ChemVL scaffold/random_scaffold split, got {split!r}.")


def metric(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, empty: int = -1) -> Dict[str, float]:
    y_true = np.array(y_true).flatten()
    y_prob = np.array(y_prob).flatten()
    flag = y_true != empty
    return {"ROCAUC": metrics.roc_auc_score(y_true[flag], y_prob[flag])}


def metric_multitask(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, num_tasks: int, empty: int = -1
) -> Dict[str, Any]:
    result_list_dict_each_task: List[Optional[Dict[str, float]]] = []
    cur_num_tasks = 0
    mean_performance: Dict[str, Any] = {"ROCAUC": 0.0}
    for i in range(num_tasks):
        flag = y_true[:, i] != empty
        if len(set(y_true[flag, i].flatten())) == 1:
            result_list_dict_each_task.append(None)
            continue
        task_result = metric(y_true[flag, i].flatten(), y_pred[flag, i].flatten(), y_prob[flag, i].flatten())
        result_list_dict_each_task.append(task_result)
        cur_num_tasks += 1
    if cur_num_tasks == 0:
        mean_performance["ROCAUC"] = float("nan")
    else:
        mean_performance["ROCAUC"] = sum(
            r["ROCAUC"] for r in result_list_dict_each_task if r is not None
        ) / cur_num_tasks
    mean_performance["result_list_dict_each_task"] = result_list_dict_each_task
    if cur_num_tasks < num_tasks:
        mean_performance["some_target_missing"] = f"{1 - float(cur_num_tasks) / num_tasks:.2f} [{cur_num_tasks}/{num_tasks}]"
    return mean_performance


def metric_reg(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    mse = metrics.mean_squared_error(y_true, y_pred)
    return {
        "MAE": metrics.mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": mse ** 0.5,
        "R2": metrics.r2_score(y_true, y_pred),
    }


def metric_reg_multitask(y_true: np.ndarray, y_pred: np.ndarray, num_tasks: int) -> Dict[str, Any]:
    result_list_dict_each_task = [metric_reg(y_true[:, i].flatten(), y_pred[:, i].flatten()) for i in range(num_tasks)]
    mean_performance: Dict[str, Any] = {}
    for key in ("MAE", "MSE", "RMSE", "R2"):
        mean_performance[key] = sum(r[key] for r in result_list_dict_each_task) / num_tasks
    mean_performance["result_list_dict_each_task"] = result_list_dict_each_task
    return mean_performance


def get_metric(cfg: Dict[str, Any], labels_train: Optional[np.ndarray] = None) -> Tuple[str, str, float, None, None]:
    task_type = cfg["dataset"]["task_type"]
    if task_type == "classification":
        return "rocauc", "max", -np.inf, None, None
    if task_type == "regression":
        if cfg["dataset"]["dataset"] in ["qm7", "qm8", "qm9"]:
            return "mae", "min", np.inf, None, None
        return "rmse", "min", np.inf, None, None
    raise ValueError(f"{task_type!r} is not supported")


def is_left_better_right(left: float, right: float, standard: str) -> bool:
    if standard == "max":
        return left > right
    if standard == "min":
        return left < right
    raise ValueError(f"Unknown comparison standard: {standard!r}")

REGRESSION_COLUMNS = {
    "esol": (
        ["smiles", "SMILES", "compound_iso_smiles"],
        ["measured log solubility in mols per litre", "measured log solubility in mols per litre ", "y", "exp"],
    ),
    "freesolv": (["smiles", "SMILES"], ["expt", "y", "exp"]),
    "lipo": (["smiles", "SMILES"], ["exp", "y", "value", "label", "lipo_exp"]),
    "lipophilicity": (["smiles", "SMILES"], ["exp", "y", "value", "label", "lipo_exp"]),
    "qm7": (
        ["smiles", "SMILES", "molecule", "mol"],
        ["u0_atom", "U0_atom", "y", "exp", "internal_energy_at_0K"],
    ),
}


def _default_data_root() -> Path:
    raw = os.environ.get("CHEMVL_DATA_ROOT", "").strip()
    if raw:
        return Path(os.path.abspath(os.path.expanduser(raw)))
    return _REPO_ROOT / "chemvl-data"


def _default_molformer_root() -> Path:
    raw = os.environ.get("MOLFORMER_REPO", "").strip()
    if raw:
        return Path(os.path.abspath(os.path.expanduser(raw)))
    return _REPO_ROOT


def _expand_string(value: str) -> str:
    replacements = {
        "CHEMVL_REPO_ROOT": str(_REPO_ROOT),
        "CHEMVL_DATA_ROOT": str(_default_data_root()),
        "MOLFORMER_REPO": str(_default_molformer_root()),
    }
    out = value
    for key, repl in replacements.items():
        out = out.replace("{" + key + "}", repl)
    out = os.path.expandvars(os.path.expanduser(out))
    return out


def expand_paths(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: expand_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_paths(v) for v in obj]
    if isinstance(obj, str):
        return _expand_string(obj)
    return obj


def apply_exp_name(cfg: Dict[str, Any]) -> None:
    exp_name = str((cfg.get("basic") or {}).get("exp_name") or "").strip()
    if not exp_name:
        return
    if ".." in exp_name or "/" in exp_name or "\\" in exp_name or os.path.isabs(exp_name):
        raise ValueError(f"Invalid basic.exp_name={exp_name!r}; use a single path segment.")
    basic = cfg.setdefault("basic", {})
    basic["log_dir_base"] = os.path.normpath(os.path.join(str(basic.get("log_dir_base", "results")), exp_name))


def timestamp() -> str:
    return datetime.fromtimestamp(time.time()).strftime("%Y_%m_%d_%H_%M")


def setup_output_dir(cfg: Dict[str, Any]) -> str:
    basic = cfg.setdefault("basic", {})
    basic["timestamp"] = timestamp()
    log_dir = os.path.join(
        basic.get("log_dir_base", "results"),
        basic.get("version", "molformer_under_chemvl"),
        cfg["dataset"]["dataset"],
        basic["timestamp"],
    )
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4)
    return log_dir


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates!r} found in columns {list(df.columns)!r}")


def _parse_processed_ac(path: str, task_type: str) -> Tuple[List[str], np.ndarray]:
    df = pd.read_csv(path)
    smiles = df["smiles"].astype(str).to_list()
    labels = np.array(df["label"].apply(lambda x: str(x).split()).to_list())
    if task_type == "classification":
        labels = labels.astype(np.int64)
    else:
        labels = labels.astype(np.float32)
    return smiles, labels


def _load_raw_moleculenet(path: str, dataset: str, task_type: str) -> Tuple[List[str], np.ndarray]:
    df = pd.read_csv(path)
    name = dataset.lower()
    if task_type == "classification":
        if name == "sider":
            smiles_col = "smiles"
            label_cols = [c for c in df.columns if c != smiles_col]
        else:
            smiles_col, label_cols = CLASSIFICATION_COLUMNS[name]
            if smiles_col not in df.columns and name == "bace" and "smiles" in df.columns:
                smiles_col = "smiles"
        smiles = df[smiles_col].astype(str).to_list()
        labels = df[label_cols].fillna(-1).values.astype(np.int64)
        return smiles, labels

    smiles_cols, y_cols = REGRESSION_COLUMNS[name]
    smiles_col = _first_existing_column(df, smiles_cols)
    y_col = _first_existing_column(df, y_cols)
    smiles = df[smiles_col].astype(str).to_list()
    labels = df[[y_col]].values.astype(np.float32)
    return smiles, labels


def load_moleculenet_for_chemvl_split(cfg: Dict[str, Any]) -> Tuple[List[str], np.ndarray, str]:
    csv_path = resolve_moleculenet_csv(cfg, _REPO_ROOT)
    task_type = cfg["dataset"]["task_type"]
    dataset = cfg["dataset"]["dataset"]
    if csv_path.lower().endswith("_processed_ac.csv"):
        smiles, labels = _parse_processed_ac(csv_path, task_type)
    else:
        smiles, labels = _load_raw_moleculenet(csv_path, dataset, task_type)
    return smiles, labels, csv_path


def canonicalize_smiles(smi: str) -> Optional[str]:
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    except Exception:
        return None


class SmilesDataset(Dataset):
    def __init__(self, smiles: Sequence[str], labels: np.ndarray):
        self.smiles = list(smiles)
        self.labels = np.asarray(labels)

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> Tuple[str, np.ndarray]:
        return self.smiles[idx], self.labels[idx]


def build_subset(smiles: Sequence[str], labels: np.ndarray, indices: Sequence[int]) -> SmilesDataset:
    out_smiles: List[str] = []
    out_labels: List[np.ndarray] = []
    dropped = 0
    for i in indices:
        c = canonicalize_smiles(smiles[int(i)])
        if c is None:
            dropped += 1
            continue
        out_smiles.append(c)
        out_labels.append(labels[int(i)])
    if dropped:
        print(f"Dropped {dropped} invalid SMILES after split canonicalization.")
    return SmilesDataset(out_smiles, np.asarray(out_labels))


def import_molformer_modules(molformer_root: Path):
    finetune_dir = molformer_root / "finetune"
    if not finetune_dir.is_dir():
        raise FileNotFoundError(f"MoLFormer finetune dir not found: {finetune_dir}")
    if str(finetune_dir) not in sys.path:
        sys.path.insert(0, str(finetune_dir))

    from fast_transformers.feature_maps import GeneralizedRandomFeatures
    from fast_transformers.masking import LengthMask
    from rotate_attention.rotate_builder import RotateEncoderBuilder
    from tokenizer.tokenizer import MolTranBertTokenizer

    return GeneralizedRandomFeatures, LengthMask, RotateEncoderBuilder, MolTranBertTokenizer


class PredictionHead(nn.Module):
    def __init__(self, n_embd: int, out_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, n_embd)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(n_embd, n_embd)
        self.dropout2 = nn.Dropout(dropout)
        self.final = nn.Linear(n_embd, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = F.gelu(self.dropout1(self.fc1(x)))
        x1 = x1 + x
        x2 = F.gelu(self.dropout2(self.fc2(x1)))
        return self.final(x2 + x1)


class MolFormerPropertyModel(nn.Module):
    def __init__(self, cfg: Dict[str, Any], tokenizer: Any, molformer_modules: Tuple[Any, Any, Any, Any]):
        super().__init__()
        GeneralizedRandomFeatures, LengthMask, RotateEncoderBuilder, _Tokenizer = molformer_modules
        from functools import partial

        model_cfg = cfg["model"]
        n_vocab = len(tokenizer.vocab)
        n_embd = int(model_cfg.get("n_embd", 768))
        n_head = int(model_cfg.get("n_head", 12))
        n_layer = int(model_cfg.get("n_layer", 12))
        num_feats = int(model_cfg.get("num_feats", 32))
        self.length_mask_cls = LengthMask

        builder = RotateEncoderBuilder.from_kwargs(
            n_layers=n_layer,
            n_heads=n_head,
            query_dimensions=n_embd // n_head,
            value_dimensions=n_embd // n_head,
            feed_forward_dimensions=n_embd,
            attention_type="linear",
            feature_map=partial(GeneralizedRandomFeatures, n_dims=num_feats),
            activation="gelu",
        )
        self.pos_emb = None
        self.tok_emb = nn.Embedding(n_vocab, n_embd)
        self.drop = nn.Dropout(float(model_cfg.get("d_dropout", 0.1)))
        self.blocks = builder.get()
        self.lang_model = self.lm_layer(n_embd, n_vocab)
        self.net = PredictionHead(n_embd, int(cfg["dataset"]["num_tasks"]), float(model_cfg.get("dropout", 0.1)))

    class lm_layer(nn.Module):
        def __init__(self, n_embd: int, n_vocab: int):
            super().__init__()
            self.embed = nn.Linear(n_embd, n_embd)
            self.ln_f = nn.LayerNorm(n_embd)
            self.head = nn.Linear(n_embd, n_vocab, bias=False)

        def forward(self, tensor: torch.Tensor) -> torch.Tensor:
            tensor = F.gelu(self.embed(tensor))
            tensor = self.ln_f(tensor)
            return self.head(tensor)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.tok_emb(input_ids))
        x = self.blocks(x, length_mask=self.length_mask_cls(attention_mask.sum(-1)))
        mask = attention_mask.unsqueeze(-1).expand_as(x).float()
        return torch.sum(x * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.net(self.encode(input_ids, attention_mask))


def torch_load(path: str, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_pretrained(model: nn.Module, ckpt_path: str) -> None:
    if not ckpt_path:
        print("No MoLFormer checkpoint configured; training from random initialization.")
        return
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"MoLFormer checkpoint not found: {ckpt_path}")
    ckpt = torch_load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(
        f"Loaded MoLFormer checkpoint: {ckpt_path} "
        f"(missing={len(missing)}, unexpected={len(unexpected)})"
    )


def maybe_freeze_encoder(model: MolFormerPropertyModel, freeze: bool) -> None:
    if not freeze:
        return
    for module in (model.tok_emb, model.blocks, model.lang_model):
        for p in module.parameters():
            p.requires_grad = False


def make_collate(tokenizer: Any, task_type: str):
    label_dtype = torch.long if task_type == "classification" else torch.float32

    def collate(batch: Sequence[Tuple[str, np.ndarray]]):
        smiles = [b[0] for b in batch]
        labels = np.asarray([b[1] for b in batch])
        tokens = tokenizer.batch_encode_plus(smiles, padding=True, add_special_tokens=True)
        return (
            torch.tensor(tokens["input_ids"], dtype=torch.long),
            torch.tensor(tokens["attention_mask"], dtype=torch.long),
            torch.tensor(labels, dtype=label_dtype),
        )

    return collate


def classification_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labels = labels.float()
    valid = labels != -1
    if not torch.any(valid):
        return logits.sum() * 0.0
    targets = labels.clamp(min=0.0)
    loss_mat = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return loss_mat[valid].mean()


def regression_loss(logits: torch.Tensor, labels: torch.Tensor, loss_name: str) -> torch.Tensor:
    logits = logits.float()
    labels = labels.float()
    if loss_name.lower() == "mse":
        return F.mse_loss(logits, labels)
    if loss_name.lower() == "smooth_l1":
        return F.smooth_l1_loss(logits, labels)
    return F.l1_loss(logits, labels)


def train_epoch(
    model: MolFormerPropertyModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task_type: str,
    loss_name: str,
    max_grad_norm: Optional[float],
) -> float:
    model.train()
    losses: List[float] = []
    for step, (input_ids, attention_mask, labels) in enumerate(loader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids, attention_mask)
        if task_type == "classification":
            loss = classification_loss(logits, labels)
        else:
            loss = regression_loss(logits, labels, loss_name)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step == 0 or (step + 1) % 50 == 0:
            print(f"step: {step}, loss: {losses[-1]}")
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def evaluate(model: MolFormerPropertyModel, loader: DataLoader, device: torch.device, task_type: str) -> Dict[str, Any]:
    model.eval()
    y_true_parts: List[np.ndarray] = []
    score_parts: List[np.ndarray] = []
    for input_ids, attention_mask, labels in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        logits = model(input_ids, attention_mask)
        y_true_parts.append(labels.cpu().numpy())
        score_parts.append(logits.detach().cpu().numpy())
    y_true = np.concatenate(y_true_parts, axis=0)
    y_scores = np.concatenate(score_parts, axis=0)

    if task_type == "classification":
        y_prob = 1.0 / (1.0 + np.exp(-y_scores))
        y_pred = (y_prob >= 0.5).astype(np.int64)
        if y_true.shape[1] == 1:
            return metric(y_true, y_pred, y_prob, empty=-1)
        return dict(metric_multitask(y_true, y_pred, y_prob, y_true.shape[1], empty=-1))

    if y_true.shape[1] == 1:
        return metric_reg(y_true, y_scores)
    return dict(metric_reg_multitask(y_true, y_scores, y_true.shape[1]))


def build_optimizer(cfg: Dict[str, Any], model: nn.Module) -> torch.optim.Optimizer:
    tr = cfg["training"]
    params = [p for p in model.parameters() if p.requires_grad]
    lr = float(tr.get("lr", 3e-5))
    weight_decay = float(tr.get("weight_decay", 0.0))
    name = str(tr.get("optimizer", "AdamW")).lower()
    if name == "fusedlamb":
        try:
            from apex import optimizers

            return optimizers.FusedLAMB(params, lr=lr, betas=(0.9, 0.99), weight_decay=weight_decay)
        except Exception as exc:
            print(f"FusedLAMB unavailable ({exc}); falling back to AdamW.")
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def main(cfg: Dict[str, Any]) -> None:
    cfg = expand_paths(cfg)
    apply_exp_name(cfg)
    dataset_cfg = cfg.setdefault("dataset", {})
    if str(dataset_cfg.get("benchmark", "moleculenet")).lower() != "moleculenet":
        raise NotImplementedError("This MoLFormer adapter currently supports MoleculeNet only.")

    gpu = str((cfg.get("basic") or {}).get("gpu", "0"))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int((cfg.get("training") or {}).get("runseed", 1)))
    cfg.setdefault("training", {}).setdefault("weighted_CE", False)

    smiles, labels, csv_path = load_moleculenet_for_chemvl_split(cfg)
    labels = np.asarray(labels)
    dataset_cfg["num_tasks"] = int(labels.shape[1])
    cfg.setdefault("regression_scheduler", {})["labels_csv_path"] = csv_path
    print(f"Loaded {len(smiles)} molecules from {csv_path}")

    names = np.array([str(i) for i in range(len(smiles))], dtype=object)
    train_idx, val_idx, test_idx = get_split(cfg, names, labels, smiles)
    train_ds = build_subset(smiles, labels, train_idx)
    val_ds = build_subset(smiles, labels, val_idx)
    test_ds = build_subset(smiles, labels, test_idx)
    print(f"Split sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    log_dir = setup_output_dir(cfg)

    molformer_root = Path(cfg["model"].get("molformer_repo", str(_default_molformer_root()))).resolve()
    modules = import_molformer_modules(molformer_root)
    _GeneralizedRandomFeatures, _LengthMask, _RotateEncoderBuilder, MolTranBertTokenizer = modules
    tokenizer = MolTranBertTokenizer(str(molformer_root / "finetune" / "bert_vocab.txt"))

    model = MolFormerPropertyModel(cfg, tokenizer, modules)
    load_pretrained(model, str(cfg["model"].get("pretrained_ckpt", "")))
    maybe_freeze_encoder(model, bool(cfg["model"].get("freeze_encoder", False)))
    model.to(device)

    batch_size = int(cfg["training"].get("batch_size", 64))
    num_workers = int((cfg.get("basic") or {}).get("num_workers", 0))
    collate = make_collate(tokenizer, cfg["dataset"]["task_type"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)

    optimizer = build_optimizer(cfg, model)
    eval_metric, valid_select, min_value, _criterion, _weights = get_metric(cfg, labels[np.asarray(train_idx)])
    metric_key = eval_metric.upper()
    task_type = cfg["dataset"]["task_type"]
    loss_name = str(cfg["training"].get("loss", "l1"))
    max_grad_norm = cfg["training"].get("max_grad_norm")
    max_grad_norm = None if max_grad_norm is None else float(max_grad_norm)

    results: Dict[str, Any] = {
        "best_valid": min_value,
        "best_valid_epoch": 0,
        "best_train_loss": np.inf,
        "best_train_epoch": 0,
    }
    history_rows: List[Dict[str, Any]] = []
    epochs = int(cfg["training"].get("epochs", 20))
    start_epoch = int(cfg["training"].get("start_epoch", 0))

    for epoch in range(start_epoch, start_epoch + epochs):
        train_step_loss = train_epoch(model, train_loader, optimizer, device, task_type, loss_name, max_grad_norm)
        val_results = evaluate(model, val_loader, device, task_type)
        test_results = evaluate(model, test_loader, device, task_type)

        valid_result = float(val_results[metric_key])
        test_result = float(test_results[metric_key])
        row = {
            "epoch": epoch,
            "train_step_loss": train_step_loss,
            f"valid_{eval_metric}": valid_result,
            f"test_{eval_metric}": test_result,
        }
        history_rows.append(row)
        print(row)

        if is_left_better_right(train_step_loss, results["best_train_loss"], standard="min"):
            results["best_train_loss"] = train_step_loss
            results["best_train_on_test"] = test_result
            results["best_train_epoch"] = epoch

        if is_left_better_right(valid_result, results["best_valid"], standard=valid_select):
            results["best_valid"] = valid_result
            results["best_valid_on_test"] = test_result
            results["best_valid_epoch"] = epoch
            if bool((cfg.get("basic") or {}).get("save_finetune_ckpt", False)):
                ckpt_path = os.path.join(log_dir, "valid_best.pth")
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "result_dict": results}, ckpt_path)
                print(f"Checkpoint saved to {ckpt_path}")

    pd.DataFrame(history_rows).to_csv(os.path.join(log_dir, "train_val_test_history.csv"), index=False)
    with open(os.path.join(log_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", action="append", required=True, help="JSON config(s), later files override earlier.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(load_config(args.config))
