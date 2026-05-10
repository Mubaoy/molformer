#!/usr/bin/env python3
"""Batch driver for MoLFormer under ChemVL MoleculeNet splits.

Runs datasets x runseeds by composing a base JSON with a tiny temporary overlay.
Defaults match the requested reporting protocol: three independent runs
(``runseed`` 1--3).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def resolve_path(path: str) -> str:
    path = path.replace("{CHEMVL_REPO_ROOT}", str(_REPO_ROOT))
    path = path.replace(
        "{CHEMVL_DATA_ROOT}",
        os.environ.get("CHEMVL_DATA_ROOT", str(_REPO_ROOT / "chemvl-data")),
    )
    path = path.replace(
        "{MOLFORMER_REPO}",
        os.environ.get("MOLFORMER_REPO", str(_REPO_ROOT)),
    )
    path = os.path.expandvars(os.path.expanduser(path))
    if os.path.isabs(path):
        return path
    return str((_REPO_ROOT / path).resolve())


def load_dataset_list(path: str) -> List[str]:
    out: List[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                out.append(s)
    return out


def load_config(config_files: List[str]) -> Dict[str, Any]:
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


def validate_exp_name(name: str) -> str:
    s = name.strip()
    if not s:
        raise ValueError("--exp-name must be non-empty")
    if ".." in s or "/" in s or "\\" in s or os.path.isabs(s):
        raise ValueError("--exp-name must be a single path segment")
    return s


def output_has_result(cfg: Dict[str, Any]) -> bool:
    basic = cfg.get("basic") or {}
    dataset = (cfg.get("dataset") or {}).get("dataset")
    runseed = (cfg.get("training") or {}).get("runseed")
    log_base = str(basic.get("log_dir_base", "results"))
    exp_name = str(basic.get("exp_name", "")).strip()
    if exp_name:
        log_base = os.path.join(log_base, exp_name)
    log_base = resolve_path(log_base)
    if not os.path.isabs(log_base):
        log_base = str((_REPO_ROOT / log_base).resolve())
    ddir = Path(log_base) / str(basic.get("version")) / str(dataset)
    if not ddir.is_dir():
        return False
    for run_dir in ddir.iterdir():
        if not run_dir.is_dir():
            continue
        cfg_path = run_dir / "config.json"
        res_path = run_dir / "result.json"
        if not cfg_path.is_file() or not res_path.is_file():
            continue
        try:
            saved = json.loads(cfg_path.read_text(encoding="utf-8"))
            if saved.get("training", {}).get("runseed") != runseed:
                continue
            result = json.loads(res_path.read_text(encoding="utf-8"))
            if "best_valid_on_test" in result:
                return True
        except (OSError, json.JSONDecodeError, TypeError):
            continue
    return False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-config", required=True)
    p.add_argument("--dataset-list", default=None)
    p.add_argument("--datasets", default="", help="Comma-separated dataset ids if --dataset-list is not used.")
    p.add_argument("--runseed-start", type=int, default=1)
    p.add_argument("--runseed-end", type=int, default=3)
    p.add_argument("--exp-name", default="molformer_under_chemvl")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--finetune-script", default=str(_SCRIPT_DIR / "finetune_molformer.py"))
    p.add_argument("--batch-size", type=int, default=None, help="Optional per-batch override.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-skip-existing", action="store_true")
    p.add_argument("--capture-output", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    base_config = resolve_path(args.base_config)
    finetune_script = resolve_path(args.finetune_script)
    exp_name = validate_exp_name(args.exp_name)

    if args.dataset_list:
        datasets = load_dataset_list(resolve_path(args.dataset_list))
    else:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if not datasets:
        print("No datasets provided.", file=sys.stderr)
        return 1

    failures = 0
    for dataset in datasets:
        for runseed in range(args.runseed_start, args.runseed_end + 1):
            overlay = {
                "basic": {"exp_name": exp_name},
                "dataset": {"dataset": dataset},
                "training": {"runseed": runseed},
            }
            if args.batch_size is not None:
                overlay["training"]["batch_size"] = args.batch_size
            fd, overlay_path = tempfile.mkstemp(
                prefix=f"molformer_chemvl_{dataset}_{runseed}_", suffix=".json"
            )
            os.close(fd)
            try:
                with open(overlay_path, "w", encoding="utf-8") as f:
                    json.dump(overlay, f, indent=2)
                merged = load_config([base_config, overlay_path])
                if not args.no_skip_existing and output_has_result(merged):
                    print(f"Skip {dataset} / runseed={runseed} (existing result.json)")
                    continue

                cmd = [args.python, finetune_script, "--config", base_config, "--config", overlay_path]
                print("Executing:", " ".join(cmd))
                if args.dry_run:
                    continue
                result = subprocess.run(
                    cmd,
                    cwd=str(_REPO_ROOT),
                    capture_output=args.capture_output,
                    text=True,
                )
                if args.capture_output:
                    if result.stdout:
                        print(result.stdout)
                    if result.stderr:
                        print(result.stderr, file=sys.stderr)
                if result.returncode != 0:
                    failures += 1
                    print(f"Failed {dataset} / runseed={runseed}: exit {result.returncode}", file=sys.stderr)
            finally:
                try:
                    os.unlink(overlay_path)
                except OSError:
                    pass
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
