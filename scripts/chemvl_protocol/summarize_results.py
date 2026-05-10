#!/usr/bin/env python3
"""
Aggregate MoLFormer-under-ChemVL batch runs under::

    <root>/<basic.version>/<dataset>/<timestamp>/

Each run should have ``config.json`` + ``result.json`` with a scalar
``best_valid_on_test``.

Writes per-(version, dataset) mean ± std, macro averages over datasets, CSVs, and a bar chart
for the ChemVL-compatible result layout produced by ``finetune_molformer.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def extract_test_metric(
    result: Dict[str, Any],
    fallback_valid: bool,
) -> Tuple[Optional[str], Optional[float]]:
    avt = result.get("best_valid_on_test")
    if isinstance(avt, (int, float)) and not isinstance(avt, bool):
        return "best_valid_on_test", float(avt)
    if isinstance(avt, dict):
        for key in ("ROCAUC", "MAE", "RMSE", "R2", "r2", "rocauc", "mae", "rmse"):
            if key in avt and avt[key] is not None:
                name = key.lower()
                return name, float(avt[key])
    if fallback_valid:
        bv = result.get("best_valid")
        if isinstance(bv, (int, float)) and not isinstance(bv, bool):
            return "best_valid", float(bv)
        if isinstance(bv, dict):
            for key in ("ROCAUC", "MAE", "RMSE", "R2", "r2", "rocauc", "mae", "rmse"):
                if key in bv and bv[key] is not None:
                    return key.lower(), float(bv[key])
    return None, None


def discover_versions(root: Path) -> List[str]:
    out: List[str] = []
    for vdir in sorted(root.iterdir()):
        if not vdir.is_dir() or vdir.name.startswith("_") or vdir.name.startswith("."):
            continue
        ok = False
        for ddir in vdir.iterdir():
            if not ddir.is_dir():
                continue
            for tdir in ddir.iterdir():
                if tdir.is_dir() and (tdir / "result.json").is_file():
                    ok = True
                    break
            if ok:
                break
        if ok:
            out.append(vdir.name)
    return out


def discover_datasets(root: Path, versions: List[str]) -> List[str]:
    found: Set[str] = set()
    for v in versions:
        vpath = root / v
        if not vpath.is_dir():
            continue
        for dpath in vpath.iterdir():
            if not dpath.is_dir():
                continue
            for tdir in dpath.iterdir():
                if tdir.is_dir() and (tdir / "result.json").is_file():
                    found.add(dpath.name)
                    break
    return sorted(found)


def scan_runs(
    root: Path,
    versions: List[str],
    datasets: List[str],
    fallback_valid: bool,
    only_versions: Optional[Set[str]] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")

    for ver in versions:
        if only_versions is not None and ver not in only_versions:
            continue
        gdir = root / ver
        if not gdir.is_dir():
            continue
        for ds in datasets:
            dpath = gdir / ds
            if not dpath.is_dir():
                continue
            for run_dir in sorted(dpath.iterdir()):
                if not run_dir.is_dir():
                    continue
                result_path = run_dir / "result.json"
                if not result_path.is_file():
                    continue
                try:
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    print(f"Skip invalid JSON: {result_path}", file=sys.stderr)
                    continue
                metric_name, metric_val = extract_test_metric(result, fallback_valid)
                if metric_name is None or metric_val is None:
                    print(
                        f"Skip (no test metric{'/valid' if not fallback_valid else ''}): {result_path}",
                        file=sys.stderr,
                    )
                    continue

                runseed: Optional[int] = None
                cfg_path = run_dir / "config.json"
                if cfg_path.is_file():
                    try:
                        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                        rs = cfg.get("training", {}).get("runseed")
                        if rs is not None:
                            runseed = int(rs)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        pass

                rows.append(
                    {
                        "version": ver,
                        "dataset": ds,
                        "run_dir": str(run_dir),
                        "timestamp": run_dir.name,
                        "runseed": runseed,
                        "metric_name": metric_name,
                        "metric_value": metric_val,
                    }
                )

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df, df

    g = (
        df.groupby(["version", "dataset", "metric_name"], as_index=False)["metric_value"]
        .agg(mean="mean", std="std", n="count")
    )
    macro_parts = []
    for grp, sub in g.groupby("version"):
        names = sub["metric_name"].unique()
        if len(names) != 1:
            print(
                f"Warning: version {grp!r} has mixed metric names {list(names)}; check --fallback-valid.",
                file=sys.stderr,
            )
        macro_parts.append(
            {
                "version": grp,
                "macro_mean": sub["mean"].mean(),
                "macro_std_across_datasets": sub["mean"].std(ddof=0),
                "metric_name": names[0] if len(names) == 1 else "mixed",
                "n_datasets": len(sub),
            }
        )
    macro = pd.DataFrame(macro_parts).sort_values("version")
    return g.sort_values(["version", "dataset"]), macro


def plot_grouped_means(per_ds: pd.DataFrame, macro: pd.DataFrame, out_path: Path, title: str) -> None:
    if per_ds.empty:
        return

    mname = per_ds["metric_name"].iloc[0]

    groups = list(per_ds["version"].unique())
    datasets = list(per_ds["dataset"].unique())
    x = np.arange(len(groups))
    n_ds = len(datasets)
    width = min(0.8 / max(n_ds, 1), 0.2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax0 = axes[0]
    for i, ds in enumerate(datasets):
        means = []
        errs = []
        for grp in groups:
            row = per_ds[(per_ds["version"] == grp) & (per_ds["dataset"] == ds)]
            if row.empty:
                means.append(np.nan)
                errs.append(0.0)
            else:
                r = row.iloc[0]
                means.append(r["mean"])
                errs.append(0.0 if pd.isna(r["std"]) else r["std"])
        offset = (i - (n_ds - 1) / 2) * width
        ax0.bar(
            x + offset,
            means,
            width,
            yerr=errs,
            capsize=2,
            label=ds,
        )
    ax0.set_xticks(x)
    ax0.set_xticklabels(groups, rotation=25, ha="right")
    ax0.set_ylabel(f"Test {mname} (mean ± std over runs)")
    ax0.set_title("Per dataset")
    ax0.legend(title="dataset", fontsize=8)
    ax0.grid(axis="y", alpha=0.3)

    ax1 = axes[1]
    if not macro.empty:
        y = macro["macro_mean"].values
        yerr = macro["macro_std_across_datasets"].values
        yerr = np.nan_to_num(yerr, nan=0.0)
        ax1.bar(range(len(macro)), y, yerr=yerr, capsize=3, color="steelblue", alpha=0.85)
        ax1.set_xticks(range(len(macro)))
        ax1.set_xticklabels(macro["version"].tolist(), rotation=25, ha="right")
        ax1.set_ylabel(f"Mean of dataset means ({mname})")
        ax1.set_title("Macro average over datasets")
        ax1.grid(axis="y", alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote plot: {out_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Batch log root: <basic.log_dir_base>/<exp_name> (contains version folders).",
    )
    p.add_argument(
        "--versions",
        default=None,
        help="Comma-separated ``basic.version`` directory names. Default: auto-discover under --root.",
    )
    p.add_argument(
        "--version-prefix",
        default=None,
        help="If set, keep only version dirs whose name starts with this prefix (after --versions or discover).",
    )
    p.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated dataset folder names. Default: auto-discover from runs under selected versions.",
    )
    p.add_argument(
        "--versions-filter",
        default=None,
        help="Optional comma-separated subset of version names (same as full directory names).",
    )
    p.add_argument(
        "--out-stem",
        default="molformer_under_chemvl",
        help="Basename for <stem>_summary_by_dataset.csv, <stem>_summary_macro.csv, <stem>_summary.png.",
    )
    p.add_argument(
        "--out-csv-detail",
        type=Path,
        default=None,
    )
    p.add_argument(
        "--out-csv-macro",
        type=Path,
        default=None,
    )
    p.add_argument(
        "--out-plot",
        type=Path,
        default=None,
    )
    p.add_argument(
        "--fallback-valid",
        action="store_true",
        help="If best_valid_on_test is missing, use best_valid.",
    )
    args = p.parse_args()

    root = args.root.resolve()
    if args.versions:
        versions = [v.strip() for v in args.versions.split(",") if v.strip()]
    else:
        versions = discover_versions(root)
    if args.version_prefix:
        pref = args.version_prefix.strip()
        versions = [v for v in versions if v.startswith(pref)]
    only_v: Optional[Set[str]] = None
    if args.versions_filter:
        only_v = {x.strip() for x in args.versions_filter.split(",") if x.strip()}

    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    else:
        datasets = discover_datasets(root, versions)

    if not versions:
        print(f"No version directories with runs under {root}", file=sys.stderr)
        return 1
    if not datasets:
        print(f"No datasets discovered under versions {versions}", file=sys.stderr)
        return 1

    stem = args.out_stem.strip() or "molformer_under_chemvl"
    out_detail = args.out_csv_detail or (root / f"{stem}_summary_by_dataset.csv")
    out_macro = args.out_csv_macro or (root / f"{stem}_summary_macro.csv")
    out_plot = args.out_plot or (root / f"{stem}_summary.png")

    df = scan_runs(root, versions, datasets, args.fallback_valid, only_v)
    if df.empty:
        print(
            f"No scorable runs under {root} for versions={versions!r} datasets={datasets!r}. "
            "Each result.json needs numeric best_valid_on_test (or pass --fallback-valid).",
            file=sys.stderr,
        )
        return 1

    per_ds, macro = summarize(df)
    per_ds.to_csv(out_detail, index=False)
    macro.to_csv(out_macro, index=False)
    print(f"Wrote {out_detail}")
    print(f"Wrote {out_macro}")

    print("\n=== Per (version, dataset): mean ± std (n runs) ===")
    for ver in per_ds["version"].unique():
        print(f"\n[{ver}]")
        sub = per_ds[per_ds["version"] == ver]
        for _, r in sub.iterrows():
            std = r["std"]
            std_s = f"{std:.4f}" if pd.notna(std) else "nan"
            print(
                f"  {r['dataset']}: {r['metric_name']} = {r['mean']:.4f} ± {std_s} (n={int(r['n'])})"
            )

    print("\n=== Macro mean over datasets (mean of per-dataset means) ===")
    for _, r in macro.iterrows():
        s = r["macro_std_across_datasets"]
        s_s = f"{s:.4f}" if pd.notna(s) else "nan"
        print(
            f"  {r['version']}: {r['macro_mean']:.4f} (std across datasets: {s_s}, n_ds={int(r['n_datasets'])})"
        )

    plot_grouped_means(
        per_ds,
        macro,
        out_plot,
        title="MoLFormer under ChemVL protocol summary (by basic.version)",
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
