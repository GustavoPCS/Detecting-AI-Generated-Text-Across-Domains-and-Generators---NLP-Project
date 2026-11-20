"""Compare validation-selected decision rules on frozen held-out predictions.

This script uses only the saved predictions. It does NOT retrain a model, select
thresholds with test labels, or adapt to test score distributions. Each rule is
fitted independently for every (method, train_domain, train_generator) group.

Usage:
    python -m src.threshold_analysis --out results/threshold_analysis
    python -m src.threshold_analysis --n-boot 1000 --out results/threshold_analysis_ci

Inputs (by default):
    results/tfidf/predictions_val.csv and predictions_test.csv
    results/roberta_seed42/predictions_val.csv and predictions_test.csv

'1%-FPR' means the validation threshold targets <=1% FPR; it does not promise
<=1% on unseen domains. Check each held-out group's observed FPR explicitly.

'Midpoint' is the midpoint between the human-class and machine-class MEDIAN
validation scores. This is a newly specified exploratory rule, not part of the
original published/proposed experiments.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src import metrics as M
from src.evaluate import TRAIN_COLS, REQUIRED

RULES = ("max_macro_f1", "low_fpr_1pct", "class_median_midpoint")


def select_thresholds(validation: pd.DataFrame, target_fpr: float = 0.01) -> dict[str, float]:
    """Fit each threshold using validation labels and scores only."""
    y = validation["label"].to_numpy(dtype=int)
    s = validation["score"].to_numpy(dtype=float)
    if len(s) == 0 or not np.isfinite(s).all() or not {0, 1} <= set(y):
        raise ValueError("Validation data must be nonempty, finite and contain both classes")
    return {
        "max_macro_f1": float(M.threshold_max_macro_f1(y, s)),
        "low_fpr_1pct": float(M.threshold_for_fpr(y, s, target_fpr)),
        "class_median_midpoint": float((np.median(s[y == 0]) + np.median(s[y == 1])) / 2),
    }


def analyze(validation: pd.DataFrame, test: pd.DataFrame, target_fpr: float = 0.01,
            n_boot: int = 0, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Produce threshold table and held-out metrics for all training/testing cells."""
    required = set(REQUIRED + TRAIN_COLS + ["domain", "generator"])
    for name, df in (("validation", validation), ("test", test)):
        if not required <= set(df):
            raise ValueError(f"{name} missing columns: {sorted(required - set(df))}")

    thresholds, rows = [], []
    for key, val in validation.groupby(TRAIN_COLS, sort=True):
        identity = dict(zip(TRAIN_COLS, key))
        candidates = select_thresholds(val, target_fpr)
        matching = test
        for col, value in identity.items():
            matching = matching[matching[col] == value]
        if matching.empty:
            raise ValueError(f"No held-out test predictions for {identity}")
        val_y = val["label"].to_numpy(dtype=int)
        val_s = val["score"].to_numpy(dtype=float)
        for rule, threshold in candidates.items():
            val_metrics = M.evaluate_arrays(val_y, val_s, threshold, target_fpr)
            thresholds.append({**identity, "rule": rule, "threshold": threshold,
                               "n_val_human": int((val_y == 0).sum()),
                               "n_val_machine": int((val_y == 1).sum()),
                               "val_fpr": val_metrics["fpr"], "val_macro_f1": val_metrics["macro_f1"]})
            for (domain, generator), g in matching.groupby(["domain", "generator"], sort=True):
                met = M.evaluate_with_ci(g, threshold, target_fpr=target_fpr,
                                         n_boot=n_boot, seed=seed)
                rows.append({**identity, "rule": rule, "domain": domain,
                             "generator": generator, **met})
    per_cell = pd.DataFrame(rows)
    per_cell["condition"] = np.select([
        per_cell.domain.eq(per_cell.train_domain) & per_cell.generator.eq(per_cell.train_generator),
        per_cell.domain.ne(per_cell.train_domain) & per_cell.generator.eq(per_cell.train_generator),
        per_cell.domain.eq(per_cell.train_domain) & per_cell.generator.ne(per_cell.train_generator),
    ], ["in_domain", "cross_domain", "cross_generator"], default="combined_shift")
    return pd.DataFrame(thresholds), per_cell


def summarize(cells: pd.DataFrame) -> pd.DataFrame:
    """Unweighted mean across conditions, matching Update 1's macro-average convention."""
    fields = ["macro_f1", "auroc", "fpr", "tpr"]
    summary = (cells.groupby(["method", "condition", "rule"], as_index=False)
                 .agg(**{f: (f, "mean") for f in fields}, n_cells=("n", "size")))
    original = summary[summary.rule == "max_macro_f1"][["method", "condition", "macro_f1"]]
    original = original.rename(columns={"macro_f1": "original_macro_f1"})
    summary = summary.merge(original, on=["method", "condition"], how="left")
    summary["delta_macro_f1_vs_original"] = summary.macro_f1 - summary.original_macro_f1
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prediction-dirs", nargs="+", default=["results/tfidf", "results/roberta_seed42"])
    p.add_argument("--target-fpr", type=float, default=0.01)
    p.add_argument("--n-boot", type=int, default=0, help="0 for fast exploratory run; use 1000 for CI study")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results/threshold_analysis")
    args = p.parse_args(argv)
    if not 0 <= args.target_fpr <= 1 or args.n_boot < 0:
        p.error("--target-fpr must be in [0,1] and --n-boot >= 0")

    threshold_tables, cells_tables = [], []
    for directory in args.prediction_dirs:
        folder = Path(directory)
        val = pd.read_csv(folder / "predictions_val.csv")
        test = pd.read_csv(folder / "predictions_test.csv")
        th, cells = analyze(val, test, args.target_fpr, args.n_boot, args.seed)
        threshold_tables.append(th)
        cells_tables.append(cells)
    thresholds = pd.concat(threshold_tables, ignore_index=True)
    cells = pd.concat(cells_tables, ignore_index=True)
    result = summarize(cells)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    thresholds.to_csv(output / "thresholds.csv", index=False)
    cells.to_csv(output / "metrics_by_condition.csv", index=False)
    result.to_csv(output / "summary.csv", index=False)
    show = result[result.condition == "cross_domain"][["method", "rule", "macro_f1", "fpr", "tpr",
                                                          "delta_macro_f1_vs_original"]]
    print("Cross-domain G1 mean (12 off-diagonal pairs per method)")
    print(show.round(3).to_string(index=False))
    print(f"Saved output in {output}")
    return result


if __name__ == "__main__":
    main()
