"""Evaluate prediction files with frozen validation thresholds.

Prediction CSV columns (one row per text):
  required: source_id, label (0 human / 1 machine), score (higher = more machine)
  optional identifiers: method, train_domain, train_generator, domain, generator, split

Examples
  python -m src.evaluate --test results/tfidf_test.csv --val results/tfidf_val.csv --out results/tfidf_metrics.csv
  python -m src.evaluate --test results/x_test.csv --val results/x_val.csv --target-fpr 0.01 --threshold-mode fpr
  python -m src.evaluate --test results/x_test.csv --threshold 0.5          # fixed threshold, no val file
"""
import argparse

import pandas as pd

from src import metrics as M

ID_COLS = ["method", "train_domain", "train_generator", "domain", "generator", "split"]
TRAIN_COLS = ["method", "train_domain", "train_generator"]
REQUIRED = ["source_id", "label", "score"]


def read_predictions(path):
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")
    return df


def pick_thresholds(val, train_cols, mode="f1", target_fpr=0.01):
    """Threshold per training source, chosen on validation data only."""
    out = {}
    groups = val.groupby(train_cols) if train_cols else [((), val)]
    for key, g in groups:
        key = key if isinstance(key, tuple) else (key,)
        y, s = g["label"].to_numpy(), g["score"].to_numpy(dtype=float)
        out[key] = (M.threshold_for_fpr(y, s, target_fpr) if mode == "fpr"
                    else M.threshold_max_macro_f1(y, s))
    return out


def run(test, val=None, threshold=0.5, mode="f1", target_fpr=0.01, n_boot=1000, seed=0):
    id_cols = [c for c in ID_COLS if c in test.columns]
    train_cols = [c for c in TRAIN_COLS if c in test.columns]
    thr = pick_thresholds(val, train_cols, mode, target_fpr) if val is not None else None
    rows = []
    groups = test.groupby(id_cols, dropna=False) if id_cols else [((), test)]
    for key, g in groups:
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(id_cols, key))
        t = threshold
        if thr is not None:
            tkey = tuple(row[c] for c in train_cols)
            if tkey not in thr:
                raise ValueError(f"No validation rows for training source {tkey}")
            t = thr[tkey]
        row.update(M.evaluate_with_ci(g, t, target_fpr, n_boot, seed))
        rows.append(row)
    return M.add_gaps(pd.DataFrame(rows))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--test", required=True)
    p.add_argument("--val", default=None, help="validation predictions used to pick the threshold")
    p.add_argument("--threshold", type=float, default=0.5, help="fixed threshold if no --val")
    p.add_argument("--threshold-mode", choices=["f1", "fpr"], default="f1")
    p.add_argument("--target-fpr", type=float, default=0.01)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    val = read_predictions(a.val) if a.val else None
    res = run(read_predictions(a.test), val, a.threshold, a.threshold_mode, a.target_fpr, a.n_boot, a.seed)
    show = [c for c in ID_COLS + ["n", "threshold", "macro_f1", "macro_f1_lo", "macro_f1_hi", "auroc",
            "fpr", "tpr", "tpr_at_target_fpr", "gap_macro_f1"] if c in res.columns]
    print(res[show].round(3).to_string(index=False))
    if a.out:
        res.to_csv(a.out, index=False)
        print(f"\nWrote {a.out}")
    return res


if __name__ == "__main__":
    main()
