"""Evaluation metrics for the detector comparison (proposal, Section 6).

Conventions
  * label 1 = machine-generated = POSITIVE class, label 0 = human.
  * score: higher = more likely machine. A text is predicted machine if score >= threshold.
  * A false positive is a HUMAN text labeled machine.
  * Thresholds are chosen on validation data and then FROZEN for the test set.
  * Bootstrap CIs resample SOURCE DOCUMENTS (groups), not individual rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CI_METRICS = ["macro_f1", "auroc", "accuracy", "tpr", "fpr"]


def _div(a, b, default=0.0):
    return float(a) / float(b) if b else default


# --------------------------------------------------------------------------- point metrics
def classification_metrics(y, pred):
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum()); tn = int(((y == 0) & (pred == 0)).sum())
    prec_m, rec_m = _div(tp, tp + fp), _div(tp, tp + fn)
    prec_h, rec_h = _div(tn, tn + fn), _div(tn, tn + fp)
    f1_m = _div(2 * prec_m * rec_m, prec_m + rec_m)
    f1_h = _div(2 * prec_h * rec_h, prec_h + rec_h)
    return {
        "accuracy": _div(tp + tn, len(y)),
        "precision_machine": prec_m, "recall_machine": rec_m, "f1_machine": f1_m,
        "precision_human": prec_h, "recall_human": rec_h, "f1_human": f1_h,
        "macro_f1": (f1_m + f1_h) / 2,
        "tpr": _div(tp, tp + fn, float("nan")),   # = recall_machine
        "fpr": _div(fp, fp + tn, float("nan")),   # human texts flagged as machine
    }


def auroc(y, s):
    """Rank-based AUROC (ties get average rank). NaN if only one class is present."""
    y, s = np.asarray(y).astype(int), np.asarray(s, dtype=float)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def tpr_at_fpr(y, s, target=0.01):
    """Best TPR achievable with FPR <= target on THIS data (descriptive benchmark metric)."""
    y, s = np.asarray(y).astype(int), np.asarray(s, dtype=float)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    ys, ss = y[order], s[order]
    tps, fps = np.cumsum(ys == 1), np.cumsum(ys == 0)
    last = np.r_[ss[1:] != ss[:-1], True]        # one point per distinct score
    tpr, fpr = tps[last] / n1, fps[last] / n0
    ok = fpr <= target
    return float(tpr[ok].max()) if ok.any() else 0.0


# --------------------------------------------------------------------------- thresholds
def threshold_for_fpr(y, s, target=0.01):
    """Smallest threshold whose FPR on these (validation) data is <= target."""
    y, s = np.asarray(y).astype(int), np.asarray(s, dtype=float)
    hs = np.sort(s[y == 0])
    if len(hs) == 0:
        raise ValueError("Need human examples to choose a low-FPR threshold")
    cands = np.unique(s)
    fprs = (len(hs) - np.searchsorted(hs, cands, side="left")) / len(hs)
    ok = cands[fprs <= target]
    return float(ok.min()) if len(ok) else float(np.nextafter(cands.max(), np.inf))


def threshold_max_macro_f1(y, s, max_candidates=500):
    """Threshold maximizing macro F1 on these (validation) data."""
    y, s = np.asarray(y).astype(int), np.asarray(s, dtype=float)
    cands = np.unique(s)
    if len(cands) > max_candidates:
        cands = np.unique(np.quantile(s, np.linspace(0, 1, max_candidates)))
    best_t, best_f = float(cands[0]), -1.0
    for t in cands:
        f = classification_metrics(y, (s >= t).astype(int))["macro_f1"]
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t


# --------------------------------------------------------------------------- full evaluation
def evaluate_arrays(y, s, threshold, target_fpr=0.01):
    y, s = np.asarray(y).astype(int), np.asarray(s, dtype=float)
    out = classification_metrics(y, (s >= threshold).astype(int))
    out["auroc"] = auroc(y, s)
    out["tpr_at_target_fpr"] = tpr_at_fpr(y, s, target_fpr)
    out["threshold"] = float(threshold)
    return out


def bootstrap_ci(y, s, groups, fn, n_boot=1000, seed=0, alpha=0.05):
    """Percentile CIs resampling GROUPS (source documents) with replacement.

    fn(y, s) -> dict of metrics. Returns {metric: (lo, hi)}.
    """
    y, s = np.asarray(y).astype(int), np.asarray(s, dtype=float)
    idx_by_group = list(pd.Series(np.arange(len(y))).groupby(np.asarray(groups)).apply(np.array))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(idx_by_group), len(idx_by_group))
        idx = np.concatenate([idx_by_group[i] for i in pick])
        draws.append(fn(y[idx], s[idx]))
    table = pd.DataFrame(draws)
    return {m: (float(np.nanpercentile(table[m], 100 * alpha / 2)),
                float(np.nanpercentile(table[m], 100 * (1 - alpha / 2)))) for m in table.columns}


def evaluate_with_ci(df, threshold, target_fpr=0.01, n_boot=1000, seed=0):
    """One row of results for a prediction frame (columns: label, score, source_id)."""
    y, s = df["label"].to_numpy(), df["score"].to_numpy(dtype=float)
    out = evaluate_arrays(y, s, threshold, target_fpr)
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    out.update(n=len(y), n_human=n0, n_machine=n1,
               fpr_resolution=_div(1, n0, float("nan")),            # smallest nonzero FPR step
               majority_baseline_acc=_div(max(n0, n1), len(y)))
    if n_boot > 0:
        cis = bootstrap_ci(y, s, df["source_id"].to_numpy(),
                           lambda a, b: evaluate_arrays(a, b, threshold, target_fpr), n_boot, seed)
        for m in CI_METRICS:
            out[f"{m}_lo"], out[f"{m}_hi"] = cis[m]
    return out


def add_gaps(table):
    """Generalization gap  delta = F1macro(A, A) - F1macro(A, B)  (proposal Eq. 1).

    Needs columns train_domain, domain, train_generator, generator, macro_f1.
    Gap is relative to the in-domain, same-generator row of the same method/training source.
    """
    need = {"train_domain", "domain", "train_generator", "generator", "macro_f1"}
    if not need <= set(table.columns):
        return table
    keys = [c for c in ["method", "train_domain", "train_generator"] if c in table.columns]
    base = table[(table["domain"] == table["train_domain"]) & (table["generator"] == table["train_generator"])]
    base = base[keys + ["macro_f1"]].rename(columns={"macro_f1": "macro_f1_in_domain"})
    out = table.merge(base, on=keys, how="left")
    out["gap_macro_f1"] = out["macro_f1_in_domain"] - out["macro_f1"]
    return out
