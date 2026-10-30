import numpy as np
import pandas as pd

from src import metrics as M
from src import evaluate as E


def close(a, b, tol=1e-9):
    return abs(a - b) < tol


def test_classification_metrics_hand_computed():
    y = [1, 1, 1, 0, 0, 0, 0, 0]
    p = [1, 1, 0, 1, 0, 0, 0, 0]          # tp=2 fn=1 fp=1 tn=4
    m = M.classification_metrics(y, p)
    assert close(m["accuracy"], 6 / 8)
    assert close(m["precision_machine"], 2 / 3) and close(m["recall_machine"], 2 / 3)
    assert close(m["precision_human"], 4 / 5) and close(m["recall_human"], 4 / 5)
    assert close(m["fpr"], 1 / 5) and close(m["tpr"], 2 / 3)
    assert close(m["macro_f1"], (2 / 3 + 4 / 5) / 2)


def test_auroc_perfect_inverted_random_and_ties():
    y = np.array([0, 0, 0, 1, 1, 1])
    assert close(M.auroc(y, [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]), 1.0)
    assert close(M.auroc(y, [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]), 0.0)
    assert close(M.auroc(y, [0.5] * 6), 0.5)                      # all ties
    rng = np.random.default_rng(0)
    big_y = rng.integers(0, 2, 4000)
    assert abs(M.auroc(big_y, rng.random(4000)) - 0.5) < 0.03
    assert np.isnan(M.auroc([1, 1], [0.2, 0.3]))                  # one class only


def test_auroc_matches_pairwise_definition():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 60)
    s = np.round(rng.random(60), 1)                               # many ties
    pos, neg = s[y == 1], s[y == 0]
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    assert close(M.auroc(y, s), wins / (len(pos) * len(neg)))


def test_tpr_at_fpr():
    y = np.array([0] * 100 + [1] * 10)
    s = np.r_[np.linspace(0, 0.5, 100), [0.4, 0.45, 0.6, 0.7, 0.8, 0.9, 0.95, 0.96, 0.97, 0.98]]
    # top human score is 0.5; at FPR<=1% (one human may be flagged) threshold sits above 0.495
    assert close(M.tpr_at_fpr(y, s, 0.01), 8 / 10)
    assert close(M.tpr_at_fpr(y, s, 0.0), 8 / 10)                 # still above top human score 0.5
    assert close(M.tpr_at_fpr(y, s, 1.0), 1.0)


def test_threshold_for_fpr_respects_target_on_validation():
    rng = np.random.default_rng(2)
    y = np.r_[np.zeros(500), np.ones(500)].astype(int)
    s = np.r_[rng.normal(0, 1, 500), rng.normal(2, 1, 500)]
    t = M.threshold_for_fpr(y, s, 0.05)
    assert ((s[y == 0] >= t).mean()) <= 0.05
    t_loose = M.threshold_for_fpr(y, s, 0.5)
    assert t_loose < t


def test_threshold_max_macro_f1_beats_default():
    rng = np.random.default_rng(3)
    y = np.r_[np.zeros(300), np.ones(300)].astype(int)
    s = np.r_[rng.normal(5, 1, 300), rng.normal(8, 1, 300)]      # default 0.5 would flag everything
    t = M.threshold_max_macro_f1(y, s)
    best = M.classification_metrics(y, (s >= t).astype(int))["macro_f1"]
    naive = M.classification_metrics(y, (s >= 0.5).astype(int))["macro_f1"]
    assert best > naive + 0.2


def test_bootstrap_ci_contains_estimate_and_is_deterministic():
    rng = np.random.default_rng(4)
    n = 300
    y = np.r_[np.zeros(n), np.ones(n)].astype(int)
    s = np.r_[rng.normal(0, 1, n), rng.normal(1.5, 1, n)]
    groups = np.r_[np.arange(n), np.arange(n)]
    fn = lambda a, b: M.evaluate_arrays(a, b, 0.75)
    ci1 = M.bootstrap_ci(y, s, groups, fn, n_boot=200, seed=7)
    ci2 = M.bootstrap_ci(y, s, groups, fn, n_boot=200, seed=7)
    assert ci1 == ci2
    point = fn(y, s)
    lo, hi = ci1["auroc"]
    assert lo < point["auroc"] < hi and hi - lo < 0.1


def test_bootstrap_resamples_groups_not_rows():
    # one source document that is wildly different: group resampling must give a wider CI
    rng = np.random.default_rng(5)
    n_docs, per_doc = 20, 25
    groups = np.repeat(np.arange(n_docs), per_doc)
    y = rng.integers(0, 2, n_docs * per_doc)
    s = y + rng.normal(0, 0.3, len(y))
    s[groups % 2 == 0] = rng.random((groups % 2 == 0).sum())      # half the docs are uninformative
    fn = lambda a, b: M.evaluate_arrays(a, b, 0.5)
    by_group = M.bootstrap_ci(y, s, groups, fn, n_boot=300, seed=1)["auroc"]
    by_row = M.bootstrap_ci(y, s, np.arange(len(y)), fn, n_boot=300, seed=1)["auroc"]
    assert (by_group[1] - by_group[0]) > (by_row[1] - by_row[0])


def _preds(n=120, seed=0, shift=2.0, **ids):
    rng = np.random.default_rng(seed)
    y = np.r_[np.zeros(n), np.ones(n)].astype(int)
    df = pd.DataFrame({"source_id": [f"d{i}" for i in range(n)] * 2, "label": y,
                       "score": np.r_[rng.normal(0, 1, n), rng.normal(shift, 1, n)]})
    for k, v in ids.items():
        df[k] = v
    return df


def test_evaluate_with_ci_fields():
    r = M.evaluate_with_ci(_preds(), threshold=1.0, n_boot=50)
    assert r["n"] == 240 and r["n_human"] == 120 and close(r["fpr_resolution"], 1 / 120)
    assert close(r["majority_baseline_acc"], 0.5)
    assert r["macro_f1_lo"] <= r["macro_f1"] <= r["macro_f1_hi"]


def test_add_gaps():
    t = pd.DataFrame({"method": ["m"] * 2, "train_domain": ["news"] * 2, "train_generator": ["g1"] * 2,
                      "domain": ["news", "reddit"], "generator": ["g1", "g1"], "macro_f1": [0.9, 0.6]})
    out = M.add_gaps(t)
    assert close(out.loc[out.domain == "news", "gap_macro_f1"].iloc[0], 0.0)
    assert close(out.loc[out.domain == "reddit", "gap_macro_f1"].iloc[0], 0.3)


def test_evaluate_end_to_end_with_frozen_val_threshold(tmp_path):
    ids = dict(method="m", train_domain="news", train_generator="g1", generator="g1", split="test")
    test = pd.concat([_preds(seed=1, domain="news", **ids), _preds(seed=2, shift=0.5, domain="reddit", **ids)])
    val = _preds(seed=3, domain="news", **{**ids, "split": "val"})
    tp, vp, op = tmp_path / "t.csv", tmp_path / "v.csv", tmp_path / "o.csv"
    test.to_csv(tp, index=False); val.to_csv(vp, index=False)
    res = E.main(["--test", str(tp), "--val", str(vp), "--n-boot", "30", "--out", str(op)])
    assert len(res) == 2 and op.exists()
    assert res["threshold"].nunique() == 1                        # same frozen threshold for both targets
    news = res[res.domain == "news"].iloc[0]; reddit = res[res.domain == "reddit"].iloc[0]
    assert news["auroc"] > reddit["auroc"]
    assert reddit["gap_macro_f1"] > 0 and close(news["gap_macro_f1"], 0.0)


def test_missing_columns_raises(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"source_id": ["a"], "label": [0]}).to_csv(p, index=False)
    try:
        E.read_predictions(p)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
