import numpy as np
import pandas as pd
import pytest

from src import threshold_analysis as T


def fixture_predictions():
    v, te = [], []
    for train_domain in ("news", "reviews"):
        scores = [-1.2, -1.0, -0.5, 0.4, 0.8, 1.5]
        labels = [0, 0, 0, 1, 1, 1]
        for i, (score, label) in enumerate(zip(scores, labels)):
            v.append(dict(source_id=f"{train_domain}-val-{i}", label=label, score=score,
                          method="tfidf_lr", train_domain=train_domain,
                          train_generator="mistral-chat", domain=train_domain,
                          generator="mistral-chat", split="val"))
        for domain in ("news", "reviews"):
            for gen in ("mistral-chat", "llama-chat"):
                for i, (score, label) in enumerate(zip(scores, labels)):
                    te.append(dict(source_id=f"{domain}-test-{i}", label=label, score=score * 0.8,
                                   method="tfidf_lr", train_domain=train_domain,
                                   train_generator="mistral-chat", domain=domain,
                                   generator=gen, split="test"))
    return pd.DataFrame(v), pd.DataFrame(te)


def test_rules_are_validation_fitted():
    v, te = fixture_predictions()
    t1, m1 = T.analyze(v, te)
    shuffled = te.copy()
    shuffled["label"] = shuffled["label"].iloc[::-1].to_numpy()
    t2, _ = T.analyze(v, shuffled)
    pd.testing.assert_frame_equal(t1, t2)  # test labels never affect thresholds
    assert set(t1.rule) == set(T.RULES)
    assert len(m1) == 2 * 3 * 2 * 2
    assert (t1[t1.rule == "low_fpr_1pct"].val_fpr <= 0.01).all()


def test_midpoint_matches_medians():
    v, _ = fixture_predictions()
    val = v[v.train_domain == "news"]
    t = T.select_thresholds(val)
    assert t["class_median_midpoint"] == pytest.approx(0.5 * (-1.0 + 0.8))


def test_outputs_have_fixed_threshold_and_four_conditions():
    v, te = fixture_predictions()
    _, m = T.analyze(v, te)
    assert (m.groupby(["method", "train_domain", "rule"])["threshold"].nunique() == 1).all()
    assert set(m.condition) == {"in_domain", "cross_domain", "cross_generator", "combined_shift"}
    s = T.summarize(m)
    assert len(s) == 1 * 4 * 3  # one method, four conditions, three rules
    assert np.allclose(s[s.rule == "max_macro_f1"].delta_macro_f1_vs_original, 0)


def test_nonfinite_validation_scores_rejected():
    v, _ = fixture_predictions()
    v.loc[v.index[0], "score"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        T.select_thresholds(v[v.train_domain == "news"])
