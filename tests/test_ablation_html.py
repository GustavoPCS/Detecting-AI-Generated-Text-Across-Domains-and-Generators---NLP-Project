import pandas as pd
import pytest

from src import ablation_html as H


def test_only_markup_removed_and_entities_unescaped():
    text = "Loved it!<br /><br/>The film &amp; soundtrack were great. <em>Really</em>!"
    assert H.strip_formatting(text) == "Loved it! The film & soundtrack were great. Really !"
    assert H.strip_formatting("x < 3 and a > b") == "x < 3 and a > b"
    assert H.strip_formatting("plain words") == "plain words"


def test_fixed_identity_and_cleaning_stats():
    df = pd.DataFrame({"text": ["Nice<br/>movie", "Plain text", "Awful <br /> movie"],
                       "label": [0, 1, 0], "source_id": ["x", "y", "z"],
                       "domain": ["reviews"] * 3, "generator": ["mistral-chat"] * 3,
                       "split": ["train"] * 3})
    cleaned, stats = H.clean_dataset(df)
    assert cleaned.source_id.equals(df.source_id)
    assert cleaned.label.equals(df.label)
    assert cleaned.split.equals(df.split)
    assert cleaned.text.tolist() == ["Nice movie", "Plain text", "Awful movie"]
    assert int(stats.n_changed.sum()) == 2
    assert int(stats.n_texts.sum()) == 3


def test_metrics_comparison_and_same_sample_count():
    keys = dict(method="tfidf_lr", train_domain="reviews", train_generator="mistral-chat",
                domain="news", generator="mistral-chat", split="test")
    a = pd.DataFrame([{**keys, "n": 100, "macro_f1": 0.5, "auroc": 0.9, "fpr": 0.4, "tpr": 0.6}])
    b = pd.DataFrame([{**keys, "n": 100, "macro_f1": 0.6, "auroc": 0.89, "fpr": 0.3, "tpr": 0.65}])
    c = H.compare_metrics(a, b)
    assert c.delta_macro_f1.iloc[0] == pytest.approx(0.1)
    assert c.delta_fpr.iloc[0] == pytest.approx(-0.1)
    with pytest.raises(AssertionError):
        H.compare_metrics(a, b.assign(n=99))
