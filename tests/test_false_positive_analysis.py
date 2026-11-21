import pandas as pd
from src.analyze_false_positives import make_report


def test_false_positives_use_frozen_validation_threshold_and_join_text():
    val = pd.DataFrame({"method": ["tfidf_lr"] * 4, "train_domain": ["reddit"] * 4,
        "train_generator": ["mistral-chat"] * 4, "label": [0, 0, 1, 1],
        "score": [0.1, 0.2, 0.8, 0.9]})
    test = pd.DataFrame({"source_id": ["a", "b", "c", "d"], "label": [0] * 4,
        "score": [0.95, 0.0, 0.15, 0.12], "method": ["tfidf_lr"] * 4,
        "train_domain": ["reddit"] * 4, "train_generator": ["mistral-chat"] * 4,
        "domain": ["news"] * 4, "generator": ["mistral-chat"] * 4,
        "split": ["test"] * 4})
    dataset = test[["source_id", "label", "domain", "generator", "split"]].copy()
    dataset["text"] = ["article report writers work", "they report local news",
                       "medical report research", "city local people said"]
    dataset["word_count"] = dataset.text.str.split().str.len()
    summary, lengths, cues = make_report(dataset, val, test, domains=["news"], methods=["tfidf_lr"])
    assert summary.iloc[0].n_human == 4
    assert summary.iloc[0].n_fp == 1
    assert summary.iloc[0].fpr == 0.25
    assert lengths.n.sum() == 4
