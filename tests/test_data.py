import json

import pandas as pd
import pytest

from src import data


@pytest.fixture(scope="module")
def raw():
    return data.filter_frame(data.make_synthetic_raid(60), decoding="greedy", repetition_penalty="no")


@pytest.fixture(scope="module")
def built(raw):
    return data.build_dataset(raw, max_docs=50, max_words=200, min_words=50, seed=1)


def test_filter_frame_drops_attacks_other_models_and_other_decoding():
    full = data.make_synthetic_raid(10)
    out = data.filter_frame(full, decoding="greedy", repetition_penalty="no")
    assert set(out["model"]) == {"human", "mistral-chat", "llama-chat"}  # chatgpt removed
    assert (out["attack"] == "none").all()
    assert set(out.loc[out.model != "human", "decoding"]) == {"greedy"}
    assert (out["model"] == "human").sum() == 40  # human rows kept despite NaN decoding


def test_truncate_words():
    assert data.truncate_words("a b c d e", 3) == "a b c"
    assert data.truncate_words("a b", 3) == "a b"


def test_dedup_removes_exact_and_near_duplicates():
    base = " ".join(f"word{i}" for i in range(80))
    df = pd.DataFrame({
        "text": [base, base, base + " extra tail", "totally different " + base],
        "model": ["human"] * 4, "domain": ["news"] * 4,
    })
    assert len(data.dedup(df)) == 2


def test_assign_splits_proportions_and_determinism():
    ids = [f"id{i}" for i in range(100)]
    a, b = data.assign_splits(ids, seed=3), data.assign_splits(ids, seed=3)
    assert a == b
    counts = pd.Series(a).value_counts()
    assert (counts["train"], counts["val"], counts["test"]) == (70, 15, 15)
    assert data.assign_splits(ids, seed=4) != a


def test_no_source_document_in_two_splits(built):
    assert built.groupby("source_id")["split"].nunique().max() == 1
    data.check_no_leakage(built)


def test_word_cap_min_and_docs_cap(built):
    assert built["word_count"].max() <= 200
    assert built["word_count"].min() >= 50
    assert built.groupby("domain")["source_id"].nunique().max() <= 50


def test_classes_balanced_per_comparison(built):
    counts = built.groupby(["domain", "generator", "split"])["label"].agg(["sum", "count"])
    assert ((counts["count"] - counts["sum"]) == counts["sum"]).all()  # humans == machines


def test_length_matched_within_bins(built):
    b = built.assign(bin=pd.cut(built["word_count"], data.DEFAULT_BINS, labels=False, include_lowest=True))
    t = b.groupby(["domain", "generator", "split", "bin"])["label"].agg(["sum", "count"])
    assert ((t["count"] - t["sum"]) == t["sum"]).all()


def test_all_domains_and_generators_present(built):
    assert set(built["domain"]) == set(data.DOMAINS)
    assert set(built["generator"]) == set(data.GENERATORS)
    assert set(built["split"]) == {"train", "val", "test"}


def test_build_is_reproducible(raw):
    a = data.build_dataset(raw, max_docs=30, seed=5)
    b = data.build_dataset(raw, max_docs=30, seed=5)
    pd.testing.assert_frame_equal(a, b)


def test_missing_generator_text_excludes_document(raw):
    r = raw[~((raw.source_id == "news-0") & (raw.model == "llama-chat"))]
    out = data.build_dataset(r, max_docs=60, seed=1)
    assert "news-0" not in set(out["source_id"])


def test_load_split_roundtrip(built, tmp_path):
    p = tmp_path / "d.csv"
    built.to_csv(p, index=False)
    sub = data.load_split("news", "mistral-chat", "train", path=p)
    assert len(sub) > 0
    assert set(sub["domain"]) == {"news"} and set(sub["split"]) == {"train"}
    assert {"text", "label", "source_id"} <= set(sub.columns)


def test_manifest_matches_data(built, tmp_path):
    data.save_manifest(built, tmp_path / "m.json")
    m = json.loads((tmp_path / "m.json").read_text())
    ids = {s for split in m.values() for dom in split.values() for s in dom}
    assert ids == set(built["source_id"])
