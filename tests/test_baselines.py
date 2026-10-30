import numpy as np
import pandas as pd

from src import baselines_tfidf as B
from src import evaluate as E


def make_dataset(domains=("news", "reddit"), gens=("g1", "g2"), n_train=40, seed=0):
    """Machine texts carry 'delve/moreover/...' markers, human texts carry 'honestly/kinda/...'."""
    rng = np.random.default_rng(seed)
    rows = []
    for dom in domains:
        vocab = [f"{dom}word{i}" for i in range(60)]
        for g in gens:
            for split, count in [("train", n_train), ("val", 15), ("test", 15)]:
                for i in range(count):
                    sid = f"{dom}-{split}-{i}"
                    for label in (0, 1):
                        marks = ["delve", "moreover", "tapestry", "crucial"] if label else ["honestly", "kinda", "gonna", "lol"]
                        words = list(rng.choice(vocab, 40)) + list(rng.choice(marks, 8))
                        rng.shuffle(words)
                        rows.append(dict(text=" ".join(words), label=label, source_id=sid, domain=dom,
                                         generator=g, model="human" if label == 0 else g, split=split,
                                         word_count=len(words)))
    return pd.DataFrame(rows)


DOMS = ["news", "reddit"]


def test_outputs_cover_all_train_test_combinations():
    df = make_dataset()
    res = B.run_baselines(df, DOMS, "g1", ["g1", "g2"])
    # 2 methods x 2 train domains x (2 test domains x 2 test generators) groups
    g = res["test"].groupby(["method", "train_domain", "domain", "generator"]).size()
    assert len(g) == 2 * 2 * 2 * 2
    assert set(res["test"]["split"]) == {"test"} and set(res["val"]["split"]) == {"val"}
    assert (res["test"]["train_generator"] == "g1").all()
    need = {"source_id", "label", "score", "method", "train_domain", "train_generator", "domain", "generator", "split"}
    assert need <= set(res["test"].columns)


def test_val_predictions_only_from_training_domain_and_generator():
    res = B.run_baselines(make_dataset(), DOMS, "g1")
    v = res["val"]
    assert (v["domain"] == v["train_domain"]).all() and (v["generator"] == "g1").all()


def test_models_learn_the_signal_in_and_across_domains():
    res = B.run_baselines(make_dataset(), DOMS, "g1")
    from src import metrics as M
    for (m, td, d), g in res["test"].groupby(["method", "train_domain", "domain"]):
        assert M.auroc(g["label"], g["score"]) > 0.9, (m, td, d)


def test_vectorizer_is_fitted_on_training_data_only():
    df = make_dataset()
    tr = df[(df.domain == "news") & (df.generator == "g1") & (df.split == "train")].copy()
    te = df[(df.domain == "news") & (df.generator == "g1") & (df.split == "test")].copy()
    te["text"] = te["text"] + " zzzonlyintest zzzonlyintest"
    vec = B.build_features()
    vec.fit(tr["text"])
    assert not any("zzzonlyintest" in n for n in vec.get_feature_names_out())
    assert vec.transform(te["text"]).shape[1] == len(vec.get_feature_names_out())


def test_hyperparameters_come_from_grid_and_are_chosen_on_validation():
    res = B.run_baselines(make_dataset(), DOMS, "g1", grid=[0.1, 1.0])
    hp = res["hyperparams"]
    assert set(hp["best_C"]) <= {0.1, 1.0}
    assert (hp["val_auroc"] > 0.9).all() and len(hp) == 4
    assert {"train_seconds", "infer_ms_per_1000_texts"} <= set(res["runtime"].columns)
    assert (res["runtime"]["train_seconds"] > 0).all()


def test_top_features_are_the_marker_words():
    res = B.run_baselines(make_dataset(), DOMS, "g1")
    tf = res["top_features"]
    news_machine = " ".join(tf[(tf.train_domain == "news") & (tf.direction == "machine")]["feature"])
    news_human = " ".join(tf[(tf.train_domain == "news") & (tf.direction == "human")]["feature"])
    assert "delve" in news_machine or "moreover" in news_machine
    assert "honestly" in news_human or "kinda" in news_human


def test_deterministic():
    df = make_dataset()
    a = B.run_baselines(df, DOMS, "g1", grid=[1.0])["test"]
    b = B.run_baselines(df, DOMS, "g1", grid=[1.0])["test"]
    pd.testing.assert_frame_equal(a, b)


def test_end_to_end_with_evaluate_and_cli(tmp_path):
    data_path = tmp_path / "dataset.csv"
    make_dataset().to_csv(data_path, index=False)
    out = tmp_path / "out"
    B.main(["--data", str(data_path), "--generator", "g1", "--test-generators", "g1", "g2",
            "--domains", "news", "reddit", "--out", str(out), "--evaluate", "--n-boot", "20"])
    for f in ["predictions_val.csv", "predictions_test.csv", "hyperparams.csv", "runtime.csv",
              "top_features.csv", "metrics.csv"]:
        assert (out / f).exists(), f
    m = pd.read_csv(out / "metrics.csv")
    assert len(m) == 16 and "gap_macro_f1" in m.columns
    # frozen thresholds: one threshold per (method, train_domain), shared across all its test sets
    assert (m.groupby(["method", "train_domain"])["threshold"].nunique() == 1).all()
    # in-domain, same-generator rows have zero gap
    ind = m[(m.domain == m.train_domain) & (m.generator == "g1")]
    assert (ind["gap_macro_f1"].abs() < 1e-9).all()
