"""Classical baselines: TF-IDF (word + character n-grams) with logistic regression / linear SVM.

Protocol (proposal, Section 4)
  * TF-IDF vocabulary and IDF weights are fitted on TRAINING data only.
  * For each training domain, C is chosen on that domain's VALIDATION split (by val AUROC,
    which is threshold-free); the decision threshold is chosen later by src.evaluate, also on val.
  * Each trained model is tested on the test split of EVERY domain (cross-domain, RQ1) and,
    optionally, other generators (cross-generator, RQ2).

Outputs in --out:
  predictions_val.csv / predictions_test.csv   (format read by src.evaluate)
  hyperparams.csv, runtime.csv, top_features.csv, [metrics.csv with --evaluate]

Example
  python -m src.baselines_tfidf --generator mistral-chat --test-generators mistral-chat llama-chat --evaluate
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from src import evaluate as E
from src import metrics as M
from src.data import DOMAINS, GENERATORS

GRID = [0.1, 1.0, 10.0]
METHODS = {
    "tfidf_lr": lambda C, seed: LogisticRegression(C=C, max_iter=2000, solver="liblinear", random_state=seed),
    "tfidf_svm": lambda C, seed: LinearSVC(C=C, max_iter=5000, random_state=seed),
}


def build_features(word_max=100_000, char_max=200_000):
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=word_max, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                 max_features=char_max, sublinear_tf=True)),
    ])


def subset(df, domain, generator, split):
    return df[(df["domain"] == domain) & (df["generator"] == generator) & (df["split"] == split)]


def pred_frame(rows, scores, method, train_domain, train_generator):
    return pd.DataFrame({
        "source_id": rows["source_id"].to_numpy(), "label": rows["label"].to_numpy(),
        "score": np.asarray(scores, dtype=float), "method": method,
        "train_domain": train_domain, "train_generator": train_generator,
        "domain": rows["domain"].to_numpy(), "generator": rows["generator"].to_numpy(),
        "split": rows["split"].to_numpy()})


def run_baselines(df, domains=DOMAINS, train_generator="mistral-chat", test_generators=None,
                  methods=None, grid=GRID, seed=42, top_k=20):
    test_generators = test_generators or [train_generator]
    methods = methods or METHODS
    val_rows, test_rows, hp_rows, rt_rows, feat_rows = [], [], [], [], []
    for td in domains:
        tr, va = subset(df, td, train_generator, "train"), subset(df, td, train_generator, "val")
        if tr.empty or va.empty:
            print(f"[warn] no train/val data for {td} / {train_generator}; skipping")
            continue
        t0 = time.perf_counter()
        vec = build_features()
        Xtr = vec.fit_transform(tr["text"])            # fitted on training data only
        vec_seconds = time.perf_counter() - t0
        Xva = vec.transform(va["text"])
        names = vec.get_feature_names_out()
        for name, make in methods.items():
            fit_seconds, best = 0.0, None
            for C in grid:
                t0 = time.perf_counter()
                clf = make(C, seed).fit(Xtr, tr["label"])
                fit_seconds += time.perf_counter() - t0
                auc = M.auroc(va["label"], clf.decision_function(Xva))
                if best is None or auc > best[0]:
                    best = (auc, C, clf)
            auc, C, clf = best
            val_rows.append(pred_frame(va, clf.decision_function(Xva), name, td, train_generator))
            infer_s, infer_n = 0.0, 0
            for d2 in domains:
                for g2 in test_generators:
                    te = subset(df, d2, g2, "test")
                    if te.empty:
                        continue
                    t0 = time.perf_counter()
                    s = clf.decision_function(vec.transform(te["text"]))
                    infer_s += time.perf_counter() - t0
                    infer_n += len(te)
                    test_rows.append(pred_frame(te, s, name, td, train_generator))
            hp_rows.append(dict(method=name, train_domain=td, train_generator=train_generator, best_C=C,
                                val_auroc=auc, n_train=len(tr), n_features=len(names)))
            rt_rows.append(dict(method=name, train_domain=td, train_generator=train_generator,
                                train_seconds=vec_seconds + fit_seconds,
                                infer_ms_per_1000_texts=(1e6 * infer_s / infer_n) if infer_n else float("nan")))
            if name == "tfidf_lr":
                w = clf.coef_.ravel()
                for direction, idx in [("machine", np.argsort(-w)[:top_k]), ("human", np.argsort(w)[:top_k])]:
                    for i in idx:
                        feat_rows.append(dict(train_domain=td, direction=direction, feature=names[i], weight=float(w[i])))
    cat = lambda rows: pd.concat(rows, ignore_index=True) if rows and isinstance(rows[0], pd.DataFrame) else pd.DataFrame(rows)
    return {"val": cat(val_rows), "test": cat(test_rows), "hyperparams": cat(hp_rows),
            "runtime": cat(rt_rows), "top_features": cat(feat_rows)}


def save_outputs(res, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {"val": "predictions_val.csv", "test": "predictions_test.csv", "hyperparams": "hyperparams.csv",
             "runtime": "runtime.csv", "top_features": "top_features.csv"}
    for k, f in files.items():
        res[k].to_csv(out / f, index=False)
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/dataset.csv")
    p.add_argument("--generator", default="mistral-chat", help="generator used for training")
    p.add_argument("--test-generators", nargs="+", default=None,
                   help="generators to test on (default: same as --generator)")
    p.add_argument("--domains", nargs="+", default=DOMAINS)
    p.add_argument("--out", default="results/tfidf")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--evaluate", action="store_true", help="also run src.evaluate and write metrics.csv")
    p.add_argument("--n-boot", type=int, default=1000)
    a = p.parse_args(argv)
    df = pd.read_csv(a.data)
    res = run_baselines(df, a.domains, a.generator, a.test_generators, seed=a.seed)
    out = save_outputs(res, a.out)
    print(res["hyperparams"].round(3).to_string(index=False))
    print(res["runtime"].round(2).to_string(index=False))
    if a.evaluate:
        metrics = E.run(res["test"], res["val"], n_boot=a.n_boot, seed=a.seed)
        metrics.to_csv(out / "metrics.csv", index=False)
        show = ["method", "train_domain", "domain", "generator", "n", "macro_f1", "macro_f1_lo",
                "macro_f1_hi", "auroc", "fpr", "gap_macro_f1"]
        print(metrics[[c for c in show if c in metrics.columns]].round(3).to_string(index=False))
    print(f"\nWrote results to {out}/")
    return res


if __name__ == "__main__":
    main()
