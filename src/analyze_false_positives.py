"""Analyze human false positives with *frozen source-validation* thresholds.

Run from repository root:
  py -m src.analyze_false_positives --out results/tfidf_false_positives

Uses the existing predictions; no retraining. Outputs only aggregates, not raw text.
The cue analysis is descriptive and is NOT used for model or threshold tuning.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re

import numpy as np
import pandas as pd
from src import evaluate as E

TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
TRAIN_COLS = ["method", "train_domain", "train_generator"]
JOIN_COLS = ["source_id", "label", "domain", "generator", "split"]


def token_document_counts(texts):
    """One count per token per document, not raw token frequency."""
    counts = Counter()
    for text in texts:
        counts.update(set(TOKEN_RE.findall(str(text).lower())))
    return counts


def make_report(dataset, val, test, *, train_domain="reddit",
                domains=("abstracts", "news"), generator="mistral-chat",
                methods=("tfidf_lr", "tfidf_svm"), max_cues=20):
    needed = set(JOIN_COLS + ["text", "word_count"])
    if not needed <= set(dataset.columns):
        raise ValueError(f"dataset missing columns {sorted(needed - set(dataset.columns))}")
    if not set(JOIN_COLS + ["score"] + TRAIN_COLS) <= set(test.columns):
        raise ValueError("test predictions missing required columns")
    if not set(TRAIN_COLS + ["score", "label"]) <= set(val.columns):
        raise ValueError("validation predictions missing required columns")
    thresholds = E.pick_thresholds(val, TRAIN_COLS, mode="f1")
    group = test[(test["train_domain"] == train_domain) &
                 test["domain"].isin(domains) &
                 (test["generator"] == generator) &
                 test["method"].isin(methods) & (test["label"] == 0)].copy()
    if group.empty:
        raise ValueError("No human test predictions match the requested filters")
    group["threshold"] = group.apply(
        lambda r: thresholds[(r["method"], r["train_domain"], r["train_generator"])], axis=1)
    group["false_positive"] = group["score"] >= group["threshold"]
    group = group.merge(dataset[JOIN_COLS + ["text", "word_count"]],
                        on=JOIN_COLS, how="left", validate="many_to_one", indicator=True)
    if (group["_merge"] != "both").any():
        raise ValueError("Missing corresponding dataset texts; verify the saved predictions and dataset match")
    group = group.drop(columns="_merge")

    summaries, lengths, cues = [], [], []
    keys = ["method", "train_domain", "domain", "generator"]
    for key, section in group.groupby(keys):
        values = dict(zip(keys, key))
        fp = section[section.false_positive]
        tn = section[~section.false_positive]
        summaries.append({**values, "n_human": len(section), "n_fp": len(fp),
                          "n_tn": len(tn), "fpr": len(fp) / len(section),
                          "threshold": float(section["threshold"].iloc[0])})
        for name, part in (("false_positive", fp), ("true_negative", tn)):
            lengths.append({**values, "group": name, "n": len(part),
                "mean_words": float(part.word_count.mean()) if len(part) else np.nan,
                "median_words": float(part.word_count.median()) if len(part) else np.nan,
                "mean_score": float(part.score.mean()) if len(part) else np.nan})
        if len(fp) >= 3 and len(tn) >= 3:
            count_fp = token_document_counts(fp.text)
            count_tn = token_document_counts(tn.text)
            scored = []
            for tok in count_fp.keys() | count_tn.keys():
                a, b = count_fp[tok], count_tn[tok]
                # Filter very rare cues; normalized difference is descriptive.
                if a + b < 3:
                    continue
                rate_fp = a / len(fp); rate_tn = b / len(tn)
                scored.append((rate_fp - rate_tn, tok, a, b, rate_fp, rate_tn))
            for delta, tok, a, b, ra, rb in sorted(scored, reverse=True)[:max_cues]:
                if delta <= 0:
                    break
                cues.append({**values, "token": tok, "fp_docs": a, "tn_docs": b,
                             "fp_doc_rate": ra, "tn_doc_rate": rb, "rate_difference": delta})
    return pd.DataFrame(summaries), pd.DataFrame(lengths), pd.DataFrame(cues)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/dataset.csv")
    p.add_argument("--val", default="results/tfidf/predictions_val.csv")
    p.add_argument("--test", default="results/tfidf/predictions_test.csv")
    p.add_argument("--train-domain", default="reddit")
    p.add_argument("--test-domains", nargs="+", default=["abstracts", "news"])
    p.add_argument("--generator", default="mistral-chat")
    p.add_argument("--out", default="results/tfidf_false_positives")
    a = p.parse_args(argv)
    result = make_report(pd.read_csv(a.data), pd.read_csv(a.val), pd.read_csv(a.test),
                         train_domain=a.train_domain, domains=a.test_domains,
                         generator=a.generator)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for name, df in zip(("false_positive_summary", "error_lengths", "error_token_cues"), result):
        df.to_csv(out / f"{name}.csv", index=False)
    print("\nFALSE POSITIVES BY DOMAIN")
    print(result[0].round(4).to_string(index=False))
    print("\nLENGTH DIFFERENCES")
    print(result[1].round(3).to_string(index=False))
    print("\nTOP TOKEN CUES (DESCRIPTIVE)")
    print(result[2].groupby(["method", "domain"]).head(7).round(3).to_string(index=False)) if not result[2].empty else print("No qualifying token cues")
    print(f"\nWrote three CSVs to {out}")

if __name__ == "__main__":
    main()
