"""Quantify RAID data attrition without changing the existing dataset pipeline.

Run from repository root:
  py -m src.diagnose_attrition --csv data/raw/train_none.csv --out results/abstracts_attrition

Requires the *raw non-adversarial RAID CSV*, not data/processed/dataset.csv.
All experiment parameters default to the project's documented build settings.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src import data as D

STAGES = [
    "01_selected_raid", "02_min_50_words", "03_deduplicated",
    "04_one_source_model", "05_complete_source_triplets", "06_max_docs_cap",
]


def make_report(raw: pd.DataFrame, *, min_words=50, max_words=256,
                max_docs=1000, seed=42):
    """Build stage-level diagnostics that mirror src.data.build_dataset."""
    df = raw[["source_id", "model", "domain", "generation"]].rename(
        columns={"generation": "text"}).copy()
    # Inspect word counts before any truncation.
    df["text"] = df["text"].astype(str).str.strip()
    df["raw_words"] = df["text"].str.split().str.len()
    trunc = (df.assign(over_word_cap=df["raw_words"] > max_words,
                       under_word_min=df["raw_words"] < min_words)
             .groupby(["domain", "model"], as_index=False)
             .agg(n_texts=("source_id", "size"),
                  n_over_word_cap=("over_word_cap", "sum"),
                  n_under_word_min=("under_word_min", "sum"),
                  median_raw_words=("raw_words", "median")))
    trunc["pct_over_word_cap"] = trunc["n_over_word_cap"] / trunc["n_texts"]
    trunc["pct_under_word_min"] = trunc["n_under_word_min"] / trunc["n_texts"]

    stages = {}
    df["text"] = df["text"].map(lambda t: D.truncate_words(t, max_words))
    df["word_count"] = df["text"].str.split().str.len()
    stages[STAGES[0]] = df.copy()
    df = df[df["word_count"] >= min_words].copy()
    stages[STAGES[1]] = df.copy()
    df = D.dedup(df)
    stages[STAGES[2]] = df.copy()
    df = df.drop_duplicates(["source_id", "model"])
    stages[STAGES[3]] = df.copy()
    # The same cohort rule used in src/data.py: all 3 texts must survive.
    required = {D.HUMAN, *D.GENERATORS}
    present = df.groupby("source_id")["model"].agg(set)
    complete = set(present[present.map(lambda ms: required <= ms)].index)
    df = df[df["source_id"].isin(complete)].copy()
    stages[STAGES[4]] = df.copy()

    kept = []
    for dom in D.DOMAINS:
        sub = df[df["domain"] == dom]
        if sub.empty:
            continue
        ids = np.array(sorted(sub["source_id"].unique()))
        ids = np.random.default_rng(seed).permutation(ids)[:max_docs]
        sub = sub[sub["source_id"].isin(ids)].copy()
        sub["split"] = sub["source_id"].map(D.assign_splits(ids, seed))
        kept.append(sub)
    capped = pd.concat(kept, ignore_index=True) if kept else df.iloc[0:0].copy()
    stages[STAGES[5]] = capped.copy()

    snapshots = []
    for stage, stage_df in stages.items():
        counts = (stage_df.groupby(["domain", "model"], as_index=False)
                  .agg(n_rows=("source_id", "size"),
                       n_source_ids=("source_id", "nunique")))
        counts.insert(0, "stage", stage)
        snapshots.append(counts)
    stage_counts = pd.concat(snapshots, ignore_index=True)
    # Include missing stage/domain/model combinations as zero rather than silently dropping.
    index = pd.MultiIndex.from_product(
        [STAGES, D.DOMAINS, [D.HUMAN] + D.GENERATORS],
        names=["stage", "domain", "model"])
    stage_counts = (stage_counts.set_index(["stage", "domain", "model"])
                    .reindex(index, fill_value=0).reset_index())
    stage_counts["stage_index"] = stage_counts["stage"].map({s: i for i, s in enumerate(STAGES)})
    stage_counts = stage_counts.sort_values(["domain", "model", "stage_index"])
    stage_counts["rows_removed_since_previous"] = (
        stage_counts.groupby(["domain", "model"])["n_rows"].shift(1)
        - stage_counts["n_rows"])
    stage_counts["sources_removed_since_previous"] = (
        stage_counts.groupby(["domain", "model"])["n_source_ids"].shift(1)
        - stage_counts["n_source_ids"])
    stage_counts = stage_counts.drop(columns="stage_index")

    # Mirror the generator-specific length-matching step of the original pipeline.
    balance_rows = []
    for gen in D.GENERATORS:
        subset = capped[capped["model"].isin([D.HUMAN, gen])].copy()
        if subset.empty:
            continue
        subset["label"] = (subset["model"] != D.HUMAN).astype(int)
        for (domain, split), group in subset.groupby(["domain", "split"]):
            matched = D.length_match(group, seed=seed)
            for label in (0, 1):
                initial = int((group["label"] == label).sum())
                final = int((matched["label"] == label).sum())
                balance_rows.append(dict(domain=domain, generator=gen, split=split,
                                         label=label, n_before_matching=initial,
                                         n_after_matching=final,
                                         n_removed_by_matching=initial-final))
    matching = pd.DataFrame(balance_rows)
    return stage_counts, trunc, matching


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Raw RAID train_none.csv (not processed dataset.csv)")
    p.add_argument("--out", default="results/abstracts_attrition")
    p.add_argument("--decoding", default="greedy")
    p.add_argument("--rep-penalty", default="no")
    p.add_argument("--min-words", type=int, default=50)
    p.add_argument("--max-words", type=int, default=256)
    p.add_argument("--max-docs", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)
    if not Path(a.csv).is_file():
        p.error(f"Raw CSV not found: {a.csv}")
    raw = D.read_raid_csv(a.csv, decoding=a.decoding, repetition_penalty=a.rep_penalty)
    stages, trunc, matching = make_report(raw, min_words=a.min_words,
        max_words=a.max_words, max_docs=a.max_docs, seed=a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    stages.to_csv(out / "stage_counts.csv", index=False)
    trunc.to_csv(out / "truncation_diagnostics.csv", index=False)
    matching.to_csv(out / "length_matching.csv", index=False)
    print("\nABSTRACTS: PREPROCESSING STAGES")
    print(stages[stages.domain == "abstracts"].to_string(index=False))
    print("\nABSTRACTS: TRUNCATION")
    print(trunc[trunc.domain == "abstracts"].round(3).to_string(index=False))
    print("\nABSTRACTS: LENGTH MATCHING")
    print(matching[matching.domain == "abstracts"].to_string(index=False))
    print(f"\nWrote three diagnostic CSVs to {out}")

if __name__ == "__main__":
    main()
