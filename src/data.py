"""Data pipeline for the AI-generated-text detection project.

Steps (see proposal, Sections 4-5):
  1. filter RAID to our domains / generators / decoding setting (no attacks)
  2. truncate to a common word cap, drop very short texts
  3. remove exact and near-duplicate texts
  4. keep only source documents that have a human text AND every generator's text
  5. cap at max_docs source documents per domain
  6. split SOURCE DOCUMENTS 70/15/15 so a human text and all its generations
     always land in the same partition (no leakage)
  7. per (generator, domain, split): balance human vs. machine by word-count bin

Output: one row per text with columns
  text, label (0 human / 1 machine), source_id, domain, generator (the
  comparison, e.g. 'mistral-chat'), model, split, word_count
Human texts appear once per generator comparison (same text, different
`generator` value) because each comparison is length-matched separately.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

DOMAINS = ["abstracts", "news", "reviews", "reddit"]
GENERATORS = ["mistral-chat", "llama-chat"]  # G1, G2
HUMAN = "human"
DEFAULT_BINS = [0, 100, 150, 200, 10_000]
RAID_COLS = ["source_id", "model", "decoding", "repetition_penalty", "attack", "domain", "generation"]
DEFAULT_PATH = "data/processed/dataset.csv"


# --------------------------------------------------------------------------- loading / filtering
def filter_frame(df, domains=DOMAINS, generators=GENERATORS, decoding=None, repetition_penalty=None):
    """Keep human rows + chosen generators, chosen domains, no adversarial attack.

    decoding / repetition_penalty (if given) apply to machine rows only, since
    human rows have no decoding setting.
    """
    keep_models = set(generators) | {HUMAN}
    mask = df["domain"].isin(domains) & df["model"].isin(keep_models)
    if "attack" in df.columns:
        mask &= df["attack"].fillna("none").eq("none")
    out = df[mask]
    is_h = out["model"].eq(HUMAN)
    if decoding is not None and "decoding" in out.columns:
        out = out[is_h | out["decoding"].eq(decoding)]
        is_h = out["model"].eq(HUMAN)
    if repetition_penalty is not None and "repetition_penalty" in out.columns:
        out = out[is_h | out["repetition_penalty"].eq(repetition_penalty)]
    return out


def read_raid_csv(path, domains=DOMAINS, generators=GENERATORS, decoding=None,
                  repetition_penalty=None, chunksize=200_000):
    """Stream a big RAID csv (e.g. train_none.csv) and keep only what we need."""
    parts = []
    for chunk in pd.read_csv(path, usecols=lambda c: c in RAID_COLS, chunksize=chunksize):
        f = filter_frame(chunk, domains, generators, decoding, repetition_penalty)
        if len(f):
            parts.append(f)
    if not parts:
        raise ValueError("No rows left after filtering. Run the `inspect` command to see what values exist.")
    return pd.concat(parts, ignore_index=True)


def read_raid_hf(domains=DOMAINS, generators=GENERATORS, decoding=None, repetition_penalty=None):
    """Load from Hugging Face (liamdugan/raid). Needs `pip install datasets`."""
    from datasets import load_dataset

    ds = load_dataset("liamdugan/raid", split="train")
    ds = ds.filter(lambda ex: ex["domain"] in domains and ex["model"] in set(generators) | {HUMAN}
                   and ex["attack"] == "none")
    return filter_frame(ds.to_pandas(), domains, generators, decoding, repetition_penalty)


# --------------------------------------------------------------------------- text cleaning
def truncate_words(text, max_words):
    words = text.split()
    return text if len(words) <= max_words else " ".join(words[:max_words])


def normalize(text):
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def dedup(df, prefix_chars=300):
    """Drop exact duplicates and near-duplicates (same first `prefix_chars`
    normalized characters) within each (model, domain). Lightweight stand-in
    for MinHash near-dup detection."""
    df = df.copy()
    norm = df["text"].map(normalize)
    df["_exact"] = norm.map(lambda s: hashlib.md5(s.encode()).hexdigest())
    df["_prefix"] = norm.str[:prefix_chars]
    df = df.drop_duplicates(["model", "domain", "_exact"])
    df = df.drop_duplicates(["model", "domain", "_prefix"])
    return df.drop(columns=["_exact", "_prefix"])


# --------------------------------------------------------------------------- splitting / matching
def assign_splits(source_ids, seed=42, fracs=(0.7, 0.15, 0.15)):
    """Deterministically map each source_id to train/val/test."""
    ids = np.array(sorted(set(source_ids)))
    ids = np.random.default_rng(seed).permutation(ids)
    n = len(ids)
    n_tr, n_va = int(round(fracs[0] * n)), int(round(fracs[1] * n))
    mapping = {}
    for i, sid in enumerate(ids):
        mapping[str(sid)] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    return mapping


def length_match(df, bins=DEFAULT_BINS, seed=42):
    """Within each word-count bin keep equal numbers of human (0) and machine (1) texts."""
    df = df.assign(_bin=pd.cut(df["word_count"], bins=bins, labels=False, include_lowest=True))
    out = []
    for _, g in df.groupby("_bin"):
        h, m = g[g["label"] == 0], g[g["label"] == 1]
        k = min(len(h), len(m))
        if k == 0:
            continue
        out.append(h.sample(k, random_state=seed))
        out.append(m.sample(k, random_state=seed))
    if not out:
        return df.iloc[0:0].drop(columns="_bin")
    return pd.concat(out).drop(columns="_bin")


# --------------------------------------------------------------------------- main builder
def build_dataset(raw, domains=DOMAINS, generators=GENERATORS, max_docs=1000, max_words=256,
                  min_words=50, bins=DEFAULT_BINS, seed=42, fracs=(0.7, 0.15, 0.15),
                  require_all_generators=True):
    df = raw[["source_id", "model", "domain", "generation"]].rename(columns={"generation": "text"}).copy()
    df["text"] = df["text"].astype(str).str.strip().map(lambda t: truncate_words(t, max_words))
    df["word_count"] = df["text"].str.split().str.len()
    df = df[df["word_count"] >= min_words]
    df = dedup(df)
    df = df.drop_duplicates(["source_id", "model"])  # one text per (source, model)

    need = {HUMAN} | (set(generators) if require_all_generators else set())
    have = df.groupby("source_id")["model"].agg(set)
    ok_ids = have[have.map(lambda s: need <= s)].index
    df = df[df["source_id"].isin(ok_ids)]
    if df.empty:
        raise ValueError("No source documents have a human text plus all generators. "
                         "Check `source_id` for human rows (run `inspect`) and the decoding filter.")

    parts = []
    for dom in domains:
        sub = df[df["domain"] == dom]
        ids = np.array(sorted(sub["source_id"].unique()))
        if len(ids) == 0:
            print(f"[warn] no usable documents for domain '{dom}'")
            continue
        ids = np.random.default_rng(seed).permutation(ids)[:max_docs]
        sub = sub[sub["source_id"].isin(ids)].copy()
        sub["split"] = sub["source_id"].map(assign_splits(ids, seed, fracs))
        parts.append(sub)
    full = pd.concat(parts, ignore_index=True)

    out = []
    for gen in generators:
        cmp_ = full[full["model"].isin([HUMAN, gen])].copy()
        cmp_["generator"] = gen
        cmp_["label"] = (cmp_["model"] != HUMAN).astype(int)
        for _, g in cmp_.groupby(["domain", "split"]):
            out.append(length_match(g, bins, seed))
    res = pd.concat(out, ignore_index=True)
    check_no_leakage(res)
    cols = ["text", "label", "source_id", "domain", "generator", "model", "split", "word_count"]
    return res[cols]


# --------------------------------------------------------------------------- checks / io
def check_no_leakage(df):
    """Every source_id must live in exactly one split."""
    n_splits = df.groupby("source_id")["split"].nunique()
    bad = n_splits[n_splits > 1]
    if len(bad):
        raise AssertionError(f"{len(bad)} source documents appear in more than one split")


def summary(df):
    return (df.groupby(["domain", "generator", "split", "label"]).size()
              .unstack("label", fill_value=0).rename(columns={0: "human", 1: "machine"}))


def save_manifest(df, path):
    """Document IDs per split/domain so the splits can be reproduced without the data."""
    manifest = {s: {d: sorted(g["source_id"].unique().tolist()) for d, g in sg.groupby("domain")}
                for s, sg in df.groupby("split")}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(manifest, indent=1))


def load_split(domain, generator, split, path=DEFAULT_PATH):
    """The interface Gustavo codes against: a DataFrame with text, label, source_id, ..."""
    df = pd.read_csv(path)
    sel = df[(df["domain"] == domain) & (df["generator"] == generator) & (df["split"] == split)]
    return sel.reset_index(drop=True)


# --------------------------------------------------------------------------- synthetic data
def make_synthetic_raid(n_per_domain=60, seed=0, domains=DOMAINS):
    """Tiny RAID-shaped frame for tests / smoke runs (no download needed)."""
    rng = np.random.default_rng(seed)
    vocab = [f"w{i}" for i in range(800)]
    models = ["mistral-chat", "llama-chat", "chatgpt"]
    rows = []

    def text(lo, hi):
        return " ".join(rng.choice(vocab, size=int(rng.integers(lo, hi))))

    for dom in domains:
        for i in range(n_per_domain):
            sid = f"{dom}-{i}"
            rows.append(dict(source_id=sid, model=HUMAN, decoding=np.nan, repetition_penalty=np.nan,
                             attack="none", domain=dom, generation=text(60, 320)))
            for m in models:
                for dec in ["greedy", "sampling"]:
                    rows.append(dict(source_id=sid, model=m, decoding=dec, repetition_penalty="no",
                                     attack="none", domain=dom, generation=text(60, 320)))
            rows.append(dict(source_id=sid, model="mistral-chat", decoding="greedy", repetition_penalty="no",
                             attack="paraphrase", domain=dom, generation=text(60, 320)))
    return pd.DataFrame(rows)
