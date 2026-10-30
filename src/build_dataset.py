"""CLI for the data pipeline.

  python -m src.build_dataset inspect --csv data/raw/train_none.csv
  python -m src.build_dataset build   --csv data/raw/train_none.csv --decoding greedy --rep-penalty no
  python -m src.build_dataset build   --hf --decoding greedy --rep-penalty no
  python -m src.build_dataset build   --synthetic --out data/processed_synth     # smoke test, no download
"""
import argparse
from pathlib import Path

import pandas as pd

from src import data


def cmd_inspect(args):
    cols = ["model", "decoding", "repetition_penalty", "attack", "domain"]
    total = None
    for chunk in pd.read_csv(args.csv, usecols=lambda c: c in cols, chunksize=args.chunksize):
        chunk = chunk[chunk["domain"].isin(data.DOMAINS)]
        keys = [c for c in cols if c in chunk.columns]
        counts = chunk.groupby(keys, dropna=False).size()
        total = counts if total is None else total.add(counts, fill_value=0)
    pd.set_option("display.max_rows", 500)
    print("Row counts by (model, decoding, repetition_penalty, attack, domain):")
    print(total.astype(int).to_string())


def cmd_build(args):
    if args.synthetic:
        raw = data.filter_frame(data.make_synthetic_raid(), decoding=args.decoding,
                                repetition_penalty=args.rep_penalty)
    elif args.hf:
        raw = data.read_raid_hf(decoding=args.decoding, repetition_penalty=args.rep_penalty)
    else:
        raw = data.read_raid_csv(args.csv, decoding=args.decoding, repetition_penalty=args.rep_penalty)
    df = data.build_dataset(raw, max_docs=args.max_docs, max_words=args.max_words,
                            min_words=args.min_words, seed=args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "dataset.csv", index=False)
    data.save_manifest(df, out / "split_manifest.json")
    summ = data.summary(df)
    summ.to_csv(out / "summary.csv")
    print(summ.to_string())
    print(f"\nWrote {len(df)} rows to {out}/dataset.csv")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("inspect")
    i.add_argument("--csv", required=True)
    i.add_argument("--chunksize", type=int, default=200_000)
    b = sub.add_parser("build")
    g = b.add_mutually_exclusive_group(required=True)
    g.add_argument("--csv")
    g.add_argument("--hf", action="store_true")
    g.add_argument("--synthetic", action="store_true")
    b.add_argument("--decoding", default=None, help="e.g. greedy or sampling (machine rows only)")
    b.add_argument("--rep-penalty", default=None, help="e.g. yes or no (machine rows only)")
    b.add_argument("--max-docs", type=int, default=1000)
    b.add_argument("--max-words", type=int, default=256)
    b.add_argument("--min-words", type=int, default=50)
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--out", default="data/processed")
    args = p.parse_args()
    {"inspect": cmd_inspect, "build": cmd_build}[args.cmd](args)


if __name__ == "__main__":
    main()
