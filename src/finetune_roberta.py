"""Fine-tune RoBERTa-base on the project's fixed splits and record GPU costs.

One model is fit per training domain using G1 (mistral-chat), then evaluated on
every requested domain and generator. Prediction CSVs follow ``src.evaluate``.
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import evaluate as E
from src.data import DOMAINS
from src.gpu_metrics import peak_memory_gib, require_torch, reset_peak, write_hardware

MODEL = "roberta-base"


def seed_everything(torch, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def select(df, domain, generator, split):
    return df[(df.domain == domain) & (df.generator == generator) & (df.split == split)].reset_index(drop=True)


def prediction_frame(rows, scores, train_domain, train_generator, seed, split):
    return pd.DataFrame({
        "source_id": rows.source_id, "label": rows.label, "score": np.asarray(scores),
        "method": f"roberta_base_seed{seed}", "train_domain": train_domain,
        "train_generator": train_generator, "domain": rows.domain, "generator": rows.generator,
        "split": split,
    })


def batched_scores(torch, model, tokenizer, rows, batch_size, max_length):
    model.eval(); scores = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows.text.iloc[start:start + batch_size].tolist()
            x = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            logits = model(**{k: v.to("cuda") for k, v in x.items()}).logits
            scores.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().tolist())
    return scores


def run(df, seeds, domains=DOMAINS, train_generator="mistral-chat", test_generators=None,
        model_name=MODEL, epochs=3, batch_size=16, learning_rate=2e-5, max_length=256):
    torch = require_torch()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    test_generators = test_generators or [train_generator]
    vals, tests, runtimes = [], [], []
    for seed in seeds:
        for train_domain in domains:
            train, val = select(df, train_domain, train_generator, "train"), select(df, train_domain, train_generator, "val")
            if train.empty or val.empty:
                print(f"[warn] missing {train_domain}/{train_generator}; skipping")
                continue
            seed_everything(torch, seed)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to("cuda")
            opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)
            steps = epochs * int(np.ceil(len(train) / batch_size))
            schedule = get_linear_schedule_with_warmup(opt, 0, steps)
            reset_peak(torch); began = time.perf_counter(); model.train()
            for _ in range(epochs):
                order = np.random.permutation(len(train))
                for ix in range(0, len(order), batch_size):
                    rows = train.iloc[order[ix:ix + batch_size]]
                    x = tokenizer(rows.text.tolist(), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
                    out = model(**{k: v.to("cuda") for k, v in x.items()}, labels=torch.tensor(rows.label.to_numpy(), device="cuda"))
                    out.loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step(); schedule.step(); opt.zero_grad(set_to_none=True)
            train_seconds, train_peak = time.perf_counter() - began, peak_memory_gib(torch)

            reset_peak(torch); infer_start = time.perf_counter(); infer_n = 0
            vals.append(prediction_frame(val, batched_scores(torch, model, tokenizer, val, batch_size, max_length), train_domain, train_generator, seed, "val"))
            for domain in domains:
                for generator in test_generators:
                    test = select(df, domain, generator, "test")
                    if not test.empty:
                        tests.append(prediction_frame(test, batched_scores(torch, model, tokenizer, test, batch_size, max_length), train_domain, train_generator, seed, "test"))
                        infer_n += len(test)
            inference_seconds = time.perf_counter() - infer_start
            runtimes.append({"method": f"roberta_base_seed{seed}", "seed": seed, "train_domain": train_domain,
                             "train_generator": train_generator, "model": model_name, "epochs": epochs,
                             "batch_size": batch_size, "max_length": max_length, "n_train": len(train),
                             "train_seconds": train_seconds, "inference_ms_per_1000_texts": 1e6 * inference_seconds / infer_n,
                             "peak_gpu_memory_gib_train": train_peak, "peak_gpu_memory_gib_inference": peak_memory_gib(torch)})
            del model; torch.cuda.empty_cache()
    return {"val": pd.concat(vals, ignore_index=True), "test": pd.concat(tests, ignore_index=True), "runtime": pd.DataFrame(runtimes)}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/dataset.csv")
    p.add_argument("--out", default="results/roberta")
    p.add_argument("--seeds", type=int, nargs="+", default=[42], help="Use 42 43 44 for the three-seed run.")
    p.add_argument("--generator", default="mistral-chat")
    p.add_argument("--test-generators", nargs="+", default=["mistral-chat", "llama-chat"])
    p.add_argument("--domains", nargs="+", default=DOMAINS); p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16); p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=256); p.add_argument("--n-boot", type=int, default=1000)
    a = p.parse_args(argv); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch = require_torch(); write_hardware(out / "hardware.json", torch)
    res = run(pd.read_csv(a.data), a.seeds, a.domains, a.generator, a.test_generators, epochs=a.epochs,
              batch_size=a.batch_size, learning_rate=a.learning_rate, max_length=a.max_length)
    res["val"].to_csv(out / "predictions_val.csv", index=False); res["test"].to_csv(out / "predictions_test.csv", index=False)
    res["runtime"].to_csv(out / "runtime.csv", index=False)
    metrics = E.run(res["test"], res["val"], n_boot=a.n_boot, seed=a.seeds[0]); metrics.to_csv(out / "metrics.csv", index=False)
    print(metrics[["method", "train_domain", "domain", "generator", "n", "macro_f1", "auroc", "gap_macro_f1"]].round(3).to_string(index=False))
    print(f"Wrote results to {out}/")


if __name__ == "__main__":
    main()
