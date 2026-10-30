"""Bounded Fast-DetectGPT pilot using GPT-Neo-2.7B.

The score is the analytic discrepancy from Fast-DetectGPT: at each token, the
observed log-probability is compared with its expected value and variance under
the reference model distribution.  GPT-Neo-2.7B is deliberately fixed as both
scoring and reference checkpoint for this initial feasibility run.  This is a
pilot, not a substitute for the final two-model Fast-DetectGPT configuration.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_split
from src.gpu_metrics import peak_memory_gib, require_torch, reset_peak, write_hardware
from src.metrics import auroc

CHECKPOINT = "EleutherAI/gpt-neo-2.7B"


def choose_dtype(torch, precision):
    if precision == "fp16": return torch.float16
    if precision == "bf16": return torch.bfloat16
    return torch.float32


def fast_detectgpt_score(torch, model, tokens):
    """One document score; higher scores indicate text more likely produced by the LM."""
    output = model(input_ids=tokens, attention_mask=torch.ones_like(tokens))
    # Predict token t from the prefix ending at t-1.
    log_probs = torch.log_softmax(output.logits[:, :-1].float(), dim=-1)
    observed = tokens[:, 1:]
    token_logp = log_probs.gather(-1, observed.unsqueeze(-1)).squeeze(-1)
    probs = log_probs.exp()
    mean = (probs * log_probs).sum(-1)
    variance = (probs * (log_probs - mean.unsqueeze(-1)).square()).sum(-1).clamp_min(1e-8)
    # Standardize the sum rather than individual positions, as in the analytic test statistic.
    return ((token_logp - mean).sum(-1) / variance.sum(-1).sqrt()).item()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/dataset.csv")
    p.add_argument("--domain", default="news"); p.add_argument("--generator", default="mistral-chat")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--max-texts", type=int, default=32, help="Bounded balanced pilot batch; 0 means all rows.")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp16")
    p.add_argument("--out", default="results/fast_detectgpt_pilot")
    p.add_argument("--memory-check-only", action="store_true", help="Load model, report peak memory, and exit.")
    a = p.parse_args(argv)
    if a.max_texts and a.max_texts < 2:
        p.error("--max-texts must be 0 (all rows) or at least 2 so both classes are represented")
    torch = require_torch()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if a.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support bf16; use --precision fp16.")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); write_hardware(out / "hardware.json", torch)
    dtype = choose_dtype(torch, a.precision)
    reset_peak(torch); started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, torch_dtype=dtype).to("cuda").eval()
    load_seconds, load_peak = time.perf_counter() - started, peak_memory_gib(torch)
    config = {"method": "fast_detectgpt_analytic_pilot", "checkpoint": CHECKPOINT,
              "reference_checkpoint": CHECKPOINT, "precision": a.precision, "max_length": a.max_length,
              "load_seconds": load_seconds, "peak_gpu_memory_gib_load": load_peak}
    if a.memory_check_only:
        (out / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
        print(json.dumps(config, indent=2)); return
    rows = load_split(a.domain, a.generator, a.split, a.data)
    if a.max_texts:
        # Keep both classes and make the pilot deterministic/balanced.
        n = min(a.max_texts // 2, *(len(rows[rows.label == label]) for label in (0, 1)))
        rows = pd.concat([rows[rows.label == label].sample(n, random_state=42) for label in (0, 1)], ignore_index=True)
    reset_peak(torch); start = time.perf_counter(); scores = []
    with torch.inference_mode():
        for text in rows.text:
            ids = tokenizer(text, truncation=True, max_length=a.max_length, return_tensors="pt").input_ids.to("cuda")
            scores.append(fast_detectgpt_score(torch, model, ids))
    seconds, peak = time.perf_counter() - start, peak_memory_gib(torch)
    pred = rows[["source_id", "label", "domain", "generator", "split"]].copy()
    pred["score"] = scores; pred.insert(2, "method", "fast_detectgpt_analytic_pilot")
    pred.to_csv(out / "predictions.csv", index=False)
    config.update({"n_texts": len(pred), "auroc": auroc(pred.label, pred.score), "inference_seconds": seconds,
                   "inference_ms_per_1000_texts": 1e6 * seconds / len(pred), "peak_gpu_memory_gib_inference": peak})
    (out / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config, indent=2)); print(f"Wrote pilot outputs to {out}/")


if __name__ == "__main__":
    main()
