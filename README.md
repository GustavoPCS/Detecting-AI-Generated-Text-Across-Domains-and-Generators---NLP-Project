# Detecting AI-Generated Text Across Domains and Generators

CS 59300 NLP project, Purdue University Fort Wayne (Shreya Sagar, Gustavo Pereira Corsi da Silva).

This repository contains an NLP research project investigating how well AI-generated text detectors generalize when the text domain or generating model changes. Using controlled human-versus-machine datasets from scientific abstracts, news, reviews, and Reddit posts, the project evaluates classical TF–IDF classifiers, fine-tuned transformer detectors such as RoBERTa, and zero-shot methods including Fast-DetectGPT.

The evaluation emphasizes realistic distribution shifts: cross-domain transfer, generator shift from Mistral Chat to Llama Chat, and combined domain-and-generator shift. Experiments use fixed source-level train/validation/test splits, report macro-F1, AUROC, false-positive rate, confidence intervals, training and inference time, and GPU-memory use.

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Tests
```bash
python -m pytest -q
python -m src.build_dataset build --synthetic --out data/processed_synth   # no download needed
```

## Data pipeline (PR 1)

**Source:** RAID (non-adversarial `train_none.csv`), domains `abstracts`, `news`, `reviews`, `reddit`. Generators: G1 = `mistral-chat`, G2 = `llama-chat`.

**Shared decoding setting:** greedy decoding, no repetition penalty (`--decoding greedy --rep-penalty no`). Chosen because it is deterministic and available for both generators in all four domains. Sampling is a candidate robustness check later.

**Build the dataset**
```bash
mkdir data\raw
curl.exe -L -o data\raw\train_none.csv https://dataset.raid-bench.xyz/train_none.csv
python -m src.build_dataset inspect --csv data\raw\train_none.csv
python -m src.build_dataset build --csv data\raw\train_none.csv --decoding greedy --rep-penalty no
```
Outputs in `data/processed/`: `dataset.csv` (git-ignored), `split_manifest.json`, `summary.csv`. The seed is fixed (42), so the splits are reproducible.

**Use it from code**
```python
from src.data import load_split
train = load_split("news", "mistral-chat", "train")  # text, label (0 human / 1 machine), source_id, ...
```

**Pipeline:** word cap 256, minimum 50 words, exact + prefix near-duplicate removal, up to 1,000 source documents per domain, 70/15/15 split by source document (a human text and all its generations stay in one partition), then human/machine balancing by word-count bin within each (domain, generator, split).

### Resulting size (texts per class; human count = machine count)

| Domain | Generator | Train | Val | Test |
|---|---|---|---|---|
| abstracts | mistral-chat | 342 | 77 | 64 |
| abstracts | llama-chat | 244 | 55 | 46 |
| news | mistral-chat | 646 | 133 | 138 |
| news | llama-chat | 617 | 133 | 134 |
| reddit | mistral-chat | 453 | 93 | 99 |
| reddit | llama-chat | 375 | 79 | 85 |
| reviews | mistral-chat | 423 | 90 | 108 |
| reviews | llama-chat | 463 | 90 | 87 |

Total: 10,148 rows.

### Known issues
- **Reviews:** RAID has only 943 non-adversarial reviews source documents, below our 1,000 target.
- **Abstracts attrition:** abstracts keep only about a third of their source documents after filtering. The cause is not yet diagnosed (candidates: 50-word minimum, deduplication, length matching).
- **Small test sets:** abstracts have only 46-64 human test texts per generator, so FPR resolution is about 2% and bootstrap CIs will be wide. Low-FPR results must report this.
- **Human texts differ by generator:** length matching is run separately per generator, so the human test set for G1 and G2 is not identical. Planned fix: one fixed human set per domain and split.
- **Near-duplicate detection** is prefix-based, not MinHash.
- Only greedy / no-repetition-penalty is built so far; the MAGE external check is not included yet.

## GPU initial results

Install the GPU dependencies in a CUDA environment (choose the matching PyTorch CUDA wheel if the standard install does not provide CUDA):

```bash
pip install -r requirements-gpu.txt
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / 2**30)"
```

Fine-tune one RoBERTa-base detector per domain on G1 (Mistral Chat). This evaluates both the in-generator condition and the required same-domain G1-to-G2 generator shift. Every run writes `hardware.json`, predictions, metrics, and `runtime.csv` with training time, inference milliseconds per 1,000 texts, and peak GPU memory.

```bash
# Initial one-seed results
python -m src.finetune_roberta --seeds 42 --out results/roberta_seed42

# Three-seed follow-up
python -m src.finetune_roberta --seeds 42 43 44 --out results/roberta_3seeds
```

Run the bounded Fast-DetectGPT feasibility check with the fixed `EleutherAI/gpt-neo-2.7B` checkpoint and FP16. The first command checks whether the model fits before scoring; the second scores a deterministic, balanced 32-text news test pilot and records AUROC and runtime telemetry. It uses the analytic discrepancy statistic and is explicitly labeled a pilot because the same checkpoint is used as scoring and reference model.

```bash
python -m src.fast_detectgpt_pilot --precision fp16 --memory-check-only
python -m src.fast_detectgpt_pilot --domain news --generator mistral-chat --split test --max-texts 32 --precision fp16
```
