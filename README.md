# perturb-lora-bench

LoRA post-training of a single-cell foundation model (scGPT, loaded through
[helical](https://github.com/helicalAI/helical)) to predict transcriptional
responses to genetic perturbations, evaluated against baselines that are hard
to beat.

> **Status:** work in progress. Environment, LoRA wiring and test scaffolding are
> done; data loading, baselines and training are next.

## Task

Given control K562 cells and a perturbation label (the gene targeted by
CRISPRa/CRISPRi), predict the mean post-perturbation expression profile.
Evaluation is on **perturbations never seen during training**.

| Dataset | Perturbations | Role |
|---|---|---|
| Adamson et al. 2016 (CRISPRi) | single | pipeline smoke test |
| Norman et al. 2019 (CRISPRa) | single + pairs | main benchmark |

Data and splits come from the GEARS data loaders (the "simulation" split), so
numbers are comparable with GEARS/scGPT papers.

## Method

- **Backbone:** scGPT whole-human checkpoint (12 layers, d=512, 8 heads), frozen.
- **Adapters:** LoRA via HuggingFace PEFT, r=8, α=16, dropout 0.05.
- **Head:** a learned perturbation-flag embedding added to the gene tokens, plus
  a per-gene expression decoder (both trained from scratch).

### Why LoRA targets `self_attn`, not `q_proj`/`v_proj`

scGPT uses `torch.nn.MultiheadAttention`, which stores Q, K and V as one stacked
`in_proj_weight` (1536 × 512). There are no separate query/value modules, so
PEFT adapts the attention block as a unit: the stacked QKV projection plus
`out_proj`. At r=8 that is 294,912 trainable adapter parameters (12 × [8·2048 + 8·1024]).

### A silent failure mode worth knowing about

In eval mode under `torch.no_grad()`, PyTorch's `TransformerEncoderLayer` takes a
fused "fast path" that reads `in_proj_weight` directly and **bypasses the LoRA
wrapper**. Evaluation then scores the base model while reporting it as the
fine-tuned one. We disable the fast path (`torch.backends.mha.set_fastpath_enabled(False)`)
and `tests/test_lora.py::test_lora_active_in_eval_mode` guards against regressions.

## Evaluation

Per held-out perturbation, on mean profiles; then averaged:

- **Pearson Δ**: correlation of predicted vs. true *change from control*
  (all genes, and top-20 DE genes). This is the headline metric.
- **MSE**: all genes, and top-20 DE genes.
- Raw-expression Pearson is reported only to show why it misleads: predicting
  "no change" already scores r ≈ 0.99.

Baselines:

1. **No change**: return the control profile.
2. **Training mean**: control + mean training perturbation effect; for Norman
   pairs, **additive**: control + Δ(A) + Δ(B).
3. **scGPT, no fine-tuning**: frozen embeddings + linear head.
4. **scGPT + LoRA** (this work).

Model selection uses the validation split only. The test split is touched once.

## Results

_To be filled._

| Model | Pearson Δ (all) | Pearson Δ (DE20) | MSE (all) | MSE (DE20) |
|---|---|---|---|---|
| No change | n/a | n/a | | |
| Training mean / additive | | | | |
| scGPT frozen + linear | | | | |
| scGPT + LoRA r=8 | | | | |

## Reproduce

```bash
conda env create -f environment.yml -p /path/to/envs/plb
conda activate /path/to/envs/plb
make test          # fast unit tests, no data or GPU needed
make all           # data -> baselines -> train -> evaluate -> figures
```

`requirements.lock` pins every package (Linux x86_64, Python 3.11, torch 2.7.0 + CUDA 12.6).
Seeds are fixed in each config under `configs/`; runs are tracked in Weights & Biases.

## What I'd do next, and what I don't trust here

_To be written once results exist._

## License

MIT
