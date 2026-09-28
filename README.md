# perturb-lora-bench

Can a single-cell foundation model, lightly fine-tuned, predict how cells respond
to a genetic perturbation it has never seen? This repo fine-tunes scGPT [1] with
LoRA [2] on Perturb-seq data and compares it against simple baselines.

Work in progress. The environment, LoRA setup and metrics are done and tested.
Data loading, baselines and training come next.

## The problem

In a Perturb-seq experiment, each cell gets a CRISPR guide that switches one
gene off (CRISPRi) or on (CRISPRa), and is then profiled with single-cell
RNA-seq. The task here: given the expression of control cells and the name of
the targeted gene, predict the average expression profile of perturbed cells.
The test set contains only perturbations the model never saw during training.

Most genes barely change after any single perturbation. So a model that
predicts "nothing changes" already matches the true profile closely on raw
expression, and a model can look good on that metric while learning nothing
about the perturbation. Ahlmann-Eltze et al. [3] showed that published deep
learning models, including scGPT, often do no better than simple additive or
mean baselines on this task. Any claim of improvement needs those baselines
next to it, and a metric that looks at what changed.

## Data

| Dataset | Cells | Perturbations | Used for |
|---|---|---|---|
| Adamson et al. 2016 [4] | K562, CRISPRi | single genes | debugging the pipeline |
| Norman et al. 2019 [5] | K562, CRISPRa | single genes and pairs | main results |

Both are loaded through the GEARS data loaders [6], using its "simulation"
split, so the numbers can be compared with GEARS and scGPT.

## Model

scGPT [1] is a transformer pretrained on over 33 million human cells, and I
load it through Helical's `helical` package. Each cell is read as a set of gene
tokens with expression values.

LoRA [2] freezes the pretrained weights and trains a small low-rank update
next to selected weight matrices, so only about 1% of the parameters change.
For a frozen matrix W it learns two thin matrices A and B and computes
`W x + (alpha / r) * B A x`, with rank r = 8 and alpha = 16 here. B starts at
zero, so training begins from exactly the pretrained model.

On top of the frozen encoder with LoRA sit two new pieces trained from scratch:
an embedding that marks which gene was perturbed, and a decoder that outputs
expression for each gene.

### Where the adapters go

The brief I started from said to put LoRA on the query and value projections
(`q_proj`, `v_proj`), the usual choice for language models. scGPT doesn't
have those layers. It uses PyTorch's `nn.MultiheadAttention`, which stores the
query, key and value weights in one stacked 1536 × 512 matrix. PEFT therefore
adapts the attention block as a whole: the stacked matrix plus the output
projection. At r = 8 that comes to 294,912 trainable parameters across the 12
layers.

### A bug that would have invalidated the results

In eval mode, PyTorch runs `TransformerEncoderLayer` through a fused fast path
that reads the attention weights directly and never calls the LoRA wrapper.
The fine-tuned model then produces exactly the base model's output, with no
error or warning. I found this by setting the LoRA weights to random values
and checking that eval output changed; it didn't. The fix is
`torch.backends.mha.set_fastpath_enabled(False)`, and
`tests/test_lora.py::test_lora_active_in_eval_mode` keeps it from coming back.

## Evaluation

Every metric is computed per test perturbation on the mean expression profile,
then averaged over perturbations.

The main metric is Pearson correlation of the delta: predicted change from
control against true change from control, over all genes and over the 20 most
differentially expressed genes. I also report MSE on the same two gene sets.
Raw-expression correlation is included only to show how misleading it is.

Baselines:

1. No change: predict the control profile.
2. Training mean: add the average effect of all training perturbations. For
   Norman pairs, the additive baseline adds the two single-gene effects.
3. scGPT without fine-tuning: frozen embeddings with a linear head.
4. scGPT with LoRA.

Hyperparameters and early stopping use the validation split. The test split is
scored once, at the end.

## Results

Not yet run.

| Model | Pearson Δ, all genes | Pearson Δ, top-20 DE | MSE, all genes | MSE, top-20 DE |
|---|---|---|---|---|
| No change | n/a | n/a | | |
| Training mean / additive | | | | |
| scGPT, frozen + linear | | | | |
| scGPT + LoRA, r = 8 | | | | |

Pearson Δ is undefined for "no change" because its predicted delta is zero
everywhere.

## Reproducing

```bash
conda env create -f environment.yml -p /path/to/envs/plb
conda activate /path/to/envs/plb
make test    # unit tests, no data or GPU needed
make all     # download data, run baselines, train, evaluate, plot
```

`requirements.lock` pins every package for Linux, Python 3.11, and torch 2.7.0
with CUDA 12.6. Seeds and settings live in `configs/`, and training runs are
logged to Weights & Biases.

## What I'd do next, and what I don't trust

To be written once there are results.

## References

1. Cui H, Wang C, Maan H, et al. scGPT: toward building a foundation model for
   single-cell multi-omics using generative AI. *Nature Methods* 21, 1470–1480 (2024).
2. Hu EJ, Shen Y, Wallis P, et al. LoRA: Low-rank adaptation of large language
   models. *ICLR* (2022).
3. Ahlmann-Eltze C, Huber W, Anders S. Deep-learning-based gene perturbation
   effect prediction does not yet outperform simple linear baselines.
   *Nature Methods* (2025). doi:10.1038/s41592-025-02772-6
4. Adamson B, Norman TM, Jost M, et al. A multiplexed single-cell CRISPR
   screening platform enables systematic dissection of the unfolded protein
   response. *Cell* 167, 1867–1882 (2016).
5. Norman TM, Horlbeck MA, Replogle JM, et al. Exploring genetic interaction
   manifolds constructed from rich single-cell phenotypes. *Science* 365,
   786–793 (2019).
6. Roohani Y, Huang K, Leskovec J. Predicting transcriptional outcomes of novel
   multigene perturbations with GEARS. *Nature Biotechnology* 42, 927–935 (2024).

## License

MIT
