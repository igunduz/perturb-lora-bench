"""README figures: delta correlation by model, and predicted vs. true delta for example perturbations."""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
INK, MUTED, MARK, GRID = "#1f2328", "#6e7781", "#2f6fb0", "#d8dee4"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False})


def delta_by_model(dataset: str, results: Path, out: Path) -> None:
    """Per-perturbation top-20 DE delta Pearson for every model, ordered by mean."""
    df = pd.concat([pd.read_csv(f) for f in sorted((results / dataset).glob("*.csv"))])
    df = df[df.model != "no_change"]
    order = df.groupby("model").pearson_delta_de20.mean().sort_values().index
    fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(order) + 1.2))
    rng = np.random.default_rng(0)
    for i, m in enumerate(order):
        v = df.loc[df.model == m, "pearson_delta_de20"].dropna()
        ax.scatter(v, i + rng.uniform(-0.15, 0.15, len(v)), s=10, color=MARK, alpha=0.35, linewidths=0)
        ax.plot([v.mean()] * 2, [i - 0.3, i + 0.3], color=INK, lw=2)
    ax.set_yticks(range(len(order)), [m.removeprefix(f"{dataset}_") for m in order])
    ax.axvline(0, color=GRID, lw=1, zorder=0)
    ax.set_xlabel("Pearson Δ, top-20 DE genes")
    ax.set_title(f"{dataset}: dots are test perturbations, bars are means", fontsize=9, color=MUTED, loc="left")
    fig.tight_layout()
    fig.savefig(out / f"delta_corr_{dataset}.png", dpi=200)
    plt.close(fig)


def examples(dataset: str, results: Path, out: Path, models: list[str], n: int = 3) -> None:
    """Predicted vs. true Δ on top-20 DE genes for the n test perturbations with the largest effects."""
    from plb.data import load_pertdata, summarise

    pd_ = load_pertdata(dataset)
    s = summarise(pd_.adata, pd_.set2conditions, pd_.subgroup)
    models = [m for m in models if (results / dataset / f"{m}_pred.npz").exists()]
    preds = {m: np.load(results / dataset / f"{m}_pred.npz") for m in models}
    size = {p: np.abs(s.means[p] - s.ctrl)[s.de_idx[p]].mean() for p in s.split["test"]}
    top = sorted(size, key=size.get, reverse=True)[:n]
    fig, axes = plt.subplots(n, len(models), figsize=(2.4 * len(models), 2.3 * n), squeeze=False)
    for r, p in enumerate(top):
        idx = s.de_idx[p]
        true = (s.means[p] - s.ctrl)[idx]
        lim = np.abs(true).max() * 1.15
        for c, m in enumerate(models):
            ax = axes[r, c]
            pred = (preds[m][p] - s.ctrl)[idx]
            ax.plot([-lim, lim], [-lim, lim], color=GRID, lw=1, zorder=0)
            ax.scatter(true, pred, s=14, color=MARK)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            if r == 0:
                ax.set_title(m.removeprefix(f"{dataset}_"), fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{p}\npredicted Δ", fontsize=8)
            if r == n - 1:
                ax.set_xlabel("true Δ", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"examples_{dataset}.png", dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("figures"))
    ap.add_argument("--datasets", nargs="+", default=["adamson", "norman"])
    ap.add_argument("--models", nargs="+", default=["additive", "frozen", "lora_r8", "random_lora_r8"])
    args = ap.parse_args()
    args.out.mkdir(exist_ok=True)
    for d in args.datasets:
        if not (args.results / d).exists():
            continue
        delta_by_model(d, args.results, args.out)
        examples(d, args.results, args.out, [m if m in ("additive", "train_mean") else f"{d}_{m}" for m in args.models])


if __name__ == "__main__":
    main()
