"""README figures: mean delta correlation per model, and predicted vs. true delta for example perturbations."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLOR = "#4c72b0"


def bar_chart(dataset: str, results: Path, out: Path) -> None:
    """Mean top-20 DE delta Pearson per model, with standard error."""
    df = pd.concat([pd.read_csv(f) for f in sorted((results / dataset).glob("*.csv"))])
    df = df[df.model != "no_change"]
    g = df.groupby("model").pearson_delta_de20
    stats = pd.DataFrame({"mean": g.mean(), "sem": g.sem()}).sort_values("mean")
    labels = [m.removeprefix(f"{dataset}_") for m in stats.index]

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.barh(labels, stats["mean"], xerr=stats["sem"], color=COLOR)
    ax.set_xlabel("Pearson Δ, top-20 DE genes")
    ax.set_title(dataset)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / f"delta_corr_{dataset}.png", dpi=150)
    plt.close(fig)


def gain_over_baseline(dataset: str, results: Path, out: Path, baseline: str = "train_mean") -> None:
    """Per-model gain in top-20 DE delta Pearson over the baseline, paired by perturbation."""
    base = pd.read_csv(results / dataset / f"{baseline}.csv").set_index("perturbation").pearson_delta_de20
    rows = []
    for f in sorted((results / dataset).glob(f"{dataset}_*.csv")):
        d = pd.read_csv(f).set_index("perturbation").pearson_delta_de20 - base
        rows.append((f.stem.removeprefix(f"{dataset}_"), d.mean(), d.std() / np.sqrt(d.notna().sum())))
    rows.sort(key=lambda r: r[1])

    fig, ax = plt.subplots(figsize=(7, 0.55 * len(rows) + 1.2))
    for y, (name, m, se) in enumerate(rows):
        color = COLOR if name.startswith("lora") else "#9aa0a6"
        ax.errorbar(m, y, xerr=se, fmt="o", color=color, ms=7, capsize=3, lw=1.5)
        ax.text(m + se + 0.005, y, f"{m:+.3f}", va="center", fontsize=8, color="#444")
    ax.axvline(0, color="#bbbbbb", lw=1)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlabel(f"Gain in Pearson Δ, top-20 DE genes, over {baseline}\n(mean ± s.e.m., paired by perturbation)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(dataset, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out / f"gain_over_baseline_{dataset}.png", dpi=150)
    plt.close(fig)


def examples(dataset: str, results: Path, out: Path, model: str, n: int = 3) -> None:
    """Predicted vs. true Δ on top-20 DE genes for the n test perturbations with the largest effects."""
    from plb.data import load_pertdata, summarise

    pd_ = load_pertdata(dataset)
    s = summarise(pd_.adata, pd_.set2conditions, pd_.subgroup)
    pred = np.load(results / dataset / f"{model}_pred.npz")
    size = {p: np.abs(s.means[p] - s.ctrl)[s.de_idx[p]].mean() for p in s.split["test"]}
    top = sorted(size, key=size.get, reverse=True)[:n]

    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    for ax, p in zip(axes, top):
        idx = s.de_idx[p]
        true = (s.means[p] - s.ctrl)[idx]
        guess = (pred[p] - s.ctrl)[idx]
        lim = max(np.abs(true).max(), np.abs(guess).max()) * 1.1
        ax.plot([-lim, lim], [-lim, lim], color="lightgray")
        ax.scatter(true, guess, s=15, color=COLOR)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title(p)
        ax.set_xlabel("true Δ")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("predicted Δ")
    fig.suptitle(f"{dataset}, {model.removeprefix(f'{dataset}_')}, top-20 DE genes")
    fig.tight_layout()
    fig.savefig(out / f"examples_{dataset}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("figures"))
    ap.add_argument("--datasets", nargs="+", default=["adamson", "norman"])
    ap.add_argument("--model", default="lora_r8")
    args = ap.parse_args()
    args.out.mkdir(exist_ok=True)
    for d in args.datasets:
        if not (args.results / d).exists():
            continue
        bar_chart(d, args.results, args.out)
        gain_over_baseline(d, args.results, args.out)
        examples(d, args.results, args.out, f"{d}_{args.model}")


if __name__ == "__main__":
    main()
