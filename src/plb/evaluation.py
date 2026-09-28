"""Score predicted mean profiles and aggregate results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from plb.data import PerturbationSummary
from plb.metrics import evaluate_perturbation

RESULTS_DIR = Path("results")
MAIN_METRICS = ["pearson_delta_de20", "pearson_delta_all", "mse_de20", "mse_all"]


def subgroup_of(pert: str, summary: PerturbationSummary) -> str:
    """GEARS test subgroup (e.g. combo_seen1, unseen_single) or 'single'."""
    for group, perts in (summary.test_subgroup or {}).items():
        if pert in perts:
            return group
    return "single"


def score(preds: dict[str, np.ndarray], summary: PerturbationSummary, model: str) -> pd.DataFrame:
    """One row of metrics per perturbation."""
    rows = []
    for p, pred in preds.items():
        m = evaluate_perturbation(pred, summary.means[p], summary.ctrl, summary.de_idx[p])
        rows.append({"model": model, "perturbation": p, "subgroup": subgroup_of(p, summary), **m})
    return pd.DataFrame(rows)


def save_run(df: pd.DataFrame, preds: dict[str, np.ndarray], dataset: str, model: str) -> Path:
    """Write per-perturbation metrics (csv) and predictions (npz) under results/<dataset>/."""
    out = RESULTS_DIR / dataset
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{model}.csv", index=False)
    np.savez_compressed(out / f"{model}_pred.npz", **preds)
    return out / f"{model}.csv"


def summarise_runs(dataset: str, by_subgroup: bool = False) -> pd.DataFrame:
    """Mean and standard error over perturbations for every saved run."""
    files = sorted((RESULTS_DIR / dataset).glob("*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files if not f.stem.startswith("summary")], ignore_index=True)
    keys = ["model", "subgroup"] if by_subgroup else ["model"]
    g = df.groupby(keys)[MAIN_METRICS]
    out = g.mean().join(g.sem(), rsuffix="_sem").join(g.size().rename("n"))
    return out.reset_index()


def markdown_table(summary: pd.DataFrame) -> str:
    """Results table as markdown, mean ± s.e.m."""
    cols = ["Model", "Pearson Δ top-20 DE", "Pearson Δ all", "MSE top-20 DE", "MSE all", "n"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in summary.iterrows():
        cells = [r["model"]]
        for m in MAIN_METRICS:
            v, s = r[m], r[f"{m}_sem"]
            cells.append("n/a" if np.isnan(v) else f"{v:.3f} ± {s:.3f}")
        cells.append(str(int(r["n"])))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
