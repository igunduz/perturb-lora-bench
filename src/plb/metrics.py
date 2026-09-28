"""Evaluation metrics. tests/test_metrics.py defines the expected behaviour.

Each metric is computed for one test perturbation on mean profiles (averaged
over cells) and then averaged over perturbations. Inputs are length-n_genes
vectors: `ctrl` is the control mean, `true` the observed perturbed mean, and
`pred` the prediction. The delta is the change from control (pred - ctrl,
true - ctrl). `de_idx` holds the top-20 differentially expressed genes for the
perturbation, taken from GEARS (`top_non_dropout_de_20`).

For the "no change" baseline the predicted delta is all zeros, so its delta
correlation is undefined. We return NaN and report it as n/a instead of
filling in 0.
"""

from __future__ import annotations

import numpy as np


def _subset(*arrays: np.ndarray, idx: np.ndarray | None):
    """Restrict every array to the genes in `idx` (no-op when idx is None)."""
    if idx is None:
        return arrays
    return tuple(a[idx] for a in arrays)


def mse(pred: np.ndarray, true: np.ndarray, idx: np.ndarray | None = None) -> float:
    """Mean squared error over all genes, or only the genes in `idx`.

    Raw and delta MSE are identical, since (pred - ctrl) - (true - ctrl) = pred - true.
    """
    pred, true = _subset(np.asarray(pred, float), np.asarray(true, float), idx=idx)
    return float(np.mean((pred - true) ** 2))


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r. Returns NaN when either vector is constant."""
    x = np.asarray(x, float) - np.mean(x)
    y = np.asarray(y, float) - np.mean(y)
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom == 0:
        return float("nan")
    return float(np.sum(x * y) / denom)


def pearson_delta(
    pred: np.ndarray, true: np.ndarray, ctrl: np.ndarray, idx: np.ndarray | None = None
) -> float:
    """Pearson r between predicted and observed change from control, optionally on genes `idx`."""
    d_pred, d_true = _subset(
        np.asarray(pred, float) - ctrl, np.asarray(true, float) - ctrl, idx=idx
    )
    return pearson(d_pred, d_true)


def evaluate_perturbation(
    pred: np.ndarray, true: np.ndarray, ctrl: np.ndarray, de_idx: np.ndarray
) -> dict[str, float]:
    """Every reported metric for one perturbation, as a dict:
    mse_all, mse_de20, pearson_delta_all, pearson_delta_de20, pearson_raw_all.
    """
    return {
        "mse_all": mse(pred, true),
        "mse_de20": mse(pred, true, de_idx),
        "pearson_delta_all": pearson_delta(pred, true, ctrl),
        "pearson_delta_de20": pearson_delta(pred, true, ctrl, de_idx),
        "pearson_raw_all": pearson(pred, true),
    }
