"""Per-perturbation metrics on mean profiles; delta = profile minus control mean."""

from __future__ import annotations

import numpy as np


def _subset(*arrays: np.ndarray, idx: np.ndarray | None):
    """Subset arrays to genes in idx."""
    if idx is None:
        return arrays
    return tuple(a[idx] for a in arrays)


def mse(pred: np.ndarray, true: np.ndarray, idx: np.ndarray | None = None) -> float:
    """MSE over all genes or genes in idx."""
    pred, true = _subset(np.asarray(pred, float), np.asarray(true, float), idx=idx)
    return float(np.mean((pred - true) ** 2))


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r; NaN if either input is constant."""
    x = np.asarray(x, float) - np.mean(x)
    y = np.asarray(y, float) - np.mean(y)
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom == 0:
        return float("nan")
    return float(np.sum(x * y) / denom)


def pearson_delta(
    pred: np.ndarray, true: np.ndarray, ctrl: np.ndarray, idx: np.ndarray | None = None
) -> float:
    """Pearson r of predicted vs. observed change from control."""
    d_pred, d_true = _subset(
        np.asarray(pred, float) - ctrl, np.asarray(true, float) - ctrl, idx=idx
    )
    return pearson(d_pred, d_true)


def evaluate_perturbation(
    pred: np.ndarray, true: np.ndarray, ctrl: np.ndarray, de_idx: np.ndarray
) -> dict[str, float]:
    """All reported metrics for one perturbation."""
    return {
        "mse_all": mse(pred, true),
        "mse_de20": mse(pred, true, de_idx),
        "pearson_delta_all": pearson_delta(pred, true, ctrl),
        "pearson_delta_de20": pearson_delta(pred, true, ctrl, de_idx),
        "pearson_raw_all": pearson(pred, true),
    }
