"""Evaluation metrics. The contract is defined by tests/test_metrics.py; implement to make those pass.

Conventions
-----------
All metrics are computed per held-out perturbation on *mean* profiles
(mean over cells), then averaged over perturbations:

    ctrl : (n_genes,)  mean expression of control cells
    true : (n_genes,)  mean expression of cells with this perturbation
    pred : (n_genes,)  model's predicted mean expression

"Delta" means the change from control: pred - ctrl, true - ctrl.
`de_idx` are the indices of the top-20 differentially expressed genes for this
perturbation (from GEARS' precomputed `top_non_dropout_de_20`).

Judgment call: for the "no change" baseline, pred - ctrl is all zeros, so the
delta Pearson correlation is undefined (0/0). We return NaN and report it as
"n/a" rather than silently substituting 0.
"""

from __future__ import annotations

import numpy as np


def mse(pred: np.ndarray, true: np.ndarray, idx: np.ndarray | None = None) -> float:
    """Mean squared error over all genes, or only the genes in `idx`."""
    raise NotImplementedError


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r; NaN if either vector is constant."""
    raise NotImplementedError


def pearson_delta(
    pred: np.ndarray, true: np.ndarray, ctrl: np.ndarray, idx: np.ndarray | None = None
) -> float:
    """Pearson r between predicted and true change from control (optionally on genes `idx`)."""
    raise NotImplementedError


def evaluate_perturbation(
    pred: np.ndarray, true: np.ndarray, ctrl: np.ndarray, de_idx: np.ndarray
) -> dict[str, float]:
    """All headline metrics for one perturbation.

    Returns keys: mse_all, mse_de20, pearson_delta_all, pearson_delta_de20, pearson_raw_all.
    """
    raise NotImplementedError
