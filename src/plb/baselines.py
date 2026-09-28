"""Baselines that involve no learning. Each returns a predicted mean profile per test perturbation.

no_change:  the control mean. Because most genes don't move, this already
            scores well on raw expression, which is why it is in the table.
train_mean: the control mean plus the average effect of all training perturbations.
additive:   for a Norman pair A+B, the control mean plus the effect of A alone
            plus the effect of B alone, both taken from training.

The scGPT baseline without fine-tuning lives in model.py.
"""

from __future__ import annotations

import numpy as np


def no_change(ctrl: np.ndarray, test_perts: list[str]) -> dict[str, np.ndarray]:
    raise NotImplementedError


def train_mean(ctrl: np.ndarray, train_means: dict[str, np.ndarray], test_perts: list[str]) -> dict[str, np.ndarray]:
    raise NotImplementedError


def additive(ctrl: np.ndarray, train_means: dict[str, np.ndarray], test_perts: list[str]) -> dict[str, np.ndarray]:
    raise NotImplementedError
