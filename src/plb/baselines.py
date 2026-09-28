"""Non-learned baselines: no change, training mean, additive."""

from __future__ import annotations

import numpy as np


def no_change(ctrl: np.ndarray, test_perts: list[str]) -> dict[str, np.ndarray]:
    raise NotImplementedError


def train_mean(ctrl: np.ndarray, train_means: dict[str, np.ndarray], test_perts: list[str]) -> dict[str, np.ndarray]:
    raise NotImplementedError


def additive(ctrl: np.ndarray, train_means: dict[str, np.ndarray], test_perts: list[str]) -> dict[str, np.ndarray]:
    raise NotImplementedError
