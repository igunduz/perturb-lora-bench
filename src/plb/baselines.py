"""Non-learned baselines. Each returns predicted mean profiles for the test perturbations.

1. no_change      : predict the control mean. Scores well on raw MSE/correlation
                    because most genes don't move; that's why it's included.
2. train_mean     : control mean + average delta over all training perturbations.
   additive       : for a combo A+B, control + delta(A) + delta(B), using the
                    single-perturbation deltas seen in training (Norman only).
3. (scGPT zero-shot / frozen-embedding + linear head lives in model.py.)
"""

from __future__ import annotations

import numpy as np


def no_change(ctrl: np.ndarray, test_perts: list[str]) -> dict[str, np.ndarray]:
    raise NotImplementedError


def train_mean(ctrl: np.ndarray, train_means: dict[str, np.ndarray], test_perts: list[str]) -> dict[str, np.ndarray]:
    raise NotImplementedError


def additive(ctrl: np.ndarray, train_means: dict[str, np.ndarray], test_perts: list[str]) -> dict[str, np.ndarray]:
    raise NotImplementedError
