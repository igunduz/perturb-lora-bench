"""Non-learned baselines: no change, training mean, additive."""

from __future__ import annotations

import numpy as np

from plb.data import perturbed_genes


def no_change(ctrl: np.ndarray, perts: list[str]) -> dict[str, np.ndarray]:
    """Predict the control mean for every perturbation."""
    return {p: ctrl.copy() for p in perts}


def mean_delta(ctrl: np.ndarray, train_means: dict[str, np.ndarray]) -> np.ndarray:
    """Average change from control over training perturbations."""
    return np.mean([m - ctrl for m in train_means.values()], axis=0)


def train_mean(ctrl: np.ndarray, train_means: dict[str, np.ndarray], perts: list[str]) -> dict[str, np.ndarray]:
    """Control mean plus the average training effect."""
    d = mean_delta(ctrl, train_means)
    return {p: ctrl + d for p in perts}


def additive(ctrl: np.ndarray, train_means: dict[str, np.ndarray], perts: list[str]) -> dict[str, np.ndarray]:
    """Control plus the sum of single-gene training effects; unseen genes get the average effect."""
    fallback = mean_delta(ctrl, train_means)
    single = {perturbed_genes(c)[0]: m - ctrl for c, m in train_means.items() if len(perturbed_genes(c)) == 1}
    return {p: ctrl + sum(single.get(g, fallback) for g in perturbed_genes(p)) for p in perts}


BASELINES = {"no_change": None, "train_mean": train_mean, "additive": additive}


def predict_baseline(name: str, ctrl: np.ndarray, train_means: dict[str, np.ndarray], perts: list[str]):
    """Run a baseline by name."""
    if name == "no_change":
        return no_change(ctrl, perts)
    return BASELINES[name](ctrl, train_means, perts)
