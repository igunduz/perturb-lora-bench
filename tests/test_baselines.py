"""Baseline tests."""

import numpy as np

from plb.baselines import additive, no_change, train_mean

CTRL = np.zeros(4)
TRAIN = {"A+ctrl": np.array([1.0, 0, 0, 0]), "B+ctrl": np.array([0, 2.0, 0, 0])}


def test_no_change():
    assert np.array_equal(no_change(CTRL, ["C+ctrl"])["C+ctrl"], CTRL)


def test_train_mean():
    assert np.allclose(train_mean(CTRL, TRAIN, ["C+ctrl"])["C+ctrl"], [0.5, 1.0, 0, 0])


def test_additive_sums_single_effects():
    assert np.allclose(additive(CTRL, TRAIN, ["A+B"])["A+B"], [1.0, 2.0, 0, 0])


def test_additive_unseen_gene_uses_mean_effect():
    assert np.allclose(additive(CTRL, TRAIN, ["A+C"])["A+C"], [1.5, 1.0, 0, 0])
