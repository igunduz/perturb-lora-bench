"""Tests for plb.metrics, written before the implementation."""

import numpy as np
import pytest

from plb.metrics import evaluate_perturbation, mse, pearson, pearson_delta

rng = np.random.default_rng(0)
N = 2000


@pytest.fixture
def toy():
    """2,000 genes; 20 of them change a lot, the rest only by noise."""
    ctrl = rng.gamma(2.0, 1.0, N)
    delta = np.zeros(N)
    de_idx = np.arange(20)
    delta[de_idx] = rng.normal(0, 2.0, 20)
    true = ctrl + delta + rng.normal(0, 0.05, N)
    return ctrl, true, de_idx


def test_perfect_prediction(toy):
    ctrl, true, de = toy
    m = evaluate_perturbation(true, true, ctrl, de)
    assert m["mse_all"] == 0 and m["mse_de20"] == 0
    assert m["pearson_delta_all"] == pytest.approx(1.0)
    assert m["pearson_delta_de20"] == pytest.approx(1.0)


def test_mse_subset():
    pred, true = np.zeros(4), np.array([1.0, 1.0, 3.0, 3.0])
    assert mse(pred, true) == pytest.approx(5.0)
    assert mse(pred, true, idx=np.array([2, 3])) == pytest.approx(9.0)


def test_pearson_constant_is_nan():
    assert np.isnan(pearson(np.zeros(5), np.arange(5.0)))


def test_no_change_baseline_looks_great_on_raw_but_undefined_on_delta(toy):
    """Predicting no change gives raw r above 0.9, which is why delta correlation is the main metric."""
    ctrl, true, de = toy
    m = evaluate_perturbation(ctrl.copy(), true, ctrl, de)
    assert m["pearson_raw_all"] > 0.9
    assert np.isnan(m["pearson_delta_all"])


def test_delta_correlation_is_scale_invariant(toy):
    """Half-size effects in the right direction: delta r stays 1, MSE does not."""
    ctrl, true, de = toy
    pred = ctrl + 0.5 * (true - ctrl)
    assert pearson_delta(pred, true, ctrl, de) == pytest.approx(1.0)
    assert mse(pred, true, de) > 0


def test_wrong_direction_is_negative(toy):
    ctrl, true, de = toy
    pred = ctrl - (true - ctrl)
    assert pearson_delta(pred, true, ctrl, de) == pytest.approx(-1.0)
