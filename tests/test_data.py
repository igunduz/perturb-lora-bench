"""Data loading and split checks. Skipped until `make data` has been run."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.data
DATA = Path("data")
needs_adamson = pytest.mark.skipif(not (DATA / "adamson").exists(), reason="run `make data` first")


@needs_adamson
def test_test_perturbations_unseen_in_train():
    """No test perturbation may appear in training."""
    from plb.data import load_pertdata

    pd_ = load_pertdata("adamson")
    train = set(pd_.set2conditions["train"]) - {"ctrl"}
    test = set(pd_.set2conditions["test"])
    assert test and not (train & test)


@needs_adamson
def test_control_cells_present():
    from plb.data import load_pertdata

    pd_ = load_pertdata("adamson")
    assert (pd_.adata.obs["condition"] == "ctrl").sum() > 100
