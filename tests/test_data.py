"""Data loading and split tests; real-data tests skip without data/."""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from plb.data import DE_KEY, condition_mean, de_indices, perturbed_genes, summarise

GENES = [f"G{i}" for i in range(30)]


@pytest.fixture
def fake_adata():
    """Fake GEARS-style AnnData: 3 conditions x 4 cells."""
    conds = ["ctrl"] * 4 + ["G1+ctrl"] * 4 + ["G1+G2"] * 4
    X = np.zeros((12, 30))
    X[0:4] = 1.0
    X[4:8] = 2.0
    X[8:12, :5] = 7.0
    obs = pd.DataFrame(
        {"condition": conds, "condition_name": [f"K562_{c}_1+1" for c in conds]},
        index=[f"cell{i}" for i in range(12)],
    )
    var = pd.DataFrame({"gene_name": GENES}, index=[f"ENSG{i:03d}" for i in range(30)])
    a = ad.AnnData(sp.csr_matrix(X), obs=obs, var=var)
    a.uns[DE_KEY] = {
        "K562_G1+ctrl_1+1": np.array(["ENSG005", "ENSG001"]),
        "K562_G1+G2_1+1": np.array(["ENSG010"]),
    }
    return a


def test_perturbed_genes():
    assert perturbed_genes("ctrl") == []
    assert perturbed_genes("FOXA1+ctrl") == ["FOXA1"]
    assert perturbed_genes("FOXA1+FOXL2") == ["FOXA1", "FOXL2"]


def test_condition_mean(fake_adata):
    assert np.allclose(condition_mean(fake_adata, "ctrl"), 1.0)
    assert np.allclose(condition_mean(fake_adata, "G1+ctrl"), 2.0)
    with pytest.raises(KeyError):
        condition_mean(fake_adata, "G9+ctrl")


def test_de_indices_map_gene_ids_to_columns(fake_adata):
    assert sorted(de_indices(fake_adata, "G1+ctrl")) == [1, 5]
    assert list(de_indices(fake_adata, "G1+G2")) == [10]


def test_summarise(fake_adata):
    s = summarise(fake_adata, {"train": ["ctrl", "G1+ctrl"], "test": ["G1+G2"]})
    assert s.split == {"train": ["G1+ctrl"], "test": ["G1+G2"]}
    assert np.allclose(s.ctrl, 1.0)
    assert s.means["G1+G2"][0] == 7.0 and s.means["G1+G2"][5] == 0.0


def test_gears_simulation_split_holds_out_genes():
    """Test genes of single perturbations never appear in training."""
    from gears.data_utils import DataSplitter

    rng = np.random.default_rng(0)
    genes = [f"G{i}" for i in range(40)]
    conds = [f"{g}+ctrl" for g in genes] + [
        f"{a}+{b}" for a, b in (rng.choice(genes, 2, replace=False) for _ in range(60))
    ]
    conds = ["ctrl"] + sorted(set(conds))
    obs = pd.DataFrame({"condition": np.repeat(conds, 3)})
    obs.index = obs.index.astype(str)
    a = ad.AnnData(np.zeros((len(obs), 2)), obs=obs)
    a, _ = DataSplitter(a, split_type="simulation").split_data(seed=1)
    by_split = a.obs.groupby("split", observed=True)["condition"].unique()
    train_genes = {g for c in by_split["train"] for g in perturbed_genes(c)}
    test_single_genes = {g for c in by_split["test"] if c.endswith("+ctrl") for g in perturbed_genes(c)}
    assert test_single_genes and not (train_genes & test_single_genes)
    assert not (set(by_split["train"]) & set(by_split["test"]))


DATA = Path("data")
needs_adamson = pytest.mark.skipif(
    not (DATA / "adamson" / "perturb_processed.h5ad").exists(), reason="run `make data` first"
)


@pytest.fixture(scope="module")
def adamson():
    from plb.data import load_pertdata

    return load_pertdata("adamson")


@pytest.mark.data
@needs_adamson
def test_adamson_split_is_disjoint(adamson):
    s = adamson.set2conditions
    train, val, test = (set(s[k]) - {"ctrl"} for k in ("train", "val", "test"))
    assert test and val and not (train & test) and not (train & val) and not (val & test)


@pytest.mark.data
@needs_adamson
def test_adamson_every_test_perturbation_has_20_de_genes(adamson):
    for c in adamson.set2conditions["test"]:
        assert len(de_indices(adamson.adata, c)) == 20
