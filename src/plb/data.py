"""Adamson and Norman data and splits from GEARS (GO-graph GNN, Roohani et al. 2024); we use only its data and split code."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

DATA_DIR = Path("data")
DATASETS = ("adamson", "norman")
DE_KEY = "top_non_dropout_de_20"


def _pertdata_class():
    """GEARS PertData without per-cell graph construction."""
    from gears import PertData

    class LightPertData(PertData):
        def create_dataset_file(self):
            self.dataset_processed = {}

    return LightPertData


def load_pertdata(name: str, data_dir: Path = DATA_DIR, split_seed: int = 1):
    """Download (first call), GO-filter and split a dataset with GEARS' simulation split."""
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; expected one of {DATASETS}")
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    pert_data = _pertdata_class()(str(data_dir))
    pert_data.load(data_name=name)
    pert_data.prepare_split(split="simulation", seed=split_seed)
    return pert_data


def _mean_rows(X) -> np.ndarray:
    return np.asarray(X.mean(axis=0)).ravel().astype(np.float64)


def condition_mean(adata, condition: str) -> np.ndarray:
    """Mean expression of cells with this condition."""
    mask = (adata.obs["condition"] == condition).to_numpy()
    if not mask.any():
        raise KeyError(f"no cells with condition {condition!r}")
    return _mean_rows(adata.X[mask])


def de_indices(adata, condition: str) -> np.ndarray:
    """Column indices of GEARS' top-20 DE genes for this condition."""
    cond_names = adata.obs.loc[adata.obs["condition"] == condition, "condition_name"].unique()
    if len(cond_names) != 1:
        raise KeyError(f"expected one condition_name for {condition!r}, got {list(cond_names)}")
    gene_ids = np.asarray(adata.uns[DE_KEY][cond_names[0]])
    idx = np.flatnonzero(adata.var_names.isin(gene_ids))
    if len(idx) != len(gene_ids):
        raise ValueError(f"{condition}: {len(gene_ids) - len(idx)} DE genes missing from var_names")
    return idx


def perturbed_genes(condition: str) -> list[str]:
    """'A+ctrl' -> ['A'], 'A+B' -> ['A', 'B'], 'ctrl' -> []."""
    return [g for g in condition.split("+") if g != "ctrl"]


@dataclass
class PerturbationSummary:
    """Per-condition mean profiles, DE indices and split."""

    genes: np.ndarray
    ctrl: np.ndarray
    means: dict[str, np.ndarray]
    de_idx: dict[str, np.ndarray]
    split: dict[str, list[str]]
    test_subgroup: dict[str, list[str]] | None = None


def summarise(adata, set2conditions: dict, subgroup: dict | None = None) -> PerturbationSummary:
    """Collapse cells into per-condition mean profiles."""
    split = {k: [c for c in v if c != "ctrl"] for k, v in set2conditions.items()}
    conds = [c for v in split.values() for c in v]
    X = adata.X if sp.issparse(adata.X) else np.asarray(adata.X)
    obs_cond = adata.obs["condition"].to_numpy()
    return PerturbationSummary(
        genes=adata.var["gene_name"].to_numpy(),
        ctrl=_mean_rows(X[obs_cond == "ctrl"]),
        means={c: _mean_rows(X[obs_cond == c]) for c in conds},
        de_idx={c: de_indices(adata, c) for c in conds},
        split=split,
        test_subgroup=(subgroup or {}).get("test_subgroup"),
    )
