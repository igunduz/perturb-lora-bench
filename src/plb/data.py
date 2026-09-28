"""Loading Adamson and Norman through GEARS (Roohani et al., Nat Biotechnol 2024).

GEARS ships preprocessed versions of both datasets on Harvard Dataverse: log-
normalised expression for 5,000 highly variable genes plus the perturbed genes,
with precomputed differential expression per perturbation. It also defines the
"simulation" split used by GEARS and scGPT. We use its download, gene filter
and split code unchanged so our numbers are comparable with theirs.

The one thing we skip is GEARS' per-cell graph construction, which only the
GEARS model needs and which takes a long time and several GB for Norman.

Conditions are named like GEARS does: "ctrl", "FOXA1+ctrl" for a single gene,
"FOXA1+FOXL2" for a pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

DATA_DIR = Path("data")
DATASETS = ("adamson", "norman")
DE_KEY = "top_non_dropout_de_20"


def _pertdata_class():
    """GEARS PertData without the per-cell graph step. Imported lazily (GEARS pulls in torch_geometric)."""
    from gears import PertData

    class LightPertData(PertData):
        def create_dataset_file(self):
            self.dataset_processed = {}

    return LightPertData


def load_pertdata(name: str, data_dir: Path = DATA_DIR, split_seed: int = 1):
    """GEARS PertData with the simulation split. Downloads the data on first use.

    The split puts 25% of perturbation genes in the test set; for Norman, test
    pairs are grouped by how many of their two genes were seen alone in
    training (`pert_data.subgroup["test_subgroup"]`).
    """
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
    """Mean expression over all cells with this condition (use "ctrl" for controls)."""
    mask = (adata.obs["condition"] == condition).to_numpy()
    if not mask.any():
        raise KeyError(f"no cells with condition {condition!r}")
    return _mean_rows(adata.X[mask])


def de_indices(adata, condition: str) -> np.ndarray:
    """Column indices of the top-20 DE genes GEARS precomputed for this condition."""
    cond_names = adata.obs.loc[adata.obs["condition"] == condition, "condition_name"].unique()
    if len(cond_names) != 1:
        raise KeyError(f"expected one condition_name for {condition!r}, got {list(cond_names)}")
    gene_ids = np.asarray(adata.uns[DE_KEY][cond_names[0]])
    idx = np.flatnonzero(adata.var_names.isin(gene_ids))
    if len(idx) != len(gene_ids):
        raise ValueError(f"{condition}: {len(gene_ids) - len(idx)} DE genes missing from var_names")
    return idx


def perturbed_genes(condition: str) -> list[str]:
    """'FOXA1+ctrl' -> ['FOXA1'], 'FOXA1+FOXL2' -> ['FOXA1', 'FOXL2'], 'ctrl' -> []."""
    return [g for g in condition.split("+") if g != "ctrl"]


@dataclass
class PerturbationSummary:
    """Everything the baselines and the evaluation need, as mean profiles.

    ctrl:   (n_genes,) mean of control cells
    means:  condition -> (n_genes,) mean of that condition's cells
    de_idx: condition -> indices of its top-20 DE genes
    split:  "train" / "val" / "test" -> list of conditions (ctrl excluded)
    """

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
