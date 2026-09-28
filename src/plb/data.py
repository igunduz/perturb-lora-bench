"""Loading Adamson and Norman through GEARS (Roohani et al., Nat Biotechnol 2024).

GEARS ships preprocessed versions of both datasets, downloaded from Harvard
Dataverse, along with the "simulation" split used by GEARS and scGPT. Using the
same files and split keeps our numbers comparable with theirs.

In the simulation split, test perturbations never appear in training. For
Norman's gene pairs, test pairs are further grouped by how many of the two
genes were seen alone in training (combo_seen0, combo_seen1, combo_seen2).

To be implemented once we have looked at the downloaded AnnData.
"""

from __future__ import annotations

from pathlib import Path

DATA_DIR = Path("data")
DATASETS = ("adamson", "norman")


def load_pertdata(name: str, data_dir: Path = DATA_DIR, split_seed: int = 1):
    """Return a GEARS PertData with the simulation split, downloading the data on first use."""
    raise NotImplementedError


def control_mean(pert_data) -> "np.ndarray":  # noqa: F821
    """Mean expression of the control cells (condition == 'ctrl')."""
    raise NotImplementedError
