"""Data loading via GEARS PertData.

Why GEARS: its Norman/Adamson files and "simulation" splits are what the
perturbation-prediction literature (GEARS, scGPT, and benchmarks after them)
reports on, so our numbers are comparable. Downloads come from Harvard Dataverse.

Split semantics ("simulation" split, seed-controlled):
  * single-gene datasets (Adamson): test perturbations are genes never seen in training.
  * Norman combos: test combos are stratified by how many of the two genes were
    seen as single perturbations in training -> "combo_seen0/1/2" subgroups.

TODO(next step): implement after we inspect the downloaded AnnData together.
"""

from __future__ import annotations

from pathlib import Path

DATA_DIR = Path("data")
DATASETS = ("adamson", "norman")


def load_pertdata(name: str, data_dir: Path = DATA_DIR, split_seed: int = 1):
    """Download (if needed) and return a GEARS PertData with the simulation split prepared."""
    raise NotImplementedError


def control_mean(pert_data) -> "np.ndarray":  # noqa: F821
    """Mean expression over control cells (condition == 'ctrl')."""
    raise NotImplementedError
