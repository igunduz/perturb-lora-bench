"""Download GEARS datasets, build splits and print a summary."""

import argparse
from collections import Counter
from pathlib import Path

from plb.data import DATA_DIR, DATASETS, load_pertdata, perturbed_genes
from plb.genes import download_hgnc


def describe(name: str, pert_data) -> None:
    adata = pert_data.adata
    conds = adata.obs["condition"]
    n_genes_per_pert = Counter(len(perturbed_genes(c)) for c in conds.unique())
    print(f"\n== {name} ==")
    print(f"cells: {adata.n_obs:,}   genes: {adata.n_vars:,}   control cells: {(conds == 'ctrl').sum():,}")
    print(f"conditions: {n_genes_per_pert[1]} single-gene, {n_genes_per_pert[2]} two-gene")
    for split, cs in pert_data.set2conditions.items():
        print(f"  {split:5s}: {len([c for c in cs if c != 'ctrl'])} perturbations")
    if pert_data.subgroup:
        for group, cs in pert_data.subgroup["test_subgroup"].items():
            print(f"  test/{group}: {len(cs)}")
    print(f"X: {type(adata.X).__name__}, max value {adata.X.max():.2f} (log-normalised if < ~15)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--split-seed", type=int, default=1)
    args = ap.parse_args()
    print(f"HGNC table: {download_hgnc()}")
    for name in args.datasets:
        describe(name, load_pertdata(name, args.data_dir, args.split_seed))


if __name__ == "__main__":
    main()
