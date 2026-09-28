"""Aggregate saved runs into results tables (csv + markdown)."""

import argparse

from plb.data import DATASETS
from plb.evaluation import RESULTS_DIR, markdown_table, summarise_runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    args = ap.parse_args()
    for name in args.datasets:
        if not (RESULTS_DIR / name).exists():
            continue
        overall = summarise_runs(name).sort_values("pearson_delta_de20", ascending=False)
        overall.to_csv(RESULTS_DIR / f"summary_{name}.csv", index=False)
        md = f"### {name}\n\n{markdown_table(overall)}\n"
        if name == "norman":
            sub = summarise_runs(name, by_subgroup=True)
            sub.to_csv(RESULTS_DIR / "summary_norman_subgroups.csv", index=False)
            for g, part in sub.groupby("subgroup"):
                md += f"\n#### {g}\n\n{markdown_table(part.sort_values('pearson_delta_de20', ascending=False))}\n"
        (RESULTS_DIR / f"summary_{name}.md").write_text(md)
        print(md)


if __name__ == "__main__":
    main()
