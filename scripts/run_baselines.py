"""Score the non-learned baselines on the test split."""

import argparse

from plb.baselines import BASELINES, predict_baseline
from plb.data import DATASETS, load_pertdata, summarise
from plb.evaluation import save_run, score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    ap.add_argument("--split-seed", type=int, default=1)
    args = ap.parse_args()
    for name in args.datasets:
        pd_ = load_pertdata(name, split_seed=args.split_seed)
        s = summarise(pd_.adata, pd_.set2conditions, pd_.subgroup)
        train_means = {p: s.means[p] for p in s.split["train"]}
        for b in BASELINES:
            preds = predict_baseline(b, s.ctrl, train_means, s.split["test"])
            df = score(preds, s, b)
            print(f"{name:8s} {b:11s} pearson_delta_de20={df.pearson_delta_de20.mean():.3f} "
                  f"mse_de20={df.mse_de20.mean():.4f} -> {save_run(df, preds, name, b)}")


if __name__ == "__main__":
    main()
