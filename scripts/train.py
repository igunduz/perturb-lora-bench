"""Train one scGPT perturbation model from a config and score it on the test split."""

import argparse
import json
import os
from pathlib import Path

import torch

from plb.data import load_pertdata, summarise
from plb.evaluation import save_run, score
from plb.model import build_model, trainable_state
from plb.train import CellData, predict_means, train
from plb.utils import load_config, seed_everything


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = args.config.stem
    dataset = cfg["data"]["name"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(cfg["seed"])

    pd_ = load_pertdata(dataset, split_seed=cfg["data"]["split_seed"])
    summary = summarise(pd_.adata, pd_.set2conditions, pd_.subgroup)
    data = CellData.from_adata(pd_.adata, summary)
    model, gmap = build_model(cfg, summary.genes, device)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    invisible = [p for p in summary.split["test"] if not gmap.pert_cols(p)]
    info = {"run": run, "device": device, "trainable_params": n_train,
            "genes_in_vocab": f"{len(gmap.cols)}/{len(gmap.genes)}",
            "test_perts_without_visible_gene": invisible}
    print(json.dumps(info, indent=1))

    import wandb
    wb = wandb.init(project=cfg["wandb"]["project"], name=run, config={**cfg, **info},
                    mode=os.environ.get("WANDB_MODE", cfg["wandb"]["mode"]))
    result = train(model, data, gmap, cfg, device, log=lambda d: (print(d), wb.log(d)))

    preds = predict_means(model, data, gmap, summary.split["test"], cfg["eval"]["n_cells"],
                          cfg["model"]["max_genes"], cfg["train"]["batch_size"], device)
    df = score(preds, summary, run)
    save_run(df, preds, dataset, run)
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(trainable_state(model), f"checkpoints/{run}.pt")

    test = {f"test_{k}": float(df[k].mean()) for k in ["pearson_delta_de20", "pearson_delta_all", "mse_de20", "mse_all"]}
    wb.summary.update({**result, **test})
    wb.finish()
    print(json.dumps({**result, **test}, indent=1))


if __name__ == "__main__":
    main()
