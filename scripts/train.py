"""Train one scGPT perturbation model from a config and score it on the test split."""

import argparse
import json
import os
import time
from pathlib import Path

import torch

from plb.data import load_pertdata, summarise
from plb.evaluation import save_run, score
from plb.genes import HGNC, HGNC_PATH
from plb.model import build_model, trainable_state
from plb.train import CellData, predict_means, train
from plb.utils import load_config, seed_everything


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--smoke", action="store_true", help="2 short epochs, 2 test perturbations, nothing saved")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = args.config.stem
    dataset = cfg["data"]["name"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and "SLURM_CPUS_PER_TASK" in os.environ:
        torch.set_num_threads(int(os.environ["SLURM_CPUS_PER_TASK"]))
    if args.smoke:
        cfg["train"].update(epochs=2, steps_per_epoch=5, patience=2)
        cfg["eval"].update(n_cells=2, n_cells_val=1)
        cfg["wandb"]["mode"] = "disabled"
    seed_everything(cfg["seed"])

    pd_ = load_pertdata(dataset, split_seed=cfg["data"]["split_seed"])
    summary = summarise(pd_.adata, pd_.set2conditions, pd_.subgroup)
    data = CellData.from_adata(pd_.adata, summary)
    hgnc = HGNC() if HGNC_PATH.exists() else None
    if hgnc is None:
        print(f"warning: {HGNC_PATH} missing, matching genes to scGPT by exact symbol only", flush=True)
    model, gmap = build_model(cfg, summary.genes, device, summary.gene_ids, hgnc)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    invisible = [p for p in summary.split["test"] if not gmap.pert_cols(p)]
    info = {"run": run, "device": device, "threads": torch.get_num_threads(), "trainable_params": n_train,
            "genes_in_vocab": f"{len(gmap.cols)}/{len(gmap.genes)}",
            "genes_matched_via_hgnc": len(gmap.renamed),
            "n_perts": {k: len(v) for k, v in summary.split.items()},
            "test_perts_without_visible_gene": invisible}
    print(json.dumps(info, indent=1), flush=True)

    import wandb
    mode = cfg["wandb"]["mode"] if args.smoke else os.environ.get("WANDB_MODE", cfg["wandb"]["mode"])
    wb = wandb.init(project=cfg["wandb"]["project"], name=run, config={**cfg, **info}, mode=mode)
    result = train(model, data, gmap, cfg, device, log=lambda d: (print(d, flush=True), wb.log(d)))

    test_perts = summary.split["test"][:2] if args.smoke else summary.split["test"]
    t0 = time.perf_counter()
    preds = predict_means(model, data, gmap, test_perts, cfg["eval"]["n_cells"],
                          cfg["model"]["max_genes"], cfg["train"]["batch_size"], device)
    per_pert = (time.perf_counter() - t0) / len(test_perts)
    print(json.dumps({"test_sec_per_perturbation": per_pert}), flush=True)
    if args.smoke:
        wb.finish()
        return

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
