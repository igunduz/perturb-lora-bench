"""Training loop and prediction for PerturbationScGPT."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch

from plb.data import PerturbationSummary
from plb.metrics import pearson_delta
from plb.model import GeneMap, PerturbationScGPT, trainable_state


@dataclass
class CellData:
    """Control cells (dense) plus per-perturbation summaries."""

    ctrl_cells: np.ndarray
    summary: PerturbationSummary

    @classmethod
    def from_adata(cls, adata, summary: PerturbationSummary) -> "CellData":
        X = adata.X[(adata.obs["condition"] == "ctrl").to_numpy()]
        X = X.toarray() if sp.issparse(X) else np.asarray(X)
        return cls(X.astype(np.float32), summary)


def make_batch(data: CellData, gmap: GeneMap, perts: list[str], cells: np.ndarray,
               cols: list[np.ndarray], device: str):
    """Tensors for (control cell, perturbation, gene columns) triples; target is the true mean delta."""
    tokens, values, flags, targets = [], [], [], []
    for p, c, cc in zip(perts, cells, cols):
        pc = gmap.pert_cols(p)
        tokens.append(gmap.token[cc])
        values.append(data.ctrl_cells[c, cc])
        flags.append(np.isin(cc, pc).astype(np.int64))
        targets.append((data.summary.means[p] - data.summary.ctrl)[cc] if p in data.summary.means else np.zeros(len(cc)))
    t = lambda a, dt: torch.as_tensor(np.stack(a), dtype=dt, device=device)
    return t(tokens, torch.long), t(values, torch.float32), t(flags, torch.long), t(targets, torch.float32)


def sample_cols(gmap: GeneMap, pert: str, max_genes: int, rng: np.random.Generator) -> np.ndarray:
    """Perturbed genes plus a random subset of other genes."""
    forced = gmap.pert_cols(pert)
    others = np.setdiff1d(gmap.cols, forced)
    k = min(max_genes - len(forced), len(others))
    return np.concatenate([forced, rng.choice(others, k, replace=False)]).astype(int)


def chunk_cols(gmap: GeneMap, pert: str, max_genes: int) -> list[np.ndarray]:
    """Cover all model genes in chunks, each containing the perturbed genes."""
    forced = gmap.pert_cols(pert)
    others = np.setdiff1d(gmap.cols, forced)
    size = max_genes - len(forced)
    return [np.concatenate([forced, others[i:i + size]]).astype(int) for i in range(0, len(others), size)]


@torch.no_grad()
def predict_means(model: PerturbationScGPT, data: CellData, gmap: GeneMap, perts: list[str],
                  n_cells: int, max_genes: int, batch_size: int, device: str, seed: int = 0) -> dict[str, np.ndarray]:
    """Predicted mean profile per perturbation: control mean + average predicted delta over control cells."""
    model.eval()
    rng = np.random.default_rng(seed)
    cells = rng.choice(len(data.ctrl_cells), min(n_cells, len(data.ctrl_cells)), replace=False)
    out = {}
    for p in perts:
        total = np.zeros(len(gmap.genes))
        count = np.zeros(len(gmap.genes))
        for cc in chunk_cols(gmap, p, max_genes):
            for i in range(0, len(cells), batch_size):
                cb = cells[i:i + batch_size]
                tok, val, flg, _ = make_batch(data, gmap, [p] * len(cb), cb, [cc] * len(cb), device)
                with _autocast(device):
                    d = model(tok, val, flg).float().cpu().numpy()
                np.add.at(total, cc, d.sum(0))
                np.add.at(count, cc, len(cb))
        delta = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
        out[p] = data.summary.ctrl + delta
    return out


def _autocast(device: str):
    if device.startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


def val_score(model, data, gmap, cfg, device) -> float:
    """Mean top-20 DE delta Pearson over validation perturbations."""
    perts = data.summary.split["val"]
    preds = predict_means(model, data, gmap, perts, cfg["eval"]["n_cells_val"], cfg["model"]["max_genes"],
                          cfg["train"]["batch_size"], device)
    s = data.summary
    return float(np.nanmean([pearson_delta(preds[p], s.means[p], s.ctrl, s.de_idx[p]) for p in perts]))


def train(model: PerturbationScGPT, data: CellData, gmap: GeneMap, cfg: dict, device: str, log=print) -> dict:
    """Train on training perturbations; keep the trainable weights with the best validation score."""
    tc = cfg["train"]
    rng = np.random.default_rng(cfg["seed"])
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tc["lr"], weight_decay=tc["weight_decay"])
    train_perts = data.summary.split["train"]
    best, best_state, bad = -np.inf, trainable_state(model), 0
    for epoch in range(tc["epochs"]):
        model.train()
        losses = []
        for _ in range(tc["steps_per_epoch"]):
            perts = list(rng.choice(train_perts, tc["batch_size"]))
            cells = rng.integers(len(data.ctrl_cells), size=len(perts))
            cols = [sample_cols(gmap, p, cfg["model"]["max_genes"], rng) for p in perts]
            tok, val, flg, tgt = make_batch(data, gmap, perts, cells, cols, device)
            with _autocast(device):
                pred = model(tok, val, flg)
            loss = torch.nn.functional.mse_loss(pred.float(), tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, tc["grad_clip"])
            opt.step()
            losses.append(loss.item())
        score = val_score(model, data, gmap, cfg, device)
        log({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_pearson_delta_de20": score})
        if score > best:
            best, best_state, bad = score, trainable_state(model), 0
        else:
            bad += 1
            if bad >= tc["patience"]:
                break
    model.load_state_dict(best_state, strict=False)
    return {"best_val_pearson_delta_de20": best, "epochs_run": epoch + 1}
