"""Model and training-loop tests on fake data with a tiny random scGPT."""

import copy

import numpy as np
import pytest
import torch

from plb.data import PerturbationSummary
from plb.model import GeneMap, build_model, reinitialise, tiny_scgpt
from plb.train import CellData, chunk_cols, predict_means, train

GENES = np.array([f"G{i}" for i in range(40)] + ["NOT_IN_VOCAB"])
CFG = {
    "seed": 0,
    "model": {"backbone": "tiny", "pretrained": True, "max_genes": 12},
    "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["self_attn"]},
    "train": {"epochs": 30, "steps_per_epoch": 10, "batch_size": 8, "lr": 3e-3,
              "weight_decay": 0.0, "grad_clip": 1.0, "patience": 30},
    "eval": {"n_cells": 8, "n_cells_val": 4},
}


@pytest.fixture
def data():
    """Perturbing Gk raises gene G(k+20) by 3; everything else unchanged."""
    rng = np.random.default_rng(0)
    ctrl = rng.gamma(2.0, 0.5, len(GENES))
    means, de = {}, {}
    for k in range(10):
        m = ctrl.copy()
        m[k + 20] += 3.0
        means[f"G{k}+ctrl"] = m
        de[f"G{k}+ctrl"] = np.arange(20, 40)
    split = {"train": [f"G{k}+ctrl" for k in range(6)], "val": ["G6+ctrl", "G7+ctrl"], "test": ["G8+ctrl", "G9+ctrl"]}
    s = PerturbationSummary(GENES, ctrl, means, de, split)
    cells = (ctrl + rng.normal(0, 0.1, (50, len(GENES)))).astype(np.float32)
    return CellData(cells, s)


def trainable_names(model):
    return {n.split(".")[0] if not "lora_" in n else "lora" for n, p in model.named_parameters() if p.requires_grad}


def test_frozen_trains_only_flag_and_head(data):
    model, _ = build_model({**CFG, "lora": {**CFG["lora"], "r": 0}}, data.summary.genes)
    assert trainable_names(model) == {"pert_flag", "head"}


def test_lora_adds_adapters(data):
    model, _ = build_model(CFG, data.summary.genes)
    assert trainable_names(model) == {"pert_flag", "head", "lora"}


def test_untrained_model_predicts_no_change(data):
    model, gmap = build_model(CFG, data.summary.genes)
    preds = predict_means(model, data, gmap, ["G8+ctrl"], 4, 12, 4, "cpu")
    assert np.allclose(preds["G8+ctrl"], data.summary.ctrl)


def test_chunks_cover_all_vocab_genes_and_carry_perturbed_gene(data):
    _, vocab = tiny_scgpt([g for g in GENES if g != "NOT_IN_VOCAB"])
    gmap = GeneMap(GENES, vocab)
    chunks = chunk_cols(gmap, "G3+ctrl", 12)
    assert set(np.concatenate(chunks)) == set(gmap.cols)
    assert all(3 in c and len(c) <= 12 for c in chunks)
    assert gmap.col["NOT_IN_VOCAB"] not in gmap.cols


def test_reinitialise_changes_attention_weights():
    net, _ = tiny_scgpt(list(GENES))
    before = copy.deepcopy(net.transformer_encoder.layers[0].self_attn.in_proj_weight.data)
    reinitialise(net)
    assert not torch.allclose(before, net.transformer_encoder.layers[0].self_attn.in_proj_weight.data)


def test_training_learns_seen_perturbations(data):
    torch.manual_seed(0)
    data.summary.split["val"] = ["G0+ctrl", "G1+ctrl"]
    cfg = {**CFG, "train": {**CFG["train"], "epochs": 60, "patience": 60}}
    model, gmap = build_model(cfg, data.summary.genes)
    train(model, data, gmap, cfg, "cpu", log=lambda d: None)
    preds = predict_means(model, data, gmap, ["G0+ctrl"], 8, 12, 8, "cpu")
    delta = preds["G0+ctrl"] - data.summary.ctrl
    assert delta[20] > 1.0 and abs(np.delete(delta, 20)).max() < delta[20]
