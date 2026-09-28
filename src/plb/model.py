"""scGPT (via helical) with PEFT LoRA adapters and a perturbation head."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from plb.data import perturbed_genes


def disable_mha_fastpath() -> None:
    """Disable PyTorch's MHA fast path, which bypasses LoRA at eval."""
    torch.backends.mha.set_fastpath_enabled(False)


def attach_lora(
    model: nn.Module,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: list[str] | None = None,
):
    """Wrap model with LoRA adapters; freeze everything else."""
    from peft import LoraConfig, get_peft_model

    disable_mha_fastpath()
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules or ["self_attn"],
        bias="none",
    )
    return get_peft_model(model, cfg)


def load_scgpt(device: str = "cpu"):
    """Load whole-human scGPT and its gene vocabulary via helical."""
    from helical.models.scgpt import scGPT, scGPTConfig

    wrapper = scGPT(configurer=scGPTConfig(device=device))
    return wrapper.model, wrapper.vocab


def tiny_scgpt(genes: list[str]):
    """Small random scGPT-architecture network over the given genes, for tests."""
    from helical.models.scgpt.model_dir.model import TransformerModel

    vocab = {"<pad>": 0, **{g: i + 1 for i, g in enumerate(genes)}}
    net = TransformerModel(ntoken=len(vocab), d_model=32, nhead=4, d_hid=32, nlayers=2,
                           vocab=vocab, dropout=0.1, pad_value=-2)
    return net, vocab


def reinitialise(net: nn.Module) -> nn.Module:
    """Reset every layer to random initial weights (the no-pretraining control)."""
    for m in net.modules():
        if m is net:
            continue
        if isinstance(m, nn.MultiheadAttention):
            m._reset_parameters()
        elif hasattr(m, "reset_parameters"):
            m.reset_parameters()
    return net


def build_backbone(model_cfg: dict, genes: list[str], device: str = "cpu"):
    """Backbone and vocab from config: pretrained or random scGPT, or a tiny test network."""
    if model_cfg["backbone"] == "tiny":
        net, vocab = tiny_scgpt(list(genes))
    else:
        net, vocab = load_scgpt(device)
    if not model_cfg.get("pretrained", True):
        reinitialise(net)
    return net, vocab


class GeneMap:
    """Maps dataset genes to scGPT token ids."""

    def __init__(self, genes: np.ndarray, vocab: dict[str, int]):
        self.genes = np.asarray(genes)
        self.col = {g: i for i, g in enumerate(self.genes)}
        self.token = np.array([vocab.get(g, -1) for g in self.genes])
        self.cols = np.flatnonzero(self.token >= 0)
        self.vocab = vocab

    def pert_cols(self, pert: str) -> list[int]:
        """Columns of the perturbed genes that the model can see."""
        return [self.col[g] for g in perturbed_genes(pert) if g in self.col and self.token[self.col[g]] >= 0]


class PerturbationScGPT(nn.Module):
    """scGPT encoder + perturbation flag embedding + per-gene delta decoder."""

    def __init__(self, backbone: nn.Module, d_model: int):
        super().__init__()
        self.backbone = backbone
        self.pert_flag = nn.Embedding(2, d_model)
        nn.init.zeros_(self.pert_flag.weight)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, tokens: torch.Tensor, values: torch.Tensor, flags: torch.Tensor) -> torch.Tensor:
        """Predicted change from control for each input gene, shape (batch, genes)."""
        b = self.backbone
        x = b.encoder(tokens) + b.value_encoder(values) + self.pert_flag(flags)
        if getattr(b, "bn", None) is not None:
            x = b.bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        pad = torch.zeros(tokens.shape, dtype=torch.bool, device=tokens.device)
        h = b.transformer_encoder(x, src_key_padding_mask=pad)
        return self.head(h).squeeze(-1)


def build_model(cfg: dict, genes: np.ndarray, device: str = "cpu"):
    """Backbone (frozen, optionally with LoRA) wrapped in PerturbationScGPT, plus its GeneMap."""
    net, vocab = build_backbone(cfg["model"], list(genes), device)
    disable_mha_fastpath()
    r = cfg["lora"]["r"]
    if r > 0:
        attach_lora(net, r=r, alpha=cfg["lora"]["alpha"], dropout=cfg["lora"]["dropout"],
                    target_modules=cfg["lora"]["target_modules"])
    else:
        net.requires_grad_(False)
    d_model = net.encoder.embedding.embedding_dim
    return PerturbationScGPT(net, d_model).to(device), GeneMap(genes, vocab)


def trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Copy of the trainable parameters only (LoRA, flag embedding, head)."""
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
