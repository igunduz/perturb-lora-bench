"""scGPT loaded through helical, with LoRA adapters and a perturbation head.

LoRA (Hu et al., ICLR 2022) keeps a pretrained weight matrix W frozen and
learns a low-rank update next to it: the layer computes

    y = W x + (alpha / r) * B A x

where A is r x d_in and B is d_out x r. B starts at zero, so before training
the model is exactly the pretrained one. For r = 8 on a 512 -> 1536 projection
that is 16,384 trainable numbers instead of 786,432.

scGPT (Cui et al., Nature Methods 2024) uses torch.nn.MultiheadAttention, which
keeps the query, key and value weights in one stacked `in_proj_weight`. There
are no separate q_proj / v_proj layers, so we target "self_attn" and PEFT
adapts the stacked projection plus out_proj.

In eval mode under torch.no_grad(), TransformerEncoderLayer takes a fused fast
path that reads in_proj_weight directly and skips the LoRA wrapper. attach_lora
turns that path off; tests/test_lora.py checks it.
"""

from __future__ import annotations

import torch
from torch import nn


def disable_mha_fastpath() -> None:
    """Make attention run through the normal Python path so LoRA is used at eval time."""
    torch.backends.mha.set_fastpath_enabled(False)


def attach_lora(
    model: nn.Module,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: list[str] | None = None,
):
    """Add PEFT LoRA adapters to `model` and freeze all other weights. Returns a PeftModel."""
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
    """Load the whole-human scGPT checkpoint and its gene vocabulary via helical.

    The first call downloads the weights to $CACHE_DIR_HELICAL_PREFIX/.cache/helical
    (~/.cache if unset; slurm/_env.sh points it at scratch).
    Returns (network, vocab), where vocab maps gene symbol to token id.
    """
    from helical.models.scgpt import scGPT, scGPTConfig

    wrapper = scGPT(configurer=scGPTConfig(device=device))
    return wrapper.model, wrapper.vocab


class PerturbationScGPT(nn.Module):
    """Predict post-perturbation expression from a control cell. Not implemented yet.

    Planned design, following the scGPT paper's perturbation setup: the input is
    one control cell's genes and expression values. Each gene token gets an
    extra learned embedding saying whether it is the perturbed gene. The frozen
    scGPT encoder (with LoRA) processes the tokens, and a small MLP predicts
    each gene's expression after the perturbation. Trained: LoRA adapters, the
    perturbation embedding, and the MLP.
    """

    def __init__(self, backbone: nn.Module, d_model: int = 512):
        super().__init__()
        self.backbone = backbone
        self.pert_flag = nn.Embedding(2, d_model)
        raise NotImplementedError("PerturbationScGPT is drafted, not implemented yet.")
