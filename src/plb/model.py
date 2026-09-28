"""scGPT (via helical) with PEFT LoRA adapters and a perturbation head."""

from __future__ import annotations

import torch
from torch import nn


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


class PerturbationScGPT(nn.Module):
    """scGPT encoder + perturbation embedding + per-gene decoder (not implemented)."""

    def __init__(self, backbone: nn.Module, d_model: int = 512):
        super().__init__()
        self.backbone = backbone
        self.pert_flag = nn.Embedding(2, d_model)
        raise NotImplementedError("PerturbationScGPT is drafted, not implemented yet.")
