"""scGPT (loaded via helical) + LoRA adapters + a perturbation head.

LoRA in one paragraph
---------------------
A frozen weight matrix W (d_out x d_in) is left untouched. Next to it we train
two small matrices, A (r x d_in) and B (d_out x r), and the layer computes

    y = W x + (alpha / r) * B A x

B is initialised to zero, so at step 0 the model is exactly the pretrained
model. Only A and B get gradients: for r=8 on a 512->1536 projection that is
8*512 + 1536*8 = 16,384 parameters instead of 786,432.

scGPT-specific wrinkle
----------------------
scGPT uses torch.nn.MultiheadAttention, which stores Q, K and V stacked in a
single `in_proj_weight` (3*d_model x d_model). There are no separate q_proj /
v_proj modules to target. PEFT supports MultiheadAttention as a unit: targeting
"self_attn" adapts the stacked QKV projection *and* out_proj.

PyTorch fast-path trap (tested in tests/test_lora.py)
------------------------------------------------------
In eval mode under torch.no_grad(), nn.TransformerEncoderLayer takes a fused
C++ "fast path" that reads `self_attn.in_proj_weight` directly and never calls
the (LoRA-wrapped) module's forward. The adapter is silently skipped and you
evaluate the base model. `attach_lora` disables the fast path globally.
"""

from __future__ import annotations

import torch
from torch import nn


def disable_mha_fastpath() -> None:
    """Force MultiheadAttention/TransformerEncoderLayer to use the Python path, so LoRA is applied at eval."""
    torch.backends.mha.set_fastpath_enabled(False)


def attach_lora(
    model: nn.Module,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: list[str] | None = None,
):
    """Wrap `model` with PEFT LoRA adapters and freeze everything else.

    Returns a peft.PeftModel. Call `.print_trainable_parameters()` on it to see the budget.
    """
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
    """Load pretrained scGPT (whole-human checkpoint) and its gene vocabulary through helical.

    Downloads ~200 MB to $CACHE_DIR_HELICAL_PREFIX/.cache/helical on first call
    (defaults to ~/.cache; on the cluster, point the prefix at scratch).
    Returns (network, vocab) where vocab maps gene symbol -> token id.
    """
    from helical.models.scgpt import scGPT, scGPTConfig

    wrapper = scGPT(configurer=scGPTConfig(device=device))
    return wrapper.model, wrapper.vocab


class PerturbationScGPT(nn.Module):
    """scGPT encoder + perturbation-flag embedding + per-gene expression decoder.

    Design (follows the original scGPT perturbation setup, not yet implemented):
      * input tokens: genes of one control cell, values = control expression
      * each gene token also gets a learned flag embedding: 0 = not perturbed,
        1 = perturbed (the CRISPRa target), added to the gene embedding
      * encoder: pretrained scGPT transformer (frozen, + LoRA)
      * head: per-gene MLP predicting post-perturbation expression

    Trainable: LoRA adapters, the flag embedding, and the decoder head.
    TODO(week 2): implement after data loading and baselines are done.
    """

    def __init__(self, backbone: nn.Module, d_model: int = 512):
        super().__init__()
        self.backbone = backbone
        self.pert_flag = nn.Embedding(2, d_model)
        raise NotImplementedError("PerturbationScGPT is drafted, not implemented yet.")
