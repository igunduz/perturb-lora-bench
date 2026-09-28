"""LoRA tests on the scGPT architecture with random weights."""

import pytest
import torch

from plb.model import attach_lora

N_GENES, VOCAB = 50, 1000


@pytest.fixture
def scgpt_net():
    """Whole-human scGPT encoder shape with a small vocabulary."""
    from helical.models.scgpt.model_dir.model import TransformerModel

    torch.manual_seed(0)
    return TransformerModel(
        ntoken=VOCAB, d_model=512, nhead=8, d_hid=512, nlayers=12,
        vocab={"<pad>": 0}, dropout=0.2, pad_value=-2,
    )


@pytest.fixture
def batch():
    g = torch.Generator().manual_seed(1)
    genes = torch.randint(1, VOCAB, (2, N_GENES), generator=g)
    vals = torch.rand(2, N_GENES, generator=g)
    pad = torch.zeros(2, N_GENES, dtype=torch.bool)
    return genes, vals, pad


def encode_eval(model, batch):
    model.eval()
    with torch.no_grad():
        return model._encode(*batch)


def test_only_lora_params_trainable(scgpt_net):
    pm = attach_lora(scgpt_net, r=8, alpha=16)
    trainable = [n for n, p in pm.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable)
    assert sum(p.numel() for p in pm.parameters() if p.requires_grad) == 12 * (8 * 2048 + 8 * 1024)


def test_lora_is_identity_at_init(scgpt_net, batch):
    """Zero-initialised B leaves the output unchanged."""
    base = encode_eval(scgpt_net, batch)
    pm = attach_lora(scgpt_net)
    assert torch.allclose(encode_eval(pm, batch), base, atol=1e-5)


def test_lora_active_in_eval_mode(scgpt_net, batch):
    """LoRA must change eval-mode output."""
    base = encode_eval(scgpt_net, batch)
    pm = attach_lora(scgpt_net)
    for n, p in pm.named_parameters():
        if "lora_B" in n:
            p.data.normal_(0, 0.05)
    assert not torch.allclose(encode_eval(pm, batch), base, atol=1e-4)
