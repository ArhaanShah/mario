"""MODEL EXECUTION TESTS: written for later CI/cloud; not run during scaffolding."""
import copy

import pytest
import torch

from mario_dino.config import Config, EncoderCfg
from mario_dino.features import store_identity
from mario_dino.losses import stage_a_loss
from mario_dino.models.encoder import build_encoder
from mario_dino.models.finetuned import encoder_spec, load_finetuned_checkpoint
from mario_dino.models.full import Task1Net
from mario_dino.models.heads import MarioHeads, OtherGateHead, heads_from_hparams

pytestmark = pytest.mark.model_execution


@pytest.mark.parametrize("fusion", ["legacy", "delta_mlp", "other_gate"])
def test_fusion_shapes_gradients_and_roundtrip(fusion):
    heads = MarioHeads(8, hidden=8, dt_dim=4, dropout=0, fusion=fusion, fusion_hidden=12)
    a = torch.randn(4, 8, requires_grad=True)
    b = torch.randn(4, 8, requires_grad=True)
    logits = heads.task1_logits(a, b)
    assert logits.shape == (4, 4) and torch.isfinite(logits).all()
    loss, _ = stage_a_loss(heads, a, b, torch.ones(4), torch.arange(4), torch.ones(4), 1.)
    loss.backward()
    assert a.grad.abs().sum() > 0 and b.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in heads.head.parameters())
    restored = heads_from_hparams(heads.hparams)
    restored.load_state_dict(heads.state_dict(), strict=True)
    torch.testing.assert_close(restored.task1_logits(a, b), logits)
    if fusion != "legacy":
        with pytest.raises(ValueError, match="Task 1 only"):
            heads.task2_logits(a)


@pytest.mark.parametrize("gate_bias", [-10000., 0., 10000.])
def test_other_gate_is_a_normalized_stable_distribution(gate_bias):
    head = OtherGateHead(8, hidden=12, p=0)
    with torch.no_grad():
        head.gate.weight.zero_()
        head.gate.bias.fill_(gate_bias)
    logp = head(torch.randn(4, 8), torch.randn(4, 8))
    assert torch.isfinite(logp).all()
    torch.testing.assert_close(logp.exp().sum(1), torch.ones(4))
    logp.sum().backward()
    assert head.gate.bias.grad is not None


def test_old_hparams_and_state_dict_remain_strictly_compatible():
    old = {"d": 8, "hidden": 8, "dt_dim": 4, "dropout": 0}
    original = MarioHeads(**old)
    assert "fusion" not in original.hparams
    restored = heads_from_hparams(original.hparams)
    restored.load_state_dict(original.state_dict(), strict=True)
    assert restored.fusion == "legacy"
    assert restored.task2_logits(torch.randn(2, 8)).shape == (2, 3)
    with pytest.raises(RuntimeError):
        MarioHeads(**old, fusion="delta_mlp").load_state_dict(original.state_dict(), strict=True)


@pytest.mark.parametrize("fusion", ["delta_mlp", "other_gate"])
def test_experimental_full_checkpoint_reconstructs_without_source_weights(fusion):
    ec = EncoderCfg(kind="convnext", name="convnext_tiny.in12k_ft_in1k", weights=None,
                    img_size=[32, 64], preprocessing="imagenet",
                    kwargs={"depths": [1, 1, 1, 1], "dims": [8, 16, 24, 32]})
    cfg = Config(encoder=ec)
    enc = build_encoder(ec)
    net = Task1Net(enc, MarioHeads(enc.out_dim, hidden=8, dt_dim=4, dropout=0, fusion=fusion, fusion_hidden=12)).eval()
    checkpoint = {"kind": "finetuned", "format_version": 1, "task": 1, "fold": 0,
                  "encoder": net.encoder.state_dict(), "heads": net.heads.state_dict(),
                  "hparams": net.heads.hparams, "encoder_spec": encoder_spec(cfg, enc, store_identity(cfg))}
    restored = load_finetuned_checkpoint(checkpoint).eval()
    x = torch.randn(2, 3, 32, 64)
    with torch.no_grad():
        torch.testing.assert_close(net(x, x.flip(-1)), restored(x, x.flip(-1)))
    bad = copy.deepcopy(checkpoint)
    bad["hparams"]["fusion"] = "legacy"
    with pytest.raises(RuntimeError):
        load_finetuned_checkpoint(bad)
