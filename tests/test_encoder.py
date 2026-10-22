import json

import pytest
import torch

from mario_dino.config import EncoderCfg
from mario_dino.models.encoder import DinoEncoder, build_timm_model, load_encoder, save_weights

SMALL = "vit_small_patch14_reg4_dinov2"


@pytest.fixture(scope="module")
def enc():
    torch.manual_seed(0)
    return DinoEncoder(build_timm_model(SMALL, (42, 84))).eval()


def test_prefix_suffix_equals_forward_features(enc):
    x = torch.randn(2, 3, 42, 84)
    with torch.no_grad():
        ref = enc.forward_features(x)
        for k in (0, 1, 6, enc.depth - 1, enc.depth):
            out = enc.forward_suffix(enc.forward_prefix(x, k), k)
            torch.testing.assert_close(out, ref, rtol=0, atol=1e-5)


def test_pooling(enc):
    x = torch.randn(2, 3, 42, 84)
    with torch.no_grad():
        tokens = enc.forward_features(x)
        v = enc(x)
    assert enc.num_prefix_tokens == 5  # CLS + 4 registers
    assert tokens.shape == (2, 5 + 3 * 6, 384)
    assert v.shape == (2, enc.out_dim) == (2, 768)
    torch.testing.assert_close(v[:, :384], tokens[:, 0])
    torch.testing.assert_close(v[:, 384:], tokens[:, 5:].mean(1))


def test_default_input_gives_518_patch_tokens():
    enc = DinoEncoder(build_timm_model(SMALL, (196, 518), depth=1)).eval()
    with torch.no_grad():
        tokens = enc.forward_features(torch.randn(1, 3, 196, 518))
    assert tokens.shape[1] == 14 * 37 + 5 == 523


def test_save_and_strict_load_roundtrip(tmp_path):
    cfg = EncoderCfg(name=SMALL, weights=str(tmp_path / "w.pth"), img_size=[42, 84], kwargs={"depth": 2})
    save_weights(cfg, pretrained=False, seed=3)
    side = json.loads((tmp_path / "w.pth.json").read_text())
    assert side["pretrained"] is False and len(side["sha256"]) == 64
    a, b = load_encoder(cfg), load_encoder(cfg)
    x = torch.randn(1, 3, 42, 84)
    with torch.no_grad():
        torch.testing.assert_close(a(x), b(x))
    assert not any(p.requires_grad for p in a.parameters())
    # strict=True: a checkpoint for a different depth must not load silently
    bad = EncoderCfg(name=SMALL, weights=cfg.weights, img_size=[42, 84], kwargs={"depth": 3})
    with pytest.raises(RuntimeError):
        load_encoder(bad)


def test_missing_weights_file_is_an_error(tmp_path):
    cfg = EncoderCfg(name=SMALL, weights=str(tmp_path / "absent.pth"), img_size=[42, 84])
    with pytest.raises(FileNotFoundError, match="fetch_weights"):
        load_encoder(cfg)
