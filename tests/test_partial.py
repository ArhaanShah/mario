import os

import numpy as np
import pandas as pd
import pytest
import torch

from mario_dino.config import load_config
from mario_dino.data.images import CachedImageDataset
from mario_dino.models.encoder import load_encoder, save_weights
from mario_dino.models.full import MarioModelDINO, merge_suffix_into_encoder
from mario_dino.models.heads import MarioHeads
from mario_dino.octip_cache import build_cache
from mario_dino.train_heads import Task1Data
from mario_dino.train_partial import build_token_cache, open_token_store, token_split, train_partial_fold
from mario_dino.utils import REPO_ROOT


@pytest.fixture()
def setup(synth, tmp_path):
    cfg = load_config(
        os.path.join(REPO_ROOT, "configs", "smoke.yaml"),
        [
            f"data.image_root={synth['images']}",
            f"preprocess.cache_dir={tmp_path / 'cache'}",
            f"encoder.weights={tmp_path / 'vits.pth'}",
            "encoder.img_size=[28, 56]",
            "encoder.kwargs={}",  # full 12-block ViT-S, random init
            f"partial.token_store_dir={tmp_path / 'tokens'}",
            "partial.n_blocks=1",
            "partial.batch_size=1",
            "partial.max_steps=2",
            "partial.max_epochs=5",
            "partial.lr_blocks=1.0e-3",
            "features.batch_size=2",
        ],
    )
    save_weights(cfg.encoder, pretrained=False, seed=0)
    # two pairs on different B-scans -> 4 distinct images
    df = pd.read_csv(synth["csv"]["task1_train"])
    df = df[df["id_patient"] == df["id_patient"].iloc[0]].drop_duplicates("BScan").head(2)
    images = list(df["image_at_ti"]) + list(df["image_at_ti+1"])
    assert len(set(images)) == 4
    build_cache(images, cfg.data.image_root, cfg.preprocess.cache_dir, "raw")
    return cfg, df, images


def test_partial_finetune_changes_only_unfrozen_params(setup):
    cfg, df, images = setup
    device = torch.device("cpu")
    encoder = load_encoder(cfg.encoder, device)
    split = token_split(encoder, cfg.partial.n_blocks)
    assert split == 11
    before = {k: v.clone() for k, v in encoder.model.state_dict().items()}

    tokens = open_token_store(cfg, encoder, split)
    build_token_cache(cfg, encoder, split, images, device, tokens)
    assert tokens.done.all() and tokens.row_shape == (1, 2 * 4 + 5, 384)
    t1 = Task1Data(
        t_rows=tokens.rows(df["image_at_ti"]),
        n_rows=tokens.rows(df["image_at_ti+1"]),
        dt=df["delta_t"].to_numpy(np.float32),
        y=df["label"].to_numpy(np.int64),
        case=df["case"].to_numpy(),
        pid=df["id_patient"].to_numpy(),
        fold=np.zeros(len(df), np.int64),
    )
    heads = MarioHeads(encoder.out_dim, hidden=16, dt_dim=8)
    heads0 = {k: v.clone() for k, v in heads.state_dict().items()}
    idx = np.arange(len(df))
    net, history, best_epoch = train_partial_fold(cfg, encoder, split, tokens, t1, idx, np.zeros(0, np.int64),
                                                  heads, device, torch.Generator().manual_seed(0))
    # 2 pairs at batch 1 = 2 steps per epoch; max_steps=2 stops after the first epoch
    assert len(history) == 1 and best_epoch == 1

    # the frozen encoder object is untouched
    for k, v in encoder.model.state_dict().items():
        assert torch.equal(v, before[k]), k
    # the fine-tuned encoder differs from the original only in blocks[split:] and the final norm
    after = merge_suffix_into_encoder(encoder, split, net.suffix.state_dict())
    changed = {k for k in after if not torch.equal(after[k], before[k])}
    unfrozen = {k for k in after if k.startswith(f"blocks.{split}.") or k.startswith("norm.")}
    assert changed and changed <= unfrozen
    assert len(changed) >= 0.8 * len(unfrozen)
    assert any(not torch.equal(v, heads0[k]) for k, v in net.heads.state_dict().items())
    tokens.close()


def test_partial_model_matches_token_path(setup):
    """MarioModelDINO with a fine-tuned suffix on images == PartialNet on the cached (fp16) prefix tokens."""
    cfg, df, images = setup
    device = torch.device("cpu")
    encoder = load_encoder(cfg.encoder, device)
    split = token_split(encoder, 1)
    tokens = open_token_store(cfg, encoder, split)
    build_token_cache(cfg, encoder, split, images, device, tokens)
    from mario_dino.train_partial import PartialNet

    net = PartialNet(encoder, split, MarioHeads(encoder.out_dim, hidden=16, dt_dim=8)).eval()
    with torch.no_grad():
        for p in net.suffix.parameters():
            p.add_(0.01 * torch.randn_like(p))  # a "fine-tuned" suffix
    model = MarioModelDINO(encoder, [net.heads], [net.suffix.state_dict()], split)
    ds = CachedImageDataset(images, cfg.preprocess.cache_dir, "raw", cfg.encoder.img_size)
    x = torch.stack([ds[i][1] for i in range(len(images))])
    with torch.no_grad():
        f_img = model.encode(x)[0, :, 0]
        f_tok = net.features(torch.from_numpy(tokens.get(images)[:, 0]).float())
    torch.testing.assert_close(f_img, f_tok, rtol=1e-2, atol=1e-2)
    tokens.close()
