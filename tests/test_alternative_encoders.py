"""Offline regression tests for cloud/CI. These tests run models; do not run during code-only edits."""
import ast
import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from mario_dino.config import Config, ConfigError, EncoderCfg, config_from_dict, load_config
from mario_dino.data.images import preprocess, preprocess_input, resolve_input_spec
from mario_dino.features import POOLING, store_identity
from mario_dino.infer import check_architecture, check_identity, checkpoint_paths
from mario_dino.losses import stage_a_loss
from mario_dino.models.encoder import build_encoder, load_encoder, save_weights
from mario_dino.models.finetuned import encoder_spec, load_finetuned_checkpoint
from mario_dino.models.full import Task1Net
from mario_dino.models.heads import MarioHeads
from mario_dino.models.vendor.visionfm_vit import VisionTransformer
from mario_dino.models.visionfm import backbone_state
from mario_dino.train_partial import train_all_partial

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def shallow_visionfm(monkeypatch):
    """Random one-block fixture, never a production pretrained architecture."""
    from mario_dino.models import visionfm

    def shallow(**kwargs):
        kwargs["depth"] = 1
        return VisionTransformer(**kwargs)

    monkeypatch.setattr(visionfm, "VisionTransformer", shallow)


def alternative_cfg(kind, tmp_path):
    conv = kind == "convnext"
    return EncoderCfg(kind=kind, name="convnext_tiny.in12k_ft_in1k" if conv else "vit_base_patch16",
                      weights=str(tmp_path / f"{kind}.pth"), img_size=[32, 64],
                      preprocessing="imagenet" if conv else "visionfm_oct",
                      kwargs={"depths": [1, 1, 1, 1], "dims": [8, 16, 24, 32]} if conv else {})


@pytest.mark.parametrize("kind", ["convnext", "visionfm"])
def test_actual_loss_changes_encoder_and_heads(kind, tmp_path, shallow_visionfm):
    ec = alternative_cfg(kind, tmp_path)
    save_weights(ec, pretrained=False, seed=7)
    encoder = load_encoder(ec, trainable=True)
    assert encoder.training and all(p.requires_grad for p in encoder.parameters())
    heads = MarioHeads(encoder.out_dim, hidden=16, dt_dim=4, dropout=0)
    net = Task1Net(encoder, heads)
    a = torch.randn(2, 3, *ec.img_size, requires_grad=True)
    b = torch.randn_like(a, requires_grad=True)
    fa, fb = net.features(a, b)
    assert fa.shape == (2, encoder.out_dim)
    assert fb.shape == fa.shape
    before_encoder = {n: p.detach().clone() for n, p in encoder.named_parameters()}
    before_heads = {n: p.detach().clone() for n, p in heads.named_parameters()}
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    loss, _ = stage_a_loss(heads, fa, fb, torch.ones(2), torch.tensor([0, 1]), torch.ones(4), 1.)
    opt.zero_grad()
    loss.backward()
    assert a.grad.abs().sum() > 0 and b.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.parameters())
    opt.step()
    assert any(not torch.equal(before_encoder[n], p) for n, p in encoder.named_parameters())
    assert any(not torch.equal(before_heads[n], p) for n, p in heads.named_parameters())
    frozen = load_encoder(ec)
    assert not frozen.training and not any(p.requires_grad for p in frozen.parameters())


@pytest.mark.parametrize("key", [None, "teacher", "student", "visionfm_state_dict"])
def test_visionfm_formats_are_strict(tmp_path, shallow_visionfm, key):
    if key is not None and not hasattr(torch.serialization, "safe_globals"):
        pytest.skip("upstream Namespace metadata import needs PyTorch >=2.4")
    ec = alternative_cfg("visionfm", tmp_path)
    ref = build_encoder(ec)
    raw = ref.model.state_dict()
    prefixed = {"module.backbone." + n: t for n, t in raw.items()}
    prefixed["module.head.last_layer.weight"] = torch.zeros(3, 3)
    prefixed["module.backbone.masked_embed"] = torch.zeros(1, 768)
    import argparse
    wrapper = {
        key: prefixed,
        "args": argparse.Namespace(arch="vit_base", modality="OCT"),
        "ibot_loss": np.float64(0.5),
        "temperature_dtype": np.dtype(np.float64),
    }
    torch.save(prefixed if key is None else wrapper, ec.weights)
    ec.checkpoint_key = key
    actual = load_encoder(ec)
    for n, tensor in actual.model.state_dict().items():
        torch.testing.assert_close(tensor, raw[n])
    broken = dict(prefixed)
    broken.pop("module.backbone.norm.weight")
    torch.save(broken if key is None else {key: broken}, ec.weights)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_encoder(ec)


def test_visionfm_rejects_collisions_wrong_wrappers_and_grids():
    pos = torch.zeros(1, 197, 768)
    with pytest.raises(ValueError, match="collision"):
        backbone_state({"pos_embed": pos, "module.backbone.pos_embed": pos})
    with pytest.raises(ValueError, match="requested key"):
        backbone_state({"pos_embed": pos}, "teacher")
    with pytest.raises(ValueError, match="square"):
        backbone_state({"pos_embed": torch.zeros(1, 198, 768)})
    with pytest.raises(ValueError, match="select the wrapper"):
        backbone_state({"teacher": {"pos_embed": pos}})


@pytest.mark.parametrize("damage", ["extra", "patch", "norm"])
def test_visionfm_unknown_or_incompatible_tensors_fail(tmp_path, shallow_visionfm, damage):
    ec = alternative_cfg("visionfm", tmp_path)
    state = build_encoder(ec).model.state_dict()
    if damage == "extra":
        state["unrecognized.weight"] = torch.zeros(1)
    elif damage == "patch":
        state["patch_embed.proj.weight"] = state["patch_embed.proj.weight"][:, :, :8, :8]
    else:
        del state["norm.weight"]
    torch.save(state, ec.weights)
    with pytest.raises(RuntimeError):
        load_encoder(ec)


def isolated_upstream():
    """Use pinned upstream source verbatim except its unavailable dependency imports.

    No external utils or CUDA FlashAttention objects are imported/constructed.
    """
    source = (ROOT / "tests/fixtures/visionfm_upstream.py.txt").read_text()
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and node.module in {"utils", "einops", "packaging"})]
    scope = {"trunc_normal_": torch.nn.init.trunc_normal_}
    exec(compile(tree, "pinned_visionfm_upstream", "exec"), scope)
    return scope["VisionTransformer"]


@pytest.mark.parametrize("shape", [(224, 224), (192, 512)])
def test_visionfm_matches_pinned_upstream_cls(shape):
    # Small random architecture for numerical parity; same upstream block/interpolation implementation.
    cls = isolated_upstream()
    kwargs = dict(img_size=[224], embed_dim=48, depth=1, num_heads=3, qkv_bias=True, use_flash=False)
    ref, actual = cls(**kwargs).eval(), VisionTransformer(**kwargs).eval()
    actual.load_state_dict(ref.state_dict(), strict=True)
    x = torch.randn(1, 3, *shape)
    with torch.no_grad():
        torch.testing.assert_close(actual(x), ref(x), rtol=0, atol=0)


def test_nondefault_native_positional_grid_roundtrip(tmp_path, shallow_visionfm):
    ec = alternative_cfg("visionfm", tmp_path)
    ref = build_encoder(ec, native_grid_size=16)
    torch.save(ref.model.state_dict(), ec.weights)
    actual = load_encoder(ec)
    assert actual.native_grid_size == 16 and actual.model.pos_embed.shape == (1, 257, 768)
    with pytest.raises(ValueError, match="divisible"):
        actual(torch.zeros(1, 3, 33, 64))


@pytest.mark.parametrize("kind", ["convnext", "visionfm"])
def test_self_contained_checkpoint_needs_no_source_weights(kind, tmp_path, shallow_visionfm):
    ec = alternative_cfg(kind, tmp_path)
    save_weights(ec, pretrained=False)
    cfg = Config(encoder=ec)
    enc = load_encoder(ec)
    net = Task1Net(enc, MarioHeads(enc.out_dim, hidden=16, dt_dim=4, dropout=0)).eval()
    identity = store_identity(cfg)
    c = {"kind": "finetuned", "format_version": 1, "task": 1, "fold": 0,
         "encoder": net.encoder.state_dict(), "heads": net.heads.state_dict(), "hparams": net.heads.hparams,
         "encoder_spec": encoder_spec(cfg, net.encoder, identity)}
    Path(ec.weights).unlink()
    Path(ec.weights + ".json").unlink()
    restored = load_finetuned_checkpoint(c)
    x = torch.randn(1, 3, *ec.img_size)
    with torch.no_grad():
        torch.testing.assert_close(net(x, x.flip(-1)), restored(x, x.flip(-1)), rtol=0, atol=0)
    bad = copy.deepcopy(c)
    bad["encoder_spec"]["out_dim"] += 1
    with pytest.raises(ValueError, match="dimension"):
        load_finetuned_checkpoint(bad)


def test_legacy_defaults_and_feature_identity_unchanged():
    # Simulate old checkpoint config with every new field absent.
    cfg = config_from_dict({"encoder": {"weights": None}})
    assert cfg.encoder.kind == "dinov2" and cfg.encoder.preprocessing == "legacy_dino"
    expected = {"encoder": cfg.encoder.name, "encoder_kwargs": {}, "img_size_hw": [196, 518],
                "weights_sha256": None, "preprocess_mode": "classic", "pooling": POOLING,
                "views": ["identity", "hflip"]}
    assert store_identity(cfg) == expected
    image = np.arange(100, dtype=np.uint8).reshape(10, 10)
    assert np.array_equal(preprocess(image), preprocess_input(image, resolve_input_spec(cfg.encoder)))


@pytest.mark.parametrize("kind", ["convnext", "visionfm"])
def test_presets_and_early_partial_guard(kind):
    cfg = load_config(str(ROOT / f"configs/task1_{kind}.yaml"))
    assert cfg.encoder.kind == kind and cfg.encoder.img_size == [192, 512]
    assert cfg.finetune.init_heads_from is None
    with pytest.raises(ValueError, match="train_finetune"):
        train_all_partial(cfg)
    with pytest.raises(ConfigError):
        load_config(str(ROOT / f"configs/task1_{kind}.yaml"), ["encoder.name=wrong"])
    if kind == "visionfm":
        assert load_config(str(ROOT / f"configs/task1_{kind}.yaml"), ["encoder.img_size=[224,224]"])


def test_new_identities_include_transform_and_kind():
    cfg = Config(encoder=EncoderCfg(kind="visionfm", name="vit_base_patch16", weights=None,
                                   img_size=[192, 512], preprocessing="visionfm_oct"))
    identity = store_identity(cfg)
    assert identity["identity_version"] == 2 and identity["pooling"] == "final-normalized CLS"
    cfg.encoder.preprocessing = "imagenet"
    with pytest.raises(ValueError, match="differ"):
        check_identity([{"store_meta": {"identity": identity}}], store_identity(cfg))
    with pytest.raises(ValueError, match="architecture"):
        check_architecture([{"hparams": {"d": 768}, "store_meta": {"identity": identity}}],
                           {**identity, "encoder_kind": "convnext"}, 768)


def test_checkpoint_inventory_rejects_mixed_kinds(tmp_path):
    for k, kind in enumerate(["heads", "finetuned"]):
        folder = tmp_path / f"fold{k}"
        folder.mkdir()
        torch.save({"kind": kind, "hparams": {"d": 768}, "fold": k, "task": 1}, folder / "task1.pt")
    with pytest.raises(ValueError, match="mixed"):
        checkpoint_paths(str(tmp_path), 1)


@pytest.mark.parametrize("preset,mean,std", [
    ("imagenet", [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ("visionfm_oct", [0.21091926, 0.21091926, 0.21091919], [0.17598894, 0.17598891, 0.17598893]),
])
def test_uint8_transforms_preserve_intensity(preset, mean, std):
    spec = resolve_input_spec(EncoderCfg(preprocessing=preset, img_size=[32, 64]))
    result = preprocess_input(np.full((12, 20), 128, np.uint8), spec)
    ref = (128 / 255 - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    np.testing.assert_allclose(result, np.broadcast_to(ref[:, None, None], result.shape), rtol=1e-6)


@pytest.mark.parametrize("kind", ["convnext", "visionfm"])
def test_image_training_and_serial_inference_pipeline(kind, tmp_path, shallow_visionfm, monkeypatch):
    import weakref

    import pandas as pd

    from mario_dino import infer
    from mario_dino.data import schema
    from mario_dino.data.images import PairedImageDataset
    from mario_dino.data.splits import assign_folds, make_splits, save_splits
    from mario_dino.synthetic import make_synthetic_dataset
    from mario_dino.train_finetune import predict_pairs, train_all_finetune

    paths = make_synthetic_dataset(str(tmp_path / "data"), n_patients=4, n_heldout_patients=1, bscans_per_visit=1)
    cfg = Config(encoder=alternative_cfg(kind, tmp_path), device="cpu")
    cfg.data.image_root = str(tmp_path / "data/images")
    cfg.data.task1_train_csvs = [paths["task1_train"]]
    cfg.data.splits = str(tmp_path / "splits.json")
    cfg.data.n_folds = 2
    cfg.features.num_workers = 0
    cfg.features.amp = False
    cfg.preprocess.mode = "raw"
    cfg.preprocess.cache_dir = str(tmp_path / "cache")
    cfg.finetune.out_dir = cfg.inference.run_dir = str(tmp_path / "finetune")
    cfg.finetune.max_epochs = cfg.finetune.max_steps = 1
    cfg.finetune.batch_size = cfg.inference.batch_size = 4
    cfg.model.hidden, cfg.model.dt_dim, cfg.model.dropout = 16, 4, 0
    cfg.inference.out_dir = str(tmp_path / "infer")
    source_frame = pd.read_csv(paths["task1_train"])
    splits = make_splits(source_frame, n_folds=2)
    save_splits(splits, cfg.data.splits)
    save_weights(cfg.encoder, pretrained=False)
    result = train_all_finetune(cfg)
    assert result["folds"] == [0, 1]
    oof = pd.read_csv(Path(cfg.finetune.out_dir) / "oof_task1.csv")
    assert len(oof) == len(source_frame)
    assert oof.groupby("id_patient")["fold"].nunique().max() == 1
    for k in range(2):
        c = torch.load(Path(cfg.finetune.out_dir) / f"fold{k}/task1.pt", weights_only=False)
        assignment = assign_folds(source_frame, splits)
        assert c["dt_median_train"] == pytest.approx(source_frame.loc[assignment != k, "delta_t"].median())
        assert c["source"]["sha256"] and c["best_epoch"] == 1

    Path(cfg.encoder.weights).unlink()
    Path(cfg.encoder.weights + ".json").unlink()

    def forbidden(*args, **kwargs):
        raise AssertionError("fine-tuned inference opened source weights or a frozen store")

    monkeypatch.setattr(infer, "load_encoder", forbidden)
    monkeypatch.setattr(infer, "open_feature_store", forbidden)
    references = []
    original_loader = infer.load_finetuned_checkpoint

    def serial_loader(checkpoint, device):
        assert all(ref() is None for ref in references), "previous full fold model was retained"
        net = original_loader(checkpoint, device)
        references.append(weakref.ref(net.encoder))
        return net

    monkeypatch.setattr(infer, "load_finetuned_checkpoint", serial_loader)
    # Checkpoint owns its transform, independent of the current encoder initialization config.
    cfg.encoder.preprocessing, cfg.encoder.img_size = "legacy_dino", [224, 224]
    r = infer.run_inference(cfg, 1, paths["task1_val"])
    assert r["finetuned"] and r["n_folds"] == 2
    frame = schema.read_csv(paths["task1_val"], 1, image_root=cfg.data.image_roots)
    total = np.zeros((len(frame), 4), np.float32)
    for k in range(2):
        c = torch.load(Path(cfg.finetune.out_dir) / f"fold{k}/task1.pt", weights_only=False)
        net = original_loader(c)
        ds = PairedImageDataset(frame, cfg.preprocess.cache_dir, cfg.preprocess.mode, c["encoder_spec"]["input_spec"])
        total += predict_pairs(net, ds, np.arange(len(frame)), cfg, torch.device("cpu")) / 2
        del net
    np.testing.assert_array_equal(pd.read_csv(r["output"])["y_pred"], total.argmax(1))


def test_frozen_convnext_features_and_legacy_head_checkpoint(tmp_path):
    import pandas as pd

    from mario_dino.data import schema
    from mario_dino.data.splits import make_splits, save_splits
    from mario_dino.features import extract_features, open_feature_store
    from mario_dino.infer import run_inference
    from mario_dino.octip_cache import build_cache
    from mario_dino.synthetic import make_synthetic_dataset
    from mario_dino.train_heads import train_all

    paths = make_synthetic_dataset(str(tmp_path / "data"), n_patients=4, n_heldout_patients=1, bscans_per_visit=1)
    cfg = Config(encoder=alternative_cfg("convnext", tmp_path), device="cpu")
    cfg.data.image_root = str(tmp_path / "data/images")
    cfg.data.task1_train_csvs = [paths["task1_train"]]
    cfg.data.task2_train_csvs = [paths["task2_train"]]
    cfg.data.n_folds, cfg.train.folds = 2, [0]
    cfg.data.splits = str(tmp_path / "splits.json")
    save_splits(make_splits(pd.read_csv(paths["task1_train"]), n_folds=2), cfg.data.splits)
    cfg.preprocess.mode, cfg.preprocess.cache_dir = "raw", str(tmp_path / "cache")
    cfg.features.store_dir = str(tmp_path / "features")
    cfg.features.num_workers, cfg.features.amp = 0, False
    cfg.train.out_dir = cfg.inference.run_dir = str(tmp_path / "heads")
    cfg.inference.out_dir = str(tmp_path / "infer")
    cfg.model.hidden, cfg.model.dt_dim = 16, 4
    cfg.train.max_epochs = 1
    save_weights(cfg.encoder, pretrained=False)
    enc = load_encoder(cfg.encoder)
    frames = [(t, schema.read_csv(paths[f"task{t}_train"], t, image_root=cfg.data.image_roots)) for t in (1, 2)]
    names = schema.unique_images(frames)
    build_cache(names, cfg.data.image_roots, cfg.preprocess.cache_dir, "raw")
    store = open_feature_store(cfg, cfg.features.store_dir, enc.out_dim)
    extract_features(enc, names, store, cfg.preprocess.cache_dir, "raw", cfg.encoder.img_size,
                     torch.device("cpu"), batch_size=8, num_workers=0, amp=False,
                     input_spec=resolve_input_spec(cfg.encoder))
    assert store.get(names).shape == (len(names), 2, enc.out_dim)
    store.close()
    train_all(cfg)
    # Missing kind remains the historical heads format.
    path = Path(cfg.train.out_dir) / "fold0/task1.pt"
    c = torch.load(path, weights_only=False)
    del c["kind"]
    torch.save(c, path)
    r = run_inference(cfg, 1, paths["task1_test"])
    assert not r["finetuned"] and not r["partial"]
    r2 = run_inference(cfg, 2, paths["task2_test"])
    assert r2["n_folds"] == 1


def test_pair_dataset_uses_the_same_view_for_both_images(tmp_path):
    import cv2
    import pandas as pd

    from mario_dino.data.images import PairedImageDataset
    from mario_dino.octip_cache import cache_path

    mode = "raw"
    for name in ("a.png", "b.png"):
        path = Path(cache_path(str(tmp_path), mode, name))
        path.parent.mkdir(parents=True, exist_ok=True)
        image = np.tile(np.arange(64, dtype=np.uint8), (32, 1))
        cv2.imwrite(str(path), image)
    frame = pd.DataFrame({"image_at_ti": ["a.png"], "image_at_ti+1": ["b.png"], "delta_t": [2], "label": [1]})
    ds = PairedImageDataset(frame, str(tmp_path), mode,
                            resolve_input_spec(EncoderCfg(preprocessing="visionfm_oct", img_size=[32, 64])))
    a, b, dt, y = ds.pair(0, 0)
    fa, fb, _, _ = ds.pair(0, 1)
    assert dt == 2 and y == 1
    torch.testing.assert_close(fa, a.flip(-1))
    torch.testing.assert_close(fb, b.flip(-1))
