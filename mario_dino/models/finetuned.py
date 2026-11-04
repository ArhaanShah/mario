"""Versioned, self-contained full-encoder Task 1 inference checkpoints."""
from ..config import EncoderCfg
from ..data.images import resolve_input_spec
from .encoder import build_encoder
from .full import Task1Net
from .heads import heads_from_hparams

FORMAT_VERSION = 1
POOLING = {"convnext": "native average pool and final normalization", "visionfm": "final-normalized CLS"}


def encoder_spec(cfg, encoder, identity):
    return {"kind": cfg.encoder.kind, "name": cfg.encoder.name, "kwargs": dict(cfg.encoder.kwargs),
            "pooling": POOLING[cfg.encoder.kind], "out_dim": encoder.out_dim,
            "native_grid_size": getattr(encoder, "native_grid_size", None),
            "img_size_hw": list(cfg.encoder.img_size), "input_spec": resolve_input_spec(cfg.encoder),
            "preprocess_mode": cfg.preprocess.mode, "weights_sha256": identity["weights_sha256"],
            "checkpoint_key": cfg.encoder.checkpoint_key}


def load_finetuned_checkpoint(checkpoint, device="cpu"):
    """Reconstruct using stored architecture; source weight files are never consulted."""
    c = checkpoint
    if c.get("kind") != "finetuned" or c.get("format_version") != FORMAT_VERSION or c.get("task") != 1:
        raise ValueError("unsupported fine-tuned checkpoint kind, format version, or task")
    spec = c["encoder_spec"]
    kind = spec["kind"]
    if kind not in POOLING or spec["pooling"] != POOLING[kind]:
        raise ValueError("unsupported fine-tuned encoder/pooling")
    ec = EncoderCfg(kind=kind, name=spec["name"], weights=None, img_size=spec["img_size_hw"],
                    kwargs=spec["kwargs"], preprocessing=spec["input_spec"]["preset"],
                    checkpoint_key=spec["checkpoint_key"])
    if resolve_input_spec(ec) != spec["input_spec"]:
        raise ValueError("checkpoint transform specification is inconsistent or unsupported")
    if any(s <= 0 or s % (16 if kind == "visionfm" else 32) for s in ec.img_size):
        raise ValueError("invalid fine-tuned input dimensions")
    enc = build_encoder(ec, native_grid_size=spec["native_grid_size"])
    if enc.out_dim != spec["out_dim"] or c["hparams"]["d"] != enc.out_dim:
        raise ValueError("fine-tuned checkpoint feature dimension mismatch")
    enc.load_state_dict(c["encoder"], strict=True)
    heads = heads_from_hparams(c["hparams"])
    heads.load_state_dict(c["heads"], strict=True)
    return Task1Net(enc, heads).to(device).eval()
