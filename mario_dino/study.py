"""Model-free preparation and validation of the locked Task 1 study.

Preparation reads CSV/JSON/YAML artifacts and writes configurations. It never
imports torch, extracts features, loads weights, or launches experiment commands.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .config import config_to_dict, load_config

VARIANTS = {
    "legacy": ("legacy", None),
    "delta_mlp": ("delta_mlp", None),
    "other_gate": ("other_gate", None),
    "legacy_no_forecast": ("legacy", 0.0),
}
SEEDS = (0, 1, 2)
COHORT_COLUMNS = ("case", "id_patient", "label", "image_at_ti", "image_at_ti+1", "delta_t")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_outer_splits(path):
    splits = json.loads(Path(path).read_text(encoding="utf-8"))
    n = splits.get("n_folds")
    mapping = splits.get("patients")
    if not isinstance(n, int) or isinstance(n, bool) or n < 2 or not isinstance(mapping, dict) or not mapping:
        raise ValueError("invalid outer patient split file")
    if any(not str(p).strip() or not isinstance(k, int) or isinstance(k, bool) or not 0 <= k < n
           for p, k in mapping.items()):
        raise ValueError("invalid outer patient assignment")
    if set(mapping.values()) != set(range(n)):
        raise ValueError("every outer fold must contain patients")
    return splits


def read_cohort(paths):
    if not paths:
        raise ValueError("Task 1 source CSVs are required")
    frame = pd.concat([pd.read_csv(p, dtype={"case": str, "id_patient": str}) for p in paths], ignore_index=True)
    missing = set(COHORT_COLUMNS) - set(frame)
    if missing or frame.empty:
        raise ValueError(f"empty cohort or missing columns: {sorted(missing)}")
    for key in COHORT_COLUMNS:
        if frame[key].isna().any() or frame[key].astype(str).str.strip().eq("").any():
            raise ValueError(f"missing cohort values: {key}")
    if frame.case.duplicated().any():
        raise ValueError("cohort cases must be unique across source CSVs")
    labels = pd.to_numeric(frame.label, errors="raise")
    gaps = pd.to_numeric(frame.delta_t, errors="raise")
    if not labels.isin(range(4)).all() or not np.isfinite(gaps).all() or (gaps <= 0).any():
        raise ValueError("invalid cohort labels or time gaps")
    frame["label"] = labels.astype(int)
    frame["delta_t"] = gaps.astype(float)
    return frame


def cohort_manifest(frame, csvs, splits_path):
    outer = read_outer_splits(splits_path)
    if set(frame.id_patient) - set(outer["patients"]):
        raise ValueError("cohort has patients missing from outer splits")
    # Include all source bytes, not just labels or image filenames.
    return {
        "format_version": 1,
        "sources": [{"path": str(Path(p).resolve()), "sha256": sha256(p)} for p in csvs],
        "outer_splits_sha256": sha256(splits_path),
        "n_cases": len(frame), "n_patients": frame.id_patient.nunique(),
        "cases": sorted(frame.case.tolist()),
        "case_patients": dict(zip(frame.case, frame.id_patient, strict=True)),
        "case_labels": dict(zip(frame.case, frame.label.map(int), strict=True)),
        "class_support": [int((frame.label == i).sum()) for i in range(4)],
    }


def make_inner_splits(outer, patients, fraction=0.2, seed=0):
    patients = sorted(set(map(str, patients)))
    if not 0 < fraction < 1 or set(patients) - set(outer["patients"]):
        raise ValueError("invalid inner fraction or unknown patients")
    folds = {}
    for k in range(outer["n_folds"]):
        test = sorted(p for p in patients if outer["patients"][p] == k)
        train = [p for p in patients if outer["patients"][p] != k]
        if not test or len(train) < 2:
            raise ValueError(f"fold {k} needs outer-test patients and at least two training patients")
        ranked = sorted(train, key=lambda p: json_hash([seed, k, p]))
        count = min(len(train) - 1, max(1, math.ceil(fraction * len(train))))
        folds[str(k)] = {"fit": sorted(ranked[count:]), "selection": sorted(ranked[:count]), "test": test}
    return {"format_version": 1, "fraction": fraction, "split_seed": seed, "folds": folds}


def validate_inner_splits(inner, outer, patients):
    patients = set(map(str, patients))
    if inner.get("format_version") != 1 or set(inner.get("folds", {})) != {str(k) for k in range(outer["n_folds"])}:
        raise ValueError("incomplete inner split manifest")
    for k in range(outer["n_folds"]):
        fold = inner["folds"][str(k)]
        groups = [fold.get(name, []) for name in ("fit", "selection", "test")]
        if any(not g or len(g) != len(set(g)) for g in groups):
            raise ValueError("inner splits must have nonempty unique patient groups")
        fit, selection, test = map(set, groups)
        expected_test = {p for p in patients if outer["patients"][p] == k}
        if fit & selection or fit & test or selection & test or fit | selection | test != patients or test != expected_test:
            raise ValueError(f"patient leakage or cohort mismatch in fold {k}")
    expected = make_inner_splits(outer, patients, fraction=0.2, seed=0)
    if inner != expected:
        raise ValueError("inner splits differ from the locked deterministic 20% patient protocol")


def check_cohort(cfg, frame):
    """Called before any model construction or retinal-cache work in research training."""
    expected = json.loads(Path(cfg.research.cohort_manifest).read_text(encoding="utf-8"))
    source = read_cohort(cfg.data.task1_train_csvs)
    actual = cohort_manifest(source, cfg.data.task1_train_csvs, cfg.data.splits)
    if expected != actual:
        raise ValueError("source CSVs or outer splits changed since study preparation")
    # Training must not silently drop cases that happened to have unreadable images.
    cases = frame.case.astype(str)
    if cases.duplicated().any() or set(cases) != set(actual["cases"]):
        raise ValueError("readable training cohort differs from the prepared study; repair data and prepare again")
    for case, patient, label in zip(cases, frame.id_patient.astype(str), frame.label, strict=True):
        if actual["case_patients"][case] != patient or actual["case_labels"][case] != int(label):
            raise ValueError("training case identity or label differs from the prepared cohort")
    outer = read_outer_splits(cfg.data.splits)
    if outer["n_folds"] != cfg.data.n_folds:
        raise ValueError("configured fold count differs from the saved outer splits")
    inner = json.loads(Path(cfg.research.inner_splits).read_text(encoding="utf-8"))
    validate_inner_splits(inner, outer, source.id_patient)
    return actual, outer, inner


def write_locked(path, content):
    """Allow identical preparation again, never overwrite different experiment definitions."""
    path = Path(path)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to overwrite different study artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)


def prepare_study(cfg, out_dir):
    if cfg.encoder.kind != "convnext" or cfg.data.n_folds != 4:
        raise ValueError("this study requires ConvNeXt and four existing patient folds")
    if cfg.train.lambda_pred <= 0 or cfg.model.head_input != "pair" or cfg.model.forecaster != "gru":
        raise ValueError("study base requires paired inputs, GRU and a positive forecasting-loss weight")
    if cfg.finetune.init_heads_from or cfg.finetune.max_steps:
        raise ValueError("study requires fresh heads and no max_steps cap")
    root = Path(out_dir).resolve()
    frame = read_cohort(cfg.data.task1_train_csvs)
    outer = read_outer_splits(cfg.data.splits)
    if outer["n_folds"] != 4:
        raise ValueError("the saved outer split file must contain four folds")
    manifest = cohort_manifest(frame, cfg.data.task1_train_csvs, cfg.data.splits)
    inner = make_inner_splits(outer, frame.id_patient)
    validate_inner_splits(inner, outer, frame.id_patient)
    write_locked(root / "cohort.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_locked(root / "inner_splits.json", json.dumps(inner, indent=2, sort_keys=True) + "\n")
    base = config_to_dict(cfg)
    experiments = []
    for seed in SEEDS:
        for name, (fusion, weight) in VARIANTS.items():
            # Roundtrip copies nested values without mutating the caller's configuration.
            spec = json.loads(json.dumps(base))
            run_dir = root / "runs" / f"seed{seed}" / name
            config_path = root / "configs" / f"seed{seed}_{name}.yaml"
            spec["seed"] = seed
            spec["data"]["task2_train_csvs"] = []
            spec["train"]["select_metric"] = "mcc"
            spec["train"]["folds"] = None
            spec["model"].update(fusion=fusion, fusion_hidden=256)
            spec["train"]["lambda_pred"] = cfg.train.lambda_pred if weight is None else weight
            spec["finetune"]["out_dir"] = str(run_dir)
            spec["inference"]["run_dir"] = str(run_dir)
            spec["inference"]["out_dir"] = str(run_dir / "inference")
            spec["research"] = {"protocol": "grouped_holdout", "inner_splits": str(root / "inner_splits.json"),
                                "cohort_manifest": str(root / "cohort.json")}
            write_locked(config_path, yaml.safe_dump(spec, sort_keys=False))
            experiments.append({"seed": seed, "variant": name, "config": str(config_path),
                                "config_hash": json_hash(spec),
                                "out_dir": str(run_dir), "outer_folds": list(range(4)),
                                "command": ["python", "-m", "mario_dino.train_finetune", "--config", str(config_path)]})
    study = {"format_version": 1, "primary_endpoint": "MCC", "inner_selection": "MCC",
             "protocol": "20% patient inner holdout; fresh refit for selected epoch count",
             "seeds": list(SEEDS), "dataset_hash": json_hash(manifest), "inner_split_hash": json_hash(inner),
             "base_config_hash": json_hash(base), "experiments": experiments}
    write_locked(root / "study.json", json.dumps(study, indent=2, sort_keys=True) + "\n")
    return study


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task1_convnext.yaml")
    parser.add_argument("--out-dir", default="outputs/convnext_study")
    args = parser.parse_args(argv)
    study = prepare_study(load_config(args.config), args.out_dir)
    print("Prepared only; no model commands were executed.")
    print("seed  variant                 outer folds")
    for item in study["experiments"]:
        print(f"{item['seed']:4}  {item['variant']:23} {item['outer_folds']}")


if __name__ == "__main__":
    main()
