"""Patient-grouped, shared cross-validation folds for MARIO Tasks 1 and 2."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from ..cli import base_parser, load_from_args
from . import schema


def make_splits(task1: pd.DataFrame, task2: pd.DataFrame | None = None, n_folds: int = 4) -> dict:
    """Assign every patient to one deterministic GroupKFold, balancing rows across both tasks."""
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {n_folds}")
    if "id_patient" not in task1:
        raise ValueError("Task 1 rows need id_patient to create patient-grouped folds")
    patient_parts = [task1["id_patient"].astype(str)]
    if task2 is not None and len(task2):
        task2 = schema.attach_task2_patients(task2, task1)
        patient_parts.append(task2["id_patient"].astype(str))
    patients_per_row = pd.concat(patient_parts, ignore_index=True)
    unique_patients = sorted(patients_per_row.unique())
    if len(unique_patients) < n_folds:
        raise ValueError(f"need at least {n_folds} patients for {n_folds} folds, got {len(unique_patients)}")
    assignments: dict[str, int] = {}
    groups = patients_per_row.to_numpy()
    splitter = GroupKFold(n_splits=n_folds)
    for fold, (_, heldout) in enumerate(splitter.split(np.zeros(len(groups)), groups=groups)):
        assignments.update({str(pid): fold for pid in np.unique(groups[heldout])})
    return {"n_folds": n_folds, "patients": assignments}


def save_splits(splits: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def load_splits(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        splits = json.load(f)
    if not isinstance(splits, dict) or not isinstance(splits.get("patients"), dict):
        raise ValueError(f"invalid patient-fold file: {path!r}")
    return splits


def assign_folds(df: pd.DataFrame, splits: dict) -> np.ndarray:
    """Map each row's patient to a fold; raise rather than silently assigning unknown patients."""
    mapping = splits["patients"]
    missing = sorted(set(df["id_patient"].astype(str)) - set(mapping))
    if missing:
        raise ValueError(f"patients missing from splits: {missing[:10]}")
    return df["id_patient"].astype(str).map(mapping).to_numpy(dtype=np.int64)


def main(argv: list[str] | None = None) -> None:
    p = base_parser("Create shared patient-grouped folds for Tasks 1 and 2.")
    args = p.parse_args(argv)
    cfg = load_from_args(args)
    task1 = pd.concat([pd.read_csv(path) for path in cfg.data.task1_train_csvs], ignore_index=True)
    task2 = (
        pd.concat([pd.read_csv(path) for path in cfg.data.task2_train_csvs], ignore_index=True)
        if cfg.data.task2_train_csvs
        else None
    )
    if task2 is not None:
        task2 = schema.attach_task2_patients(task2, task1)
    splits = make_splits(task1, task2, cfg.data.n_folds)
    save_splits(splits, cfg.data.splits)
    print(f"wrote {len(splits['patients'])} patient assignments across {splits['n_folds']} folds to {cfg.data.splits}")


if __name__ == "__main__":
    main()
