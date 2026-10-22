"""Synthetic MARIO-shaped data for tests and the smoke run.

``make_synthetic_dataset`` writes grayscale B-scan PNGs (height 200, width 512 or 768, horizontal bands plus
noise), localizer PNGs and CSVs with the exact DF41 column sets. Labels follow the DF41 val class balance.
The output is a pure function of the seed.
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import pandas as pd

from .data.schema import TASK1_COLUMNS, TASK2_COLUMNS

# DF41 val label counts, indexed by class
TASK1_LABEL_P = np.array([1033, 4764, 698, 515], dtype=float) / 7010
TASK2_LABEL_P = np.array([351, 2821, 650], dtype=float) / 3822

BSCAN_HEIGHT = 200


def _bscan(rng: np.random.Generator, width: int) -> np.ndarray:
    y = np.arange(BSCAN_HEIGHT, dtype=np.float32)[:, None]
    x = np.linspace(-1, 1, width, dtype=np.float32)[None, :]
    img = np.full((BSCAN_HEIGHT, width), 20.0, dtype=np.float32)
    top = rng.uniform(50, 90)
    curve = rng.uniform(-15, 15)
    for _ in range(int(rng.integers(3, 7))):
        centre = top + rng.uniform(0, 80) + curve * x**2
        sigma = rng.uniform(1.5, 6.0)
        img += rng.uniform(60, 180) * np.exp(-0.5 * ((y - centre) / sigma) ** 2)
    img += rng.normal(0, 12, size=img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def _name(rng: np.random.Generator, used: set[str]) -> str:
    while True:
        n = f"{int(rng.integers(0, 16**8)):08X}.png"
        if n not in used:
            used.add(n)
            return n


def make_synthetic_dataset(
    out_dir: str,
    seed: int = 0,
    n_patients: int = 12,
    n_heldout_patients: int = 3,
    bscans_per_visit: int = 3,
    write_images: bool = True,
) -> dict[str, str]:
    """Write images and CSVs under ``out_dir``; returns {name: path} of the CSVs.

    Training patients go to ``df_task{1,2}_train.csv``. Held-out patients go to ``df_task{1,2}_val.csv``
    (with labels) and ``df_task{1,2}_test.csv`` (same rows, no ``label`` column). ``write_images=False``
    writes the CSVs only (for training on a synthetic feature store).
    """
    rng = np.random.default_rng(seed)
    img_dir = os.path.join(out_dir, "images")
    csv_dir = os.path.join(out_dir, "csv")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    used: set[str] = set()

    def write(name: str, img: np.ndarray) -> None:
        if write_images:
            cv2.imwrite(os.path.join(img_dir, name), img)

    rows1: dict[str, list[dict]] = {"train": [], "val": []}
    rows2: dict[str, list[dict]] = {"train": [], "val": []}
    case1 = case2 = 0
    for p in range(n_patients + n_heldout_patients):
        split = "train" if p < n_patients else "val"
        pid = int(p + 1)
        sex = "M" if rng.random() < 0.5 else "F"
        age0 = int(rng.integers(55, 90))
        eyes = ["OD", "OS"][: int(rng.integers(1, 3))]
        for eye in eyes:
            n_visits = int(rng.integers(3, 6))
            width = int(rng.choice([512, 768]))
            bscan_ids = sorted(int(b) for b in rng.choice(25, size=bscans_per_visit, replace=False))
            visits = []
            age = age0
            for v in range(n_visits):
                loc = _name(rng, used)
                write(loc, rng.integers(0, 256, size=(96, 96), dtype=np.uint8))
                imgs = {}
                for b in bscan_ids:
                    n = _name(rng, used)
                    write(n, _bscan(rng, width))
                    imgs[b] = n
                dt = int(rng.integers(8, 127)) if v > 0 else 0
                age += int(rng.random() < 0.3) if v > 0 else 0
                visits.append({"loc": loc, "imgs": imgs, "dt": dt, "age": age, "num": v + 1})
            for v in range(n_visits - 1):
                a, b_ = visits[v], visits[v + 1]
                for b in bscan_ids:
                    rows1[split].append(
                        {
                            "id_patient": pid,
                            "side_eye": eye,
                            "BScan": b,
                            "image_at_ti": a["imgs"][b],
                            "image_at_ti+1": b_["imgs"][b],
                            "split_type": split,
                            "LOCALIZER_at_ti+1": b_["loc"],
                            "LOCALIZER_at_ti": a["loc"],
                            "sex": sex,
                            "age_at_ti+1": b_["age"],
                            "age_at_ti": a["age"],
                            "num_current_visit_at_i+1": b_["num"],
                            "num_current_visit_at_i": a["num"],
                            "delta_t": b_["dt"],
                            "label": int(rng.choice(4, p=TASK1_LABEL_P)),
                            "case": case1,
                        }
                    )
                    case1 += 1
                    # Task 2 covers every image that starts a pair
                    rows2[split].append(
                        {
                            "id_patient": pid,
                            "side_eye": eye,
                            "BScan": b,
                            "image": a["imgs"][b],
                            "split_type": split,
                            "LOCALIZER": a["loc"],
                            "sex": sex,
                            "age": a["age"],
                            "num_current_visit": a["num"],
                            "label": int(rng.choice(3, p=TASK2_LABEL_P)),
                            "case": case2,
                        }
                    )
                    case2 += 1

    paths = {}
    for split in ("train", "val"):
        t1 = pd.DataFrame(rows1[split], columns=TASK1_COLUMNS)
        t2 = pd.DataFrame(rows2[split], columns=TASK2_COLUMNS)
        paths[f"task1_{split}"] = os.path.join(csv_dir, f"df_task1_{split}.csv")
        paths[f"task2_{split}"] = os.path.join(csv_dir, f"df_task2_{split}.csv")
        t1.to_csv(paths[f"task1_{split}"], index=False, lineterminator="\n")
        t2.to_csv(paths[f"task2_{split}"], index=False, lineterminator="\n")
        if split == "val":
            paths["task1_test"] = os.path.join(csv_dir, "df_task1_test.csv")
            paths["task2_test"] = os.path.join(csv_dir, "df_task2_test.csv")
            t1.assign(split_type="test").drop(columns="label").to_csv(
                paths["task1_test"], index=False, lineterminator="\n"
            )
            t2.assign(split_type="test").drop(columns="label").to_csv(
                paths["task2_test"], index=False, lineterminator="\n"
            )
    return paths


def make_planted_feature_store(
    root: str,
    task1: pd.DataFrame,
    task2: pd.DataFrame | None = None,
    dim: int = 64,
    signal: float = 2.0,
    noise: float = 1.0,
    seed: int = 0,
    dtype: str = "float32",
):
    """A FeatureStore whose Task 1 label is planted in the direction of f_next - f_t.

    Pairs are walked in visit order so that f_next of one pair is f_t of the next; f_next = f_t +
    signal * u_label + noise. View 1 is view 0 plus small noise. Task 2 labels carry no planted signal.
    """
    from .data.schema import unique_images
    from .store import FeatureStore

    rng = np.random.default_rng(seed)
    u = rng.normal(size=(4, dim))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    f: dict[str, np.ndarray] = {}
    order = task1.sort_values(["id_patient", "side_eye", "BScan", "num_current_visit_at_i"])
    for a, b, y in zip(order["image_at_ti"], order["image_at_ti+1"], order["label"]):
        if a not in f:
            f[a] = rng.normal(size=dim)
        f[b] = f[a] + signal * u[int(y)] + noise * rng.normal(size=dim)
    dfs = [(1, task1)] + ([(2, task2)] if task2 is not None else [])
    images = unique_images(dfs)
    for im in images:
        if im not in f:
            f[im] = rng.normal(size=dim)
    store = FeatureStore(root, (2, dim), dtype, identity={"synthetic": "planted", "seed": seed, "dim": dim})
    rows = store.ensure(images)
    v0 = np.stack([f[im] for im in images])
    v1 = v0 + 0.05 * rng.normal(size=v0.shape)
    store.write(rows, np.stack([v0, v1], axis=1))
    store.mark_done(rows)
    return store


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Write a synthetic MARIO-shaped dataset (images + CSVs).")
    p.add_argument("--out", default="smoke_out/data")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-patients", type=int, default=12)
    p.add_argument("--n-heldout-patients", type=int, default=3)
    args = p.parse_args(argv)
    paths = make_synthetic_dataset(args.out, args.seed, args.n_patients, args.n_heldout_patients)
    for k, v in paths.items():
        print(f"{k}: {v} ({len(pd.read_csv(v))} rows)")


if __name__ == "__main__":
    main()
