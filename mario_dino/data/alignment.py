"""Task 1/Task 2 label agreement for examples sharing the same starting B-scan."""

from __future__ import annotations

import pandas as pd


def label_alignment(task1: pd.DataFrame, task2: pd.DataFrame) -> dict:
    """Return a class crosstab and agreement counts for Task 2 images matching Task 1 starts."""
    starts = task1[["image_at_ti", "label"]].drop_duplicates("image_at_ti")
    matched = task2[["image", "label"]].merge(
        starts, how="left", left_on="image", right_on="image_at_ti", suffixes=("_task2", "_task1")
    )
    matched = matched.dropna(subset=["label_task1", "label_task2"])
    n_matched = len(matched)
    n_agree = int((matched["label_task1"] == matched["label_task2"]).sum())
    crosstab = pd.crosstab(matched["label_task1"], matched["label_task2"]).reindex(
        index=range(4), columns=range(3), fill_value=0
    )
    return {
        "crosstab": crosstab,
        "n_task2": len(task2),
        "n_matched": n_matched,
        "n_agree": n_agree,
        "agreement": n_agree / len(task2) if len(task2) else 0.0,
    }
