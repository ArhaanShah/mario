"""Validation and image lookup for the DF41/MARIO Task 1 and Task 2 CSV formats."""

from __future__ import annotations

import os
from collections.abc import Sequence

import cv2
import pandas as pd

TASK1_COLUMNS = [
    "id_patient",
    "side_eye",
    "BScan",
    "image_at_ti",
    "image_at_ti+1",
    "split_type",
    "LOCALIZER_at_ti+1",
    "LOCALIZER_at_ti",
    "sex",
    "age_at_ti+1",
    "age_at_ti",
    "num_current_visit_at_i+1",
    "num_current_visit_at_i",
    "delta_t",
    "label",
    "case",
]
TASK2_COLUMNS = [
    "id_patient",
    "side_eye",
    "BScan",
    "image",
    "split_type",
    "LOCALIZER",
    "sex",
    "age",
    "num_current_visit",
    "label",
    "case",
]
# Some released MARIO Task 2 CSVs omit ``id_patient``.  The training pipeline derives it from the
# corresponding Task 1 start image, whose patient identifier is authoritative.
TASK2_REQUIRED_COLUMNS = [column for column in TASK2_COLUMNS if column != "id_patient"]

_IMAGE_COLUMNS = {1: ("image_at_ti", "image_at_ti+1"), 2: ("image",)}
_LOCALIZER_COLUMNS = {1: ("LOCALIZER_at_ti", "LOCALIZER_at_ti+1"), 2: ("LOCALIZER",)}


class SchemaError(ValueError):
    """Raised when a challenge CSV is missing required columns or contains invalid rows."""


def detect_task(df: pd.DataFrame) -> int:
    """Infer the task from its B-scan filename columns."""
    columns = set(df.columns)
    if set(_IMAGE_COLUMNS[1]).issubset(columns):
        return 1
    if set(_IMAGE_COLUMNS[2]).issubset(columns):
        return 2
    raise SchemaError("cannot detect task: expected Task 1 image_at_ti/image_at_ti+1 or Task 2 image")


def _image_roots(image_root: str | os.PathLike | Sequence[str | os.PathLike]) -> list[str]:
    if isinstance(image_root, (str, os.PathLike)):
        roots = [os.fspath(image_root)]
    else:
        roots = [os.fspath(root) for root in image_root]
    if not roots:
        raise ValueError("at least one image root is required")
    return roots


def image_path(image_root: str | os.PathLike | Sequence[str | os.PathLike], image: str) -> str:
    """Resolve a CSV image name under the first configured directory containing it."""
    candidates = [os.path.join(root, str(image)) for root in _image_roots(image_root)]
    return next((path for path in candidates if os.path.isfile(path)), candidates[0])


def readable_image_path(
    image_root: str | os.PathLike | Sequence[str | os.PathLike], image: str
) -> str | None:
    for root in _image_roots(image_root):
        path = os.path.join(root, str(image))
        if os.path.isfile(path) and cv2.imread(path, cv2.IMREAD_GRAYSCALE) is not None:
            return path
    return None


def has_labels(df: pd.DataFrame) -> bool:
    return "label" in df.columns and not df["label"].isna().any()


def validate(
    df: pd.DataFrame,
    task: int,
    *,
    image_root: str | os.PathLike | Sequence[str | os.PathLike] | None = None,
    require_label: bool = True,
    check_images: bool = True,
    check_localizers: bool = False,
    source: str = "CSV",
) -> pd.DataFrame:
    """Validate challenge columns, labels, and optionally referenced files; return the input frame."""
    if task not in (1, 2):
        raise ValueError(f"task must be 1 or 2, got {task!r}")
    expected = TASK1_COLUMNS if task == 1 else TASK2_REQUIRED_COLUMNS
    required = [c for c in expected if c != "label"] if not require_label else expected
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(f"{source}: missing required column(s): {', '.join(missing)}")
    if df.empty:
        raise SchemaError(f"{source}: no rows")

    image_columns = list(_IMAGE_COLUMNS[task])
    for column in image_columns:
        bad = df[column].isna() | (df[column].astype(str).str.strip() == "")
        if bad.any():
            raise SchemaError(f"{source}: missing image filename in {column}")

    if "label" in df.columns:
        labels = pd.to_numeric(df["label"], errors="coerce")
        n_classes = 4 if task == 1 else 3
        bad = labels.isna() | (labels % 1 != 0) | (labels < 0) | (labels >= n_classes)
        if bad.any():
            raise SchemaError(f"{source}: label must be an integer in [0, {n_classes - 1}]")
        df["label"] = labels.astype("int64")

    if task == 1:
        dt = pd.to_numeric(df["delta_t"], errors="coerce")
        if dt.isna().any() or (dt <= 0).any():
            raise SchemaError(f"{source}: delta_t must contain positive numeric values")
        df["delta_t"] = dt

    if (check_images or check_localizers) and image_root is not None:
        columns = image_columns[:]
        if check_localizers:
            columns.extend(_LOCALIZER_COLUMNS[task])
        missing_files = sorted(
            {
                str(name)
                for column in columns
                for name in df[column].dropna().astype(str)
                if not any(os.path.isfile(os.path.join(root, str(name))) for root in _image_roots(image_root))
            }
        )
        if missing_files:
            raise SchemaError(
                f"{source}: {len(missing_files)} referenced image(s) missing under "
                f"{_image_roots(image_root)!r}, "
                f"e.g. {missing_files[:5]}"
            )
    return df


def attach_task2_patients(task2: pd.DataFrame, task1: pd.DataFrame, *, source: str = "Task 2 CSV") -> pd.DataFrame:
    """Return Task 2 rows with ``id_patient`` populated from matching Task 1 start images.

    Task 2 describes the image at time ``t``. When its source CSV omits patient IDs, each image must map to
    exactly one Task 1 ``image_at_ti`` row. This preserves patient-grouped folds without inventing a
    row-level split that could leak a patient's scans into validation.
    """
    if "id_patient" in task2.columns and task2["id_patient"].notna().all():
        return task2
    if "image" not in task2.columns:
        raise SchemaError(f"{source}: cannot derive id_patient without the image column")
    required_task1 = {"image_at_ti", "id_patient"}
    missing_task1 = sorted(required_task1 - set(task1.columns))
    if missing_task1:
        raise SchemaError(f"cannot derive Task 2 patients: Task 1 is missing {', '.join(missing_task1)}")

    mapping = task1[["image_at_ti", "id_patient"]].dropna().copy()
    patient_counts = mapping.groupby("image_at_ti")["id_patient"].nunique()
    ambiguous = patient_counts[patient_counts > 1].index.tolist()
    if ambiguous:
        raise SchemaError(
            "cannot derive Task 2 patients: Task 1 start images map to multiple patients, "
            f"e.g. {ambiguous[:5]}"
        )
    mapping = mapping.drop_duplicates("image_at_ti")
    out = task2.drop(columns=["id_patient"], errors="ignore").merge(
        mapping, how="left", left_on="image", right_on="image_at_ti", validate="many_to_one"
    ).drop(columns=["image_at_ti"])
    missing_images = out.loc[out["id_patient"].isna(), "image"].astype(str).unique().tolist()
    if missing_images:
        raise SchemaError(
            f"{source}: cannot derive id_patient for {len(missing_images)} image(s) from Task 1 start images, "
            f"e.g. {missing_images[:5]}"
        )
    return out


def read_csv(
    path: str,
    task: int,
    *,
    image_root: str | os.PathLike | Sequence[str | os.PathLike] | None = None,
    require_label: bool = True,
    check_images: bool = True,
    check_localizers: bool = False,
) -> pd.DataFrame:
    """Read and validate one challenge CSV."""
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as e:
        raise SchemaError(f"cannot read {path!r}: {e}") from e
    return validate(
        df,
        task,
        image_root=image_root,
        require_label=require_label,
        check_images=check_images,
        check_localizers=check_localizers,
        source=path,
    )


def read_csvs(
    paths: Sequence[str],
    task: int,
    *,
    image_root: str | os.PathLike | Sequence[str | os.PathLike] | None = None,
    require_label: bool = True,
    check_images: bool = True,
) -> pd.DataFrame:
    """Read, validate, and concatenate multiple CSVs for one task."""
    if not paths:
        raise SchemaError(f"no Task {task} CSV paths supplied")
    frames = [
        read_csv(
            path,
            task,
            image_root=image_root,
            require_label=require_label,
            check_images=check_images,
        )
        for path in paths
    ]
    return pd.concat(frames, ignore_index=True)


def unique_images(dfs: Sequence[tuple[int, pd.DataFrame]]) -> list[str]:
    """Sorted unique B-scan filenames from (task, frame) pairs; localizer images are not encoded."""
    images: set[str] = set()
    for task, df in dfs:
        if task not in (1, 2):
            raise ValueError(f"task must be 1 or 2, got {task!r}")
        for column in _IMAGE_COLUMNS[task]:
            if column not in df:
                raise SchemaError(f"Task {task} frame missing image column {column!r}")
            images.update(df[column].dropna().astype(str))
    return sorted(images)


def exclude_unreadable_rows(
    df: pd.DataFrame, task: int, image_root: str | os.PathLike | Sequence[str | os.PathLike]
) -> tuple[pd.DataFrame, list[str]]:
    """Drop rows that reference B-scans OpenCV cannot decode; return the excluded filenames."""
    if task not in (1, 2):
        raise ValueError(f"task must be 1 or 2, got {task!r}")
    roots = _image_roots(image_root)
    if not any(os.path.isdir(root) for root in roots):
        return df, []
    columns = _IMAGE_COLUMNS[task]
    images = {str(name) for column in columns for name in df[column].dropna().unique()}
    unreadable = sorted(
        name for name in images if readable_image_path(roots, name) is None
    )
    if not unreadable:
        return df, []
    keep = ~df[list(columns)].astype(str).isin(unreadable).any(axis=1)
    return df.loc[keep].copy(), unreadable
