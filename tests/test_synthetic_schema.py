import hashlib
import os

import cv2
import pandas as pd
import pytest

from mario_dino.data import schema
from mario_dino.synthetic import make_synthetic_dataset


def _digest(root):
    h = hashlib.sha256()
    for dirpath, _, files in sorted(os.walk(root)):
        for f in sorted(files):
            h.update(f.encode())
            with open(os.path.join(dirpath, f), "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()


def test_generator_is_deterministic(tmp_path):
    a = make_synthetic_dataset(str(tmp_path / "a"), seed=7, n_patients=3, n_heldout_patients=1)
    b = make_synthetic_dataset(str(tmp_path / "b"), seed=7, n_patients=3, n_heldout_patients=1)
    make_synthetic_dataset(str(tmp_path / "c"), seed=8, n_patients=3, n_heldout_patients=1)
    assert _digest(tmp_path / "a") == _digest(tmp_path / "b")
    assert _digest(tmp_path / "a") != _digest(tmp_path / "c")
    assert set(a) == set(b)


def test_generated_csvs_have_exact_columns_and_validate(synth):
    t1 = pd.read_csv(synth["csv"]["task1_train"])
    t2 = pd.read_csv(synth["csv"]["task2_train"])
    assert list(t1.columns) == schema.TASK1_COLUMNS
    assert list(t2.columns) == schema.TASK2_COLUMNS
    schema.validate(t1, 1, image_root=synth["images"], check_localizers=True)
    schema.validate(t2, 2, image_root=synth["images"], check_localizers=True)
    assert t1["delta_t"].between(8, 126).all()
    # Task 2 covers every image that starts a Task 1 pair
    assert set(t2["image"]) == set(t1["image_at_ti"])


def test_generated_images_shape(synth):
    t1 = pd.read_csv(synth["csv"]["task1_train"])
    for name in t1["image_at_ti"].head(5):
        img = cv2.imread(os.path.join(synth["images"], name), cv2.IMREAD_UNCHANGED)
        assert img.ndim == 2 and img.shape[0] == 200 and img.shape[1] in (512, 768)


def test_test_csvs_are_label_free(synth):
    for task in (1, 2):
        df = pd.read_csv(synth["csv"][f"task{task}_test"])
        assert "label" not in df.columns
        schema.validate(df, task, image_root=synth["images"], require_label=False)
        with pytest.raises(schema.SchemaError, match="label"):
            schema.validate(df, task, image_root=synth["images"], require_label=True)


def test_validator_rejects_missing_column(synth):
    df = pd.read_csv(synth["csv"]["task1_train"]).drop(columns=["delta_t"])
    with pytest.raises(schema.SchemaError, match="delta_t"):
        schema.validate(df, 1, image_root=synth["images"])


def test_validator_rejects_missing_image(synth):
    df = pd.read_csv(synth["csv"]["task2_train"])
    df.loc[0, "image"] = "DOES_NOT_EXIST.png"
    with pytest.raises(schema.SchemaError, match="missing"):
        schema.validate(df, 2, image_root=synth["images"])


def test_validator_rejects_bad_labels(synth):
    df = pd.read_csv(synth["csv"]["task2_train"])
    df.loc[0, "label"] = 3  # Task 2 has 3 classes
    with pytest.raises(schema.SchemaError, match="label"):
        schema.validate(df, 2, image_root=synth["images"])


def test_df41_val_csvs_match_schema():
    from mario_dino.utils import THIRD_PARTY

    csv_dir = os.path.join(THIRD_PARTY, "csv")
    if not os.path.isdir(csv_dir):
        pytest.skip("third_party/mario_challenge_code not checked out (git submodule update --init)")
    t1 = pd.read_csv(os.path.join(csv_dir, "df_task1_val_challenge.csv"))
    t2 = pd.read_csv(os.path.join(csv_dir, "df_task2_val_challenge.csv"))
    assert list(t1.columns) == schema.TASK1_COLUMNS
    assert list(t2.columns) == schema.TASK2_COLUMNS
    schema.validate(t1, 1, check_images=False)
    schema.validate(t2, 2, check_images=False)
