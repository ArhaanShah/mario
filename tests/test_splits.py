import os

import pandas as pd
import pytest

from mario_dino.data import schema
from mario_dino.data.alignment import label_alignment
from mario_dino.data.splits import assign_folds, load_splits, make_splits, save_splits
from mario_dino.data.splits import main as splits_main
from mario_dino.utils import REPO_ROOT, THIRD_PARTY


def _train(synth):
    return pd.read_csv(synth["csv"]["task1_train"]), pd.read_csv(synth["csv"]["task2_train"])


def test_no_patient_in_two_folds_and_tasks_share_assignment(synth, tmp_path):
    t1, t2 = _train(synth)
    splits = make_splits(t1, t2, n_folds=4)
    path = str(tmp_path / "splits.json")
    save_splits(splits, path)
    splits = load_splits(path)

    f1, f2 = assign_folds(t1, splits), assign_folds(t2, splits)
    rows = pd.concat(
        [pd.DataFrame({"pid": t1["id_patient"], "fold": f1}), pd.DataFrame({"pid": t2["id_patient"], "fold": f2})]
    )
    assert (rows.groupby("pid")["fold"].nunique() == 1).all()
    assert set(rows["fold"]) == {0, 1, 2, 3}
    # one assignment for both tasks: a patient's Task 1 and Task 2 folds agree
    m1 = dict(zip(t1["id_patient"], f1))
    m2 = dict(zip(t2["id_patient"], f2))
    assert all(m1[p] == m2[p] for p in m2)


def test_splits_reject_unknown_patient(synth):
    t1, t2 = _train(synth)
    splits = make_splits(t1, t2, n_folds=4)
    other = t2.head(1).assign(id_patient=99999)
    with pytest.raises(ValueError, match="missing from splits"):
        assign_folds(other, splits)


def test_splits_cli(tmp_path):
    from mario_dino.synthetic import make_synthetic_dataset

    paths = make_synthetic_dataset(str(tmp_path / "d"), seed=1, n_patients=5, n_heldout_patients=1)
    out = tmp_path / "splits.json"
    splits_main(
        [
            "--config",
            os.path.join(REPO_ROOT, "configs", "smoke.yaml"),
            "--set",
            f"data.task1_train_csvs=[{paths['task1_train']}]",
            "--set",
            f"data.task2_train_csvs=[{paths['task2_train']}]",
            "--set",
            f"data.splits={out}",
        ]
    )
    assert len(load_splits(str(out))["patients"]) == 5


def test_label_alignment_on_df41_val_csvs():
    csv_dir = os.path.join(THIRD_PARTY, "csv")
    if not os.path.isdir(csv_dir):
        pytest.skip("third_party/mario_challenge_code not checked out")
    t1 = schema.read_csv(os.path.join(csv_dir, "df_task1_val_challenge.csv"), 1, check_images=False)
    t2 = schema.read_csv(os.path.join(csv_dir, "df_task2_val_challenge.csv"), 2, check_images=False)
    r = label_alignment(t1, t2)
    assert (r["n_task2"], r["n_matched"], r["n_agree"]) == (3822, 3822, 2140)


def test_label_alignment_synthetic(synth):
    t1, t2 = _train(synth)
    r = label_alignment(t1, t2)
    assert r["n_matched"] == len(t2)
    assert r["crosstab"].to_numpy().sum() == len(t2)
