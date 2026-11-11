"""Artifact-only tests. No fixtures construct, load, train or execute models."""
import copy
import json
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef

from mario_dino.benchmark import (
    LOGITS,
    align_runs,
    attach_metadata,
    benchmark_runs,
    benchmark_study,
    bootstrap,
    confusion,
    hard_metrics,
    patient_bootstrap_indices,
    per_class_metrics,
    probability_metrics,
    validate_oof,
)
from mario_dino.config import ConfigError, load_config
from mario_dino.study import (
    check_cohort,
    make_inner_splits,
    prepare_study,
    read_cohort,
    read_outer_splits,
    validate_inner_splits,
)

ROOT = Path(__file__).resolve().parents[1]
CONVNEXT_CM = np.array([[1605, 330, 174, 47], [559, 8010, 477, 364], [200, 265, 1103, 62], [138, 524, 135, 498]])


@pytest.fixture
def tmp_path():
    # Windows managed sandboxes cannot reopen pytest's mode-0700 directories.
    # Keep disposable artifact fixtures in an ignored workspace cache instead.
    path = ROOT / ".cache" / "artifact-tests" / uuid.uuid4().hex
    path.mkdir(parents=True)
    return path


@pytest.fixture
def artifacts():
    frame = pd.DataFrame({"case": [f"case{i:03}" for i in range(16)],
                          "id_patient": [f"p{i // 2}" for i in range(16)],
                          "fold": [i // 8 for i in range(16)], "y_true": np.arange(16) % 4})
    frame["y_pred"] = frame.y_true
    frame.loc[[0, 4, 9], "y_pred"] = [1, 2, 3]
    logits = np.full((16, 4), -2.)
    logits[np.arange(16), frame.y_pred] = 2.
    frame[LOGITS] = logits
    splits = {"n_folds": 2, "patients": {f"p{i}": i // 4 for i in range(8)}}
    return frame, splits


def test_matrix_reproduces_notebook_and_independent_metrics():
    y, pred = [], []
    for (i, j), count in np.ndenumerate(CONVNEXT_CM):
        y.extend([i] * count)
        pred.extend([j] * count)
    actual = hard_metrics(CONVNEXT_CM)
    assert actual["mcc"] == pytest.approx(0.5860071300207579)
    assert actual["mcc"] == pytest.approx(matthews_corrcoef(y, pred))
    assert actual["micro_f1"] == pytest.approx(f1_score(y, pred, average="micro"))
    assert actual["macro_f1"] == pytest.approx(f1_score(y, pred, average="macro"))
    assert actual["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(y, pred))
    assert actual["task1_composite"] == pytest.approx(0.5 * actual["micro_f1"] + 0.25 * (actual["mcc"] + actual["specificity"]))
    assert per_class_metrics(CONVNEXT_CM)[3]["recall"] == pytest.approx(498 / 1295)


@pytest.mark.parametrize("y,pred", [([0, 1, 2, 3], [1, 1, 2, 3]), ([0, 0, 1], [0, 1, 1]), ([1, 1], [1, 1])])
def test_challenge_specificity_parity(y, pred):
    from mario_dino.scoring import specificity

    assert hard_metrics(confusion(y, pred))["specificity"] == pytest.approx(specificity(y, pred))


@pytest.mark.parametrize("damage", ["duplicate", "nan_logits", "infinite_logits", "prediction", "label", "fractional_fold", "patient", "missing", "blank"])
def test_invalid_predictions_rejected(artifacts, damage):
    frame, splits = artifacts
    if damage == "duplicate":
        frame.loc[1, "case"] = frame.loc[0, "case"]
    elif damage == "nan_logits":
        frame.loc[0, "logit_0"] = np.nan
    elif damage == "infinite_logits":
        frame.loc[0, "logit_0"] = np.inf
    elif damage == "prediction":
        frame.loc[0, "y_pred"] = 0
    elif damage == "label":
        frame.loc[0, "y_true"] = 4
    elif damage == "fractional_fold":
        frame["fold"] = frame.fold.astype(float)
        frame.loc[0, "fold"] = 0.5
    elif damage == "patient":
        frame.loc[0, "id_patient"] = "p7"
    elif damage == "missing":
        frame = frame.drop(columns="logit_3")
    else:
        frame.loc[0, "case"] = " "
    with pytest.raises(ValueError):
        validate_oof(frame, splits)


def test_alignment_is_paired_and_explicit(artifacts):
    frame, _ = artifacts
    partial = frame.iloc[1:]
    with pytest.raises(ValueError, match="case sets"):
        align_runs({"base": frame, "partial": partial})
    aligned, excluded = align_runs({"base": frame.sample(frac=1, random_state=0), "partial": partial}, True)
    assert aligned["base"]["case"].equals(aligned["partial"]["case"])
    assert excluded == {"base": ["case000"], "partial": []}
    bad = frame.copy()
    bad.loc[0, "y_true"] = 2
    with pytest.raises(ValueError, match="disagreement"):
        align_runs({"base": frame, "bad": bad})


def test_missing_classes_have_reasons():
    cm = confusion([1, 1], [1, 1])
    classes = per_class_metrics(cm)
    assert classes[3]["recall"] is None and classes[3]["undefined_reasons"]["recall"]
    assert hard_metrics(cm)["mcc"] == 0
    assert hard_metrics(cm)["macro_f1"] == 0.25
    p = probability_metrics(np.array([1, 1]), np.zeros((2, 4)))
    assert all(row["auroc"] is None and row["undefined_reason"] for row in p["ovr"])
    assert p["nll"] == pytest.approx(np.log(4))
    assert p["brier"] == pytest.approx(0.75)


def test_probability_metrics_are_shift_invariant_and_unweighted(artifacts):
    frame, _ = artifacts
    a = probability_metrics(frame.y_true.to_numpy(), frame[LOGITS].to_numpy())
    b = probability_metrics(frame.y_true.to_numpy(), frame[LOGITS].to_numpy() + 10000)
    for metric in ("nll", "brier", "ece_15"):
        assert a[metric] == pytest.approx(b[metric])
    assert sum(row["n"] for row in a["calibration"]) == len(frame)


def test_bootstrap_preserves_patient_clusters_and_fold_sizes(artifacts):
    frame, _ = artifacts
    indices = patient_bootstrap_indices(frame, np.random.default_rng(7))
    sampled = frame.loc[indices]
    assert sampled.groupby("fold").size().to_dict() == frame.groupby("fold").size().to_dict()
    counts = sampled.case.value_counts()
    for _, group in frame.groupby("id_patient"):
        assert counts.reindex(group.case, fill_value=0).nunique() == 1
    report = bootstrap({"base": frame, "same": frame.copy()}, "base", replicates=31, seed=9)
    assert report == bootstrap({"base": frame, "same": frame.copy()}, "base", replicates=31, seed=9)
    delta = report["paired_differences"]["same"]
    assert delta["lower"] == delta["upper"] == delta["estimate"] == 0


def test_metadata_must_not_relabel_or_duplicate_cases(artifacts):
    frame, _ = artifacts
    meta = frame[["case", "id_patient"]].assign(side_eye="R", BScan=1, delta_t=42)
    assert "side_eye" in attach_metadata(frame, meta)
    with pytest.raises(ValueError, match="unique"):
        attach_metadata(frame, pd.concat([meta, meta.iloc[:1]]))
    with pytest.raises(ValueError, match="labels"):
        attach_metadata(frame, meta.assign(label=0))


def test_complete_artifact_export_has_no_model_imports(artifacts, tmp_path):
    frame, splits = artifacts
    run = tmp_path / "run"
    run.mkdir()
    frame.to_csv(run / "oof_task1.csv", index=False)
    (run / "metrics.json").write_text('{"task1": {}}')
    (run / "fold0").mkdir()
    (run / "fold0/history.json").write_text(json.dumps([
        {"epoch": 1, "train_ce": 1.2, "val_loss": 1.3},
        {"epoch": 2, "train_ce": 0.8, "val_loss": 1.5}]))
    split_path = tmp_path / "splits.json"
    split_path.write_text(json.dumps(splits))
    output = tmp_path / "report"
    report = benchmark_runs({"base": run, "same": run}, split_path, output, replicates=20)
    assert report["runs"]["base"]["histories"]["assessments"][0]["possible_divergence"]
    assert (output / "report.md").is_file() and (output / "run0_confusion.png").is_file()
    assert (output / "metrics.csv").is_file() and (output / "per_class.csv").is_file()
    raw = (output / "report.json").read_text()
    assert "NaN" not in raw and "Infinity" not in raw
    assert not any(name.startswith("mario_dino.models") for name in sys.modules)


@pytest.fixture
def prepared_study(tmp_path):
    rows = []
    for p in range(8):
        for label in range(4):
            rows.append({"case": f"c{p}_{label}", "id_patient": str(p), "label": label,
                         "image_at_ti": f"{p}_{label}_a.png", "image_at_ti+1": f"{p}_{label}_b.png", "delta_t": 42})
    csv = tmp_path / "cohort.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    splits_path = tmp_path / "splits.json"
    splits_path.write_text(json.dumps({"n_folds": 4, "patients": {str(p): p % 4 for p in range(8)}}))
    cfg = load_config(str(ROOT / "configs/task1_convnext_study.yaml"))
    cfg.data.task1_train_csvs = [str(csv)]
    cfg.data.splits = str(splits_path)
    root = tmp_path / "study"
    study = prepare_study(cfg, root)
    generated = load_config(study["experiments"][0]["config"])
    return generated, root, study


def test_study_preparation_is_deterministic_isolated_and_model_free(prepared_study):
    cfg, root, study = prepared_study
    assert len(study["experiments"]) == 12
    assert len({item["out_dir"] for item in study["experiments"]}) == 12
    assert not (root / "runs").exists()
    inner = json.loads((root / "inner_splits.json").read_text())
    outer = read_outer_splits(cfg.data.splits)
    assert inner == make_inner_splits(outer, outer["patients"])
    for group in inner["folds"].values():
        fit, selection, test = (set(group[k]) for k in ("fit", "selection", "test"))
        assert not fit & selection and not fit & test and not selection & test
    for item in study["experiments"]:
        c = load_config(item["config"])
        assert c.research.protocol == "grouped_holdout" and c.train.select_metric == "mcc"
        assert not c.data.task2_train_csvs
        assert c.train.lambda_pred == (0 if item["variant"] == "legacy_no_forecast" else 1)


def test_study_rejects_patient_leakage_and_changed_sources(prepared_study):
    cfg, root, _ = prepared_study
    frame = read_cohort(cfg.data.task1_train_csvs)
    check_cohort(cfg, frame)
    inner = json.loads((root / "inner_splits.json").read_text())
    bad = copy.deepcopy(inner)
    bad["folds"]["0"]["selection"].append(bad["folds"]["0"]["test"][0])
    with pytest.raises(ValueError, match="leakage"):
        validate_inner_splits(bad, read_outer_splits(cfg.data.splits), frame.id_patient)
    with pytest.raises(ValueError, match="readable training cohort"):
        check_cohort(cfg, frame.iloc[1:])
    with open(cfg.data.task1_train_csvs[0], "a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="changed"):
        check_cohort(cfg, frame)


def test_three_seed_summary_keeps_seeds_separate(prepared_study, tmp_path, monkeypatch):
    cfg, root, study = prepared_study
    cohort = read_cohort(cfg.data.task1_train_csvs)
    outer = read_outer_splits(cfg.data.splits)
    frame = cohort[["case", "id_patient"]].assign(y_true=cohort.label, y_pred=cohort.label)
    frame["fold"] = frame.id_patient.map(outer["patients"])
    logits = np.full((len(frame), 4), -2.)
    logits[np.arange(len(frame)), frame.y_pred] = 2.
    frame[LOGITS] = logits
    for item in study["experiments"]:
        directory = Path(item["out_dir"])
        directory.mkdir(parents=True)
        frame.to_csv(directory / "oof_task1.csv", index=False)
        (directory / "provenance.json").write_text(json.dumps({
            "protocol": "grouped_holdout", "seed": item["seed"], "folds": list(range(4)),
            "dataset_hash": study["dataset_hash"], "inner_split_hash": study["inner_split_hash"],
            "config_hash": item["config_hash"], "identity": {"weights_sha256": "0" * 64},
        }))
    # Figure generation is covered by the complete export test; avoid 24 duplicate plots here.
    monkeypatch.setattr("mario_dino.benchmark.plot_run", lambda *args: None)
    report = benchmark_study(root / "study.json", cfg.data.splits, tmp_path / "summary", replicates=5)
    assert len(report["variants"]) == 4
    for variant in report["variants"]:
        assert variant["mcc_mean"] == 1 and variant["mcc_sd"] == 0
    assert (tmp_path / "summary/seed2/paired_differences.csv").is_file()
    provenance_path = Path(study["experiments"][0]["out_dir"]) / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["config_hash"] = "tampered"
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="provenance"):
        benchmark_study(root / "study.json", cfg.data.splits, tmp_path / "invalid-summary", replicates=5)


@pytest.mark.parametrize("override", ["model.fusion=invalid", "model.fusion_hidden=0",
                                     "data.task2_train_csvs=[task2.csv]", "train.select_metric=f1",
                                     "finetune.init_heads_from=outputs/old", "finetune.max_steps=1", "encoder.weights=null"])
def test_research_configuration_guards(override):
    with pytest.raises(ConfigError):
        load_config(str(ROOT / "configs/task1_convnext_study.yaml"), [override])


def test_new_presets_and_legacy_defaults_load_without_models():
    legacy = load_config(str(ROOT / "configs/task1_convnext.yaml"))
    assert legacy.model.fusion == "legacy" and legacy.research.protocol == "legacy"
    for fusion in ("delta_mlp", "other_gate"):
        cfg = load_config(str(ROOT / f"configs/task1_convnext_{fusion}.yaml"))
        assert cfg.model.fusion == fusion and not cfg.data.task2_train_csvs
    code = "import sys; import mario_dino.benchmark, mario_dino.study; assert 'torch' not in sys.modules; assert 'timm' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_notebook_evidence_can_be_regenerated(tmp_path):
    from scripts.notebook_evidence import build_evidence

    report = build_evidence(ROOT / "notebooks/mario_training_dashboard.ipynb", tmp_path)
    assert report["runs"]["Fine-tuned ConvNeXt"]["confusion_matrix"][3][1] == 524
    assert report["runs"]["ConvNeXt embeddings + retrained heads"]["n"] is None
    assert (tmp_path / "notebook_evidence.md").is_file()
