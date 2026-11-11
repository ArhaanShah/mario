"""Artifact-only Task 1 evaluation; no torch, encoder, checkpoint or image imports.

python -m mario_dino.benchmark --run baseline=outputs/convnext_finetune \
    --run candidate=outputs/candidate --splits outputs/splits.json --out-dir outputs/benchmark
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .study import read_outer_splits, sha256

CLASS_NAMES = ("REDUCED", "STABLE", "WORSENED", "OTHER")
HARD_METRICS = ("mcc", "micro_f1", "specificity", "macro_f1", "balanced_accuracy", "task1_composite")
LOGITS = [f"logit_{i}" for i in range(4)]
KEYS = ["case", "id_patient", "fold", "y_true"]


def validate_oof(frame, splits):
    required = {*KEYS, "y_pred", *LOGITS}
    if required - set(frame) or frame.empty:
        raise ValueError(f"empty predictions or missing OOF columns: {sorted(required - set(frame))}")
    frame = frame.copy()
    for key in ("case", "id_patient"):
        if frame[key].isna().any():
            raise ValueError(f"missing {key}")
        frame[key] = frame[key].astype(str)
        if frame[key].str.strip().eq("").any() or not frame[key].eq(frame[key].str.strip()).all():
            raise ValueError(f"blank or whitespace-padded {key}")
    if frame.case.duplicated().any():
        raise ValueError("OOF cases must be unique")
    for key, limit in (("fold", splits["n_folds"]), ("y_true", 4), ("y_pred", 4)):
        values = pd.to_numeric(frame[key], errors="raise")
        if values.isna().any() or not values.isin(range(limit)).all():
            raise ValueError(f"invalid integer {key}")
        frame[key] = values.astype(int)
    logits = frame[LOGITS].apply(pd.to_numeric, errors="raise").to_numpy(float)
    if not np.isfinite(logits).all():
        raise ValueError("OOF logits must be finite")
    frame[LOGITS] = logits
    if not np.array_equal(frame.y_pred, logits.argmax(axis=1)):
        raise ValueError("y_pred is inconsistent with logit argmax")
    assignment = frame.id_patient.map(splits["patients"])
    if assignment.isna().any() or not np.array_equal(assignment, frame.fold):
        raise ValueError("patient fold membership differs from the saved outer splits")
    return frame.sort_values("case").reset_index(drop=True)


def read_oof(path, splits):
    return validate_oof(pd.read_csv(path, dtype={"case": str, "id_patient": str}), splits)


def align_runs(runs, intersection=False):
    if not runs:
        raise ValueError("at least one run is required")
    sets = {name: set(frame.case) for name, frame in runs.items()}
    common = set.intersection(*sets.values())
    if not common:
        raise ValueError("runs have no common cases")
    if not intersection and any(cases != common for cases in sets.values()):
        raise ValueError("run case sets differ; use --intersection for an explicit restricted analysis")
    aligned = {name: frame.loc[frame.case.isin(common)].sort_values("case").reset_index(drop=True)
               for name, frame in runs.items()}
    reference = next(iter(aligned.values()))[KEYS]
    for name, frame in aligned.items():
        if not reference.equals(frame[KEYS]):
            raise ValueError(f"patient, label or fold disagreement between runs: {name}")
    exclusions = {name: sorted(cases - common) for name, cases in sets.items()}
    return aligned, exclusions


def confusion(y, pred):
    return np.bincount(np.asarray(y, int) * 4 + np.asarray(pred, int), minlength=16).reshape(4, 4)


def hard_metrics(cm):
    """Fixed four-class report, with the pinned challenge's specificity convention."""
    cm = np.asarray(cm, float)
    row, col = cm.sum(1), cm.sum(0)
    tp, n = np.diag(cm), cm.sum()
    if n <= 0:
        raise ValueError("cannot score an empty confusion matrix")
    recall = np.divide(tp, row, out=np.zeros(4), where=row > 0)
    f1 = np.divide(2 * tp, row + col, out=np.zeros(4), where=row + col > 0)
    denominator = np.sqrt(max(0., (n * n - col @ col) * (n * n - row @ row)))
    mcc = float((tp.sum() * n - row @ col) / denominator) if denominator > 0 else 0.
    # DF41 uses sklearn confusion_matrix without explicit labels: classes absent
    # from BOTH truth and prediction are omitted; retain its epsilon exactly.
    active = row + col > 0
    tn = n - row - col + tp
    specificity = float((tn[active] / (n - row[active] + 1e-6)).mean())
    micro = float(tp.sum() / n)
    return {"mcc": mcc, "micro_f1": micro, "specificity": specificity,
            "macro_f1": float(f1.mean()), "balanced_accuracy": float(recall[row > 0].mean()),
            "task1_composite": 0.5 * micro + 0.25 * mcc + 0.25 * specificity}


def per_class_metrics(cm):
    rows = []
    for k, name in enumerate(CLASS_NAMES):
        support, predicted, tp = int(cm[k].sum()), int(cm[:, k].sum()), int(cm[k, k])
        rows.append({"class": k, "name": name, "support": support, "predicted": predicted,
                     "precision": tp / predicted if predicted else None,
                     "recall": tp / support if support else None,
                     "f1": 2 * tp / (support + predicted) if support + predicted else None,
                     "undefined_reasons": {
                         **({"precision": "no predictions for this class"} if not predicted else {}),
                         **({"recall": "no true examples of this class"} if not support else {}),
                         **({"f1": "class absent from truth and predictions"} if not support + predicted else {}),
                     }})
    return rows


def probability_metrics(y, logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    p = np.exp(logp)
    confidence, predicted = p.max(1), p.argmax(1)
    bins = np.minimum((confidence * 15).astype(int), 14)
    calibration, ece = [], 0.
    for i in range(15):
        mask = bins == i
        count = int(mask.sum())
        acc = float((predicted[mask] == y[mask]).mean()) if count else None
        conf = float(confidence[mask].mean()) if count else None
        if count:
            ece += count / len(y) * abs(acc - conf)
        calibration.append({"bin": i, "lower": i / 15, "upper": (i + 1) / 15,
                            "n": count, "accuracy": acc, "confidence": conf})
    auc = []
    for k in range(4):
        target = y == k
        defined = target.any() and (~target).any()
        auc.append({"class": k, "auroc": float(roc_auc_score(target, p[:, k])) if defined else None,
                    "auprc": float(average_precision_score(target, p[:, k])) if defined else None,
                    "undefined_reason": None if defined else "requires both positive and negative examples"})
    return {"nll": float(-logp[np.arange(len(y)), y].mean()),
            "brier": float(((p - np.eye(4)[y]) ** 2).sum(1).mean()),
            "ece_15": float(ece), "ovr": auc, "calibration": calibration}


def patient_bootstrap_indices(frame, rng):
    """One paired draw; repeated patient draws repeat ALL that patient's rows."""
    chunks = []
    for _, group in frame.groupby("fold", sort=True):
        patients = sorted(group.id_patient.unique())
        lookup = {p: group.index[group.id_patient == p].to_numpy() for p in patients}
        chunks.extend(lookup[p] for p in rng.choice(patients, size=len(patients), replace=True))
    return np.concatenate(chunks)


def bootstrap(runs, baseline, replicates=2000, seed=0):
    if replicates < 1 or baseline not in runs:
        raise ValueError("bootstrap needs a positive replicate count and a known baseline")
    reference = next(iter(runs.values()))
    rng = np.random.default_rng(seed)
    scores = {name: {m: [] for m in HARD_METRICS} for name in runs}
    # Precompute a confusion matrix per patient per run: bootstrap cost is then
    # independent of the number of B-scans, and patient multiplicity is exact.
    groups = list(reference.groupby(["fold", "id_patient"], sort=True))
    matrices = {name: np.stack([confusion(frame.loc[g.index, "y_true"], frame.loc[g.index, "y_pred"])
                                for _, g in groups]) for name, frame in runs.items()}
    strata = {}
    for i, ((fold, _), _) in enumerate(groups):
        strata.setdefault(fold, []).append(i)
    for _ in range(replicates):
        drawn = np.concatenate([rng.choice(indices, size=len(indices), replace=True) for indices in strata.values()])
        for name in runs:
            values = hard_metrics(matrices[name][drawn].sum(0))
            for metric, value in values.items():
                scores[name][metric].append(value)
    intervals, comparisons = {}, {}
    for name in runs:
        intervals[name] = {metric: {"lower": float(np.percentile(values, 2.5)),
                                    "upper": float(np.percentile(values, 97.5)), "valid_replicates": len(values)}
                           for metric, values in scores[name].items()}
        if name != baseline:
            delta = np.asarray(scores[name]["mcc"]) - scores[baseline]["mcc"]
            comparisons[name] = {"baseline": baseline, "endpoint": "paired MCC difference",
                                  "estimate": hard_metrics(confusion(runs[name].y_true, runs[name].y_pred))["mcc"]
                                  - hard_metrics(confusion(runs[baseline].y_true, runs[baseline].y_pred))["mcc"],
                                  "lower": float(np.percentile(delta, 2.5)),
                                  "upper": float(np.percentile(delta, 97.5)), "valid_replicates": replicates}
    return {"replicates": replicates, "seed": seed,
            "method": "paired patient-cluster percentile bootstrap, stratified by outer fold",
            "interpretation": "conditional on saved predictions; excludes training and model-selection variability",
            "intervals": intervals, "paired_differences": comparisons}


def attach_metadata(frame, metadata):
    if "case" not in metadata or metadata.case.isna().any() or metadata.case.astype(str).duplicated().any():
        raise ValueError("metadata requires unique nonmissing case identifiers")
    metadata = metadata.copy()
    metadata["case"] = metadata.case.astype(str)
    if set(frame.case) - set(metadata.case):
        raise ValueError("metadata is missing evaluated cases")
    joined = frame.merge(metadata, on="case", how="left", suffixes=("", "_metadata"), validate="one_to_one")
    for key in ("id_patient", "fold", "y_true"):
        other = f"{key}_metadata"
        if other in joined and not joined[key].astype(str).eq(joined[other].astype(str)).all():
            raise ValueError(f"metadata disagrees on {key}")
    if "label" in metadata and not joined.y_true.eq(pd.to_numeric(joined.label, errors="raise")).all():
        raise ValueError("metadata labels disagree with OOF truth")
    return joined


def slice_reports(frame):
    slices = []
    for field in ("side_eye", "BScan", "delta_t"):
        if field not in frame:
            continue
        if field == "delta_t":
            # Explicit fixed bins; no cut points are learned on held-out labels.
            groups = pd.cut(pd.to_numeric(frame[field], errors="raise"), [0, 30, 60, 90, np.inf],
                            labels=["(0,30]", "(30,60]", "(60,90]", ">90"])
        else:
            groups = frame[field].fillna("missing").astype(str)
        for value in sorted(groups.dropna().unique(), key=str):
            subset = frame.loc[groups == value]
            slices.append({"field": field, "value": str(value), "n": len(subset),
                           "n_patients": subset.id_patient.nunique(),
                           "class_support": np.bincount(subset.y_true, minlength=4).tolist(),
                           **hard_metrics(confusion(subset.y_true, subset.y_pred))})
        missing = frame.loc[groups.isna()]
        if len(missing):
            slices.append({"field": field, "value": "missing/outside positive time bins", "n": len(missing),
                           "n_patients": missing.id_patient.nunique(),
                           **hard_metrics(confusion(missing.y_true, missing.y_pred))})
    return slices


def history_report(run_dir):
    rows, reasons = [], []
    for path in sorted(Path(run_dir).glob("fold*/history.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        stages = data if isinstance(data, dict) else {"finetune": data}
        for stage, records in stages.items():
            if stage == "stage_b":
                continue
            if not isinstance(records, list):
                raise ValueError(f"invalid history records in {path}")
            for record in records:
                rows.append({**record, "fold": path.parent.name, "stage": stage})
    if not rows:
        reasons.append("training histories unavailable; train/validation divergence cannot be assessed")
    assessments = []
    if rows:
        table = pd.DataFrame(rows)
        for (fold, stage), group in table.groupby(["fold", "stage"]):
            valid = group.dropna(subset=["train_ce", "val_loss"]) if {"train_ce", "val_loss"} <= set(group) else pd.DataFrame()
            if len(valid) >= 2:
                first, last = valid.sort_values("epoch").iloc[[0, -1]].to_dict("records")
                assessments.append({"fold": fold, "stage": stage,
                                    "train_ce_change": last["train_ce"] - first["train_ce"],
                                    "val_loss_change": last["val_loss"] - first["val_loss"],
                                    "possible_divergence": last["train_ce"] < first["train_ce"] and last["val_loss"] > first["val_loss"]})
            else:
                reasons.append(f"{fold}/{stage}: insufficient comparable training CE and validation loss records")
    return {"records": rows, "assessments": assessments, "limitations": reasons,
            "interpretation": "descriptive trends only; weighted training and validation CE can differ in composition"}


def describe_run(frame, run_dir, metadata=None):
    cm = confusion(frame.y_true, frame.y_pred)
    augmented = attach_metadata(frame, metadata) if metadata is not None else frame
    wrong = frame.assign(error=frame.y_true != frame.y_pred)
    patients = wrong.groupby("id_patient").agg(n=("case", "size"), errors=("error", "sum"))
    patients["error_rate"] = patients.errors / patients.n
    total_errors = int(patients.errors.sum())
    top = patients.sort_values(["errors", "error_rate"], ascending=False).reset_index()
    patient_weights = 1 / frame.id_patient.map(frame.id_patient.value_counts()).to_numpy()
    equal_patient_cm = np.bincount(frame.y_true * 4 + frame.y_pred, weights=patient_weights, minlength=16).reshape(4, 4)
    diagnostics = {
        "other_to_stable": int(cm[3, 1]), "other_support": int(cm[3].sum()),
        "reduced_to_worsened": int(cm[0, 2]), "worsened_to_reduced": int(cm[2, 0]),
        "class_support": cm.sum(1).astype(int).tolist(),
        "majority_class_accuracy": float(cm.sum(1).max() / cm.sum()),
        "majority_baseline_note": "descriptive evaluated-cohort prevalence, not a fitted baseline",
        "top_5_patients_error_share": float(top.head(5).errors.sum() / total_errors) if total_errors else 0.,
    }
    reasons = {}
    if (cm.sum(0) > 0).sum() == 1 or (cm.sum(1) > 0).sum() == 1:
        reasons["mcc"] = "zero contingency variance: reported as 0 using the sklearn/challenge convention"
    return {"n": len(frame), "n_patients": frame.id_patient.nunique(),
            "folds": sorted(frame.fold.unique().tolist()), "metrics": hard_metrics(cm),
            "undefined_or_conventional": reasons, "per_class": per_class_metrics(cm), "confusion_matrix": cm.tolist(),
            "probability_metrics": probability_metrics(frame.y_true.to_numpy(), frame[LOGITS].to_numpy()),
            "per_fold": {str(k): {"n": len(g), "n_patients": g.id_patient.nunique(),
                                   **hard_metrics(confusion(g.y_true, g.y_pred))} for k, g in frame.groupby("fold")},
            "equal_patient_weight_metrics": hard_metrics(equal_patient_cm), "diagnostics": diagnostics,
            "patient_errors": top.to_dict("records"), "slices": slice_reports(augmented),
            "histories": history_report(run_dir),
            "missing_metadata": [key for key in ("side_eye", "BScan", "delta_t") if key not in augmented]}


def _clean(value):
    """Strict JSON: unavailable values are null, never nonstandard NaN/Infinity."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(_clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def plot_run(run, root, prefix):
    # Imported only when exporting figures; never import torchvision or a model.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(run["confusion_matrix"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, matrix, title in zip(axes, [cm, np.divide(cm, cm.sum(1, keepdims=True),
                                                      out=np.zeros_like(cm, dtype=float), where=cm.sum(1, keepdims=True) > 0)],
                                 ["Counts", "Recall by true class"], strict=True):
        ax.imshow(matrix, cmap="Blues")
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, f"{value:.2f}" if title != "Counts" else str(value), ha="center", va="center")
        ax.set(xticks=range(4), yticks=range(4), xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
               xlabel="Predicted", ylabel="True", title=title)
        ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(root / f"{prefix}_confusion.png", dpi=180)
    plt.close(fig)
    calibration = run["probability_metrics"]["calibration"]
    used = [b for b in calibration if b["n"]]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.scatter([b["confidence"] for b in used], [b["accuracy"] for b in used],
               s=[20 + 80 * b["n"] / run["n"] for b in used])
    ax.set(xlabel="Mean confidence", ylabel="Accuracy", title="Calibration (15 bins)", xlim=(0, 1), ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(root / f"{prefix}_calibration.png", dpi=180)
    plt.close(fig)
    records = run["histories"]["records"]
    if records:
        table = pd.DataFrame(records)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for (fold, stage), group in table.groupby(["fold", "stage"]):
            for ax, keys in zip(axes, [("train_ce", "val_loss"), ("val_Rk-correlation",)], strict=True):
                for key in keys:
                    if key in group:
                        ax.plot(group.epoch, group[key], label=f"{fold}/{stage}/{key}")
        for ax in axes:
            ax.set_xlabel("Epoch")
            if ax.lines:
                ax.legend(fontsize=6)
        axes[0].set_title("CE trends (composition can differ)")
        axes[1].set_title("Selection validation MCC")
        fig.tight_layout()
        fig.savefig(root / f"{prefix}_histories.png", dpi=180)
        plt.close(fig)


def export_report(report, out_dir):
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "report.json", report)
    rows, perclass, slices = [], [], []
    text = ["# Task 1 artifact benchmark", "", "MCC is the primary endpoint. Micro-F1 equals accuracy for this single-label task.",
            "", "95% intervals condition on saved predictions and exclude training variability. Probability metrics have point estimates only.",
            "Architecture comparisons are predeclared; intervals are descriptive and are not multiplicity-adjusted significance tests.", "",
            "| Run | Cases | Patients | MCC (95% CI) | Micro-F1 | Macro-F1 | OTHER recall | Composite |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for i, (name, run) in enumerate(report["runs"].items()):
        m = run["metrics"]
        interval = report["bootstrap"]["intervals"][name]["mcc"]
        recall = run["per_class"][3]["recall"]
        other = "undefined" if recall is None else f"{recall:.4f}"
        safe_name = name.replace("|", "/").replace("\n", " ")
        text.append(f"| {safe_name} | {run['n']} | {run['n_patients']} | {m['mcc']:.4f} [{interval['lower']:.4f}, {interval['upper']:.4f}] "
                    f"| {m['micro_f1']:.4f} | {m['macro_f1']:.4f} | {other} | {m['task1_composite']:.4f} |")
        rows.append({"run": name, "n": run["n"], "n_patients": run["n_patients"], **m,
                     **{key: run["probability_metrics"][key] for key in ("nll", "brier", "ece_15")}})
        perclass.extend({"run": name, **row} for row in run["per_class"])
        slices.extend({"run": name, **row} for row in run["slices"])
        pd.DataFrame(run["patient_errors"]).to_csv(root / f"run{i}_patient_errors.csv", index=False)
        plot_run(run, root, f"run{i}")
    pd.DataFrame(rows).to_csv(root / "metrics.csv", index=False)
    pd.DataFrame(perclass).to_csv(root / "per_class.csv", index=False)
    pd.DataFrame(slices).to_csv(root / "slices.csv", index=False)
    differences = report["bootstrap"]["paired_differences"]
    pd.DataFrame([{"run": name, **value} for name, value in differences.items()]).to_csv(root / "paired_differences.csv", index=False)
    text.extend(["", "## Paired MCC differences", ""])
    for name, delta in differences.items():
        text.append(f"- {name} versus {delta['baseline']}: {delta['estimate']:+.4f}, 95% CI [{delta['lower']:+.4f}, {delta['upper']:+.4f}].")
    text.extend(["", "## Diagnostics and limitations", ""])
    for i, (name, run) in enumerate(report["runs"].items()):
        d = run["diagnostics"]
        text.append(f"- {name}: OTHER → STABLE {d['other_to_stable']}/{d['other_support']}; REDUCED → WORSENED {d['reduced_to_worsened']}; "
                    f"WORSENED → REDUCED {d['worsened_to_reduced']}. Top five patients account for {d['top_5_patients_error_share']:.1%} of errors.")
        text.append(f"- {name}: unavailable metadata: {', '.join(run['missing_metadata']) or 'none'}; "
                    f"history limitations: {'; '.join(run['histories']['limitations']) or 'none'}. Figures use prefix run{i}.")
    text.extend(["", f"Case alignment: {report['alignment']}. Excluded case IDs are listed in report.json.",
                 "Missing classes: macro-F1 uses four classes with zero for undefined class F1; balanced accuracy averages classes present in truth.",
                 "Specificity reproduces DF41's active-class and epsilon convention. Per-class undefined quantities are null with reasons.",
                 "Historical OOF selected on the reported validation fold remains exploratory; bootstrap cannot remove selection bias.",
                 "No leaderboard ranking or score-improvement claim follows from this report.", ""])
    (root / "report.md").write_text("\n".join(text), encoding="utf-8")


def benchmark_runs(paths, splits_path, out_dir, metadata_path=None, intersection=False, replicates=2000, seed=0, baseline=None):
    if not paths:
        raise ValueError("at least one named run is required")
    splits = read_outer_splits(splits_path)
    frames = {name: read_oof(Path(path) / "oof_task1.csv", splits) for name, path in paths.items()}
    frames, excluded = align_runs(frames, intersection)
    metadata = pd.read_csv(metadata_path, dtype={"case": str, "id_patient": str}) if metadata_path else None
    report = {"format_version": 1, "task": 1, "primary_endpoint": "MCC",
              "alignment": "explicit intersection" if intersection else "identical case sets",
              "excluded_cases": excluded, "outer_splits_sha256": sha256(splits_path),
              "metadata_sha256": sha256(metadata_path) if metadata_path else None,
              "inputs": {name: {"directory": str(Path(path).resolve()),
                                "oof_sha256": sha256(Path(path) / "oof_task1.csv"),
                                "artifacts": {str(p.relative_to(path)): sha256(p) for p in sorted(Path(path).rglob("*"))
                                              if p.is_file() and p.name in ("history.json", "metrics.json", "provenance.json", "config.resolved.yaml")}}
                         for name, path in paths.items()},
              "runs": {name: describe_run(frame, paths[name], metadata) for name, frame in frames.items()},
              "bootstrap": bootstrap(frames, baseline or next(iter(paths)), replicates, seed)}
    export_report(report, out_dir)
    return report


def benchmark_study(study_path, splits_path, out_dir, metadata_path=None, replicates=2000, seed=0):
    """Separate per-seed paired studies; summarize seed variability without pooling logits."""
    from .config import load_config
    from .study import SEEDS, VARIANTS, json_hash

    study = json.loads(Path(study_path).read_text(encoding="utf-8"))
    experiments = study["experiments"]
    expected = {(seed, variant) for seed in SEEDS for variant in VARIANTS}
    if len(experiments) != len(expected) or {(item["seed"], item["variant"]) for item in experiments} != expected:
        raise ValueError("study must contain each of the four locked variants for seeds 0, 1 and 2 exactly once")
    if splits_path is None:
        splits_path = load_config(experiments[0]["config"]).data.splits
    splits = read_outer_splits(splits_path)
    cohort = json.loads((Path(study_path).parent / "cohort.json").read_text(encoding="utf-8"))
    inner = json.loads((Path(study_path).parent / "inner_splits.json").read_text(encoding="utf-8"))
    if (json_hash(cohort) != study["dataset_hash"] or json_hash(inner) != study["inner_split_hash"]
            or sha256(splits_path) != cohort["outer_splits_sha256"]):
        raise ValueError("study cohort, inner splits or outer split hash changed")
    missing = [str(Path(item["out_dir"]) / "oof_task1.csv") for item in experiments
               if not (Path(item["out_dir"]) / "oof_task1.csv").is_file()]
    if missing:
        raise FileNotFoundError("incomplete study OOF artifacts: " + ", ".join(missing))
    frames = {f"{item['seed']}/{item['variant']}": read_oof(Path(item["out_dir"]) / "oof_task1.csv", splits)
              for item in experiments}
    align_runs(frames)
    if any(set(frame.fold) != set(range(4)) for frame in frames.values()):
        raise ValueError("study aggregation requires all four outer folds for every seed and variant")
    identity = None
    for item in experiments:
        frame = frames[f"{item['seed']}/{item['variant']}"]
        if (set(frame.case) != set(cohort["cases"])
                or dict(zip(frame.case, frame.id_patient, strict=True)) != cohort["case_patients"]
                or dict(zip(frame.case, frame.y_true, strict=True)) != cohort["case_labels"]):
            raise ValueError("study OOF cases, patients or labels differ from the locked cohort")
        provenance = json.loads((Path(item["out_dir"]) / "provenance.json").read_text(encoding="utf-8"))
        if (provenance.get("protocol") != "grouped_holdout" or provenance.get("seed") != item["seed"]
                or set(provenance.get("folds", [])) != set(range(4))
                or provenance.get("dataset_hash") != study["dataset_hash"]
                or provenance.get("inner_split_hash") != study["inner_split_hash"]
                or provenance.get("config_hash") != item["config_hash"]):
            raise ValueError("run provenance differs from its locked experiment definition")
        source = provenance.get("identity", {})
        if not source.get("weights_sha256") or (identity is not None and source != identity):
            raise ValueError("study runs require identical pretrained encoder and preprocessing identities")
        identity = source
    root = Path(out_dir)
    per_seed = {}
    for training_seed in SEEDS:
        paths = {item["variant"]: item["out_dir"] for item in experiments if item["seed"] == training_seed}
        per_seed[training_seed] = benchmark_runs(paths, splits_path, root / f"seed{training_seed}", metadata_path,
                                                 False, replicates, seed, "legacy")
    rows = []
    for variant in VARIANTS:
        row = {"variant": variant, "n_seeds": len(SEEDS)}
        for metric in (*HARD_METRICS, "other_recall"):
            values = [per_seed[s]["runs"][variant]["per_class"][3]["recall"] if metric == "other_recall"
                      else per_seed[s]["runs"][variant]["metrics"][metric] for s in SEEDS]
            defined = [v for v in values if v is not None]
            row[f"{metric}_mean"] = float(np.mean(defined)) if defined else None
            row[f"{metric}_sd"] = float(np.std(defined, ddof=1)) if len(defined) > 1 else None
            row[f"{metric}_valid_seeds"] = len(defined)
        rows.append(row)
    summary = {"study_sha256": sha256(study_path), "seeds": list(SEEDS), "variants": rows,
               "interpretation": "mean and sample SD across matched training seeds; no logit pooling; paired conditional CIs are in each seed report",
               "limitations": "three seeds do not estimate full training-population uncertainty; no automatic winner or multiplicity-adjusted significance claim"}
    write_json(root / "seed_summary.json", summary)
    pd.DataFrame(rows).to_csv(root / "seed_summary.csv", index=False)
    lines = ["# ConvNeXt Task 1: three-seed study", "", summary["interpretation"], "",
             "| Variant | MCC mean ± SD | Macro-F1 mean ± SD | OTHER recall mean ± SD |", "|---|---:|---:|---:|"]
    for row in rows:
        def display(metric, row=row):
            mean, sd = row[f"{metric}_mean"], row[f"{metric}_sd"]
            return f"{mean:.4f} ± {sd:.4f}" if mean is not None and sd is not None else "undefined"

        lines.append(f"| {row['variant']} | {display('mcc')} | {display('macro_f1')} | {display('other_recall')} |")
    lines.extend(["", summary["limitations"], "", *[f"- [Seed {s}: matched patient intervals and paired differences](seed{s}/report.md)" for s in SEEDS], ""])
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", action="append", metavar="NAME=DIRECTORY")
    source.add_argument("--study", help="prepared study.json; requires all four variants and three seeds")
    parser.add_argument("--splits", help="required with --run; optional override with --study")
    parser.add_argument("--metadata", help="Task 1 source CSV with case and optional eye/BScan/delta_t columns")
    parser.add_argument("--intersection", action="store_true")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baseline", help="named reference run; defaults to the first --run")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    if args.study:
        if args.intersection or args.baseline:
            parser.error("study mode requires complete matching cohorts and the locked legacy baseline")
        benchmark_study(args.study, args.splits, args.out_dir, args.metadata, args.bootstrap_replicates, args.seed)
        print(f"Artifact-only three-seed study: {Path(args.out_dir) / 'report.md'}")
        return
    if not args.splits:
        parser.error("--splits is required with --run")
    paths = {}
    for item in args.run:
        name, separator, path = item.partition("=")
        if not separator or not name.strip() or not path.strip() or name in paths:
            parser.error("--run requires a unique nonempty NAME=DIRECTORY")
        paths[name] = path
    report = benchmark_runs(paths, args.splits, args.out_dir, args.metadata, args.intersection,
                            args.bootstrap_replicates, args.seed, args.baseline)
    print(f"Artifact-only report: {Path(args.out_dir) / 'report.md'} ({len(report['runs'])} runs)")


if __name__ == "__main__":
    main()
