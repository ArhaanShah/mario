"""Recover preliminary Task 1 evidence without model execution or invented OOF rows.

Matrices are manually transcribed from cell 27's embedded figures and checked
against saved aggregate outputs. Individual predictions cannot be recovered.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mario_dino.benchmark import CLASS_NAMES, hard_metrics, per_class_metrics, write_json
from mario_dino.study import sha256

MATRICES = {
    "Frozen DINOv2": [[1226, 385, 262, 282], [1185, 6240, 803, 1182], [353, 298, 721, 257], [193, 209, 123, 770]],
    "Fine-tuned ConvNeXt": [[1605, 330, 174, 47], [559, 8010, 477, 364], [200, 265, 1103, 62], [138, 524, 135, 498]],
}


def saved_text(cell):
    return "\n".join("".join(output.get("text", [])) or
                     "".join(output.get("data", {}).get("text/plain", [])) for output in cell.get("outputs", []))


def build_evidence(notebook, out_dir):
    data = json.loads(Path(notebook).read_text(encoding="utf-8"))
    cells = data["cells"]
    comparison = saved_text(cells[27])
    if "0.586007" not in comparison or "14491" not in comparison or "14489" not in comparison:
        raise ValueError("notebook outputs changed; re-inspect the embedded confusion matrices before updating the transcription")
    final_text = saved_text(cells[29])
    marker = "Fine-tuned ConvNeXt embedding head training finished. Overall out-of-fold metrics:"
    if marker not in final_text:
        raise ValueError("saved combined embedding-head results are unavailable")
    retrained, _ = json.JSONDecoder().raw_decode(final_text.split(marker, 1)[1].lstrip())
    rows = []
    evidence = {"notebook": str(Path(notebook).as_posix()), "notebook_sha256": sha256(notebook),
                "matrix_source": "manual transcription of cell 27 embedded figures; rows=true, columns=predicted",
                "comparison_cell": 27, "retrained_heads_cell": 29, "runs": {},
                "limitations": ["no raw OOF predictions, patient splits, checkpoints or histories in this checkout",
                                "different cohorts: DINOv2 14489 vs ConvNeXt 14491 cases",
                                "best epochs were selected using the folds supplying the reported scores",
                                "no paired inference, uncertainty, individual examples or causal architecture conclusions"]}
    for name, values in MATRICES.items():
        cm = np.asarray(values)
        metrics = hard_metrics(cm)
        expected = 0.39756614676523133 if name == "Frozen DINOv2" else 0.5860071300207579
        if not np.isclose(metrics["mcc"], expected, atol=1e-12, rtol=0):
            raise ValueError("matrix transcription does not reproduce the saved MCC")
        classes = per_class_metrics(cm)
        evidence["runs"][name] = {"n": int(cm.sum()), "metrics": metrics, "per_class": classes,
                                  "confusion_matrix": values}
        rows.append({"run": name, "n": int(cm.sum()), **metrics, "other_recall": classes[3]["recall"]})
    m = retrained["task1"]
    evidence["runs"]["ConvNeXt embeddings + retrained heads"] = {
        "n": None, "micro_f1": m["F1_score"], "mcc": m["Rk-correlation"], "specificity": m["Specificity"],
        "note": "combined summary only; per-class metrics and row count not independently available"}
    rows.append({"run": "ConvNeXt embeddings + retrained heads", "n": None,
                 "micro_f1": m["F1_score"], "mcc": m["Rk-correlation"], "specificity": m["Specificity"],
                 "task1_composite": 0.5 * m["F1_score"] + 0.25 * m["Rk-correlation"] + 0.25 * m["Specificity"]})
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "notebook_evidence.json", evidence)
    pd.DataFrame(rows).to_csv(root / "notebook_metrics.csv", index=False)
    text = ["# ConvNeXt Task 1: preliminary notebook evidence", "",
            "These results were recovered from saved notebook outputs. No model was run. They are exploratory cross-validation results, not an independent test benchmark.", "",
            "| Model | Rows | Micro-F1 / accuracy | MCC | Macro-F1 | OTHER recall |",
            "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        n = str(row["n"]) if row["n"] is not None else "unverified"
        macro = f"{row['macro_f1']:.4f}" if "macro_f1" in row else "unavailable"
        recall = f"{row['other_recall']:.1%}" if "other_recall" in row else "unavailable"
        text.append(f"| {row['run']} | {n} | {row['micro_f1']:.4f} | {row['mcc']:.4f} | {macro} | {recall} |")
    text.extend(["", "## Observed failure modes", "",
                 "- ConvNeXt predicts STABLE for 524 of 1,295 OTHER cases (40.5%). OTHER recall is 38.5%, versus 59.5% with frozen DINOv2.",
                 "- STABLE accounts for 64.9% of the ConvNeXt cohort. Its recall is 85.1%; balanced accuracy across classes is 66.4%.",
                 "- ConvNeXt confuses REDUCED with WORSENED in 174 cases and WORSENED with REDUCED in 200 cases.",
                 "- Retraining heads adds approximately 0.0043 MCC. Significance cannot be assessed without matched case predictions.", "",
                 "These observations motivate a nonlinear comparison head and a separate OTHER gate. They do not establish that either architecture will improve performance.",
                 "OTHER combines uninterpretable scans and mixed changes; it is not a pure quality label. "
                 "[Official Task 1 definitions](https://youvenz.github.io/MARIO_challenge.github.io/docs/compose/task1/)", "",
                 "## Evidence and limits", "", f"Notebook SHA-256: `{evidence['notebook_sha256']}`.", "",
                 "Confusion matrices were transcribed from cell 27's embedded figures and reproduce the saved row counts, accuracy and MCC. The retrained-head summary comes from cell 29.", ""])
    text.extend(f"- {reason}." for reason in evidence["limitations"])
    text.extend(["", "The current trainer uses validation-fold MCC for epoch selection and then reports that fold. The new study separates inner selection and fresh refit from outer evaluation.",
                 "Confidence intervals on saved OOF predictions would condition on the trained models and would not remove this selection bias. "
                 "[Cross-validation uncertainty](https://arxiv.org/abs/2104.00673)", "",
                 "The Task 1 composite is 0.5 × micro-F1 + 0.25 × MCC + 0.25 × specificity. Published leaderboard results use different cohorts and are contextual only. "
                 "[Challenge paper](https://arxiv.org/html/2506.02976v1)", "",
                 "Awaiting artifacts: paired patient bootstrap, calibration, patient error concentration, metadata slices, and learning-curve analysis.", "",
                 "![Saved matrix counts and class recalls](notebook_confusion.png)", ""])
    (root / "notebook_evidence.md").write_text("\n".join(text), encoding="utf-8")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (name, values) in zip(axes, MATRICES.items(), strict=True):
        cm = np.asarray(values)
        ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for (i, j), value in np.ndenumerate(cm):
            ax.text(j, i, str(value), ha="center", va="center")
        ax.set(title=name, xlabel="Predicted", ylabel="True", xticks=range(4), yticks=range(4),
               xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        ax.tick_params(axis="x", rotation=35)
    fig.suptitle("Notebook evidence: counts annotated; color = recall fraction")
    fig.tight_layout()
    fig.savefig(root / "notebook_confusion.png", dpi=180)
    plt.close(fig)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", default="notebooks/mario_training_dashboard.ipynb")
    parser.add_argument("--out-dir", default="docs/results/convnext_task1")
    args = parser.parse_args()
    build_evidence(args.notebook, args.out_dir)
    print(f"Notebook evidence only: {args.out_dir}")


if __name__ == "__main__":
    main()
