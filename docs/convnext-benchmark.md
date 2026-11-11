# ConvNeXt Task 1 benchmark and improvement scaffold

The [notebook evidence report](results/convnext_task1/notebook_evidence.md) records the available
historical results. ConvNeXt reports aggregate MCC 0.5860, but OTHER recall is 38.5%; 524 OTHER cases
are called STABLE. The compared pipelines used different cohorts and training regimes. Neither
uncertainty nor a statistically supported architecture improvement can be established from those summaries.

This extension adds artifact reporting, two optional heads, and an opt-in inner-selection/refit training protocol. No training, extraction, inference, model construction, forward/backward pass, or optimizer step was executed while implementing it. Model tests are provided for later execution.

## Artifact-only tools

For a reporting-only environment, install:

```bash
pip install -e ".[benchmark]" pytest
```

Alternatively, install `requirements/benchmark.txt` plus pytest and run from the repository root.
Neither option requires encoder or retinal-preprocessing dependencies. Existing main environments
also need the benchmark plotting dependencies; the historical main/docker lock files are unchanged.
Artifact test parity checks require the pinned DF41 submodule from a recursive checkout, but no
weights, model execution or TensorFlow.

Run commands from the repository root. These commands do not import torch or encoders:

```bash
# Regenerate the committed evidence report using saved notebook output only.
python -m scripts.notebook_evidence

# Read existing OOF CSVs, never run inference to fill missing files.
python -m mario_dino.benchmark \
  --run convnext=outputs/convnext_finetune \
  --run retrained_heads=outputs/convnext_finetuned_heads \
  --baseline convnext --splits outputs/splits.json \
  --metadata Task_1/df_task1_train.csv --out-dir outputs/task1_benchmark
```

Each run directory requires `oof_task1.csv` with this header:

```text
case,id_patient,fold,y_true,y_pred,logit_0,logit_1,logit_2,logit_3
```

Cases must be unique, labels integer 0–3, logits finite, and predictions equal logit argmax.
The split file uses the existing `n_folds`/`patients` format. Patient IDs are read as strings and
must agree exactly with the split file. Duplicate cases, inconsistent patient ownership or labels,
unknown patients, and incorrect fold assignments fail. `--baseline` defaults to the first named run.
The `metrics.json` file is not required for generic reporting: metrics are recomputed from OOF.

Run cohorts must match exactly. `--intersection` explicitly restricts every run to common cases, recomputes scores, and records every excluded case ID. Patient/label/fold disagreements are errors even in intersection mode. Partial-fold reports state which folds are present; use the complete study mode for paper comparisons.

Optional metadata must contain one row per `case`, cover all evaluated cases, and agree with OOF labels and patient identities. Reports include eye and B-scan groups and fixed positive time-gap bins `(0,30]`, `(30,60]`, `(60,90]`, `>90` when those fields exist. These are descriptive slices, not additional hypothesis tests.

Outputs include `report.md`, strict `report.json`, aggregate and per-class CSVs, paired differences, patient error tables, calibration plots, confusion matrices, and available history curves. Input artifacts and split/metadata files receive SHA-256 hashes. Patient identifiers remain in local reports, as they already do in OOF artifacts.

### Metrics and uncertainty

- Primary endpoint: multiclass MCC. Secondary endpoints: micro-F1 (equal to accuracy), macro-F1, balanced accuracy, specificity, class precision/recall/F1, and the Task 1 composite `0.5 F1 + 0.25 MCC + 0.25 specificity`.
- Specificity matches the pinned DF41 epsilon and active-class convention. Macro-F1 fixes the label set to four classes and uses zero for undefined class F1; balanced accuracy averages classes present in truth. Per-class undefined results are null with reasons. Degenerate MCC follows sklearn's zero convention and is annotated.
- Softmax of saved logits supplies unweighted NLL, multiclass Brier score (sum across classes), 15 equal-width confidence-bin ECE, and one-versus-rest AUROC/AUPRC. These receive point estimates only; AUC quantities without both positive and negative examples are null with reasons. No calibration or thresholds are fitted on evaluated predictions.
- The default 2,000 bootstrap replicates sample patients with replacement separately within each outer fold. Every sampled patient's scans stay together, including multiplicity. The exact same draws apply to all compared runs. Reports give percentile 95% intervals for the six aggregate hard-label metrics and paired MCC differences.
- These intervals condition on saved predictions. They do not include retraining/model-selection uncertainty or correct optimism in historical validation-selected OOF. Comparisons and slices are descriptive; no multiplicity-adjusted significance or automatic winner is claimed. `--seed` controls bootstrap reproducibility; `--bootstrap-replicates` can lower the count for artifact testing.
- Per-patient error concentration and equal-patient-weight metric sensitivity expose results dominated by patients with many B-scans. The displayed majority-class accuracy is evaluated-cohort prevalence, not a fitted baseline.

Training-curve diagnostics compare recorded training CE and validation CE when available. Different weighting/composition limits direct interpretation; auxiliary forecasting loss is not equated with classification CE. Refit has no validation curve by design. Missing histories and metadata are reported explicitly.

## Prepare the locked future study

```bash
# Reads CSVs and existing outer splits; writes definitions, prints the matrix.
# Does not launch any of the printed model commands.
python -m mario_dino.study --config configs/task1_convnext_study.yaml \
  --out-dir outputs/convnext_study
```

Preparation requires the real Task 1 CSVs and existing four-fold patient split file. It hashes source
CSV bytes and outer splits, records the cohort, saves deterministic inner partitions, and generates
12 configs: seeds 0/1/2 × legacy, delta_mlp, other_gate, legacy_no_forecast. Each config specifies four
outer folds, giving 48 future fold experiments, each with selection and refit phases. No jobs are launched.

Set data paths, device and training settings in the preparation config before generating the study.
Invoke the exact experiment commands saved in `study.json`; changing generated configs or adding CLI
overrides changes the config hash and prevents strict study aggregation. Prepare a separate study
directory for changed definitions. Identical preparation can be repeated; different existing definitions
are never overwritten. Generated paths are absolute, while inherited relative data paths still resolve
from the repository root.

Image readability is checked when training is later invoked. Cases are never silently removed from
a prepared study: repair unreadable images, or explicitly curate the source CSV before preparing a
new cohort. CSV-byte changes, even if labels are unchanged, invalidate the saved cohort manifest.

The preparation directory contains:

| Artifact | Purpose |
| --- | --- |
| `cohort.json` | Source CSV hashes, outer-split hash, case/patient/label membership and class support |
| `inner_splits.json` | Deterministic fit/selection/test patient groups for each outer fold |
| `study.json` | Seeds, variants, config hashes, run paths and future command arguments |
| `configs/seed{seed}_{variant}.yaml` | Resolved, Task 1-only experiment configs |

For each outer fold, 20% of the remaining patients (rounded up, with at least one patient left for fitting) form the inner selection set. Hash-based patient ordering with split seed 0 makes this partition independent of training seed and architecture. Existing outer assignments are preserved, and fit/selection/test membership is validated for disjointness and complete cohort coverage.

The default legacy trainer is unchanged. Configs with `research.protocol: grouped_holdout` use the
new path and require both saved manifest paths. Use the generated configs to keep the study definitions
and per-run provenance aligned:

1. Load fresh pretrained ConvNeXt and fresh heads; fit on inner-fit patients, using inner-selection MCC for early stopping.
2. Discard that model. Reload the same pretrained initialization, create fresh heads and optimizer, and train on all outer-training patients for exactly the selected epoch count.
3. Evaluate the outer test fold once, using the existing identity/paired-horizontal-flip score averaging. Outer scores never enter fitting or epoch selection.

Class weights and median time gap are recomputed separately from the applicable fitting partition. Retinal preprocessing is deterministic per image. Pretrained-encoder, input-transform, cohort, split, config and saved-checkpoint provenance are recorded. Fine-tuned feature caches and warm-start heads are prohibited in this protocol; historical checkpoints selected on the outer fold cannot be reused for inner selection. Step-capped smoke runs are also prohibited in the paper protocol.

Generated configs can later be passed to `python -m mario_dino.train_finetune --config <generated-config>`. **That command runs models and was not executed here.** `fold{k}/selection/` stores the inner-selected checkpoint and history; `fold{k}/refit/` stores the refit history and checkpoint reference; `fold{k}/task1.pt` remains the self-contained final inference checkpoint. Each phase records its fitting patients, class support, time scale, parameter count and checkpoint hash. Nonempty run directories are refused; optimizer resume is not implemented.

When all runs have finished elsewhere:

```bash
python -m mario_dino.benchmark --study outputs/convnext_study/study.json \
  --metadata Task_1/df_task1_train.csv --out-dir outputs/convnext_study_report
```

Study reporting requires all four variants and all three seeds, with matching case sets and all four outer folds. Each seed receives paired comparisons against its own legacy baseline. `seed_summary.csv` and the top-level report show means and sample SD across seeds, without pooling logits. Conditional intervals remain in the per-seed reports; three seeds do not establish full training-population uncertainty. Choose hyperparameters using inner selection only; further tuning after inspecting outer results requires a new, explicitly exploratory study or fresh independent evaluation.

Study aggregation verifies the locked cohort's cases/patients/labels, cohort-manifest and inner-split
hashes, outer-split file hash, each run's protocol/seed/config provenance, and identical pretrained
encoder/preprocessing identities across arms. Source CSV bytes are checked when training starts;
reporting checks their recorded manifest rather than reopening the source CSVs. Aggregation refuses
changed configs, random-source encoders and incompatible representations. Generic `--run` mode remains
available for explicitly exploratory historical comparisons with less complete provenance.

The study report requires `study.json`, its sibling cohort/inner-split manifests, every run's OOF and
`provenance.json`, and the outer-split file. Histories are optional. By default it reads the outer-split
path from the first generated config; use `--splits PATH` to supply a relocated file with the same hash.
Partial studies and `--intersection` are not supported in study mode.

## Experimental fusion interfaces

`model.fusion` defaults to `legacy`. Old checkpoint hyperparameters and state-dict keys are preserved. `model.fusion_hidden` defaults to 256 for experimental heads.

| Fusion | Behavior | Hypothesis |
|---|---|---|
| `legacy` | Existing LayerNorm/dropout/linear on `[a,b,b−a]` | Reference |
| `delta_mlp` | `[a,b,b−a,abs(b−a),a*b]` → LayerNorm → Linear(256) → GELU → dropout → Linear(4) | Test whether nonlinear pair interactions improve comparison |
| `other_gate` | Same trunk, with OTHER sigmoid and conditional 3-class softmax, composed as stable four-class log probabilities | Test whether a separate OTHER decision reduces confusion with STABLE |

Weighted cross-entropy and the existing forecasting auxiliary objective remain unchanged. The no-forecast arm changes only `train.lambda_pred` to zero. All other architecture arms retain the base forecasting-loss weight and training settings.

The OTHER gate is not a quality detector: OTHER includes uninterpretable scans and mixed changes. There is no assumed ordinal position for this class, no quality supervision, and no new auxiliary gate loss. New heads require paired inputs and explicitly reject Task 2. Standalone architecture presets disable Task 2 training CSVs; paper runs should use generated study configs.

Nonlegacy fusion settings are persisted in checkpoint head hyperparameters. Legacy checkpoints retain
their existing payload; absent fusion fields reconstruct the legacy head. Full inference reconstructs
from saved encoder/head states without source weights and checks compatible heads across folds.
The existing score-averaging contract is retained: for the gate, log probabilities are averaged and
then normalized when probabilities are needed (a geometric probability combination). No spatial
pooling or encoder architecture changes were introduced.

## Validation boundary

Scaffolding verification on November 12, 2026 completed: **31 artifact-only tests passed**; repository Ruff checks,
Python syntax, all YAML configurations, notebook JSON structure, CLI help and diff whitespace
checks passed. The single sklearn warning comes from the intentional single-class specificity
parity fixture. No model execution tests or model runs were performed.

Artifact-only validation:

```bash
python -m pytest --noconftest tests/test_benchmark_artifacts.py -q -p no:cacheprovider
```

`make benchmark-check` runs the same artifact-only selection. `--noconftest` avoids the shared
image/synthetic setup, allowing the reporting-only environment. Disposable artifact fixtures are
written under ignored `.cache/artifact-tests/` directories. These checks cover independent metric
reconstruction, challenge specificity parity, malformed OOF, case alignment, stable probability
metrics, missing classes, cluster multiplicity, paired bootstrap reproducibility, report export,
three-seed summaries, deterministic splits, cohort hashes, configuration guards and evidence
regeneration. They do not construct models.

`tests/test_fusion_models.py` contains model construction/execution tests for shapes, gradients, gate normalization at extreme logits, legacy compatibility, and self-contained checkpoint reconstruction. They carry the `model_execution` marker and have **not** been executed. The selection/refit training path and real checkpoints still require later CI/cloud model validation. `make test`, `make smoke`, training, extraction and inference all execute models and are outside scaffolding validation.

The marker applies only to the new fusion test file. Other model tests are not all marked, so
`pytest -m "not model_execution"` is not a substitute for the explicit artifact-only command.

The committed evidence report preserves saved notebook outputs; the notebooks were not rerun or changed. DINOv2 versus ConvNeXt is a comparison of full pipelines with different cohorts and training/preprocessing regimes, not an isolated encoder ablation. Published challenge scores use different evaluation cohorts and provide context only. No improvement is claimed for either new head until matched future evaluations support it.

Sources: [official labels](https://youvenz.github.io/MARIO_challenge.github.io/docs/compose/task1/), [challenge metrics and composite](https://arxiv.org/html/2506.02976v1), [cross-validation uncertainty](https://arxiv.org/abs/2104.00673).
