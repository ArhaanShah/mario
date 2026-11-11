# MARIO Tasks 1–2: encoders + GRU forecaster — runbook

For Task 1 artifact-only benchmarking, the saved ConvNeXt result audit, experimental fusion heads,
and preparation of the future selection/refit study, see [the benchmark guide](docs/convnext-benchmark.md).
Its reporting and preparation commands do not execute models. The model commands below do.

What this repo does, in one line per task:

- **Task 1** (4 classes): `head(f_t, f_{t+Δt})` on encoder features of both visits.
- **Task 2** (3 classes): `head(f_t, f̂_{t+Δt})[:, :3]`, with `f̂ = forecaster(f_t, Δt₀)`, a GRU forecast in
  feature space. This replaces DF41's pixel-space PPMAE.

The default DINOv2 frozen workflow uses **step 3** for one encoder pass per unique image/view;
head training and ablations can then run on CPU. ConvNeXt and VisionFM OCT support the same frozen
workflow and a separate full Task 1 fine-tuning path whose encoder compute grows with every epoch/fold.
For that path, go to [alternative encoders](#alternative-task-1-encoders-convnext-and-visionfm-oct);
it does not require a feature or token cache. GPU is recommended for encoder work; CPU is supported.

Run every command from the repo root. Paths below are the defaults in `configs/base.yaml`.

The design rationale (GPU budget, DF41 facts, milestones) is in
[docs/implementation-plan.md](docs/implementation-plan.md), retained as a historical design with a
scope note. The current adapter/config/checkpoint contract is in [docs/encoders.md](docs/encoders.md).
Other departures appear under [Deviations from the plan](#deviations-from-the-plan-and-why) below.

## Models and where they come from

Every model the default pipeline needs is either public or not needed:

| Model | Used for | Source | Access |
| --- | --- | --- | --- |
| DINOv2 ViT-B/14 with 4 registers (`vit_base_patch14_reg4_dinov2.lvd142m`) | frozen encoder | Hugging Face [`timm/vit_base_patch14_reg4_dinov2.lvd142m`](https://huggingface.co/timm/vit_base_patch14_reg4_dinov2.lvd142m) | public, not gated, Apache-2.0; `scripts/fetch_weights.py` downloads it once |
| DINOv2 ViT-S/14 with 4 registers (`vit_small_patch14_reg4_dinov2.lvd142m`) | smaller-budget switch | Hugging Face [`timm/vit_small_patch14_reg4_dinov2.lvd142m`](https://huggingface.co/timm/vit_small_patch14_reg4_dinov2.lvd142m) | public, not gated, Apache-2.0 |
| ConvNeXt Tiny (`convnext_tiny.in12k_ft_in1k`) | alternative frozen encoder or full Task 1 fine-tuning | pinned timm 1.0.30 | prepare headless local weights with `fetch_weights.py --config configs/task1_convnext.yaml` |
| VisionFM OCT ViT-B/16 (`vit_base_patch16`) | alternative frozen encoder or full Task 1 fine-tuning | official OCT download linked in the alternative-encoder section | manual download; explicit wrapper selection; vendored backbone under CC BY-NC 4.0 |
| OCTIP retina localizers (`FPN_efficientnetb6_384x384.hdf5`, `FPN_efficientnetb7_320x320.hdf5`) | `--mode octip` only | by email request to the OCTIP author ([leto-atreides-2/octip](https://github.com/leto-atreides-2/octip)); DF41 does not ship them | **not needed**: `--mode classic` (the default) replaces them, see step 2 |
| EfficientNet-B6/B7 ImageNet weights | `segmentation_models` asks for them when OCTIP builds its FPNs | GitHub releases | **not needed**: OCTIP's .hdf5 overwrites them, so `octip_cache` builds the FPNs with `encoder_weights=None` and stays offline |

DF41's own checkpoints (ResNet50 fusion models, PPMAE) are not released and not used. The MARIO images and
CSVs come from the challenge organisers.

For DINOv2/ConvNeXt, if the Hugging Face Hub is blocked, run `fetch_weights.py` anywhere with access and copy
`weights/*.pth` and `weights/*.pth.json`, or point `HF_ENDPOINT` at a mirror.

---

## 0. Setup

```bash
git clone --recursive https://github.com/ArhaanShah/mario.git && cd mario
git submodule update --init          # if cloned without --recursive; pins DF41 at 4d4a11b
```

**Main env** (torch/timm; never imports TensorFlow), Python 3.10:

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # pick the CUDA build for your driver
pip install -r requirements/main.txt
make test                            # CPU only; no GPU, network, MARIO data or OCTIP weights needed
```

**OCTIP env** (optional; only for `--mode octip`, which needs the on-request OCTIP weights), Python 3.10,
CPU TensorFlow:

```bash
python3.10 -m venv .venv-octip && source .venv-octip/bin/activate
pip install -r requirements/octip.lock.txt
```

The two envs exchange data only through `cache/<mode>/` (PNG files). `requirements/*.lock.txt` are exact
pins resolved for Python 3.10 / Linux x86_64; `requirements/*.txt` are the loose specs they came from.

## 1. Data, weights, splits

Lay out the data like this, or change `data.*` in `configs/base.yaml`:

```
data/images/<B-scan>.png                         # names as they appear in the CSVs
data/csv/df_task1_train_challenge.csv            # data.task1_train_csvs
data/csv/df_task2_train_challenge.csv            # data.task2_train_csvs
data/csv/df_task{1,2}_val_challenge.csv          # data.extra_csvs (encoded too, so inference reuses them)
data/csv/df_task{1,2}_test_challenge.csv         # inference.task{1,2}_csv in configs/task{1,2}.yaml
```

```bash
# once, on a machine with network access (writes weights/vit_base_patch14_reg4_dinov2.lvd142m.pth + .json)
python scripts/fetch_weights.py --config configs/base.yaml

# Task 2 vs Task 1 label agreement on the training CSVs (DF41 val: 2140/3822 = 56%).
# Set train.task2_aux_from_task1_pairs: true only if agreement here is high.
python scripts/check_label_alignment.py --config configs/base.yaml

# 4 patient-grouped folds shared by both tasks -> outputs/splits.json
python -m mario_dino.data.splits --config configs/base.yaml
```

Smaller budget: set `encoder.name: vit_small_patch14_reg4_dinov2.lvd142m` and
`encoder.weights: weights/vit_small_patch14_reg4_dinov2.lvd142m.pth`, then re-run `fetch_weights.py`.

## 2. OCTIP preprocessing cache (CPU)

Default, no weights needed (main env):

```bash
python -m mario_dino.octip_cache --config configs/base.yaml        # preprocess.mode: classic
```

The base config also sets `data.include_image_dirs: [Task_2/train]`, so every supported image file in that
folder is cached even if it is absent from the training CSVs. The cache key is the image filename, and
already-cached files are skipped on subsequent runs.

`classic` mode runs DF41's unmodified OCTIP `PreProcessor`, which flattens the retina, crops it to 200 rows,
drops bad A-scans and normalises intensity. Its retina masks come from two learning-free localizers in
`mario_dino/retina.py` instead of OCTIP's two FPN networks. The PreProcessor only uses each mask's top
retina boundary per A-scan. It keeps an A-scan only when the two masks agree within 5 px, as it does with
the FPN pair:

- **threshold**: blur, then an Otsu threshold capped at background + 30% of the range, then largest
  tissue components;
- **gradient**: blur, then a vertical Sobel, then the first strong dark-to-bright edge.

On synthetic B-scans with a known retina top (`tests/test_retina.py`) both land within 1–3 px. This
says nothing yet about real Spectralis B-scans. Before step 3, spot-check a few `cache/classic/*.png`
against the originals, and check `cache/classic/failures.csv` (images that fell back to raw).

If you obtain the OCTIP weights, put `FPN_efficientnetb6_384x384.hdf5` and `FPN_efficientnetb7_320x320.hdf5`
in `weights/octip/`, set `preprocess.mode: octip` in `configs/base.yaml`, and run in the octip env:

```bash
source .venv-octip/bin/activate
python -m mario_dino.octip_cache --config configs/base.yaml --mode octip
```

To skip preprocessing entirely, set `preprocess.mode: raw` (originals are copied).

The rest of the pipeline reads `preprocess.mode` from the config, so keep it the same for steps 3–6. The
feature store records the mode and refuses to mix modes. The step is idempotent: it skips images already
cached. Images where the PreProcessor fails, or keeps under 25% of the A-scans, are cached raw, as DF41
does, and listed in `cache/<mode>/failures.csv`.

## 3. Frozen feature extraction (GPU recommended)

```bash
source .venv/bin/activate
python -m mario_dino.features --config configs/base.yaml --device cuda
```

- Encodes each unique image once per view (identity, horizontal flip), batch 64, bf16 autocast (fp16 if bf16
  is unsupported), into `features/store/` (fp16 memmap, 3 KB per image-view for ViT-B; ~52 MB for 8,450 images).
- **Resumable**: if interrupted, run the same command again. Finished chunks (`features.chunk_size`, 1,024
  images) are not recomputed.
- The store refuses to mix features: its `meta.json` records the encoder, weights sha256, input size and
  preprocessing mode. To change any of these, use a new `features.store_dir`.

## 4. Train forecaster + head (CPU), with ablations

```bash
# main model: stage A (Task 1) then stage B (Task 2) for each of the 4 folds -> outputs/heads/
python -m mario_dino.train_heads --config configs/base.yaml --device cpu

# ablations, from the same feature cache
python -m mario_dino.train_heads --config configs/base.yaml --device cpu \
    --set model.forecaster=identity --set train.out_dir=outputs/abl_identity
python -m mario_dino.train_heads --config configs/base.yaml --device cpu \
    --set model.head_input=ft_only --set train.out_dir=outputs/abl_ft_only
python -m mario_dino.train_heads --config configs/base.yaml --device cpu \
    --set train.lambda_pred=0 --set train.out_dir=outputs/abl_nopred

python scripts/compare_runs.py outputs/heads outputs/abl_identity outputs/abl_ft_only outputs/abl_nopred
```

Each run directory holds `fold{k}/task1.pt`, `fold{k}/task2.pt`, `fold{k}/history.json`,
`oof_task{1,2}.csv` (out-of-fold logits), `metrics.json` (per-fold and OOF micro-F1, MCC, specificity,
plus QWK for Task 2, with DF41's metric keys), `config.resolved.yaml` and `provenance.json`.
Checkpoints are selected on fold-validation MCC (`train.select_metric`).

Useful flags: `--folds 0 1`, `--set train.delta_t0=<days>` (Task 2 horizon; default = training-fold median
`delta_t`), `--set train.task2_aux_from_task1_pairs=true`.

## 5. Optional: DINOv2 partial fine-tune (GPU recommended)

Run this after reviewing step 4's scores. This path is DINOv2-only and trains from cached prefix tokens.
For ConvNeXt/VisionFM, use `train_finetune` with image pairs as described below.

```bash
python -m mario_dino.train_partial --config configs/base.yaml --device cuda
```

- Caches the tokens after block `11−N` once (N = `partial.n_blocks`, default 2) into
  `features/tokens/` (fp16, ~0.8 MB per image-view, ~6.8 GB for 8,450 images and one view; resumable).
- Then trains blocks `12−N … 11`, the final norm and the heads with stage A's loss; lr 1e-5 for the blocks
  and 1e-4 for the heads. Heads warm-start from `outputs/heads/fold{k}/task1.pt`. Output: `outputs/partial/`.
- This trains **Task 1 only**. Use it for inference with
  `python -m mario_dino.infer --config configs/task1.yaml --task 1 --run-dir outputs/partial --method 2`.

## 6. Inference and Docker

Without Docker:

```bash
python -m mario_dino.infer --config configs/task1.yaml --task 1    # -> outputs/infer_task1/
python -m mario_dino.infer --config configs/task2.yaml --task 2    # -> outputs/infer_task2/
```

For frozen head runs, inference caches its CSV's images (OCTIP step, skipping cached files), encodes any image not yet in the
feature store (once per view), averages logits over the 4 folds and both views, and writes
`results_task{t}_team_dino_method_{m}.csv` with DF41's columns: `y_true` (only when the CSV has labels),
`y_pred`, `cases`. With labels it also writes `scores_task{t}_method_{m}.json`. Task 2 images that were
already encoded for Task 1 or training are reused. Set the team name with `inference.team`.

DINO partial runs share the frozen prefix and apply each fold's saved suffix. Full fine-tuned runs
load each fold's encoder/heads sequentially and average that fold's own image/view logits; they do not
use frozen feature stores or require the original pretrained encoder file. See the
[checkpoint reference](docs/encoders.md#checkpoint-and-cache-compatibility).

Docker (CPU; both envs in one image; weights and trained heads copied in; no network at runtime):

```bash
python scripts/fetch_weights.py --config configs/base.yaml      # weights/ must exist before building
docker build -f docker/Dockerfile -t mario-dino --build-arg GIT_COMMIT=$(git rev-parse HEAD) .
# mounting cache/ and features/ lets the Task 2 run reuse what the Task 1 run (or step 3) computed
# (add -v $PWD/weights/octip:/app/weights/octip only for preprocess.mode: octip)
MOUNTS="-v $PWD/data:/app/data -v $PWD/cache:/app/cache \
        -v $PWD/features:/app/features -v $PWD/outputs/infer:/app/outputs/infer"
docker run --rm --network none $MOUNTS mario-dino --config configs/task1.yaml --task 1 --out-dir outputs/infer
docker run --rm --network none $MOUNTS mario-dino --config configs/task2.yaml --task 2 --out-dir outputs/infer

# self-test inside the image on synthetic data (random-init encoder, CPU)
docker run --rm --network none --entrypoint python mario-dino scripts/smoke.py --skip-partial
```

---

## Task 1 reports and study preparation

Use the reporting-only environment for artifact analysis and study definitions:

```bash
pip install -e ".[benchmark]" pytest
make benchmark-check
python -m scripts.notebook_evidence
```

Without `make`, the artifact check is
`python -m pytest --noconftest tests/test_benchmark_artifacts.py -q -p no:cacheprovider`.
These commands execute no models. The committed
[evidence report](docs/results/convnext_task1/notebook_evidence.md) records the saved ConvNeXt Task 1
scores and OTHER-class failure; it cannot supply patient uncertainty or calibration without raw OOF.

After configuring the real Task 1 CSVs and existing outer splits, prepare the study with
`python -m mario_dino.study --config configs/task1_convnext_study.yaml`.
Preparation saves the cohort and inner patient partitions, writes 12 configs for three seeds and
four variants, and prints the experiment matrix. It does not launch those experiments.
Generated runs use inner MCC for epoch selection, then fresh refit and one outer-fold evaluation.
Passing a generated config to `train_finetune` later **does execute models**.

For existing artifacts, use `python -m mario_dino.benchmark --run NAME=RUN_DIR --splits SPLIT_FILE --out-dir REPORT_DIR`.
For the complete prepared study, use
`python -m mario_dino.benchmark --study outputs/convnext_study/study.json --out-dir outputs/convnext_study_report`.
Both reporting commands read artifacts only. The
[benchmark guide](docs/convnext-benchmark.md) specifies matching cohorts, optional metadata,
patient bootstrap uncertainty, strict study provenance and output formats. Raw real-run artifacts
are not included here, so these full reports await those files.

## Tests and smoke run

```bash
make test     # pytest on CPU (tests needing OCTIP weights + TensorFlow skip with a reason)
make smoke    # whole pipeline on synthetic data: data -> weights -> splits -> cache -> features ->
              # train (+ identity ablation) -> partial fine-tune -> inference for both tasks
make lint
```

On Windows without `make`: `python -m pytest -q` and `python scripts/smoke.py`.
`make test`, `make smoke` and their Python equivalents execute CPU models. `make lint` does not.
`make smoke` remains the original DINO pipeline; it does not exercise ConvNeXt/VisionFM full
fine-tuning. Model tests are in `tests/test_alternative_encoders.py` and `tests/test_fusion_models.py`,
with an old-schema DINO regression in `tests/test_infer.py`. These model checks were not executed
for the scaffolding change; run them later on cloud/CI. The `model_execution` marker covers the new
fusion test file only; filtering on that marker does not make the whole suite model-free.
`OCTIP_WEIGHTS_DIR=/path/to/octip_models make test` runs the real OCTIP test when the octip packages are
installed in the same env.

## Status: what was and was not run

The November 12, 2026 benchmark/scaffolding validation completed:

- **31 artifact-only tests passed**, including metric reconstruction, challenge specificity parity,
  paired patient sampling, report generation, three-seed summaries, cohort/provenance guards and split isolation.
- Repository Ruff checks, Python syntax, configuration loading, notebook JSON structure, CLI help
  and diff whitespace checks passed.
- The notebook-evidence Markdown, JSON, CSV and figure were generated from saved outputs. Neither
  notebook was executed or modified. No encoder or model was constructed for these checks.

Saved notebook outputs record earlier real-data runs: ConvNeXt Task 1 MCC 0.5860 and OTHER recall
38.5%; retraining heads reports MCC 0.5903. The audit is in
[the evidence report](docs/results/convnext_task1/notebook_evidence.md). These are historical,
validation-selected results from different model/cohort settings, not new independent benchmark
measurements or evidence that the experimental heads improve scores. Raw OOF files, histories,
checkpoints and the real patient split file are absent from this checkout.

Historical DINO baseline results recorded before the alternative-encoder extension
(Windows, Python 3.12, torch 2.14.1+cpu, timm 1.0.30):

- `pytest`: 60 passed, 1 skipped (`test_octip_mode_real_weights`: no OCTIP weights or TensorFlow).
- `scripts/smoke.py`: the full pipeline on synthetic data with `classic` preprocessing, including partial
  fine-tune and inference for both tasks (0 images fell back to raw).
- `ruff check` (target py310).
- `uv pip compile` of `requirements/{main,octip,docker}.txt` for Python 3.10 / manylinux x86_64. All three
  resolve (`requirements/*.lock.txt`).
- `check_label_alignment.py` on DF41's val CSVs reproduces 2140/3822 (56%).

The 60-test count is historical, not a result for the current full suite. The initial ConvNeXt/VisionFM
implementation received static checks, including unchanged DINO helpers/default transforms and pinned
VisionFM positional interpolation. The later 31 artifact tests validate reporting and split logic;
they do not validate encoder execution, fusion gradients, checkpoint reconstruction or selection/refit training.

Not performed for the scaffolding change; still requiring later validation:

- Model construction/loading, forwards/backwards, optimizer steps, training, feature extraction,
  inference and the model execution tests. No new scores were measured for the experimental heads.
- Pretrained downloads and real-binary compatibility checks, including the released VisionFM OCT layout.
- Real OCTIP execution with request-only weights/TensorFlow. Historical local `classic` checks used
  synthetic scans; the saved notebook separately records preprocessing performed elsewhere on real data.
- GPU memory, throughput and execution of the selection/refit protocol on real MARIO data.
- `docker build` (downloads ~2 GB of TensorFlow + torch wheels) and the in-image run.
- A Python 3.10 interpreter: code targets 3.10 (ruff `target-version = "py310"`) but was executed on 3.12.

## Deviations from the plan, and why

Relative to [docs/implementation-plan.md](docs/implementation-plan.md):

- **OCTIP env opencv pin**: DF41 pins `opencv-python==4.4.0.42`, which has no Python 3.10 wheel. Uses
  `opencv-python-headless==4.10.0.84`, the newest release that still supports numpy 1.x, as TF 2.10 needs.
- **DF41 scoring import**: DF41's `utils/scoring.py` imports seaborn, matplotlib and torchvision for
  plotting helpers. `mario_dino/scoring.py` stubs those three while importing the file and then restores
  `sys.modules`. `specificity` itself is DF41's code, imported from the pinned clone and not copied.
- **Feature store reuse**: `inference.store_dir` defaults to `features.store_dir`, so val/test images encoded
  in step 3 (via `data.extra_csvs`) are not encoded again at inference. The store appends new images.
- **Task 2 auxiliary loss** (`task2_aux_from_task1_pairs`, off by default): forecasts each Task 1 pair at its
  own `delta_t`, not at Δt₀.
- **`classic` preprocessing mode (new, default)**: OCTIP's localizer weights are request-only, so two
  learning-free localizers stand in for them in front of DF41's own PreProcessor (step 2). `octip` mode is
  unchanged apart from building its FPNs without the ImageNet download.
- **Synthetic B-scans**: the plan asked for horizontal bands plus noise. Those bands dropped back to vitreous
  darkness between layers, which no real B-scan does and which broke retina localization. They now have
  continuous tissue below a curved retinal surface, with layered bands inside, an RPE-like band and a fading
  choroid. `synthetic_bscan` also returns the true retina top that the localizer tests check against.
- **Partial fine-tune inference**: fold checkpoints carry their own encoder suffix, so `infer.py` runs the
  shared prefix once per image and view and each fold's suffix on top.
- **Alternative encoders**: ConvNeXt/VisionFM adapters, shared transform specs, image-based Task 1
  full fine-tuning and self-contained fold checkpoints extend the original frozen DINO scope.
  Existing DINO defaults and identity payloads are retained. See [docs/encoders.md](docs/encoders.md).
- **Task 1 research scaffold**: artifact-only evaluation, nonlinear fusion and OTHER-gate heads,
  and inner-selection/fresh-refit study definitions extend the encoder work. Model runs remain
  deferred; [the benchmark guide](docs/convnext-benchmark.md) defines the protocol and evidence limits.

## Open questions (resolve before step 3)

- [ ] Unit of `delta_t`, and the Task 2 prediction horizon. Δt₀ defaults to the training median until known
      (`train.delta_t0`).
- [x] Task 1 class names for indices 0–3: reduced, stable, worsened, other, confirmed against the
      [official Task 1 definitions](https://youvenz.github.io/MARIO_challenge.github.io/docs/compose/task1/).
      OTHER includes uninterpretable scans and mixed changes. Code uses indices only.
- [ ] Task 2 label definition, given only 56% agreement with the Task 1 label of the same starting image.
- [ ] Optional: request the two OCTIP weight files from the OCTIP author (gwenole.quellec@inserm.fr, per
      [leto-atreides-2/octip](https://github.com/leto-atreides-2/octip)) to compare `octip` against `classic`.
      Not needed to run anything.
- [ ] Whether challenge rules permit the selected pretrained encoder and its source/license terms.

## Code map

| Path | What |
| --- | --- |
| `mario_dino/config.py` | YAML → dataclasses; `base:` inheritance, `--set` overrides, unknown keys rejected |
| `mario_dino/data/schema.py` | DF41 column sets, CSV/dtype/image validation |
| `mario_dino/data/splits.py` | GroupKFold(4) on `id_patient` over Task 1 ∪ Task 2 → `splits.json` |
| `mario_dino/octip_cache.py` | CLI: `classic`, batched OCTIP (`octip`) or copy (`raw`) → `cache/<mode>/` |
| `mario_dino/retina.py` | weight-free retina localizers + DF41's OCTIP PreProcessor loaded without TensorFlow |
| `mario_dino/data/images.py` | versioned shared input specs, legacy/new transforms, cached images and paired-image dataset, views 0/1 |
| `mario_dino/models/encoder.py` | explicit offline factory/local loading; preserved DINO pooling and prefix/suffix helpers |
| `mario_dino/models/convnext.py` | differentiable headless timm ConvNeXt adapter |
| `mario_dino/models/visionfm.py` | differentiable VisionFM CLS adapter and strict external checkpoint import |
| `mario_dino/models/vendor/` | isolated pinned VisionFM backbone, attribution and upstream license |
| `mario_dino/models/finetuned.py` | resolved encoder specifications and self-contained checkpoint reconstruction |
| `mario_dino/store.py` | append-only resumable memmap store (features and token caches) |
| `mario_dino/features.py` | CLI: unique images → feature store |
| `mario_dino/models/heads.py` | forecaster, legacy/experimental Task 1 fusion heads, ablations, `MarioHeads` |
| `mario_dino/losses.py` | class weights, forecasting MSE (detached target), stage A/B losses |
| `mario_dino/train_heads.py` | CLI: CPU training per fold, OOF predictions, metrics |
| `mario_dino/train_partial.py` | CLI: DINO-only partial fine-tune on cached prefix tokens |
| `mario_dino/train_finetune.py` | CLI: ConvNeXt/VisionFM full Task 1 fine-tuning from image pairs, OOF/metrics/checkpoints |
| `mario_dino/train_research.py` | opt-in Task 1 inner selection, fresh refit and one outer evaluation |
| `mario_dino/study.py` | model-free cohort/split validation, study manifests and generated configs |
| `mario_dino/benchmark.py` | model-free OOF metrics, patient bootstrap, failure reports and seed summaries |
| `mario_dino/models/full.py` | `MarioModelDINO` inference and differentiable encoder/head `Task1Net` |
| `mario_dino/infer.py` | CLI: frozen/partial/full-fold ensemble → results CSV (DF41 format) |
| `mario_dino/scoring.py` | DF41 `specificity` + challenge metrics |
| `mario_dino/synthetic.py` | synthetic dataset and planted feature store |
| `scripts/` | data/weight helpers, alignment/run comparison, notebook evidence regeneration, smoke |

## Alternative Task 1 encoders: ConvNeXt and VisionFM OCT

Use `configs/task1_convnext.yaml` or `configs/task1_visionfm.yaml`. Both support frozen
feature extraction followed by `train_heads`, or full encoder/head training with
`train_finetune`. Fine-tuning is Task 1 only. `train_partial` remains DINOv2-only.
Existing DINO configurations, pooling, feature identities and checkpoint defaults remain compatible.
These original presets retain legacy validation-fold checkpoint selection. For separate inner selection
and outer evaluation, use the [prepared research study](docs/convnext-benchmark.md#prepare-the-locked-future-study).

Run commands below from the repository root on the cloud instance. Use the existing
main environment (`pip install -r requirements/main.txt` and `pip install -e .`),
with a CUDA-enabled PyTorch build installed for the instance. Keep `timm==1.0.30`.
Wrapped VisionFM files containing upstream `argparse.Namespace` metadata need
PyTorch >=2.4; raw backbone tensor dictionaries retain PyTorch >=2.1 support.
No upstream VisionFM environment, einops, FlashAttention or extra encoder dependencies
are required. The existing retinal cache setup still applies; `preprocess.mode=raw`
is available as an explicit ablation.

Prepare pretrained weights explicitly; training and inference never download them:

```bash
python scripts/fetch_weights.py --config configs/task1_convnext.yaml
```

For VisionFM, manually download the official **OCT** encoder from the
[VisionFM OCT release](https://drive.google.com/file/d/1o6E-ine2QLx2pxap-c77u-SU0FjxwypA/view?usp=sharing)
and place it at `weights/VFM_OCT_weights.pth`. The
[upstream weights table](https://github.com/ABILab-CUHK/VisionFM#31-download-the-pretrained-weights-of-visionfm)
is the source of that link. Release bytes have **not** been downloaded or inspected
in this checkout. The preset explicitly assumes a **raw backbone state dict**
(`checkpoint_key: null`). If your file is wrapped, explicitly select `teacher`,
`student`, or `visionfm_state_dict` using `encoder.checkpoint_key`. The loader does
not guess from filenames. Missing wrapper keys, unknown tensors, incomplete backbones
and incompatible shapes fail. Known leading `module.`/`backbone.` prefixes,
pretraining/classification heads and the unused masking embedding are handled;
discarded tensors are logged. Missing/malformed files never become random models.

Optional local conversion saves a canonical raw backbone and SHA-256 sidecar:

```bash
# Example for a teacher wrapper: select the key actually present in your file.
python scripts/fetch_weights.py --config configs/task1_visionfm.yaml \
  --source /path/to/original_oct.pth --set encoder.checkpoint_key=teacher \
  --out weights/VFM_OCT_weights.pth
# The output is RAW: use encoder.checkpoint_key=null thereafter.
```

Create/reuse the existing patient-grouped split file, then fine-tune:

```bash
python -m mario_dino.data.splits --config configs/task1_convnext.yaml
python -m mario_dino.train_finetune --config configs/task1_convnext.yaml --device cuda
python -m mario_dino.infer --config configs/task1_convnext.yaml --task 1 --csv /path/to/task1.csv --device cuda

# Raw OCT weights; add --set encoder.checkpoint_key=teacher for a teacher wrapper.
python -m mario_dino.train_finetune --config configs/task1_visionfm.yaml --device cuda
python -m mario_dino.infer --config configs/task1_visionfm.yaml --task 1 --csv /path/to/task1.csv --device cuda
```

Each fold reloads the same pretrained initialization, creates fresh heads and AdamW
state, and uses training-fold class frequencies and median time gap. Both images
receive the same sampled flip. The existing loss is unchanged: Task 1 CE propagates
through both images; only the forecasting target is detached. Validation averages
both flip views with the existing metric selection and patience semantics
(`train.select_metric: mcc` by default). Training uses CUDA bf16 when supported and
enabled by `features.amp`, otherwise fp32. Images load on demand from the retinal cache.
Lower `finetune.batch_size` and `inference.batch_size` if needed; 8 is a starting
preset, not a measured memory guarantee. `--folds 0` selects a fold;
`finetune.max_steps` optionally caps steps for a later cloud smoke run.

Outputs go to `outputs/convnext_finetune` or `outputs/visionfm_finetune`, including
`fold{k}/task1.pt`, history, out-of-fold predictions, `summary.json`, and `metrics.json`.
Checkpoints use `kind: finetuned`, `format_version: 1`, strict full adapter/head state,
resolved architecture, positional grid, pooling, transform, and source provenance.
Inference reconstructs from the checkpoint after the original weights and sidecar
have been removed. Each fold encodes its own images/views; logits are averaged
across folds/views. One full fold model is loaded at a time. Frozen feature stores
are never used in this path. Fold encoder specifications must agree. The stored
transform overrides current `encoder.*` initialization settings at inference;
retinal `preprocess.mode` must match. Optimizer resume is not implemented.

For a frozen baseline, use the separate feature/head paths in either preset:

```bash
python -m mario_dino.octip_cache --config configs/task1_convnext.yaml
python -m mario_dino.features --config configs/task1_convnext.yaml --device cuda
python -m mario_dino.train_heads --config configs/task1_convnext.yaml
python -m mario_dino.infer --config configs/task1_convnext.yaml --task 1 \
  --run-dir outputs/convnext_heads --csv /path/to/task1.csv --device cuda
# Substitute task1_visionfm.yaml and outputs/visionfm_heads for VisionFM.
```

DINO heads/forecasters cannot be reused with the alternatives, even when dimensions
match. `finetune.init_heads_from` defaults to null independently of the inherited
`partial.init_heads_from`. An optional warm start must be a frozen Task 1 run with
the same fold, encoder, pretrained weights, transform identity, dimension and head
architecture. Keep the default separate feature/run directories.

ConvNeXt uses native average pooling and final normalization (768 standard Tiny
features). VisionFM uses final-normalized CLS (768), a MARIO adapter choice rather
than its upstream multi-block downstream decoders. Its square pretrained positional
parameter is inferred from the weights, retained, and interpolated at runtime using
upstream logic. New presets use `[192, 512]` for approximate B-scan aspect ratio;
VisionFM also supports `[224, 224]`. Dimensions are never silently rounded/cropped.
OCT stays grayscale repeated into three channels. `legacy_dino` retains linear
resize/min-max scaling/ImageNet normalization exactly. `imagenet` uses OpenCV bicubic
uint8 resize then /255 and ImageNet mean/std. `visionfm_oct` uses that resize/scaling
with mean `(0.21091926, 0.21091926, 0.21091919)` and std
`(0.17598894, 0.17598891, 0.17598893)`. Retinal preprocessing and rectangular resize
are MARIO adaptations, not an exact reproduction of upstream experiments.

The minimal vendored backbone is pinned to VisionFM commit
`648ab0895f8e80e2c937f658bd2f1629ea6d27e7`, with upstream CC BY-NC 4.0 license,
attribution and local edits recorded in `mario_dino/models/vendor/`. Ordinary
attention has no CUDA requirement. The unmodified pinned source is an offline
test oracle. Docker's build context includes both new preset head/finetune output
directories. Mount custom run paths or explicitly include them in `.dockerignore`.

Alternative-encoder and fusion model execution remain unvalidated here. The current 31 passing
artifact-only tests are separate from the following model checks, written for later cloud/CI execution:

```bash
# These commands RUN CPU models: execute only when ready on the cloud/CI machine.
python -m pytest tests/test_encoder.py tests/test_partial.py tests/test_store_features.py \
  tests/test_heads_training.py tests/test_infer.py tests/test_images.py tests/test_alternative_encoders.py \
  tests/test_fusion_models.py
```

New tests use offline random fixtures: small ConvNeXt and one-block VisionFM.
Coverage includes actual Task 1 gradients/updates, strict import, positional grids,
square/rectangular upstream parity, transforms/identities, frozen extraction, joint
training, serial fold inference and removal of source weights. These tests have not
been executed here. Real release compatibility and GPU memory/throughput remain unmeasured.
