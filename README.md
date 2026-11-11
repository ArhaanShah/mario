# mario

Code for Tasks 1 and 2 of the MARIO challenge (monitoring of AMD progression on OCT). Choose DINOv2,
ConvNeXt Tiny, or VisionFM OCT features, with a GRU that forecasts next-visit features and a fusion head
shared by both tasks. DINOv2 remains the default. ConvNeXt and VisionFM also support joint Task 1
encoder/head fine-tuning directly from image pairs.

Task 1 also has optional nonlinear fusion and OTHER-gate heads, artifact-only benchmarking,
and a separate patient inner-selection/refit protocol. Start with the
[benchmark guide](docs/convnext-benchmark.md) for that workflow.

- **Task 1** (4 classes, two visits): `head(f_t, f_{t+Δt})` on the encoder features of both B-scans.
- **Task 2** (3 classes, one visit): `head(f_t, f̂_{t+Δt})[:, :3]`, where `f̂ = forecaster(f_t, Δt₀)`. This
  replaces DF41's pixel-space masked-autoencoder forecast with a forecast in feature space.

OCTIP preprocessing, scoring and the output format come from team DF41's code
([pzhangwj/mario_challenge_code](https://github.com/pzhangwj/mario_challenge_code)), included as a git
submodule pinned at `4d4a11b` and imported, never copied or edited.

## Quickstart

```bash
git clone --recursive https://github.com/ArhaanShah/mario.git && cd mario
pip install -r requirements/main.txt
make test      # pytest on CPU: no GPU, network, MARIO data or extra weights needed
make smoke     # the whole pipeline end to end on synthetic data with a small random-init encoder
```

No `make` (e.g. Windows): `python -m pytest -q` and `python scripts/smoke.py`.

These commands execute CPU models. The alternative-encoder and experimental-head model tests
remain unexecuted in this checkout; the completed artifact-only checks are listed under [Status](#status).
`make smoke` exercises the existing DINO workflow, not full ConvNeXt/VisionFM fine-tuning.

For reporting and study preparation without encoder dependencies or model execution:

```bash
pip install -e ".[benchmark]" pytest
make benchmark-check
python -m scripts.notebook_evidence
```

Without `make`, use `python -m pytest --noconftest tests/test_benchmark_artifacts.py -q -p no:cacheprovider`.
The evidence command reads saved notebook outputs. A full benchmark additionally requires the
real run directories and patient split file, which are not included in this checkout.

## Interactive training dashboard

Open [notebooks/mario_training_dashboard.ipynb](notebooks/mario_training_dashboard.ipynb) for the complete
Task 1 + Task 2 workflow. It previews source B-scans and generated OCTIP/cache views, then trains both stages
from the feature store with a live update after every validation epoch. Each run also writes
`training_progress.jsonl` line by line to `train.out_dir`; every line records the fold, stage, epoch, mean training losses,
and validation metrics, so the live view can be reproduced after a notebook or terminal session ends.

The original sections default to `configs/base.yaml` and use frozen DINO features. Later sections
include selected ConvNeXt/VisionFM feature extraction and full fine-tuning, saved-history plots,
and fold-specific ConvNeXt embedding/head retraining. Set data paths and review the encoder selections
before executing those cells. Full fine-tuning runs through a subprocess rather than the live
`train_heads` callback. The new research selection/refit workflow uses generated CLI configs, not
the historical notebook training cells. The older `task1_training_dashboard.ipynb` remains an
exploratory, Task-1-only DINO notebook.

## Pipeline

One CLI per model-pipeline stage. Each takes `--config`, `--device {auto,cpu,cuda}` and `--set key=value`
overrides. The artifact-only benchmark and study-preparation CLIs have their own arguments;
see [the benchmark guide](docs/convnext-benchmark.md).

| Step | Command | Hardware |
| --- | --- | --- |
| DINOv2/ConvNeXt weights (once) | `python scripts/fetch_weights.py` | network; VisionFM uses manual OCT download/local import |
| Folds | `python -m mario_dino.data.splits` | CPU |
| OCTIP preprocessing cache | `python -m mario_dino.octip_cache` | CPU |
| Frozen feature cache (all encoders, 2 views, resumable) | `python -m mario_dino.features` | GPU recommended; CPU supported |
| Forecaster + head, 4 folds, ablations | `python -m mario_dino.train_heads` | CPU |
| DINOv2 partial fine-tune of the last N blocks | `python -m mario_dino.train_partial` | GPU recommended |
| ConvNeXt/VisionFM full fine-tune (Task 1 only) | `python -m mario_dino.train_finetune` | GPU recommended; images loaded per batch |
| Challenge inference | `python -m mario_dino.infer --task {1,2}` | GPU or CPU |

[RUNBOOK.md](RUNBOOK.md) has the exact real-data command sequence, the Docker image, and the ablation
commands.

## Alternative encoders

| Encoder | Preset | Feature dimension | Training options |
| --- | --- | --- | --- |
| DINOv2 ViT-B/14 with registers | `configs/base.yaml` | 1536: CLS + mean patch tokens | Frozen heads for Tasks 1–2; Task 1 partial fine-tuning |
| ConvNeXt Tiny | `configs/task1_convnext.yaml` | 768: native average pool + final normalization | Frozen heads for Tasks 1–2; Task 1 full fine-tuning |
| VisionFM OCT ViT-B/16 | `configs/task1_visionfm.yaml` | 768: final-normalized CLS | Frozen heads for Tasks 1–2; Task 1 full fine-tuning |

Configure data paths in the inherited `configs/base.yaml`, create/reuse patient-grouped splits, then run
these commands on the cloud instance:

```bash
python scripts/fetch_weights.py --config configs/task1_convnext.yaml
python -m mario_dino.data.splits --config configs/task1_convnext.yaml
python -m mario_dino.train_finetune --config configs/task1_convnext.yaml --device cuda
python -m mario_dino.infer --config configs/task1_convnext.yaml --task 1 --csv /path/to/task1.csv --device cuda
```

For VisionFM, manually place the official OCT checkpoint at `weights/VFM_OCT_weights.pth`, then use
`configs/task1_visionfm.yaml`. Its preset assumes a raw state dict; explicitly select
`encoder.checkpoint_key=teacher`, `student`, or `visionfm_state_dict` for wrapped files.
The released OCT binary's layout remains unverified. Wrapped files containing upstream Namespace
metadata require PyTorch >=2.4; raw tensor files retain PyTorch >=2.1 support. Keep `timm==1.0.30`.

New fine-tuned checkpoints contain the encoder, heads and resolved input transform; inference works
without the original pretrained file and loads one fold encoder at a time. Existing DINO heads/partial
checkpoints retain their format. Heads trained on DINO features cannot be transferred to either new
encoder based on feature dimension alone. See the [encoder reference](docs/encoders.md) and
[cloud instructions](RUNBOOK.md#alternative-task-1-encoders-convnext-and-visionfm-oct) for weights,
preprocessing, frozen baselines, output paths and checkpoint compatibility.

## Models

Encoder weights are prepared locally from the sources below; retinal preprocessing has a weight-free default:

- **DINOv2** ViT-B/14 with registers (or ViT-S for a smaller budget): ungated on the Hugging Face Hub
  under Apache-2.0, downloaded once by `scripts/fetch_weights.py`. Nothing downloads at runtime.
- **ConvNeXt** Tiny (`convnext_tiny.in12k_ft_in1k`): prepared explicitly through the pinned timm
  version and saved as a headless local state dict.
- **VisionFM OCT**: manually downloaded OCT encoder weights. The isolated vendored backbone and its
  upstream CC BY-NC 4.0 license live in `mario_dino/models/vendor/`; no upstream training stack is needed.
- **OCTIP retina localizers**: available only on request from their author. By default preprocessing
  uses `classic` mode instead: two learning-free retina localizers feed DF41's unmodified OCTIP
  PreProcessor. If you obtain the weights, `preprocess.mode: octip` uses the original pipeline.

Details and sources: [RUNBOOK.md § Models](RUNBOOK.md#models-and-where-they-come-from).

## Status

The November 12, 2026 scaffolding validation passed **31 artifact-only tests**, repository Ruff checks,
Python syntax, configuration loading, notebook JSON structure, CLI help and diff whitespace checks.
No training, feature extraction, inference, model construction or model execution tests were run for
this change. The experimental heads and selection/refit trainer await model validation on cloud/CI.

Saved notebook outputs from earlier runs report ConvNeXt Task 1 MCC **0.5860** and OTHER recall
**38.5%**. Those outputs are audited in the [evidence report](docs/results/convnext_task1/notebook_evidence.md);
raw predictions, histories and checkpoints are absent here. The historical scores remain exploratory:
the reported folds also selected epochs, and the DINOv2 and ConvNeXt cohorts differ. No measured
improvement is claimed for either new fusion head.

The original DINO CPU test/smoke results predate these extensions. See
[RUNBOOK.md § Status](RUNBOOK.md#status-what-was-and-was-not-run) for the historical results,
current verification boundary and remaining model checks.

## Docs

| Document | What it is |
| --- | --- |
| [RUNBOOK.md](RUNBOOK.md) | how to run it: setup, the GPU sequence, Docker, model sources, status, deviations from the plan, open questions |
| [docs/encoders.md](docs/encoders.md) | supported encoders, configuration, adapter API, checkpoint formats and compatibility |
| [docs/convnext-benchmark.md](docs/convnext-benchmark.md) | artifact reports, metrics, patient uncertainty, experimental heads and future selection/refit study |
| [Notebook evidence](docs/results/convnext_task1/notebook_evidence.md) | recovered historical scores, confusion matrices, failure modes and evidence limitations |
| [docs/implementation-plan.md](docs/implementation-plan.md) | historical DINO design, with a scope note linking the implemented encoder extension |

## Repository layout

```
mario_dino/            package: config, data/, models/, one module per CLI stage
configs/               base/task presets, alternative encoders, Task 1 fusion and study presets
scripts/               weight/data helpers, run comparison, notebook evidence, smoke
tests/                 artifact-only checks and offline CPU model tests (model tests await execution)
requirements/          main/octip/docker specs and locks; separate artifact-only benchmark specs
docker/Dockerfile      CPU inference image (both envs, weights baked in, no network at runtime)
third_party/           DF41's repo, git submodule at 4d4a11b
docs/                  implementation history, encoder/benchmark references and notebook evidence
```

DF41's repository has no licence file; it is referenced as a submodule, not redistributed.
