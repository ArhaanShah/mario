# mario

Code for Tasks 1 and 2 of the MARIO challenge (monitoring of AMD progression on OCT). Choose DINOv2,
ConvNeXt Tiny, or VisionFM OCT features, with a GRU that forecasts next-visit features and a fusion head
shared by both tasks. DINOv2 remains the default. ConvNeXt and VisionFM also support joint Task 1
encoder/head fine-tuning directly from image pairs.

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

These commands execute CPU models. The alternative-encoder changes received static checks only;
their runtime tests and released pretrained weights have not been validated in this session.
`make smoke` exercises the existing DINO workflow, not full ConvNeXt/VisionFM fine-tuning.

## Interactive training dashboard

Open [notebooks/mario_training_dashboard.ipynb](notebooks/mario_training_dashboard.ipynb) for the complete
Task 1 + Task 2 workflow. It previews source B-scans and generated OCTIP/cache views, then trains both stages
from the feature store with a live update after every validation epoch. Each run also writes
`training_progress.jsonl` line by line to `train.out_dir`; every line records the fold, stage, epoch, mean training losses,
and validation metrics, so the live view can be reproduced after a notebook or terminal session ends.

The notebook defaults to `configs/base.yaml`; its executable cells assume the frozen DINO workflow,
including the preprocessing previews and weight preparation. Set the real image and CSV paths there
before running it. Use the CLI examples below for alternative encoders and full fine-tuning; the dashboard
does not run `train_finetune` or stream its history. The older `task1_training_dashboard.ipynb` is an
exploratory, Task-1-only notebook with the same DINO assumptions.

## Pipeline

One CLI per stage. Each takes `--config`, `--device {auto,cpu,cuda}` and `--set key=value` overrides.

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

Every model the default pipeline needs is public:

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

The original DINO pipeline has historical CPU synthetic-data test and smoke results. Those results
predate the alternative encoders. For the ConvNeXt/VisionFM implementation, syntax, configuration-schema,
lint and diff checks passed; model construction/loading, CPU/GPU execution, pytest and pretrained binary
downloads were not run. New regression tests are written for later cloud/CI execution. Nothing has been
trained or scored on real MARIO data in this work. See
[RUNBOOK.md § Status](RUNBOOK.md#status-what-was-and-was-not-run) for the full list and the open
questions to settle before the GPU run.

## Docs

| Document | What it is |
| --- | --- |
| [RUNBOOK.md](RUNBOOK.md) | how to run it: setup, the GPU sequence, Docker, model sources, status, deviations from the plan, open questions |
| [docs/encoders.md](docs/encoders.md) | supported encoders, configuration, adapter API, checkpoint formats and compatibility |
| [docs/implementation-plan.md](docs/implementation-plan.md) | historical DINO design, with a scope note linking the implemented encoder extension |

## Repository layout

```
mario_dino/            package: config, data/, models/, one module per CLI stage
configs/               base/task1/task2/smoke + task1_convnext.yaml and task1_visionfm.yaml
scripts/               fetch_weights, make_synthetic_data, check_label_alignment, compare_runs, smoke
tests/                 offline CPU suite; alternative-encoder tests await execution
requirements/          loose specs and Python 3.10 lock files: main, octip, docker
docker/Dockerfile      CPU inference image (both envs, weights baked in, no network at runtime)
third_party/           DF41's repo, git submodule at 4d4a11b
docs/                  historical implementation plan and current encoder reference
```

DF41's repository has no licence file; it is referenced as a submodule, not redistributed.
