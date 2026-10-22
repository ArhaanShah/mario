# mario

Code for Tasks 1 and 2 of the MARIO challenge (monitoring of AMD progression on OCT). It uses a frozen
DINOv2 encoder, a GRU that forecasts next-visit features, and one fusion head shared by both tasks:

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

## Pipeline

One CLI per stage. Each takes `--config`, `--device {auto,cpu,cuda}` and `--set key=value` overrides.

| Step | Command | Hardware |
| --- | --- | --- |
| Encoder weights (once) | `python scripts/fetch_weights.py` | network |
| Folds | `python -m mario_dino.data.splits` | CPU |
| OCTIP preprocessing cache | `python -m mario_dino.octip_cache` | CPU |
| Feature cache (2 views per image, resumable) | `python -m mario_dino.features` | **GPU, the only required GPU job** |
| Forecaster + head, 4 folds, ablations | `python -m mario_dino.train_heads` | CPU |
| Partial fine-tune of the last N blocks (optional) | `python -m mario_dino.train_partial` | GPU |
| Challenge inference | `python -m mario_dino.infer --task {1,2}` | GPU or CPU |

[RUNBOOK.md](RUNBOOK.md) has the exact real-data command sequence, the Docker image, and the ablation
commands.

## Models

Every model the default pipeline needs is public:

- **DINOv2** ViT-B/14 with registers (or ViT-S for a smaller budget): ungated on the Hugging Face Hub
  under Apache-2.0, downloaded once by `scripts/fetch_weights.py`. Nothing downloads at runtime.
- **OCTIP retina localizers**: available only on request from their author. By default preprocessing
  uses `classic` mode instead: two learning-free retina localizers feed DF41's unmodified OCTIP
  PreProcessor. If you obtain the weights, `preprocess.mode: octip` uses the original pipeline.

Details and sources: [RUNBOOK.md § Models](RUNBOOK.md#models-and-where-they-come-from).

## Status

Everything has been run on CPU on synthetic data: the test suite, the end-to-end smoke run, and the
parity test showing the end-to-end model matches the cached-feature path. Nothing has been trained or
scored on real MARIO data yet, and no GPU step has been run. See
[RUNBOOK.md § Status](RUNBOOK.md#status-what-was-and-was-not-run) for the full list and the open
questions to settle before the GPU run.

## Docs

| Document | What it is |
| --- | --- |
| [RUNBOOK.md](RUNBOOK.md) | how to run it: setup, the GPU sequence, Docker, model sources, status, deviations from the plan, open questions |
| [docs/implementation-plan.md](docs/implementation-plan.md) | the original design and implementation plan (milestones, GPU budget, DF41 facts), unchanged; RUNBOOK lists where the implementation departs from it |

## Repository layout

```
mario_dino/            package: config, data/, models/, one module per CLI stage
configs/               base.yaml (real data), task1.yaml, task2.yaml (inference), smoke.yaml (synthetic)
scripts/               fetch_weights, make_synthetic_data, check_label_alignment, compare_runs, smoke
tests/                 pytest suite (CPU only)
requirements/          loose specs and Python 3.10 lock files: main, octip, docker
docker/Dockerfile      CPU inference image (both envs, weights baked in, no network at runtime)
third_party/           DF41's repo, git submodule at 4d4a11b
docs/                  the implementation plan
```

DF41's repository has no licence file; it is referenced as a submodule, not redistributed.
