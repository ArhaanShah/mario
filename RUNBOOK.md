# MARIO Tasks 1–2: DINOv2 + GRU forecaster — runbook

What this repo does, in one line per task:

- **Task 1** (4 classes): `head(f_t, f_{t+Δt})` on frozen DINOv2 features of both visits.
- **Task 2** (3 classes): `head(f_t, f̂_{t+Δt})[:, :3]`, with `f̂ = forecaster(f_t, Δt₀)`, a GRU forecast in
  feature space. This replaces DF41's pixel-space PPMAE.

The only mandatory GPU job is **step 3** (one frozen encoder pass per unique image and view). Training the
forecaster and head, the folds and the ablations all run on CPU from cached features.

Run every command from the repo root. Paths below are the defaults in `configs/base.yaml`.

---

## 0. Setup

```bash
git clone --recursive <this repo> mario-dino && cd mario-dino
git submodule update --init          # if cloned without --recursive; pins DF41 at 4d4a11b
```

**Main env** (torch/timm; never imports TensorFlow), Python 3.10:

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # pick the CUDA build for your driver
pip install -r requirements/main.txt
make test                            # CPU only; no GPU, network, MARIO data or OCTIP weights needed
```

**OCTIP env** (optional; only for `--mode octip`), Python 3.10, CPU TensorFlow:

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

## 2. OCTIP cache (CPU, octip env)

Put `FPN_efficientnetb6_384x384.hdf5` and `FPN_efficientnetb7_320x320.hdf5` in `weights/octip/`, then:

```bash
source .venv-octip/bin/activate
python -m mario_dino.octip_cache --config configs/base.yaml --mode octip
```

Without the OCTIP weights, set `preprocess.mode: raw` in `configs/base.yaml` and run (in either env):

```bash
python -m mario_dino.octip_cache --config configs/base.yaml --mode raw
```

The rest of the pipeline reads `preprocess.mode` from the config, so keep it the same for steps 3–6. The step
is idempotent: it skips images already cached. Images where OCTIP's PreProcessor fails are cached raw,
as DF41 does, and listed in `cache/<mode>/failures.csv`; check that file.

## 3. Feature extraction (GPU, the only mandatory GPU job)

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

## 5. Optional: partial fine-tune (GPU)

Run this only after step 4 is scored: it is the only step whose GPU cost grows with the number of epochs.

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

Each run caches its CSV's images (OCTIP step, skipping cached files), encodes any image not yet in the
feature store (once per view), averages logits over the 4 folds and both views, and writes
`results_task{t}_team_dino_method_{m}.csv` with DF41's columns: `y_true` (only when the CSV has labels),
`y_pred`, `cases`. With labels it also writes `scores_task{t}_method_{m}.json`. Task 2 images that were
already encoded for Task 1 or training are reused. Set the team name with `inference.team`.

Docker (CPU; both envs in one image; weights and trained heads copied in; no network at runtime):

```bash
python scripts/fetch_weights.py --config configs/base.yaml      # weights/ must exist before building
docker build -f docker/Dockerfile -t mario-dino --build-arg GIT_COMMIT=$(git rev-parse HEAD) .
# mounting cache/ and features/ lets the Task 2 run reuse what the Task 1 run (or step 3) computed
MOUNTS="-v $PWD/data:/app/data -v $PWD/weights/octip:/app/weights/octip -v $PWD/cache:/app/cache \
        -v $PWD/features:/app/features -v $PWD/outputs/infer:/app/outputs/infer"
docker run --rm --network none $MOUNTS mario-dino --config configs/task1.yaml --task 1 --out-dir outputs/infer
docker run --rm --network none $MOUNTS mario-dino --config configs/task2.yaml --task 2 --out-dir outputs/infer

# self-test inside the image on synthetic data (random-init encoder, CPU)
docker run --rm --network none --entrypoint python mario-dino scripts/smoke.py --skip-partial
```

---

## Tests and smoke run

```bash
make test     # pytest on CPU (tests needing OCTIP weights + TensorFlow skip with a reason)
make smoke    # whole pipeline on synthetic data: data -> weights -> splits -> cache -> features ->
              # train (+ identity ablation) -> partial fine-tune -> inference for both tasks
make lint
```

On Windows without `make`: `python -m pytest -q` and `python scripts/smoke.py`.
`OCTIP_WEIGHTS_DIR=/path/to/octip_models make test` runs the real OCTIP test when the octip packages are
installed in the same env.

## Status: what was and was not run

Run on CPU (Windows, Python 3.12, torch 2.14.1+cpu, timm 1.0.30):

- `pytest`: all tests pass except `test_octip_mode_real_weights`, which skips (no OCTIP weights or
  TensorFlow).
- `scripts/smoke.py`: the full pipeline on synthetic data, including partial fine-tune and inference for
  both tasks.
- `ruff check` (target py310).
- `uv pip compile` of `requirements/{main,octip,docker}.txt` for Python 3.10 / manylinux x86_64. All three
  resolve (`requirements/*.lock.txt`).
- `check_label_alignment.py` on DF41's val CSVs reproduces 2140/3822 (56%).

Not run, by design or for lack of assets:

- Pretrained weight download (`fetch_weights.py` without `--random-init`): left for the GPU machine.
- `--mode octip`: no OCTIP weight files and no TensorFlow env here.
- Any GPU step, and anything on real MARIO data. No metrics on real data exist in this repo.
- `docker build` (downloads ~2 GB of TensorFlow + torch wheels) and the in-image run.
- A Python 3.10 interpreter: code targets 3.10 (ruff `target-version = "py310"`) but was executed on 3.12.

## Deviations from the plan, and why

- **OCTIP env opencv pin**: DF41 pins `opencv-python==4.4.0.42`, which has no Python 3.10 wheel. Uses
  `opencv-python-headless==4.10.0.84`, the newest release that still supports numpy 1.x, as TF 2.10 needs.
- **DF41 scoring import**: DF41's `utils/scoring.py` imports seaborn, matplotlib and torchvision for
  plotting helpers. `mario_dino/scoring.py` stubs those three while importing the file and then restores
  `sys.modules`. `specificity` itself is DF41's code, imported from the pinned clone and not copied.
- **Feature store reuse**: `inference.store_dir` defaults to `features.store_dir`, so val/test images encoded
  in step 3 (via `data.extra_csvs`) are not encoded again at inference. The store appends new images.
- **Task 2 auxiliary loss** (`task2_aux_from_task1_pairs`, off by default): forecasts each Task 1 pair at its
  own `delta_t`, not at Δt₀.
- **Partial fine-tune inference**: fold checkpoints carry their own encoder suffix, so `infer.py` runs the
  shared prefix once per image and view and each fold's suffix on top.

## Open questions (resolve before step 3)

- [ ] Unit of `delta_t`, and the Task 2 prediction horizon. Δt₀ defaults to the training median until known
      (`train.delta_t0`).
- [ ] Class names for indices 0–3. `class_names` in `configs/base.yaml` holds the usual MARIO reading
      (reduced, stable, worsened, other), unconfirmed. Code uses indices only.
- [ ] Task 2 label definition, given only 56% agreement with the Task 1 label of the same starting image.
- [ ] Where to get the two OCTIP weight files (OCTIP README: author's Dropbox; DF41 README links
      leto-atreides-2/octip).
- [ ] Whether the challenge rules allow DINOv2's public pretrained weights.

## Code map

| Path | What |
| --- | --- |
| `mario_dino/config.py` | YAML → dataclasses; `base:` inheritance, `--set` overrides, unknown keys rejected |
| `mario_dino/data/schema.py` | DF41 column sets, CSV/dtype/image validation |
| `mario_dino/data/splits.py` | GroupKFold(4) on `id_patient` over Task 1 ∪ Task 2 → `splits.json` |
| `mario_dino/octip_cache.py` | CLI: batched OCTIP (`octip`) or copy (`raw`) → `cache/<mode>/` |
| `mario_dino/data/images.py` | resize 518×196, min-max, 3 channels, ImageNet norm, views 0/1 |
| `mario_dino/models/encoder.py` | timm DINOv2 from local weights; CLS + mean-patch pooling; prefix/suffix split |
| `mario_dino/store.py` | append-only resumable memmap store (features and token caches) |
| `mario_dino/features.py` | CLI: unique images → feature store |
| `mario_dino/models/heads.py` | `Forecaster`, `FusionHead`, ablation heads, `MarioHeads` |
| `mario_dino/losses.py` | class weights, forecasting MSE (detached target), stage A/B losses |
| `mario_dino/train_heads.py` | CLI: CPU training per fold, OOF predictions, metrics |
| `mario_dino/train_partial.py` | CLI: optional partial fine-tune on cached prefix tokens |
| `mario_dino/models/full.py` | `MarioModelDINO`: encoder + fold heads (+ suffixes) for inference |
| `mario_dino/infer.py` | CLI: challenge CSV → results CSV (DF41 format) |
| `mario_dino/scoring.py` | DF41 `specificity` + challenge metrics |
| `mario_dino/synthetic.py` | synthetic dataset and planted feature store |
| `scripts/` | `make_synthetic_data`, `fetch_weights`, `check_label_alignment`, `compare_runs`, `smoke` |
