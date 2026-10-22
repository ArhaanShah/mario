# MARIO Tasks 1–2: DINOv2 + RNN implementation plan

Oct 22, 2026 · @Arhaan

## Brief for the implementing agent

Build a CPU-testable codebase for MARIO Tasks 1 and 2: a frozen DINOv2 encoder, a GRU that forecasts next-visit features, and one fusion head shared by both tasks. Reuse the DF41 repo for OCTIP preprocessing, scoring and the output format. You have no GPU and no MARIO data, so implement everything, test it on synthetic data with small randomly initialised models, and leave the GPU runs as documented commands.

The model, in one line per task:

- Task 1 (4 classes): `head(f_t, f_{t+Δt})` on the encoder features of both visits.
- Task 2 (3 classes): `head(f_t, f̂_{t+Δt})[:, :3]`, where `f̂ = forecaster(f_t, Δt)`. This replaces DF41's pixel-space PPMAE with a feature-space forecast.

Deliverables:

- Python package `mario_dino/` with one CLI per pipeline stage, YAML configs, and a synthetic-data generator.
- `pytest` suite that passes on CPU, plus `make smoke`, which runs the whole pipeline end to end on synthetic data.
- Inference Dockerfile and `RUNBOOK.md` listing the GPU commands the user will run.

Out of scope: obtaining MARIO data or the OCTIP weight files, training on real data, and continued self-supervised pretraining of DINOv2.

## Facts from the DF41 repo

These come from [pzhangwj/mario\_challenge\_code](https://github.com/pzhangwj/mario_challenge_code) at commit `4d4a11bc8fa1ee0ccf54d3337a2797390ff32ad0` and its two validation CSVs in `csv/`. Pin that commit.

| Fact | Value | Design consequence |
| --- | --- | --- |
| Task 1 CSV columns | `id_patient, side_eye, BScan, image_at_ti, image_at_ti+1, split_type, LOCALIZER_at_ti+1, LOCALIZER_at_ti, sex, age_at_ti+1, age_at_ti, num_current_visit_at_i+1, num_current_visit_at_i, delta_t, label, case` | Schema for loaders and the synthetic generator |
| Task 2 CSV columns | `id_patient, side_eye, BScan, image, split_type, LOCALIZER, sex, age, num_current_visit, label, case` | No `delta_t`: Task 2 uses a constant Δt |
| Task 1 val size | 7,010 pairs, 8,450 unique images | Encoding unique images instead of pairs cuts encoder passes by 40% |
| Task 2 val images | All 3,822 also appear in Task 1 val pairs | Task 2 adds no encoder passes once Task 1 is cached |
| Task 1 val labels | 1: 4,764 (68%), 0: 1,033, 2: 698, 3: 515 | Class-weighted loss; select checkpoints on MCC, not accuracy |
| Task 2 val labels | 1: 2,821 (74%), 2: 650, 0: 351 | Same |
| `delta_t` (Task 1 val) | 8 to 126, median 42; unit not stated | Scale Δt by the training median |
| Patients | All 29 Task 2 patients are among the 34 Task 1 patients | Folds group by `id_patient` across both tasks |
| Label agreement | Task 2 label equals the Task 1 label of the pair starting at the same image in 2,140 of 3,822 rows (56%) | Do not train the Task 2 branch on Task 1 labels by default |
| OCTIP | Needs TensorFlow 2.10, `segmentation_models` 1.0.1 and two weight files not in the repo: `FPN_efficientnetb6_384x384.hdf5`, `FPN_efficientnetb7_320x320.hdf5` | Separate env; a `raw` fallback mode so everything runs without weights |
| Licence | The repo has no LICENSE file | Import from the pinned clone; do not copy its code into the package |

OCTIP API, as DF41 calls it in `utils/dataset.py`:

- `RetinaLocalizer('FPN', 'efficientnetb6', (384, 384), model_directory=d)` and `RetinaLocalizer('FPN', 'efficientnetb7', (320, 320), model_directory=d)`.
- `names, segs = localizer(RetinaLocalizationDataset(paths, batch_size, localizer))` returns segmentations as floats in \[0, 1\].
- `PreProcessor(200, min_height=100, normalize_intensities=True)(path, [seg1_uint8, seg2_uint8], output_directory='')` returns the flattened B-scan as a uint8 array.

DF41 runs this per sample inside `__getitem__`. Batch it over many paths instead.

DF41 defects to avoid repeating: `MARIO_DS_T2` leaves `img_t0` undefined when `processing_octip=False`; t0 and t1 receive independent random augmentations; the training notebook uses `processing_octip=False` while inference uses `True`; `data_processing.octip_preprocess_input` references an undefined `img_path`; checkpoints are selected on accuracy.

## GPU budget design

The only required GPU work is one frozen forward pass per unique image and view. All training of the forecaster and head runs on CPU from cached features, so folds, ablations and hyperparameter sweeps cost no GPU time.

| Stage | Hardware | Cost driver | Notes |
| --- | --- | --- | --- |
| OCTIP cache | CPU (TF) | Once per unique image | Batched localizer calls; idempotent, skips cached files |
| Feature cache | GPU, once | Unique images × 2 views | The only mandatory GPU step |
| Train forecaster + head | CPU | Cached vectors only | 4 folds, ablations, sweeps |
| Partial fine-tune (optional) | GPU | Epochs × last N blocks | Gated; see the runbook |
| Inference | GPU or CPU | One encoder pass per unique test image per view | Task 2 images reuse Task 1 features when they overlap |

Feature cache settings:

- Encoder `vit_base_patch14_reg4_dinov2.lvd142m` (timm), frozen. `vit_small_patch14_reg4_dinov2.lvd142m` is the config switch for a smaller budget.
- Input 518 × 196 (W × H), both multiples of 14, giving 37 × 14 = 518 patch tokens plus 5 prefix tokens.
- `torch.inference_mode()`, autocast to bf16 where supported, otherwise fp16; batch 64.
- Two views per image: identity and horizontal flip. A pair always uses the same view for t and t+Δt.
- Stored vector per view: CLS token concatenated with the mean of the patch tokens (prefix and register tokens excluded), 1,536 values in fp16, 3 KB. This is the pooling the DINOv2 paper uses for linear evaluation. At the Task 1 val size (8,450 images × 2 views) the cache is about 52 MB.

Partial fine-tune, if run, also avoids full passes. Run blocks `0 … 11−N` once and cache their output tokens in fp16: 523 tokens × 768 × 2 bytes ≈ 0.8 MB per image-view, about 6.8 GB for 8,450 images and one view. Training then backpropagates through only the last N blocks, roughly N/12 of the encoder's compute per step. Default N = 2.

DF41's ResNet50 baseline needs full GPU training and has no released weights, so it is optional. The default comparisons run on CPU from the same cache: head on `f_t` alone, identity forecaster (`f̂ = f_t`), and `λ_pred = 0`.

## Repo layout and environments

```
mario-dino/
  third_party/mario_challenge_code/   # git clone, pinned commit; never edited
  mario_dino/
    config.py            # YAML -> dataclasses
    data/schema.py       # column names, CSV validation
    data/splits.py       # GroupKFold on id_patient over Task 1 ∪ Task 2
    data/images.py       # cache lookup, resize, normalise, views
    octip_cache.py       # CLI; runs in the octip env
    features.py          # CLI; unique images -> feature store
    store.py             # fp16 memmap + index CSV
    models/encoder.py    # timm DINOv2 from local weights; pooling; token-cache split
    models/heads.py      # Forecaster, FusionHead
    models/full.py       # MarioModelDINO: encoder + heads, for inference
    losses.py
    train_heads.py       # CLI; CPU training on cached features
    train_partial.py     # CLI; optional GPU partial fine-tune
    infer.py             # CLI; --task 1|2, writes challenge CSVs
    scoring.py           # thin wrapper importing DF41's specificity
  configs/  base.yaml  task1.yaml  task2.yaml  smoke.yaml
  scripts/  make_synthetic_data.py  check_label_alignment.py  fetch_weights.py
  tests/
  docker/Dockerfile
  Makefile  RUNBOOK.md  pyproject.toml
```

Use Python 3.10 everywhere: it is the newest version TensorFlow 2.10 supports, and torch and timm support it too.

- **Main env**: torch, timm 1.x (pin the installed version), numpy, pandas, scikit-learn, opencv-python-headless, pyyaml, pytest. Never imports TensorFlow.
- **octip env** (optional extra): DF41's pins, with `tensorflow-cpu==2.10.0` in place of `tensorflow==2.10.0`, plus `segmentation_models==1.0.1`. If `segmentation_models` fails to import, set `SM_FRAMEWORK=tf.keras`.
- The envs exchange data only through the OCTIP cache directory of PNG files.
- Import OCTIP by adding `third_party/mario_challenge_code/utils/octip_segmentation_2d` to `sys.path` and importing `octip.retina_localizer` and `octip.pre_processor`. Do not import DF41's `utils.dataset`: it pulls in TensorFlow through relative imports.
- The inference Docker image installs both envs' packages into one Python 3.10 environment.

## Milestones

Implement in this order. Each milestone ends with its tests passing on CPU; commit after each.

1. **Scaffold.** `pyproject.toml` with `main` and `octip` extras, the pinned clone in `third_party/`, `Makefile` targets `test`, `smoke`, `lint`. Global seed helper; every CLI takes `--config` and `--device {auto,cpu,cuda}`.
   - Done when `make test` runs an empty suite and `import mario_dino` does not import TensorFlow.
2. **Synthetic data and schema.** `make_synthetic_data.py` writes grayscale PNG B-scans (height 200, width 512 or 768, horizontal bands plus noise) and CSVs with the exact column sets above: about 12 patients, 1–2 eyes, 3–5 visits, 3 B-scans per visit. Task 1 rows link consecutive visits with `delta_t` drawn from 8–126; Task 2 rows cover every image that starts a pair. Labels follow the val imbalance. Also write label-free test CSVs. `schema.py` validates columns and dtypes and that every referenced image exists.
   - Done when the generator is deterministic for a seed and the validator rejects a CSV with a missing column or image.
3. **Splits and label check.** `splits.py`: `GroupKFold(4)` on `id_patient` over the union of Task 1 and Task 2 patients, saved to `splits.json`. `check_label_alignment.py` prints the Task 2 vs Task 1 label crosstab for images that start a Task 1 pair (it reproduces the 56% agreement on DF41's val CSVs).
   - Done when a test asserts no patient appears in two folds and both tasks share one assignment.
4. **OCTIP cache.** `octip_cache.py --mode {octip,raw}` takes the unique image set from all CSVs, writes `cache/<mode>/<image>.png`, skips existing files, and records failures in `failures.csv`. In `octip` mode it calls both localizers in batches (default 16 paths) and `PreProcessor` per image, falling back to the raw image on exception as DF41 does. `raw` mode copies the original.
   - Done when `raw` mode is tested on synthetic data and `octip` mode has a test that skips unless the two weight files are present.
5. **Image pipeline.** Load from the cache, resize to 518 × 196 with `cv2.resize` (W, H), min-max to 0–255, replicate to 3 channels, apply ImageNet mean and std. View 0 is identity, view 1 is horizontal flip; a pair always shares its view. No other augmentation.
   - Done when tests check output shape `(3, 196, 518)`, normalisation, and that view 1 is the exact flip of view 0.
6. **Encoder and feature store.** `fetch_weights.py` creates the timm model with `pretrained=True` and the same kwargs as `encoder.py` (`num_classes=0, img_size=(196, 518), dynamic_img_size=True`) and saves its `state_dict` to `weights/`; `encoder.py` builds with `pretrained=False` and loads it with `strict=True`, so nothing downloads at runtime. Pooling: `tokens = model.forward_features(x)`; vector = `concat(tokens[:, 0], tokens[:, model.num_prefix_tokens:].mean(1))`. Also implement `forward_prefix(x, upto)` and `forward_suffix(tokens, start)` around timm's `patch_embed → _pos_embed → norm_pre → blocks → norm`. `features.py` encodes each unique image once per view into `features.f16` (memmap, shape `[n_images, 2, 2D]`) plus `index.csv`, writes in chunks, and resumes from a done-mask. Use bf16 or fp16 autocast on CUDA, fp32 on CPU.
   - Done when tests with random-init `vit_small_patch14_reg4_dinov2` confirm `forward_suffix(forward_prefix(x, k), k)` equals `forward_features(x)`, and an interrupted extraction resumes without recomputing finished chunks. If the weight download is blocked, document it and continue.
7. **Heads, losses, CPU training.** `heads.py`, with `d = 2D` (1,536 for ViT-B):

   ```python
   class Forecaster(nn.Module):           # f̂ = f_t + W_out·GRU(Δt, W_in·f_t)
       def __init__(self, d, hidden=512, dt_dim=64, dt_scale=42.0):
           super().__init__()
           self.dt_scale = dt_scale           # set to the training median of delta_t
           self.dt_embed = nn.Sequential(nn.Linear(1, dt_dim), nn.GELU())
           self.w_in = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.Tanh())
           self.rnn = nn.GRUCell(dt_dim, hidden)
           self.w_out = nn.Linear(hidden, d)
           nn.init.zeros_(self.w_out.weight); nn.init.zeros_(self.w_out.bias)  # f̂ = f_t at init
       def forward(self, f_t, dt, steps=1):
           e = self.dt_embed(dt.float().view(-1, 1) / self.dt_scale)
           h = self.w_in(f_t)
           for _ in range(steps):
               h = self.rnn(e, h)
           return f_t + self.w_out(h)
   
   class FusionHead(nn.Module):           # logits over [a, b, b - a]; Task 2 keeps [:, :3]
       def __init__(self, d, n_classes=4, p=0.2):
           super().__init__()
           self.fc = nn.Sequential(nn.LayerNorm(3 * d), nn.Dropout(p), nn.Linear(3 * d, n_classes))
       def forward(self, a, b):
           return self.fc(torch.cat([a, b, b - a], dim=1))
   ```

   `train_heads.py` runs per fold from the feature store:
   - Stage A, Task 1 pairs: `CE_w(head(f_t, f_next), y) + λ_pred · MSE(LN(f̂), LN(f_next.detach()))`, λ\_pred = 1. Save the Task 1 checkpoint.
   - Stage B, Task 2 rows: start from the stage A checkpoint and train `CE_w(head(f_t, forecaster(f_t, Δt₀))[:, :3], y)`, with Δt₀ = median training `delta_t`. Save a separate Task 2 checkpoint so Task 1 is unaffected.
   - Flag `task2_aux_from_task1_pairs` (default off) adds the Task 2 loss on Task 1 pairs with y < 3. Enable only if step 3 shows high agreement on the real training CSVs.
   - AdamW, lr 1e-3, weight decay 1e-4, batch 256, up to 50 epochs, early stopping (patience 8). Inverse-frequency class weights from the training fold. Sample one view per pair per step; at evaluation average logits over both views.
   - Model selection on fold-validation MCC (configurable). Write out-of-fold predictions and a metrics JSON with micro-F1, MCC, specificity, and quadratic-weighted kappa for Task 2, matching DF41's `scoring()`.
   - The ablations from the GPU section (head on `f_t` only, identity forecaster, λ\_pred = 0) are config flags in this same script.
   - Done when, on a synthetic feature store where the label is planted in the direction of `f_next − f_t`, stage A beats the majority-class MCC of 0 by a clear margin on CPU, and stage B runs.
8. **Inference.** `models/full.py` wraps encoder, pooling, forecaster and head. `infer.py --task {1,2}` reads a challenge CSV, runs the OCTIP cache step for its images, encodes each unique image once per view, and averages logits over folds and views as DF41's `simple_ensemble_inference` does. It writes `y_pred, cases` plus `y_true` when labels exist, and scores when they do.
   - Done when a parity test shows `full.py` on an image matches the heads applied to that image's cached vector (fp32, atol 1e-4), and `make smoke` writes both result CSVs from synthetic data.
9. **Partial fine-tune (optional, GPU).** `train_partial.py`: cache the tokens after block `11−N` with `forward_prefix`, then train blocks `12−N … 11`, the final norm and the heads with stage A's loss. Two param groups: lr 1e-5 for the blocks, 1e-4 for the heads. The `detach()` on the forecasting target now matters; keep it.
   - Done when a CPU smoke test (random ViT-S, N = 1, 4 images, 2 steps) runs and changes only the unfrozen parameters.
10. **Docker and runbook.** `docker/Dockerfile` on `python:3.10-slim` with both envs' packages, weights copied in, no network at runtime. `RUNBOOK.md` holds the GPU sequence below with exact commands.
    - Done when the image builds and `infer.py` runs inside it on synthetic data on CPU.

## GPU runbook for the user

After the agent finishes, one GPU job is required (step 3); everything else runs on CPU. The agent writes these as exact commands in `RUNBOOK.md`.

1. Put the MARIO CSVs and images in place and set their paths in `configs/base.yaml`. Run `scripts/fetch_weights.py` once on a machine with network access.
2. In the octip env, build the OCTIP cache for train, val and test images. Without the two OCTIP weight files, use `--mode raw`; the rest of the pipeline is unchanged. This runs on CPU.
3. Extract features on GPU for all unique images, two views each. This is the only mandatory GPU job, and it resumes if interrupted.
4. Train stages A and B on CPU for all 4 folds, plus the ablations. Compare fold MCC against the identity-forecaster and `f_t`-only baselines.
5. Optional: partial fine-tune on GPU. Run it only after step 4 is scored, because it is the only step whose GPU cost grows with the number of epochs.
6. Build the Docker image and run `infer.py` for both tasks. Inference encodes each test image once per view.

## Rules and open questions

Rules for the agent:

- Never edit or copy code from `third_party/`; import from the pinned clone.
- No test may need a GPU, network access, MARIO data or the OCTIP weights. Tests that need optional assets skip with a stated reason.
- No network calls at import time or during inference.
- Code uses class indices only; names live in the config.
- Every output directory stores the resolved config, git commit and installed package versions.
- Do not write metrics for real data anywhere. Report anything you could not run as not run, with the reason.

Open questions for the user, to resolve before step 3 of the runbook:

- [ ] Unit of `delta_t`, and the Task 2 prediction horizon. Δt₀ defaults to the training median until this is known.
- [ ] Class names for indices 0–3. The usual MARIO reading is reduced, stable, worsened, other; confirm on the challenge page.
- [ ] Task 2 label definition, given only 56% agreement with the Task 1 label of the same starting image.
- [ ] Where to get the two OCTIP weight files. The OCTIP README says they were distributed from its author's Dropbox; the DF41 README links [leto-atreides-2/octip](https://github.com/leto-atreides-2/octip).
- [ ] Whether the challenge rules allow DINOv2's public pretrained weights.
