# Encoder reference

This describes the implemented encoder extension. For setup, downloads and cloud commands, see
[the runbook](../RUNBOOK.md#alternative-task-1-encoders-convnext-and-visionfm-oct).
The [original implementation plan](implementation-plan.md) describes the historical DINO baseline.
For experimental Task 1 heads, artifact reporting and the inner-selection/refit protocol, see
[the benchmark guide](convnext-benchmark.md). The completed checks include 31 artifact-only tests;
alternative-encoder/fusion model tests and real pretrained compatibility remain unvalidated here.

## Supported workflows

| `encoder.kind` | Supported model | Frozen features + `train_heads` | `train_partial` | `train_finetune` |
| --- | --- | --- | --- | --- |
| `dinov2` (default) | timm DINOv2 names used by existing configs | Tasks 1–2 | Task 1, last N blocks + final norm | Use `train_partial` |
| `convnext` | `convnext_tiny.in12k_ft_in1k` | Tasks 1–2 | Rejected; use `train_finetune` | Task 1, full encoder + heads |
| `visionfm` | `vit_base_patch16` | Tasks 1–2 | Rejected; use `train_finetune` | Task 1, full encoder + heads |

Task 2 image-based end-to-end training is outside this extension. The original
`task1_convnext.yaml` and `task1_visionfm.yaml` presets inherit the base Task 2 CSV: optional frozen
`train_heads` training runs both stages when that CSV is configured. Experimental fusion and study
presets disable Task 2 CSVs. Full image fine-tuning reads Task 1 pairs only.

## Configuration and preprocessing

`EncoderCfg` retains `name`, `weights`, `img_size` (height, width), and `kwargs`. New fields are:

| Field | Default | Meaning |
| --- | --- | --- |
| `kind` | `dinov2` | Explicit architecture family; names must agree with the family |
| `checkpoint_key` | `null` | Raw weights; only VisionFM accepts `teacher`, `student`, or `visionfm_state_dict` wrappers |
| `preprocessing` | `legacy_dino` | Shared input-transform preset; independent of retinal `preprocess.mode` |

| Preset | Resize and scaling | Normalization |
| --- | --- | --- |
| `legacy_dino` | OpenCV linear resize, per-image min/max scaling | ImageNet mean/std; preserves historical numerical behavior |
| `imagenet` | OpenCV bicubic resize of uint8, then /255 | ImageNet mean/std |
| `visionfm_oct` | OpenCV bicubic resize of uint8, then /255 | OCT mean `(0.21091926, 0.21091926, 0.21091919)`, std `(0.17598894, 0.17598891, 0.17598893)` |

All paths read grayscale B-scans and repeat them into three channels. `preprocess.mode` still selects
the retinal cache (`classic`, `octip`, or `raw`). Extraction, paired training, validation and inference
share the versioned input-spec resolver in `data/images.py`. New presets use `[192, 512]`;
`[224, 224]` is also supported. Configured dimensions must be positive multiples of 14 for DINOv2,
32 for ConvNeXt, or 16 for VisionFM. No automatic rounding/crop is applied.

`FinetuneCfg` is separate from `TrainCfg` and `PartialCfg`:

| Field | Default |
| --- | --- |
| `out_dir` | `outputs/finetune` (overridden separately by each new preset) |
| `batch_size` | `8` |
| `lr_encoder`, `lr_heads` | `1e-5`, `1e-4` |
| `weight_decay` | `1e-4` |
| `max_epochs`, `patience` | `10`, `3` |
| `max_steps` | `null` (optional per-fold optimizer-step cap) |
| `init_heads_from` | `null` (optional compatible frozen Task 1 run directory) |

`train.folds`/`--folds`, `train.lambda_pred`, `train.select_metric`, `train.delta_t0`,
`model.*`, and `features.amp` also apply to full fine-tuning. Images are loaded on demand, with the
same sampled horizontal flip for both pair members. Each fold reloads the source encoder and creates
its own heads/optimizer. Class weights and time scaling use training-fold rows. CUDA bf16 is used
when enabled and supported; otherwise training uses fp32. These defaults are starting settings,
with no measured accuracy or cloud memory guarantee.

### Task 1 fusion and research configuration

| Field | Default | Behavior |
| --- | --- | --- |
| `model.fusion` | `legacy` | Existing pair head; `delta_mlp` and `other_gate` opt into experimental Task 1 heads |
| `model.fusion_hidden` | `256` | Hidden width for experimental fusion heads |
| `research.protocol` | `legacy` | Existing fold-validation selection; `grouped_holdout` enables inner selection and fresh refit |
| `research.inner_splits` | `null` | Saved deterministic patient partitions produced by study preparation |
| `research.cohort_manifest` | `null` | Saved cohort/source hashes produced by study preparation |

Experimental heads require `model.head_input=pair` and no Task 2 training CSVs. Their encoder features
and pooling remain unchanged. The OTHER gate composes four-class log probabilities from an OTHER
decision and conditional three-class progression probabilities; the label is not a quality-only label.

The research protocol requires ConvNeXt, four outer patient folds, local pretrained weights, MCC
selection, fresh heads and no step cap. It uses 20% of outer-training patients for epoch selection,
then refits a fresh model on all outer-training patients for the selected epoch count. Outer evaluation
runs once after refit. Warm starts and cached-feature/partial trainers are rejected for this protocol.
Use `python -m mario_dino.study --config configs/task1_convnext_study.yaml` to prepare the 12
seed/variant configs without running models. The [benchmark guide](convnext-benchmark.md#prepare-the-locked-future-study)
documents preparation, provenance checks, future execution and reporting.

## Adapter interface and weights

`models/encoder.py` remains the public entry point. `build_encoder(enc_cfg, native_grid_size=14)`
constructs a model without consulting source files; `load_encoder(enc_cfg, device, trainable=False)`
loads local weights strictly, returning frozen/eval by default. `trainable=True` enables gradients
and training mode. Every adapter exposes `out_dim` and a differentiable
`forward(images: [B, 3, H, W]) -> [B, out_dim]`. `Task1Net` provides the differentiable encoder/head
composition; `MarioModelDINO` retains its inference methods.

- **DINOv2:** unchanged final-normalized CLS + mean patch pooling, excluding register/prefix tokens.
  ViT-B yields 1536 features. Prefix/suffix methods and token caches remain DINO-specific.
- **ConvNeXt:** headless timm model (`num_classes=0`, `global_pool="avg"`), native final normalization,
  with `out_dim` read from the model (768 for standard Tiny). DINO image-size kwargs are not passed.
- **VisionFM:** isolated ViT-B/16 (768 dimensions, 12 blocks/heads, biased QKV, LayerNorm epsilon 1e-6).
  Final-normalized CLS is the MARIO pooling choice. The native square positional grid is inferred from
  `pos_embed`, retained, and interpolated for runtime rectangles using the pinned upstream algorithm.
  Masked pretraining and FlashAttention are disabled. Vendored code and license are in
  [`models/vendor`](../mario_dino/models/vendor/visionfm_vit.py), pinned to upstream commit
  `648ab0895f8e80e2c937f658bd2f1629ea6d27e7` under CC BY-NC 4.0.

DINOv2/ConvNeXt preparation uses `scripts/fetch_weights.py` with network access; runtime construction
uses `pretrained=False`. ConvNeXt supports the headless timm state dict written by that script,
not an arbitrary torchvision classifier checkpoint. VisionFM accepts a raw backbone tensor dict or
an explicitly selected supported wrapper. Its helper strips only known leading `module.`/`backbone.`
prefixes with collision checks, logs discarded known heads/masking tensors, and strictly loads the
remaining full backbone. Unknown extras, missing keys and incompatible shapes fail.

Optional VisionFM `--source` conversion writes canonical raw weights and a SHA-256 sidecar; use
`checkpoint_key=null` for its output. Upstream Namespace metadata is narrowly allowlisted when a
wrapper is selected (PyTorch >=2.4); there is no unrestricted-pickle fallback for external encoder
weights. Production missing/malformed weights never silently become random initialization.
Explicit null weights or `--random-init` remain test-fixture options.
The research protocol requires a local pretrained source and rejects null weights.

## Checkpoint and cache compatibility

| Run type | Saved kind | Runtime dependencies and inference |
| --- | --- | --- |
| Frozen heads | `heads`, or missing kind for historical files | Matching local source encoder weights and transform; unique-image features shared across folds |
| DINO partial | `partial` | Matching local DINO source weights plus saved fold suffix; prefix shared across folds |
| Full alternative fine-tuning | `finetuned`, `format_version: 1` | Self-contained encoder + heads + input specification; one full fold loaded at a time |

Fine-tuned checkpoints retain `task`, `fold`, `heads`, `hparams`, `config`, `best_epoch`, and
training-fold median, plus full adapter `encoder` state, resolved `encoder_spec`, source provenance
(`source.sha256`, `source.checkpoint_key`) and feature identity. `encoder_spec` records kind/name,
kwargs, pooling, dimension, native positional grid, runtime size, input transform, retinal mode and
source identity. Inference reconstructs and strictly loads from this data without the original
pretrained file or sidecar. Saved transform settings take precedence over `encoder.*` initialization
settings; retinal `preprocess.mode` must agree. Optimizer resume is not implemented.

Legacy heads retain their existing hyperparameter payload and state-dict keys. Experimental heads
store `fusion` and `fusion_hidden` in `hparams`; missing fusion fields reconstruct the legacy head.
Strict head-state loading and cross-fold architecture checks still apply. Checkpoint format version 1
is retained. Gate outputs use the existing score-averaging path, so averaging across views/folds
combines log probabilities geometrically before normalization.

Fine-tuned inference averages logits over both views and all folds, computing image features in
memory per fold without persistent frozen caches. Mixed checkpoint kinds or incompatible fold
specifications are rejected. `--allow-identity-mismatch` cannot bypass hard architecture/shape checks
in legacy inference or the fine-tuned specification checks.

Default DINO feature identities retain their exact historical payload. New encoder kinds and
non-default DINO transforms use richer versioned identities covering family, architecture kwargs,
pooling, input transform/size, source SHA-256 and wrapper selection. Feature stores compare full
identities and row shapes. Use separate stores/runs when switching encoders or transforms.

DINO heads are not compatible with alternative embeddings, even if dimensions match. A head warm
start for full fine-tuning must be a frozen Task 1 checkpoint for the same fold, full available
feature identity, feature dimension and configured head architecture. The new
`finetune.init_heads_from` does not inherit `partial.init_heads_from`.

## Outputs and verification

The presets keep `features/{convnext,visionfm}`, `outputs/{convnext,visionfm}_heads`,
`outputs/{convnext,visionfm}_finetune`, and `outputs/infer_{convnext,visionfm}` separate. Full training
writes per-fold `task1.pt`, `history.json`, and `oof_task1.csv`, plus combined OOF, `summary.json`,
`metrics.json`, resolved config and provenance. For frozen inference, override `--run-dir` to the
corresponding `_heads` directory. Full inference uses the preset's `_finetune` default.

Standalone experimental presets use `outputs/convnext_delta_mlp` and `outputs/convnext_other_gate`.
Prepared study runs use `outputs/convnext_study/runs/seed{seed}/{variant}` by default. Within a study
fold, `selection/` stores the inner-selected checkpoint/history, `refit/` stores the refit history and
checkpoint reference, and `task1.pt` is the final self-contained checkpoint. Each phase records fitting
patients, class support, time scale, parameter count and checkpoint hash. Nonempty research run
directories are refused; study preparation never launches training.

Current scaffolding verification passed 31 artifact-only tests plus syntax, configuration, lint,
notebook-structure, CLI-help and diff checks. Those tests validate reporting and patient-split logic;
they do not validate encoders or training. The written model tests cover gradients/updates, strict
imports, upstream parity, transforms, identities, paired views, training, inference reconstruction,
legacy checkpoints and gate normalization. They remain unexecuted here. Actual released OCT format,
pretrained compatibility, GPU memory/throughput and the new trainer require later cloud/CI validation.
See [the runbook verification commands](../RUNBOOK.md#tests-and-smoke-run) and
[the evidence report](results/convnext_task1/notebook_evidence.md) for the separately recorded historical
real-data results; those saved outputs do not establish correctness of the new scaffolding.
