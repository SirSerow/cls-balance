# Phase 3 — FLAT training and review

The operational entry point is `python -m cls_balance.train --config configs/flat.yaml`.
It fine-tunes ImageNet V1 EfficientNet-B0 on every official training image and
selects the earliest checkpoint with the highest validation variant macro recall.
It does not run test inference. The supplied recipe fixes seeds 42/43/44, FP32,
224×224, AdamW (LR 1e-4, weight decay 0.01), 30 epochs and patience 7.

## Environment and execution

Preserve the validated CPU environment. Create a separate Python 3.12 GPU
environment with the official matching CUDA builds, as documented by
[PyTorch](https://pytorch.org/get-started/previous-versions/):

```powershell
.venv/Scripts/python.exe -m venv .venv-cuda
.venv-cuda/Scripts/python.exe -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
.venv-cuda/Scripts/python.exe -m pip install -e '.[test]' -r requirements/phase3-plots.txt
.venv-cuda/Scripts/python.exe -m pip check
.venv-cuda/Scripts/python.exe -c "import torch; print(torch.cuda.is_available())"
.venv-cuda/Scripts/python.exe -m cls_balance.train --config configs/flat.yaml
```

All YAML paths resolve relative to the configuration file. Set `prepared_run` to
the completed local Phase 1 directory and `data_root` to the extracted dataset.
The checked-in configuration names this workspace's existing validated preparation;
other machines must substitute their own immutable preparation run. CPU execution
is supported by setting `device: cpu`; do not mix hardware for later latency comparisons.
ImageNet weights download on the first training construction. Checkpoint reloads
never download pretrained weights.

Before training, the entry point validates the full official edition, hierarchy
and split disjointness, compares manifests with the prepared run, and hashes every
image against its validated checksum inventory. Missing/modified data fail before
training. Test labels are used only for dataset-integrity checks. Training,
validation selection, and final generalization evaluation consume train/val only.

Each seed creates a new `runs/flat/flat-s<seed>-<UTC>-<random>/` directory. Epoch
records are written once. The best state is retained in memory until training ends;
interrupted runs have no completion marker and must be rerun from the beginning.
Resume is not implemented. Persistent DataLoader workers are disabled so each
epoch receives the correct augmentation epoch. No images are dropped from batches.

## Artifacts and invariants

- `flat.pt` and `flat.json`: selected 100-way full model and file/state/version hashes.
- `encoder.pt` and `encoder.json`: separate frozen 1,280-dimensional pooled feature
  extractor, excluding the classifier. Its metadata binds the parent FLAT checksum.
- `config.resolved.json`, `environment.json`, `source_hashes.json`, taxonomy,
  preprocessing, dataset hashes: complete training provenance.
- `epoch-*.json`, `history.csv`, `resources.json`: actual loss/recall, selection,
  optimizer steps, image exposures and synchronized wall time.
- `evaluations/`: full deterministic train and validation predictions, produced
  by the common Phase 2 batch-1 adapter and verified by its saved-run auditor.
- `checkpoint_manifest.json`: exact selected-model reload logits and encoder
  embedding checks. `summary.json` marks successful completion and hashes files.

All model tensors use FP32. TF32 is explicitly disabled. Unsupported deterministic
CUDA kernels raise. Validation runs in eval mode without gradients. Selection uses
strict macro-recall improvement; a tie retains the earlier epoch. The final
deterministic training evaluation differs from augmented, training-mode epoch
statistics. Batched epoch validation and final batch-1 inference may differ in
floating-point rounding; both measurements are retained with their boundaries.

`load_checkpoint(path, taxonomy_sha256=..., preprocessing_sha256=...,
dataset_manifest_sha256=..., kind="FLAT")` verifies file bytes, semantic metadata,
tensor state, and version contracts before returning a model. Use
`kind="FT-FROZEN-encoder"` for the encoder. `FrozenEncoder.train(True)` keeps every
submodule in eval mode; encoder parameters have no gradients and own separate
storage from the FLAT model. There is no optimizer state/resume claim.

## Review package and figures

```powershell
.venv-cuda/Scripts/python.exe -m cls_balance.report_flat --runs <seed42-run> <seed43-run> <seed44-run> --output-root results/stage3
```

Use the three completed training directory paths printed by the command. The
reporter rejects incomplete or altered runs and inconsistent configuration,
taxonomy, preprocessing, dataset or hardware. It creates a new immutable package
with learning curves, selected-checkpoint generalization, per-class validation
recalls, and row-normalized confusion matrices. Figures use measured values from
audited artifacts. SVG figures include selectable labels and require no image
generation service. No fake latency/Pareto graph is created in this phase.

Publish aggregate training metrics, validation predictions, configuration,
environment and checkpoint references. Do not publish training-image ID lists,
aircraft images, archives, embeddings or weights. The reporter copies no such
files. Reports explicitly distinguish validation accuracy from final test results.

## Verification

```powershell
.venv/Scripts/python.exe scripts/freeze_tests.py --check
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m pytest -q checks/
```

The Phase 2 test suite, runner and validated freeze manifest remain unchanged.
Phase 3 adds `checks/` and a separate CI workflow. Its tests cover real
EfficientNet save/load, metadata and corruption rejection, frozen encoder
batch-normalization/gradient/storage behavior, full-batch accounting, macro
recall, train/eval state changes, invalid configurations and data tampering.

Training GPU-hours include all epoch training, epoch validation and in-memory
selection time. They exclude initialization, final evaluation and report drawing.
This is elapsed GPU-resident work, not metered GPU utilization. Neither epoch
times nor final-evaluation duration are deployment latency benchmarks. The common
latency harness remains Phase 6.
