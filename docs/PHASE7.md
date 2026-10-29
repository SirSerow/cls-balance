# Phase 7 — incremental specialist update

This experiment trains new initial models. Phase 3–6 encoders and specialists
have already seen all 100 variants and are ineligible as initial checkpoints.
The operational entry points separate training, test evaluation and publication.

## Fixed protocol

`configs/incremental.yaml` uses the existing FP32 EfficientNet-B0/ImageNet V1
recipe, seeds 42/43/44, 224×224 preprocessing, AdamW, a 30-epoch cap and
validation-only early stopping with patience 7. No test-driven selection or
hyperparameter changes are permitted.

The deterministic rule chooses the lowest canonical training-taxonomy variant ID
whose manufacturer has at least three variants and whose family has at least
two variants. The manufacturer therefore keeps a trained specialist and the
family remains represented. For the validated edition this is **global ID 2,
Boeing 737-200**. Boeing has 22 variants, so its initial specialist has 21
outputs. The existing 737 family remains present.

All 737-200 **train and validation** images are removed from every initial
supervised dataset: the FLAT model that provides the task-adapted encoder, the
manufacturer CNN, independent specialists and shared heads. No initial selection
uses the withheld validation images. Source annotations, manifests and original
image checksums are verified against the complete official preparation.

Initial output maps contain 99 active IDs over the original 100-class taxonomy.
The implementation deliberately retains canonical IDs instead of renumbering
labels. This is the reduced active taxonomy; neither official data nor scoring
IDs are changed. Every manufacturer/family remains represented.

At introduction, both systems replay all affected-branch training examples,
including the new class. Validation-only selection uses that branch's complete
validation examples. Both replacements preserve all old feature/head tensors
and old output rows at initialization, add one deterministically seeded output
row, then train only the affected specialist. The independent replacement trains
the full CNN; the shared replacement trains its MLP. Each head has its own
optimizer, shuffle generator and early stopping state. Shared initial fitting
extracts one fresh augmented view per training image per epoch; no persistent
embedding cache is used.

The shared encoder's weights, buffers and mode remain frozen, including while
extracting features of introduced-class images during the update. Manufacturer
routers and every other specialist retain their saved bytes. Original affected
specialists are also retained so both systems remain evaluable before/after.
New checkpoint paths are immutable; nothing is overwritten in place.

## Commands

Use the validated CUDA environment and optional plotting dependency from Phase 3:

```powershell
.venv-cuda/Scripts/python.exe -m pip install -r requirements/phase3-plots.txt
.venv-cuda/Scripts/python.exe scripts/train_incremental_awake.py train --config configs/incremental.yaml
```

The wrapper requests idle-sleep prevention only for its own process lifetime,
releases it on exit, and leaves OS/driver power plans unchanged. It imports the
training module normally, keeping custom datasets compatible with Windows
DataLoader spawn workers. The direct module command is also available:

```powershell
.venv-cuda/Scripts/python.exe -m cls_balance.incremental train --config configs/incremental.yaml
```

Training completes **all seeds and both replacements before any test inference**.
Use the emitted immutable training run ID for evaluation and publication:

```powershell
.venv-cuda/Scripts/python.exe scripts/train_incremental_awake.py evaluate --suite runs/incremental/<training-run-id>
.venv-cuda/Scripts/python.exe -m cls_balance.report_incremental --training runs/incremental/<training-run-id> --evaluation runs/incremental/<evaluation-run-id> --output-root results/stage7
.venv-cuda/Scripts/python.exe -m cls_balance.report_incremental --audit results/stage7/<review-run-id>
```

Every before/after, predicted/oracle evaluation contains **all 3,333 official
test images**. The initial known-class score uses the 99 known variants as its
explicit denominator. Retrospective all-100 scores and zero new-class-before
recall are diagnostics, not headline initial performance. Test inference occurs
only after all checkpoints and update choices are frozen.

No new latency/FPS measurement is inferred from update duration. Update timing
includes shared image feature extraction, optimization, validation, selection,
checkpoint export and exact reload; model initialization/loading and test
inference are excluded. Initial training costs are retained separately, including
the shared encoder's full FLAT training cost. Replacement sizes use decimal MB.

## Resume and audit contracts

```powershell
.venv-cuda/Scripts/python.exe scripts/train_incremental_awake.py train --config configs/incremental.yaml --resume runs/incremental/<incomplete-training-run-id>
```

Resume requires identical resolved configuration, exact source hashes,
preparation and environment. Completed nodes/stages are checksummed and reused;
incomplete fits restart in a new immutable attempt. Optimizer/epoch resume is
not implemented. A completed training suite cannot be resumed or overwritten.
If sources/configuration change, start a new suite. Failed attempts have no
completion summary and contribute no reported measurements.

The training auditor independently derives exact supervised IDs and local/global
maps for the reduced backbone, each initial classifier and each replacement.
It verifies seed, source/config/data/preprocessing hashes, encoder parent,
checkpoint tensor/file hashes, class exclusion, earliest best validation epoch
and epoch support. The report also compares every untouched module's logits on
the same synthetic fixed input and shared encoder features exactly.
Exact training and inference source files are preserved with their hash manifests,
so later report/tooling changes do not obscure the implementation that produced
the measured checkpoints and predictions.

Before/after predicted routes must have identical manufacturer decisions. Every
image routed outside the affected branch must have an identical final prediction
and path. **True** outside-branch accuracy is measured independently: images
mistakenly routed into Boeing can change even though their correct specialists
are untouched. Oracle routing remains a named, separate diagnostic.

## Review artifacts

The report includes new-class recall, existing affected-branch macro recall,
outside-branch macro/micro accuracy, known-class and overall metrics, update
duration, trainable parameters, replacement size, per-seed measurements and
complete test predictions. Graphs show before/after accuracy, update costs,
learning/selection curves and all 100 class recalls.

Five full network schematics expand initial FLAT and both hierarchical systems
before/after. They contain all 30 manufacturers, every trained specialist,
singleton bypasses, every active global output, and per-seed selected epochs.
The replacement is highlighted. Companion JSONs enumerate exact checkpoint
hashes/output maps; the EfficientNet detail expands the actual Torchvision
module inventory. These are generated directly from trained artifacts, not
illustrative estimated networks.

Rebuild the report and all figures without local dataset images or weights:

```powershell
.venv-cuda/Scripts/python.exe -m cls_balance.report_incremental --rebuild results/stage7/<review-run-id> --rebuild-output .tmp/rebuilt-phase7
```

Publication uses an explicit artifact selection. Test IDs/predictions and
aggregate train/validation histories are permitted; supervised train/validation
image-ID lists, images, caches and weights are rejected. Public audits verify
nested checksums and recompute group metrics from saved test predictions.

## Checks

```powershell
.venv/Scripts/python.exe scripts/freeze_tests.py --check
.venv/Scripts/python.exe -m pytest -q tests checks
.venv-cuda/Scripts/python.exe -m pytest -q checks/test_phase7.py
```

The original frozen tests are unchanged. Phase 7 checks cover deterministic
taxonomy selection, family retention, train/validation exclusion with unchanged
test manifests, invalid masks, global-ID output expansion, full miniature
training/reload/replacement, frozen weights/buffers, immutable artifact reuse,
semantic audits, corruption, fixed group denominators, predicted-route locality
and full network inventories. Synthetic tests are never experimental results.

## Measured run and review

The [three-seed experiment report](../results/stage7/incremental-20261029T102022Z-dd4acc20/REPORT.md)
contains all 24 official full-test evaluations, 79,992 saved predictions, five
complete network schematics, every specialist's validation curves, and update
cost/accuracy plots. The shared update averages 192.89 seconds versus 375.96
seconds for the independent CNN. New-class recall averages 58.82% / 54.90%,
respectively. Existing Boeing-class macro recall drops by 3.48 / 0.37 pp;
outside-branch accuracy is measured to remain unchanged in both systems.

The [validation publication](../results/stage7/validation-20261029T103720Z-f75088da/REPORT.md)
retains actual execution/check logs: 305 passing CPU tests, 73 passing CUDA
checks, the verified 172-test original freeze, all-image checksum revalidation,
and a byte-identical rebuild of all 432 review files. All 120 checkpoint artifacts
were audited, including the initial 99-class backbone exclusion and unchanged
unrelated parameters. Initial seed-42 fitting overlapped CPU checks; update timing
and test inference ran without those foreground workloads. No new inference
latency or broader class-introduction claim is made.
