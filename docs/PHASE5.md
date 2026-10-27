# Phase 5 — frozen shared backbone and independent heads

Measured FT-FROZEN-2 validation macro recall for seeds 42/43/44 is **63.43% ±
0.25 pp** (sample SD). See the [report and figures](../results/stage5/shared-20261027T145809Z-ee48fbad/REPORT.md)
and [validation/reproduction evidence](../results/stage5/validation-20261027T150049Z-777f9bfa/REPORT.md).
These publications include validation predictions and aggregate training histories;
dataset images, training-ID lists, embeddings and checkpoint tensors stay local.

The operational primary configuration is [shared.yaml](../configs/shared.yaml).
It fits FT-FROZEN-2 for seeds 42/43/44 using the corresponding selected Phase 3
encoder. Each seed trains 18 heads: a manufacturer router and 17 manufacturer
variant specialists. Thirteen single-variant manufacturer branches bypass a
classifier. The common adapter returns global variant IDs and executes exactly
one frozen EfficientNet feature pass, including singleton and oracle routes.

```powershell
.venv-cuda/Scripts/python.exe -m cls_balance.train_shared --config configs/shared.yaml
.venv/Scripts/python.exe scripts/freeze_tests.py --check
.venv/Scripts/python.exe -m pytest -q tests checks
.venv-cuda/Scripts/python.exe -m cls_balance.report_shared --runs <seed42-run> <seed43-run> <seed44-run> --output-root results/stage5
```

Replace placeholders with the new run directories printed by training. Dataset
preparation is separately invocable; training verifies its complete image and
annotation checksums before fitting. Publication only consumes completed runs.
Optional `--flat-runs <three-runs>` and `--independent-runs <three-runs>` add audited
paired validation comparisons to the figures. Matplotlib uses the existing pinned
`requirements/phase3-plots.txt` plotting dependency.

On Windows, the measured command can use
`python scripts/train_shared_awake.py --config configs/shared.yaml` to hold a
temporary process-lifetime idle-sleep request. It releases on exit and does not
alter the power plan or GPU clocks. `--smoke-test` checks Windows spawn behavior.

## Fitting and feature contracts

Each head is Linear(1280,256) → ReLU → Linear(n), configurable with
`hidden_dimension`. It has its own stable initialization seed, parameters,
AdamW state, shuffling generator, local validation macro-recall selection,
early stopping and checkpoint. A stable hash of `(experiment seed, node key)`
sets its initialization/shuffle seed, avoiding dependence on traversal order.
Training uses all official training images appropriate to that node's true
parent, with explicit global-to-local targets. Inference uses predicted parents.

Default settings retain the existing 30-epoch cap, patience 7, batch size 16,
FP32 and AdamW weight decay 0.01, with the planned head LR 0.001. Validation
selection uses strict improvement and chooses the earliest tie. All validation
classes must have support. There is no test inference or test-label selection.

The encoder stays in eval mode without gradients. Parameter and buffer hashes
(including BatchNorm running statistics) are checked throughout training. Each
epoch extracts a fresh seeded augmented view for every training image once;
all heads consume these features read-only. Deterministic validation features
are held in memory for local selection. Each head's early stopping is independent
of other heads. A selected head is saved immediately; completed heads receive
no further optimizer updates. Encoder and head reloads must exactly reproduce
embeddings/logits on a fixed validation input on the training device.

`cache_root: null` uses memory only. To persist training-only augmented feature
views, set an ignored directory such as `../embeddings/shared`. Cache contracts
include exact training IDs, dataset manifest, backbone state, preprocessing
profile, transform implementation hash, augmentation policy, seed and epoch.
Corrupt/incomplete caches fail. Every epoch has its own cache version; a fixed
deterministic cache cannot silently replace augmented training. Validation/test
features are never written to this disk cache. Full final evaluation always
processes images through the real inference adapter.

## Generic features and depth ablations

Use a separate YAML/output root with `backbone: FROZEN` to build a frozen
ImageNet V1 EfficientNet; `flat_runs` is then unused and may be omitted. Set
`depths: [3]` or `[2, 3]` for conditional family routers and family variant heads.
Node plans omit every singleton classifier. With both depths, only an identical
node task (the manufacturer router) is reused. Every head has an independent
optimizer and checkpoint; independently trained heads never share weights.

Generic ImageNet-frozen and three-stage **measurements** remain gated after the
primary FT-FROZEN-2/IND-2 comparison and Phase 6 benchmark, following the execution
order. Both model types and both depths are exercised in synthetic integration
checks; these checks are not reported as aircraft-training results.

## Deployment and specialist replacement

```python
from pathlib import Path
from cls_balance.train_shared import load_shared

run = Path("runs/shared/<completed-run-id>")
model = load_shared(run, depth=2)
prediction = model.predict(preprocessed_image)  # CHW FP32, 224 × 224
updated = load_shared(run, depth=2, replacements={
    "variant-manufacturer-1": Path("<compatible-replacement-head.pt>"),
})
```

Replacement keys are canonical taxonomy node keys. The loader rejects wrong
taxonomy, dataset, preprocessing, encoder hash, seed, dimensions, global-ID map,
checkpoint hashes and unknown node keys. Loading an override constructs a new
adapter; it neither changes the original bundle nor retrains any modules. The
replacement must retain the original head's exact training and selection ID
lists; a shape-compatible head trained or selected on test images is rejected.
The caller records replacement file hashes and the new `evaluate.model_hash` when
creating new evaluation provenance. Tests change a specialist and compare the
exact weights **and logits** of every untouched head on fixed features.

This mechanism does not implement the Phase 7 introduced-class study: adding
labels requires a new compatible taxonomy and initial backbone training that
excluded the introduced variant.

## Audit and publication

Each immutable run contains source/config/environment records, backbone origin,
encoder and head checkpoint manifests, local histories/selection/exposures,
resources, deployment bundles, and complete predicted/oracle validation CSVs.
The final summary exists only after export/reload and evaluation audits succeed.
Interrupted runs are incomplete and must restart under new IDs; optimizer resume
is not implemented.

```python
from cls_balance.report_shared import audit_training_run
audit_training_run(run)
```

The audit verifies nested file checksums, recomputes saved prediction metrics,
checks node selection/exposures, reloads the deployment bundle and binds the
evaluations to its model fingerprint. Publication validates matching data,
settings, training sources and environments across seeds. Optional baselines
must match validation image IDs, preprocessing, seeds and evaluation hardware.

The public allowlist includes validation predictions/labels, aggregate training
metrics/histories, hashes and SVG plots. Aircraft images, training-image lists,
embeddings and tensors remain in ignored local storage. Resources distinguish
shared feature extraction, head epochs, total fitting wall time/GPU-hours, head
exposures, trainable parameters and encoder/all-head stored bytes. The Phase 3
FLAT training cost is separately recorded, and must be included in total
training-compute comparisons. No time recorded here is a p95/FPS benchmark.
