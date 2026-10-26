# Implementation roadmap and acceptance checklist

**Status:** Phases 0–4 implemented, with IND-3 measurement deferred by the execution-order gate. See README for operational commands, [PHASE2.md](PHASE2.md) for inference/evaluation contracts, [PHASE3.md](PHASE3.md) for FLAT training/export/reporting, [PHASE4.md](PHASE4.md) for independent cascades, and `results/` for measured evidence. Shared-head training and the full test benchmark remain **proposed**. Treat [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) as the experiment's scientific contract.

## Proposed repository structure

```text
cls-balance/
├── README.md
├── AGENTS.md
├── configs/
│   └── benchmark.example.yaml
├── docs/
│   ├── EXPERIMENT_PLAN.md
│   ├── ROADMAP.md
│   └── RESULTS_TEMPLATE.md
├── src/cls_balance/                  # To implement
│   ├── data.py                       # FGVC-Aircraft loader, banner handling, manifest, transforms
│   ├── taxonomy.py                   # Global IDs and validated routing mappings
│   ├── models/
│   │   ├── backbone.py               # EfficientNet-B0 encoder and model version
│   │   ├── flat.py                   # FLAT baseline
│   │   ├── cascade.py                # Separate full EfficientNet routing/variant models
│   │   └── shared.py                 # Frozen encoder, independent heads
│   ├── train.py                      # Reproducible training/checkpointing
│   ├── infer.py                      # Uniform production hard-route interface
│   ├── evaluate.py                   # End-to-end & oracle classifications
│   ├── benchmark.py                  # CUDA-safe latency / FPS / memory metrics
│   ├── incremental.py                # Holdout-new-variant update protocol
│   ├── artifacts.py                  # Run IDs, manifests, checkpoint versioning
│   └── plots.py                      # Confusion, Pareto and resource figures
├── scripts/                           # To implement
│   ├── prepare_data.py
│   ├── train_all.py
│   ├── evaluate_all.py
│   ├── benchmark_all.py
│   └── report.py
├── tests/                             # To implement
│   ├── test_data.py
│   ├── test_taxonomy.py
│   ├── test_routing.py
│   ├── test_metrics.py
│   ├── test_checkpoint.py
│   └── test_benchmark_smoke.py
├── pyproject.toml                     # To implement; pin/lock final versions
└── .gitignore                         # To implement; exclude image/model/output files
```

Do not create fake datasets or check in training results just to populate this structure. Create folders when their implementation starts.

## Phase 0 — Environment and reproducibility

- [x] Add `pyproject.toml` with pinned compatible `torch`, `torchvision`, `numpy`, `Pillow`, `scikit-learn`, `PyYAML`, `pandas`. Optional profiling/plotting dependencies deferred until needed.
- [x] Add installation instructions and a reproducible CPU environment export.
- [x] Add `.gitignore` for data, runs, checkpoints, embeddings and generated media.
- [x] Log Python/Torch/CUDA/driver/GPU details, config hash, git SHA, random seed, input preprocessing hash for data runs.
- [x] Implement deterministic seeding; unsupported deterministic Torch kernels raise errors.
- [x] Add CPU-only CI smoke tests with synthetic metadata and tiny tensors (no dataset download in CI).

**Exit criterion:** A clean environment can import the package and execute deterministic unit tests without requiring a GPU or licensed dataset download.

## Phase 1 — Dataset, labels, and hierarchy

- [x] Implement official `train`, `val`, `test` dataset loading via `FGVCAircraft`.
- [x] Build split manifests joined by **image ID** across manufacturer/family/variant annotations.
- [x] Verify expected **Torchvision-edition** class cardinality and disjoint split IDs.
- [x] Generate canonical `variant_to_family`, `family_to_manufacturer` and class name/ID maps from training only.
- [x] Implement banner removal, aspect-preserving resize/pad to 224, ImageNet normalization and reproducible train augmentation.
- [x] Count and record multi-child/single-child/empty branches; assert no branch is mislabeled. Bypass values are true-route potential until models exist.
- [x] Write a dataset summary with counts and SHA256 of manifest/taxonomy.

**Exit criterion:** Every test image has exactly one valid variant → family → manufacturer path; sample-visualization spot checks retain entire aircraft geometry.

**Validation:** Passed on all 10,000 official images, with 56 CPU tests and local geometry inspection. See the [stage-one report](../results/stage1/data-20261024T212625Z-4cf9b394/REPORT.md), including the documented archive-checksum source discrepancy.

## Phase 2 — Unified model and evaluation interfaces

Implemented interface (see [PHASE2.md](PHASE2.md) for construction and provenance):

```python
prediction = model.predict(image_tensor)
# prediction: global variant ID, selected ancestors, decisions, executed counts
result = evaluator.evaluate(model, split="val", routing="predicted")
oracle_result = evaluator.evaluate(model, split="val", routing="oracle")
```

- [x] Common single-image inference adapter for FLAT, IND-2, IND-3, and shared models.
- [x] Hard-route selectors use global label maps, not arbitrary local classifier output indices.
- [x] Single-class branch bypasses redundant head inference.
- [x] Oracle mode injects ground-truth ancestor labels **only** for diagnostic evaluation.
- [x] End-to-end metric module computes variant macro recall, top-1, macro-F1, family/manufacturer accuracy, and router diagnostics.
- [x] Save per-image predictions (including selected branch and ground-truth route).
- [x] Add tests verifying global label correctness, oracle behavior, wrong-route handling, and zero test/train overlap.

**Exit criterion:** A dummy model with known labels yields exact expected metrics; errors caused by upstream routing are attributed to the correct stage.

Adapters accept injected modules; EfficientNet factories and training remain in
Phases 3–5. Phase-two validation uses deterministic test models, not experimental
measurements. The checked-in `tests/FROZEN.json` freezes the validated suite and
runner inputs; CI verifies it before executing the tests.

**Validation:** 172 passing CPU tests, including exact dummy-model metrics and
routing error attribution. The complete official metadata also passes evaluator
preflight. See the [phase-two report](../results/stage2/validation-20261025T024200Z-cc4d2a5a/REPORT.md).

## Phase 3 — Baseline FLAT

- [x] Implement ImageNet-pretrained EfficientNet-B0 with 100-variant head.
- [x] Train on official train split; choose checkpoint using validation variant macro recall.
- [x] Save complete model config, seed, optimizer settings, preprocessing and taxonomy versions.
- [x] Evaluate train/validation generalization; delay final test set report until models are frozen.
- [x] Extract the final trained backbone as a separate, immutable `FT-FROZEN` encoder artifact.

**Exit criterion:** All seeds produce loadable full and encoder-only checkpoints. A reloaded checkpoint exactly reproduces inference on a fixed input.

**Validation:** Passed for seeds 42/43/44 on the complete official train/validation
splits. Selected epochs: 17/23/26; completed epochs: 24/30/30. Validation macro recall:
65.58% ± 0.10 pp (sample standard deviation). Exact training-device reloads and
independent CPU encoder/source audits passed for all seeds. See the
[measured report](../results/stage3/flat-20261025T045223Z-788dc5f4/REPORT.md) and
[203-test validation report](../results/stage3/validation-20261025T042101Z-bcc17404/REPORT.md).

## Phase 4 — Independent hierarchical baselines

- [ ] Train manufacturer classifier with full EfficientNet.
- [ ] For `IND-2`, train a full EfficientNet variant specialist for each manufacturer with >1 variants.
- [ ] For `IND-3`, train per-manufacturer family routing and per-family variant specialists where the branch has >1 child.
- [x] Implement IND-3 node planning, fitting, bundle loading and full predicted/oracle evaluation; integration-check singleton and conditional family paths. Genuine IND-3 training is gated until after the main two-stage comparison and benchmark.
- [x] Initialize all independent CNNs from the same published ImageNet weights; no shared trainable weights.
- [x] Implement cold-disk, CPU-preloaded and all-resident model states and a synchronized validation residency probe; declare policy and cache boundaries.
- [ ] Report each specialist's validation score, training exposure, seed, and artifact checksum.

**Exit criterion:** Normal inference uses predicted upstream labels, all executed full-backbone passes are counted, and no extra CNN runs for singleton branches.

Operational commands and audit/publication contracts are in [PHASE4.md](PHASE4.md).
The default training configuration runs IND-2 for all three seeds. Full test
latency/throughput remains Phase 6; residency probes cannot substitute for it.

## Phase 5 — Shared frozen backbone with independent heads

- [ ] Build frozen ImageNet backbone `FROZEN`; train standalone manufacturer, family and variant heads using the correct local training samples.
- [ ] Build fine-tuned-then-frozen `FT-FROZEN` from the corresponding seed's Phase 3 backbone.
- [ ] Use a fixed, configurable MLP structure (default 1280 → 256 → n) and independent optimizers/checkpoints.
- [ ] Implement optional cached pooled embeddings using training-only images; version by backbone+transform hash.
- [ ] Assert backbone gradients and weights never change during head training.
- [ ] Support replacing a specific specialist checkpoint without retraining other heads.
- [ ] Compare `FROZEN-2`, `FROZEN-3`, `FT-FROZEN-2`, `FT-FROZEN-3`.

**Exit criterion:** Each inference calls the shared EfficientNet exactly once; swapping a single head preserves weights and logits of all untouched heads.

## Phase 6 — Benchmark harness

- [ ] Measure model-path and full-pipeline latency (p50/mean/p95/p99), plus sustained FPS.
- [ ] Include preprocessing, host/device transfer, routing, and final label conversion in the declared full-pipeline boundary.
- [ ] Warm up models/representative routes and synchronize GPU before/after timed regions.
- [ ] Run on every test image, batch size 1, at least 5 timed passes; retain per-image latency and branch length.
- [ ] Capture peak GPU memory, total model storage, per-path executed backbone count, and conditional MAC estimates.
- [ ] Record device name, clocks/power regime, framework versions, dtype (FP32 first), model residency and any OOM cases.
- [ ] Add an optional separate FP16 pass on the same checkpoints.
- [ ] Compare inference by actual image paths, not just mean classifier cost.

**Exit criterion:** Repeated runs on a fixed set of images produce plausible distributions and all models share the same measurement boundary; no unsynchronized CUDA timing.

## Phase 7 — Incremental specialist update

- [ ] Preselect a new variant from an existing multi-variant manufacturer using training taxonomy only.
- [ ] Exclude that variant's initial train/val data from **all initial models and backbone fine-tuning**.
- [ ] Train reduced-label initial system; save untouched backbone and unrelated head checksums.
- [ ] Introduce new class with training data; retrain **only** affected variant specialist in independent and shared systems.
- [ ] Evaluate new-class recall, pre-existing within-branch accuracy, outside-branch accuracy, update duration, trainable parameters and replacement checkpoint size.
- [ ] Assert no test-image leakage and that unrelated parameters are unchanged.

**Exit criterion:** A single-head update can be demonstrated and audited. The effect on unaffected classes is measured, not assumed.

## Phase 8 — Analysis and publication

- [ ] Execute seeds 42, 43, 44 with frozen experiment settings.
- [ ] Compute per-seed aggregate metrics and **paired stratified bootstrap** intervals for macro-recall differences.
- [ ] Plot confusion matrices, first-wrong-stage distribution, routing penalty, per-branch latency and accuracy–latency Pareto frontier.
- [ ] Write `docs/RESULTS.md` using `docs/RESULTS_TEMPLATE.md`, linking run IDs and per-image predictions.
- [ ] Report exclusions, failed runs, resource constraints and training budgets.
- [ ] Publish exact commands for downloading dataset, training, evaluating and regenerating all report figures.

**Exit criterion:** A collaborator can follow a single README sequence to reproduce the table/plots (subject to owning the dataset and compatible hardware), and can trace each value back to versioned artifacts.

## Proposed commands (interface to implement, not operational today)

Data preparation is operational as `python -m cls_balance.prepare --config configs/data.yaml` (see README). Phase-two evaluation is an operational Python API; a saved run can be audited with `python -m cls_balance.evaluate --run-dir <existing-eval-run-directory>`. FLAT training is operational as `python -m cls_balance.train --config configs/flat.yaml`; reporting is `python -m cls_balance.report_flat --runs <completed-run-directories>`. Independent training is `python -m cls_balance.train_independent --config configs/independent.yaml`, reporting/auditing is `python -m cls_balance.report_independent --runs <completed-run-directories>`, and residency probes are `python -m cls_balance.residency --run <completed-run-directory>`. The all-architecture and benchmark commands below remain planned:

```bash
python -m scripts.prepare_data --config configs/benchmark.yaml
python -m scripts.train_all --config configs/benchmark.yaml --seeds 42 43 44
python -m scripts.evaluate_all --config configs/benchmark.yaml --split test
python -m scripts.benchmark_all --config configs/benchmark.yaml --dtype fp32
python -m scripts.report --runs-dir runs/
pytest -q
```

The implementation may replace these with a `cls-balance` CLI; update this document when that decision is made.

## Work sequencing and priorities

**Minimal viable comparison:** Phases 0–3, 4 (IND-2 only), 5 (FT-FROZEN-2 only), 6, and the primary portion of Phase 8. This is the fastest credible answer to the main research question.

**Full study:** Add the other 2-/3-stage baselines, oracle-routing breakdowns, multiple training seeds, uncertainty intervals, and the incremental-update experiment.

**Future extensions:** Higher input resolution, intermediate-map spatial specialists, calibrated soft routing/top-k branches, structured backbone compression, ONNX Runtime and TensorRT exports. These are **outside the primary protocol** and should not alter its headline comparison after seeing test results.

## Data artifacts and naming

Each run should have a unique ID (architecture, seed, backbone hash, config hash, hardware+dtype) and preserve:

```text
runs/<run_id>/
├── config.resolved.yaml
├── environment.json
├── dataset_manifest_hashes.json
├── checkpoint_manifest.json
├── predictions_test.csv
├── metrics.json
├── classwise_metrics.csv
├── routing_diagnostics.json
├── latency_test.csv
├── resources.json
└── plots/
```

Do not create empty result files that appear to contain completed measurements.
