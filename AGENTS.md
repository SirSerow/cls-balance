# AGENTS.md — cls-balance

This file guides Codex and other coding agents implementing the experiment.

## Mission

Build a rigorous, reproducible comparison of **independently trained hierarchical EfficientNet cascades** and **one frozen shared EfficientNet feature extractor with independently trained hierarchical specialist heads** on FGVC-Aircraft. Optimize for the validity of the comparison before optimizer tricks or runtime speed.

## Mandatory documents

Read [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md) before architecture, training, metrics, or data decisions. Follow [docs/ROADMAP.md](docs/ROADMAP.md) phase gates, and publish genuine measurements using [docs/RESULTS_TEMPLATE.md](docs/RESULTS_TEMPLATE.md). \`configs/benchmark.example.yaml\` is a proposed configuration schema, not an implemented program.

## Scientific invariants

1. Use official FGVC train/val/test splits and construct all hierarchy maps **by image ID**, not by parsing label strings or relying on label-number coincidences.
2. Standard Torchvision target is **100 variants, 70 families, 30 manufacturers**; the original FGVC site describes a different broader edition. Validate the actual downloaded dataset edition.
3. Use **the same test image set, preprocessing, input resolution, EfficientNet-B0 basis, seeds, precision and hardware** across architectural comparisons.
4. Never use test labels for training, routing at normal inference, calibration, hyperparameter tuning or model selection. Oracle routing is a named diagnostic only.
5. Main accuracy metric = **variant macro recall** (mean per-class accuracy); main latency metric = **batch-1 p95 full-pipeline time**.
6. Measure the complete predicted-route pipeline, including manufacturer/optional family routing and the chosen specialist; avoid evaluating only the selected specialist.
7. Independent cascade nodes each have their own full CNN weights. Shared variants run the EfficientNet backbone **exactly once** per image and train small heads independently with its weights frozen.
8. Single-child taxonomy branches are deterministic outputs, not extra trained classifiers. Report bypass rates.
9. For introduced-variant experiments, exclude the new variant from **initial backbone** training as well as initial specialist training.
10. Never fabricate or copy estimated experimental metrics. All reported figures must be traceable to run IDs, configurations and saved per-image predictions/timings.

## Coding style

- Prefer small modules with explicit types, docstrings and deterministic input/output contracts.
- Centralize taxonomy IDs and map specialist-local logits to globally unique variant IDs before scoring.
- Fail early on missing labels, invalid hierarchy, unexpected dataset edition, split leakage and mismatched checkpoint/preprocessing hashes.
- Keep data preparation, training, evaluation and benchmarking separately invocable.
- Make results machine readable: JSON for aggregate/run metadata, CSV or Parquet for per-image outputs.
- Log seed, git SHA, data manifest hash, environment, torch/torchvision/CUDA versions, device and runtime/model residency policy.
- Cache frozen embeddings only with explicit backbone and preprocessing hash; deterministic caches do not replace augmented training.
- Use a standard inference adapter for every architecture; output final global variant ID and path diagnostics.
- In PyTorch CUDA timing, synchronize before/after wall-clock measurements; do not benchmark asynchronous launch overhead accidentally.
- Measure sustained FPS from elapsed processing time; do not equate \`1/median_latency\` with observed throughput.
- Distinguish total stored-model footprint from per-image executed computation.
- Add tests for label mapping, singleton shortcuts, hard vs oracle routing, dataset split disjointness, save/load predictions and unchanged frozen weights.

## Execution order

Implement dataset and tests first; then FLAT baseline; then IND-2 and FT-FROZEN-2; then common benchmarking/results; only afterward add 3-stage ablations, raw ImageNet-frozen variants, and incremental updates. Keep each stage runnable and documented.

## Safety and repository hygiene

- Do **not** commit FGVC-Aircraft images, downloaded archives, generated embeddings or trained weights.
- Respect non-commercial research and attribution conditions for the dataset.
- Never alter test data or silently substitute a subset for the official benchmark.
- Do not overwrite published results when rerunning; use immutable run IDs.
- If a design or result conflicts with the protocol, document the deviation rather than hiding it.
- Keep \`README.md\` commands accurate; mark planned commands explicitly until implemented.
- Prefer focused PRs after the initial documentation, with unit-test output and exact reproducibility steps.
