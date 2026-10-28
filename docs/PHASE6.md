# Phase 6 — complete test benchmark and review

See the [measured three-seed test report and complete network diagrams](../results/stage6/benchmark-20261028T022812Z-fa52c222/REPORT.md).
FT-FROZEN-2 achieved **63.02% ± 0.93 pp** test macro recall versus
**53.51% ± 1.02 pp** for IND-2. Mean seed-p95 full-pipeline latency is
**34.26 ms versus 57.15 ms**, a **40.05% reduction**, across five complete
official-test passes per checkpoint. FLAT remains a strong baseline at
**65.59% ± 0.37 pp** and **32.47 ms**. All uncertainty shown here is sample SD
across seeds; paired image-bootstrap intervals and limitations are in the report.

[Validation and reproduction evidence](../results/stage6/validation-20261028T023745Z-a38164e6/REPORT.md)
contains the 270-test CPU pass (one CUDA-only skip), 36-test CUDA pass, full
benchmark log, JUnit results, image-byte recheck and artifact-only rebuild proof.

The operational configuration is [benchmark.yaml](../configs/benchmark.yaml).
It fixes the existing validation-selected FLAT, IND-2 and FT-FROZEN-2 checkpoints
for seeds 42/43/44. No training or model selection occurs in this phase.

```powershell
.venv-cuda/Scripts/python.exe -m cls_balance.benchmark --config configs/benchmark.yaml
.venv-cuda/Scripts/python.exe -m cls_balance.report_benchmark --suite <completed-suite-directory> --output-root results/stage6 --preview-root .tmp/phase6-previews
.venv/Scripts/python.exe scripts/freeze_tests.py --check
.venv/Scripts/python.exe -m pytest -q tests checks
```

Use the suite directory printed by the benchmark. Each invocation creates new
immutable IDs; interrupted runs have no completion summary. Run CPU tests before
or after timing, avoiding competing CPU/GPU work during the benchmark. Plotting
uses the existing pinned `requirements/phase3-plots.txt` dependency. Substitute
your reproduced training run IDs on another machine; all YAML paths resolve
relative to the configuration file.

## Frozen inputs and provenance

Before opening test images for inference, validate the complete official dataset
against the preparation run, including every image checksum, split disjointness,
100/70/30 edition and preprocessing/taxonomy/manifest hashes. Audit every training
run and selected node, verify checkpoint bytes and global output maps, and check
training/selection IDs against official train/validation IDs. Shared exports bind
the original encoder source instead of duplicating supervised ID lists: audit that
original FLAT run and its encoder state, seed and parent checkpoint checksums.
When a paired FLAT appears in the comparison, it must be that exact source run.

Freeze the whole checkpoint matrix in `frozen_matrix.json` before test inference.
Each architecture must have exactly the same seed set. Sources, configurations,
environments, checkpoint checksums and original aggregate training costs are
retained. Training resources for FT-FROZEN include a separate source-FLAT record;
include that cost when comparing total training compute.

The benchmark has no subset option. All 3,333 official test images are processed
at batch size one in at least five timed passes. Every pass uses a deterministic
image permutation, identical across architectures/seeds/precision. It changes
between passes. Architecture order rotates across seeds in the supplied YAML.
No labels enter predicted inference; labels are attached only for scoring.
Untimed oracle diagnostics run afterward and remain explicitly labeled.

## Timing and deployment contract

- **Full pipeline:** file open/decode, banner removal, letterbox and normalization,
  dtype conversion and host-to-device transfer, completion synchronization,
  complete predicted-route adapter, final device-to-host/Python global ID, and
  completion synchronization. Normal OS filesystem caching is retained. Image
  files are reopened and processed on every pass; no input/embedding cache.
- **Model path:** nested timer after input-transfer synchronization, through the
  entire adapter, routing and final ID conversion, ending after completion sync.
  Adapter input/logit checks are part of deployed cost. One inference supplies
  both boundaries; the H2D boundary sync is charged to full pipeline.
- **Sustained FPS:** total processed images divided by elapsed continuous timed
  pass time, including per-image timing, synchronization and in-memory record
  collection. Setup, warmup, telemetry, CSV writes and oracle are excluded.
  This is observed sequential processing throughput, never reciprocal latency.

Every timed operation runs under `torch.inference_mode()` with models in eval
mode. CUDA synchronizes before the outer timer, after transfer, and after the
model. All trained nodes are warmed directly using a synthetic zero input,
including specialists not selected by the warmup router; the complete adapter is
also warmed. Warmup uses no labels. Conv2d/Linear profiling hooks are removed
before timing. FP32 is primary; TF32 is disabled and deterministic kernels enabled.

All weights are resident on the inference device, including inactive specialist
weights. Failure/OOM is retained in `failure.json`; no silent fallback, different
subset, changed residency policy or substituted precision. Cold-start, cold OS
file-cache and on-demand deployment latency are unmeasured by this harness;
[residency probes](PHASE4.md) remain separate diagnostics.

Peak GPU allocated and reserved memory covers all timed passes after warmup,
with all model weights resident; oracle is outside that measurement. CPU RSS and
ambient temperature are explicitly unavailable. Capture device/driver/framework
versions and before/after telemetry with clocks, power, temperature and utilization.
The Windows idle-sleep guard is temporary and releases on exit; OS power scheme
and GPU clocks remain unchanged. Desktop activity and unlocked laptop clocks are
limitations, even when the run has no competing regression suite.

Set `precision: fp16` in a **separate YAML invocation** for the optional deployment
ablation. It uses the same source checkpoints and full image path, saves its own
accuracy and timing, and never overwrites the FP32 run. FP16 is CUDA-only.
Do not combine FP32 and FP16 runs into one primary report.

## Saved observations and auditing

Each immutable child run retains predicted and oracle test CSVs, classwise metrics,
configuration, dataset/preprocessing/taxonomy contracts, environments, source and
checkpoint hashes, per-pass elapsed times/telemetry, resources and timings.
`latency_test.csv` includes every image/pass, predicted route, classifier count,
backbone count, bypass count, conditional MAC estimate and both time boundaries.
Repeated passes must reproduce each global prediction and full path exactly.
Model-state fingerprints must remain unchanged.

```powershell
.venv/Scripts/python.exe -m cls_balance.benchmark --audit <completed-child-run>
```

The auditor verifies exact inventories and hashes, complete unique image/pass
coverage, predictions/paths, MACs and nested timer ordering, then recomputes
classification, routing, per-pass/per-route distributions and elapsed-time FPS.
All latency quantiles use NumPy's linear interpolation. Within-seed quantiles pool
all image/pass observations; across seeds, report mean ± sample standard deviation
of those per-seed statistics. Retain pass distributions to expose drift.

MAC estimates count one Conv2d/Linear multiply-accumulate as one MAC, with grouped,
depthwise and squeeze/excitation convolutions. Exclude BN, activations, pooling,
elementwise operations, routing and transfers. Attribute MACs only to executed
nodes; every stored node still counts toward checkpoint bytes and parameters.

## Publication and graphs

Reporting copies a strict allowlist of benchmark results, then audits the copies.
No training IDs, aircraft pixels, tensors or embedding caches enter publication.
`publication.json` binds every published artifact by SHA256. Reports follow the
[results template](RESULTS_TEMPLATE.md), marking unmeasured experiments explicitly.

Figures cover accuracy/latency and the observed Pareto frontier, full timing CDFs,
pass drift, every observed predicted branch, nested timing means, sustained FPS,
storage/memory, routing penalty/error stage, class recalls and nine confusion
matrices. Every trained router and specialist has training/validation recall and
loss curves, with its selected checkpoint marked. Full SVG network diagrams show
all 17 specialists, all 13 singleton branches, all 100 global output labels,
EfficientNet stage dimensions and selected specialist epochs for all three seeds.
JSON inventories expose the exact trained node set. FLAT output grouping is
display-only and does not imply manufacturer routing.

The B0 detail figure expands every stage and explains MBConv expansion,
depthwise convolution, squeeze/excitation, projection and residual connections.
Its JSON companion lists every actual Torchvision module, convolution kernel,
stride, group and parameter count without exporting parameter values. Stage
specifications are verified against the installed implementation.

```powershell
# Audit committed evidence or regenerate all figures without data/checkpoints:
.venv-cuda/Scripts/python.exe -m cls_balance.report_benchmark --audit <published-review-directory>
.venv-cuda/Scripts/python.exe -m cls_balance.report_benchmark --rebuild <published-review-directory> --output-root .tmp/rebuilt-phase6
```

Paired accuracy uncertainty is an image-paired stratified bootstrap: sample within
every variant, use the same sampled IDs for both architectures and every fixed
seed, and average class/seed differences. Use 2,000 replicates, RNG seed 6006,
percentile 95% intervals. This captures test-image uncertainty conditional on the
three trained seeds, not population-of-training-seeds uncertainty. The exploratory
2 pp / 30% thresholds are descriptive; no formal noninferiority test is claimed.

Phase 7 introduced-class updates, generic ImageNet-frozen and three-stage
measurements, FP16 measurements and broader Phase 8 publication remain separate.
