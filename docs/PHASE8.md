# Phase 8 — analysis and publication

The primary matrix is the completed FLAT / IND-2 / FT-FROZEN-2 study for seeds
42, 43 and 44. Phase 8 consumes immutable Phase 6 and 7 publications and performs
no retraining, model selection, image loading or inference. The broader generic
ImageNet-frozen, three-stage and FP16 ablations remain explicitly unmeasured.

## Frozen sources and statistical contract

`configs/analysis.yaml` names the two source publications and binds their manifest
SHA256 values. Source audits recompute every test metric from predictions and
every timing summary from retained observations, verify hardware/preprocessing
compatibility, and check the full official dataset edition. Analysis additionally
requires the exact primary architecture/seed matrix and identical test IDs/truth.
The incremental study remains a separate reduced-label experiment.

For each fixed seed pair, the bootstrap samples image IDs with replacement
within each true variant, preserving every class's test support. Both models use
the same sampled IDs, and the statistic averages class recall differences with
equal class weights. The percentile 95% interval uses 10,000 replicates and RNG
seed 8008. Differences are FT-FROZEN-2 minus IND-2 in percentage points.

Per-seed intervals are reported individually. Sample SD across the three
observed differences describes training-seed variation. A separate descriptive
interval for the mean of these fixed pairs uses identical sampled image IDs
across seeds to preserve image dependence. It is conditional on the fixed
checkpoints and is **not** a population-of-training-seeds interval, a formal
noninferiority test, or a reason to tune the models on test data.

Each seed's latency percentile pools all five complete test passes; the main
table gives mean ± sample SD of these seed percentiles. Per-pass min/max/SD and
all 45 pass summaries describe repeatability. Five passes on one interactive
laptop do not establish a latency population confidence interval. Sustained FPS
uses observed elapsed processing time. No reciprocal-latency throughput is used.

The Pareto frontier minimizes mean seed-p95 and maximizes mean seed macro recall
over the three measured architectures. Ties remain on the frontier. Training
costs for shared systems include their source FLAT fit; these are cumulative
fitting/validation/selection wall times, not independently instrumented GPU
activity or an equal-compute comparison.

## Commands

From the repository root, with the locked environment and plotting dependency:

```powershell
.venv-cuda/Scripts/python.exe -m pip install -r requirements/phase3-plots.txt
.venv-cuda/Scripts/python.exe -m cls_balance.report_analysis --config configs/analysis.yaml --preview-root .tmp/phase8-previews
.venv-cuda/Scripts/python.exe -m cls_balance.report_analysis --audit <phase8-review-directory>
.venv-cuda/Scripts/python.exe -m cls_balance.report_analysis --rebuild <phase8-review-directory> --output .tmp/rebuilt-phase8
.venv/Scripts/python.exe scripts/freeze_tests.py --check
.venv/Scripts/python.exe -m pytest -q tests checks
.venv-cuda/Scripts/python.exe -m pytest -q checks/test_phase8.py
```

The first command creates a unique `results/stage8/analysis-<UTC>-<random>/`
publication. `--output` can specify a new directory; existing directories are
never overwritten. `--repo` locates the repository if running elsewhere. Paths
inside the configuration resolve relative to that repository. Previews remain
ignored and never contain aircraft pixels. The full study reproduction sequence
is in the README; no dataset or model weights are needed to regenerate this
analysis and its figures from committed evidence.

## Published artifacts and audit

- `analysis.json`: all per-seed statistics, paired intervals, frontier membership,
  route/depth/pass timing tables, family error counts and measured updates.
- `per_seed.csv`: classification, latency, resource and completed-fit cost table.
- `branch_latency.csv`: predicted-route latency and observation counts per seed.
- `depth_latency.csv`: distinct image counts and repeated observations grouped
  by actual executed backbone count; shared systems always execute one backbone.
- `pass_latency.csv`: all per-pass p50/p95 values, elapsed seconds and FPS.
- `family_error_breakdown.csv`: correct, within-family, cross-family within
  manufacturer, and cross-manufacturer counts; denominator is every test image.
- Five SVGs: accuracy/latency and paired intervals, routing/repeatability,
  family error types, latency by executed depth, and mean row-normalized
  confusion matrices.
- `REPORT.md`: all nine sections of RESULTS_TEMPLATE, with source run IDs,
  checkpoint/config hashes, linked per-image predictions/timings, limitations,
  exclusions and previous measured learning curves/network diagrams.
- `config.resolved.json`, `provenance.json`, `summary.json`: frozen input
  identities, analysis source/environment and SHA256 inventory.

Audits reject missing/extra artifacts, changed source manifests, modified output
bytes and changed statistics even if output checksums have been re-sealed.
They recompute the tables, bootstrap and report from the audited input evidence.
Rebuild regenerates every table, report and SVG and verifies byte equality.
Keep Matplotlib 3.10.7 and the recorded runtime for byte-identical SVG rebuilds;
other rendering versions may produce different bytes despite equal statistics.

`--results-doc docs/RESULTS.md` additionally exports all nine report sections to
the documentation file, rebasing every relative link; it explicitly replaces
that documentation file while leaving all published runs immutable.
`docs/RESULTS.md` is the human entry point to the immutable final report. Existing
Phase 3–7 publications remain unchanged, including historically pending items
in those earlier reports. Dataset images, archives, training-ID lists, embeddings
and checkpoints remain ignored. Results contain test evidence and aggregate
training histories only.
