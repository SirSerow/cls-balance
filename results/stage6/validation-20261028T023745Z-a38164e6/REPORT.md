# Phase 6 validation and reproduction evidence

The full CPU suite passed **270 tests**, with the CUDA-only FP16 check
explicitly skipped when CUDA is unavailable. The focused CUDA suite passed
**36 Phase 6 checks**, including a real synchronized FP16 pipeline
and saved-output replay on generated test images. Synthetic check timings are
never experimental results. The original 172-test suite freeze is unchanged.

[Measured test results, graphs and complete network schematics](../benchmark-20261028T022812Z-fa52c222/REPORT.md).
The publication auditor verified 9 complete benchmark
runs and 108 hierarchical classifier annotations.
All official test images appear in every one of five timed passes per checkpoint.
Exact per-image global predictions and paths agree across passes; model state is
unchanged. Shared encoder sources match the seed-paired FLAT checkpoints.
All 10,000 dataset images were rechecked against their original preparation
checksums after timing; the hashes and counts match the benchmark contracts.
An artifact-only rebuild preserved all 189 benchmark files byte for byte,
retained all 149,985 timings, and produced 23 valid SVGs with all 100 global
variant labels present. All report links exist. Four network schematics remain
byte-identical; accuracy/latency seed-label placement was improved after visual
review. No model weights or dataset images were needed for that rebuild.

The published suite/matrix bind the measured checkpoints before test inference.
`benchmark.log` is the actual complete execution log. `publication_audit.json`
records independent recomputation of saved classification/timing evidence.
`source_hashes.json` binds checked code; per-run inference source hashes bind the
measured implementation. No dataset pixels, training IDs, embeddings or checkpoint
tensors are included.

An early incomplete timing attempt was stopped because the CPU regression suite
was still active. It contributed no completed pass or published measurement.
Its discarded-attempt record is retained. Two preflight-only development errors
(shared encoder provenance layout and independent resource-file layout) were
fixed before the successful suite froze its matrix; they produced no test results.
Timed passes used no competing regression/plotting workload. OS/driver
clocks remained unlocked and desktop background activity remains a limitation.

An initial report audit caught omitted independent specialist histories, whose selected checkpoints live under nested attempt IDs. The final report follows exact deployment-bundle paths and checks all 108 selected hierarchical classifiers. This reporting correction did not change any measured predictions or timings. A regression check covers nested attempts, and completion metadata is now written only after the final publication audit passes.


```powershell
.venv/Scripts/python.exe scripts/freeze_tests.py --check
.venv/Scripts/python.exe -m pytest -q tests checks
.venv-cuda/Scripts/python.exe -m pytest -q checks/test_phase6.py
.venv-cuda/Scripts/python.exe -m cls_balance.benchmark --config configs/benchmark.yaml
.venv-cuda/Scripts/python.exe scripts/audit_phase6_data.py --config configs/benchmark.yaml --output .tmp-phase6-data-audit.json
.venv-cuda/Scripts/python.exe -m cls_balance.report_benchmark --suite runs/benchmark/benchmark-suite-20261028T002829Z-095436bf --output-root results/stage6 --preview-root .tmp/phase6-previews
.venv-cuda/Scripts/python.exe -m cls_balance.report_benchmark --audit results/stage6/benchmark-20261028T022812Z-fa52c222
.venv-cuda/Scripts/python.exe -m cls_balance.report_benchmark --rebuild results/stage6/benchmark-20261028T022812Z-fa52c222 --output-root .tmp/rebuilt-phase6
```

Use your newly reproduced local run IDs in `configs/benchmark.yaml`; never rerun
into an existing result directory. Final CPU/CUDA checks and visual review occur
after timing, so they cannot inflate measured latency. FP16 experimental results,
generic features, three-stage ablations and introduced-class updates remain
unmeasured. Full plotting rebuild requires only the published result bundle and
the pinned plotting dependency, not data images or trained weights.
