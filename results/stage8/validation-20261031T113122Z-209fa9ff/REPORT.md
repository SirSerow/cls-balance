# Phase 8 validation and reproduction evidence

The complete CPU regression suite passed **324 tests**; explicit
skips and reasons are preserved in [the log](cpu-tests.txt). A final run of the
affected benchmark, incremental-publication and Phase 8 statistics/figure checks
passed **60 checks** in the CUDA/plotting environment, including
the CUDA-only FP16 benchmark integration check. These generated fixtures test
software contracts; they are not experimental performance measurements.
After the final rebuild API refinement, all **20 Phase 8 checks**
also passed again; [the final log](final-statistics-tests.txt) is retained.

[Measured consolidated results](../analysis-20261031T112159Z-ee01689c/REPORT.md).

The original 172-test suite remains unchanged. Phase 8 consumes the existing
three-seed Phase 6/7 measurements without training, image loading or inference.
The source audits replay all primary test predictions/timings and all incremental
before/after predicted/oracle evaluations. They enforce frozen source manifests,
official class/image counts, matching seed/image identities and measurement
contracts. The final analysis recomputes paired stratified intervals, tables and
report text, and rejects changed statistics even with re-sealed checksums.

All 14 final report artifacts rebuilt byte for byte from the
committed source evidence without data or weights. [Rebuild evidence](rebuild-audit.json)
binds the exact final source publication. All five new SVGs were rendered to
local PNG previews and visually reviewed. Original Phase 3–7 publications remain
unchanged. No dataset pixels, training-ID lists, embeddings or tensors were added.

```powershell
.venv/Scripts/python.exe -m pytest -q tests checks
.venv/Scripts/python.exe scripts/freeze_tests.py --check
.venv-cuda/Scripts/python.exe -m pytest -q checks/test_phase6.py checks/test_phase7_report.py checks/test_phase8.py
.venv-cuda/Scripts/python.exe -m cls_balance.report_analysis --config configs/analysis.yaml --preview-root .tmp/phase8-previews --results-doc docs/RESULTS.md
.venv-cuda/Scripts/python.exe -m cls_balance.report_analysis --audit results/stage8/analysis-20261031T112159Z-ee01689c
.venv-cuda/Scripts/python.exe -m cls_balance.report_analysis --rebuild results/stage8/analysis-20261031T112159Z-ee01689c --output .tmp/rebuilt-phase8
```

Use a fresh output directory for every rebuild. Source hashes and environment
are retained alongside the real test logs. Analysis source may have a dirty-tree
flag because this publication was produced before the final result commit; the
recorded source SHA256 values identify the exact files. The bootstrap holds the
trained model pairs fixed and does not estimate future-training-seed uncertainty.
Unmeasured generic frozen, three-stage, FP16 experimental and cold-start results
remain clearly labeled follow-up work.
