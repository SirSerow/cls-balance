# Phase 7 validation and reproduction evidence

The complete CPU suite passed **305 tests**. The CUDA-focused
suite passed **73 checks**, including actual EfficientNet
class introduction/reload on generated pixels. The plotting suite passed
**4 checks**, including artifact-only byte-stable regeneration.
Synthetic checks are never experimental performance. The original 172-test
freeze remains verified.

[Measured specialist-update results and complete networks](../incremental-20261029T102022Z-dd4acc20/REPORT.md).
All three seeds have fresh initial 99-class FLAT sources and independent/shared
systems, followed by only the affected specialist update. Every before/after,
predicted/oracle evaluation retains all 3,333 official test images. The published
review covers 6 seed/architecture pairs.
The semantic audit checks exact supervised-ID exclusion, seed/source/config/data
hashes, encoder ancestry, checkpoint states, global logit maps, local validation
selection, frozen weights/buffers and unchanged unrelated logits. Router decisions
and predictions routed outside the affected branch are unchanged.

All 10,000 source images were rechecked against the original preparation after
the experiment. The public review and all its figures/reports were rebuilt
without images, supervised IDs, caches or trained weights; per-file comparisons
are in `rebuild-audit.json`. The execution logs and nested evaluation summaries
provide measurement lineage. No training image IDs or dataset pixels are copied.

An initial preflight attempt failed at Windows worker creation before completing
any epoch. It is explicitly discarded. Normal module import fixed the spawn
contract. A protocol review before the successful experiment also tightened class
selection to preserve the held-out class's existing family; the successful
training-only rule selects 737-200. No test results informed this selection.

Initial seed-42 backbone training overlapped CPU regression/report checks, which
can affect its initial wall-time accounting. Specialist update timings and test
inference ran without competing foreground test/plot workloads. Clocks remained
unlocked; desktop background activity remains a limitation. Update duration is
not an inference-latency or throughput measurement.

```powershell
.venv/Scripts/python.exe scripts/freeze_tests.py --check
.venv/Scripts/python.exe -m pytest -q tests checks
.venv-cuda/Scripts/python.exe -m pytest -q checks/test_phase6.py checks/test_phase7.py checks/test_phase7_report.py
.venv-cuda/Scripts/python.exe scripts/train_incremental_awake.py train --config configs/incremental.yaml
.venv-cuda/Scripts/python.exe scripts/train_incremental_awake.py evaluate --suite S:/Projects/cls-balance/runs/incremental/incremental-20261029T040144Z-860727e6
.venv-cuda/Scripts/python.exe -m cls_balance.report_incremental --training S:/Projects/cls-balance/runs/incremental/incremental-20261029T040144Z-860727e6 --evaluation S:/Projects/cls-balance/runs/incremental/incremental-evaluation-20261029T093143Z-069c1926 --output-root results/stage7 --preview-root .tmp/phase7-previews
.venv-cuda/Scripts/python.exe -m cls_balance.report_incremental --audit S:/Projects/cls-balance/results/stage7/incremental-20261029T102022Z-dd4acc20
.venv-cuda/Scripts/python.exe -m cls_balance.report_incremental --rebuild S:/Projects/cls-balance/results/stage7/incremental-20261029T102022Z-dd4acc20 --rebuild-output .tmp/rebuilt-phase7
.venv-cuda/Scripts/python.exe scripts/audit_phase7_data.py --training S:/Projects/cls-balance/runs/incremental/incremental-20261029T040144Z-860727e6 --output .tmp-phase7-data-audit.json
```

Use your reproduced immutable run IDs; never overwrite an existing result.
Phase 8 broader analysis, generic ImageNet-frozen and three-stage measurements
remain separate work.
