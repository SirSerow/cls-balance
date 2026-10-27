# Phase 5 validation evidence

The original frozen suite is unchanged. CPU regression output: **235 passing tests**.
After publication/config-audit refinements, the current Phase 5 suite passed **22 checks**.
The CUDA environment also passed **12 focused checks**. These synthetic checks are implementation evidence, not aircraft accuracy measurements.
The final report refinements passed **2 end-to-end publication checks**, covering both backbone types and depths.

All three real FT-FROZEN-2 runs passed nested checkpoint, local selection/exposure,
model-fingerprint and complete predicted/oracle validation audits. Their publication
hashes were independently rechecked. Each run used all official training/validation
images; there was no test inference. No discarded training attempts were observed
in this execution. The process-lifetime Windows idle-sleep guard was spawn-tested;
it did not change power plans or GPU clocks.

Measured report: [shared-head results](../shared-20261027T145809Z-ee48fbad/REPORT.md).
`audit.json` binds run IDs, the report, frozen suite and current source/config hashes.
The report's per-seed records preserve the exact training source hashes and Git
SHA/dirty state; this validation environment describes the later audit.

Commands executed from the repository root:

```powershell
.venv/Scripts/python.exe scripts/freeze_tests.py --check
.venv/Scripts/python.exe -m pytest -q tests checks
.venv/Scripts/python.exe -m pytest -q checks/test_phase5.py
.venv/Scripts/python.exe -m pytest -q checks/test_phase5.py -k train_reload_route_replace_and_publish
.venv-cuda/Scripts/python.exe -m pytest -q checks/test_phase5.py -k 'head_checkpoint or training_cache or frozen_guard or source_encoder or head_dimensions'
.venv-cuda/Scripts/python.exe scripts/train_shared_awake.py --smoke-test
.venv-cuda/Scripts/python.exe -m pip check
.venv-cuda/Scripts/python.exe scripts/train_shared_awake.py --config configs/shared.yaml
.venv-cuda/Scripts/python.exe -m cls_balance.report_shared --runs runs/shared/shared-ft-frozen-s42-20261027T140957Z-b392fc3a runs/shared/shared-ft-frozen-s43-20261027T142909Z-e9730962 runs/shared/shared-ft-frozen-s44-20261027T144302Z-c8142ff8 --output-root results/stage5 --flat-runs S:/Projects/cls-balance/runs/flat/flat-s42-20261025T031340Z-5385e556 S:/Projects/cls-balance/runs/flat/flat-s43-20261025T034152Z-d43e9130 S:/Projects/cls-balance/runs/flat/flat-s44-20261025T041556Z-40c093b2 --independent-runs runs/independent/ind-s42-20261026T125759Z-6d87251e runs/independent/ind-s43-20261026T141824Z-b8c12b2c runs/independent/ind-s44-20261026T153237Z-2b462c75 --preview-root .tmp/phase5-figures
```

Images, archives, embeddings, training-ID lists and weights remain local and ignored.
Only validation predictions and aggregate metrics/curves are published. Generic
ImageNet-frozen and three-stage measurements retain the execution-order gate;
Phase 6 test accuracy, full-pipeline latency/FPS and Phase 7 introduction remain deferred.
