# Phase 3 — validation and reproducibility evidence

**203 CPU tests passed**: the original frozen 172 tests plus 31 phase-three checks.
There were no failures, errors or skips. Both the CPU and CUDA environment dependency
checks passed. The original test freeze and its validation evidence are unchanged.

The FLAT baseline was also trained on every official training image for seeds
42/43/44. Each selected checkpoint was evaluated on all 3,334 train and 3,333
validation images. All GPU reload checks and independent CPU export audits passed.
See the [measured results and figures](../flat-20261025T045223Z-788dc5f4/REPORT.md).

## Checks and artifacts

- [JUnit](junit.xml) and [test output](pytest.txt): 203 passed in 48.88 seconds.
- [Frozen-suite integrity](freeze-check.txt): 12 frozen inputs, original 172-test validation verified.
- [CPU dependencies](pip-check-cpu.txt) and [CUDA dependencies](pip-check-cuda.txt).
- [Independent export verification](export-verification.json): all 15 recorded source files for each training run match commit `aa18ef15283ba9a7abcec22ac213c1f842717966`; full and encoder checkpoints reload exactly on the training device; CPU encoder output exactly matches pooled FLAT features; frozen parameters/buffers remain unchanged and all encoder modules stay in eval mode.
- [Training console](training-console.txt), [validation metadata](validation.json), [environment](environment.json) and [source hashes](source_hashes.json).

The source snapshots were recorded before training each seed. Uncommitted test and
documentation work explains later dirty flags; training sources match the recorded
commit. The report writer was subsequently hardened for LF SVG output and clearer
legends; report-generation sources have separate hashes in its publication metadata.
SVGs and all other published artifact hashes were checked, and the final learning,
generalization, classwise and confusion figures were visually inspected. Text
outputs use LF so checksums survive Git checkouts on Windows and Unix.

## Reproduce the checks

```powershell
.venv/Scripts/python.exe scripts/freeze_tests.py --check
.venv/Scripts/python.exe -m pytest -q tests checks
.venv/Scripts/python.exe -m pip check
.venv-cuda/Scripts/python.exe -m pip check
.venv/Scripts/python.exe results/stage3/validation-20261025T042101Z-bcc17404/verify_exports.py --runs runs/flat/flat-s42-20261025T031340Z-5385e556 runs/flat/flat-s43-20261025T034152Z-d43e9130 runs/flat/flat-s44-20261025T041556Z-40c093b2
```

Use your own completed run directories when reproducing training. Checkpoints stay
local; the standalone export audit requires them. Published validation evaluations
can be audited without weights or dataset files using
`python -m cls_balance.evaluate --run-dir <report>/seed-42/validation` (and similarly
for seeds 43/44). The zero input used by `verify_exports.py` checks artifact
equivalence and is not an aircraft accuracy measurement.

Dataset images, archives, training-image ID lists, embeddings and weights are not
published. Official test inference and latency measurement remain deferred by the
protocol. The measured train/validation results do not establish the main
hierarchical comparison.
