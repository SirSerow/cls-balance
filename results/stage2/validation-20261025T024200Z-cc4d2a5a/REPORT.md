# Phase-two validation — validation-20261025T024200Z-cc4d2a5a

This report records software validation, not trained aircraft-classification
results. Phase 2 implements common inference/evaluation interfaces; no models
were trained, and no experimental accuracy, latency or throughput is reported.
The experimental comparison tables in `docs/RESULTS_TEMPLATE.md` remain unmeasured.

All **172 CPU tests passed**, with zero failures, errors or skips. The final suite
finished in 9.409 seconds on the recorded Windows/Python 3.12 CPU environment.
See [pytest output](pytest.txt), [JUnit evidence](junit.xml),
[machine-readable validation](validation.json), and [environment](environment.json).
The editable install and installed evaluation entry point passed checks;
[pip check](pip-check.txt) found no broken requirements.

The tests cover FLAT, two-/three-stage independent and shared adapters; explicit
local-to-global maps; selected-branch-only execution; singleton shortcuts;
hard/oracle separation and wrong routes; exact macro recall/top-1/macro-F1
(including an independent scikit-learn reference); first wrong stage and
conditional family denominators; frozen encoder parameters/buffers;
complete-split evaluation contracts; leakage/version mismatch rejection;
immutable prediction/run round trips; artifact corruption; and the test freeze.
All image fixtures are generated during testing and are not committed.

The [official-data preflight](official_data_preflight.json) also passed on the
existing complete FGVC-Aircraft metadata: 3334 train, 3333 validation and 3333 test
images; 100 variants, 70 families and 30 manufacturers; zero split-ID overlap.
It exercised evaluator construction/provenance guards without model inference.
No official image, per-image label manifest, archive, embedding or weight is
included in this publication.

After the final successful test run, [tests/FROZEN.json](../../../tests/FROZEN.json)
froze nine test/fixture source files plus three runner/configuration files.
The [read-only verification](freeze-check.txt) passed. CI now runs that check
before executing tests. The freeze script refuses to overwrite a baseline;
intentional future test changes require explicit review and new passing evidence.
No test source or frozen runner input was edited after this freeze.

The working-tree implementation was validated against base commit
`6724461b87f382ab2c650e9a739f26399ccb2c6d` before the implementation/evidence commit. Its exact file bytes
are recorded in [source_hashes.json](source_hashes.json); the tree hash is
`194842c8a3ec8831312fb5a421cc316ed4d7e1431ed360bc3f7ee16a41db0939`. The environment records the dirty working tree
explicitly, so the base SHA alone is not presented as the implemented revision.
The commit containing this report also contains those sources and frozen tests.

To reproduce the checks without changing the frozen baseline:

```powershell
.venv/Scripts/python.exe -m pip check
.venv/Scripts/python.exe scripts/freeze_tests.py --check
.venv/Scripts/python.exe -m pytest -q --junitxml=.tmp/phase2-recheck.xml
.venv/Scripts/cls-balance-evaluate.exe --help
```

Saved evaluation runs can be audited separately with
`python -m cls_balance.evaluate --run-dir <existing-eval-run-directory>`.
See [the operational API](../../../docs/PHASE2.md) for evaluation construction,
provenance contracts, score units and diagnostic boundaries. Reproduction must
write new evidence to a new directory; do not overwrite this publication.

Limitations: validation used CPU and deterministic injected modules. CUDA,
trained EfficientNet checkpoints and latency benchmarking belong to later phases.
Remote CI was not yet observed when this report was written. Model training
provenance remains a declaration that is checked for split consistency; hashes
cannot independently prove its history. No scientific protocol deviation is
introduced by this tooling phase.
