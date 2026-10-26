# Phase 4 paused progress — 2026-10-27 JST

Paused at the user's request. Training and its worker processes are stopped.
Implementation is saved locally on `codex/phase4-independent-baselines`; the
final publication and Git push are pending the next user command.

## Completed work

Independent full EfficientNet node training, IND-2/IND-3 node plans, checkpoint
contracts/reloads, predicted/oracle validation, all-resident / CPU-preloaded /
cold-disk adapters, residency probes, audited report generation, review plots,
documentation and Phase 4 CI are implemented. IND-3 real-data measurements remain
gated behind the main two-stage shared comparison, as required by AGENTS.md.

The full CPU suite passed **213 tests**; the separate Phase 4 run passed **10
tests**. Frozen baseline checks (172 tests) and CPU/CUDA `pip check` passed.
Rerun the full suite after resuming before final publication. The latest report
changes also passed a genuine two-seed combined audit/publication smoke check.

Both completed seeds have 18 full CNNs, exact checkpoint reload checks, and
complete predicted/oracle evaluations on all 3,333 official validation images.
The two-seed publication audit verifies matching training/evaluation source,
configuration, hardware/runtime, data/preprocessing/taxonomy and ImageNet feature
initialization. Test inference has not been performed.

| Seed | Local run directory under `runs/independent/` | State | Hard-route variant macro recall | Oracle diagnostic |
| --- | --- | --- | --- | --- |
| 42 | `ind-s42-20261026T125759Z-6d87251e` | Complete | 56.40% | 66.73% |
| 43 | `ind-s43-20261026T141824Z-b8c12b2c` | Complete | 54.85% | 65.19% |
| 44 | `ind-s44-20261026T153237Z-2b462c75` | Incomplete; 0/18 nodes complete | Unmeasured | Unmeasured |

Seed 44's manufacturer router completed seven epochs before the pause. Per-epoch
metrics remain on disk, but its in-memory selected weights/optimizer were not
exported. Partial-node optimizer resume is not implemented: this node must
restart in a new immutable attempt. Completed seeds 42/43 do not need retraining.
Any partially processed epoch after epoch seven is unmeasured.

## Resume precisely

Do not alter `src/` before resuming: source/configuration/environment/data hashes
are checked strictly. The command below retries only the incomplete seed 44 run,
preserves previous attempts, and appends to the existing console capture:

```powershell
& .\.venv-cuda\Scripts\python.exe -u -m cls_balance.train_independent --config configs/independent.yaml --resume runs/independent/ind-s44-20261026T153237Z-2b462c75 *>> .tmp-phase4-training-v2.log
```

The configuration still contains all three seeds, but `--resume` selects seed 44
from the saved run. Running without `--resume` would unnecessarily retrain all
three seeds. All dataset bytes are rechecked before fitting.

## Remaining steps

1. Finish seed 44, including its complete hard/oracle validation and saved-run audit.
2. Run the final CPU suite and retain the exact artifacts used by the validation publisher:

   ```powershell
   & .\.venv\Scripts\python.exe -m pytest -q tests checks --junitxml=.tmp-phase4-final-junit.xml *> .tmp-phase4-final-pytest.log
   ```

3. After GPU training stops, measure the three residency policies on seed 42:

   ```powershell
   & .\.venv-cuda\Scripts\python.exe -m cls_balance.residency --run runs/independent/ind-s42-20261026T125759Z-6d87251e --depth 2 --device cuda:0 --output-root runs/residency
   ```

4. Publish all three genuine runs:

   ```powershell
   & .\.venv-cuda\Scripts\python.exe -m cls_balance.report_independent --runs runs/independent/ind-s42-20261026T125759Z-6d87251e runs/independent/ind-s43-20261026T141824Z-b8c12b2c runs/independent/ind-s44-20261026T153237Z-2b462c75 --output-root results/stage4
   ```

5. Update README/roadmap/experiment status with genuine evidence and links.
   Run `.tmp/publish_validation.py --report <report-dir> --residency <probe-dir>`
   using the CUDA Python environment to package checks, normalized LF console
   captures, residency plots and source hashes. **Extend that helper first to
   document this user-requested seed-44 interruption**, including seven completed
   epochs, no complete node, unmeasured partial epoch and excluded failed-attempt
   costs. It currently documents only the earlier serial-decoding interruption.
6. Render/review actual final figures, verify publication hashes and staged Git
   blob hashes, complete documentation checkboxes accurately, commit, and push.
   Keep all images, archives, checkpoints, embeddings and training-ID contracts
   outside Git. Mark IND-3 measurement and test/latency endpoints as deferred.

## Local evidence and helpers

- Training console: `.tmp-phase4-training-v2.log` (preserved; append on resume).
- Passing full-suite evidence: `.tmp-phase4-pytest-v2.log`, `.tmp-phase4-junit-v2.xml`.
- Passing Phase 4 checks: `.tmp-phase4-checks.log`, `.tmp-phase4-checks.xml`.
- Audited two-seed review: `runs/review-phase4/independent-20261026T153404Z-3f05e081/REPORT.md`.
- Actual seed-42 visual review: `.tmp/review-s42/`.
- `.tmp/plot_smoke.py --source <actual-report-dir> --output <new-preview-dir>`
  regenerates SVG/PNG previews from a review bundle. **Always supply the actual
  report with `--source`**; its default uses generated unit-test fixtures, which
  must never enter a measured publication.
- `.tmp/status.py` reads the console and on-disk completion markers. On resume,
  the repeated run-start line may cause duplicate seed-44 status lines; deduping
  that helper is optional and does not affect training.
- Earlier interrupted run: `runs/independent/ind-s42-20261026T125510Z-3fa0423c`.
  One serial-decoding epoch completed before restart. Its scientific epoch-one
  observations exactly match the parallel-decoding restart; wall times differ.
  Preserve and disclose the attempt rather than reporting it as a finished model.

All local runs, checkpoints, images and temporary helpers remain ignored by Git.
No background automation or scheduled restart was created.
