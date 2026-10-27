# Phase 4 execution record — 2026-10-27

The independent cascade implementation and three-seed IND-2 validation are
complete on branch codex/phase4-independent-baselines. All 54 full CNNs passed
exact selected-checkpoint reload checks. Every seed was evaluated on the same
3,333 official validation images with both predicted and oracle routing.

| Seed | Run ID | Hard-route variant macro recall | Oracle diagnostic |
| --- | --- | --- | --- |
| 42 | ind-s42-20261026T125759Z-6d87251e | 56.40% | 66.73% |
| 43 | ind-s43-20261026T141824Z-b8c12b2c | 54.85% | 65.19% |
| 44 | ind-s44-20261026T153237Z-2b462c75 | 57.13% | 67.74% |

IND-2 validation macro recall is 56.13% ± 1.17 percentage points (sample SD).
See the [measured report](../results/stage4/independent-20261027T071605Z-64675e2e/REPORT.md)
for histories, per-image predictions, confusion matrices, per-class recalls,
routing penalties, bypass rates, executed CNN counts and artifact references.
Each standalone deployment stores 18 CNNs, 72,285,741 parameters and 294.43 MB.

The [validation publication](../results/stage4/validation-20261027T071705Z-2b712381/REPORT.md)
contains the final 213-test regression result (155.38 seconds, no failures or
skips), frozen-baseline audit, environment checks, raw console evidence and
CUDA residency diagnostics. The 30 identical validation inputs per policy
produced identical predictions and paths under all-resident, preloaded-CPU and
cold-disk policies. These are synchronized, un-warmed ready-tensor probes;
CUDA runtime startup is excluded. They are not test p95/FPS measurements.

## Interruptions and reproducibility

- An initial serial-decoding attempt completed one router epoch before restart.
  Epoch-one loss, accuracy, support and optimizer-step observations exactly
  reproduced after parallel decoding. Unmeasured partial work is disclosed.
- The user requested a pause after seven seed-44 router epochs. Those metrics
  remained local; its incomplete node restarted on explicit continuation. All
  seven scientific epoch observations reproduced exactly, excluding timings.
- Windows slept for 22,371.2739459 seconds during seed-44 Airbus validation.
  Its incomplete 20-epoch attempt was retained and discarded from successful-run
  costs. Only that node restarted; all 38 completed CNNs were preserved.
- The first auxiliary sleep guard lacked the Windows multiprocessing import
  guard. Two zero-epoch attempts were retained. A corrected main guard passed a
  separate spawn smoke check, followed by successful real DataLoader training.
  Its temporary idle-sleep request ended with the training process and changed
  no persistent power plan, display setting or clock policy.
- The first CUDA residency probe failed before recording measurements because
  allocator initialization had not preceded peak-stat reset. A released sentinel
  allocation now initializes the allocator before policy timers. The corrected
  probe and final regression suite passed; the failed-start log is retained.

Successful-run resources exclude all discarded attempts. Sleep-contaminated
wall time is preserved as raw wall time, not described as active GPU time.
No partial-node optimizer resume is claimed. Initial run metadata records Git
base 1af6c0f with dirty source snapshots; resumption followed local commit
46a5b00. All fitting/evaluation module hashes, dataset/taxonomy/preprocessing
contracts, ImageNet initialization and hardware/runtime contracts matched.
The resumed evaluation Git SHA is recorded separately in the validation evidence.

## Scope and local artifacts

Training images, archives, embeddings, weights and contracts with training-image
ID lists remain in ignored local storage. Published validation CSVs and aggregate
histories contain genuine measurements. The full recipe is in
[PHASE4.md](PHASE4.md) and the measured report.

IND-3 is implemented and integration-tested; its real-data ablation follows the
main two-stage shared comparison and common benchmark. Test inference, shared
architectures, full-pipeline latency/FPS and incremental updates remain deferred.
No automatic restart, background monitor or scheduled follow-up was created.
