# Phase 4 — Independent hierarchical baselines

`python -m cls_balance.train_independent --config configs/independent.yaml` trains
independent full EfficientNet-B0 nodes, initialized from the same published
`EfficientNet_B0_Weights.IMAGENET1K_V1` features. Each node owns its own parameters,
buffers, optimizer, validation selection and checkpoint. There is no embedding
cache or shared backbone. The existing Phase 2 inference adapters enforce storage
independence and explicit global label maps.

## Training and branch contracts

The operational YAML uses official train/validation data, seeds 42/43/44, FP32,
AdamW at 1e-4, 30 epochs and patience 7. All paths resolve against the YAML.
Data preparation, all image checksums and every official split are validated
before training. Test pixels and labels never enter node fitting or selection.
Node filters join the canonical image-ID taxonomy; output columns are explicitly
mapped from local indices to global manufacturer/family/variant IDs.

`depths: [2]` trains a full manufacturer router and one full variant CNN for every
manufacturer with multiple variants. This edition requires **18 CNNs**. A
single-variant manufacturer returns that variant deterministically.

`depths: [3]` trains manufacturer → conditional family → conditional variant
nodes, requiring **33 CNNs** in this edition. Single-family manufacturers and
single-variant families bypass their classifiers. `depths: [2, 3]` produces both
bundles and reuses the selected manufacturer router **across architectures within
the same seed**; CNNs within each cascade remain independent. This avoids
retraining the identical supervised task and is declared in run metadata. Each
architecture's resource totals include its router; the unique experiment total
must count it only once.

The default remains IND-2: AGENTS.md and the roadmap require the main two-stage
shared comparison and common benchmark before three-stage ablations. IND-3 is
implemented and integration-tested, while its genuine measurements are deferred
until that gate. Do not describe its generated-fixture tests as aircraft results.

Each epoch visits every image in the supervised branch, with the same augmentation
and deterministic `(seed, epoch, image_id)` policy as FLAT. Local macro recall
uses the complete fixed branch class list. Strict improvements select a new
checkpoint; ties retain the earlier epoch. No test-set hyperparameter choices,
balanced sampling, mixed precision or learning-rate scheduler are introduced.
Configured worker processes are used for branches with at least 256 train
images; smaller branches use zero workers to avoid Windows startup costs. Actual
worker count is logged per node. The scientific sample/augmentation contract
does not depend on worker count.

## Saved runs, retries and audits

Runs use immutable IDs under ignored `runs/independent/`. Each node attempt saves:

- A contract with explicit local/global IDs, exact train/selection ID lists,
  seed, configuration/source/data/preprocessing/taxonomy hashes.
- Per-epoch JSON and `history.csv`, selected checkpoint and adjacent SHA256
  manifest, resource accounting and a completion summary.
- Exact selected/reloaded logits on a deterministic validation input, checked on
  the training device before marking the node complete.

Each full run saves environment/driver/device/power/dtype metadata, preprocessing,
taxonomy, dataset manifest hashes, source hashes and one deployment bundle per
depth. Training keeps one node and its optimizer on the device at a time; the
run environment's all-resident policy describes validation/deployment. Both
predicted-route and separately named oracle evaluations run on **all
3,333 validation images** and save per-image decisions and metrics. IND-3 also
collects conditional family accuracy under the true manufacturer in a separate
diagnostic pass. No test inference is performed. The final run summary appears
only after all nodes, reloads and evaluations succeed.

An interrupted node is never continued from a half-completed optimizer state.
To retry an incomplete run, use the same configuration and source/environment:

```powershell
& .\.venv-cuda\Scripts\python.exe -m cls_balance.train_independent --config configs/independent.yaml --resume runs/independent/<incomplete-run-id>
```

Verified completed nodes are reused; incomplete attempts remain intact and are
retrained under a new attempt ID. Source, configuration, package, device, data or
preprocessing changes reject reuse. A complete run cannot be resumed/overwritten.
An unchanged retry may create another immutable final evaluation; it never edits
previous evaluation evidence.

```powershell
& .\.venv-cuda\Scripts\python.exe -m cls_balance.report_independent --runs runs/independent/<run-id> --audit-only
& .\.venv-cuda\Scripts\python.exe -m pip install -r requirements/phase3-plots.txt
& .\.venv-cuda\Scripts\python.exe -m cls_balance.report_independent --runs runs/independent/<seed42-run> runs/independent/<seed43-run> runs/independent/<seed44-run> --output-root results/stage4
```

The report audits nested checkpoint bytes, node contracts and saved validation
predictions, rejects incompatible seed bundles, and generates graphs directly
from measured histories and per-image results. Review figures show router
learning/loss, node validation scores and costs, hard vs oracle penalties,
first-wrong-stage counts, executed CNN passes, class recalls and confusion
matrices. A code-native architecture image explains the prediction path. The
report follows the results template while explicitly marking test/latency/shared
comparisons and uncertainty as unmeasured. Single-seed reports do not claim seed
variance; multi-seed scores use sample standard deviation.

Published bundles contain validation predictions, aggregate training histories
and costs, source/configuration hashes, and model checksum references. Contracts
with training image lists, dataset images, checkpoints and embeddings remain
local. Reports are immutable; rerunning creates a new report directory. Figures
are SVG with LF line endings so artifact hashes survive Git checkouts.

## Model residency

`load_cascade(run, depth)` constructs the standard all-resident adapter;
`cascade_with_policy(run, depth, policy)` also supports:

| Policy | Before prediction | Inside each selected node | After selected node |
| --- | --- | --- | --- |
| `all-resident` | All complete CNNs on target device | Execute selected CNN | All remain resident |
| `preloaded-cpu` | All complete CNNs held on CPU | Transfer selected CNN to input device and execute | Move it back to CPU; synchronize |
| `cold-disk` | Paths and checkpoint hashes; no cached CNNs | Verify/load checkpoint, transfer and execute | Evict CNN; synchronize |

Cold-disk does **not** flush the OS filesystem cache or CUDA allocator. Hash and
checkpoint checks are included in disk loading overhead. Preloaded-CPU cache
weights are deliberately excluded from `.to(device)` recursion and identified
by checkpoint digest buffers. Use fresh adapters for different policies; treat
all policies as single-threaded inference objects. These policies do not alter
predicted/oracle paths, and singleton branches never trigger a load or CNN call.
Deployment storage counts all full CNN checkpoints in every policy.

```powershell
& .\.venv-cuda\Scripts\python.exe -m cls_balance.residency --run runs/independent/<run-id> --depth 2 --device cuda:0 --output-root runs/residency
```

This is a separate synchronized **validation residency diagnostic**: the first
validation image by manifest order for each true manufacturer is supplied to all
three policies as an identical ready tensor. It measures setup, un-warmed model
path time and peak allocated GPU memory, and asserts identical predictions and
execution paths. It excludes decoding, preprocessing and input transfer, does
not guarantee predicted-route coverage and is **not** the Phase 6 full-pipeline
p95/FPS benchmark. The latter still requires the whole official test split,
representative warmup, repeated passes and a fixed measurement boundary.

## Checks

```powershell
& .\.venv\Scripts\python.exe scripts/freeze_tests.py --check
& .\.venv\Scripts\python.exe -m pytest -q tests checks
```

The original suite and its runner remain frozen. Phase 4 checks cover real
EfficientNet checkpoint reloads and separate parameter storage, local target
mapping, node planning, absent singleton classifiers, configuration validation,
full training/export/evaluation orchestration on generated fixtures, predicted
versus oracle routing, all residency policies, completion gating, verified node
reuse, tamper rejection and publication exclusion of training-ID contracts.
