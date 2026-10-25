# Phase 2 — inference and evaluation contract

Phase 2 supplies adapters for **injected PyTorch modules** and complete-split
evaluation. It does not construct or train EfficientNet checkpoints. Those
factories and training commands begin in Phase 3. The tests use known-output
modules and generated images; their scores are correctness checks, not aircraft
classification results. Latency, throughput and memory benchmarking remain
Phase 6.

## Model adapters

`FlatAdapter(taxonomy, LogitNode(classifier, global_ids))` wraps an all-variant
classifier. `HierarchicalAdapter(taxonomy, nodes, depth=2)` wraps independent full
CNN nodes. Set `depth=3` for the family stage. For shared models, supply
`backbone=encoder` and optionally `fine_tuned=True` for `FT-FROZEN-*`.

`LogitNode.global_ids` declares the global class corresponding to each output
column. Its order may differ from canonical order, but its set must exactly
match the branch's allowed children. Forward output must be finite floating
logits of shape `[1, number_of_children]`. The common input to `predict` is one
preprocessed floating tensor with shape `[3, 224, 224]`.

Hierarchical node keys are:

| Key | Output global IDs | Input |
| --- | --- | --- |
| `manufacturer` | All manufacturer IDs | Image or shared embedding |
| `family:<manufacturer_id>` | That manufacturer's family IDs, depth 3 only | Image or shared embedding |
| `variant:<manufacturer_id>` | That manufacturer's variant IDs, depth 2 only | Image or shared embedding |
| `variant:<family_id>` | That family's variant IDs, depth 3 only | Image or shared embedding |

Obtain child IDs from `taxonomy.branches()`. Supply **only** multi-child nodes;
single-child nodes have no classifier. Independent nodes must own separate
modules and parameter/buffer storage. Full EfficientNet construction is the
responsibility of the later model factories. Shared heads must also be distinct.
The shared encoder returns `[1, feature_dimension]`, is frozen, and stays in eval
mode when the adapter enters train mode. Each shared `predict` invokes it once
and passes the same embedding to only the chosen heads.

All predictions run in `torch.inference_mode()` and eval mode. The returned
`Prediction` contains the final global `variant_id`, its `family_id` and
`manufacturer_id`, and ordered `Decision` records with stage, parent, selected
global ID, and source (`classifier`, `singleton`, `oracle`). `backbone_calls`,
`classifier_calls` and `singleton_bypasses` describe the executed path. For FLAT,
the family/manufacturer IDs are projected from the final variant and are not
router outputs. No conditional softmax scores are presented as global confidence.

Normal inference is `model.predict(image)`. It rejects truth arguments. The
explicit diagnostic form is
`model.predict(image, routing="oracle", truth=Ancestors(manufacturer_id, family_id))`.
It skips the ancestor classifiers and injects only the ancestors needed at the
chosen depth; variant truth is never an argument. FLAT rejects oracle mode.

## Evaluation API

The following function is runnable integration code once a selected adapter and
its training/validation ID provenance exist; it does not train a model:

```python
from pathlib import Path

from cls_balance.artifacts import object_hash
from cls_balance.data import AircraftData
from cls_balance.evaluate import (
    Evaluator, ModelProvenance, compare_evaluations, model_hash,
)
from cls_balance.infer import InferenceAdapter
from cls_balance.transforms import Preprocessing


def evaluate_selected_model(
    model: InferenceAdapter,
    data_root: Path,
    model_id: str,
    training_ids: tuple[str, ...],
    selection_ids: tuple[str, ...],
    seed: int,
    output_root: Path,
):
    data = AircraftData.load(data_root, download=False)
    preprocessing = Preprocessing()
    provenance = ModelProvenance(
        model_id=model_id,
        model_sha256=model_hash(model),
        taxonomy_sha256=object_hash(model.taxonomy.to_dict()),
        preprocessing_sha256=preprocessing.sha256,
        seed=seed,
        training_image_ids=training_ids,
        selection_image_ids=selection_ids,
    )
    evaluator = Evaluator(data, preprocessing, provenance, device="cpu")
    hard = evaluator.evaluate(model, split="val", conditional_family=model.depth == 3)
    saved = hard.save(output_root)
    if model.depth > 1:
        oracle = evaluator.evaluate(model, split="val", routing="oracle")
        oracle.save(output_root)
        paired = compare_evaluations(hard, oracle)
        return saved, paired
    return saved, hard.metrics
```

Use the preprocessing and fingerprints saved with the selected checkpoint in
production, rather than recomputing expected versions after changing an artifact.
The API verifies the model's current tensor bytes, module types and logit-ID maps
against its declared fingerprint, before and after inference. It checks taxonomy
and preprocessing hashes, official dataset edition, all split IDs for leakage,
declared training IDs against train, and selection IDs against val. A model's
declared training history remains an auditable claim; a hash cannot prove it.

Evaluation consumes every image of the requested split with deterministic
preprocessing. Subset selection and augmented evaluation are not exposed.
Validation is the default; use `split="test"` only after model selection and
experiment settings are frozen. The first supported evaluation policy is FP32,
batch 1, all models resident, on an explicitly selected device.

`conditional_family=True` performs a **separate diagnostic** at depth 3: predict
the family under the true manufacturer. Its denominator is all evaluated images,
including deterministic branches. This measures the actual family router; it
does not count truth-forced oracle family labels as successful predictions. These
extra diagnostic calls are excluded from the recorded production-path counts.
The normal family accuracy conditional on a correct predicted manufacturer has
its own, explicitly smaller denominator. Neither diagnostic is timed here.

## Metrics and artifacts

`score(records, taxonomy)` computes fixed-class macro recall, top-1, macro-F1,
variant confusion matrix, classwise recall/precision/F1/support, and family and
manufacturer accuracy projected from the final variant. Scores are **fractions**
in `[0, 1]`; paired routing penalties are **percentage points**. Missing variant
support, duplicate IDs, mixed splits/modes, and impossible paths are errors.

Routing diagnostics include manufacturer accuracy and macro recall, both family
diagnostic denominators, first wrong stage (manufacturer/family/variant), observed
singleton bypass rates, selected-branch support, and executed classifier/backbone
count distributions. FLAT router metrics and truth-forced oracle router metrics
are null. `compare_evaluations` checks model/data/preprocessing contracts and
requires exactly matching image IDs/truth before computing oracle-minus-hard
macro-recall and top-1 penalties. `paired_scores` is the lower-level record API;
use `compare_evaluations` when comparing saved evaluation contexts.

`EvaluationResult.save` creates a unique `eval-<UTC>-<random>` directory. It saves
per-image CSV predictions with both truth and selected route, metrics JSON,
classwise CSV, routing diagnostics, taxonomy, preprocessing, environment and
metadata. `summary.json` is written last and hashes all required artifacts. A run
without that marker is incomplete. Results are not overwritten on rerun.

Metadata includes seed, Git SHA/dirty state, package/CUDA/device environment,
precision/residency, model/preprocessing/taxonomy/configuration hashes, and
manifest/training/selection/evaluation-ID hashes. Evaluation manifest hashes use
canonical JSON rows; preparation's manifest hashes use CSV bytes. These are
explicitly distinct hash schemes. Runs and any later model artifacts stay ignored.

The saved-run audit is operational and requires no dataset or model reload:

```powershell
.venv/Scripts/python.exe -m cls_balance.evaluate --run-dir runs/evaluation/<existing-eval-run-id>
```

Replace the path with the directory returned by `save`. The installed
`cls-balance-evaluate` entry point accepts the same flag. The audit checks artifact
hashes and version contracts, reloads predictions, recomputes metrics, and verifies
the redundant classwise/diagnostic/environment outputs.

## Frozen tests

After successful validation, `tests/FROZEN.json` records SHA256 values for every
test/fixture file, the package/test configuration, CI workflow and freeze script.
It also hashes the successful committed JUnit report. CI checks this manifest.

```powershell
.venv/Scripts/python.exe scripts/freeze_tests.py --check
.venv/Scripts/python.exe -m pytest -q
```

The check is read-only and fails on changed, added or missing inputs or altered
validation evidence. Tests still execute normally. `--write --junit <report>` is
only for an initial, validated baseline and refuses to overwrite an existing one.
Future test changes need explicit review and a new passing baseline; do not
silently refresh it to make a failing check disappear.
