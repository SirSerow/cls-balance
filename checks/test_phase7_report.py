"""Artifact-only Phase 7 publication audit and rebuild on synthetic predictions.

These perfect synthetic predictions exercise file contracts, never experimental
performance. All fixtures remain under pytest's temporary directory.
"""

from functools import lru_cache

import pytest

from cls_balance import incremental as inc
from cls_balance import report_incremental as report
from cls_balance.artifacts import object_hash, write_json
from cls_balance.evaluate import EvaluationResult, audit_run
from cls_balance.infer import Decision, Prediction
from cls_balance.metrics import PredictionRecord, score
from cls_balance.models.cascade import node_plan
from cls_balance.prepare import write_csv
from cls_balance.transforms import Preprocessing


@pytest.fixture
def public_inputs(tmp_path, full_metadata, monkeypatch):
    # Scoring validates the same immutable synthetic taxonomy repeatedly. Cache
    # its pure branch derivation here; production code and results are unchanged.
    from cls_balance.taxonomy import Taxonomy
    monkeypatch.setattr(Taxonomy, "branches", lru_cache(maxsize=8)(Taxonomy.branches))
    splits, taxonomy, _ = full_metadata
    selection = inc.select_introduction(taxonomy)
    root = tmp_path / "synthetic-check-only"
    root.mkdir()
    seed = 42
    manifests = {s: object_hash([r.image_id for r in rows]) for s, rows in splits.items()}
    for name, value in {"taxonomy.json": taxonomy.to_dict(), "selection.json": selection,
        "source.json": {"training_run_id": "synthetic-training", "evaluation_run_id": "synthetic-evaluation"},
        "config.resolved.json": {"seeds": [seed], "hidden_dimension": 4},
        "environment.json": {"git_sha": "synthetic", "git_dirty": True, "gpus": ["synthetic checks"],
                              "torch": "synthetic", "cuda_runtime": None, "nvidia_smi": None},
        "compatibility.json": {"taxonomy_sha256": object_hash(taxonomy.to_dict()),
                               "dataset_manifest_sha256": object_hash(manifests), "preprocessing_sha256": Preprocessing().sha256}}.items():
        write_json(root / name, value)
    heldout = selection["variant_id"]
    alternate = next(v for v in selection["branch_variant_ids"] if v != heldout)
    branches = taxonomy.branches()
    tables = []
    for architecture in report.ARCHITECTURES:
        for stage in ("before", "after"):
            for routing in ("predicted", "oracle"):
                records = []
                for row in splits["test"]:
                    v, f, m = taxonomy.ids(row)
                    predicted = alternate if stage == "before" and v == heldout else v
                    pf = taxonomy.variant_to_family[predicted]
                    ids = branches["manufacturer_to_variant"][m]
                    decisions = (Decision("manufacturer", None, m, "oracle" if routing == "oracle" else "classifier"),
                                 Decision("variant", m, predicted, "singleton" if len(ids) == 1 else "classifier"))
                    p = Prediction(predicted, m, pf, decisions,
                        1 if "FROZEN" in architecture else sum(d.source == "classifier" for d in decisions))
                    records.append(PredictionRecord(row.image_id, "test", architecture, routing, v, f, m, p))
                config = {"seed": seed, "split": "test", "architecture": architecture, "routing": routing,
                          "device": "cpu", "precision": "fp32", "residency": "all-resident"}
                metadata = {"config": config, "config_sha256": object_hash(config),
                    "model_id": f"synthetic-training/s{seed}/{architecture}/{stage}",
                    "taxonomy_sha256": object_hash(taxonomy.to_dict()), "preprocessing_sha256": Preprocessing().sha256,
                    "evaluation_ids_sha256": object_hash([r.image_id for r in records]),
                    "manifest_records_sha256": manifests, "dataset_manifest_records_sha256": object_hash(manifests),
                    "environment": {"torch": "synthetic", "gpus": []}}
                result = EvaluationResult(tuple(records), score(records, taxonomy), metadata, taxonomy, Preprocessing())
                saved = result.save(root / "evaluations")
                tables.append({"seed": seed, "architecture": architecture, "stage": stage, "routing": routing,
                               "path": saved.relative_to(root).as_posix(), "audit": audit_run(saved)})
    write_json(root / "evaluations.json", tables)
    target = root / f"seed-{seed}"
    target.mkdir()
    for architecture in report.ARCHITECTURES:
        write_json(target / f"resources-{architecture}.json", {"update_wall_seconds": 1.,
            "trainable_parameters": 100, "replacement_bytes": 1000})
    write_json(target / "initial-costs.json", {"IND-2": [{"training_wall_seconds": 1.}],
        "FLAT": {"update_wall_seconds": 1.}, "shared_training_wall_seconds": 1.})
    nodes = []
    for architecture in ("FLAT", *report.ARCHITECTURES):
        for stage in (("before",) if architecture == "FLAT" else ("before", "after")):
            specs = (None,) if architecture == "FLAT" else inc.initial_specs(taxonomy, heldout) if stage == "before" else (
                next(s for s in node_plan(taxonomy, 2) if s.adapter_key == f"variant:{selection['manufacturer_id']}"),)
            for spec in specs:
                key = spec.key if spec else "variant"
                history = target / f"{architecture}-{stage}-{key}.csv"
                write_csv(history, [{"epoch": i, "train_macro_recall": .25 * i, "val_macro_recall": .2 * i}
                                    for i in (1, 2)])
                nodes.append({"architecture": architecture, "stage": stage, "seed": seed, "key": key,
                    "classes": len(spec.global_ids) if spec else 99, "best_epoch": 2,
                    "validation_macro_recall": .4, "hidden_dimension": 4,
                    "history": history.relative_to(root).as_posix()})
    write_json(root / "nodes.json", nodes)
    write_json(root / "aggregate.json", report.calculate(root))
    write_json(root / "summary.json", {"status": "complete", "artifacts_sha256": inc.inventory(root)})
    return root, taxonomy, selection


def test_artifact_only_recomputes_all_test_images(public_inputs):
    root, _, _ = public_inputs
    audit = report.audit_publication(root)
    assert audit["artifact_only_recomputed"] and audit["data_and_weights_excluded"]
    values = report.calculate(root)
    assert len(values) == 2
    assert values[0]["predicted"]["groups"]["all"]["after"]["images"] == 3333
    assert values[0]["predicted"]["groups"]["new"]["before"]["macro_recall"] == 0


def test_resealed_aggregate_tampering_rejected(public_inputs):
    root, _, _ = public_inputs
    values = inc.read_json(root / "aggregate.json")
    values[0]["predicted"]["groups"]["new"]["after"]["macro_recall"] = .123
    (root / "aggregate.json").unlink()
    write_json(root / "aggregate.json", values)
    (root / "summary.json").unlink()
    write_json(root / "summary.json", {"status": "complete", "artifacts_sha256": inc.inventory(root)})
    with pytest.raises(ValueError, match="aggregates disagree"):
        report.audit_publication(root)


def test_missing_evaluation_rejected(public_inputs):
    root, _, _ = public_inputs
    entries = inc.read_json(root / "evaluations.json")[:-1]
    (root / "evaluations.json").unlink()
    write_json(root / "evaluations.json", entries)
    with pytest.raises(ValueError, match="Incomplete"):
        report.calculate(root)


def test_rebuild_without_data_or_weights_is_byte_stable(public_inputs, tmp_path):
    pytest.importorskip("matplotlib")
    root, taxonomy, selection = public_inputs
    # Build a complete synthetic review before exercising the public rebuild API.
    (root / "summary.json").unlink()
    report.draw_figures(root, inc.read_json(root / "aggregate.json"), inc.read_json(root / "nodes.json"), taxonomy, selection)
    report.render_report(root, inc.read_json(root / "aggregate.json"))
    write_json(root / "summary.json", {"status": "complete", "kind": "phase7-review", "artifacts_sha256": inc.inventory(root)})
    destination = tmp_path / "rebuilt"
    report.rebuild(root, destination)
    assert inc.inventory(root) == inc.inventory(destination)
