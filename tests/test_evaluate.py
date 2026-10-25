"""Synthetic integration evidence only; never a trained-model experiment."""

from dataclasses import replace
import json

import pytest
import torch

from cls_balance.artifacts import object_hash
from cls_balance import evaluate as module
from cls_balance.evaluate import Evaluator, ModelProvenance, audit_run, compare_evaluations, model_hash
from cls_balance.infer import FlatAdapter, HierarchicalAdapter, LogitNode
from cls_balance.taxonomy import Taxonomy
from cls_balance.transforms import Preprocessing
from test_routing import Constant, nodes_for


def provenance(data, model):
    return ModelProvenance("synthetic-fixture-only", model_hash(model), object_hash(data.taxonomy.to_dict()),
        Preprocessing().sha256, 42, tuple(r.image_id for r in data.splits["train"]),
        tuple(r.image_id for r in data.splits["val"]))


@pytest.fixture
def evaluation_fixture(tiny_data, monkeypatch):
    # The production edition guard is tested separately. Only that guard is
    # replaced for tiny generated images; parsing/transforms/inference/CSV/metrics
    # and all leakage/provenance contracts still execute for real.
    monkeypatch.setattr(module, "validate_edition", lambda *args: None)
    monkeypatch.setattr(module, "environment", lambda repo: {"scope": "synthetic unit test", "torch": torch.__version__})
    model = HierarchicalAdapter(tiny_data.taxonomy, nodes_for(tiny_data.taxonomy, 2, manufacturer=1, variant=2))
    contract = provenance(tiny_data, model)
    return tiny_data, model, contract


def test_production_evaluator_rejects_nonofficial_edition(tiny_data):
    model = HierarchicalAdapter(tiny_data.taxonomy, nodes_for(tiny_data.taxonomy, 2))
    with pytest.raises(ValueError, match="Unexpected dataset edition"):
        Evaluator(tiny_data, Preprocessing(), provenance(tiny_data, model))


def test_complete_split_predictions_saved_and_audited(tmp_path, evaluation_fixture):
    data, model, contract = evaluation_fixture
    evaluator = Evaluator(data, Preprocessing(), contract)
    result = evaluator.evaluate(model)
    assert result.metrics["images"] == 3 and result.metrics["variant_macro_recall"] == 1 / 3
    assert [r.image_id for r in result.records] == [r.image_id for r in data.splits["val"]]
    assert result.metadata["preprocessing_sha256"] == Preprocessing().sha256
    assert result.metadata["model_sha256"] == model_hash(model)
    assert result.metadata["config"]["precision"] == "fp32"
    first, second = result.save(tmp_path), result.save(tmp_path)
    assert first != second
    assert audit_run(first)["status"] == audit_run(second)["status"] == "verified"
    assert audit_run(first)["images"] == 3
    assert (first / "classwise_metrics.csv").is_file()


def test_paired_evaluation_contract(evaluation_fixture):
    data, model, contract = evaluation_fixture
    evaluator = Evaluator(data, Preprocessing(), contract)
    hard = evaluator.evaluate(model, split="test")
    oracle = evaluator.evaluate(model, split="test", routing="oracle")
    paired = compare_evaluations(hard, oracle)
    assert paired["predicted"]["routing"] == "predicted"
    assert paired["oracle_diagnostic"]["diagnostic_only"]
    assert paired["routing_penalty_macro_recall_pp"] == pytest.approx(100 / 3)
    changed = {**oracle.metadata, "preprocessing_sha256": "0" * 64}
    with pytest.raises(ValueError, match="preprocessing_sha256"):
        compare_evaluations(hard, replace(oracle, metadata=changed))


def test_flat_evaluation_receives_no_truth(evaluation_fixture):
    data, _, _ = evaluation_fixture
    model = FlatAdapter(data.taxonomy, LogitNode(Constant([0, 1, 0]), (0, 1, 2)))
    evaluator = Evaluator(data, Preprocessing(), provenance(data, model))
    assert evaluator.evaluate(model).metrics["variant_macro_recall"] == 1 / 3
    with pytest.raises(ValueError, match="unavailable"):
        evaluator.evaluate(model, routing="oracle")


def test_three_stage_diagnostics_are_separate(evaluation_fixture):
    data, _, _ = evaluation_fixture
    model = HierarchicalAdapter(data.taxonomy, nodes_for(data.taxonomy, 3, manufacturer=0, variant=2), depth=3)
    evaluator = Evaluator(data, Preprocessing(), provenance(data, model))
    result = evaluator.evaluate(model, conditional_family=True)
    assert all(r.conditional_family_id == r.true_family_id for r in result.records)
    assert result.metrics["routing_diagnostics"]["family_under_true_manufacturer"]["accuracy"] == 1
    assert result.metrics["routing_diagnostics"]["manufacturer_router"]["accuracy"] == 1 / 3
    assert all(r.prediction.backbone_calls == 1 for r in result.records)


@pytest.mark.parametrize("field", ["model_sha256", "taxonomy_sha256", "preprocessing_sha256"])
def test_hash_mismatch_rejected(evaluation_fixture, field):
    data, model, contract = evaluation_fixture
    with pytest.raises(ValueError, match="hash mismatch|fingerprint mismatch"):
        Evaluator(data, Preprocessing(), replace(contract, **{field: "0" * 64})).evaluate(model)


@pytest.mark.parametrize("field,ids", [("training_image_ids", ("0000020",)),
    ("selection_image_ids", ("0000020",)), ("training_image_ids", ()),
    ("training_image_ids", ("0000000", "0000000"))])
def test_declared_training_or_selection_leakage(evaluation_fixture, field, ids):
    data, model, contract = evaluation_fixture
    with pytest.raises(ValueError, match="invalid .* IDs|leakage"):
        Evaluator(data, Preprocessing(), replace(contract, **{field: ids}))


def test_overlapping_splits_rejected(evaluation_fixture):
    data, _, contract = evaluation_fixture
    data.splits["test"][0] = replace(data.splits["test"][0], image_id=data.splits["train"][0].image_id)
    with pytest.raises(ValueError, match="split leakage"):
        Evaluator(data, Preprocessing(), contract)


def test_manifest_mutation_rejected(evaluation_fixture):
    data, model, contract = evaluation_fixture
    evaluator = Evaluator(data, Preprocessing(), contract)
    data.splits["test"].pop()
    with pytest.raises(ValueError, match="manifests changed"):
        evaluator.evaluate(model)


@pytest.mark.parametrize("mutation", ["tensor", "logit_map"])
def test_model_mutation_rejected(evaluation_fixture, mutation):
    data, model, contract = evaluation_fixture
    evaluator = Evaluator(data, Preprocessing(), contract)
    if mutation == "tensor":
        model.nodes["manufacturer"].classifier.logits.add_(1)
    else:
        model.nodes["manufacturer"].global_ids = (0, 1)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        evaluator.evaluate(model)


@pytest.mark.parametrize("field,value", [("seed", -1), ("seed", True), ("precision", "fp16"),
                                       ("residency", "on-demand"), ("model_id", "")])
def test_unsupported_contract(evaluation_fixture, field, value):
    data, _, contract = evaluation_fixture
    with pytest.raises(ValueError):
        Evaluator(data, Preprocessing(), replace(contract, **{field: value}))


@pytest.mark.parametrize("artifact", ["predictions.csv", "metrics.json", "taxonomy.json", "metadata.json"])
def test_audit_detects_corruption(tmp_path, evaluation_fixture, artifact):
    data, model, contract = evaluation_fixture
    path = Evaluator(data, Preprocessing(), contract).evaluate(model).save(tmp_path)
    target = path / artifact
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        audit_run(path)


def test_audit_recomputes_metrics_even_with_updated_byte_hash(tmp_path, evaluation_fixture):
    from cls_balance.artifacts import file_hash

    data, model, contract = evaluation_fixture
    path = Evaluator(data, Preprocessing(), contract).evaluate(model).save(tmp_path)
    metrics = json.loads((path / "metrics.json").read_text())
    metrics["variant_macro_recall"] = 1
    (path / "metrics.json").write_text(json.dumps(metrics))
    summary = json.loads((path / "summary.json").read_text())
    summary["artifacts_sha256"]["metrics.json"] = file_hash(path / "metrics.json")
    (path / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="disagree with per-image"):
        audit_run(path)


def test_taxonomy_artifact_roundtrip_and_inconsistent_maps(tiny_data):
    value = tiny_data.taxonomy.to_dict()
    assert Taxonomy.from_dict(value) == tiny_data.taxonomy
    value["branches"]["family_to_variant"] = ((0,), (1, 2))
    with pytest.raises(ValueError, match="inconsistent maps"):
        Taxonomy.from_dict(value)


@pytest.mark.parametrize("mapping", [(0,), (0, 2), (0, True), (0, 0)])
def test_invalid_taxonomy_direct_construction(mapping):
    with pytest.raises(ValueError):
        Taxonomy(("V0", "V1"), ("F0", "F1"), ("M",), mapping, (0, 0))
