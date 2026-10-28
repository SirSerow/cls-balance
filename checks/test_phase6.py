"""Benchmark boundary, complete-pass, compute accounting and publication checks."""

import csv
from dataclasses import replace

import pytest
import torch
from torch import nn

from cls_balance import benchmark as b
from cls_balance.artifacts import file_hash, write_json
from cls_balance.infer import FlatAdapter, HierarchicalAdapter, LogitNode
from cls_balance.models.cascade import node_plan
from cls_balance.transforms import Preprocessing


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


class Fixed(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.register_buffer("logits", torch.arange(classes, dtype=torch.float32).unsqueeze(0))
        self.calls = 0

    def forward(self, x):
        assert torch.is_inference_mode_enabled()
        self.calls += 1
        return self.logits


def adapter(data, shared=False):
    nodes = {s.adapter_key: LogitNode(Fixed(len(s.global_ids)), s.global_ids) for s in node_plan(data.taxonomy, 2)}
    encoder = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten()) if shared else None
    return HierarchicalAdapter(data.taxonomy, nodes, backbone=encoder, fine_tuned=shared)


@pytest.mark.parametrize("kwargs", [{"passes": 4}, {"passes": True}, {"precision": "bf16"},
    {"warmup_iterations": 0}, {"cpu_threads": 0}, {"runs": ("r", "r")}, {"depth": 4},
    {"precision": "fp16", "device": "cpu"}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        b.BenchmarkConfig("d", "p", "o", **({"runs": ("r",)} | kwargs))


def test_nested_timers_include_decode_and_conversion(tiny_data, monkeypatch):
    events = []
    model = FlatAdapter(tiny_data.taxonomy, LogitNode(Fixed(3), (0, 1, 2)))
    original = model.predict
    def predict(image):
        events.append("predict")
        return original(image)
    model.predict = predict
    monkeypatch.setattr(b, "synchronize", lambda d: events.append("sync"))
    ticks = iter([1., 2., 3.])
    monkeypatch.setattr(b.time, "perf_counter", lambda: next(ticks))
    p, image, timing = b.timed_image(model, tiny_data.image_path(tiny_data.splits["test"][0].image_id),
                                     Preprocessing(), torch.device("cpu"), torch.float32)
    assert events == ["sync", "sync", "predict", "sync"]
    assert timing == {"full_pipeline_ms": 2000., "model_path_ms": 1000.}
    assert image.shape == (3, 224, 224) and p.variant_id == 2


def test_conv_depthwise_linear_mac_convention_and_hook_cleanup():
    module = nn.Sequential(nn.Conv2d(3, 3, 3, groups=3, bias=False), nn.Flatten(), nn.Linear(27, 2)).eval()
    assert b.module_macs(module, torch.zeros(1, 3, 5, 5)) == 27 * 9 + 2 * 27
    assert all(not m._forward_hooks for m in module.modules())
    class Broken(nn.Module):
        def forward(self, x):
            raise ValueError("broken")
    broken = nn.Sequential(nn.Linear(2, 2), Broken())
    with pytest.raises(ValueError):
        b.module_macs(broken, torch.zeros(1, 2))
    assert not broken[0]._forward_hooks


@pytest.mark.parametrize("shared", [False, True])
def test_warmup_covers_unselected_nodes_and_paths(tiny_data, shared):
    model = adapter(tiny_data, shared)
    costs = b.warmup(model, torch.zeros(3, 224, 224), 2)
    assert set(costs) == set(model.nodes) | ({"backbone"} if shared else set())
    assert all(n.classifier.calls >= 3 for n in model.nodes.values())
    p = model.predict(torch.zeros(3, 224, 224))
    custom = {k: 10 for k in costs}
    assert b.path_macs(p, custom) == (p.classifier_calls + int(shared)) * 10
    assert p.backbone_calls == (1 if shared else p.classifier_calls)


@pytest.mark.parametrize("values", [[], [0], [float("nan")], [-1], [float("inf")]])
def test_bad_distribution(values):
    with pytest.raises(ValueError):
        b.distribution(values)


def test_fps_uses_elapsed_processing_time():
    rows = [{"pass": 0, "full_pipeline_ms": 1, "model_path_ms": .5, "route": "flat",
             "executed_macs": 3, "backbone_calls": 1, "singleton_bypasses": 0}] * 2
    result = b.aggregate_timings(rows, [{"pass": 0, "images": 2, "elapsed_seconds": 4, "sustained_fps": .5}])
    assert result["sustained_fps"] == .5
    assert result["full_pipeline_ms"]["p95"] == 1


@pytest.fixture
def measured(tmp_path, tiny_data, monkeypatch):
    model = adapter(tiny_data, True)
    provenance = {"seed": 42, "architecture": model.architecture, "training_run_id": "synthetic",
                  "stored_checkpoint_bytes": 0, "total_model_parameters": 0, "checkpoint_files": []}
    monkeypatch.setattr(b, "load_selected", lambda *a: (model, provenance))
    monkeypatch.setattr(b, "telemetry", lambda: {})
    monkeypatch.setattr(b, "environment", lambda repo: {"device": "cpu"})
    cfg = b.BenchmarkConfig("d", "p", str(tmp_path), runs=("r",), device="cpu", warmup_iterations=1)
    output = b.measure(cfg, tiny_data, tmp_path, tmp_path / "results")
    return output


def test_complete_benchmark_replay_and_oracle_roundtrip(measured):
    summary = b.audit_run(measured)
    assert summary["images_per_pass"] == 3 and summary["passes"] == 5
    assert b.read_json(measured / "resources.json")["model_state_unchanged"]
    assert (measured / "predictions_oracle_test.csv").exists()


def rehash(output, name):
    summary = b.read_json(output / "summary.json")
    summary["artifacts_sha256"][name] = file_hash(output / name)
    (output / "summary.json").unlink()
    write_json(output / "summary.json", summary)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "prediction", "macs", "timer", "route"])
def test_audit_rejects_inconsistent_timing_records(measured, mutation):
    path = measured / "latency_test.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[1]["image_id"] = rows[0]["image_id"]
    elif mutation == "prediction":
        rows[0]["variant_id"] = "999"
    elif mutation == "macs":
        rows[0]["executed_macs"] = "999"
    elif mutation == "timer":
        rows[0]["model_path_ms"] = str(float(rows[0]["full_pipeline_ms"]) + 1)
    else:
        rows[0]["route"] = "wrong"
    path.unlink()
    b.write_csv(path, rows)
    rehash(measured, path.name)
    with pytest.raises(ValueError):
        b.audit_run(measured)


def test_failures_preserved_without_completion(tmp_path, tiny_data, monkeypatch):
    model = adapter(tiny_data)
    monkeypatch.setattr(b, "load_selected", lambda *a: (model, {"seed": 42}))
    monkeypatch.setattr(b, "environment", lambda r: {})
    monkeypatch.setattr(b, "telemetry", lambda: {})
    monkeypatch.setattr(b, "warmup", lambda *a: (_ for _ in ()).throw(torch.cuda.OutOfMemoryError("fixture")))
    cfg = b.BenchmarkConfig("d", "p", "o", runs=("r",), device="cpu")
    with pytest.raises(torch.cuda.OutOfMemoryError):
        b.measure(cfg, tiny_data, tmp_path, tmp_path / "out")
    failed = next((tmp_path / "out").iterdir())
    assert not (failed / "summary.json").exists()
    assert b.read_json(failed / "failure.json")["oom"] is True


def test_report_replays_published_metrics_and_pairs_identical_images(measured):
    from cls_balance.report_benchmark import branch_review, load_records, paired_bootstrap
    records, taxonomy = load_records([measured])
    branches = branch_review(records, taxonomy)
    assert sum(r["test_support"] for r in branches) == len(records[0]["predictions"])
    assert sum(r["variants"] * r["variant_macro_recall"] for r in branches) / len(taxonomy.variants) == pytest.approx(records[0]["metrics"]["variant_macro_recall"])
    same = paired_bootstrap(records, records)
    assert same["difference_pp"] == 0 and same["ci95_pp"] == [0, 0]
    changed = dict(records[0])
    changed["predictions"] = tuple(replace(r, image_id="9999999") for r in changed["predictions"])
    with pytest.raises(ValueError, match="identical image"):
        paired_bootstrap(records, [changed])


def test_full_schematics_include_all_nodes_and_variants(tmp_path, tiny_data):
    from cls_balance.network_diagrams import draw_network, network_inventory
    from xml.etree import ElementTree
    for arch in ("FLAT", "IND-2", "FT-FROZEN-2"):
        draw_network(tmp_path, tiny_data.taxonomy, arch, [])
        root = ElementTree.parse(tmp_path / f"network-{arch}.svg").getroot()
        text = " ".join(root.itertext())
        assert all(v in text for v in tiny_data.taxonomy.variants)
        assert all(m in text for m in tiny_data.taxonomy.manufacturers)
        inventory = network_inventory(tiny_data.taxonomy, arch)
        if arch != "FLAT":
            assert len(inventory["nodes"]) == len(node_plan(tiny_data.taxonomy, 2))
            assert sum(b["singleton"] for b in inventory["branches"]) == 1
        assert inventory["backbones_stored"] == (2 if arch == "IND-2" else 1)


def test_full_backbone_schematic_matches_installed_model(tmp_path):
    from cls_balance.network_diagrams import draw_backbone
    from xml.etree import ElementTree
    draw_backbone(tmp_path)
    ElementTree.parse(tmp_path / "efficientnet-b0-detail.svg")
    inventory = b.read_json(tmp_path / "efficientnet-b0-modules.json")
    assert inventory["total_parameters"] == 4135648
    convolutions = [m for m in inventory["modules"] if m["type"] == "Conv2d"]
    assert any(m["groups"] > 1 for m in convolutions)
    assert all("state_dict" not in m for m in inventory["modules"])


def test_synthetic_or_subset_results_cannot_be_published_as_official(measured, tmp_path):
    from cls_balance.report_benchmark import publish
    with pytest.raises(ValueError, match="complete official benchmark"):
        publish([measured], tmp_path / "publication", [], saved_nodes=[])
    assert not (tmp_path / "publication").exists()


def test_training_review_follows_selected_nested_attempts(tmp_path, tiny_data):
    from cls_balance.report_benchmark import training_nodes
    run = tmp_path / "source"
    run.mkdir()
    write_json(run / "taxonomy.json", tiny_data.taxonomy.to_dict())
    write_json(run / "config.resolved.json", {})
    entries = {}
    for spec in node_plan(tiny_data.taxonomy, 2):
        folder = run / "nodes" / spec.key / "selected-attempt"
        folder.mkdir(parents=True)
        b.write_csv(folder / "history.csv", [{"epoch": 1, "val_macro_recall": .5}, {"epoch": 2, "val_macro_recall": 1.}])
        write_json(folder / "node.json", {"file_sha256": "a" * 64, "metadata": {
            "node": spec.to_dict(), "seed": 42, "best_epoch": 2, "validation_macro_recall": 1.}})
        write_json(folder / "summary.json", {"node": spec.to_dict(), "best_epoch": 2, "validation_macro_recall": 1.,
            "artifacts_sha256": {name: file_hash(folder / name) for name in ("history.csv", "node.json")}})
        rejected = folder.parent / "discarded-attempt"
        rejected.mkdir()
        write_json(rejected / "summary.json", {"must_not_be_read": True})
        entries[spec.key] = {"path": folder.relative_to(run).as_posix(), "checkpoint_sha256": "a" * 64}
    write_json(run / "bundle-ind2.json", {"nodes": entries})
    write_json(run / "summary.json", {"artifacts_sha256": {"bundle-ind2.json": file_hash(run / "bundle-ind2.json")}})
    records = [{"architecture": "IND-2", "seed": 42, "resources": {"training_run_id": "source",
                "training_summary_sha256": file_hash(run / "summary.json")}}]
    nodes = training_nodes(records, [tmp_path])
    assert {n["key"] for n in nodes} == set(entries)
    assert all(n["best_epoch"] == 2 for n in nodes)


def test_failed_publication_audit_leaves_no_completion_marker(measured, full_metadata, tmp_path, monkeypatch):
    import cls_balance.report_benchmark as report
    records, _ = report.load_records([measured])
    records[0] = records[0] | {"architecture": "FLAT", "images_per_pass": 3333}
    monkeypatch.setattr(report, "load_records", lambda runs: (records, full_metadata[1]))
    for function in ("draw_network", "draw_backbone", "draw_figures", "draw_training", "write_report"):
        monkeypatch.setattr(report, function, lambda *a, **k: None)
    monkeypatch.setattr(report, "branch_review", lambda *a: [{"synthetic_check": True}])
    def reject(*args, **kwargs):
        raise ValueError("audit rejected")
    monkeypatch.setattr(report, "audit_publication", reject)
    with pytest.raises(ValueError, match="audit rejected"):
        report.publish([measured], tmp_path / "out", [], saved_nodes=[])
    assert not any((tmp_path / "out").rglob("publication.json"))


def test_paired_bootstrap_detects_known_accuracy_difference(measured):
    from cls_balance.report_benchmark import load_records, paired_bootstrap
    records, taxonomy = load_records([measured])
    left, right = dict(records[0]), dict(records[0])
    left["predictions"] = tuple(replace(r, prediction=replace(r.prediction,
        variant_id=(r.true_variant_id + 1) % len(taxonomy.variants))) for r in left["predictions"])
    right["predictions"] = tuple(replace(r, prediction=replace(r.prediction,
        variant_id=r.true_variant_id)) for r in right["predictions"])
    result = paired_bootstrap([left], [right])
    assert result["difference_pp"] == pytest.approx(100)
    assert result["ci95_pp"] == pytest.approx([100, 100])


@pytest.mark.parametrize("mutation", ["classwise", "fps"])
def test_report_rejects_inconsistent_redundant_artifacts(measured, mutation):
    from cls_balance.report_benchmark import load_records
    if mutation == "classwise":
        path = measured / "classwise_metrics.csv"
        rows = b.read_json(measured / "metrics.json")["classwise"]
        rows[0]["recall"] = 999
        path.unlink()
        b.write_csv(path, rows)
        rehash(measured, path.name)
    else:
        path = measured / "pass-00.json"
        value = b.read_json(path)
        value["sustained_fps"] = 999
        path.unlink()
        write_json(path, value)
        rehash(measured, path.name)
        timing_path = measured / "timing_summary.json"
        timing = b.read_json(timing_path)
        timing["passes"][0]["sustained_fps"] = 999
        timing_path.unlink()
        write_json(timing_path, timing)
        rehash(measured, timing_path.name)
    with pytest.raises(ValueError, match="Classwise|elapsed time/FPS"):
        load_records([measured])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Optional CUDA FP16 deployment check")
def test_cuda_fp16_full_pipeline_and_saved_replay(tmp_path, tiny_data, monkeypatch):
    model = adapter(tiny_data, True)
    provenance = {"seed": 42, "architecture": model.architecture, "training_run_id": "synthetic",
                  "stored_checkpoint_bytes": 0, "total_model_parameters": 0, "checkpoint_files": []}
    monkeypatch.setattr(b, "load_selected", lambda *a: (model, provenance))
    monkeypatch.setattr(b, "telemetry", lambda: {})
    config = b.BenchmarkConfig("d", "p", str(tmp_path), runs=("r",), device="cuda:0", precision="fp16", warmup_iterations=2)
    output = b.measure(config, tiny_data, tmp_path, tmp_path / "out")
    summary = b.audit_run(output)
    assert summary["precision"] == "fp16"
    assert b.read_json(output / "resources.json")["peak_gpu_allocated_bytes"] > 0
