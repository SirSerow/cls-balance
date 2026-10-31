"""Phase 8 statistical contracts and publication checks on synthetic evidence.

These generated fixtures are software checks, never experimental measurements.
"""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from cls_balance import report_analysis as report
from cls_balance.artifacts import write_json
from cls_balance.benchmark import aggregate_timings
from cls_balance.infer import Decision, Prediction
from cls_balance.metrics import PredictionRecord, score


def pair(labels, before, after, seed=42):
    def rows(correct):
        return [SimpleNamespace(image_id=str(i), true_variant_id=v, split="test", routing="predicted",
                prediction=SimpleNamespace(variant_id=v if ok else -1))
                for i, (v, ok) in enumerate(zip(labels, correct, strict=True))]
    return {"seed": seed, "predictions": rows(before)}, {"seed": seed, "predictions": rows(after)}


def bootstrap(left, right, **kwargs):
    return report.bootstrap_difference(left, right, **({"class_ids": (0, 1), "replicates": 1000, "rng_seed": 8} | kwargs))


def test_equal_class_weighting_unequal_support():
    a, b = pair([0, 0, 0, 1], [False] * 4, [False, False, False, True])
    result = bootstrap([a], [b])
    assert result["per_seed"][0]["difference_pp"] == 50  # Micro difference would be 25.
    assert result["per_seed"][0]["ci95_pp"] == [50, 50]


def test_bootstrap_matches_reference_stratified_draws():
    a, b = pair([0, 0, 1, 1], [True, False, False, True], [False, True, True, True])
    result = bootstrap([a], [b])
    rng = np.random.default_rng(8)
    delta = np.array([-1, 1, 1, 0])
    samples = sum(delta[rng.choice(g, size=(1000, len(g)))].mean(1) / 2 for g in ([0, 1], [2, 3]))
    assert result["per_seed"][0]["difference_pp"] == 25
    assert result["per_seed"][0]["ci95_pp"] == pytest.approx(np.percentile(100 * samples, [2.5, 97.5]))


def test_shared_image_draws_preserve_cross_seed_dependence():
    a, b = pair([0, 0, 1, 1], [False] * 4, [True, False, True, False])
    c, d = deepcopy(a), deepcopy(b)
    c["seed"] = d["seed"] = 43
    result = bootstrap([a, c], [b, d])
    assert result["fixed_seed_mean"]["ci95_pp"] == result["per_seed"][0]["ci95_pp"]
    assert result["training_seed_difference_sd_pp"] == 0
    # Duplicating identical fixed seeds must not narrow the image interval.
    assert result["per_seed"][0]["ci95_pp"] == result["per_seed"][1]["ci95_pp"]


def test_pairing_is_by_image_id_and_deterministic():
    a, b = pair([0, 0, 1, 1], [True, False, False, True], [False, True, True, True])
    expected = bootstrap([a], [b])
    b["predictions"].reverse()
    assert bootstrap([a], [b]) == expected
    assert bootstrap([a], [a])["per_seed"][0]["ci95_pp"] == [0, 0]


@pytest.mark.parametrize("mutation", ["ids", "truth", "duplicate", "seed", "missing_class", "oracle", "validation"])
def test_bootstrap_rejects_unpaired_or_ineligible_data(mutation):
    a, b = pair([0, 0, 1, 1], [True] * 4, [True] * 4)
    if mutation == "ids":
        b["predictions"][0].image_id = "missing"
    elif mutation == "truth":
        b["predictions"][0].true_variant_id = 1
    elif mutation == "duplicate":
        b["predictions"][0].image_id = b["predictions"][1].image_id
    elif mutation == "seed":
        b["seed"] = 43
    elif mutation == "missing_class":
        a["predictions"] = a["predictions"][:2]
        b["predictions"] = b["predictions"][:2]
    elif mutation == "oracle":
        b["predictions"][0].routing = "oracle"
    else:
        b["predictions"][0].split = "val"
    with pytest.raises(ValueError):
        bootstrap([a], [b])


@pytest.mark.parametrize("settings", [{"replicates": 0}, {"replicates": True}, {"rng_seed": -1},
    {"class_ids": ()}, {"class_ids": (0, 0)}])
def test_invalid_bootstrap_settings(settings):
    a, b = pair([0, 1], [True, True], [True, True])
    with pytest.raises(ValueError):
        bootstrap([a], [b], **settings)


def test_pareto_dominance_ties_and_tradeoffs():
    points = [{"architecture": a, "p95_ms": x, "recall_percent": y}
              for a, x, y in [("fast", 1, 80), ("same", 1, 80), ("slow", 3, 70), ("accurate", 2, 90)]]
    assert report.pareto_frontier(points) == ["fast", "same", "accurate"]
    with pytest.raises(ValueError):
        report.pareto_frontier([{"architecture": "invalid", "p95_ms": float("nan"), "recall_percent": 1}])


@pytest.fixture
def synthetic_sources(tmp_path, tiny_data, monkeypatch):
    taxonomy = tiny_data.taxonomy
    path = tmp_path / "synthetic-input"
    path.mkdir()
    write_json(path / "data_contract.json", {"manifest_records_sha256": {"test": "synthetic"},
        "taxonomy_sha256": "synthetic", "preprocessing_sha256": "synthetic"})
    config = {"benchmark": "synthetic-benchmark", "incremental": "synthetic-incremental", "seeds": list(report.SEEDS),
              "bootstrap_replicates": 100, "bootstrap_seed": 8}
    records = []
    for architecture in report.ARCHITECTURES:
        for seed in report.SEEDS:
            predictions = []
            for row in tiny_data.splits["test"]:
                v, f, m = taxonomy.ids(row)
                ids = taxonomy.branches()["manufacturer_to_variant"][m]
                decisions = ((Decision("variant", None, v, "classifier"),) if architecture == "FLAT" else
                    (Decision("manufacturer", None, m, "classifier"), Decision("variant", m, v, "singleton" if len(ids) == 1 else "classifier")))
                prediction = Prediction(v, m, f, decisions,
                    1 if architecture != "IND-2" else sum(d.source == "classifier" for d in decisions))
                predictions.append(PredictionRecord(row.image_id, "test", architecture, "predicted", v, f, m, prediction))
            metrics = score(predictions, taxonomy)
            timings = [{"pass": i, "image_id": p.image_id, "full_pipeline_ms": 2., "model_path_ms": 1.,
                "route": "synthetic", "executed_macs": 10, "backbone_calls": p.prediction.backbone_calls, "singleton_bypasses": 0}
                for i in range(5) for p in predictions]
            passes = [{"pass": i, "images": len(predictions), "elapsed_seconds": 1., "sustained_fps": len(predictions)} for i in range(5)]
            records.append({"architecture": architecture, "seed": seed, "path": path, "run_id": f"synthetic-{architecture}-{seed}",
                "config_sha256": "synthetic", "model_sha256": "synthetic", "metrics": metrics, "predictions": predictions,
                "oracle": None if architecture == "FLAT" else metrics,
                "timing": aggregate_timings(timings, passes), "latencies": timings,
                "resources": {"training_run_id": "synthetic", "training_resources": {"training_wall_seconds": 2.},
                    "peak_gpu_allocated_bytes": 100, "stored_checkpoint_bytes": 100, "total_model_parameters": 10},
                "environment": {"torch": "synthetic", "cuda_runtime": None, "cudnn": None},
                "full_pipeline_boundary": "synthetic", "model_path_boundary": "synthetic", "fps_boundary": "synthetic"})
    groups = {name: {stage: {"macro_recall": .5, "micro_accuracy": .5} for stage in ("before", "after")}
              for name in ("new", "old_within_branch", "outside_branch")}
    updates = [{"architecture": a, "seed": seed, "predicted": {"groups": groups},
                "update": {"update_wall_seconds": 2., "trainable_parameters": 10, "replacement_bytes": 100}}
               for a in report.ARCHITECTURES[1:] for seed in report.SEEDS]
    monkeypatch.setattr(report, "load_sources", lambda *args: (records, taxonomy, updates))
    monkeypatch.setattr(report, "environment", lambda repo: {"git_sha": "synthetic-test-only"})
    return records, taxonomy, config


def test_depth_and_pass_tables_keep_observation_denominators(synthetic_sources):
    records, taxonomy, config = synthetic_sources
    values = report.calculate(records, taxonomy, [], config)
    assert sum(r["observations"] for r in values["depth_latency"]) == 5 * sum(len(r["predictions"]) for r in records)
    assert len(values["pass_latency"]) == 45
    assert all(sum(row[k] for k in ("correct", "within_family_error", "cross_family_same_manufacturer", "cross_manufacturer_error"))
               == len(records[0]["predictions"]) for row in values["family_error_breakdown"])


def test_publication_rebuild_and_resealed_tampering(synthetic_sources, tmp_path):
    pytest.importorskip("matplotlib")
    _, _, config = synthetic_sources
    published = report.publish(tmp_path, config, tmp_path / "published")
    assert report.audit_publication(tmp_path, published)["per_image_recomputed"]
    rebuilt = tmp_path / "rebuilt"
    report.rebuild(tmp_path, published, rebuilt)
    assert report.inventory(published) == report.inventory(rebuilt)
    doc = tmp_path / "docs/RESULTS.md"
    doc.parent.mkdir()
    report.export_results_doc(published, doc)
    text = doc.read_text(encoding="utf-8")
    assert "## 9. Reproduction" in text and "../published/accuracy-latency-bootstrap.svg" in text
    # Re-sealing checksums must not make a changed scientific value trustworthy.
    values = report.read_json(published / "analysis.json")
    values["bootstrap"]["per_seed"][0]["difference_pp"] = -90
    (published / "analysis.json").unlink()
    write_json(published / "analysis.json", values)
    (published / "summary.json").unlink()
    write_json(published / "summary.json", {"status": "complete", "kind": "phase8-analysis", "artifacts_sha256": report.inventory(published)})
    with pytest.raises(ValueError, match="per-image source"):
        report.audit_publication(tmp_path, published)


def test_source_manifest_change_fails_before_reading_runs(tmp_path):
    path = tmp_path / "source"
    path.mkdir()
    write_json(path / "publication.json", {"status": "synthetic"})
    config = {"benchmark": "source", "incremental": "unused", "benchmark_manifest_sha256": "changed"}
    with pytest.raises(ValueError, match="manifest changed"):
        report.load_sources(tmp_path, config)
