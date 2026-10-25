from dataclasses import replace

import pytest

from cls_balance.infer import Decision, Prediction
from cls_balance.metrics import PredictionRecord, paired_scores, score, validate_record
from cls_balance.taxonomy import Taxonomy


@pytest.fixture
def metric_taxonomy():
    return Taxonomy(("V0", "V1", "V2", "V3"), ("F0", "F1", "F2"), ("M0", "M1"), (0, 0, 1, 2), (0, 0, 1))


def record(taxonomy, image_id, truth, predicted, *, architecture="IND-3", routing="predicted", conditional=None):
    f = taxonomy.variant_to_family[predicted]
    m = taxonomy.family_to_manufacturer[f]
    tf = taxonomy.variant_to_family[truth]
    tm = taxonomy.family_to_manufacturer[tf]
    if architecture == "FLAT":
        decisions = (Decision("variant", None, predicted, "classifier"),)
    else:
        branches = taxonomy.branches()
        decisions = [Decision("manufacturer", None, m, "oracle" if routing == "oracle" else "classifier")]
        if architecture.endswith("3"):
            decisions.append(Decision("family", m, f, "singleton" if len(branches["manufacturer_to_family"][m]) == 1
                                      else "oracle" if routing == "oracle" else "classifier"))
            parent, children = f, branches["family_to_variant"][f]
        else:
            parent, children = m, branches["manufacturer_to_variant"][m]
        decisions.append(Decision("variant", parent, predicted, "singleton" if len(children) == 1 else "classifier"))
        decisions = tuple(decisions)
    calls = 1 if architecture == "FLAT" or "FROZEN" in architecture else sum(d.source == "classifier" for d in decisions)
    p = Prediction(predicted, m, f, decisions, calls)
    return PredictionRecord(f"{image_id:07d}", "test", architecture, routing, truth, tf, tm, p, conditional)


@pytest.fixture
def known_records(metric_taxonomy):
    # Imbalanced support: V0 twice, others once. First errors: M, F, V.
    return tuple(record(metric_taxonomy, i, t, p, conditional=c) for i, (t, p, c) in enumerate(
        [(0, 0, 0), (0, 3, 0), (1, 0, 1), (2, 0, 1), (3, 3, 2)]))


def test_exact_metrics_and_error_attribution(metric_taxonomy, known_records):
    m = score(known_records, metric_taxonomy)
    assert m["variant_macro_recall"] == pytest.approx(0.375)
    assert m["variant_top1"] == pytest.approx(0.4)
    assert m["variant_macro_f1"] == pytest.approx((0.4 + 0 + 0 + 2 / 3) / 4)
    assert m["confusion_matrix"] == [[1, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
    assert m["classwise"][1]["precision"] == 0
    assert m["family_from_variant"]["accuracy"] == 3 / 5
    assert m["manufacturer_from_variant"]["accuracy"] == 4 / 5
    d = m["routing_diagnostics"]
    assert d["first_wrong_stage"] == {"manufacturer": 1, "family": 1, "variant": 1, "correct": 2}
    assert d["manufacturer_router_macro_recall"] == (3 / 4 + 1) / 2
    assert d["family_under_true_manufacturer"] == {"correct": 4, "support": 5, "accuracy": 4 / 5}
    assert d["family_given_correct_predicted_manufacturer"] == {"correct": 3, "support": 4, "accuracy": 3 / 4}
    assert d["singleton_bypasses"]["family"]["bypassed"] == 2
    assert d["singleton_bypasses"]["variant"]["rate"] == 2 / 5
    assert d["executed_backbone_calls"] == {1: 2, 3: 3}


def test_exact_oracle_penalty(metric_taxonomy, known_records):
    oracle = tuple(record(metric_taxonomy, i, t, p, routing="oracle") for i, (t, p) in enumerate(
        [(0, 0), (0, 0), (1, 0), (2, 2), (3, 3)]))
    result = paired_scores(known_records, tuple(reversed(oracle)), metric_taxonomy)
    assert result["oracle_diagnostic"]["variant_macro_recall"] == 3 / 4
    assert result["routing_penalty_macro_recall_pp"] == 37.5
    assert result["routing_penalty_top1_pp"] == 40
    d = result["oracle_diagnostic"]["routing_diagnostics"]
    assert d["manufacturer_router"] is None
    assert d["family_given_correct_predicted_manufacturer"] is None
    assert d["first_wrong_stage"] == {"manufacturer": 0, "family": 0, "variant": 1, "correct": 4}


@pytest.mark.parametrize("architecture", ["FLAT", "IND-2", "IND-3", "FROZEN-2", "FROZEN-3", "FT-FROZEN-2", "FT-FROZEN-3"])
def test_perfect_metrics(metric_taxonomy, architecture):
    rows = [record(metric_taxonomy, i, i, i, architecture=architecture) for i in range(4)]
    result = score(rows, metric_taxonomy)
    assert result["variant_macro_recall"] == result["variant_top1"] == result["variant_macro_f1"] == 1
    assert result["routing_diagnostics"]["first_wrong_stage"]["correct"] == 4
    if architecture == "FLAT":
        assert result["routing_diagnostics"]["manufacturer_router"] is None


@pytest.mark.parametrize("failure", ["empty", "missing_class", "duplicate", "mixed_mode", "mixed_split", "partial_conditional"])
def test_invalid_record_sets(metric_taxonomy, known_records, failure):
    rows = list(known_records)
    if failure == "empty":
        rows = []
    elif failure == "missing_class":
        rows.pop()
    elif failure == "duplicate":
        rows.append(rows[0])
    elif failure == "mixed_mode":
        rows[0] = record(metric_taxonomy, 0, 0, 0, routing="oracle")
    elif failure == "mixed_split":
        rows[0] = replace(rows[0], split="val")
    else:
        rows[0] = replace(rows[0], conditional_family_id=None)
    with pytest.raises(ValueError):
        score(rows, metric_taxonomy)


@pytest.mark.parametrize("failure", ["wrong_global", "inconsistent_route", "bad_source", "bad_parent", "bad_calls", "bad_truth", "bad_oracle", "bad_conditional"])
def test_invalid_record_paths(metric_taxonomy, known_records, failure):
    row = known_records[0]
    p = row.prediction
    if failure == "wrong_global":
        p = replace(p, variant_id=99)
    elif failure == "inconsistent_route":
        p = replace(p, manufacturer_id=1)
    elif failure in ("bad_source", "bad_parent"):
        decision = replace(p.decisions[-1], **({"source": "oracle"} if failure == "bad_source" else {"parent_id": 2}))
        p = replace(p, decisions=(*p.decisions[:-1], decision))
    elif failure == "bad_calls":
        p = replace(p, backbone_calls=1)
    elif failure == "bad_truth":
        row = replace(row, true_family_id=2)
    elif failure == "bad_oracle":
        row = record(metric_taxonomy, 0, 0, 3, routing="oracle")
        p = row.prediction
    else:
        row = replace(row, conditional_family_id=2)
    with pytest.raises(ValueError):
        validate_record(replace(row, prediction=p), metric_taxonomy)


def test_pairing_rejects_different_image_sets(metric_taxonomy, known_records):
    oracle = tuple(record(metric_taxonomy, i + 10, t, t, routing="oracle") for i, t in enumerate([0, 0, 1, 2, 3]))
    with pytest.raises(ValueError, match="same image IDs"):
        paired_scores(known_records, oracle, metric_taxonomy)


def test_prediction_csv_roundtrip_immutable(tmp_path, metric_taxonomy, known_records):
    from cls_balance.predictions import read_predictions, write_predictions

    path = tmp_path / "predictions.csv"
    write_predictions(path, known_records, metric_taxonomy)
    loaded = read_predictions(path, metric_taxonomy)
    assert loaded == known_records
    assert loaded[0].image_id == "0000000"
    assert score(loaded, metric_taxonomy) == score(known_records, metric_taxonomy)
    with pytest.raises(FileExistsError):
        write_predictions(path, known_records, metric_taxonomy)


@pytest.mark.parametrize("failure", ["header", "version", "duplicate", "global_id", "json", "empty", "extra_column", "missing_column"])
def test_corrupt_prediction_csv(tmp_path, metric_taxonomy, known_records, failure):
    import csv
    from cls_balance.predictions import COLUMNS, read_predictions, write_predictions

    path = tmp_path / "predictions.csv"
    write_predictions(path, known_records, metric_taxonomy)
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    if failure == "header":
        rows[0][0] = "unknown"
    elif failure == "version":
        rows[1][0] = "999"
    elif failure == "duplicate":
        rows.append(rows[1])
    elif failure == "global_id":
        rows[1][COLUMNS.index("predicted_variant_id")] = "99"
    elif failure == "json":
        rows[1][COLUMNS.index("decisions_json")] = "null"
    elif failure == "empty":
        rows = rows[:1]
    elif failure == "extra_column":
        rows[1].append("extra")
    else:
        rows[1].pop()
    with path.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows(rows)
    with pytest.raises(ValueError):
        read_predictions(path, metric_taxonomy)


def test_macro_metrics_match_independent_reference(metric_taxonomy):
    from sklearn.metrics import accuracy_score, f1_score, recall_score

    truth = [0, 1, 2, 3] + [0] * 17 + [1] * 9 + [2] * 2
    prediction = [(i * 3 + 2) % 4 for i in range(len(truth))]
    rows = [record(metric_taxonomy, i, t, p, architecture="FLAT") for i, (t, p) in enumerate(zip(truth, prediction, strict=True))]
    metrics = score(rows, metric_taxonomy)
    assert metrics["variant_top1"] == accuracy_score(truth, prediction)
    assert metrics["variant_macro_recall"] == recall_score(truth, prediction, labels=[0, 1, 2, 3], average="macro")
    assert metrics["variant_macro_f1"] == f1_score(truth, prediction, labels=[0, 1, 2, 3], average="macro")
