"""Globally aligned classification metrics and auditable routing diagnostics.

All scores are fractions in [0, 1]. Missing class support is an error, not a
silently shortened macro average. Oracle scores are diagnostic only.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Sequence

from .infer import Prediction
from .taxonomy import Taxonomy

ARCHITECTURES = {"FLAT": 1, "IND-2": 2, "IND-3": 3, "FROZEN-2": 2,
                 "FROZEN-3": 3, "FT-FROZEN-2": 2, "FT-FROZEN-3": 3}


@dataclass(frozen=True)
class PredictionRecord:
    image_id: str
    split: str
    architecture: str
    routing: str
    true_variant_id: int
    true_family_id: int
    true_manufacturer_id: int
    prediction: Prediction
    conditional_family_id: int | None = None


def validate_record(row: PredictionRecord, taxonomy: Taxonomy) -> None:
    """Reject impossible paths, malformed diagnostics and local/global ID mixups."""
    if (not re.fullmatch(r"[0-9]{7}", row.image_id) or row.split not in ("train", "val", "test")
            or row.architecture not in ARCHITECTURES or row.routing not in ("predicted", "oracle")):
        raise ValueError("Invalid prediction identity, split, architecture or routing")
    depth = ARCHITECTURES[row.architecture]
    if depth == 1 and row.routing != "predicted":
        raise ValueError("FLAT has no oracle routing")
    p = row.prediction
    for v, f, m in ((row.true_variant_id, row.true_family_id, row.true_manufacturer_id),
                    (p.variant_id, p.family_id, p.manufacturer_id)):
        if (any(type(i) is not int for i in (v, f, m)) or not 0 <= v < len(taxonomy.variants)
                or not 0 <= f < len(taxonomy.families) or not 0 <= m < len(taxonomy.manufacturers)
                or taxonomy.variant_to_family[v] != f or taxonomy.family_to_manufacturer[f] != m):
            raise ValueError("Invalid global variant/family/manufacturer path")
    branches = taxonomy.branches()
    if depth == 1:
        expected = [("variant", None, p.variant_id, tuple(range(len(taxonomy.variants))))]
    else:
        expected = [("manufacturer", None, p.manufacturer_id, tuple(range(len(taxonomy.manufacturers))))]
        if depth == 3:
            expected.append(("family", p.manufacturer_id, p.family_id,
                             branches["manufacturer_to_family"][p.manufacturer_id]))
        parent = p.manufacturer_id if depth == 2 else p.family_id
        children = branches["manufacturer_to_variant" if depth == 2 else "family_to_variant"][parent]
        expected.append(("variant", parent, p.variant_id, children))
    if len(p.decisions) != len(expected):
        raise ValueError("Prediction has an invalid decision depth")
    for decision, (stage, parent, selected, ids) in zip(p.decisions, expected, strict=True):
        source = ("singleton" if len(ids) == 1 and depth > 1 else
                  "oracle" if row.routing == "oracle" and stage != "variant" else "classifier")
        if (type(decision.global_id) is not int or (decision.parent_id is not None and type(decision.parent_id) is not int)
                or (decision.stage, decision.parent_id, decision.global_id, decision.source) != (stage, parent, selected, source)):
            raise ValueError("Invalid selected branch, logit mapping or decision source")
    calls = 1 if depth == 1 or "FROZEN" in row.architecture else p.classifier_calls
    if type(p.backbone_calls) is not int or p.backbone_calls != calls:
        raise ValueError("Executed backbone count disagrees with the architecture/path")
    if row.routing == "oracle" and (p.manufacturer_id != row.true_manufacturer_id or
                                   (depth == 3 and p.family_id != row.true_family_id)):
        raise ValueError("Oracle route must use the true ancestors")
    if row.conditional_family_id is not None:
        ids = branches["manufacturer_to_family"][row.true_manufacturer_id]
        if depth != 3 or type(row.conditional_family_id) is not int or row.conditional_family_id not in ids:
            raise ValueError("Invalid conditional family diagnostic")


def _accuracy(correct: int, support: int) -> dict:
    return {"correct": correct, "support": support, "accuracy": correct / support if support else None}


def score(records: Sequence[PredictionRecord], taxonomy: Taxonomy) -> dict:
    """Score one complete split/mode with a fixed global class list."""
    if not records:
        raise ValueError("Cannot score empty predictions")
    identities = {(r.split, r.architecture, r.routing) for r in records}
    if len(identities) != 1 or len({r.image_id for r in records}) != len(records):
        raise ValueError("Predictions must have unique IDs and one split, architecture and routing mode")
    for row in records:
        validate_record(row, taxonomy)
    count = len(taxonomy.variants)
    matrix = [[0] * count for _ in range(count)]
    for r in records:
        matrix[r.true_variant_id][r.prediction.variant_id] += 1
    supports = [sum(row) for row in matrix]
    if not all(supports):
        raise ValueError("Missing variant support; refusing a partial macro average")
    classwise = []
    for v, support in enumerate(supports):
        tp = matrix[v][v]
        predicted = sum(row[v] for row in matrix)
        classwise.append({"variant_id": v, "variant_name": taxonomy.variants[v], "support": support,
                          "predicted_support": predicted, "true_positive": tp, "recall": tp / support,
                          "precision": tp / predicted if predicted else 0.0,
                          "f1": 2 * tp / (support + predicted)})
    n = len(records)
    depth = ARCHITECTURES[records[0].architecture]
    oracle = records[0].routing == "oracle"
    wrong = Counter({"manufacturer": 0, "family": 0, "variant": 0, "correct": 0})
    for r in records:
        p = r.prediction
        if depth > 1 and p.manufacturer_id != r.true_manufacturer_id:
            wrong["manufacturer"] += 1
        elif depth == 3 and p.family_id != r.true_family_id:
            wrong["family"] += 1
        elif p.variant_id != r.true_variant_id:
            wrong["variant"] += 1
        else:
            wrong["correct"] += 1
    manufacturer = _accuracy(sum(r.prediction.manufacturer_id == r.true_manufacturer_id for r in records), n)
    family = _accuracy(sum(r.prediction.family_id == r.true_family_id for r in records), n)
    by_manufacturer = []
    for m in range(len(taxonomy.manufacturers)):
        rows = [r for r in records if r.true_manufacturer_id == m]
        by_manufacturer.append({"manufacturer_id": m, **_accuracy(
            sum(r.prediction.manufacturer_id == m for r in rows), len(rows))})
    conditional = [r for r in records if r.conditional_family_id is not None]
    if conditional and len(conditional) != n:
        raise ValueError("Conditional family diagnostics must cover all images or none")
    correct_manufacturer = [r for r in records if r.prediction.manufacturer_id == r.true_manufacturer_id]
    bypass = {}
    for stage in ("manufacturer", "family", "variant"):
        decisions = [d for r in records for d in r.prediction.decisions if d.stage == stage]
        bypass[stage] = {"bypassed": sum(d.source == "singleton" for d in decisions),
                         "support": len(decisions),
                         "rate": sum(d.source == "singleton" for d in decisions) / len(decisions) if decisions else None}
    branches = {}
    for stage in ("manufacturer", "family", "variant"):
        counts = Counter((d.parent_id, d.global_id) for r in records for d in r.prediction.decisions if d.stage == stage)
        branches[stage] = [{"parent_id": p, "selected_global_id": g, "support": support}
                           for (p, g), support in sorted(counts.items(), key=lambda item: (
                               -1 if item[0][0] is None else item[0][0], item[0][1]))]
    diagnostics = {
        "manufacturer_router": manufacturer if depth > 1 and not oracle else None,
        "manufacturer_router_macro_recall": (sum(r["accuracy"] for r in by_manufacturer) / len(by_manufacturer)
                                              if depth > 1 and not oracle else None),
        "manufacturer_router_classwise": by_manufacturer if depth > 1 and not oracle else None,
        "family_under_true_manufacturer": (_accuracy(sum(r.conditional_family_id == r.true_family_id for r in conditional), n)
                                           if conditional else None),
        "family_given_correct_predicted_manufacturer": (_accuracy(
            sum(r.prediction.family_id == r.true_family_id for r in correct_manufacturer), len(correct_manufacturer))
            if depth == 3 and not oracle else None),
        "first_wrong_stage": dict(wrong), "singleton_bypasses": bypass, "selected_branch_support": branches,
        "executed_classifier_calls": dict(sorted(Counter(r.prediction.classifier_calls for r in records).items())),
        "executed_backbone_calls": dict(sorted(Counter(r.prediction.backbone_calls for r in records).items())),
        "conditional_family_note": "Separate family-router diagnostic under true manufacturer; excluded from production path counts.",
    }
    return {"schema_version": 1, "score_units": "fraction", "images": n, "split": records[0].split,
            "architecture": records[0].architecture, "routing": records[0].routing,
            "diagnostic_only": oracle, "variant_macro_recall": sum(c["recall"] for c in classwise) / count,
            "variant_top1": sum(matrix[v][v] for v in range(count)) / n,
            "variant_macro_f1": sum(c["f1"] for c in classwise) / count,
            "family_from_variant": family, "manufacturer_from_variant": manufacturer,
            "confusion_matrix": matrix, "classwise": classwise, "routing_diagnostics": diagnostics}


def paired_scores(predicted: Sequence[PredictionRecord], oracle: Sequence[PredictionRecord], taxonomy: Taxonomy) -> dict:
    """Compute oracle minus hard-route penalties only on exactly paired images."""
    hard, diagnostic = score(predicted, taxonomy), score(oracle, taxonomy)
    if (hard["routing"] != "predicted" or diagnostic["routing"] != "oracle"
            or hard["architecture"] != diagnostic["architecture"] or hard["split"] != diagnostic["split"]):
        raise ValueError("Routing penalty requires a matching hard/oracle architecture and split")
    def truths(rows: Sequence[PredictionRecord]) -> dict:
        return {r.image_id: (r.true_variant_id, r.true_family_id, r.true_manufacturer_id) for r in rows}
    if truths(predicted) != truths(oracle):
        raise ValueError("Hard/oracle predictions must contain exactly the same image IDs and truth")
    return {"predicted": hard, "oracle_diagnostic": diagnostic,
            "routing_penalty_macro_recall_pp": 100 * (diagnostic["variant_macro_recall"] - hard["variant_macro_recall"]),
            "routing_penalty_top1_pp": 100 * (diagnostic["variant_top1"] - hard["variant_top1"])}
