from dataclasses import replace

import pytest

from cls_balance.artifacts import object_hash
from cls_balance.taxonomy import Taxonomy


def test_canonical_ids_and_hash(tiny_data):
    taxonomy = tiny_data.taxonomy
    assert taxonomy.variants == ("Alpha jet", "Middle jet", "Zulu jet")
    assert taxonomy.variant_to_family == (1, 0, 0)
    assert taxonomy.family_to_manufacturer == (1, 0)
    reversed_taxonomy = Taxonomy.from_training(reversed(tiny_data.splits["train"]))
    assert object_hash(taxonomy.to_dict()) == object_hash(reversed_taxonomy.to_dict())


@pytest.mark.parametrize("level", ["family", "manufacturer"])
def test_conflicting_hierarchy(tiny_data, level):
    rows = tiny_data.splits["train"]
    with pytest.raises(ValueError, match="multiple"):
        Taxonomy.from_training([*rows, replace(rows[0], **{level: "conflict"})])


def test_test_labels_cannot_extend_taxonomy(tiny_data):
    with pytest.raises(ValueError, match="Unknown label"):
        tiny_data.taxonomy.ids(replace(tiny_data.splits["test"][0], variant="unseen"))


def test_branches_and_local_logit_order(tiny_data):
    branches = tiny_data.taxonomy.branches()
    assert branches["manufacturer_to_variant"] == ((0,), (1, 2))
    assert branches["family_to_variant"] == ((1, 2), (0,))
    summary = tiny_data.taxonomy.branch_summary(tiny_data.splits)
    assert summary["family_to_variant"]["empty"] == 0
    assert summary["family_to_variant"]["single_child"] == 1
    assert summary["family_to_variant"]["multi_child"] == 1
    assert summary["family_to_variant"]["true_route_bypass_potential"]["test"]["rate"] == 1 / 3
