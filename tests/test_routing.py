"""Known-output modules verify routing without downloading pretrained weights."""

import pytest
import torch
from torch import nn

from cls_balance.infer import Ancestors, FlatAdapter, HierarchicalAdapter, LogitNode
from cls_balance.taxonomy import Taxonomy


class Constant(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", torch.tensor([logits], dtype=torch.float32))
        self.calls = 0
        self.inputs = []

    def forward(self, value):
        assert not self.training and torch.is_inference_mode_enabled()
        self.calls += 1
        self.inputs.append(value)
        return self.logits


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm1d(3)
        self.calls = 0

    def forward(self, image):
        assert not self.training and torch.is_inference_mode_enabled()
        self.calls += 1
        return self.bn(self.pool(image).flatten(1))


@pytest.fixture
def route_taxonomy():
    # Deliberately interleaved global IDs; M1 owns variants 1, 3, 4.
    return Taxonomy(("V0", "V1", "V2", "V3", "V4"), ("F0", "F1", "F2", "F3"),
                    ("M0", "M1"), (0, 1, 2, 1, 3), (0, 1, 0, 1))


@pytest.fixture
def image():
    return torch.zeros(3, 224, 224)


def nodes_for(taxonomy, depth, *, manufacturer=1, family=1, variant=3):
    def node(ids, chosen):
        # Reverse column order to catch assumed canonical/local ID coincidences.
        ids = tuple(reversed(ids))
        return LogitNode(Constant([float(i == chosen) for i in ids]), ids)
    nodes = {"manufacturer": node(range(len(taxonomy.manufacturers)), manufacturer)}
    branches = taxonomy.branches()
    if depth == 3:
        nodes.update({f"family:{p}": node(ids, family) for p, ids in enumerate(branches["manufacturer_to_family"])
                      if len(ids) > 1})
    kind = "manufacturer_to_variant" if depth == 2 else "family_to_variant"
    nodes.update({f"variant:{p}": node(ids, variant) for p, ids in enumerate(branches[kind]) if len(ids) > 1})
    return nodes


@pytest.mark.parametrize("depth", [2, 3])
@pytest.mark.parametrize("shared,fine_tuned", [(False, False), (True, False), (True, True)])
def test_global_ids_and_only_selected_nodes(route_taxonomy, image, depth, shared, fine_tuned):
    nodes = nodes_for(route_taxonomy, depth)
    encoder = Encoder() if shared else None
    adapter = HierarchicalAdapter(route_taxonomy, nodes, depth=depth, backbone=encoder, fine_tuned=fine_tuned)
    p = adapter.predict(image)
    assert (p.variant_id, p.family_id, p.manufacturer_id) == (3, 1, 1)
    assert p.backbone_calls == (1 if shared else depth)
    assert p.classifier_calls == depth
    called = {key for key, node in nodes.items() if node.classifier.calls}
    assert called == ({"manufacturer", "variant:1"} if depth == 2 else {"manufacturer", "family:1", "variant:1"})
    if shared:
        assert encoder.calls == 1
        selected_inputs = [node.classifier.inputs[0] for node in nodes.values() if node.classifier.calls]
        assert all(value is selected_inputs[0] for value in selected_inputs)
        assert selected_inputs[0].shape == (1, 3)
        assert all(not p.requires_grad for p in encoder.parameters())
    else:
        assert all(node.classifier.inputs[0].shape == (1, 3, 224, 224)
                   for node in nodes.values() if node.classifier.calls)


@pytest.mark.parametrize("depth", [2, 3])
def test_wrong_route_and_oracle(route_taxonomy, image, depth):
    nodes = nodes_for(route_taxonomy, depth, manufacturer=0)
    adapter = HierarchicalAdapter(route_taxonomy, nodes, depth=depth)
    hard = adapter.predict(image)
    assert hard.variant_id in (0, 2) and hard.variant_id != 3
    counts = {key: node.classifier.calls for key, node in nodes.items()}
    oracle = adapter.predict(image, routing="oracle", truth=Ancestors(1, 1))
    assert oracle.variant_id == 3
    assert nodes["manufacturer"].classifier.calls == counts["manufacturer"]
    assert oracle.backbone_calls == 1
    assert [d.source for d in oracle.decisions] == ["oracle"] * (depth - 1) + ["classifier"]


@pytest.mark.parametrize("depth", [2, 3])
def test_perfect_router_equals_oracle(route_taxonomy, image, depth):
    adapter = HierarchicalAdapter(route_taxonomy, nodes_for(route_taxonomy, depth), depth=depth)
    assert adapter.predict(image).variant_id == adapter.predict(image, routing="oracle", truth=Ancestors(1, 1)).variant_id


def test_singletons_execute_no_classifier(image):
    taxonomy = Taxonomy(("V",), ("F",), ("M",), (0,), (0,))
    independent = HierarchicalAdapter(taxonomy, {}, depth=3)
    p = independent.predict(image)
    assert p.variant_id == 0 and p.classifier_calls == p.backbone_calls == 0 and p.singleton_bypasses == 3
    shared = HierarchicalAdapter(taxonomy, {}, depth=3, backbone=Encoder())
    assert shared.predict(image).backbone_calls == shared.backbone.calls == 1
    assert shared.diagnose_family(image, 0).source == "singleton"
    assert shared.backbone.calls == 1


def test_singleton_variant_uses_global_id(route_taxonomy, image):
    adapter = HierarchicalAdapter(route_taxonomy, nodes_for(route_taxonomy, 3, family=3), depth=3)
    p = adapter.predict(image)
    assert p.variant_id == 4 and p.singleton_bypasses == 1 and p.backbone_calls == 2
    assert p.decisions[-1].source == "singleton"


def test_conditional_family_is_not_oracle_family(route_taxonomy, image):
    adapter = HierarchicalAdapter(route_taxonomy, nodes_for(route_taxonomy, 3, manufacturer=0, family=3), depth=3)
    assert adapter.predict(image).manufacturer_id == 0
    assert adapter.diagnose_family(image, 1).global_id == 3
    assert adapter.predict(image, routing="oracle", truth=Ancestors(1, 1)).family_id == 1


def test_frozen_encoder_parameters_and_buffers_unchanged(route_taxonomy, image):
    encoder = Encoder()
    before = {k: v.clone() for k, v in encoder.state_dict().items()}
    adapter = HierarchicalAdapter(route_taxonomy, nodes_for(route_taxonomy, 2), backbone=encoder)
    adapter.train()
    assert not encoder.training and all(not p.requires_grad for p in encoder.parameters())
    adapter.predict(image)
    for key, value in encoder.state_dict().items():
        assert torch.equal(value, before[key])


def test_flat_explicit_column_map(route_taxonomy, image):
    node = LogitNode(Constant([0, 1, 0, 0, 0]), (4, 3, 2, 1, 0))
    p = FlatAdapter(route_taxonomy, node).predict(image)
    assert (p.variant_id, p.family_id, p.manufacturer_id) == (3, 1, 1)


@pytest.mark.parametrize("kwargs", [{"truth": Ancestors(1, 1)}, {"routing": "oracle"},
    {"routing": "bad"}, {"routing": "oracle", "truth": Ancestors(0, 1)},
    {"routing": "oracle", "truth": Ancestors(True, 1)}])
def test_invalid_truth_and_routing(route_taxonomy, image, kwargs):
    adapter = HierarchicalAdapter(route_taxonomy, nodes_for(route_taxonomy, 2))
    with pytest.raises(ValueError):
        adapter.predict(image, **kwargs)


@pytest.mark.parametrize("value", [torch.zeros(1, 3, 224, 224), torch.zeros(3, 32, 32),
    torch.zeros(3, 224, 224, dtype=torch.int64), torch.full((3, 224, 224), float("nan"))])
def test_invalid_image(route_taxonomy, value):
    with pytest.raises(ValueError, match="preprocessed"):
        HierarchicalAdapter(route_taxonomy, nodes_for(route_taxonomy, 2)).predict(value)


@pytest.mark.parametrize("logits", [[1.0], [1.0, float("nan")], [1.0, float("inf")]])
def test_invalid_logits(route_taxonomy, image, logits):
    nodes = nodes_for(route_taxonomy, 2)
    nodes["manufacturer"] = LogitNode(Constant(logits), (0, 1))
    with pytest.raises(ValueError, match="finite batch-1 logits"):
        HierarchicalAdapter(route_taxonomy, nodes).predict(image)


@pytest.mark.parametrize("failure", ["missing", "extra", "wrong_map", "shared_module", "shared_storage"])
def test_invalid_nodes(route_taxonomy, failure):
    nodes = nodes_for(route_taxonomy, 2)
    if failure == "missing":
        del nodes["variant:1"]
    elif failure == "extra":
        nodes["extra"] = LogitNode(Constant([1]), (0,))
    elif failure == "wrong_map":
        nodes["variant:1"] = LogitNode(Constant([0, 1, 0]), (0, 1, 2))
    elif failure == "shared_module":
        nodes["variant:0"].classifier = nodes["manufacturer"].classifier
    else:
        nodes["variant:0"].classifier.logits = nodes["manufacturer"].classifier.logits.view(1, 2)
    with pytest.raises(ValueError):
        HierarchicalAdapter(route_taxonomy, nodes)


@pytest.mark.parametrize("ids", [(), (1, 1), (-1,), (True,)])
def test_invalid_logit_ids(ids):
    with pytest.raises(ValueError):
        LogitNode(Constant([1]), ids)


@pytest.mark.parametrize("kwargs", [{"depth": 1}, {"depth": 4}, {"fine_tuned": True}])
def test_invalid_architecture(route_taxonomy, kwargs):
    with pytest.raises(ValueError):
        HierarchicalAdapter(route_taxonomy, nodes_for(route_taxonomy, 2), **kwargs)


def test_reject_unpooled_shared_features(route_taxonomy, image):
    adapter = HierarchicalAdapter(route_taxonomy, nodes_for(route_taxonomy, 2), backbone=nn.Identity())
    with pytest.raises(ValueError, match="pooled embedding"):
        adapter.predict(image)


def test_flat_rejects_truth_and_incomplete_map(route_taxonomy, image):
    with pytest.raises(ValueError, match="all global variants"):
        FlatAdapter(route_taxonomy, LogitNode(Constant([0, 1]), (0, 1)))
    adapter = FlatAdapter(route_taxonomy, LogitNode(Constant([0, 1, 0, 0, 0]), (0, 1, 2, 3, 4)))
    with pytest.raises(ValueError, match="no oracle routing"):
        adapter.predict(image, truth=Ancestors(1, 1))
    with pytest.raises(ValueError, match="no oracle routing"):
        adapter.predict(image, routing="oracle", truth=Ancestors(1, 1))
