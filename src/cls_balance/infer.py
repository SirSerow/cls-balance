"""Single-image adapters for injected classifiers; training is a later phase.

Inputs are preprocessed CHW tensors. Logit columns carry explicit global IDs.
Oracle routing is a separate, named diagnostic and never accepts variant truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn

from .taxonomy import Taxonomy


@dataclass(frozen=True)
class Ancestors:
    manufacturer_id: int
    family_id: int


@dataclass(frozen=True)
class Decision:
    stage: str
    parent_id: int | None
    global_id: int
    source: str  # classifier, singleton, or oracle


@dataclass(frozen=True)
class Prediction:
    variant_id: int
    manufacturer_id: int
    family_id: int
    decisions: tuple[Decision, ...]
    backbone_calls: int

    @property
    def classifier_calls(self) -> int:
        return sum(d.source == "classifier" for d in self.decisions)

    @property
    def singleton_bypasses(self) -> int:
        return sum(d.source == "singleton" for d in self.decisions)


class LogitNode(nn.Module):
    """Classifier whose output column order is explicitly mapped to global IDs."""

    def __init__(self, classifier: nn.Module, global_ids: tuple[int, ...]) -> None:
        super().__init__()
        if (not global_ids or any(type(i) is not int or i < 0 for i in global_ids)
                or len(set(global_ids)) != len(global_ids)):
            raise ValueError("Logit IDs must be unique nonnegative integers")
        self.classifier = classifier
        self.global_ids = tuple(global_ids)

    def select(self, value: torch.Tensor) -> int:
        logits = self.classifier(value)
        if (not isinstance(logits, torch.Tensor) or logits.shape != (1, len(self.global_ids))
                or not logits.is_floating_point() or not torch.isfinite(logits).all()):
            raise ValueError("Expected finite batch-1 logits matching the explicit global ID map")
        return self.global_ids[int(logits.argmax(dim=1).item())]


def _independent(modules: list[nn.Module]) -> None:
    """Reject shared modules/parameter or buffer storage between independent nodes."""
    objects: set[int] = set()
    storage: set[tuple[str, int]] = set()
    for module in modules:
        current_objects = {id(m) for m in module.modules()}
        current_storage = {(str(t.device), t.untyped_storage().data_ptr())
                           for t in (*module.parameters(), *module.buffers()) if t.numel()}
        if objects & current_objects or storage & current_storage:
            raise ValueError("Independent nodes must not share modules, parameters or buffers")
        objects.update(current_objects)
        storage.update(current_storage)


class InferenceAdapter(nn.Module):
    """Common contract: one preprocessed image, one globally valid prediction."""

    architecture: str
    depth: int

    def __init__(self, taxonomy: Taxonomy) -> None:
        super().__init__()
        self.taxonomy = taxonomy

    @staticmethod
    def _input(image: torch.Tensor) -> torch.Tensor:
        if (not isinstance(image, torch.Tensor) or image.shape != (3, 224, 224)
                or not image.is_floating_point() or not torch.isfinite(image).all()):
            raise ValueError("Expected one finite, preprocessed floating-point 3x224x224 image")
        return image.unsqueeze(0)

    def predict(self, image: torch.Tensor, *, routing: str = "predicted",
                truth: Ancestors | None = None) -> Prediction:
        raise NotImplementedError


class FlatAdapter(InferenceAdapter):
    architecture = "FLAT"
    depth = 1

    def __init__(self, taxonomy: Taxonomy, node: LogitNode) -> None:
        super().__init__(taxonomy)
        if set(node.global_ids) != set(range(len(taxonomy.variants))):
            raise ValueError("FLAT logits must cover all global variants")
        self.node = node

    @torch.inference_mode()
    def predict(self, image: torch.Tensor, *, routing: str = "predicted",
                truth: Ancestors | None = None) -> Prediction:
        if routing != "predicted" or truth is not None:
            raise ValueError("FLAT has no oracle routing; predicted inference cannot receive truth")
        self.eval()
        v = self.node.select(self._input(image))
        f = self.taxonomy.variant_to_family[v]
        return Prediction(v, self.taxonomy.family_to_manufacturer[f], f,
                          (Decision("variant", None, v, "classifier"),), 1)


class HierarchicalAdapter(InferenceAdapter):
    """Hard routing over independently supplied full CNNs or shared-feature heads.

    Keys are manufacturer, family:<manufacturer ID>, variant:<parent ID>.
    Variant parents are manufacturers at depth 2 and families at depth 3.
    Only multi-child nodes may have classifiers. Shared heads receive the same
    pooled feature tensor; the frozen encoder stays in eval even in train mode.
    """

    def __init__(self, taxonomy: Taxonomy, nodes: Mapping[str, LogitNode], *,
                 depth: int = 2, backbone: nn.Module | None = None,
                 fine_tuned: bool = False) -> None:
        super().__init__(taxonomy)
        if depth not in (2, 3) or (fine_tuned and backbone is None):
            raise ValueError("Depth must be 2 or 3; fine-tuned shared models require a backbone")
        self.depth = depth
        self.architecture = f'{"FT-FROZEN" if fine_tuned else "FROZEN" if backbone is not None else "IND"}-{depth}'
        self.branch_ids = taxonomy.branches()
        expected = {"manufacturer": tuple(range(len(taxonomy.manufacturers)))}
        if depth == 3:
            expected.update({f"family:{m}": ids for m, ids in enumerate(self.branch_ids["manufacturer_to_family"])})
        branch = "manufacturer_to_variant" if depth == 2 else "family_to_variant"
        expected.update({f"variant:{p}": ids for p, ids in enumerate(self.branch_ids[branch])})
        required = {key: ids for key, ids in expected.items() if len(ids) > 1}
        if set(nodes) != set(required):
            raise ValueError(f"Expected precisely the multi-child nodes: {sorted(required)}")
        for key, ids in required.items():
            if set(nodes[key].global_ids) != set(ids):
                raise ValueError(f"Invalid global logit map for {key}")
        _independent([*nodes.values(), *([] if backbone is None else [backbone])])
        self.nodes = nn.ModuleDict(nodes)
        self.backbone = backbone
        if self.backbone is not None:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode: bool = True) -> HierarchicalAdapter:
        super().train(mode)
        if self.backbone is not None:
            self.backbone.eval()
        return self

    def _value(self, image: torch.Tensor) -> torch.Tensor:
        value = self._input(image)
        if self.backbone is not None:
            value = self.backbone(value)
            if (not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[0] != 1
                    or value.shape[1] < 1 or not torch.isfinite(value).all()):
                raise ValueError("Shared backbone must return one finite pooled embedding [1, features]")
        return value

    def _select(self, stage: str, parent: int | None, ids: tuple[int, ...],
                value: torch.Tensor, forced: int | None = None) -> Decision:
        if len(ids) == 1:
            return Decision(stage, parent, ids[0], "singleton")
        if forced is not None:
            if forced not in ids:
                raise ValueError("Oracle ancestor is outside the selected branch")
            return Decision(stage, parent, forced, "oracle")
        key = stage if parent is None else f"{stage}:{parent}"
        return Decision(stage, parent, self.nodes[key].select(value), "classifier")

    def _truth(self, truth: Ancestors) -> None:
        m, f = truth.manufacturer_id, truth.family_id
        if (type(m) is not int or type(f) is not int or not 0 <= m < len(self.taxonomy.manufacturers)
                or not 0 <= f < len(self.taxonomy.families) or self.taxonomy.family_to_manufacturer[f] != m):
            raise ValueError("Invalid oracle ancestor path")

    @torch.inference_mode()
    def predict(self, image: torch.Tensor, *, routing: str = "predicted",
                truth: Ancestors | None = None) -> Prediction:
        if routing not in ("predicted", "oracle") or ((routing == "oracle") != (truth is not None)):
            raise ValueError("Only named oracle diagnostics receive ancestor truth")
        if truth is not None:
            self._truth(truth)
        self.eval()
        value = self._value(image)
        decisions = [self._select("manufacturer", None, tuple(range(len(self.taxonomy.manufacturers))),
                                  value, None if truth is None else truth.manufacturer_id)]
        m = decisions[0].global_id
        if self.depth == 3:
            decisions.append(self._select("family", m, self.branch_ids["manufacturer_to_family"][m],
                                          value, None if truth is None else truth.family_id))
            f = decisions[-1].global_id
            decisions.append(self._select("variant", f, self.branch_ids["family_to_variant"][f], value))
        else:
            decisions.append(self._select("variant", m, self.branch_ids["manufacturer_to_variant"][m], value))
        v = decisions[-1].global_id
        f = self.taxonomy.variant_to_family[v]
        calls = 1 if self.backbone is not None else sum(d.source == "classifier" for d in decisions)
        return Prediction(v, m, f, tuple(decisions), calls)

    @torch.inference_mode()
    def diagnose_family(self, image: torch.Tensor, manufacturer_id: int) -> Decision:
        """Separate diagnostic pass: family router under the TRUE manufacturer.

        Excluded from production path counts and any future latency boundary.
        This is not the family output obtained by forcing both oracle ancestors.
        """
        if self.depth != 3 or type(manufacturer_id) is not int or not 0 <= manufacturer_id < len(self.taxonomy.manufacturers):
            raise ValueError("Conditional family diagnosis requires depth 3 and a valid manufacturer")
        self.eval()
        ids = self.branch_ids["manufacturer_to_family"][manufacturer_id]
        if len(ids) == 1:
            self._input(image)
            return Decision("family", manufacturer_id, ids[0], "singleton")
        return self._select("family", manufacturer_id, ids, self._value(image))
