"""Independent full-CNN node plans and train/validation-only local targets."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from torch.utils.data import Dataset

from ..data import AircraftDataset
from ..taxonomy import Taxonomy


@dataclass(frozen=True)
class NodeSpec:
    """Classifier columns are explicit global IDs at one taxonomy level."""

    key: str
    stage: str
    parent_stage: str | None
    parent_id: int | None
    global_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if (self.stage not in {"manufacturer", "family", "variant"}
                or len(self.global_ids) < 2
                or tuple(sorted(set(self.global_ids))) != self.global_ids
                or any(type(i) is not int or i < 0 for i in self.global_ids)):
            raise ValueError("Invalid multi-child node/global ID map")
        if self.stage == "manufacturer":
            valid = self.key == "manufacturer" and self.parent_stage is None and self.parent_id is None
        else:
            valid = (self.parent_stage in ({"manufacturer"} if self.stage == "family" else {"manufacturer", "family"})
                     and type(self.parent_id) is int and self.parent_id >= 0
                     and self.key == f"{self.stage}-{self.parent_stage}-{self.parent_id}")
        if not valid:
            raise ValueError("Invalid node parent/key")

    def to_dict(self) -> dict:
        return {**asdict(self), "global_ids": list(self.global_ids)}

    @classmethod
    def from_dict(cls, value: dict) -> NodeSpec:
        return cls(**{**value, "global_ids": tuple(value["global_ids"])})

    @property
    def adapter_key(self) -> str:
        return self.stage if self.parent_id is None else f"{self.stage}:{self.parent_id}"


def node_plan(taxonomy: Taxonomy, depth: int) -> tuple[NodeSpec, ...]:
    """Only multi-child nodes; canonical order is independent of annotation order."""
    if depth not in (2, 3):
        raise ValueError("Independent depth must be 2 or 3")
    branches = taxonomy.branches()
    nodes = []
    if len(taxonomy.manufacturers) > 1:
        nodes.append(NodeSpec("manufacturer", "manufacturer", None, None,
                              tuple(range(len(taxonomy.manufacturers)))))
    levels = [("family", "manufacturer", "manufacturer_to_family")] if depth == 3 else []
    levels.append(("variant", "manufacturer" if depth == 2 else "family",
                   "manufacturer_to_variant" if depth == 2 else "family_to_variant"))
    for stage, parent, branch in levels:
        for parent_id, ids in enumerate(branches[branch]):
            if len(ids) > 1:
                nodes.append(NodeSpec(f"{stage}-{parent}-{parent_id}", stage, parent, parent_id, ids))
    return tuple(nodes)


class LocalDataset(Dataset):
    """Filter by true TRAIN/VAL parent, remap global targets to classifier columns.

    Inference never uses this supervised branch filter. Test use is forbidden.
    All samples within the official branch are used; no replacement/subsampling.
    """

    def __init__(self, base: AircraftDataset, spec: NodeSpec, *, split: str) -> None:
        if split not in {"train", "val"}:
            raise ValueError("Node training/selection accepts train/val only")
        self.base, self.spec = base, spec
        self.indices = []
        self.local_ids = {global_id: local for local, global_id in enumerate(spec.global_ids)}
        self.targets = []
        for index, row in enumerate(base.rows):
            v, f, m = base.taxonomy.ids(row)
            labels = {"variant": v, "family": f, "manufacturer": m}
            if spec.parent_stage is None or labels[spec.parent_stage] == spec.parent_id:
                if labels[spec.stage] not in self.local_ids:
                    raise ValueError("Node labels disagree with taxonomy")
                self.indices.append(index)
                self.targets.append(self.local_ids[labels[spec.stage]])
        if set(self.targets) != set(range(len(spec.global_ids))):
            raise ValueError("Missing local class support")

    @property
    def image_ids(self) -> list[str]:
        return [self.base.rows[i].image_id for i in self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        return {**self.base[self.indices[index]], "variant_id": self.targets[index]}
