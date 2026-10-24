"""Canonical global label IDs built from image-ID-joined training annotations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class Labels:
    image_id: str
    variant: str
    family: str
    manufacturer: str


@dataclass(frozen=True)
class Taxonomy:
    """Names are sorted globally; maps use explicit IDs, never label syntax."""

    variants: tuple[str, ...]
    families: tuple[str, ...]
    manufacturers: tuple[str, ...]
    variant_to_family: tuple[int, ...]
    family_to_manufacturer: tuple[int, ...]

    @classmethod
    def from_training(cls, labels: Iterable[Labels]) -> Taxonomy:
        records = list(labels)
        if not records:
            raise ValueError("Training annotations are empty")
        variants = tuple(sorted({r.variant for r in records}))
        families = tuple(sorted({r.family for r in records}))
        manufacturers = tuple(sorted({r.manufacturer for r in records}))
        vf: dict[str, str] = {}
        fm: dict[str, str] = {}
        for row in records:
            if vf.setdefault(row.variant, row.family) != row.family:
                raise ValueError(f"Variant has multiple families: {row.variant}")
            if fm.setdefault(row.family, row.manufacturer) != row.manufacturer:
                raise ValueError(f"Family has multiple manufacturers: {row.family}")
        return cls(variants, families, manufacturers,
                   tuple(families.index(vf[v]) for v in variants),
                   tuple(manufacturers.index(fm[f]) for f in families))

    def ids(self, row: Labels) -> tuple[int, int, int]:
        """Validate a complete path against the training-only taxonomy."""
        try:
            v = self.variants.index(row.variant)
            f = self.families.index(row.family)
            m = self.manufacturers.index(row.manufacturer)
        except ValueError as exc:
            raise ValueError(f"Unknown label for image {row.image_id}") from exc
        if self.variant_to_family[v] != f or self.family_to_manufacturer[f] != m:
            raise ValueError(f"Hierarchy mismatch for image {row.image_id}")
        return v, f, m

    def branches(self) -> dict[str, tuple[tuple[int, ...], ...]]:
        """Return child global IDs in deterministic specialist-logit order."""
        return {
            "manufacturer_to_family": tuple(tuple(i for i, p in enumerate(self.family_to_manufacturer) if p == m)
                                             for m in range(len(self.manufacturers))),
            "manufacturer_to_variant": tuple(tuple(v for v, f in enumerate(self.variant_to_family)
                                                  if self.family_to_manufacturer[f] == m)
                                              for m in range(len(self.manufacturers))),
            "family_to_variant": tuple(tuple(v for v, p in enumerate(self.variant_to_family) if p == f)
                                      for f in range(len(self.families))),
        }

    def to_dict(self) -> dict:
        names = {"variant": self.variants, "family": self.families, "manufacturer": self.manufacturers}
        return {
            "schema_version": 1, "id_policy": "lexicographic names from train only",
            "classes": {level: {"id_to_name": list(values), "name_to_id": {n: i for i, n in enumerate(values)}}
                        for level, values in names.items()},
            "variant_to_family": list(self.variant_to_family),
            "family_to_manufacturer": list(self.family_to_manufacturer),
            "branches": self.branches(),
        }

    def branch_summary(self, splits: Mapping[str, list[Labels]]) -> dict:
        result = {}
        for kind, branches in self.branches().items():
            parent_is_family = kind == "family_to_variant"
            bypass = {}
            for split, rows in splits.items():
                parents = [self.ids(r)[1 if parent_is_family else 2] for r in rows]
                count = sum(len(branches[p]) == 1 for p in parents)
                bypass[split] = {"images": count, "total": len(rows), "rate": count / len(rows)}
            result[kind] = {"empty": sum(not b for b in branches),
                            "single_child": sum(len(b) == 1 for b in branches),
                            "multi_child": sum(len(b) > 1 for b in branches),
                            "true_route_bypass_potential": bypass}
        return result
