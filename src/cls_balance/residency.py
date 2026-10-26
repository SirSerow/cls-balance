"""Explicit independent-cascade loading policies and validation residency probes.

These synchronized, un-warmed probes are diagnostics, not Phase 6 test latency.
Cold-disk means no model object cached between calls; OS file caches are retained.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch
from torch import nn

from .artifacts import file_hash, object_hash, seed_everything, write_json
from .checkpoints import load_checkpoint
from .data import AircraftData, AircraftDataset
from .infer import HierarchicalAdapter, LogitNode
from .models.cascade import node_plan
from .prepare import write_csv
from .report_flat import read_json
from .train import synchronize
from .train_independent import audit_node, compatibility, load_cascade, unique_id
from .transforms import Preprocessing

POLICIES = ("all-resident", "preloaded-cpu", "cold-disk")


class OnDemandCNN(nn.Module):
    """One selected full CNN; load/transfer and evict inside the forward boundary."""

    def __init__(self, path: Path, hashes: dict, *, policy: str, checkpoint_sha256: str) -> None:
        super().__init__()
        if policy not in POLICIES[1:]:
            raise ValueError("On-demand policy must be preloaded-cpu or cold-disk")
        self.path, self.hashes, self.policy = path, hashes, policy
        if file_hash(path) != checkpoint_sha256:
            raise ValueError("Residency checkpoint checksum mismatch")
        # Bind the external weights into the common adapter's model fingerprint.
        self.register_buffer("checkpoint_digest", torch.tensor(list(bytes.fromhex(checkpoint_sha256)), dtype=torch.uint8))
        # CPU cache deliberately excluded from nn.Module registration so .to(cuda)
        # does not move every cached node to GPU. Its weights are never mutated.
        cached = load_checkpoint(path, **hashes, kind="IND-node")[0] if policy == "preloaded-cpu" else None
        object.__setattr__(self, "_cpu_model", cached)
        self.loads = self.transfers = self.calls = 0

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.policy == "cold-disk":
            model, _ = load_checkpoint(self.path, **self.hashes, kind="IND-node")
            self.loads += 1
        else:
            model = self._cpu_model
        self.calls += 1
        try:
            model.to(value.device).eval()
            self.transfers += int(value.device.type != "cpu")
            with torch.inference_mode():
                return model(value)
        finally:
            # Eviction completes before this node returns. GPU allocator caching
            # is retained and must not be mistaken for tensor residency.
            model.cpu()
            synchronize(value.device)


def cascade_with_policy(run: Path, depth: int, policy: str) -> HierarchicalAdapter:
    if policy not in POLICIES:
        raise ValueError("Unknown residency policy")
    if policy == "all-resident":
        return load_cascade(run, depth)
    from .taxonomy import Taxonomy

    bundle = read_json(run / f"bundle-ind{depth}.json")
    taxonomy = Taxonomy.from_dict(read_json(run / "taxonomy.json"))
    specs = node_plan(taxonomy, depth)
    if set(bundle["nodes"]) != {s.key for s in specs}:
        raise ValueError("Residency bundle node map mismatch")
    nodes = {}
    for spec in specs:
        entry = bundle["nodes"][spec.key]
        path = (run / entry["path"]).resolve()
        if not path.is_relative_to(run.resolve()):
            raise ValueError("Node path escapes run")
        audit_node(path)
        metadata = read_json(path / "node.json")["metadata"]
        if (object_hash(metadata["node"]) != object_hash(spec.to_dict())
                or metadata["seed"] != bundle["seed"] or metadata["run_id"] != bundle["run_id"]
                or any(metadata[k] != v for k, v in bundle["compatibility"].items())):
            raise ValueError("Residency node contract mismatch")
        nodes[spec.adapter_key] = LogitNode(OnDemandCNN(path / "node.pt", bundle["compatibility"],
            policy=policy, checkpoint_sha256=entry["checkpoint_sha256"]), spec.global_ids)
    return HierarchicalAdapter(taxonomy, nodes, depth=depth)


def probe(run: Path, data: AircraftData, depth: int, device: str, output_root: Path) -> Path:
    """One fixed validation image per true manufacturer; identical across policies.

    These images do not guarantee coverage of predicted routes. They are explicitly
    not a warmup, a complete split evaluation, a cold-filesystem test or a benchmark.
    """
    preprocessing = Preprocessing()
    bundle = read_json(run / f"bundle-ind{depth}.json")
    if bundle["compatibility"] != compatibility(data, preprocessing):
        raise ValueError("Residency probe data/preprocessing mismatch")
    seed_everything(bundle["seed"])
    dataset = AircraftDataset(data, "val", preprocessing)
    chosen = {}
    for index, row in enumerate(dataset.rows):
        chosen.setdefault(data.taxonomy.ids(row)[2], index)
    output = output_root / unique_id(f"residency-ind{depth}")
    output.mkdir(parents=True, exist_ok=False)
    dev = torch.device(device)
    results, references = [], {}
    for policy in POLICIES:
        if dev.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(dev)
        synchronize(dev)
        start = time.perf_counter()
        adapter = cascade_with_policy(run, depth, policy).to(dev).eval()
        synchronize(dev)
        setup_seconds = time.perf_counter() - start
        for index in chosen.values():
            sample = dataset[index]
            # Fixed ready tensor: excludes file decoding, preprocessing and input transfer.
            image = sample["image"].to(dev)
            synchronize(dev)
            start = time.perf_counter()
            prediction = adapter.predict(image)
            synchronize(dev)
            elapsed = time.perf_counter() - start
            if policy == "all-resident":
                references[sample["image_id"]] = prediction
            elif references[sample["image_id"]] != prediction:
                raise ValueError("Residency policies changed prediction or path")
            results.append({"policy": policy, "image_id": sample["image_id"], "split": "val",
                "variant_id": prediction.variant_id, "backbone_calls": prediction.backbone_calls,
                "singleton_bypasses": prediction.singleton_bypasses, "setup_seconds": setup_seconds,
                "unwarmed_model_path_ms": 1000 * elapsed,
                "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(dev) if dev.type == "cuda" else None})
        del adapter
    write_csv(output / "probes.csv", results)
    from .artifacts import environment
    env = environment(Path(__file__).resolve().parents[2])
    env.update(device=device, runtime_policy="validation residency probes: all-resident, preloaded-cpu, cold-disk; FP32; TF32 disabled",
               cpu_threads=torch.get_num_threads(), power_policy="OS/driver defaults; clocks not locked")
    write_json(output / "environment.json", env)
    write_json(output / "summary.json", {"status": "complete", "run_id": run.name,
        "architecture": f"IND-{depth}", "device": device, "precision": "fp32", "seed": bundle["seed"],
        "bundle_sha256": file_hash(run / f"bundle-ind{depth}.json"),
        "compatibility": bundle["compatibility"], "prediction_and_path_equality": True,
        "scope": "validation residency diagnostic; fixed tensor; no warmup; no test latency or FPS",
        "cold_disk_definition": "no model cache; OS filesystem cache and CUDA allocator cache retained",
        "images_per_policy": len(chosen), "policies": POLICIES,
        "artifacts_sha256": {name: file_hash(output / name) for name in ("probes.csv", "environment.json")}})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--depth", type=int, choices=(2, 3), default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=Path("runs/residency"))
    args = parser.parse_args()
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print(probe(args.run, AircraftData.load(args.data_root), args.depth, args.device, args.output_root))


if __name__ == "__main__":
    main()
