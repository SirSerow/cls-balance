"""Exact SVG schematics derived from deployed taxonomy and trained-node inventory."""

from __future__ import annotations

from html import escape
from pathlib import Path
import textwrap

from .artifacts import write_json
from .models.cascade import node_plan
from .taxonomy import Taxonomy


def network_inventory(taxonomy: Taxonomy, architecture: str, *, active_variant_ids: tuple[int, ...] | None = None) -> dict:
    """Every classifier and deterministic branch, with explicit global outputs."""
    if architecture == "FLAT":
        return {"architecture": architecture, "backbones_stored": 1,
                "nodes": [{"key": "variant", "kind": "EfficientNet-B0", "global_ids": list(active_variant_ids if active_variant_ids is not None else range(len(taxonomy.variants)))}],
                "branches": []}
    depth = int(architecture[-1])
    shared = not architecture.startswith("IND")
    specs = node_plan(taxonomy, depth)
    if active_variant_ids is not None:
        from dataclasses import replace
        if depth != 2:
            raise ValueError("Active-label inventory requires two-stage architecture")
        specs = tuple(replace(s, global_ids=tuple(v for v in s.global_ids if v in active_variant_ids))
                      if s.stage == "variant" else s for s in specs
                      if s.stage != "variant" or sum(v in active_variant_ids for v in s.global_ids) > 1)
    branches = []
    for m, variants in enumerate(taxonomy.branches()["manufacturer_to_variant"]):
        if active_variant_ids is not None:
            variants = tuple(v for v in variants if v in active_variant_ids)
        branches.append({"manufacturer_id": m, "manufacturer": taxonomy.manufacturers[m],
                         "global_variant_ids": list(variants), "variants": [taxonomy.variants[v] for v in variants],
                         "singleton": len(variants) == 1})
    return {"architecture": architecture, "backbones_stored": 1 if shared else len(specs),
            "nodes": [{**s.to_dict(), "kind": "MLP head" if shared else "EfficientNet-B0"} for s in specs],
            "branches": branches}


def draw_network(output: Path, taxonomy: Taxonomy, architecture: str, trained: list[dict], *,
                 active_variant_ids: tuple[int, ...] | None = None, stage: str | None = None,
                 introduced_variant_id: int | None = None) -> None:
    """Show all stored trained specialists, singleton outputs, and all 100 labels.

    Arrows from a shared embedding represent available input, not eager execution
    of every head. Selection uses only the manufacturer router's predicted ID.
    """
    inventory = network_inventory(taxonomy, architecture, active_variant_ids=active_variant_ids)
    count = len(active_variant_ids) if active_variant_ids is not None else len(taxonomy.variants)
    if stage is not None:
        inventory.update(stage=stage, active_variant_ids=list(active_variant_ids) if active_variant_ids is not None else list(range(len(taxonomy.variants))),
                         introduced_variant_id=introduced_variant_id,
                         trained_checkpoints=[n for n in trained if n["architecture"] == architecture])
    write_json(output / f"network-{architecture}.json", inventory)
    if architecture != "FLAT" and not architecture.endswith("-2"):
        raise ValueError("Full branch-layout renderer currently supports two-stage trained networks")
    flat, shared = architecture == "FLAT", "FROZEN" in architecture
    hidden_dimensions = {n.get("hidden_dimension", 256) for n in trained if n["architecture"] == architecture}
    hidden = next(iter(hidden_dimensions), 256) if shared else None
    if shared and (len(hidden_dimensions) > 1 or type(hidden) is not int or hidden < 1):
        raise ValueError("Shared schematic requires one actual MLP width")
    width = 1600
    all_branches = inventory["branches"] if not flat else [
        {"manufacturer_id": m, "manufacturer": name,
         "variants": [taxonomy.variants[v] for v in ids if active_variant_ids is None or v in active_variant_ids],
         "global_variant_ids": [v for v in ids if active_variant_ids is None or v in active_variant_ids], "singleton": False}
        for m, (name, ids) in enumerate(zip(taxonomy.manufacturers, taxonomy.branches()["manufacturer_to_variant"]))]
    y = 390
    layouts = []
    for branch in all_branches:
        text = " · ".join(f"{v}: {name}" for v, name in zip(branch["global_variant_ids"], branch["variants"]))
        lines = textwrap.wrap(text, width=88, break_long_words=False)
        height = max(80, 34 + len(lines) * 20)
        layouts.append((branch, y, height, lines))
        y += height + 12
    height = y + 80
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#53677e"/></marker></defs>',
        '<rect width="100%" height="100%" fill="#f5f7fb"/>',
        '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#17283b}.title{font-size:30px;font-weight:700}.label{font-size:16px;font-weight:600}.small{font-size:14px}.muted{fill:#53677e}</style>']
    def text(x, yy, value, cls="small"):
        parts.append(f'<text x="{x}" y="{yy}" class="{cls}">{escape(str(value))}</text>')
    def rect(x, yy, w, h, fill):
        parts.append(f'<rect x="{x}" y="{yy}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="#b8c9d9"/>')
    def line(x, yy, xx, yyy, dashed=False):
        parts.append(f'<path d="M{x} {yy} L{xx} {yyy}" stroke="#53677e" stroke-width="1.5" fill="none" marker-end="url(#arrow)" {"stroke-dasharray=\"5 4\"" if dashed else ""}/>')
    text(30, 48, f"{architecture} · complete deployed network" + (f" · {stage}" if stage else ""), "title")
    seeds = sorted({n["seed"] for n in trained})
    text(30, 80, f"Same graph across measured seeds {seeds}; each seed has its own independently trained weights.")
    text(30, 105, f"All output labels below are globally unique variant IDs. All {count} logits are computed; grouping below is display only." if flat else
         "All output labels below are globally unique variant IDs. Only the predicted branch executes per image.")
    router_epochs = [f"s{n['seed']}: epoch {n['best_epoch']}" for n in trained
                     if n["architecture"] == architecture and n["key"] == "manufacturer"]
    if introduced_variant_id is not None:
        text(30, 126, f"Introduced global ID {introduced_variant_id}: {taxonomy.variants[introduced_variant_id]} · " +
             ("excluded from ALL initial train/validation, including backbone selection" if stage != "after" else
              "orange branch = the ONLY replaced classifier; encoder and router unchanged"))
    elif router_epochs:
        text(30, 126, "Manufacturer router selected checkpoints: " + " · ".join(router_epochs))
    rect(30, 135, 290, 90, "#e8eef7")
    text(48, 164, "RGB image → 224 × 224", "label")
    text(48, 191, "Remove 20 px banner; letterbox")
    text(48, 212, "Normalize; batch 1; FP32 primary")
    rect(380, 135, 570, 90, "#d7eee9" if shared else "#e0e8fa")
    text(400, 163, "Frozen EfficientNet-B0 → pooled 1280-D" if shared else f"EfficientNet-B0 → {count}-way classifier" if flat else "Independent EfficientNet-B0 → 30-way manufacturer", "label")
    text(400, 190, "Exactly ONE feature pass per image" if shared else "One full CNN per executed classifier")
    text(400, 212, "FLAT classifier removed; backbone weights + BN buffers fixed" if shared else "Fine-tuned from ImageNet V1; selected using validation only")
    line(320, 180, 380, 180)
    rect(1010, 135, 555, 90, "#fff0d2")
    text(1030, 163, f"Manufacturer MLP: 1280 → {hidden} → 30" if shared else "Global variant prediction" if flat else "Predicted manufacturer selects ONE branch", "label")
    text(1030, 190, "ReLU; independent AdamW + checkpoint" if shared else "Argmax local column → explicit global ID")
    text(1030, 212, "No test labels used for routing")
    line(950, 180, 1010, 180)
    text(30, 265, f"Stored backbones: {inventory['backbones_stored']} · trained classifiers: {len(inventory['nodes'])} · deterministic singleton branches: {sum(b['singleton'] for b in inventory['branches'])}", "label")
    text(30, 293, "B0 stages (channels / output size / repeats): stem 32/112/1 → 16/112/1 → 24/56/2 → 40/28/2 → 80/14/3")
    text(30, 317, "→ 112/14/3 → 192/7/4 → 320/7/1 → pointwise 1280/7 → global average pool → classifier or independent MLP heads.")
    text(30, 341, "Solid branch arrows: predicted routing. Dashed arrows: the SAME per-image embedding; other specialist heads remain idle." if shared else
         "One full CNN computes every output column. Manufacturer groups below organize labels for display; there is no routing." if flat else
         "Solid arrows select a branch. Dashed arrows supply the SAME original image to its OWN full B0; CNN weights are independent.")
    text(30, 369, "Manufacturer branch / global ID", "label")
    text(360, 369, "Trained specialist or deterministic output" if not flat else "Output group (display only)", "label")
    text(875, 369, "All globally mapped variant outputs", "label")
    last_center = layouts[-1][1] + layouts[-1][2] / 2
    parts.append(f'<path d="M1285 225 V238 H15 V{last_center}" stroke="#53677e" fill="none" stroke-width="1.5"/>')
    if not flat:
        source_x = 665 if shared else 175
        parts.append(f'<path d="M{source_x} 225 V232 H5 V380 H335 V{last_center}" stroke="#168475" fill="none" stroke-width="1.5" stroke-dasharray="5 4"/>')
    for branch, yy, hh, lines in layouts:
        m = branch["manufacturer_id"]
        rect(30, yy, 270, hh, "#e8eef7")
        text(45, yy + 29, f"{m}: {branch['manufacturer']}", "label")
        text(45, yy + 53, f"{len(branch['variants'])} variant(s)")
        singleton = branch["singleton"]
        replaced = stage == "after" and introduced_variant_id in branch["global_variant_ids"]
        rect(350, yy, 480, hh, "#ffd6ad" if replaced else "#fdf0d9" if singleton else "#dcf0e9" if shared else "#e3eafb")
        label = "FLAT output columns" if flat else "Deterministic singleton · NO classifier" if singleton else f"MLP 1280 → {hidden} → {len(branch['variants'])}" if shared else f"Full B0 → Linear(1280, {len(branch['variants'])})"
        text(365, yy + 27, label, "label")
        if not flat and not singleton:
            key = f"variant-manufacturer-{m}"
            text(365, yy + 48, key + (" · REPLACED" if replaced else ""))
            selections = [f"s{n['seed']}: epoch {n['best_epoch']}" for n in trained if n["architecture"] == architecture and n["key"] == key]
            text(365, yy + 68, " · ".join(selections))
        elif singleton:
            text(365, yy + 52, "Known global variant; zero specialist MACs")
        else:
            text(365, yy + 52, f"{count}-way global argmax; no manufacturer routing")
        rect(870, yy, 695, hh, "#ffffff")
        for i, value in enumerate(lines):
            text(885, yy + 26 + i * 20, value)
        line(300, yy + hh / 2, 350, yy + hh / 2)
        line(15, yy + hh / 2, 30, yy + hh / 2)
        line(830, yy + hh / 2, 870, yy + hh / 2)
        if not flat and not singleton:
            line(335, yy + 13, 350, yy + 13, True)
    text(30, height - 35, "Source: versioned training-only taxonomy + audited deployment node plans. Selection epochs are validation measurements; timing is reported separately.")
    parts.append("</svg>")
    with (output / f"network-{architecture}.svg").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(parts) + "\n")


def draw_backbone(output: Path, *, shared_hidden: int = 256, flat_classes: int = 100) -> None:
    """Save every real Torchvision module and explain its repeated MBConv stages.

    Construct without pretrained download; no checkpoint tensors or random
    parameter values are exported. Module inventory is architecture metadata.
    """
    from torch import nn
    from .models.flat import build_flat
    model = build_flat(pretrained=False, classes=flat_classes)
    modules = []
    for name, module in model.named_modules():
        row = {"name": name, "type": type(module).__name__, "own_parameters": sum(p.numel() for p in module.parameters(recurse=False))}
        if isinstance(module, nn.Conv2d):
            row.update(in_channels=module.in_channels, out_channels=module.out_channels,
                       kernel_size=list(module.kernel_size), stride=list(module.stride),
                       groups=module.groups, padding=list(module.padding), bias=module.bias is not None)
        elif isinstance(module, nn.Linear):
            row.update(in_features=module.in_features, out_features=module.out_features, bias=module.bias is not None)
        elif isinstance(module, nn.BatchNorm2d):
            row.update(features=module.num_features, eps=module.eps, momentum=module.momentum)
        elif isinstance(module, nn.Dropout):
            row["p"] = module.p
        elif type(module).__name__ == "StochasticDepth":
            row.update(p=module.p, mode=module.mode)
        modules.append(row)
    write_json(output / "efficientnet-b0-modules.json", {"architecture": "Torchvision EfficientNet-B0 with 100-way FLAT classifier",
        "input": [1, 3, 224, 224], "modules": modules, "total_parameters": sum(p.numel() for p in model.parameters()),
        "shared_export": "features + avgpool + flatten; remove classifier; freeze parameters and BN buffers"})
    # The repeated stage specifications below are verified against the actual
    # constructed module's depthwise kernels, strides and output channels.
    stages = [(1, 1, 3, 1, 16, 112), (2, 2, 3, 2, 24, 56), (3, 2, 5, 2, 40, 28),
              (4, 3, 3, 2, 80, 14), (5, 3, 5, 1, 112, 14), (6, 4, 5, 2, 192, 7), (7, 1, 3, 1, 320, 7)]
    for stage, repeats, kernel, stride, channels, _ in stages:
        blocks = model.features[stage]
        depthwise = [m for m in blocks[0].modules() if isinstance(m, nn.Conv2d) and m.groups > 1]
        convolutions = [m for m in blocks[-1].modules() if isinstance(m, nn.Conv2d)]
        if len(blocks) != repeats or depthwise[0].kernel_size != (kernel, kernel) or depthwise[0].stride != (stride, stride) or convolutions[-1].out_channels != channels:
            raise ValueError("EfficientNet stage diagram does not match installed architecture")
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1460" height="1450" viewBox="0 0 1460 1450">',
        '<rect width="100%" height="100%" fill="#f5f7fb"/>',
        '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#17283b;font-size:17px}.h{font-size:28px;font-weight:bold}.b{font-weight:bold}</style>']
    def text(x, y, value, cls=""):
        parts.append(f'<text x="{x}" y="{y}" class="{cls}">{escape(value)}</text>')
    def box(x, y, width, height, fill):
        parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="12" fill="{fill}" stroke="#b8c9d9"/>')
    text(35, 50, "EfficientNet-B0 · internal architecture and export points", "h")
    text(35, 83, "Common basis for every independent full CNN and the one shared frozen encoder. Batch-one shape convention: C × H × W.")
    text(35, 111, "Exact named-module inventory, convolution groups, kernels, strides and parameter counts: efficientnet-b0-modules.json")
    box(35, 145, 660, 80, "#e0e8fa")
    text(55, 175, "Input 3 × 224 × 224 → stem Conv 3×3, stride 2", "b")
    text(55, 204, "32 × 112 × 112 · BatchNorm + SiLU")
    for i, (stage, repeats, kernel, stride, channels, size) in enumerate(stages):
        yy = 248 + i * 100
        box(35, yy, 660, 82, "#e0e8fa")
        text(55, yy + 28, f"Stage {stage}: MBConv{'1' if stage == 1 else '6'} · kernel {kernel}×{kernel} · repeat ×{repeats}", "b")
        text(55, yy + 56, f"First-block stride {stride}; remaining stride 1 → {channels} × {size} × {size}")
        parts.append(f'<path d="M365 {yy-18} V{yy}" stroke="#53677e" stroke-width="2"/>')
    box(35, 950, 660, 80, "#d7eee9")
    text(55, 980, "Pointwise Conv 1×1 → 1280 × 7 × 7", "b")
    text(55, 1009, "BatchNorm + SiLU → adaptive global average pool → flatten")
    box(35, 1050, 660, 72, "#d7eee9")
    text(55, 1080, "1280-dimensional pooled feature vector", "b")
    text(55, 1107, "Shared encoder export ENDS HERE; classifier is removed")
    box(745, 145, 680, 370, "#ffffff")
    text(765, 180, "Inside each MBConv block", "h")
    for i, line in enumerate([
        "1. Optional 1×1 expansion → BatchNorm + SiLU",
        "   MBConv6 expands channels ×6; MBConv1 omits expansion.",
        "2. Depthwise k×k convolution → BatchNorm + SiLU",
        "3. Squeeze/excitation: global pool → 1×1 → SiLU → 1×1",
        "   → sigmoid gate → multiply input features",
        "4. 1×1 projection → BatchNorm (linear bottleneck)",
        "5. Residual add if stride=1 and input/output channels match",
        "Stochastic depth is a training operation; disabled in eval.",
        "All shared encoder weights AND BatchNorm buffers stay frozen."]):
        text(765, 219 + i * 31, line)
    box(745, 550, 680, 225, "#fff0d2")
    text(765, 588, "Independent CNN classifier / FLAT", "h")
    for i, line in enumerate(["Pooled 1280-D → Dropout(p=0.2) → Linear(1280, n)",
        "Dropout disabled in inference. Each CNN has separate weights.",
        f"n={flat_classes} for FLAT; n=30 for the manufacturer router;",
        "n=number of allowed global variants for each specialist.",
        "Argmax local column → saved explicit global ID map."]):
        text(765, 628 + i * 29, line)
    box(745, 810, 680, 225, "#d7eee9")
    text(765, 848, "Shared frozen independent MLP head", "h")
    for i, line in enumerate([f"SAME pooled 1280-D → Linear(1280, {shared_hidden}) → ReLU",
        f"→ Linear({shared_hidden}, n) → argmax → explicit global ID",
        "One independently trained router; one selected specialist.",
        "Each head has its own optimizer and selected checkpoint.",
        "A singleton branch returns its only variant without a head."]):
        text(765, 888 + i * 29, line)
    text(35, 1180, "Deployment distinction", "h")
    text(35, 1220, "FLAT: one full B0. IND-2: 18 stored full B0 CNNs; execute router + selected non-singleton specialist.")
    text(35, 1255, "FT-FROZEN-2: one frozen B0 encoder + 18 independent MLPs; execute backbone once, router, then one selected head.")
    text(35, 1290, "All 30 manufacturer branches and all 100 variant labels are expanded in the companion full-network SVGs.")
    text(35, 1340, "Source: actual installed Torchvision model construction; no parameter tensors or dataset pixels exported.")
    parts.append("</svg>")
    with (output / "efficientnet-b0-detail.svg").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(parts) + "\n")
