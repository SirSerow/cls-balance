"""Publish measured Phase 7 test predictions, update costs and full networks.

Only test predictions, aggregate training histories and architecture metadata are
exported. Dataset pixels, supervised image-ID lists, caches and weights stay local.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil
import statistics

from .artifacts import file_hash, object_hash, write_json
from .data import AircraftData
from .evaluate import audit_run, model_hash
from .incremental import audit_inventory, audit_training, inventory, load_system, local_path
from .network_diagrams import draw_backbone, draw_network
from .predictions import read_predictions
from .prepare import write_csv
from .report_flat import read_json
from .taxonomy import Taxonomy
from .train_independent import unique_id

ARCHITECTURES = ("IND-2", "FT-FROZEN-2")


def group_metrics(records, variant_ids: tuple[int, ...]) -> dict:
    """Fixed group denominator; normal pipeline predictions, never branch-only."""
    selected = [r for r in records if r.true_variant_id in variant_ids]
    supports = {v: sum(r.true_variant_id == v for r in selected) for v in variant_ids}
    if not variant_ids or any(n == 0 for n in supports.values()):
        raise ValueError("Missing predeclared incremental group support")
    correct = {v: sum(r.true_variant_id == v and r.prediction.variant_id == v for r in selected) for v in variant_ids}
    return {"variant_ids": list(variant_ids), "images": len(selected), "correct": sum(correct.values()),
            "macro_recall": statistics.mean(correct[v] / supports[v] for v in variant_ids),
            "micro_accuracy": sum(correct.values()) / len(selected)}


def compare_records(before, after, selection: dict, taxonomy: Taxonomy) -> dict:
    """Measure all true groups and functional locality under predicted routing.

    Outside true-branch images can still change when mistakenly routed into the
    updated branch. Exact invariance is required for images routed elsewhere.
    """
    if [r.image_id for r in before] != [r.image_id for r in after]:
        raise ValueError("Before/after official test IDs/order differ")
    if any(a.routing != b.routing or a.architecture != b.architecture or a.true_variant_id != b.true_variant_id
           for a, b in zip(before, after, strict=True)):
        raise ValueError("Before/after prediction identities differ")
    heldout, parent = selection["variant_id"], selection["manufacturer_id"]
    groups = {"new": (heldout,), "old_within_branch": tuple(v for v in selection["branch_variant_ids"] if v != heldout),
              "outside_branch": tuple(v for v in range(len(taxonomy.variants)) if v not in selection["branch_variant_ids"]),
              "known": tuple(v for v in range(len(taxonomy.variants)) if v != heldout),
              "all": tuple(range(len(taxonomy.variants)))}
    if any(r.prediction.variant_id == heldout for r in before):
        raise ValueError("Pre-update deployment can predict the excluded class")
    routed_elsewhere = changed_elsewhere = changed_outside_truth = 0
    for a, b in zip(before, after, strict=True):
        if a.prediction.manufacturer_id != b.prediction.manufacturer_id:
            raise ValueError("Router changed during single-specialist update")
        if a.prediction.manufacturer_id != parent:
            routed_elsewhere += 1
            changed_elsewhere += a.prediction != b.prediction
        if a.true_variant_id not in selection["branch_variant_ids"]:
            changed_outside_truth += a.prediction.variant_id != b.prediction.variant_id
    if changed_elsewhere:
        raise ValueError("A prediction changed outside the executed affected branch")
    return {"groups": {name: {"before": group_metrics(before, ids), "after": group_metrics(after, ids)}
                       for name, ids in groups.items()},
            "functional_locality": {"routed_outside_affected_branch": routed_elsewhere,
                "changed_predictions_routed_outside": changed_elsewhere,
                "changed_predictions_with_true_manufacturer_outside": changed_outside_truth,
                "router_decisions_identical": True}}


def calculate(output: Path) -> list[dict]:
    """Artifact-only recomputation from published test prediction CSVs."""
    taxonomy = Taxonomy.from_dict(read_json(output / "taxonomy.json"))
    compatibility = read_json(output / "compatibility.json")
    selection = read_json(output / "selection.json")
    entries = read_json(output / "evaluations.json")
    expected_ids = None
    common_contract = None
    tables = {}
    for entry in entries:
        evaluation = local_path(output, entry["path"])
        if audit_run(evaluation) != entry["audit"]:
            raise ValueError("Incremental nested evaluation audit mismatch")
        meta = read_json(evaluation / "metadata.json")
        if any(meta[key] != compatibility[expected] for key, expected in (
                ("taxonomy_sha256", "taxonomy_sha256"), ("preprocessing_sha256", "preprocessing_sha256"),
                ("dataset_manifest_records_sha256", "dataset_manifest_sha256"))):
            raise ValueError("Published evaluation/data/preprocessing contract mismatch")
        key = (entry["seed"], entry["architecture"], entry["stage"], entry["routing"])
        if key in tables or (meta["config"]["split"], meta["config"]["seed"], meta["config"]["architecture"],
                             meta["config"]["routing"]) != ("test", key[0], key[1], key[3]):
            raise ValueError("Duplicate or incorrectly labeled incremental evaluation")
        contract = {k: meta[k] for k in ("taxonomy_sha256", "preprocessing_sha256", "dataset_manifest_records_sha256")}
        contract["runtime"] = {k: meta["config"][k] for k in ("device", "precision", "residency")}
        contract["environment"] = {k: meta["environment"].get(k) for k in
                                   ("torch", "cuda_runtime", "cudnn", "gpus", "packages", "platform", "nvidia_smi")}
        if common_contract is None:
            common_contract = contract
        if common_contract != contract:
            raise ValueError("Incremental evaluations differ in data, preprocessing, hardware or precision")
        source = read_json(output / "source.json")
        if meta["model_id"] != f"{source['training_run_id']}/s{key[0]}/{key[1]}/{key[2]}":
            raise ValueError("Evaluation before/after model identity mismatch")
        if "inference_source_sha256" in source and meta.get("inference_source_sha256") != source["inference_source_sha256"]:
            raise ValueError("Inference implementation hash mismatch")
        records = read_predictions(evaluation / "predictions.csv", taxonomy)
        ids = [r.image_id for r in records]
        if expected_ids is None:
            expected_ids = ids
        if ids != expected_ids:
            raise ValueError("Architectures/seeds evaluated different test images")
        tables[key] = records
    config = read_json(output / "config.resolved.json")
    expected = {(s, a, stage, routing) for s in config["seeds"] for a in ARCHITECTURES
                for stage in ("before", "after") for routing in ("predicted", "oracle")}
    if set(tables) != expected:
        raise ValueError("Incomplete seed/architecture/stage/routing evaluation inventory")
    result = []
    for seed in config["seeds"]:
        for architecture in ARCHITECTURES:
            record = {"seed": seed, "architecture": architecture}
            for routing in ("predicted", "oracle"):
                record[routing] = compare_records(tables[(seed, architecture, "before", routing)],
                                                 tables[(seed, architecture, "after", routing)], selection, taxonomy)
            resources = read_json(output / f"seed-{seed}" / f"resources-{architecture}.json")
            record["update"] = resources
            result.append(record)
    return result


def mean_sd(values) -> str:
    values = list(values)
    return f"{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}" if len(values) > 1 else f"{values[0]:.2f}"


def draw_figures(output: Path, records: list[dict], nodes: list[dict], taxonomy: Taxonomy, selection: dict,
                 preview_root: Path | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "svg.hashsalt": "cls-balance-phase7", "figure.facecolor": "white"})
    colors = {"IND-2": "#3266ad", "FT-FROZEN-2": "#158779"}
    plots = output / "plots"
    plots.mkdir(exist_ok=True)

    def save(fig, name):
        fig.tight_layout()
        if preview_root is not None:
            preview_root.mkdir(parents=True, exist_ok=True)
            fig.savefig(preview_root / (Path(name).stem + ".png"), dpi=110)
        # Stable metadata supports byte-for-byte artifact-only rebuilds.
        with (plots / name).open("x", encoding="utf-8", newline="\n") as stream:
            fig.savefig(stream, format="svg", metadata={"Date": None})
        plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    for ax, group, label in zip(axes, ("new", "old_within_branch", "outside_branch", "known"),
                               ("New variant", "Existing variants in updated branch", "True manufacturer outside branch", "All 99 previously known variants")):
        for i, architecture in enumerate(ARCHITECTURES):
            selected = [r for r in records if r["architecture"] == architecture]
            for j, stage in enumerate(("before", "after")):
                values = [100 * r["predicted"]["groups"][group][stage]["macro_recall"] for r in selected]
                x = i * 1.3 + j * .44
                ax.bar(x, statistics.mean(values), width=.38, color=colors[architecture], alpha=.42 if j == 0 else 1,
                       yerr=statistics.stdev(values) if len(values) > 1 else None, capsize=3)
                ax.scatter([x] * len(values), values, color="#203246", s=12, zorder=4)
                ax.text(x, max(values) + 3, f"{statistics.mean(values):.1f}", ha="center", fontsize=8)
        ax.set_xticks([0, .44, 1.3, 1.74], ["IND\nbefore", "IND\nafter", "Shared\nbefore", "Shared\nafter"], fontsize=8)
        ax.set(title=label, ylabel="Test variant macro recall (%)", ylim=(0, 108))
        ax.grid(axis="y", alpha=.15)
    fig.suptitle(f"Introduce {selection['variant']} · complete predicted-route pipeline\nBars = seed mean; whiskers = sample SD; dots = individual seeds", fontsize=13)
    fig.subplots_adjust(top=.8)
    save(fig, "accuracy-before-after.svg")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, key, divisor, title, unit in zip(axes,
            ("update_wall_seconds", "trainable_parameters", "replacement_bytes"), (1, 1e6, 1e6),
            ("Measured update duration", "Parameters retrained", "Replacement checkpoint"), ("Seconds", "Million parameters", "Decimal MB")):
        for i, architecture in enumerate(ARCHITECTURES):
            values = [r["update"][key] / divisor for r in records if r["architecture"] == architecture]
            ax.bar(i, statistics.mean(values), color=colors[architecture], width=.6,
                   yerr=statistics.stdev(values) if len(values) > 1 else None, capsize=4)
            ax.scatter([i] * len(values), values, color="#203246", s=16, zorder=4)
            ax.text(i, max(values) * 1.06, f"{statistics.mean(values):.2f}", ha="center")
        ax.set_xticks([0, 1], ["IND-2", "FT-FROZEN-2"])
        ax.set(title=title, ylabel=unit, ylim=(0, ax.get_ylim()[1] * 1.18))
        ax.grid(axis="y", alpha=.15)
    fig.suptitle("Replace only the Boeing variant specialist\nShared duration includes image feature extraction; both include validation, selection, export and reload", fontsize=12)
    fig.subplots_adjust(top=.78)
    save(fig, "update-cost.svg")

    seeds = sorted({r["seed"] for r in records})
    fig, axes = plt.subplots(len(seeds), 3, figsize=(15, 3.3 * len(seeds)), squeeze=False)
    titles = ["Reduced-label FLAT encoder source", "Independent affected specialist", "Shared affected specialist"]
    for row, seed in enumerate(seeds):
        for col, category in enumerate(("FLAT", "IND-2", "FT-FROZEN-2")):
            relevant = [n for n in nodes if n["seed"] == seed and n["architecture"] == category
                        and (category == "FLAT" or n["key"] == f"variant-manufacturer-{selection['manufacturer_id']}")]
            for n in relevant:
                with (output / n["history"]).open(encoding="utf-8", newline="") as stream:
                    history = list(csv.DictReader(stream))
                color = "#cb7528" if n["stage"] == "after" else "#3266ad"
                for split, style in (("train", "--"), ("val", "-")):
                    axes[row, col].plot([int(h["epoch"]) for h in history],
                        [100 * float(h[f"{split}_macro_recall"]) for h in history], style,
                        color=color, alpha=.6 if split == "train" else 1,
                        label=f"{n['stage']} {n['classes']}-way {'augmented train' if split == 'train' else 'validation'}")
                axes[row, col].axvline(n["best_epoch"], color=color, alpha=.5, linestyle=":")
            axes[row, col].set(title=f"Seed {seed} · {titles[col]}", xlabel="Epoch", ylabel="Local macro recall (%)", ylim=(0, 100))
            axes[row, col].legend(fontsize=7, loc="lower right")
            axes[row, col].grid(alpha=.15)
    fig.suptitle("Learning curves · local validation selection; dotted lines = selected epochs\nInitial specialist has 21 outputs; replacement has 22. Local scores have different class denominators.", fontsize=13)
    save(fig, "learning-curves.svg")

    from .models.cascade import node_plan
    specs = node_plan(taxonomy, 2)
    fig, axes = plt.subplots(len(specs), 2, figsize=(15, 2.65 * len(specs)), squeeze=False)
    seed_colors = ("#3266ad", "#158779", "#a74f83")
    for row, spec in enumerate(specs):
        for col, architecture in enumerate(ARCHITECTURES):
            ax = axes[row, col]
            relevant = [n for n in nodes if n["architecture"] == architecture and n["key"] == spec.key]
            for n in relevant:
                with (output / n["history"]).open(encoding="utf-8", newline="") as stream:
                    history = list(csv.DictReader(stream))
                color = seed_colors[seeds.index(n["seed"]) % len(seed_colors)]
                epochs = [int(h["epoch"]) for h in history]
                values = [100 * float(h["val_macro_recall"]) for h in history]
                ax.plot(epochs, values, "--" if n["stage"] == "after" else "-", color=color,
                        label=f"s{n['seed']} {n['stage']}, {n['classes']}-way")
                ax.scatter(n["best_epoch"], 100 * n["validation_macro_recall"], s=18, color=color)
            branch = "manufacturer router" if spec.parent_id is None else taxonomy.manufacturers[spec.parent_id]
            ax.set(title=f"{architecture} · {branch}", xlabel="Epoch", ylabel="Local validation macro recall (%)", ylim=(0, 100))
            ax.grid(alpha=.15)
            ax.legend(fontsize=6, ncol=2)
    fig.suptitle("Every trained routing/specialist classifier · validation learning curves\nDots = selected checkpoints; dashed = replacement; singleton branches have no trained classifier", fontsize=13)
    save(fig, "every-specialist-validation.svg")

    # One row per canonical class, never a selected or sorted high-performing subset.
    entries = read_json(output / "evaluations.json")
    classwise = {}
    for a in ARCHITECTURES:
        for stage in ("before", "after"):
            metrics = [read_json(output / e["path"] / "metrics.json") for e in entries
                       if e["architecture"] == a and e["stage"] == stage and e["routing"] == "predicted"]
            classwise[a, stage] = [100 * statistics.mean(m["classwise"][v]["recall"] for m in metrics)
                                   for v in range(len(taxonomy.variants))]
    fig, axes = plt.subplots(1, 2, figsize=(15, 25), sharey=True)
    y = np.arange(len(taxonomy.variants))
    rows = []
    for ax, a in zip(axes, ARCHITECTURES):
        old, new = classwise[a, "before"], classwise[a, "after"]
        ax.barh(y - .17, old, height=.32, color=colors[a], alpha=.4, label="Before")
        ax.barh(y + .17, new, height=.32, color=colors[a], label="After")
        ax.set(title=f"{a} · all 100 canonical variant IDs", xlabel="Test recall (%), mean across seeds", xlim=(0, 100))
        ax.grid(axis="x", alpha=.15)
        ax.legend(loc="upper right")
        for v in range(len(taxonomy.variants)):
            rows.append({"architecture": a, "variant_id": v, "variant": taxonomy.variants[v],
                         "before_recall_percent": old[v], "after_recall_percent": new[v], "delta_pp": new[v] - old[v]})
    axes[0].set_yticks(y, [f"{v}: {name}" + ("  [NEW]" if v == selection["variant_id"] else "")
                             for v, name in enumerate(taxonomy.variants)], fontsize=8)
    axes[0].invert_yaxis()
    save(fig, "all-variant-recalls.svg")
    write_csv(output / "classwise-before-after.csv", rows)

    ids = tuple(v for v in range(len(taxonomy.variants)) if v != selection["variant_id"])
    for stage in ("before", "after"):
        target = plots / stage
        target.mkdir()
        for a in ARCHITECTURES:
            current = [n for n in nodes if n["architecture"] == a and (n["stage"] == "before"
                       and (stage == "before" or n["key"] != f"variant-manufacturer-{selection['manufacturer_id']}")
                       or n["stage"] == stage == "after")]
            draw_network(target, taxonomy, a, current, active_variant_ids=ids if stage == "before" else None,
                         stage=stage, introduced_variant_id=selection["variant_id"])
    draw_network(plots / "before", taxonomy, "FLAT", [n for n in nodes if n["architecture"] == "FLAT"],
                 active_variant_ids=ids, stage="initial 99-class encoder source", introduced_variant_id=selection["variant_id"])
    draw_backbone(plots, shared_hidden=read_json(output / "config.resolved.json")["hidden_dimension"], flat_classes=len(ids))


def render_report(output: Path, records: list[dict]) -> None:
    selection = read_json(output / "selection.json")
    config = read_json(output / "config.resolved.json")
    env = read_json(output / "environment.json")
    source = read_json(output / "source.json")
    hashes = read_json(output / "compatibility.json")
    text = [f"# Phase 7 — introduce {selection['variant']} with one specialist replacement", "",
        "Measured incremental experiment. Initial weights are trained afresh on 99 variants; the Phase 3–6 100-class encoders are never reused.", "",
        "## Experimental context", "",
        "| Field | Recorded value |", "| --- | --- |",
        f"| Training run | `{source['training_run_id']}` |",
        f"| Test evaluation run | `{source['evaluation_run_id']}` |",
        f"| Training git SHA | `{env['git_sha']}`; dirty={env['git_dirty']}; exact source hashes included |",
        f"| Config hash | `{object_hash(config)}` |",
        f"| Dataset / taxonomy hash | `{hashes['dataset_manifest_sha256']}` / `{hashes['taxonomy_sha256']}` |",
        f"| Preprocessing hash | `{hashes['preprocessing_sha256']}` |",
        "| Edition / official splits | Torchvision 100 variants / 70 families / 30 manufacturers; 3334 / 3333 / 3333 |",
        f"| Seeds | {config['seeds']} |",
        f"| Hardware / framework | {', '.join(env['gpus'])}; torch {env['torch']}; CUDA {env['cuda_runtime']} |",
        f"| Driver | {env['nvidia_smi']} |",
        "| Precision / input / residency | FP32; 224×224; all resident; batch-one test inference; TF32 disabled |",
        "| Power | OS/driver defaults; clocks not locked; temporary process-lifetime idle-sleep request |", "",
        "## Predeclared introduction protocol", "",
        f"Selection: **{selection['selection_rule']}**. This selects global ID **{selection['variant_id']} ({selection['variant']})**, manufacturer **{selection['manufacturer']}**.", "",
        "The manufacturer initially retains 21 variants, and the held-out variant's family retains existing variants. All of its train **and validation** images are absent from initial FLAT fine-tuning, manufacturer training, independent specialist training, shared head fitting and every initial checkpoint selection. The initial shared encoder is exported only from that seed's new 99-way FLAT checkpoint.", "",
        "The introduction adds the class's official training examples to the affected specialist, replaying all existing branch training examples. Only the corresponding validation branch is used for early stopping. Both architectures warm-start existing feature/head tensors and output rows; a new output row is seeded deterministically. The independent update optimizes the affected full CNN; the shared update optimizes only its MLP. Routers, shared encoder and every other specialist retain their exact bytes.", "",
        "**Explicit implementation choices:** Active initial labels are a 99-ID mask over the canonical 100-ID taxonomy, preserving global IDs instead of renumbering. All test inference occurs only after every seed's before/after checkpoint has been selected. Before-stage headline accuracy below uses the 99 known classes. Retrospective all-100 and new-class-before scores are diagnostics; new-class recall is necessarily zero before its output exists. Every evaluation still retains the complete official test set, with no hidden subset benchmark.", "",
        "## Specialist update / introduction (predicted full pipeline)", "",
        "Values are seed mean ± sample SD. Macro recall is mean class recall; outside-branch micro accuracy is also given below. These are classification/update measurements, with no new inference-latency claim.", "",
        "| Metric | IND-2 | FT-FROZEN-2 |", "| --- | --- | --- |"]
    grouped = {a: [r for r in records if r["architecture"] == a] for a in ARCHITECTURES}
    def row(label, fn):
        text.append("| " + label + " | " + " | ".join(mean_sd(fn(r) for r in grouped[a]) for a in ARCHITECTURES) + " |")
    for group, label in (("new", "New-class recall"), ("old_within_branch", "Existing updated-branch macro recall"),
                         ("outside_branch", "Outside true branch macro recall"), ("known", "All 99 known variants macro recall"),
                         ("all", "All 100 variants macro recall (before diagnostic)")):
        for stage in ("before", "after"):
            row(f"{label}, {stage} (%)", lambda r, g=group, s=stage: 100 * r["predicted"]["groups"][g][s]["macro_recall"])
    for stage in ("before", "after"):
        row(f"Outside true branch micro accuracy, {stage} (%)", lambda r, s=stage:
            100 * r["predicted"]["groups"]["outside_branch"][s]["micro_accuracy"])
    row("Update wall time (s)", lambda r: r["update"]["update_wall_seconds"])
    row("Parameters retrained (M)", lambda r: r["update"]["trainable_parameters"] / 1e6)
    row("Replacement checkpoint (decimal MB)", lambda r: r["update"]["replacement_bytes"] / 1e6)
    initial = {seed: read_json(output / f"seed-{seed}/initial-costs.json") for seed in config["seeds"]}
    text.append("| Initial fitting wall time including encoder source (h) | " + mean_sd(
        sum(n["training_wall_seconds"] for n in initial[s]["IND-2"]) / 3600 for s in config["seeds"]) + " | " + mean_sd(
        (initial[s]["FLAT"]["update_wall_seconds"] + initial[s]["shared_training_wall_seconds"]) / 3600 for s in config["seeds"]) + " |")
    text += ["", "Update duration includes image feature extraction for the shared system, epoch optimization, validation, checkpoint selection, export and exact reload. Model initialization/loading and test inference are excluded. Replay/training exposure and step counts are in the per-seed resources JSONs. Validation-only selected epochs and complete learning curves are preserved.", "",
        "![Accuracy before and after](plots/accuracy-before-after.svg)", "![Update duration, parameters and storage](plots/update-cost.svg)", "",
        "## Per-seed results and routing diagnostics", "",
        "| Seed / architecture | New recall after (%) | Old branch before → after (%) | Outside before → after (%) | Oracle new recall after (%) | Update (s) | Changed outside true branch |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in records:
        g = r["predicted"]["groups"]
        text.append(f"| {r['seed']} / {r['architecture']} | {100*g['new']['after']['macro_recall']:.2f} | "
            f"{100*g['old_within_branch']['before']['macro_recall']:.2f} → {100*g['old_within_branch']['after']['macro_recall']:.2f} | "
            f"{100*g['outside_branch']['before']['macro_recall']:.2f} → {100*g['outside_branch']['after']['macro_recall']:.2f} | "
            f"{100*r['oracle']['groups']['new']['after']['macro_recall']:.2f} | {r['update']['update_wall_seconds']:.2f} | "
            f"{r['predicted']['functional_locality']['changed_predictions_with_true_manufacturer_outside']} |")
    text += ["", "The manufacturer router produces identical before/after decisions. Every image routed outside the affected manufacturer has an identical complete prediction. Images whose **true** manufacturer lies outside Boeing can still change when the fixed router mistakenly selects Boeing; the count is measured above. Unchanged module bytes alone do not justify assuming unchanged outside-class accuracy.", "",
        "Oracle routing is a named diagnostic only. Every nested evaluation retains full routing diagnostics, per-class recall, confusion matrix, bypass counts and per-image path decisions. Hard-route and oracle scores are never pooled.", "",
        "## Training review and complete architectures", "",
        "![Learning curves and selection](plots/learning-curves.svg)", "",
        "[Validation curves for every trained router and specialist](plots/every-specialist-validation.svg). Every selected classifier's full aggregate train/validation history is linked from `nodes.json`.", "",
        "The diagrams expand **all 30 manufacturer branches**, every trained specialist, all active global variant outputs, singleton bypasses, and per-seed selected epochs. Only the predicted branch executes. Orange marks the replacement; the other nodes remain unchanged.", "",
        "| Network | Before | After |", "| --- | --- | --- |",
        "| IND-2, 18 separate full CNNs | [99-class network](plots/before/network-IND-2.svg) | [100-class network](plots/after/network-IND-2.svg) |",
        "| FT-FROZEN-2, one encoder + 18 independent heads | [99-class network](plots/before/network-FT-FROZEN-2.svg) | [100-class network](plots/after/network-FT-FROZEN-2.svg) |",
        "| Initial FLAT encoder source | [99-way classifier](plots/before/network-FLAT.svg) | No FLAT update; used only to export the initial encoder |", "",
        "![Complete updated shared network](plots/after/network-FT-FROZEN-2.svg)", "",
        "[EfficientNet-B0 layer detail](plots/efficientnet-b0-detail.svg) · [every variant's before/after recall](plots/all-variant-recalls.svg) · [classwise CSV](classwise-before-after.csv) · [checkpoint and node inventory](nodes.json).", "",
        "## Audits, interpretation and limits", "",
        "The training audit re-derives each node's exact supervised IDs from official train/validation annotations; checks initial holdout exclusion, encoder parent, seed/config/data/preprocessing hashes, local-to-global maps, earliest best validation epoch, and per-epoch support. Checkpoint reloads are exact. Unrelated weights/buffers, encoder and fixed-input logits are checked before publication; no test image enters optimization or selection.", "",
        "The result tests one deterministic class introduction on one dataset/hardware configuration. With roughly 33 test examples per class, new-class recall has limited resolution. Seed SD describes training variability and is not a confidence interval. No post-test hyperparameter changes or class reselection are permitted. Broader paired-bootstrap inference and Phase 8 publication remain separate.", "",
        "The primary Phase 6 latency comparison is unchanged. Update duration is measured separately from inference latency and cannot be interpreted as FPS. Shared update cost excludes the already paid initial FLAT fine-tuning, which is retained separately in initial-cost metadata. Initial fitting scopes differ slightly (independent node epochs exclude export; reduced FLAT includes export/reload), and initial seed-42 backbone training overlapped CPU checks. Initial-cost totals are accounting evidence, not a controlled training-speed claim.", "",
        "## Reproduction and artifact trace", "", "```powershell",
        ".venv-cuda/Scripts/python.exe scripts/train_incremental_awake.py train --config configs/incremental.yaml",
        f".venv-cuda/Scripts/python.exe scripts/train_incremental_awake.py evaluate --suite runs/incremental/{source['training_run_id']}",
        f".venv-cuda/Scripts/python.exe -m cls_balance.report_incremental --training runs/incremental/{source['training_run_id']} --evaluation runs/incremental/{source['evaluation_run_id']} --output-root results/stage7",
        ".venv/Scripts/python.exe -m pytest -q tests checks", "```", "",
        "Every table value is reproducible from [aggregate.json](aggregate.json), [evaluations.json](evaluations.json), per-test-image CSVs under `evaluations/`, and per-seed update resource files. [Training audit](training_audit.json), [functional audit](functional_audit.json), exact source hashes, configuration, environment and publication checksums are included. Weights, caches, training IDs and dataset images are excluded.", ""]
    with (output / "REPORT.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(text))


def audit_functional(training: Path, config: dict) -> list[dict]:
    """Exact fixed-input logits for every untouched module; no dataset pixels."""
    import torch
    selection = read_json(training / "selection.json")
    affected = f"variant:{selection['manufacturer_id']}"
    results = []
    for seed in config["seeds"]:
        for a in ARCHITECTURES:
            before, after = load_system(training, seed, a, "before").eval(), load_system(training, seed, a, "after").eval()
            image = torch.zeros(1, 3, 224, 224)
            with torch.inference_mode():
                if before.backbone is not None:
                    feature = before.backbone(image)
                    torch.testing.assert_close(feature, after.backbone(image), rtol=0, atol=0)
                else:
                    feature = image
                checked = []
                for key in before.nodes:
                    if key != affected:
                        torch.testing.assert_close(before.nodes[key].classifier(feature), after.nodes[key].classifier(feature), rtol=0, atol=0)
                        checked.append(key)
            results.append({"seed": seed, "architecture": a, "untouched_modules_exact_logits": checked,
                            "shared_encoder_exact_features": before.backbone is not None,
                            "probe": "all-zero preprocessed tensor; CPU FP32; exact rtol=atol=0"})
    return results


def publish(training: Path, evaluation: Path, output_root: Path, *, preview_root: Path | None = None) -> Path:
    config = read_json(training / "config.resolved.json")
    data = AircraftData.load(Path(config["data_root"]))
    audit = audit_training(training, data)
    audit_inventory(evaluation)
    source = read_json(evaluation / "training_source.json")
    if source["run_id"] != training.name or source["summary_sha256"] != file_hash(training / "summary.json"):
        raise ValueError("Evaluation trained-checkpoint source mismatch")
    functional = audit_functional(training, config)
    output = output_root / unique_id("incremental")
    output.mkdir(parents=True, exist_ok=False)
    for name in ("taxonomy.json", "selection.json", "config.resolved.json", "environment.json", "preprocessing.json",
                 "source_hashes.json", "dataset_manifest_hashes.json", "compatibility.json"):
        shutil.copyfile(training / name, output / name)
    if (training / "training_sources").exists():
        shutil.copytree(training / "training_sources", output / "training_sources")
    shutil.copyfile(evaluation / "source_hashes.json", output / "inference_source_hashes.json")
    shutil.copytree(evaluation / "inference_sources", output / "inference_sources")
    write_json(output / "training_audit.json", audit)
    write_json(output / "functional_audit.json", functional)
    write_json(output / "source.json", {"training_run_id": training.name, "evaluation_run_id": evaluation.name,
               "training_summary_sha256": file_hash(training / "summary.json"), "evaluation_summary_sha256": file_hash(evaluation / "summary.json"),
               "inference_source_sha256": object_hash(read_json(evaluation / "source_hashes.json"))})
    entries = read_json(evaluation / "entries.json")
    published_entries = []
    for e in entries:
        model = load_system(training, e["seed"], e["architecture"], e["stage"])
        meta = read_json(local_path(evaluation, e["path"]) / "metadata.json")
        if meta["model_sha256"] != model_hash(model):
            raise ValueError("Test prediction is not bound to selected incremental weights")
        if meta["config"]["device"] != config["device"]:
            raise ValueError("Review requires the configured inference hardware")
        if meta["evaluation_ids_sha256"] != object_hash([r.image_id for r in data.splits["test"]]):
            raise ValueError("Evaluation does not contain every original official test image")
        del model
        target = output / "evaluations" / f"seed-{e['seed']}" / e["architecture"] / e["stage"] / e["routing"]
        shutil.copytree(local_path(evaluation, e["path"]), target)
        published_entries.append({**e, "path": target.relative_to(output).as_posix()})
    write_json(output / "evaluations.json", published_entries)
    nodes = []
    for seed in config["seeds"]:
        bundle = read_json(training / f"seed-{seed}/bundle.json")
        target = output / f"seed-{seed}"
        target.mkdir()
        references = [("FLAT", "before", "variant", bundle["flat"])]
        for a in ARCHITECTURES:
            references += [(a, "before", key, entry) for key, entry in bundle[a]["initial"].items()]
            references.append((a, "after", bundle["affected_key"], bundle[a]["replacement"]))
            replacement = local_path(training, bundle[a]["replacement"]["path"])
            shutil.copyfile(replacement.parent / "resources.json", target / f"resources-{a}.json")
        for a, stage, key, entry in references:
            local = local_path(training, entry["path"])
            meta = read_json(local.with_suffix(".json"))["metadata"]
            spec = meta.get("node")
            node_key = spec["key"] if spec else "variant"
            history = target / f"{a}-{stage}-{node_key}.csv"
            shutil.copyfile(local.parent / "history.csv", history)
            if (local.parent / "resources.json").exists():
                node_resources = read_json(local.parent / "resources.json")
            else:
                selected = read_json(local.parent / "selection.json")
                node_resources = {k: selected[k] for k in ("trainable_parameters", "optimizer_steps", "training_image_exposures")}
            nodes.append({"seed": seed, "architecture": a, "stage": stage, "key": node_key,
                "best_epoch": meta["best_epoch"], "validation_macro_recall": meta["validation_macro_recall"],
                "global_ids": spec["global_ids"] if spec else meta["variant_ids"],
                "classes": len(spec["global_ids"] if spec else meta["variant_ids"]),
                "hidden_dimension": meta.get("hidden_dimension"), "backbone_sha256": meta.get("backbone_sha256"),
                "checkpoint_sha256": entry["sha256"], "state_sha256": entry["state_sha256"],
                "training_images": len(meta["training_image_ids"]), "selection_images": len(meta["selection_image_ids"]),
                "training_ids_sha256": object_hash(meta["training_image_ids"]),
                "selection_ids_sha256": object_hash(meta["selection_image_ids"]),
                "resources": node_resources,
                "history": history.relative_to(output).as_posix()})
        initial_costs = {"FLAT": read_json(local_path(training, bundle["flat"]["path"]).parent / "resources.json"),
                         "encoder": bundle["encoder"], "initial_checkpoint_hashes": {a: bundle[a]["initial"] for a in ARCHITECTURES}}
        # Node contracts contain private supervised IDs; publish resource totals only.
        initial_costs["IND-2"] = [read_json(local_path(training, entry["path"]).parent / "resources.json")
                                   for entry in bundle["IND-2"]["initial"].values()]
        shared_first = next(iter(bundle["FT-FROZEN-2"]["initial"].values()))
        initial_costs["shared_training_wall_seconds"] = read_json(local_path(training, shared_first["path"]).parent.parent / "summary.json")["training_wall_seconds"]
        write_json(target / "initial-costs.json", initial_costs)
    write_json(output / "nodes.json", nodes)
    records = calculate(output)
    write_json(output / "aggregate.json", records)
    taxonomy = Taxonomy.from_dict(read_json(output / "taxonomy.json"))
    draw_figures(output, records, nodes, taxonomy, read_json(output / "selection.json"), preview_root)
    render_report(output, records)
    write_json(output / "summary.json", {"status": "complete", "kind": "phase7-review", "artifacts_sha256": inventory(output)})
    audit_publication(output)
    return output


def audit_publication(output: Path) -> dict:
    """No local data/weights needed; reject tampering and recompute all scores."""
    audit_inventory(output)
    taxonomy = Taxonomy.from_dict(read_json(output / "taxonomy.json"))
    if (len(taxonomy.variants), len(taxonomy.families), len(taxonomy.manufacturers)) != (100, 70, 30):
        raise ValueError("Publication requires the validated 100/70/30 edition")
    for directory, manifest in (("training_sources", "source_hashes.json"), ("inference_sources", "inference_source_hashes.json")):
        if (output / manifest).exists() != (output / directory).exists():
            raise ValueError("Missing exact source snapshot or manifest")
        if (output / directory).exists():
            hashes = {p.relative_to(output / directory).as_posix(): file_hash(p) for p in (output / directory).rglob("*.py")}
            if hashes != read_json(output / manifest):
                raise ValueError("Published exact source snapshot mismatch")
    actual = calculate(output)
    if actual != read_json(output / "aggregate.json"):
        raise ValueError("Published incremental aggregates disagree with test predictions")
    if any(r["predicted"]["groups"]["all"][s]["images"] != 3333 for r in actual for s in ("before", "after")):
        raise ValueError("Publication requires all 3,333 official test images")
    forbidden = {".pt", ".pth", ".npy", ".npz", ".jpg", ".jpeg", ".png", ".tar", ".gz"}
    for path in output.rglob("*"):
        if path.is_file() and path.suffix in forbidden:
            raise ValueError("Publication includes data, cache or weights")
        if path.suffix == ".json":
            raw = path.read_text(encoding="utf-8")
            if '"training_image_ids"' in raw or '"selection_image_ids"' in raw:
                raise ValueError("Publication contains supervised image-ID lists")
    return {"publication_run_id": output.name, "seed_architecture_pairs": len(actual),
            "artifact_only_recomputed": True, "data_and_weights_excluded": True}


def rebuild(source: Path, output: Path, *, preview_root: Path | None = None) -> None:
    """Rebuild every generated figure/report from an audited public review bundle."""
    audit_publication(source)
    output.mkdir(parents=True, exist_ok=False)
    generated = {"plots", "REPORT.md", "classwise-before-after.csv", "summary.json"}
    for path in source.iterdir():
        if path.name not in generated:
            if path.is_dir():
                shutil.copytree(path, output / path.name)
            else:
                shutil.copyfile(path, output / path.name)
    records = calculate(output)
    draw_figures(output, records, read_json(output / "nodes.json"), Taxonomy.from_dict(read_json(output / "taxonomy.json")), read_json(output / "selection.json"), preview_root)
    render_report(output, records)
    write_json(output / "summary.json", {"status": "complete", "kind": "phase7-review", "artifacts_sha256": inventory(output)})
    audit_publication(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("results/stage7"))
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--rebuild", type=Path)
    parser.add_argument("--rebuild-output", type=Path)
    parser.add_argument("--preview-root", type=Path)
    args = parser.parse_args()
    if args.audit:
        print(audit_publication(args.audit))
    elif args.rebuild and args.rebuild_output:
        rebuild(args.rebuild, args.rebuild_output, preview_root=args.preview_root)
        print(args.rebuild_output)
    elif args.training and args.evaluation:
        print(publish(args.training, args.evaluation, args.output_root, preview_root=args.preview_root))
    else:
        parser.error("Supply --training and --evaluation, --audit, or --rebuild and --rebuild-output")


if __name__ == "__main__":
    main()
