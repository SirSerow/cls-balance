"""Audit and publish Phase 5 validation results, excluding data and weights."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil
import statistics

from .artifacts import file_hash, object_hash, write_json
from .evaluate import audit_run, model_hash
from .models.cascade import node_plan
from .report_flat import read_json, save_svg as write_svg
from .taxonomy import Taxonomy
from .train_shared import load_shared
from .train_independent import unique_id


def csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def audit_training_run(path: Path) -> dict:
    """Recompute metrics, local selection/exposures, and deployment fingerprints."""
    summary = read_json(path / "summary.json")
    if (summary.get("status") != "complete" or summary.get("test_evaluated") is not False
            or summary.get("frozen_weights_and_buffers_unchanged") is not True):
        raise ValueError("Incomplete Phase 5 run")
    inventory = {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file() and p != path / "summary.json"}
    if inventory != set(summary["artifacts_sha256"]):
        raise ValueError("Incomplete/extra run artifacts")
    for name, digest in summary["artifacts_sha256"].items():
        target = (path / name).resolve()
        if not target.is_relative_to(path.resolve()) or file_hash(target) != digest:
            raise ValueError(f"Run checksum mismatch: {name}")
    for name, key in (("taxonomy.json", "taxonomy_sha256"), ("preprocessing.json", "preprocessing_sha256"),
                      ("config.resolved.json", "config_sha256"), ("source_hashes.json", "source_hashes_sha256")):
        # Resolved configs contain integer seed keys before JSON serialization.
        # Bind canonical written bytes; reparsing turns those keys into strings
        # and can change their sorted order (e.g. seeds 2 and 10).
        digest = file_hash(path / name) if key == "config_sha256" else object_hash(read_json(path / name))
        if digest != summary[key]:
            raise ValueError(f"Run contract mismatch: {key}")
    manifests = read_json(path / "dataset_manifest_hashes.json")["canonical_records"]
    if object_hash(manifests) != summary["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest mismatch")
    taxonomy = Taxonomy.from_dict(read_json(path / "taxonomy.json"))
    specs = {s.key: s for d in summary["depths"] for s in node_plan(taxonomy, d)}
    if len(specs) != summary["unique_trained_heads"]:
        raise ValueError("Head inventory mismatch")
    for key, spec in specs.items():
        node = path / "nodes" / key
        selected = read_json(node / "summary.json")
        metadata = read_json(node / "head.json")["metadata"]
        history = csv_rows(node / "history.csv")
        best = max(history, key=lambda h: float(h["val_macro_recall"]))
        if (metadata["node"] != spec.to_dict() or selected["node"] != spec.to_dict()
                or any(metadata[k] != summary[k] for k in ("run_id", "seed", "backbone_sha256", "config_sha256"))
                or not selected["exact_reload_logits"]
                or int(best["epoch"]) != selected["best_epoch"] or metadata["best_epoch"] != selected["best_epoch"]
                or float(best["val_macro_recall"]) != selected["validation_macro_recall"]
                or len(history) != selected["epochs_completed"]
                or sum(int(h["train_optimizer_steps"]) for h in history) != selected["optimizer_steps"]
                or sum(int(h["train_images"]) for h in history) != selected["training_image_exposures"]
                or set(metadata["training_image_ids"]) & set(metadata["selection_image_ids"])):
            raise ValueError("Head selection/provenance mismatch")
    expected = {f"{summary['backbone']}-{d}/{r}" for d in summary["depths"] for r in ("predicted", "oracle")}
    if set(summary["evaluations"]) != expected:
        raise ValueError("Missing hard/oracle evaluations")
    for depth in summary["depths"]:
        adapter = load_shared(path, depth)
        fingerprint = model_hash(adapter)
        paired = []
        for routing in ("predicted", "oracle"):
            entry = summary["evaluations"][f"{summary['backbone']}-{depth}/{routing}"]
            evaluation = (path / entry["path"]).resolve()
            if not evaluation.is_relative_to(path.resolve()) or audit_run(evaluation) != entry["audit"]:
                raise ValueError("Evaluation audit mismatch")
            meta = read_json(evaluation / "metadata.json")
            if (meta["model_sha256"] != fingerprint or meta["model_id"] != summary["run_id"]
                    or meta["manifest_records_sha256"] != manifests
                    or meta["config"]["architecture"] != f"{summary['backbone']}-{depth}"
                    or meta["config"]["routing"] != routing or meta["config"]["split"] != "val"
                    or meta["config"]["seed"] != summary["seed"]):
                raise ValueError("Evaluation model binding mismatch")
            paired.append(read_json(evaluation / "metrics.json"))
        penalty = 100 * (paired[1]["variant_macro_recall"] - paired[0]["variant_macro_recall"])
        if penalty != summary["routing_penalties_pp"][str(depth)]:
            raise ValueError("Routing penalty mismatch")
    return summary


def draw_figures(output: Path, records: list[dict], nodes: list[dict], baselines: list[dict],
                 preview_root: Path | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import FancyBboxPatch

    def save_svg(figure, path: Path) -> None:
        write_svg(figure, path)
        if preview_root is not None:
            preview_root.mkdir(parents=True, exist_ok=True)
            preview = preview_root / (path.stem + ".png")
            with preview.open("xb") as stream:
                figure.savefig(stream, format="png", dpi=120)

    plt.rcParams.update({"svg.fonttype": "none", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(records))
    for offset, route, color in ((-.18, "hard", "#126782"), (.18, "oracle", "#d68a38")):
        bars = ax.bar(x + offset, [100 * r[route]["variant_macro_recall"] for r in records], .36,
                      label="Predicted pipeline" if route == "hard" else "Oracle diagnostic", color=color)
        ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.set_xticks(x, [f"{r['architecture']}\ns{r['seed']}" for r in records])
    ax.set(ylabel="Validation variant macro recall (%)", ylim=(0, 100), title="Shared frozen features · complete official validation split")
    ax.legend(); fig.tight_layout(); save_svg(fig, output / "routing-penalty.svg"); plt.close(fig)
    keys = list(dict.fromkeys(n["key"] for n in nodes))
    for metric, ylabel, scale in (("macro_recall", "Local recall (%)", 100), ("loss", "Cross-entropy", 1)):
        rows = (len(keys) + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=(14, 2.7 * rows), squeeze=False)
        for ax, key in zip(axes.flat, keys):
            for node in (n for n in nodes if n["key"] == key):
                epochs = [int(h["epoch"]) for h in node["history"]]
                line, = ax.plot(epochs, [scale * float(h[f"val_{metric}"]) for h in node["history"]], label=f"{node['backbone']} s{node['seed']} val")
                ax.plot(epochs, [scale * float(h[f"train_{metric}"]) for h in node["history"]], "--", color=line.get_color(), alpha=.6)
                best = node["history"][node["best_epoch"] - 1]
                ax.plot(node["best_epoch"], scale * float(best[f"val_{metric}"]), "o", color=line.get_color(), ms=4)
            ax.set(title=next(n["label"] for n in nodes if n["key"] == key), xlabel="Epoch", ylabel=ylabel)
            if metric == "macro_recall":
                ax.set_ylim(0, 100)
            ax.grid(alpha=.15); ax.legend(fontsize=6)
        for ax in list(axes.flat)[len(keys):]:
            ax.axis("off")
        fig.suptitle("Independent heads · solid = validation; dashed = augmented training; dot = selected epoch")
        fig.tight_layout(rect=(0, 0, 1, .98)); save_svg(fig, output / f"head-{metric}.svg"); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    bottom = np.zeros(len(records))
    for stage, color in (("manufacturer", "#bd403b"), ("family", "#e6a03b"), ("variant", "#6c71a1"), ("correct", "#377f6e")):
        values = [100 * r["hard"]["routing_diagnostics"]["first_wrong_stage"].get(stage, 0) / r["hard"]["images"] for r in records]
        axes[0].bar(x, values, bottom=bottom, label=stage, color=color); bottom += values
    axes[0].set(ylabel="Validation images (%)", title="First wrong decision", ylim=(0, 100)); axes[0].legend()
    axes[1].bar(x, [100 * r["hard"]["routing_diagnostics"]["singleton_bypasses"]["variant"]["rate"] for r in records], color="#126782")
    axes[1].set(ylabel="Validation images (%)", title="Deterministic variant singleton bypass", ylim=(0, 100))
    for ax in axes:
        ax.set_xticks(x, [f"{r['architecture']}\ns{r['seed']}" for r in records])
    fig.tight_layout(); save_svg(fig, output / "routing-paths.svg"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(records))
    categories = (
        ("Correct variant", "#377f6e", lambda m: m["variant_top1"]),
        ("Wrong variant, correct family", "#6c71a1", lambda m: m["family_from_variant"]["accuracy"] - m["variant_top1"]),
        ("Wrong family, correct manufacturer", "#e6a03b", lambda m: m["manufacturer_from_variant"]["accuracy"] - m["family_from_variant"]["accuracy"]),
        ("Wrong manufacturer", "#bd403b", lambda m: 1 - m["manufacturer_from_variant"]["accuracy"]),
    )
    for label, color, fraction in categories:
        values = [100 * fraction(r["hard"]) for r in records]
        ax.bar(x, values, bottom=bottom, label=label, color=color); bottom += values
    ax.set_xticks(x, [f"{r['architecture']}\ns{r['seed']}" for r in records])
    ax.set(ylabel="Validation images (%)", ylim=(0, 100), title="Final variant errors projected through the taxonomy")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    fig.tight_layout(); save_svg(fig, output / "hierarchy-errors.svg"); plt.close(fig)
    for record in records:
        matrix = np.asarray(record["hard"]["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(matrix / matrix.sum(1, keepdims=True), vmin=0, vmax=1, cmap="Blues")
        ax.set(xlabel="Predicted global variant ID", ylabel="True global variant ID", title=f"{record['architecture']} seed {record['seed']} · validation")
        fig.colorbar(im, ax=ax, label="Per-class recall fraction"); fig.tight_layout()
        save_svg(fig, output / f"confusion-{record['architecture']}-s{record['seed']}.svg"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(15, 6))
    order = sorted(range(len(records[0]["hard"]["classwise"])), key=lambda i: statistics.mean(r["hard"]["classwise"][i]["recall"] for r in records))
    for r in records:
        ax.plot(range(len(order)), [100 * r["hard"]["classwise"][i]["recall"] for i in order], ".", label=f"{r['architecture']} s{r['seed']}")
    ax.set_xticks(range(len(order)), [records[0]["hard"]["classwise"][i]["variant_name"] for i in order], rotation=90, fontsize=6)
    ax.set(ylabel="Validation variant recall (%)", ylim=(-2, 102), title="Per-variant recall · full predicted pipeline"); ax.legend()
    fig.tight_layout(); save_svg(fig, output / "classwise-validation.svg"); plt.close(fig)
    combined = baselines + [{"architecture": r["architecture"], "seed": r["seed"], "macro_recall": r["hard"]["variant_macro_recall"]} for r in records]
    fig, ax = plt.subplots(figsize=(9, 5))
    architectures = list(dict.fromkeys(r["architecture"] for r in combined))
    for i, architecture in enumerate(architectures):
        subset = [r for r in combined if r["architecture"] == architecture]
        values = [100 * r["macro_recall"] for r in subset]
        ax.bar(i, statistics.mean(values), yerr=statistics.stdev(values) if len(values) > 1 else None, color="#126782", alpha=.7, capsize=6)
        ax.scatter([i] * len(values), values, color="#b45328", zorder=3)
        ax.text(i, max(values) + 2, f"{statistics.mean(values):.2f}%", ha="center")
    ax.set_xticks(range(len(architectures)), architectures)
    ax.set(ylabel="Validation variant macro recall (%)", ylim=(0, 100), title="Complete predicted pipelines · dots = seeds; whiskers = sample SD")
    fig.tight_layout(); save_svg(fig, output / "comparison.svg"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 4)); ax.set(xlim=(0, 12), ylim=(0, 4)); ax.axis("off")
    boxes = [(0.15, 1.4, "224 × 224 image\nFixed preprocessing"), (3.15, 1.4, "Frozen EfficientNet\nONE feature pass"),
             (6.15, 2.3, "Independent MLP\nManufacturer router"), (6.15, .45, "Selected MLP or\nsingleton bypass"),
             (9.15, .45, "Global variant ID\nScore entire route")]
    for left, bottom, label in boxes:
        ax.add_patch(FancyBboxPatch((left, bottom), 2.55, 1.0, boxstyle="round,pad=.06", facecolor="#e9f2f5", edgecolor="#126782"))
        ax.text(left + 1.275, bottom + .5, label, ha="center", va="center")
    for start, end in (((2.75, 1.9), (3.05, 1.9)), ((5.75, 1.9), (6.05, 2.8)), ((5.75, 1.9), (6.05, .95)),
                       ((7.425, 2.23), (7.425, 1.53)), ((8.75, .95), (9.05, .95))):
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": "#126782"})
    ax.text(6, 3.8, "Shared frozen features: one encoder, separately fitted heads", ha="center", fontsize=14)
    ax.text(6, .02, "Two-stage path shown. Depth 3 adds a conditional family head using the same features. Encoder weights and buffers stay fixed.", ha="center", fontsize=9)
    fig.tight_layout(); save_svg(fig, output / "architecture.svg"); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    targets = list({r["run_id"]: r for r in records}.values())
    labels = [f"{r['architecture'].rsplit('-', 1)[0]}\ns{r['seed']}" for r in targets]
    positions = range(len(targets))
    extraction = [r["resources"]["feature_extraction_seconds"] / 60 for r in targets]
    heads = [r["resources"]["head_epoch_seconds"] / 60 for r in targets]
    axes[0].bar(positions, extraction, label="Shared feature extraction", color="#126782")
    axes[0].bar(positions, heads, bottom=extraction, label="Head training + validation epochs", color="#d68a38")
    axes[0].set_xticks(positions, labels)
    axes[0].set(ylabel="Measured minutes", title="Shared fitting components · excludes source FLAT"); axes[0].legend(fontsize=8)
    encoder_bytes = [r["resources"]["encoder_bytes"] / 1e6 for r in targets]
    head_bytes = [r["resources"]["head_bytes"] / 1e6 for r in targets]
    axes[1].bar(positions, encoder_bytes, label="Encoder", color="#126782")
    axes[1].bar(positions, head_bytes, bottom=encoder_bytes, label="All trained heads", color="#d68a38")
    axes[1].set_xticks(positions, labels)
    axes[1].set(ylabel="Stored artifacts (decimal MB)", title="Total stored weights · all trained depths"); axes[1].legend(fontsize=8)
    fig.tight_layout(); save_svg(fig, output / "resources.svg"); plt.close(fig)


def publish(runs: list[Path], output_root: Path, *, flat_runs: list[Path] = (), independent_runs: list[Path] = (),
            preview_root: Path | None = None) -> Path:
    summaries = [audit_training_run(p) for p in runs]
    if len({(s["backbone"], s["seed"]) for s in summaries}) != len(summaries):
        raise ValueError("Duplicate backbone/seed")
    seed_sets = {backbone: {s["seed"] for s in summaries if s["backbone"] == backbone}
                 for backbone in {s["backbone"] for s in summaries}}
    if len({tuple(sorted(seeds)) for seeds in seed_sets.values()}) != 1:
        raise ValueError("Backbone comparisons require matching seed sets")
    for key in ("depths", "taxonomy_sha256", "preprocessing_sha256", "dataset_manifest_sha256"):
        if len({object_hash(s[key]) for s in summaries}) != 1:
            raise ValueError(f"Incompatible runs: {key}")
    configs = [read_json(p / "config.resolved.json") for p in runs]
    if len({object_hash({k: v for k, v in c.items() if k not in {"seed", "seeds", "output_root", "flat_runs", "backbone"}}) for c in configs}) != 1:
        raise ValueError("Different training recipes")
    environments = [read_json(p / "environment.json") for p in runs]
    for key in ("device", "gpus", "torch", "cuda_runtime", "cudnn", "packages", "runtime_policy"):
        if len({object_hash(e[key]) for e in environments}) != 1:
            raise ValueError(f"Incompatible environments: {key}")
    # Later gated benchmarks/plots may be added without changing fitting or
    # classification. Every module used for those operations must still match.
    unused = {f"src/cls_balance/{name}.py" for name in
              ("report_shared", "report_independent", "residency", "benchmark", "plots", "incremental")}
    sources = [{k: v for k, v in read_json(p / "source_hashes.json").items() if k not in unused} for p in runs]
    if len({object_hash(s) for s in sources}) != 1:
        raise ValueError("Training sources differ across seeds")
    output = output_root / unique_id("shared"); output.mkdir(parents=True, exist_ok=False)
    records, nodes, baselines = [], [], []
    for run, summary in zip(runs, summaries, strict=True):
        target = output / f"{summary['backbone']}-seed-{summary['seed']}"; target.mkdir()
        for name in ("config.resolved.json", "environment.json", "source_hashes.json", "taxonomy.json", "preprocessing.json",
                     "dataset_manifest_hashes.json", "backbone_source.json", "resources.json"):
            shutil.copyfile(run / name, target / name)
        write_json(target / "checkpoint_references.json", {"run_id": summary["run_id"],
            "backbone_sha256": summary["backbone_sha256"], "artifacts_sha256": summary["artifacts_sha256"]})
        taxonomy = Taxonomy.from_dict(read_json(run / "taxonomy.json"))
        for spec in {s.key: s for d in summary["depths"] for s in node_plan(taxonomy, d)}.values():
            node = run / "nodes" / spec.key
            selected = read_json(node / "summary.json")
            label = "Manufacturer router" if spec.parent_id is None else f"{spec.stage}: {getattr(taxonomy, 'manufacturers' if spec.parent_stage == 'manufacturer' else 'families')[spec.parent_id]}"
            nodes.append({**selected, "key": spec.key, "label": label, "backbone": summary["backbone"],
                          "seed": summary["seed"], "history": csv_rows(node / "history.csv")})
            destination = target / "heads" / spec.key; destination.mkdir(parents=True)
            for name in ("summary.json", "history.csv"):
                shutil.copyfile(node / name, destination / name)
        for depth in summary["depths"]:
            architecture = f"{summary['backbone']}-{depth}"
            record = {"architecture": architecture, "seed": summary["seed"], "run_id": summary["run_id"],
                      "publication_directory": target.name, "resources": summary["resources"]}
            for routing, label in (("predicted", "hard"), ("oracle", "oracle")):
                evaluation = run / summary["evaluations"][f"{architecture}/{routing}"]["path"]
                destination = target / architecture / routing
                shutil.copytree(evaluation, destination)
                record[label] = read_json(evaluation / "metrics.json")
            records.append(record)
    from .report_flat import audit_training_run as audit_flat
    from .report_independent import audit_training_run as audit_independent
    for architecture, paths, audit in (("FLAT", flat_runs, audit_flat), ("IND-2", independent_runs, audit_independent)):
        baseline_seeds = []
        for run in paths:
            summary = audit(run)
            baseline_seeds.append(summary["seed"])
            if summary["seed"] not in {s["seed"] for s in summaries} or any(summary[k] != summaries[0][k] for k in
                    ("taxonomy_sha256", "preprocessing_sha256", "dataset_manifest_sha256")):
                raise ValueError("Baseline seed/data mismatch")
            key = "val" if architecture == "FLAT" else "IND-2/predicted"
            evaluation = run / summary["evaluations"][key]["path"]
            reference = next(r for r in records if r["seed"] == summary["seed"])
            candidate_meta = read_json(evaluation / "metadata.json")
            shared_meta = read_json(output / reference["publication_directory"] / reference["architecture"] / "predicted/metadata.json")
            for field in ("evaluation_ids_sha256", "preprocessing_sha256"):
                if candidate_meta[field] != shared_meta[field]:
                    raise ValueError("Unpaired baseline images/preprocessing")
            for field in ("gpus", "torch", "cuda_runtime", "device"):
                if candidate_meta["environment"][field] != shared_meta["environment"][field]:
                    raise ValueError("Baseline evaluation hardware/software mismatch")
            metrics = read_json(evaluation / "metrics.json")
            baselines.append({"architecture": architecture, "seed": summary["seed"], "run_id": summary["run_id"],
                              "macro_recall": metrics["variant_macro_recall"], "evaluation_ids_sha256": candidate_meta["evaluation_ids_sha256"]})
        if paths and sorted(baseline_seeds) != sorted({s["seed"] for s in summaries}):
            raise ValueError("Baseline comparisons require exactly matching seeds")
    write_json(output / "baselines.json", baselines)
    aggregates = {}
    for architecture in sorted({r["architecture"] for r in records}):
        values = [100 * r["hard"]["variant_macro_recall"] for r in records if r["architecture"] == architecture]
        aggregates[architecture] = {"mean_percent": statistics.mean(values), "sample_sd_pp": statistics.stdev(values) if len(values) > 1 else None}
    write_json(output / "aggregate.json", {"test_evaluated": False, "architectures": aggregates, "source_run_ids": [s["run_id"] for s in summaries]})
    draw_figures(output, records, nodes, baselines, preview_root)
    aggregate_rows = ["| Architecture | Mean validation macro recall (%) | Sample SD (pp) |",
                      "| --- | ---: | ---: |"]
    for architecture, values in aggregates.items():
        spread = f"{values['sample_sd_pp']:.2f}" if values["sample_sd_pp"] is not None else "N/A (one seed)"
        aggregate_rows.append(f"| {architecture} | {values['mean_percent']:.2f} | {spread} |")
    rows = ["| Architecture | Seed | Hard macro recall (%) | Oracle (%) | Penalty (pp) | Top-1 (%) |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    secondary = ["| Architecture / seed | Macro-F1 (%) | Family from final variant (%) | Manufacturer from final variant (%) | Manufacturer router (%) |",
                 "| --- | ---: | ---: | ---: | ---: |"]
    costs = ["| Backbone / seed | Head fitting incl. extraction (min) | Source FLAT (min) | Combined training (min) | Stored encoder + all heads (MB) |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for r in records:
        hard, oracle = (100 * r[k]["variant_macro_recall"] for k in ("hard", "oracle"))
        rows.append(f"| {r['architecture']} | {r['seed']} | {hard:.2f} | {oracle:.2f} | {oracle-hard:.2f} | {100*r['hard']['variant_top1']:.2f} |")
        m = r["hard"]
        secondary.append(f"| {r['architecture']} / {r['seed']} | {100*m['variant_macro_f1']:.2f} | "
                         f"{100*m['family_from_variant']['accuracy']:.2f} | {100*m['manufacturer_from_variant']['accuracy']:.2f} | "
                         f"{100*m['routing_diagnostics']['manufacturer_router']['accuracy']:.2f} |")
    for run, summary in zip(runs, summaries, strict=True):
        source = read_json(run / "backbone_source.json")
        source_time = source.get("flat_resources", {}).get("training_wall_seconds")
        shared_time = summary["resources"]["shared_training_wall_seconds"]
        source_label = f"{source_time/60:.2f}" if source_time is not None else "External ImageNet pretraining (unmeasured)"
        combined_label = f"{(shared_time+source_time)/60:.2f}" if source_time is not None else "Not measurable here"
        stored = (summary["resources"]["encoder_bytes"] + summary["resources"]["head_bytes"]) / 1e6
        costs.append(f"| {summary['backbone']} / {summary['seed']} | {shared_time/60:.2f} | {source_label} | {combined_label} | {stored:.2f} |")
    text = f"""# Phase 5 — measured frozen shared-backbone heads

Complete official FGVC-Aircraft train/validation splits: 3,334 / 3,333 images, 100 variants / 70 families / 30 manufacturers. These are validation measurements; test inference and the Phase 6 latency/FPS benchmark remain deferred. No test labels entered training or selection.

## Experimental context and controls

Seed-matched Phase 3 FLAT encoders supply FT-FROZEN features; generic FROZEN uses published ImageNet V1 features. Each head is a separately initialized Linear(1280,{configs[0]['hidden_dimension']}) → ReLU → Linear(n) with independent AdamW state, shuffle generator, early stopping and checkpoint. Learning rate {configs[0]['learning_rate']}, weight decay {configs[0]['weight_decay']}, cap {configs[0]['epochs']}, patience {configs[0]['patience']}; strict local validation macro-recall improvement, earliest tie. The same selected router can be reused across depths. Singleton nodes have no classifier.

FP32, TF32 disabled; {environments[0]['gpus']}; Torch {environments[0]['torch']}, CUDA {environments[0]['cuda_runtime']}. Per-seed environments record Git SHA/dirty state and source hashes bind the actual training code. Dataset manifest: `{summaries[0]['dataset_manifest_sha256']}`. Preprocessing: `{summaries[0]['preprocessing_sha256']}`. Remove the bottom 20 pixels, preserve image aspect ratio, black-pad to 224×224 and normalize with ImageNet V1 statistics.

Fresh seeded flip/photometric augmentation is extracted once per training image per epoch, then shared read-only among independently optimized heads. Validation features are deterministic and held in memory for local selection. Disk training cache enabled: **{configs[0]['cache_root'] is not None}**. Full predicted/oracle evaluation reads images through the common batch-1 adapter; no deployment embedding cache is used.

![Architecture](architecture.svg)

## Classification and routing measurements

{chr(10).join(rows)}

Across-seed aggregate (sample standard deviation, not a confidence interval):

{chr(10).join(aggregate_rows)}

{chr(10).join(secondary)}

![Comparison](comparison.svg)

Optional FLAT/IND-2 comparison bars are audited paired validation results. Baseline run IDs and exact values are in `baselines.json`; their complete prediction evidence is in the Phase 3/4 publications. These validation comparisons do not establish the preregistered test endpoint or a latency advantage.

![Routing penalty](routing-penalty.svg)

![Error attribution and singleton paths](routing-paths.svg)

![Within-family and cross-family errors](hierarchy-errors.svg)

## Learning, local support and per-class performance

![Head learning](head-macro_recall.svg)

![Head losses](head-loss.svg)

Each head's selected epoch, local validation macro recall, optimizer steps, training exposures and history accompany the report. Local scores use true-parent supervised subsets; only the complete predicted-route scores above represent deployable performance. Augmented training curves and deterministic validation curves have different input/mode conditions. Confusion matrices for every seed accompany the report; global names/support are in the per-image and classwise files.

![Per-class recall](classwise-validation.svg)

## Resources and exit checks

All selected heads reproduce logits exactly after reload. Encoder parameters, BatchNorm buffers, eval mode and absent gradients are checked before training and after every extraction/fit sequence. The common adapter executes the backbone exactly once per image, including singleton routes. Synthetic regression checks exercise two-/three-stage hard/oracle routing, cache corruption/version rejection and compatible replacement of one head while all untouched head weights/logits remain bit-for-bit unchanged. Replacement is a compatibility mechanism; the held-out-new-variant experiment remains Phase 7.

`resources.json` distinguishes head fitting and shared extraction time, unique feature-image exposures, GPU-hours, encoder bytes and all stored head bytes. `backbone_source.json` separately reports the source FLAT training cost; it must be included when comparing total training compute. Head training is not compute matched against independent CNN fitting. OS/driver clocks are unlocked and desktop load can affect wall time. Training duration and stored bytes do not substitute for measured inference latency or memory.

{chr(10).join(costs)}

Costs count the source FLAT once per seed and shared fitting once across all trained depths. Stored bytes include the encoder and every trained head; this is an artifact-footprint measurement, not executed computation or peak GPU memory. Generic ImageNet pretraining cost is external and unavailable.

![Measured training components and storage](resources.svg)

## Limitations and deferred measurements

This bundle measures **{', '.join(aggregates)}**. Configurations absent from this bundle remain unmeasured here. Generic ImageNet-frozen and three-stage runs follow the primary two-stage comparison/benchmark execution gate; both paths are implemented and integration checked. Test accuracy, confidence intervals, full-pipeline p95/FPS, memory/MAC measurements and introduced-class updates remain unmeasured here. No images, training-ID lists, embeddings or checkpoint tensors are published. The Phase 1 archive-checksum discrepancy remains documented.

## Reproduction and audit links

```powershell
.venv-cuda/Scripts/python.exe -m cls_balance.train_shared --config configs/shared.yaml
.venv-cuda/Scripts/python.exe -m cls_balance.report_shared --runs <seed42-run> <seed43-run> <seed44-run> --output-root results/stage5
.venv/Scripts/python.exe scripts/freeze_tests.py --check
.venv/Scripts/python.exe -m pytest -q tests checks
```

Replace placeholders with the immutable run IDs below; optional `--flat-runs` and `--independent-runs` add audited paired validation comparisons. Full exact publication commands are recorded in the validation evidence. Repeated publication creates a new directory. Each nested evaluation includes per-image predictions, classwise metrics and recomputable aggregate metrics.

"""
    for r in records:
        text += f"- {r['architecture']} seed {r['seed']}: `{r['run_id']}`; [predicted metrics]({r['publication_directory']}/{r['architecture']}/predicted/metrics.json); [confusion](confusion-{r['architecture']}-s{r['seed']}.svg).\n"
    (output / "REPORT.md").write_text(text, encoding="utf-8", newline="\n")
    write_json(output / "publication.json", {"source_run_ids": [s["run_id"] for s in summaries],
        "report_source_sha256": file_hash(Path(__file__)),
        "artifacts_sha256": {p.relative_to(output).as_posix(): file_hash(p) for p in sorted(output.rglob("*")) if p.is_file()}})
    print(output, flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("results/stage5"))
    parser.add_argument("--flat-runs", nargs="*", type=Path, default=[])
    parser.add_argument("--independent-runs", nargs="*", type=Path, default=[])
    parser.add_argument("--preview-root", type=Path, help="Optional local PNG previews; use an ignored directory")
    args = parser.parse_args()
    publish(args.runs, args.output_root, flat_runs=args.flat_runs, independent_runs=args.independent_runs,
            preview_root=args.preview_root)


if __name__ == "__main__":
    main()
