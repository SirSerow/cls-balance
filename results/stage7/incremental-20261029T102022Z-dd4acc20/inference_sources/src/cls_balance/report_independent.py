"""Audit Phase 4 bundles and publish validation-only evidence and SVG figures.

No images, training-ID lists, checkpoints or embeddings enter the publication.
Requires requirements/phase3-plots.txt. Test inference stays deferred.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil
import statistics

from .artifacts import file_hash, object_hash, write_json
from .evaluate import audit_run
from .models.cascade import node_plan
from .report_flat import read_json, save_svg
from .taxonomy import Taxonomy
from .train_independent import audit_node, load_cascade, unique_id


def audit_training_run(path: Path) -> dict:
    """Verify nested checkpoint and per-image evidence before publishing figures."""
    summary = read_json(path / "summary.json")
    required = {"config.resolved.json", "source_hashes.json", "taxonomy.json", "preprocessing.json",
                "dataset_manifest_hashes.json", "environment.json"}
    depths = summary.get("depths", [])
    required |= {f"bundle-ind{d}.json" for d in depths}
    if (summary.get("status") != "complete" or summary.get("test_evaluated") is not False
            or not depths or set(depths) - {2, 3}
            or set(summary.get("artifacts_sha256", {})) != required):
        raise ValueError("Incomplete Phase 4 run")
    for name, digest in summary["artifacts_sha256"].items():
        if Path(name).name != name or file_hash(path / name) != digest:
            raise ValueError(f"Run checksum mismatch: {name}")
    taxonomy = Taxonomy.from_dict(read_json(path / "taxonomy.json"))
    for name, key in (("taxonomy.json", "taxonomy_sha256"), ("preprocessing.json", "preprocessing_sha256"),
                      ("config.resolved.json", "config_sha256"), ("source_hashes.json", "source_hashes_sha256")):
        if object_hash(read_json(path / name)) != summary[key]:
            raise ValueError(f"Run contract mismatch: {key}")
    manifests = read_json(path / "dataset_manifest_hashes.json")["canonical_records"]
    if object_hash(manifests) != summary["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest binding mismatch")
    expected_evals = {f"IND-{d}/{r}" for d in depths for r in ("predicted", "oracle")}
    if set(summary["evaluations"]) != expected_evals:
        raise ValueError("Missing hard/oracle evaluations")
    initializations = set()
    for depth in depths:
        resources = {key: 0 for key in ("training_wall_seconds", "optimizer_steps", "training_image_exposures",
                                        "trainable_parameters", "deployment_bytes")}
        bundle = read_json(path / f"bundle-ind{depth}.json")
        if (bundle["seed"] != summary["seed"] or bundle["run_id"] != summary["run_id"]
                or any(summary[k] != v for k, v in bundle["compatibility"].items())
                or set(bundle["nodes"]) != {s.key for s in node_plan(taxonomy, depth)}):
            raise ValueError("Bundle contract mismatch")
        for spec in node_plan(taxonomy, depth):
            entry = bundle["nodes"][spec.key]
            node = (path / entry["path"]).resolve()
            if not node.is_relative_to(path.resolve()):
                raise ValueError("Node path escapes run")
            checked = audit_node(node)
            contract = read_json(node / "contract.json")
            checkpoint_metadata = read_json(node / "node.json")["metadata"]
            if (checked["checkpoint_sha256"] != entry["checkpoint_sha256"]
                    or object_hash(contract["node"]) != object_hash(spec.to_dict())
                    or any(contract[k] != summary[k] for k in
                           ("run_id", "seed", "config_sha256", "source_hashes_sha256", *bundle["compatibility"]))
                    or set(contract["training_image_ids"]) & set(contract["selection_image_ids"])):
                raise ValueError("Node identity/data binding mismatch")
            for key in ("best_epoch", "validation_macro_recall", "initialization_features_sha256"):
                if checked[key] != checkpoint_metadata[key]:
                    raise ValueError("Node summary disagrees with checkpoint")
            with (node / "history.csv").open(encoding="utf-8", newline="") as stream:
                history = list(csv.DictReader(stream))
            selected = max(history, key=lambda h: float(h["val_macro_recall"]))
            if (len(history) != checked["epochs_completed"] or int(selected["epoch"]) != checked["best_epoch"]
                    or float(selected["val_macro_recall"]) != checked["validation_macro_recall"]
                    or sum(int(h["train_images"]) for h in history) != checked["resources"]["training_image_exposures"]
                    or sum(int(h["train_optimizer_steps"]) for h in history) != checked["resources"]["optimizer_steps"]
                    or any(int(h["train_images"]) != len(contract["training_image_ids"])
                           or int(h["val_images"]) != len(contract["selection_image_ids"]) for h in history)):
                raise ValueError("Node history/selection/exposure mismatch")
            if read_json(node / "resources.json") != checked["resources"]:
                raise ValueError("Node resource records disagree")
            for key in resources:
                resources[key] += checked["resources"][key]
            initializations.add(checked["initialization_features_sha256"])
        declared = summary["bundles"][f"IND-{depth}"]
        if declared["nodes"] != len(bundle["nodes"]) or any(declared["resources"][k] != v for k, v in resources.items()):
            raise ValueError("Bundle resource totals disagree")
        from .evaluate import model_hash
        fingerprint = model_hash(load_cascade(path, depth))
        paired_metadata = []
        for routing in ("predicted", "oracle"):
            entry = summary["evaluations"][f"IND-{depth}/{routing}"]
            evaluation = (path / entry["path"]).resolve()
            if not evaluation.is_relative_to(path.resolve()):
                raise ValueError("Evaluation path escapes run")
            if audit_run(evaluation) != entry["audit"]:
                raise ValueError("Saved evaluation audit mismatch")
            metadata = read_json(evaluation / "metadata.json")
            if (metadata["model_id"] != summary["run_id"] or metadata["manifest_records_sha256"] != manifests
                    or metadata["model_sha256"] != fingerprint
                    or metadata["config"]["routing"] != routing or metadata["config"]["split"] != "val"
                    or metadata["config"]["architecture"] != f"IND-{depth}"
                    or metadata["config"]["seed"] != summary["seed"]
                    or metadata["taxonomy_sha256"] != summary["taxonomy_sha256"]
                    or metadata["preprocessing_sha256"] != summary["preprocessing_sha256"]):
                raise ValueError("Evaluation identity mismatch")
            paired_metadata.append(metadata)
        for key in ("model_sha256", "evaluation_ids_sha256", "training_ids_sha256", "selection_ids_sha256"):
            if paired_metadata[0][key] != paired_metadata[1][key]:
                raise ValueError("Unpaired hard/oracle evaluation")
        metrics = [read_json(path / summary["evaluations"][f"IND-{depth}/{r}"]["path"] / "metrics.json")
                   for r in ("predicted", "oracle")]
        penalty = 100 * (metrics[1]["variant_macro_recall"] - metrics[0]["variant_macro_recall"])
        if declared["routing_penalty_macro_recall_pp"] != penalty:
            raise ValueError("Routing penalty disagrees with per-image evidence")
    if len(initializations) != 1:
        raise ValueError("ImageNet initialization mismatch")
    return summary


def draw_figures(output: Path, records: list[dict], nodes: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "figure.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(records))
    hard_bars = ax.bar(x - .18, [100 * r["hard"]["variant_macro_recall"] for r in records], .36, label="Predicted route", color="#126782")
    oracle_bars = ax.bar(x + .18, [100 * r["oracle"]["variant_macro_recall"] for r in records], .36, label="Oracle diagnostic", color="#db8b35")
    ax.bar_label(hard_bars, fmt="%.2f", padding=3)
    ax.bar_label(oracle_bars, fmt="%.2f", padding=3)
    ax.set_xticks(x, [f"{r['architecture']} / seed {r['seed']}" for r in records])
    ax.set(ylabel="Validation variant macro recall (%)", ylim=(0, 100), title="Full predicted pipeline vs oracle routing · all 3,333 validation images")
    ax.legend(); fig.tight_layout(); save_svg(fig, output / "routing-penalty.svg"); plt.close(fig)

    fig, axes = plt.subplots(len(records), 2, figsize=(12, 3 * len(records)), squeeze=False)
    for row, record in enumerate(records):
        router = next(n for n in nodes if n["seed"] == record["seed"] and n["key"] == "manufacturer")
        epochs = [int(h["epoch"]) for h in router["history"]]
        for split, color in (("train", "#126782"), ("val", "#c85c2c")):
            axes[row, 0].plot(epochs, [100 * float(h[f"{split}_macro_recall"]) for h in router["history"]], color=color, label=split)
            axes[row, 1].plot(epochs, [float(h[f"{split}_loss"]) for h in router["history"]], color=color, label=split)
        for col in (0, 1):
            axes[row, col].axvline(router["best_epoch"], linestyle="--", color="#555555", label="Selected")
            axes[row, col].set_xlabel("Epoch"); axes[row, col].legend(fontsize=8); axes[row, col].grid(alpha=.2)
        axes[row, 0].set(ylabel="Manufacturer macro recall (%)", title=f"Seed {record['seed']} · manufacturer router", ylim=(0, 100))
        axes[row, 1].set(ylabel="Cross-entropy", title="Augmented training vs deterministic validation")
    fig.tight_layout(); save_svg(fig, output / "router-learning.svg"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, max(5, .28 * len({n['key'] for n in nodes}))))
    keys = list(dict.fromkeys(n["key"] for n in nodes))
    for seed in sorted({n["seed"] for n in nodes}):
        subset = {n["key"]: n for n in nodes if n["seed"] == seed}
        axes[0].plot([100 * subset[k]["validation_macro_recall"] for k in keys], range(len(keys)), "o", label=f"Seed {seed}")
    axes[0].set_yticks(range(len(keys)), [next(n["label"] for n in nodes if n["key"] == k) for k in keys], fontsize=8)
    axes[0].invert_yaxis(); axes[0].set(xlabel="Local validation macro recall (%)", xlim=(0, 100), title="Selected nodes · true-parent diagnostic")
    axes[0].legend(fontsize=8)
    axes[1].barh(range(len(keys)), [statistics.mean(n["training_wall_seconds"] for n in nodes if n["key"] == k) / 60 for k in keys], color="#126782")
    axes[1].invert_yaxis(); axes[1].set_yticks(range(len(keys)), []); axes[1].set(xlabel="Mean training + selection minutes / seed", title="Measured node training cost")
    fig.tight_layout(); save_svg(fig, output / "specialists.svg"); plt.close(fig)

    # Every specialist's trajectory remains visible, including early plateaus.
    specialist_keys = [k for k in keys if k != "manufacturer"]
    if specialist_keys:
        rows = (len(specialist_keys) + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=(14, 2.7 * rows), squeeze=False)
        for ax, key in zip(axes.flat, specialist_keys):
            for node in (n for n in nodes if n["key"] == key):
                epochs = [int(h["epoch"]) for h in node["history"]]
                line, = ax.plot(epochs, [100 * float(h["val_macro_recall"]) for h in node["history"]],
                                label=f"Seed {node['seed']}")
                ax.plot(node["best_epoch"], 100 * node["validation_macro_recall"], "o", color=line.get_color(), markersize=4)
            ax.set(title=next(n["label"] for n in nodes if n["key"] == key), xlabel="Epoch", ylabel="Local val recall (%)", ylim=(0, 100))
            ax.grid(alpha=.15); ax.legend(fontsize=6)
        for ax in list(axes.flat)[len(specialist_keys):]:
            ax.axis("off")
        fig.suptitle("Independent specialists · complete local validation · dots mark selected epochs")
        fig.tight_layout(rect=(0, 0, 1, .98)); save_svg(fig, output / "specialist-learning.svg"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    stages = ("manufacturer", "family", "variant", "correct")
    bottom = np.zeros(len(records))
    for stage, color in zip(stages, ("#bd403b", "#e6a03b", "#6c71a1", "#377f6e")):
        values = [100 * r["hard"]["routing_diagnostics"]["first_wrong_stage"].get(stage, 0) / r["hard"]["images"] for r in records]
        axes[0].bar(x, values, bottom=bottom, label=stage, color=color); bottom += values
    axes[0].set_xticks(x, [f"{r['architecture']}\ns{r['seed']}" for r in records]); axes[0].set(ylabel="Validation images (%)", title="First wrong decision · hard routing", ylim=(0, 100)); axes[0].legend(fontsize=8)
    for calls in sorted({int(k) for r in records for k in r["hard"]["routing_diagnostics"]["executed_backbone_calls"]}):
        axes[1].plot(x, [100 * r["hard"]["routing_diagnostics"]["executed_backbone_calls"].get(str(calls), r["hard"]["routing_diagnostics"]["executed_backbone_calls"].get(calls, 0)) / r["hard"]["images"] for r in records], "o-", label=f"{calls} full CNN passes")
    axes[1].set_xticks(x, [f"{r['architecture']}\ns{r['seed']}" for r in records]); axes[1].set(ylabel="Validation images (%)", title="Executed paths · singleton classifiers bypassed", ylim=(0, 100)); axes[1].legend(fontsize=8)
    fig.tight_layout(); save_svg(fig, output / "routing-paths.svg"); plt.close(fig)

    for r in records:
        matrix = np.asarray(r["hard"]["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(matrix / matrix.sum(1, keepdims=True), vmin=0, vmax=1, cmap="Blues", interpolation="nearest")
        ax.set(xlabel="Predicted global variant ID", ylabel="True global variant ID", title=f"{r['architecture']} seed {r['seed']} · validation confusion")
        fig.colorbar(im, ax=ax, label="Per-class recall fraction"); fig.tight_layout()
        save_svg(fig, output / f"confusion-{r['architecture']}-s{r['seed']}.svg"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(15, 7))
    classwise = records[0]["hard"]["classwise"]
    order = sorted(range(len(classwise)), key=lambda i: statistics.mean(r["hard"]["classwise"][i]["recall"] for r in records))
    for r in records:
        ax.plot(range(len(order)), [100 * r["hard"]["classwise"][i]["recall"] for i in order], ".", label=f"{r['architecture']} s{r['seed']}")
    ax.set_xticks(range(len(order)), [classwise[i]["variant_name"] for i in order], rotation=90, fontsize=6)
    ax.set(ylabel="Validation recall (%)", ylim=(-2, 102), title="Full predicted route · per-variant recall sorted by mean")
    ax.legend(); fig.tight_layout(); save_svg(fig, output / "classwise-validation.svg"); plt.close(fig)

    # Code-native explanatory image: no generated or estimated experimental values.
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(12, 4)); ax.set(xlim=(0, 12), ylim=(0, 4)); ax.axis("off")
    boxes = [(0.1, 1.3, "224 × 224 image\nFixed preprocessing"), (3.1, 1.3, "Full EfficientNet\nManufacturer router"),
             (6.1, 2.1, "Selected full CNN\nVariant specialist"), (6.1, .3, "Single variant?\nDeterministic bypass"),
             (9.1, 1.3, "Global variant ID\nScore whole path")]
    for x0, y0, label in boxes:
        ax.add_patch(FancyBboxPatch((x0, y0), 2.6, 1.0, boxstyle="round,pad=0.06", facecolor="#e9f2f5", edgecolor="#126782"))
        ax.text(x0 + 1.3, y0 + .5, label, ha="center", va="center")
    for start, end in (((2.75, 1.8), (3.0, 1.8)), ((5.75, 1.8), (6.0, 2.55)), ((5.75, 1.8), (6.0, .8)),
                       ((8.75, 2.55), (9.0, 1.8)), ((8.75, .8), (9.0, 1.8))):
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": "#126782"})
    ax.text(6, 3.65, "IND-2: every trained node owns separate full CNN weights", ha="center", fontsize=14)
    ax.text(6, .03, "Hard routing uses predictions. Oracle routing is a separately labeled diagnostic. No test images used here.", ha="center", fontsize=9)
    fig.tight_layout(); save_svg(fig, output / "architecture.svg"); plt.close(fig)


def publish(runs: list[Path], output_root: Path) -> Path:
    summaries = [audit_training_run(p) for p in runs]
    if len({s["seed"] for s in summaries}) != len(summaries):
        raise ValueError("Duplicate seed")
    for key in ("taxonomy_sha256", "preprocessing_sha256", "dataset_manifest_sha256", "depths"):
        if len({object_hash(s[key]) for s in summaries}) != 1:
            raise ValueError(f"Runs disagree on {key}")
    configs = [read_json(p / "config.resolved.json") for p in runs]
    if len({object_hash({k: v for k, v in c.items() if k not in {"seed", "seeds", "output_root"}}) for c in configs}) != 1:
        raise ValueError("Training settings differ")
    # Reporting/residency tools may be finished while long training runs execute;
    # every module actually used for fitting and evaluation must stay identical.
    unused_during_training = {"src/cls_balance/report_independent.py", "src/cls_balance/residency.py"}
    sources = [{k: v for k, v in read_json(p / "source_hashes.json").items() if k not in unused_during_training}
               for p in runs]
    if len({object_hash(s) for s in sources}) != 1:
        raise ValueError("Training/evaluation source code differs across seeds")
    environments = [read_json(p / "environment.json") for p in runs]
    for key in ("torch", "cuda_runtime", "cudnn", "gpus", "nvidia_smi", "platform", "device", "runtime_policy", "git_sha", "packages"):
        if len({object_hash(e.get(key)) for e in environments}) != 1:
            raise ValueError(f"Environment differs: {key}")
    output = output_root / unique_id("independent")
    output.mkdir(parents=True, exist_ok=False)
    records, node_records, initializations = [], [], set()
    for run, summary in zip(runs, summaries, strict=True):
        target = output / f"seed-{summary['seed']}"; target.mkdir()
        for name in ("config.resolved.json", "environment.json", "source_hashes.json", "dataset_manifest_hashes.json"):
            shutil.copyfile(run / name, target / name)
        seen_nodes = set()
        for depth in summary["depths"]:
            architecture = f"IND-{depth}"
            evaluations = {}
            for routing in ("predicted", "oracle"):
                source = run / summary["evaluations"][f"{architecture}/{routing}"]["path"]
                shutil.copytree(source, target / f"{architecture}-{routing}")
                evaluations[routing] = read_json(source / "metrics.json")
            bundle = read_json(run / f"bundle-ind{depth}.json")
            references = {}
            taxonomy = Taxonomy.from_dict(read_json(run / "taxonomy.json"))
            for spec in node_plan(taxonomy, depth):
                entry = bundle["nodes"][spec.key]; source = run / entry["path"]
                references[spec.key] = {"checkpoint_sha256": entry["checkpoint_sha256"],
                    "local_global_ids": list(spec.global_ids), "local_run_path": source.relative_to(run).as_posix()}
                if spec.key in seen_nodes:
                    continue
                seen_nodes.add(spec.key)
                node_target = target / "nodes" / spec.key; node_target.mkdir(parents=True)
                shutil.copyfile(source / "history.csv", node_target / "history.csv")
                ns = read_json(source / "summary.json")
                initializations.add(ns["initialization_features_sha256"])
                # Summaries contain aggregate costs and hashes; contracts with training IDs stay local.
                shutil.copyfile(source / "summary.json", node_target / "summary.json")
                with (source / "history.csv").open(encoding="utf-8", newline="") as stream:
                    history = list(csv.DictReader(stream))
                names = taxonomy.manufacturers if spec.parent_stage == "manufacturer" else taxonomy.families
                label = "Manufacturer router" if spec.parent_id is None else f"{spec.stage}: {names[spec.parent_id]}"
                node_records.append({"seed": summary["seed"], "key": spec.key, "label": label,
                    "classes": len(spec.global_ids),
                    "best_epoch": ns["best_epoch"], "validation_macro_recall": ns["validation_macro_recall"],
                    "epochs_completed": ns["epochs_completed"], "history": history, **ns["resources"]})
            write_json(target / f"{architecture}-checkpoints.json", references)
            records.append({"seed": summary["seed"], "run_id": summary["run_id"], "architecture": architecture,
                "hard": evaluations["predicted"], "oracle": evaluations["oracle"], **summary["bundles"][architecture]})
    if len(initializations) != 1:
        raise ValueError("ImageNet feature initialization differs across seeds")
    draw_figures(output, records, node_records)
    from .prepare import write_csv
    write_csv(output / "nodes.csv", [{k: v for k, v in n.items() if k != "history"} for n in node_records])
    aggregate = {"schema_version": 1, "split": "val", "test_evaluated": False, "records": records,
                 "initialization_features_sha256": next(iter(initializations)),
                 "units": "accuracy fractions; penalty percentage points; storage bytes; time seconds"}
    write_json(output / "aggregate.json", aggregate)
    lines = ["# Phase 4 — Independent hierarchical validation", "", "## 1. Experimental context", "",
        "Official Torchvision edition: 100 variants / 70 families / 30 manufacturers; 3,334 train / 3,333 validation / 3,333 held-out test images.",
        "FP32 EfficientNet-B0, ImageNet V1 initialization, 224×224 banner removal + aspect-preserving resize/pad. Each node is trained independently on its true supervised training branch and selected by local validation macro recall.", "",
        f"Seeds: {', '.join(str(s['seed']) for s in summaries)}. Device: {environments[0]['device']}; {environments[0]['gpus']}. Torch {environments[0]['torch']}, CUDA {environments[0]['cuda_runtime']}. OS power defaults; clocks not locked. All classifiers resident during validation.", "",
        f"Publication ID: `{output.name}` (UTC timestamp). Training source base Git SHA: `{environments[0]['git_sha']}`; dirty working tree: `{environments[0]['git_dirty']}`. Exact training/evaluation source hashes are saved per seed; the final Git commit additionally includes reporting and documentation.", "",
        f"Taxonomy SHA256: `{summaries[0]['taxonomy_sha256']}`. Preprocessing: `{summaries[0]['preprocessing_sha256']}`. Manifest: `{summaries[0]['dataset_manifest_sha256']}`.", "",
        "## 2. Validation comparison (test endpoints deferred)", "",
        "These are complete validation predictions, not held-out test results. Oracle values are diagnostics. The shared-backbone comparison and synchronized full-pipeline p95/FPS benchmark remain Phases 5–6; no speed claim is made.", "",
        "| Architecture / seed | Variant macro recall % | Top-1 % | Macro-F1 % | Family from variant % | Manufacturer from variant % | Oracle macro recall % | Oracle − hard pp |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in records:
        h = r["hard"]
        lines.append(f"| {r['architecture']} / {r['seed']} | {100*h['variant_macro_recall']:.2f} | {100*h['variant_top1']:.2f} | {100*h['variant_macro_f1']:.2f} | {100*h['family_from_variant']['accuracy']:.2f} | {100*h['manufacturer_from_variant']['accuracy']:.2f} | {100*r['oracle']['variant_macro_recall']:.2f} | {r['routing_penalty_macro_recall_pp']:.2f} |")
    for arch in sorted({r["architecture"] for r in records}):
        values = [100*r["hard"]["variant_macro_recall"] for r in records if r["architecture"] == arch]
        deviation = f" ± {statistics.stdev(values):.2f} pp sample SD" if len(values) > 1 else "; single seed, no seed SD"
        lines += ["", f"{arch} validation macro recall: {statistics.mean(values):.2f}%{deviation}."]
    lines += ["", "## 3. Paired differences and uncertainty", "", "No shared/independent test comparison or bootstrap interval is available at this phase. Engineering thresholds are unassessed.", "",
        "## 4. Routing and singleton paths", "", "![Architecture](architecture.svg)", "", "![Routing penalty](routing-penalty.svg)", "", "![First errors and paths](routing-paths.svg)", "",
        "All normal routes use predicted ancestors. Oracle forces ancestors only and still classifies the final variant. Per-image decisions, bypasses and executed CNN counts are saved in each validation CSV. Conditional family diagnostics, when present, use the true manufacturer in a separate pass.", "",
        "## 5. Resources and residency", "", "| Architecture / seed | Full CNN nodes | Stored MB | Trainable parameters | Train + selection hours | Training image exposures | Optimizer steps |",
        "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in records:
        c = r["resources"]
        lines.append(f"| {r['architecture']} / {r['seed']} | {r['nodes']} | {c['deployment_bytes']/1e6:.2f} | {c['trainable_parameters']} | {c['training_wall_seconds']/3600:.3f} | {c['training_image_exposures']} | {c['optimizer_steps']} |")
    lines += ["", "Storage counts all classifier artifacts, including inactive branches; execution counts include only the selected path. Training hours include epoch validation and checkpoint selection, excluding initialization/export/final evaluation. Reused manufacturer training is counted in each architecture's standalone cost; do not sum architectures to infer unique experiment GPU hours.", "",
        "Cold-disk and preloaded-CPU are separately invocable residency policies. Disk mode retains OS filesystem caches and performs integrity-checked loading on every selected node call; CPU mode retains full CPU nodes and transfers/evicts selected weights inside prediction. The Phase 6 full-pipeline benchmark must include those operations when using either policy.", "",
        "## 6. Specialist training review", "", "Each row in [nodes.csv](nodes.csv) gives branch name, support, seed, selected epoch, local validation score, exposures, optimizer steps and measured cost. Per-node histories and checkpoint hashes are under each seed directory. Local true-parent scores are not whole-pipeline accuracy.", "",
        "![Router learning](router-learning.svg)", "", "![Specialist scores and costs](specialists.svg)", "", "![Specialist learning](specialist-learning.svg)", "", "![Classwise validation](classwise-validation.svg)", "",
        "## 7. Confusion matrices", ""]
    for r in records:
        lines += [f"![{r['architecture']} seed {r['seed']}](confusion-{r['architecture']}-s{r['seed']}.svg)", ""]
    lines += ["## 8. Limitations and phase gates", "", "Test inference, final uncertainty, latency/FPS/MACs, the shared architectures and incremental updates remain unmeasured. IND-3 is implemented and checked on generated fixtures; training it is opt-in after the primary two-stage comparison, following the repository execution order. No training dataset images, training-image ID lists, embeddings or weights are published.", "",
        "## 9. Reproduction and evidence", "", "```powershell", "& .\\.venv-cuda\\Scripts\\python.exe -m cls_balance.train_independent --config configs/independent.yaml",
        "& .\\.venv-cuda\\Scripts\\python.exe -m cls_balance.report_independent --runs " + " ".join("runs/independent/" + s["run_id"] for s in summaries) + " --output-root results/stage4", "```", "",
        "| Seed | Run ID | Config SHA256 | Source snapshot |", "| --- | --- | --- | --- |"]
    for s in summaries:
        lines.append(f"| {s['seed']} | `{s['run_id']}` | `{s['config_sha256']}` | [hashes](seed-{s['seed']}/source_hashes.json) |")
    with (output / "REPORT.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines) + "\n")
    write_json(output / "publication.json", {"status": "complete", "source_run_ids": [s["run_id"] for s in summaries],
        "contents": "validation predictions, aggregate node training evidence, checkpoint references and SVG plots; no training IDs or model files",
        "artifacts_sha256": {p.relative_to(output).as_posix(): file_hash(p) for p in sorted(output.rglob("*")) if p.is_file()}})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("results/stage4"))
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        for run in args.runs:
            print(audit_training_run(run)["run_id"])
    else:
        print(publish(args.runs, args.output_root))


if __name__ == "__main__":
    main()
