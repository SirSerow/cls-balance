"""Publish audited Phase 6 measurements, review plots and complete network SVGs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import shutil
import statistics

import numpy as np

from .artifacts import file_hash, object_hash, write_json
from .benchmark import audit_run
from .data import EXPECTED_CLASSES, EXPECTED_SPLITS
from .network_diagrams import draw_backbone, draw_network
from .predictions import read_predictions
from .prepare import write_csv
from .report_flat import read_json, save_svg
from .taxonomy import Taxonomy
from .train_independent import unique_id


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def paired_bootstrap(independent: list[dict], shared: list[dict], *, replicates: int = 2000) -> dict:
    """Image-paired, class-stratified percentile CI, averaging matched fixed seeds.

    The same within-class sampled image IDs are used for both architectures AND
    every seed. This captures test-image uncertainty conditional on these seeds;
    it is not a population-of-training-seeds interval or a noninferiority test.
    """
    independent = sorted(independent, key=lambda r: r["seed"])
    shared = sorted(shared, key=lambda r: r["seed"])
    if [r["seed"] for r in independent] != [r["seed"] for r in shared]:
        raise ValueError("Bootstrap requires paired seeds")
    delta, labels, ids = [], None, None
    for left, right in zip(independent, shared):
        l = sorted(left["predictions"], key=lambda r: r.image_id)
        r = sorted(right["predictions"], key=lambda r: r.image_id)
        current = [(x.image_id, x.true_variant_id) for x in l]
        if current != [(x.image_id, x.true_variant_id) for x in r] or (ids is not None and ids != current):
            raise ValueError("Bootstrap requires identical image IDs/truth across seeds and architectures")
        ids = current
        labels = np.array([x.true_variant_id for x in l])
        delta.append(np.array([int(y.prediction.variant_id == y.true_variant_id) - int(x.prediction.variant_id == x.true_variant_id)
                               for x, y in zip(l, r)]))
    mean_delta = np.mean(delta, axis=0)
    groups = [np.flatnonzero(labels == c) for c in sorted(set(labels))]
    rng = np.random.default_rng(6006)
    samples = np.zeros(replicates)
    for group in groups:
        samples += mean_delta[rng.choice(group, size=(replicates, len(group)), replace=True)].mean(1) / len(groups)
    return {"difference_pp": float(100 * np.mean([mean_delta[g].mean() for g in groups])),
            "ci95_pp": [float(v) for v in np.percentile(100 * samples, [2.5, 97.5])],
            "replicates": replicates, "rng_seed": 6006,
            "method": "paired image bootstrap within every variant; same resample across fixed seeds; percentile CI; not training-seed uncertainty"}


def load_records(runs: list[Path]) -> tuple[list[dict], Taxonomy]:
    records, common, keys = [], None, set()
    for run in runs:
        summary = audit_run(run)
        env, cfg = read_json(run / "environment.json"), read_json(run / "config.resolved.json")
        contract = read_json(run / "data_contract.json")
        comparable = {"data": contract, "device": env["device"], "gpus": env.get("gpus"),
                      "torch": env.get("torch"), "cuda": env.get("cuda_runtime"), "cudnn": env.get("cudnn"),
                      "platform": env.get("platform"), "packages": env.get("packages"),
                      "threads": cfg["cpu_threads"], "precision": cfg["precision"], "passes": cfg["passes"],
                      "residency": cfg["residency"], "warmup": cfg["warmup_iterations"],
                      "boundaries": [summary[k] for k in ("full_pipeline_boundary", "model_path_boundary", "fps_boundary")],
                      "inference_sources": {k: v for k, v in read_json(run / "source_hashes.json").items()
                          if "/report_" not in k and not k.endswith("network_diagrams.py")}}
        if common is not None and object_hash(comparable) != object_hash(common):
            raise ValueError("Benchmark data/hardware/environment/boundary mismatch")
        common = comparable
        key = (summary["architecture"], summary["seed"])
        if key in keys:
            raise ValueError("Duplicate architecture/seed")
        keys.add(key)
        taxonomy = Taxonomy.from_dict(read_json(run / "taxonomy.json"))
        records.append({"path": run, **summary, "metrics": read_json(run / "metrics.json"),
                        "timing": read_json(run / "timing_summary.json"), "environment": env,
                        "resources": read_json(run / "resources.json"),
                        "predictions": read_predictions(run / "predictions_test.csv", taxonomy),
                        "oracle": read_json(run / "oracle_metrics.json") if (run / "oracle_metrics.json").exists() else None,
                        "latencies": read_csv(run / "latency_test.csv")})
        record = records[-1]
        expected_classwise = [{k: str(v) for k, v in c.items()} for c in record["metrics"]["classwise"]]
        if read_csv(run / "classwise_metrics.csv") != expected_classwise:
            raise ValueError("Classwise CSV disagrees with prediction metrics")
        for measured_pass in record["timing"]["passes"]:
            elapsed = measured_pass["elapsed_seconds"]
            observed = [r for r in record["latencies"] if int(r["pass"]) == measured_pass["pass"]]
            if (not math.isfinite(elapsed) or elapsed <= 0 or measured_pass["images"] != len(observed)
                    or not math.isclose(measured_pass["sustained_fps"], len(observed) / elapsed, rel_tol=1e-12)
                    or elapsed + 1e-9 < sum(float(r["full_pipeline_ms"]) for r in observed) / 1000):
                raise ValueError("Per-pass elapsed time/FPS contract mismatch")
    if not records:
        raise ValueError("No completed benchmarks")
    seeds = {r["seed"] for r in records}
    if any({r["seed"] for r in records if r["architecture"] == arch} != seeds for arch in {r["architecture"] for r in records}):
        raise ValueError("Require identical seed sets")
    for arch in {r["architecture"] for r in records}:
        if len({r["resources"]["total_model_parameters"] for r in records if r["architecture"] == arch}) != 1:
            raise ValueError("Architecture parameter count changed across seeds")
    return records, taxonomy


def training_nodes(records: list[dict], roots: list[Path]) -> list[dict]:
    """Read only node summaries/histories, publish no training IDs or tensors."""
    nodes = []
    for record in records:
        resource = record["resources"]
        run_id = resource["training_run_id"]
        if record["architecture"] == "FLAT":
            continue
        candidates = [root / run_id for root in roots if (root / run_id / "summary.json").exists()]
        if len(candidates) != 1:
            raise ValueError(f"Need exactly one local training run for network annotation: {run_id}")
        run = candidates[0]
        if file_hash(run / "summary.json") != resource["training_summary_sha256"]:
            raise ValueError("Training annotation source changed")
        taxonomy = Taxonomy.from_dict(read_json(run / "taxonomy.json"))
        hidden = read_json(run / "config.resolved.json").get("hidden_dimension")
        from .models.cascade import node_plan
        shared = "FROZEN" in record["architecture"]
        bundle_path = run / f"bundle-{'shared' if shared else 'ind'}{record['architecture'][-1]}.json"
        source_summary = read_json(run / "summary.json")
        if file_hash(bundle_path) != source_summary["artifacts_sha256"][bundle_path.name]:
            raise ValueError("Selected training bundle changed")
        bundle = read_json(bundle_path)
        specs = node_plan(taxonomy, int(record["architecture"][-1]))
        if set(bundle["nodes"]) != {s.key for s in specs}:
            raise ValueError("Selected specialist bundle inventory mismatch")
        for planned in specs:
            entry = bundle["nodes"][planned.key]
            target = (run / entry["path"]).resolve()
            if not target.is_relative_to(run.resolve()):
                raise ValueError("Selected node path escapes training run")
            folder = target.parent if shared else target
            path = folder / "summary.json"
            summary = read_json(path)
            spec = summary["node"]
            manifest = read_json(folder / ("head.json" if shared else "node.json"))
            metadata = manifest["metadata"]
            if (spec != planned.to_dict() or manifest["file_sha256"] != entry["checkpoint_sha256"]
                    or metadata["node"] != spec or metadata["seed"] != record["seed"]
                    or metadata["best_epoch"] != summary["best_epoch"]
                    or metadata["validation_macro_recall"] != summary["validation_macro_recall"]):
                raise ValueError("Selected training annotation/checkpoint mismatch")
            for artifact in (path, folder / "history.csv", folder / ("head.json" if shared else "node.json")):
                relative = artifact.relative_to(run.resolve()).as_posix()
                expected_hash = source_summary["artifacts_sha256"].get(relative)
                if expected_hash is None:
                    expected_hash = summary.get("artifacts_sha256", {}).get(artifact.name)
                if expected_hash is not None and file_hash(artifact) != expected_hash:
                    raise ValueError("Selected training history/summary changed")
            names = taxonomy.manufacturers if spec["parent_stage"] == "manufacturer" else taxonomy.families
            label = ("Manufacturer router" if spec["parent_id"] is None else
                     f"{names[spec['parent_id']]} · {len(spec['global_ids'])} outputs")
            nodes.append({"architecture": record["architecture"], "seed": record["seed"], "training_run_id": run_id,
                          "key": spec["key"], "label": label, "hidden_dimension": hidden, "best_epoch": summary["best_epoch"],
                          "validation_macro_recall": summary["validation_macro_recall"],
                          "history": read_csv(path.parent / "history.csv"), "summary_sha256": file_hash(path)})
    return nodes


def branch_review(records: list[dict], taxonomy: Taxonomy) -> list[dict]:
    """True-branch variant recall includes upstream mistakes; no oracle filtering."""
    rows = []
    for record in records:
        for m, variants in enumerate(taxonomy.branches()["manufacturer_to_variant"]):
            selected = [p for p in record["predictions"] if p.true_manufacturer_id == m]
            correct = sum(p.prediction.variant_id == p.true_variant_id for p in selected)
            rows.append({"architecture": record["architecture"], "seed": record["seed"],
                "manufacturer_id": m, "manufacturer_name": taxonomy.manufacturers[m],
                "variants": len(variants), "test_support": len(selected), "variant_correct": correct,
                "variant_top1": correct / len(selected),
                "variant_macro_recall": statistics.mean(record["metrics"]["classwise"][v]["recall"] for v in variants),
                "correct_manufacturer": sum(p.prediction.manufacturer_id == m for p in selected),
                "predicted_manufacturer_count": sum(p.prediction.manufacturer_id == m for p in record["predictions"]),
                "singleton_taxonomy_branch": len(variants) == 1})
    return rows


def draw_figures(output: Path, records: list[dict], preview_root: Path | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.fonttype": "none", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"FLAT": "#5c6bc0", "IND-2": "#cf7040", "FT-FROZEN-2": "#168475"}
    architectures = sorted({r["architecture"] for r in records})
    def save(fig, name):
        fig.tight_layout()
        save_svg(fig, output / f"{name}.svg")
        if preview_root:
            preview_root.mkdir(parents=True, exist_ok=True)
            fig.savefig(preview_root / f"{name}.png", dpi=110)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 6))
    means = []
    for arch in architectures:
        group = [r for r in records if r["architecture"] == arch]
        x = [r["timing"]["full_pipeline_ms"]["p95"] for r in group]
        y = [100 * r["metrics"]["variant_macro_recall"] for r in group]
        color = colors.get(arch)
        ax.scatter(x, y, color=color, alpha=.65)
        ax.errorbar(statistics.mean(x), statistics.mean(y), xerr=statistics.stdev(x) if len(x) > 1 else 0,
                    yerr=statistics.stdev(y) if len(y) > 1 else 0, marker="D", capsize=5, color=color, label=arch)
        means.append((statistics.mean(x), statistics.mean(y)))
        ranks = {int(index): rank for rank, index in enumerate(np.argsort(y))}
        for index, (r, xx, yy) in enumerate(zip(group, x, y)):
            offset = ((7, -16), (-12, 2), (9, 12))[ranks[index] % 3]
            ax.annotate(f"s{r['seed']}", (xx, yy), xytext=offset, textcoords="offset points", fontsize=8,
                        ha="right" if offset[0] < 0 else "left", va="center",
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8, "pad": .6},
                        arrowprops={"arrowstyle": "-", "color": color, "lw": .6, "alpha": .6})
    frontier = sorted((x, y) for x, y in means if not any(xx <= x and yy >= y and (xx < x or yy > y) for xx, yy in means))
    ax.plot([p[0] for p in frontier], [p[1] for p in frontier], "--*", markersize=12, color="#445566", label="Observed mean Pareto frontier")
    ax.set(xlabel="Full pipeline p95 (ms) · lower is better", ylabel="Test variant macro recall (%)",
           title="Accuracy / latency · full official test split · diamonds = mean ± seed SD")
    ax.legend(); ax.grid(alpha=.2); save(fig, "accuracy-latency")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for arch in architectures:
        group = [r for r in records if r["architecture"] == arch]
        values = np.sort([float(row["full_pipeline_ms"]) for r in group for row in r["latencies"]])
        axes[0].plot(values, np.arange(1, len(values) + 1) / len(values), label=arch, color=colors.get(arch))
        for r in group:
            axes[1].plot([p["pass"] + 1 for p in r["timing"]["passes"]],
                         [p["full_pipeline_ms"]["p95"] for p in r["timing"]["passes"]],
                         "o-", color=colors.get(arch), alpha=.7, label=f"{arch} s{r['seed']}")
    axes[0].set(xlabel="Full pipeline time (ms; log scale)", ylabel="Cumulative fraction", title="All measured image/pass observations", xscale="log")
    axes[1].set(xlabel="Timed pass", ylabel="Full pipeline p95 (ms)", title="Repeatability and drift", xticks=range(1, records[0]["passes"] + 1))
    for ax in axes:
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    save(fig, "latency-distributions")
    labels = [f"{r['architecture']}\ns{r['seed']}" for r in records]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, key, divisor, title in ((axes[0], "stored_checkpoint_bytes", 1e6, "All deployed checkpoints (decimal MB)"),
                                  (axes[1], "total_model_parameters", 1e6, "All stored parameters (M)"),
                                  (axes[2], "peak_gpu_allocated_bytes", 1024**2, "Peak allocated GPU memory (MiB)")):
        values = [r["resources"][key] / divisor if r["resources"][key] is not None else 0 for r in records]
        ax.bar(range(len(records)), values, color=[colors.get(r["architecture"]) for r in records])
        ax.set_title(title); ax.set_xticks(range(len(records)), labels, rotation=55, ha="right", fontsize=8)
    save(fig, "storage-memory")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(records))
    path_mean = np.array([r["timing"]["model_path_ms"]["mean"] for r in records])
    overhead = np.array([r["timing"]["full_pipeline_ms"]["mean"] for r in records]) - path_mean
    axes[0].bar(x, path_mean, label="Complete model path", color="#168475")
    axes[0].bar(x, overhead, bottom=path_mean, label="Decode + preprocess + H2D + boundary sync", color="#dda848")
    axes[0].set(ylabel="Mean milliseconds", title="Measured nested boundaries (means are additive)"); axes[0].legend(fontsize=8)
    axes[1].bar(x, [r["timing"]["sustained_fps"] for r in records], color=[colors.get(r["architecture"]) for r in records])
    axes[1].set(ylabel="Images / elapsed processing second", title="Observed sustained batch-one throughput")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=50, ha="right", fontsize=8)
    save(fig, "timing-throughput")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bottom = np.zeros(len(records))
    for stage, color in (("manufacturer", "#bf5547"), ("family", "#dda848"), ("variant", "#7985bc"), ("correct", "#168475")):
        vals = [100 * r["metrics"]["routing_diagnostics"]["first_wrong_stage"].get(stage, 0) / r["metrics"]["images"] for r in records]
        axes[0].bar(x, vals, bottom=bottom, label=stage, color=color); bottom += vals
    axes[0].set(ylabel="Test images (%)", ylim=(0, 100), title="First wrong decision"); axes[0].legend(fontsize=8)
    axes[1].bar(x - .18, [100 * r["metrics"]["variant_macro_recall"] for r in records], .36, label="Predicted pipeline", color="#168475")
    axes[1].bar(x + .18, [100 * r["oracle"]["variant_macro_recall"] if r["oracle"] else 0 for r in records], .36, label="Oracle diagnostic (FLAT: N/A)", color="#dda848")
    axes[1].set(ylabel="Test variant macro recall (%)", ylim=(0, 100), title="Routing cost; oracle excluded from all timings"); axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(x, labels, rotation=50, ha="right", fontsize=8)
    save(fig, "routing-errors")
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    for ax, arch in zip(axes, architectures):
        groups = {}
        for r in records:
            if r["architecture"] != arch:
                continue
            for row in r["latencies"]:
                groups.setdefault(row["route"], []).append(float(row["full_pipeline_ms"]))
        keys = sorted(groups)
        ax.bar(range(len(keys)), [np.percentile(groups[k], 95) for k in keys], color=colors.get(arch))
        names = []
        taxonomy = Taxonomy.from_dict(read_json(records[0]["path"] / "taxonomy.json"))
        for key in keys:
            names.append("FLAT" if key == "flat" else taxonomy.manufacturers[int(key.split(":")[1])])
        ax.set_xticks(range(len(keys)), names, rotation=65, ha="right", fontsize=7)
        ax.set(ylabel="Full pipeline p95 (ms)", title=f"{arch} · grouped by ACTUAL predicted route; all seeds/passes")
    save(fig, "branch-latency")
    for r in records:
        matrix = np.asarray(r["metrics"]["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(matrix / matrix.sum(1, keepdims=True), vmin=0, vmax=1, cmap="Blues")
        ax.set(xlabel="Predicted global variant ID", ylabel="True global variant ID", title=f"{r['architecture']} seed {r['seed']} · official test")
        fig.colorbar(im, ax=ax, label="Row-normalized fraction")
        save(fig, f"confusion-{r['architecture']}-s{r['seed']}")
    fig, ax = plt.subplots(figsize=(17, 6))
    for arch in architectures:
        group = [r for r in records if r["architecture"] == arch]
        ax.plot(range(len(group[0]["metrics"]["classwise"])),
                np.mean([[100 * c["recall"] for c in r["metrics"]["classwise"]] for r in group], axis=0),
                ".-", lw=.8, color=colors.get(arch), label=arch)
    ax.set_xticks(range(len(records[0]["metrics"]["classwise"])), [c["variant_name"] for c in records[0]["metrics"]["classwise"]], rotation=90, fontsize=6)
    ax.set(ylabel="Test recall (%) · mean across seeds", ylim=(-2, 102), title="Every variant · explicit global ID order")
    ax.legend(); save(fig, "class-recall")
    taxonomy = Taxonomy.from_dict(read_json(records[0]["path"] / "taxonomy.json"))
    branches = branch_review(records, taxonomy)
    fig, axes = plt.subplots(2, 1, figsize=(18, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    x = np.arange(len(taxonomy.manufacturers))
    width = .8 / len(architectures)
    for i, arch in enumerate(architectures):
        groups = [[100 * b["variant_macro_recall"] for b in branches if b["architecture"] == arch and b["manufacturer_id"] == m] for m in x]
        axes[0].bar(x + (i - (len(architectures) - 1) / 2) * width, [statistics.mean(v) for v in groups], width,
                    yerr=[statistics.stdev(v) if len(v) > 1 else 0 for v in groups], capsize=2,
                    color=colors.get(arch), label=arch)
    support = [next(b["test_support"] for b in branches if b["manufacturer_id"] == m) for m in x]
    axes[1].bar(x, support, color="#8093a8")
    axes[0].set(ylabel="Variant macro recall (%)", ylim=(0, 110), title="Within TRUE manufacturer · complete predicted pipeline, including routing mistakes · mean ± seed SD")
    axes[0].legend(ncol=len(architectures)); axes[0].grid(axis="y", alpha=.15)
    axes[1].set(ylabel="Test images", xlabel="Manufacturer ground truth (support identical across architectures)")
    axes[1].set_xticks(x, taxonomy.manufacturers, rotation=70, ha="right", fontsize=8)
    save(fig, "branch-accuracy-support")


def draw_training(output: Path, nodes: list[dict], preview_root: Path | None) -> None:
    """Every trained router/specialist, with independently selected checkpoints."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = list(dict.fromkeys(n["key"] for n in nodes))
    if not keys:
        return
    rows = (len(keys) + 2) // 3
    seeds = sorted({n["seed"] for n in nodes})
    palettes = {"IND": ["#b34b24", "#d77d44", "#efa76e"], "shared": ["#116d62", "#2b9c8a", "#74bba9"]}
    for metric, ylabel, scale in (("macro_recall", "Local macro recall (%)", 100), ("loss", "Cross-entropy", 1)):
        fig, axes = plt.subplots(rows, 3, figsize=(17, rows * 3.2), squeeze=False)
        for ax, key in zip(axes.flat, keys):
            for node in (n for n in nodes if n["key"] == key):
                history = node["history"]
                epochs = [int(h["epoch"]) for h in history]
                palette = palettes["IND" if node["architecture"].startswith("IND") else "shared"]
                color = palette[seeds.index(node["seed"]) % len(palette)]
                ax.plot(epochs, [scale * float(h[f"train_{metric}"]) for h in history], "--", color=color, alpha=.25, lw=.8)
                ax.plot(epochs, [scale * float(h[f"val_{metric}"]) for h in history], color=color, alpha=.65, lw=1,
                        label=f"{node['architecture']} s{node['seed']}")
                selected = history[node["best_epoch"] - 1]
                ax.plot(node["best_epoch"], scale * float(selected[f"val_{metric}"]), "o", ms=3, color=color)
            label = next(n.get("label", key) for n in nodes if n["key"] == key)
            ax.set(title=f"{label}\n{key}", xlabel="Epoch", ylabel=ylabel)
            if metric == "macro_recall":
                ax.set_ylim(0, 102)
            ax.grid(alpha=.15)
        for ax in list(axes.flat)[len(keys):]:
            ax.axis("off")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=6, fontsize=8, bbox_to_anchor=(.5, .983))
        fig.suptitle("Every trained classifier · solid: validation; dashed: augmented training; dot: independently selected epoch", y=.998)
        fig.tight_layout(rect=(0, 0, 1, .97))
        save_svg(fig, output / f"all-specialist-{metric}.svg")
        if preview_root:
            fig.savefig(preview_root / f"all-specialist-{metric}.png", dpi=110)
        plt.close(fig)


def publish(runs: list[Path], output_root: Path, training_roots: list[Path], preview_root: Path | None = None,
            *, saved_nodes: list[dict] | None = None) -> Path:
    records, taxonomy = load_records(runs)
    if ((len(taxonomy.variants), len(taxonomy.families), len(taxonomy.manufacturers)) != EXPECTED_CLASSES
            or any(r["images_per_pass"] != EXPECTED_SPLITS["test"] for r in records)):
        raise ValueError("Publication requires the complete official benchmark edition")
    nodes = training_nodes(records, training_roots) if saved_nodes is None else saved_nodes
    output = output_root / unique_id("benchmark")
    output.mkdir(parents=True, exist_ok=False)
    # Strict named allowlist: never traverse training runs when copying results.
    allowed = {"summary.json", "config.resolved.json", "provenance.json", "source_hashes.json", "environment.json",
        "data_contract.json", "taxonomy.json", "preprocessing.json", "predictions_test.csv", "metrics.json",
        "classwise_metrics.csv", "latency_test.csv", "timing_summary.json", "predictions_oracle_test.csv",
        "oracle_metrics.json", "routing_comparison.json", "resources.json"}
    allowed |= {f"pass-{i:02d}.json" for i in range(records[0]["passes"])}
    for r in records:
        target = output / r["run_id"]
        target.mkdir()
        for source in r["path"].iterdir():
            if source.name in allowed:
                shutil.copyfile(source, target / source.name)
        # The final publication audit recomputes copied predictions/timings.
        # Bind copied bytes here without repeating that expensive rescore twice.
        for name, digest in r["artifacts_sha256"].items():
            if file_hash(target / name) != digest:
                raise ValueError("Copied benchmark bytes changed")
    write_json(output / "specialist_training.json", nodes)
    write_csv(output / "branch_accuracy.csv", branch_review(records, taxonomy))
    if nodes:
        columns = ("architecture", "seed", "training_run_id", "key", "label", "hidden_dimension",
                   "best_epoch", "validation_macro_recall", "summary_sha256")
        write_csv(output / "specialist_inventory.csv", [{k: node[k] for k in columns} for node in nodes])
    for arch in sorted({r["architecture"] for r in records}):
        draw_network(output, taxonomy, arch, nodes)
    hidden_dimensions = {n["hidden_dimension"] for n in nodes if n.get("hidden_dimension") is not None}
    if len(hidden_dimensions) > 1:
        raise ValueError("Shared comparison requires a fixed MLP width")
    draw_backbone(output, shared_hidden=next(iter(hidden_dimensions), 256))
    draw_figures(output, records, preview_root)
    draw_training(output, nodes, preview_root)
    paired = None
    if {"IND-2", "FT-FROZEN-2"} <= {r["architecture"] for r in records}:
        paired = paired_bootstrap([r for r in records if r["architecture"] == "IND-2"], [r for r in records if r["architecture"] == "FT-FROZEN-2"])
        write_json(output / "paired_comparison.json", paired)
    write_report(output, records, paired)
    hashes = {p.relative_to(output).as_posix(): file_hash(p) for p in sorted(output.rglob("*")) if p.is_file()}
    publication = {"status": "complete", "run_id": output.name,
        "benchmark_runs": [r["run_id"] for r in records], "artifacts_sha256": hashes,
        "scope": "official test predictions/timings, aggregate training histories and exact taxonomy schematics; no training IDs, pixels, embeddings or tensors"}
    audit_publication(output, _manifest=publication)
    write_json(output / "publication.json", publication)
    return output


def audit_publication(path: Path, *, _manifest: dict | None = None) -> dict:
    """Verify copied evidence and that graph/training inventories match deployments."""
    from .models.cascade import node_plan
    from .network_diagrams import network_inventory
    publication = read_json(path / "publication.json") if _manifest is None else _manifest
    inventory = {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file() and p != path / "publication.json"}
    if publication.get("status") != "complete" or set(publication["artifacts_sha256"]) != inventory:
        raise ValueError("Incomplete/extra publication artifacts")
    for relative, digest in publication["artifacts_sha256"].items():
        target = (path / relative).resolve()
        if not target.is_relative_to(path.resolve()) or file_hash(target) != digest:
            raise ValueError("Publication checksum mismatch")
    records, taxonomy = load_records([path / name for name in publication["benchmark_runs"]])
    if ((len(taxonomy.variants), len(taxonomy.families), len(taxonomy.manufacturers)) != EXPECTED_CLASSES
            or any(r["images_per_pass"] != EXPECTED_SPLITS["test"] for r in records)):
        raise ValueError("Publication requires the complete official benchmark edition")
    nodes = read_json(path / "specialist_training.json")
    expected = {(r["architecture"], r["seed"], spec.key) for r in records if r["architecture"] != "FLAT"
                for spec in node_plan(taxonomy, int(r["architecture"][-1]))}
    if len(nodes) != len(expected) or {(n["architecture"], n["seed"], n["key"]) for n in nodes} != expected:
        raise ValueError("Trained specialist inventory mismatch")
    for node in nodes:
        selected = max(node["history"], key=lambda h: float(h["val_macro_recall"]))
        if int(selected["epoch"]) != node["best_epoch"] or float(selected["val_macro_recall"]) != node["validation_macro_recall"]:
            raise ValueError("Training graph checkpoint selection mismatch")
    for arch in {r["architecture"] for r in records}:
        if read_json(path / f"network-{arch}.json") != network_inventory(taxonomy, arch):
            raise ValueError("Network graph inventory mismatch")
    if read_csv(path / "branch_accuracy.csv") != [{k: str(v) for k, v in r.items()} for r in branch_review(records, taxonomy)]:
        raise ValueError("Branch accuracy CSV disagrees with test predictions")
    return {"status": "verified", "benchmark_runs": len(records), "trained_classifiers": len(nodes),
            "artifacts": len(inventory), "publication_sha256": file_hash(path / "publication.json") if _manifest is None else object_hash(publication)}


def write_report(output: Path, records: list[dict], paired: dict | None) -> None:
    def mean_sd(values):
        if any(v is None for v in values):
            return "N/A"
        return f"{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}" if len(values) > 1 else f"{values[0]:.2f}"
    first = records[0]
    env = first["environment"]
    contract = read_json(first["path"] / "data_contract.json")
    architectures = sorted({r["architecture"] for r in records})
    lines = ["# Phase 6 — measured official test benchmark", "", "## 1. Experimental context", "",
        f"Immutable publication `{output.name}`. All {first['images_per_pass']:,} official test images, seeds {sorted({r['seed'] for r in records})}, batch 1, {first['precision'].upper()}, {first['passes']} timed passes per checkpoint. Fixed Phase 3–5 validation-selected checkpoints; no retraining or test-based selection.", "",
        "| Context | Value |", "| --- | --- |",
        f"| Dataset | Torchvision edition: 100 variants / 70 families / 30 manufacturers; 3334 / 3333 / 3333 official train / val / test |",
        f"| Taxonomy SHA256 | `{contract['taxonomy_sha256']}` |",
        f"| Test manifest SHA256 | `{contract['manifest_records_sha256']['test']}` |",
        f"| Preprocessing SHA256 | `{contract['preprocessing_sha256']}` |",
        f"| Runtime commit | `{env.get('git_sha')}`; dirty source tree recorded; exact per-run source hashes retained |",
        f"| Framework | PyTorch {env.get('torch')}; CUDA {env.get('cuda_runtime')}; cuDNN {env.get('cudnn')} |",
        f"| Device | {env.get('gpus')}; {env.get('nvidia_smi')} |",
        "| Residency | All deployment weights GPU-resident, including idle specialists; no embedding cache |",
        "| System | Interactive Windows desktop, unlocked driver/OS clocks; ambient temperature unmeasured. Per-pass clocks/power/load snapshots retained. |", "",
        "**Preprocessing:** remove bottom 20 px, preserve full remaining image with bicubic 224×224 letterbox and black padding, ImageNet V1 normalization. Worker count 0; six CPU threads; TF32 disabled; deterministic kernels; `eval()` and `torch.inference_mode()`.", "",
        f"**Full pipeline:** {first['full_pipeline_boundary']}.", "",
        f"**Model path:** {first['model_path_boundary']}.", "",
        f"**Sustained FPS:** {first['fps_boundary']}.", "",
        "Warmup exercises every trained node 32 times plus label-free predicted routes. Full/model timers share one inference with CUDA completion synchronization; the extra H2D synchronization is included in full time. Cold start and OS cold-file-cache latency are unmeasured. Memory is peak allocated/reserved during timed passes, including resident weights; no OOM occurred in these completed runs. CPU process RSS is unmeasured.", "",
        "## 2. Main comparison", "",
        "All table cells show mean ± sample SD across the fixed seeds. Within each seed, latency percentiles pool all image/pass observations; FPS uses all processed images divided by summed elapsed pass time. Per-pass distributions are retained. MACs are conditional Conv2d/Linear estimates (1 multiply-accumulate = 1 MAC), excluding normalization, activations, pooling, routing and transfer.", "",
        "| Metric | " + " | ".join(architectures) + " |", "| --- | " + " | ".join("---:" for _ in architectures) + " |"]
    fields = [
        ("Variant macro recall (%)", lambda r: 100 * r["metrics"]["variant_macro_recall"]),
        ("Variant top-1 (%)", lambda r: 100 * r["metrics"]["variant_top1"]),
        ("Variant macro-F1 (%)", lambda r: 100 * r["metrics"]["variant_macro_f1"]),
        ("Manufacturer accuracy from final variant (%)", lambda r: 100 * r["metrics"]["manufacturer_from_variant"]["accuracy"]),
        ("Family accuracy from final variant (%)", lambda r: 100 * r["metrics"]["family_from_variant"]["accuracy"]),
        ("Full pipeline p50 (ms)", lambda r: r["timing"]["full_pipeline_ms"]["p50"]),
        ("Full pipeline mean (ms)", lambda r: r["timing"]["full_pipeline_ms"]["mean"]),
        ("**Full pipeline p95 (ms)**", lambda r: r["timing"]["full_pipeline_ms"]["p95"]),
        ("Full pipeline p99 (ms)", lambda r: r["timing"]["full_pipeline_ms"]["p99"]),
        ("Model-path p95 (ms)", lambda r: r["timing"]["model_path_ms"]["p95"]),
        ("Observed sustained FPS", lambda r: r["timing"]["sustained_fps"]),
        ("Peak GPU allocated memory (MiB)", lambda r: r["resources"]["peak_gpu_allocated_bytes"] / 1024**2 if r["resources"]["peak_gpu_allocated_bytes"] is not None else None),
        ("Deployment checkpoints (decimal MB)", lambda r: r["resources"]["stored_checkpoint_bytes"] / 1e6),
        ("All model parameters (M)", lambda r: r["resources"]["total_model_parameters"] / 1e6),
        ("Executed MACs / image (G)", lambda r: r["timing"]["mean_executed_macs"] / 1e9),
        ("Executed backbones / image", lambda r: r["timing"]["mean_backbone_calls"]),
        ("Singleton bypass images (%)", lambda r: 100 * r["timing"]["singleton_bypass_rate"])]
    def training_total(r, key):
        resources = r["resources"]
        local = resources["training_resources"]
        if key == "training_wall_seconds":
            value = local.get(key, local.get("shared_training_wall_seconds", 0))
        else:
            value = local[key]
        if resources.get("backbone_source"):
            value += resources["backbone_source"].get("flat_resources", {}).get(key, 0)
        return value
    fields += [("Cumulative fitting wall time incl. source FLAT (h)", lambda r: training_total(r, "training_wall_seconds") / 3600),
               ("Optimizer steps incl. source FLAT", lambda r: training_total(r, "optimizer_steps")),
               ("Training-image exposures incl. source FLAT", lambda r: training_total(r, "training_image_exposures"))]
    for label, fn in fields:
        lines.append("| " + label + " | " + " | ".join(mean_sd([fn(r) for r in records if r["architecture"] == a]) for a in architectures) + " |")
    lines += ["", "![Accuracy and latency](accuracy-latency.svg)", "", "![Storage and memory](storage-memory.svg)", "",
        "Generic ImageNet-frozen, three-stage and FP16 results are unmeasured in this primary publication. FP16 is a separately invocable deployment ablation using the same checkpoints.", "",
        "## 3. Paired differences and engineering objective", ""]
    if paired:
        independent = statistics.mean(r["timing"]["full_pipeline_ms"]["p95"] for r in records if r["architecture"] == "IND-2")
        shared = statistics.mean(r["timing"]["full_pipeline_ms"]["p95"] for r in records if r["architecture"] == "FT-FROZEN-2")
        reduction = 100 * (1 - shared / independent)
        lines += [f"FT-FROZEN-2 − IND-2 macro recall: **{paired['difference_pp']:+.2f} pp**, paired 95% image-bootstrap CI **[{paired['ci95_pp'][0]:+.2f}, {paired['ci95_pp'][1]:+.2f}] pp**. Mean seed-p95 difference: **{shared-independent:+.2f} ms**; reduction: **{reduction:.2f}%**.", "",
            f"Exploratory engineering thresholds: observed accuracy within 2 pp degradation: **{'yes' if paired['difference_pp'] >= -2 else 'no'}**; observed p95 reduction ≥30%: **{'yes' if reduction >= 30 else 'no'}**. These observations are not a formal noninferiority claim.", "",
            "Bootstrap: 2,000 replicates, RNG seed 6006, paired image resampling within every variant; use identical resampled IDs for both architectures and all fixed seeds, then average per-class differences across seeds. Percentile CI captures test-image uncertainty conditional on these checkpoints; training-seed population uncertainty and latency confidence intervals are not estimated."]
    lines += ["", "## 4. Routing error attribution", "", "| Architecture / seed | Hard macro recall (%) | Oracle diagnostic (%) | Oracle − hard (pp) |", "| --- | ---: | ---: | ---: |"]
    for r in records:
        hard = 100 * r["metrics"]["variant_macro_recall"]
        oracle = 100 * r["oracle"]["variant_macro_recall"] if r["oracle"] else None
        lines.append(f"| {r['architecture']} / {r['seed']} | {hard:.2f} | {oracle:.2f} | {oracle-hard:.2f} |" if oracle is not None else f"| {r['architecture']} / {r['seed']} | {hard:.2f} | N/A | N/A |")
    lines += ["", "![Routing errors](routing-errors.svg)", "", "Oracle inference is a separate untimed diagnostic. Actual inference receives no labels. Two-stage routing has 17 trained variant specialists and 13 deterministic single-variant manufacturer branches; bypass rates are predicted-route frequencies. All manufacturer router and first-wrong-stage counts are retained in each metrics JSON.", "",
        "## 5. Latency by actual executed path", "", "![Latency distributions](latency-distributions.svg)", "", "![Mean boundaries and observed throughput](timing-throughput.svg)", "", "![Every executed manufacturer route](branch-latency.svg)", "",
        "Per-image CSVs retain every pass, selected manufacturer route, classifier/backbone counts, singleton bypasses, conditional MACs and both latency boundaries. `timing_summary.json` contains p50/mean/p95/p99 for every observed route and pass. Stored inactive models count toward storage and residency, but not executed MACs.", "",
        "## 6. Specialist update / introduction", "", "Unmeasured; Phase 7 remains pending. This benchmark makes no introduced-class or update-cost claim.", "",
        "## 7. Full networks and training-result review", "",
        "Each schematic includes every trained specialist, all singleton branches, all 100 global variant labels, EfficientNet stage dimensions, and each specialist's selected epoch for all seeds. JSON inventories permit checking that the graph contains exactly the deployed nodes. Schematics show stored alternatives; only one branch executes.", "",
        "![EfficientNet internals and classifier export points](efficientnet-b0-detail.svg)", "",
        "[Every actual Torchvision module, kernel, stride, convolution group and parameter count](efficientnet-b0-modules.json).", ""]
    for arch in architectures:
        lines += [f"### {arch}", "", f"![Complete {arch} network](network-{arch}.svg)", ""]
    lines += ["[All specialist selections and validation scores](specialist_inventory.csv); [full per-head training histories](specialist_training.json). FLAT learning curves and source-encoder checks remain in the [Phase 3 review](../../stage3/flat-20261025T045223Z-788dc5f4/REPORT.md).", "", "![Every trained classifier learning curve](all-specialist-macro_recall.svg)", "", "![Every trained classifier loss curve](all-specialist-loss.svg)", "", "![Every variant recall](class-recall.svg)", "", "![Manufacturer-branch accuracy and test support](branch-accuracy-support.svg)", "", "[Branch accuracy/support CSV](branch_accuracy.csv). Accuracy groups by true manufacturer and includes routing mistakes; latency groups by the actually predicted route. These have different denominators.", "",
        "Confusion matrices are available per architecture and seed:", ""]
    for r in records:
        lines.append(f"- [{r['architecture']} seed {r['seed']}](confusion-{r['architecture']}-s{r['seed']}.svg)")
    lines += ["", "## 8. Discussion and limits", ""]
    if paired:
        lines += [f"**RQ1 (accuracy):** task-adapted shared heads change variant macro recall by {paired['difference_pp']:+.2f} pp relative to independent cascades on these fixed checkpoints; the paired image interval is reported above.", "",
                  f"**RQ2 (latency):** the shared pipeline changes mean seed-p95 latency by {shared-independent:+.2f} ms ({reduction:.2f}% reduction) under the same full boundary and all-resident policy.", ""]
        if "FLAT" in architectures:
            flat_recall = 100 * statistics.mean(r["metrics"]["variant_macro_recall"] for r in records if r["architecture"] == "FLAT")
            shared_recall = 100 * statistics.mean(r["metrics"]["variant_macro_recall"] for r in records if r["architecture"] == "FT-FROZEN-2")
            flat_p95 = statistics.mean(r["timing"]["full_pipeline_ms"]["p95"] for r in records if r["architecture"] == "FLAT")
            lines += [f"The FLAT sanity baseline reaches {flat_recall:.2f}% macro recall at {flat_p95:.2f} ms mean seed-p95; the shared cascade reaches {shared_recall:.2f}% at {shared:.2f} ms. These baseline results are part of the accuracy/latency comparison. Specialist-update outcomes remain pending Phase 7.", ""]
    lines += ["**RQ3 (routing):** compare hard versus oracle rows and first-wrong-stage counts to isolate lost reachability from local discrimination. Oracle results are diagnostic and cannot be deployed. **RQ4 (updates)** and **RQ5 (generic versus adapted frozen features)** remain unmeasured.", "",
        "The primary observations apply to the complete official FGVC test split on this GPU, preprocessing and residency policy. Routing errors, bypass frequency, all-model storage and conditional compute are exposed separately. Interactive desktop load, unlocked laptop clocks, normal filesystem caching, and nested synchronization limit portability of latency values. Adapter finite-value checks and per-call eval-mode maintenance are included, so latency reflects this deployed implementation as well as backbone arithmetic. Training costs are copied from audited training resources; task-adapted shared models additionally require their source FLAT training, recorded in `backbone_source.flat_resources`. Cumulative fitting time includes training and validation/selection; parallel work and total GPU activity are not inferred from wall time. These are completed-fit costs; discarded/interrupted training attempts remain documented in the [Phase 4 evidence](../../stage4/validation-20261027T071705Z-2b712381/REPORT.md) and are not added to this table. No equal-training-compute claim is made.", "",
        "## 9. Reproduction and audit links", "", "```powershell", ".venv-cuda/Scripts/python.exe -m cls_balance.benchmark --config configs/benchmark.yaml", ".venv-cuda/Scripts/python.exe -m cls_balance.report_benchmark --suite <completed-benchmark-suite> --output-root results/stage6", ".venv/Scripts/python.exe scripts/freeze_tests.py --check", ".venv/Scripts/python.exe -m pytest -q tests checks", "```", "",
        "The exact local checkpoint paths are in `configs/benchmark.yaml`; substitute your reproduced training run IDs. Set `precision: fp16` in a separate YAML for the optional separate precision pass. Models/data remain ignored. Rebuild training runs using the README sequence before benchmarking on another machine.", "",
        "| Architecture / seed | Training run | Predictions | Timings | Metrics / resources |", "| --- | --- | --- | --- | --- |"]
    for r in records:
        base = r["run_id"]
        lines.append(f"| {r['architecture']} / {r['seed']} | `{r['resources']['training_run_id']}` | [CSV]({base}/predictions_test.csv) | [CSV]({base}/latency_test.csv) | [metrics]({base}/metrics.json), [resources]({base}/resources.json), [environment]({base}/environment.json), [provenance]({base}/provenance.json) |")
    lines += ["", "Audit any copied run independently with `python -m cls_balance.benchmark --audit <run-directory>`. `publication.json` binds every published artifact by SHA256. Training and inference sources are versioned by per-run hashes. Published CSVs contain test labels/predictions and aggregate training histories; no training image IDs, aircraft images, embedding caches or checkpoint tensors are included.", ""]
    with (output / "REPORT.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--suite", type=Path)
    group.add_argument("--rebuild", type=Path, help="Regenerate from published evidence; no dataset/weights required")
    group.add_argument("--audit", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("results/stage6"))
    parser.add_argument("--training-roots", nargs="+", type=Path, default=[Path("runs/flat"), Path("runs/independent"), Path("runs/shared")])
    parser.add_argument("--preview-root", type=Path)
    args = parser.parse_args()
    if args.audit:
        print(audit_publication(args.audit))
        return
    if args.rebuild:
        audit_publication(args.rebuild)
        publication = read_json(args.rebuild / "publication.json")
        print(publish([args.rebuild / name for name in publication["benchmark_runs"]], args.output_root, [],
                      args.preview_root, saved_nodes=read_json(args.rebuild / "specialist_training.json")))
        return
    summary = read_json(args.suite / "summary.json")
    if (summary["status"] != "complete" or file_hash(args.suite / "frozen_matrix.json") != summary["frozen_matrix_sha256"]
            or set(summary["runs"]) != set(summary["run_summaries_sha256"])):
        raise ValueError("Incomplete or changed benchmark suite")
    for name, digest in summary["run_summaries_sha256"].items():
        if Path(name).name != name or file_hash(args.suite / name / "summary.json") != digest:
            raise ValueError("Suite child checksum mismatch")
    print(publish([args.suite / name for name in summary["runs"]], args.output_root, args.training_roots, args.preview_root))


if __name__ == "__main__":
    main()
