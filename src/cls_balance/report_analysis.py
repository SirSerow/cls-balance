"""Phase 8: paired statistics and reproducible publication from measured evidence.

No training, model selection, image access or inference occurs in this module.
Sources remain in their immutable Phase 6/7 publications; hashes bind every link.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
from pathlib import Path
import re
import shutil
import statistics

import numpy as np
import yaml

from .artifacts import environment, file_hash, object_hash, write_json
from .incremental import inventory, local_path
from .report_benchmark import audit_publication as audit_benchmark
from .report_flat import read_json
from .report_incremental import audit_publication as audit_incremental
from .taxonomy import Taxonomy
from .train_independent import unique_id

ARCHITECTURES = ("FLAT", "IND-2", "FT-FROZEN-2")
SEEDS = (42, 43, 44)


def bootstrap_difference(left: list[dict], right: list[dict], *, class_ids: tuple[int, ...],
                         replicates: int, rng_seed: int) -> dict:
    """Percentile intervals for fixed model pairs, stratified by true class.

    Resample image IDs with replacement within each class, preserving its support.
    Use each draw for both models and all fixed seeds; average classes equally.
    Never resample seeds or treat seed/image combinations as independent images.
    """
    if type(replicates) is not int or replicates < 100 or type(rng_seed) is not int or rng_seed < 0:
        raise ValueError("Require at least 100 bootstrap replicates and a nonnegative integer RNG seed")
    if not class_ids or len(set(class_ids)) != len(class_ids):
        raise ValueError("Require a unique, explicit class list")
    left, right = sorted(left, key=lambda r: r["seed"]), sorted(right, key=lambda r: r["seed"])
    seeds = [r["seed"] for r in left]
    if not seeds or len(set(seeds)) != len(seeds) or seeds != [r["seed"] for r in right]:
        raise ValueError("Require unique paired seeds")
    deltas, identities = [], None
    for a, b in zip(left, right, strict=True):
        arows, brows = (sorted(r["predictions"], key=lambda p: p.image_id) for r in (a, b))
        ids = [(p.image_id, p.true_variant_id) for p in arows]
        if (not ids or len({i for i, _ in ids}) != len(ids)
                or ids != [(p.image_id, p.true_variant_id) for p in brows]
                or (identities is not None and ids != identities)):
            raise ValueError("Require identical unique image IDs/truth across model pairs and seeds")
        if any(p.split != "test" or p.routing != "predicted" for p in (*arows, *brows)):
            raise ValueError("Primary bootstrap requires predicted-route test predictions")
        identities = ids
        deltas.append([int(b.prediction.variant_id == b.true_variant_id)
                       - int(a.prediction.variant_id == a.true_variant_id) for a, b in zip(arows, brows, strict=True)])
    labels = np.array([v for _, v in identities])
    if set(labels) != set(class_ids):
        raise ValueError("Missing or unexpected class support")
    delta = np.asarray(deltas, dtype=float)
    groups = [np.flatnonzero(labels == c) for c in sorted(class_ids)]
    rng = np.random.default_rng(rng_seed)
    samples = np.zeros((len(seeds), replicates))
    points = np.zeros(len(seeds))
    for group in groups:
        indices = rng.choice(group, size=(replicates, len(group)), replace=True)
        samples += delta[:, indices].mean(axis=2) / len(groups)
        points += delta[:, group].mean(axis=1) / len(groups)
    def interval(point: float, values: np.ndarray) -> dict:
        return {"difference_pp": float(100 * point),
                "ci95_pp": [float(x) for x in np.percentile(100 * values, [2.5, 97.5])]}
    return {"per_seed": [{"seed": seed, **interval(points[i], samples[i])} for i, seed in enumerate(seeds)],
            "fixed_seed_mean": interval(points.mean(), samples.mean(axis=0)),
            "training_seed_difference_sd_pp": float(np.std(100 * points, ddof=1)) if len(seeds) > 1 else None,
            "replicates": replicates, "rng_seed": rng_seed, "images": len(labels), "classes": len(groups),
            "method": "paired image IDs within true variant; fixed support; equal class weights; percentile 95% CI",
            "scope": "test-image uncertainty conditional on each fixed model pair; shared draws across seeds for the descriptive fixed-seed mean; no training-seed population CI"}


def pareto_frontier(points: list[dict]) -> list[str]:
    """Return all nondominated architectures: minimize latency, maximize recall."""
    if not points or any(not np.isfinite(p[k]) for p in points for k in ("p95_ms", "recall_percent")):
        raise ValueError("Pareto frontier requires finite measurements")
    return [p["architecture"] for p in sorted(points, key=lambda p: (p["p95_ms"], -p["recall_percent"]))
            if not any(q["p95_ms"] <= p["p95_ms"] and q["recall_percent"] >= p["recall_percent"]
                       and (q["p95_ms"] < p["p95_ms"] or q["recall_percent"] > p["recall_percent"])
                       for q in points)]


def load_sources(repo: Path, config: dict) -> tuple[list[dict], Taxonomy, list[dict]]:
    """Audit frozen public inputs, then enforce the complete primary matrix."""
    benchmark, incremental = (local_path(repo, config[k]) for k in ("benchmark", "incremental"))
    for root, name, key in ((benchmark, "publication.json", "benchmark_manifest_sha256"),
                            (incremental, "summary.json", "incremental_manifest_sha256")):
        if file_hash(root / name) != config[key]:
            raise ValueError("Frozen source publication manifest changed")
    _, records, taxonomy = audit_benchmark(benchmark, _return_records=True)
    _, updates = audit_incremental(incremental, _return_records=True)
    if config["seeds"] != list(SEEDS) or {(r["architecture"], r["seed"]) for r in records} != {
            (a, s) for a in ARCHITECTURES for s in SEEDS}:
        raise ValueError("Phase 8 requires the frozen three-seed primary architecture matrix")
    signatures = [[(p.image_id, p.true_variant_id) for p in sorted(r["predictions"], key=lambda p: p.image_id)] for r in records]
    if any(s != signatures[0] for s in signatures[1:]):
        raise ValueError("Primary test image IDs/truth differ")
    if {(r["architecture"], r["seed"]) for r in updates} != {(a, s) for a in ARCHITECTURES[1:] for s in SEEDS}:
        raise ValueError("Incomplete incremental seed matrix")
    return sorted(records, key=lambda r: (ARCHITECTURES.index(r["architecture"]), r["seed"])), taxonomy, updates


def training_cost(record: dict, key: str) -> float:
    """Count the source FLAT fit for each task-adapted shared deployment."""
    resources = record["resources"]
    local = resources["training_resources"]
    value = local.get(key, local.get("shared_training_wall_seconds", 0) if key == "training_wall_seconds" else 0)
    return value + (resources.get("backbone_source") or {}).get("flat_resources", {}).get(key, 0)


def calculate(records: list[dict], taxonomy: Taxonomy, updates: list[dict], config: dict) -> dict:
    """Derive tables and statistics only from audited per-image measurements."""
    paired = bootstrap_difference([r for r in records if r["architecture"] == "IND-2"],
        [r for r in records if r["architecture"] == "FT-FROZEN-2"],
        class_ids=tuple(range(len(taxonomy.variants))), replicates=config["bootstrap_replicates"], rng_seed=config["bootstrap_seed"])
    rows, branches, depths, passes, error_types = [], [], [], [], []
    for r in records:
        m, t, resource = r["metrics"], r["timing"], r["resources"]
        row = {"architecture": r["architecture"], "seed": r["seed"], "run_id": r["run_id"],
               "training_run_id": resource["training_run_id"], "config_sha256": r["config_sha256"],
               "model_sha256": r["model_sha256"],
               "macro_recall_percent": 100 * m["variant_macro_recall"], "top1_percent": 100 * m["variant_top1"],
               "macro_f1_percent": 100 * m["variant_macro_f1"],
               "manufacturer_percent": 100 * m["manufacturer_from_variant"]["accuracy"],
               "family_percent": 100 * m["family_from_variant"]["accuracy"],
               "oracle_macro_percent": 100 * r["oracle"]["variant_macro_recall"] if r["oracle"] else None,
               "router_percent": 100 * m["routing_diagnostics"]["manufacturer_router"]["accuracy"] if r["oracle"] else None,
               **{f"full_{k}_ms": v for k, v in t["full_pipeline_ms"].items()},
               "model_p95_ms": t["model_path_ms"]["p95"], "sustained_fps": t["sustained_fps"],
               "peak_gpu_mib": resource["peak_gpu_allocated_bytes"] / 1024**2,
               "stored_mb": resource["stored_checkpoint_bytes"] / 1e6,
               "parameters_m": resource["total_model_parameters"] / 1e6,
               "macs_g": t["mean_executed_macs"] / 1e9, "backbone_calls": t["mean_backbone_calls"],
               "bypass_percent": 100 * t["singleton_bypass_rate"],
               "training_hours": training_cost(r, "training_wall_seconds") / 3600,
               "optimizer_steps": training_cost(r, "optimizer_steps"),
               "training_exposures": training_cost(r, "training_image_exposures")}
        row["routing_penalty_pp"] = row["oracle_macro_percent"] - row["macro_recall_percent"] if r["oracle"] else None
        pass_p95 = [p["full_pipeline_ms"]["p95"] for p in t["passes"]]
        row.update({"pass_p95_min_ms": min(pass_p95), "pass_p95_max_ms": max(pass_p95),
                    "pass_p95_sd_ms": statistics.stdev(pass_p95)})
        rows.append(row)
        for p in t["passes"]:
            passes.append({"architecture": r["architecture"], "seed": r["seed"], "pass": p["pass"],
                           "p50_ms": p["full_pipeline_ms"]["p50"], "p95_ms": p["full_pipeline_ms"]["p95"],
                           "elapsed_seconds": p["elapsed_seconds"], "sustained_fps": p["sustained_fps"]})
        for route, values in sorted(t["paths"].items()):
            branches.append({"architecture": r["architecture"], "seed": r["seed"], "predicted_route": route,
                             "observations": values["observations"], **values["full_pipeline_ms"]})
        for calls in sorted({int(p["backbone_calls"]) for p in r["latencies"]}):
            selected = [p for p in r["latencies"] if int(p["backbone_calls"]) == calls]
            values = [float(p["full_pipeline_ms"]) for p in selected]
            depths.append({"architecture": r["architecture"], "seed": r["seed"], "backbone_calls": calls,
                           "images": len({p["image_id"] for p in selected}), "observations": len(values),
                           "p50_ms": float(np.percentile(values, 50)), "p95_ms": float(np.percentile(values, 95))})
        counts = {"correct": 0, "within_family_error": 0, "cross_family_same_manufacturer": 0, "cross_manufacturer_error": 0}
        for p in r["predictions"]:
            kind = ("correct" if p.prediction.variant_id == p.true_variant_id else
                    "within_family_error" if p.prediction.family_id == p.true_family_id else
                    "cross_family_same_manufacturer" if p.prediction.manufacturer_id == p.true_manufacturer_id else
                    "cross_manufacturer_error")
            counts[kind] += 1
        error_types.append({"architecture": r["architecture"], "seed": r["seed"], **counts})
    points = [{"architecture": a, "p95_ms": statistics.mean(r["full_p95_ms"] for r in rows if r["architecture"] == a),
               "recall_percent": statistics.mean(r["macro_recall_percent"] for r in rows if r["architecture"] == a)} for a in ARCHITECTURES]
    independent, shared = points[1:]
    return {"per_seed": rows, "bootstrap": paired, "pareto_points": points, "pareto_frontier": pareto_frontier(points),
            "p95_difference_ms": shared["p95_ms"] - independent["p95_ms"],
            "p95_reduction_percent": 100 * (1 - shared["p95_ms"] / independent["p95_ms"]),
            "branch_latency": branches, "depth_latency": depths, "pass_latency": passes,
            "family_error_breakdown": error_types, "incremental": updates}


def csv_text(rows: list[dict]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def draw_figures(output: Path, analysis: dict, records: list[dict], preview: Path | None) -> None:
    """Render deterministic, data-derived review graphics; no aircraft pixels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "cls-balance-phase8", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    colors = dict(zip(ARCHITECTURES, ("#5c6bc0", "#cf7040", "#168475")))
    def save(fig, name):
        fig.tight_layout()
        # Suppress timestamps so artifact-only rebuilds are byte stable.
        with (output / f"{name}.svg").open("x", encoding="utf-8", newline="\n") as stream:
            fig.savefig(stream, format="svg", metadata={"Date": None})
        if preview:
            preview.mkdir(parents=True, exist_ok=True)
            fig.savefig(preview / f"{name}.png", dpi=130)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for a in ARCHITECTURES:
        group = [r for r in analysis["per_seed"] if r["architecture"] == a]
        x, y = [r["full_p95_ms"] for r in group], [r["macro_recall_percent"] for r in group]
        axes[0].scatter(x, y, color=colors[a], alpha=.5, s=24)
        axes[0].errorbar(np.mean(x), np.mean(y), xerr=np.std(x, ddof=1), yerr=np.std(y, ddof=1),
                         color=colors[a], marker="D", capsize=4, label=a)
    frontier = [p for p in analysis["pareto_points"] if p["architecture"] in analysis["pareto_frontier"]]
    axes[0].plot([p["p95_ms"] for p in frontier], [p["recall_percent"] for p in frontier],
                 "o--", color="black", ms=12, markerfacecolor="none", label="Mean Pareto frontier")
    axes[0].set(xlabel="Full-pipeline p95 (ms; lower is better)", ylabel="Variant macro recall (%)",
                title="Accuracy and latency · bars = sample SD across seeds")
    axes[0].legend(fontsize=8)
    intervals = analysis["bootstrap"]["per_seed"]
    for i, row in enumerate(intervals):
        axes[1].plot(row["ci95_pp"], [i, i], color=colors["FT-FROZEN-2"], lw=3)
        axes[1].plot(row["difference_pp"], i, "o", color=colors["FT-FROZEN-2"])
    axes[1].axvline(0, color="grey", ls=":")
    axes[1].axvline(-2, color=colors["IND-2"], ls="--", label="Exploratory −2 pp limit")
    axes[1].set(yticks=range(len(intervals)), yticklabels=[f"Seed {r['seed']}" for r in intervals],
                xlabel="FT-FROZEN-2 − IND-2 macro recall (pp)", title="Paired stratified 95% image-bootstrap intervals")
    axes[1].legend(fontsize=8)
    save(fig, "accuracy-latency-bootstrap")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    names = ("correct", "manufacturer", "family", "variant")
    for i, a in enumerate(ARCHITECTURES):
        group = [r for r in records if r["architecture"] == a]
        bottom = 0
        for stage, color in zip(names, ("#168475", "#bf4b45", "#cba748", "#7685b3")):
            value = np.mean([100 * r["metrics"]["routing_diagnostics"]["first_wrong_stage"][stage] / len(r["predictions"]) for r in group])
            axes[0].bar(i, value, bottom=bottom, color=color, label=stage if i == 0 else None)
            bottom += value
        if a != "FLAT":
            axes[1].bar(a, np.mean([r["routing_penalty_pp"] for r in analysis["per_seed"] if r["architecture"] == a]), color=colors[a])
    axes[0].set(xticks=range(3), xticklabels=ARCHITECTURES, ylabel="All test images (%)", title="First wrong decision · fixed-seed mean")
    axes[0].legend(fontsize=8)
    axes[1].set(ylabel="Oracle − predicted macro recall (pp)", title="Routing penalty · untimed oracle diagnostic")
    for i, a in enumerate(ARCHITECTURES):
        rows = [r for r in analysis["pass_latency"] if r["architecture"] == a]
        for seed in SEEDS:
            values = [r["p95_ms"] for r in rows if r["seed"] == seed]
            axes[2].plot(range(1, 6), values, "o-", color=colors[a], alpha=.5, label=a if seed == 42 else None)
    axes[2].set(xlabel="Timed pass", ylabel="Full-pipeline p95 (ms)", title="Five measured passes per fixed checkpoint")
    axes[2].legend(fontsize=8)
    save(fig, "routing-and-repeatability")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bottom = np.zeros(len(ARCHITECTURES))
    for key, label, color in (("within_family_error", "Wrong variant, correct family", "#7685b3"),
                              ("cross_family_same_manufacturer", "Wrong family, correct manufacturer", "#cba748"),
                              ("cross_manufacturer_error", "Wrong manufacturer", "#bf4b45")):
        values = [np.mean([100 * r[key] / sum(r[k] for k in ("correct", "within_family_error",
                    "cross_family_same_manufacturer", "cross_manufacturer_error"))
                    for r in analysis["family_error_breakdown"] if r["architecture"] == a]) for a in ARCHITECTURES]
        ax.bar(ARCHITECTURES, values, bottom=bottom, color=color, label=label)
        bottom += values
    ax.set(ylabel="All test images (%)", title="Where the final prediction differs · mean over fixed seeds")
    ax.legend(fontsize=9)
    save(fig, "family-errors")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    labels, values = [], []
    for a in ARCHITECTURES:
        for calls in sorted({r["backbone_calls"] for r in analysis["depth_latency"] if r["architecture"] == a}):
            selected = [float(p["full_pipeline_ms"]) for r in records if r["architecture"] == a
                        for p in r["latencies"] if int(p["backbone_calls"]) == calls]
            labels.append(f"{a}\n{calls} backbone pass{'es' if calls > 1 else ''}")
            values.append(selected)
    ax.boxplot(values, tick_labels=labels, showfliers=False, whis=(5, 95))
    ax.set(ylabel="Full-pipeline latency (ms)", title="Observed execution depth · all seeds/passes; whiskers = p5–p95")
    save(fig, "latency-by-depth")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, a in zip(axes, ARCHITECTURES):
        matrices = [np.asarray(r["metrics"]["confusion_matrix"]) for r in records if r["architecture"] == a]
        matrix = np.mean([m / m.sum(1, keepdims=True) for m in matrices], axis=0)
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues", interpolation="nearest")
        ax.set(title=a, xlabel="Predicted global variant ID", ylabel="True global variant ID")
    fig.colorbar(im, ax=axes.tolist(), fraction=.025, pad=.02, label="Mean within-class proportion")
    # Explicit margins leave space for a shared colorbar.
    fig.subplots_adjust(left=.05, right=.88, bottom=.15, top=.9, wspace=.3)
    with (output / "confusion-matrices.svg").open("x", encoding="utf-8", newline="\n") as stream:
        fig.savefig(stream, format="svg", metadata={"Date": None})
    if preview:
        fig.savefig(preview / "confusion-matrices.png", dpi=130)
    plt.close(fig)


def mean_sd(values: list[float | None]) -> str:
    if any(v is None for v in values):
        return "N/A"
    return f"{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}"


def render_report(analysis: dict, records: list[dict], config: dict, provenance: dict) -> str:
    """Follow all nine sections of RESULTS_TEMPLATE using linked measured values."""
    root_link = provenance["repository_link"]
    source = root_link + "/" + config["benchmark"]
    update = root_link + "/" + config["incremental"]
    first = records[0]
    env = first["environment"]
    contract = read_json(first["path"] / "data_contract.json")
    rows, bootstrap = analysis["per_seed"], analysis["bootstrap"]
    lines = ["# FGVC-Aircraft — measured results", "",
        "## 1. Experimental context", "",
        "Primary study: FLAT, IND-2 and FT-FROZEN-2, seeds **42/43/44**, all **3,333 official test images**. "
        "Phase 8 reuses the completed frozen Phase 3–7 runs; it performs no fitting, inference or test-based model selection.", "",
        "| Field | Recorded value |", "| --- | --- |",
        f"| Analysis date / source commit | {provenance['created_utc']}; `{provenance['git_sha']}`; exact analysis source hashes in provenance.json |",
        "| Dataset | Torchvision 100 variants / 70 families / 30 manufacturers; train/val/test 3334/3333/3333 |",
        f"| Dataset manifest SHA256 | `{object_hash(contract['manifest_records_sha256'])}`; individual split hashes in linked data contracts |",
        f"| Taxonomy SHA256 | `{contract['taxonomy_sha256']}` |",
        f"| Preprocessing SHA256 | `{contract['preprocessing_sha256']}` |",
        f"| Runtime framework / driver / GPU | PyTorch {env['torch']}; CUDA {env['cuda_runtime']}; cuDNN {env['cudnn']}; {env.get('nvidia_smi')} |",
        f"| Analysis configuration SHA256 | `{object_hash(config)}` |",
        "| Precision / residency | FP32; batch 1; all deployment weights GPU resident; no inference embedding cache |",
        "| Power regime | Interactive Windows laptop; driver/OS clocks unlocked; per-pass telemetry retained; ambient temperature unmeasured |", "",
        "Preprocessing removes the bottom 20 px, bicubic letterboxes the complete remaining image to 224×224 with black padding, then applies ImageNet V1 normalization. "
        "All architectures share data, EfficientNet-B0 initialization family, preprocessing, device and measurement boundaries. "
        "Training uses AdamW, a 30-epoch cap, patience 7 and validation macro-recall checkpoint selection. Full CNN LR 1e-4; head LR 1e-3; shared MLP width 256. Resolved per-run configurations are authoritative.", "",
        f"**Full pipeline:** {first['full_pipeline_boundary']}.", "",
        f"**Model path:** {first['model_path_boundary']}. **FPS:** {first['fps_boundary']}.", "",
        "Warmup covers every trained node 32 times and label-free predicted routes. Timers synchronize CUDA. "
        "The benchmark has five full passes per checkpoint (149,985 image timings). Oracle routing is a separate untimed diagnostic.", "",
        "**Protocol deviations and scope:** no new primary experiment settings were selected in Phase 8. Generic ImageNet-frozen, three-stage, and FP16 results are unmeasured follow-ups. "
        "Initial incremental systems preserve global IDs through a 99-class active mask; the held-out family retains another variant. "
        "Archive checksum discrepancies, interruptions and discarded attempts remain documented in the earlier phase evidence.", "",
        "## 2. Main comparison", "",
        "Mean ± sample SD across seeds. Latency percentiles pool all 5×3,333 observations within each seed, then report mean ± SD of seed percentiles. "
        "FPS divides observed image counts by elapsed processing time. MB is decimal; MiB is binary. MACs count executed Conv2d/Linear multiply-accumulates only.", "",
        "| Metric | FLAT | IND-2 | FT-FROZEN-2 |", "| --- | ---: | ---: | ---: |"]
    fields = [("Variant macro recall (%)", "macro_recall_percent"), ("Variant top-1 (%)", "top1_percent"),
        ("Variant macro-F1 (%)", "macro_f1_percent"), ("Manufacturer accuracy from final variant (%)", "manufacturer_percent"),
        ("Family accuracy from final variant (%)", "family_percent"), ("Oracle variant macro recall (%)", "oracle_macro_percent"),
        ("First-stage routing accuracy (%)", "router_percent"), ("Full-pipeline p50 (ms)", "full_p50_ms"),
        ("Full-pipeline mean (ms)", "full_mean_ms"), ("**Full-pipeline p95 (ms)**", "full_p95_ms"),
        ("Full-pipeline p99 (ms)", "full_p99_ms"), ("Model-path p95 (ms)", "model_p95_ms"),
        ("Continuous throughput (FPS)", "sustained_fps"), ("Peak GPU allocated memory (MiB)", "peak_gpu_mib"),
        ("Deployment checkpoint storage (MB)", "stored_mb"), ("All model parameters (M)", "parameters_m"),
        ("Mean executed MACs/image (G)", "macs_g"), ("Mean executed backbone calls", "backbone_calls"),
        ("Singleton-bypass images (%)", "bypass_percent"), ("Cumulative fitting wall time incl. source FLAT (h)", "training_hours"),
        ("Optimizer steps incl. source FLAT", "optimizer_steps"), ("Training-image exposures incl. source FLAT", "training_exposures")]
    for label, field in fields:
        lines.append(f"| {label} | " + " | ".join(mean_sd([r[field] for r in rows if r["architecture"] == a]) for a in ARCHITECTURES) + " |")
    lines += ["", "FROZEN-2 / IND-3 / FT-FROZEN-3 / FROZEN-3: **unmeasured**. "
        "Shared cost includes its seed-matched source FLAT fitting. Training hours are elapsed fitting/validation/selection time on the recorded GPU, not independently instrumented GPU activity or equal-compute budgets.", "",
        "## 3. Paired differences, uncertainty and engineering objective", "",
        "| Fixed seed | Δ macro recall, shared − independent (pp) | Paired 95% bootstrap CI (pp) |", "| --- | ---: | --- |"]
    for r in bootstrap["per_seed"]:
        lines.append(f"| {r['seed']} | {r['difference_pp']:+.2f} | [{r['ci95_pp'][0]:+.2f}, {r['ci95_pp'][1]:+.2f}] |")
    fixed = bootstrap["fixed_seed_mean"]
    accuracy_ok = fixed["difference_pp"] >= -2
    latency_ok = analysis["p95_reduction_percent"] >= 30
    intervals_ok = all(r["ci95_pp"][0] >= -2 for r in bootstrap["per_seed"])
    lines += ["", f"Across the three fixed seed pairs, mean difference **{fixed['difference_pp']:+.2f} pp**; sample SD of differences **{bootstrap['training_seed_difference_sd_pp']:.2f} pp**. "
        f"The descriptive fixed-seed-mean image interval is [{fixed['ci95_pp'][0]:+.2f}, {fixed['ci95_pp'][1]:+.2f}] pp.", "",
        f"Bootstrap uses {bootstrap['replicates']:,} replicates, NumPy RNG seed {bootstrap['rng_seed']}, paired IDs sampled with replacement separately within each true variant, "
        "preserved class support and equal class weights. Each per-seed interval holds that model pair fixed. The fixed-seed mean uses identical image draws across the pairs; "
        "it does not treat seed/image combinations as independent observations. These intervals describe test-image sampling uncertainty, not uncertainty over future training seeds.", "",
        f"Mean seed-p95 difference: **{analysis['p95_difference_ms']:+.2f} ms**; reduction **{analysis['p95_reduction_percent']:.2f}%**. "
        f"Exploratory engineering checks: ≤2 pp observed degradation: **{accuracy_ok}**; ≥30% observed p95 reduction: **{latency_ok}**; "
        f"every fixed-pair accuracy CI lower bound ≥−2 pp: **{intervals_ok}**. This is not a formal noninferiority study, and no latency population confidence interval is claimed. "
        "Five-pass min/max/SD and every pass percentile are retained in the CSVs to expose repeatability.", "",
        "![Accuracy, Pareto frontier and per-seed uncertainty](accuracy-latency-bootstrap.svg)", "",
        f"Mean accuracy–latency Pareto frontier among measured systems: **{', '.join(analysis['pareto_frontier'])}**. "
        "FLAT has higher recall and lower p95 than either measured hierarchy; the modular update question is a separate benefit/cost axis. No model was selected using this frontier.", "",
        "## 4. Routing error attribution", "",
        "| Architecture / seed | Manufacturer router (%) | Hard macro recall (%) | Oracle macro recall (%) | Oracle − hard (pp) |", "| --- | ---: | ---: | ---: | ---: |"]
    for r in rows:
        if r["architecture"] != "FLAT":
            lines.append(f"| {r['architecture']} / {r['seed']} | {r['router_percent']:.2f} | {r['macro_recall_percent']:.2f} | {r['oracle_macro_percent']:.2f} | {r['routing_penalty_pp']:.2f} |")
    lines += ["", "![First wrong stage, routing penalty and repeated passes](routing-and-repeatability.svg)", "",
        "First-wrong-stage proportions use all test images per seed, counting manufacturer errors before downstream variant errors. "
        "For FLAT, mistakes are assigned to its single variant decision. Two-stage systems have no family router, so conditional family-router accuracy is N/A. "
        "There are 17 trained variant specialists and 13 deterministic single-variant manufacturer branches. Bypass rates above use actual predicted routes. "
        "Within-family, cross-family within manufacturer and cross-manufacturer error counts are mutually exclusive and available in family_error_breakdown.csv.", "",
        "![Within-family and cross-family errors](family-errors.svg)", "",
        "## 5. Latency by path", "",
        "| Architecture / seed | Executed backbone calls | Distinct test images | Observations | p50 (ms) | p95 (ms) |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in analysis["depth_latency"]:
        lines.append(f"| {r['architecture']} / {r['seed']} | {r['backbone_calls']} | {r['images']} | {r['observations']} | {r['p50_ms']:.2f} | {r['p95_ms']:.2f} |")
    lines += ["", "![Latency grouped by executed backbone count](latency-by-depth.svg)", "",
        f"[Every predicted manufacturer branch p50/p95](branch_latency.csv); [per-pass repeatability](pass_latency.csv); [original per-branch graph]({source}/branch-latency.svg). "
        "Latency groups by predicted route; accuracy/support groups by true manufacturer and includes routing mistakes. Path-specific GPU peaks were not measured: only per-run peaks appear above. "
        "All stored independent models count toward residency/storage even when idle. Cold start, cold filesystem cache and separate preprocessing/backbone/router/transfer component timers are unmeasured. "
        "The full and model-path boundaries are nested; their mean difference includes preprocessing, transfer and extra synchronization, not an isolated component profile.", "",
        "## 6. Specialist update / introduction", "",
        "Initial systems know 99 variants. Boeing 737-200 (global ID 2) was chosen using training taxonomy only and excluded from initial train/validation for **every initial backbone and specialist**. "
        "After introduction, only the affected Boeing specialist is retrained with new-class training examples and existing-class replay; validation selects the replacement. "
        "The manufacturer router, shared backbone and unrelated heads retain their parameters and functional logits. No extra dataset is used.", "",
        "| Metric | Independent cascade | Shared frozen backbone |", "| --- | ---: | ---: |"]
    def update_values(fn):
        return " | ".join(mean_sd([fn(r) for r in analysis["incremental"] if r["architecture"] == a]) for a in ARCHITECTURES[1:])
    update_fields = [("New-class recall after (%)", lambda r: 100 * r["predicted"]["groups"]["new"]["after"]["macro_recall"]),
        ("Old Boeing-class macro recall before (%)", lambda r: 100 * r["predicted"]["groups"]["old_within_branch"]["before"]["macro_recall"]),
        ("Old Boeing-class macro recall after (%)", lambda r: 100 * r["predicted"]["groups"]["old_within_branch"]["after"]["macro_recall"]),
        ("Outside-branch accuracy before (%)", lambda r: 100 * r["predicted"]["groups"]["outside_branch"]["before"]["micro_accuracy"]),
        ("Outside-branch accuracy after (%)", lambda r: 100 * r["predicted"]["groups"]["outside_branch"]["after"]["micro_accuracy"]),
        ("Update wall time (s)", lambda r: r["update"]["update_wall_seconds"]),
        ("Parameters retrained", lambda r: r["update"]["trainable_parameters"]),
        ("Replacement checkpoint (MB)", lambda r: r["update"]["replacement_bytes"] / 1e6)]
    for label, fn in update_fields:
        lines.append(f"| {label} | {update_values(fn)} |")
    lines += ["", f"[Full measured update report, per-image before/after predictions, costs and checksum audits]({update}/REPORT.md). "
        "Update time includes shared feature extraction, fitting, selection, export and reload, excluding initialization/loading. "
        "This is a separately trained reduced-label experiment; its results must not be mixed with the 100-class primary systems. "
        "Unchanged outside-branch accuracy does not imply every wrong label is unchanged for images incorrectly routed into Boeing.", "",
        "## 7. Plots to publish", "",
        "![Mean row-normalized variant confusion matrices](confusion-matrices.svg)", "",
        f"- [Per-seed confusion matrices and every class recall]({source}/REPORT.md#7-full-networks-and-training-result-review).",
        f"- [Every specialist learning curve]({source}/all-specialist-macro_recall.svg) and [loss curve]({source}/all-specialist-loss.svg).",
        f"- [Per-class recall]({source}/class-recall.svg), [branch support]({source}/branch-accuracy-support.svg), [storage and GPU memory]({source}/storage-memory.svg).",
        f"- [Complete independent network]({source}/network-IND-2.svg), [shared network]({source}/network-FT-FROZEN-2.svg), [EfficientNet internals]({source}/efficientnet-b0-detail.svg).",
        f"- [Incremental cost, before/after accuracy and training plots]({update}/REPORT.md).", "",
        "All graphics derive from measured numeric artifacts or exact network topology. No generated aircraft pictures are required for interpreting the results.", "",
        "## 8. Discussion and conclusion", "",
        "**RQ1 — Accuracy:** FT-FROZEN-2 exceeds IND-2 on each fixed seed pair, while FLAT achieves the highest measured variant macro recall. "
        "The per-seed paired intervals quantify image-sampling uncertainty; three seeds do not establish a population-of-training-seeds guarantee.", "",
        "**RQ2 — Latency:** one shared backbone reduces the hierarchical pipeline p95 under the same all-resident FP32 boundary. "
        "FLAT remains faster than the shared hierarchy. Differences include this implementation's routing, finite-value checks, eval-mode maintenance and synchronization overhead.", "",
        "**RQ3 — Routing error:** oracle gaps and first-wrong-stage counts expose lost reachability as well as residual within-branch discrimination. "
        "Oracle diagnostics are not deployable accuracy. Singleton bypass frequencies depend on the taxonomy and learned router.", "",
        "**RQ4 — Modularity:** both approaches replace only one specialist with audited unchanged unrelated parameters. "
        "The shared replacement is cheaper here but loses more recall on existing Boeing classes. Outside-branch accuracy is measured unchanged.", "",
        "**RQ5 — Feature suitability:** generic ImageNet-frozen ablations remain unmeasured, so these results cannot isolate the gain from adapting the encoder before freezing.", "",
        "Claims apply to this official FGVC-Aircraft edition, 224×224 preprocessing, EfficientNet-B0, three seeds and one GPU/residency policy. "
        "They do not establish generality to other datasets, larger models, hardware, class introductions or equal training-compute budgets. "
        "Laptop clocks/load, caching, only five timing passes, small class test supports and unmeasured cold-start/component times limit deployment extrapolation.", "",
        "## 9. Reproduction commands and audit links", "",
        "Follow the single **Reproduce the complete study** sequence in the repository README. The analysis commands run entirely from committed test evidence:", "",
        "```powershell", ".venv-cuda/Scripts/python.exe -m cls_balance.report_analysis --config configs/analysis.yaml",
        ".venv-cuda/Scripts/python.exe -m cls_balance.report_analysis --audit <phase8-review-directory>",
        ".venv-cuda/Scripts/python.exe -m cls_balance.report_analysis --rebuild <phase8-review-directory> --output .tmp/rebuilt-phase8", "```", "",
        "| Architecture / seed | Training run / selected checkpoint identity | Config SHA256 | Test artifacts |", "| --- | --- | --- | --- |"]
    for r in rows:
        base = f"{source}/{r['run_id']}"
        lines.append(f"| {r['architecture']} / {r['seed']} | `{r['training_run_id']}` / `{r['model_sha256']}` | `{r['config_sha256']}` | [predictions]({base}/predictions_test.csv), [timings]({base}/latency_test.csv), [metrics]({base}/metrics.json), [provenance]({base}/provenance.json), [environment]({base}/environment.json) |")
    lines += ["", "Per-run provenance binds the source encoder, selected full CNN/head checkpoint files and training summary hashes. "
        "The source publication manifests bind every linked per-image artifact. Phase 8 summary.json binds every generated table/figure; provenance.json binds the analysis sources and environment.", "",
        "**Exclusions, failures and resource limits:** no test images or classes are excluded. "
        "No completed seed was dropped. Interrupted/discarded Phase 4 fits (including user pause and system sleep) were restarted, "
        "and are excluded from completed-fit training costs, not hidden or counted as successful training. "
        "Shared fitting includes the source FLAT cost; no equal GPU-hour budget was imposed. No OOM occurred in completed benchmarks. "
        "Raw GPU activity, CPU RSS, cold-start latency and broader ablations are unmeasured. "
        "The dataset authors' archive checksum differs from the independently recorded archive checksum; preparation preserves this discrepancy and verifies decoded image/annotation checksums.", "",
        f"[Phase 4 failures and residency evidence]({root_link}/results/stage4/validation-20261027T071705Z-2b712381/REPORT.md); "
        f"[Phase 6 validation]({root_link}/results/stage6/validation-20261028T023745Z-a38164e6/REPORT.md); "
        f"[Phase 7 validation]({root_link}/results/stage7/validation-20261029T103720Z-f75088da/REPORT.md). "
        "Dataset pixels, training image-ID lists, embeddings and weights stay outside Git.", ""]
    return "\n".join(lines)


TABLES = ("per_seed", "branch_latency", "depth_latency", "pass_latency", "family_error_breakdown")
FIGURES = ("accuracy-latency-bootstrap", "routing-and-repeatability", "family-errors", "latency-by-depth", "confusion-matrices")
OUTPUT_FILES = {"analysis.json", "config.resolved.json", "provenance.json", "REPORT.md"} | {
    f"{key}.csv" for key in TABLES} | {f"{name}.svg" for name in FIGURES}


def write_outputs(output: Path, analysis: dict, records: list[dict], config: dict, provenance: dict,
                  preview: Path | None) -> None:
    write_json(output / "analysis.json", analysis)
    for key in TABLES:
        with (output / f"{key}.csv").open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(csv_text(analysis[key]))
    with (output / "REPORT.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(render_report(analysis, records, config, provenance))
    draw_figures(output, analysis, records, preview)


def publish(repo: Path, config: dict, output: Path, preview: Path | None = None) -> Path:
    records, taxonomy, updates = load_sources(repo, config)
    analysis = calculate(records, taxonomy, updates, config)
    output.mkdir(parents=True, exist_ok=False)
    env = environment(repo)
    from datetime import datetime, timezone
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(), "git_sha": env["git_sha"],
                  "repository_link": Path(os.path.relpath(repo, output)).as_posix(),
                  "environment": env, "config_sha256": object_hash(config),
                  "source_sha256": {p.relative_to(repo).as_posix(): file_hash(p) for p in sorted((repo / "src/cls_balance").rglob("*.py"))},
                  "benchmark_runs": [r["run_id"] for r in records]}
    write_json(output / "config.resolved.json", config)
    write_json(output / "provenance.json", provenance)
    write_outputs(output, analysis, records, config, provenance, preview)
    write_json(output / "summary.json", {"status": "complete", "kind": "phase8-analysis", "artifacts_sha256": inventory(output)})
    return output


def _audit_and_load(repo: Path, output: Path) -> tuple[dict, dict, list[dict], dict, dict]:
    """Verify every output and return the recomputed evidence for rebuilding."""
    summary = read_json(output / "summary.json")
    if (summary.get("status") != "complete" or summary.get("kind") != "phase8-analysis"
            or set(summary["artifacts_sha256"]) != OUTPUT_FILES or inventory(output) != summary["artifacts_sha256"]):
        raise ValueError("Incomplete/corrupt Phase 8 publication")
    config = read_json(output / "config.resolved.json")
    provenance = read_json(output / "provenance.json")
    if object_hash(config) != provenance["config_sha256"]:
        raise ValueError("Analysis configuration changed")
    records, taxonomy, updates = load_sources(repo, config)
    analysis = calculate(records, taxonomy, updates, config)
    if object_hash(analysis) != object_hash(read_json(output / "analysis.json")):
        raise ValueError("Analysis disagrees with per-image source evidence")
    for key in TABLES:
        if (output / f"{key}.csv").read_text(encoding="utf-8") != csv_text(analysis[key]):
            raise ValueError("Analysis CSV disagrees with recomputed evidence")
    if (output / "REPORT.md").read_text(encoding="utf-8") != render_report(analysis, records, config, provenance):
        raise ValueError("Analysis report disagrees with recomputed evidence")
    audit = {"status": "verified", "benchmark_runs": len(records), "test_images_per_run": len(records[0]["predictions"]),
            "fixed_seeds": list(SEEDS), "per_image_recomputed": True, "source_publications_verified": True,
            "artifacts": len(summary["artifacts_sha256"])}
    return audit, analysis, records, config, provenance


def audit_publication(repo: Path, output: Path) -> dict:
    """Verify source publications and recompute all derived scientific values."""
    return _audit_and_load(repo, output)[0]


def rebuild(repo: Path, source: Path, output: Path, preview: Path | None = None) -> dict:
    audit, analysis, records, config, provenance = _audit_and_load(repo, source)
    output.mkdir(parents=True, exist_ok=False)
    for name in ("config.resolved.json", "provenance.json"):
        shutil.copyfile(source / name, output / name)
    write_outputs(output, analysis, records, config, provenance, preview)
    write_json(output / "summary.json", {"status": "complete", "kind": "phase8-analysis", "artifacts_sha256": inventory(output)})
    if inventory(source) != inventory(output):
        raise ValueError("Phase 8 rebuild changed published bytes")
    return {**audit, "source_review_id": source.name, "byte_identical": True,
            "source_summary_sha256": file_hash(source / "summary.json")}


def export_results_doc(publication: Path, destination: Path) -> None:
    """Export the nine-section report, rebasing links to its documentation location."""
    def rebase(match: re.Match) -> str:
        target = match[1]
        if "://" in target or target.startswith("#"):
            return match[0]
        relative = Path(os.path.relpath((publication / target).resolve(), destination.parent.resolve())).as_posix()
        return f"]({relative})"
    content = re.sub(r"\]\(([^)\n]+)\)", rebase, (publication / "REPORT.md").read_text(encoding="utf-8"))
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(f"<!-- Generated from immutable publication {publication.name}/REPORT.md. -->\n\n" + content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=Path)
    group.add_argument("--audit", type=Path)
    group.add_argument("--rebuild", type=Path)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preview-root", type=Path)
    parser.add_argument("--results-doc", type=Path, help="Also export the nine-section report to docs/RESULTS.md (explicitly replaces that documentation file)")
    args = parser.parse_args()
    if args.audit:
        if args.results_doc:
            parser.error("--results-doc is supported with --config only")
        print(audit_publication(args.repo, args.audit))
    elif args.rebuild:
        if args.results_doc:
            parser.error("--results-doc is supported with --config only")
        if args.output is None:
            parser.error("--rebuild requires a new --output directory")
        rebuild(args.repo, args.rebuild, args.output, args.preview_root)
        print(args.output)
    else:
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        output = publish(args.repo, config, args.output or args.repo / "results/stage8" / unique_id("analysis"), args.preview_root)
        if args.results_doc:
            export_results_doc(output, args.results_doc)
        print(output)


if __name__ == "__main__":
    main()
