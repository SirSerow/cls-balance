"""Audit completed FLAT runs and publish aggregate results and validation plots.

No aircraft images, training-ID lists, embeddings or weights are copied.
Requires the optional ``requirements/phase3-plots.txt`` dependency.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import statistics
from typing import TYPE_CHECKING
import uuid

from .artifacts import file_hash, object_hash, write_json
from .evaluate import audit_run

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_svg(figure: Figure, path: Path) -> None:
    """Write immutable LF text so artifact hashes survive Git checkouts on Windows."""
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        figure.savefig(stream, format="svg")


def audit_training_run(path: Path) -> dict:
    """Require completion and verify all saved bytes and nested evaluation evidence."""
    summary = read_json(path / "summary.json")
    required = {"flat.pt", "flat.json", "encoder.pt", "encoder.json", "history.csv",
                "resources.json", "checkpoint_manifest.json", "config.resolved.json", "environment.json",
                "dataset_manifest_hashes.json", "taxonomy.json", "preprocessing.json", "source_hashes.json"}
    if (summary.get("status") != "complete" or summary.get("test_evaluated") is not False
            or not required <= set(summary.get("artifacts_sha256", {}))):
        raise ValueError("Incomplete Phase 3 run")
    for name, digest in summary["artifacts_sha256"].items():
        if Path(name).name != name or file_hash(path / name) != digest:
            raise ValueError(f"Training artifact checksum mismatch: {name}")
    for split in ("train", "val"):
        relative = Path(summary["evaluations"][split]["path"])
        evaluation = (path / relative).resolve()
        if not evaluation.is_relative_to(path.resolve()):
            raise ValueError("Evaluation path escapes training run")
        verified = audit_run(evaluation)
        metadata = read_json(evaluation / "metadata.json")
        metrics = read_json(evaluation / "metrics.json")
        if (verified != summary["evaluations"][split]["audit"] or metrics["split"] != split
                or metadata["model_id"] != summary["run_id"]):
            raise ValueError("Evaluation does not belong to this selected checkpoint")
    return summary


def draw_figures(output: Path, seeds: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "figure.facecolor": "white"})
    colors = ("#126782", "#c85c2c", "#527d39")
    figure, axes = plt.subplots(len(seeds), 2, figsize=(12, 3.1 * len(seeds)), squeeze=False)
    for row, seed in enumerate(seeds):
        history = seed["history"]
        epochs = [int(h["epoch"]) for h in history]
        for split, color in (("train", colors[0]), ("val", colors[1])):
            axes[row, 0].plot(epochs, [100 * float(h[f"{split}_macro_recall"]) for h in history],
                              label="Augmented train" if split == "train" else "Validation", color=color)
            axes[row, 1].plot(epochs, [float(h[f"{split}_loss"]) for h in history], label=split, color=color)
        for column in range(2):
            axes[row, column].axvline(seed["best_epoch"], color="#525252", linestyle="--", label="Selected epoch")
            axes[row, column].set_xlabel("Epoch")
            axes[row, column].grid(alpha=0.18)
            axes[row, column].legend(loc="lower right" if column == 0 else "upper right", fontsize=8)
        axes[row, 0].set(ylabel="Variant macro recall (%)", ylim=(0, 100), title=f"Seed {seed['seed']} · learning")
        axes[row, 1].set(ylabel="Cross-entropy", title=f"Seed {seed['seed']} · loss")
    figure.suptitle("FLAT EfficientNet-B0 · official train / validation · FP32\nSelection uses validation macro recall only", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    save_svg(figure, output / "learning-curves.svg")
    plt.close(figure)

    for seed in seeds:
        matrix = np.asarray(seed["validation"]["confusion_matrix"])
        normalized = matrix / matrix.sum(axis=1, keepdims=True)
        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues", interpolation="nearest")
        ax.set(xlabel="Predicted global variant ID", ylabel="True global variant ID",
               title=f"Seed {seed['seed']} · validation confusion\nRow normalized; all 3,333 validation images")
        fig.colorbar(im, ax=ax, label="Recall fraction")
        fig.tight_layout()
        save_svg(fig, output / f"confusion-seed-{seed['seed']}.svg")
        plt.close(fig)

    classwise = seeds[0]["validation"]["classwise"]
    order = sorted(range(len(classwise)), key=lambda i: statistics.mean(
        s["validation"]["classwise"][i]["recall"] for s in seeds))
    fig, ax = plt.subplots(figsize=(15, 7))
    for index, seed in enumerate(seeds):
        ax.plot(range(100), [100 * seed["validation"]["classwise"][i]["recall"] for i in order],
                ".", alpha=0.8, color=colors[index % len(colors)], label=f"Seed {seed['seed']}")
    ax.set_xticks(range(100), [classwise[i]["variant_name"] for i in order], rotation=90, fontsize=6)
    ax.set(ylabel="Validation recall (%)", ylim=(-2, 102), title="Per-variant recall · sorted by mean across seeds")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    save_svg(fig, output / "classwise-validation.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    positions = np.arange(len(seeds))
    for offset, split, color in ((-0.18, "train", colors[0]), (0.18, "validation", colors[1])):
        ax.bar(positions + offset, [100 * s[split]["variant_macro_recall"] for s in seeds],
               width=0.36, color=color, label="Deterministic train" if split == "train" else "Validation")
    ax.set_xticks(positions, [f"Seed {s['seed']}\nselected epoch {s['best_epoch']}" for s in seeds])
    ax.set(ylabel="Variant macro recall (%)", ylim=(0, 100), title="Selected checkpoints · generalization")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    save_svg(fig, output / "generalization.svg")
    plt.close(fig)


def publish(runs: list[Path], output_root: Path) -> Path:
    """Create a new review bundle from audited, immutable training runs."""
    summaries = [audit_training_run(path) for path in runs]
    seeds = [s["seed"] for s in summaries]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Each seed must have one selected FLAT run")
    for key in ("taxonomy_sha256", "preprocessing_sha256", "dataset_manifest_sha256"):
        if len({s[key] for s in summaries}) != 1:
            raise ValueError(f"Runs disagree on {key}")
    configs = [read_json(p / "config.resolved.json") for p in runs]
    comparable = [{k: v for k, v in c.items() if k not in {"seed", "seeds", "output_root"}} for c in configs]
    if len({object_hash(c) for c in comparable}) != 1:
        raise ValueError("Runs have different training settings")
    envs = [read_json(p / "environment.json") for p in runs]
    for key in ("torch", "cuda_runtime", "gpus", "device", "runtime_policy", "git_sha"):
        if len({object_hash(e.get(key)) for e in envs}) != 1:
            raise ValueError(f"Runs disagree on environment {key}")
    report_id = datetime.now(timezone.utc).strftime("flat-%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
    output = output_root / report_id
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for path, summary in zip(runs, summaries, strict=True):
        target = output / f"seed-{summary['seed']}"
        target.mkdir()
        for name in ("history.csv", "resources.json", "environment.json", "config.resolved.json", "dataset_manifest_hashes.json", "source_hashes.json"):
            shutil.copyfile(path / name, target / name)
        train_eval = path / summary["evaluations"]["train"]["path"]
        val_eval = path / summary["evaluations"]["val"]["path"]
        shutil.copyfile(train_eval / "metrics.json", target / "metrics_train.json")
        shutil.copytree(val_eval, target / "validation")
        with (path / "history.csv").open(encoding="utf-8", newline="") as stream:
            history = list(csv.DictReader(stream))
        references = {"run_id": summary["run_id"], "summary_sha256": file_hash(path / "summary.json"),
                      "checkpoints": {name: {"file_sha256": read_json(path / f"{name}.json")["file_sha256"],
                                             "state_sha256": read_json(path / f"{name}.json")["state_sha256"]}
                                      for name in ("flat", "encoder")},
                      "weights_location": "ignored local training run; not redistributed",
                      "exact_reload_logits": True, "exact_reload_embeddings": True}
        write_json(target / "checkpoint_references.json", references)
        records.append({"seed": summary["seed"], "run_id": summary["run_id"],
                        "best_epoch": summary["best_epoch"], "epochs_completed": summary["epochs_completed"],
                        "history": history, "resources": summary["resources"],
                        "train": read_json(train_eval / "metrics.json"), "validation": read_json(val_eval / "metrics.json")})
    draw_figures(output, records)
    values = [100 * r["validation"]["variant_macro_recall"] for r in records]
    aggregate = {"validation_macro_recall_percent_mean": statistics.mean(values),
                 "validation_macro_recall_percent_sample_std": statistics.stdev(values) if len(values) > 1 else None,
                 "seeds": seeds, "test_evaluated": False,
                 "scope": "Phase 3 FLAT train/validation only; no architecture or latency comparison"}
    write_json(output / "aggregate.json", aggregate)
    rows = ["| Seed | Selected / completed epochs | Train macro recall (%) | Validation macro recall (%) | Validation top-1 (%) | Training hours |",
            "| --- | --- | --- | --- | --- | --- |"]
    for r in records:
        rows.append(f"| {r['seed']} | {r['best_epoch']} / {r['epochs_completed']} | {100*r['train']['variant_macro_recall']:.2f} | "
                    f"{100*r['validation']['variant_macro_recall']:.2f} | {100*r['validation']['variant_top1']:.2f} | "
                    f"{r['resources']['training_wall_seconds']/3600:.3f} |")
    std = aggregate["validation_macro_recall_percent_sample_std"]
    context = summaries[0]
    text = f"""# Phase 3 — measured FLAT baseline

FLAT EfficientNet-B0 was trained on all 3,334 official training images and selected using macro recall on all 3,333 validation images. Test inference was deferred until the comparison settings are frozen. These are validation results; they do not establish the main IND-2 versus FT-FROZEN-2 claim.

## Experimental context

- Dataset: Torchvision FGVC-Aircraft, 100 variants / 70 families / 30 manufacturers; 3,334 / 3,333 / 3,333 official splits.
- Training Git SHA: `{envs[0]['git_sha']}`; dirty flags by seed: `{dict(zip(seeds, [e['git_dirty'] for e in envs], strict=True))}`. Per-seed source hashes record the training snapshot; publication metadata fingerprints the report-generation sources.
- Device: `{envs[0]['device']}`; GPU: `{envs[0]['gpus']}`; Torch: `{envs[0]['torch']}`; CUDA: `{envs[0]['cuda_runtime']}`.
- Dataset manifest (canonical JSON records): `{context['dataset_manifest_sha256']}`.
- Taxonomy: `{context['taxonomy_sha256']}`; preprocessing: `{context['preprocessing_sha256']}`.
- FP32, TF32 disabled, deterministic algorithms, OS/driver power defaults. Training uses shuffled batches of {configs[0]['batch_size']}; final evaluation uses the common batch-1 adapter.
- ImageNet V1 initialization; full fine-tuning with AdamW, LR {configs[0]['learning_rate']}, weight decay {configs[0]['weight_decay']}; cap {configs[0]['epochs']} epochs; patience {configs[0]['patience']}; earliest epoch wins a validation-score tie.
- Remove bottom 20-pixel banner, preserve aspect ratio, center black padding to 224×224, ImageNet normalization. Training adds deterministic mild flips and photometric jitter.

## Selected-checkpoint measurements

{chr(10).join(rows)}

Validation macro recall across the recorded seeds: **{statistics.mean(values):.2f}%**{f' ± {std:.2f} pp (sample standard deviation)' if std is not None else ' (one seed only)'}. This spread measures training-seed variability, not a confidence interval or test performance.

![Training curves](learning-curves.svg)

Dashed lines mark selected epochs. Training curves use augmentation and training mode; final train scores below use deterministic preprocessing and eval mode. A train/validation gap indicates generalization limits; no settings were selected with test labels.

![Generalization](generalization.svg)

![Per-class validation recall](classwise-validation.svg)

Per-class support and global class names are in each seed's validation CSV. Recall estimates have only 33 or 34 validation examples per class.

## Artifact and encoder checks

All seeds produced hash-checked full and encoder-only checkpoints. Reloaded logits and pooled embeddings matched exactly on a fixed validation input on the same device. Encoders copy only the selected features/pooling, return 1,280 dimensions, disable gradients and remain in eval mode. Checkpoint files stay local under ignored `runs/`; `checkpoint_references.json` records their file/state hashes. Validation predictions and aggregate train metrics are included; dataset images and training-image ID lists are excluded.

## Limitations and protocol deviations

Only Phase 3 is measured. IND-2, FT-FROZEN-2, oracle routing, p95 latency, sustained FPS, inference memory, MACs and incremental updates remain unmeasured. Training wall time includes epoch training, validation and checkpoint selection; it is not an inference benchmark. GPU clocks were not locked, and other desktop applications may affect elapsed time. The archive-checksum discrepancy documented in Phase 1 remains applicable. No dataset subset, resolution change, test-label tuning, mixed precision or optimizer recipe change was introduced.

## Reproduction and audit

From the repository root:

```powershell
.venv-cuda/Scripts/python.exe -m cls_balance.train --config configs/flat.yaml
.venv-cuda/Scripts/python.exe -m cls_balance.report_flat --runs <run-seed-42> <run-seed-43> <run-seed-44> --output-root results/stage3
.venv/Scripts/python.exe scripts/freeze_tests.py --check
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m pytest -q checks/
```

Replace report placeholders with the local run directories recorded below. Full environment/configuration, histories, resource accounting, audited per-image validation predictions, classwise metrics and row-normalized confusion matrices accompany this report. Reproduction creates new immutable run IDs.

Implementation test output and additional export-audit commands are linked from the [repository README](../../../README.md).

"""
    for r in records:
        text += f"- Seed {r['seed']}: `{r['run_id']}`; [validation metrics](seed-{r['seed']}/validation/metrics.json); [confusion](confusion-seed-{r['seed']}.svg).\n"
    (output / "REPORT.md").write_text(text, encoding="utf-8", newline="\n")
    repo = Path(__file__).resolve().parents[2]
    sources = [p for folder in ("src", "checks", "configs") for p in (repo / folder).rglob("*")
               if p.is_file() and p.suffix in {".py", ".yaml"}]
    write_json(output / "publication.json", {"report_id": report_id, "source_run_ids": [s["run_id"] for s in summaries],
               "source_sha256": {p.relative_to(repo).as_posix(): file_hash(p) for p in sorted(sources)},
               "artifacts_sha256": {p.relative_to(output).as_posix(): file_hash(p)
                                    for p in sorted(output.rglob("*")) if p.is_file()}})
    print(output, flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("results/stage3"))
    args = parser.parse_args()
    publish(args.runs, args.output_root)


if __name__ == "__main__":
    main()
