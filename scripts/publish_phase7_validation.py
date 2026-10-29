"""Publish genuine Phase 7 checks, execution logs and artifact-rebuild evidence."""

import argparse
from pathlib import Path
import re
from xml.etree import ElementTree

from cls_balance.artifacts import environment, file_hash, write_json
from cls_balance.incremental import audit_inventory, inventory
from cls_balance.report_flat import read_json
from cls_balance.report_incremental import audit_publication
from cls_balance.train_independent import unique_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training", "evaluation", "report", "cpu-log", "cuda-log", "plot-log", "cpu-junit",
                 "freeze-log", "training-log", "evaluation-log", "rebuild-audit", "data-audit"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--discarded-log", type=Path)
    parser.add_argument("--discarded-run", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("results/stage7"))
    args = parser.parse_args()
    training, evaluation = audit_inventory(args.training), audit_inventory(args.evaluation)
    review = audit_publication(args.report)
    source = read_json(args.report / "source.json")
    if (source["training_run_id"] != args.training.name or source["evaluation_run_id"] != args.evaluation.name
            or source["training_summary_sha256"] != file_hash(args.training / "summary.json")
            or source["evaluation_summary_sha256"] != file_hash(args.evaluation / "summary.json")):
        raise ValueError("Phase 7 validation lineage mismatch")
    logs = {name: getattr(args, name.replace("-", "_")).read_text(encoding="utf-8")
            for name in ("cpu-log", "cuda-log", "plot-log")}
    counts = {}
    for name, value in logs.items():
        count = re.search(r"(\d+) passed", value)
        if count is None or "FAILED " in value or "ERROR " in value:
            raise ValueError("Require passing CPU/CUDA/plot check logs")
        counts[name] = int(count[1])
    suites = list(ElementTree.parse(args.cpu_junit).getroot().iter("testsuite"))
    if (sum(int(s.get("failures", 0)) + int(s.get("errors", 0)) for s in suites)
            or sum(int(s.get("tests", 0)) - int(s.get("skipped", 0)) for s in suites) != counts["cpu-log"]):
        raise ValueError("CPU test/JUnit mismatch")
    freeze, rebuilt, data = map(read_json, (args.freeze_log, args.rebuild_audit, args.data_audit))
    if (freeze["status"] != "verified" or rebuilt["status"] != "verified"
            or rebuilt["source_review_id"] != args.report.name or data["status"] != "verified"
            or data["images_checked"] != 10000 or data["training_run_id"] != args.training.name):
        raise ValueError("Freeze/rebuild/data audit mismatch")
    discarded = None
    if args.discarded_run:
        if (args.discarded_run / "summary.json").exists() or list(args.discarded_run.rglob("epoch-*.json")):
            raise ValueError("Discarded preflight attempt unexpectedly contains completed epochs")
        discarded = {"run_id": args.discarded_run.name, "completed_epochs": 0,
                     "reason": "Windows spawn could not pickle FlatTargets through runpy; fixed normal module import before successful run",
                     "used_for_experimental_results": False, "artifacts_sha256": inventory(args.discarded_run)}
    output = args.output_root / unique_id("validation")
    output.mkdir(parents=True, exist_ok=False)
    def copy_text(target: str, source: Path) -> None:
        with (output / target).open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(source.read_text(encoding="utf-8").replace("\r\n", "\n"))
    for target, source_path in (("cpu-tests.txt", args.cpu_log), ("cuda-tests.txt", args.cuda_log),
        ("plot-checks.txt", args.plot_log), ("cpu-tests.xml", args.cpu_junit),
        ("training.log", args.training_log), ("evaluation.log", args.evaluation_log)):
        copy_text(target, source_path)
    for name, value in {"frozen-suite.json": freeze, "publication-audit.json": review,
                        "rebuild-audit.json": rebuilt, "post-training-data-audit.json": data,
                        "training-summary.json": training, "evaluation-summary.json": evaluation}.items():
        write_json(output / name, value)
    repo = Path(__file__).resolve().parents[1]
    env = environment(repo)
    env["runtime_policy"] = "post-experiment validation snapshot; measured environments retained in run artifacts"
    write_json(output / "environment.json", env)
    sources = sorted({*repo.glob("src/cls_balance/**/*.py"), *repo.glob("checks/*.py"),
                      *repo.glob("scripts/*phase7*.py"), repo / "scripts/train_incremental_awake.py"})
    write_json(output / "validation_source_hashes.json", {p.relative_to(repo).as_posix(): file_hash(p) for p in sources})
    if discarded is not None:
        write_json(output / "discarded-preflight.json", discarded)
    if args.discarded_log:
        copy_text("discarded-preflight.log", args.discarded_log)
    text = f"""# Phase 7 validation and reproduction evidence

The complete CPU suite passed **{counts['cpu-log']} tests**. The CUDA-focused
suite passed **{counts['cuda-log']} checks**, including actual EfficientNet
class introduction/reload on generated pixels. The plotting suite passed
**{counts['plot-log']} checks**, including artifact-only byte-stable regeneration.
Synthetic checks are never experimental performance. The original 172-test
freeze remains verified.

[Measured specialist-update results and complete networks](../{args.report.name}/REPORT.md).
All three seeds have fresh initial 99-class FLAT sources and independent/shared
systems, followed by only the affected specialist update. Every before/after,
predicted/oracle evaluation retains all 3,333 official test images. The published
review covers {review['seed_architecture_pairs']} seed/architecture pairs.
The semantic audit checks exact supervised-ID exclusion, seed/source/config/data
hashes, encoder ancestry, checkpoint states, global logit maps, local validation
selection, frozen weights/buffers and unchanged unrelated logits. Router decisions
and predictions routed outside the affected branch are unchanged.

All 10,000 source images were rechecked against the original preparation after
the experiment. The public review and all its figures/reports were rebuilt
without images, supervised IDs, caches or trained weights; per-file comparisons
are in `rebuild-audit.json`. The execution logs and nested evaluation summaries
provide measurement lineage. No training image IDs or dataset pixels are copied.

An initial preflight attempt failed at Windows worker creation before completing
any epoch. It is explicitly discarded. Normal module import fixed the spawn
contract. A protocol review before the successful experiment also tightened class
selection to preserve the held-out class's existing family; the successful
training-only rule selects 737-200. No test results informed this selection.

Initial seed-42 backbone training overlapped CPU regression/report checks, which
can affect its initial wall-time accounting. Specialist update timings and test
inference ran without competing foreground test/plot workloads. Clocks remained
unlocked; desktop background activity remains a limitation. Update duration is
not an inference-latency or throughput measurement.

```powershell
.venv/Scripts/python.exe scripts/freeze_tests.py --check
.venv/Scripts/python.exe -m pytest -q tests checks
.venv-cuda/Scripts/python.exe -m pytest -q checks/test_phase6.py checks/test_phase7.py checks/test_phase7_report.py
.venv-cuda/Scripts/python.exe scripts/train_incremental_awake.py train --config configs/incremental.yaml
.venv-cuda/Scripts/python.exe scripts/train_incremental_awake.py evaluate --suite {args.training.as_posix()}
.venv-cuda/Scripts/python.exe -m cls_balance.report_incremental --training {args.training.as_posix()} --evaluation {args.evaluation.as_posix()} --output-root results/stage7 --preview-root .tmp/phase7-previews
.venv-cuda/Scripts/python.exe -m cls_balance.report_incremental --audit {args.report.as_posix()}
.venv-cuda/Scripts/python.exe -m cls_balance.report_incremental --rebuild {args.report.as_posix()} --rebuild-output .tmp/rebuilt-phase7
.venv-cuda/Scripts/python.exe scripts/audit_phase7_data.py --training {args.training.as_posix()} --output .tmp-phase7-data-audit.json
```

Use your reproduced immutable run IDs; never overwrite an existing result.
Phase 8 broader analysis, generic ImageNet-frozen and three-stage measurements
remain separate work.
"""
    with (output / "REPORT.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    write_json(output / "summary.json", {"status": "complete", "kind": "phase7-validation", "artifacts_sha256": inventory(output)})
    print(output)


if __name__ == "__main__":
    main()
