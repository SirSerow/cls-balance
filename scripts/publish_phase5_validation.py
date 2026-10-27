"""Publish immutable check logs and audit links without training data/weights."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys

from cls_balance.artifacts import environment, file_hash, write_json
from cls_balance.report_flat import read_json
from cls_balance.report_shared import audit_training_run
from cls_balance.train_independent import unique_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("results/stage5"))
    args = parser.parse_args()
    summaries = [audit_training_run(p) for p in args.runs]
    publication = read_json(args.report / "publication.json")
    if publication["source_run_ids"] != [s["run_id"] for s in summaries]:
        raise ValueError("Report source runs differ")
    for name, digest in publication["artifacts_sha256"].items():
        target = (args.report / name).resolve()
        if not target.is_relative_to(args.report.resolve()) or file_hash(target) != digest:
            raise ValueError("Publication checksum mismatch")
    output = args.output_root / unique_id("validation")
    output.mkdir(parents=True, exist_ok=False)
    logs = {"cpu-tests.txt": ".tmp/phase5-cpu-tests.log", "phase5-regression.txt": ".tmp/phase5-regression.log",
            "publication-checks.txt": ".tmp/phase5-publication-tests.log",
            "cuda-checks.txt": ".tmp/phase5-cuda-checks.log", "training-log.txt": ".tmp/phase5-training.log"}
    counts = {}
    for name, source in logs.items():
        text = Path(source).read_text(encoding="utf-8-sig")
        if name != "training-log.txt":
            match = re.search(r"(\d+) passed(?:, \d+ deselected)? in", text)
            if not match or re.search(r"\d+ (failed|error)", text):
                raise ValueError(f"No successful test summary in {source}")
            counts[name] = int(match[1])
        # Normalize line endings for Git-stable hashes.
        (output / name).write_text(text, encoding="utf-8", newline="\n")
    freeze = subprocess.run([sys.executable, "scripts/freeze_tests.py", "--check"], capture_output=True, text=True, check=True)
    (output / "frozen-suite.txt").write_text(freeze.stdout, encoding="utf-8", newline="\n")
    repo = Path(__file__).resolve().parents[1]
    write_json(output / "environment.json", environment(repo))
    write_json(output / "audit.json", {"test_counts": counts, "source_run_ids": [s["run_id"] for s in summaries],
        "report": args.report.as_posix(), "publication_sha256": file_hash(args.report / "publication.json"),
        "frozen_test_manifest_sha256": file_hash(repo / "tests/FROZEN.json"),
        "all_runs_audited": True, "publication_hashes_verified": True,
        "test_evaluated": False, "failures_or_discarded_training_attempts": [],
        "source_sha256": {p.relative_to(repo).as_posix(): file_hash(p) for folder in ("src", "checks", "scripts", "configs")
                          for p in sorted((repo / folder).rglob("*")) if p.is_file() and p.suffix in {".py", ".yaml"}}})
    reproduction = f"""# Phase 5 validation evidence

The original frozen suite is unchanged. CPU regression output: **{counts['cpu-tests.txt']} passing tests**.
After publication/config-audit refinements, the current Phase 5 suite passed **{counts['phase5-regression.txt']} checks**.
The CUDA environment also passed **{counts['cuda-checks.txt']} focused checks**. These synthetic checks are implementation evidence, not aircraft accuracy measurements.
The final report refinements passed **{counts['publication-checks.txt']} end-to-end publication checks**, covering both backbone types and depths.

All three real FT-FROZEN-2 runs passed nested checkpoint, local selection/exposure,
model-fingerprint and complete predicted/oracle validation audits. Their publication
hashes were independently rechecked. Each run used all official training/validation
images; there was no test inference. No discarded training attempts were observed
in this execution. The process-lifetime Windows idle-sleep guard was spawn-tested;
it did not change power plans or GPU clocks.

Measured report: [shared-head results](../{args.report.name}/REPORT.md).
`audit.json` binds run IDs, the report, frozen suite and current source/config hashes.
The report's per-seed records preserve the exact training source hashes and Git
SHA/dirty state; this validation environment describes the later audit.

Commands executed from the repository root:

```powershell
.venv/Scripts/python.exe scripts/freeze_tests.py --check
.venv/Scripts/python.exe -m pytest -q tests checks
.venv/Scripts/python.exe -m pytest -q checks/test_phase5.py
.venv/Scripts/python.exe -m pytest -q checks/test_phase5.py -k train_reload_route_replace_and_publish
.venv-cuda/Scripts/python.exe -m pytest -q checks/test_phase5.py -k 'head_checkpoint or training_cache or frozen_guard or source_encoder or head_dimensions'
.venv-cuda/Scripts/python.exe scripts/train_shared_awake.py --smoke-test
.venv-cuda/Scripts/python.exe -m pip check
.venv-cuda/Scripts/python.exe scripts/train_shared_awake.py --config configs/shared.yaml
.venv-cuda/Scripts/python.exe -m cls_balance.report_shared --runs {' '.join(p.as_posix() for p in args.runs)} --output-root results/stage5 --flat-runs {' '.join(Path(p).as_posix() for p in read_json(args.runs[0] / 'config.resolved.json')['flat_runs'].values())} --independent-runs runs/independent/ind-s42-20261026T125759Z-6d87251e runs/independent/ind-s43-20261026T141824Z-b8c12b2c runs/independent/ind-s44-20261026T153237Z-2b462c75 --preview-root .tmp/phase5-figures
```

Images, archives, embeddings, training-ID lists and weights remain local and ignored.
Only validation predictions and aggregate metrics/curves are published. Generic
ImageNet-frozen and three-stage measurements retain the execution-order gate;
Phase 6 test accuracy, full-pipeline latency/FPS and Phase 7 introduction remain deferred.
"""
    (output / "REPORT.md").write_text(reproduction, encoding="utf-8", newline="\n")
    write_json(output / "publication.json", {"artifacts_sha256": {p.name: file_hash(p) for p in output.iterdir() if p.is_file()}})
    print(output, flush=True)


if __name__ == "__main__":
    main()
