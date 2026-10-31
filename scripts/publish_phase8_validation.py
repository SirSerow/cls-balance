"""Publish genuine Phase 8 regression logs and artifact-only rebuild evidence."""

import argparse
from pathlib import Path
import re

from cls_balance.artifacts import environment, file_hash, write_json
from cls_balance.incremental import inventory
from cls_balance.report_analysis import rebuild
from cls_balance.train_independent import unique_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cpu-log", type=Path, required=True)
    parser.add_argument("--focused-log", type=Path, required=True)
    parser.add_argument("--final-log", type=Path, required=True)
    parser.add_argument("--freeze-log", type=Path, required=True)
    parser.add_argument("--rebuild-output", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("results/stage8"))
    args = parser.parse_args()
    logs, counts = {}, {}
    for name in ("cpu", "focused", "final"):
        logs[name] = getattr(args, f"{name}_log").read_text(encoding="utf-8")
        match = re.search(r"(\d+) passed", logs[name])
        if match is None or re.search(r"\b(?:FAILED|ERROR|failed|errors)\b", logs[name]):
            raise ValueError("Require passing complete CPU and affected-module/plot check logs")
        counts[name] = int(match[1])
    freeze = args.freeze_log.read_text(encoding="utf-8")
    if '"status": "verified"' not in freeze and "'status': 'verified'" not in freeze:
        raise ValueError("Require the unchanged frozen-suite verification")
    repo = Path(__file__).resolve().parents[1]
    # Rebuild includes both source publication audits and semantic recomputation.
    # It verifies every regenerated report/table/figure byte against the source.
    audit = rebuild(repo, args.report, args.rebuild_output)
    output = args.output_root / unique_id("validation")
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("cpu-tests.txt", logs["cpu"]), ("focused-tests.txt", logs["focused"]),
                        ("final-statistics-tests.txt", logs["final"]), ("frozen-suite.txt", freeze)):
        with (output / name).open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
    write_json(output / "rebuild-audit.json", audit)
    write_json(output / "environment.json", environment(repo))
    sources = sorted({*repo.glob("src/cls_balance/**/*.py"), *repo.glob("checks/*.py"),
                      repo / "scripts/publish_phase8_validation.py"})
    write_json(output / "source_hashes.json", {p.relative_to(repo).as_posix(): file_hash(p) for p in sources})
    text = f"""# Phase 8 validation and reproduction evidence

The complete CPU regression suite passed **{counts['cpu']} tests**; explicit
skips and reasons are preserved in [the log](cpu-tests.txt). A final run of the
affected benchmark, incremental-publication and Phase 8 statistics/figure checks
passed **{counts['focused']} checks** in the CUDA/plotting environment, including
the CUDA-only FP16 benchmark integration check. These generated fixtures test
software contracts; they are not experimental performance measurements.
After the final rebuild API refinement, all **{counts['final']} Phase 8 checks**
also passed again; [the final log](final-statistics-tests.txt) is retained.

[Measured consolidated results](../{args.report.name}/REPORT.md).

The original 172-test suite remains unchanged. Phase 8 consumes the existing
three-seed Phase 6/7 measurements without training, image loading or inference.
The source audits replay all primary test predictions/timings and all incremental
before/after predicted/oracle evaluations. They enforce frozen source manifests,
official class/image counts, matching seed/image identities and measurement
contracts. The final analysis recomputes paired stratified intervals, tables and
report text, and rejects changed statistics even with re-sealed checksums.

All {audit['artifacts']} final report artifacts rebuilt byte for byte from the
committed source evidence without data or weights. [Rebuild evidence](rebuild-audit.json)
binds the exact final source publication. All five new SVGs were rendered to
local PNG previews and visually reviewed. Original Phase 3–7 publications remain
unchanged. No dataset pixels, training-ID lists, embeddings or tensors were added.

```powershell
.venv/Scripts/python.exe -m pytest -q tests checks
.venv/Scripts/python.exe scripts/freeze_tests.py --check
.venv-cuda/Scripts/python.exe -m pytest -q checks/test_phase6.py checks/test_phase7_report.py checks/test_phase8.py
.venv-cuda/Scripts/python.exe -m cls_balance.report_analysis --config configs/analysis.yaml --preview-root .tmp/phase8-previews --results-doc docs/RESULTS.md
.venv-cuda/Scripts/python.exe -m cls_balance.report_analysis --audit {args.report.as_posix()}
.venv-cuda/Scripts/python.exe -m cls_balance.report_analysis --rebuild {args.report.as_posix()} --output .tmp/rebuilt-phase8
```

Use a fresh output directory for every rebuild. Source hashes and environment
are retained alongside the real test logs. Analysis source may have a dirty-tree
flag because this publication was produced before the final result commit; the
recorded source SHA256 values identify the exact files. The bootstrap holds the
trained model pairs fixed and does not estimate future-training-seed uncertainty.
Unmeasured generic frozen, three-stage, FP16 experimental and cold-start results
remain clearly labeled follow-up work.
"""
    with (output / "REPORT.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    write_json(output / "summary.json", {"status": "complete", "kind": "phase8-validation", "artifacts_sha256": inventory(output)})
    print(output)


if __name__ == "__main__":
    main()
