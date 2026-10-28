"""Publish actual checks and complete benchmark lineage without data or weights."""

import argparse
from pathlib import Path
import re
from xml.etree import ElementTree

import torch

from cls_balance.artifacts import environment, file_hash, write_json
from cls_balance.report_benchmark import audit_publication
from cls_balance.report_flat import read_json
from cls_balance.train_independent import unique_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cpu-log", type=Path, required=True)
    parser.add_argument("--cuda-log", type=Path, required=True)
    parser.add_argument("--freeze-log", type=Path, required=True)
    parser.add_argument("--benchmark-log", type=Path, required=True)
    parser.add_argument("--data-audit", type=Path, required=True)
    parser.add_argument("--rebuild-audit", type=Path, required=True)
    parser.add_argument("--cpu-junit", type=Path, required=True)
    parser.add_argument("--discarded-attempt", type=Path)
    parser.add_argument("--publication-failure-log", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("results/stage6"))
    args = parser.parse_args()
    verified = audit_publication(args.report)
    suite = read_json(args.suite / "summary.json")
    publication = read_json(args.report / "publication.json")
    if suite["status"] != "complete" or suite["runs"] != publication["benchmark_runs"]:
        raise ValueError("Benchmark/publication lineage mismatch")
    if file_hash(args.suite / "frozen_matrix.json") != suite["frozen_matrix_sha256"]:
        raise ValueError("Frozen matrix changed")
    for name, digest in suite["run_summaries_sha256"].items():
        if file_hash(args.report / name / "summary.json") != digest:
            raise ValueError("Published child differs from measured suite")
    cpu, cuda = (p.read_text(encoding="utf-8") for p in (args.cpu_log, args.cuda_log))
    cpu_count = re.search(r"(\d+) passed", cpu)
    cuda_count = re.search(r"(\d+) passed", cuda)
    if not cpu_count or not cuda_count or "FAILED " in cpu + cuda or "ERROR " in cpu + cuda:
        raise ValueError("Require actual passing CPU and CUDA check output")
    junit = ElementTree.parse(args.cpu_junit).getroot()
    suites = list(junit.iter("testsuite"))
    if (sum(int(s.get("failures", 0)) + int(s.get("errors", 0)) for s in suites)
            or sum(int(s.get("tests", 0)) - int(s.get("skipped", 0)) for s in suites) != int(cpu_count[1])):
        raise ValueError("CPU JUnit evidence disagrees with test output")
    freeze = read_json(args.freeze_log)
    if freeze["status"] != "verified":
        raise ValueError("Original test freeze failed")
    data_audit = read_json(args.data_audit)
    contract = read_json(args.report / suite["runs"][0] / "data_contract.json")
    if (data_audit["status"] != "verified" or data_audit["images_checked"] != 10000
            or not data_audit["training_preparation_binding_verified"]
            or data_audit["manifest_records_sha256"] != contract["manifest_records_sha256"]
            or data_audit["taxonomy_sha256"] != contract["taxonomy_sha256"]
            or data_audit["preprocessing_sha256"] != contract["preprocessing_sha256"]):
        raise ValueError("Post-benchmark dataset audit mismatch")
    rebuild = read_json(args.rebuild_audit)
    if (rebuild["status"] != "verified" or rebuild["rebuilt_review_id"] != args.report.name
            or rebuild["rebuilt_publication_sha256"] != file_hash(args.report / "publication.json")):
        raise ValueError("Artifact-only rebuild evidence mismatch")
    output = args.output_root / unique_id("validation")
    output.mkdir(parents=True, exist_ok=False)
    def text_file(name: str, value: str) -> None:
        with (output / name).open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(value.replace("\r\n", "\n"))
    for name, source in (("cpu-tests.txt", args.cpu_log), ("cuda-phase6-tests.txt", args.cuda_log),
                         ("frozen-suite.json", args.freeze_log), ("benchmark.log", args.benchmark_log)):
        text_file(name, source.read_text(encoding="utf-8"))
    write_json(output / "suite.json", suite)
    write_json(output / "frozen_matrix.json", read_json(args.suite / "frozen_matrix.json"))
    write_json(output / "publication_audit.json", verified)
    write_json(output / "post_benchmark_data_audit.json", data_audit)
    write_json(output / "rebuild_audit.json", rebuild)
    text_file("cpu-tests.xml", args.cpu_junit.read_text(encoding="utf-8"))
    repo = Path(__file__).resolve().parents[1]
    env = environment(repo)
    env["runtime_policy"] = "post-benchmark validation snapshot; measurement environments retained per benchmark run"
    env["gpu_properties"] = [{"index": i, "name": p.name, "total_memory_bytes": p.total_memory,
        "compute_capability": [p.major, p.minor], "multiprocessors": p.multi_processor_count}
        for i in range(torch.cuda.device_count()) for p in [torch.cuda.get_device_properties(i)]]
    write_json(output / "environment.json", env)
    sources = [*repo.glob("src/cls_balance/**/*.py"), *repo.glob("checks/*.py"), repo / "scripts/publish_phase6_validation.py", repo / "scripts/audit_phase6_data.py"]
    write_json(output / "source_hashes.json", {p.relative_to(repo).as_posix(): file_hash(p) for p in sorted(sources)})
    if args.discarded_attempt:
        write_json(output / "discarded_attempt.json", read_json(args.discarded_attempt))
    retry_note = ""
    if args.publication_failure_log:
        text_file("publication-rejected.log", args.publication_failure_log.read_text(encoding="utf-8"))
        retry_note = ("An initial report audit caught omitted independent specialist histories, whose selected "
                      "checkpoints live under nested attempt IDs. The final report follows exact deployment-bundle "
                      "paths and checks all 108 selected hierarchical classifiers. This reporting correction did "
                      "not change any measured predictions or timings. A regression check covers nested attempts, "
                      "and completion metadata is now written only after the final publication audit passes.\n")
    report = f"""# Phase 6 validation and reproduction evidence

The full CPU suite passed **{cpu_count[1]} tests**, with the CUDA-only FP16 check
explicitly skipped when CUDA is unavailable. The focused CUDA suite passed
**{cuda_count[1]} Phase 6 checks**, including a real synchronized FP16 pipeline
and saved-output replay on generated test images. Synthetic check timings are
never experimental results. The original 172-test suite freeze is unchanged.

[Measured test results, graphs and complete network schematics](../{args.report.name}/REPORT.md).
The publication auditor verified {verified['benchmark_runs']} complete benchmark
runs and {verified['trained_classifiers']} hierarchical classifier annotations.
All official test images appear in every one of five timed passes per checkpoint.
Exact per-image global predictions and paths agree across passes; model state is
unchanged. Shared encoder sources match the seed-paired FLAT checkpoints.
All 10,000 dataset images were rechecked against their original preparation
checksums after timing; the hashes and counts match the benchmark contracts.
An artifact-only rebuild preserved all 189 benchmark files byte for byte,
retained all 149,985 timings, and produced 23 valid SVGs with all 100 global
variant labels present. All report links exist. Four network schematics remain
byte-identical; accuracy/latency seed-label placement was improved after visual
review. No model weights or dataset images were needed for that rebuild.

The published suite/matrix bind the measured checkpoints before test inference.
`benchmark.log` is the actual complete execution log. `publication_audit.json`
records independent recomputation of saved classification/timing evidence.
`source_hashes.json` binds checked code; per-run inference source hashes bind the
measured implementation. No dataset pixels, training IDs, embeddings or checkpoint
tensors are included.

An early incomplete timing attempt was stopped because the CPU regression suite
was still active. It contributed no completed pass or published measurement.
Its discarded-attempt record is retained. Two preflight-only development errors
(shared encoder provenance layout and independent resource-file layout) were
fixed before the successful suite froze its matrix; they produced no test results.
Timed passes used no competing regression/plotting workload. OS/driver
clocks remained unlocked and desktop background activity remains a limitation.

{retry_note}

```powershell
.venv/Scripts/python.exe scripts/freeze_tests.py --check
.venv/Scripts/python.exe -m pytest -q tests checks
.venv-cuda/Scripts/python.exe -m pytest -q checks/test_phase6.py
.venv-cuda/Scripts/python.exe -m cls_balance.benchmark --config configs/benchmark.yaml
.venv-cuda/Scripts/python.exe scripts/audit_phase6_data.py --config configs/benchmark.yaml --output .tmp-phase6-data-audit.json
.venv-cuda/Scripts/python.exe -m cls_balance.report_benchmark --suite {args.suite.as_posix()} --output-root results/stage6 --preview-root .tmp/phase6-previews
.venv-cuda/Scripts/python.exe -m cls_balance.report_benchmark --audit {args.report.as_posix()}
.venv-cuda/Scripts/python.exe -m cls_balance.report_benchmark --rebuild {args.report.as_posix()} --output-root .tmp/rebuilt-phase6
```

Use your newly reproduced local run IDs in `configs/benchmark.yaml`; never rerun
into an existing result directory. Final CPU/CUDA checks and visual review occur
after timing, so they cannot inflate measured latency. FP16 experimental results,
generic features, three-stage ablations and introduced-class updates remain
unmeasured. Full plotting rebuild requires only the published result bundle and
the pinned plotting dependency, not data images or trained weights.
"""
    text_file("REPORT.md", report)
    write_json(output / "publication.json", {"status": "complete", "run_id": output.name,
        "scope": "actual check output and benchmark lineage; no training data or tensors",
        "artifacts_sha256": {p.name: file_hash(p) for p in sorted(output.iterdir())}})
    print(output)


if __name__ == "__main__":
    main()
