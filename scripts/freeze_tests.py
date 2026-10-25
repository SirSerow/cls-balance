"""Freeze a validated test suite once, or verify its exact files and hashes.

The default is read-only checking. Creating a baseline requires a successful
JUnit report and --write. Existing baselines cannot be overwritten by this tool.
Intentional future test changes require a separately reviewed baseline update.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

RUNNER_FILES = ("pyproject.toml", ".github/workflows/tests.yml", "scripts/freeze_tests.py")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(root: Path) -> dict[str, str]:
    paths = [p for p in (root / "tests").rglob("*") if p.is_file()
             and not {"__pycache__", ".pytest_cache"}.intersection(p.parts) and p != root / "tests/FROZEN.json"]
    if not any(p.name.startswith("test_") and p.suffix == ".py" for p in paths):
        raise ValueError("No test suite found")
    paths.extend(root / name for name in RUNNER_FILES)
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(paths)}


def successful_junit(path: Path) -> dict:
    tree = ET.parse(path).getroot()
    suites = [tree] if tree.tag == "testsuite" else list(tree.iter("testsuite"))
    counts = {key: sum(int(s.get(key, "0")) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
    if counts["tests"] <= 0 or any(counts[k] for k in ("failures", "errors", "skipped")):
        raise ValueError("Freezing requires a nonempty JUnit report without failures, errors or skips")
    return {**counts, "junit_sha256": sha256(path)}


def freeze(root: Path, junit: Path) -> dict:
    report = successful_junit(junit)
    manifest = {"schema_version": 1,
                "policy": "Frozen after validation. Changes require explicit review and a new validated baseline.",
                "files": snapshot(root), "validation": {**report, "junit_path": junit.relative_to(root).as_posix()}}
    with (root / "tests/FROZEN.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return {"status": "frozen", "files": len(manifest["files"]), "validated_tests": report["tests"]}


def check(root: Path) -> dict:
    manifest = json.loads((root / "tests/FROZEN.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unknown test-freeze schema")
    actual = snapshot(root)
    expected = manifest["files"]
    changed = sorted(name for name in set(actual) | set(expected) if actual.get(name) != expected.get(name))
    if changed:
        raise ValueError("Frozen test inputs changed: " + ", ".join(changed))
    report_path = (root / manifest["validation"]["junit_path"]).resolve()
    if not report_path.is_relative_to(root):
        raise ValueError("Validation evidence must stay inside the repository")
    validated = successful_junit(report_path)
    if any(manifest["validation"].get(key) != value for key, value in validated.items()):
        raise ValueError("Frozen validation evidence changed")
    return {"status": "verified", "files": len(actual), "validated_tests": validated["tests"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Read-only integrity check (default)")
    mode.add_argument("--write", action="store_true", help="Create baseline once after successful validation")
    parser.add_argument("--junit", type=Path, help="Successful validation report required with --write")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    if args.write and args.junit is None:
        parser.error("--write requires --junit")
    root = args.root.resolve()
    result = freeze(root, args.junit.resolve()) if args.write else check(root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
