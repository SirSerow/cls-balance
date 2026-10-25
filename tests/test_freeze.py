"""The frozen suite must detect changed, added and missing test inputs."""

from pathlib import Path
import shutil
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/freeze_tests.py"


@pytest.fixture
def frozen_repo(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_example.py").write_text("def test_example():\n    assert True\n")
    (tmp_path / "scripts").mkdir()
    shutil.copyfile(SCRIPT, tmp_path / "scripts/freeze_tests.py")
    (tmp_path / ".github/workflows").mkdir(parents=True)
    (tmp_path / ".github/workflows/tests.yml").write_text("name: tests\n")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'example'\n")
    (tmp_path / "junit.xml").write_text('<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0"/></testsuites>')
    return tmp_path


def run(root, *args):
    return subprocess.run([sys.executable, str(SCRIPT), "--root", str(root), *args],
                          capture_output=True, text=True, check=False)


def test_freeze_requires_successful_validation_and_never_overwrites(frozen_repo):
    report = str(frozen_repo / "junit.xml")
    assert run(frozen_repo, "--write").returncode != 0
    assert run(frozen_repo, "--write", "--junit", report).returncode == 0
    assert run(frozen_repo, "--check").returncode == 0
    assert run(frozen_repo, "--write", "--junit", report).returncode != 0


@pytest.mark.parametrize("failure", ["change", "add", "delete", "runner", "report"])
def test_freeze_detects_tampering(frozen_repo, failure):
    assert run(frozen_repo, "--write", "--junit", str(frozen_repo / "junit.xml")).returncode == 0
    if failure == "change":
        (frozen_repo / "tests/test_example.py").write_text("def test_example():\n    pass\n")
    elif failure == "add":
        (frozen_repo / "tests/test_new.py").write_text("def test_new():\n    assert True\n")
    elif failure == "delete":
        (frozen_repo / "tests/test_example.py").unlink()
    elif failure == "runner":
        (frozen_repo / "pyproject.toml").write_text("changed config\n")
    else:
        (frozen_repo / "junit.xml").write_text('<testsuite tests="2" failures="0" errors="0" skipped="0"/>')
    assert run(frozen_repo, "--check").returncode != 0


@pytest.mark.parametrize("counts", ['tests="0"', 'tests="1" failures="1"', 'tests="1" errors="1"', 'tests="1" skipped="1"'])
def test_freeze_rejects_unsuccessful_validation(frozen_repo, counts):
    report = frozen_repo / "junit.xml"
    report.write_text(f"<testsuite {counts}/>")
    assert run(frozen_repo, "--write", "--junit", str(report)).returncode != 0
    assert not (frozen_repo / "tests/FROZEN.json").exists()
