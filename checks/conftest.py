"""Reuse the frozen generated-data fixture without modifying its source."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("phase2_fixtures", Path(__file__).parents[1] / "tests/conftest.py")
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
tiny_data = fixtures.tiny_data
full_metadata = fixtures.full_metadata
