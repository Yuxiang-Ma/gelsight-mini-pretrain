"""phash/grey_center must stay bit-identical to the legacy implementation,
because the published dedupe decisions were made with it."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from gsmp import config, filters

LEGACY = config.repo_root() / "legacy" / "make_parquet_v2.py"


@pytest.fixture(scope="module")
def legacy_mod():
    if not LEGACY.exists():
        pytest.skip("legacy/make_parquet_v2.py not present")
    spec = importlib.util.spec_from_file_location("legacy_mpv2", LEGACY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["legacy_mpv2"] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:                      # noqa: BLE001
        pytest.skip(f"legacy module not importable: {exc}")
    return mod


def test_grey_center_matches_legacy(legacy_mod):
    rng = np.random.default_rng(7)
    img = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    np.testing.assert_array_equal(
        filters.grey_center(img), legacy_mod.grey_center(img)
    )


def test_phash_matches_legacy(legacy_mod):
    rng = np.random.default_rng(7)
    for _ in range(5):
        img = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
        assert filters.phash(img) == legacy_mod.phash(img)
