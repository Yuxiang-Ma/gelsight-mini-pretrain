from __future__ import annotations

import importlib

import pytest

TIER2 = ["feats", "fota_labeled", "fota_unlabeled"]


@pytest.mark.parametrize("name", TIER2)
def test_tier2_declares_spec_but_no_dry_run(name):
    """tier-2 sources are wrapped, not migrated.

    They must NOT expose dry_run_keys, because there is no join key to
    regression-test them with and a dry_run would imply a verified migration.
    """
    mod = importlib.import_module(f"gsmp.sources.{name}")
    assert hasattr(mod, "SPEC")
    assert not hasattr(mod, "dry_run_keys")
    assert "tier-2" in (mod.__doc__ or "")


@pytest.mark.parametrize("name", TIER2)
def test_tier2_spec_records_why(name):
    mod = importlib.import_module(f"gsmp.sources.{name}")
    assert mod.SPEC.notes.strip(), f"{name} must document why it is tier-2"
