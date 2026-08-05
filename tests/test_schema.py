from __future__ import annotations

import pytest

from gsmp import config, schema


def test_schema_has_30_columns():
    assert len(schema.SCHEMA) == 30
    assert schema.COLUMNS[0] == "image"
    assert "domain" in schema.COLUMNS
    assert "gel_variant" in schema.COLUMNS


@pytest.mark.parametrize(
    "source",
    ["gelslam", "tactile_tracking", "feats", "feelanyforce", "threedcal",
     "real_tactile_mnist", "sim_starstruck", "sim_tactile_mnist",
     "tacquad", "unit"],
)
def test_conforming_sources_match_schema(source):
    """These 10 sources were written with the full 30-column schema."""
    if not (config.PARQUET_MAIN / source).is_dir():
        pytest.skip(f"{source} not present on this machine")
    assert schema.published_columns(source) == list(schema.COLUMNS)


@pytest.mark.parametrize("source", ["fota_labeled", "fota_unlabeled"])
def test_fota_is_known_to_deviate(source):
    """REGRESSION GUARD, not an aspiration.

    fota_labeled and fota_unlabeled were published with 26 columns, missing
    episode / frame_idx / digit_class / gel_variant. This contradicts both
    PIPELINE.md and the user-facing README, and breaks the README's own
    concatenate_datasets quick-start example.

    This test pins the defect so it cannot silently change. When the data is
    eventually republished with the full schema, this test should start
    failing -- at which point move `source` out of LEGACY_26_SOURCES.
    """
    if not (config.PARQUET_MAIN / source).is_dir():
        pytest.skip(f"{source} not present on this machine")
    cols = schema.published_columns(source)
    assert source in schema.LEGACY_26_SOURCES
    assert len(cols) == 26
    assert set(schema.COLUMNS) - set(cols) == {
        "episode", "frame_idx", "digit_class", "gel_variant",
    }
