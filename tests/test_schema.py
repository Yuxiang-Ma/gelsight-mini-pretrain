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
     "tacquad", "unit", "fota_labeled", "fota_unlabeled"],
)
def test_conforming_sources_match_schema(source):
    """All 12 main-repo configs share the full 30-column schema.

    fota_labeled and fota_unlabeled joined this list on 2026-08-06, when they
    were widened from 26 columns and republished.
    """
    if not (config.PARQUET_MAIN / source).is_dir():
        pytest.skip(f"{source} not present on this machine")
    assert schema.published_columns(source) == list(schema.COLUMNS)


@pytest.mark.parametrize("source", ["fota_labeled", "fota_unlabeled"])
def test_fota_schema_fix_held(source):
    """The 26-column defect is fixed; this guards against regressing it.

    These two configs shipped missing episode, frame_idx, digit_class and
    gel_variant, which broke the dataset README's own concatenate_datasets
    example across configs (93,155 rows affected). They were widened and
    republished on 2026-08-06. The three columns that were never recorded
    remain present-but-null; gel_variant carries a real derived value.
    """
    if not (config.PARQUET_MAIN / source).is_dir():
        pytest.skip(f"{source} not present on this machine")
    cols = schema.published_columns(source)
    assert cols == list(schema.COLUMNS)
    assert source not in schema.LEGACY_26_SOURCES
