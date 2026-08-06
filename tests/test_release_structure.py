"""The declared structure must match the actual release.

The dataset README's YAML front-matter is what HuggingFace uses to build the
config/split index. It is a *claim* about the data, and this project has
already shipped one claim that was false for months: the README advertised a
uniform 30-column schema while two configs carried 26, which broke its own
`concatenate_datasets` quick-start.

These tests check the claim against the parquet, so the next divergence
surfaces as a red test instead of a user's traceback.
"""
from __future__ import annotations

import collections
import glob
import re

import pyarrow.parquet as pq
import pytest

from gsmp import config, schema

_README = config.repo_root() / "docs" / "_readme_new.md"


def _declared():
    """{config_name: {split, ...}} parsed from the README front-matter."""
    fm = _README.read_text().split("---")[1]
    out, cur = {}, None
    for line in fm.splitlines():
        m = re.match(r"\s*- config_name:\s*(\S+)", line)
        if m:
            cur = m.group(1)
            out[cur] = set()
            continue
        m = re.match(r"\s*- split:\s*(\S+)", line)
        if m and cur:
            out[cur].add(m.group(1))
    return out


def _actual_splits(cfg):
    counts = collections.Counter()
    for f in sorted(glob.glob(str(config.PARQUET_MAIN / cfg / "*.parquet"))):
        counts.update(pq.read_table(f, columns=["split"]).column("split").to_pylist())
    return set(counts)


DECLARED = _declared()


def test_readme_declares_every_config_on_disk():
    on_disk = {
        p.name for p in config.PARQUET_MAIN.iterdir()
        if p.is_dir() and not p.name.startswith((".", "assets"))
    }
    assert on_disk - set(DECLARED) == set(), "config on disk is undeclared"


@pytest.mark.parametrize("cfg", sorted(DECLARED))
def test_declared_config_exists_with_matching_splits(cfg):
    d = config.PARQUET_MAIN / cfg
    if not d.is_dir():
        pytest.fail(f"README declares {cfg} but no directory exists")
    assert _actual_splits(cfg) == DECLARED[cfg]


@pytest.mark.parametrize("cfg", sorted(DECLARED))
def test_every_declared_config_shares_the_schema(cfg):
    """The claim the README makes in prose: one schema, every row identical."""
    if not (config.PARQUET_MAIN / cfg).is_dir():
        pytest.skip(f"{cfg} not present")
    assert schema.published_columns(cfg) == list(schema.COLUMNS)


def test_split_column_values_are_the_declared_split_names():
    """A row's `split` value must equal the split it is filed under.

    Three configs use domain-specific split names rather than train/val/test
    -- tacquad (data_indoor/outdoor/fine), gelslam (train/recon), and sparsh
    in the NC repo (flat/sharp/sphere). Those names are DATA, not a naming
    slip: they carry the domain or indenter identity. Renaming them to
    train/val/test would destroy information and break parity with the
    release, so this test pins them instead of "fixing" them.
    """
    for cfg, declared in DECLARED.items():
        if not (config.PARQUET_MAIN / cfg).is_dir():
            continue
        assert _actual_splits(cfg) <= declared, (
            f"{cfg} has split values not declared in the README"
        )
