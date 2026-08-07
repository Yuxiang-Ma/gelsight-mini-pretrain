"""The tier table in docs/source_tiers.md must match what the code measures.

That table is the deliverable's central honesty claim -- it says, per source,
how strong a parity proof is possible. It is hand-maintained prose, and prose
drifts: fixing the FoTA schema on 2026-08-06 silently made its "26 columns"
rows false, and nothing failed. The only reason it was caught is that someone
happened to re-read the document against the code.

This test removes that dependence on noticing.
"""
from __future__ import annotations

import re

import pytest

from gsmp import config, determinism, schema

_DOC = config.repo_root() / "docs" / "source_tiers.md"

#: Rows of the generated table: | source | repo | rows | keys | ... | tier |
_ROW = re.compile(
    r"^\|\s*`?(?P<source>[a-z_]+)`?\s*\|\s*(?P<repo>main|nc)\s*\|"
    r"\s*(?P<rows>[\d,]+)\s*\|",
    re.M,
)


def _documented_rows():
    """{source: (repo, n_rows)} from every table row that declares them."""
    out = {}
    for m in _ROW.finditer(_DOC.read_text()):
        rows = m.group("rows").replace(",", "")
        if rows.isdigit():
            out[m.group("source")] = (m.group("repo"), int(rows))
    return out


DOCUMENTED = _documented_rows()


def test_the_doc_actually_contains_a_parsable_table():
    """Guard the guard: a regex that matches nothing would pass vacuously."""
    assert len(DOCUMENTED) >= 10, (
        f"parsed only {len(DOCUMENTED)} rows from {_DOC.name}; the table "
        f"format changed and this test is no longer checking anything"
    )


@pytest.mark.parametrize("source", sorted(DOCUMENTED))
def test_documented_row_count_matches_the_release(source):
    repo, documented = DOCUMENTED[source]
    if not config.published_dir(source, repo).is_dir():
        pytest.skip(f"{source} not present on this machine")
    actual = schema.join_key_quality(source, repo).n_rows
    assert actual == documented, (
        f"{source}: doc says {documented} rows, release has {actual}"
    )


@pytest.mark.parametrize("source", sorted(DOCUMENTED))
def test_documented_column_count_matches_the_release(source):
    """The column count is what went stale when FoTA was widened."""
    repo, _ = DOCUMENTED[source]
    if not config.published_dir(source, repo).is_dir():
        pytest.skip(f"{source} not present on this machine")
    text = _DOC.read_text()
    row = next(
        line for line in text.splitlines()
        if re.match(rf"^\|\s*`?{source}`?\s*\|", line)
    )
    cells = [c.strip() for c in row.strip("|").split("|")]
    documented_cols = int(cells[2])
    actual_cols = len(schema.published_columns(source, repo))
    assert actual_cols == documented_cols, (
        f"{source}: doc says {documented_cols} columns, release has {actual_cols}"
    )


def test_hash_seeded_producers_listed_in_doc_match_the_code():
    text = _DOC.read_text()
    for source in determinism.HASH_SEEDED_PRODUCERS:
        assert source in text, (
            f"{source} is hash-seeded in code but absent from {_DOC.name}"
        )
