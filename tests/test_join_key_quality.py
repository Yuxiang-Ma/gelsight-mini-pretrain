"""Pin what a set-based regression can actually prove, per source.

`tools/regress.py` compares SETS of (capture, frame_idx). Where that key is
not unique, set equality does NOT establish row-level equality -- two rows
sharing a key are indistinguishable. These tests pin which sources are in
which category so a summary can never silently promote the weaker claim.

Measured against the published release on 2026-08-06.
"""
from __future__ import annotations

import pytest

from gsmp import config, schema

#: Key is unique -> set equality IS row-level equality.
UNIQUE_KEY = [
    ("gelslam", "main"),
    ("tactile_tracking", "main"),
    ("unit", "main"),
    ("threedcal", "main"),
    ("feelanyforce", "main"),
    ("real_tactile_mnist", "main"),
    ("sparsh", "nc"),
]

#: Key is NOT unique -> set equality is a strictly weaker claim.
#: (source, license_repo, n_rows, n_keys)
NON_UNIQUE_KEY = [
    ("tacquad", "main", 12195, 11887),
    ("sim_tactile_mnist", "main", 150601, 87913),
    ("sim_starstruck", "main", 166104, 31096),
]


def _skip_if_absent(source, repo):
    if not config.published_dir(source, repo).is_dir():
        pytest.skip(f"{source} not present on this machine")


@pytest.mark.parametrize("source,repo", UNIQUE_KEY)
def test_unique_key_sources_support_row_level_claims(source, repo):
    _skip_if_absent(source, repo)
    q = schema.join_key_quality(source, repo)
    assert q.present
    assert q.unique, f"{source} key stopped being unique: {q.proof_strength}"
    assert q.n_rows == q.n_keys
    assert "row-level" in q.proof_strength


@pytest.mark.parametrize("source,repo,n_rows,n_keys", NON_UNIQUE_KEY)
def test_non_unique_key_sources_are_pinned_as_weaker(source, repo, n_rows, n_keys):
    """REGRESSION GUARD, not an aspiration.

    These three cannot support a row-level parity claim from set equality
    alone. The test pins the exact collision counts so that a change in the
    release -- or in how keys are derived -- surfaces instead of silently
    strengthening or weakening what the deliverable claims.
    """
    _skip_if_absent(source, repo)
    q = schema.join_key_quality(source, repo)
    assert q.present
    assert not q.unique
    assert (q.n_rows, q.n_keys) == (n_rows, n_keys)
    assert "does NOT" in q.proof_strength


def test_has_join_key_only_reports_presence():
    """has_join_key() must not be mistaken for a proof-strength check."""
    _skip_if_absent("sim_starstruck", "main")
    assert schema.has_join_key("sim_starstruck") is True
    assert schema.join_key_quality("sim_starstruck").unique is False


@pytest.mark.parametrize("source", ["fota_labeled", "fota_unlabeled"])
def test_fota_has_the_columns_but_still_no_usable_key(source):
    """Widening the schema did NOT make these regression-testable.

    The 2026-08-06 backfill gave both configs all 30 columns, so
    `frame_idx` now EXISTS -- but it is null for every row, because the value
    was never recorded upstream and is not recoverable. A present-but-null key
    identifies nothing, so these two stay tier-2: moved verbatim, no parity
    proof attempted. This test exists because "the column is there now" is
    exactly the kind of surface change that could be mistaken for a promotion.
    """
    _skip_if_absent(source, "main")
    assert "frame_idx" in schema.published_columns(source)
    q = schema.join_key_quality(source, "main")
    assert not q.present
    assert q.n_rows > 0                     # rows are counted even without a key
    assert q.proof_strength.startswith("none")
