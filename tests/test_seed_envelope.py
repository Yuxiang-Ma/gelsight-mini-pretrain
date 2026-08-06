"""Verification for producers whose RNG seed is unrecoverable.

Three legacy producers seed with `random.Random(hash(<str>) & 0xFFFFFFFF)`
(ingest_sparsh.py:78, ingest_tacquad_full.py:88, extract_rtm_video.py:123).
Python 3 randomises `hash(str)` per process, so the published artifact encodes
one per-process seed that was never recorded. Exact parity is impossible in
principle -- re-running the ORIGINAL script would not reproduce it either.

"Cannot prove exact" is not the same as "cannot verify". A faithful port
should differ from the release by about as much as it differs from ITSELF
under a different seed. If the published set sits inside our own run-to-run
spread, that is evidence of behavioural equivalence under seed uncertainty;
if it sits far outside, there is a real difference beyond the seed.
"""
from __future__ import annotations

import pytest

from gsmp.determinism import evaluate_envelope


def _keys(n, offset=0):
    return {("c", i + offset) for i in range(n)}


def test_published_inside_self_spread_is_consistent():
    runs = [_keys(1000), _keys(1000, 5), _keys(1000, 9)]  # self symdiffs ~10-18
    published = _keys(1000, 4)
    e = evaluate_envelope("demo", runs, published)
    assert e.published_symdiff <= max(e.self_symdiffs)
    assert e.consistent
    assert "consistent with seed uncertainty" in e.verdict


def test_published_far_outside_self_spread_is_a_real_difference():
    runs = [_keys(1000), _keys(1000, 1), _keys(1000, 2)]  # tight self spread
    published = _keys(1000, 800)  # massively different
    e = evaluate_envelope("demo", runs, published)
    assert not e.consistent
    assert "exceeds" in e.verdict


def test_identical_runs_still_report_a_usable_envelope():
    """A producer that turns out deterministic must not divide by zero."""
    runs = [_keys(100), _keys(100), _keys(100)]
    e = evaluate_envelope("demo", runs, _keys(100))
    assert e.self_symdiffs == [0, 0, 0]
    assert e.published_symdiff == 0
    assert e.consistent


def test_deterministic_runs_with_differing_published_is_not_consistent():
    runs = [_keys(100), _keys(100), _keys(100)]
    e = evaluate_envelope("demo", runs, _keys(100, 50))
    assert e.published_symdiff > 0
    assert not e.consistent


def test_requires_at_least_two_runs():
    with pytest.raises(ValueError, match="at least 2"):
        evaluate_envelope("demo", [_keys(10)], _keys(10))


def test_reports_row_counts_alongside_symdiff():
    runs = [_keys(1000), _keys(1000, 5)]
    e = evaluate_envelope("demo", runs, _keys(1000, 4))
    assert e.n_published == 1000
    assert e.mean_produced == pytest.approx(1000.0)
