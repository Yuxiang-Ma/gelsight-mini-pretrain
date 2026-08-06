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
    runs = [_keys(1000), _keys(1000, 5), _keys(1000, 9)]  # self symdiffs 8-18
    published = _keys(1000, 4)                            # gap 2, below floor 8
    e = evaluate_envelope("demo", runs, published)
    assert e.published_symdiff <= max(e.self_symdiffs)
    assert e.consistent


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


def test_limited_runs_need_capture_scoping_to_be_meaningful():
    """Without scoping, a --limit run looks catastrophically wrong."""
    runs = [{("a", i) for i in range(100)}, {("a", i) for i in range(100)}]
    published = {("a", i) for i in range(100)} | {("b", i) for i in range(5000)}

    unscoped = evaluate_envelope("demo", runs, published)
    assert not unscoped.consistent  # 5000 unvisited keys read as missing

    scoped = evaluate_envelope("demo", runs, published,
                               restrict_to_covered_captures=True)
    assert scoped.consistent
    assert scoped.capture_coverage == pytest.approx(0.5)


def test_capture_coverage_exposes_a_port_that_drops_captures():
    """Scoping hides dropped captures, so coverage must surface them."""
    runs = [{("a", 1)}, {("a", 1)}]
    published = {("a", 1), ("b", 1), ("c", 1), ("d", 1)}
    e = evaluate_envelope("demo", runs, published,
                          restrict_to_covered_captures=True)
    assert e.consistent                       # within the scoped view
    assert e.capture_coverage == pytest.approx(0.25)  # but 3 of 4 never visited


def test_gap_below_own_minimum_spread_is_reported_as_strong():
    """A wide self-spread makes '<= max' trivially satisfiable, so the verdict
    must distinguish 'below our own floor' from 'barely inside our ceiling'."""
    runs = [_keys(1000), _keys(1000, 400), _keys(1000, 800)]  # wide spread
    published = _keys(1000, 1)  # very close to run 1
    e = evaluate_envelope("demo", runs, published)
    assert e.consistent
    assert e.verdict.startswith("strong")
    assert e.published_symdiff < min(e.self_symdiffs)


def test_gap_inside_spread_is_flagged_as_a_weak_bar():
    # runs at offsets 0/400/800 -> self symdiffs 800, 1600, 800 (floor 800).
    # published at 1300 is 1000 from run@800: inside [800, 1600], not below it.
    runs = [_keys(1000), _keys(1000, 400), _keys(1000, 800)]
    published = _keys(1000, 1300)
    e = evaluate_envelope("demo", runs, published)
    assert min(e.self_symdiffs) <= e.published_symdiff <= max(e.self_symdiffs)
    assert e.consistent
    assert "weak bar" in e.verdict
