from __future__ import annotations

from gsmp.regress import compare


def test_identical_sets_pass():
    keys = {("c0", i) for i in range(100)}
    rep = compare(keys, keys, bg_keep_rate=0.015, n_candidates=1000)
    assert rep.deterministic_ok
    assert rep.bg_within_tolerance
    assert not rep.missing and not rep.extra


def test_small_symmetric_difference_within_bg_budget_passes():
    published = {("c0", i) for i in range(1000)}
    produced = set(published)
    produced.discard(("c0", 0))
    produced.add(("c0", 5000))
    rep = compare(published, produced, bg_keep_rate=0.015, n_candidates=100_000)
    assert rep.bg_within_tolerance


def test_large_difference_fails():
    published = {("c0", i) for i in range(1000)}
    produced = {("c0", i) for i in range(500)}
    rep = compare(published, produced, bg_keep_rate=0.015, n_candidates=1000)
    assert not rep.deterministic_ok
    assert len(rep.missing) == 500


def test_report_summary_is_human_readable():
    keys = {("c0", 1)}
    rep = compare(keys, keys, bg_keep_rate=0.015, n_candidates=10)
    assert "PASS" in rep.summary
