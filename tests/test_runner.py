from __future__ import annotations

import numpy as np

from gsmp.baseline import FirstNFrames, NoBaseline
from gsmp.runner import FrameRecord, run
from gsmp.spec import SourceSpec


def _spec(**kw):
    base = dict(name="demo", domain="real", gel_variant="markerless",
                license_repo="main", baseline=FirstNFrames(10), i_min=10.0,
                phash_dist=None, bg_keep_rate=0.0)
    base.update(kw)
    return SourceSpec(**base)


def _unit(values, capture="c0"):
    """One frame per value. runner.BASE_FRAMES (=10) of them are consumed."""
    return capture, [
        FrameRecord(rgb=np.full((80, 120, 3), v, dtype=np.uint8),
                    capture=capture, frame_idx=i)
        for i, v in enumerate(values)
    ]


def test_first_ten_frames_are_consumed_as_baseline_and_never_emitted():
    """Legacy's fingerprint: every published GelSLAM capture starts at
    frame_idx 10, because the head of each capture builds the baseline."""
    unit = _unit([10] * 10 + [200, 200])
    res = run(_spec(), [unit], dry_run=True)
    assert res.kept_keys == {("c0", 10), ("c0", 11)}
    assert all(idx >= 10 for _, idx in res.kept_keys)


def test_keeps_only_frames_passing_the_filter():
    unit = _unit([10] * 10 + [200, 10, 200])
    res = run(_spec(), [unit], dry_run=True)
    assert res.kept_keys == {("c0", 10), ("c0", 12)}


def test_background_quota_is_deterministic_not_random():
    """bg_keep_rate is a running ratio cap, not a probability. Two identical
    runs must produce identical sets, and the count must obey the quota."""
    unit = _unit([10] * 10 + [10] * 100)
    a = run(_spec(bg_keep_rate=0.015), [unit], dry_run=True)
    b = run(_spec(bg_keep_rate=0.015), [unit], dry_run=True)
    assert a.kept_keys == b.kept_keys
    # With no passing frames, the quota admits at most one empty frame
    # (n_empty_kept >= 0.015 * max(n_kept,1) blocks the rest).
    assert a.n_empty_kept == len(a.kept_keys)
    assert len(a.kept_keys) <= 2


def test_bg_keep_rate_zero_keeps_nothing():
    res = run(_spec(bg_keep_rate=0.0), [_unit([10] * 14)], dry_run=True)
    assert res.kept_keys == set()
    assert res.n_empty_kept == 0


def test_no_baseline_source_keeps_every_frame_including_the_head():
    """NoBaseline sources consume no head frames, so frame 0 survives."""
    res = run(_spec(baseline=NoBaseline()), [_unit([1, 2, 3])], dry_run=True)
    assert res.kept_keys == {("c0", 0), ("c0", 1), ("c0", 2)}


def test_budget_caps_kept_frames():
    unit = _unit([10] * 10 + [200] * 50)
    res = run(_spec(budget=10), [unit], dry_run=True)
    assert res.n_kept == 10


def test_dedupe_window_resets_between_captures():
    """Legacy cleared cap_phashes per capture; a global window would wrongly
    suppress an identical frame appearing in a different capture."""
    rng = np.random.default_rng(3)
    noise = rng.integers(0, 255, (80, 120, 3), dtype=np.uint8)

    def unit(name):
        return name, [
            FrameRecord(rgb=noise.copy(), capture=name, frame_idx=0),
            FrameRecord(rgb=noise.copy(), capture=name, frame_idx=1),
        ]

    res = run(_spec(baseline=NoBaseline(), phash_dist=4, phash_lookback=30),
              [unit("c0"), unit("c1")], dry_run=True)
    # Within a capture the second copy is a duplicate; across captures it is not.
    assert ("c0", 0) in res.kept_keys
    assert ("c0", 1) not in res.kept_keys
    assert ("c1", 0) in res.kept_keys
    assert res.n_dup_dropped == 2


def test_streaming_does_not_materialise_the_unit():
    """The runner must consume an iterator, not require a list -- a GelSLAM
    episode is 45,557 frames (10.5 GB) and cannot be buffered."""
    def gen():
        for i in range(30):
            yield FrameRecord(rgb=np.full((80, 120, 3), 10 if i < 10 else 200,
                                          dtype=np.uint8),
                              capture="c0", frame_idx=i)

    res = run(_spec(), [("c0", gen())], dry_run=True)
    assert res.n_kept == 20
