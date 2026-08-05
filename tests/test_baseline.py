from __future__ import annotations

import numpy as np
import pytest

from gsmp import baseline


def _frames(values):
    """One (80,120,3) frame per value, so grey_center gives (40,60)."""
    out = []
    for v in values:
        a = np.full((80, 120, 3), v, dtype=np.uint8)
        out.append(a)
    return out


def test_first_n_frames_uses_only_the_head():
    frames = _frames([10, 10, 10, 200, 200, 200])
    b = baseline.FirstNFrames(3).compute(frames)
    assert b.shape == (40, 60)
    assert b.mean() == pytest.approx(10.0)


def test_first_n_frames_handles_fewer_frames_than_n():
    b = baseline.FirstNFrames(10).compute(_frames([7, 7]))
    assert b.mean() == pytest.approx(7.0)


def test_global_median_is_robust_to_contact_outliers():
    # 8 at-rest frames, 2 heavy-contact frames -> median ignores the outliers
    frames = _frames([20] * 8 + [250] * 2)
    b = baseline.GlobalMedian(n=10, seed=0).compute(frames)
    assert b.mean() == pytest.approx(20.0)


def test_per_capture_median_samples_deterministically():
    frames = _frames(list(range(100)))
    a = baseline.PerCaptureMedian(n=30, seed=0).compute(frames)
    b = baseline.PerCaptureMedian(n=30, seed=0).compute(frames)
    np.testing.assert_array_equal(a, b)


def test_per_touch_median_uses_head_of_each_touch():
    b = baseline.PerTouchMedian(n=5).compute(_frames([3] * 5 + [255] * 20))
    assert b.mean() == pytest.approx(3.0)


def test_explicit_reference_reads_the_shipped_blank(tmp_path):
    from PIL import Image

    ref = tmp_path / "blank.png"
    Image.fromarray(np.full((80, 120, 3), 33, dtype=np.uint8)).save(ref)
    b = baseline.ExplicitReference(ref).compute([])
    assert b.shape == (40, 60)
    assert b.mean() == pytest.approx(33.0, abs=1.0)


def test_explicit_reference_ignores_the_frames_it_is_given():
    """The reference is the baseline; frames must not influence it."""
    from PIL import Image
    import tempfile, pathlib

    with tempfile.TemporaryDirectory() as d:
        ref = pathlib.Path(d) / "blank.png"
        Image.fromarray(np.full((80, 120, 3), 33, dtype=np.uint8)).save(ref)
        s = baseline.ExplicitReference(ref)
        np.testing.assert_array_equal(
            s.compute([]), s.compute(_frames([250] * 20))
        )


def test_no_baseline_returns_none():
    assert baseline.NoBaseline().compute(_frames([1, 2, 3])) is None


def test_needs_frames_is_false_only_for_no_baseline():
    assert not baseline.needs_frames(baseline.NoBaseline())
    assert baseline.needs_frames(baseline.FirstNFrames(10))
    assert baseline.needs_frames(baseline.GlobalMedian(200))


def test_empty_frame_list_returns_none():
    assert baseline.FirstNFrames(10).compute([]) is None
    assert baseline.GlobalMedian(10).compute([]) is None
