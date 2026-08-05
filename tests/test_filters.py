from __future__ import annotations

import numpy as np
import pytest

from gsmp import filters


def test_grey_center_takes_central_50_percent():
    arr = np.zeros((80, 120, 3), dtype=np.uint8)
    arr[20:60, 30:90] = 255          # exactly the central 50% box
    g = filters.grey_center(arr)
    assert g.shape == (40, 60)
    assert g.dtype == np.float32
    assert g.min() == 255.0


def test_contact_metrics_counts_only_pixels_above_thresh():
    baseline = np.zeros((40, 60), dtype=np.float32)
    rgb = np.zeros((80, 120, 3), dtype=np.uint8)
    # 100 central pixels lifted to 50 grey-levels; 5 lifted to 3 (below thresh)
    rgb[20:22, 30:80] = 50           # 2*50 = 100 px inside the central crop
    rgb[22:23, 30:35] = 3            # 5 px, below PIXEL_THRESH=10
    area, inten = filters.contact_metrics(rgb, baseline)
    assert area == 100
    assert inten == pytest.approx(50.0)


def test_passes_filter_requires_both_area_and_intensity():
    baseline = np.zeros((40, 60), dtype=np.float32)

    big_but_faint = np.zeros((80, 120, 3), dtype=np.uint8)
    big_but_faint[20:40, 30:90] = 11          # area huge, intensity 11
    assert filters.passes_filter(big_but_faint, baseline, a_min=40, i_min=10)
    assert not filters.passes_filter(big_but_faint, baseline, a_min=40, i_min=15)

    bright_but_tiny = np.zeros((80, 120, 3), dtype=np.uint8)
    bright_but_tiny[20:21, 30:35] = 200       # 5 px, very bright
    assert not filters.passes_filter(bright_but_tiny, baseline, a_min=40, i_min=10)


def test_channel_check_sign_flags_bgr_storage():
    rgb_like = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb_like[..., 2] = 200                     # B > R  -> at-rest GelSight Mini
    assert filters.channel_check(rgb_like) < 0

    bgr_like = np.zeros((4, 4, 3), dtype=np.uint8)
    bgr_like[..., 0] = 200                     # R > B  -> stored BGR
    assert filters.channel_check(bgr_like) > 0


def test_maybe_swap_channels_modes():
    bgr_like = np.zeros((2, 2, 3), dtype=np.uint8)
    bgr_like[..., 0] = 200

    assert filters.maybe_swap_channels(bgr_like, "rgb")[0, 0, 0] == 200
    assert filters.maybe_swap_channels(bgr_like, "bgr")[0, 0, 2] == 200
    assert filters.maybe_swap_channels(bgr_like, "auto")[0, 0, 2] == 200

    rgb_like = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb_like[..., 2] = 200
    assert filters.maybe_swap_channels(rgb_like, "auto")[0, 0, 2] == 200


def test_phash_identical_images_have_zero_distance():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    assert filters.hamming(filters.phash(img), filters.phash(img.copy())) == 0


def test_phash_differs_for_unrelated_images():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    b = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    assert filters.hamming(filters.phash(a), filters.phash(b)) > 4


def test_hamming():
    assert filters.hamming(0b1011, 0b1001) == 1
    assert filters.hamming(0, 0) == 0
