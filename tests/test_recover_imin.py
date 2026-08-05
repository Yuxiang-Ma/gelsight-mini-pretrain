from __future__ import annotations

import numpy as np
import pytest

from gsmp.tools_imin import ImInEstimate, estimate_from_intensities


def test_p01_ignores_background_contamination():
    # 990 real contacts at >=12, 10 background frames far below
    kept = np.concatenate([np.full(990, 12.0), np.full(10, 1.0)])
    est = estimate_from_intensities("demo", kept, bg_keep_rate=0.015)
    assert est.min_kept_intensity == pytest.approx(1.0)
    assert est.p01 == pytest.approx(12.0, abs=0.5)
    assert est.verdict.startswith("likely i_min")


def test_flags_ambiguous_when_distribution_has_no_floor():
    kept = np.linspace(0.0, 60.0, 1000)
    est = estimate_from_intensities("demo", kept, bg_keep_rate=0.015)
    assert est.verdict == "ambiguous"


def test_rejects_empty_input():
    with pytest.raises(ValueError):
        estimate_from_intensities("demo", np.array([]), bg_keep_rate=0.015)
