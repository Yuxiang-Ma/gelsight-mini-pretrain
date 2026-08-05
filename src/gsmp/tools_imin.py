"""Recover the effective i_min of a published source from its own data.

The published parquet contains only KEPT frames. Under the rule
`keep iff area >= a_min and intensity >= i_min`, the lower edge of the kept
intensity distribution is an upper-bound estimate of i_min -- except that
bg_keep_rate (1.5%) of frames were kept *despite* failing, which contaminates
the very bottom. So the estimate uses the 1st percentile, not the minimum,
and reports both.
"""
from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class ImInEstimate:
    source: str
    n_sampled: int
    min_kept_intensity: float
    p01: float
    p05: float
    verdict: str


def estimate_from_intensities(
    source: str,
    intensities: np.ndarray,
    bg_keep_rate: float,
) -> ImInEstimate:
    if intensities.size == 0:
        raise ValueError(f"no intensities sampled for {source}")

    arr = np.sort(np.asarray(intensities, dtype=np.float64))
    p01 = float(np.percentile(arr, 1))
    p05 = float(np.percentile(arr, 5))
    lo = float(arr[0])

    # A real threshold shows up as a sharp floor: the 1st and 5th percentiles
    # sit close together well above the contaminated minimum.
    spread = p05 - p01
    if spread <= max(1.0, 0.1 * p01) and p01 > lo + 0.5:
        verdict = f"likely i_min = {p01:.1f}"
    elif spread <= max(1.0, 0.1 * p01):
        verdict = f"likely i_min = {p01:.1f} (no bg contamination detected)"
    else:
        verdict = "ambiguous"

    return ImInEstimate(
        source=source,
        n_sampled=int(arr.size),
        min_kept_intensity=lo,
        p01=p01,
        p05=p05,
        verdict=verdict,
    )
