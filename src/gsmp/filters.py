"""Contact filter, channel-order normalization, and perceptual hashing.

All functions are pure. `grey_center`, `phash` and `hamming` are lifted
verbatim (behaviour-preserving) from legacy/make_parquet_v2.py; the filter
is generalised so a_min and i_min are always explicit arguments -- the
legacy code carried three different defaults for i_min in three places.
"""
from __future__ import annotations

import io
from typing import Tuple

import numpy as np
from PIL import Image

#: Sensor-noise floor in grey-levels. A pixel must differ from the baseline
#: by more than this to count as "lit".
PIXEL_THRESH = 10


def grey_center(arr: np.ndarray) -> np.ndarray:
    """Central 50% crop, greyscale, float32.

    Crops to central 50% region first, then averages channels (if RGB). This
    is 4x cheaper than the legacy mean-then-crop form: the channel mean is
    computed independently per-pixel, so cropping before averaging is
    mathematically identical. Bit-identical to legacy output; pinned by
    test_grey_center_matches_legacy.
    """
    h, w = arr.shape[:2]
    crop = arr[h // 4:3 * h // 4, w // 4:3 * w // 4]
    g = crop.mean(axis=2) if crop.ndim == 3 else crop
    return g.astype(np.float32)


def contact_metrics(
    rgb: np.ndarray,
    baseline: np.ndarray,
    pixel_thresh: int = PIXEL_THRESH,
) -> Tuple[int, float]:
    """Return (contact_area, contact_intensity) against `baseline`.

    area      = number of central-crop pixels differing by > pixel_thresh
    intensity = mean absolute difference over exactly those pixels
                (0.0 when area == 0)
    """
    diff = np.abs(grey_center(rgb) - baseline)
    mask = diff > pixel_thresh
    area = int(mask.sum())
    if area == 0:
        return 0, 0.0
    return area, float(diff[mask].mean())


def passes_filter(
    rgb: np.ndarray,
    baseline: np.ndarray,
    a_min: int,
    i_min: float,
    pixel_thresh: int = PIXEL_THRESH,
) -> bool:
    """The unified validity rule: area >= a_min AND intensity >= i_min.

    a_min and i_min are required. There is deliberately no default for
    i_min -- see docs/PIPELINE.md on why the legacy value is ambiguous.
    """
    area, inten = contact_metrics(rgb, baseline, pixel_thresh)
    return area >= a_min and inten >= i_min


def channel_check(rgb: np.ndarray) -> float:
    """Signed R-B channel mean difference.

    A GelSight Mini at rest is lit by three coloured LEDs such that B > R,
    so a positive value means the frame is probably stored BGR.
    """
    return float(rgb[..., 0].mean()) - float(rgb[..., 2].mean())


def maybe_swap_channels(rgb: np.ndarray, mode: str) -> np.ndarray:
    """Normalize channel order to RGB.

    mode:
      'rgb'          never swap
      'bgr'          always swap
      'auto'/'mixed' swap only when channel_check(rgb) > 0
    """
    if mode == "rgb":
        return rgb
    if mode == "bgr":
        return rgb[..., ::-1].copy()
    if mode in ("auto", "mixed"):
        return rgb[..., ::-1].copy() if channel_check(rgb) > 0 else rgb
    raise ValueError(f"unknown channel mode: {mode!r}")


def phash(rgb: np.ndarray) -> int:
    """8x8 DCT-low-frequency perceptual hash as a 64-bit int.

    `dct1` mirrors along axis 0 (`x[::-1]`) while concatenating along
    axis -1. That is not the axis pairing a textbook DCT-II mirror uses,
    but it is exactly what produced every dedupe decision in the published
    release, so it is reproduced verbatim. Changing it to the "correct"
    `x[..., ::-1]` yields a hash 20 bits different out of 64 -- a different
    hash function, and a different dataset.
    """
    im = Image.fromarray(rgb).convert("L").resize((32, 32), Image.LANCZOS)
    a = np.array(im, dtype=np.float32)

    def dct1(x: np.ndarray) -> np.ndarray:
        return np.fft.fft(
            np.concatenate([x, x[::-1]], axis=-1)
        ).real[..., :x.shape[-1]]

    d = dct1(dct1(a).T).T
    low = d[:8, :8].flatten()
    med = np.median(low[1:])          # skip DC term
    h = 0
    for bit in (low > med):
        h = (h << 1) | int(bit)
    return h


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")
