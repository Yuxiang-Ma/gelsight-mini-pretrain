"""JPEG encoding. Quality 92 at native resolution -- no resizing."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

JPEG_QUALITY = 92


def encode_jpeg(rgb: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
