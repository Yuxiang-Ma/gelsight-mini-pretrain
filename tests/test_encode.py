from __future__ import annotations

import io

import numpy as np
from PIL import Image

from gsmp import encode


def test_encode_jpeg_roundtrips_shape():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    blob = encode.encode_jpeg(img)
    assert isinstance(blob, bytes)
    assert blob[:2] == b"\xff\xd8"            # JPEG SOI marker
    back = np.array(Image.open(io.BytesIO(blob)))
    assert back.shape == (240, 320, 3)


def test_default_quality_is_92():
    assert encode.JPEG_QUALITY == 92


def test_higher_quality_is_larger():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    assert len(encode.encode_jpeg(img, 92)) > len(encode.encode_jpeg(img, 40))
