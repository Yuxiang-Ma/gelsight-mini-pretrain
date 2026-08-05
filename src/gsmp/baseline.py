"""Baseline strategies -- how a source decides what "gel at rest" looks like.

The legacy code expressed this as a SKIP_EMPTY_FILTER boolean: True meant
"this source filters inside its own iterator, leave it alone". 7 of 9 sources
were True, so the "unified filter" was really only unified for 2 of them, and
the actual variation -- six different ways of computing the reference frame --
was invisible.

Making the strategy a first-class object removes that escape hatch: every
source now declares how its baseline is computed, and the runner is generic.

All strategies return a greyscale central-crop array (the same space
gsmp.filters.contact_metrics compares against), or None when the source does
not use a pixel baseline at all.
"""
from __future__ import annotations

import abc
import os
from typing import List, Optional, Sequence, Union

import numpy as np

from gsmp.filters import grey_center


class BaselineStrategy(abc.ABC):
    """Computes a per-unit reference image from that unit's frames."""

    @abc.abstractmethod
    def compute(self, frames: Sequence[np.ndarray]) -> Optional[np.ndarray]:
        ...

    def __repr__(self) -> str:                      # pragma: no cover
        args = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"{type(self).__name__}({args})"


def _median_of(frames: Sequence[np.ndarray]) -> Optional[np.ndarray]:
    if len(frames) == 0:
        return None
    return np.median(np.stack([grey_center(f) for f in frames]), axis=0)


class FirstNFrames(BaselineStrategy):
    """Median of the first n frames -- the pre-contact prologue of a video."""

    def __init__(self, n: int) -> None:
        self.n = n

    def compute(self, frames: Sequence[np.ndarray]) -> Optional[np.ndarray]:
        return _median_of(list(frames)[: self.n])


class _RandomSampleMedian(BaselineStrategy):
    """Median over a deterministic random sample of n frames."""

    def __init__(self, n: int, seed: int = 0) -> None:
        self.n = n
        self.seed = seed

    def compute(self, frames: Sequence[np.ndarray]) -> Optional[np.ndarray]:
        frames = list(frames)
        if not frames:
            return None
        if len(frames) <= self.n:
            picked: List[np.ndarray] = frames
        else:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(len(frames), size=self.n, replace=False)
            picked = [frames[i] for i in sorted(idx)]
        return _median_of(picked)


class PerCaptureMedian(_RandomSampleMedian):
    """Median of a random sample within one capture (object x pose x side)."""


class GlobalMedian(_RandomSampleMedian):
    """Median of a random sample across the whole source."""


class PerGroupMedian(_RandomSampleMedian):
    """Median of a random sample within one group (indenter shape, split, ...).

    The grouping key lives on the source module; this strategy only sees the
    frames of a single group.
    """


class PerTouchMedian(BaselineStrategy):
    """Median of the first n frames of one touch (a touch = one short video)."""

    def __init__(self, n: int) -> None:
        self.n = n

    def compute(self, frames: Sequence[np.ndarray]) -> Optional[np.ndarray]:
        return _median_of(list(frames)[: self.n])


class ExplicitReference(BaselineStrategy):
    """Baseline read from a gel-at-rest reference image shipped upstream.

    py3DCal ships `blank_images/blank.png`. The legacy iterator uses it
    directly and never computes a median -- PIPELINE.md's claim that
    threedcal uses a "cross-image median over a random 200-frame sample" is
    simply wrong. Verified against legacy/make_parquet_v2.py:529-547.
    """

    def __init__(self, path: Union[str, "os.PathLike[str]"]) -> None:
        self.path = path

    def compute(self, frames: Sequence[np.ndarray]) -> Optional[np.ndarray]:
        from PIL import Image

        blank = np.asarray(Image.open(self.path).convert("L"), dtype=np.float32)
        h, w = blank.shape
        return blank[h // 4:3 * h // 4, w // 4:3 * w // 4]


class NoBaseline(BaselineStrategy):
    """No pixel baseline.

    Used by sources that are pre-curated upstream (every frame is a contact
    moment, so a median would include contact and poison the reference), and
    by markered sources where pixel diffing is unreliable and a force
    threshold is used instead.
    """

    def compute(self, frames: Sequence[np.ndarray]) -> Optional[np.ndarray]:
        return None


def needs_frames(strategy: BaselineStrategy) -> bool:
    """True if the runner must buffer frames before it can filter."""
    return not isinstance(strategy, NoBaseline)
