"""GelSLAM -- tracking and reconstruction episodes, one .avi per episode.

tier-1: published rows carry capture + frame_idx, so tools/regress.py can
verify this migration against the release.

`iter_units` is derived from `legacy/make_parquet_v2.py::iter_gelslam`
(lines 141-172), not from a from-scratch reading of the raw tree layout.
Two details matter and are easy to get wrong by "cleaning up" the legacy
code instead of copying its behaviour:

  - capture = f"{sub}/{episode}" where `episode` is the *parent directory*
    of gelsight.avi, not a path derived from the video filename. Published
    capture values look like "reconstruction_dataset/handle_of_hammer" --
    no "/gelsight" suffix. Globbing "**/*.avi" and stripping the suffix
    would produce "reconstruction_dataset/handle_of_hammer/gelsight",
    which does not match a single published row.
  - the raw HF download can land the two dataset subdirectories either
    directly under `_ROOT` or one level deeper under `_ROOT/dataset/`
    (legacy's own comment: "The HF download puts content under root or
    root/dataset/ depending on extraction"), so both locations are
    globbed.
"""
from __future__ import annotations

import os
from glob import glob
from typing import Iterator, Optional, Tuple

import cv2

from gsmp import config
from gsmp.baseline import FirstNFrames
from gsmp.runner import FrameRecord
from gsmp.spec import SourceSpec, register

SPEC = register(SourceSpec(
    name="gelslam",
    domain="real",
    gel_variant="markerless",
    license_repo="main",
    baseline=FirstNFrames(10),
    a_min=40,
    i_min=10.0,           # docs/imin_from_code.md: make_parquet_v2.py:53
    channel_mode="auto",
    phash_dist=4,
    phash_lookback=30,
    notes="MIT. arXiv:2508.15990. One .avi per tracking/recon episode.",
))

_ROOT = config.RAW_ROOT / "markerless" / "GelSLAM"

#: (sub-directory name, published split label) -- mirrors legacy's
#: `split = "train" if sub == "tracking_dataset" else "recon"`.
_SUBS = (
    ("tracking_dataset", "train"),
    ("reconstruction_dataset", "recon"),
)


def _episode_frames(avi_path: str, capture: str, obj_name: str,
                     split: str) -> Iterator[FrameRecord]:
    """Lazily decode one episode's frames. Never buffers more than one."""
    cap = cv2.VideoCapture(avi_path)
    try:
        idx = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            yield FrameRecord(
                rgb=bgr[:, :, ::-1].copy(),
                capture=capture,
                obj_name=obj_name,
                split=split,
                episode=capture,
                frame_idx=idx,
            )
            idx += 1
    finally:
        cap.release()


def iter_units(limit: Optional[int] = None) -> Iterator[Tuple[str, Iterator[FrameRecord]]]:
    """Yield (capture, frame_generator) per episode .avi, one at a time.

    The frame generator is not materialised here: GelSLAM's largest episode
    is 45,557 frames, 10.5 GB as uint8 RGB, and the runner consumes units
    streaming (see gsmp.runner.run's docstring, property 1).
    """
    n = 0
    for sub, split in _SUBS:
        # Deliberately NOT sorted. Legacy iter_gelslam (make_parquet_v2.py:
        # 147-150) concatenates the two globs in raw filesystem order and
        # never sorts them. That order is load-bearing: the runner's
        # background-keep quota (n_empty_kept >= bg_keep_rate *
        # max(n_kept, 1)) is a running count over the *whole* source, so
        # changing episode order changes which empty frames land inside the
        # quota and which don't -- a different, still-valid-looking, but
        # non-matching kept set. Sorting here would make the code more
        # deterministic-looking but less faithful to the release. Contrast
        # with iter_tactile_tracking (make_parquet_v2.py:178), which DOES
        # sort, and whose gsmp.sources.tactile_tracking counterpart sorts
        # too and matches exactly for that reason.
        candidates = (
            glob(str(_ROOT / sub / "*" / "gelsight.avi"))
            + glob(str(_ROOT / "dataset" / sub / "*" / "gelsight.avi"))
        )
        for vid in candidates:
            episode = os.path.basename(os.path.dirname(vid))
            capture = f"{sub}/{episode}"
            yield capture, _episode_frames(vid, capture, episode, split)
            n += 1
            if limit is not None and n >= limit:
                return


def dry_run_keys(limit: Optional[int] = None):
    """Return (kept_keys, n_seen) without writing anything."""
    from gsmp.runner import run

    res = run(SPEC, iter_units(limit=limit), dry_run=True)
    return res.kept_keys, res.n_seen
