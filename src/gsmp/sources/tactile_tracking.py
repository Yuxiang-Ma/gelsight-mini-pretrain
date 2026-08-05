"""TactileTracking / NormalFlow -- one .avi per trial.

tier-1: published rows carry capture + frame_idx, so tools/regress.py can
verify this migration against the release.

`iter_units` is derived from `legacy/make_parquet_v2.py::iter_tactile_tracking`
(lines 174-203), not from a from-scratch reading of the raw tree layout.
Two details matter and are easy to get wrong by "cleaning up" the legacy
code instead of copying its behaviour:

  - capture = f"{obj}/{trial}" where `obj`/`trial` come from splitting the
    trial directory name (the parent of gelsight.avi) on a trailing run of
    digits via `^([a-zA-Z]+)(\\d+)$`, e.g. "corner3" -> obj="corner",
    trial="3" -> capture "corner/3". Directory names with no trailing digit
    (e.g. "avocado") fall back to trial "0" -> "avocado/0". Globbing
    "**/*.avi" and deriving capture from the file path (as a naive reading
    would) produces "normalflow_dataset/corner3" instead, which does not
    match a single published row.
  - only `normalflow_dataset/*/gelsight.avi` is globbed, not `**/*.avi`:
    each trial directory also holds a `webcam.avi` that is never published.
"""
from __future__ import annotations

import os
import re
from glob import glob
from typing import Iterator, Optional, Tuple

import cv2

from gsmp import config
from gsmp.baseline import FirstNFrames
from gsmp.runner import FrameRecord
from gsmp.spec import SourceSpec, register

SPEC = register(SourceSpec(
    name="tactile_tracking",
    domain="real",
    gel_variant="markerless",
    license_repo="main",
    baseline=FirstNFrames(10),
    a_min=40,
    i_min=10.0,           # docs/imin_from_code.md: make_parquet_v2.py:54
    channel_mode="auto",
    phash_dist=4,
    phash_lookback=30,
    notes="MIT. RA-L 2024. One normalflow .avi per trial.",
))

_ROOT = config.RAW_ROOT / "markerless" / "TactileTracking"
_OBJ_RE = re.compile(r"^([a-zA-Z]+)(\d+)$")


def _trial_frames(avi_path: str, capture: str, obj_name: str) -> Iterator[FrameRecord]:
    """Lazily decode one trial's frames. Never buffers more than one."""
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
                split="train",
                episode=capture,
                frame_idx=idx,
            )
            idx += 1
    finally:
        cap.release()


def iter_units(limit: Optional[int] = None) -> Iterator[Tuple[str, Iterator[FrameRecord]]]:
    """Yield (capture, frame_generator) per trial .avi, one at a time.

    The frame generator is not materialised here: the runner consumes units
    streaming (see gsmp.runner.run's docstring, property 1).
    """
    n = 0
    candidates = sorted(glob(str(_ROOT / "normalflow_dataset" / "*" / "gelsight.avi")))
    for vid in candidates:
        trial_dir = os.path.basename(os.path.dirname(vid))
        m = _OBJ_RE.match(trial_dir)
        if m:
            obj, trial = m.group(1), m.group(2)
        else:
            obj, trial = trial_dir, "0"
        capture = f"{obj}/{trial}"
        yield capture, _trial_frames(vid, capture, obj)
        n += 1
        if limit is not None and n >= limit:
            return


def dry_run_keys(limit: Optional[int] = None):
    """Return (kept_keys, n_seen) without writing anything."""
    from gsmp.runner import run

    res = run(SPEC, iter_units(limit=limit), dry_run=True)
    return res.kept_keys, res.n_seen
