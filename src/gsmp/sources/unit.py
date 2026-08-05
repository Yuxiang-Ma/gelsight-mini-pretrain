"""UniT -- 3D-pose replay buffer, one zarr array for the whole capture.

tier-1: published rows carry capture + frame_idx, so tools/regress.py can
verify this migration against the release.

`unit` is NOT a `make_parquet_v2.process()` source. It was ingested by its
own standalone script, `legacy/ingest_unit.py` (lines 62-136), with a
baseline strategy and a background-keep decision that do not match the
generic pipeline `gsmp.runner.run()` reproduces for the FirstNFrames
sources (gelslam, tactile_tracking). Two details are load-bearing:

  - BASELINE. `legacy/ingest_unit.py:74-78` builds the reference frame from
    the median of 120 frames sampled *uniformly at random from the whole
    11,340-frame array* (`rng.sample(range(n_total), 120)`,
    `rng = random.Random(20260520)`), not from the first N frames of a
    stream. The 120 baseline-sample frames are NOT excluded from the
    candidate pool afterwards -- baseline sampling and output eligibility
    are two independent draws over the same population, unlike gelslam/
    tactile_tracking where the first 10 frames of every capture are
    consumed and can never appear in the output (see gsmp.runner.run's
    docstring, property 2). Every published `unit` row can, in principle,
    also be one of the 120 baseline frames.

  - BACKGROUND KEEP. `legacy/ingest_unit.py:106` is a genuine Bernoulli
    draw, `elif rng.random() < BG_RATE`, on the SAME rng instance used for
    the baseline sample -- not the deterministic running quota
    (`n_empty_kept < bg_keep_rate * max(n_kept, 1)`) gsmp.runner.run()
    implements. Reproducing the published set exactly requires calling
    `rng.random()` in precisely the order and on precisely the candidates
    legacy did: only for frames that (a) are not near-black
    (`rgb.mean() < 10`, checked first and does NOT consume the rng) and
    (b) fail the area/intensity filter.

Because both of these diverge from what the generic runner does, this
module reproduces the filter inline in `_unit_frames` instead of relying on
`gsmp.runner.run()`'s baseline/bg-keep machinery. The SPEC declares
`baseline=NoBaseline()`: `gsmp.baseline.needs_frames(NoBaseline())` is
False, so the runner never buffers frames to build its own baseline, `base`
stays `None` for every record, and `is_empty` is therefore always False --
the runner accepts every FrameRecord this module yields exactly as given.
`phash_dist=None` disables the runner's dedupe pass for the same reason:
`legacy/ingest_unit.py` never deduplicates.

`dry_run_keys` still routes through `gsmp.runner.run()` (mirroring
gelslam/tactile_tracking) purely for its (capture, frame_idx) bookkeeping
and RunResult shape -- it never exercises the runner's filtering branches
for this source, since those are neutralised by NoBaseline + phash_dist=None
as described above.
"""
from __future__ import annotations

import random
from typing import Iterator, Optional, Tuple

import numpy as np
import zarr

from gsmp import config
from gsmp.baseline import NoBaseline
from gsmp.filters import PIXEL_THRESH, grey_center
from gsmp.runner import FrameRecord
from gsmp.spec import SourceSpec, register

SPEC = register(SourceSpec(
    name="unit",
    domain="real",
    gel_variant="markered",
    license_repo="main",
    baseline=NoBaseline(),
    a_min=40,
    i_min=12.0,           # docs/imin_from_code.md: legacy/ingest_unit.py:44
    channel_mode="rgb",   # legacy never swaps channels for unit
    phash_dist=None,      # legacy never dedupes unit
    bg_keep_rate=0.015,   # legacy/ingest_unit.py:45 BG_RATE
    rng_seed=20260520,    # legacy/ingest_unit.py:63
    notes=(
        "arXiv:2408.06481. Standalone ingest (ingest_unit.py), not "
        "make_parquet_v2.process(): random-sample baseline + Bernoulli "
        "bg-keep, both on one rng instance."
    ),
))

_ZARR_PATH = (
    config.RAW_ROOT / "markerless" / "UniT" / "UniT_dataset" /
    "3D_pose_gelsight" / "replay_buffer.zarr"
)

_CAPTURE = "3D_pose_gelsight"       # legacy/ingest_unit.py:118 -- a constant,
_OBJ_NAME = "unit_pose_target"      # not a per-recording grouping.
_N_BASELINE = 120
_NEAR_BLACK_THRESH = 10             # legacy/ingest_unit.py:95: rgb.mean() < 10
_CHUNK_SIZE = 256


def _unit_frames(limit: Optional[int]) -> Iterator[FrameRecord]:
    """Lazily filter the zarr array frame-by-frame, legacy decision-for-decision.

    Streams in chunks of `_CHUNK_SIZE` (matching legacy) so no more than one
    chunk of the 11,340 x 240 x 320 x 3 array is materialised at a time.
    """
    images = zarr.open_array(
        store=zarr.DirectoryStore(str(_ZARR_PATH / "data" / "tactile_image")),
        mode="r",
    )
    poses = zarr.open_array(
        store=zarr.DirectoryStore(str(_ZARR_PATH / "data" / "3Dpose")),
        mode="r",
    )
    n_total = images.shape[0]

    rng = random.Random(SPEC.rng_seed)
    base_idxs = sorted(rng.sample(range(n_total), _N_BASELINE))
    grays = [grey_center(images[i][:]) for i in base_idxs]
    baseline = np.median(np.stack(grays), axis=0)

    n_yielded = 0
    for start in range(0, n_total, _CHUNK_SIZE):
        end = min(start + _CHUNK_SIZE, n_total)
        chunk = images[start:end][:]
        chunk_poses = poses[start:end][:]
        for k, rgb in enumerate(chunk):
            i = start + k

            # Drop near-black frames (LED off / no signal). Does not
            # consume the rng -- legacy checks this before the coin flip.
            if rgb.mean() < _NEAR_BLACK_THRESH:
                continue

            g = grey_center(rgb)
            diff = np.abs(g - baseline)
            mask = diff > PIXEL_THRESH
            area = int(mask.sum())
            inten = float(diff[mask].mean()) if area > 0 else 0.0
            passes = area >= SPEC.a_min and inten >= SPEC.i_min

            if passes:
                pass
            elif rng.random() < SPEC.bg_keep_rate:
                pass
            else:
                continue

            pose = chunk_poses[k]
            yield FrameRecord(
                rgb=rgb,
                capture=_CAPTURE,
                obj_name=_OBJ_NAME,
                split="train",
                episode=_CAPTURE,
                frame_idx=i,
                extra=dict(
                    x_mm=float(pose[0]),
                    y_mm=float(pose[1]),
                    z_mm=float(pose[2]),
                    quat_z=float(pose[3]),  # actually yaw; legacy abuses quat_z
                ),
            )
            n_yielded += 1
            if limit is not None and n_yielded >= limit:
                return


def iter_units(limit: Optional[int] = None) -> Iterator[Tuple[str, Iterator[FrameRecord]]]:
    """Yield the single (capture, frame_generator) unit for `unit`.

    Unlike the .avi-per-trial sources, UniT is one zarr array of 11,340
    frames with a single hardcoded capture label (see module docstring).
    All filtering happens inline in `_unit_frames`, so there is exactly one
    unit to yield; `limit` therefore caps yielded *frames*, not units --
    the only cap that means anything when there is only one unit.
    """
    yield _CAPTURE, _unit_frames(limit)


def dry_run_keys(limit: Optional[int] = None):
    """Return (kept_keys, n_seen) without writing anything."""
    from gsmp.runner import run

    res = run(SPEC, iter_units(limit=limit), dry_run=True)
    return res.kept_keys, res.n_seen
