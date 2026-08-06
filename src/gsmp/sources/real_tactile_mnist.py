"""RealTactileMNIST -- 3D-printed digit imprints, one peak frame per touch.

tier-1: published rows carry capture + frame_idx, so tools/regress.py can
verify this migration against the release.

READ THIS BEFORE TRUSTING `legacy/make_parquet_v2.py::iter_real_tactile_mnist`
(lines 207-307) FOR THIS SOURCE. That function -- and its near-identical
sibling `legacy/parallel_rtm.py::process_one_row` -- describe a *different*
algorithm than the one that produced the published `real_tactile_mnist`
release: `random.Random(42)`, `I_MIN=15`, and a per-frame K=0.30 Bernoulli
draw inside a "touch window" that can keep 0, 1, 2 or more frames from the
same touch. Three empirical facts rule that algorithm out:

  1. EVERY published capture has EXACTLY ONE row. Both splits (25,829 train
     + 5,127 test = 30,956 rows) were read back and grouped by `capture`;
     every group has size 1, none has 0, 2, or more. A K=0.30 Bernoulli
     draw over a multi-frame window would routinely produce 0, 1, 2, 3+
     kept frames per touch -- not a hard, universal cap of exactly 1.
  2. Published `capture` strings have NO "r" prefix ("2023-09-28_15-50-43_
     7-0_t4", not "r2023-09-28_15-50-43_7-0_t4"). `iter_real_tactile_mnist`
     and `parallel_rtm.py` both literally write `f"r{round_id}_t{tj}"`.
  3. `legacy/extract_rtm_video.py` -- "Re-extract real_tactile_mnist from
     the VIDEO upstream (vs the prior single-frame extract)" -- writes
     `capture=f"{id_}_t{ti}"` (no "r"), `I_MIN=12`, and `K_PER_TOUCH=1`: for
     each touch it decodes every frame, scores each one
     (`score = intensity if area >= A_MIN else 0.0` on the central-50%-crop,
     `PIXEL_THRESH=10`, `A_MIN=40`), and keeps only the single highest-
     scoring frame if that score clears `I_MIN=12` -- structurally
     guaranteeing at most one kept frame per touch. This matches fact 1
     exactly and fact 2 exactly.

I verified `extract_rtm_video.py`'s algorithm directly against the raw
touch videos rather than trusting its docstring: for every published
(capture, frame_idx) whose winning frame actually clears the A_MIN/I_MIN
gate, re-running the score-and-argmax procedure on the raw video
reproduces the published frame_idx exactly -- 626/626 in one measured
test-split sample (10 rounds x up to 256 touches, area>=40-and-score>=12
subset), 100% agreement, zero cases where a *different*, non-null frame_idx
was predicted. This is 15 (`docs/imin_from_code.md`'s claimed I_MIN) failing
outright at this same check: at I_MIN=15 the match rate collapses (61-104
of 654 in the same sample, vs 626/654 at I_MIN=12) because most winning
frames' intensity sits in [12, 15). `docs/imin_from_code.md` recorded 15 by
reading `make_parquet_v2.py:231` alone, without cross-checking that
`iter_real_tactile_mnist` is not the function that actually ran -- the same
class of error its own "tacquad conflict" and "unit conflict" sections
already document for other sources, just not (yet) called out for this one.

Two further details, mirroring `gsmp.sources.unit`/`sparsh`/`tacquad`'s
precedent:

  - BASELINE. Per TOUCH (not per capture-stream-prologue): the median of
    `grey_center` over *every* decoded frame of that one touch's clip
    (`legacy/extract_rtm_video.py:99,148-149`), not the first N frames.
    A touch is typically 58-90 frames; baseline-sample frames are not
    excluded from being the winning frame afterwards.
  - BACKGROUND KEEP -- SEED NOT RECOVERABLE. When no frame in a touch
    clears the A_MIN/I_MIN gate, `legacy/extract_rtm_video.py:151-155`
    keeps one uniformly-random frame from that touch with probability
    `BG_RATE=0.015`, on `rng = random.Random(hash(fname) & 0xFFFFFFFF)` --
    one rng per upstream parquet SHARD (`hash(fname)`, the shard's
    basename), consumed in row/touch order across that whole shard. Exactly
    like `sparsh.py`/`tacquad.py`'s `hash(indenter)`/`hash(domain)` seeds,
    `hash(str)` is SipHash-randomised per Python process unless
    `PYTHONHASHSEED` is pinned in the environment; no such pin exists
    anywhere in this repo, its predecessor MultimodalData tree, or the
    shell/conda profiles on this machine (same search `sparsh.py`/
    `tacquad.py` already performed). This module reproduces the formula
    verbatim (`hash(fname) & 0xFFFFFFFF`) -- the faithful transcription --
    which reproduces the deterministic peak-frame set exactly but draws a
    different 1.5%-of-empty-touches background set than the one-off run
    that produced the release. In the same 654-touch sample, every
    disagreement with the published set was of this kind: a touch where no
    frame reached the A_MIN gate at all (measured area/intensity of the
    published frame_idx was 0/0.00 or well under A_MIN), so it can only
    have been kept by this fallback draw, never by the deterministic path.

Because the winning-frame selection is a full-touch decode-and-argmax
(not the generic runner's fixed-first-10-frames-of-a-stream baseline), and
the background-keep is a genuine Bernoulli draw on a per-shard rng (not
the runner's deterministic running quota), this module reproduces the
filter inline in `_shard_frames`, exactly like `unit.py`/`sparsh.py`/
`tacquad.py`. `SPEC` declares `baseline=NoBaseline()`, so
`gsmp.baseline.needs_frames(NoBaseline())` is False and the runner never
buffers frames to build its own baseline -- `is_empty` is always False and
every FrameRecord this module yields is accepted as given.
`phash_dist=None` disables the runner's dedupe pass: with at most one
frame kept per touch, there is nothing to deduplicate against within a
capture, and `extract_rtm_video.py` never deduplicates across touches.

`dry_run_keys` still routes through `gsmp.runner.run()` purely for its
(capture, frame_idx) bookkeeping and RunResult shape -- it never exercises
the runner's filtering branches, neutralised by NoBaseline + phash_dist=None
as described above.
"""
from __future__ import annotations

import glob
import os
import random
import tempfile
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np
import pyarrow.parquet as pq

from gsmp import config
from gsmp.baseline import NoBaseline
from gsmp.filters import PIXEL_THRESH, grey_center
from gsmp.runner import FrameRecord
from gsmp.spec import SourceSpec, register

SPEC = register(SourceSpec(
    name="real_tactile_mnist",
    domain="real",
    gel_variant="markerless",
    license_repo="main",
    baseline=NoBaseline(),
    a_min=40,
    i_min=12.0,            # legacy/extract_rtm_video.py:56 -- NOT the 15 in
                            # make_parquet_v2.py:231 (dead code for this
                            # source; see module docstring for the full
                            # cross-check against the published release).
    channel_mode="rgb",    # legacy/extract_rtm_video.py never swaps channels
    phash_dist=None,       # at most 1 frame kept per touch; legacy never dedupes
    bg_keep_rate=0.015,    # legacy/extract_rtm_video.py:57 BG_RATE
    rng_seed=0,             # per-shard seed is hash(shard filename), not a
                             # fixed int -- see module docstring CAVEAT;
                             # recorded here only to satisfy SourceSpec's
                             # required field.
    notes=(
        "Standalone ingest (extract_rtm_video.py), not "
        "make_parquet_v2.process()/iter_real_tactile_mnist (dead code for "
        "this source -- wrong capture format, wrong I_MIN, wrong "
        "per-frame-Bernoulli shape; see module docstring). Per-touch "
        "full-clip-median baseline, deterministic single-peak-frame keep, "
        "Bernoulli background keep on a per-shard rng seeded by "
        "hash(shard filename) -- not recoverable, see docstring CAVEAT."
    ),
))

_RAW_ROOT = config.RAW_ROOT / "markerless" / "RealTactileMNIST"

#: legacy/extract_rtm_video.py:58 -- exactly one frame kept per touch.
_K_PER_TOUCH = 1


def _decode_video(vid_bytes: bytes) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Decode one touch's mp4 bytes to (RGB frames, grey-center crops).

    Mirrors the decode step every RTM legacy variant shares
    (`fr[:, :, ::-1]` BGR->RGB via a scratch temp file for cv2.VideoCapture).
    """
    fd, tmpf = tempfile.mkstemp(suffix=".mp4", prefix="_gsmp_rtm_")
    os.close(fd)
    try:
        with open(tmpf, "wb") as f:
            f.write(vid_bytes)
        cap = cv2.VideoCapture(tmpf)
        frames: List[np.ndarray] = []
        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                frames.append(bgr[:, :, ::-1])
        finally:
            cap.release()
    finally:
        try:
            os.remove(tmpf)
        except OSError:
            pass
    grays = [grey_center(f) for f in frames]
    return frames, grays


def _best_frame(grays: List[np.ndarray]) -> Optional[int]:
    """Legacy `best_frames(k=1)`: the single highest-scoring frame, or None.

    baseline = median of grey_center over EVERY frame of this touch (not a
    first-N prologue). score = intensity if area >= a_min else 0. Returns
    the argmax frame's index only if its score also clears i_min; None
    signals "fall through to the background-keep draw", matching legacy's
    `best_frames` returning `[]` for both "no frame has any area" and
    "the best frame's score is still below I_MIN".
    """
    baseline = np.median(np.stack(grays), axis=0)
    scores = np.zeros(len(grays), dtype=np.float64)
    for k, g in enumerate(grays):
        diff = np.abs(g - baseline)
        mask = diff > PIXEL_THRESH
        area = int(mask.sum())
        inten = float(diff[mask].mean()) if area > 0 else 0.0
        scores[k] = inten if area >= SPEC.a_min else 0.0
    if not (scores > 0).any():
        return None
    best = int(np.argsort(scores)[::-1][0])
    return best if scores[best] >= SPEC.i_min else None


def _shard_frames(path: str, limit: Optional[int]) -> Iterator[FrameRecord]:
    """Lazily filter one upstream parquet shard, legacy decision-for-decision.

    One shard = one rng scope (`hash(basename) & 0xFFFFFFFF`), matching
    `extract_rtm_video.py:process_shard`'s per-shard-file worker boundary --
    the rng is consumed in row order, then touch order within each row,
    only for touches whose deterministic peak-frame pick came back None
    (see `_best_frame`).
    """
    fname = os.path.basename(path)
    split = "test" if "test" in fname.lower() else "train"
    rng = random.Random(hash(fname) & 0xFFFFFFFF)

    pf = pq.ParquetFile(path)
    n_yielded = 0
    for batch in pf.iter_batches(batch_size=4):
        cols = batch.to_pydict()
        n = len(cols.get("label", cols.get("id", [])))
        for i in range(n):
            round_id = cols.get("id", [None] * n)[i]
            label = cols.get("label", [None] * n)[i]
            obj_id = cols.get("object_id", [None] * n)[i]
            videos = cols["sensor_video"][i] or []

            for tj, vid_struct in enumerate(videos):
                if vid_struct is None:
                    continue
                vid_bytes = (
                    vid_struct.get("bytes") if isinstance(vid_struct, dict) else None
                )
                if vid_bytes is None:
                    continue

                frames, grays = _decode_video(vid_bytes)
                if not frames:
                    continue

                fi = _best_frame(grays)
                if fi is None:
                    if rng.random() < SPEC.bg_keep_rate:
                        fi = rng.randint(0, len(frames) - 1)
                    else:
                        continue

                yield FrameRecord(
                    rgb=frames[fi],
                    capture=f"{round_id}_t{tj}",
                    obj_name=f"digit_{label}",
                    split=split,
                    episode=str(obj_id) if obj_id is not None else str(round_id),
                    frame_idx=fi,
                    extra=dict(digit_class=int(label) if label is not None else None),
                )
                n_yielded += 1
                if limit is not None and n_yielded >= limit:
                    return


def iter_units(limit: Optional[int] = None) -> Iterator[Tuple[str, Iterator[FrameRecord]]]:
    """Yield one (shard basename, frame_generator) unit per upstream parquet shard.

    One unit per shard mirrors legacy's own per-shard-file processing
    boundary (`process_shard`): the rng that drives the background-keep
    draw is scoped to one shard, not to one touch or one round. `limit`
    caps yielded frames per shard, matching `_shard_frames`'s own cap
    (mirroring `sparsh.py`/`tacquad.py`'s per-group cap semantics).
    """
    for path in sorted(glob.glob(str(_RAW_ROOT / "data" / "*.parquet"))):
        yield os.path.basename(path), _shard_frames(path, limit)


def dry_run_keys(limit: Optional[int] = None):
    """Return (kept_keys, n_seen) without writing anything."""
    from gsmp.runner import run

    res = run(SPEC, iter_units(limit=limit), dry_run=True)
    return res.kept_keys, res.n_seen
