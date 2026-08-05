"""The generic ingest driver.

One code path for every source. What used to vary per source -- baseline
recipe, thresholds, dedupe strictness, licence destination -- now arrives as
a SourceSpec, so this module has no per-source branches.

Pipeline order matches docs/PIPELINE.md:
    iter_frames -> baseline -> contact filter -> channel norm
                -> phash dedupe -> budget cap -> parquet
"""
from __future__ import annotations

import dataclasses
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from gsmp import filters
from gsmp.baseline import needs_frames
from gsmp.encode import encode_jpeg
from gsmp.spec import SourceSpec
from gsmp.writer import ShardWriter

Key = Tuple[str, int]


@dataclasses.dataclass
class FrameRecord:
    """One decoded source frame, ready to filter."""

    rgb: np.ndarray
    capture: str = ""
    obj_name: str = ""
    split: str = "train"
    episode: str = ""
    frame_idx: int = 0
    extra: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class RunResult:
    kept_keys: Set[Key]
    n_seen: int
    n_kept: int
    n_empty_kept: int
    n_dup_dropped: int


def _row(spec: SourceSpec, rec: FrameRecord, rgb: np.ndarray) -> Dict[str, Any]:
    h, w = rgb.shape[:2]
    row = {
        "image": encode_jpeg(rgb),
        "image_format": "jpeg",
        "source": spec.name,
        "domain": spec.domain,
        "markered": spec.markered,
        "gel_variant": spec.gel_variant,
        "capture": rec.capture,
        "split": rec.split,
        "episode": rec.episode,
        "frame_idx": rec.frame_idx,
        "obj_name": rec.obj_name,
        "height": h,
        "width": w,
    }
    row.update(rec.extra)
    return row


#: Frames consumed per capture to build the baseline. They are NOT emitted.
BASE_FRAMES = 10


def run(
    spec: SourceSpec,
    units: Iterable[Tuple[str, Iterable[FrameRecord]]],
    writer: Optional[ShardWriter] = None,
    dry_run: bool = False,
) -> RunResult:
    """Process `units`, where each unit is (unit_id, iterable_of_frames).

    Mirrors legacy make_parquet_v2.process() decision-for-decision. Four
    properties are load-bearing and were each verified against the published
    release; none may be "simplified":

    1. STREAMING. Frames are consumed one at a time and only BASE_FRAMES
       greyscale centre-crops are held. Buffering a whole unit is not an
       option: GelSLAM's largest episode is 45,557 frames, which is 10.5 GB
       as uint8 RGB.

    2. BASELINE FRAMES ARE CONSUMED. The first BASE_FRAMES frames of each
       capture build the baseline and are never emitted. Every published
       GelSLAM capture has minimum frame_idx == 10, which is this rule's
       fingerprint.

    3. THE BACKGROUND KEEP IS A DETERMINISTIC QUOTA, NOT A COIN FLIP. A
       failing frame is kept only while n_empty_kept < bg_keep_rate *
       max(n_kept, 1). Legacy used EMPTY_BUDGET as a running ratio cap. A
       Bernoulli draw at the same rate keeps a different set of frames, so
       for sources whose baseline needs no sampling this pipeline is fully
       deterministic and regression must match EXACTLY.

    4. DEDUPE STATE RESETS PER CAPTURE. Legacy cleared cap_phashes on every
       capture change; a global window would suppress across captures.

    Legacy also had a live stride rate-limiter, deliberately not reproduced:
    it only rises above 1.0 once n_kept exceeds BUDGET * 0.95 (190,000), and
    the largest published source is sim_starstruck at 166,104, so it never
    activated in the build that produced the release.
    """
    kept: Set[Key] = set()
    n_seen = 0
    n_kept = 0
    n_empty_kept = 0
    n_dup = 0

    for _unit_id, frames in units:
        cap_buffer: List[np.ndarray] = []
        base: Optional[np.ndarray] = None
        cap_phashes: List[int] = []
        uses_baseline = needs_frames(spec.baseline)

        for rec in frames:
            if n_kept >= spec.budget:
                break

            # (2) Consume the head of each capture to form the baseline.
            if uses_baseline and base is None:
                cap_buffer.append(filters.grey_center(rec.rgb))
                n_seen += 1
                if len(cap_buffer) >= BASE_FRAMES:
                    base = np.median(np.stack(cap_buffer, axis=0), axis=0)
                    cap_buffer = []
                continue

            n_seen += 1

            if base is None:
                is_empty = False
            else:
                area, inten = filters.contact_metrics(rec.rgb, base)
                is_empty = area < spec.a_min or inten < spec.i_min

            # (3) Deterministic quota, not a Bernoulli draw.
            if is_empty and n_empty_kept >= spec.bg_keep_rate * max(n_kept, 1):
                continue

            rgb = filters.maybe_swap_channels(rec.rgb, spec.channel_mode)

            # (4) Dedupe window is per-capture.
            if spec.dedupe_enabled:
                h = filters.phash(rgb)
                window = cap_phashes[-spec.phash_lookback:]
                if any(filters.hamming(h, p) <= spec.phash_dist for p in window):
                    n_dup += 1
                    continue
                cap_phashes.append(h)

            kept.add((rec.capture, rec.frame_idx))
            n_kept += 1
            if is_empty:
                n_empty_kept += 1
            if not dry_run and writer is not None:
                writer.add(_row(spec, rec, rgb))

        if n_kept >= spec.budget:
            break

    return RunResult(
        kept_keys=kept,
        n_seen=n_seen,
        n_kept=n_kept,
        n_empty_kept=n_empty_kept,
        n_dup_dropped=n_dup,
    )
