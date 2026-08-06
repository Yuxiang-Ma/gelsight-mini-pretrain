"""FeelAnyForce -- loose PNGs per object under dataset/dataset/<obj>/tactile/.

tier-1: published rows carry capture + frame_idx, so tools/regress.py can
verify this migration against the release.

`iter_units` is derived from `legacy/make_parquet_v2.py::iter_feelanyforce`
(lines 308-367), not from a from-scratch reading of the raw tree layout.
Three details are load-bearing and were each checked against the
executable code, not the (partly stale) docstring:

  - BASELINE SCOPE. Legacy's docstring claims "1 image from each of the 42
    objects", but the executable line is `rng.sample(files, min(3,
    len(files)))` -- 3 images per object, not 1. The baseline is a single
    **global cross-object** median (not per-object -- FeelAnyForce is
    force-controlled, so every frame in an object folder has contact, and a
    per-object median would be poisoned by contact). It is built entirely
    from `objects/*/tactile/*.png` BEFORE any frame is considered for
    output, over every object with a non-empty `tactile/` dir, in sorted
    object order. Sampled baseline frames are drawn with `random.sample`
    (no replacement *within* one object's file list) and are NOT excluded
    from the output pool afterwards -- a frame can be both a baseline
    sample and a kept output row, exactly like `unit` (see
    `gsmp/sources/unit.py`'s docstring on this point) and unlike
    gelslam/tactile_tracking's BASE_FRAMES-consumed prologue.

  - BASELINE GREYSCALE DIFFERS FROM THE DIFF GREYSCALE. Baseline frames are
    read via `Image.open(...).convert("L")` -- PIL's luma-weighted
    grayscale (0.299 R + 0.587 G + 0.114 B) -- then centre-cropped. The
    per-frame diff, in the *second* pass, instead opens each candidate as
    RGB and greyscales with a plain channel mean
    (`rgb.mean(axis=2)`, i.e. `gsmp.filters.grey_center`), then centre-crops
    and diffs against that baseline. These are two different greyscale
    formulas applied to the same underlying pixels; reproducing only one of
    them (e.g. using `grey_center` for the baseline too) shifts every
    baseline value slightly and desyncs the area/intensity filter.

  - RNG SEQUENCING. `random.Random(0)` is seeded once and used for BOTH
    passes, in order: pass 1 calls `rng.sample(files, ...)` once per object
    with a non-empty `tactile/` dir (sorted object order, skipping objects
    whose `tactile/` dir is missing or empty -- those objects consume no
    rng draw at all); pass 2, after pass 1 has fully finished for every
    object, calls `rng.random()` once per candidate frame that FAILS the
    area/intensity filter (frames that pass never touch the rng). Only
    `tactile/` is read -- `tactile_nobg/` (same file count, present
    alongside `tactile/` and `depth/` in every object folder) is never
    opened.

  - DEDUPE LIVES IN `process()`, NOT IN `iter_feelanyforce`. The area+
    intensity filter above is only half of legacy's pipeline for this
    source: `legacy/make_parquet_v2.py::process()` (lines 784-916) wraps
    EVERY `SOURCE_ITERS[sub]()` generator -- including feelanyforce's --
    with a per-capture phash dedupe pass (`PHASH_DIST = 4`,
    `cap_phashes[-30:]`, reset whenever `meta["capture"]` changes), applied
    unconditionally regardless of `SKIP_EMPTY_FILTER[sub]`. Measured
    directly against the raw tree: the area/intensity filter alone passes
    ~95.8% of a 3,000-frame sample (FeelAnyForce is force-controlled --
    almost every frame has real contact), but published rows are only
    48,197 / 101,883 raw PNGs = 47.3% -- 0.473 / 0.958 = 0.494, matching a
    per-capture dedupe survival rate around 49%: consecutive indentation
    frames on a fixed object are near-identical, so roughly half dedupe
    away. `SPEC` therefore carries `phash_dist=4, phash_lookback=30` so
    `gsmp.runner.run()` performs this stage -- it must NOT be
    reimplemented inline here (that would double-apply it, since the
    runner unconditionally dedupes when `phash_dist is not None`,
    independent of `baseline=NoBaseline()`). This also means `iter_units`
    must yield one unit PER OBJECT (not one global unit spanning all 42
    objects, as an earlier draft did): the runner resets its phash window
    on every new unit, exactly mirroring legacy's per-capture reset, since
    `capture` here is the object.
"""
from __future__ import annotations

import os
import random
from typing import Iterator, Optional, Tuple

import numpy as np
from PIL import Image

from gsmp import config
from gsmp.baseline import NoBaseline
from gsmp.filters import PIXEL_THRESH, contact_metrics
from gsmp.runner import FrameRecord
from gsmp.spec import SourceSpec, register

SPEC = register(SourceSpec(
    name="feelanyforce",
    domain="real",
    gel_variant="markerless",
    license_repo="main",
    baseline=NoBaseline(),
    a_min=40,
    i_min=10.0,           # docs/imin_from_code.md: make_parquet_v2.py:319
    channel_mode="rgb",   # legacy never swaps channels for feelanyforce
    phash_dist=4,          # legacy/make_parquet_v2.py:46 PHASH_DIST -- applied
                           # by process() to EVERY source, not skipped for
                           # feelanyforce despite SKIP_EMPTY_FILTER=True; see
                           # "DEDUPE LIVES IN process()" in the module
                           # docstring. Measured: ~95.8% area/intensity pass
                           # rate x ~49% per-capture dedupe survival =~ the
                           # published 47.3% keep rate (48,197 / 101,883).
    phash_lookback=30,     # legacy: cap_phashes[-30:]
    bg_keep_rate=0.015,   # legacy/make_parquet_v2.py:320 BG
    rng_seed=0,           # legacy/make_parquet_v2.py:318 random.Random(0)
    notes=(
        "arXiv:2410.02048 (FeelAnyForce). make_parquet_v2.iter_feelanyforce: "
        "global cross-object baseline (3 PIL-L samples/object) + Bernoulli "
        "bg-keep on one rng instance, decision-for-decision. Per-capture "
        "phash dedupe (dist<=4, 30-frame window) is applied by process(), "
        "reproduced here via the runner's dedupe stage."
    ),
))

_ROOT = config.RAW_ROOT / "markerless" / "FeelAnyForce" / "dataset" / "dataset"
_N_BASELINE_SAMPLE = 3          # legacy: min(3, len(files))
_MIN_BASELINE_FRAMES = 30       # legacy: `if len(grays) < 30: return`


def _grey_center_L(path: str) -> Optional[np.ndarray]:
    """PIL-L (luma-weighted) greyscale, centre-cropped, float32.

    Deliberately NOT `gsmp.filters.grey_center`: legacy builds the baseline
    via `Image.open(...).convert("L")`, a different formula from the plain
    RGB-channel mean used for the per-frame diff. See module docstring.
    """
    try:
        im = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    except Exception:
        return None
    h, w = im.shape
    return im[h // 4:3 * h // 4, w // 4:3 * w // 4]


def _build_baseline(objects, rng: random.Random) -> Optional[np.ndarray]:
    """Pass 1: global cross-object median from up to 3 samples per object.

    Must run to completion -- and consume its `rng.sample` draws for every
    qualifying object, in order -- before pass 2 (`_object_frames`) makes
    its first `rng.random()` call, since both share one rng instance.
    """
    grays = []
    for obj in objects:
        p = _ROOT / obj / "tactile"
        if not p.is_dir():
            continue
        files = sorted(os.listdir(p))
        if not files:
            continue
        for fn in rng.sample(files, min(_N_BASELINE_SAMPLE, len(files))):
            g = _grey_center_L(str(p / fn))
            if g is not None:
                grays.append(g)
    if len(grays) < _MIN_BASELINE_FRAMES:
        return None
    return np.median(np.stack(grays), axis=0)


def _object_frames(
    obj: str,
    p,
    baseline: np.ndarray,
    rng: random.Random,
    counter: list,
    limit: Optional[int],
) -> Iterator[FrameRecord]:
    """Pass 2, one object: iterate `tactile/` in file-sorted order.

    `rng.random()` is drawn only for frames that FAIL the area/intensity
    filter, in exactly this iteration order, on the same rng instance
    `_build_baseline` already advanced. `counter` is a 1-element list
    shared across every object's generator so `limit` caps total yielded
    frames across the whole source, matching `unit.py`'s convention, even
    though (unlike `unit`) each object here is its own unit.
    """
    obj_name = obj.split("_")[0]
    for fi, fn in enumerate(sorted(os.listdir(p))):
        try:
            rgb = np.array(Image.open(str(p / fn)).convert("RGB"))
        except Exception:
            continue
        area, inten = contact_metrics(rgb, baseline, PIXEL_THRESH)
        passes = area >= SPEC.a_min and inten >= SPEC.i_min
        if not passes and rng.random() >= SPEC.bg_keep_rate:
            continue

        yield FrameRecord(
            rgb=rgb,
            capture=obj,
            obj_name=obj_name,
            split="train",
            episode=obj,
            frame_idx=fi,
        )
        counter[0] += 1
        if limit is not None and counter[0] >= limit:
            return


def iter_units(limit: Optional[int] = None) -> Iterator[Tuple[str, Iterator[FrameRecord]]]:
    """Yield one (capture, frame_generator) unit per object, in sorted order.

    All 42 objects share one global baseline (see `_build_baseline`),
    computed to completion here -- consuming its `rng.sample` draws --
    before the first unit is yielded. One unit per object (rather than one
    global unit spanning everything, as `unit.py` uses for its single
    zarr array) is required so `gsmp.runner.run()` resets its per-capture
    phash dedupe window on every object change, matching legacy's
    `cap_phashes` reset on `meta["capture"]` change in `process()`.
    """
    objects = sorted(d for d in os.listdir(_ROOT) if (_ROOT / d).is_dir())
    rng = random.Random(SPEC.rng_seed)
    baseline = _build_baseline(objects, rng)
    if baseline is None:
        return
    counter = [0]
    for obj in objects:
        p = _ROOT / obj / "tactile"
        if not p.is_dir():
            continue
        if limit is not None and counter[0] >= limit:
            return
        yield obj, _object_frames(obj, p, baseline, rng, counter, limit)


def dry_run_keys(limit: Optional[int] = None):
    """Return (kept_keys, n_seen) without writing anything."""
    from gsmp.runner import run

    res = run(SPEC, iter_units(limit=limit), dry_run=True)
    return res.kept_keys, res.n_seen
