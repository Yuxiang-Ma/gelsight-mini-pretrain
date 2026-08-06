"""py3DCal -- sphere-indentation grid, one loose PNG per (x, y, z) probe.

tier-1: published rows carry capture + frame_idx, so tools/regress.py can
verify this migration against the release.

`threedcal` IS a `make_parquet_v2.process()` source
(`SOURCE_ITERS["threedcal"] = iter_threedcal`,
`legacy/make_parquet_v2.py:529-576`), unlike `unit` -- but it is one of the
seven sources marked `SKIP_EMPTY_FILTER["threedcal"] = True`
(`legacy/make_parquet_v2.py:677`), meaning `iter_threedcal` decides
keep/drop for the contact-validity rule itself; `process()` only forces
`is_empty = False` for every frame it receives (`legacy/make_parquet_v2.py:
845-846`) and then still runs its live stride limiter and per-capture phash
dedupe on top (`legacy/make_parquet_v2.py:857-878`). Three details are
load-bearing:

  - BASELINE. PIPELINE.md used to describe threedcal's baseline as "a
    cross-image median over a random 200-frame sample". That is false. py3DCal
    ships an explicit gel-at-rest reference, `blank_images/blank.png`
    (`legacy/make_parquet_v2.py:541,545-547`), and the legacy iterator reads
    it directly -- no sampling, no median, and no randomness in the baseline
    at all. This module uses `gsmp.baseline.ExplicitReference` for exactly
    that (`tests/test_baseline.py` covers the strategy in isolation); see
    `_baseline()` below.

  - RNG. `legacy/make_parquet_v2.py:537` seeds `random.Random(0)`. Unlike
    `tacquad_mini`/`faf_force_estimation` in the same file, this rng is
    *never* used to sample baseline frames (there is nothing to sample --
    the baseline is the shipped blank.png). Its only use is the background
    keep coin flip at line 565: for a frame that fails the area/intensity
    rule, `rng.random()` is drawn and the frame is kept iff the draw is
    `< BG` (0.015). A frame that *passes* the rule is kept unconditionally
    and never touches the rng. Reproducing the published set exactly
    requires drawing in precisely this order, on precisely the frames that
    fail, in the same iteration order legacy used
    (`sorted(glob(f"{probe_dir}/*.png"))`, i.e. lexicographic sort on the
    full path string -- NOT numeric sort on the leading index, and NOT
    grouped by capture; consecutive files interleave many different
    (x, y) captures, see `iter_units` below).

  - CAPTURE GROUPING FOR DEDUPE. `legacy/make_parquet_v2.py:822-830` resets
    `cap_phashes` (and the stride/baseline state, irrelevant here since
    `skip_empty=True`) every time `meta["capture"]` differs from the
    previous frame's capture -- a *consecutive-run* reset, not a global
    group-by-capture reset. Because probe_images/*.png filenames are
    `{global_index}_X{x}Y{y}Z{z}.png` and sort lexicographically by
    `global_index`-as-a-string (not numerically), frames of one (x, y)
    capture are usually contiguous runs but are frequently interrupted by
    single frames from a different capture (verified empirically: e.g.
    `x10.5_y4.0` x10, `x16.5_y0.0` x1, `x10.5_y4.0` x10, `x16.5_y0.0` x1,
    `x11.0_y4.0` x7, ... in the actual sorted file list). `gsmp.runner.run`
    resets its own `cap_phashes` once per *unit* (`_unit_id, frames in
    units`), not per capture value seen inside a unit -- so reproducing
    legacy's per-transition reset requires `iter_units` to yield one unit
    per *consecutive run* of same-capture frames, via
    `itertools.groupby(frames, key=capture)`, not one unit per distinct
    capture.

Because the contact-validity + bg-keep decision happens inline in
`_threedcal_frames`, `SPEC` declares `baseline=NoBaseline()`:
`gsmp.baseline.needs_frames(NoBaseline())` is False, so the runner never
buffers frames to build its own baseline, `base` stays `None` for every
record and `is_empty` is therefore always False -- mirroring
`legacy/make_parquet_v2.py:845-846`'s `if skip_empty: is_empty = False`.
Unlike `unit` (a standalone script that never dedupes), `threedcal` runs
through legacy `process()`, which *does* apply per-capture-run phash dedupe
unconditionally (`legacy/make_parquet_v2.py:872-878`) -- so, unlike
`unit.py`, this module leaves `phash_dist`/`phash_lookback` at the SPEC
defaults (4 / 30) rather than disabling dedupe.

`dry_run_keys` still routes through `gsmp.runner.run()` (mirroring
gelslam/tactile_tracking/unit) purely for its (capture, frame_idx)
bookkeeping and RunResult shape -- it never exercises the runner's baseline
or bg-keep-quota branches for this source, since those are neutralised by
NoBaseline as described above; only the runner's phash dedupe is live.
"""
from __future__ import annotations

import glob
import itertools
import os
import random
import re
from typing import Iterator, Optional, Tuple

import numpy as np
from PIL import Image

from gsmp import config
from gsmp.baseline import ExplicitReference, NoBaseline
from gsmp.filters import passes_filter
from gsmp.runner import FrameRecord
from gsmp.spec import SourceSpec, register

SPEC = register(SourceSpec(
    name="threedcal",
    domain="real",
    gel_variant="markerless",
    license_repo="main",
    baseline=NoBaseline(),
    a_min=40,
    i_min=10.0,            # docs/imin_from_code.md: legacy/make_parquet_v2.py:538
    channel_mode="rgb",    # legacy never swaps channels in process()
    phash_dist=4,          # legacy DOES dedupe threedcal (process() line 872-878)
    phash_lookback=30,     # legacy/make_parquet_v2.py:874 cap_phashes[-30:]
    bg_keep_rate=0.015,    # legacy/make_parquet_v2.py:539 BG
    rng_seed=0,            # legacy/make_parquet_v2.py:537 random.Random(0)
    notes=(
        "arXiv py3DCal sphere-indentation grid. make_parquet_v2.process() "
        "source with SKIP_EMPTY_FILTER=True: contact-validity + bg-keep "
        "filtering happens inline against an explicit blank.png reference, "
        "not a sampled median; process() still applies its per-capture-run "
        "phash dedupe on top."
    ),
))

_ROOT = config.RAW_ROOT / "markerless" / "3DCal" / "gsmini_calibration_data"
_BLANK_PATH = _ROOT / "blank_images" / "blank.png"
_PROBE_DIR = _ROOT / "probe_images"

_NAME_RE = re.compile(r"(\d+)_X([0-9.]+)Y([0-9.]+)Z([0-9.]+)\.png")


def _baseline() -> Optional[np.ndarray]:
    """Explicit gel-at-rest reference: central 50% crop of blank.png.

    Not a median over any sample -- see module docstring. Returns None
    (matching legacy's `if not os.path.isfile(blank_path): return`, i.e. no
    frames at all) when the reference is missing.
    """
    if not os.path.isfile(_BLANK_PATH):
        return None
    return ExplicitReference(_BLANK_PATH).compute([])


def _threedcal_frames(limit: Optional[int]) -> Iterator[FrameRecord]:
    """Lazily filter probe_images/*.png, legacy decision-for-decision.

    Iteration order is `sorted(glob(...))` -- lexicographic on the full
    path string, exactly matching `legacy/make_parquet_v2.py:549` -- because
    both the rng draw order (for frames failing the validity rule) and the
    capture-run boundaries `iter_units` groups on depend on it.
    """
    baseline = _baseline()
    if baseline is None:
        return

    rng = random.Random(SPEC.rng_seed)
    n_yielded = 0

    for fp in sorted(glob.glob(str(_PROBE_DIR / "*.png"))):
        m = _NAME_RE.search(os.path.basename(fp))
        if not m:
            continue
        idx_s, x_s, y_s, z_s = m.groups()
        x_mm, y_mm, z_mm = float(x_s), float(y_s), float(z_s)

        try:
            rgb = np.array(Image.open(fp).convert("RGB"))
        except Exception:
            continue

        passes = passes_filter(rgb, baseline, SPEC.a_min, SPEC.i_min)
        # Only a failing frame draws from the rng -- a passing frame is
        # kept unconditionally, matching `if not passes and rng.random()
        # >= BG: continue` (legacy/make_parquet_v2.py:565).
        if not passes and rng.random() >= SPEC.bg_keep_rate:
            continue

        yield FrameRecord(
            rgb=rgb,
            capture=f"x{x_mm}_y{y_mm}",
            split="train",
            frame_idx=int(idx_s),
            extra=dict(x_mm=x_mm, y_mm=y_mm, z_mm=z_mm),
        )
        n_yielded += 1
        if limit is not None and n_yielded >= limit:
            return


def iter_units(limit: Optional[int] = None) -> Iterator[Tuple[str, Iterator[FrameRecord]]]:
    """Yield one unit per consecutive run of same-capture frames.

    threedcal has no natural per-capture grouping in its source layout --
    every probe PNG lives loose in one flat directory, and legacy's dedupe
    reset triggers on a capture *transition* in iteration order, not on a
    capture *value* (see module docstring). `itertools.groupby` reproduces
    that transition-based reset: `gsmp.runner.run` resets its own
    `cap_phashes` once per unit, so one unit per consecutive run gives it
    the same reset points legacy's `if cap != cur_capture` produced.
    """
    frames = _threedcal_frames(limit)
    for capture, group in itertools.groupby(frames, key=lambda rec: rec.capture):
        yield capture, group


def dry_run_keys(limit: Optional[int] = None):
    """Return (kept_keys, n_seen) without writing anything."""
    from gsmp.runner import run

    res = run(SPEC, iter_units(limit=limit), dry_run=True)
    return res.kept_keys, res.n_seen
