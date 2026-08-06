"""Sparsh GelSight -- Meta's TacBench pickles, split by indenter shape.

tier-1: published rows carry capture + frame_idx, so tools/regress.py can
verify this migration against the release.

`sparsh` is the one source in the **nc** repo, not the main one -- see
`SPEC.license_repo="nc"`, which routes `gsmp.config.published_dir()` (and
`tools/regress.py`, which passes `SPEC.license_repo` straight through) to
`PARQUET_NC` instead of `PARQUET_MAIN`.

It is also, like `unit`, a standalone script (`legacy/ingest_sparsh.py`),
not a `make_parquet_v2.process()` source, and its own docstring states the
threshold outright: `I_MIN = 12` at `legacy/ingest_sparsh.py:46` -- the
`docs/imin_from_code.md` calibration anchor for the empirical i_min work.
Three details are load-bearing:

  - RAW LAYOUT. `mini_data/markerless_nc/SparshGelSight/{flat,sharp,sphere}/
    batch_*/dataset_gelsight_NN.pkl`. Each pkl is a Python list of
    JPEG-encoded bytes (~5000 frames). Sibling `org_dataset_gelsight_NN.pkl`
    files are the *pre-filter* raw data and MUST be skipped
    (`legacy/ingest_sparsh.py:64-67`, `fname.startswith("org_")`).

  - SPLIT IS THE INDENTER, NOT train/val. `split=indenter` (flat/sharp/
    sphere), and the published shards are named accordingly
    (`sparsh/flat-00000-of-00001.parquet` etc.,
    `legacy/ingest_sparsh.py:131,157`). Do not "normalise" this to
    train/val -- it is deliberate and matches the release.

  - BASELINE + BG-KEEP, PER INDENTER GROUP, ONE RNG INSTANCE.
    `legacy/ingest_sparsh.py:77-102` loads *every* frame of one indenter
    (all batches, all files -- 7/8/24 pkls, 29,809 / 35,356 / 109,701
    frames) into memory first, so the total count `n` is known before any
    sampling happens. The baseline is the per-pixel median of
    `grey_center` over `rng.sample(range(n), min(N_BASELINE=100, n))`
    frames drawn *uniformly across the whole indenter* -- not the first N
    frames of a stream, and not scoped to one pkl/batch. Baseline-sample
    frames are NOT excluded from the candidate pool afterwards: exactly
    like `unit` (see `src/gsmp/sources/unit.py`'s docstring, "property 2"),
    baseline sampling and output eligibility are independent draws over the
    same population. The background keep
    (`legacy/ingest_sparsh.py:121: elif rng.random() < BG_RATE`) is a
    genuine Bernoulli draw on the *same* `rng` instance used for the
    baseline sample, in strict iteration order over all frames of the
    indenter -- not the deterministic running quota
    (`n_empty_kept < bg_keep_rate * max(n_kept, 1)`) `gsmp.runner.run()`
    implements. Because both diverge from what the generic runner does,
    this module reproduces the filter inline (`_group_frames`) instead of
    relying on `gsmp.runner.run()`'s baseline/bg-keep machinery, exactly as
    `unit.py` does. `SPEC` declares `baseline=NoBaseline()`:
    `gsmp.baseline.needs_frames(NoBaseline())` is False, so `base` stays
    `None` for every record the runner sees and `is_empty` is therefore
    always False -- the runner accepts every `FrameRecord` this module
    yields exactly as given. `phash_dist=None` disables the runner's dedupe
    pass because `legacy/ingest_sparsh.py` never deduplicates.

  - CHANNEL ORDER. `legacy/ingest_sparsh.py` never touches channel order --
    it writes `image=b`, the *original* JPEG bytes, verbatim
    (`legacy/ingest_sparsh.py:127`). The published parquet's pixel bytes
    were later modified in place by a *separate* post-hoc step
    (`finalize_v9.sh` step 5's commit message: "sparsh conditional per-image
    R<->B swap (mixed RGB/BGR per Facebook upstream)"), which rewrote image
    bytes but did not touch which `(capture, frame_idx)` keys are kept --
    confirmed by reading `finalize_v9.sh`, which only re-uploads the
    already-existing parquet files, it does not re-run `ingest_sparsh.py`
    or any other row-selection pass. Since `tools/regress.py` only compares
    `(capture, frame_idx)` keys, not pixel bytes, this module matches
    `ingest_sparsh.py` itself: `channel_mode="rgb"` (never swap).

  KNOWN GAP -- THE RNG SEED IS NOT RECOVERABLE. `legacy/ingest_sparsh.py:78`
  seeds each indenter's rng from `random.Random(hash(indenter) &
  0xFFFFFFFF)`, i.e. Python's built-in `hash()` of the indenter name. As of
  CPython 3.3, `hash(str)` is randomised per-process (SipHash keyed from
  `os.urandom()` at interpreter start) unless `PYTHONHASHSEED` is fixed in
  the environment. I searched for any pin of `PYTHONHASHSEED` in this
  machine's shell profiles (`~/.bashrc`, `~/.profile`, `~/.zshrc`),
  `/etc/environment`, every `conda`/`venv` activation script under
  `$HOME`, and every orchestration shell script that shipped alongside
  `ingest_sparsh.py` (`finalize*.sh`) -- none sets it, and the script itself
  never logs the resolved seed. This module calls `hash(indenter) &
  0xFFFFFFFF` exactly as legacy did (the most faithful reproduction of the
  actual code, rather than substituting some other arbitrary deterministic
  formula that would be no more likely to match); with hash randomisation
  active by default in this environment, that reproduces the DETERMINISTIC
  area/intensity-pass set exactly (verified: of 8,056 published `flat`
  rows, 7,680 pass the filter unconditionally against a baseline rebuilt
  from an independently-drawn 100-frame sample -- the median is robust to
  *which* 100 frames were drawn) but draws a *different* 1.5%
  background-keep set than the one actually published, because that draw
  depends on the exact, never-recorded, per-process `hash("flat")` value
  from the one-off run that produced the release. This is a strictly worse
  version of the caveat `unit.py` documents for its own rng: `unit`'s seed
  is a literal constant (`random.Random(20260520)`), fully reproducible;
  `sparsh`'s is derived from unrecoverable interpreter state. See the
  regression run recorded in the commit that adds this file for the actual
  (non-exact) outcome.

`dry_run_keys` still routes through `gsmp.runner.run()` (mirroring
gelslam/tactile_tracking/unit/threedcal) purely for its (capture,
frame_idx) bookkeeping and RunResult shape -- it never exercises the
runner's filtering branches for this source, since those are neutralised by
NoBaseline + phash_dist=None as described above.
"""
from __future__ import annotations

import glob
import io
import os
import pickle
import random
from typing import Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image

from gsmp import config
from gsmp.baseline import NoBaseline
from gsmp.filters import PIXEL_THRESH, grey_center
from gsmp.runner import FrameRecord
from gsmp.spec import SourceSpec, register

SPEC = register(SourceSpec(
    name="sparsh",
    domain="real",
    gel_variant="markerless",
    license_repo="nc",     # NC repo: yxma/gelsight-mini-pretrain-nc (CC-BY-NC-4.0)
    baseline=NoBaseline(),
    a_min=40,
    i_min=12.0,             # docs/imin_from_code.md: legacy/ingest_sparsh.py:46
    channel_mode="rgb",     # legacy never swaps channels in ingest_sparsh.py
    phash_dist=None,        # legacy never dedupes sparsh
    bg_keep_rate=0.015,     # legacy/ingest_sparsh.py:47 BG_RATE
    notes=(
        "facebook/SparshGelSight, CC-BY-NC-4.0. Standalone ingest "
        "(ingest_sparsh.py), not make_parquet_v2.process(): per-indenter "
        "random-sample baseline + Bernoulli bg-keep on one rng instance "
        "seeded from hash(indenter) -- see module docstring, seed not "
        "recoverable."
    ),
))

_RAW_BASE = config.RAW_ROOT / "markerless_nc" / "SparshGelSight"
_INDENTERS = ("flat", "sharp", "sphere")
_N_BASELINE = 100


def _collect_pkl_files(indenter: str) -> List[Tuple[str, int, str]]:
    """(batch, file_idx, path) for one indenter, skipping org_*.pkl.

    Mirrors legacy/ingest_sparsh.py:61-74 exactly, including the sort (glob
    is not order-stable, and rng.sample's baseline draw depends on `n`
    being computed over frames appended in this same order).
    """
    out: List[Tuple[str, int, str]] = []
    pattern = str(_RAW_BASE / indenter / "batch_*" / "dataset_gelsight_*.pkl")
    for pkl in sorted(glob.glob(pattern)):
        fname = os.path.basename(pkl)
        if fname.startswith("org_"):
            continue
        batch = os.path.basename(os.path.dirname(pkl))
        file_idx = int(fname.split("_")[-1].split(".")[0])
        out.append((batch, file_idx, pkl))
    return out


def _grey_center_from_jpeg(b: bytes) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Decode + central-crop-greyscale one JPEG. (None, None) on decode failure."""
    try:
        rgb = np.array(Image.open(io.BytesIO(b)).convert("RGB"))
    except Exception:
        return None, None
    return rgb, grey_center(rgb)


def _group_frames(indenter: str, limit: Optional[int]) -> Iterator[FrameRecord]:
    """Lazily filter one indenter's frames, legacy decision-for-decision.

    Legacy itself is not streaming here: it loads every frame's JPEG bytes
    for the whole indenter into memory before it can know `n` for
    `rng.sample(range(n), ...)`. This function does the same -- there is no
    way to know `n` (needed to seed the baseline sample) without first
    unpickling every file.
    """
    rng = random.Random(hash(indenter) & 0xFFFFFFFF)

    pkls = _collect_pkl_files(indenter)
    all_frames: List[Tuple[str, int, int, bytes]] = []
    for batch, file_idx, pkl in pkls:
        with open(pkl, "rb") as f:
            data = pickle.load(f)
        for fi, b in enumerate(data):
            all_frames.append((batch, file_idx, fi, b))
    n = len(all_frames)
    if n == 0:
        return

    base_idxs = rng.sample(range(n), min(_N_BASELINE, n))
    grays = []
    for i in base_idxs:
        _, g = _grey_center_from_jpeg(all_frames[i][3])
        if g is not None:
            grays.append(g)
    baseline = np.median(np.stack(grays), axis=0)

    n_yielded = 0
    for batch, file_idx, frame_idx, b in all_frames:
        rgb, g = _grey_center_from_jpeg(b)
        if g is None:
            continue

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

        yield FrameRecord(
            rgb=rgb,
            capture=f"{indenter}_{batch}_f{file_idx:02d}",
            obj_name=f"indenter_{indenter}",
            split=indenter,               # split-by-indenter, not train/val
            episode=f"{batch}_f{file_idx:02d}",
            frame_idx=frame_idx,
            extra=dict(indenter=indenter),
        )
        n_yielded += 1
        if limit is not None and n_yielded >= limit:
            return


def iter_units(limit: Optional[int] = None) -> Iterator[Tuple[str, Iterator[FrameRecord]]]:
    """Yield one unit per indenter group (flat, sharp, sphere).

    One unit per group mirrors legacy's own per-indenter processing
    boundary (`process_indenter`): the baseline and the rng instance are
    both scoped to one indenter, not to one capture. `limit` caps yielded
    *frames per group*, matching `_group_frames`'s own cap, since capping
    "units" here would only ever mean "stop after 0-3 indenters" -- not a
    useful knob for a 3-group source.
    """
    n = 0
    for indenter in _INDENTERS:
        yield indenter, _group_frames(indenter, limit)
        n += 1


def dry_run_keys(limit: Optional[int] = None):
    """Return (kept_keys, n_seen) without writing anything."""
    from gsmp.runner import run

    res = run(SPEC, iter_units(limit=limit), dry_run=True)
    return res.kept_keys, res.n_seen
