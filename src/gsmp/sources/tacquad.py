"""TacQuad -- quad-sensor benchmark, GelSight Mini stream only.

tier-1: published rows carry capture + frame_idx, so tools/regress.py can
verify this migration against the release.

TacQuad (arXiv:2503.16578) records four tactile sensors simultaneously
(`gelsight`, `digit`, `duragel`, `tac3d`) plus external scene cameras
(`img_*`). Only the `gelsight/*.png` subdirectory belongs in this corpus --
pulling frames from `digit/`, `duragel/`, or `tac3d/` would silently mix in
a physically different sensor.

`tacquad` is NOT a `make_parquet_v2.process()` source, and it is NOT
`iter_tacquad_mini()` (`make_parquet_v2.py:457-528`, `I_MIN=10`,
`source="tacquad_mini"`) either -- that iterator is dead code, superseded
before publication (its own successor's docstring says so verbatim) and
never reached the released files; see docs/imin_from_code.md's "tacquad
conflict" section for the full cross-check against the published shard
names/row counts. The published `tacquad` config was produced by the
standalone script `legacy/ingest_tacquad_full.py`, `I_MIN=12`, one shard
per domain (`tacquad/{domain}-00000-of-00001.parquet`).

Two details are load-bearing, mirroring gsmp.sources.unit's precedent:

  - BASELINE. `legacy/ingest_tacquad_full.py:98-107` builds one reference
    frame *per domain* (not per object/capture), from the median of
    `N_BASELINE=100` frames sampled uniformly at random from every frame in
    that domain (`rng.sample(range(n), 100)`), where `n` ranges from ~6,000
    (data_fine) to ~12,600 (data_indoor). This does not fit
    `gsmp.baseline.PerGroupMedian`'s numpy-Generator-based sampling: legacy
    uses `random.Random`, not `numpy.random.default_rng`, and the same rng
    instance is reused afterwards for the background-keep draw (see below),
    so the draw order matters and must be reproduced with stdlib `random`.
  - BACKGROUND KEEP. `legacy/ingest_tacquad_full.py:130` is a genuine
    Bernoulli draw, `elif rng.random() < BG_RATE`, on the SAME rng instance
    used for the baseline sample -- not the deterministic running quota
    (`n_empty_kept < bg_keep_rate * max(n_kept, 1)`) `gsmp.runner.run()`
    implements. The rng for each domain is seeded with
    `random.Random(hash(domain) & 0xFFFFFFFF)` (line 88) -- i.e. keyed off
    the domain name string, not a fixed integer literal like `unit`'s
    `random.Random(20260520)`.

Because both of these diverge from what the generic runner does, this
module reproduces the filter inline in `_domain_frames`, exactly like
`gsmp.sources.unit._unit_frames`. The SPEC declares `baseline=NoBaseline()`:
`gsmp.baseline.needs_frames(NoBaseline())` is False, so the runner never
buffers frames to build its own baseline, `base` stays `None` for every
record, and `is_empty` is therefore always False -- the runner accepts
every FrameRecord this module yields exactly as given. `phash_dist=None`
disables the runner's dedupe pass for the same reason: legacy never
deduplicates tacquad.

`dry_run_keys` still routes through `gsmp.runner.run()` (mirroring
unit/gelslam/tactile_tracking) purely for its (capture, frame_idx)
bookkeeping and RunResult shape -- it never exercises the runner's
filtering branches for this source, since those are neutralised by
NoBaseline + phash_dist=None as described above.

CAVEAT -- `hash(domain) & 0xFFFFFFFF` is not reproducible across Python
processes. CPython randomizes `hash()` of `str` objects per-process
(SipHash keyed by a value chosen at interpreter start, unless
`PYTHONHASHSEED` is set in the environment that launched the process); no
`PYTHONHASHSEED` pin was found anywhere in this repo, its predecessor
MultimodalData tree, or the shell/conda environment. This means the exact
rng draw sequence -- both the 100-frame baseline sample AND the background
Bernoulli keeps -- used for the published release depended on whatever
random per-process hash seed happened to be live when
`legacy/ingest_tacquad_full.py` was actually executed, which cannot be
recovered from the code alone. This module reproduces the formula
verbatim (`hash(domain) & 0xFFFFFFFF`), which is the best-faithful
transcription of the spec; see the module's task report for the measured
regression gap this causes and why it is not a threshold-tuning problem.
"""
from __future__ import annotations

import glob
import os
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
    name="tacquad",
    domain="real",
    gel_variant="markerless",
    license_repo="main",
    baseline=NoBaseline(),
    a_min=40,
    i_min=12.0,           # docs/imin_from_code.md: legacy/ingest_tacquad_full.py:50
    channel_mode="rgb",   # legacy never swaps channels for tacquad
    phash_dist=None,      # legacy never dedupes tacquad
    bg_keep_rate=0.015,   # legacy/ingest_tacquad_full.py:51 BG_RATE
    rng_seed=0,            # per-domain seed is hash(domain), not a fixed int
                            # -- see module docstring CAVEAT; recorded here
                            # only to satisfy SourceSpec's required field.
    notes=(
        "arXiv:2503.16578. Standalone ingest (ingest_tacquad_full.py), not "
        "make_parquet_v2.process() / iter_tacquad_mini() (dead code, "
        "I_MIN=10, never published -- docs/imin_from_code.md). Per-domain "
        "random-sample baseline + Bernoulli bg-keep, one rng instance "
        "seeded by hash(domain), GelSight Mini stream only "
        "(gelsight/*.png; digit/duragel/tac3d and img_* excluded)."
    ),
))

_RAW_BASE = (
    config.RAW_ROOT / "multi_sensor" / "TacQuad" / "tacquad_extracted"
)

_DOMAINS = ("data_indoor", "data_outdoor", "data_fine")
_N_BASELINE = 100


def _list_frames(domain: str) -> List[Tuple[str, str]]:
    """Return [(obj_name, frame_path), ...] for one domain, legacy order.

    Only `<domain>/<obj>/gelsight/*.png` is read -- the GelSight Mini
    stream. `digit/`, `duragel/`, `tac3d/`, and the `img_*` external scene
    camera directories that sit alongside `gelsight/` in the same object
    directory are never touched.
    """
    out: List[Tuple[str, str]] = []
    domain_dir = str(_RAW_BASE / domain)
    for obj in sorted(os.listdir(domain_dir)):
        gs_dir = os.path.join(domain_dir, obj, "gelsight")
        if not os.path.isdir(gs_dir):
            continue
        for p in sorted(
            glob.glob(os.path.join(gs_dir, "*.png")),
            key=lambda x: int(os.path.basename(x).split(".")[0])
            if os.path.basename(x).split(".")[0].isdigit() else 0,
        ):
            out.append((obj, p))
    return out


def _decode(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _domain_frames(domain: str, limit: Optional[int]) -> Iterator[FrameRecord]:
    """Lazily filter one domain's frame list, legacy decision-for-decision.

    Reproduces `legacy/ingest_tacquad_full.py:process_domain` exactly:
    one rng (seeded `hash(domain) & 0xFFFFFFFF`) draws the 100-frame
    baseline sample, then the same rng instance is consulted for every
    frame that fails the area/intensity filter (the Bernoulli
    background-diversity keep).
    """
    frames = _list_frames(domain)
    n = len(frames)
    if n == 0:
        return

    rng = random.Random(hash(domain) & 0xFFFFFFFF)
    idxs = rng.sample(range(n), min(_N_BASELINE, n))
    grays = []
    for i in idxs:
        try:
            grays.append(grey_center(_decode(frames[i][1])))
        except Exception:
            pass
    baseline = np.median(np.stack(grays), axis=0)

    n_yielded = 0
    for i, (obj, path) in enumerate(frames):
        try:
            rgb = _decode(path)
        except Exception:
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

        base = os.path.basename(path).split(".")[0]
        frame_idx = int(base) if base.isdigit() else i
        yield FrameRecord(
            rgb=rgb,
            capture=obj,
            obj_name=obj,
            split=domain,
            episode=obj,
            frame_idx=frame_idx,
        )
        n_yielded += 1
        if limit is not None and n_yielded >= limit:
            return


def iter_units(limit: Optional[int] = None) -> Iterator[Tuple[str, Iterator[FrameRecord]]]:
    """Yield one (domain, frame_generator) unit per TacQuad domain.

    Unlike the per-object avi sources, TacQuad's baseline is computed once
    per *domain* (across all objects in that domain), so the natural unit
    boundary here is the domain, not the object/capture. All filtering
    happens inline in `_domain_frames`; `limit` caps yielded frames per
    domain (mirroring `unit.py`'s per-frame cap semantics), not the number
    of domains.
    """
    for domain in _DOMAINS:
        yield domain, _domain_frames(domain, limit)


def dry_run_keys(limit: Optional[int] = None):
    """Return (kept_keys, n_seen) without writing anything."""
    from gsmp.runner import run

    res = run(SPEC, iter_units(limit=limit), dry_run=True)
    return res.kept_keys, res.n_seen
