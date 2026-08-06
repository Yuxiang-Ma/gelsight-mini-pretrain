"""Sim Star-Struck -- Taxim-rendered, Mini-calibrated star-shape imprints.

tier-1: published rows carry capture + frame_idx, so tools/regress.py can
verify this migration against the release.

Same recipe as `sim_tactile_mnist` (see `gsmp.sources._sim_reprocessed` for
the full algorithm and the evidence that `reprocess_upstream.py`, not
`legacy/make_parquet_v2.py`'s `_iter_sim_parquet_filtered` /
`iter_sim_starstruck`, actually produced the published parquet), just with
this source's own `split_alloc`/`force_stride`/object naming.
"""
from __future__ import annotations

from typing import Iterator, Optional, Tuple

from gsmp.baseline import NoBaseline
from gsmp.runner import FrameRecord
from gsmp.sources import _sim_reprocessed as shared
from gsmp.spec import SourceSpec, register

_NAME = "sim_starstruck"

SPEC = register(SourceSpec(
    name=_NAME,
    domain="sim",
    gel_variant="markerless",
    license_repo="main",
    baseline=NoBaseline(),
    a_min=shared.A_MIN,
    i_min=shared.CONFIGS[_NAME].i_min,   # reprocess_upstream.py:93 -- NOT the
                                          # 15 in docs/imin_from_code.md; see
                                          # _sim_reprocessed module docstring.
    channel_mode="rgb",                  # reprocess_upstream.py never swaps
    phash_dist=None,                     # reprocess_upstream.py never dedupes
    bg_keep_rate=shared.BG_RATE,
    rng_seed=shared.CONFIGS[_NAME].seed,
    notes=(
        "Taxim-rendered star-shape imprints, Mini-calibrated. Produced by "
        "reprocess_upstream.py (2026-05-19), superseding the "
        "make_parquet_v2.py/parallel_sim.py path the task brief points at -- "
        "see gsmp.sources._sim_reprocessed for the evidence."
    ),
))


def iter_units(limit: Optional[int] = None) -> Iterator[Tuple[str, Iterator[FrameRecord]]]:
    return shared.iter_units(_NAME, limit=limit)


def dry_run_keys(limit: Optional[int] = None):
    """Return (kept_keys, n_seen) without writing anything."""
    from gsmp.runner import run

    res = run(SPEC, iter_units(limit=limit), dry_run=True)
    return res.kept_keys, res.n_seen
