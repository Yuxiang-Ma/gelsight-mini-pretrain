"""FEATS -- markered gel, force-based filtering.

tier-2: WRAPPED VERBATIM, NOT MIGRATED.

Why: all 1363 published rows have frame_idx = NULL, so there is no join key
to regression-test a rewrite against. The source also filters on force
(|f_z| >= 0.4 N) rather than pixel diff, because tracking dots make pixel
diffing unreliable -- there is no pixel baseline to reconstruct.

The original implementation stays in legacy/convert_feats.py and
legacy/make_parquet_v2.py. This module only declares metadata.
"""
from __future__ import annotations

from gsmp.baseline import NoBaseline
from gsmp.spec import SourceSpec, register

SPEC = register(SourceSpec(
    name="feats",
    domain="real",
    gel_variant="markered",
    license_repo="main",
    baseline=NoBaseline(),
    i_min=0.0,           # unused: force-based filter, not pixel-based; feats
                          # is gated on |f_z| >= 0.4 N and has no i_min by
                          # design -- do not read this 0.0 as a real threshold
    phash_dist=None,
    notes=(
        "tier-2. Force filter |f_z| >= 0.4 N + 1.5% bg keep. frame_idx is "
        "NULL in all published rows, so no regression join key exists. "
        "Implementation: legacy/convert_feats.py."
    ),
))


def legacy_entrypoint() -> str:
    return "legacy/convert_feats.py"
