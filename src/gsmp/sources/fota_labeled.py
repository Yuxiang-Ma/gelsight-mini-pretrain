"""FoTA labeled -- 6-DoF pose + object, mixed markered/markerless gels.

tier-2: WRAPPED VERBATIM, NOT MIGRATED.

Why: published with the 26-column legacy schema, which has no frame_idx
column at all (gsmp.schema.LEGACY_26_SOURCES), so there is no join key.
"""
from __future__ import annotations

from gsmp.baseline import PerCaptureMedian
from gsmp.spec import SourceSpec, register

SPEC = register(SourceSpec(
    name="fota_labeled",
    domain="real",
    gel_variant="mixed",
    license_repo="main",
    baseline=PerCaptureMedian(30),
    i_min=10.0,
    phash_dist=4,
    phash_lookback=30,
    resolution=(640, 480),
    notes=(
        "tier-2. Published with the 26-column schema (no frame_idx), so no "
        "regression join key. Implementation: archive/reprocess_fota.py."
    ),
))


def legacy_entrypoint() -> str:
    return "archive/reprocess_fota.py"
