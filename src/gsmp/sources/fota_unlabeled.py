"""FoTA unlabeled -- 516K raw frames subsampled to the published set.

tier-2: WRAPPED VERBATIM, NOT MIGRATED.

Why: two independent reasons. (1) Published with the 26-column legacy schema,
which has no frame_idx column. (2) It went through the v9 channel-order fix
(archive/fix_channel_order.py), so its intermediate state is not reproducible
from the current raw data even if a join key existed.
"""
from __future__ import annotations

from gsmp.baseline import PerCaptureMedian
from gsmp.spec import SourceSpec, register

SPEC = register(SourceSpec(
    name="fota_unlabeled",
    domain="real",
    gel_variant="mixed",
    license_repo="main",
    baseline=PerCaptureMedian(30),
    i_min=10.0,
    phash_dist=1,        # loose, to retain 200K of 516K raw
    phash_lookback=5,
    resolution=(640, 480),
    notes=(
        "tier-2. 26-column schema (no frame_idx) AND post-v9 channel fix, so "
        "the intermediate state is unreproducible. Implementation: "
        "archive/redo_fota_unlabeled.py + archive/fix_channel_order.py."
    ),
))


def legacy_entrypoint() -> str:
    return "archive/redo_fota_unlabeled.py"
