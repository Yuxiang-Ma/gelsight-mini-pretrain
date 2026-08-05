"""The unified parquet schema.

This module is the single source of truth for the column set. The legacy
`make_parquet_v2.SCHEMA` and the `sys.path` hack in `legacy/pipeline.py`
that imported it are both replaced by this.

Known deviation: fota_labeled and fota_unlabeled were published with 26
columns (see LEGACY_26_SOURCES). Verified 2026-08-04 against the released
parquet. tests/test_schema.py pins this.
"""
from __future__ import annotations

import glob
from typing import List

import pyarrow as pa
import pyarrow.parquet as pq

from gsmp import config

SCHEMA = pa.schema([
    ("image", pa.binary()),
    ("image_format", pa.string()),
    ("source", pa.string()),
    ("markered", pa.bool_()),
    ("capture", pa.string()),
    ("split", pa.string()),
    ("height", pa.int32()),
    ("width", pa.int32()),
    ("obj_name", pa.string()),
    ("init_pose", pa.int32()),
    ("side", pa.string()),
    ("x_mm", pa.float32()),
    ("y_mm", pa.float32()),
    ("z_mm", pa.float32()),
    ("quat_x", pa.float32()),
    ("quat_y", pa.float32()),
    ("quat_z", pa.float32()),
    ("quat_w", pa.float32()),
    ("indenter", pa.string()),
    ("indenter_param", pa.string()),
    ("f_x", pa.float32()),
    ("f_y", pa.float32()),
    ("f_z", pa.float32()),
    ("grid_z_max", pa.float32()),
    ("grid_z_mean", pa.float32()),
    ("episode", pa.string()),
    ("frame_idx", pa.int32()),
    ("digit_class", pa.int32()),
    ("gel_variant", pa.string()),
    ("domain", pa.string()),
])

COLUMNS = tuple(f.name for f in SCHEMA)

#: Sources published with the older 26-column schema.
LEGACY_26_SOURCES = frozenset({"fota_labeled", "fota_unlabeled"})

#: Columns absent from the 26-column sources.
LEGACY_26_MISSING = frozenset({
    "episode", "frame_idx", "digit_class", "gel_variant",
})


def published_columns(source: str, license_repo: str = "main") -> List[str]:
    """Column names of the first published shard of `source` (read-only)."""
    d = config.published_dir(source, license_repo)
    shards = sorted(glob.glob(str(d / "*.parquet")))
    if not shards:
        raise FileNotFoundError(f"no published parquet under {d}")
    return list(pq.ParquetFile(shards[0]).schema_arrow.names)


def has_join_key(source: str, license_repo: str = "main") -> bool:
    """True if published shards carry both `capture` and a non-null `frame_idx`.

    This is the precondition for regression-testing a source against the
    published release (see tools/regress.py).
    """
    cols = published_columns(source, license_repo)
    if "capture" not in cols or "frame_idx" not in cols:
        return False
    d = config.published_dir(source, license_repo)
    shard = sorted(glob.glob(str(d / "*.parquet")))[0]
    t = pq.read_table(shard, columns=["frame_idx"])
    return t.num_rows > 0 and t.column("frame_idx").null_count < t.num_rows
