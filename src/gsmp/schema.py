"""The unified parquet schema.

This module is the single source of truth for the column set. The legacy
`make_parquet_v2.SCHEMA` and the `sys.path` hack in `legacy/pipeline.py`
that imported it are both replaced by this.

All 13 published configs share this 30-column schema. fota_labeled and
fota_unlabeled shipped with 26 until 2026-08-06; they were widened and
republished, and LEGACY_26_SOURCES is now empty. LEGACY_26_MISSING is kept
because gsmp.backfill still needs to know which columns were absent.
"""
from __future__ import annotations

import dataclasses
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
#:
#: Empty since 2026-08-06: fota_labeled and fota_unlabeled were widened to the
#: full 30 columns and republished, so every config now shares one schema and
#: cross-config concatenate_datasets works (verified by actually running the
#: dataset README's quick-start, not by inspecting column lists). The three
#: columns that were never recorded for these subsets -- frame_idx, episode,
#: digit_class -- are present but null.
LEGACY_26_SOURCES = frozenset()

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


def join_key_quality(source: str, license_repo: str = "main") -> "JoinKeyQuality":
    """Describe how well `(capture, frame_idx)` identifies a published row.

    Presence of the columns is NOT the same as usability. `tools/regress.py`
    compares SETS of keys, so a non-unique key silently weakens the claim: two
    rows sharing a key are indistinguishable to a set comparison. Measured on
    this release, `sim_starstruck` has 166,104 rows over 31,096 distinct keys —
    matching that key set is equally consistent with emitting either count.

    Reads every shard, not just the first: uniqueness is a property of the
    whole source and a per-shard check would miss cross-shard collisions.
    """
    cols = published_columns(source, license_repo)
    d = config.published_dir(source, license_repo)
    shards = sorted(glob.glob(str(d / "*.parquet")))
    has_cols = "capture" in cols and "frame_idx" in cols

    # Always count rows, even when the key columns are absent -- reporting
    # n_rows=0 for a source that has 66,761 published rows would be a false
    # statement in the tier table this feeds.
    n_rows = 0
    keys = set()
    for shard in shards:
        pf = pq.ParquetFile(shard)
        if not has_cols:
            n_rows += pf.metadata.num_rows
            continue
        t = pq.read_table(shard, columns=["capture", "frame_idx"])
        n_rows += t.num_rows
        keys.update(
            (c, i)
            for c, i in zip(t.column("capture").to_pylist(),
                            t.column("frame_idx").to_pylist())
            if c is not None and i is not None
        )

    present = has_cols and n_rows > 0 and len(keys) > 0
    return JoinKeyQuality(
        source=source,
        present=present,
        unique=present and len(keys) == n_rows,
        n_rows=n_rows,
        n_keys=len(keys),
    )


@dataclasses.dataclass(frozen=True)
class JoinKeyQuality:
    """What a set-based regression on this source can and cannot prove."""

    source: str
    present: bool
    unique: bool
    n_rows: int
    n_keys: int

    @property
    def proof_strength(self) -> str:
        if not self.present:
            return "none: no usable (capture, frame_idx) key"
        if self.unique:
            return "row-level: key is unique, so set equality IS row equality"
        return (
            f"key-set only: {self.n_rows} rows collapse to {self.n_keys} keys "
            f"({self.n_rows - self.n_keys} duplicates); set equality does NOT "
            f"establish row-level equality -- also compare row counts"
        )


def has_join_key(source: str, license_repo: str = "main") -> bool:
    """True if published shards carry a usable `(capture, frame_idx)` key.

    Retained for callers that only need presence. **Presence is not
    sufficient for a row-level parity claim** -- use `join_key_quality()` and
    report `proof_strength` alongside any regression verdict.
    """
    return join_key_quality(source, license_repo).present
