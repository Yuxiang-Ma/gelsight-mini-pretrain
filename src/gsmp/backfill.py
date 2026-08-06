"""Backfill the 26-column FoTA shards to the full 30-column schema.

fota_labeled and fota_unlabeled were published missing episode, frame_idx,
digit_class and gel_variant, which breaks the concatenate_datasets example in
the dataset's own README (93,155 frames, ~11% of the corpus).

Only gel_variant can be given a real value -- it is a function of the
markered column. The other three are unrecoverable from the published rows
and are written as NULL. That is enough to fix the user-visible breakage:
HF feature compatibility requires matching column names and types, not
non-null values.

JPEG bytes are never re-encoded. Every original column passes through
untouched, and verify_backfill() enforces that.
"""
from __future__ import annotations

import pyarrow as pa

from gsmp.schema import COLUMNS, LEGACY_26_MISSING, SCHEMA


def backfill_table(table: pa.Table) -> pa.Table:
    """Return `table` widened to the full 30-column schema."""
    n = table.num_rows
    present = set(table.schema.names)
    cols = {}

    for name in COLUMNS:
        field = SCHEMA.field(name)
        if name in present:
            cols[name] = table.column(name).cast(field.type)
        elif name == "gel_variant":
            markered = table.column("markered").to_pylist()
            cols[name] = pa.array(
                ["markered" if m else "markerless" for m in markered],
                pa.string(),
            )
        else:
            cols[name] = pa.nulls(n, field.type)

    return pa.table(cols, schema=SCHEMA)


def verify_backfill(old: pa.Table, new: pa.Table) -> None:
    """Raise unless `new` is `old` widened, with nothing else changed."""
    if new.num_rows != old.num_rows:
        raise ValueError(
            f"row count changed: {old.num_rows} -> {new.num_rows}"
        )
    if not new.schema.equals(SCHEMA):
        raise ValueError("result does not match the canonical schema")

    for name in old.schema.names:
        if name in LEGACY_26_MISSING:
            continue
        if new.column(name).to_pylist() != old.column(name).to_pylist():
            raise ValueError(f"column {name!r} was modified")

    for name in LEGACY_26_MISSING:
        if name == "gel_variant":
            continue
        if new.column(name).null_count != new.num_rows:
            raise ValueError(f"{name!r} should be entirely null")
