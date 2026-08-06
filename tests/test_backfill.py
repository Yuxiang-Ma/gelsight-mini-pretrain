from __future__ import annotations

import pyarrow as pa
import pytest

from gsmp import schema
from gsmp.backfill import backfill_table, verify_backfill

_OLD_COLS = [c for c in schema.COLUMNS if c not in schema.LEGACY_26_MISSING]


def _old_table(n=3, markered=(True, False, True)):
    data = {}
    for name in _OLD_COLS:
        field = schema.SCHEMA.field(name)
        if name == "image":
            data[name] = pa.array([b"\xff\xd8jpg%d" % i for i in range(n)], pa.binary())
        elif name == "markered":
            data[name] = pa.array(list(markered[:n]), pa.bool_())
        elif pa.types.is_string(field.type):
            data[name] = pa.array([f"{name}{i}" for i in range(n)], pa.string())
        elif pa.types.is_boolean(field.type):
            data[name] = pa.array([False] * n, pa.bool_())
        elif pa.types.is_int32(field.type):
            data[name] = pa.array(list(range(n)), pa.int32())
        else:
            data[name] = pa.array([float(i) for i in range(n)], pa.float32())
    return pa.table(data)


def test_backfill_produces_the_full_schema():
    new = backfill_table(_old_table())
    assert new.schema.names == list(schema.COLUMNS)
    assert new.schema.equals(schema.SCHEMA)


def test_gel_variant_is_derived_from_markered():
    new = backfill_table(_old_table(markered=(True, False, True)))
    assert new.column("gel_variant").to_pylist() == [
        "markered", "markerless", "markered",
    ]


def test_unrecoverable_columns_are_null():
    new = backfill_table(_old_table())
    for col in ("frame_idx", "episode", "digit_class"):
        assert new.column(col).null_count == new.num_rows


def test_image_bytes_pass_through_untouched():
    old = _old_table()
    new = backfill_table(old)
    assert new.column("image").to_pylist() == old.column("image").to_pylist()


def test_all_original_columns_are_preserved_exactly():
    old = _old_table()
    new = backfill_table(old)
    for name in _OLD_COLS:
        assert new.column(name).to_pylist() == old.column(name).to_pylist(), name


def test_verify_passes_for_a_correct_backfill():
    old = _old_table()
    verify_backfill(old, backfill_table(old))


def test_verify_rejects_row_count_change():
    old = _old_table()
    bad = backfill_table(old).slice(0, 2)
    with pytest.raises(ValueError, match="row count"):
        verify_backfill(old, bad)


def test_verify_rejects_mutated_image_bytes():
    old = _old_table()
    new = backfill_table(old)
    cols = {n: new.column(n) for n in new.schema.names}
    cols["image"] = pa.array([b"tampered"] * new.num_rows, pa.binary())
    with pytest.raises(ValueError, match="image"):
        verify_backfill(old, pa.table(cols).cast(schema.SCHEMA))


def test_backfill_is_idempotent():
    old = _old_table()
    once = backfill_table(old)
    assert backfill_table(once).equals(once)
