from __future__ import annotations

import pyarrow.parquet as pq

from gsmp import schema
from gsmp.writer import ShardWriter


def _row(image=b"\xff\xd8fake", **kw):
    r = {"image": image, "image_format": "jpeg", "source": "t",
         "domain": "real", "markered": False, "capture": "c0",
         "split": "train", "height": 240, "width": 320, "frame_idx": 0}
    r.update(kw)
    return r


def test_missing_columns_are_filled_with_none(tmp_path):
    w = ShardWriter(tmp_path, "train")
    w.add(_row())
    paths = w.close()
    t = pq.read_table(paths[0])
    assert t.schema.names == list(schema.COLUMNS)
    assert t.column("obj_name").to_pylist() == [None]


def test_close_renames_with_of_total_suffix(tmp_path):
    w = ShardWriter(tmp_path, "train")
    for i in range(3):
        w.add(_row(frame_idx=i))
    paths = w.close()
    assert len(paths) == 1
    assert paths[0].name == "train-00000-of-00001.parquet"


def test_rolls_to_new_shard_when_byte_budget_exceeded(tmp_path):
    w = ShardWriter(tmp_path, "train", shard_bytes=100)
    for i in range(4):
        w.add(_row(image=b"x" * 60, frame_idx=i))
    paths = w.close()
    assert len(paths) >= 2
    assert [p.name for p in paths] == sorted(p.name for p in paths)
    total = sum(pq.read_table(p).num_rows for p in paths)
    assert total == 4


def test_pickle_recovery_point_written_then_removed(tmp_path, monkeypatch):
    import gsmp.writer as writer_mod

    recovery = tmp_path / "_all_rows.pkl"
    seen_present = {"flag": False}
    real_write_table = writer_mod.pq.write_table

    def wrapper(*args, **kwargs):
        # Strengthened per ambiguity resolution #4: assert the pickle
        # genuinely exists at the moment of the write, not just that it's
        # absent before/after close() (which would pass even if the pickle
        # were never written at all).
        assert recovery.exists()
        seen_present["flag"] = True
        return real_write_table(*args, **kwargs)

    monkeypatch.setattr(writer_mod.pq, "write_table", wrapper)

    w = ShardWriter(tmp_path, "train")
    w.add(_row())
    assert not recovery.exists()
    w.close()
    assert not recovery.exists()
    assert seen_present["flag"], "pyarrow.parquet.write_table was never called"


def test_empty_writer_produces_no_files(tmp_path):
    w = ShardWriter(tmp_path, "train")
    assert w.close() == []
    assert list(tmp_path.glob("*.parquet")) == []
