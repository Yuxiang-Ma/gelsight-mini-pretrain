"""Sharded parquet writer.

Two safety properties carried over from the legacy ShardWriter:

1. Each shard is capped at ~2 GB. pyarrow overflows when concatenating a
   binary column past 2 GB, which is reachable at ~470K rows of ~50 KB JPEG.
2. Rows are pickled to disk before the parquet write and the pickle is
   removed only after the write succeeds, so a crash mid-write leaves a
   complete recovery point.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List

import pyarrow as pa
import pyarrow.parquet as pq

from gsmp.schema import COLUMNS, SCHEMA

SHARD_TARGET_BYTES = 2 * 1024 ** 3

_PICKLE_NAME = "_all_rows.pkl"


class ShardWriter:
    """Accumulate rows and flush them as size-capped parquet shards."""

    def __init__(
        self,
        out_dir: Path,
        prefix: str,
        shard_bytes: int = SHARD_TARGET_BYTES,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.prefix = prefix
        self.shard_bytes = shard_bytes
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._rows: List[Dict[str, Any]] = []
        self._shard_idx = 0
        self._pending_bytes = 0
        self.total_rows = 0
        self._closed = False
        self._final_paths: List[Path] = []

    def add(self, row: Dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError(
                "ShardWriter is closed: add() cannot be called after "
                "close(). ShardWriter has a one-shot lifecycle — create a "
                "new instance to write more rows."
            )
        full = {name: row.get(name) for name in COLUMNS}
        self._rows.append(full)
        img = full.get("image")
        self._pending_bytes += len(img) if img else 0
        self.total_rows += 1
        if self._pending_bytes >= self.shard_bytes:
            self._flush()

    def _flush(self) -> None:
        if not self._rows:
            return
        recovery = self.out_dir / _PICKLE_NAME
        with open(recovery, "wb") as fh:
            pickle.dump(self._rows, fh, protocol=pickle.HIGHEST_PROTOCOL)

        cols = {name: [r[name] for r in self._rows] for name in COLUMNS}
        table = pa.Table.from_pydict(cols, schema=SCHEMA)
        path = self.out_dir / f"{self.prefix}-{self._shard_idx:05d}.parquet"
        pq.write_table(table, path, compression="snappy")

        recovery.unlink()
        self._shard_idx += 1
        self._rows = []
        self._pending_bytes = 0

    def close(self) -> List[Path]:
        """Flush the tail and rename shards to `prefix-NNNNN-of-NNNNN`.

        One-shot: once closed, subsequent `close()` calls are a safe no-op
        that return the same path list without touching disk again (this is
        what makes it safe for `add()` after `close()` to raise instead of
        silently reopening and clobbering already-renamed shards).
        """
        if self._closed:
            return self._final_paths
        self._flush()
        staged = sorted(self.out_dir.glob(f"{self.prefix}-?????.parquet"))
        total = len(staged)
        final: List[Path] = []
        for i, src in enumerate(staged):
            dst = self.out_dir / f"{self.prefix}-{i:05d}-of-{total:05d}.parquet"
            if src != dst:
                src.rename(dst)
            final.append(dst)
        self._closed = True
        self._final_paths = final
        return final
