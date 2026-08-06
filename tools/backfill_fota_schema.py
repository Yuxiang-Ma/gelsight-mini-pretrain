#!/usr/bin/env python3
"""Widen the published FoTA shards to the 30-column schema.

Writes to config.OUT_ROOT/fota_backfill/<source>/ -- never in place, so the
published tree stays untouched until the upload step is run separately.

    python tools/backfill_fota_schema.py --source fota_labeled
    python tools/backfill_fota_schema.py --all
"""
from __future__ import annotations

import argparse
import glob
import pathlib
import sys

import pyarrow.parquet as pq

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from gsmp import config                                    # noqa: E402
from gsmp.backfill import backfill_table, verify_backfill  # noqa: E402

SOURCES = ("fota_labeled", "fota_unlabeled")


def run(source: str) -> None:
    src_dir = config.published_dir(source)
    out_dir = config.OUT_ROOT / "fota_backfill" / source
    out_dir.mkdir(parents=True, exist_ok=True)

    shards = sorted(glob.glob(str(src_dir / "*.parquet")))
    if not shards:
        raise FileNotFoundError(f"no shards under {src_dir}")

    for path in shards:
        name = pathlib.Path(path).name
        old = pq.read_table(path)
        new = backfill_table(old)
        verify_backfill(old, new)
        pq.write_table(new, out_dir / name, compression="snappy")
        print(f"  {name}: {old.num_rows} rows, "
              f"{len(old.schema.names)} -> {len(new.schema.names)} cols")

    print(f"{source}: wrote {len(shards)} shards to {out_dir}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=SOURCES)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    targets = SOURCES if args.all else ([args.source] if args.source else [])
    if not targets:
        ap.error("pass --source NAME or --all")
    for s in targets:
        run(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
