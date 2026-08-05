#!/usr/bin/env python3
"""Stride-subsample fota_unlabeled in place to ~60 K total frames.

Why: fota_unlabeled is ~70 % of the aggregated mini real-tactile pool
(516 K frames after the area+intensity filter) but contains only **11
objects across 60 captures** (~8,600 frames/capture, gripper-trajectory
near-duplicates). After stride downsampling to 800 train + 200 val
frames per capture, fota_unlabeled drops to ~60 K total (~20 % share)
while every object/init_pose/side combination is still represented.

Strategy:
  1. Load all parquet shards for fota_unlabeled into memory (metadata
     + image bytes, single source ~few GB).
  2. Group row indices by (split, capture).
  3. Per group, keep `max(1, round(len(idxs)/N_TARGET)) ` strided indices
     where N_TARGET is the per-capture target (800 train / 200 val).
  4. Re-shard the kept rows into the same number of parquet files as
     before, preserving original train/val file counts.
  5. Replace existing files atomically.

Usage:
  python subsample_fota_unlabeled.py                # dry-run (counts only)
  python subsample_fota_unlabeled.py --apply        # rewrite files

The script is idempotent in spirit but NOT in counts — running it again
after --apply will further downsample (each capture is restrided from
whatever's left). So only run --apply once.
"""
import argparse
import glob
import os
import sys
from collections import defaultdict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

BASE = "/media/yxma/Disk1/yuxiang/mini_data_parquet"
SUB = "fota_unlabeled"
TRAIN_PER_CAP = 800
VAL_PER_CAP = 200


def stride_indices(n: int, target: int) -> np.ndarray:
    """Return `min(n, target)` indices uniformly spaced over [0, n)."""
    if n <= target:
        return np.arange(n, dtype=np.int64)
    # linspace endpoints inclusive, rounded — gives even coverage
    return np.linspace(0, n - 1, num=target, dtype=np.int64)


def list_shards(split: str):
    paths = sorted(glob.glob(f"{BASE}/{SUB}/{split}-*.parquet"))
    if not paths:
        raise SystemExit(f"no {split} shards at {BASE}/{SUB}/")
    return paths


def load_captures_only(paths):
    """Cheap pass: read only the `capture` column from all shards."""
    print(f"  scanning captures from {len(paths)} shard(s)...", flush=True)
    cols = [pq.read_table(p, columns=["capture"]) for p in paths]
    combined = pa.concat_tables(cols)
    print(f"    total rows: {combined.num_rows:,}", flush=True)
    return combined


def subsample_split(split: str, per_cap: int, apply: bool):
    print(f"\n=== {split} (target {per_cap}/capture) ===", flush=True)
    paths = list_shards(split)

    # ---- Pass 1: cheap capture-only scan to decide global keep set ----
    cap_tbl = load_captures_only(paths)
    n_in = cap_tbl.num_rows
    captures = cap_tbl.column("capture").to_pylist()
    del cap_tbl

    by_cap: dict = defaultdict(list)
    for i, c in enumerate(captures):
        by_cap[c].append(i)
    print(f"    {len(by_cap)} captures", flush=True)

    keep_idx = []
    for cap, idxs in sorted(by_cap.items()):
        idxs_arr = np.asarray(idxs, dtype=np.int64)
        local = stride_indices(len(idxs_arr), per_cap)
        keep_idx.append(idxs_arr[local])
    keep_idx = np.concatenate(keep_idx)
    keep_idx.sort()  # preserve original global row order
    n_out = len(keep_idx)
    print(f"    keeping {n_out:,}/{n_in:,} rows ({100*n_out/n_in:.1f}%)",
          flush=True)

    if not apply:
        return n_in, n_out

    # ---- Pass 2: shard-by-shard filter. Map global keep_idx → per-shard
    # local indices, take rows from each shard, accumulate. ----
    shard_offsets = []
    offset = 0
    for p in paths:
        n = pq.read_metadata(p).num_rows
        shard_offsets.append((offset, offset + n))
        offset += n
    assert offset == n_in, f"shard count mismatch: {offset} vs {n_in}"

    filtered_tables = []
    for p, (lo, hi) in zip(paths, shard_offsets):
        # global indices in [lo, hi)
        l = np.searchsorted(keep_idx, lo, side="left")
        r = np.searchsorted(keep_idx, hi, side="left")
        local = keep_idx[l:r] - lo
        if len(local) == 0:
            continue
        print(f"  reading {os.path.basename(p)} -> taking {len(local):,} rows",
              flush=True)
        t = pq.read_table(p)
        # Avoid offset-overflow on take() when image binary chunks exceed 2 GB
        if "image" in t.column_names and pa.types.is_binary(t.schema.field("image").type):
            idx = t.column_names.index("image")
            t = t.set_column(idx, "image",
                             t.column("image").cast(pa.large_binary()))
        filtered_tables.append(t.take(pa.array(local)))
        del t

    # Now total bytes are tiny (~60K JPEGs ~ <1 GB), safe to concat.
    out = pa.concat_tables(filtered_tables)
    del filtered_tables
    assert out.num_rows == n_out, (out.num_rows, n_out)

    # Cast image back to plain binary so the parquet schema matches other
    # shards / other subsets in the repo.
    if "image" in out.column_names and pa.types.is_large_binary(
        out.schema.field("image").type
    ):
        idx = out.column_names.index("image")
        out = out.set_column(idx, "image",
                             out.column("image").cast(pa.binary()))

    # ---- Re-shard into the same number of files, balanced row counts. ----
    n_shards = len(paths)
    chunk = (n_out + n_shards - 1) // n_shards
    for k, p in enumerate(paths):
        lo, hi = k * chunk, min((k + 1) * chunk, n_out)
        if lo >= hi:
            shard = out.slice(0, 0)
        else:
            shard = out.slice(lo, hi - lo)
        tmp = p + ".tmp"
        pq.write_table(shard, tmp, compression="snappy")
        os.replace(tmp, p)
        print(f"    wrote {os.path.basename(p)}  rows={shard.num_rows:,}",
              flush=True)

    return n_in, n_out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="actually overwrite parquet files (default: dry run)")
    args = ap.parse_args()

    if not args.apply:
        print("DRY RUN — no files will be modified. Re-run with --apply.\n")

    grand_in = grand_out = 0
    for split, per_cap in [("train", TRAIN_PER_CAP), ("val", VAL_PER_CAP)]:
        n_in, n_out = subsample_split(split, per_cap, args.apply)
        grand_in += n_in
        grand_out += n_out

    print(f"\n=== {SUB} done ===")
    print(f"  before: {grand_in:>10,}")
    print(f"  after : {grand_out:>10,}  ({100*grand_out/grand_in:.1f}%)")
    if not args.apply:
        print("  (dry run — no files modified)")


if __name__ == "__main__":
    main()
