#!/usr/bin/env python3
"""Streaming per-shard dedupe + budget cap for FoTA parquet.

Avoids pyarrow's 2 GB binary-chunk overflow by processing shards one at a
time. Across shards we share a per-capture rolling-phash state.

Two passes:
  Pass 1: read every shard, compute phash, mark keep/dup; collect all
          (shard_idx, row_idx, capture) tuples we'd keep.
  Pass 2: if total kept > BUDGET, stride-down the keep list uniformly.
  Pass 3: re-read each shard and write filtered rows to new shards.
"""
import glob, io, os, random, sys, time
from collections import defaultdict
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

BASE = "/media/yxma/Disk1/yuxiang/mini_data_parquet"
BUDGET = 200_000
PHASH_LOOKBACK = 30
PHASH_DIST = 4

import sys
sys.path.insert(0, "/home/yxma/MultimodalData")
from make_parquet_v2 import phash, hamming


def main():
    if len(sys.argv) < 2: print(__doc__); sys.exit(1)
    sub = sys.argv[1]
    paths = sorted(glob.glob(f"{BASE}/{sub}/*.parquet"))
    if not paths: print(f"no parquet at {BASE}/{sub}/"); sys.exit(1)

    # ----- Pass 1: compute phash per row, decide keep vs dup -----
    print(f"PASS 1: phash + per-capture dedupe over {len(paths)} shards...",
          flush=True)
    last_hashes = defaultdict(list)
    keeps = []           # list of (shard_idx, row_idx_within_shard)
    n_total = n_dup = 0
    t0 = time.time()
    for si, p in enumerate(paths):
        pf = pq.ParquetFile(p)
        ri_base = 0
        for batch in pf.iter_batches(batch_size=512,
                                     columns=["image", "capture"]):
            caps = batch.column("capture").to_pylist()
            imgs = batch.column("image").to_pylist()
            for i in range(len(imgs)):
                cap = caps[i]
                try:
                    rgb = np.array(Image.open(io.BytesIO(imgs[i])).convert("RGB"))
                    h = phash(rgb)
                except Exception:
                    keeps.append((si, ri_base + i))
                    n_total += 1
                    continue
                recent = last_hashes[cap][-PHASH_LOOKBACK:]
                if any(hamming(h, hh) <= PHASH_DIST for hh in recent):
                    n_dup += 1
                else:
                    keeps.append((si, ri_base + i))
                    last_hashes[cap].append(h)
                n_total += 1
            ri_base += len(imgs)
        print(f"  shard {si+1}/{len(paths)} done · "
              f"kept-so-far={len(keeps):,} · dup={n_dup:,} · "
              f"{n_total/(time.time()-t0):.0f} fps", flush=True)
    print(f"  PASS 1 done: {n_total:,} seen, {len(keeps):,} kept, "
          f"{n_dup:,} dup", flush=True)

    # ----- Pass 2: stride-cap to BUDGET -----
    if len(keeps) > BUDGET:
        stride = len(keeps) / BUDGET
        sampled = [keeps[int(i * stride)] for i in range(BUDGET)]
        keeps = sampled
        print(f"  cap to BUDGET={BUDGET}: stride={stride:.2f} → {len(keeps):,}",
              flush=True)
    # Group by shard
    by_shard = defaultdict(set)
    for si, ri in keeps: by_shard[si].add(ri)

    # ----- Pass 3: rewrite shards -----
    print(f"PASS 3: rewriting shards with mask...", flush=True)
    out_dir = f"{BASE}/{sub}"
    tmpdir = f"{BASE}/{sub}_tmp"
    os.makedirs(tmpdir, exist_ok=True)
    shard_idx_out = 0
    SHARD_ROWS = 60_000   # ~2 GB at 50 KB/row JPEG
    buf_rows = []
    schema = pq.read_schema(paths[0])

    def flush_buf():
        nonlocal shard_idx_out, buf_rows
        if not buf_rows: return
        cols = {f.name: [r[f.name] for r in buf_rows] for f in schema}
        t = pa.Table.from_pydict(cols, schema=schema)
        outp = f"{tmpdir}/train-{shard_idx_out:05d}.parquet"
        pq.write_table(t, outp, compression="snappy")
        print(f"  wrote {os.path.basename(outp)} rows={len(buf_rows):,}",
              flush=True)
        shard_idx_out += 1
        buf_rows = []

    for si, p in enumerate(paths):
        mask = by_shard.get(si, set())
        if not mask: continue
        pf = pq.ParquetFile(p)
        ri_base = 0
        for batch in pf.iter_batches(batch_size=512):
            for i in range(batch.num_rows):
                if ri_base + i in mask:
                    row = {col: batch.column(col)[i].as_py() for col in batch.column_names}
                    buf_rows.append(row)
                    if len(buf_rows) >= SHARD_ROWS:
                        flush_buf()
            ri_base += batch.num_rows
    flush_buf()

    # Replace old shards with new
    for p in paths: os.remove(p)
    total_shards = len(sorted(glob.glob(f"{tmpdir}/train-?????.parquet")))
    for i, fp in enumerate(sorted(glob.glob(f"{tmpdir}/train-?????.parquet"))):
        new = f"{out_dir}/train-{i:05d}-of-{total_shards:05d}.parquet"
        os.rename(fp, new)
    os.rmdir(tmpdir)
    final = sum(pq.read_metadata(p).num_rows
                for p in sorted(glob.glob(f"{out_dir}/*.parquet")))
    print(f"\n=== {sub} dedupe+cap done ===")
    print(f"  {n_total:,} → {final:,}  (dup {n_dup:,} dropped, "
          f"cap budget={BUDGET:,})")


if __name__ == "__main__":
    main()
