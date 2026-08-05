#!/usr/bin/env python3
"""Re-filter FoTA labeled & unlabeled parquet shards in place using the
unified area+intensity rule.

FoTA was originally processed without a validity filter. We can't re-extract
from raw WebDataset tars cheaply, so we filter the existing parquet:

  1. Read existing parquet, group rows by `capture`.
  2. For each capture, sample ≤30 random frames → decode → median = baseline.
     (Per-pixel contact is rare across many frames at varying gripper poses,
      so cross-frame median ≈ gel-at-rest.)
  3. For each row, decode image, compute area+intensity vs baseline, apply
     unified A=40, I=15 filter + 1.5% bg-keep.
  4. Write filtered parquet, replacing the original shard.

Usage:
  python reprocess_fota.py fota_labeled
  python reprocess_fota.py fota_unlabeled
"""
import glob, io, os, random, sys, time
from collections import defaultdict
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

BASE = "/media/yxma/Disk1/yuxiang/mini_data_parquet"
PIXEL_THRESH = 10
A_MIN = 40
I_MIN = 10
BG_RATE = 0.015
N_BASELINE = 30      # frames per capture used to estimate baseline

import sys
sys.path.insert(0, "/home/yxma/MultimodalData")
from make_parquet_v2 import SCHEMA


def grey_center(rgb):
    g = rgb.mean(axis=2).astype(np.float32)
    h, w = g.shape
    return g[h//4:3*h//4, w//4:3*w//4]


def filter_shard(in_path, out_path, rng):
    print(f"  reading {os.path.basename(in_path)} ...", flush=True)
    t = pq.read_table(in_path)
    n = t.num_rows
    captures = t.column("capture").to_pylist()
    images = t.column("image").to_pylist()

    # Group row indices by capture
    by_cap = defaultdict(list)
    for i, c in enumerate(captures):
        by_cap[c].append(i)
    print(f"    {n:,} rows · {len(by_cap)} captures", flush=True)

    # Compute per-capture baseline from a random sample
    baselines = {}
    t0 = time.time()
    for cap, idxs in by_cap.items():
        sample = idxs if len(idxs) <= N_BASELINE else rng.sample(idxs, N_BASELINE)
        grays = []
        for i in sample:
            try:
                rgb = np.array(Image.open(io.BytesIO(images[i])).convert("RGB"))
                grays.append(grey_center(rgb))
            except Exception:
                pass
        if len(grays) >= 5:
            baselines[cap] = np.median(np.stack(grays), axis=0)
    print(f"    baselines for {len(baselines)}/{len(by_cap)} captures "
          f"({time.time()-t0:.0f}s)", flush=True)

    # Filter every row
    keep_mask = np.zeros(n, dtype=bool)
    n_pass = n_bg = 0
    t0 = time.time()
    for i in range(n):
        c = captures[i]
        bl = baselines.get(c)
        if bl is None:
            keep_mask[i] = True  # safety: keep if no baseline
            continue
        try:
            rgb = np.array(Image.open(io.BytesIO(images[i])).convert("RGB"))
        except Exception:
            continue
        g_c = grey_center(rgb)
        diff = np.abs(g_c - bl)
        mask = diff > PIXEL_THRESH
        area = int(mask.sum())
        inten = float(diff[mask].mean()) if area > 0 else 0.0
        passes = (area >= A_MIN) and (inten >= I_MIN)
        if passes:
            keep_mask[i] = True; n_pass += 1
        elif rng.random() < BG_RATE:
            keep_mask[i] = True; n_bg += 1
        if (i+1) % 5000 == 0:
            print(f"    {i+1}/{n} processed  "
                  f"({(i+1)/(time.time()-t0):.0f} fps)", flush=True)

    kept = int(keep_mask.sum())
    print(f"    kept {kept}/{n} ({100*kept/n:.1f}%); pass={n_pass} bg={n_bg}",
          flush=True)

    # Write filtered shard
    t2 = t.filter(pa.array(keep_mask))
    # Ensure domain column set to "real"
    if "domain" in t2.column_names:
        idx = t2.column_names.index("domain")
        t2 = t2.set_column(idx, "domain", pa.array(["real"]*kept, type=pa.string()))
    else:
        t2 = t2.append_column("domain", pa.array(["real"]*kept, type=pa.string()))
    tmp = out_path + ".tmp"
    pq.write_table(t2, tmp, compression="snappy")
    os.replace(tmp, out_path)
    print(f"    wrote {os.path.basename(out_path)}  rows={kept}", flush=True)
    return n, kept, n_bg


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    sub = sys.argv[1]
    paths = sorted(glob.glob(f"{BASE}/{sub}/*.parquet"))
    if not paths:
        print(f"no parquet at {BASE}/{sub}/"); sys.exit(1)
    rng = random.Random(0)
    grand = dict(seen=0, kept=0, bg=0)
    for p in paths:
        n, kept, bg = filter_shard(p, p, rng)
        grand["seen"] += n; grand["kept"] += kept; grand["bg"] += bg
    print(f"\n=== {sub} done ===")
    print(f"  seen={grand['seen']:,}  kept={grand['kept']:,}  "
          f"bg={grand['bg']:,}  kept-rate={100*grand['kept']/grand['seen']:.1f}%")


if __name__ == "__main__":
    main()
