#!/usr/bin/env python3
"""Fix RGB/BGR channel-order inconsistencies across the dataset.

Based on the diagnose_channel_order.py analysis:
  - GelSight Mini at-rest gel has B > R (camera sensor response bias).
  - 10 subsets cluster with R-B < 0 (correctly RGB).
  - 4 subsets cluster with R-B > 0 (BGR; need swap):
      * fota_unlabeled  (full dataset BGR)
      * unit            (full dataset BGR)
      * faf_force_estimation (full dataset BGR; NC repo)
      * sparsh          (MIXED — Facebook published with both RGB and BGR)

Strategy:
  - For globally-BGR subsets: read each row's JPEG → decode → swap R↔B →
    re-encode JPEG → write back.
  - For sparsh: per-image conditional swap. Compute the mean (R, G, B) of
    each image; if R > B, swap; else keep. Since the contact patch is
    only ~20% of the image, the at-rest gel background dominates the
    whole-image mean, making this a reliable per-image discriminator.

Multiprocessing: one worker per (subset, shard) pair. Each worker reads
one parquet, processes all rows, writes back in place via temp file.

Usage:
  python fix_channel_order.py
"""
import argparse
import glob
import io
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

MAIN_BASE = "/media/yxma/Disk1/yuxiang/mini_data_parquet"
NC_BASE = "/media/yxma/Disk1/yuxiang/mini_data_parquet_nc"

JPEG_Q = 92

# Mode per subset:
#   "unconditional" → swap every image's R and B channels
#   "conditional"   → swap only images where R-channel mean > B-channel mean
SUBSETS = {
    "main/fota_unlabeled": ("unconditional", MAIN_BASE),
    "main/unit": ("unconditional", MAIN_BASE),
    "nc/faf_force_estimation": ("unconditional", NC_BASE),
    "nc/sparsh": ("conditional", NC_BASE),
}


def swap_image_bytes(jpeg_bytes):
    """Decode JPEG → swap R↔B → re-encode."""
    rgb = np.array(Image.open(io.BytesIO(jpeg_bytes)).convert("RGB"))
    bgr = rgb[..., ::-1]
    buf = io.BytesIO()
    Image.fromarray(bgr).save(buf, format="JPEG", quality=JPEG_Q)
    return buf.getvalue()


def conditional_swap(jpeg_bytes):
    """Decode → if R > B (indicates BGR-stored), swap; else keep."""
    rgb = np.array(Image.open(io.BytesIO(jpeg_bytes)).convert("RGB"))
    R_mean = rgb[..., 0].mean()
    B_mean = rgb[..., 2].mean()
    if R_mean > B_mean:
        bgr = rgb[..., ::-1]
        buf = io.BytesIO()
        Image.fromarray(bgr).save(buf, format="JPEG", quality=JPEG_Q)
        return buf.getvalue(), True   # swapped
    else:
        return jpeg_bytes, False       # kept


def process_shard(path, mode):
    fname = os.path.basename(path)
    t0 = time.time()
    print(f"  [{fname}] reading...", flush=True)
    t = pq.read_table(path)
    n = t.num_rows
    # Cast image -> large_binary so we can do row-level updates safely
    if "image" in t.column_names and pa.types.is_binary(t.schema.field("image").type):
        i = t.column_names.index("image")
        t = t.set_column(i, "image", t.column("image").cast(pa.large_binary()))
    img_col = t.column("image")
    new_imgs = []
    n_swapped = 0
    for i in range(n):
        b = img_col[i].as_py()
        if b is None:
            new_imgs.append(b); continue
        try:
            if mode == "unconditional":
                new_b = swap_image_bytes(b)
                n_swapped += 1
            else:  # conditional
                new_b, swapped = conditional_swap(b)
                if swapped: n_swapped += 1
            new_imgs.append(new_b)
        except Exception as e:
            new_imgs.append(b)  # leave unchanged on decode error
        if (i + 1) % 5000 == 0:
            fps = (i + 1) / max(0.01, time.time() - t0)
            print(f"  [{fname}] {i+1:,}/{n:,}  swapped={n_swapped:,}  "
                  f"({fps:.0f} fps)", flush=True)

    # Replace image column. Build the new binary array in CHUNKS so each
    # chunk stays under PyArrow's 2 GB pa.binary() limit. This avoids the
    # cast-overflow error that hit large fota_unlabeled shards (>2 GB of
    # JPEG bytes per shard).
    idx = t.column_names.index("image")
    CHUNK = 5000
    chunks = []
    for i in range(0, len(new_imgs), CHUNK):
        chunks.append(pa.array(new_imgs[i:i + CHUNK], type=pa.binary()))
    new_col = pa.chunked_array(chunks, type=pa.binary())
    t = t.set_column(idx, "image", new_col)

    tmp = path + ".tmp"
    pq.write_table(t, tmp, compression="snappy")
    os.replace(tmp, path)
    print(f"  [{fname}] DONE  swapped={n_swapped:,}/{n:,} ({100*n_swapped/n:.1f}%)  "
          f"in {time.time()-t0:.0f}s", flush=True)
    return path, n, n_swapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    # Collect all (path, mode) pairs to process
    jobs = []
    for key, (mode, base) in SUBSETS.items():
        sub = key.split("/", 1)[1]
        paths = sorted(glob.glob(f"{base}/{sub}/*.parquet"))
        for p in paths:
            jobs.append((p, mode))
    print(f"=== Fix channel order ===")
    print(f"Jobs: {len(jobs)} parquet shards across {len(SUBSETS)} subsets")
    print(f"Workers: {args.workers}")
    for j, m in jobs:
        print(f"  {m:14s}  {j}")
    print()

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_shard, p, m): (p, m) for p, m in jobs}
        for fut in as_completed(futs):
            try:
                results.append(fut.result())
            except Exception as e:
                import traceback; traceback.print_exc()

    print("\n=== SUMMARY ===")
    for path, n, sw in results:
        print(f"  {path}  swapped={sw:,}/{n:,}")


if __name__ == "__main__":
    main()
