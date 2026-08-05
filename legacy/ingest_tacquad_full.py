#!/usr/bin/env python3
"""Re-ingest TacQuad full Mini stream → unified-schema parquet.

TacQuad (https://arxiv.org/abs/2503.16578) is a quad-sensor benchmark
recorded simultaneously with 4 tactile sensors. We only ingest the
GelSight Mini stream (gelsight/ subfolder per object). The dataset has
three domains:
  - data_indoor:  101 objects, 12,629 frames total
  - data_outdoor:  50 objects,  6,240 frames total
  - data_fine:     30 objects,  5,997 frames total
  Total:          181 objects, 24,866 frames

This script:
  1. Walks each domain's object directories
  2. Reads gelsight/*.png frames per object
  3. Applies unified intensity filter (PIXEL_THRESH=10, A_MIN=40,
     I_MIN=12, BG_RATE=0.015) using a per-domain baseline
  4. Writes tacquad/{domain}-00000-of-00001.parquet
     (3 splits: data_indoor / data_outdoor / data_fine — these are
     conceptually different *environments*, not train/test; we name
     them split=data_indoor etc. to mirror gelslam's train/recon
     conceptual-split pattern)

Outputs go to /media/yxma/Disk1/yuxiang/mini_data_parquet/tacquad/.
Replaces the existing tacquad_mini/ which only had 4,289 rows.
"""
import argparse
import glob
import io
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, "/home/yxma/MultimodalData")
from make_parquet_v2 import SCHEMA

RAW_BASE = ("/media/yxma/Disk1/yuxiang/mini_data/multi_sensor/TacQuad/"
            "tacquad_extracted")
OUT_BASE = "/media/yxma/Disk1/yuxiang/mini_data_parquet/tacquad"

PIXEL_THRESH = 10
A_MIN = 40
I_MIN = 12
BG_RATE = 0.015
N_BASELINE = 100
JPEG_Q = 92


def grey_center(rgb):
    g = rgb.mean(axis=2).astype(np.float32)
    h, w = g.shape
    return g[h // 4:3 * h // 4, w // 4:3 * w // 4]


def decode_path(p):
    return np.array(Image.open(p).convert("RGB"))


def to_jpeg(rgb, q=JPEG_Q):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=q)
    return buf.getvalue()


def list_frames(domain):
    """Return [(object_name, frame_path), ...] for one domain."""
    out = []
    domain_dir = f"{RAW_BASE}/{domain}"
    for obj in sorted(os.listdir(domain_dir)):
        gs_dir = f"{domain_dir}/{obj}/gelsight"
        if not os.path.isdir(gs_dir):
            continue
        for p in sorted(glob.glob(f"{gs_dir}/*.png"),
                        key=lambda x: int(os.path.basename(x).split(".")[0])
                        if os.path.basename(x).split(".")[0].isdigit() else 0):
            out.append((obj, p))
    return out


def process_domain(domain):
    rng = random.Random(hash(domain) & 0xFFFFFFFF)
    print(f"\n=== {domain} ===  listing frames...", flush=True)
    frames = list_frames(domain)
    n = len(frames)
    n_obj = len(set(o for o, _ in frames))
    print(f"  {n_obj} objects, {n:,} frames", flush=True)

    if n == 0:
        return domain, 0

    # Baseline: median over N_BASELINE random frames
    t0 = time.time()
    idxs = rng.sample(range(n), min(N_BASELINE, n))
    grays = []
    for i in idxs:
        try:
            grays.append(grey_center(decode_path(frames[i][1])))
        except Exception:
            pass
    baseline = np.median(np.stack(grays), axis=0)
    print(f"  [{domain}] baseline from {len(grays)} frames in "
          f"{time.time()-t0:.0f}s", flush=True)

    # Filter pass
    rows = []
    n_pass = n_bg = 0
    t0 = time.time()
    for i, (obj, path) in enumerate(frames):
        try:
            rgb = decode_path(path)
        except Exception:
            continue
        g = grey_center(rgb)
        diff = np.abs(g - baseline)
        mask = diff > PIXEL_THRESH
        area = int(mask.sum())
        inten = float(diff[mask].mean()) if area > 0 else 0.0
        passes = (area >= A_MIN) and (inten >= I_MIN)
        keep = False
        if passes:
            keep = True
            n_pass += 1
        elif rng.random() < BG_RATE:
            keep = True
            n_bg += 1
        if keep:
            frame_idx = int(os.path.basename(path).split(".")[0]) \
                if os.path.basename(path).split(".")[0].isdigit() else i
            row = {f.name: None for f in SCHEMA}
            row.update(dict(
                image=to_jpeg(rgb),
                image_format="jpeg",
                source="tacquad",
                markered=False,
                capture=f"{obj}",
                split=domain,           # data_indoor / data_outdoor / data_fine
                height=rgb.shape[0],
                width=rgb.shape[1],
                obj_name=obj,
                episode=obj,
                frame_idx=frame_idx,
                gel_variant="markerless",
                domain="real",
            ))
            rows.append(row)
        if (i + 1) % 2000 == 0:
            fps = (i + 1) / max(0.01, time.time() - t0)
            print(f"  [{domain}] {i+1:,}/{n:,}  kept={len(rows):,}  "
                  f"pass={n_pass:,}  bg={n_bg:,}  ({fps:.0f} fps)", flush=True)

    rate = 100 * len(rows) / max(1, n)
    print(f"  [{domain}] DONE  kept={len(rows):,}/{n:,} ({rate:.1f}%)  "
          f"pass={n_pass:,}  bg={n_bg:,}", flush=True)

    # Build table
    cols = {f.name: [r.get(f.name) for r in rows] for f in SCHEMA}
    arrays = [pa.array(cols[f.name], type=f.type) for f in SCHEMA]
    table = pa.Table.from_arrays(arrays, schema=SCHEMA)

    os.makedirs(OUT_BASE, exist_ok=True)
    out_path = f"{OUT_BASE}/{domain}-00000-of-00001.parquet"
    tmp = out_path + ".tmp"
    pq.write_table(table, tmp, compression="snappy")
    os.replace(tmp, out_path)
    print(f"  [{domain}] wrote {out_path}  rows={table.num_rows:,}",
          flush=True)
    return domain, table.num_rows


def main():
    domains = ["data_indoor", "data_outdoor", "data_fine"]
    print(f"TacQuad full ingest  PIXEL_THRESH={PIXEL_THRESH}  "
          f"A_MIN={A_MIN}  I_MIN={I_MIN}  BG_RATE={BG_RATE}")

    with ProcessPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(process_domain, d): d for d in domains}
        totals = {}
        for fut in as_completed(futs):
            try:
                d, n = fut.result()
                totals[d] = n
            except Exception as e:
                print(f"FAIL {futs[fut]}: {e}")
                import traceback; traceback.print_exc()

    print("\n=== SUMMARY ===")
    total = 0
    for d in domains:
        n = totals.get(d, 0)
        total += n
        print(f"  {d:15s}  {n:>8,}")
    print(f"  {'TOTAL':15s}  {total:>8,}")


if __name__ == "__main__":
    main()
