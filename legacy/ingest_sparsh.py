#!/usr/bin/env python3
"""Ingest Sparsh GelSight SparshGelSight pickles → NC-licensed parquet.

Sparsh dataset (https://sparsh-ssl.github.io/) is a force/slip benchmark
for the GelSight Mini sensor, distributed under CC-BY-NC-4.0. It must
go to the **nc** repo (yxma/gelsight-mini-pretrain-nc) not the main one.

Layout:
  SparshGelSight/{flat,sharp,sphere}/batch_*/dataset_gelsight_NN.pkl
    Each pkl = list of JPEG-encoded RGB bytes, ~5000 frames per file.
    (org_dataset_*.pkl are pre-filter; we skip those.)

Per indenter, total available:
  flat:    7 files,  29,809 frames
  sharp:   8 files,  35,356 frames
  sphere: 24 files, 109,701 frames
  TOTAL:                174,866 frames

We apply the unified intensity filter (PIXEL_THRESH=10, A_MIN=40,
I_MIN=12) per-indenter baseline + 1.5% bg-diversity, then write to:
  /media/yxma/Disk1/yuxiang/mini_data_parquet_nc/sparsh/{indenter}-00000-of-00001.parquet
"""
import argparse
import glob
import io
import os
import pickle
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

RAW_BASE = "/media/yxma/Disk1/yuxiang/mini_data/markerless_nc/SparshGelSight"
OUT_BASE = "/media/yxma/Disk1/yuxiang/mini_data_parquet_nc/sparsh"

PIXEL_THRESH = 10
A_MIN = 40
I_MIN = 12
BG_RATE = 0.015
N_BASELINE = 100


def grey_center_from_jpeg(b):
    try:
        rgb = np.array(Image.open(io.BytesIO(b)).convert("RGB"))
    except Exception:
        return None, None
    g = rgb.mean(axis=2).astype(np.float32)
    h, w = g.shape
    return rgb, g[h // 4:3 * h // 4, w // 4:3 * w // 4]


def collect_pkl_files(indenter):
    """Return list of (batch, file_idx, pkl_path) for one indenter."""
    out = []
    for pkl in sorted(glob.glob(f"{RAW_BASE}/{indenter}/batch_*/dataset_gelsight_*.pkl")):
        # Skip org_dataset_*.pkl (pre-filter raw data)
        fname = os.path.basename(pkl)
        if fname.startswith("org_"):
            continue
        # batch_N
        batch = os.path.basename(os.path.dirname(pkl))
        # file_idx = NN from dataset_gelsight_NN.pkl
        file_idx = int(fname.split("_")[-1].split(".")[0])
        out.append((batch, file_idx, pkl))
    return out


def process_indenter(indenter):
    rng = random.Random(hash(indenter) & 0xFFFFFFFF)
    print(f"\n=== sparsh/{indenter} ===  collecting pkls...", flush=True)
    pkls = collect_pkl_files(indenter)
    print(f"  {len(pkls)} pkl files", flush=True)

    # Load all to estimate total count and pick baseline frames
    all_frames = []  # list of (batch, file_idx, frame_idx, jpeg_bytes)
    t0 = time.time()
    for batch, file_idx, pkl in pkls:
        with open(pkl, "rb") as f:
            data = pickle.load(f)
        for fi, b in enumerate(data):
            all_frames.append((batch, file_idx, fi, b))
    n = len(all_frames)
    print(f"  [{indenter}] loaded {n:,} frames in {time.time()-t0:.0f}s",
          flush=True)

    # Baseline
    base_idxs = rng.sample(range(n), min(N_BASELINE, n))
    grays = []
    for i in base_idxs:
        _, g = grey_center_from_jpeg(all_frames[i][3])
        if g is not None:
            grays.append(g)
    baseline = np.median(np.stack(grays), axis=0)
    print(f"  [{indenter}] baseline from {len(grays)} frames", flush=True)

    rows = []
    n_pass = n_bg = 0
    t0 = time.time()
    for i, (batch, file_idx, frame_idx, b) in enumerate(all_frames):
        rgb, g = grey_center_from_jpeg(b)
        if g is None:
            continue
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
            r = {f.name: None for f in SCHEMA}
            r.update(dict(
                image=b,                      # already JPEG bytes
                image_format="jpeg",
                source="sparsh",
                markered=False,
                capture=f"{indenter}_{batch}_f{file_idx:02d}",
                split=indenter,               # split-by-indenter (matches faf pattern)
                height=rgb.shape[0],
                width=rgb.shape[1],
                obj_name=f"indenter_{indenter}",
                indenter=indenter,
                episode=f"{batch}_f{file_idx:02d}",
                frame_idx=frame_idx,
                gel_variant="markerless",
                domain="real",
            ))
            rows.append(r)
        if (i + 1) % 5000 == 0:
            fps = (i + 1) / max(0.01, time.time() - t0)
            print(f"  [{indenter}] {i+1:,}/{n:,}  kept={len(rows):,}  "
                  f"pass={n_pass:,}  bg={n_bg:,}  ({fps:.0f} fps)", flush=True)

    rate = 100 * len(rows) / max(1, n)
    print(f"  [{indenter}] DONE  kept={len(rows):,}/{n:,} ({rate:.1f}%)",
          flush=True)

    # Build + write
    cols = {f.name: [r.get(f.name) for r in rows] for f in SCHEMA}
    arrays = [pa.array(cols[f.name], type=f.type) for f in SCHEMA]
    table = pa.Table.from_arrays(arrays, schema=SCHEMA)
    os.makedirs(OUT_BASE, exist_ok=True)
    out_path = f"{OUT_BASE}/{indenter}-00000-of-00001.parquet"
    tmp = out_path + ".tmp"
    pq.write_table(table, tmp, compression="snappy")
    os.replace(tmp, out_path)
    sz = os.path.getsize(out_path) / 1e6
    print(f"  [{indenter}] wrote {out_path}  rows={table.num_rows:,}  {sz:.1f} MB",
          flush=True)
    return indenter, table.num_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    indenters = ["flat", "sharp", "sphere"]
    print(f"Sparsh ingest  I_MIN={I_MIN}  A_MIN={A_MIN}  BG_RATE={BG_RATE}  "
          f"workers={args.workers}")
    totals = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_indenter, ind): ind for ind in indenters}
        for fut in as_completed(futs):
            try:
                ind, n = fut.result()
                totals[ind] = n
            except Exception as e:
                print(f"FAIL {futs[fut]}: {e}")
                import traceback; traceback.print_exc()

    print("\n=== SUMMARY ===")
    grand = 0
    for ind in indenters:
        n = totals.get(ind, 0)
        grand += n
        print(f"  {ind:10s}  {n:>8,}")
    print(f"  {'TOTAL':10s}  {grand:>8,}")


if __name__ == "__main__":
    main()
