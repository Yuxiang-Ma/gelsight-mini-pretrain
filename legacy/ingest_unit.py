#!/usr/bin/env python3
"""Ingest UniT zarr replay_buffer → unified-schema parquet.

UniT (https://arxiv.org/abs/2408.06481) provides paired tactile RGB +
3D-pose target data on the GelSight Mini sensor. The 3D_pose_gelsight
replay buffer contains 11,340 frames as a (11340, 240, 320, 3) uint8
array with zstd compression, plus a (11340, 4) float32 pose array.

This script:
  1. Reads each frame from zarr
  2. Re-encodes RGB → JPEG quality 92 (matches repo schema)
  3. Fills unified SCHEMA columns:
       source="unit", domain="real", sensor_id="mini_unit",
       gel_variant="markerless", episode="3D_pose_gelsight",
       capture="3D_pose_gelsight", obj_name="unit_pose_target",
       frame_idx=row_index
  4. Writes /media/yxma/Disk1/yuxiang/mini_data_parquet/unit/train-*.parquet

Applies the unified intensity filter (PIXEL_THRESH=10, A_MIN=40,
I_MIN=12) with 1.5% bg-diversity keep rate. UniT data is real-domain
markerless captures so the real-side threshold applies.
"""
import io
import os
import random
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr
from PIL import Image

sys.path.insert(0, "/home/yxma/MultimodalData")
from make_parquet_v2 import SCHEMA

BASE_OUT = "/media/yxma/Disk1/yuxiang/mini_data_parquet"
ZARR_PATH = ("/media/yxma/Disk1/yuxiang/mini_data/markerless/UniT/"
             "UniT_dataset/3D_pose_gelsight/replay_buffer.zarr")

PIXEL_THRESH = 10
A_MIN = 40
I_MIN = 12
BG_RATE = 0.015
N_BASELINE = 120
JPEG_Q = 92


def grey_center(rgb):
    g = rgb.mean(axis=2).astype(np.float32)
    h, w = g.shape
    return g[h // 4:3 * h // 4, w // 4:3 * w // 4]


def to_jpeg(rgb, q=JPEG_Q):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=q)
    return buf.getvalue()


def main():
    rng = random.Random(20260520)
    print(f"Opening zarr at {ZARR_PATH}", flush=True)
    # No top-level .zgroup; open each array directly via DirectoryStore.
    images_store = zarr.DirectoryStore(f"{ZARR_PATH}/data/tactile_image")
    images = zarr.open_array(store=images_store, mode="r")
    poses_store = zarr.DirectoryStore(f"{ZARR_PATH}/data/3Dpose")
    poses = zarr.open_array(store=poses_store, mode="r")
    n_total = images.shape[0]
    print(f"  total frames: {n_total:,}  image shape: {images.shape}",
          flush=True)

    # Baseline: median over N_BASELINE random frames
    t0 = time.time()
    base_idxs = sorted(rng.sample(range(n_total), N_BASELINE))
    grays = [grey_center(images[i][:]) for i in base_idxs]
    baseline = np.median(np.stack(grays), axis=0)
    print(f"  baseline from {N_BASELINE} frames in {time.time()-t0:.0f}s",
          flush=True)

    # Apply contact filter — UniT has many near-black frames (sensor LED
    # off or aperture closed during ~86% of the recording). Drop those.
    rows = []
    n_pass = n_bg = 0
    t0 = time.time()
    chunk_size = 256
    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)
        chunk = images[start:end][:]
        chunk_poses = poses[start:end][:]
        for k, rgb in enumerate(chunk):
            i = start + k
            # Drop near-black frames (LED off / no signal)
            if rgb.mean() < 10:
                continue
            # Apply contact filter
            g = grey_center(rgb)
            diff = np.abs(g - baseline)
            mask = diff > PIXEL_THRESH
            area = int(mask.sum())
            inten = float(diff[mask].mean()) if area > 0 else 0.0
            passes = (area >= A_MIN) and (inten >= I_MIN)
            if passes:
                n_pass += 1
            elif rng.random() < BG_RATE:
                n_bg += 1
            else:
                continue
            if True:
                pose = chunk_poses[k]
                row = {f.name: None for f in SCHEMA}
                row.update(dict(
                    image=to_jpeg(rgb),
                    image_format="jpeg",
                    source="unit",
                    markered=True,
                    capture="3D_pose_gelsight",
                    split="train",
                    height=rgb.shape[0],
                    width=rgb.shape[1],
                    obj_name="unit_pose_target",
                    episode="3D_pose_gelsight",
                    frame_idx=i,
                    x_mm=float(pose[0]),
                    y_mm=float(pose[1]),
                    z_mm=float(pose[2]),
                    quat_z=float(pose[3]),  # actually yaw, abuse quat_z
                    gel_variant="markered",
                    domain="real",
                ))
                rows.append(row)
        if end % 1024 == 0 or end == n_total:
            fps = end / max(0.01, time.time() - t0)
            print(f"  {end:,}/{n_total:,}  pass={n_pass:,}  bg={n_bg:,}  "
                  f"kept={len(rows):,}  ({fps:.0f} fps)", flush=True)

    print(f"\nfinal: kept={len(rows):,}/{n_total:,} ({100*len(rows)/n_total:.1f}%)",
          flush=True)

    # Build table by column
    cols = {f.name: [r.get(f.name) for r in rows] for f in SCHEMA}
    arrays = [pa.array(cols[f.name], type=f.type) for f in SCHEMA]
    table = pa.Table.from_arrays(arrays, schema=SCHEMA)

    out_dir = f"{BASE_OUT}/unit"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/train-00000-of-00001.parquet"
    tmp = out_path + ".tmp"
    pq.write_table(table, tmp, compression="snappy")
    os.replace(tmp, out_path)
    print(f"wrote {out_path}  rows={table.num_rows:,}  "
          f"size={os.path.getsize(out_path)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
