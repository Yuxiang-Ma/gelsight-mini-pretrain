#!/usr/bin/env python3
"""Re-extract real_tactile_mnist from the VIDEO upstream (vs the prior
single-frame extract).

Upstream layout:
  RealTactileMNIST/data/{train,test}-*-of-*.parquet
  Each row = one (digit_print, gel_id) combo with 256 touches.
  Each touch is an embedded AVI byte-string of ~58-90 frames.

For each touch:
  1. Decode the AVI byte-string with imageio (ffmpeg backend).
  2. Score every decoded frame's contact intensity vs the row baseline
     (median over all frames in that touch).
  3. Keep the K_PER_TOUCH frames with highest intensity (default K=1).

Then aggregate across all touches → target ~150K kept frames.

Schema columns populated:
  source="real_tactile_mnist"
  domain="real"
  sensor_id="mini_rtm"           # via gel_variant=markerless
  capture=f"{id}_t{touch_idx}"   # e.g. "2023-10-07_22-09-26_6-3_t42"
  obj_name=f"digit_{label}"
  digit_class=label
  episode=str(object_id)         # the print id (1..600)
  frame_idx=peak_frame_idx_in_clip
  height=240, width=320
  markered=False, gel_variant="markerless"

Workers: one process per upstream parquet shard (48 train + 8 test = 56).
"""
import argparse
import glob
import io
import os
import random
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import imageio.v3 as iio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, "/home/yxma/MultimodalData")
from make_parquet_v2 import SCHEMA

RAW_BASE = "/media/yxma/Disk1/yuxiang/mini_data/markerless/RealTactileMNIST"
OUT_BASE = "/media/yxma/Disk1/yuxiang/mini_data_parquet/real_tactile_mnist"

PIXEL_THRESH = 10
A_MIN = 40
I_MIN = 12
BG_RATE = 0.015
K_PER_TOUCH = 1            # peak frame per touch
JPEG_Q = 92


def grey_center(rgb):
    g = rgb.mean(axis=2).astype(np.float32)
    h, w = g.shape
    return g[h // 4:3 * h // 4, w // 4:3 * w // 4]


def to_jpeg(rgb, q=JPEG_Q):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=q)
    return buf.getvalue()


def decode_clip(b: bytes):
    """Return a list of (H,W,3) uint8 numpy arrays."""
    # imageio_ffmpeg can read AVI bytes via a tmp file. Use named tmp to
    # avoid keeping the bytes in RAM twice.
    with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as f:
        f.write(b)
        tmppath = f.name
    try:
        frames = list(iio.imiter(tmppath, format_hint=".avi"))
    except Exception:
        frames = []
    finally:
        try:
            os.unlink(tmppath)
        except Exception:
            pass
    return frames


def best_frames(frames, baseline, k=K_PER_TOUCH):
    """Score each frame's contact intensity, return top-k indices."""
    scores = []
    for i, rgb in enumerate(frames):
        if rgb.shape != baseline.shape if rgb.ndim == 2 else False:
            pass
        g = grey_center(rgb)
        diff = np.abs(g - baseline)
        mask = diff > PIXEL_THRESH
        area = int(mask.sum())
        inten = float(diff[mask].mean()) if area > 0 else 0.0
        score = inten if area >= A_MIN else 0.0
        scores.append(score)
    scores = np.asarray(scores)
    if (scores > 0).sum() == 0:
        return []
    order = np.argsort(scores)[::-1]
    return [(int(i), float(scores[i])) for i in order[:k]
            if scores[int(i)] >= I_MIN]


def process_shard(in_path, split):
    """Decode every touch in one upstream shard, keep K_PER_TOUCH peak frames."""
    fname = os.path.basename(in_path)
    print(f"[{fname}] reading metadata...", flush=True)
    pf = pq.ParquetFile(in_path)
    n_rows = pf.metadata.num_rows
    print(f"[{fname}] {n_rows} rows × 256 touches each", flush=True)

    rows = []
    rng = random.Random(hash(fname) & 0xFFFFFFFF)
    t0 = time.time()
    # Read columns we need
    wanted = ["sensor_video", "id", "label", "object_id", "info.gel_id"]
    available = [c for c in wanted if c in pf.schema_arrow.names]
    table = pq.read_table(in_path, columns=available)

    for ri in range(n_rows):
        row = {c: table.column(c)[ri].as_py() for c in available}
        clips = row["sensor_video"]
        id_ = row["id"]
        label = row["label"]
        object_id = row.get("object_id")
        gel_id = row.get("info.gel_id")

        for ti, clip in enumerate(clips):
            if clip is None:
                continue
            b = clip.get("bytes") if isinstance(clip, dict) else None
            if b is None:
                continue
            frames = decode_clip(b)
            if not frames:
                continue
            # Per-clip baseline = median of all frames in the clip
            grays = [grey_center(f) for f in frames]
            baseline = np.median(np.stack(grays), axis=0)
            top = best_frames(frames, baseline, k=K_PER_TOUCH)
            if not top:
                # bg-diversity: keep one random frame from this touch
                if rng.random() < BG_RATE:
                    fi = rng.randint(0, len(frames) - 1)
                    top = [(fi, 0.0)]
            for fi, _ in top:
                rgb = frames[fi]
                r = {f.name: None for f in SCHEMA}
                r.update(dict(
                    image=to_jpeg(rgb),
                    image_format="jpeg",
                    source="real_tactile_mnist",
                    markered=False,
                    capture=f"{id_}_t{ti}",
                    split=split,
                    height=rgb.shape[0],
                    width=rgb.shape[1],
                    obj_name=f"digit_{label}",
                    digit_class=label,
                    episode=str(object_id) if object_id is not None else id_,
                    frame_idx=fi,
                    gel_variant="markerless",
                    domain="real",
                ))
                rows.append(r)
        if (ri + 1) % 1 == 0:
            fps = (ri + 1) / max(0.01, time.time() - t0)
            eta = (n_rows - ri - 1) / max(0.01, fps)
            if (ri + 1) % 1 == 0:
                print(f"[{fname}] row {ri+1}/{n_rows}  kept={len(rows):,}  "
                      f"({fps:.1f} rows/s  eta={eta:.0f}s)", flush=True)

    print(f"[{fname}] DONE  kept={len(rows):,} in {time.time()-t0:.0f}s",
          flush=True)
    return fname, split, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--target", type=int, default=150_000)
    args = ap.parse_args()

    train_shards = sorted(glob.glob(f"{RAW_BASE}/data/train-*.parquet"))
    test_shards = sorted(glob.glob(f"{RAW_BASE}/data/test-*.parquet"))

    all_jobs = [(p, "train") for p in train_shards] + \
               [(p, "test") for p in test_shards]
    print(f"RTM video re-extract  workers={args.workers}  "
          f"target={args.target:,}  K_PER_TOUCH={K_PER_TOUCH}")
    print(f"jobs: {len(train_shards)} train + {len(test_shards)} test "
          f"= {len(all_jobs)} shards")

    by_split = {"train": [], "test": []}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_shard, p, s): (p, s) for p, s in all_jobs}
        for fut in as_completed(futs):
            try:
                fname, split, rows = fut.result()
                by_split[split].extend(rows)
                print(f"  ACCUM  {split}={len(by_split['train'])+len(by_split['test']):,} total",
                      flush=True)
            except Exception as e:
                p, s = futs[fut]
                print(f"FAIL {p}: {e}")
                import traceback; traceback.print_exc()

    # Stride-subsample down to target per split (proportional allocation)
    total_kept = sum(len(v) for v in by_split.values())
    print(f"\n=== Pre-cap: train={len(by_split['train']):,}  "
          f"test={len(by_split['test']):,}  total={total_kept:,} ===")

    train_target = int(args.target * 0.83)  # ~83% train (153K source -> 125K train)
    test_target = args.target - train_target
    for split, target in [("train", train_target), ("test", test_target)]:
        rows = by_split[split]
        if len(rows) > target:
            idx = np.linspace(0, len(rows) - 1, target, dtype=np.int64)
            by_split[split] = [rows[int(i)] for i in idx]
        print(f"  {split}: {len(by_split[split]):,} (target {target:,})")

    # Write parquet
    os.makedirs(OUT_BASE, exist_ok=True)
    for split, rows in by_split.items():
        cols = {f.name: [r.get(f.name) for r in rows] for f in SCHEMA}
        arrays = [pa.array(cols[f.name], type=f.type) for f in SCHEMA]
        table = pa.Table.from_arrays(arrays, schema=SCHEMA)
        out_path = f"{OUT_BASE}/{split}-00000-of-00001.parquet"
        tmp = out_path + ".tmp"
        pq.write_table(table, tmp, compression="snappy")
        os.replace(tmp, out_path)
        sz = os.path.getsize(out_path) / 1e6
        print(f"wrote {out_path}  rows={table.num_rows:,}  {sz:.1f} MB")


if __name__ == "__main__":
    main()
