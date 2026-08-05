#!/usr/bin/env python3
"""Multiprocessing RTM processor — ~6-8× speedup vs sequential.

Splits work by parquet ROW (each row = 1 round = 256 touches).
Each worker reads one row from one parquet file, decodes its 256 touch videos,
applies the unified A=40, I=15 + K=0.3 Bernoulli rule, returns list of
(rgb_bytes, meta) tuples. Main process collects and writes shards.

Usage:
  python parallel_rtm.py --workers 8 --output main      # writes to mini_data_parquet/real_tactile_mnist/
  python parallel_rtm.py --workers 8 --output video     # writes to mini_data_parquet_video/real_tactile_mnist/
"""
import argparse, glob, io, os, pickle, random, sys, time
from collections import defaultdict
import multiprocessing as mp

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

ROOT = "/media/yxma/Disk1/yuxiang/mini_data/markerless/RealTactileMNIST"
PIXEL_THRESH = 10
A_MIN = 40
I_MIN = 15
K_SAMPLE = 0.30
BG_KEEP_RATE = 0.015
BUDGET = 200_000

sys.path.insert(0, "/home/yxma/MultimodalData")
from make_parquet_v2 import SCHEMA as MAIN_SCHEMA, encode_jpeg


def encode_one(rgb):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=92, optimize=True)
    return buf.getvalue()


def process_one_row(args):
    """Worker entry. Args is (parquet_file, row_index, mode, worker_seed).
    Returns list of dict rows (image bytes + metadata).
    """
    parquet_file, row_index, mode, worker_seed = args
    pf = pq.ParquetFile(parquet_file)
    # Read just this row
    target_row = None
    cum = 0
    for batch in pf.iter_batches(batch_size=4):
        n = batch.num_rows
        if row_index < cum + n:
            row_in_batch = row_index - cum
            d = {c: batch.column(c)[row_in_batch].as_py() for c in batch.column_names}
            target_row = d
            break
        cum += n
    if target_row is None: return []

    split = "test" if "test" in os.path.basename(parquet_file).lower() else "train"
    label = target_row.get("label")
    obj_id = target_row.get("object_id")
    round_id = target_row.get("id")
    videos = target_row.get("sensor_video") or []
    ts_seq = target_row.get("time_stamp_rel_seq") or []
    t_start = target_row.get("touch_start_time_rel") or []
    t_end = target_row.get("touch_end_time_rel") or []

    rng = random.Random(worker_seed * 1000 + row_index)
    out = []
    for tj, vs in enumerate(videos):
        if not vs: continue
        vb = vs.get("bytes") if isinstance(vs, dict) else None
        if not vb: continue
        tmpf = f"/tmp/_rtm_p_{os.getpid()}_{tj}.mp4"
        with open(tmpf, "wb") as f: f.write(vb)
        cap = cv2.VideoCapture(tmpf)
        frames, grays = [], []
        while True:
            ok, fr = cap.read()
            if not ok: break
            rgb = fr[:, :, ::-1]
            frames.append(rgb)
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            h, w = g.shape
            grays.append(g[h//4:3*h//4, w//4:3*w//4])
        cap.release()
        try: os.remove(tmpf)
        except: pass
        if len(frames) < 8: continue

        baseline = np.median(np.stack(grays[:5]), axis=0)
        # Touch window
        in_window = list(range(len(frames)))
        try:
            ts = ts_seq[tj] if tj < len(ts_seq) else None
            ts0 = t_start[tj] if tj < len(t_start) else None
            ts1 = t_end[tj] if tj < len(t_end) else None
            if ts and ts0 and ts1 and len(ts) == len(frames):
                in_window = [k for k, t in enumerate(ts) if ts0 <= t <= ts1]
                if not in_window: in_window = list(range(len(frames)))
        except Exception: pass

        kept_within = []
        for k in in_window:
            diff = np.abs(grays[k] - baseline)
            mask = diff > PIXEL_THRESH
            area = int(mask.sum())
            inten = float(diff[mask].mean()) if area > 0 else 0.0
            passes = (area >= A_MIN) and (inten >= I_MIN)
            if passes:
                if mode == "main":
                    if rng.random() >= K_SAMPLE: continue  # K=0.3 Bernoulli sampling
                # video mode keeps all passing frames within window
            else:
                if rng.random() >= BG_KEEP_RATE: continue
            kept_within.append(k)

        for new_idx, k in enumerate(kept_within):
            img_bytes = encode_one(frames[k])
            meta = {
                "image": img_bytes, "image_format": "jpeg",
                "source": "real_tactile_mnist", "markered": False, "domain": "real",
                "capture": f"r{round_id}_t{tj}", "split": split,
                "obj_name": f"digit_{label}",
                "digit_class": int(label) if label is not None else None,
                "episode": str(obj_id) if obj_id is not None else None,
                "frame_idx": k,
                "height": int(frames[k].shape[0]),
                "width":  int(frames[k].shape[1]),
            }
            if mode == "video":
                meta["sequence_id"] = f"r{round_id}_t{tj}"
                meta["frame_in_seq"] = new_idx
                meta["sequence_length"] = len(kept_within)
                meta["fps"] = 25.0
            out.append(meta)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--output", choices=["main", "video"], default="main")
    args = ap.parse_args()

    pq_files = sorted(glob.glob(f"{ROOT}/data/*.parquet"))
    # Build full list of (file, row_idx) tasks
    tasks = []
    for p in pq_files:
        n = pq.read_metadata(p).num_rows
        for i in range(n):
            tasks.append((p, i, args.output, hash(p) & 0xFFFF))
    print(f"total tasks: {len(tasks):,} parquet-rows across {len(pq_files)} files",
          flush=True)

    out_dir = (
        "/media/yxma/Disk1/yuxiang/mini_data_parquet/real_tactile_mnist"
        if args.output == "main"
        else "/media/yxma/Disk1/yuxiang/mini_data_parquet_video/real_tactile_mnist"
    )
    os.makedirs(out_dir, exist_ok=True)
    for f in glob.glob(f"{out_dir}/*.parquet"): os.remove(f)

    # Schema
    if args.output == "video":
        schema = pa.schema(list(MAIN_SCHEMA) + [
            ("sequence_id", pa.string()),
            ("frame_in_seq", pa.int32()),
            ("sequence_length", pa.int32()),
            ("fps", pa.float32()),
        ])
    else:
        schema = MAIN_SCHEMA

    SHARD_TGT = 2 * 1024 ** 3
    shard_buf = []
    shard_bytes = 0
    shard_idx = 0
    total_kept = 0
    t0 = time.time()

    def flush():
        nonlocal shard_buf, shard_bytes, shard_idx
        if not shard_buf: return
        cols = {f.name: [r.get(f.name) for r in shard_buf] for f in schema}
        t = pa.Table.from_pydict(cols, schema=schema)
        if args.output == "video":
            outp = f"{out_dir}/train-{shard_idx:05d}.parquet"
        else:
            # mix train/test depending on split column
            train_rows = [r for r in shard_buf if r.get("split") != "test"]
            test_rows = [r for r in shard_buf if r.get("split") == "test"]
            if train_rows:
                tcols = {f.name: [r.get(f.name) for r in train_rows] for f in schema}
                pq.write_table(pa.Table.from_pydict(tcols, schema=schema),
                               f"{out_dir}/train-{shard_idx:05d}.parquet",
                               compression="snappy")
            if test_rows:
                tcols = {f.name: [r.get(f.name) for r in test_rows] for f in schema}
                pq.write_table(pa.Table.from_pydict(tcols, schema=schema),
                               f"{out_dir}/test-{shard_idx:05d}.parquet",
                               compression="snappy")
            shard_buf = []
            shard_bytes = 0
            shard_idx += 1
            return
        pq.write_table(t, outp, compression="snappy")
        print(f"  wrote {outp.split('/')[-1]} rows={len(shard_buf):,}", flush=True)
        shard_buf = []
        shard_bytes = 0
        shard_idx += 1

    print(f"starting pool with {args.workers} workers...", flush=True)
    with mp.Pool(args.workers) as pool:
        for batch_result in pool.imap_unordered(process_one_row, tasks, chunksize=2):
            for row in batch_result:
                shard_buf.append(row)
                shard_bytes += len(row["image"]) if row.get("image") else 0
                total_kept += 1
                if total_kept >= BUDGET and args.output == "main":
                    print(f"hit BUDGET={BUDGET}, stopping", flush=True)
                    flush()
                    return
                if shard_bytes >= SHARD_TGT:
                    flush()
            if total_kept % 5000 < 100 and total_kept > 0:
                dt = time.time() - t0
                print(f"  kept={total_kept:,} ({total_kept/max(dt,0.01):.0f} fps)",
                      flush=True)
    flush()

    # Rename to of-NN format
    files = sorted(glob.glob(f"{out_dir}/train-?????.parquet"))
    n = len(files)
    for i, fp in enumerate(files):
        new = f"{out_dir}/train-{i:05d}-of-{n:05d}.parquet"
        if fp != new: os.rename(fp, new)
    files = sorted(glob.glob(f"{out_dir}/test-?????.parquet"))
    n = len(files)
    for i, fp in enumerate(files):
        new = f"{out_dir}/test-{i:05d}-of-{n:05d}.parquet"
        if fp != new: os.rename(fp, new)

    dt = time.time() - t0
    print(f"DONE: total_kept={total_kept:,} in {dt:.0f}s "
          f"({total_kept/max(dt,0.01):.0f} fps)", flush=True)


if __name__ == "__main__":
    main()
