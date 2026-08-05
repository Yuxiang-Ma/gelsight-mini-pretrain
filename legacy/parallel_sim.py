#!/usr/bin/env python3
"""Multiprocessing sim_tactile_mnist / sim_starstruck processor.

Same logic as `_iter_sim_parquet_filtered` in make_parquet_v2.py but parallelised
across parquet rows. Each row = one digit/star object with 32 rendered touches.

Usage:
  python parallel_sim.py --source sim_tactile_mnist --workers 8
  python parallel_sim.py --source sim_starstruck    --workers 8
"""
import argparse, glob, io, os, random, sys, time
import multiprocessing as mp

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

PIXEL_THRESH = 10
A_MIN = 40
I_MIN = 15
BG_KEEP_RATE = 0.015
BUDGET = 200_000

sys.path.insert(0, "/home/yxma/MultimodalData")
from make_parquet_v2 import SCHEMA, BASE_DATA, encode_jpeg


SOURCES = {
    "sim_tactile_mnist": dict(
        root=f"{BASE_DATA}/markerless/SimTactileMNIST",
        obj_name_fn=lambda lbl: f"digit_{lbl}",
        has_digit=True),
    "sim_starstruck": dict(
        root=f"{BASE_DATA}/markerless/SimStarStruck",
        obj_name_fn=lambda lbl: "starstruck",
        has_digit=False),
}


def process_one_row(args):
    parquet_file, row_index, source_tag, worker_seed = args
    cfg = SOURCES[source_tag]
    pf = pq.ParquetFile(parquet_file)
    target_row = None
    cum = 0
    for batch in pf.iter_batches(batch_size=4):
        n = batch.num_rows
        if row_index < cum + n:
            row_in_batch = row_index - cum
            d = {c: batch.column(c)[row_in_batch].as_py() for c in batch.column_names}
            target_row = d; break
        cum += n
    if target_row is None: return []

    fn = os.path.basename(parquet_file).lower()
    if "test" in fn: split = "test"
    elif "val" in fn: split = "val"
    else: split = "train"

    round_id = target_row.get("id")
    label = target_row.get("label")
    obj_id = target_row.get("object_id")
    images = target_row.get("sensor_image") or []
    rng = random.Random(worker_seed * 1000 + row_index)

    decoded = []
    for tj, img_s in enumerate(images):
        if not img_s: continue
        b = img_s.get("bytes") if isinstance(img_s, dict) else None
        if not b: continue
        try:
            rgb = np.array(Image.open(io.BytesIO(b)).convert("RGB"))
            g = rgb.mean(axis=2).astype(np.float32)
            h, w = g.shape
            decoded.append((rgb, g[h//4:3*h//4, w//4:3*w//4], tj))
        except Exception:
            pass
    if len(decoded) < 5: return []
    baseline = np.median(np.stack([d[1] for d in decoded]), axis=0)

    out = []
    for rgb, g_center, tj in decoded:
        diff = np.abs(g_center - baseline)
        mask = diff > PIXEL_THRESH
        area = int(mask.sum())
        inten = float(diff[mask].mean()) if area > 0 else 0.0
        passes = (area >= A_MIN) and (inten >= I_MIN)
        if not passes and rng.random() >= BG_KEEP_RATE: continue
        img_bytes = encode_jpeg(rgb)
        meta = {
            "image": img_bytes, "image_format": "jpeg",
            "source": source_tag, "markered": False, "domain": "sim",
            "capture": f"r{round_id}_t{tj}", "split": split,
            "obj_name": cfg["obj_name_fn"](label),
            "episode": str(obj_id) if obj_id is not None else None,
            "frame_idx": tj,
            "height": int(rgb.shape[0]),
            "width": int(rgb.shape[1]),
        }
        if cfg["has_digit"]:
            meta["digit_class"] = int(label) if label is not None else None
        out.append(meta)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=list(SOURCES.keys()))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    root = SOURCES[args.source]["root"]
    pq_files = sorted(glob.glob(f"{root}/data/*.parquet"))
    tasks = []
    for p in pq_files:
        n = pq.read_metadata(p).num_rows
        for i in range(n):
            tasks.append((p, i, args.source, hash(p) & 0xFFFF))
    print(f"{args.source}: {len(tasks):,} tasks across {len(pq_files)} files",
          flush=True)

    out_dir = f"/media/yxma/Disk1/yuxiang/mini_data_parquet/{args.source}"
    os.makedirs(out_dir, exist_ok=True)
    for f in glob.glob(f"{out_dir}/*.parquet"): os.remove(f)

    SHARD_TGT = 2 * 1024 ** 3
    shard_buf = []
    shard_bytes = 0
    shard_idx = 0
    total_kept = 0
    t0 = time.time()

    def flush(force_split=None):
        nonlocal shard_buf, shard_bytes, shard_idx
        if not shard_buf: return
        # Split by 'split' column
        from collections import defaultdict as _dd
        by_split = _dd(list)
        for r in shard_buf:
            by_split[r.get("split","train")].append(r)
        for sp, rows in by_split.items():
            cols = {f.name: [r.get(f.name) for r in rows] for f in SCHEMA}
            t = pa.Table.from_pydict(cols, schema=SCHEMA)
            outp = f"{out_dir}/{sp}-{shard_idx:05d}.parquet"
            pq.write_table(t, outp, compression="snappy")
            print(f"  wrote {os.path.basename(outp)} rows={len(rows):,}", flush=True)
        shard_buf = []; shard_bytes = 0; shard_idx += 1

    with mp.Pool(args.workers) as pool:
        for batch_result in pool.imap_unordered(process_one_row, tasks, chunksize=4):
            for row in batch_result:
                shard_buf.append(row)
                shard_bytes += len(row["image"]) if row.get("image") else 0
                total_kept += 1
                if total_kept >= BUDGET:
                    print(f"hit BUDGET={BUDGET}", flush=True)
                    flush();
                    # rename
                    for split in ["train", "test", "val"]:
                        files = sorted(glob.glob(f"{out_dir}/{split}-?????.parquet"))
                        n = len(files)
                        for i, fp in enumerate(files):
                            new = f"{out_dir}/{split}-{i:05d}-of-{n:05d}.parquet"
                            if fp != new: os.rename(fp, new)
                    return
                if shard_bytes >= SHARD_TGT: flush()
            if total_kept % 5000 < 100:
                dt = time.time() - t0
                print(f"  kept={total_kept:,} ({total_kept/max(dt,0.01):.0f} fps)",
                      flush=True)
    flush()
    for split in ["train", "test", "val"]:
        files = sorted(glob.glob(f"{out_dir}/{split}-?????.parquet"))
        n = len(files)
        for i, fp in enumerate(files):
            new = f"{out_dir}/{split}-{i:05d}-of-{n:05d}.parquet"
            if fp != new: os.rename(fp, new)
    dt = time.time() - t0
    print(f"DONE: total_kept={total_kept:,} in {dt:.0f}s "
          f"({total_kept/max(dt,0.01):.0f} fps)", flush=True)


if __name__ == "__main__":
    main()
