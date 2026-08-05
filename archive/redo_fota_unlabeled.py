#!/usr/bin/env python3
"""Re-extract fota_unlabeled from raw JPEGs with loose dedupe → 200K frames.

The existing mini_data_parquet/fota_unlabeled/ shards are post strict dedupe
(PHASH_DIST=4, lookback=30) = 66K rows. To grow to 200K we re-process from
raw JPEGs with **looser dedupe** (PHASH_DIST=2, lookback=10) and stride-cap.

Per-capture worker:
  1. List all frame_*.jpg under <cap>/{train,val}/
  2. Sample 30 random frames → median → baseline (central crop, grey)
  3. For each frame:
       - validity (A ≥ 40, I ≥ 10) + 1.5% bg-keep
       - phash, drop if hamming ≤ 2 to any of last 10 kept
  4. Return rows (image bytes + metadata)
Main:
  - imap_unordered over captures, collect into one list
  - stride-cap to 200K
  - write shards (train + val split)

Usage:
  python redo_fota_unlabeled.py --workers 16
"""
import argparse, glob, io, os, random, sys, time
import multiprocessing as mp
from collections import defaultdict

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

# cv2 thread limit — each subprocess should not spawn its own thread pool
cv2.setNumThreads(1)


def decode_bgr(path):
    """Fast JPEG decode via cv2. Returns HxWx3 uint8 BGR (cv2 native)."""
    return cv2.imread(path, cv2.IMREAD_COLOR)


def grey_center_fast(bgr):
    """Crop center 50% first, then convert to grey float32. ~10x faster than
    mean(axis=2) on full image."""
    h, w = bgr.shape[:2]
    crop = bgr[h//4:3*h//4, w//4:3*w//4]
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return g.astype(np.float32)


def encode_jpeg_cv2(bgr, q=92):
    """cv2 JPEG encode — ~2x faster than PIL."""
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
    return buf.tobytes() if ok else b""

sys.path.insert(0, "/home/yxma/MultimodalData")
from make_parquet_v2 import SCHEMA, phash, hamming, encode_jpeg, grey_center

RAW_ROOT = "/media/yxma/Disk1/yuxiang/mini_data/markerless/FoTa_unlabeled"
OUT_DIR = "/media/yxma/Disk1/yuxiang/mini_data_parquet/fota_unlabeled"
PIXEL_THRESH = 10
A_MIN = 40
I_MIN = 10
BG_RATE = 0.015
PHASH_DIST = 1          # was 4 — only drop near-identical
PHASH_LOOKBACK = 5      # was 30 — short window
N_BASELINE = 30
BUDGET = 200_000


def process_capture(args):
    capture, worker_seed = args
    # Pin each worker to a single thread for cv2/numpy so 8 workers don't
    # collectively spawn 32+ contending threads on a 4-core CPU.
    cv2.setNumThreads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    rng = random.Random(worker_seed)
    # Collect frames from both train and val
    files = []
    for split in ("train", "val"):
        for f in sorted(glob.glob(f"{RAW_ROOT}/{capture}/{split}/frame_*.jpg")):
            files.append((f, split))
    if not files: return []

    # Baseline: random sample
    sample = rng.sample(files, min(N_BASELINE, len(files)))
    grays_sample = []
    for f, _ in sample:
        bgr = decode_bgr(f)
        if bgr is None: continue
        grays_sample.append(grey_center_fast(bgr))
    if len(grays_sample) < 5:
        return []
    baseline = np.median(np.stack(grays_sample), axis=0)

    # Parse pose/side from capture name
    # e.g. "blackclamp_Mini_Mini_init_pose_3_unlabeled_left"
    side = "left" if capture.endswith("_left") else "right" if capture.endswith("_right") else None
    init_pose = None
    parts = capture.split("_")
    for i, p in enumerate(parts):
        if p == "pose" and i + 1 < len(parts):
            try: init_pose = int(parts[i + 1])
            except: pass
            break
    obj_name = parts[0]  # e.g. blackclamp, foldingknife

    out = []
    recent_hashes = []
    n_pass = n_bg = n_dup = 0
    for f, split in files:
        bgr = decode_bgr(f)
        if bgr is None: continue
        g = grey_center_fast(bgr)
        diff = np.abs(g - baseline)
        mask = diff > PIXEL_THRESH
        area = int(mask.sum())
        inten = float(diff[mask].mean()) if area > 0 else 0.0
        passes = (area >= A_MIN) and (inten >= I_MIN)
        if passes:
            n_pass += 1
        else:
            if rng.random() >= BG_RATE: continue
            n_bg += 1

        # phash dedupe (phash needs RGB)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h = phash(rgb)
        if any(hamming(h, hh) <= PHASH_DIST for hh in recent_hashes[-PHASH_LOOKBACK:]):
            n_dup += 1
            continue
        recent_hashes.append(h)

        frame_idx = None
        try:
            frame_idx = int(os.path.basename(f).split("_")[1].split(".")[0])
        except Exception:
            pass

        out.append({
            "image": encode_jpeg_cv2(bgr),
            "image_format": "jpeg",
            "source": "fota_unlabeled",
            "markered": False,
            "domain": "real",
            "capture": capture,
            "split": split,
            "height": int(bgr.shape[0]),
            "width": int(bgr.shape[1]),
            "obj_name": obj_name,
            "init_pose": init_pose,
            "side": side,
            "frame_idx": frame_idx,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--budget", type=int, default=BUDGET)
    args = ap.parse_args()

    captures = sorted(os.listdir(RAW_ROOT))
    captures = [c for c in captures if os.path.isdir(f"{RAW_ROOT}/{c}")]
    print(f"captures: {len(captures)} | workers: {args.workers} | budget: {args.budget:,}",
          flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    # Clean only the parquet files (don't touch caches)
    for fp in glob.glob(f"{OUT_DIR}/*.parquet"): os.remove(fp)

    tasks = [(c, hash(c) & 0xFFFF) for c in captures]
    t0 = time.time()

    all_rows = []
    with mp.Pool(args.workers) as pool:
        for i, rows in enumerate(pool.imap_unordered(process_capture, tasks)):
            all_rows.extend(rows)
            dt = time.time() - t0
            print(f"  [{i+1}/{len(tasks)}] cap done · "
                  f"got={len(rows):,} · total={len(all_rows):,} · "
                  f"{len(all_rows)/max(dt,0.01):.0f} fps · {dt:.0f}s",
                  flush=True)

    print(f"\nALL CAPTURES DONE: {len(all_rows):,} rows ({time.time()-t0:.0f}s)",
          flush=True)

    # Stride-cap to budget
    if len(all_rows) > args.budget:
        stride = len(all_rows) / args.budget
        sampled = [all_rows[int(i * stride)] for i in range(args.budget)]
        print(f"stride-cap: {len(all_rows):,} → {len(sampled):,} (stride {stride:.2f})",
              flush=True)
        all_rows = sampled

    # Safety: pickle the row list before writing parquet
    import pickle
    pkl_path = f"{OUT_DIR}/_all_rows.pkl"
    print(f"pickling {len(all_rows):,} rows to {pkl_path} ...", flush=True)
    with open(pkl_path, "wb") as fh:
        pickle.dump(all_rows, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  pickled ok ({os.path.getsize(pkl_path)/1e9:.1f} GB)", flush=True)

    # Write shards, separating train/val
    by_split = defaultdict(list)
    for r in all_rows:
        by_split[r.get("split", "train")].append(r)
    print("split counts:", {k: len(v) for k, v in by_split.items()}, flush=True)

    SHARD_ROWS = 60_000  # ~1.5-2 GB per shard
    for sp, rows in by_split.items():
        n_shards = (len(rows) + SHARD_ROWS - 1) // SHARD_ROWS
        for si in range(n_shards):
            chunk = rows[si * SHARD_ROWS:(si + 1) * SHARD_ROWS]
            cols = {f.name: [r.get(f.name) for r in chunk] for f in SCHEMA}
            t = pa.Table.from_pydict(cols, schema=SCHEMA)
            outp = f"{OUT_DIR}/{sp}-{si:05d}-of-{n_shards:05d}.parquet"
            pq.write_table(t, outp, compression="snappy")
            print(f"  wrote {os.path.basename(outp)} rows={len(chunk):,}",
                  flush=True)

    total = sum(pq.read_metadata(p).num_rows
                for p in sorted(glob.glob(f"{OUT_DIR}/*.parquet")))
    dt = time.time() - t0
    print(f"\n=== DONE: {total:,} rows in {OUT_DIR} ({dt:.0f}s) ===",
          flush=True)


if __name__ == "__main__":
    main()
