#!/usr/bin/env python3
"""Re-filter parquet shards with the noise-normalized z-score validity rule.

Replaces the legacy fixed-threshold rule:
    diff      = |frame - baseline|
    mask      = diff > 10
    intensity = diff[mask].mean()
    keep iff area >= 40 AND intensity >= 15

with the per-pixel z-score rule:
    baseline    = median over a global pool of N_BASELINE frames
    sigma_pixel = std over the same pool (floored at EPSILON)
    z           = |frame - baseline| / sigma_pixel
    mask        = z > Z_PIX
    signal      = z[mask].mean()                # in units of σ
    keep iff area >= A_MIN AND signal >= S_MIN

The z-score is dimensionless and self-calibrates per-pixel, so the same
S_MIN threshold works for both **real** (where σ ≈ 6–10 grey-levels) and
**sim** (where σ ≈ EPSILON ≈ 1).

A small fraction (BG_RATE) of frames that fail the gate are still kept
to preserve background diversity for the VAE.

Usage:
    python reprocess_v7_zscore.py real_tactile_mnist \
        sim_tactile_mnist sim_starstruck --workers 3

Runs each subset in its own process so the 3 reprocesses are parallel.
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

BASE = "/media/yxma/Disk1/yuxiang/mini_data_parquet"

# Z-score gating parameters
N_BASELINE = 100        # frames used for baseline + sigma estimation
EPSILON    = 1.0        # floor for sigma_pixel (1 grey-level)
Z_PIX      = 3.0        # per-pixel z-threshold (mask = z > Z_PIX)
A_MIN      = 40         # minimum number of "active" pixels
S_MIN      = 4.0        # minimum mean z over active pixels (σ units)
BG_RATE    = 0.015      # keep this fraction of rejected frames as background


# ---------- helpers ----------------------------------------------------------

def grey_center(rgb: np.ndarray) -> np.ndarray:
    g = rgb.mean(axis=2).astype(np.float32)
    h, w = g.shape
    return g[h // 4:3 * h // 4, w // 4:3 * w // 4]


def decode_grey(img_bytes: bytes):
    try:
        rgb = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
        return grey_center(rgb)
    except Exception:
        return None


def list_paths(sub: str):
    return sorted(glob.glob(f"{BASE}/{sub}/*.parquet"))


def compute_global_baseline(paths, rng):
    """Sample N_BASELINE random rows across all shards; compute pixel-wise
    median (baseline) and std (sigma_pixel)."""
    counts = [pq.read_metadata(p).num_rows for p in paths]
    total = sum(counts)
    idxs = sorted(rng.sample(range(total), min(N_BASELINE, total)))
    # Walk shards to collect those rows efficiently
    grays = []
    cum = 0
    it = iter(idxs)
    nxt = next(it, None)
    for p, c in zip(paths, counts):
        if nxt is None:
            break
        if nxt >= cum + c:
            cum += c
            continue
        local = []
        while nxt is not None and nxt < cum + c:
            local.append(nxt - cum)
            nxt = next(it, None)
        if local:
            t = pq.read_table(p, columns=["image"])
            for li in local:
                g = decode_grey(t.column("image")[li].as_py())
                if g is not None:
                    grays.append(g)
        cum += c
    stack = np.stack(grays).astype(np.float32)
    baseline = np.median(stack, axis=0)
    sigma_pixel = stack.std(axis=0)
    sigma_pixel = np.maximum(sigma_pixel, EPSILON)
    return baseline, sigma_pixel, len(grays)


# ---------- per-shard filter -------------------------------------------------

def filter_shard(in_path, baseline, sigma_pixel, rng, a_min=A_MIN, s_min=S_MIN):
    """Read one parquet shard, filter rows by z-score rule, write back."""
    fname = os.path.basename(in_path)
    print(f"  [{fname}] reading...", flush=True)
    t = pq.read_table(in_path)
    n = t.num_rows
    # Cast image bytes column to large_binary to allow safe take() later
    if "image" in t.column_names and pa.types.is_binary(t.schema.field("image").type):
        i = t.column_names.index("image")
        t = t.set_column(i, "image", t.column("image").cast(pa.large_binary()))
    images = t.column("image")

    keep_mask = np.zeros(n, dtype=bool)
    n_pass = n_bg = 0
    t0 = time.time()
    for i in range(n):
        b = images[i].as_py()
        g = decode_grey(b)
        if g is None:
            continue
        diff = np.abs(g - baseline)
        z = diff / sigma_pixel
        mask = z > Z_PIX
        area = int(mask.sum())
        signal = float(z[mask].mean()) if area > 0 else 0.0
        passes = (area >= A_MIN) and (signal >= S_MIN)
        if passes:
            keep_mask[i] = True
            n_pass += 1
        elif rng.random() < BG_RATE:
            keep_mask[i] = True
            n_bg += 1
        if (i + 1) % 10000 == 0:
            fps = (i + 1) / (time.time() - t0)
            print(f"  [{fname}] {i+1:>7,}/{n:>7,}  ({fps:.0f} fps)  "
                  f"pass={n_pass:>6,}  bg={n_bg:>5,}", flush=True)

    kept = int(keep_mask.sum())
    print(f"  [{fname}] DONE  kept={kept:,}/{n:,} ({100*kept/n:.1f}%)  "
          f"pass={n_pass:,}  bg={n_bg:,}  in {time.time()-t0:.0f}s",
          flush=True)

    # Apply filter, cast image back to plain binary for repo-wide parity
    t2 = t.filter(pa.array(keep_mask))
    if "image" in t2.column_names and pa.types.is_large_binary(
        t2.schema.field("image").type
    ):
        i = t2.column_names.index("image")
        t2 = t2.set_column(i, "image", t2.column("image").cast(pa.binary()))

    tmp = in_path + ".tmp"
    pq.write_table(t2, tmp, compression="snappy")
    os.replace(tmp, in_path)
    return n, kept, n_pass, n_bg


# ---------- per-subset worker (one process per subset) -----------------------

def process_subset(sub: str, seed: int):
    rng = random.Random(seed)
    t_total = time.time()
    paths = list_paths(sub)
    if not paths:
        return sub, 0, 0, "no parquet"
    print(f"\n=== {sub} ({len(paths)} shards) ===  starting...", flush=True)

    # 1. Global baseline + sigma
    t0 = time.time()
    baseline, sigma_pixel, n_used = compute_global_baseline(paths, rng)
    print(f"  baseline from {n_used} frames  "
          f"sigma stats: min={sigma_pixel.min():.2f} "
          f"mean={sigma_pixel.mean():.2f} max={sigma_pixel.max():.2f}  "
          f"({time.time()-t0:.0f}s)", flush=True)

    # 2. Filter each shard in place
    grand = dict(seen=0, kept=0, pass_=0, bg=0)
    for p in paths:
        n, kept, n_pass, n_bg = filter_shard(p, baseline, sigma_pixel, rng)
        grand["seen"] += n
        grand["kept"] += kept
        grand["pass_"] += n_pass
        grand["bg"] += n_bg

    rate = 100 * grand["kept"] / max(1, grand["seen"])
    print(f"\n=== {sub} done in {time.time()-t_total:.0f}s ===")
    print(f"  seen={grand['seen']:,}  kept={grand['kept']:,}  ({rate:.1f}%)  "
          f"pass={grand['pass_']:,}  bg={grand['bg']:,}", flush=True)
    return sub, grand["seen"], grand["kept"], "ok"


# ---------- main -------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subsets", nargs="+",
                    help="subset names under "+BASE)
    ap.add_argument("--workers", type=int, default=3,
                    help="number of subsets to process in parallel")
    ap.add_argument("--seed", type=int, default=20260518)
    args = ap.parse_args()

    print(f"z-score reprocess: Z_PIX={Z_PIX}  A_MIN={A_MIN}  S_MIN={S_MIN}  "
          f"EPSILON={EPSILON}  BG_RATE={BG_RATE}")
    print(f"subsets: {args.subsets}  workers: {args.workers}")

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_subset, sub, args.seed + i): sub
                for i, sub in enumerate(args.subsets)}
        for fut in as_completed(futs):
            try:
                results.append(fut.result())
            except Exception as e:
                results.append((futs[fut], 0, 0, f"FAIL: {e}"))

    print("\n=== SUMMARY ===")
    print(f"{'subset':22s} {'before':>10s} {'after':>10s} {'ret':>6s}  msg")
    for sub, n_in, n_out, msg in results:
        ret = f"{100*n_out/max(1,n_in):.1f}%"
        print(f"{sub:22s} {n_in:>10,} {n_out:>10,} {ret:>6s}  {msg}")


if __name__ == "__main__":
    main()
