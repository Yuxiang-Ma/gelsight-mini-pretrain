#!/usr/bin/env python3
"""Re-filter parquet shards in place using the legacy area+intensity rule
with per-subset configurable thresholds.

    diff      = |frame - baseline|  (greyscale, central 50% crop)
    mask      = diff > PIXEL_THRESH(=10)
    area      = mask.sum()
    intensity = diff[mask].mean()                          (grey-levels)
    keep iff area >= A_MIN AND intensity >= I_MIN
    else keep with probability BG_RATE

Each subset gets its own I_MIN (sim datasets have ~zero noise floor and
can use a lower threshold; real datasets need a higher one to clear
sensor noise).

Usage:
    python reprocess_legacy.py \\
        real_tactile_mnist:12 \\
        sim_tactile_mnist:10 \\
        sim_starstruck:10 \\
        --workers 3

Argument format is `subset_name:I_MIN`. Each subset runs in its own
process; --workers limits concurrency.
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

PIXEL_THRESH = 10        # noise-floor gate on |frame - baseline|
A_MIN = 40               # minimum number of pixels passing the gate
BG_RATE = 0.015          # keep this fraction of rejected frames as bg
N_BASELINE = 100         # frames used to estimate gel-at-rest median


# ---------- helpers ----------------------------------------------------------

def grey_center(rgb):
    g = rgb.mean(axis=2).astype(np.float32)
    h, w = g.shape
    return g[h // 4:3 * h // 4, w // 4:3 * w // 4]


def decode_grey(img_bytes):
    try:
        rgb = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
        return grey_center(rgb)
    except Exception:
        return None


def list_paths(sub):
    return sorted(glob.glob(f"{BASE}/{sub}/*.parquet"))


def compute_baseline(paths, rng):
    counts = [pq.read_metadata(p).num_rows for p in paths]
    total = sum(counts)
    idxs = sorted(rng.sample(range(total), min(N_BASELINE, total)))
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
    return np.median(stack, axis=0), len(grays)


def filter_shard(in_path, baseline, i_min, rng):
    fname = os.path.basename(in_path)
    print(f"  [{fname}] reading...", flush=True)
    t = pq.read_table(in_path)
    n = t.num_rows
    # Cast image -> large_binary for safe take() if needed
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
        mask = diff > PIXEL_THRESH
        area = int(mask.sum())
        inten = float(diff[mask].mean()) if area > 0 else 0.0
        passes = (area >= A_MIN) and (inten >= i_min)
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


def process_subset(sub, i_min, seed):
    rng = random.Random(seed)
    t_total = time.time()
    paths = list_paths(sub)
    if not paths:
        return sub, 0, 0, "no parquet"
    print(f"\n=== {sub} ({len(paths)} shards, I_MIN={i_min}) ===  starting...",
          flush=True)
    t0 = time.time()
    baseline, n_used = compute_baseline(paths, rng)
    print(f"  [{sub}] baseline from {n_used} frames in {time.time()-t0:.0f}s",
          flush=True)

    grand = dict(seen=0, kept=0, pass_=0, bg=0)
    for p in paths:
        n, kept, n_pass, n_bg = filter_shard(p, baseline, i_min, rng)
        grand["seen"] += n
        grand["kept"] += kept
        grand["pass_"] += n_pass
        grand["bg"] += n_bg

    rate = 100 * grand["kept"] / max(1, grand["seen"])
    print(f"\n=== {sub} done in {time.time()-t_total:.0f}s ===  "
          f"kept={grand['kept']:,}/{grand['seen']:,} ({rate:.1f}%)",
          flush=True)
    return sub, grand["seen"], grand["kept"], "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subsets", nargs="+",
                    help="subset:i_min pairs, e.g. real_tactile_mnist:12")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260518)
    args = ap.parse_args()

    parsed = []
    for s in args.subsets:
        if ":" not in s:
            raise SystemExit(f"bad arg {s!r}, expected subset:i_min")
        sub, im = s.split(":")
        parsed.append((sub, int(im)))
    print(f"reprocess_legacy: PIXEL_THRESH={PIXEL_THRESH}  A_MIN={A_MIN}  "
          f"BG_RATE={BG_RATE}")
    for sub, im in parsed:
        print(f"  {sub:25s}  I_MIN={im}")

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_subset, sub, im, args.seed + i): sub
                for i, (sub, im) in enumerate(parsed)}
        for fut in as_completed(futs):
            try:
                results.append(fut.result())
            except Exception as e:
                results.append((futs[fut], 0, 0, f"FAIL: {e}"))

    print("\n=== SUMMARY ===")
    for sub, n_in, n_out, msg in results:
        ret = f"{100*n_out/max(1,n_in):.1f}%"
        print(f"  {sub:25s}  {n_in:>10,} -> {n_out:>10,}  ({ret})  {msg}")


if __name__ == "__main__":
    main()
