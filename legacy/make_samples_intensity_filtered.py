#!/usr/bin/env python3
"""Re-sample sample-grid PNGs with an intensity floor so empty frames are
excluded. Used when a subset's parquet still contains low-/zero-contact
frames (e.g. RTM / sim subsets that haven't been through the unified
area+intensity filter yet).

For each subset:
  1. Compute a per-source baseline = median of N_BASELINE random frames
     (cross-frame median ≈ gel-at-rest because per-pixel contact is rare).
  2. Stream another POOL random frames, compute area+intensity vs baseline
     (central 50% crop, greyscale, pixel_diff > 10).
  3. Keep frames with intensity ≥ I_MIN, randomly pick N for the grid.

Runs the 4 subsets in parallel via ProcessPoolExecutor.
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
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

BASE = "/media/yxma/Disk1/yuxiang/mini_data_parquet"
OUT = os.path.join(BASE, "assets")

PIXEL_THRESH = 10           # noise floor for |frame - baseline|
A_MIN_FOR_VIZ = 40          # area floor (same as unified pipeline)
N_BASELINE = 30             # frames used to estimate gel-at-rest median
POOL = 1200                 # candidate pool per subset (random reads)
N_GRID = 40                 # frames per output grid
COLS = 10
THUMB = 144


def grey_center(rgb):
    g = rgb.mean(axis=2).astype(np.float32)
    h, w = g.shape
    return g[h // 4:3 * h // 4, w // 4:3 * w // 4]


def list_parquets(sub):
    return sorted(glob.glob(os.path.join(BASE, sub, "*.parquet")))


def random_global_indices(counts, k, rng):
    total = sum(counts)
    return sorted(rng.sample(range(total), min(k, total)))


def fetch_rows(paths, counts, idxs):
    """Yield (global_idx, jpeg_bytes) for global indices in idxs (sorted)."""
    cum = 0
    it = iter(idxs)
    nxt = next(it, None)
    for p, c in zip(paths, counts):
        if nxt is None:
            return
        if nxt >= cum + c:
            cum += c
            continue
        local = []
        while nxt is not None and nxt < cum + c:
            local.append((nxt, nxt - cum))
            nxt = next(it, None)
        if local:
            t = pq.read_table(p, columns=["image"])
            for gi, li in local:
                yield gi, t.column("image")[li].as_py()
        cum += c


def thumbnail(img_bytes, side=THUMB):
    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    return im.resize((side, side), Image.LANCZOS)


def make_grid(images, title, cols=COLS, side=THUMB, pad=4, title_h=44):
    rows = (len(images) + cols - 1) // cols
    W = pad + cols * (side + pad)
    H = title_h + rows * (side + pad) + pad
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    try:
        f = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
    except Exception:
        f = ImageFont.load_default()
    d.text((pad, 8), title, fill=(0, 0, 0), font=f)
    for i, im in enumerate(images):
        r, c = i // cols, i % cols
        x = pad + c * (side + pad)
        y = title_h + r * (side + pad)
        canvas.paste(im, (x, y))
    return canvas


def worker(sub, i_min, seed):
    t0 = time.time()
    rng = random.Random(seed)
    paths = list_parquets(sub)
    if not paths:
        return sub, 0, 0, 0, f"no parquet under {sub}/"
    counts = [pq.read_metadata(p).num_rows for p in paths]
    total = sum(counts)

    # Step 1: baseline
    base_idxs = random_global_indices(counts, N_BASELINE, rng)
    grays = []
    for gi, img_bytes in fetch_rows(paths, counts, base_idxs):
        try:
            rgb = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
            grays.append(grey_center(rgb))
        except Exception:
            pass
    if len(grays) < 5:
        return sub, total, 0, 0, "not enough frames for baseline"
    baseline = np.median(np.stack(grays), axis=0)

    # Step 2: scan POOL random candidates, keep those passing the intensity floor
    cand_idxs = random_global_indices(counts, POOL, rng)
    survivors = []  # list of jpeg bytes
    n_scanned = n_kept = 0
    for gi, img_bytes in fetch_rows(paths, counts, cand_idxs):
        n_scanned += 1
        try:
            rgb = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
        except Exception:
            continue
        g = grey_center(rgb)
        diff = np.abs(g - baseline)
        mask = diff > PIXEL_THRESH
        area = int(mask.sum())
        inten = float(diff[mask].mean()) if area > 0 else 0.0
        if area >= A_MIN_FOR_VIZ and inten >= i_min:
            survivors.append(img_bytes)
            n_kept += 1

    # Step 3: pick N for the grid
    if len(survivors) < N_GRID:
        msg = f"only {len(survivors)} contact frames in pool of {POOL}"
    else:
        survivors = rng.sample(survivors, N_GRID)
        msg = "ok"

    if len(survivors) == 0:
        return sub, total, n_scanned, n_kept, "no survivors"

    thumbs = [thumbnail(b) for b in survivors]
    title = f"{sub} — {len(thumbs)} random contact samples  (i≥{i_min})"
    grid = make_grid(thumbs, title)
    out_path = os.path.join(OUT, f"samples_{N_GRID}_{sub}.png")
    grid.save(out_path, optimize=True)
    return sub, total, n_scanned, n_kept, f"{msg}; wrote in {time.time()-t0:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--i_min", type=float, default=12.0,
                    help="intensity floor (mean |frame-baseline| on masked pixels)")
    ap.add_argument("--subs", nargs="+", default=[
        "real_tactile_mnist", "feelanyforce",
        "sim_tactile_mnist", "sim_starstruck"])
    ap.add_argument("--seed", type=int, default=20260518)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    print(f"Re-sampling {len(args.subs)} subsets with intensity floor "
          f"i_min={args.i_min}, pool={POOL}, seed={args.seed}")
    print(f"Workers: {args.workers}")

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(worker, sub, args.i_min, args.seed + i): sub
                for i, sub in enumerate(args.subs)}
        for fut in as_completed(futs):
            sub, total, scanned, kept, msg = fut.result()
            print(f"  {sub:22s}  total={total:>7,}  scanned={scanned:>4}  "
                  f"kept_in_pool={kept:>4}  ::  {msg}")


if __name__ == "__main__":
    main()
