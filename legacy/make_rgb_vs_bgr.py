#!/usr/bin/env python3
"""Generate side-by-side RGB-vs-BGR sample grids per subset.

For each subset (main repo + NC repo), pick K random samples from the
parquet, decode JPEG as RGB (current pipeline assumption), and render two
side-by-side panels per sample:
  - Left: RGB interpretation (as-stored)
  - Right: BGR interpretation (channels swapped R↔B)

Output: samples_rgb_vs_bgr_<subset>.png per subset.

Saves to assets/ in the respective repo's local dir for later push.
"""
import argparse
import glob
import io
import os
import random
import sys

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

MAIN_BASE = "/media/yxma/Disk1/yuxiang/mini_data_parquet"
NC_BASE = "/media/yxma/Disk1/yuxiang/mini_data_parquet_nc"

K = 6              # samples per subset
THUMB = 200        # pixels per panel
PAD = 8
TITLE_H = 32
LABEL_H = 22


def decode_rgb(b):
    return np.array(Image.open(io.BytesIO(b)).convert("RGB"))


def swap_rb(rgb):
    """Treat array as BGR by swapping channels 0 and 2 → effectively
    'what it would look like if the source was stored as BGR'."""
    return rgb[..., ::-1].copy()


def thumbnail(rgb, side=THUMB):
    im = Image.fromarray(rgb)
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    return im.resize((side, side), Image.LANCZOS)


def make_comparison(images_rgb, subset_label):
    """Build a 2-row × K-column grid:
       row 0 = RGB, row 1 = BGR(R↔B swapped).
       Pre/post images for the same source bytes — visually identical
       contact pattern, different colors."""
    cols = len(images_rgb)
    rows = 2
    W = PAD + cols * (THUMB + PAD)
    H = TITLE_H + rows * (THUMB + LABEL_H + PAD) + PAD
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    try:
        f_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
        f_label = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
    except Exception:
        f_title = ImageFont.load_default()
        f_label = ImageFont.load_default()
    d.text((PAD, 8), f"{subset_label} — RGB (as stored) vs BGR (R↔B swap)",
           fill=(0, 0, 0), font=f_title)

    for row_idx, (label, transform) in enumerate(
        [("interpret as RGB", lambda x: x),
         ("interpret as BGR (R↔B)", swap_rb)]
    ):
        y0 = TITLE_H + row_idx * (THUMB + LABEL_H + PAD)
        d.text((PAD, y0), label, fill=(0, 0, 0), font=f_label)
        for c, rgb in enumerate(images_rgb):
            im = thumbnail(transform(rgb))
            x = PAD + c * (THUMB + PAD)
            y = y0 + LABEL_H
            canvas.paste(im, (x, y))
    return canvas


def collect_samples(parquet_paths, n, seed):
    rng = random.Random(seed)
    counts = [pq.read_metadata(p).num_rows for p in parquet_paths]
    total = sum(counts)
    if total == 0: return []
    idxs = sorted(rng.sample(range(total), min(n, total)))
    out = []
    cum = 0
    it = iter(idxs); nxt = next(it, None)
    for p, c in zip(parquet_paths, counts):
        if nxt is None: break
        if nxt >= cum + c: cum += c; continue
        local = []
        while nxt is not None and nxt < cum + c:
            local.append(nxt - cum); nxt = next(it, None)
        if local:
            t = pq.read_table(p, columns=["image"])
            for li in local:
                out.append(decode_rgb(t.column("image")[li].as_py()))
        cum += c
    return out


def process_subset(base_dir, sub, label, out_dir):
    paths = sorted(glob.glob(f"{base_dir}/{sub}/*.parquet"))
    if not paths:
        return None
    images = collect_samples(paths, K, seed=hash(sub) & 0xFFFFFFFF)
    if not images:
        return None
    canvas = make_comparison(images, label)
    out_p = f"{out_dir}/samples_rgb_vs_bgr_{sub.replace('/', '_')}.png"
    canvas.save(out_p, optimize=True)
    return out_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true",
                    help="also push to HF after generation")
    args = ap.parse_args()

    # Main repo subsets
    main_subs = ["fota_labeled", "fota_unlabeled", "threedcal", "feats",
                 "gelslam", "tactile_tracking", "real_tactile_mnist",
                 "feelanyforce", "unit", "tacquad",
                 "sim_tactile_mnist", "sim_starstruck"]
    main_out_dir = f"{MAIN_BASE}/assets"
    os.makedirs(main_out_dir, exist_ok=True)
    print(f"=== Main repo: {len(main_subs)} subsets ===")
    main_results = []
    for sub in main_subs:
        p = process_subset(MAIN_BASE, sub, sub, main_out_dir)
        if p:
            print(f"  wrote {os.path.basename(p)}")
            main_results.append(p)

    # NC repo subsets (sparsh and faf_force_estimation)
    nc_out_dir = f"{NC_BASE}/assets"
    os.makedirs(nc_out_dir, exist_ok=True)
    print(f"\n=== NC repo ===")
    nc_results = []
    for sub in ["sparsh", "faf_force_estimation"]:
        p = process_subset(NC_BASE, sub, sub, nc_out_dir)
        if p:
            print(f"  wrote {os.path.basename(p)}")
            nc_results.append(p)

    if not args.push:
        return

    # Push
    print("\n=== Pushing to HF ===")
    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi()

    # Main repo
    ops = []
    for p in main_results:
        rel = "assets/" + os.path.basename(p)
        ops.append(CommitOperationAdd(path_in_repo=rel, path_or_fileobj=p))
    if ops:
        info = api.create_commit(
            repo_id="yxma/gelsight-mini-pretrain", repo_type="dataset",
            operations=ops,
            commit_message=("add RGB-vs-BGR comparison grids for each subset "
                            "(side-by-side: top=RGB as stored, bottom=BGR R↔B swap)"))
        print(f"main: {info.commit_url}")

    # NC repo
    ops = []
    for p in nc_results:
        rel = "assets/" + os.path.basename(p)
        ops.append(CommitOperationAdd(path_in_repo=rel, path_or_fileobj=p))
    if ops:
        info = api.create_commit(
            repo_id="yxma/gelsight-mini-pretrain-nc", repo_type="dataset",
            operations=ops,
            commit_message=("add RGB-vs-BGR comparison grids for sparsh + faf"))
        print(f"nc:   {info.commit_url}")


if __name__ == "__main__":
    main()
