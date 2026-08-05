#!/usr/bin/env python3
"""Classify each FoTA Mini capture as markered or markerless.

Approach: average ~50 frames per capture to cancel out time-varying tactile
impressions and reveal the static gel surface. Then detect dark dots (markers
appear as ~80 small dark circular spots arranged in a grid; markerless gels
have <~20 spurious dark blobs from vignetting/specular highlights).

For each capture:
  - sample N frames evenly across train split
  - compute mean grayscale image
  - find connected dark blobs at marker-typical size range
  - count blobs; report verdict + score

Output: /home/yxma/MultimodalData/fota_marker_classification.json
Also: visual receipt /tmp/fota_marker_grid.png with mean-image thumbnails per capture.
"""

import glob
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

BASE = "/media/yxma/Disk1/yuxiang/mini_data/markerless"
SUBSETS = ["FoTa_labeled", "FoTa_unlabeled"]
SAMPLES_PER_CAPTURE = 50
MARKER_AREA_MIN_640 = 30    # px area on 640x480 image
MARKER_AREA_MAX_640 = 400
DOT_THRESHOLD_OFFSET = 12   # dark threshold below 5th-percentile pixel value
MARKER_COUNT_THRESHOLD = 30  # ≥30 valid dark dots in mean image → markered


def process_capture(args):
    sub, cap_name, cap_dir = args
    train_dir = os.path.join(cap_dir, "train")
    if not os.path.isdir(train_dir):
        return None
    paths = sorted([
        os.path.join(train_dir, f)
        for f in os.listdir(train_dir)
        if f.endswith(".jpg")
    ])
    if not paths:
        return None
    # Sample evenly
    n = min(SAMPLES_PER_CAPTURE, len(paths))
    idxs = np.linspace(0, len(paths) - 1, n, dtype=int)
    samples = [paths[i] for i in idxs]

    accum = None
    cnt = 0
    img_shape = None
    for p in samples:
        try:
            im = np.asarray(Image.open(p).convert("L"), dtype=np.float32)
            if accum is None:
                accum = im.copy()
                img_shape = im.shape
            else:
                accum += im
            cnt += 1
        except Exception:
            continue
    if cnt == 0:
        return None
    mean_img = accum / cnt

    # Adaptive thresholding: dark pixels are blob candidates
    p5 = np.percentile(mean_img, 5)
    mask = mean_img < (p5 - DOT_THRESHOLD_OFFSET)

    # Scale area bounds for image resolution
    h, w = img_shape
    scale_factor = (h * w) / (480 * 640)
    area_min = int(MARKER_AREA_MIN_640 * scale_factor)
    area_max = int(MARKER_AREA_MAX_640 * scale_factor)

    lab, nblob = ndimage.label(mask)
    if nblob == 0:
        valid_dots = 0
        all_blobs = 0
    else:
        sizes = ndimage.sum(mask, lab, range(1, nblob + 1))
        valid_dots = int(((sizes >= area_min) & (sizes <= area_max)).sum())
        all_blobs = int(nblob)

    return {
        "subset": sub,
        "capture": cap_name,
        "n_samples": cnt,
        "img_h": int(img_shape[0]),
        "img_w": int(img_shape[1]),
        "valid_dots": valid_dots,
        "all_blobs": all_blobs,
        "markered": bool(valid_dots >= MARKER_COUNT_THRESHOLD),
        "mean_brightness": float(mean_img.mean()),
        "mean_std": float(mean_img.std()),
    }


def main():
    jobs = []
    for sub in SUBSETS:
        sub_dir = os.path.join(BASE, sub)
        if not os.path.isdir(sub_dir):
            continue
        for cap in sorted(os.listdir(sub_dir)):
            if "_Mini_Mini_" not in cap:
                continue
            cap_dir = os.path.join(sub_dir, cap)
            if not os.path.isdir(cap_dir):
                continue
            jobs.append((sub, cap, cap_dir))
    print(f"Processing {len(jobs)} captures...", flush=True)

    results = []
    workers = max(2, min(6, (os.cpu_count() or 4)))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(process_capture, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r:
                results.append(r)
            if i % 20 == 0:
                print(f"  {i}/{len(jobs)}  ({time.time()-t0:.0f}s)", flush=True)

    # Summarize
    results.sort(key=lambda r: (r["subset"], r["capture"]))
    print(f"\n=== Summary ===\nTotal captures classified: {len(results)}")
    by_sub = Counter()
    by_marker = Counter()
    for r in results:
        by_sub[r["subset"]] += 1
        by_marker[(r["subset"], r["markered"])] += 1
    for sub in SUBSETS:
        m = by_marker[(sub, True)]
        u = by_marker[(sub, False)]
        print(f"  {sub}: {m} markered, {u} markerless")

    out = "/home/yxma/MultimodalData/fota_marker_classification.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out}")

    # Distribution of valid_dots
    counts = [r["valid_dots"] for r in results]
    print(f"\nvalid_dots distribution:")
    print(f"  min={min(counts)} max={max(counts)} mean={np.mean(counts):.1f} median={np.median(counts):.1f}")
    bins = [0, 5, 10, 20, 30, 50, 100, 200]
    hist, _ = np.histogram(counts, bins=bins + [10000])
    for i, b in enumerate(bins):
        next_b = bins[i + 1] if i + 1 < len(bins) else "∞"
        print(f"  [{b:4d}, {next_b}): {hist[i]} captures")


if __name__ == "__main__":
    main()
