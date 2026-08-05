#!/usr/bin/env python3
"""Marker classifier reading FoTA from parquet (no loose-jpg IO contention).

Groups rows by `capture` column, averages ~50 frames per capture, then runs
dark-blob detection on the mean image. A markered Mini gel produces ~80
regular dark dots in the average; markerless gels produce <20 scattered
dark blobs from vignetting/specular shadows.

Output: /home/yxma/MultimodalData/fota_marker_classification.json
"""

import glob
import io
import json
import os
import time
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from scipy import ndimage

PARQUET_BASE = "/media/yxma/Disk1/yuxiang/mini_data_parquet"
SUBSETS = ["fota_labeled", "fota_unlabeled"]
SAMPLES_PER_CAPTURE = 50
DOT_THRESHOLD_OFFSET = 12
MARKER_AREA_MIN_640 = 30
MARKER_AREA_MAX_640 = 400
MARKER_COUNT_THRESHOLD = 30  # ≥30 dark dots → markered


def classify_mean(mean_img: np.ndarray) -> tuple[int, int]:
    h, w = mean_img.shape
    p5 = np.percentile(mean_img, 5)
    mask = mean_img < (p5 - DOT_THRESHOLD_OFFSET)
    lab, n = ndimage.label(mask)
    if n == 0:
        return 0, 0
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    scale = (h * w) / (480 * 640)
    a_min = int(MARKER_AREA_MIN_640 * scale)
    a_max = int(MARKER_AREA_MAX_640 * scale)
    valid = int(((sizes >= a_min) & (sizes <= a_max)).sum())
    return valid, n


def process_subset(sub: str) -> list[dict]:
    paths = sorted(glob.glob(os.path.join(PARQUET_BASE, sub, "*.parquet")))
    if not paths:
        return []
    print(f"  reading {len(paths)} shards for {sub}", flush=True)

    # Collect: per capture, accumulate sum and count using a streaming approach
    accums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = defaultdict(int)
    target_per_cap: dict[str, int] = {}
    shapes: dict[str, tuple] = {}

    # Two passes:
    # Pass 1: count per capture so we know how to space samples
    print("  pass 1: counting rows per capture", flush=True)
    cap_total = defaultdict(int)
    for p in paths:
        t = pq.read_table(p, columns=["capture"])
        captures = t.column("capture").to_pylist()
        for c in captures:
            cap_total[c] += 1
    for c, n in cap_total.items():
        target_per_cap[c] = min(SAMPLES_PER_CAPTURE, n)

    print(f"  found {len(cap_total)} captures, sampling up to {SAMPLES_PER_CAPTURE} each", flush=True)
    print("  pass 2: accumulating mean images", flush=True)

    # Pass 2: stream rows; for each capture, take every nth row to get N samples
    seen_per_cap = defaultdict(int)
    stride: dict[str, int] = {c: max(1, n // SAMPLES_PER_CAPTURE) for c, n in cap_total.items()}
    visit_count = defaultdict(int)

    for p in paths:
        pf = pq.ParquetFile(p)
        for batch in pf.iter_batches(batch_size=256,
                                     columns=["image", "capture"]):
            n_batch = batch.num_rows
            caps = batch.column("capture").to_pylist()
            imgs = batch.column("image")
            for i in range(n_batch):
                cap = caps[i]
                # Decide if we should sample this row: every stride-th visit
                if (visit_count[cap] % stride[cap]) == 0 and seen_per_cap[cap] < target_per_cap[cap]:
                    img_bytes = imgs[i].as_py()
                    arr = np.asarray(Image.open(io.BytesIO(img_bytes)).convert("L"),
                                     dtype=np.float32)
                    if cap not in accums:
                        accums[cap] = arr.copy()
                        shapes[cap] = arr.shape
                    else:
                        accums[cap] += arr
                    counts[cap] += 1
                    seen_per_cap[cap] += 1
                visit_count[cap] += 1

    # Classify
    results = []
    for cap, accum in accums.items():
        mean = accum / counts[cap]
        valid, all_b = classify_mean(mean)
        results.append({
            "subset": sub,
            "capture": cap,
            "n_samples": counts[cap],
            "img_h": int(shapes[cap][0]),
            "img_w": int(shapes[cap][1]),
            "valid_dots": valid,
            "all_blobs": all_b,
            "markered": bool(valid >= MARKER_COUNT_THRESHOLD),
            "mean_brightness": float(mean.mean()),
            "mean_std": float(mean.std()),
        })
    results.sort(key=lambda r: r["capture"])
    return results


def main():
    all_results = []
    for sub in SUBSETS:
        t0 = time.time()
        print(f"\n=== {sub} ===", flush=True)
        rs = process_subset(sub)
        print(f"  done in {time.time()-t0:.0f}s", flush=True)
        all_results.extend(rs)

    # Summary
    print(f"\n=== Summary ===")
    print(f"Total captures: {len(all_results)}")
    for sub in SUBSETS:
        ms = [r for r in all_results if r["subset"] == sub]
        m = sum(1 for r in ms if r["markered"])
        u = len(ms) - m
        print(f"  {sub}: {m} markered, {u} markerless (of {len(ms)})")

    counts = [r["valid_dots"] for r in all_results]
    if counts:
        print(f"\nvalid_dots distribution:")
        print(f"  min={min(counts)} max={max(counts)} "
              f"mean={np.mean(counts):.1f} median={np.median(counts):.1f}")
        for bound in [0, 5, 10, 20, 30, 50, 100]:
            n = sum(1 for c in counts if c >= bound)
            print(f"  >= {bound:3d} dots: {n}")

    out = "/home/yxma/MultimodalData/fota_marker_classification.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
