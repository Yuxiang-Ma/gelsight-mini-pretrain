#!/usr/bin/env python3
"""Diagnose RGB vs BGR channel order across all subsets.

Logic: GelSight Mini has 3 colored LEDs in a known geometric configuration.
The gel's "at-rest" illumination is roughly the same across all sensors,
so the mean (R, G, B) channel values should be roughly consistent across
all subsets — IF the channel order is the same.

If a subset has (R, G, B) means that look like (B_ref, G_ref, R_ref) of
the others, the R and B channels are swapped — i.e. that subset was
stored in BGR.

This script:
  1. Samples N frames per subset
  2. Computes channel means + stds
  3. Reports a table sorted so similar profiles cluster together
  4. Flags suspected channel-swapped subsets
"""
import argparse
import glob
import io
import os
import random

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

MAIN_BASE = "/media/yxma/Disk1/yuxiang/mini_data_parquet"
NC_BASE = "/media/yxma/Disk1/yuxiang/mini_data_parquet_nc"

N_SAMPLES = 200


def sample_means(parquet_paths, n=N_SAMPLES, seed=42):
    """Return mean (R, G, B) across N random images from a subset."""
    rng = random.Random(seed)
    counts = [pq.read_metadata(p).num_rows for p in parquet_paths]
    total = sum(counts)
    if total == 0: return None
    idxs = sorted(rng.sample(range(total), min(n, total)))
    rs, gs, bs = [], [], []
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
                try:
                    rgb = np.array(Image.open(
                        io.BytesIO(t.column("image")[li].as_py())).convert("RGB"))
                    rs.append(rgb[..., 0].mean())
                    gs.append(rgb[..., 1].mean())
                    bs.append(rgb[..., 2].mean())
                except Exception:
                    pass
        cum += c
    if not rs: return None
    return dict(R=np.mean(rs), G=np.mean(gs), B=np.mean(bs),
                R_std=np.std(rs), G_std=np.std(gs), B_std=np.std(bs),
                n=len(rs))


def main():
    subs = [
        ("main", "fota_labeled"), ("main", "fota_unlabeled"),
        ("main", "threedcal"), ("main", "feats"),
        ("main", "gelslam"), ("main", "tactile_tracking"),
        ("main", "real_tactile_mnist"), ("main", "feelanyforce"),
        ("main", "unit"), ("main", "tacquad"),
        ("main", "sim_tactile_mnist"), ("main", "sim_starstruck"),
        ("nc", "sparsh"), ("nc", "faf_force_estimation"),
    ]

    results = {}
    print(f"{'subset':30s} {'N':>4s}    {'R':>6s}     {'G':>6s}     {'B':>6s}    R-B")
    print("-" * 80)
    for repo, sub in subs:
        base = MAIN_BASE if repo == "main" else NC_BASE
        paths = sorted(glob.glob(f"{base}/{sub}/*.parquet"))
        if not paths: continue
        stats = sample_means(paths)
        if not stats: continue
        results[(repo, sub)] = stats
        rb_gap = stats["R"] - stats["B"]
        rb_sign = "+" if rb_gap > 0 else " "
        print(f"{repo+'/'+sub:30s} {stats['n']:>4d}   "
              f"{stats['R']:6.1f}    {stats['G']:6.1f}    {stats['B']:6.1f}   "
              f"{rb_sign}{rb_gap:+5.1f}")

    # Diagnose
    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    print("\nInterpretation of R-B column:")
    print("  R-B > 0  →  red channel value is higher than blue channel value")
    print("              (the channel labeled 'R' carries more LED energy)")
    print("  R-B < 0  →  blue channel value is higher than red")
    print()
    print("If GelSight Mini's reference RGB has R-B ≈ +ε (consistent sign across")
    print("most subsets), then any subset with R-B of opposite sign is likely")
    print("stored in BGR (R and B channels swapped). Subsets clustered on the")
    print("majority sign side are likely consistent.")
    print()

    # Cluster by sign of R-B
    positive = [k for k, v in results.items() if v["R"] - v["B"] > 0]
    negative = [k for k, v in results.items() if v["R"] - v["B"] <= 0]
    print(f"\n  Subsets with R > B ({len(positive)}):")
    for k in positive:
        v = results[k]
        print(f"    {k[0]}/{k[1]:30s}  R={v['R']:.1f}  G={v['G']:.1f}  "
              f"B={v['B']:.1f}  R-B={v['R']-v['B']:+.1f}")
    print(f"\n  Subsets with R ≤ B ({len(negative)}):")
    for k in negative:
        v = results[k]
        print(f"    {k[0]}/{k[1]:30s}  R={v['R']:.1f}  G={v['G']:.1f}  "
              f"B={v['B']:.1f}  R-B={v['R']-v['B']:+.1f}")

    # Save JSON
    import json
    out = {f"{r}/{s}": v for (r, s), v in results.items()}
    out_p = f"{MAIN_BASE}/assets/channel_order_diagnosis.json"
    with open(out_p, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nsaved {out_p}")


if __name__ == "__main__":
    main()
