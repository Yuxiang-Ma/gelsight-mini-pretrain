#!/usr/bin/env python3
"""Generate a per-channel pixel-value distribution plot, comparing frames
that PASS the area+intensity filter ("with contact") vs frames that FAIL
("without contact"), for the 3 most recently reprocessed subsets.

Mirrors the structure of the TWM mode1_v1 plot the user shared:
    rows = subsets ; cols = R, G, B channels
    red   = with contact   (pass filter)
    blue  = without contact (fail filter)

Since the local parquet has already been filtered (= pass), we pull a
sample of the *unfiltered* version straight from HuggingFace (the remote
parquet still holds the full pre-filter population) and re-apply the
same baseline + threshold logic to split kept-vs-rejected.

Output:  assets/pixel_value_distribution.png
"""
import io
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

BASE = "/media/yxma/Disk1/yuxiang/mini_data_parquet"
REPO = "yxma/gelsight-mini-pretrain"
OUT = f"{BASE}/assets/pixel_value_distribution.png"

PIXEL_THRESH = 10
A_MIN = 40
N_BASELINE = 80
N_SAMPLE = 2500          # frames to scan per subset
N_BINS = 64              # histogram bins over [0, 256)


def grey_center(rgb):
    g = rgb.mean(axis=2).astype(np.float32)
    h, w = g.shape
    return g[h // 4:3 * h // 4, w // 4:3 * w // 4]


def fetch_unfiltered_table(sub, split_glob):
    """Download one parquet shard of the *unfiltered* subset from HF.

    For our 3 subsets, each split is a single ~600 MB file; we use the
    largest (train) for stratified random sampling.
    """
    fname = f"{sub}/{split_glob}-00000-of-00001.parquet"
    print(f"  [{sub}] downloading {fname} from HF ...", flush=True)
    t0 = time.time()
    path = hf_hub_download(repo_id=REPO, repo_type="dataset", filename=fname)
    print(f"  [{sub}] downloaded in {time.time()-t0:.0f}s -> {path}", flush=True)
    return pq.read_table(path)


def compute_subset(sub, i_min, seed):
    """For one subset, sample N_SAMPLE rows from the unfiltered HF parquet,
    compute area+intensity per frame, split kept-vs-rejected, accumulate
    per-channel histograms."""
    rng = random.Random(seed)
    # Download train shard (largest); for our 3 subsets each train shard
    # has 100K+ rows so sampling 2500 is easy.
    table = fetch_unfiltered_table(sub, "train")
    n_total = table.num_rows
    img_col = table.column("image")

    # Pick N_SAMPLE random indices
    idxs = rng.sample(range(n_total), min(N_SAMPLE, n_total))

    # Step 1: baseline from first N_BASELINE indices (random sample)
    grays = []
    for i in idxs[:N_BASELINE]:
        try:
            rgb = np.array(Image.open(io.BytesIO(img_col[i].as_py())).convert("RGB"))
            grays.append(grey_center(rgb))
        except Exception:
            pass
    if len(grays) < 10:
        return sub, None
    baseline = np.median(np.stack(grays), axis=0)

    # Step 2: per-frame area+intensity + per-channel histograms
    edges = np.arange(N_BINS + 1) * (256 / N_BINS)
    h_pass = np.zeros((3, N_BINS), dtype=np.float64)
    h_fail = np.zeros((3, N_BINS), dtype=np.float64)
    n_pass = n_fail = 0
    t0 = time.time()
    for k, i in enumerate(idxs):
        try:
            rgb = np.array(Image.open(io.BytesIO(img_col[i].as_py())).convert("RGB"))
        except Exception:
            continue
        g = grey_center(rgb)
        diff = np.abs(g - baseline)
        mask = diff > PIXEL_THRESH
        area = int(mask.sum())
        inten = float(diff[mask].mean()) if area > 0 else 0.0
        passes = (area >= A_MIN) and (inten >= i_min)

        # Histogram each channel of the full RGB frame (not just centre crop;
        # the user's reference plot histograms the whole frame).
        for c in range(3):
            h, _ = np.histogram(rgb[..., c], bins=edges)
            if passes:
                h_pass[c] += h
            else:
                h_fail[c] += h
        if passes:
            n_pass += 1
        else:
            n_fail += 1
        if (k + 1) % 500 == 0:
            print(f"  [{sub}] {k+1}/{len(idxs)}  ({(k+1)/(time.time()-t0):.0f} fps)",
                  flush=True)

    print(f"  [{sub}] done  pass={n_pass:,}  fail={n_fail:,}  "
          f"in {time.time()-t0:.0f}s", flush=True)
    return sub, dict(h_pass=h_pass, h_fail=h_fail,
                     n_pass=n_pass, n_fail=n_fail, edges=edges, i_min=i_min)


def main():
    subsets = [
        ("real_tactile_mnist", 12),
        ("sim_tactile_mnist", 10),
        ("sim_starstruck", 10),
    ]
    rng_seed = 42

    results = {}
    with ProcessPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(compute_subset, sub, i, rng_seed + k): sub
                for k, (sub, i) in enumerate(subsets)}
        for fut in as_completed(futs):
            sub, data = fut.result()
            results[sub] = data

    # ----- plot -----
    n_rows = len(subsets)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 3.4 * n_rows), squeeze=False)
    chan_names = ["R channel", "G channel", "B channel"]
    for r, (sub, i_min) in enumerate(subsets):
        d = results.get(sub)
        for c in range(3):
            ax = axes[r, c]
            if d is None:
                ax.set_title(f"{sub} — {chan_names[c]}  (no data)")
                continue
            # Per-frame normalized: divide each histogram by (n_frames * pixels_per_frame)
            # so the y-axis is "average frequency contributed per frame".
            # Each frame contributes (640*480) pixels = 307200 to its channel sum.
            pixels_per_frame = 640 * 480
            y_pass = d["h_pass"][c] / max(1, d["n_pass"]) / pixels_per_frame
            y_fail = d["h_fail"][c] / max(1, d["n_fail"]) / pixels_per_frame
            x = (d["edges"][:-1] + d["edges"][1:]) / 2.0
            if d["n_pass"] > 0:
                ax.plot(x, y_pass, color="tab:red",
                        label=f"with contact (n={d['n_pass']:,})", lw=1.6)
            if d["n_fail"] > 0:
                ax.plot(x, y_fail, color="tab:blue",
                        label=f"without contact (n={d['n_fail']:,})", lw=1.6)
            ax.set_title(f"{sub} — {chan_names[c]}",
                         fontsize=11, fontweight="bold")
            if r == n_rows - 1:
                ax.set_xlabel("pixel value (uint8)")
            if c == 0:
                ax.set_ylabel("frequency (per-frame normalized)")
            ax.legend(loc="upper right", fontsize=8)
            ax.set_xlim(0, 256)
            ax.grid(alpha=0.25)

    title = ("GelSight-Mini pixel-value distribution — contact metric = "
             "area≥40 ∧ intensity≥I_min  (red = passes filter, blue = fails)")
    fig.suptitle(title, fontsize=13, y=1.001)
    fig.tight_layout()
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
