#!/usr/bin/env python3
"""Generate analytical plots for the dataset card:
  - composition.png             frames per subset, stacked by markered/markerless
  - resolution_distribution.png 320x240 vs 640x480 per subset
  - force_distribution.png      FEATS f_z histogram + indenter shapes
  - threedcal_coverage.png      (x,y) probe-position heatmap
  - rtm_digit_distribution.png  digit-class counts
"""

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

BASE = "/media/yxma/Disk1/yuxiang/mini_data_parquet"
OUT  = f"{BASE}/assets"

SUBSETS = ["fota_labeled", "fota_unlabeled", "threedcal", "feats",
           "feelanyforce", "gelslam", "tactile_tracking", "real_tactile_mnist",
           "sim_tactile_mnist", "sim_starstruck", "tacquad_mini"]


def scan_subset(sub, columns):
    paths = sorted(glob.glob(f"{BASE}/{sub}/*.parquet"))
    out = {c: [] for c in columns}
    for p in paths:
        t = pq.read_table(p, columns=columns)
        for c in columns:
            out[c].extend(t.column(c).to_pylist())
    return out


# 1. composition stacked bar
def plot_composition():
    counts = {}
    for sub in SUBSETS:
        d = scan_subset(sub, ["markered"])
        n_m = sum(1 for x in d["markered"] if x)
        n_u = sum(1 for x in d["markered"] if not x)
        counts[sub] = (n_m, n_u)

    labels = SUBSETS
    m = np.array([counts[s][0] for s in labels])
    u = np.array([counts[s][1] for s in labels])
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(labels))
    ax.bar(x, u, label="markerless", color="#4c95d6")
    ax.bar(x, m, bottom=u, label="markered", color="#d6794c")
    for i, s in enumerate(labels):
        total = counts[s][0] + counts[s][1]
        ax.text(i, total + max(m+u)*0.005, f"{total:,}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Frames")
    ax.set_title("gelsight-mini-pretrain · frames per subset (stacked by gel variant)")
    ax.legend(loc="upper right", frameon=False)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{OUT}/composition.png", dpi=140)
    plt.close()
    print("wrote composition.png")


# 2. resolution distribution
def plot_resolution():
    cols = {}
    for sub in SUBSETS:
        d = scan_subset(sub, ["height", "width"])
        from collections import Counter
        c = Counter(zip(d["width"], d["height"]))
        cols[sub] = c

    all_dims = sorted({k for sub in cols.values() for k in sub.keys()})
    colors = {(640,480): "#4c95d6", (320,240): "#d6794c", (160,120): "#9e8fcc"}
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(SUBSETS))
    bottom = np.zeros(len(SUBSETS))
    totals = np.array([sum(cols[s].values()) for s in SUBSETS])
    for d in all_dims:
        vals = np.array([cols[s].get(d, 0) for s in SUBSETS])
        ax.bar(x, vals, bottom=bottom,
               color=colors.get(d, "#999999"),
               label=f"{d[0]}×{d[1]}  ({vals.sum():,})")
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(SUBSETS, rotation=20, ha="right")
    ax.set_ylabel("Frames (log scale)")
    ax.set_yscale("log")
    ax.set_ylim(1, max(totals) * 2)
    # Annotate each bar with its total count
    for i, t in enumerate(totals):
        if t > 0:
            ax.text(i, t * 1.15, f"{t:,}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Image resolution per subset (GelSight Mini native modes; log y-axis)")
    ax.legend(loc="lower center", frameon=False, ncol=3)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{OUT}/resolution_distribution.png", dpi=140)
    plt.close()
    print("wrote resolution_distribution.png")


# 3. FEATS force distribution + indenter mix
def plot_feats_force():
    d = scan_subset("feats", ["f_z", "f_x", "f_y", "indenter"])
    fz = np.array([x for x in d["f_z"] if x is not None])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(fz, bins=60, color="#4c95d6", edgecolor="white")
    axes[0].axvline(0, color="#444", linestyle="--", linewidth=1)
    axes[0].set_xlabel("normal force  f_z  (N)")
    axes[0].set_ylabel("Frames")
    axes[0].set_title(f"FEATS normal force distribution (n={len(fz):,})")
    axes[0].spines[["top","right"]].set_visible(False)
    # indenter mix
    from collections import Counter
    c = Counter(x or "unknown" for x in d["indenter"])
    items = sorted(c.items(), key=lambda kv: -kv[1])
    keys = [k for k,_ in items]; vals = [v for _,v in items]
    axes[1].barh(range(len(keys)), vals, color="#d6794c", edgecolor="white")
    axes[1].set_yticks(range(len(keys)))
    axes[1].set_yticklabels(keys)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Frames")
    axes[1].set_title("FEATS indenter-shape mix")
    axes[1].spines[["top","right"]].set_visible(False)
    for i, v in enumerate(vals):
        axes[1].text(v + max(vals)*0.005, i, f"{v:,}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUT}/force_distribution.png", dpi=140)
    plt.close()
    print("wrote force_distribution.png")


# 4. 3DCal probe-position heatmap
def plot_threedcal_coverage():
    d = scan_subset("threedcal", ["x_mm", "y_mm"])
    x = np.array([v for v in d["x_mm"] if v is not None])
    y = np.array([v for v in d["y_mm"] if v is not None])
    # Grid is at 0.5-mm spacing — bin edges must straddle each grid point
    x_edges = np.arange(-0.25, x.max() + 0.5, 0.5)
    y_edges = np.arange(-0.25, y.max() + 0.5, 0.5)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    H, xe, ye = np.histogram2d(x, y, bins=[x_edges, y_edges])
    im = ax.pcolormesh(xe, ye, H.T, cmap="magma")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(f"py3DCal calibration grid — probe coverage "
                 f"(n={len(x):,}; 0.5-mm grid spacing)")
    plt.colorbar(im, ax=ax, label="frames per (x,y) cell")
    plt.tight_layout()
    plt.savefig(f"{OUT}/threedcal_coverage.png", dpi=140)
    plt.close()
    print("wrote threedcal_coverage.png")


# 5. RTM digit-class distribution
def plot_rtm_digits():
    d = scan_subset("real_tactile_mnist", ["digit_class"])
    from collections import Counter
    c = Counter(x for x in d["digit_class"] if x is not None)
    keys = list(range(10))
    vals = [c.get(k, 0) for k in keys]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(keys, vals, color="#4c95d6", edgecolor="white")
    for k, v in zip(keys, vals):
        ax.text(k, v + max(vals)*0.005, f"{v:,}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(keys)
    ax.set_xlabel("digit class")
    ax.set_ylabel("frames")
    ax.set_title(f"Real Tactile MNIST · digit-class balance (total {sum(vals):,})")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{OUT}/rtm_digit_distribution.png", dpi=140)
    plt.close()
    print("wrote rtm_digit_distribution.png")


if __name__ == "__main__":
    plot_composition()
    plot_resolution()
    plot_feats_force()
    plot_threedcal_coverage()
    plot_rtm_digits()
    print("done")
