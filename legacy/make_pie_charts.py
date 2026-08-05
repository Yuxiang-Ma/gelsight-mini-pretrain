#!/usr/bin/env python3
"""Generate a 4-panel pie-chart summary for the dataset card.

Panels:
  (A) Image resolution — 320×240 vs 640×480
  (B) Gel variant — markered vs markerless
  (C) Source contribution — 8 subsets (legend instead of inline labels
                                       to avoid overlap)
  (D) Gel variant × source — donut: outer ring = gel variant,
                                    inner ring = subset within variant
"""
import glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

BASE = "/media/yxma/Disk1/yuxiang/mini_data_parquet"
OUT  = f"{BASE}/assets"
SUBSETS = ["fota_unlabeled", "sim_tactile_mnist", "sim_starstruck",
           "gelslam", "feelanyforce", "threedcal", "fota_labeled",
           "real_tactile_mnist", "feats", "tactile_tracking", "tacquad_mini"]
# Reordered descending by typical size so colors line up sensibly.

def aggregate():
    agg = {}
    for sub in SUBSETS:
        paths = sorted(glob.glob(f"{BASE}/{sub}/*.parquet"))
        m = u = 0
        res = {}
        for p in paths:
            t = pq.read_table(p, columns=["markered", "width", "height"])
            for mki, wi, hi in zip(t.column("markered").to_pylist(),
                                   t.column("width").to_pylist(),
                                   t.column("height").to_pylist()):
                if mki: m += 1
                else: u += 1
                res[(wi, hi)] = res.get((wi, hi), 0) + 1
        agg[sub] = {"markered": m, "markerless": u, "res": res, "total": m + u}
    return agg


def main():
    agg = aggregate()
    fig, axes = plt.subplots(2, 2, figsize=(13, 12))

    # (A) by resolution
    tot_res = {}
    for sub in SUBSETS:
        for k, v in agg[sub]["res"].items():
            tot_res[k] = tot_res.get(k, 0) + v
    ax = axes[0, 0]
    items = sorted(tot_res.items(), key=lambda kv: -kv[1])
    labels = [f"{w}×{h}\n{n:,}  ({100*n/sum(tot_res.values()):.1f}%)"
              for (w, h), n in items]
    vals = [n for (_, _), n in items]
    ax.pie(vals, labels=labels, colors=["#4c95d6", "#d6794c"],
           startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2},
           textprops={"fontsize": 12})
    ax.set_title("Image resolution\n(GelSight Mini native modes)",
                 fontsize=14, pad=14)

    # (B) by gel variant
    m = sum(agg[s]["markered"] for s in SUBSETS)
    u = sum(agg[s]["markerless"] for s in SUBSETS)
    ax = axes[0, 1]
    ax.pie([u, m],
           labels=[f"markerless\n{u:,}  ({100*u/(m+u):.1f}%)",
                   f"markered\n{m:,}  ({100*m/(m+u):.1f}%)"],
           colors=["#4c95d6", "#d6794c"], startangle=90,
           wedgeprops={"edgecolor": "white", "linewidth": 2},
           textprops={"fontsize": 12})
    ax.set_title("Gel variant", fontsize=14, pad=14)

    # (C) by source — legend instead of inline labels
    ax = axes[1, 0]
    src_vals = [agg[s]["total"] for s in SUBSETS]
    src_total = sum(src_vals)
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(SUBSETS), 10)))
    wedges, _ = ax.pie(src_vals, colors=cmap, startangle=90,
                       wedgeprops={"edgecolor": "white", "linewidth": 2})
    legend_labels = [f"{s:<19s} {n:>7,d}  ({100*n/src_total:5.2f}%)"
                     for s, n in zip(SUBSETS, src_vals)]
    ax.legend(wedges, legend_labels, loc="center left",
              bbox_to_anchor=(1.0, 0.5), fontsize=10,
              prop={"family": "monospace", "size": 10},
              frameon=False, title="subset (frames)",
              title_fontsize=11)
    ax.set_title("Source contribution", fontsize=14, pad=14)

    # (D) gel variant × source donut
    ax = axes[1, 1]
    outer_sizes = [u, m]
    outer_colors = ["#4c95d6", "#d6794c"]
    ax.pie(outer_sizes,
           labels=[f"markerless\n{u:,}", f"markered\n{m:,}"],
           colors=outer_colors, radius=1.0,
           wedgeprops={"edgecolor": "white", "linewidth": 2, "width": 0.35},
           startangle=90, textprops={"fontsize": 12, "fontweight": "bold"},
           labeldistance=1.1)
    inner_sizes, inner_colors, inner_lbls = [], [], []
    blue_pal = plt.cm.Blues(np.linspace(0.4, 0.9, len(SUBSETS)))
    orng_pal = plt.cm.Oranges(np.linspace(0.4, 0.9, len(SUBSETS)))
    for i, s in enumerate(SUBSETS):
        if agg[s]["markerless"] > 0:
            inner_sizes.append(agg[s]["markerless"])
            inner_colors.append(blue_pal[i])
            inner_lbls.append(s if agg[s]["markerless"] > u * 0.06 else "")
    for i, s in enumerate(SUBSETS):
        if agg[s]["markered"] > 0:
            inner_sizes.append(agg[s]["markered"])
            inner_colors.append(orng_pal[i])
            inner_lbls.append(s if agg[s]["markered"] > m * 0.06 else "")
    ax.pie(inner_sizes, radius=0.62, colors=inner_colors,
           wedgeprops={"edgecolor": "white", "linewidth": 1, "width": 0.30},
           startangle=90, labels=inner_lbls, labeldistance=0.78,
           textprops={"fontsize": 8})
    ax.set_title("Gel variant × source (donut)", fontsize=14, pad=14)

    fig.suptitle(f"gelsight-mini-pretrain · summary pie charts  "
                 f"(total {m+u:,} frames)", fontsize=16, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = f"{OUT}/summary_pies.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved {out}  total={m+u:,}  markered={m:,}  markerless={u:,}")


if __name__ == "__main__":
    main()
