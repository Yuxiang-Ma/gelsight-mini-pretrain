#!/usr/bin/env python3
"""Compute dataset statistics + generate per-subset sample image grids.

Outputs into /media/yxma/Disk1/yuxiang/mini_data_parquet/assets/:
  - samples_fota_labeled.png
  - samples_fota_unlabeled.png
  - samples_threedcal.png
  - samples_feats.png
  - combined_overview.png  (one figure showing all four)
  - stats.md  (markdown table with per-subset stats)
  - stats.json (machine-readable stats)
"""

import glob
import io
import json
import os
import random
from collections import Counter, defaultdict

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

BASE = "/media/yxma/Disk1/yuxiang/mini_data_parquet"
OUT = os.path.join(BASE, "assets")
os.makedirs(OUT, exist_ok=True)

SUBSETS = {
    "fota_labeled":   "fota_labeled",
    "fota_unlabeled": "fota_unlabeled",
    "threedcal":      "threedcal",
    "feats":          "feats",
}

THUMB = 192  # thumbnail side length for grids

# ---------------- helpers ----------------
def list_parquets(sub: str, split: str | None = None) -> list[str]:
    pat = os.path.join(BASE, sub,
                       f"{split or '*'}-*.parquet")
    return sorted(glob.glob(pat))


def iter_rows(paths, columns=None, sample_n=None, seed=0):
    """Iterate selected columns from parquet shards.
    If sample_n is given, return a uniform random sample across all rows."""
    if sample_n is None:
        for p in paths:
            t = pq.read_table(p, columns=columns)
            for i in range(t.num_rows):
                yield {c: t.column(c)[i].as_py() for c in (columns or t.column_names)}
        return
    # uniform sampling: get row counts per shard, sample indices, then read
    rng = random.Random(seed)
    counts = []
    for p in paths:
        counts.append(pq.read_metadata(p).num_rows)
    total = sum(counts)
    n = min(sample_n, total)
    idxs = sorted(rng.sample(range(total), n))
    # walk
    cum = 0
    idx_iter = iter(idxs)
    nxt = next(idx_iter, None)
    for p, c in zip(paths, counts):
        if nxt is None:
            return
        if nxt >= cum + c:
            cum += c
            continue
        local_idxs = []
        while nxt is not None and nxt < cum + c:
            local_idxs.append(nxt - cum)
            nxt = next(idx_iter, None)
        if local_idxs:
            t = pq.read_table(p, columns=columns)
            for li in local_idxs:
                yield {col: t.column(col)[li].as_py()
                       for col in (columns or t.column_names)}
        cum += c


def thumbnail(img_bytes: bytes, side: int = THUMB) -> Image.Image:
    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h = im.size
    # center-crop to square then resize
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    im = im.resize((side, side), Image.LANCZOS)
    return im


def make_grid(images: list[Image.Image], labels: list[str], cols: int,
              title: str, side: int = THUMB,
              pad: int = 8, label_h: int = 22, title_h: int = 40) -> Image.Image:
    rows = (len(images) + cols - 1) // cols
    W = pad + cols * (side + pad)
    H = title_h + rows * (side + label_h + pad) + pad
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    try:
        f_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
        f_lbl = ImageFont.truetype("DejaVuSans.ttf", 12)
    except Exception:
        f_title = ImageFont.load_default()
        f_lbl = ImageFont.load_default()
    d.text((pad, 8), title, fill=(0, 0, 0), font=f_title)
    for i, (im, lab) in enumerate(zip(images, labels)):
        r = i // cols
        c = i % cols
        x = pad + c * (side + pad)
        y = title_h + r * (side + label_h + pad)
        canvas.paste(im, (x, y))
        # truncate label
        if len(lab) > 28:
            lab = lab[:25] + "..."
        d.text((x, y + side + 4), lab, fill=(60, 60, 60), font=f_lbl)
    return canvas


# ---------------- per-subset routines ----------------

def stats_fota_labeled():
    paths = list_parquets("fota_labeled")
    total = sum(pq.read_metadata(p).num_rows for p in paths)
    # iterate metadata once
    obj_counter: Counter = Counter()
    pose_counter: Counter = Counter()
    side_counter: Counter = Counter()
    split_counter: Counter = Counter()
    xs, ys, zs = [], [], []
    for r in iter_rows(paths,
                       columns=["obj_name", "init_pose", "side",
                                "split", "x_mm", "y_mm", "z_mm"]):
        obj_counter[r["obj_name"]] += 1
        pose_counter[r["init_pose"]] += 1
        side_counter[r["side"]] += 1
        split_counter[r["split"]] += 1
        if r["x_mm"] is not None:
            xs.append(r["x_mm"])
            ys.append(r["y_mm"])
            zs.append(r["z_mm"])
    return {
        "total": total,
        "splits": dict(split_counter),
        "n_objects": len(obj_counter),
        "top_objects": obj_counter.most_common(10),
        "n_init_poses": len(pose_counter),
        "side_counts": dict(side_counter),
        "x_mm_range": [float(min(xs)), float(max(xs))] if xs else None,
        "y_mm_range": [float(min(ys)), float(max(ys))] if ys else None,
        "z_mm_range": [float(min(zs)), float(max(zs))] if zs else None,
    }


def stats_fota_unlabeled():
    paths = list_parquets("fota_unlabeled")
    total = sum(pq.read_metadata(p).num_rows for p in paths)
    obj_counter: Counter = Counter()
    side_counter: Counter = Counter()
    split_counter: Counter = Counter()
    for r in iter_rows(paths,
                       columns=["obj_name", "side", "split"]):
        obj_counter[r["obj_name"]] += 1
        side_counter[r["side"]] += 1
        split_counter[r["split"]] += 1
    return {
        "total": total,
        "splits": dict(split_counter),
        "n_objects": len(obj_counter),
        "top_objects": obj_counter.most_common(10),
        "side_counts": dict(side_counter),
    }


def stats_threedcal():
    paths = list_parquets("threedcal")
    total = sum(pq.read_metadata(p).num_rows for p in paths)
    xs, ys, zs = [], [], []
    for r in iter_rows(paths, columns=["x_mm", "y_mm", "z_mm"]):
        if r["x_mm"] is not None:
            xs.append(r["x_mm"])
            ys.append(r["y_mm"])
            zs.append(r["z_mm"])
    z_counter = Counter(round(z, 2) for z in zs)
    return {
        "total": total,
        "x_mm_range": [float(min(xs)), float(max(xs))] if xs else None,
        "y_mm_range": [float(min(ys)), float(max(ys))] if ys else None,
        "depth_mm_range": [float(min(zs)), float(max(zs))] if zs else None,
        "n_unique_xy_grid": len({(round(x,1), round(y,1)) for x,y in zip(xs, ys)}),
        "depth_distribution": dict(sorted(z_counter.items())),
    }


def stats_feats():
    paths = list_parquets("feats")
    total = sum(pq.read_metadata(p).num_rows for p in paths)
    indent_counter: Counter = Counter()
    indent_param: defaultdict = defaultdict(Counter)
    split_counter: Counter = Counter()
    fx, fy, fz = [], [], []
    for r in iter_rows(paths,
                       columns=["indenter", "indenter_param", "split",
                                "f_x", "f_y", "f_z"]):
        indent_counter[r["indenter"]] += 1
        indent_param[r["indenter"]][r["indenter_param"]] += 1
        split_counter[r["split"]] += 1
        if r["f_x"] is not None:
            fx.append(r["f_x"]); fy.append(r["f_y"]); fz.append(r["f_z"])
    return {
        "total": total,
        "splits": dict(split_counter),
        "indenters": dict(indent_counter),
        "indenter_params": {k: dict(v) for k, v in indent_param.items()},
        "f_x_range": [float(min(fx)), float(max(fx))] if fx else None,
        "f_y_range": [float(min(fy)), float(max(fy))] if fy else None,
        "f_z_range": [float(min(fz)), float(max(fz))] if fz else None,
        "f_z_mean": float(np.mean(fz)) if fz else None,
        "f_z_std": float(np.std(fz)) if fz else None,
    }


# ---------------- sample grids ----------------

def samples_fota_labeled():
    paths = list_parquets("fota_labeled")
    # one sample per object
    seen = {}
    for r in iter_rows(paths, columns=["image", "obj_name", "side"]):
        key = r["obj_name"]
        if key in seen:
            continue
        seen[key] = (r["image"], f"{r['obj_name']}/{r['side']}")
        if len(seen) >= 16:
            break
    images = [thumbnail(b) for b, _ in seen.values()]
    labels = [lab for _, lab in seen.values()]
    return make_grid(images, labels, cols=4,
                     title=f"fota_labeled — {len(seen)} distinct objects shown")


def samples_fota_unlabeled():
    paths = list_parquets("fota_unlabeled")
    # one sample per object (using object info)
    seen = {}
    for r in iter_rows(paths, columns=["image", "obj_name", "side"]):
        key = r["obj_name"]
        if key in seen:
            continue
        seen[key] = (r["image"], f"{r['obj_name']}/{r['side']}")
        if len(seen) >= 16:
            break
    images = [thumbnail(b) for b, _ in seen.values()]
    labels = [lab for _, lab in seen.values()]
    return make_grid(images, labels, cols=4,
                     title=f"fota_unlabeled — {len(seen)} distinct objects shown")


def samples_threedcal():
    paths = list_parquets("threedcal")
    # uniform random
    rows = list(iter_rows(paths, columns=["image", "x_mm", "y_mm", "z_mm"],
                          sample_n=16, seed=42))
    images = [thumbnail(r["image"]) for r in rows]
    labels = [f"x={r['x_mm']:.1f} y={r['y_mm']:.1f} z={r['z_mm']:.1f}"
              if r["x_mm"] is not None else "blank"
              for r in rows]
    return make_grid(images, labels, cols=4,
                     title="threedcal — random sphere indentations at varying (x,y,z)")


def samples_feats():
    paths = list_parquets("feats")
    # one per (indenter, param) pair for diversity
    by_key: dict = {}
    for r in iter_rows(paths,
                       columns=["image", "indenter", "indenter_param", "f_z"]):
        key = (r["indenter"], r["indenter_param"])
        if key in by_key:
            continue
        by_key[key] = (r["image"],
                       f"{r['indenter']}-{r['indenter_param']}  fz={r['f_z']:.2f}"
                       if r["f_z"] is not None else f"{r['indenter']}-{r['indenter_param']}")
        if len(by_key) >= 16:
            break
    images = [thumbnail(b) for b, _ in by_key.values()]
    labels = [lab for _, lab in by_key.values()]
    return make_grid(images, labels, cols=4,
                     title=f"feats (markered gel) — {len(by_key)} indenter shape/size combos")


def combined_overview():
    """4-panel poster: one row per subset, 4 example images each."""
    grids = []
    for name, fn in [
        ("fota_labeled", samples_fota_labeled),
        ("fota_unlabeled", samples_fota_unlabeled),
        ("threedcal", samples_threedcal),
        ("feats", samples_feats),
    ]:
        # Reuse fn? Too slow. We just build a small 4-image strip.
        pass
    # Cheaper: load 4 random images per subset
    pad = 8
    side = 160
    title_h = 36
    row_label_w = 130
    cols = 8
    rows = 4
    W = row_label_w + pad + cols * (side + pad)
    H = title_h + rows * (side + pad) + pad
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    try:
        f_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
        f_row = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
    except Exception:
        f_title = ImageFont.load_default()
        f_row = ImageFont.load_default()
    d.text((pad, 8), "gelsight-mini-pretrain · subset overview", fill=(0,0,0), font=f_title)
    for ri, sub in enumerate(["fota_labeled", "fota_unlabeled", "threedcal", "feats"]):
        paths = list_parquets(sub)
        rows_iter = list(iter_rows(paths, columns=["image"], sample_n=cols, seed=7+ri))
        y = title_h + ri * (side + pad)
        d.text((pad, y + side // 2 - 7), sub, fill=(0,0,0), font=f_row)
        for ci, r in enumerate(rows_iter):
            im = thumbnail(r["image"], side=side)
            x = row_label_w + pad + ci * (side + pad)
            canvas.paste(im, (x, y))
    return canvas


# ---------------- main ----------------

def main():
    print("Computing stats...", flush=True)
    stats = {
        "fota_labeled":   stats_fota_labeled(),
        "fota_unlabeled": stats_fota_unlabeled(),
        "threedcal":      stats_threedcal(),
        "feats":          stats_feats(),
    }
    with open(os.path.join(OUT, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(json.dumps({k: {k2: v2 for k2, v2 in v.items() if not isinstance(v2, dict)}
                      for k, v in stats.items()}, indent=2, default=str))

    print("Generating sample grids...", flush=True)
    for name, fn in [
        ("samples_fota_labeled.png", samples_fota_labeled),
        ("samples_fota_unlabeled.png", samples_fota_unlabeled),
        ("samples_threedcal.png", samples_threedcal),
        ("samples_feats.png", samples_feats),
    ]:
        img = fn()
        out = os.path.join(OUT, name)
        img.save(out, optimize=True)
        print(f"  wrote {out}  ({img.size})", flush=True)

    print("Generating combined overview...", flush=True)
    overview = combined_overview()
    overview.save(os.path.join(OUT, "combined_overview.png"), optimize=True)
    print(f"  wrote combined_overview.png  ({overview.size})", flush=True)


if __name__ == "__main__":
    main()
