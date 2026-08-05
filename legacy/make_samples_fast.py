#!/usr/bin/env python3
"""Fast sample-grid generation using batched parquet iteration.
Reads only enough to get diverse samples, not whole tables."""

import glob
import io
import os

import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

BASE = "/media/yxma/Disk1/yuxiang/mini_data_parquet"
OUT = os.path.join(BASE, "assets")
os.makedirs(OUT, exist_ok=True)

THUMB = 192


def list_parquets(sub, split=None):
    pat = os.path.join(BASE, sub, f"{split or '*'}-*.parquet")
    return sorted(glob.glob(pat))


def thumbnail(img_bytes, side=THUMB):
    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    return im.resize((side, side), Image.LANCZOS)


def make_grid(images, labels, cols, title, side=THUMB, pad=8,
              label_h=22, title_h=40):
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
        r, c = i // cols, i % cols
        x = pad + c * (side + pad)
        y = title_h + r * (side + label_h + pad)
        canvas.paste(im, (x, y))
        if len(lab) > 28:
            lab = lab[:25] + "..."
        d.text((x, y + side + 4), lab, fill=(60, 60, 60), font=f_lbl)
    return canvas


def collect_diverse(sub, key_cols, label_fn, max_n=16):
    """Stream parquet batches, keep one row per unique key_cols tuple."""
    seen = {}
    for p in list_parquets(sub):
        pf = pq.ParquetFile(p)
        for batch in pf.iter_batches(batch_size=256,
                                     columns=["image"] + list(key_cols)):
            cols = {c: batch.column(c).to_pylist() for c in batch.column_names}
            n = len(cols["image"])
            for i in range(n):
                key = tuple(cols[c][i] for c in key_cols)
                if key in seen:
                    continue
                row = {c: cols[c][i] for c in batch.column_names}
                seen[key] = (row["image"], label_fn(row))
                if len(seen) >= max_n:
                    return seen
        # if we already have >= max_n we'd've returned; else keep scanning files
    return seen


def collect_random(sub, columns, label_fn, n=16, max_scan_rows=8000):
    """Just take first N samples from the first parquet shard (fast)."""
    paths = list_parquets(sub)
    out = []
    scanned = 0
    for p in paths:
        pf = pq.ParquetFile(p)
        for batch in pf.iter_batches(batch_size=256,
                                     columns=["image"] + list(columns)):
            cols = {c: batch.column(c).to_pylist() for c in batch.column_names}
            for i in range(len(cols["image"])):
                if scanned % max(1, max_scan_rows // n) == 0 and len(out) < n:
                    row = {c: cols[c][i] for c in batch.column_names}
                    out.append((row["image"], label_fn(row)))
                scanned += 1
                if len(out) >= n:
                    return out
            if scanned >= max_scan_rows:
                return out
    return out


def main():
    # fota_labeled: one image per object
    print("fota_labeled samples...", flush=True)
    seen = collect_diverse(
        "fota_labeled",
        ["obj_name", "side"],
        lambda r: f"{r['obj_name']}/{r['side']}",
        max_n=16)
    items = list(seen.values())[:16]
    grid = make_grid([thumbnail(b) for b, _ in items],
                     [lab for _, lab in items],
                     cols=4,
                     title=f"fota_labeled — {len(items)} distinct (object, side) pairs")
    grid.save(os.path.join(OUT, "samples_fota_labeled.png"), optimize=True)

    print("skip fota_unlabeled (HDD busy)") ; SKIP_UNLAB = True
    print("fota_unlabeled samples...", flush=True)
    seen = collect_diverse(
        "fota_unlabeled",
        ["obj_name", "side"],
        lambda r: f"{r['obj_name']}/{r['side']}",
        max_n=16)
    items = list(seen.values())[:16]
    grid = make_grid([thumbnail(b) for b, _ in items],
                     [lab for _, lab in items],
                     cols=4,
                     title=f"fota_unlabeled — {len(items)} distinct (object, side) pairs")
    grid.save(os.path.join(OUT, "samples_fota_unlabeled.png"), optimize=True)

    # threedcal: sample by unique x,y position
    print("threedcal samples...", flush=True)
    seen = collect_diverse(
        "threedcal",
        ["x_mm", "y_mm", "z_mm"],
        lambda r: f"x={r['x_mm']:.1f} y={r['y_mm']:.1f} z={r['z_mm']:.1f}",
        max_n=16)
    items = list(seen.values())[:16]
    grid = make_grid([thumbnail(b) for b, _ in items],
                     [lab for _, lab in items],
                     cols=4,
                     title="threedcal — sphere indentation at varying (x,y) probe positions")
    grid.save(os.path.join(OUT, "samples_threedcal.png"), optimize=True)

    # feats: sample by (indenter, param)
    print("feats samples...", flush=True)
    seen = collect_diverse(
        "feats",
        ["indenter", "indenter_param", "f_z"],
        lambda r: f"{r['indenter']}-{r['indenter_param']}  fz={r['f_z']:.2f}"
                  if r["f_z"] is not None else f"{r['indenter']}-{r['indenter_param']}",
        max_n=16)
    items = list(seen.values())[:16]
    grid = make_grid([thumbnail(b) for b, _ in items],
                     [lab for _, lab in items],
                     cols=4,
                     title=f"feats (markered gel) — {len(items)} indenter shape/size combos")
    grid.save(os.path.join(OUT, "samples_feats.png"), optimize=True)

    # combined overview: 4 rows × 8 cols, one row per subset
    print("combined overview...", flush=True)
    pad = 8
    side = 160
    title_h = 36
    row_label_w = 140
    cols = 8
    rows = 4
    W = row_label_w + pad + cols * (side + pad)
    H = title_h + rows * (side + pad) + pad
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    try:
        f_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
        f_row = ImageFont.truetype("DejaVuSans-Bold.ttf", 15)
    except Exception:
        f_title = ImageFont.load_default()
        f_row = ImageFont.load_default()
    d.text((pad, 8), "gelsight-mini-pretrain · subset overview",
           fill=(0, 0, 0), font=f_title)
    for ri, sub in enumerate(["fota_labeled", "fota_unlabeled",
                              "threedcal", "feats"]):
        # First 8 unique images we can find
        items = collect_random(sub, [], lambda r: "", n=cols)
        if len(items) < cols:
            # fallback: not enough, scan more
            items = collect_random(sub, [], lambda r: "", n=cols,
                                   max_scan_rows=20000)
        y = title_h + ri * (side + pad)
        d.text((pad, y + side // 2 - 8), sub, fill=(0, 0, 0), font=f_row)
        for ci, (b, _) in enumerate(items[:cols]):
            im = thumbnail(b, side=side)
            x = row_label_w + pad + ci * (side + pad)
            canvas.paste(im, (x, y))
    canvas.save(os.path.join(OUT, "combined_overview.png"), optimize=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
