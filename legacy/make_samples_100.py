#!/usr/bin/env python3
"""Sample 100 images per subset into a 10x10 grid."""

import glob
import io
import os
import random
import sys

import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

BASE = "/media/yxma/Disk1/yuxiang/mini_data_parquet"
OUT = os.path.join(BASE, "assets")
os.makedirs(OUT, exist_ok=True)

N = 40
COLS = 10
THUMB = 144  # smaller to keep image manageable


def list_parquets(sub, split_pat="*"):
    return sorted(glob.glob(os.path.join(BASE, sub, f"{split_pat}-*.parquet")))


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
        f = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
    except Exception:
        f = ImageFont.load_default()
    d.text((pad + 4, 8), title, fill=(0, 0, 0), font=f)
    for i, im in enumerate(images):
        r, c = i // cols, i % cols
        x = pad + c * (side + pad)
        y = title_h + r * (side + pad)
        canvas.paste(im, (x, y))
    return canvas


def collect_uniform(sub, n=N, key_cols=None, seed=42):
    """Sample n images uniformly across the parquet.

    If key_cols given, also try to balance across the (key_cols) keyspace.
    """
    rng = random.Random(seed)
    paths = list_parquets(sub)
    # First, count total rows
    counts = [pq.read_metadata(p).num_rows for p in paths]
    total = sum(counts)
    if total == 0:
        return []
    # Pick global indices uniformly
    n = min(n, total)
    idxs = sorted(rng.sample(range(total), n))
    # Walk shards reading only needed rows
    out = []
    cum = 0
    idx_iter = iter(idxs)
    nxt = next(idx_iter, None)
    for p, c in zip(paths, counts):
        if nxt is None:
            break
        if nxt >= cum + c:
            cum += c
            continue
        # collect local indices in this shard
        local = []
        while nxt is not None and nxt < cum + c:
            local.append(nxt - cum)
            nxt = next(idx_iter, None)
        if local:
            t = pq.read_table(p, columns=["image"])
            for li in local:
                out.append(t.column("image")[li].as_py())
        cum += c
    return out


def main():
    subs = sys.argv[1:] if len(sys.argv) > 1 else [
        "fota_labeled", "fota_unlabeled", "threedcal", "feats"]
    for sub in subs:
        if not list_parquets(sub):
            print(f"skip {sub} (no parquet found)")
            continue
        print(f"sampling {N} from {sub}...", flush=True)
        imgs = collect_uniform(sub, n=N)
        thumbs = [thumbnail(b) for b in imgs]
        title = f"{sub} — {N} random samples"
        if sub == "feats":
            title += " (includes black_dot + different gel variants)"
        g = make_grid(thumbs, title)
        out = os.path.join(OUT, f"samples_{N}_{sub}.png")
        g.save(out, optimize=True)
        print(f"  saved {out}  size={g.size}", flush=True)


if __name__ == "__main__":
    main()
