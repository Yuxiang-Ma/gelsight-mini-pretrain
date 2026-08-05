#!/usr/bin/env python3
"""Probe gelslam and tactile_tracking with the proper per-capture pipeline
baseline (median of first 10 raw video frames), measure area + intensity
of every subsequent frame, and report distributions + sample grids at
candidate (A_min, I_min) operating points.

Outputs:
  rtm-style scatter (area, intensity) for each source
  samples_100_<source>_op_<name>.png  for a few candidate operating points
"""
import io, os, random, time
from glob import glob

import cv2
import numpy as np
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

BASE_DATA = "/media/yxma/Disk1/yuxiang/mini_data"
OUT       = "/media/yxma/Disk1/yuxiang/mini_data_parquet/assets"
PIXEL_THRESH = 10
BASE_FRAMES  = 10

OPS = [
    ("strict",   dict(A_min=400, I_min=15)),
    ("balanced", dict(A_min=200, I_min=12)),
    ("lenient",  dict(A_min=100, I_min=10)),
]


def grey_center(arr_bgr):
    g = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = g.shape
    return g[h//4:3*h//4, w//4:3*w//4]


def iter_video_frames(path):
    cap = cv2.VideoCapture(path)
    while True:
        ok, fr = cap.read()
        if not ok: break
        yield fr
    cap.release()


def probe_video(path, max_frames_per_clip=None):
    """Yield (rgb, area, intensity) for each non-baseline frame in this clip."""
    buf = []
    baseline = None
    fi = 0
    for fr in iter_video_frames(path):
        fi += 1
        if max_frames_per_clip and fi > max_frames_per_clip: break
        g = grey_center(fr)
        if baseline is None:
            buf.append(g)
            if len(buf) >= BASE_FRAMES:
                baseline = np.median(np.stack(buf, axis=0), axis=0)
            continue
        diff = np.abs(g - baseline)
        mask = diff > PIXEL_THRESH
        area = int(mask.sum())
        inten = float(diff[mask].mean()) if area > 0 else 0.0
        yield fr[:, :, ::-1], area, inten


def probe_source(sub, max_clips=40, max_frames_per_clip=80):
    if sub == "gelslam":
        vids = sorted(glob(f"{BASE_DATA}/markerless/GelSLAM/tracking_dataset/*/gelsight.avi")) \
             + sorted(glob(f"{BASE_DATA}/markerless/GelSLAM/reconstruction_dataset/*/gelsight.avi"))
    elif sub == "tactile_tracking":
        vids = sorted(glob(f"{BASE_DATA}/markerless/TactileTracking/normalflow_dataset/*/gelsight.avi"))
    else:
        return [], []
    rng = random.Random(7)
    rng.shuffle(vids)
    vids = vids[:max_clips]
    bucket = []   # (area, intensity, rgb)
    t0 = time.time()
    print(f"probing {sub}: {len(vids)} clips...", flush=True)
    for i, v in enumerate(vids):
        for rgb, area, inten in probe_video(v, max_frames_per_clip=max_frames_per_clip):
            bucket.append((area, inten, rgb))
        if (i+1) % 5 == 0:
            print(f"  {i+1}/{len(vids)} clips, {len(bucket)} frames, "
                  f"{(i+1)/(time.time()-t0):.1f} clips/s", flush=True)
    A = np.array([b[0] for b in bucket])
    I = np.array([b[1] for b in bucket])
    print(f"  {sub} ({len(bucket)} frames):")
    print(f"    area:      min={A.min()} median={int(np.median(A))} mean={int(A.mean())} max={A.max()}")
    print(f"    intensity: min={I.min():.1f} median={np.median(I):.1f} mean={I.mean():.1f} max={I.max():.1f}")
    for name, op in OPS:
        kept = ((A >= op["A_min"]) & (I >= op["I_min"])).sum()
        print(f"    {name:8s} A>={op['A_min']}, I>={op['I_min']}: {kept}/{len(bucket)} ({100*kept/len(bucket):.1f}%)")
    return bucket, OPS


def render_grid(bucket, op, name, sub, out_path):
    A_min, I_min = op["A_min"], op["I_min"]
    kept = [(a, i, fr) for a, i, fr in bucket if a >= A_min and i >= I_min]
    pct = 100 * len(kept) / len(bucket)
    rng = random.Random(0)
    sample = rng.sample(kept, min(100, len(kept)))
    if not sample:
        print(f"  {name}: 0 kept, skip"); return
    side = 144; cols = 10; pad = 4; title_h = 44
    rows = (len(sample) + cols - 1) // cols
    W = pad + cols * (side + pad)
    H = title_h + rows * (side + pad) + pad
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    try: f_t = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except: f_t = ImageFont.load_default()
    d.text((pad + 4, 8),
           f"{sub}  ·  '{name}' op:  A>={A_min}, I>={I_min}  ·  "
           f"keep {pct:.1f}%  ·  {len(sample)} samples shown",
           fill=(0, 0, 0), font=f_t)
    for i, (a, intensity, fr) in enumerate(sample):
        r, c = i // cols, i % cols
        x = pad + c * (side + pad)
        y = title_h + r * (side + pad)
        im = Image.fromarray(fr)
        w, h = im.size; s = min(w, h)
        im = im.crop(((w-s)//2, (h-s)//2, (w+s)//2, (h+s)//2)).resize((side, side), Image.LANCZOS)
        canvas.paste(im, (x, y))
    canvas.save(out_path, optimize=True)
    print(f"  saved {out_path}  ({pct:.1f}% kept)")


def main():
    for sub in ["gelslam", "tactile_tracking"]:
        bucket, ops = probe_source(sub, max_clips=30, max_frames_per_clip=120)
        if not bucket: continue
        # render grids for each op
        for name, op in ops:
            render_grid(bucket, op, name, sub,
                        f"{OUT}/samples_100_{sub}_op_{name}.png")
        # scatter
        A = np.array([b[0] for b in bucket])
        I = np.array([b[1] for b in bucket])
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(A, I, s=6, alpha=0.4, color="#4c95d6")
        for (name, op), col in zip(ops, ["#d62728", "#2ca02c", "#1f77b4"]):
            ax.axvline(op["A_min"], color=col, ls="--", alpha=0.5)
            ax.axhline(op["I_min"], color=col, ls="--", alpha=0.5)
            kept = ((A >= op["A_min"]) & (I >= op["I_min"])).sum()
            ax.text(op["A_min"] + 50, op["I_min"] + 0.3,
                    f"{name}: A≥{op['A_min']}, I≥{op['I_min']}  → "
                    f"{100*kept/len(A):.0f}%",
                    fontsize=9, color=col, fontweight="bold")
        ax.set_xlabel(f"contact_area (# pixels with |diff| > {PIXEL_THRESH})")
        ax.set_ylabel("contact_intensity (mean |diff| over those pixels)")
        ax.set_title(f"{sub}  ·  area vs intensity  ({len(bucket)} probed frames)")
        ax.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(f"{OUT}/{sub}_area_intensity_scatter.png", dpi=140)
        plt.close()
        print(f"  saved {OUT}/{sub}_area_intensity_scatter.png")


if __name__ == "__main__":
    main()
