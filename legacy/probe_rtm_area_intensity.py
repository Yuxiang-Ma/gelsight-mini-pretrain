#!/usr/bin/env python3
"""Probe RTM with a joint (contact_area × contact_intensity) keep rule.

For each touch, we pick the peak-deformation frame within the touch-window
(same as before), then compute two scalars on the central 50% crop:

    pixel_diff   = |frame - baseline|                # grey-level per pixel
    mask         = pixel_diff > PIXEL_THRESH         # ignore sensor noise
    contact_area = sum(mask)                         # in pixels
    contact_int  = mean(pixel_diff[mask])  if mask.any() else 0

A touch is kept iff `contact_area >= A_min  AND  contact_int >= I_min`.

Outputs:
  - rtm_area_intensity_scatter.png  : 2D scatter of (area, intensity)
                                       with operating-point lines drawn
  - samples_100_rtm_op_<name>.png   : 10x10 grid for each candidate
                                       (A_min, I_min) operating point
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

ROOT = "/media/yxma/Disk1/yuxiang/mini_data/markerless/RealTactileMNIST"
OUT  = "/media/yxma/Disk1/yuxiang/mini_data_parquet/assets"

PIXEL_THRESH = 10          # per-pixel grey-level threshold (noise floor)
PROBE_TOUCHES = 2500

# Candidate operating points (A_min in pixels, I_min in grey-levels)
# Central crop is 240 wide × 120 tall ≈ 28800 px ... actually our crop is
# 1/2 × 1/2 = 1/4 of frame -> for 320x240 that's 160x120 = 19,200 px
OPERATING_POINTS = [
    ("strict",         dict(A_min=200,  I_min=20)),  # large + strong
    ("balanced",       dict(A_min=100,  I_min=18)),  # default rec.
    ("lenient",        dict(A_min=50,   I_min=15)),  # accept smaller
    ("area-only",      dict(A_min=100,  I_min=0)),   # area, no intensity bar
    ("intensity-only", dict(A_min=0,    I_min=20)),  # intensity, no area bar
]


def decode_touch(vid_bytes):
    tmpf = f"/tmp/_rtm_ai_{os.getpid()}.mp4"
    with open(tmpf, "wb") as f: f.write(vid_bytes)
    cap = cv2.VideoCapture(tmpf)
    frames, grays = [], []
    while True:
        ok, fr = cap.read()
        if not ok: break
        frames.append(fr[:, :, ::-1])
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = g.shape
        grays.append(g[h//4:3*h//4, w//4:3*w//4])
    cap.release()
    try: os.remove(tmpf)
    except: pass
    if len(frames) < 8: return None
    return frames, grays


def analyse(frames, grays, ts, ts0, ts1):
    """Return (peak_idx, area, intensity, peak_frame_rgb, mean_diff)."""
    baseline = np.median(np.stack(grays[:5]), axis=0)
    # mean-diff (the old scalar metric) for cross-comparison
    deforms = [float(np.abs(g - baseline).mean()) for g in grays]
    in_window = list(range(len(frames)))
    try:
        if ts is not None and ts0 is not None and ts1 is not None \
                and len(ts) == len(frames):
            in_window = [k for k, t in enumerate(ts) if ts0 <= t <= ts1]
            if not in_window: in_window = list(range(len(frames)))
    except Exception:
        pass
    peak_idx = in_window[int(np.argmax([deforms[k] for k in in_window]))]

    diff = np.abs(grays[peak_idx] - baseline)
    mask = diff > PIXEL_THRESH
    area = int(mask.sum())
    intensity = float(diff[mask].mean()) if area > 0 else 0.0
    return peak_idx, area, intensity, frames[peak_idx], deforms[peak_idx]


def main():
    rng = random.Random(11)
    pq_files = sorted(glob(f"{ROOT}/data/*.parquet"))
    bucket = []   # list of (area, intensity, mean_diff, peak_rgb, label)
    t0 = time.time()
    print(f"probing up to {PROBE_TOUCHES} touches...", flush=True)
    for p in pq_files:
        if len(bucket) >= PROBE_TOUCHES: break
        pf = pq.ParquetFile(p)
        for batch in pf.iter_batches(batch_size=4):
            if len(bucket) >= PROBE_TOUCHES: break
            cols = batch.to_pydict()
            n = len(cols["label"])
            for i in range(n):
                if rng.random() > 0.06: continue
                videos = cols["sensor_video"][i] or []
                ts_seq = cols.get("time_stamp_rel_seq", [None]*n)[i] or []
                t_start = cols.get("touch_start_time_rel", [None]*n)[i] or []
                t_end = cols.get("touch_end_time_rel", [None]*n)[i] or []
                label = cols["label"][i]
                for tj, vs in enumerate(videos):
                    if rng.random() > 0.3: continue
                    if len(bucket) >= PROBE_TOUCHES: break
                    vb = vs.get("bytes") if isinstance(vs, dict) else None
                    if not vb: continue
                    out = decode_touch(vb)
                    if out is None: continue
                    frames, grays = out
                    ts = ts_seq[tj] if tj < len(ts_seq) else None
                    ts0 = t_start[tj] if tj < len(t_start) else None
                    ts1 = t_end[tj] if tj < len(t_end) else None
                    pidx, area, intensity, peak_rgb, md = analyse(frames, grays, ts, ts0, ts1)
                    bucket.append((area, intensity, md, peak_rgb, label))
                    if len(bucket) % 200 == 0:
                        dt = time.time() - t0
                        print(f"  {len(bucket)} touches "
                              f"({len(bucket)/max(dt,0.01):.1f}/s)",
                              flush=True)
    n_total = len(bucket)
    print(f"\ncollected {n_total} touches in {time.time()-t0:.0f}s")
    A = np.array([b[0] for b in bucket])
    I = np.array([b[1] for b in bucket])
    MD = np.array([b[2] for b in bucket])
    print(f"area:       min={A.min()} median={int(np.median(A))} mean={int(A.mean())} max={A.max()}")
    print(f"intensity:  min={I.min():.1f} median={np.median(I):.1f} mean={I.mean():.1f} max={I.max():.1f}")
    print(f"correlation(area, intensity) = {np.corrcoef(A, I)[0,1]:.3f}")
    print(f"correlation(area, mean_diff) = {np.corrcoef(A, MD)[0,1]:.3f}")

    # ------------------------------------------------------------------
    # 2D scatter
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    sc = ax.scatter(A, I, c=MD, s=8, alpha=0.55, cmap="magma",
                    edgecolor="none")
    ax.set_xlabel(f"contact_area  (# pixels with |diff| > {PIXEL_THRESH})",
                  fontsize=11)
    ax.set_ylabel(f"contact_intensity  (mean |diff| over those pixels)",
                  fontsize=11)
    ax.set_title(f"Real Tactile MNIST · peak-frame (area, intensity)  "
                 f"for {n_total} touches\n"
                 f"point colour = current mean-diff scalar  (no longer "
                 f"used as the keep rule)",
                 fontsize=11, pad=10)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("mean |diff| (old metric)", fontsize=10)
    # Draw operating-point boundaries
    colors_op = ["#d62728", "#2ca02c", "#1f77b4", "#9467bd", "#ff7f0e"]
    for (name, op), col in zip(OPERATING_POINTS, colors_op):
        kept = (A >= op["A_min"]) & (I >= op["I_min"])
        pct = 100 * kept.sum() / n_total
        ax.axvline(op["A_min"], color=col, linestyle="--", alpha=0.5, linewidth=1)
        ax.axhline(op["I_min"], color=col, linestyle="--", alpha=0.5, linewidth=1)
        ax.text(op["A_min"] + 5, op["I_min"] + 0.3,
                f"{name}: A≥{op['A_min']}, I≥{op['I_min']}  → {pct:.0f}%",
                fontsize=8, color=col, fontweight="bold")
    ax.set_xlim(0, max(800, A.max()*1.05))
    ax.set_ylim(0, max(40, I.max()*1.05))
    ax.grid(alpha=0.2)
    plt.tight_layout()
    out_scatter = f"{OUT}/rtm_area_intensity_scatter.png"
    plt.savefig(out_scatter, dpi=140)
    plt.close()
    print(f"saved {out_scatter}")

    # ------------------------------------------------------------------
    # Per-operating-point 10x10 sample grid
    # ------------------------------------------------------------------
    try:
        f_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except Exception:
        f_title = ImageFont.load_default()

    for name, op in OPERATING_POINTS:
        kept = [b for b in bucket if b[0] >= op["A_min"] and b[1] >= op["I_min"]]
        if not kept:
            print(f"op {name}: 0 kept, skip"); continue
        sample = rng.sample(kept, min(100, len(kept)))
        side = 144; cols = 10; pad = 4; title_h = 44
        rows = (len(sample) + cols - 1) // cols
        W = pad + cols * (side + pad)
        H = title_h + rows * (side + pad) + pad
        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        d = ImageDraw.Draw(canvas)
        pct = 100 * len(kept) / n_total
        d.text((pad + 4, 8),
               f"real_tactile_mnist  ·  '{name}' op:  "
               f"area ≥ {op['A_min']}  &  intensity ≥ {op['I_min']}  "
               f"·  would keep {pct:.1f}% of touches",
               fill=(0, 0, 0), font=f_title)
        for i, (area, intensity, md, fr, lbl) in enumerate(sample):
            r, c = i // cols, i % cols
            x = pad + c * (side + pad)
            y = title_h + r * (side + pad)
            im = Image.fromarray(fr)
            w, h = im.size
            s = min(w, h)
            im = im.crop(((w-s)//2, (h-s)//2, (w+s)//2, (h+s)//2))
            im = im.resize((side, side), Image.LANCZOS)
            canvas.paste(im, (x, y))
        out = f"{OUT}/samples_100_rtm_op_{name}.png"
        canvas.save(out, optimize=True)
        print(f"saved {out}  ({pct:.1f}% kept · {len(sample)} shown)")


if __name__ == "__main__":
    main()
