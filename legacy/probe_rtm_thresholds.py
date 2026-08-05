#!/usr/bin/env python3
"""Probe RTM at multiple peak-deformation thresholds.

Samples ~1500 touches, computes peak-within-window deformation for each,
then for each threshold (0.5, 0.7, 1.0, 1.5, 2.0) randomly draws 100
touches from the kept subset and assembles a 10x10 sample grid.

Output: /media/yxma/Disk1/yuxiang/mini_data_parquet/assets/samples_100_rtm_tau_{tau}.png
"""

import io, os, random, time
from glob import glob

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

ROOT = "/media/yxma/Disk1/yuxiang/mini_data/markerless/RealTactileMNIST"
OUT  = "/media/yxma/Disk1/yuxiang/mini_data_parquet/assets"
N_TOUCHES = 2500          # how many touches to sample for the probe
TAUS = [0.5, 0.7, 1.0, 1.5, 2.0]
GRID_SIDE = 144
COLS = 10
ROWS = 10


def pick_peak(vid_bytes, ts, ts0, ts1):
    """Decode video bytes; return (peak_rgb, peak_deform, peak_idx) or None."""
    tmpf = f"/tmp/_rtm_probe_{os.getpid()}.mp4"
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
    baseline = np.median(np.stack(grays[:5]), axis=0)
    deforms = [float(np.abs(g - baseline).mean()) for g in grays]
    in_window = list(range(len(frames)))
    try:
        if ts is not None and ts0 is not None and ts1 is not None \
                and len(ts) == len(frames):
            in_window = [k for k, t in enumerate(ts) if ts0 <= t <= ts1]
            if not in_window: in_window = list(range(len(frames)))
    except Exception: pass
    peak_idx = in_window[int(np.argmax([deforms[k] for k in in_window]))]
    return frames[peak_idx], deforms[peak_idx], peak_idx


def main():
    rng = random.Random(42)
    pq_files = sorted(glob(f"{ROOT}/data/*.parquet"))
    # Subsample 0.5% of rows -> each row has 256 touches -> we hit plenty
    SUBSAMPLE_ROW = 0.05

    bucket = []   # list of (peak_deform, frame_rgb, label)
    t0 = time.time()
    for p in pq_files:
        if len(bucket) >= N_TOUCHES: break
        pf = pq.ParquetFile(p)
        for batch in pf.iter_batches(batch_size=4):
            if len(bucket) >= N_TOUCHES: break
            cols = batch.to_pydict()
            n = len(cols["label"])
            for i in range(n):
                if rng.random() > SUBSAMPLE_ROW: continue
                if len(bucket) >= N_TOUCHES: break
                videos = cols["sensor_video"][i] or []
                ts_seq = cols.get("time_stamp_rel_seq", [None]*n)[i] or []
                t_start = cols.get("touch_start_time_rel", [None]*n)[i] or []
                t_end = cols.get("touch_end_time_rel", [None]*n)[i] or []
                label = cols["label"][i]
                for tj, vs in enumerate(videos):
                    if rng.random() > 0.3: continue
                    if len(bucket) >= N_TOUCHES: break
                    vb = vs.get("bytes") if isinstance(vs, dict) else None
                    if not vb: continue
                    ts = ts_seq[tj] if tj < len(ts_seq) else None
                    ts0 = t_start[tj] if tj < len(t_start) else None
                    ts1 = t_end[tj] if tj < len(t_end) else None
                    out = pick_peak(vb, ts, ts0, ts1)
                    if out is None: continue
                    fr, d, idx = out
                    bucket.append((d, fr, label))
                    if len(bucket) % 200 == 0:
                        dt = time.time() - t0
                        print(f"collected {len(bucket)} touches "
                              f"({len(bucket)/max(dt,0.01):.1f}/s)", flush=True)
    print(f"\ntotal collected: {len(bucket)} touches in {time.time()-t0:.0f}s")
    deforms = np.array([b[0] for b in bucket])
    print(f"peak-deform stats: min={deforms.min():.2f} "
          f"median={np.median(deforms):.2f} mean={deforms.mean():.2f} "
          f"max={deforms.max():.2f}")

    # Build grids
    try:
        f_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except Exception:
        f_title = ImageFont.load_default()

    for tau in TAUS:
        kept = [b for b in bucket if b[0] >= tau]
        n_kept = len(kept)
        if n_kept == 0:
            print(f"tau={tau}: 0 frames kept, skip")
            continue
        sample = rng.sample(kept, min(COLS*ROWS, n_kept))
        n_pick = len(sample)
        rows = (n_pick + COLS - 1) // COLS
        pad = 4
        title_h = 44
        W = pad + COLS * (GRID_SIDE + pad)
        H = title_h + rows * (GRID_SIDE + pad) + pad
        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        d = ImageDraw.Draw(canvas)
        pct = 100 * n_kept / len(bucket)
        d.text((pad + 4, 8),
               f"real_tactile_mnist  ·  peak-deform τ ≥ {tau}  ·  "
               f"would keep {pct:.1f}% of touches  ·  showing {n_pick} "
               f"randomly drawn samples",
               fill=(0, 0, 0), font=f_title)
        for i, (deform, fr, lbl) in enumerate(sample):
            r, c = i // COLS, i % COLS
            x = pad + c * (GRID_SIDE + pad)
            y = title_h + r * (GRID_SIDE + pad)
            im = Image.fromarray(fr)
            w, h = im.size
            s = min(w, h)
            im = im.crop(((w-s)//2, (h-s)//2, (w+s)//2, (h+s)//2))
            im = im.resize((GRID_SIDE, GRID_SIDE), Image.LANCZOS)
            canvas.paste(im, (x, y))
        out = f"{OUT}/samples_100_rtm_tau_{tau:.1f}.png"
        canvas.save(out, optimize=True)
        print(f"saved {out}  (kept fraction={pct:.1f}% · {n_pick} samples shown)")


if __name__ == "__main__":
    main()
