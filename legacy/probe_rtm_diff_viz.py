#!/usr/bin/env python3
"""Render side-by-side (peak-frame | deformation-heatmap) visualisations
across multiple RTM tau thresholds.

For each tau in {0.5, 0.7, 1.0, 1.5, 2.0}, find touches whose
peak_deform falls just above that threshold, and render:

   (RGB peak frame  |  |frame - baseline|  heatmap)

side-by-side, with the peak_deform scalar printed. Output one PNG
per tau plus a combined "tau_grid_comparison.png" with one row per tau.
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

TAUS = [0.5, 0.7, 1.0, 1.5, 2.0]
PER_TAU = 8         # how many examples to render per tau
PROBE_TOUCHES = 2000


def decode_touch(vid_bytes):
    """Return (frames_rgb, grays_centercrop) or None."""
    tmpf = f"/tmp/_rtm_dvz_{os.getpid()}.mp4"
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
    """Return (peak_idx, peak_deform, baseline_full, frame_full_at_peak, diff_central, deform_central)."""
    baseline_center = np.median(np.stack(grays[:5]), axis=0)   # shape (Hc, Wc)
    deforms = [float(np.abs(g - baseline_center).mean()) for g in grays]
    in_window = list(range(len(frames)))
    try:
        if ts is not None and ts0 is not None and ts1 is not None \
                and len(ts) == len(frames):
            in_window = [k for k, t in enumerate(ts) if ts0 <= t <= ts1]
            if not in_window: in_window = list(range(len(frames)))
    except Exception:
        pass
    peak_idx = in_window[int(np.argmax([deforms[k] for k in in_window]))]
    peak_deform = deforms[peak_idx]

    # full-image baseline + diff for visualisation
    H, W = frames[0].shape[:2]
    grays_full = [cv2.cvtColor(cv2.cvtColor(f, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY).astype(np.float32)
                  for f in frames]
    baseline_full = np.median(np.stack(grays_full[:5]), axis=0)
    diff_full = np.abs(grays_full[peak_idx] - baseline_full)
    return peak_idx, peak_deform, frames[peak_idx], diff_full


def main():
    rng = random.Random(7)
    pq_files = sorted(glob(f"{ROOT}/data/*.parquet"))

    # Collect ~PROBE_TOUCHES touches with their peak_deform
    bucket = []   # list of (peak_deform, peak_frame_rgb, diff_full, label)
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
                if rng.random() > 0.05: continue
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
                    pidx, pdef, peak_rgb, diff = analyse(frames, grays, ts, ts0, ts1)
                    bucket.append((pdef, peak_rgb, diff, label))
                    if len(bucket) % 200 == 0:
                        dt = time.time() - t0
                        print(f"  {len(bucket)} touches  ({len(bucket)/max(dt,0.01):.1f}/s)",
                              flush=True)
    print(f"collected {len(bucket)} touches in {time.time()-t0:.0f}s")

    # ------------------------------------------------------------------
    # Build a 5-row comparison figure: each row = one tau, showing
    # PER_TAU (peak | diff) pairs from touches with peak_deform near tau.
    # ------------------------------------------------------------------
    fig_w = PER_TAU * 2.2 + 1.0
    fig_h = len(TAUS) * 2.3
    fig, axes = plt.subplots(len(TAUS), PER_TAU * 2,
                             figsize=(fig_w, fig_h),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.08})

    for ri, tau in enumerate(TAUS):
        # Touches whose peak_deform is in a narrow band around tau
        # (so the visualisation actually shows what 'tau' looks like)
        lo, hi = tau, tau * 1.4
        band = [b for b in bucket if lo <= b[0] < hi]
        if len(band) < PER_TAU:
            band = sorted(bucket, key=lambda b: abs(b[0] - tau))[:PER_TAU]
        rng.shuffle(band)
        chosen = band[:PER_TAU]

        for ci, (pdef, rgb, diff, label) in enumerate(chosen):
            ax_rgb = axes[ri, ci * 2]
            ax_dif = axes[ri, ci * 2 + 1]
            ax_rgb.imshow(rgb)
            ax_rgb.set_xticks([]); ax_rgb.set_yticks([])
            # show diff heatmap, fixed scale 0..20 grey-levels so all rows
            # compare on identical colour mapping
            im = ax_dif.imshow(diff, cmap="magma", vmin=0, vmax=20)
            ax_dif.set_xticks([]); ax_dif.set_yticks([])
            ax_dif.set_title(f"Δ={pdef:.2f}", fontsize=8, pad=1)
            if ci == 0:
                ax_rgb.set_ylabel(f"τ≈{tau}", fontsize=11,
                                  rotation=0, labelpad=22, va="center",
                                  fontweight="bold")

    fig.suptitle(
        "Real Tactile MNIST · peak frame  ↔  |frame − baseline|  heatmap\n"
        "(8 example touches per tau band · diff scale fixed 0–20 grey-levels · "
        "Δ = mean absolute deformation in central crop)",
        fontsize=11, y=0.995)
    out = f"{OUT}/rtm_tau_diff_comparison.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")

    # ------------------------------------------------------------------
    # Also: aggregate-level row showing the DIFFERENCE between baseline
    # and peak averaged over many kept touches at each tau.
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, len(TAUS), figsize=(3 * len(TAUS), 6),
                             gridspec_kw={"hspace": 0.15})
    for ci, tau in enumerate(TAUS):
        kept = [b for b in bucket if b[0] >= tau]
        if not kept: continue
        # mean RGB across kept peak frames
        mean_rgb = np.mean(np.stack([b[1] for b in kept]), axis=0).astype(np.uint8)
        # mean diff
        mean_diff = np.mean(np.stack([b[2] for b in kept]), axis=0)
        axes[0, ci].imshow(mean_rgb)
        axes[0, ci].set_title(f"τ ≥ {tau}\n(mean of {len(kept)} kept peaks)", fontsize=10)
        axes[0, ci].set_xticks([]); axes[0, ci].set_yticks([])
        im = axes[1, ci].imshow(mean_diff, cmap="magma", vmin=0, vmax=8)
        axes[1, ci].set_xticks([]); axes[1, ci].set_yticks([])
        axes[1, ci].set_title("mean |Δ| heatmap (vmax=8)", fontsize=9)
    fig.suptitle(
        "Mean tactile imprint across kept touches at each tau\n"
        "top: mean RGB · bottom: mean |peak − baseline| (the higher tau, the "
        "more clearly the imprint emerges from averaging)", fontsize=11, y=0.995)
    out2 = f"{OUT}/rtm_tau_mean_diff.png"
    plt.savefig(out2, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"saved {out2}")


if __name__ == "__main__":
    main()
