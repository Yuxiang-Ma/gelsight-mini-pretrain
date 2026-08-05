#!/usr/bin/env python3
"""Per-sensor I_min calibrator for the VBTS pipeline.

Given a source of RGB frames (an HF dataset config, a directory of
images, a video file, or a zarr array), this script:

  1. Samples N_SAMPLE frames at random.
  2. Builds a per-resolution median baseline.
  3. For each frame, computes |center_grey - baseline| -> the
     per-pixel signed deviation. Reports:
       - background-diff distribution (5/50/95/99 pct)
       - "contact-frame" peak (frames with mean diff > 2 * bg p95)
       - recommended I_min = round(bg_p99) + 2
       - recommended A_min = 40 (we keep this fixed across sensors)
  4. Plots histogram + sample frames at threshold boundary so we
     can sanity-check by eye.

Outputs both a JSON report and a PNG to ./calibration/<source>/.

Usage:
  python calibrate_imin.py --hf yxma/gelsight-mini-pretrain --config fota_unlabeled --split train
  python calibrate_imin.py --dir /path/to/images
  python calibrate_imin.py --video /path/to/clip.mp4
  python calibrate_imin.py --zarr /path/to/replay_buffer.zarr --zarr-key data/tactile_image

The script does NOT modify any data; it just reports recommended
thresholds we can plug into per-sensor IngestPipeline configs.
"""
import argparse
import glob
import io
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

N_SAMPLE = 1000          # sample size for calibration
N_BASELINE = 200         # frames used to build median baseline
A_MIN = 40               # area floor; not tuned per sensor
PIXEL_THRESH = 10        # per-pixel diff threshold (fixed)
CENTER_FRAC = 0.5        # central 50% ROI (same as Mini pipeline)


def center_grey(rgb):
    g = rgb.mean(axis=2).astype(np.float32)
    h, w = g.shape
    yh, yw = int(h * (1 - CENTER_FRAC) / 2), int(w * (1 - CENTER_FRAC) / 2)
    return g[yh:h - yh, yw:w - yw]


# ---------- frame sources ----------

def iter_hf(dataset_name, config, split):
    from datasets import load_dataset
    ds = load_dataset(dataset_name, config, split=split, streaming=False)
    idxs = list(range(len(ds)))
    random.shuffle(idxs)
    for i in idxs:
        yield np.asarray(ds[i]["image"])


def iter_dir(root, include=None, exclude=None):
    """Walk root for image files. `include`/`exclude` are substrings that
    paths must contain / must not contain (e.g. include='tactile' to skip
    paired vision images in TVL)."""
    paths = []
    for ext in ("jpg", "jpeg", "png", "bmp", "tif"):
        paths.extend(glob.glob(f"{root}/**/*.{ext}", recursive=True))
        paths.extend(glob.glob(f"{root}/**/*.{ext.upper()}", recursive=True))
    if include:
        paths = [p for p in paths if include in p]
    if exclude:
        paths = [p for p in paths if exclude not in p]
    random.shuffle(paths)
    for p in paths:
        try:
            yield np.asarray(Image.open(p).convert("RGB"))
        except Exception:
            continue


def iter_video(path):
    import imageio.v3 as iio
    frames = list(iio.imiter(path))
    random.shuffle(frames)
    for f in frames:
        yield np.asarray(f)


def iter_zarr(path, key):
    import zarr
    store = zarr.DirectoryStore(f"{path}/{key}")
    arr = zarr.open_array(store=store, mode="r")
    n = arr.shape[0]
    idxs = list(range(n))
    random.shuffle(idxs)
    for i in idxs:
        yield arr[i][:]


def collect_frames(source_iter, n):
    out = []
    for f in source_iter:
        if f is None or f.ndim != 3 or f.shape[2] != 3:
            continue
        out.append(f)
        if len(out) >= n:
            break
    return out


# ---------- calibration ----------

def calibrate(frames, name, out_dir):
    if not frames:
        sys.exit("No frames collected; check the source path.")
    print(f"  collected {len(frames)} frames", flush=True)

    # Group by resolution (handles mixed-res sources like TVL)
    by_shape = {}
    for f in frames:
        by_shape.setdefault(f.shape, []).append(f)

    print(f"  resolutions: {[(s, len(v)) for s, v in by_shape.items()]}",
          flush=True)

    # Build per-shape baseline
    baselines = {}
    for shape, fs in by_shape.items():
        nb = min(N_BASELINE, len(fs))
        sample = random.sample(fs, nb)
        baselines[shape] = np.median(
            np.stack([center_grey(s) for s in sample]), axis=0)

    # Per-frame intensity stats
    intens, areas, means = [], [], []
    for f in frames:
        bl = baselines[f.shape]
        diff = np.abs(center_grey(f) - bl)
        mask = diff > PIXEL_THRESH
        area = int(mask.sum())
        inten = float(diff[mask].mean()) if area > 0 else 0.0
        intens.append(inten)
        areas.append(area)
        means.append(float(diff.mean()))

    intens = np.asarray(intens)
    areas = np.asarray(areas)
    means = np.asarray(means)

    # Recommendation logic: I_min = round(bg_p99) + 2.
    # We treat the lowest-25%-by-mean-diff frames as background (no contact).
    # Using mean-diff (not mask-area) because mask-area = 0 frames produce
    # intens = 0 by construction, which biases bg_intens toward 0.
    bg_mask = means < np.percentile(means, 25)
    bg_intens = intens[bg_mask]
    # Exclude zero-intensity frames (no pixels above PIXEL_THRESH) — they
    # are not informative for finding the noise ceiling.
    bg_intens_nz = bg_intens[bg_intens > 0]
    if bg_intens_nz.size >= 10:
        bg_p95 = float(np.percentile(bg_intens_nz, 95))
        bg_p99 = float(np.percentile(bg_intens_nz, 99))
        recommended = int(round(bg_p99)) + 2
    else:
        # Fall back to the global intensity p25 as a noise estimate
        bg_p95 = float(np.percentile(intens[intens > 0], 25)) if (intens > 0).any() else 0.0
        bg_p99 = bg_p95
        recommended = max(int(round(bg_p95)) + 2, PIXEL_THRESH + 2)

    report = {
        "source": name,
        "n_frames": len(frames),
        "resolutions": {str(k): len(v) for k, v in by_shape.items()},
        "intensity_pct": {
            "p05": float(np.percentile(intens, 5)),
            "p50": float(np.percentile(intens, 50)),
            "p95": float(np.percentile(intens, 95)),
            "p99": float(np.percentile(intens, 99)),
        },
        "area_pct": {
            "p05": float(np.percentile(areas, 5)),
            "p50": float(np.percentile(areas, 50)),
            "p95": float(np.percentile(areas, 95)),
        },
        "background_diff": {
            "bg_intensity_p95": bg_p95,
            "bg_intensity_p99": bg_p99,
        },
        "recommended_I_min": recommended,
        "fixed_A_min": A_MIN,
        "fixed_PIXEL_THRESH": PIXEL_THRESH,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}_calibration.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].hist(intens, bins=60)
        ax[0].axvline(recommended, color="red", linestyle="--",
                      label=f"I_min={recommended}")
        ax[0].set_xlabel("intensity (px diff)")
        ax[0].set_ylabel("frames")
        ax[0].set_title(f"{name}: intensity distribution")
        ax[0].legend()
        ax[1].hist(areas, bins=60)
        ax[1].axvline(A_MIN, color="red", linestyle="--", label=f"A_min={A_MIN}")
        ax[1].set_xlabel("contact area (px)")
        ax[1].set_title(f"{name}: area distribution")
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}_calibration.png", dpi=100)
        plt.close(fig)
        print(f"  plot saved to {out_dir / f'{name}_calibration.png'}",
              flush=True)
    except Exception as e:
        print(f"  plot skipped ({e})")

    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None,
                    help="output filename stem (default derived from source)")
    ap.add_argument("--n", type=int, default=N_SAMPLE)
    ap.add_argument("--seed", type=int, default=20260521)
    ap.add_argument("--out", default="./calibration")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--hf", help="HF dataset name, e.g. yxma/gelsight-mini-pretrain")
    src.add_argument("--dir", help="directory of image files")
    src.add_argument("--video", help="video file")
    src.add_argument("--zarr", help="path to zarr root")
    ap.add_argument("--config", default=None, help="HF config name")
    ap.add_argument("--split", default="train", help="HF split")
    ap.add_argument("--zarr-key", default="data/tactile_image",
                    help="key within zarr root")
    ap.add_argument("--include", default=None,
                    help="(--dir only) substring paths must contain")
    ap.add_argument("--exclude", default=None,
                    help="(--dir only) substring paths must NOT contain")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.hf:
        name = args.name or f"hf_{args.hf.replace('/', '__')}_{args.config or 'default'}"
        it = iter_hf(args.hf, args.config, args.split)
    elif args.dir:
        name = args.name or f"dir_{Path(args.dir).name}"
        it = iter_dir(args.dir, include=args.include, exclude=args.exclude)
    elif args.video:
        name = args.name or f"vid_{Path(args.video).stem}"
        it = iter_video(args.video)
    elif args.zarr:
        name = args.name or f"zarr_{Path(args.zarr).stem}"
        it = iter_zarr(args.zarr, args.zarr_key)

    print(f"Calibrating {name} from {args.n} samples...", flush=True)
    frames = collect_frames(it, args.n)
    calibrate(frames, name, Path(args.out))


if __name__ == "__main__":
    main()
