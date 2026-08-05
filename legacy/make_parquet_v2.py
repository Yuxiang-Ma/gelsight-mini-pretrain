#!/usr/bin/env python3
"""V2 pipeline: pack GelSLAM, TactileTracking, RealTactileMNIST, FeelAnyForce
into the unified parquet schema used by `yxma/gelsight-mini-pretrain`.

Adaptive subsampling per source:
  - per-source frame budget = min(200_000, contact_frames_available)
  - dynamism-aware stride (high inter-frame delta -> denser sample)
  - validity filter: drop frames where central deformation < threshold
  - perceptual-hash dedupe

Outputs sit alongside existing fota_*/threedcal/feats:
  /media/yxma/Disk1/yuxiang/mini_data_parquet/<sub>/{train|...}-####-of-####.parquet

Usage:
  python make_parquet_v2.py probe <sub>       # measure dynamism + empty fraction
  python make_parquet_v2.py process <sub>     # full pipeline + write
  python make_parquet_v2.py stats             # show row counts across all subs
"""

import io
import json
import os
import sys
import time
from collections import defaultdict
from glob import glob
from typing import Iterator, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

BASE_DATA   = "/media/yxma/Disk1/yuxiang/mini_data"
BASE_OUT    = "/media/yxma/Disk1/yuxiang/mini_data_parquet"
BASE_OUT_NC = "/media/yxma/Disk1/yuxiang/mini_data_parquet_nc"
NC_SOURCES  = {"faf_force_estimation"}  # sources whose license requires NC-repo output
BUDGET      = 200_000                # max kept frames per source
SHARD_TGT   = 2 * 1024 ** 3          # 2 GB shard target

# Validity filter — area + intensity rule (replaces former mean-deform tau).
# A frame is valid iff, on the central-50%-crop |frame - baseline| diff:
#     n_pixels_above(PIXEL_THRESH) >= A_min   AND   their mean diff >= I_min
PIXEL_THRESH = 10                    # sensor-noise floor, grey-levels
EMPTY_BUDGET = 0.015                 # keep ~1-2 % of below-threshold frames as background diversity
PHASH_DIST   = 4                     # max hamming distance for "duplicate"

# UNIFIED area+intensity rule applied to every source (markerless).
# Per-source thresholds — defaults to A=40, I=15 unless overridden.
A_MIN_DEFAULT = 40
I_MIN_DEFAULT = 10
VALIDITY_THRESH = {
    "gelslam":           dict(A_min=40, I_min=10),
    "tactile_tracking":  dict(A_min=40, I_min=10),
}

# Existing schema + 3 new nullable cols
SCHEMA = pa.schema([
    ("image", pa.binary()),
    ("image_format", pa.string()),
    ("source", pa.string()),
    ("markered", pa.bool_()),
    ("capture", pa.string()),
    ("split", pa.string()),
    ("height", pa.int32()),
    ("width", pa.int32()),
    ("obj_name", pa.string()),
    ("init_pose", pa.int32()),
    ("side", pa.string()),
    ("x_mm", pa.float32()),
    ("y_mm", pa.float32()),
    ("z_mm", pa.float32()),
    ("quat_x", pa.float32()),
    ("quat_y", pa.float32()),
    ("quat_z", pa.float32()),
    ("quat_w", pa.float32()),
    ("indenter", pa.string()),
    ("indenter_param", pa.string()),
    ("f_x", pa.float32()),
    ("f_y", pa.float32()),
    ("f_z", pa.float32()),
    ("grid_z_max", pa.float32()),
    ("grid_z_mean", pa.float32()),
    # NEW columns (nullable for old data)
    ("episode", pa.string()),
    ("frame_idx", pa.int32()),
    ("digit_class", pa.int32()),
    ("gel_variant", pa.string()),
    # v3 — distinguish real-world capture vs synthetic
    ("domain", pa.string()),    # "real" | "sim"
])


# ─────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────

def encode_jpeg(arr_rgb: np.ndarray, q=92) -> bytes:
    im = Image.fromarray(arr_rgb)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=q, optimize=True)
    return buf.getvalue()


def grey_center(arr: np.ndarray) -> np.ndarray:
    """Central 50% crop, greyscale, float32."""
    if arr.ndim == 3:
        g = arr.mean(axis=2)
    else:
        g = arr
    h, w = g.shape
    return g[h//4:3*h//4, w//4:3*w//4].astype(np.float32)


def phash(arr_rgb: np.ndarray) -> int:
    """8x8 DCT-low-freq perceptual hash, returned as 64-bit int."""
    im = Image.fromarray(arr_rgb).convert("L").resize((32, 32), Image.LANCZOS)
    a = np.array(im, dtype=np.float32)
    # 2D DCT via numpy
    def dct1(x): return np.fft.fft(np.concatenate([x, x[::-1]], axis=-1)).real[..., :x.shape[-1]]
    d = dct1(dct1(a).T).T
    low = d[:8, :8].flatten()
    med = np.median(low[1:])  # skip DC
    h = 0
    for bit in (low > med):
        h = (h << 1) | int(bit)
    return h


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# ─────────────────────────────────────────────────────────────────
# Per-source iterators
# Each yields (frame_rgb_np, base_meta_dict) for ONE frame at a time.
# Within each episode/capture, frames are emitted in order so we can
# compute a baseline image for the validity filter.
# ─────────────────────────────────────────────────────────────────

def iter_gelslam():
    """GelSLAM: gelsight.avi per episode under tracking_dataset/ and reconstruction_dataset/."""
    import cv2
    root = f"{BASE_DATA}/markerless/GelSLAM"
    # The HF download puts content under root or root/dataset/ depending on extraction
    for sub in ("tracking_dataset", "reconstruction_dataset"):
        candidates = (
            glob(f"{root}/{sub}/*/gelsight.avi") +
            glob(f"{root}/dataset/{sub}/*/gelsight.avi")
        )
        for vid in candidates:
            episode = os.path.basename(os.path.dirname(vid))
            split = "train" if sub == "tracking_dataset" else "recon"
            cap = cv2.VideoCapture(vid)
            fi = 0
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                fr = fr[:, :, ::-1]  # BGR->RGB
                yield fr, {
                    "source": "gelslam",
                    "markered": False,
                    "capture": f"{sub}/{episode}",
                    "split": split,
                    "obj_name": episode,
                    "episode": episode,
                    "frame_idx": fi,
                }
                fi += 1
            cap.release()


def iter_tactile_tracking():
    import cv2
    import re
    root = f"{BASE_DATA}/markerless/TactileTracking"
    candidates = sorted(glob(f"{root}/normalflow_dataset/*/gelsight.avi"))
    obj_re = re.compile(r"^([a-zA-Z]+)(\d+)$")
    for vid in candidates:
        trial_dir = os.path.basename(os.path.dirname(vid))  # e.g. 'corner3'
        m = obj_re.match(trial_dir)
        if m:
            obj, trial = m.group(1), m.group(2)
        else:
            obj, trial = trial_dir, "0"
        cap = cv2.VideoCapture(vid)
        fi = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            fr = fr[:, :, ::-1]
            yield fr, {
                "source": "tactile_tracking",
                "markered": False,
                "capture": f"{obj}/{trial}",
                "split": "train",
                "obj_name": obj,
                "episode": trial,
                "frame_idx": fi,
            }
            fi += 1
        cap.release()


def iter_real_tactile_mnist():
    """RTM seq-320x240: parquet with 'sensor_video' list-of-struct{bytes,path}.

    Frame-picking strategy (v5 — per-frame Bernoulli within touch window):
      1. Decode all frames in the touch video clip (~60-73 frames).
      2. Compute a per-clip baseline = median of first 5 frames.
      3. Use `touch_start_time_rel`/`touch_end_time_rel` to find frames in
         the contact window (typically ~6 frames near the end of the clip).
      4. For EACH frame in the touch window:
           pixel_diff   = |frame - baseline|  on central 50% crop
           mask         = pixel_diff > PIXEL_THRESH   (10 grey-levels)
           contact_area = mask.sum()                 (pixels)
           contact_int  = pixel_diff[mask].mean()
         If (contact_area >= A_MIN AND contact_int >= I_MIN):
             keep with probability K=0.3   (≈ 2 of 6 frames per window)
         Else:
             keep with probability BG_RATE=0.015  (background diversity)
    """
    import cv2
    import random
    root = f"{BASE_DATA}/markerless/RealTactileMNIST"
    pq_files = sorted(glob(f"{root}/data/*.parquet"))
    PIXEL_THRESH = 10
    A_MIN = 40
    I_MIN = 15           # RTM uses stricter I (digit imprints are inherently subtle)
    K_SAMPLE = 0.30               # per-valid-frame sampling probability inside touch window
    BG_KEEP_RATE = 0.015          # per-frame-in-window background keep for failed frames
    rng = random.Random(42)
    for p in pq_files:
        split = "test" if "test" in os.path.basename(p).lower() else "train"
        pf = pq.ParquetFile(p)
        for batch in pf.iter_batches(batch_size=4):
            cols = batch.to_pydict()
            n = len(cols.get("label", cols.get("id", [])))
            for i in range(n):
                round_id = cols.get("id", [None]*n)[i]
                label = cols.get("label", [None]*n)[i]
                obj_id = cols.get("object_id", [None]*n)[i]
                videos = cols["sensor_video"][i] or []
                ts_seq = cols.get("time_stamp_rel_seq", [None]*n)[i] or []
                t_start = cols.get("touch_start_time_rel", [None]*n)[i] or []
                t_end = cols.get("touch_end_time_rel", [None]*n)[i] or []
                for tj, vid_struct in enumerate(videos):
                    if not vid_struct: continue
                    vid_bytes = vid_struct.get("bytes") if isinstance(vid_struct, dict) else None
                    if not vid_bytes: continue
                    tmpf = f"/tmp/_rtm_{os.getpid()}.mp4"
                    with open(tmpf, "wb") as f: f.write(vid_bytes)
                    cap = cv2.VideoCapture(tmpf)
                    frames = []; grays = []
                    while True:
                        ok, fr = cap.read()
                        if not ok: break
                        rgb = fr[:, :, ::-1]
                        frames.append(rgb)
                        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
                        h, w = g.shape
                        grays.append(g[h//4:3*h//4, w//4:3*w//4])
                    cap.release()
                    try: os.remove(tmpf)
                    except: pass
                    if len(frames) < 8: continue

                    baseline = np.median(np.stack(grays[:5]), axis=0)

                    in_window = list(range(len(frames)))
                    try:
                        ts = ts_seq[tj] if tj < len(ts_seq) else None
                        ts0 = t_start[tj] if tj < len(t_start) else None
                        ts1 = t_end[tj] if tj < len(t_end) else None
                        if ts is not None and ts0 is not None and ts1 is not None \
                                and len(ts) == len(frames):
                            in_window = [k for k, t in enumerate(ts) if ts0 <= t <= ts1]
                            if not in_window:
                                in_window = list(range(len(frames)))
                    except Exception:
                        pass

                    # Per-frame Bernoulli sampling inside the touch window
                    for k in in_window:
                        pixel_diff = np.abs(grays[k] - baseline)
                        mask = pixel_diff > PIXEL_THRESH
                        contact_area = int(mask.sum())
                        contact_int = float(pixel_diff[mask].mean()) if contact_area > 0 else 0.0
                        passes = (contact_area >= A_MIN) and (contact_int >= I_MIN)
                        if passes:
                            if rng.random() >= K_SAMPLE: continue
                        else:
                            if rng.random() >= BG_KEEP_RATE: continue
                        yield frames[k], {
                            "source": "real_tactile_mnist",
                            "markered": False,
                            "capture": f"r{round_id}_t{tj}",
                            "split": split,
                            "obj_name": f"digit_{label}",
                            "digit_class": int(label) if label is not None else None,
                            "episode": str(obj_id) if obj_id is not None else None,
                            "frame_idx": k,
                        }


def iter_feelanyforce():
    """FAF: loose 320x240 PNG per indentation under dataset/<object>/tactile/.

    Validity-filter strategy: per-object median fails because EVERY FAF frame
    has contact (force-controlled). So we use a **global cross-OBJECT median**
    — sample 1 image from each of the 42 objects (touches at different
    positions/objects), median ≈ gel-at-rest. Then apply unified A=40, I=15
    + 1.5% bg-keep.
    """
    import random
    rng = random.Random(0)
    A_MIN, I_MIN = 40, 10
    BG = 0.015
    root = f"{BASE_DATA}/markerless/FeelAnyForce/dataset/dataset"
    objects = sorted(d for d in os.listdir(root) if os.path.isdir(f"{root}/{d}"))
    # Build GLOBAL baseline from 1 random image per object
    grays = []
    for obj in objects:
        p = f"{root}/{obj}/tactile"
        if not os.path.isdir(p): continue
        files = sorted(os.listdir(p))
        if not files: continue
        for fn in rng.sample(files, min(3, len(files))):
            try:
                im = np.asarray(Image.open(f"{p}/{fn}").convert("L"), dtype=np.float32)
                h, w = im.shape
                grays.append(im[h//4:3*h//4, w//4:3*w//4])
            except Exception:
                pass
    if len(grays) < 30:
        return
    baseline = np.median(np.stack(grays), axis=0)
    # Iterate every frame, apply filter
    for obj in objects:
        p = f"{root}/{obj}/tactile"
        if not os.path.isdir(p): continue
        for fi, fn in enumerate(sorted(os.listdir(p))):
            try:
                rgb = np.array(Image.open(f"{p}/{fn}").convert("RGB"))
            except Exception:
                continue
            g = rgb.mean(axis=2).astype(np.float32)
            h, w = g.shape
            g_center = g[h//4:3*h//4, w//4:3*w//4]
            diff = np.abs(g_center - baseline)
            mask = diff > PIXEL_THRESH
            area = int(mask.sum())
            inten = float(diff[mask].mean()) if area > 0 else 0.0
            passes = (area >= A_MIN) and (inten >= I_MIN)
            if not passes and rng.random() >= BG: continue
            yield rgb, {
                "source": "feelanyforce",
                "markered": False,
                "domain": "real",
                "capture": obj,
                "split": "train",
                "obj_name": obj.split("_")[0],
                "episode": obj,
                "frame_idx": fi,
            }


def _iter_sim_parquet_filtered(root, source_tag, label_to_obj_name):
    """Shared logic for sim_tactile_mnist + sim_starstruck:
       per-row baseline = median of all 32 touches (touches are at different
       positions, so the median approximates the gel-at-rest background).
       Filter each touch with area+intensity rule + 1.5% bg-keep.
    """
    import random
    rng = random.Random(0)
    A_MIN = 40
    I_MIN = 15           # stricter I to match RTM (sim imprints also vary in strength)
    BG_KEEP_RATE = 0.015
    pq_files = sorted(glob(f"{root}/data/*.parquet"))
    for p in pq_files:
        fn = os.path.basename(p).lower()
        if "test" in fn: split = "test"
        elif "val" in fn: split = "val"
        else: split = "train"
        pf = pq.ParquetFile(p)
        for batch in pf.iter_batches(batch_size=8):
            cols = batch.to_pydict()
            n = len(cols.get("label", cols.get("id", [])))
            for i in range(n):
                round_id = cols.get("id", [None]*n)[i]
                label = cols.get("label", [None]*n)[i]
                obj_id = cols.get("object_id", [None]*n)[i]
                images = cols["sensor_image"][i] or []
                # Decode all touches once
                decoded = []  # list of (rgb, gray_center, tj)
                for tj, img_struct in enumerate(images):
                    if not img_struct: continue
                    img_bytes = img_struct.get("bytes") if isinstance(img_struct, dict) else None
                    if not img_bytes: continue
                    try:
                        rgb = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
                    except Exception:
                        continue
                    g = rgb.mean(axis=2).astype(np.float32)
                    h, w = g.shape
                    decoded.append((rgb, g[h//4:3*h//4, w//4:3*w//4], tj))
                if len(decoded) < 5: continue
                # Cross-touch median baseline (each pixel rarely under contact)
                baseline = np.median(np.stack([d[1] for d in decoded]), axis=0)
                for rgb, g_center, tj in decoded:
                    diff = np.abs(g_center - baseline)
                    mask = diff > PIXEL_THRESH
                    area = int(mask.sum())
                    inten = float(diff[mask].mean()) if area > 0 else 0.0
                    passes = (area >= A_MIN) and (inten >= I_MIN)
                    if not passes and rng.random() >= BG_KEEP_RATE:
                        continue
                    meta = {
                        "source": source_tag,
                        "markered": False,
                        "domain": "sim",
                        "capture": f"r{round_id}_t{tj}",
                        "split": split,
                        "episode": str(obj_id) if obj_id is not None else None,
                        "frame_idx": tj,
                    }
                    meta.update(label_to_obj_name(label))
                    yield rgb, meta


def iter_sim_tactile_mnist():
    """Sim Tactile MNIST seq-320x240 (Taxim Mini-calibrated).
    Each row = one digit object × 32 touches at different positions.
    Filter each touch with area+intensity (A≥40, I≥15) against a per-row
    cross-touch median baseline; keep 1.5% of failed touches as background.
    """
    return _iter_sim_parquet_filtered(
        f"{BASE_DATA}/markerless/SimTactileMNIST",
        "sim_tactile_mnist",
        lambda lbl: {"obj_name": f"digit_{lbl}",
                     "digit_class": int(lbl) if lbl is not None else None},
    )


def iter_sim_starstruck():
    """Sim Star-Struck (Taxim Mini-calibrated). Same recipe as sim_tactile_mnist
    but objects are star-shapes."""
    return _iter_sim_parquet_filtered(
        f"{BASE_DATA}/markerless/SimStarStruck",
        "sim_starstruck",
        lambda lbl: {"obj_name": "starstruck"},
    )


def iter_tacquad_mini():
    """TacQuad: extract only the `gelsight/` subfolder (the GelSight Mini
    sensor data) from each object. Source images are stored as 120×160 PNGs
    in PORTRAIT orientation; we rotate 90° clockwise to landscape (160×120,
    which is a half-size Mini-native aspect ratio).

    Layout after extraction:
      tacquad_extracted/<scene>/<object>/gelsight/<idx>.png

    Baseline strategy: global cross-object median (touches at different
    object/positions ⇒ contact rare per-pixel). Apply unified A=40, I=15 + 1.5% bg.
    """
    import random
    rng = random.Random(0)
    A_MIN, I_MIN = 40, 10
    BG = 0.015
    root = f"{BASE_DATA}/multi_sensor/TacQuad/tacquad_extracted"
    gelsight_dirs = sorted(glob(f"{root}/**/gelsight", recursive=True))
    if not gelsight_dirs: return
    # Build global baseline: sample 3 images from each scene/object
    grays = []
    for d in gelsight_dirs:
        files = sorted(glob(f"{d}/*.png"))
        if not files: continue
        for fn in rng.sample(files, min(3, len(files))):
            try:
                rgb = np.array(Image.open(fn).convert("RGB"))
                # Rotate 90° clockwise: (120,160) portrait → (160,120) landscape
                rgb = np.rot90(rgb, k=-1)
                g = rgb.mean(axis=2).astype(np.float32)
                h, w = g.shape
                grays.append(g[h//4:3*h//4, w//4:3*w//4])
            except Exception:
                pass
    if len(grays) < 30: return
    baseline = np.median(np.stack(grays), axis=0)
    # Iterate all and apply filter
    for d in gelsight_dirs:
        parts = d.replace(root, "").strip("/").split("/")
        scene = parts[0] if len(parts) > 1 else "scene"
        obj_name = parts[-2] if len(parts) > 1 else "obj"
        for fp in sorted(glob(f"{d}/*.png")):
            try:
                rgb = np.array(Image.open(fp).convert("RGB"))
                rgb = np.rot90(rgb, k=-1)  # to landscape
            except Exception:
                continue
            g = rgb.mean(axis=2).astype(np.float32)
            h, w = g.shape
            g_center = g[h//4:3*h//4, w//4:3*w//4]
            diff = np.abs(g_center - baseline)
            mask = diff > PIXEL_THRESH
            area = int(mask.sum())
            inten = float(diff[mask].mean()) if area > 0 else 0.0
            passes = (area >= A_MIN) and (inten >= I_MIN)
            if not passes and rng.random() >= BG: continue
            try:
                idx = int(os.path.splitext(os.path.basename(fp))[0])
            except Exception:
                idx = None
            yield rgb, {
                "source": "tacquad_mini",
                "markered": False,
                "domain": "real",
                "capture": f"{scene}/{obj_name}",
                "split": scene,  # data_indoor / data_outdoor / data_fine_grained
                "obj_name": obj_name,
                "episode": obj_name,
                "frame_idx": idx,
            }


def iter_threedcal():
    """py3DCal: sphere-indentation grid, loose PNG per (x,y,z) position.

    Baseline strategy: use the explicit **`blank.png`** reference shipped
    with the release as the gel-at-rest baseline. Then apply unified
    A=40, I=15 + 1.5% bg-keep filter on every probe image.
    """
    import random, re as _re
    rng = random.Random(0)
    A_MIN, I_MIN = 40, 10
    BG = 0.015
    root = f"{BASE_DATA}/markerless/3DCal/gsmini_calibration_data"
    blank_path = f"{root}/blank_images/blank.png"
    probe_dir = f"{root}/probe_images"
    if not os.path.isfile(blank_path):
        return
    blank = np.asarray(Image.open(blank_path).convert("L"), dtype=np.float32)
    bh, bw = blank.shape
    baseline = blank[bh//4:3*bh//4, bw//4:3*bw//4]
    name_re = _re.compile(r"(\d+)_X([0-9.]+)Y([0-9.]+)Z([0-9.]+)\.png")
    for fp in sorted(glob(f"{probe_dir}/*.png")):
        m = name_re.search(os.path.basename(fp))
        if not m: continue
        idx, x_mm, y_mm, z_mm = m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4))
        try:
            rgb = np.array(Image.open(fp).convert("RGB"))
        except Exception:
            continue
        g = rgb.mean(axis=2).astype(np.float32)
        h, w = g.shape
        g_center = g[h//4:3*h//4, w//4:3*w//4]
        diff = np.abs(g_center - baseline)
        mask = diff > PIXEL_THRESH
        area = int(mask.sum())
        inten = float(diff[mask].mean()) if area > 0 else 0.0
        passes = (area >= A_MIN) and (inten >= I_MIN)
        if not passes and rng.random() >= BG: continue
        yield rgb, {
            "source": "3dcal",
            "markered": False,
            "domain": "real",
            "capture": f"x{x_mm}_y{y_mm}",
            "split": "train",
            "x_mm": x_mm, "y_mm": y_mm, "z_mm": z_mm,
            "frame_idx": int(idx),
        }


def iter_faf_force_estimation():
    """Meta Sparsh / TacBench `facebook/gelsight-force-estimation` →
    a single source for the NC-licensed extension repo.

    Layout: <shape>/<batch>/dataset_gelsight_<NN>.pkl (excluding `org_`),
    each pkl is a list of 5000 image-bytes blobs at 240×320 portrait.
    We rotate 90° to landscape (320×240, Mini native), then apply the
    unified A=40, I=15 filter with a global cross-pkl median baseline.
    """
    import pickle
    import random
    rng = random.Random(0)
    A_MIN, I_MIN = 40, 10
    BG = 0.015
    root = f"{BASE_DATA}/markerless_nc/SparshGelSight"
    shapes = ["sharp", "sphere", "flat"]
    all_pkls = []
    for sh in shapes:
        for p in sorted(glob(f"{root}/{sh}/**/dataset_gelsight_*.pkl", recursive=True)):
            if "org_" in os.path.basename(p): continue
            all_pkls.append((sh, p))
    if not all_pkls: return
    # Global baseline from 3 random images per pkl
    grays = []
    for sh, p in rng.sample(all_pkls, min(30, len(all_pkls))):
        try:
            with open(p, "rb") as f: d = pickle.load(f)
            for b in rng.sample(d, min(3, len(d))):
                im = Image.open(io.BytesIO(b)).convert("RGB")
                rgb = np.rot90(np.array(im), k=-1)   # portrait → landscape
                g = rgb.mean(axis=2).astype(np.float32)
                h, w = g.shape
                grays.append(g[h//4:3*h//4, w//4:3*w//4])
        except Exception:
            pass
    if len(grays) < 30: return
    baseline = np.median(np.stack(grays), axis=0)
    # Iterate all pkl, all images, apply filter
    fi_global = 0
    for sh, p in all_pkls:
        batch_name = os.path.basename(os.path.dirname(p))
        pkl_name = os.path.splitext(os.path.basename(p))[0]
        try:
            with open(p, "rb") as f: d = pickle.load(f)
        except Exception:
            continue
        for i, b in enumerate(d):
            try:
                im = Image.open(io.BytesIO(b)).convert("RGB")
                rgb = np.rot90(np.array(im), k=-1)
            except Exception:
                continue
            g = rgb.mean(axis=2).astype(np.float32)
            h, w = g.shape
            g_center = g[h//4:3*h//4, w//4:3*w//4]
            diff = np.abs(g_center - baseline)
            mask = diff > PIXEL_THRESH
            area = int(mask.sum())
            inten = float(diff[mask].mean()) if area > 0 else 0.0
            passes = (area >= A_MIN) and (inten >= I_MIN)
            if not passes and rng.random() >= BG: continue
            yield rgb, {
                "source": "faf_force_estimation",
                "markered": False,
                "domain": "real",
                "capture": f"{sh}/{batch_name}/{pkl_name}",
                "split": sh,
                "obj_name": sh,
                "indenter": sh,
                "episode": batch_name,
                "frame_idx": i,
            }
            fi_global += 1


SOURCE_ITERS = {
    "gelslam":              iter_gelslam,
    "tactile_tracking":     iter_tactile_tracking,
    "real_tactile_mnist":   iter_real_tactile_mnist,
    "feelanyforce":         iter_feelanyforce,
    "sim_tactile_mnist":    iter_sim_tactile_mnist,
    "sim_starstruck":       iter_sim_starstruck,
    "tacquad_mini":         iter_tacquad_mini,
    "threedcal":            iter_threedcal,
    "faf_force_estimation": iter_faf_force_estimation,
}

# Per-source overrides for the validity filter.
# FAF: data is pre-curated, every frame is an indentation moment, baseline median
#      includes contact frames -> filter must be disabled.
# RTM: we already pick 1 middle frame per video, every frame is peak contact ->
#      filter unnecessary.
# Video sources: keep filter active (baseline = median of first 10 frames
#      typically captures the pre-contact prologue).
SKIP_EMPTY_FILTER = {
    # Sources that filter inside their iterator with a custom baseline strategy
    "real_tactile_mnist": True,   # peak-frame + area/intensity inside touch window
    "sim_tactile_mnist":  True,   # cross-touch median baseline (32 touches per row)
    "sim_starstruck":     True,   # cross-touch median baseline
    "feelanyforce":       True,   # cross-image median per object folder (in iterator)
    "threedcal":          True,   # global cross-image median (in iterator)
    "tacquad_mini":       True,
    "faf_force_estimation": True,  # global cross-pkl median (in iterator)
    # Sources that filter via process() with first-10-frames-of-capture baseline
    "gelslam":            False,
    "tactile_tracking":   False,
}


# ─────────────────────────────────────────────────────────────────
# Probe pass: measure dynamism + empty fraction per source.
# Samples N frames per capture and computes |frame - capture_baseline|.
# ─────────────────────────────────────────────────────────────────

def probe(sub: str, n_per_capture=30, max_total=2000):
    print(f"probe {sub}...", flush=True)
    by_cap = defaultdict(list)
    total = 0
    for fr, meta in SOURCE_ITERS[sub]():
        c = meta["capture"]
        if len(by_cap[c]) < n_per_capture:
            by_cap[c].append(fr)
            total += 1
        if total >= max_total:
            break

    # Per-capture baseline = median over the sampled frames
    deltas, dynamisms = [], []
    for c, frames in by_cap.items():
        if len(frames) < 3:
            continue
        stack = np.stack([grey_center(f) for f in frames], axis=0)
        baseline = np.median(stack, axis=0)
        deformation = np.abs(stack - baseline).mean(axis=(1, 2))
        deltas.extend(deformation.tolist())
        # dynamism = mean inter-frame |Δ|
        diffs = np.abs(stack[1:] - stack[:-1]).mean(axis=(1, 2))
        dynamisms.extend(diffs.tolist())

    deltas = np.array(deltas)
    dyn = np.array(dynamisms) if dynamisms else np.array([0.0])
    empty_frac = float((deltas < 4.0).mean()) if len(deltas) else 0.0   # 4 grey-levels heuristic
    return {
        "n_probed_frames":   total,
        "n_probed_captures": len(by_cap),
        "mean_dynamism":     float(dyn.mean()),
        "median_dynamism":   float(np.median(dyn)),
        "mean_deformation":  float(deltas.mean()) if len(deltas) else 0.0,
        "empty_frac":        empty_frac,
        "n_captures_with_3plus_frames": int(sum(1 for c, fs in by_cap.items() if len(fs) >= 3)),
    }


# ─────────────────────────────────────────────────────────────────
# Two-pass full processing:
#   pass 1: scan ALL frames; per capture, store baseline + count valid + collect phashes
#   pass 2: re-iterate, apply stride+validity+dedupe; write parquet
# To avoid two full scans (RTM is large), we do a single streaming pass:
#   - maintain a per-capture rolling baseline from first 10 frames
#   - then for subsequent frames in same capture, compute validity online
#   - apply stride based on a target_keep_per_capture pre-computed at probe time
# ─────────────────────────────────────────────────────────────────

class ShardWriter:
    def __init__(self, out_dir, prefix):
        self.out_dir = out_dir
        self.prefix = prefix
        os.makedirs(out_dir, exist_ok=True)
        self.rows = []
        self.shard_idx = 0
        self.total = 0
        self.bytes_in = 0

    def add(self, row):
        # ensure all keys present
        row = {f.name: row.get(f.name) for f in SCHEMA}
        self.rows.append(row)
        self.bytes_in += len(row["image"]) if row["image"] else 0
        self.total += 1
        if self.bytes_in >= SHARD_TGT:
            self._flush()

    def _flush(self):
        if not self.rows: return
        # Build columns
        cols = {f.name: [r[f.name] for r in self.rows] for f in SCHEMA}
        t = pa.Table.from_pydict(cols, schema=SCHEMA)
        path = f"{self.out_dir}/{self.prefix}-{self.shard_idx:05d}.parquet"
        pq.write_table(t, path, compression="snappy")
        print(f"  wrote {path}  rows={len(self.rows)}  bytes={self.bytes_in/1e9:.2f}GB",
              flush=True)
        self.shard_idx += 1
        self.rows = []
        self.bytes_in = 0

    def close(self):
        self._flush()
        # rename shards to "PREFIX-NNNNN-of-NNNNN.parquet"
        files = sorted(glob(f"{self.out_dir}/{self.prefix}-?????.parquet"))
        total_shards = len(files)
        for i, fp in enumerate(files):
            base = os.path.dirname(fp)
            new = f"{base}/{self.prefix}-{i:05d}-of-{total_shards:05d}.parquet"
            if fp != new:
                os.rename(fp, new)


def process(sub: str, probe_info: dict):
    """Run pipeline on one source and write parquet shards."""
    print(f"\n=== processing {sub} ===", flush=True)

    # Decide global stride
    dyn = probe_info["mean_dynamism"]
    # Estimate effective contact frames after empty filter
    # (We'll re-tune as we go since we don't know R_total precisely)
    # Target = BUDGET. We adjust the stride live by checking running fill rate.
    target = BUDGET

    # Group by split, write to <sub>/<split>-####.parquet
    base = BASE_OUT_NC if sub in NC_SOURCES else BASE_OUT
    out_dir = f"{base}/{sub}"
    writers: dict[str, ShardWriter] = {}

    n_seen = 0
    n_empty_dropped = 0
    n_dup_dropped = 0
    n_stride_dropped = 0
    n_kept = 0
    n_empty_passed = 0
    cur_capture = None
    cap_seen_within = 0
    cap_baseline = None
    cap_buffer: list[np.ndarray] = []
    cap_phashes: list[int] = []

    # We compute capture-baseline as the median of the first 10 frames seen
    BASE_FRAMES = 10

    def finish_capture():
        # nothing to do per se; arrays cleared on capture change
        pass

    t0 = time.time()
    stride_state = {"stride": 1.0, "accum": 0.0}

    for fr, meta in SOURCE_ITERS[sub]():
        cap = meta["capture"]
        if cap != cur_capture:
            finish_capture()
            cur_capture = cap
            cap_seen_within = 0
            cap_baseline = None
            cap_buffer = []
            cap_phashes = []

        g_center = grey_center(fr)
        skip_empty = SKIP_EMPTY_FILTER.get(sub, False)

        # Build baseline from first BASE_FRAMES (only when validity filter is active)
        if not skip_empty and cap_baseline is None:
            cap_buffer.append(g_center)
            cap_seen_within += 1
            n_seen += 1
            if len(cap_buffer) >= BASE_FRAMES:
                cap_baseline = np.median(np.stack(cap_buffer, axis=0), axis=0)
            continue

        n_seen += 1
        if skip_empty:
            is_empty = False
        else:
            # Unified area + intensity validity rule
            pixel_diff = np.abs(g_center - cap_baseline)
            mask = pixel_diff > PIXEL_THRESH
            contact_area = int(mask.sum())
            contact_intensity = float(pixel_diff[mask].mean()) if contact_area > 0 else 0.0
            thresh = VALIDITY_THRESH.get(sub, dict(A_min=A_MIN_DEFAULT, I_min=I_MIN_DEFAULT))
            is_empty = (contact_area < thresh["A_min"]) \
                       or (contact_intensity < thresh["I_min"])

        # Stride decision (uniform stride based on target/total estimate later)
        # We use a live rate-limiter: every K frames, keep 1 (K adjusted live)
        stride_state["accum"] += 1.0
        if stride_state["accum"] < stride_state["stride"]:
            n_stride_dropped += 1
            continue
        stride_state["accum"] -= stride_state["stride"]

        # Empty-budget: allow up to EMPTY_BUDGET fraction of kept frames to be empty
        if is_empty:
            # Allow only if we're under budget
            if n_empty_passed >= EMPTY_BUDGET * max(n_kept, 1):
                n_empty_dropped += 1
                continue

        # Dedupe via phash within capture
        h = phash(fr)
        is_dup = any(hamming(h, hh) <= PHASH_DIST for hh in cap_phashes[-30:])
        if is_dup:
            n_dup_dropped += 1
            continue
        cap_phashes.append(h)

        # Keep!
        img_bytes = encode_jpeg(fr)
        row = dict(meta)
        row["image"] = img_bytes
        row["image_format"] = "jpeg"
        row["height"] = int(fr.shape[0])
        row["width"] = int(fr.shape[1])
        if is_empty:
            n_empty_passed += 1
        n_kept += 1

        split = row.get("split", "train") or "train"
        if split not in writers:
            writers[split] = ShardWriter(out_dir, split)
        writers[split].add(row)

        # Adapt stride live: aim for target frames over the source.
        # We don't know R_total, but we tweak so that fill rate is sensible.
        # If n_kept exceeds BUDGET, raise stride aggressively.
        if n_kept > 0 and n_kept % 5000 == 0:
            if n_kept > BUDGET:
                stride_state["stride"] = max(1.0, stride_state["stride"] * 1.5)
            elif n_kept > BUDGET * 0.95:
                stride_state["stride"] = max(1.0, stride_state["stride"] * 1.2)

        # Hard cap
        if n_kept >= BUDGET:
            print(f"  reached BUDGET={BUDGET}, stopping iteration", flush=True)
            break

        if n_seen % 20000 == 0:
            dt = time.time() - t0
            print(f"  seen={n_seen:,}  kept={n_kept:,}  "
                  f"empty_drop={n_empty_dropped:,}  dup_drop={n_dup_dropped:,}  "
                  f"stride_drop={n_stride_dropped:,}  "
                  f"({n_seen/max(dt,0.01):.0f} fps)", flush=True)

    for w in writers.values():
        w.close()

    stats = {
        "source":           sub,
        "n_seen":           n_seen,
        "n_kept":           n_kept,
        "n_empty_dropped":  n_empty_dropped,
        "n_empty_passed":   n_empty_passed,
        "n_dup_dropped":    n_dup_dropped,
        "n_stride_dropped": n_stride_dropped,
        "splits":           {k: w.total for k, w in writers.items()},
        "wall_time_sec":    time.time() - t0,
    }
    with open(f"/home/yxma/MultimodalData/stats_v2_{sub}.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  {sub} stats: {stats}", flush=True)
    return stats


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def cmd_probe(sub):
    info = probe(sub)
    out = f"/home/yxma/MultimodalData/probe_{sub}.json"
    with open(out, "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))
    print(f"saved {out}")


def cmd_process(sub):
    probe_file = f"/home/yxma/MultimodalData/probe_{sub}.json"
    if os.path.exists(probe_file):
        info = json.load(open(probe_file))
    else:
        info = probe(sub)
        with open(probe_file, "w") as f:
            json.dump(info, f, indent=2)
    process(sub, info)


def cmd_stats():
    totals = {}
    for sub in os.listdir(BASE_OUT):
        p = f"{BASE_OUT}/{sub}"
        if not os.path.isdir(p): continue
        paths = sorted(glob(f"{p}/*.parquet"))
        n = sum(pq.read_metadata(x).num_rows for x in paths)
        bytes_total = sum(os.path.getsize(x) for x in paths)
        totals[sub] = {"rows": n, "bytes": bytes_total, "shards": len(paths)}
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "probe":
        cmd_probe(sys.argv[2])
    elif cmd == "process":
        cmd_process(sys.argv[2])
    elif cmd == "stats":
        cmd_stats()
    else:
        print(f"unknown command: {cmd}"); sys.exit(1)
