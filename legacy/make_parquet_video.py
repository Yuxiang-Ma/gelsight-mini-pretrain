#!/usr/bin/env python3
"""Video-preserving variant of make_parquet_v2.py.

Same validity filter (area + intensity), but **no phash dedupe** — the small
inter-frame changes that the main-repo pipeline drops as redundant are the
temporal signal we want to learn here. Adds sequence-level metadata.

Outputs to /media/yxma/Disk1/yuxiang/mini_data_parquet_video/<source>/

Usage:
  python make_parquet_video.py process gelslam
  python make_parquet_video.py process tactile_tracking
  python make_parquet_video.py process fota_unlabeled   # reads from raw
  python make_parquet_video.py process real_tactile_mnist
"""
import io, json, os, sys, time
from collections import defaultdict
from glob import glob

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, "/home/yxma/MultimodalData")
from make_parquet_v2 import (
    SCHEMA as BASE_SCHEMA, BASE_DATA, encode_jpeg, grey_center,
    PIXEL_THRESH, A_MIN_DEFAULT, I_MIN_DEFAULT,
)

BASE_OUT_VIDEO = "/media/yxma/Disk1/yuxiang/mini_data_parquet_video"
SHARD_TGT = 2 * 1024 ** 3
A_MIN = A_MIN_DEFAULT
I_MIN = I_MIN_DEFAULT
RTM_I_MIN = 15   # RTM uses stricter I to keep only visibly-imprinted touches
BG_RATE = 0.015
FPS_DEFAULT = 25.0

# Extended schema: existing + sequence columns
VIDEO_SCHEMA = pa.schema(list(BASE_SCHEMA) + [
    ("sequence_id", pa.string()),
    ("frame_in_seq", pa.int32()),
    ("sequence_length", pa.int32()),
    ("fps", pa.float32()),
])


class ShardWriter:
    def __init__(self, out_dir, prefix):
        self.out_dir = out_dir; self.prefix = prefix
        os.makedirs(out_dir, exist_ok=True)
        self.rows = []; self.shard_idx = 0; self.bytes_in = 0
        self.schema = VIDEO_SCHEMA
    def add(self, row):
        row = {f.name: row.get(f.name) for f in self.schema}
        self.rows.append(row)
        self.bytes_in += len(row["image"]) if row["image"] else 0
        if self.bytes_in >= SHARD_TGT: self._flush()
    def _flush(self):
        if not self.rows: return
        cols = {f.name: [r[f.name] for r in self.rows] for f in self.schema}
        t = pa.Table.from_pydict(cols, schema=self.schema)
        path = f"{self.out_dir}/{self.prefix}-{self.shard_idx:05d}.parquet"
        pq.write_table(t, path, compression="snappy")
        print(f"  wrote {os.path.basename(path)} rows={len(self.rows)}", flush=True)
        self.shard_idx += 1; self.rows = []; self.bytes_in = 0
    def close(self):
        self._flush()
        files = sorted(glob(f"{self.out_dir}/{self.prefix}-?????.parquet"))
        n = len(files)
        for i, fp in enumerate(files):
            new = f"{self.out_dir}/{self.prefix}-{i:05d}-of-{n:05d}.parquet"
            if fp != new: os.rename(fp, new)


def filter_seq(frames_rgb, source_tag, sub):
    """Apply validity filter to one sequence. Returns list of (frame, in_seq_idx)
    pairs for the kept frames, preserving order."""
    import random
    rng = random.Random()
    if len(frames_rgb) < 10: return []
    grays = [grey_center(np.asarray(f, dtype=np.float32)) for f in frames_rgb]
    baseline = np.median(np.stack(grays[:10]), axis=0)
    kept = []
    for i, (rgb, g) in enumerate(zip(frames_rgb, grays)):
        diff = np.abs(g - baseline)
        mask = diff > PIXEL_THRESH
        area = int(mask.sum())
        inten = float(diff[mask].mean()) if area > 0 else 0.0
        passes = (area >= A_MIN) and (inten >= I_MIN)
        if not passes and rng.random() >= BG_RATE: continue
        kept.append((rgb, i))
    return kept


def iter_gelslam_video():
    import cv2
    root = f"{BASE_DATA}/markerless/GelSLAM"
    for sub in ("tracking_dataset", "reconstruction_dataset"):
        for vid in glob(f"{root}/{sub}/*/gelsight.avi"):
            episode = os.path.basename(os.path.dirname(vid))
            split = "train" if sub == "tracking_dataset" else "recon"
            cap = cv2.VideoCapture(vid)
            fps = cap.get(cv2.CAP_PROP_FPS) or FPS_DEFAULT
            frames = []
            while True:
                ok, fr = cap.read()
                if not ok: break
                frames.append(fr[:, :, ::-1])
            cap.release()
            kept = filter_seq(frames, "gelslam", sub)
            seq_id = f"{sub}/{episode}"
            for new_idx, (rgb, orig_idx) in enumerate(kept):
                yield rgb, {
                    "source": "gelslam",
                    "markered": False,
                    "domain": "real",
                    "capture": seq_id,
                    "split": split,
                    "obj_name": episode,
                    "episode": episode,
                    "frame_idx": orig_idx,
                    "sequence_id": seq_id,
                    "frame_in_seq": new_idx,
                    "sequence_length": len(kept),
                    "fps": fps,
                }


def iter_tactile_tracking_video():
    import cv2, re
    root = f"{BASE_DATA}/markerless/TactileTracking/normalflow_dataset"
    obj_re = re.compile(r"^([a-zA-Z]+)(\d+)$")
    for vid in sorted(glob(f"{root}/*/gelsight.avi")):
        trial_dir = os.path.basename(os.path.dirname(vid))
        m = obj_re.match(trial_dir)
        obj, trial = (m.group(1), m.group(2)) if m else (trial_dir, "0")
        cap = cv2.VideoCapture(vid)
        fps = cap.get(cv2.CAP_PROP_FPS) or FPS_DEFAULT
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok: break
            frames.append(fr[:, :, ::-1])
        cap.release()
        kept = filter_seq(frames, "tactile_tracking", "")
        seq_id = trial_dir
        for new_idx, (rgb, orig_idx) in enumerate(kept):
            yield rgb, {
                "source": "tactile_tracking",
                "markered": False,
                "domain": "real",
                "capture": seq_id, "split": "train",
                "obj_name": obj, "episode": trial,
                "frame_idx": orig_idx,
                "sequence_id": seq_id,
                "frame_in_seq": new_idx,
                "sequence_length": len(kept),
                "fps": fps,
            }


def iter_real_tactile_mnist_video():
    """For each upstream touch video, decode all frames, find touch window via
    timestamps, validity-filter within window, keep ALL passing frames as a
    short sequence (typically 2-6 frames per touch).
    """
    import cv2
    import random
    rng = random.Random(0)
    root = f"{BASE_DATA}/markerless/RealTactileMNIST"
    for p in sorted(glob(f"{root}/data/*.parquet")):
        fn = os.path.basename(p).lower()
        split = "test" if "test" in fn else "train"
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
                for tj, vs in enumerate(videos):
                    if not vs: continue
                    vb = vs.get("bytes") if isinstance(vs, dict) else None
                    if not vb: continue
                    tmpf = f"/tmp/_rtmv_{os.getpid()}.mp4"
                    with open(tmpf, "wb") as f: f.write(vb)
                    cap = cv2.VideoCapture(tmpf)
                    fps = cap.get(cv2.CAP_PROP_FPS) or FPS_DEFAULT
                    frames, grays = [], []
                    while True:
                        ok, fr = cap.read()
                        if not ok: break
                        rgb = fr[:, :, ::-1]; frames.append(rgb)
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
                        if ts and ts0 and ts1 and len(ts) == len(frames):
                            in_window = [k for k, t in enumerate(ts) if ts0 <= t <= ts1]
                            if not in_window: in_window = list(range(len(frames)))
                    except Exception: pass
                    # Within touch window, keep all that pass validity
                    kept_within = []
                    for k in in_window:
                        diff = np.abs(grays[k] - baseline)
                        mask = diff > PIXEL_THRESH
                        area = int(mask.sum())
                        inten = float(diff[mask].mean()) if area > 0 else 0.0
                        passes = (area >= A_MIN) and (inten >= RTM_I_MIN)
                        if not passes and rng.random() >= BG_RATE: continue
                        kept_within.append(k)
                    seq_id = f"r{round_id}_t{tj}"
                    for new_idx, k in enumerate(kept_within):
                        yield frames[k], {
                            "source": "real_tactile_mnist",
                            "markered": False,
                            "domain": "real",
                            "capture": seq_id, "split": split,
                            "obj_name": f"digit_{label}",
                            "digit_class": int(label) if label is not None else None,
                            "episode": str(obj_id) if obj_id is not None else None,
                            "frame_idx": k,
                            "sequence_id": seq_id,
                            "frame_in_seq": new_idx,
                            "sequence_length": len(kept_within),
                            "fps": fps,
                        }


VIDEO_SOURCE_ITERS = {
    "gelslam":            iter_gelslam_video,
    "tactile_tracking":   iter_tactile_tracking_video,
    "real_tactile_mnist": iter_real_tactile_mnist_video,
}


def process(sub):
    print(f"=== processing {sub} (video mode, no dedupe) ===", flush=True)
    iter_fn = VIDEO_SOURCE_ITERS.get(sub)
    if not iter_fn:
        print(f"no iterator for {sub}"); return
    out_dir = f"{BASE_OUT_VIDEO}/{sub}"
    if os.path.isdir(out_dir):
        for fp in glob(f"{out_dir}/*.parquet"): os.remove(fp)
    writer = ShardWriter(out_dir, "train")
    n_kept = 0; n_seq = set()
    t0 = time.time()
    for rgb, meta in iter_fn():
        rgb = np.asarray(rgb)
        if rgb.ndim != 3: continue
        img_bytes = encode_jpeg(rgb.astype(np.uint8))
        row = dict(meta)
        row["image"] = img_bytes
        row["image_format"] = "jpeg"
        row["height"] = int(rgb.shape[0])
        row["width"] = int(rgb.shape[1])
        writer.add(row)
        n_kept += 1; n_seq.add(meta.get("sequence_id"))
        if n_kept % 5000 == 0:
            print(f"  kept={n_kept:,} sequences={len(n_seq):,} "
                  f"({n_kept/max(time.time()-t0,0.01):.0f} fps)", flush=True)
    writer.close()
    print(f"=== done: kept={n_kept:,} sequences={len(n_seq):,} "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] != "process":
        print(__doc__); sys.exit(1)
    process(sys.argv[2])
