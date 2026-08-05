#!/usr/bin/env python3
"""Convert mini_data sources into unified parquet shards for HuggingFace.

Layout output:
  /media/yxma/Disk1/yuxiang/mini_data_parquet/
    fota_labeled/
      train-00000-of-NN.parquet
      val-00000-of-NN.parquet
    fota_unlabeled/
      train-...
      val-...
    threedcal/
      train-...
    feats/
      train-... val-... test-...

Schema (one row per image, unified across sources):
  image:           binary      raw JPEG/PNG bytes
  image_format:    string      "jpeg" or "png"
  source:          string      "fota_labeled" | "fota_unlabeled" | "3dcal" | "feats"
  markered:        bool        gel has tracking dots?
  capture:         string      capture/scene/object identifier
  split:           string      "train" | "val" | "test" | "test_diff_sensor_new_gel" | ...
  height:          int32
  width:           int32
  # Optional metadata (null where N/A):
  obj_name:        string      FoTA object name
  init_pose:       int32       FoTA pose index
  side:            string      "left" | "right" (FoTA)
  x_mm:            float32
  y_mm:            float32
  z_mm:            float32
  quat_x,y,z,w:    float32
  indenter:        string      FEATS indenter shape
  indenter_param:  string      FEATS indenter size
  f_x, f_y, f_z:   float32     FEATS force
  grid_z_max:      float32     FEATS deepest indentation
  grid_z_mean:     float32

Run with one source at a time, e.g.:
  python make_parquet.py fota_labeled
  python make_parquet.py fota_unlabeled
  python make_parquet.py 3dcal
  python make_parquet.py feats
"""

import csv
import io
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

BASE = "/media/yxma/Disk1/yuxiang/mini_data"
OUT_BASE = "/media/yxma/Disk1/yuxiang/mini_data_parquet"

SHARD_TARGET_BYTES = 2 * 1024 * 1024 * 1024  # ~2 GB per shard

# Unified schema
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
])

CAPTURE_RE = re.compile(
    r"^(?P<obj>.+)_Mini_Mini_init_pose_(?P<pose>\d+)_"
    r"(?P<status>labeled|unlabeled)_(?P<side>left|right)$"
)

# Known FEATS indenter shapes
FEATS_SHAPES = {"sphere", "cuboid", "cylinder", "cone", "hemisphere", "torus",
                "prism", "pyramid", "cross", "ellipsoid", "flat"}


def parse_fota_capture(name: str):
    m = CAPTURE_RE.match(name)
    if not m:
        return None
    return {
        "obj_name": m.group("obj"),
        "init_pose": int(m.group("pose")),
        "side": m.group("side"),
    }


def parse_feats_stem(stem: str):
    """e.g. '113_cuboid_12' -> ('cuboid', '12'); fallback ('unknown', '')."""
    tokens = stem.split("_")
    for i, t in enumerate(tokens):
        if t.lower() in FEATS_SHAPES:
            param = tokens[i + 1] if i + 1 < len(tokens) else ""
            return t.lower(), param
    return "unknown", ""


# ============================================================
# Image loading workers (return per-row dict)
# ============================================================

def _load_jpeg_file(args):
    """Open a loose .jpg/.png, return image bytes (re-encoded as JPEG for size) + size."""
    path, base_meta = args
    try:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92, optimize=True)
        img_bytes = buf.getvalue()
        row = dict(base_meta)
        row["image"] = img_bytes
        row["image_format"] = "jpeg"
        row["height"] = h
        row["width"] = w
        return row
    except Exception as e:
        return {"_error": f"{path}: {e}"}


def _load_feats_npy(args):
    """Open a FEATS .npy dict, encode gs_img as JPEG."""
    path, base_meta = args
    try:
        d = np.load(path, allow_pickle=True).item()
        img_arr = d.get("gs_img")
        if img_arr is None or img_arr.dtype != np.uint8 or img_arr.ndim != 3:
            return {"_error": f"{path}: bad gs_img"}
        h, w = img_arr.shape[:2]
        buf = io.BytesIO()
        Image.fromarray(img_arr).save(buf, format="JPEG", quality=92, optimize=True)
        img_bytes = buf.getvalue()
        grid_z = d.get("grid_z")
        row = dict(base_meta)
        row["image"] = img_bytes
        row["image_format"] = "jpeg"
        row["height"] = h
        row["width"] = w
        row["f_x"] = float(d.get("f_x", 0.0) or 0.0)
        row["f_y"] = float(d.get("f_y", 0.0) or 0.0)
        row["f_z"] = float(d.get("f_z", 0.0) or 0.0)
        if grid_z is not None:
            row["grid_z_max"] = float(np.max(grid_z))
            row["grid_z_mean"] = float(np.mean(grid_z))
        return row
    except Exception as e:
        return {"_error": f"{path}: {e}"}


# ============================================================
# Shard writer
# ============================================================

class ShardWriter:
    """Buffers rows and flushes to parquet shards capped at ~SHARD_TARGET_BYTES."""

    def __init__(self, out_dir: str, prefix: str):
        self.out_dir = out_dir
        self.prefix = prefix
        os.makedirs(out_dir, exist_ok=True)
        self.shard_idx = 0
        self.rows: list[dict] = []
        self.bytes_in_buffer = 0
        self.total_rows = 0
        self.shard_paths: list[str] = []

    def add(self, row: dict):
        # Skip error rows
        if "_error" in row:
            print(f"  SKIP: {row['_error']}", flush=True)
            return
        self.rows.append(row)
        self.bytes_in_buffer += len(row.get("image", b""))
        if self.bytes_in_buffer >= SHARD_TARGET_BYTES:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        # Pad missing fields with None per schema
        cols = {f.name: [] for f in SCHEMA}
        for r in self.rows:
            for name in cols:
                cols[name].append(r.get(name))
        tbl = pa.Table.from_pydict(cols, schema=SCHEMA)
        path = os.path.join(self.out_dir, f"{self.prefix}-{self.shard_idx:05d}.parquet")
        pq.write_table(tbl, path, compression="snappy")
        size = os.path.getsize(path)
        print(f"  wrote {path} ({len(self.rows)} rows, {size/1e9:.2f} GB)",
              flush=True)
        self.shard_paths.append(path)
        self.total_rows += len(self.rows)
        self.rows = []
        self.bytes_in_buffer = 0
        self.shard_idx += 1

    def finalize(self):
        self.flush()
        # Rename shards to <prefix>-NNNNN-of-MMMMM.parquet
        m = len(self.shard_paths)
        if m == 0:
            return
        for old in self.shard_paths:
            base = os.path.basename(old)
            # base was "<prefix>-NNNNN.parquet"
            n = int(base.split("-")[1].split(".")[0])
            new = os.path.join(self.out_dir,
                               f"{self.prefix}-{n:05d}-of-{m:05d}.parquet")
            os.rename(old, new)


# ============================================================
# Per-source job collection
# ============================================================

def fota_jobs(labeled: bool):
    """Return list[(path, base_meta), ...] grouped by split."""
    src_dir = os.path.join(
        BASE, "markerless",
        "FoTa_labeled" if labeled else "FoTa_unlabeled",
    )
    captures = sorted(d for d in os.listdir(src_dir)
                      if d.endswith("_left") or d.endswith("_right"))
    # Build pose lookup for labeled subset
    pose_lookup: dict[tuple[str, str], dict] = {}
    if labeled:
        for cap in captures:
            cap_dir = os.path.join(src_dir, cap)
            csv_path = os.path.join(cap_dir, "poses.csv")
            if not os.path.isfile(csv_path):
                continue
            with open(csv_path) as f:
                rdr = csv.DictReader(f)
                for r in rdr:
                    key = (cap, r["filename"])
                    pose_lookup[key] = r
    jobs_by_split: dict[str, list] = {"train": [], "val": []}
    for cap in captures:
        info = parse_fota_capture(cap)
        if info is None:
            continue
        for split in ("train", "val"):
            split_dir = os.path.join(src_dir, cap, split)
            if not os.path.isdir(split_dir):
                continue
            for f in sorted(os.listdir(split_dir)):
                if not f.endswith(".jpg"):
                    continue
                meta = {
                    "source": "fota_labeled" if labeled else "fota_unlabeled",
                    "markered": False,
                    "capture": cap,
                    "split": split,
                    "obj_name": info["obj_name"],
                    "init_pose": info["init_pose"],
                    "side": info["side"],
                }
                if labeled:
                    pose = pose_lookup.get((cap, f))
                    if pose:
                        for k_out, k_in in (
                            ("x_mm", "x_mm"), ("y_mm", "y_mm"), ("z_mm", "z_mm"),
                            ("quat_x", "quat_x"), ("quat_y", "quat_y"),
                            ("quat_z", "quat_z"), ("quat_w", "quat_w"),
                        ):
                            v = pose.get(k_in, "")
                            try:
                                meta[k_out] = float(v) if v != "" else None
                            except ValueError:
                                meta[k_out] = None
                jobs_by_split[split].append(
                    (os.path.join(split_dir, f), meta))
    return jobs_by_split


def threedcal_jobs():
    src_dir = os.path.join(BASE, "markerless", "3DCal", "gsmini_calibration_data")
    # Use probe_images as the main data
    annot_path = os.path.join(src_dir, "annotations", "annotations.csv")
    pose_lookup: dict[str, dict] = {}
    if os.path.isfile(annot_path):
        with open(annot_path) as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                pose_lookup[r["img_name"]] = r
    probe_dir = os.path.join(src_dir, "probe_images")
    jobs = []
    for f in sorted(os.listdir(probe_dir)):
        if not f.endswith(".png"):
            continue
        path = os.path.join(probe_dir, f)
        meta = {
            "source": "3dcal",
            "markered": False,
            "capture": os.path.splitext(f)[0],
            "split": "train",
        }
        pose = pose_lookup.get(f)
        if pose:
            try:
                meta["x_mm"] = float(pose.get("x_mm") or 0.0)
                meta["y_mm"] = float(pose.get("y_mm") or 0.0)
                meta["z_mm"] = float(pose.get("penetration_depth_mm") or 0.0)
            except ValueError:
                pass
        jobs.append((path, meta))
    return {"train": jobs}


def feats_jobs():
    src_dir = os.path.join(BASE, "markered", "FEATS", "v24_labels_24_32")
    splits = ["train", "val", "test", "test_diff_sensor_new_gel",
              "test_diff_sensor_old_gel", "test_unknown_indenters"]
    jobs_by_split: dict[str, list] = {}
    for split in splits:
        d = os.path.join(src_dir, split)
        if not os.path.isdir(d):
            continue
        ls = []
        for root, _, files in os.walk(d):
            for f in files:
                if not f.endswith(".npy"):
                    continue
                stem = os.path.splitext(f)[0]
                shape, param = parse_feats_stem(stem)
                meta = {
                    "source": "feats",
                    "markered": True,
                    "capture": stem,
                    "split": split,
                    "indenter": shape,
                    "indenter_param": param,
                }
                ls.append((os.path.join(root, f), meta))
        ls.sort()
        if ls:
            jobs_by_split[split] = ls
    return jobs_by_split


# ============================================================
# Main runner
# ============================================================

def run_source(source: str):
    if source == "fota_labeled":
        jobs_by_split = fota_jobs(labeled=True)
        worker = _load_jpeg_file
        out_dir = os.path.join(OUT_BASE, "fota_labeled")
    elif source == "fota_unlabeled":
        jobs_by_split = fota_jobs(labeled=False)
        worker = _load_jpeg_file
        out_dir = os.path.join(OUT_BASE, "fota_unlabeled")
    elif source in ("3dcal", "threedcal"):
        jobs_by_split = threedcal_jobs()
        worker = _load_jpeg_file
        out_dir = os.path.join(OUT_BASE, "threedcal")
    elif source == "feats":
        jobs_by_split = feats_jobs()
        worker = _load_feats_npy
        out_dir = os.path.join(OUT_BASE, "feats")
    else:
        raise ValueError(f"unknown source: {source}")

    print(f"=== {source} ===", flush=True)
    for split, jobs in jobs_by_split.items():
        print(f"  {split}: {len(jobs)} files", flush=True)

    workers = max(2, min(8, (os.cpu_count() or 4)))
    print(f"workers={workers}", flush=True)

    for split, jobs in jobs_by_split.items():
        if not jobs:
            continue
        writer = ShardWriter(out_dir, split)
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            # imap-like over chunks to keep memory bounded
            CHUNK = 256
            done = 0
            for i in range(0, len(jobs), CHUNK):
                batch = jobs[i:i + CHUNK]
                futs = [ex.submit(worker, j) for j in batch]
                for fut in as_completed(futs):
                    writer.add(fut.result())
                done += len(batch)
                if done % 4096 == 0 or done == len(jobs):
                    elapsed = time.time() - t0
                    rate = done / max(elapsed, 1e-6)
                    eta = (len(jobs) - done) / max(rate, 1e-6)
                    print(f"  [{split}] {done}/{len(jobs)}  "
                          f"({rate:.0f}/s, eta {eta:.0f}s)", flush=True)
        writer.finalize()
        print(f"  {split} done: {writer.total_rows} rows in "
              f"{writer.shard_idx} shards, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: make_parquet.py {fota_labeled|fota_unlabeled|3dcal|feats}")
        sys.exit(1)
    for src in sys.argv[1:]:
        run_source(src)
