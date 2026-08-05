#!/usr/bin/env python3
"""Reprocess FEATS from raw .npy with relaxed filter:
   - drop |f_z| < 0.4 (was 0.5 in v1)
   - keep ~1.5% of below-threshold frames as background diversity

Reads:  /media/yxma/Disk1/yuxiang/mini_data/markered/FEATS/v24_labels_24_32/<split>/*.npy
Writes: /media/yxma/Disk1/yuxiang/mini_data_parquet/feats/<split>-####-of-####.parquet
"""
import glob, io, os, random, re, time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

RAW = "/media/yxma/Disk1/yuxiang/mini_data/markered/FEATS/v24_labels_24_32"
OUT = "/media/yxma/Disk1/yuxiang/mini_data_parquet/feats"
SHARD_TGT = 2 * 1024 ** 3
FZ_THRESH = 0.4
BG_RATE   = 0.015

SPLITS = ["train", "val", "test", "test_diff_sensor_new_gel",
          "test_diff_sensor_old_gel", "test_unknown_indenters"]
GEL_VARIANT = {
    "train":                       "black_dot",
    "val":                         "black_dot",
    "test":                        "black_dot",
    "test_unknown_indenters":      "black_dot",
    "test_diff_sensor_old_gel":    "black_dot",
    "test_diff_sensor_new_gel":    "different",
}

FEATS_SHAPES = {"sphere","cuboid","cylinder","cone","hemisphere","torus",
                "prism","pyramid","cross","ellipse","ellipsoid","polygon","flat"}

# Schema must match make_parquet_v2.py SCHEMA
import sys
sys.path.insert(0, "/home/yxma/MultimodalData")
from make_parquet_v2 import SCHEMA


def parse_stem(stem):
    tokens = stem.split("_")
    for i, t in enumerate(tokens):
        if t.lower() in FEATS_SHAPES:
            param = tokens[i+1] if i+1 < len(tokens) else ""
            return t.lower(), param
    return "unknown", ""


def encode_jpeg(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=92, optimize=True)
    return buf.getvalue()


class ShardWriter:
    def __init__(self, out_dir, prefix):
        self.out_dir = out_dir; self.prefix = prefix
        os.makedirs(out_dir, exist_ok=True)
        self.rows = []; self.shard_idx = 0; self.bytes_in = 0; self.total = 0

    def add(self, row):
        row = {f.name: row.get(f.name) for f in SCHEMA}
        self.rows.append(row)
        self.bytes_in += len(row["image"]) if row["image"] else 0
        self.total += 1
        if self.bytes_in >= SHARD_TGT:
            self._flush()

    def _flush(self):
        if not self.rows: return
        cols = {f.name: [r[f.name] for r in self.rows] for f in SCHEMA}
        t = pa.Table.from_pydict(cols, schema=SCHEMA)
        path = f"{self.out_dir}/{self.prefix}-{self.shard_idx:05d}.parquet"
        pq.write_table(t, path, compression="snappy")
        print(f"  wrote {path} rows={len(self.rows)}")
        self.shard_idx += 1; self.rows = []; self.bytes_in = 0

    def close(self):
        self._flush()
        files = sorted(glob.glob(f"{self.out_dir}/{self.prefix}-?????.parquet"))
        total = len(files)
        for i, fp in enumerate(files):
            new = f"{self.out_dir}/{self.prefix}-{i:05d}-of-{total:05d}.parquet"
            if fp != new: os.rename(fp, new)


def main():
    # delete any old shards
    for f in glob.glob(f"{OUT}/*.parquet"):
        os.remove(f)
    rng = random.Random(0)
    t0 = time.time()
    stats = {}
    for split in SPLITS:
        split_dir = f"{RAW}/{split}"
        if not os.path.isdir(split_dir): continue
        npy_files = sorted(glob.glob(f"{split_dir}/*.npy"))
        writer = ShardWriter(OUT, split)
        n_kept = n_bg = n_dropped = n_err = 0
        for p in npy_files:
            try:
                d = np.load(p, allow_pickle=True).item()
            except Exception:
                n_err += 1; continue
            img = d.get("gs_img")
            if img is None or img.dtype != np.uint8 or img.ndim != 3:
                n_err += 1; continue
            fx = float(d.get("f_x", 0.0) or 0.0)
            fy = float(d.get("f_y", 0.0) or 0.0)
            fz = float(d.get("f_z", 0.0) or 0.0)
            grid_z = d.get("grid_z")
            passes = abs(fz) >= FZ_THRESH
            if not passes:
                # keep with probability BG_RATE
                if rng.random() >= BG_RATE:
                    n_dropped += 1; continue
                n_bg += 1
            else:
                n_kept += 1
            stem = os.path.splitext(os.path.basename(p))[0]
            indenter, param = parse_stem(stem)
            row = {
                "image": encode_jpeg(img),
                "image_format": "jpeg",
                "source": "feats",
                "markered": True,
                "capture": stem,
                "split": split,
                "height": int(img.shape[0]),
                "width": int(img.shape[1]),
                "indenter": indenter,
                "indenter_param": param,
                "f_x": fx, "f_y": fy, "f_z": fz,
                "grid_z_max":  float(np.max(grid_z))  if grid_z is not None else None,
                "grid_z_mean": float(np.mean(grid_z)) if grid_z is not None else None,
                "gel_variant": GEL_VARIANT.get(split, "black_dot"),
                "domain": "real",
            }
            writer.add(row)
        writer.close()
        stats[split] = dict(kept=n_kept, bg=n_bg, dropped=n_dropped, err=n_err,
                            total=n_kept + n_bg)
        print(f"{split}: kept={n_kept} bg={n_bg} dropped(empty)={n_dropped} err={n_err}")
    print(f"\ntook {time.time()-t0:.0f}s")
    print(f"summary: {stats}")
    total_kept = sum(s["total"] for s in stats.values())
    total_bg = sum(s["bg"] for s in stats.values())
    print(f"grand total: {total_kept} kept ({total_bg} background, "
          f"{100*total_bg/total_kept:.1f}%)")


if __name__ == "__main__":
    main()
