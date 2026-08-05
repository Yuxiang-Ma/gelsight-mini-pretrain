#!/usr/bin/env python3
"""Convert FEATS .npy dicts into loose .png images + per-split force CSV.

Each .npy in FEATS is a pickled dict with keys:
  gs_img:  (240, 320, 3) uint8 RGB  -> the GelSight Mini frame
  grid_x:  (32, 24) force grid x   |
  grid_y:  (32, 24) force grid y   |  per-image force/shear info
  grid_z:  (32, 24) force grid z   |
  f_x/f_y/f_z: scalar total forces  /

For the unified image dataset, we extract gs_img -> <stem>.png.
The force info is summarized into a per-split CSV so users who want it later don't
have to reopen the .npy files.

After conversion completes successfully, the original v24_labels_24_32/ tree is
left in place; the user can delete it manually if they want only the .png images
to live on disk.
"""

import csv
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image

SRC = "/media/yxma/Disk1/yuxiang/mini_data/markered/FEATS/v24_labels_24_32"
DST = "/media/yxma/Disk1/yuxiang/mini_data/markered/FEATS/images"

# Splits we care about
SPLITS = [
    "train",
    "val",
    "test",
    "test_diff_sensor_new_gel",
    "test_diff_sensor_old_gel",
    "test_unknown_indenters",
]


def parse_indenter(stem: str):
    """Try to extract indenter shape and size from filename.
    Examples seen: '113_cuboid_12', '100_1744012609911964307_sphere_10'.
    Returns (indenter_shape, indenter_param) or ('unknown', '').
    """
    tokens = stem.split("_")
    # Shapes we know FEATS uses: sphere, cuboid, cylinder, cone, hemisphere, etc.
    known = {"sphere", "cuboid", "cylinder", "cone", "hemisphere", "torus",
             "prism", "pyramid", "cross", "ellipsoid", "flat"}
    for i, t in enumerate(tokens):
        if t.lower() in known:
            param = tokens[i + 1] if i + 1 < len(tokens) else ""
            return t.lower(), param
    return "unknown", ""


def convert_one(args):
    src_path, dst_path = args
    try:
        d = np.load(src_path, allow_pickle=True).item()
    except Exception as e:
        return (src_path, None, f"load_failed: {e}")
    img = d.get("gs_img")
    if img is None or img.dtype != np.uint8 or img.ndim != 3:
        return (src_path, None, f"bad gs_img: shape={None if img is None else img.shape}")
    try:
        Image.fromarray(img).save(dst_path, optimize=False)
    except Exception as e:
        return (src_path, None, f"save_failed: {e}")

    grid_z = d.get("grid_z")
    summary = {
        "filename": os.path.basename(dst_path),
        "f_x": float(d.get("f_x", 0.0) or 0.0),
        "f_y": float(d.get("f_y", 0.0) or 0.0),
        "f_z": float(d.get("f_z", 0.0) or 0.0),
        "grid_z_max": float(np.max(grid_z)) if grid_z is not None else 0.0,
        "grid_z_mean": float(np.mean(grid_z)) if grid_z is not None else 0.0,
    }
    return (src_path, summary, "ok")


def main():
    os.makedirs(DST, exist_ok=True)
    overall_rows = []

    workers = max(2, min(6, (os.cpu_count() or 4)))
    print(f"Using {workers} workers", flush=True)

    for split in SPLITS:
        src_split = os.path.join(SRC, split)
        if not os.path.isdir(src_split):
            continue
        dst_split = os.path.join(DST, split)
        os.makedirs(dst_split, exist_ok=True)

        jobs = []
        for root, _, files in os.walk(src_split):
            rel = os.path.relpath(root, src_split)
            out_root = dst_split if rel == "." else os.path.join(dst_split, rel)
            os.makedirs(out_root, exist_ok=True)
            for f in files:
                if not f.endswith(".npy"):
                    continue
                src_path = os.path.join(root, f)
                stem = os.path.splitext(f)[0]
                dst_path = os.path.join(out_root, stem + ".png")
                jobs.append((src_path, dst_path))

        print(f"\n[{split}] {len(jobs)} files", flush=True)
        if not jobs:
            continue

        rows = []
        errors = []
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(convert_one, j) for j in jobs]
            for i, fut in enumerate(as_completed(futs), 1):
                src_path, summary, status = fut.result()
                if summary is None:
                    errors.append((src_path, status))
                else:
                    stem = os.path.splitext(os.path.basename(src_path))[0]
                    indenter, param = parse_indenter(stem)
                    row = {"split": split, "indenter": indenter, "indenter_param": param}
                    row.update(summary)
                    rows.append(row)
                if i % 500 == 0:
                    print(f"  [{split}] {i}/{len(jobs)} ({time.time()-t0:.0f}s)", flush=True)

        print(f"  [{split}] done in {time.time()-t0:.0f}s.  ok={len(rows)}  errors={len(errors)}", flush=True)
        if errors[:3]:
            for src_path, status in errors[:3]:
                print(f"    ERR {src_path}: {status}", flush=True)

        # Write per-split CSV
        out_csv = os.path.join(dst_split, "forces.csv")
        if rows:
            fields = ["filename", "split", "indenter", "indenter_param",
                      "f_x", "f_y", "f_z", "grid_z_max", "grid_z_mean"]
            with open(out_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for r in rows:
                    w.writerow({k: r.get(k, "") for k in fields})
        overall_rows.extend(rows)

    # Top-level manifest
    manifest = os.path.join(DST, "all_forces.csv")
    if overall_rows:
        fields = ["filename", "split", "indenter", "indenter_param",
                  "f_x", "f_y", "f_z", "grid_z_max", "grid_z_mean"]
        with open(manifest, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in overall_rows:
                w.writerow({k: r.get(k, "") for k in fields})
    print(f"\nWrote {len(overall_rows)} rows to {manifest}")


if __name__ == "__main__":
    sys.exit(main())
