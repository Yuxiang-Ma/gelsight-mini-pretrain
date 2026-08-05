#!/usr/bin/env python3
"""Ingest TVL (Touch-Vision-Language, Fu et al. 2024) into the main repo
using the unified pipeline library.

Upstream layout (from yoorhim/TVL-revise, the corrected fork):
  tvl_dataset/
    hct/                                     # hand-collected touches
      data{1,2,3}/
        contact.json, not_contact.json,
        train.csv, test.csv, finetune.json,
        <timestamp>/
          tactile/<frame_id>.jpg
          vision/<frame_id>.jpg
    ssvtp/                                   # SSVTP subset
      train.csv, test.csv, finetune.json,
      images_tac/image_<N>_tac.jpg
      images_rgb/image_<N>_rgb.jpg
      text/labels_<N>.txt

We only ingest the **tactile** images (Mini stream). Each tactile image
has a paired RGB camera image and (sometimes) a language caption, but
we don't need those for VAE pretraining; we just keep the tactile.

Usage:
  python ingest_tvl.py            # full pipeline + push
  python ingest_tvl.py --dry-run  # filter + write parquet, skip push
"""
import argparse
import csv
import glob
import io
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "/home/yxma/MultimodalData")
from pipeline import IngestPipeline, FrameRecord

TVL_BASE = "/media/yxma/Disk1/yuxiang/mini_data/markerless/TVL/tvl_dataset"


def decode(p):
    return np.array(Image.open(p).convert("RGB"))


def yield_hct_frames():
    """HCT (hand-collected touches): walk data{1,2,3}/<timestamp>/tactile/*.jpg.
    Use train.csv / test.csv to assign splits; default to train if not
    listed."""
    for data_dir_name in sorted(os.listdir(f"{TVL_BASE}/hct")):
        if not data_dir_name.startswith("data"):
            continue
        data_dir = f"{TVL_BASE}/hct/{data_dir_name}"
        # Load split assignments from train.csv / test.csv
        test_set = set()
        for fname in ["test.csv"]:
            csv_p = f"{data_dir}/{fname}"
            if os.path.exists(csv_p):
                with open(csv_p) as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if row:
                            test_set.add(row[0])
        # Iterate timestamped touches
        for ts_dir in sorted(os.listdir(data_dir)):
            ts_path = f"{data_dir}/{ts_dir}"
            if not os.path.isdir(ts_path):
                continue
            tactile_dir = f"{ts_path}/tactile"
            if not os.path.isdir(tactile_dir):
                continue
            for tactile_jpg in sorted(os.listdir(tactile_dir)):
                if not tactile_jpg.endswith(".jpg"):
                    continue
                full_path = f"{tactile_dir}/{tactile_jpg}"
                # Relative key for csv lookup
                rel_key = f"hct/{data_dir_name}/{ts_dir}/tactile/{tactile_jpg}"
                split = "test" if rel_key in test_set else "train"
                try:
                    rgb = decode(full_path)
                except Exception:
                    continue
                frame_id = tactile_jpg.replace(".jpg", "")
                # Parse "165-0.0253031..." → frame_idx + relative time
                fi = 0
                try:
                    fi = int(frame_id.split("-")[0])
                except Exception:
                    pass
                yield FrameRecord(
                    rgb=rgb,
                    obj_name=f"hct_{data_dir_name}",
                    capture=f"hct_{data_dir_name}_{ts_dir}",
                    episode=ts_dir,
                    frame_idx=fi,
                    split=split,
                )


def yield_ssvtp_frames():
    """SSVTP: images_tac/image_<N>_tac.jpg + paired language captions."""
    tac_dir = f"{TVL_BASE}/ssvtp/images_tac"
    if not os.path.isdir(tac_dir):
        return
    # Read train.csv / test.csv to assign splits
    train_set, test_set = set(), set()
    for fname, target in [("train.csv", train_set), ("test.csv", test_set)]:
        csv_p = f"{TVL_BASE}/ssvtp/{fname}"
        if os.path.exists(csv_p):
            with open(csv_p) as f:
                reader = csv.reader(f)
                for row in reader:
                    if row: target.add(row[0])
    for tactile_jpg in sorted(os.listdir(tac_dir)):
        if not tactile_jpg.endswith("_tac.jpg"):
            continue
        full_path = f"{tac_dir}/{tactile_jpg}"
        # image_42_tac.jpg → frame 42
        try:
            fi = int(tactile_jpg.split("_")[1])
        except Exception:
            fi = 0
        # Decide split
        rel_key = f"ssvtp/images_tac/{tactile_jpg}"
        split = "test" if rel_key in test_set or tactile_jpg in test_set else "train"
        try:
            rgb = decode(full_path)
        except Exception:
            continue
        yield FrameRecord(
            rgb=rgb,
            obj_name=f"ssvtp_image_{fi}",
            capture=f"ssvtp_image_{fi}",
            episode="ssvtp",
            frame_idx=fi,
            split=split,
        )


def yield_all():
    yield from yield_hct_frames()
    yield from yield_ssvtp_frames()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="filter + write parquet, skip HF push")
    args = ap.parse_args()

    pipe = IngestPipeline(
        source_name="tvl",
        repo="yxma/gelsight-mini-pretrain",
        domain="real",
        gel_variant="markerless",
        i_min=12.0,
        channel_mode="auto",   # auto-detect per image
        apply_filter=True,
        sample_seed=20260520,
    )

    if not os.path.isdir(TVL_BASE):
        sys.exit(f"TVL data not found at {TVL_BASE}. "
                 f"Run snapshot_download from yoorhim/TVL-revise + "
                 f"unzip first.")

    result = pipe.run(yield_all(), push=not args.dry_run)
    print(f"\n=== TVL ingest done ===")
    print(f"  kept: {result['n_kept']:,}  (bg: {result['n_bg']:,})")
    print(f"  parquets: {list(result['parquet_paths'].keys())}")
    if result["commit_url"]:
        print(f"  commit: {result['commit_url']}")


if __name__ == "__main__":
    main()
