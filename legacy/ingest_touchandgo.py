#!/usr/bin/env python3
"""Ingest Touch-and-Go (Yang et al., NeurIPS 2022) into the main repo
using the unified pipeline.

Upstream layout (from Google Drive folder
https://drive.google.com/drive/folders/1NDasyshDCL9aaQzxjn_-Q5MBURRT360B):

  TouchAndGo/
    <session_id>/
      video.mp4      # paired RGB camera (we IGNORE — not Mini sensor)
      gelsight.mp4   # tactile RGB video, GelSight Mini
      time1.npy      # timestamps for video.mp4 frames
      time2.npy      # timestamps for gelsight.mp4 frames

We extract the gelsight.mp4 frames, apply our unified contact filter,
and ingest as the `touchandgo` config in the main repo.

License: CC-BY-4.0 → main repo (compatible with our CC-BY-4.0 aggregate).

Usage:
  python ingest_touchandgo.py            # full pipeline + push
  python ingest_touchandgo.py --dry-run  # filter + parquet, skip push
"""
import argparse
import glob
import io
import os
import sys

import numpy as np
from PIL import Image
import imageio.v3 as iio

sys.path.insert(0, "/home/yxma/MultimodalData")
from pipeline import IngestPipeline, FrameRecord

TG_BASE = "/media/yxma/Disk1/yuxiang/mini_data/markerless/TouchAndGo"

# Sample every Nth frame from each gelsight.mp4 to avoid near-duplicate
# consecutive frames. Touch-and-Go videos are typically 30 fps with
# multi-second contact events.
FRAME_STRIDE = 5


def find_gelsight_videos():
    """Return list of (session_id, gelsight.mp4 path). Walks recursively
    because Drive layout is touch_and_go/dataset/<session>/gelsight.mp4."""
    out = []
    for root, dirs, files in os.walk(TG_BASE):
        if "gelsight.mp4" in files:
            session_id = os.path.basename(root)
            out.append((session_id, f"{root}/gelsight.mp4"))
    return sorted(out)


def yield_touchandgo_frames():
    videos = find_gelsight_videos()
    print(f"Found {len(videos)} Touch-and-Go gelsight.mp4 sessions",
          flush=True)
    for session_id, mp4 in videos:
        try:
            for fi, rgb in enumerate(iio.imiter(mp4)):
                if fi % FRAME_STRIDE != 0:
                    continue
                yield FrameRecord(
                    rgb=np.asarray(rgb),
                    obj_name=f"touchandgo_{session_id}",
                    capture=session_id,
                    episode=session_id,
                    frame_idx=fi,
                    split="train",
                )
        except Exception as e:
            print(f"  skip {session_id} ({e})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pipe = IngestPipeline(
        source_name="touchandgo",
        repo="yxma/gelsight-mini-pretrain",
        domain="real",
        gel_variant="markerless",
        i_min=12.0,
        channel_mode="auto",
        apply_filter=True,
        sample_seed=20260521,
    )

    if not find_gelsight_videos():
        sys.exit(f"No Touch-and-Go videos found under {TG_BASE}. "
                 f"Download from the Google Drive folder first.")

    result = pipe.run(yield_touchandgo_frames(), push=not args.dry_run)
    print(f"\n=== Touch-and-Go ingest done ===")
    print(f"  kept: {result['n_kept']:,}  (bg: {result['n_bg']:,})")
    print(f"  parquets: {list(result['parquet_paths'].keys())}")
    if result["commit_url"]:
        print(f"  commit: {result['commit_url']}")


if __name__ == "__main__":
    main()
