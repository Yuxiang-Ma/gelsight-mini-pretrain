#!/usr/bin/env python3
"""Sample published shards, recompute contact intensity, estimate i_min.

Usage:
    python tools/recover_imin.py --source gelslam --sample 2000
    python tools/recover_imin.py --all

Read-only: touches only config.PARQUET_MAIN / PARQUET_NC.
"""
from __future__ import annotations

import argparse
import glob
import io
import sys

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))

from gsmp import config, filters                      # noqa: E402
from gsmp.tools_imin import estimate_from_intensities  # noqa: E402

ALL_SOURCES = [
    "gelslam", "tactile_tracking", "real_tactile_mnist", "feelanyforce",
    "threedcal", "tacquad", "unit", "sim_tactile_mnist", "sim_starstruck",
    "feats", "fota_labeled", "fota_unlabeled", "sparsh",
]

#: Sources published to the NC repo rather than the main one. Until the
#: source modules exist (Task 13-16) the registry cannot answer this, so the
#: tools carry the mapping. After Task 16, spec.get(name).license_repo is
#: authoritative and this constant should agree with it.
SOURCE_LICENSE_REPO = {"sparsh": "nc"}


def sample_intensities(source: str, n: int) -> np.ndarray:
    """Decode up to n published frames and measure intensity vs a per-capture
    median baseline built from that same shard."""
    repo = SOURCE_LICENSE_REPO.get(source, "main")
    shards = sorted(glob.glob(str(config.published_dir(source, repo) / "*.parquet")))
    if not shards:
        raise FileNotFoundError(f"no shards for {source}")

    cols = pq.ParquetFile(shards[0]).schema_arrow.names
    want = ["image"] + (["capture"] if "capture" in cols else [])
    table = pq.read_table(shards[0], columns=want)
    total = table.num_rows
    step = max(1, total // n)
    idx = list(range(0, total, step))[:n]

    images = table.column("image").to_pylist()
    captures = (
        table.column("capture").to_pylist() if "capture" in want
        else ["_"] * total
    )

    by_cap = {}
    for i in idx:
        arr = np.array(Image.open(io.BytesIO(images[i])).convert("RGB"))
        by_cap.setdefault(captures[i], []).append(arr)

    out = []
    for frames in by_cap.values():
        greys = np.stack([filters.grey_center(f) for f in frames])
        base = np.median(greys, axis=0)
        for f in frames:
            _, inten = filters.contact_metrics(f, base)
            out.append(inten)
    return np.array(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--sample", type=int, default=2000)
    args = ap.parse_args()

    targets = ALL_SOURCES if args.all else [args.source]
    if not targets or targets == [None]:
        ap.error("pass --source NAME or --all")

    print(f"{'source':22s} {'n':>6s} {'min':>7s} {'p01':>7s} {'p05':>7s}  verdict")
    for src in targets:
        try:
            vals = sample_intensities(src, args.sample)
            est = estimate_from_intensities(src, vals, bg_keep_rate=0.015)
        except Exception as exc:                        # noqa: BLE001
            print(f"{src:22s} {'-':>6s} {'-':>7s} {'-':>7s} {'-':>7s}  ERROR: {exc}")
            continue
        print(f"{est.source:22s} {est.n_sampled:6d} {est.min_kept_intensity:7.2f} "
              f"{est.p01:7.2f} {est.p05:7.2f}  {est.verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
