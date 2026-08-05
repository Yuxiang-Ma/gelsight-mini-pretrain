#!/usr/bin/env python3
"""Re-ingest the 3 subsets (RTM, sim_tactile_mnist, sim_starstruck) **from
the raw upstream parquets** instead of from our already-aggregated parquets.

Why: our aggregated `gelsight-mini-pretrain/{real,sim}_*` parquets were
each made by a *single-frame-per-touch* extractor before the area+intensity
contact filter was tuned. That means anything the middle-frame heuristic
missed is gone forever. The raw upstream has every frame of every touch,
so filtering from raw lets us pick the highest-contact frames per touch
trajectory.

Raw upstream layout (one row per touch sequence; inner list = frames):
    sim_tactile_mnist:  207,680 rows × 32 frames =  ~6.65 M frames
    sim_starstruck:      52,800 rows × 32 frames =  ~1.69 M frames
    RealTactileMNIST_single: 600 rows × 256 frames = 153,600 frames

Filter rule (legacy area+intensity, per-subset I_min from user):
    diff      = |frame - baseline|   (greyscale, central 50% crop)
    mask      = diff > 10
    area      = mask.sum()
    intensity = diff[mask].mean()
    KEEP iff area >= 40 AND intensity >= I_min
    ELSE   keep with probability 1.5% (background-diversity)

Then hard-cap the kept pool per (subset, split) at 200,000 rows total by
stride-sampling at the upstream-row level so coverage stays uniform.

Output: writes mini_data_parquet/{subset}/train-*.parquet and
            mini_data_parquet/{subset}/test-*.parquet
using the unified SCHEMA from make_parquet_v2.SCHEMA.

Run via concurrent.futures.ProcessPoolExecutor with one worker per subset.
"""
import argparse
import glob
import io
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, "/home/yxma/MultimodalData")
from make_parquet_v2 import SCHEMA

BASE = "/media/yxma/Disk1/yuxiang/mini_data_parquet"
RAW_BASE = "/media/yxma/Disk1/yuxiang/mini_data/markerless"

PIXEL_THRESH = 10
A_MIN = 40
BG_RATE = 0.015
JPEG_Q = 92


# ---------- per-subset config ------------------------------------------------

SOURCES = {
    "real_tactile_mnist": dict(
        raw_dir=f"{RAW_BASE}/RealTactileMNIST_single",
        i_min=12,
        cap=200_000,
        frames_per_row=256,
        image_field="sensor_image",
        domain="real",
        obj_pattern="digit_{label}",
        digit_class=True,
        split_globs={"train": "train-*.parquet", "test": "test-*.parquet"},
        split_alloc={"train": 0.80, "test": 0.20},
    ),
    "sim_tactile_mnist": dict(
        raw_dir=f"{RAW_BASE}/SimTactileMNIST",
        i_min=10,
        cap=200_000,
        frames_per_row=32,
        image_field="sensor_image",
        domain="sim",
        obj_pattern="digit_{label}",
        digit_class=True,
        # Skip printed_train per upstream conventions; train+test only.
        split_globs={"train": "train-*.parquet", "test": "test-*.parquet"},
        split_alloc={"train": 0.51, "test": 0.49},
        expected_pass=0.19,
        force_stride=2,        # scan every other row → ~1M frames seen
    ),
    "sim_starstruck": dict(
        raw_dir=f"{RAW_BASE}/SimStarStruck",
        i_min=10,
        cap=200_000,
        frames_per_row=32,
        image_field="sensor_image",
        domain="sim",
        obj_pattern="starstruck",
        digit_class=False,
        split_globs={"train": "train-*.parquet", "test": "test-*.parquet"},
        split_alloc={"train": 0.75, "test": 0.25},
        expected_pass=0.33,
        force_stride=3,        # scan every 3rd row → ~660K frames seen
    ),
}


# ---------- helpers ----------------------------------------------------------

def grey_center(rgb):
    g = rgb.mean(axis=2).astype(np.float32)
    h, w = g.shape
    return g[h // 4:3 * h // 4, w // 4:3 * w // 4]


def decode(b):
    return np.array(Image.open(io.BytesIO(b)).convert("RGB"))


def to_jpeg(rgb, q=JPEG_Q):
    """Re-encode an RGB array as JPEG bytes (matches repo image_format)."""
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=q)
    return buf.getvalue()


def compute_baseline(paths, image_field, frames_per_row, n=120, seed=0):
    """Sample N frames spread across paths; return greyscale-central median."""
    rng = random.Random(seed)
    counts = [pq.read_metadata(p).num_rows for p in paths]
    total_rows = sum(counts)
    n_rows = min(max(8, n // frames_per_row + 1), total_rows)
    row_idxs = sorted(rng.sample(range(total_rows), n_rows))
    grays = []
    cum = 0
    rii = iter(row_idxs)
    nxt = next(rii, None)
    for p, c in zip(paths, counts):
        if nxt is None:
            break
        if nxt >= cum + c:
            cum += c
            continue
        local = []
        while nxt is not None and nxt < cum + c:
            local.append(nxt - cum)
            nxt = next(rii, None)
        if local:
            t = pq.read_table(p, columns=[image_field])
            for li in local:
                imgs = t.column(image_field)[li].as_py()
                # Pick one random frame per row
                idx = rng.randint(0, len(imgs) - 1)
                try:
                    rgb = decode(imgs[idx]["bytes"])
                    grays.append(grey_center(rgb))
                    if len(grays) >= n:
                        return np.median(np.stack(grays), axis=0)
                except Exception:
                    pass
        cum += c
    return np.median(np.stack(grays), axis=0)


def stride_for_target(n_rows, frames_per_row, expected_pass_rate, target):
    """Pick a row-stride so that scanning every Nth row yields ~target passing
    frames after the filter."""
    expected_pass_total = n_rows * frames_per_row * expected_pass_rate
    if expected_pass_total <= target:
        return 1
    return max(1, int(expected_pass_total / target))


# ---------- per-split worker -------------------------------------------------

def process_split(sub, cfg, split, target_count, seed):
    rng = random.Random(seed)
    paths = sorted(glob.glob(f"{cfg['raw_dir']}/data/{cfg['split_globs'][split]}"))
    if not paths:
        return sub, split, 0, 0, 0, "no raw parquets"
    counts = [pq.read_metadata(p).num_rows for p in paths]
    n_rows = sum(counts)
    # Use realistic expected pass rates derived from prior runs:
    # - real subsets at I_min=12: ~20-30%
    # - sim subsets at I_min=10 from RAW upstream: ~20-35% (NOT 95%;
    #   the 95% was from filter-on-filter double-dip)
    expected_pass = cfg.get("expected_pass",
                            0.25 if cfg["domain"] == "real" else 0.25)
    # Allow override of stride directly
    if "force_stride" in cfg:
        stride = cfg["force_stride"]
    else:
        stride = stride_for_target(n_rows, cfg["frames_per_row"],
                                   expected_pass, target_count)
    print(f"  [{sub}/{split}] rows={n_rows:,} stride={stride} "
          f"expected_pass={expected_pass:.0%} target={target_count:,}",
          flush=True)

    # Baseline (computed once per subset, but pass here per worker for simplicity)
    t0 = time.time()
    baseline = compute_baseline(paths, cfg["image_field"],
                                cfg["frames_per_row"], n=100, seed=seed)
    print(f"  [{sub}/{split}] baseline in {time.time()-t0:.0f}s", flush=True)

    kept = []
    n_scanned = n_pass = n_bg = 0
    cum = 0
    t0 = time.time()
    for p, c in zip(paths, counts):
        t = None
        for local in range(c):
            global_idx = cum + local
            if global_idx % stride != 0:
                continue
            if t is None:
                # Lazy-load just the needed columns
                wanted = [cfg["image_field"], "label", "object_id", "id",
                          "info.run_id"]
                wanted = [w for w in wanted
                          if w in pq.read_schema(p).names]
                t = pq.read_table(p, columns=wanted)
            row_imgs = t.column(cfg["image_field"])[local].as_py()
            label = (t.column("label")[local].as_py()
                     if "label" in t.column_names else None)
            object_id = (t.column("object_id")[local].as_py()
                         if "object_id" in t.column_names else None)
            row_id = (t.column("id")[local].as_py()
                      if "id" in t.column_names else "")
            run_id = (t.column("info.run_id")[local].as_py()
                      if "info.run_id" in t.column_names else "")

            for fi, frame_struct in enumerate(row_imgs):
                if frame_struct is None:
                    continue
                b = frame_struct["bytes"]
                try:
                    rgb = decode(b)
                except Exception:
                    continue
                n_scanned += 1
                g = grey_center(rgb)
                diff = np.abs(g - baseline)
                mask = diff > PIXEL_THRESH
                area = int(mask.sum())
                inten = float(diff[mask].mean()) if area > 0 else 0.0
                passes = (area >= A_MIN) and (inten >= cfg["i_min"])

                keep = False
                if passes:
                    keep = True
                    n_pass += 1
                elif rng.random() < BG_RATE:
                    keep = True
                    n_bg += 1

                if keep:
                    # Re-encode as JPEG to match repo schema; raw is usually
                    # already JPEG bytes but unifies formats anyway.
                    out_bytes = to_jpeg(rgb)
                    obj_name = (
                        cfg["obj_pattern"].format(label=label)
                        if "{label}" in cfg["obj_pattern"]
                        else cfg["obj_pattern"]
                    )
                    kept.append(dict(
                        image=out_bytes,
                        image_format="jpeg",
                        source=sub,
                        markered=False,
                        capture=f"{row_id}_t{fi}" if row_id else f"row{global_idx}_t{fi}",
                        split=split,
                        height=rgb.shape[0],
                        width=rgb.shape[1],
                        obj_name=obj_name,
                        digit_class=label if cfg["digit_class"] else None,
                        episode=str(object_id) if object_id is not None else run_id,
                        frame_idx=fi,
                        domain=cfg["domain"],
                    ))
        cum += c
        if t is not None:
            del t
        if (cum % max(1, n_rows // 20) == 0) and cum > 0:
            fps = n_scanned / max(0.01, time.time() - t0)
            print(f"  [{sub}/{split}] row {cum:,}/{n_rows:,}  scanned={n_scanned:,}  "
                  f"kept={len(kept):,}  ({fps:.0f} fps)", flush=True)
        # Early exit if we're already 1.5× over target (lets us truncate)
        if len(kept) > target_count * 1.5:
            print(f"  [{sub}/{split}] early-exit: kept={len(kept):,} exceeds "
                  f"target by 50%", flush=True)
            break

    # Stride-subsample down to target_count if oversubscribed
    if len(kept) > target_count:
        idx = np.linspace(0, len(kept) - 1, target_count, dtype=np.int64)
        kept = [kept[int(i)] for i in idx]
        print(f"  [{sub}/{split}] capped {n_pass + n_bg:,} -> {len(kept):,}",
              flush=True)

    print(f"  [{sub}/{split}] DONE  kept={len(kept):,}  pass={n_pass:,}  "
          f"bg={n_bg:,}  scanned={n_scanned:,}  in {time.time()-t0:.0f}s",
          flush=True)
    return sub, split, len(kept), n_pass, n_bg, kept


# ---------- per-subset orchestrator ------------------------------------------

def process_subset(sub, cfg, seed):
    print(f"\n=== {sub} starting ===  raw={cfg['raw_dir']}  "
          f"I_min={cfg['i_min']}  cap={cfg['cap']:,}", flush=True)
    results_by_split = {}
    for split, alloc in cfg["split_alloc"].items():
        target = int(cfg["cap"] * alloc)
        ret = process_split(sub, cfg, split, target, seed)
        sub_, split_, n_kept, n_pass, n_bg, kept = ret
        results_by_split[split] = kept

    # Build a single table per subset and write per-split shards.
    out_subset_dir = f"{BASE}/{sub}"
    os.makedirs(out_subset_dir, exist_ok=True)
    for split, rows in results_by_split.items():
        # Build columns matching the unified SCHEMA
        cols = {f.name: [] for f in SCHEMA}
        for r in rows:
            for fname in cols.keys():
                cols[fname].append(r.get(fname))
        arrays = []
        names = []
        for f in SCHEMA:
            arr = pa.array(cols[f.name], type=f.type)
            arrays.append(arr)
            names.append(f.name)
        table = pa.Table.from_arrays(arrays, schema=SCHEMA)
        # Single shard per split (consistent with current repo layout)
        out_path = f"{out_subset_dir}/{split}-00000-of-00001.parquet"
        tmp = out_path + ".tmp"
        pq.write_table(table, tmp, compression="snappy")
        os.replace(tmp, out_path)
        print(f"  wrote {out_path}  rows={table.num_rows:,}", flush=True)

    total = sum(len(v) for v in results_by_split.values())
    print(f"=== {sub} done  total kept={total:,} ===", flush=True)
    return sub, total


# ---------- main -------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subsets", nargs="*",
                    default=list(SOURCES.keys()),
                    help="subset names to reprocess (default: all 3)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260519)
    args = ap.parse_args()

    for s in args.subsets:
        if s not in SOURCES:
            raise SystemExit(f"unknown subset {s!r}; known: {list(SOURCES)}")

    print(f"reprocess_upstream  PIXEL_THRESH={PIXEL_THRESH}  A_MIN={A_MIN}  "
          f"BG_RATE={BG_RATE}")
    for s in args.subsets:
        cfg = SOURCES[s]
        print(f"  {s:25s}  raw={cfg['raw_dir'].split('/')[-1]:30s}  "
              f"I_min={cfg['i_min']}  cap={cfg['cap']:,}")

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_subset, s, SOURCES[s], args.seed + i): s
                for i, s in enumerate(args.subsets)}
        for fut in as_completed(futs):
            try:
                results.append(fut.result())
            except Exception as e:
                results.append((futs[fut], 0))
                import traceback; traceback.print_exc()

    print("\n=== SUMMARY ===")
    for s, n in results:
        print(f"  {s:25s}  {n:>8,}")


if __name__ == "__main__":
    main()
