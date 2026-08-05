#!/usr/bin/env python3
"""Apply per-bucket caps to compose the final balanced ~1M dataset.

Reads every parquet from BASE/<source>/, derives the same bucket
4-tuple (domain, sensor_id, object_id, gel_variant) as
compute_balance.py, then applies a **per-domain per-object cap** to
each row independently:

    keep_per_object_real = 250       # cap each object's frame count in real
    keep_per_object_sim  = 50        # cap each object's frame count in sim

These caps are calibrated so that:
  - Real total ≈ 600K (with ~600 RTM digits × 250 ≈ 150K dominating)
  - Sim total ≈ 400K (with ~1300 sim objects × ~300 ≈ 400K)
  - 60/40 real/sim split
  - ~1M grand total

In addition, a **per-domain hard cap** of 600K/400K (no overflow):

    cap_real_total = 600_000
    cap_sim_total  = 400_000

If after object-capping a domain is still over its hard cap, stride-
subsample uniformly within each bucket.

Output: writes new parquets to BASE_OUT (default = same as BASE,
overwriting in place). Skips Sparsh (which lives in the NC repo).
Skips tacquad_mini (already removed by an earlier cleanup).
"""
import argparse
import glob
import os
import random
import sys
from collections import defaultdict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, "/home/yxma/MultimodalData")
from compute_balance import sensor_id_for, object_id_for
from make_parquet_v2 import SCHEMA

BASE = "/media/yxma/Disk1/yuxiang/mini_data_parquet"

# Per-object cap (per-source × object-instance)
CAP_PER_OBJ_REAL = 250
CAP_PER_OBJ_SIM = 50

# Per-domain absolute cap (post-object-cap pass)
CAP_REAL_TOTAL = 600_000
CAP_SIM_TOTAL = 400_000


def derive_buckets(table, source_name):
    """Vectorised bucket derivation. Returns dict with arrays: domain,
    sensor, object_id, bucket."""
    n = table.num_rows
    schema = table.column_names
    def col(name, default=None):
        if name in schema:
            return table.column(name).to_pylist()
        return [default] * n

    src = col("source", source_name)
    side = col("side")
    indenter = col("indenter")
    split = col("split")
    obj_name = col("obj_name")
    capture = col("capture")
    episode = col("episode")
    x_mm = col("x_mm")
    y_mm = col("y_mm")
    markered = col("markered", False)
    domain = col("domain")

    out_dom, out_sensor, out_obj, out_gel = [], [], [], []
    for i in range(n):
        s = src[i] or source_name
        d = domain[i] or ("sim" if s.startswith("sim_") else "real")
        sensor = sensor_id_for(s, side[i], indenter[i], split[i])
        obj = object_id_for(s, obj_name[i], capture[i], side[i], split[i],
                            episode[i], indenter[i], x_mm[i], y_mm[i])
        gel = "markered" if markered[i] else "markerless"
        out_dom.append(d)
        out_sensor.append(sensor)
        out_obj.append(obj)
        out_gel.append(gel)
    return out_dom, out_sensor, out_obj, out_gel


def per_object_cap(rows_keep, sources_seen, rng):
    """Apply per-object cap. rows_keep is a dict source -> table.
    Returns dict source -> filtered_table."""
    # Build a global per-object index list (source, row_idx_within_source)
    obj_to_rows = defaultdict(list)
    for sub, t in rows_keep.items():
        dom, sensor, obj, gel = derive_buckets(t, sub)
        for i in range(t.num_rows):
            obj_to_rows[(dom[i], obj[i])].append((sub, i))

    # Per object: shuffle rows, cap to per-domain limit, mark kept
    kept_per_source = defaultdict(list)
    for (dom, obj), entries in obj_to_rows.items():
        cap = CAP_PER_OBJ_SIM if dom == "sim" else CAP_PER_OBJ_REAL
        if len(entries) > cap:
            # uniform stride to preserve ordering across sequence
            idx = np.linspace(0, len(entries) - 1, cap, dtype=np.int64)
            entries = [entries[int(i)] for i in idx]
        for sub, i in entries:
            kept_per_source[sub].append(i)

    out = {}
    for sub, t in rows_keep.items():
        idxs = sorted(kept_per_source[sub])
        if len(idxs) == t.num_rows:
            out[sub] = t
        else:
            mask = np.zeros(t.num_rows, dtype=bool)
            mask[idxs] = True
            out[sub] = t.filter(pa.array(mask))
        print(f"  per-obj cap: {sub:25s}  {t.num_rows:>8,} -> {out[sub].num_rows:>8,}",
              flush=True)
    return out


def per_domain_cap(tables_by_source, rng):
    """If a domain is still over total cap after per-object cap, stride-
    subsample uniformly across sources within that domain."""
    # Bucket sources by their domain (peek first row)
    domain_for_source = {}
    for sub, t in tables_by_source.items():
        if "domain" in t.column_names and t.num_rows > 0:
            d = t.column("domain")[0].as_py()
        else:
            d = "sim" if sub.startswith("sim_") else "real"
        domain_for_source[sub] = d

    out = dict(tables_by_source)
    for dom, total_cap in [("real", CAP_REAL_TOTAL), ("sim", CAP_SIM_TOTAL)]:
        dom_subs = [s for s, d in domain_for_source.items() if d == dom]
        dom_n = sum(out[s].num_rows for s in dom_subs)
        if dom_n <= total_cap:
            print(f"  {dom} domain: {dom_n:,} ≤ cap {total_cap:,} (no truncate)",
                  flush=True)
            continue
        # Subsample each source proportionally
        ratio = total_cap / dom_n
        for sub in dom_subs:
            t = out[sub]
            target = max(1, int(round(t.num_rows * ratio)))
            if target < t.num_rows:
                idx = np.linspace(0, t.num_rows - 1, target, dtype=np.int64)
                out[sub] = t.take(pa.array(idx))
                print(f"  domain cap: {sub:25s}  {t.num_rows:>8,} -> "
                      f"{out[sub].num_rows:>8,}  (ratio {ratio:.3f})",
                      flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--out", default=BASE,
                    help="Output dir (overwrites in place by default)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rng = random.Random(20260520)
    sources = sorted([d for d in os.listdir(args.base)
                      if os.path.isdir(f"{args.base}/{d}")
                      and not d.startswith(".") and d != "assets"])
    print(f"Found {len(sources)} sources: {sources}")

    # Load every parquet
    by_source_paths = {}     # source -> list of (path, split_label)
    by_source_table = {}     # source -> concatenated table (we'll re-shard by split later)
    for sub in sources:
        paths = sorted(glob.glob(f"{args.base}/{sub}/*.parquet"))
        if not paths:
            continue
        # Cast image -> large_binary so we can concat safely
        tables = []
        for p in paths:
            t = pq.read_table(p)
            if "image" in t.column_names and pa.types.is_binary(
                t.schema.field("image").type
            ):
                i = t.column_names.index("image")
                t = t.set_column(i, "image",
                                 t.column("image").cast(pa.large_binary()))
            tables.append(t)
        merged = pa.concat_tables(tables)
        by_source_table[sub] = merged
        by_source_paths[sub] = [(p, os.path.basename(p).rsplit("-", 2)[0])
                                for p in paths]
        print(f"  loaded {sub:25s}  {merged.num_rows:>8,}")

    n_before = sum(t.num_rows for t in by_source_table.values())
    print(f"\nTOTAL before rebalance: {n_before:,}")

    # Pass 1: per-object cap
    print("\n--- Pass 1: per-object cap ---")
    after_obj = per_object_cap(by_source_table, sources, rng)
    n_after_obj = sum(t.num_rows for t in after_obj.values())
    print(f"After per-object cap: {n_after_obj:,}")

    # Pass 2: per-domain cap
    print("\n--- Pass 2: per-domain cap ---")
    after_domain = per_domain_cap(after_obj, rng)
    n_after = sum(t.num_rows for t in after_domain.values())
    print(f"After per-domain cap: {n_after:,}")

    if args.dry_run:
        print("\n=== DRY RUN — no files modified ===")
        return

    # Write back per-source, split-routed by the original split column
    print("\n--- Writing rebalanced parquets ---")
    for sub, t in after_domain.items():
        # Group by split column value
        splits = t.column("split").to_pylist() if "split" in t.column_names else None
        if splits is None:
            # No split column → write as one shard
            out_path = f"{args.out}/{sub}/train-00000-of-00001.parquet"
            tmp = out_path + ".tmp"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            # Cast back to plain binary
            if "image" in t.column_names and pa.types.is_large_binary(
                t.schema.field("image").type
            ):
                i = t.column_names.index("image")
                t = t.set_column(i, "image", t.column("image").cast(pa.binary()))
            pq.write_table(t, tmp, compression="snappy")
            os.replace(tmp, out_path)
            print(f"  wrote {out_path}  rows={t.num_rows:,}")
            continue

        # Partition by split
        unique_splits = sorted(set(splits))
        for sp in unique_splits:
            mask = np.array([s == sp for s in splits])
            sub_t = t.filter(pa.array(mask))
            if sub_t.num_rows == 0:
                continue
            if "image" in sub_t.column_names and pa.types.is_large_binary(
                sub_t.schema.field("image").type
            ):
                i = sub_t.column_names.index("image")
                sub_t = sub_t.set_column(i, "image",
                                          sub_t.column("image").cast(pa.binary()))
            # Use existing split shard naming if one matches
            old_paths = [p for p, lbl in by_source_paths[sub] if lbl == sp]
            if old_paths:
                # Reuse first path's basename pattern (single shard)
                out_path = old_paths[0]
                # Remove other existing shards for this split
                for p in old_paths[1:]:
                    if os.path.exists(p):
                        os.remove(p)
                        print(f"    removed extra shard {p}")
            else:
                out_path = f"{args.out}/{sub}/{sp}-00000-of-00001.parquet"
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
            tmp = out_path + ".tmp"
            pq.write_table(sub_t, tmp, compression="snappy")
            os.replace(tmp, out_path)
            print(f"  wrote {out_path}  rows={sub_t.num_rows:,}")

    print(f"\n=== Rebalance done: {n_before:,} -> {n_after:,} ===")


if __name__ == "__main__":
    main()
