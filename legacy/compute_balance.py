#!/usr/bin/env python3
"""Compute dataset balance metrics across all subsets.

For each row in every parquet, derive its bucket = (domain, sensor_id,
object_id, gel_variant), where sensor_id is inferred from source +
optional `side` / split columns.

Then compute two scalar balance scores along each axis:

  H̃ = H / log(B)   # normalized Shannon entropy in [0, 1]; 1 = uniform
  ESS = (Σ n_b)² / Σ n_b²   # effective sample size

These tell us how skewed the dataset is. We also report:
  - per-source row counts
  - per-bucket frequencies along each axis (domain, sensor, gel)
  - top/bottom-10 buckets in the combined 4-tuple space

Output: prints to stdout AND saves to assets/balance_report.json
"""
import argparse
import glob
import json
import math
import os
from collections import Counter, defaultdict

import pyarrow.parquet as pq

BASE = "/media/yxma/Disk1/yuxiang/mini_data_parquet"


def sensor_id_for(source, side=None, indenter=None, split=None):
    """Map (source, side, indenter, split) → physical sensor_id string."""
    if source == "fota_labeled" or source == "fota_unlabeled":
        return f"mini_fota_{side or 'unk'}"
    if source == "feats":
        if split and split.startswith("test_diff_sensor_new_gel"):
            return "mini_feats_new_gel"
        if split and split.startswith("test_diff_sensor_old_gel"):
            return "mini_feats_old_sensor"
        return "mini_feats_main"
    if source == "gelslam":
        return "mini_gelslam"
    if source == "tactile_tracking":
        return "mini_tracking"
    if source == "real_tactile_mnist":
        return "mini_rtm"
    if source == "feelanyforce":
        return "mini_feelanyforce"
    if source == "threedcal":
        return "mini_threedcal"
    if source == "unit":
        return "mini_unit"
    if source == "tacquad":
        return "mini_tacquad"
    if source == "sim_tactile_mnist" or source == "sim_starstruck":
        return "sim_taxim_mini"
    if source == "sparsh":
        return f"mini_sparsh_{indenter}" if indenter else "mini_sparsh"
    return f"mini_{source}"


def object_id_for(source, obj_name, capture, side=None, split=None,
                  episode=None, indenter=None, x_mm=None, y_mm=None):
    """Map row metadata → object_id for bucketing."""
    if source in ("fota_labeled", "fota_unlabeled"):
        return f"{obj_name}/{capture or ''}"
    if source == "threedcal":
        # Each (x,y) is a different "object"-like calibration point
        try:
            return f"sphere/x={float(x_mm):.1f}_y={float(y_mm):.1f}"
        except Exception:
            return "sphere/unknown"
    if source == "feats":
        return f"{indenter or obj_name or 'unknown'}_{split or ''}"
    if source == "real_tactile_mnist":
        # episode = the digit-print id (1..600)
        return f"digit_{obj_name}/ep_{episode or 'unk'}"
    if source in ("sim_tactile_mnist",):
        return f"{obj_name}/ep_{episode or 'unk'}"
    if source == "sim_starstruck":
        return f"starstruck/ep_{episode or 'unk'}"
    if source == "tacquad":
        return f"{obj_name}/{split}"
    if source == "sparsh":
        return f"{indenter or obj_name or 'unk'}"
    return obj_name or "unknown"


def H_and_ESS(counts):
    """Normalized Shannon entropy + effective sample size."""
    n = sum(counts.values())
    if n == 0:
        return 0.0, 0.0
    B = len(counts)
    H = 0.0
    sumsq = 0.0
    for c in counts.values():
        p = c / n
        if p > 0:
            H -= p * math.log(p)
        sumsq += c * c
    Htilde = H / math.log(B) if B > 1 else 1.0
    ESS = (n * n) / sumsq if sumsq > 0 else 0.0
    return Htilde, ESS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--out", default=f"{BASE}/assets/balance_report.json")
    args = ap.parse_args()

    print(f"Scanning {args.base}/<source>/*.parquet ...")
    sources = sorted([d for d in os.listdir(args.base)
                      if os.path.isdir(f"{args.base}/{d}")
                      and not d.startswith(".")
                      and d not in ("assets", "scripts")])

    # Per-axis counters
    by_source = Counter()
    by_domain = Counter()
    by_sensor = Counter()
    by_gel = Counter()
    by_object = Counter()
    by_bucket = Counter()    # full 4-tuple
    by_split = Counter()

    n_total = 0
    for sub in sources:
        paths = sorted(glob.glob(f"{args.base}/{sub}/*.parquet"))
        if not paths:
            continue
        for p in paths:
            wanted = ["source", "domain", "obj_name", "capture", "side",
                      "split", "episode", "indenter", "x_mm", "y_mm",
                      "markered"]
            schema = pq.read_schema(p).names
            cols = [c for c in wanted if c in schema]
            t = pq.read_table(p, columns=cols)
            n = t.num_rows
            n_total += n
            # Vectorised pull
            d = {c: t.column(c).to_pylist() if c in cols else [None] * n
                 for c in wanted}
            for i in range(n):
                source = d["source"][i] or sub
                domain = d["domain"][i] or ("sim" if source.startswith("sim_") else "real")
                gel = "markered" if d["markered"][i] else "markerless"
                sensor = sensor_id_for(source, d["side"][i],
                                       d["indenter"][i], d["split"][i])
                obj = object_id_for(source, d["obj_name"][i], d["capture"][i],
                                    d["side"][i], d["split"][i],
                                    d["episode"][i], d["indenter"][i],
                                    d["x_mm"][i], d["y_mm"][i])
                bucket = (domain, sensor, obj, gel)
                by_source[source] += 1
                by_domain[domain] += 1
                by_sensor[sensor] += 1
                by_gel[gel] += 1
                by_object[obj] += 1
                by_bucket[bucket] += 1
                by_split[d["split"][i]] += 1

    print(f"\n=== Total rows scanned: {n_total:,} ===\n")

    def report(name, counter, max_buckets=20):
        H, ESS = H_and_ESS(counter)
        n = sum(counter.values())
        B = len(counter)
        # ESS interpretation: effective number of equally-weighted buckets.
        # 100% means perfectly uniform across all B buckets.
        print(f"--- by {name} ---  B={B:,}  H̃={H:.3f}  ESS={ESS:,.0f} "
              f"({100*ESS/B:.1f}% of B)")
        for k, v in counter.most_common(max_buckets):
            print(f"  {str(k):60s}  {v:>8,}  {100*v/n:5.2f}%")
        if B > max_buckets:
            tail = sum(v for _, v in counter.most_common()[max_buckets:])
            print(f"  ... + {B - max_buckets} more buckets totaling {tail:,}")
        print()

    report("source", by_source, max_buckets=30)
    report("domain", by_domain)
    report("sensor", by_sensor, max_buckets=30)
    report("gel",    by_gel)
    print(f"--- by object ---  B={len(by_object):,}  "
          f"H̃={H_and_ESS(by_object)[0]:.3f}  "
          f"ESS={H_and_ESS(by_object)[1]:,.0f}")
    print(f"  (showing top-10 only)")
    for k, v in by_object.most_common(10):
        print(f"  {str(k):60s}  {v:>8,}")
    print(f"  ... + {len(by_object) - 10} more objects\n")
    print(f"--- by 4-tuple bucket ---  B={len(by_bucket):,}  "
          f"H̃={H_and_ESS(by_bucket)[0]:.3f}  "
          f"ESS={H_and_ESS(by_bucket)[1]:,.0f}")

    # Save JSON report
    report_data = dict(
        total_rows=n_total,
        by_source=dict(by_source),
        by_domain=dict(by_domain),
        by_sensor=dict(by_sensor),
        by_gel=dict(by_gel),
        n_unique_objects=len(by_object),
        n_buckets=len(by_bucket),
        metrics=dict(
            source=H_and_ESS(by_source),
            domain=H_and_ESS(by_domain),
            sensor=H_and_ESS(by_sensor),
            gel=H_and_ESS(by_gel),
            object=H_and_ESS(by_object),
            bucket=H_and_ESS(by_bucket),
        ),
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
