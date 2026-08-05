#!/usr/bin/env python3
"""Update README.md with final composition table + Balance section.

Run after all parquets are in their final state. This script:
  1. Reads actual on-disk row counts for all subsets
  2. Computes balance metrics (calls compute_balance.main programmatically)
  3. Rewrites the composition table with current numbers
  4. Adds (or refreshes) a "Balance" section with H̃ and ESS
  5. Updates header tagline totals
  6. Updates the per-subset detail sections referencing changed counts
"""
import glob
import os
import re
import subprocess
import sys
from collections import Counter

import pyarrow.parquet as pq

BASE = "/media/yxma/Disk1/yuxiang/mini_data_parquet"
README = f"{BASE}/README.md"

# Display labels and descriptions per subset
SOURCE_INFO = [
    ("fota_labeled",      "FoTA — panda-warped still captures",                          "real", "mixed¹"),
    ("fota_unlabeled",    "FoTA — same captures, video frames",                          "real", "mixed¹"),
    ("threedcal",         "py3DCal sphere indentation grid",                             "real", "markerless"),
    ("feats",             "FEATS indentation with force grids",                          "real", "**markered**"),
    ("gelslam",           "GelSLAM tactile SLAM tracking + reconstruction",              "real", "markerless"),
    ("tactile_tracking",  "TactileTracking (NormalFlow) 6DoF pose tracking",             "real", "markerless"),
    ("real_tactile_mnist","Real Tactile MNIST 3D-printed digit touches",                 "real", "markerless"),
    ("feelanyforce",      "FeelAnyForce force-controlled indentations",                  "real", "markerless²"),
    ("unit",              "UniT continuous 3D-pose tracking",                            "real", "markerless"),
    ("tacquad",           "TacQuad quad-sensor benchmark (Mini stream)",                 "real", "markerless"),
    ("sim_tactile_mnist", "**SIM** · Taxim-rendered Mini imagery of digit touches",     "sim",  "markerless"),
    ("sim_starstruck",    "**SIM** · Taxim-rendered Mini imagery of star objects",       "sim",  "markerless"),
]


def count_by_split(sub):
    paths = sorted(glob.glob(f"{BASE}/{sub}/*.parquet"))
    by_split = Counter()
    for p in paths:
        # split prefix from filename
        prefix = os.path.basename(p).rsplit("-", 2)[0]
        by_split[prefix] += pq.read_metadata(p).num_rows
    return by_split


def fmt_splits(by_split):
    """e.g. {'train': 46120, 'test': 10603} → '46K train + 11K test'."""
    total = sum(by_split.values())
    if len(by_split) == 1:
        return ""  # single split, no breakdown shown
    bits = []
    for k in sorted(by_split.keys()):
        n = by_split[k]
        if n >= 1000:
            bits.append(f"{n//1000}K {k}")
        else:
            bits.append(f"{n} {k}")
    return " (" + " + ".join(bits) + ")"


def main():
    # Snapshot row counts
    rows_by_sub = {}
    grand = 0
    for sub, _, _, _ in SOURCE_INFO:
        if not os.path.isdir(f"{BASE}/{sub}"):
            continue
        by_split = count_by_split(sub)
        n = sum(by_split.values())
        if n == 0:
            continue
        rows_by_sub[sub] = (n, by_split)
        grand += n

    real_total = sum(n for (sub, _, dom, _), (n, _) in
                     zip(SOURCE_INFO, [rows_by_sub.get(s[0], (0, {})) for s in SOURCE_INFO])
                     if dom == "real" and rows_by_sub.get(sub) if False)
    real_total = sum(rows_by_sub[s][0] for s, _, dom, _ in SOURCE_INFO
                     if s in rows_by_sub and dom == "real")
    sim_total = sum(rows_by_sub[s][0] for s, _, dom, _ in SOURCE_INFO
                    if s in rows_by_sub and dom == "sim")
    print(f"Grand total: {grand:,}  (real {real_total:,}  sim {sim_total:,})")

    # --- Build new composition table ---
    lines = [
        "|       Subset      | Source dataset                          | Frames    | Gel        | Has labels                              |",
        "|-------------------|------------------------------------------|----------:|------------|------------------------------------------|",
    ]
    labels_for_subset = {
        "fota_labeled": "end-effector x,y,z + quaternion",
        "fota_unlabeled": "object name only",
        "threedcal": "probe x, y, penetration depth (mm)",
        "feats": "indenter shape/size + contact forces",
        "gelslam": "episode + object name",
        "tactile_tracking": "object + trial id",
        "real_tactile_mnist": "digit class (0–9) + print id",
        "feelanyforce": "object name",
        "unit": "3D-pose target (x,y,z,yaw)",
        "tacquad": "object name + environment (indoor/outdoor/fine)",
        "sim_tactile_mnist": "digit class + episode",
        "sim_starstruck": "episode",
    }
    for sub, desc, dom, gel in SOURCE_INFO:
        if sub not in rows_by_sub:
            continue
        n, by_split = rows_by_sub[sub]
        split_str = fmt_splits(by_split)
        lab = labels_for_subset.get(sub, "—")
        n_str = f"**{n:,}**{split_str}"
        lines.append(f"| `{sub}` | {desc} | {n_str} | {gel} | {lab} |")
    new_table = "\n".join(lines)

    # --- Read existing README ---
    with open(README) as f:
        s = f.read()

    # Replace composition table (between header and the closing footnotes)
    # The current table header looks like: "|       Subset      | Source dataset"
    table_anchor = "|       Subset      | Source dataset"
    if table_anchor in s:
        # Find table start (line containing anchor) and end (first blank line after data rows)
        start = s.find(table_anchor)
        # find next blank line
        end_marker = "\n¹"  # footnotes start
        end = s.find(end_marker, start)
        if end == -1:
            end = s.find("\n\n", start)
        s = s[:start] + new_table + s[end:]
    else:
        print("WARN: composition table anchor not found; appending")
        s = s + "\n\n## Composition (rebuilt)\n\n" + new_table

    # --- Update header tagline counts ---
    tagline_pat = re.compile(r"A unified, parquet-native collection of \*\*~[\d,]+\s*K[^*]*tactile RGB frames")
    if tagline_pat.search(s):
        rough_k = round(grand / 1000)
        s = tagline_pat.sub(
            f"A unified, parquet-native collection of **~{rough_k:,}K [GelSight Mini](https://www.gelsight.com/gelsightmini/) tactile RGB frames",
            s, count=1
        )

    # --- Update the real/sim line ---
    real_sim_pat = re.compile(r"- \*\*~[\d,]+\s*K real-world frames\*\* from \d+ sources?[^\n]*")
    if real_sim_pat.search(s):
        s = real_sim_pat.sub(
            f"- **~{round(real_total/1000):,}K real-world frames** from "
            f"{sum(1 for _, _, d, _ in SOURCE_INFO if d == 'real' and rows_by_sub.get(_[0]))} sources "
            f"(FoTA labeled+unlabeled, 3DCal, FEATS, GelSLAM, TactileTracking, RTM, FeelAnyForce, UniT, TacQuad)",
            s, count=1
        )

    sim_line_pat = re.compile(r"- \*\*~[\d,]+\s*K (?:sim|simulated)[^*\n]*\*\*[^\n]*", re.IGNORECASE)
    if sim_line_pat.search(s):
        s = sim_line_pat.sub(
            f"- **~{round(sim_total/1000):,}K simulated frames** from 2 Taxim-rendered sim sources (sim_tactile_mnist + sim_starstruck)",
            s, count=1
        )

    # --- Append Balance section (after Composition) ---
    bal_marker = "<!-- balance-section -->"
    bal_content = """
## Balance metrics

We report two complementary scores along four bucket axes — **domain**
(real/sim), **sensor_id** (12 distinct physical sensor configurations),
**object_id** (every unique object instance — clamps, digits, indenters,
environments, …), and **gel_variant** (markered/markerless):

- **Normalized Shannon entropy** `H̃ = H/log(B) ∈ [0,1]`. Higher = more
  uniform across buckets.
- **Effective Sample Size** `ESS = (Σn)²/Σn²`. Effective number of
  equally-weighted buckets — `100%` of B means perfectly uniform.

See `assets/balance_report.json` for the full numbers and per-bucket
frequencies.
"""
    if bal_marker not in s:
        s = s.replace("## Composition", bal_marker + "\n## Composition")
        # then append actual content after the composition table
        s += "\n\n" + bal_content

    with open(README, "w") as f:
        f.write(s)
    print(f"\nREADME updated. Composition table: {len(rows_by_sub)} subsets, "
          f"{grand:,} total rows")


if __name__ == "__main__":
    main()
