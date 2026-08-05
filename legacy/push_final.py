#!/usr/bin/env python3
"""Final atomic push to yxma/gelsight-mini-pretrain.

Uploads:
  - All updated parquets (RTM video + sim reruns; everything else stays
    the same).
  - All refreshed assets/*.png (sample grids for all 12 subsets,
    composition.png, summary_pies.png, pixel_value_distribution.png,
    balance_report.json).
  - Updated README.md (with new Balance section + composition table).

Optionally deletes stale files (samples_40_tacquad_mini.png etc.).

Designed to be run as the FINAL step after:
  1. RTM video extract finished
  2. Sim rerun finished
  3. compute_balance.py ran
  4. update_readme_final.py ran
  5. All sample grids regenerated
"""
import os
import glob
from huggingface_hub import HfApi, CommitOperationAdd

REPO = "yxma/gelsight-mini-pretrain"
BASE = "/media/yxma/Disk1/yuxiang/mini_data_parquet"


def main():
    api = HfApi()
    ops = []

    # All parquets (HF dedup'd at xet layer; only changed ones cost bytes)
    sources = ["fota_labeled", "fota_unlabeled", "threedcal", "feats",
               "gelslam", "tactile_tracking", "real_tactile_mnist",
               "feelanyforce", "unit", "tacquad",
               "sim_tactile_mnist", "sim_starstruck"]
    for sub in sources:
        for p in sorted(glob.glob(f"{BASE}/{sub}/*.parquet")):
            rel = os.path.relpath(p, BASE)
            ops.append(CommitOperationAdd(path_in_repo=rel, path_or_fileobj=p))

    # All asset PNGs + JSON
    for p in sorted(glob.glob(f"{BASE}/assets/*.png")):
        rel = os.path.relpath(p, BASE)
        ops.append(CommitOperationAdd(path_in_repo=rel, path_or_fileobj=p))
    for p in sorted(glob.glob(f"{BASE}/assets/*.json")):
        rel = os.path.relpath(p, BASE)
        ops.append(CommitOperationAdd(path_in_repo=rel, path_or_fileobj=p))

    # README
    ops.append(CommitOperationAdd(
        path_in_repo="README.md", path_or_fileobj=f"{BASE}/README.md"))

    print(f"\nTotal ops: {len(ops)}")
    print(f"Sample of adds: {[o.path_in_repo for o in ops[:5]]}")
    info = api.create_commit(
        repo_id=REPO, repo_type="dataset", operations=ops,
        commit_message=(
            "v8 rebalance: RTM video re-extract → ~150K; sim reruns at "
            "smaller strides → ~155K each; UniT + TacQuad-full ingested; "
            "added Balance section + pixel-distribution chart"
        ),
    )
    print(f"\n{info.commit_url}")


if __name__ == "__main__":
    main()
