#!/bin/bash
# Post-channel-fix pipeline: waits for fix_channel_order.py to finish,
# then verifies via re-diagnostic, regenerates affected sample grids,
# and pushes to both HF repos.
set -e

cd /home/yxma/MultimodalData

echo "=== Waiting for fix_channel_order.py to finish ==="
while pgrep -f "fix_channel_order.py" > /dev/null; do
  sleep 60
done
echo "  fix_channel_order.py done"

echo ""
echo "=== 1. Re-run diagnose_channel_order.py to verify ==="
python3 diagnose_channel_order.py 2>&1 | tee /tmp/post_fix_diagnosis.log | tail -25

echo ""
echo "=== 2. Regenerate sample grids for 4 fixed subsets ==="
python3 make_samples_100.py fota_unlabeled unit 2>&1
echo ""
echo "Regenerating RGB-vs-BGR comparison grids (now all should look correct in TOP row)..."
python3 make_rgb_vs_bgr.py 2>&1 | tail -20

echo ""
echo "=== 3. Regenerate NC sparsh + faf sample grids ==="
python3 << 'PYEOF'
import sys, os, random
sys.path.insert(0, ".")
import make_samples_100 as ms
import pyarrow.parquet as pq

NC_BASE = "/media/yxma/Disk1/yuxiang/mini_data_parquet_nc"

def collect_uniform_from(p, n=40, seed=42):
    rng = random.Random(seed)
    total = pq.read_metadata(p).num_rows
    idxs = sorted(rng.sample(range(total), min(n, total)))
    t = pq.read_table(p, columns=["image"])
    return [t.column("image")[i].as_py() for i in idxs]

for source, indenters in [("sparsh", ["flat","sharp","sphere"]),
                          ("faf_force_estimation", ["flat","sharp","sphere"])]:
    for ind in indenters:
        p = f"{NC_BASE}/{source}/{ind}-00000-of-00001.parquet"
        if not os.path.exists(p): continue
        imgs = collect_uniform_from(p, n=40,
                                     seed=20260520 + hash(ind) % 1000)
        thumbs = [ms.thumbnail(b) for b in imgs]
        title = f"{source}/{ind} — 40 random samples (CC-BY-NC)"
        g = ms.make_grid(thumbs, title)
        out_p = f"{NC_BASE}/assets/samples_40_{source}_{ind}.png" if source == "faf_force_estimation" \
                else f"{NC_BASE}/assets/samples_40_sparsh_{ind}.png"
        g.save(out_p, optimize=True)
        print(f"  saved {out_p}")
PYEOF

echo ""
echo "=== 4. Push channel-corrected parquets + refreshed grids to HF ==="
python3 << 'PYEOF'
from huggingface_hub import HfApi, CommitOperationAdd
import os, glob

MAIN_BASE = "/media/yxma/Disk1/yuxiang/mini_data_parquet"
NC_BASE = "/media/yxma/Disk1/yuxiang/mini_data_parquet_nc"
api = HfApi()

# === MAIN repo: fota_unlabeled + unit + refreshed comparison grids ===
ops = []
for sub in ["fota_unlabeled", "unit"]:
    for p in sorted(glob.glob(f"{MAIN_BASE}/{sub}/*.parquet")):
        rel = os.path.relpath(p, MAIN_BASE)
        ops.append(CommitOperationAdd(path_in_repo=rel, path_or_fileobj=p))
        print(f"  ADD main/{rel}  ({os.path.getsize(p)/1e6:.0f} MB)")
# Refreshed sample grids + RGB-vs-BGR grids
for f in [f"assets/samples_40_fota_unlabeled.png",
          f"assets/samples_40_unit.png"]:
    p = f"{MAIN_BASE}/{f}"
    if os.path.exists(p):
        ops.append(CommitOperationAdd(path_in_repo=f, path_or_fileobj=p))
for p in sorted(glob.glob(f"{MAIN_BASE}/assets/samples_rgb_vs_bgr_*.png")):
    rel = os.path.relpath(p, MAIN_BASE)
    ops.append(CommitOperationAdd(path_in_repo=rel, path_or_fileobj=p))
for p in sorted(glob.glob(f"{MAIN_BASE}/assets/channel_order_diagnosis.json")):
    rel = os.path.relpath(p, MAIN_BASE)
    ops.append(CommitOperationAdd(path_in_repo=rel, path_or_fileobj=p))
info = api.create_commit(
    repo_id="yxma/gelsight-mini-pretrain", repo_type="dataset",
    operations=ops,
    commit_message=("fix channel order: unconditional R↔B swap for "
                    "fota_unlabeled + unit (were BGR-stored); refresh "
                    "affected sample grids + RGB-vs-BGR comparison + "
                    "publish channel_order_diagnosis.json"))
print(f"main: {info.commit_url}")

# === NC repo: faf + sparsh fixed parquets + refreshed grids ===
ops = []
for sub in ["faf_force_estimation", "sparsh"]:
    for p in sorted(glob.glob(f"{NC_BASE}/{sub}/*.parquet")):
        rel = os.path.relpath(p, NC_BASE)
        ops.append(CommitOperationAdd(path_in_repo=rel, path_or_fileobj=p))
        print(f"  ADD nc/{rel}  ({os.path.getsize(p)/1e6:.0f} MB)")
for p in sorted(glob.glob(f"{NC_BASE}/assets/samples_40_*.png")):
    rel = os.path.relpath(p, NC_BASE)
    ops.append(CommitOperationAdd(path_in_repo=rel, path_or_fileobj=p))
for p in sorted(glob.glob(f"{NC_BASE}/assets/samples_rgb_vs_bgr_*.png")):
    rel = os.path.relpath(p, NC_BASE)
    ops.append(CommitOperationAdd(path_in_repo=rel, path_or_fileobj=p))
info = api.create_commit(
    repo_id="yxma/gelsight-mini-pretrain-nc", repo_type="dataset",
    operations=ops,
    commit_message=("fix channel order: unconditional R↔B swap for "
                    "faf_force_estimation (BGR-stored); conditional "
                    "per-image swap for sparsh (mixed RGB/BGR per "
                    "Facebook upstream); refreshed grids"))
print(f"nc:   {info.commit_url}")
PYEOF

echo ""
echo "=== ALL DONE ==="

echo ""
echo "=== 5. Drop redundant faf_force_estimation/ (duplicate of Sparsh) ==="
python3 << 'PYEOF'
from huggingface_hub import HfApi, CommitOperationDelete
import os, shutil, glob

REPO = "yxma/gelsight-mini-pretrain-nc"
NC_BASE = "/media/yxma/Disk1/yuxiang/mini_data_parquet_nc"
api = HfApi()

ops = [
    CommitOperationDelete(path_in_repo="faf_force_estimation/flat-00000-of-00001.parquet"),
    CommitOperationDelete(path_in_repo="faf_force_estimation/sharp-00000-of-00001.parquet"),
    CommitOperationDelete(path_in_repo="faf_force_estimation/sphere-00000-of-00001.parquet"),
    CommitOperationDelete(path_in_repo="assets/samples_40_faf_force_estimation_flat.png"),
    CommitOperationDelete(path_in_repo="assets/samples_40_faf_force_estimation_sharp.png"),
    CommitOperationDelete(path_in_repo="assets/samples_40_faf_force_estimation_sphere.png"),
    CommitOperationDelete(path_in_repo="assets/samples_rgb_vs_bgr_faf_force_estimation.png"),
]
info = api.create_commit(
    repo_id=REPO, repo_type="dataset", operations=ops,
    commit_message=("drop faf_force_estimation/ — redundant snapshot of Sparsh "
                    "(same protocol, same indenters, smaller subset, "
                    "saved under different upstream repo name)"))
print(f"nc cleanup: {info.commit_url}")

# Also clean local
if os.path.isdir(f"{NC_BASE}/faf_force_estimation"):
    shutil.rmtree(f"{NC_BASE}/faf_force_estimation")
    print("  removed local faf_force_estimation/")
for p in glob.glob(f"{NC_BASE}/assets/*faf_force*"):
    os.remove(p)
    print(f"  removed local {p}")
PYEOF

echo ""
echo "=== ALL DONE (including faf cleanup) ==="
