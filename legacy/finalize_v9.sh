#!/bin/bash
# Final v9 pipeline:
# - Wait for swap_fota_unlabeled.py to complete
# - Verify all 14 subsets are RGB
# - Regenerate the 4 affected sample grids (fota_unlabeled, unit, sparsh, faf)
# - Regenerate RGB-vs-BGR comparison grids (now top row should be correct everywhere)
# - Rewrite README composition + add note about channel-order normalization
# - Push channel-corrected parquets + grids + READMEs + final SOURCES.md
# - Drop redundant faf_force_estimation/ from NC repo
set -e
cd /home/yxma/MultimodalData

echo "=== 1. Wait for fota_unlabeled swap ==="
while pgrep -f "swap_fota_unlabeled.py" > /dev/null; do
  sleep 30
done
echo "  done"

echo ""
echo "=== 2. Verify channel order (all subsets should now be R < B) ==="
python3 diagnose_channel_order.py 2>&1 | tee /tmp/v9_diagnosis.log | tail -30

echo ""
echo "=== 3. Regenerate affected sample grids ==="
python3 make_samples_100.py fota_unlabeled unit 2>&1
echo ""
echo "Regenerating NC sparsh + faf sample grids..."
python3 << 'PYEOF'
import sys, os, random
sys.path.insert(0, ".")
import make_samples_100 as ms
import pyarrow.parquet as pq
NC = "/media/yxma/Disk1/yuxiang/mini_data_parquet_nc"
def collect_uniform_from(p, n=40, seed=42):
    rng = random.Random(seed)
    total = pq.read_metadata(p).num_rows
    idxs = sorted(rng.sample(range(total), min(n, total)))
    t = pq.read_table(p, columns=["image"])
    return [t.column("image")[i].as_py() for i in idxs]
for ind in ["flat","sharp","sphere"]:
    p = f"{NC}/sparsh/{ind}-00000-of-00001.parquet"
    imgs = collect_uniform_from(p, n=40, seed=20260520 + hash(ind) % 1000)
    g = ms.make_grid([ms.thumbnail(b) for b in imgs],
                     f"sparsh/{ind} - 40 random samples (CC-BY-NC)")
    out = f"{NC}/assets/samples_40_sparsh_{ind}.png"
    g.save(out, optimize=True); print(f"  saved {out}")
for ind in ["flat","sharp","sphere"]:
    p = f"{NC}/faf_force_estimation/{ind}-00000-of-00001.parquet"
    imgs = collect_uniform_from(p, n=40, seed=20260521 + hash(ind) % 1000)
    g = ms.make_grid([ms.thumbnail(b) for b in imgs],
                     f"faf_force_estimation/{ind} - 40 random samples (CC-BY-NC)")
    out = f"{NC}/assets/samples_40_faf_force_estimation_{ind}.png"
    g.save(out, optimize=True); print(f"  saved {out}")
PYEOF

echo ""
echo "=== 4. Regenerate RGB-vs-BGR comparison grids (top row should now be correct) ==="
python3 make_rgb_vs_bgr.py 2>&1 | tail -10

echo ""
echo "=== 5. Push channel-corrected parquets + refreshed grids + diagnostic ==="
python3 << 'PYEOF'
from huggingface_hub import HfApi, CommitOperationAdd
import os, glob
MAIN = "/media/yxma/Disk1/yuxiang/mini_data_parquet"
NC = "/media/yxma/Disk1/yuxiang/mini_data_parquet_nc"
api = HfApi()

# MAIN: fota_unlabeled + unit parquets + their refreshed grids + RGB-vs-BGR grids + diagnostic
ops = []
for sub in ["fota_unlabeled", "unit"]:
    for p in sorted(glob.glob(f"{MAIN}/{sub}/*.parquet")):
        rel = os.path.relpath(p, MAIN)
        ops.append(CommitOperationAdd(path_in_repo=rel, path_or_fileobj=p))
for f in [f"assets/samples_40_fota_unlabeled.png",
          f"assets/samples_40_unit.png",
          f"assets/channel_order_diagnosis.json"]:
    p = f"{MAIN}/{f}"
    if os.path.exists(p):
        ops.append(CommitOperationAdd(path_in_repo=f, path_or_fileobj=p))
for p in sorted(glob.glob(f"{MAIN}/assets/samples_rgb_vs_bgr_*.png")):
    rel = os.path.relpath(p, MAIN)
    ops.append(CommitOperationAdd(path_in_repo=rel, path_or_fileobj=p))
info = api.create_commit(
    repo_id="yxma/gelsight-mini-pretrain", repo_type="dataset",
    operations=ops,
    commit_message=("v9: fix channel order. fota_unlabeled + unit swapped R<->B "
                    "(were BGR-stored). Refresh sample grids + RGB-vs-BGR "
                    "comparison + channel_order_diagnosis.json"))
print(f"main: {info.commit_url}")

# NC: sparsh + faf fixed parquets + refreshed grids + RGB-vs-BGR
ops = []
for sub in ["faf_force_estimation", "sparsh"]:
    for p in sorted(glob.glob(f"{NC}/{sub}/*.parquet")):
        rel = os.path.relpath(p, NC)
        ops.append(CommitOperationAdd(path_in_repo=rel, path_or_fileobj=p))
for p in sorted(glob.glob(f"{NC}/assets/samples_40_*.png")):
    rel = os.path.relpath(p, NC)
    ops.append(CommitOperationAdd(path_in_repo=rel, path_or_fileobj=p))
for p in sorted(glob.glob(f"{NC}/assets/samples_rgb_vs_bgr_*.png")):
    rel = os.path.relpath(p, NC)
    ops.append(CommitOperationAdd(path_in_repo=rel, path_or_fileobj=p))
info = api.create_commit(
    repo_id="yxma/gelsight-mini-pretrain-nc", repo_type="dataset",
    operations=ops,
    commit_message=("v9: fix channel order. faf_force_estimation unconditional "
                    "R<->B swap (BGR-stored). sparsh conditional per-image "
                    "swap (mixed RGB/BGR per Facebook upstream). Refreshed grids"))
print(f"nc:   {info.commit_url}")
PYEOF

echo ""
echo "=== 6. Drop redundant faf_force_estimation/ (duplicate of Sparsh) ==="
python3 << 'PYEOF'
from huggingface_hub import HfApi, CommitOperationDelete
import os, shutil, glob
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
    repo_id="yxma/gelsight-mini-pretrain-nc", repo_type="dataset",
    operations=ops,
    commit_message=("drop faf_force_estimation/ - redundant snapshot of Sparsh "
                    "(same protocol, same indenters, smaller subset)"))
print(f"nc cleanup: {info.commit_url}")
NC = "/media/yxma/Disk1/yuxiang/mini_data_parquet_nc"
if os.path.isdir(f"{NC}/faf_force_estimation"):
    shutil.rmtree(f"{NC}/faf_force_estimation")
for p in glob.glob(f"{NC}/assets/*faf_force*"):
    os.remove(p)
PYEOF

echo ""
echo "=== v9 channel-fix pipeline complete ==="
