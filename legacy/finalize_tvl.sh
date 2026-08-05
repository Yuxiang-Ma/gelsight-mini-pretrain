#!/bin/bash
# Wait for TVL download, unzip the multi-part archive, ingest via
# unified pipeline, push to HF.
set -e
cd /home/yxma/MultimodalData

echo "=== 1. Wait for TVL download to finish ==="
while pgrep -f "huggingface_hub.*TVL-revise" > /dev/null \
   || pgrep -f "snapshot_download" > /dev/null; do
  sleep 60
done
# Also wait for the original task script to fully exit
while pgrep -f "from huggingface_hub import snapshot_download" > /dev/null; do
  sleep 30
done
echo "  TVL download complete"

# Verify all 8 shards present
TVL_DIR="/media/yxma/Disk1/yuxiang/mini_data/markerless/TVL"
n=$(ls $TVL_DIR/tvl_dataset_sharded.* 2>/dev/null | wc -l)
echo "  Found $n shard files"
if [ $n -lt 8 ]; then
  echo "  ERROR: expected 8 shards, found $n"
  exit 1
fi

echo ""
echo "=== 2. Unzip the multi-part archive ==="
cd $TVL_DIR
# Combine multi-part archive into single zip
if [ ! -f tvl_dataset_fixed.zip ] && [ ! -d tvl_dataset ]; then
  zip -s 0 tvl_dataset_sharded.zip --out tvl_dataset_fixed.zip
fi
if [ ! -d tvl_dataset ]; then
  unzip -q tvl_dataset_fixed.zip
fi
echo "  Unzipped to $TVL_DIR/tvl_dataset/"
ls $TVL_DIR/tvl_dataset/

echo ""
echo "=== 3. Run unified ingest pipeline ==="
cd /home/yxma/MultimodalData
python3 ingest_tvl.py 2>&1 | tail -50

echo ""
echo "=== TVL ingest pipeline complete ==="
