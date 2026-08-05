#!/bin/bash
# Wait for Touch-and-Go gdown to finish, run ingest, push to HF.
set -e
cd /home/yxma/MultimodalData

echo "=== 1. Wait for Touch-and-Go gdown to finish ==="
while pgrep -f "gdown.*1NDasyshDCL9aaQzxjn_-Q5MBURRT360B" > /dev/null \
   || pgrep -f "gdown --folder" > /dev/null; do
  sleep 60
done
echo "  gdown complete"

TG_BASE="/media/yxma/Disk1/yuxiang/mini_data/markerless/TouchAndGo"
echo "  layout:"
ls $TG_BASE | head -20
n_mp4=$(find $TG_BASE -name 'gelsight.mp4' 2>/dev/null | wc -l)
echo "  Found $n_mp4 gelsight.mp4 files"

if [ $n_mp4 -lt 1 ]; then
  echo "  ERROR: no gelsight.mp4 found"
  exit 1
fi

echo ""
echo "=== 2. Run unified ingest pipeline ==="
python3 ingest_touchandgo.py 2>&1 | tail -50

echo ""
echo "=== Touch-and-Go ingest pipeline complete ==="
