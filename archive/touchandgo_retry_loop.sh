#!/bin/bash
# Retry Touch-and-Go gdown periodically. Google Drive rate-limits public
# folder downloads after N file fetches; the limit typically resets after
# 1–24 hours. This loop tries every 30 minutes for the next 18 hours.
set -e
cd /media/yxma/Disk1/yuxiang/mini_data/markerless/TouchAndGo

end=$(($(date +%s) + 18 * 3600))   # 18 hours from now
attempt=0
while [ $(date +%s) -lt $end ]; do
  attempt=$((attempt + 1))
  echo "[$(date +%H:%M)] gdown attempt #$attempt"
  before=$(find . -name 'gelsight.mp4' 2>/dev/null | wc -l)
  gdown --folder --continue \
    https://drive.google.com/drive/folders/1NDasyshDCL9aaQzxjn_-Q5MBURRT360B \
    > /tmp/touchandgo_attempt_$attempt.log 2>&1 || true
  after=$(find . -name 'gelsight.mp4' 2>/dev/null | wc -l)
  delta=$((after - before))
  echo "[$(date +%H:%M)] attempt #$attempt: $before → $after gelsight.mp4 files (+$delta)"
  if [ $delta -eq 0 ]; then
    echo "  no progress, sleeping 30 min"
    sleep 1800
  else
    echo "  progress made, sleeping 5 min before next attempt"
    sleep 300
  fi
done
echo "Done retry loop"
