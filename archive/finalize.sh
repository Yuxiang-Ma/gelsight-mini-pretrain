#!/bin/bash
# Final pipeline step — run after RTM video extract + sim rerun complete.
# Computes balance metric, regenerates all sample grids, updates README,
# refreshes composition/pies charts, then pushes everything atomic to HF.
set -e

cd /home/yxma/MultimodalData
echo "=== 1. Compute balance metric ==="
python3 compute_balance.py 2>&1 | tee /tmp/balance_report.log | head -50

echo ""
echo "=== 2. Regenerate sample grids for all subsets ==="
python3 make_samples_100.py fota_labeled fota_unlabeled threedcal feats \
    gelslam tactile_tracking real_tactile_mnist feelanyforce \
    unit tacquad sim_tactile_mnist sim_starstruck 2>&1 | head -30

echo ""
echo "=== 3. Refresh composition/pie charts ==="
python3 make_analytical_plots.py 2>&1 | tail -8
python3 make_pie_charts.py 2>&1 | tail -3
python3 make_stats_and_samples.py 2>&1 | tail -8

echo ""
echo "=== 4. Update README ==="
python3 update_readme_final.py 2>&1 | head -10

echo ""
echo "=== 5. Final atomic push to HF ==="
python3 push_final.py 2>&1 | tail -10

echo ""
echo "=== DONE ==="
