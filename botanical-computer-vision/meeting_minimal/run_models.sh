#!/usr/bin/env bash
# Rerun the vector-only experiments (1, 4, 5, 6, 7) on BioCLIP 2.5 and DINOv3 raw vectors -> results/<model>/.
# Exp 2 already covers all three models via cluster IDs; exp 3 needs model weights (EfficientNet only here).
set -euo pipefail
cd "$(dirname "$0")"
for m in bioclip dinov3; do
  for e in exp1_nsc exp4_soft exp5_axes exp6_pca_fidelity exp7_retrieval; do
    echo "== $m $e"; FGV_MODEL=$m python3 "$e.py" > /dev/null
  done
done
python3 build_report.py
python3 build_report_en.py
