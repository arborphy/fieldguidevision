#!/usr/bin/env bash
# Rerun the vector-only experiments (1, 4, 4b, 5, 6, 7) on BioCLIP 2.5, DINOv3 and - if its embeddings exist - YOLO
# (data/embeddings/yolo_strict/embeddings.npz, made by extract_yolo_embeddings.py or your own run in the same layout).
# Outputs -> results/<model>/; both reports are rebuilt. Exp 2 uses stored cluster IDs; exp 3 needs model weights.
set -euo pipefail
cd "$(dirname "$0")"
models="bioclip dinov3"
[ -f ../data/embeddings/yolo_strict/embeddings.npz ] && models="$models yolo"
for m in $models; do
  for e in exp1_nsc exp4b_k_sweep exp4_soft exp5_axes exp6_pca_fidelity exp7_retrieval; do
    echo "== $m $e"; FGV_MODEL=$m python3 "$e.py" > /dev/null
  done
done
python3 build_report.py
python3 build_report_en.py
