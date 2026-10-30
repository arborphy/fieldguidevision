#!/usr/bin/env bash
# Reproduce all seven minimal experiments (CPU, ~5 min) and rebuild index.html.
set -euo pipefail
cd "$(dirname "$0")"
for e in exp1_nsc exp1b_nsc_viz exp2_enrichment exp3_activation exp4_soft exp5_axes exp6_pca_fidelity exp7_retrieval; do
  echo "== $e"; python3 "$e.py"
done
python3 build_report.py
python3 build_report_en.py
