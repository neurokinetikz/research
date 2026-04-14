#!/bin/bash
# Run the IRASA method-independence replication (Paper Part V).
#
# This re-runs the full analysis pipeline on IRASA-extracted peaks
# (exports_irasa_v4) to verify method-independence of key findings.
#
# Usage:
#   bash scripts/run_irasa_replication.sh
#
# Outputs to: outputs/f0_760_reanalysis/ (overwrites FOOOF results,
#             so run FOOOF analyses first or save outputs separately)

set -e

echo "=== IRASA Method-Independence Replication ==="
echo "Peak base: exports_irasa_v4"
echo ""

python scripts/run_all_f0_760_analyses.py \
    --step all \
    --peak-base exports_irasa_v4

echo ""
echo "=== IRASA replication complete ==="
echo "Compare outputs/f0_760_reanalysis/ against FOOOF results"
