#!/bin/bash
# Launch the Q4 audit batch on sie-sharpen-session VM.
#
# Two batches:
#   1. HBN R5/R7/R8/R10/R11 Q4 coupling re-runs (5 jobs, ~5 min each)
#   2. Window enrichment Q4 re-runs (14 datasets, ~30 min each)
#
# All jobs are launched detached. Logs at ~/research/logs/ on VM.

set -e
cd "$(dirname "$0")/.."

echo ">>> BATCH 1: HBN R5/R7/R8/R10/R11 Q4 coupling re-runs"
for r in R5 R7 R8 R10 R11; do
  echo "  Launching: sie_beta_peak_iaf_coupling_composite.py --cohort hbn_$r --q4"
  bash scripts/vm_run.sh scripts/sie_beta_peak_iaf_coupling_composite.py --cohort hbn_$r --q4 2>&1 | grep -E "^>>>" | head -3
  echo ""
done

echo ""
echo ">>> BATCH 2: Window enrichment Q4 re-runs (14 dataset×conditions)"
declare -a WIN_JOBS=(
  "--dataset lemon"
  "--dataset lemon --condition EO"
  "--dataset hbn --release R1"
  "--dataset hbn --release R2"
  "--dataset hbn --release R3"
  "--dataset hbn --release R4"
  "--dataset hbn --release R5"
  "--dataset hbn --release R6"
  "--dataset hbn --release R7"
  "--dataset hbn --release R8"
  "--dataset hbn --release R10"
  "--dataset hbn --release R11"
  "--dataset chbmp"
  "--dataset dortmund"
)
for args in "${WIN_JOBS[@]}"; do
  echo "  Launching: sie_window_enrichment.py $args --q4 --window 20"
  bash scripts/vm_run.sh scripts/sie_window_enrichment.py $args --q4 --window 20 2>&1 | grep -E "^>>>" | head -3
  echo ""
done

echo ""
echo ">>> All Q4 jobs launched. Logs at ~/research/logs/ on VM."
echo ">>> Monitor with:"
echo "  gcloud compute ssh sie-sharpen-session --zone=us-central1-a --tunnel-through-iap \\"
echo "    --command='ls -t ~/research/logs/ | head -20 && echo --- && ls ~/research/logs/*.done 2>/dev/null | wc -l'"
