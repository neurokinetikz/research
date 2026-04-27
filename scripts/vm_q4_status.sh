#!/bin/bash
# Check status of Q4 audit batch jobs on VM.
#
# Usage:
#   bash scripts/vm_q4_status.sh              # quick status table
#   bash scripts/vm_q4_status.sh --tail       # tail latest log
#   bash scripts/vm_q4_status.sh --pull       # rsync outputs back when done

set -e
export PATH="/opt/homebrew/share/google-cloud-sdk/bin:$PATH"
export GOOGLE_APPLICATION_CREDENTIALS="/Users/neurokinetikz/Code/research/.gcp/claude-493017-ad29d1cd661b.json"
VM=sie-sharpen-session
ZONE=us-central1-a
IAP="--tunnel-through-iap"

if [ "$1" = "--tail" ]; then
  gcloud compute ssh $VM --zone=$ZONE $IAP --command='ls -t ~/research/logs/*.log | head -1 | xargs tail -30'
  exit 0
fi

if [ "$1" = "--pull" ]; then
  echo ">>> Syncing VM outputs back..."
  gcloud compute scp --zone=$ZONE $IAP --recurse \
    $VM:~/research/outputs/sie_window_enrichment_q4_*.csv outputs/ 2>&1 | tail -3
  gcloud compute scp --zone=$ZONE $IAP --recurse \
    $VM:~/research/outputs/schumann/images/psd_timelapse/ outputs/schumann/images/ 2>&1 | tail -3
  echo ">>> Done."
  exit 0
fi

# Default: status table
gcloud compute ssh $VM --zone=$ZONE $IAP --command='
echo "=== HBN coupling Q4 jobs ==="
for f in $(ls -t ~/research/logs/sie_beta_peak_iaf_coupling_composite-*.log 2>/dev/null); do
  done_file=${f%.log}.done
  status="RUNNING"
  if [ -f $done_file ]; then status="DONE (exit $(cat $done_file))"; fi
  cohort=$(grep -E "^Cohort:" $f | head -1 | awk -F" composite" "{print \$1}" | sed "s/Cohort: //")
  rho=$(grep "SR × β" $f -A 3 | grep "Spearman" | tail -1 | awk -F"=" "{print \$2}" | awk -F"p =" "{print \$1}" | xargs)
  printf "  %-30s  %-25s  ρ_SR×β = %s\n" "$(basename $f .log)" "$status" "$rho"
done
echo ""
echo "=== Window enrichment Q4 jobs ==="
for f in $(ls -t ~/research/logs/sie_window_enrichment-*.log 2>/dev/null); do
  done_file=${f%.log}.done
  status="RUNNING"
  if [ -f $done_file ]; then status="DONE (exit $(cat $done_file))"; fi
  label=$(grep -E "Window Enrichment:" $f | head -1 | awk -F"Window Enrichment: " "{print \$2}")
  progress=$(grep -E "\[[0-9]+/" $f | tail -1)
  printf "  %-40s  %-22s  %s\n" "$(basename $f .log)" "$status" "$label"
  if [ -n "$progress" ]; then printf "    progress: %s\n" "$progress"; fi
done
echo ""
echo "Total jobs running: $(ls ~/research/logs/*.log 2>/dev/null | wc -l) launched, $(ls ~/research/logs/*.done 2>/dev/null | wc -l) done"
'
