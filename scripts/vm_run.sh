#!/bin/bash
# Run a Python analysis script on the session VM (sie-sharpen-session).
#
# Usage:
#   bash scripts/vm_run.sh <script_path_relative_to_research> [args...]
#
# Examples:
#   bash scripts/vm_run.sh scripts/sie_perionset_multistream.py
#   bash scripts/vm_run.sh scripts/sie_composite_detector_v2.py --subject sub-010249
#
# Behavior:
#   1. rsyncs scripts/ and lib/ to the VM (via IAP tunnel, incremental)
#   2. Launches the script on the VM via nohup, detaches, returns immediately.
#   3. Writes a log to ~/research/logs/<script-base>-<timestamp>.log on the VM
#      and prints a poll snippet the caller can use to wait for completion.
#   4. Poll mode: if $VM_POLL is set, this script blocks until the remote process
#      exits, then rsyncs outputs/schumann/ back.
#
# Data is mounted at /mnt/eeg-data/T9 on the VM with /Volumes/T9 symlink.
#
# Notes:
#   - Uses --tunnel-through-iap because direct SSH is firewalled.
#   - SIE_WORKERS is forwarded from local env (default 28).

set -e

VM=sie-sharpen-session
ZONE=us-central1-a
PROJECT=claude-493017
IAP="--tunnel-through-iap"

export PATH="/opt/homebrew/share/google-cloud-sdk/bin:$PATH"
export GOOGLE_APPLICATION_CREDENTIALS="/Users/neurokinetikz/Code/research/.gcp/claude-493017-ad29d1cd661b.json"

SCRIPT=${1:?"Usage: bash scripts/vm_run.sh <script_path> [args...]"}
shift
ARGS="$@"

cd "$(dirname "$0")/.."

SCRIPT_BASE=$(basename "$SCRIPT" .py)
TS=$(date +%Y%m%d-%H%M%S)
REMOTE_LOG="logs/${SCRIPT_BASE}-${TS}.log"
REMOTE_DONE="logs/${SCRIPT_BASE}-${TS}.done"
VM_WORKERS="${SIE_WORKERS:-28}"
# Forward additional env vars to the remote process
EXTRA_ENV=""
for v in LEMON_COHORT QUALITY_COL QUALITY_TOP_Q N_SUBJECTS \
         OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS \
         SCOPE; do
  if [ -n "${!v}" ]; then
    EXTRA_ENV="$EXTRA_ENV $v=${!v}"
  fi
done

echo ">>> Syncing scripts/ and lib/ to VM (IAP)..."
gcloud compute scp --zone=$ZONE $IAP --recurse scripts/*.py $VM:~/research/scripts/ 2>&1 | tail -2
gcloud compute scp --zone=$ZONE $IAP --recurse lib/*.py    $VM:~/research/lib/     2>&1 | tail -2

echo ">>> Launching (detached) on VM: $SCRIPT $ARGS"
gcloud compute ssh $VM --zone=$ZONE $IAP --command="
  mkdir -p ~/research/logs
  cd ~/research
  nohup bash -c 'source ~/eeg_env/bin/activate && SIE_WORKERS=$VM_WORKERS $EXTRA_ENV python3 -u $SCRIPT $ARGS; echo \$? > $REMOTE_DONE' > $REMOTE_LOG 2>&1 &
  echo LAUNCHED pid=\$!
  sleep 2
  head -5 $REMOTE_LOG 2>/dev/null || true
" 2>&1

echo ""
echo ">>> Remote log:  $REMOTE_LOG"
echo ">>> Done marker: $REMOTE_DONE (exit code on completion)"
echo ""
echo "To tail the log:"
echo "  gcloud compute ssh $VM --zone=$ZONE $IAP --command='tail -f ~/research/$REMOTE_LOG'"
echo ""
echo "To check completion:"
echo "  gcloud compute ssh $VM --zone=$ZONE $IAP --command='test -f ~/research/$REMOTE_DONE && cat ~/research/$REMOTE_DONE'"
echo ""
echo "To pull outputs back when done:"
echo "  gcloud compute scp --zone=$ZONE $IAP --recurse $VM:~/research/outputs/schumann/ outputs/"

# Poll mode: wait for completion, then pull outputs.
if [ -n "$VM_POLL" ]; then
  echo ""
  echo ">>> VM_POLL set — waiting for remote completion..."
  while true; do
    STATUS=$(gcloud compute ssh $VM --zone=$ZONE $IAP --command="cat ~/research/$REMOTE_DONE 2>/dev/null" 2>/dev/null || true)
    if [ -n "$STATUS" ]; then
      echo ">>> Remote exit code: $STATUS"
      break
    fi
    sleep 20
  done
  echo ">>> Tail of log:"
  gcloud compute ssh $VM --zone=$ZONE $IAP --command="tail -30 ~/research/$REMOTE_LOG" 2>&1 | tail -30
  echo ">>> Syncing outputs back to local..."
  gcloud compute scp --zone=$ZONE $IAP --recurse $VM:~/research/outputs/schumann/ outputs/ 2>&1 | tail -5
  echo ">>> Done."
fi
