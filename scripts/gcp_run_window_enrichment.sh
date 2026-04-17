#!/bin/bash
# GCP runner for SIE window enrichment analysis
# Spawns VM, runs window enrichment on a dataset, pushes CSV to GCS, deletes VM.
#
# Usage:
#   bash scripts/gcp_run_window_enrichment.sh <dataset> [condition] [session]
#
# Example:
#   bash scripts/gcp_run_window_enrichment.sh lemon
#   bash scripts/gcp_run_window_enrichment.sh hbn_r4

set -e

DATASET=${1:?Usage: $0 <dataset> [condition] [session]}
CONDITION=${2:-""}
SESSION=${3:-"1"}
PROJECT="claude-493017"
ZONE="us-central1-a"
BUCKET="gs://eeg-extraction-data"
MACHINE_TYPE="c2d-standard-32"
IMAGE="eeg-extraction-image-100gb"
DATA_DISK="eeg-data-disk"

VM_NAME="eeg-winenr-${DATASET}${CONDITION:+-$CONDITION}${SESSION:+-ses$SESSION}"
VM_NAME=$(echo "$VM_NAME" | tr '[:upper:]' '[:lower:]' | tr '_' '-' | head -c 60)

# Build extraction args for window enrichment runner
case $DATASET in
    hbn_r[0-9])
        RELEASE=$(echo $DATASET | sed 's/hbn_r/R/')
        EXTRACT_ARGS="--dataset hbn --release $RELEASE"
        ;;
    *)
        EXTRACT_ARGS="--dataset $DATASET"
        [ -n "$CONDITION" ] && EXTRACT_ARGS="$EXTRACT_ARGS --condition $CONDITION"
        [ "$SESSION" != "1" ] && EXTRACT_ARGS="$EXTRACT_ARGS --session $SESSION"
        ;;
esac

export PATH="/opt/homebrew/share/google-cloud-sdk/bin:$PATH"
export GOOGLE_APPLICATION_CREDENTIALS="/Users/neurokinetikz/Code/research/.gcp/claude-493017-ad29d1cd661b.json"

echo "============================================================"
echo "  Window Enrichment: $DATASET $CONDITION ses-$SESSION"
echo "  VM: $VM_NAME ($MACHINE_TYPE)"
echo "  Started: $(date)"
echo "============================================================"

# 1. Create VM
echo ">>> Creating VM..."
gcloud compute instances create $VM_NAME \
    --project=$PROJECT \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd \
    --image=$IMAGE \
    --image-project=$PROJECT \
    --scopes=storage-full \
    --disk=name=$DATA_DISK,device-name=eeg-data,mode=ro \
    2>&1

# 2. Wait for SSH
echo ">>> Waiting for VM..."
for i in $(seq 1 30); do
    if gcloud compute ssh $VM_NAME --zone=$ZONE --command="echo ready" 2>/dev/null; then
        break
    fi
    sleep 5
done

# 3. Run analysis
echo ">>> Running window enrichment..."
gcloud compute ssh $VM_NAME --zone=$ZONE --command="
    source ~/eeg_env/bin/activate
    cd ~/research && git pull 2>/dev/null || (cd ~ && git clone https://github.com/neurokinetikz/research.git)
    cd ~/research

    sudo mkdir -p /Volumes
    sudo mount -o ro /dev/disk/by-id/google-eeg-data /Volumes
    echo '>>> Data disk mounted:'
    ls /Volumes/T9/ | head -5

    # Need SIE events locally too -- pull from GCS
    mkdir -p exports_sie
    gcloud storage cp -r ${BUCKET}/results/exports_sie/* exports_sie/ 2>&1 | tail -3

    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1

    echo '>>> Starting window enrichment...'
    python scripts/sie_window_enrichment.py $EXTRACT_ARGS --window 20 --buffer 5 --min-events 3 2>&1

    echo '>>> Pushing results to GCS...'
    gcloud storage cp outputs/sie_window_enrichment_*.csv ${BUCKET}/results/sie_window_enrichment/ 2>&1 | tail -3

    echo '>>> DONE'
" 2>&1

# 4. Delete VM
echo ">>> Deleting VM..."
gcloud compute instances delete $VM_NAME --zone=$ZONE --quiet 2>&1

echo "============================================================"
echo "  COMPLETE: $DATASET $CONDITION ses-$SESSION"
echo "  Finished: $(date)"
echo "============================================================"
