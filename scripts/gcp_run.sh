#!/bin/bash
# GCP Extraction Runner
# Spawns a VM from custom image, runs extraction for one dataset, saves to GCS, deletes VM.
#
# Usage:
#   bash scripts/gcp_run.sh eegmmidb
#   bash scripts/gcp_run.sh dortmund
#
# Run multiple datasets in parallel:
#   bash scripts/gcp_run.sh eegmmidb &
#   bash scripts/gcp_run.sh dortmund &
#   bash scripts/gcp_run.sh lemon &
#   bash scripts/gcp_run.sh chbmp &
#   wait

set -e

DATASET=${1:?Usage: gcp_run.sh <dataset> [condition] [session]}
CONDITION=${2:-""}
SESSION=${3:-"1"}
PROJECT="claude-493017"
ZONE="us-central1-a"
BUCKET="gs://eeg-extraction-data"
MACHINE_TYPE="c2d-standard-32"
IMAGE="eeg-extraction-image"

# Build VM name and extraction args
VM_NAME="eeg-${DATASET}${CONDITION:+-$CONDITION}${SESSION:+-ses$SESSION}"
VM_NAME=$(echo "$VM_NAME" | tr '[:upper:]' '[:lower:]' | tr '_' '-' | head -c 60)

# Map dataset to GCS data path
case $DATASET in
    eegmmidb) DATA_PATHS="eegmmidb";               DISK_GB=50 ;;
    lemon)    DATA_PATHS="lemon_data";              DISK_GB=100 ;;
    dortmund) DATA_PATHS="dortmund_data_dl dortmund_data"; DISK_GB=100 ;;
    chbmp)    DATA_PATHS="CHBMP";                   DISK_GB=50 ;;
    hbn)      DATA_PATHS="hbn_data";                DISK_GB=500 ;;
    *)        echo "Unknown dataset: $DATASET"; exit 1 ;;
esac

# Build extraction args
EXTRACT_ARGS="--dataset $DATASET"
[ -n "$CONDITION" ] && EXTRACT_ARGS="$EXTRACT_ARGS --condition $CONDITION"
[ "$SESSION" != "1" ] && EXTRACT_ARGS="$EXTRACT_ARGS --session $SESSION"

export PATH="/opt/homebrew/share/google-cloud-sdk/bin:$PATH"
export GOOGLE_APPLICATION_CREDENTIALS="/Users/neurokinetikz/Code/research/.gcp/claude-493017-ad29d1cd661b.json"

echo "============================================================"
echo "  GCP Run: $DATASET $CONDITION ses-$SESSION"
echo "  VM: $VM_NAME ($MACHINE_TYPE)"
echo "  Started: $(date)"
echo "============================================================"

# 1. Create VM from custom image (deps pre-installed)
echo ">>> Creating VM from custom image..."
gcloud compute instances create $VM_NAME \
    --project=$PROJECT \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --boot-disk-size=${DISK_GB}GB \
    --boot-disk-type=pd-ssd \
    --image=$IMAGE \
    --image-project=$PROJECT \
    --scopes=storage-full \
    2>&1

# 2. Wait for SSH
echo ">>> Waiting for VM..."
for i in $(seq 1 30); do
    if gcloud compute ssh $VM_NAME --zone=$ZONE --command="echo ready" 2>/dev/null; then
        break
    fi
    sleep 5
done

# 3. Run extraction
echo ">>> Running extraction..."
gcloud compute ssh $VM_NAME --zone=$ZONE --command="
    source ~/eeg_env/bin/activate

    # Get latest code
    cd ~/research && git pull 2>/dev/null || (cd ~ && git clone git@github.com:neurokinetikz/research.git)
    cd ~/research

    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1

    # Pull data from GCS into /Volumes/T9 (matching hardcoded paths)
    echo '>>> Pulling data...'
    sudo mkdir -p /Volumes/T9
    sudo chmod 777 /Volumes/T9
    for path in $DATA_PATHS; do
        echo \"  Pulling \$path...\"
        gcloud storage cp -r $BUCKET/\$path /Volumes/T9/ 2>&1 | tail -1
    done
    echo '  Data:'
    ls /Volumes/T9/

    echo '>>> Starting extraction (parallel=28)...'
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    python scripts/run_f0_760_extraction.py $EXTRACT_ARGS --parallel 28 2>&1

    echo '>>> Pushing results to GCS...'
    gcloud storage cp -r exports_adaptive_v4 $BUCKET/results/ 2>&1 | tail -3

    echo '>>> DONE'
" 2>&1

# 4. Delete VM
echo ">>> Deleting VM..."
gcloud compute instances delete $VM_NAME \
    --zone=$ZONE \
    --quiet \
    2>&1

echo "============================================================"
echo "  COMPLETE: $DATASET $CONDITION ses-$SESSION"
echo "  Results: $BUCKET/results/exports_adaptive_v4/"
echo "  Finished: $(date)"
echo "============================================================"
