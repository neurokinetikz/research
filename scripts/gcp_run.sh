#!/bin/bash
# GCP Extraction Runner
# Spawns a VM from custom image with persistent data disk, runs extraction, saves to GCS, deletes VM.
#
# The persistent data disk (eeg-data-disk, 1TB) contains all raw EEG datasets
# pre-loaded. VMs attach it read-only -- no data download needed.
#
# Usage:
#   bash scripts/gcp_run.sh eegmmidb              # FOOOF (default)
#   bash scripts/gcp_run.sh eegmmidb "" 1 irasa   # IRASA
#
# Run all datasets in parallel (8 VMs, 256 CPU quota):
#   bash scripts/gcp_run.sh tdbrain "" 1 fooof &
#   bash scripts/gcp_run.sh tdbrain "" 1 irasa &
#   bash scripts/gcp_run.sh lemon "" 1 fooof &
#   bash scripts/gcp_run.sh lemon "" 1 irasa &
#   bash scripts/gcp_run.sh dortmund "" 1 fooof &
#   bash scripts/gcp_run.sh dortmund "" 1 irasa &
#   bash scripts/gcp_run.sh hbn_r1 "" 1 fooof &
#   bash scripts/gcp_run.sh hbn_r2 "" 1 fooof &
#   wait

set -e

DATASET=${1:?Usage: gcp_run.sh <dataset> [condition] [session] [method]}
CONDITION=${2:-""}
SESSION=${3:-"1"}
METHOD=${4:-"fooof"}
PROJECT="claude-493017"
ZONE="us-central1-a"
BUCKET="gs://eeg-extraction-data"
MACHINE_TYPE="c2d-standard-32"
IMAGE="eeg-extraction-image-100gb"
DATA_DISK="eeg-data-disk"

# Build VM name and extraction args
METHOD_SUFFIX=""
[ "$METHOD" = "irasa" ] && METHOD_SUFFIX="-irasa"
VM_NAME="eeg-${DATASET}${CONDITION:+-$CONDITION}${SESSION:+-ses$SESSION}${METHOD_SUFFIX}"
VM_NAME=$(echo "$VM_NAME" | tr '[:upper:]' '[:lower:]' | tr '_' '-' | head -c 60)

# Build extraction args
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

EXPORT_DIR="exports_adaptive_v4"
[ "$METHOD" = "irasa" ] && EXPORT_DIR="exports_irasa_v4"

echo "============================================================"
echo "  GCP Run: $DATASET $CONDITION ses-$SESSION method=$METHOD"
echo "  VM: $VM_NAME ($MACHINE_TYPE)"
echo "  Data disk: $DATA_DISK (read-only)"
echo "  Started: $(date)"
echo "============================================================"

# 1. Create VM with persistent data disk attached read-only
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

# 3. Run extraction
echo ">>> Running extraction..."
gcloud compute ssh $VM_NAME --zone=$ZONE --command="
    source ~/eeg_env/bin/activate

    # Get latest code
    cd ~/research && git pull 2>/dev/null || (cd ~ && git clone https://github.com/neurokinetikz/research.git)
    cd ~/research

    # Mount persistent data disk at /Volumes/T9
    sudo mkdir -p /Volumes/T9
    sudo mount -o ro /dev/disk/by-id/google-eeg-data /Volumes/T9
    echo '>>> Data disk mounted:'
    ls /Volumes/T9/
    echo '  Disk usage:'
    df -h /Volumes/T9 | tail -1

    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1

    echo '>>> Starting extraction (parallel=28)...'
    python scripts/run_f0_760_extraction.py $EXTRACT_ARGS --method $METHOD --parallel 28 2>&1

    echo '>>> Pushing results to GCS...'
    if [ '$METHOD' = 'irasa' ]; then
        gcloud storage cp -r exports_irasa_v4 $BUCKET/results/ 2>&1 | tail -3
    else
        gcloud storage cp -r exports_adaptive_v4 $BUCKET/results/ 2>&1 | tail -3
    fi

    echo '>>> DONE'
" 2>&1

# 4. Delete VM (disk persists)
echo ">>> Deleting VM..."
gcloud compute instances delete $VM_NAME \
    --zone=$ZONE \
    --quiet \
    2>&1

echo "============================================================"
echo "  COMPLETE: $DATASET $CONDITION ses-$SESSION method=$METHOD"
echo "  Results: $BUCKET/results/$EXPORT_DIR/"
echo "  Finished: $(date)"
echo "============================================================"
