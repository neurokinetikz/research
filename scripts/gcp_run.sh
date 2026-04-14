#!/bin/bash
# GCP Extraction Runner
# Spawns a VM from custom image, runs extraction for one dataset, saves to GCS, deletes VM.
#
# Usage:
#   bash scripts/gcp_run.sh eegmmidb              # FOOOF (default)
#   bash scripts/gcp_run.sh eegmmidb "" 1 irasa   # IRASA
#
# Run all datasets with IRASA in parallel:
#   bash scripts/gcp_run.sh eegmmidb "" 1 irasa &
#   bash scripts/gcp_run.sh dortmund "" 1 irasa &
#   bash scripts/gcp_run.sh lemon "" 1 irasa &
#   bash scripts/gcp_run.sh chbmp "" 1 irasa &
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

# Build VM name and extraction args
METHOD_SUFFIX=""
[ "$METHOD" = "irasa" ] && METHOD_SUFFIX="-irasa"
VM_NAME="eeg-${DATASET}${CONDITION:+-$CONDITION}${SESSION:+-ses$SESSION}${METHOD_SUFFIX}"
VM_NAME=$(echo "$VM_NAME" | tr '[:upper:]' '[:lower:]' | tr '_' '-' | head -c 60)

# Map dataset to GCS data path
case $DATASET in
    eegmmidb) DATA_PATHS="eegmmidb";               DISK_GB=100 ;;
    lemon)    DATA_PATHS="lemon_data";              DISK_GB=150 ;;
    dortmund) DATA_PATHS="dortmund_data_dl dortmund_data"; DISK_GB=150 ;;
    chbmp)    DATA_PATHS="CHBMP";                   DISK_GB=100 ;;
    hbn)      DATA_PATHS="hbn_data";                DISK_GB=800 ;;
    hbn_r1)   DATA_PATHS="hbn_data/cmi_bids_R1";   DISK_GB=200 ;;
    hbn_r2)   DATA_PATHS="hbn_data/cmi_bids_R2";   DISK_GB=200 ;;
    hbn_r3)   DATA_PATHS="hbn_data/cmi_bids_R3";   DISK_GB=250 ;;
    hbn_r4)   DATA_PATHS="hbn_data/cmi_bids_R4";   DISK_GB=350 ;;
    hbn_r6)   DATA_PATHS="hbn_data/cmi_bids_R6";   DISK_GB=200 ;;
    tdbrain)       DATA_PATHS="tdbrain/derivatives";                DISK_GB=200 ;;
    tdbrain_val)   DATA_PATHS="tdbrain/adult_validation";         DISK_GB=100 ;;
    *)        echo "Unknown dataset: $DATASET"; exit 1 ;;
esac

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
    cd ~/research && git pull 2>/dev/null || (cd ~ && git clone https://github.com/neurokinetikz/research.git)
    cd ~/research

    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1

    # Direct copy from GCS (gcsfuse drops subjects silently)
    echo '>>> Copying data from GCS...'
    sudo mkdir -p /Volumes/T9
    sudo chmod 777 /Volumes/T9
    for path in $DATA_PATHS; do
        parent=\$(dirname \$path)
        if [ \"\$parent\" != \".\" ]; then
            mkdir -p /Volumes/T9/\$parent
        fi
        echo \"  Copying \$path...\"
        gcloud storage cp -r $BUCKET/\$path /Volumes/T9/\$parent/ 2>&1 | tail -1
    done
    # Also copy dortmund_data for demographics if extracting dortmund
    case $DATASET in dortmund*)
        echo '  Copying dortmund_data (demographics)...'
        gcloud storage cp -r $BUCKET/dortmund_data /Volumes/T9/ 2>&1 | tail -1
    ;; esac
    echo '  Data:'
    ls /Volumes/T9/

    echo '>>> Starting extraction (parallel=28)...'
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    python scripts/run_f0_760_extraction.py $EXTRACT_ARGS --method $METHOD --parallel 28 2>&1

    echo '>>> Pushing results to GCS...'
    if [ '$METHOD' = 'irasa' ]; then
        gcloud storage cp -r exports_irasa_v4 $BUCKET/results/ 2>&1 | tail -3
    else
        gcloud storage cp -r exports_adaptive_v4 $BUCKET/results/ 2>&1 | tail -3
    fi

    echo '>>> DONE'
" 2>&1

# 4. Delete VM
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
