#!/bin/bash
# Download an OpenNeuro dataset to GCS via a temporary GCP VM.
#
# Creates a small VM, uses the OpenNeuro CLI (or AWS S3 sync) to download
# the dataset, pushes it to GCS, then deletes the VM.
#
# Usage:
#   bash scripts/gcp_download_openneuro.sh ds004584   # Iowa PD resting-state EEG
#
# The dataset lands at gs://eeg-extraction-data/datasets/<dataset_id>/

set -e

DATASET_ID=${1:?Usage: gcp_download_openneuro.sh <openneuro_dataset_id>}
PROJECT="claude-493017"
ZONE="us-central1-a"
BUCKET="gs://eeg-extraction-data"
MACHINE_TYPE="e2-standard-4"
VM_NAME="download-${DATASET_ID}"
VM_NAME=$(echo "$VM_NAME" | tr '[:upper:]' '[:lower:]' | tr '_' '-' | head -c 60)

export PATH="/opt/homebrew/share/google-cloud-sdk/bin:$PATH"
export GOOGLE_APPLICATION_CREDENTIALS="/Users/neurokinetikz/Code/research/.gcp/claude-493017-ad29d1cd661b.json"

echo "============================================================"
echo "  Download OpenNeuro dataset: $DATASET_ID"
echo "  VM: $VM_NAME ($MACHINE_TYPE)"
echo "  Destination: $BUCKET/datasets/$DATASET_ID/"
echo "  Started: $(date)"
echo "============================================================"

# 1. Create a lightweight VM with a large disk for staging
echo ">>> Creating VM..."
gcloud compute instances create $VM_NAME \
    --project=$PROJECT \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-ssd \
    --image-family=debian-12 \
    --image-project=debian-cloud \
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

# 3. Install tools, download dataset, push to GCS
echo ">>> Downloading $DATASET_ID and pushing to GCS..."
gcloud compute ssh $VM_NAME --zone=$ZONE --command="
    set -e

    # Install Node.js (for OpenNeuro CLI) and AWS CLI (for S3 fallback)
    echo '>>> Installing tools...'
    sudo apt-get update -qq
    sudo apt-get install -y -qq nodejs npm awscli 2>&1 | tail -3

    # Install OpenNeuro CLI
    sudo npm install -g @openneuro/cli 2>&1 | tail -3

    mkdir -p /tmp/data

    # Download via AWS S3 (OpenNeuro mirrors to s3://openneuro.org)
    echo '>>> Downloading $DATASET_ID from OpenNeuro S3 mirror...'
    aws s3 sync --no-sign-request \
        s3://openneuro.org/$DATASET_ID \
        /tmp/data/$DATASET_ID \
        2>&1

    echo '>>> Download complete. Contents:'
    du -sh /tmp/data/$DATASET_ID
    ls /tmp/data/$DATASET_ID/ | head -20

    # Push to GCS
    echo '>>> Pushing to GCS...'
    gcloud storage cp -r /tmp/data/$DATASET_ID $BUCKET/datasets/ 2>&1 | tail -5

    echo '>>> Verifying GCS upload...'
    gcloud storage ls $BUCKET/datasets/$DATASET_ID/ | head -20
    echo '>>> DONE'
" 2>&1

# 4. Delete VM
echo ">>> Deleting VM..."
gcloud compute instances delete $VM_NAME \
    --zone=$ZONE \
    --quiet \
    2>&1

echo "============================================================"
echo "  COMPLETE: $DATASET_ID"
echo "  Location: $BUCKET/datasets/$DATASET_ID/"
echo "  Finished: $(date)"
echo "============================================================"
