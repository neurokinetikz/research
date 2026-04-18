#!/bin/bash
# Sync discovery-cohort datasets from GCS to the eeg-data-disk persistent disk.
#
# Prerequisite: discovery folders already uploaded to gs://eeg-extraction-data/T9/
# via `gcloud storage cp -r /Volumes/T9/<ds> gs://eeg-extraction-data/T9/` for
# each of: muse, epoc, insight, PhySF, mpeng, vep.
#
# This script:
#   1. Creates a temporary e2-standard-2 VM with eeg-data-disk attached read-write
#   2. Mounts the disk and copies discovery datasets from GCS to /Volumes/T9/
#   3. Detaches and deletes the sync VM (disk persists with new data)
#
# Usage: bash scripts/gcp_sync_discovery_to_disk.sh

set -e

PROJECT="claude-493017"
ZONE="us-central1-a"
BUCKET="gs://eeg-extraction-data/T9"
VM_NAME="eeg-discovery-sync"
DATA_DISK="eeg-data-disk"
IMAGE="eeg-extraction-image-100gb"

export PATH="/opt/homebrew/share/google-cloud-sdk/bin:$PATH"
export GOOGLE_APPLICATION_CREDENTIALS="/Users/neurokinetikz/Code/research/.gcp/claude-493017-ad29d1cd661b.json"

echo "============================================================"
echo "  Discovery dataset sync: GCS → eeg-data-disk"
echo "  Datasets: muse, epoc, insight, PhySF, mpeng, vep"
echo "  Started: $(date)"
echo "============================================================"

# 1. Create sync VM with eeg-data-disk attached read-write
echo ">>> Creating sync VM with disk attached read-write..."
gcloud compute instances create $VM_NAME \
    --project=$PROJECT \
    --zone=$ZONE \
    --machine-type=e2-standard-2 \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd \
    --image=$IMAGE \
    --image-project=$PROJECT \
    --scopes=storage-full \
    --disk=name=$DATA_DISK,device-name=eeg-data,mode=rw \
    2>&1

# 2. Wait for SSH
echo ">>> Waiting for VM to accept SSH..."
for i in $(seq 1 30); do
    if gcloud compute ssh $VM_NAME --zone=$ZONE --command="echo ready" 2>/dev/null; then
        break
    fi
    sleep 5
done

# 3. Mount disk read-write, copy discovery datasets from GCS
echo ">>> Syncing GCS → disk..."
gcloud compute ssh $VM_NAME --zone=$ZONE --command="
    set -e
    sudo mkdir -p /Volumes
    sudo mount /dev/disk/by-id/google-eeg-data /Volumes
    echo '>>> Disk mounted:'
    df -h /Volumes | tail -1
    ls /Volumes/T9/ | head -20
    echo ''
    for ds in muse epoc insight PhySF mpeng vep; do
        echo \">>> Copying \$ds...\"
        sudo gcloud storage cp -r $BUCKET/\$ds /Volumes/T9/ 2>&1 | tail -3
    done
    echo ''
    echo '>>> Verification — discovery folders on disk:'
    for ds in muse epoc insight PhySF mpeng vep; do
        count=\$(ls /Volumes/T9/\$ds 2>/dev/null | wc -l)
        size=\$(du -sh /Volumes/T9/\$ds 2>/dev/null | cut -f1)
        echo \"  \$ds: \$count files, \$size\"
    done
    sudo umount /Volumes
" 2>&1

# 4. Delete sync VM (disk persists)
echo ">>> Deleting sync VM (disk persists with discovery data)..."
gcloud compute instances delete $VM_NAME \
    --zone=$ZONE \
    --quiet \
    2>&1

echo "============================================================"
echo "  SYNC COMPLETE"
echo "  Disk $DATA_DISK now contains discovery datasets"
echo "  Ready for: bash scripts/gcp_run_all_sie.sh"
echo "  Finished: $(date)"
echo "============================================================"
