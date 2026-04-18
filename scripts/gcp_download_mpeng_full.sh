#!/bin/bash
# Download full MultiPENG archive on a GCP VM, extract EEG + metadata to the
# eeg-data-disk persistent disk. Skips the large OBS video files (H.264 .mp4)
# to save disk space.
#
# Result: /Volumes/T9/mpeng/ replaced with canonical per-subject structure:
#   /Volumes/T9/mpeng/Questionnaire/{participants,submissions}.csv
#   /Volumes/T9/mpeng/Samples/<id>/EEG/*.csv
#
# Usage: bash scripts/gcp_download_mpeng_full.sh [mode]
#   mode: "eeg" (default, ~1-2 GB) | "nonvideo" (EEG+EYE+HR+OpenFace+XBOX ~3-5 GB)

set -e

MODE=${1:-"eeg"}

PROJECT="claude-493017"
ZONE="us-central1-a"
VM_NAME="eeg-mpeng-download"
DATA_DISK="eeg-data-disk"
IMAGE="eeg-extraction-image-100gb"
KAGGLE_URL="https://www.kaggle.com/api/v1/datasets/download/ammarrashed23/multimodal-player-engagement"

export PATH="/opt/homebrew/share/google-cloud-sdk/bin:$PATH"
export GOOGLE_APPLICATION_CREDENTIALS="/Users/neurokinetikz/Code/research/.gcp/claude-493017-ad29d1cd661b.json"

echo "============================================================"
echo "  MultiPENG full-archive download (VM-side)"
echo "  Mode: $MODE (eeg=EEG only; nonvideo=all except OBS video)"
echo "  Started: $(date)"
echo "============================================================"

# 1. Create VM with disk RW. e2-standard-4 (4 vCPU, 16 GB RAM) is enough for
#    curl + unzip; boot disk 100GB to accommodate the temp archive + extraction.
echo ">>> Creating download VM..."
gcloud compute instances create $VM_NAME \
    --project=$PROJECT \
    --zone=$ZONE \
    --machine-type=e2-standard-4 \
    --boot-disk-size=200GB \
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

# 3. Build the extraction filter per mode
if [ "$MODE" = "nonvideo" ]; then
    # Keep everything except OBS video files (.mp4)
    EXTRACT_PATTERN='*'
    EXCLUDE_PATTERN='-x "Samples/*/OBS/*.mp4"'
else
    # EEG-only: just Samples/*/EEG/ and Questionnaire/
    EXTRACT_PATTERN='"Samples/*/EEG/*" "Questionnaire/*"'
    EXCLUDE_PATTERN=''
fi

echo ">>> Downloading + extracting on VM..."
gcloud compute ssh $VM_NAME --zone=$ZONE --command="
    set -e
    # eeg-extraction-image-100gb lacks 'unzip' by default
    sudo apt-get install -y unzip 2>&1 | tail -2
    sudo mkdir -p /Volumes
    sudo mount /dev/disk/by-id/google-eeg-data /Volumes
    echo '>>> Disk mounted. Free space:'
    df -h /Volumes | tail -1

    cd /tmp
    echo '>>> Downloading MultiPENG archive (~13 GB)...'
    time curl -L -o mpeng.zip '$KAGGLE_URL'
    echo '>>> Download complete. Size:'
    ls -lh mpeng.zip

    echo '>>> Inspecting archive top-level structure...'
    unzip -l mpeng.zip 2>/dev/null | head -20

    # Backup existing flat structure then replace
    if [ -d /Volumes/T9/mpeng ] && [ ! -d /Volumes/T9/mpeng/Samples ]; then
        echo '>>> Backing up existing flat /Volumes/T9/mpeng → /Volumes/T9/mpeng_flat_backup'
        sudo mv /Volumes/T9/mpeng /Volumes/T9/mpeng_flat_backup
    fi
    sudo mkdir -p /Volumes/T9/mpeng

    echo '>>> Extracting ($MODE mode)...'
    sudo unzip -q -o mpeng.zip $EXTRACT_PATTERN $EXCLUDE_PATTERN -d /Volumes/T9/mpeng/

    echo '>>> Cleanup temp zip...'
    rm -f mpeng.zip

    echo '>>> Verification:'
    echo 'Questionnaire/:'
    ls /Volumes/T9/mpeng/Questionnaire/ 2>/dev/null || echo '  (missing)'
    echo ''
    echo 'Samples/ (first 5 subjects):'
    ls /Volumes/T9/mpeng/Samples/ 2>/dev/null | head -5
    echo '  ...total subjects:' \$(ls /Volumes/T9/mpeng/Samples/ 2>/dev/null | wc -l)
    echo ''
    echo 'Size:'
    du -sh /Volumes/T9/mpeng/ 2>/dev/null
    echo ''
    echo 'Sample EEG file counts (first 3 subjects):'
    for s in \$(ls /Volumes/T9/mpeng/Samples/ 2>/dev/null | head -3); do
        n=\$(ls /Volumes/T9/mpeng/Samples/\$s/EEG/ 2>/dev/null | wc -l)
        echo \"  \$s: \$n EEG files\"
    done

    sudo umount /Volumes
" 2>&1

# 4. Delete VM (disk persists)
echo ">>> Deleting download VM..."
gcloud compute instances delete $VM_NAME \
    --zone=$ZONE \
    --quiet \
    2>&1

echo "============================================================"
echo "  MultiPENG DOWNLOAD COMPLETE"
echo "  Disk now has canonical /Volumes/T9/mpeng/ structure"
echo "  Finished: $(date)"
echo "============================================================"
