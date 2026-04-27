#!/bin/bash
# Launch one source-space worker VM for a given cohort.
#
# Each worker: c2d-standard-32 with eeg-data-disk attached RO.
# Runs source-space for its cohort, pushes results to GCS, self-terminates.
#
# Usage: bash scripts/launch_source_worker.sh <COHORT> <GCS_SUBDIR> [EXTRACT_NAME]
#   COHORT: cohort name passed to --cohort (e.g. tdbrain, lemon_EO, dortmund)
#   GCS_SUBDIR: bucket subdir for extraction data (e.g. tdbrain_EC, lemon_EO, dortmund_EC_pre_s1)
#   EXTRACT_NAME: (optional) name of extraction subdir under GCS_SUBDIR. Defaults to ${COHORT}_composite.
#     Example: for gs://.../dortmund_EO_pre_s1/dortmund_EO_pre_composite/, pass "dortmund_EO_pre_composite".

set -e
export PATH="/opt/homebrew/share/google-cloud-sdk/bin:$PATH"
export GOOGLE_APPLICATION_CREDENTIALS="/Users/neurokinetikz/Code/research/.gcp/claude-493017-ad29d1cd661b.json"

COHORT=${1:?Need COHORT}
GCS_SUBDIR=${2:?Need GCS_SUBDIR}
EXTRACT_NAME=${3:-${COHORT}_composite}
ZONE=us-central1-a
PROJECT=claude-493017
IMAGE=sie-worker-base-composite-v2
MACHINE=c2d-standard-32
VM_NAME=source-$(echo "$COHORT" | tr 'A-Z_' 'a-z-')-$(date +%H%M)

# Build startup script for this VM
STARTUP=$(cat <<EOF
#!/bin/bash
set -e
exec > >(tee -a /tmp/startup.log) 2>&1
echo "=== Worker startup: cohort=$COHORT ==="
date

# Mount data disk RO (noload: skip journal replay since disk was prev RW-mounted)
mkdir -p /mnt/eeg-data
for i in 1 2 3 4 5; do
  if mount -o ro,noload /dev/sdb /mnt/eeg-data 2>&1; then break; fi
  echo "mount attempt \$i failed, retrying..."; sleep 3
done
mkdir -p /Volumes
ln -sfn /mnt/eeg-data/T9 /Volumes/T9
if ! ls /Volumes/T9 >/dev/null 2>&1; then
  echo "FATAL: T9 not accessible after mount — aborting"
  exit 1
fi
ls /Volumes/T9 | head -3

# Switch to user
sudo -u neurokinetikz bash <<'USERSH'
set -e
cd ~
mkdir -p research
cd research

# Pull scripts
gcloud storage cp gs://sie-composite-v2-extractions/source-space-scripts/scripts.tar.gz /tmp/s.tar.gz
tar xzf /tmp/s.tar.gz

# Pull extraction data for this cohort
mkdir -p exports_sie/${COHORT}_composite
gcloud storage cp -r gs://sie-composite-v2-extractions/${GCS_SUBDIR}/${EXTRACT_NAME}/* exports_sie/${COHORT}_composite/ 2>&1 | tail -2
mkdir -p outputs/schumann/images/quality outputs/schumann/images/source
QUAL_LOCAL=outputs/schumann/images/quality/per_event_quality_${COHORT}_composite.csv
# Try to pull an external quality CSV. If unavailable, Python script falls back
# to sr_score_canonical in events CSV (per-subject on-the-fly Q4).
set +e
gcloud storage cp gs://sie-composite-v2-extractions/analysis-results/${GCS_SUBDIR}/quality.csv \$QUAL_LOCAL 2>/dev/null
if [ ! -s \$QUAL_LOCAL ]; then
  gcloud storage cp gs://sie-composite-v2-extractions/analysis-results/${COHORT}/quality.csv \$QUAL_LOCAL 2>/dev/null
fi
set -e
if [ ! -s \$QUAL_LOCAL ]; then
  echo "No external quality CSV — will use sr_score_canonical fallback in events CSV"
  # Write an empty placeholder so the script's os.path.isfile check passes
  echo "subject_id,t0_net,template_rho" > \$QUAL_LOCAL
fi

ls -la \$QUAL_LOCAL
wc -l exports_sie/${COHORT}_composite/extraction_summary.csv

# Run source-space
source ~/eeg_env/bin/activate
SIE_WORKERS=20 python3 -u scripts/sie_source_localization_composite.py --cohort ${COHORT} 2>&1 | tee logs_source_${COHORT}.log || true

# Push results to GCS (even if partial)
gcloud storage cp -r outputs/schumann/images/source/${COHORT}_composite gs://sie-composite-v2-extractions/source-space-results/ 2>&1 | tail -3
gcloud storage cp logs_source_${COHORT}.log gs://sie-composite-v2-extractions/source-space-results/${COHORT}_composite/ 2>&1 | tail -1

# Mark done
echo "done" > /tmp/_DONE
gcloud storage cp /tmp/_DONE gs://sie-composite-v2-extractions/source-space-results/${COHORT}_composite/_DONE
USERSH

# Self-terminate
echo "=== self-terminating ==="
INSTANCE=\$(curl -s metadata.google.internal/computeMetadata/v1/instance/name -H "Metadata-Flavor: Google")
gcloud compute instances delete \$INSTANCE --zone=us-central1-a --quiet
EOF
)

echo "=== Creating VM $VM_NAME for cohort $COHORT (GCS subdir: $GCS_SUBDIR) ==="
STARTUP_FILE=/tmp/startup_${COHORT}.sh
echo "$STARTUP" > $STARTUP_FILE

if ! gcloud compute instances create $VM_NAME \
  --project=$PROJECT --zone=$ZONE \
  --image=$IMAGE --image-project=$PROJECT \
  --machine-type=$MACHINE \
  --disk=name=eeg-data-disk,mode=ro,boot=no \
  --metadata-from-file=startup-script=$STARTUP_FILE \
  --scopes=cloud-platform \
  --no-user-output-enabled; then
  echo "ERROR: VM creation failed for $VM_NAME" >&2
  exit 1
fi

echo "VM $VM_NAME created. Watch for _DONE marker at:"
echo "  gs://sie-composite-v2-extractions/source-space-results/${COHORT}_composite/_DONE"
