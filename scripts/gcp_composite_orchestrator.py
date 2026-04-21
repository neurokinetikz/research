#!/usr/bin/env python3
"""
GCP multi-VM orchestrator for composite v2 extraction of all datasets.

Architecture:
- Up to 8 concurrent c2d-standard-32 worker VMs (GCP quota limit)
- Each worker: mounts eeg-data-disk read-only, runs extraction for ONE
  dataset/condition/release, pushes results to GCS bucket, self-terminates
- Coordinator: maintains FIFO queue of dataset jobs, polls VM status,
  spawns new VMs as old ones terminate

Pre-requisites (manual, one-time):
  1. Stop sie-sharpen-session VM (so its RW disk attachment is released)
  2. Detach eeg-data-disk
  3. Create worker image from sie-sharpen-session's BOOT disk:
     gcloud compute images create sie-worker-base-composite \
         --source-disk=sie-sharpen-session \
         --source-disk-zone=us-central1-a \
         --family=sie-workers
  4. Create GCS bucket for staged results:
     gcloud storage buckets create gs://sie-composite-v2-extractions \
         --project=claude-493017 --location=us-central1

Usage:
  python scripts/gcp_composite_orchestrator.py --dry-run   # preview jobs
  python scripts/gcp_composite_orchestrator.py             # live run
  python scripts/gcp_composite_orchestrator.py --only lemon hbn_R10  # subset
"""
from __future__ import annotations
import argparse
import subprocess
import time
import os
import json
import sys

PROJECT = 'claude-493017'
ZONE = 'us-central1-a'
IMAGE = 'sie-worker-base-composite-v2'
IMAGE_FAMILY = 'sie-workers'
MACHINE_TYPE = 'c2d-standard-32'
DATA_DISK = 'eeg-data-disk'
GCS_BUCKET = 'gs://sie-composite-v2-extractions'
MAX_CONCURRENT = 8

# ===== JOB QUEUE (longest jobs first) =====
# Tuple fields: (job_id, dataset, cli_args_list, estimated_min[, detector])
# detector defaults to 'composite' (S₄). Use 'composite_s3' to drop MSC
# (required for 128-ch HBN and long-record EEGMMIDB).
JOB_QUEUE = [
    # --- S₃ validation (re-extracts LEMON EC with MSC-free detector to
    # confirm S₃ reproduces the B62 Figure-3 / B48 numbers before scaling
    # to HBN) ---
    ('lemon_s3_validate', 'lemon', [], 20, 'composite_s3'),

    # --- HBN (128-ch EGI, 11 releases) — S₃ only, MSC dropped ---
    ('hbn_R1', 'hbn', ['--release', 'R1'], 60, 'composite_s3'),
    ('hbn_R2', 'hbn', ['--release', 'R2'], 60, 'composite_s3'),
    ('hbn_R3', 'hbn', ['--release', 'R3'], 60, 'composite_s3'),
    ('hbn_R4', 'hbn', ['--release', 'R4'], 60, 'composite_s3'),
    ('hbn_R5', 'hbn', ['--release', 'R5'], 60, 'composite_s3'),
    ('hbn_R6', 'hbn', ['--release', 'R6'], 60, 'composite_s3'),
    ('hbn_R7', 'hbn', ['--release', 'R7'], 60, 'composite_s3'),
    ('hbn_R8', 'hbn', ['--release', 'R8'], 60, 'composite_s3'),
    ('hbn_R9', 'hbn', ['--release', 'R9'], 60, 'composite_s3'),
    ('hbn_R10', 'hbn', ['--release', 'R10'], 60, 'composite_s3'),
    ('hbn_R11', 'hbn', ['--release', 'R11'], 60, 'composite_s3'),

    # --- Already-extracted S₄ jobs (still listed so re-runs skip via
    # _SUCCESS marker) ---
    ('tdbrain_EC', 'tdbrain', ['--condition', 'EC'], 120),
    ('tdbrain_EO', 'tdbrain', ['--condition', 'EO'], 120),
    ('lemon_EO', 'lemon', ['--condition', 'EO'], 20),
    ('dortmund_EC_pre_s1', 'dortmund', ['--condition', 'EC-pre', '--session', '1'], 20),
    ('dortmund_EC_pre_s2', 'dortmund', ['--condition', 'EC-pre', '--session', '2'], 20),
    ('dortmund_EC_post_s1', 'dortmund', ['--condition', 'EC-post', '--session', '1'], 20),
    ('dortmund_EC_post_s2', 'dortmund', ['--condition', 'EC-post', '--session', '2'], 20),
    ('dortmund_EO_pre_s1', 'dortmund', ['--condition', 'EO-pre', '--session', '1'], 20),
    ('dortmund_EO_pre_s2', 'dortmund', ['--condition', 'EO-pre', '--session', '2'], 20),
    ('dortmund_EO_post_s1', 'dortmund', ['--condition', 'EO-post', '--session', '1'], 20),
    ('dortmund_EO_post_s2', 'dortmund', ['--condition', 'EO-post', '--session', '2'], 20),
    ('chbmp', 'chbmp', [], 20),
    ('srm', 'srm', [], 20),
    ('vep', 'vep', [], 15),
    ('physf', 'physf', [], 15),
    ('epoc_self', 'epoc_self', [], 10),
    ('insight_self', 'insight_self', [], 10),
]


def sh(cmd, check=True, capture=True):
    """Run shell command."""
    if capture:
        return subprocess.run(cmd, shell=True, check=check,
                               capture_output=True, text=True)
    return subprocess.run(cmd, shell=True, check=check)


def vm_name(job_id):
    """Deterministic worker VM name (GCE requires lowercase, dashes only)."""
    return f'sie-worker-{job_id.replace("_", "-").lower()}'


def build_startup_script(job_id, dataset, cli_args, detector='composite'):
    """Return a bash startup script the worker runs on boot."""
    args_str = ' '.join(cli_args)
    out_suffix = '_composite_s3' if detector == 'composite_s3' else '_composite'
    return f"""#!/bin/bash
set -e
LOG=/var/log/sie-worker.log
exec > >(tee -a $LOG) 2>&1
echo "=== SIE worker {job_id} start: $(date -u) ==="

# Mount eeg-data-disk read-only at /mnt/eeg-data (matches base image's
# /Volumes/T9 -> /mnt/eeg-data/T9 symlink)
mkdir -p /mnt/eeg-data
sleep 15  # wait for block device to settle
DEV=$(lsblk -no NAME,SIZE | awk '$2 == "2.5T" {{print "/dev/"$1; exit}}')
if [ -z "$DEV" ]; then
  DEV=/dev/sdb
fi
echo "Mounting $DEV (read-only) at /mnt/eeg-data"
mount -o ro $DEV /mnt/eeg-data || {{ echo "mount failed"; exit 2; }}
ls /mnt/eeg-data/T9/lemon_data >/dev/null || {{ echo "T9 not readable"; exit 3; }}

# Activate eeg_env (preserved in base image)
source /home/neurokinetikz/eeg_env/bin/activate || true

# Pull latest research repo from git remote (assumes base image has ~/research cloned)
cd /home/neurokinetikz/research
# git config needs HOME set; startup runs as root with HOME unset
export HOME=/root
git config --global --add safe.directory /home/neurokinetikz/research 2>&1 || true
git fetch --all && git reset --hard origin/main || echo "git pull failed, using baked-in code"

# Clean stale _composite and _composite_s3 leftovers from base image so
# upload glob only finds this job's output
rm -rf /home/neurokinetikz/research/exports_sie/*_composite \
       /home/neurokinetikz/research/exports_sie/*_composite_s3 2>/dev/null

# Cap BLAS/MKL/OpenMP threads so 16 Python workers don't spawn 16×N threads
# and saturate the 32-vCPU VM (prior runs hit load 124 on 32 cores).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Run extraction with {detector} detector (suffix {out_suffix})
SIE_WORKERS=16 python3 -u scripts/run_sie_extraction.py \\
    --dataset {dataset} \\
    --detector {detector} \\
    --out_suffix {out_suffix} \\
    --parallel 16 \\
    {args_str}
EXIT=$?

# Push results to GCS
OUT_DIR=/home/neurokinetikz/research/exports_sie
if [ $EXIT -eq 0 ]; then
  # Find the output dir name (depends on dataset/condition)
  for d in $OUT_DIR/*{out_suffix}; do
    if [ -d "$d" ]; then
      NAME=$(basename "$d")
      gcloud storage cp -r "$d" {GCS_BUCKET}/{job_id}/
      echo "Uploaded $NAME to {GCS_BUCKET}/{job_id}/"
    fi
  done
  gcloud storage cp /dev/null {GCS_BUCKET}/{job_id}/_SUCCESS
else
  gcloud storage cp /dev/null {GCS_BUCKET}/{job_id}/_FAILED
fi

# Self-terminate
NAME=$(curl -s -H 'Metadata-Flavor: Google' \\
       http://metadata.google.internal/computeMetadata/v1/instance/name)
gcloud compute instances delete "$NAME" --zone={ZONE} --quiet
"""


def launch_vm(job_id, dataset, cli_args, dry_run=False, detector='composite'):
    vm = vm_name(job_id)
    # Skip if VM already exists — treat as active (will poll for _SUCCESS)
    r = sh(f"gcloud compute instances describe {vm} "
           f"--zone={ZONE} --format='value(name)' 2>/dev/null",
            check=False)
    if r.returncode == 0 and r.stdout.strip() == vm:
        print(f"  [EXISTS] {vm} already running — track as active")
        return 'existing'
    # Skip if already succeeded in bucket
    if job_done(job_id) == 'success':
        print(f"  [SKIP] {job_id} already has _SUCCESS in bucket")
        return None  # signal: don't re-enqueue, just drop
    startup = build_startup_script(job_id, dataset, cli_args, detector=detector)
    startup_file = f'/tmp/startup_{job_id}.sh'
    with open(startup_file, 'w') as f:
        f.write(startup)
    cmd = (f"gcloud compute instances create {vm} "
           f"--project={PROJECT} --zone={ZONE} "
           f"--machine-type={MACHINE_TYPE} "
           f"--image={IMAGE} --image-project={PROJECT} "
           f"--disk=name={DATA_DISK},mode=ro,boot=no "
           f"--metadata-from-file=startup-script={startup_file} "
           f"--scopes=cloud-platform "
           f"--no-user-output-enabled")
    if dry_run:
        print(f"[dry-run] would launch {vm}:")
        print(f"  {cmd}")
        return True
    print(f"Launching {vm}...")
    r = sh(cmd, check=False)
    if r.returncode != 0:
        print(f"ERROR launching {vm}: {r.stderr}")
        return False
    print(f"  launched {vm}")
    return True


def list_running_workers():
    """Return list of currently-running sie-worker VM names."""
    r = sh(f"gcloud compute instances list "
            f"--project={PROJECT} --zones={ZONE} "
            f"--filter='name ~ ^sie-worker-' "
            f"--format='value(name)'")
    return [n.strip() for n in r.stdout.splitlines() if n.strip()]


def job_done(job_id):
    """Check if _SUCCESS or _FAILED marker exists in GCS for this job."""
    r = sh(f"gcloud storage ls {GCS_BUCKET}/{job_id}/_SUCCESS 2>/dev/null",
            check=False)
    if r.returncode == 0: return 'success'
    r = sh(f"gcloud storage ls {GCS_BUCKET}/{job_id}/_FAILED 2>/dev/null",
            check=False)
    if r.returncode == 0: return 'failed'
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true',
                    help='Preview jobs without launching')
    ap.add_argument('--only', nargs='+', default=None,
                    help='Run only specific job_ids')
    ap.add_argument('--poll-sec', type=int, default=60,
                    help='Seconds between status polls')
    args = ap.parse_args()

    queue = list(JOB_QUEUE)
    if args.only:
        queue = [j for j in queue if j[0] in args.only]
        print(f"Filtered to: {[j[0] for j in queue]}")

    print(f"Total jobs: {len(queue)}")
    print(f"Max concurrent VMs: {MAX_CONCURRENT}")
    print(f"GCS bucket: {GCS_BUCKET}")

    pending = list(queue)
    active = {}  # job_id -> launch_time
    finished = {}

    while pending or active:
        # Reap finished workers
        for job_id in list(active):
            status = job_done(job_id)
            if status is not None:
                elapsed = (time.time() - active[job_id]) / 60
                print(f"  [{status.upper()}] {job_id} after {elapsed:.1f} min")
                finished[job_id] = status
                del active[job_id]

        # Launch new workers up to MAX_CONCURRENT
        running_vms = list_running_workers() if not args.dry_run else list(active)
        capacity = MAX_CONCURRENT - len(running_vms)
        while capacity > 0 and pending:
            entry = pending.pop(0)
            job_id, dataset, cli_args, est = entry[:4]
            detector = entry[4] if len(entry) >= 5 else 'composite'
            ok = launch_vm(job_id, dataset, cli_args,
                            dry_run=args.dry_run, detector=detector)
            if ok is True:
                active[job_id] = time.time()
                capacity -= 1
            elif ok == 'existing':
                active[job_id] = time.time()
                capacity -= 1
            elif ok is None:
                finished[job_id] = 'success'
            else:
                print(f"  [REQUEUE] {job_id} after launch failure")
                pending.append(entry)
                break

        if args.dry_run:
            print("Dry run complete.")
            return

        if not active and not pending:
            break
        print(f"  active={len(active)}  pending={len(pending)}  "
              f"finished={len(finished)}  (poll in {args.poll_sec}s)")
        time.sleep(args.poll_sec)

    print(f"\n=== Summary ===")
    for j, s in finished.items():
        print(f"  {j}: {s}")
    success = sum(1 for v in finished.values() if v == 'success')
    print(f"\nSucceeded: {success}/{len(finished)}")
    print(f"Results in {GCS_BUCKET}")


if __name__ == '__main__':
    main()
