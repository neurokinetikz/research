#!/usr/bin/env python3
"""
GCP multi-VM orchestrator for composite v2 cross-cohort analysis sweep.

Architecture:
- Each worker VM owns ONE cohort and runs all 44 composite analysis scripts
  sequentially via scripts/composite_cohort_runner.py.
- Up to 8 concurrent c2d-standard-32 workers (GCP quota).
- Coordinator: maintains FIFO queue of cohort jobs, polls GCS for _DONE
  markers, launches new VMs as old ones finish.

Job flow (per VM):
  1. Startup script mounts eeg-data-disk read-only
  2. Pulls scripts tarball + lib from GCS staging path
  3. Pulls extraction CSV for this cohort from the extractions bucket
  4. (Optional) Pulls pre-computed quality CSV from staging
  5. Runs composite_cohort_runner.py --cohort <cohort>, which:
       a. Generates quality CSV if missing (step 0)
       b. Runs each of the 44 analysis scripts
       c. Pushes outputs + progress markers to bucket
       d. Writes _DONE marker
  6. VM self-terminates

Usage:
  # Preview
  python scripts/gcp_analysis_orchestrator.py --dry-run

  # Test on one cohort
  python scripts/gcp_analysis_orchestrator.py --only chbmp

  # Live full run
  python scripts/gcp_analysis_orchestrator.py
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import tarfile
import tempfile
import time


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.composite_analysis_manifest import COHORTS


PROJECT = 'claude-493017'
ZONE = 'us-central1-a'
IMAGE = 'sie-worker-base-composite-v2'
MACHINE_TYPE = 'c2d-standard-32'
DATA_DISK = 'eeg-data-disk'
GCS_BUCKET = 'gs://sie-composite-v2-extractions'
MAX_CONCURRENT = 8

# Map cohort name (as used by analysis scripts) → (GCS bucket subdir, extraction subdir name)
# Defaults to (cohort, f"{cohort}_composite") for cohorts not listed here.
COHORT_GCS_MAP = {
    'tdbrain':       ('tdbrain_EC',           'tdbrain_composite'),
    'tdbrain_EO':    ('tdbrain_EO',           'tdbrain_EO_composite'),
    'lemon_EO':      ('lemon_EO',             'lemon_EO_composite'),
    'chbmp':         ('chbmp',                'chbmp_composite'),
    # 'dortmund' is an alias for one canonical Dortmund sub-condition
    'dortmund':      ('dortmund_EC_pre_s1',   'dortmund_composite'),
}


def gcs_paths(cohort):
    """Return (bucket_subdir, extraction_subdir) for a cohort."""
    if cohort in COHORT_GCS_MAP:
        return COHORT_GCS_MAP[cohort]
    # Default: cohort matches bucket subdir; extraction is <cohort>_composite
    return cohort, f"{cohort}_composite"


def sh(cmd, check=False, capture=True):
    if capture:
        return subprocess.run(cmd, shell=True, check=check,
                               capture_output=True, text=True)
    return subprocess.run(cmd, shell=True, check=check)


def vm_name(cohort):
    return f'sie-analysis-{cohort.replace("_", "-").lower()}'


def stage_scripts_bundle():
    """Tar up scripts/ + lib/ and upload to GCS staging path.

    Returns (gcs_path, local_tarball_path).
    """
    ts = time.strftime('%Y%m%d-%H%M%S')
    tar_name = f'analysis-code-{ts}.tar.gz'
    tar_local = os.path.join(tempfile.gettempdir(), tar_name)
    print(f"Packing {tar_local} ...")
    with tarfile.open(tar_local, 'w:gz') as tf:
        for d in ['scripts', 'lib']:
            tf.add(os.path.join(ROOT, d), arcname=d,
                   filter=lambda ti: ti if not ti.name.endswith(('.pyc', '__pycache__'))
                                     else None)
    gcs_path = f"{GCS_BUCKET}/analysis-code/{tar_name}"
    print(f"Uploading → {gcs_path}")
    r = sh(f"gcloud storage cp '{tar_local}' '{gcs_path}'", check=True)
    # Also upload any pre-existing per-cohort quality CSVs
    qual_dir = os.path.join(ROOT, 'outputs', 'schumann', 'images', 'quality')
    if os.path.isdir(qual_dir):
        for f in sorted(os.listdir(qual_dir)):
            if f.startswith('per_event_quality_') and '_composite' in f:
                local = os.path.join(qual_dir, f)
                gcs = f"{GCS_BUCKET}/analysis-code/quality/{f}"
                sh(f"gcloud storage cp '{local}' '{gcs}' 2>/dev/null",
                   check=False)
    return gcs_path, tar_local


def build_startup_script(cohort, scripts_tarball_gcs):
    """Return bash startup script for the cohort worker VM."""
    # Guard cohort name against shell injection
    assert all(c.isalnum() or c in '_-' for c in cohort), f"bad cohort {cohort!r}"
    bucket_subdir, extract_subdir = gcs_paths(cohort)
    target_subdir = f'{cohort}_composite'
    # If the extraction subdir name on GCS differs from what the analysis
    # scripts expect on local disk, we rename it during copy.
    return f"""#!/bin/bash
set -e
LOG=/var/log/sie-analysis.log
exec > >(tee -a $LOG) 2>&1
echo "=== SIE analysis worker {cohort} start: $(date -u) ==="

# ---- Mount eeg-data-disk read-only ----
mkdir -p /mnt/eeg-data
sleep 15
DEV=$(lsblk -no NAME,SIZE | awk '$2 == "2.5T" {{print "/dev/"$1; exit}}')
if [ -z "$DEV" ]; then DEV=/dev/sdb; fi
mount -o ro $DEV /mnt/eeg-data || {{ echo "mount failed"; exit 2; }}
ls /mnt/eeg-data/T9/lemon_data >/dev/null || {{ echo "T9 not readable"; exit 3; }}

# ---- Activate env ----
source /home/neurokinetikz/eeg_env/bin/activate || true

cd /home/neurokinetikz/research
export HOME=/root

# ---- Pull latest scripts/ + lib/ bundle ----
gcloud storage cp '{scripts_tarball_gcs}' /tmp/analysis-code.tar.gz
tar -xzf /tmp/analysis-code.tar.gz -C /home/neurokinetikz/research
chown -R neurokinetikz:neurokinetikz /home/neurokinetikz/research/scripts /home/neurokinetikz/research/lib

# ---- Pull extraction CSV for this cohort ----
mkdir -p /home/neurokinetikz/research/exports_sie
gcloud storage cp -r '{GCS_BUCKET}/{bucket_subdir}/{extract_subdir}' \\
    /home/neurokinetikz/research/exports_sie/ 2>&1 | tail -3
# Rename if GCS subdir differs from cohort's expected local dir
if [ "{extract_subdir}" != "{target_subdir}" ]; then
  mv /home/neurokinetikz/research/exports_sie/{extract_subdir} \\
     /home/neurokinetikz/research/exports_sie/{target_subdir}
fi

# ---- Pull pre-existing quality CSV if present ----
mkdir -p /home/neurokinetikz/research/outputs/schumann/images/quality
gcloud storage cp '{GCS_BUCKET}/analysis-code/quality/per_event_quality_{cohort}_composite.csv' \\
    /home/neurokinetikz/research/outputs/schumann/images/quality/ 2>/dev/null || true

# ---- BLAS thread caps ----
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export SIE_WORKERS=28

# ---- Run analysis runner (don't exit on failure — we still want to self-terminate) ----
set +e
python3 -u /home/neurokinetikz/research/scripts/composite_cohort_runner.py \\
    --cohort {cohort} --bucket {GCS_BUCKET}
EXIT=$?
set -e

echo "=== runner exit $EXIT ==="

# Self-terminate
NAME=$(curl -s -H 'Metadata-Flavor: Google' \\
       http://metadata.google.internal/computeMetadata/v1/instance/name)
gcloud compute instances delete "$NAME" --zone={ZONE} --quiet
"""


def launch_vm(cohort, scripts_tarball_gcs, dry_run=False):
    vm = vm_name(cohort)
    r = sh(f"gcloud compute instances describe {vm} "
           f"--zone={ZONE} --format='value(name)' 2>/dev/null",
            check=False)
    if r.returncode == 0 and r.stdout.strip() == vm:
        print(f"  [EXISTS] {vm} already running — track as active")
        return 'existing'
    # Skip if already done
    if job_done(cohort) == 'success':
        print(f"  [SKIP] {cohort} already has _DONE marker")
        return None
    startup = build_startup_script(cohort, scripts_tarball_gcs)
    startup_file = f'/tmp/startup_{cohort}.sh'
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
        print(f"[dry-run] would launch {vm}")
        return True
    print(f"Launching {vm}...")
    r = sh(cmd, check=False)
    if r.returncode != 0:
        print(f"ERROR launching {vm}: {r.stderr}")
        return False
    print(f"  launched {vm}")
    return True


def list_running_workers():
    r = sh(f"gcloud compute instances list "
            f"--project={PROJECT} --zones={ZONE} "
            f"--filter='name ~ ^sie-analysis-' "
            f"--format='value(name)'")
    return [n.strip() for n in r.stdout.splitlines() if n.strip()]


def job_done(cohort):
    """Check GCS for _DONE or _ABORT marker."""
    r = sh(f"gcloud storage ls "
            f"{GCS_BUCKET}/analysis-results/{cohort}/_DONE 2>/dev/null",
            check=False)
    if r.returncode == 0:
        return 'success'
    r = sh(f"gcloud storage ls "
            f"{GCS_BUCKET}/analysis-results/{cohort}/_ABORT 2>/dev/null",
            check=False)
    if r.returncode == 0:
        return 'aborted'
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--only', nargs='+', default=None,
                    help='Restrict to specific cohorts')
    ap.add_argument('--poll-sec', type=int, default=60)
    args = ap.parse_args()

    queue = list(COHORTS)
    if args.only:
        queue = [c for c in queue if c in args.only]
    print(f"Cohorts: {queue}")
    print(f"Max concurrent VMs: {MAX_CONCURRENT}")
    print(f"Bucket: {GCS_BUCKET}")

    if not args.dry_run:
        scripts_tarball_gcs, _ = stage_scripts_bundle()
    else:
        scripts_tarball_gcs = f'{GCS_BUCKET}/analysis-code/analysis-code-DRYRUN.tar.gz'

    pending = list(queue)
    active = {}
    finished = {}

    while pending or active:
        # Reap finished workers
        for cohort in list(active):
            st = job_done(cohort)
            if st is not None:
                elapsed = (time.time() - active[cohort]) / 60
                print(f"  [{st.upper()}] {cohort} after {elapsed:.1f} min")
                finished[cohort] = st
                del active[cohort]

        # Launch new VMs up to capacity
        running = list_running_workers() if not args.dry_run else list(active)
        cap = MAX_CONCURRENT - len(running)
        while cap > 0 and pending:
            cohort = pending.pop(0)
            ok = launch_vm(cohort, scripts_tarball_gcs, dry_run=args.dry_run)
            if ok is True or ok == 'existing':
                active[cohort] = time.time()
                cap -= 1
            elif ok is None:
                finished[cohort] = 'success'
            else:
                print(f"  [REQUEUE] {cohort}")
                pending.append(cohort)
                break

        if args.dry_run:
            print("Dry run complete.")
            return

        if not active and not pending:
            break
        print(f"  active={len(active)}  pending={len(pending)}  "
              f"finished={len(finished)}  (poll in {args.poll_sec}s)")
        time.sleep(args.poll_sec)

    print("\n=== Summary ===")
    for c, s in finished.items():
        print(f"  {c}: {s}")
    ok = sum(1 for s in finished.values() if s == 'success')
    print(f"Succeeded: {ok}/{len(finished)}")
    print(f"Results in {GCS_BUCKET}/analysis-results/")


if __name__ == '__main__':
    main()
