#!/usr/bin/env python3
"""Pull composite extractions from GCS bucket into local exports_sie/.

Bucket layout: gs://sie-composite-v2-extractions/<job_id>/<dataset>_composite/*.csv
Local target:  exports_sie/<dataset>_composite/*.csv

Flattens the bucket's job_id wrapper directory so downstream scripts
can read directly from exports_sie/<dataset>_composite/ like any other
extraction.
"""
from __future__ import annotations
import os
import subprocess
import sys

BUCKET = 'gs://sie-composite-v2-extractions'
LOCAL = os.path.join(os.path.dirname(__file__), '..', 'exports_sie')


def sh(cmd, check=True):
    return subprocess.run(cmd, shell=True, check=check, capture_output=True,
                           text=True)


def list_jobs():
    r = sh(f'gcloud storage ls {BUCKET}/')
    jobs = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.endswith('/'):
            name = line.rstrip('/').rsplit('/', 1)[-1]
            jobs.append(name)
    return jobs


def has_success(job):
    r = sh(f'gcloud storage ls {BUCKET}/{job}/_SUCCESS 2>/dev/null',
            check=False)
    return r.returncode == 0


def find_composite_subdir(job):
    """Return the name of the *_composite subdir for this job
    (skipping leftover lemon_composite when the job isn't lemon_EC)."""
    r = sh(f'gcloud storage ls {BUCKET}/{job}/')
    subdirs = [line.strip().rstrip('/').rsplit('/', 1)[-1]
                for line in r.stdout.splitlines()
                if line.strip().endswith('/')]
    candidates = [s for s in subdirs if s.endswith('_composite')]
    if not candidates:
        return None
    # If lemon_composite is present AND the job is lemon_EC, we want it
    if job == 'lemon_EC' and 'lemon_composite' in candidates:
        return 'lemon_composite'
    # Otherwise pick the non-lemon_composite candidate
    non_lemon = [c for c in candidates if c != 'lemon_composite']
    return non_lemon[0] if non_lemon else candidates[0]


def pull_job(job, overwrite=False):
    if not has_success(job):
        print(f"  [skip] {job} — no _SUCCESS marker")
        return False
    sub = find_composite_subdir(job)
    if not sub:
        print(f"  [skip] {job} — no _composite subdir")
        return False
    src = f'{BUCKET}/{job}/{sub}'
    # Target local name matches subdir (e.g., tdbrain_EO_composite)
    tgt = os.path.join(LOCAL, sub)
    if os.path.isdir(tgt) and not overwrite:
        existing = len([f for f in os.listdir(tgt)
                         if f.endswith('_sie_events.csv')])
        print(f"  [skip] {tgt} exists with {existing} CSVs")
        return True
    os.makedirs(tgt, exist_ok=True)
    print(f"  pulling {job}: {src} -> {tgt}")
    r = sh(f'gcloud storage cp -r "{src}/*" "{tgt}/" 2>&1 | tail -2',
            check=False)
    if r.returncode != 0:
        print(f"    ERROR: {r.stdout} {r.stderr}")
        return False
    n = len([f for f in os.listdir(tgt)
              if f.endswith('_sie_events.csv')])
    print(f"    pulled {n} event CSVs")
    return True


def main():
    jobs = list_jobs()
    print(f"Found {len(jobs)} job(s) in bucket")
    for job in sorted(jobs):
        pull_job(job)


if __name__ == '__main__':
    main()
