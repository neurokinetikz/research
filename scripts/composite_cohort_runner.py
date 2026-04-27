#!/usr/bin/env python3
"""
Runs all composite v2 analysis scripts for one cohort, sequentially.

Designed to be invoked on a GCE worker VM after the VM has pulled:
  - scripts/ + lib/ code
  - exports_sie/<cohort>_composite/ extraction CSVs
  - optionally: outputs/schumann/images/quality/per_event_quality_<cohort>_composite.csv

If the quality CSV is missing, this runner generates it first via
sie_template_rho_crosscohort.py (step 0). Then iterates through the 44
scripts in composite_analysis_manifest.ANALYSIS_SCRIPTS, running each
with --cohort <cohort> and SIE_WORKERS=28.

After each script completes (pass or fail), the runner:
  - writes a per-script pass/fail marker to GCS under
    gs://<bucket>/analysis-results/<cohort>/_progress/<idx>_<tag>.{ok|fail}
  - pushes outputs/schumann/images/coupling/<cohort>_composite/ to
    gs://<bucket>/analysis-results/<cohort>/coupling/  (incremental cp)

When all scripts have been attempted, writes _DONE marker.

Usage on VM:
    python3 scripts/composite_cohort_runner.py --cohort hbn_R1

Environment:
    GCS_BUCKET: target bucket, default gs://sie-composite-v2-extractions
    SIE_WORKERS: Pool size for inner scripts, default 28
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
import traceback


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.composite_analysis_manifest import ANALYSIS_SCRIPTS


def sh(cmd, check=False, timeout=None):
    """Run shell command, return CompletedProcess."""
    return subprocess.run(cmd, shell=True, check=check,
                           capture_output=True, text=True,
                           timeout=timeout)


def gcs_push(local_path, gcs_path):
    """Push local path (file or dir) to GCS, non-recursive for files."""
    if os.path.isdir(local_path):
        return sh(f"gcloud storage cp -r '{local_path}' '{gcs_path}' 2>&1 | tail -3")
    return sh(f"gcloud storage cp '{local_path}' '{gcs_path}' 2>&1 | tail -3")


def mark_progress(bucket, cohort, idx, tag, status):
    """Write an empty progress marker to GCS."""
    marker = f"{idx:02d}_{tag.replace(' ', '_').replace('/', '-')}.{status}"
    local = f"/tmp/{marker}"
    open(local, 'a').close()
    gcs_push(local, f"{bucket}/analysis-results/{cohort}/_progress/{marker}")


def ensure_quality_csv(cohort):
    """Generate per-event quality CSV if missing."""
    qcsv = os.path.join(ROOT, 'outputs', 'schumann', 'images', 'quality',
                        f'per_event_quality_{cohort}_composite.csv')
    if os.path.isfile(qcsv):
        print(f"[step-0] quality CSV already present at {qcsv}")
        return True
    print(f"[step-0] generating quality CSV for {cohort} ...")
    os.makedirs(os.path.dirname(qcsv), exist_ok=True)
    cmd = (f"cd {ROOT} && SIE_WORKERS={os.environ.get('SIE_WORKERS', '28')} "
           f"python3 scripts/sie_template_rho_crosscohort.py "
           f"--cohort {cohort}_composite")
    p = sh(cmd, timeout=60 * 60)
    print(p.stdout[-2000:] if p.stdout else '')
    print(p.stderr[-2000:] if p.stderr else '')
    if not os.path.isfile(qcsv):
        print(f"[step-0] FAILED to produce {qcsv}")
        return False
    return True


def run_analysis_script(cohort, script, tag, idx, bucket):
    """Run one analysis script, time it, capture output, push progress marker."""
    t0 = time.time()
    log_path = f"/tmp/analysis_{cohort}_{idx:02d}_{os.path.splitext(script)[0]}.log"
    cmd = (f"cd {ROOT} && SIE_WORKERS={os.environ.get('SIE_WORKERS', '28')} "
           f"python3 -u scripts/{script} --cohort {cohort} > '{log_path}' 2>&1")
    print(f"\n[{idx:02d}/{len(ANALYSIS_SCRIPTS)}] {tag}  {script}")
    # per-script budget: 30 min (composite scripts are mostly Pool(28)
    # one-shot over ~100-300 subjects, should finish well under 20 min)
    try:
        p = sh(cmd, timeout=30 * 60)
        status = 'ok' if p.returncode == 0 else 'fail'
    except subprocess.TimeoutExpired:
        status = 'fail'
    dt = time.time() - t0
    print(f"    status={status}  elapsed={dt/60:.1f} min")
    # Push log
    gcs_push(log_path,
             f"{bucket}/analysis-results/{cohort}/_logs/{idx:02d}_{script}.log")
    # Push any new outputs across ALL known output subdirs
    for out_subdir in ['coupling', 'perionset', 'mechanism_battery', 'iei',
                       'psd_timelapse', 'single_event', 'multistream',
                       'source', 'if_corrections', 'spectrum', 'quality']:
        out_dir = os.path.join(ROOT, 'outputs', 'schumann', 'images',
                               out_subdir, f'{cohort}_composite')
        if os.path.isdir(out_dir):
            sh(f"gcloud storage cp -r '{out_dir}' "
               f"'{bucket}/analysis-results/{cohort}/{out_subdir}/' 2>&1 | tail -2")
    mark_progress(bucket, cohort, idx, tag, status)
    return status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', required=True)
    ap.add_argument('--bucket', default='gs://sie-composite-v2-extractions',
                    help='GCS bucket root for progress + results')
    args = ap.parse_args()

    bucket = args.bucket.rstrip('/')
    cohort = args.cohort

    print(f"=== composite_cohort_runner: {cohort} ===")
    print(f"Scripts: {len(ANALYSIS_SCRIPTS)}")
    print(f"Bucket:  {bucket}")
    print(f"Workers: {os.environ.get('SIE_WORKERS', '28')}")

    # Step 0: ensure quality CSV
    if not ensure_quality_csv(cohort):
        print("[step-0] ABORT: quality CSV not available, cannot stratify by Q4")
        mark_progress(bucket, cohort, 0, 'quality_csv', 'fail')
        open('/tmp/_ABORT', 'a').close()
        gcs_push('/tmp/_ABORT', f"{bucket}/analysis-results/{cohort}/_ABORT")
        sys.exit(2)
    # Push quality CSV to GCS for later cohort-level sharing
    qcsv = os.path.join(ROOT, 'outputs', 'schumann', 'images', 'quality',
                        f'per_event_quality_{cohort}_composite.csv')
    gcs_push(qcsv, f"{bucket}/analysis-results/{cohort}/quality.csv")
    mark_progress(bucket, cohort, 0, 'quality_csv', 'ok')

    # Steps 1..N: run each analysis script
    results = []
    for i, (script, tag) in enumerate(ANALYSIS_SCRIPTS, start=1):
        try:
            status = run_analysis_script(cohort, script, tag, i, bucket)
        except Exception as e:
            print(f"    EXCEPTION: {e}")
            traceback.print_exc()
            status = 'fail'
            mark_progress(bucket, cohort, i, tag, status)
        results.append((script, tag, status))

    # Summary
    n_ok = sum(1 for _, _, s in results if s == 'ok')
    n_fail = sum(1 for _, _, s in results if s == 'fail')
    print(f"\n=== SUMMARY {cohort} ===")
    print(f"  ok:   {n_ok}/{len(results)}")
    print(f"  fail: {n_fail}/{len(results)}")

    # Write final _DONE marker
    # Final verify-and-push: ensure every local output file is in GCS
    # before writing _DONE. Retries once if any files are missing.
    print(f"\n=== Final sync: {cohort} ===")
    subdirs = ['coupling', 'perionset', 'mechanism_battery', 'iei',
               'psd_timelapse', 'single_event', 'multistream',
               'source', 'if_corrections', 'spectrum', 'quality']
    for attempt in range(2):
        missing = []
        for subdir in subdirs:
            local_dir = os.path.join(ROOT, 'outputs', 'schumann', 'images',
                                     subdir, f'{cohort}_composite')
            if not os.path.isdir(local_dir):
                continue
            # Ensure GCS copy of this subdir is up to date
            sh(f"gcloud storage cp -r '{local_dir}' "
               f"'{bucket}/analysis-results/{cohort}/{subdir}/' 2>&1 | tail -1")
            # Enumerate local files
            local_files = set()
            for r, _, fs in os.walk(local_dir):
                for f in fs:
                    rel = os.path.relpath(os.path.join(r, f), local_dir)
                    local_files.add(rel)
            # Enumerate GCS files
            r_ls = sh(f"gcloud storage ls -r "
                      f"'{bucket}/analysis-results/{cohort}/{subdir}/"
                      f"{cohort}_composite/**' 2>/dev/null")
            prefix = (f"{bucket}/analysis-results/{cohort}/{subdir}/"
                      f"{cohort}_composite/")
            gcs_files = set()
            for line in r_ls.stdout.splitlines():
                line = line.strip()
                if line.startswith(prefix):
                    rel = line[len(prefix):]
                    if rel and not rel.endswith('/'):
                        gcs_files.add(rel)
            # Diff
            missing_here = local_files - gcs_files
            if missing_here:
                for f in sorted(missing_here):
                    missing.append(f"{subdir}/{cohort}_composite/{f}")
        if not missing:
            print(f"  Sync complete on attempt {attempt+1}: all local outputs verified in GCS")
            break
        print(f"  [attempt {attempt+1}] {len(missing)} files missing in GCS; "
              f"will retry. Examples: {missing[:3]}")
    else:
        # Both attempts had missing files — log but still write _DONE
        # (partial data is better than nothing)
        print(f"  WARNING: after 2 attempts, {len(missing)} files still missing")
        with open('/tmp/_MISSING', 'w') as f:
            f.write('\n'.join(missing))
        gcs_push('/tmp/_MISSING', f"{bucket}/analysis-results/{cohort}/_MISSING")

    open('/tmp/_DONE', 'a').close()
    gcs_push('/tmp/_DONE', f"{bucket}/analysis-results/{cohort}/_DONE")


if __name__ == '__main__':
    main()
