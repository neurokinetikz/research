#!/usr/bin/env python3
"""
HBN Phi-Lattice Replication — Per-Release Runner
=================================================

Runs the locked phi_replication protocol on a specified HBN release.
Identical preprocessing to R1: pick EEG → notch [60,120] → resample 250 Hz.

Usage:
    python scripts/run_hbn_release_phi_replication.py R2
    python scripts/run_hbn_release_phi_replication.py R3
"""

import sys
import os
import glob
import numpy as np
import pandas as pd
import mne
import warnings
warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from phi_replication import (process_subjects_from_eeg, run_full_protocol,
                             test_theta_ec_convergence,
                             generate_preregistration_report,
                             run_statistics, run_14position_enrichment)

# ── Config ────────────────────────────────────────────────────────────
HBN_BASE = '/Volumes/T9/hbn_data'
TARGET_FS = 250
MIN_CONDITION_DURATION = 10.0


def discover_subjects(data_dir):
    """Find all subjects with RestingState .set files (excludes macOS ._files)."""
    pattern = os.path.join(data_dir, 'sub-*', 'eeg',
                           '*_task-RestingState_eeg.set')
    files = sorted(glob.glob(pattern))
    subjects = []
    seen = set()
    for f in files:
        # Skip macOS resource forks
        basename = os.path.basename(f)
        if basename.startswith('._'):
            continue
        parts = f.split(os.sep)
        for p in parts:
            if p.startswith('sub-'):
                if p not in seen:
                    subjects.append(p)
                    seen.add(p)
                break
    return subjects


def parse_events(data_dir, sub_id):
    """Parse events.tsv to get EO/EC segment boundaries."""
    events_path = os.path.join(data_dir, sub_id, 'eeg',
                                f'{sub_id}_task-RestingState_events.tsv')
    if not os.path.isfile(events_path):
        return None

    events = pd.read_csv(events_path, sep='\t')

    eo_onsets = events[events['event_code'] == 20]['onset'].values
    ec_onsets = events[events['event_code'] == 30]['onset'].values

    all_onsets = sorted(list(eo_onsets) + list(ec_onsets))
    segments = {'EO': [], 'EC': []}

    for onset in eo_onsets:
        later = [t for t in all_onsets if t > onset]
        offset = later[0] if later else onset + 20.0
        segments['EO'].append((onset, offset))

    for onset in ec_onsets:
        later = [t for t in all_onsets if t > onset]
        offset = later[0] if later else onset + 20.0
        segments['EC'].append((onset, offset))

    return segments


def make_loader(data_dir):
    """Create a loader function bound to a specific data directory."""

    def load_hbn_subject(sub_id, condition='combined'):
        set_path = os.path.join(data_dir, sub_id, 'eeg',
                                 f'{sub_id}_task-RestingState_eeg.set')
        if not os.path.isfile(set_path):
            raise FileNotFoundError(f"No RestingState file for {sub_id}")

        raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)
        raw.pick_types(eeg=True, exclude='bads')

        # Notch filter at 60 Hz + harmonics (US power line)
        raw.notch_filter(freqs=[60, 120], verbose=False)

        # Downsample to 250 Hz
        if raw.info['sfreq'] > TARGET_FS:
            raw.resample(TARGET_FS, verbose=False)

        # Extract condition-specific segments
        if condition in ('EO', 'EC'):
            segments = parse_events(data_dir, sub_id)
            if segments and condition in segments and len(segments[condition]) > 0:
                raws = []
                for onset, offset in segments[condition]:
                    try:
                        segment = raw.copy().crop(tmin=onset, tmax=min(offset, raw.times[-1]))
                        if len(segment.times) > 0:
                            raws.append(segment)
                    except Exception:
                        continue
                if len(raws) > 0:
                    raw = mne.concatenate_raws(raws)
                    total_dur = raw.times[-1] - raw.times[0]
                    if total_dur < MIN_CONDITION_DURATION:
                        raise ValueError(
                            f"{sub_id}: {condition} duration {total_dur:.1f}s "
                            f"< minimum {MIN_CONDITION_DURATION}s")
                else:
                    raise ValueError(f"No valid {condition} segments for {sub_id}")

        # Load metadata
        metadata = {'subject': sub_id}
        participants_path = os.path.join(data_dir, 'participants.tsv')
        if os.path.isfile(participants_path):
            try:
                ptable = pd.read_csv(participants_path, sep='\t')
                prow = ptable[ptable['participant_id'] == sub_id]
                if len(prow) > 0:
                    metadata['age'] = prow.iloc[0].get('age', None)
                    metadata['sex'] = prow.iloc[0].get('sex', None)
            except Exception:
                pass

        return (raw.get_data(), raw.ch_names, raw.info['sfreq'], metadata)

    return load_hbn_subject


def run_release(release):
    """Run full phi replication protocol on a single HBN release."""
    data_dir = os.path.join(HBN_BASE, f'cmi_bids_{release}')
    out_dir = f'exports_hbn_{release}'

    if not os.path.isdir(data_dir):
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"  HBN {release} — Phi-Lattice Replication")
    print(f"{'#'*70}")
    print(f"Data: {data_dir}")
    print(f"Output: {out_dir}")

    subjects = discover_subjects(data_dir)
    print(f"Found {len(subjects)} subjects with RestingState data")

    if len(subjects) < 10:
        print("ERROR: Too few subjects!")
        return None

    loader_fn = make_loader(data_dir)

    # Run 3 conditions: combined (primary), EO, EC
    condition_dfs = {}
    for condition in ['combined', 'EO', 'EC']:
        print(f"\n\n{'='*60}")
        print(f"  {release} — Condition: {condition}")
        print(f"{'='*60}")

        loader = lambda sub, cond=condition: loader_fn(sub, cond)
        dom_df = process_subjects_from_eeg(
            subjects, loader,
            out_dir=os.path.join(out_dir, condition),
            label=f'HBN {release} {condition}')

        condition_dfs[condition] = dom_df

    # Theta EC convergence test
    if 'EO' in condition_dfs and 'EC' in condition_dfs:
        try:
            convergence = test_theta_ec_convergence(
                condition_dfs['EO'], condition_dfs['EC'])
        except Exception as e:
            print(f"  Theta convergence test failed: {e}")
            convergence = None

    # Pre-registration scoring
    print(f"\n\n{'='*60}")
    print(f"  {release} — PRE-REGISTRATION SCORING")
    print(f"{'='*60}")

    dom_df = condition_dfs.get('combined', pd.DataFrame())
    if len(dom_df) > 0:
        stats_result = run_statistics(dom_df, f'HBN {release} combined [OT]')
        enrichment_df = run_14position_enrichment(dom_df)

        if 'EO' in condition_dfs and 'EC' in condition_dfs:
            try:
                conv = test_theta_ec_convergence(
                    condition_dfs['EO'], condition_dfs['EC'])
                stats_result['theta_ec_p'] = conv.get('theta_ec_p', 1.0)
            except Exception:
                pass

        report = generate_preregistration_report(
            dom_df, enrichment_df, stats_result, label=f'HBN {release}')

        report_df = pd.DataFrame([report])
        report_df.to_csv(os.path.join(out_dir, 'preregistration_report.csv'),
                         index=False)

    print(f"\n\nDone with {release}! Results saved to {out_dir}")
    return condition_dfs


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_hbn_release_phi_replication.py <RELEASE>")
        print("  e.g.: python scripts/run_hbn_release_phi_replication.py R2")
        sys.exit(1)

    release = sys.argv[1].upper()
    if not release.startswith('R'):
        release = f'R{release}'

    run_release(release)
