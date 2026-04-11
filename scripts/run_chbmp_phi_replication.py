#!/usr/bin/env python3
"""
CHBMP Phi-Lattice Replication
==============================

Runs the locked phi_replication protocol on the Cuban Human Brain Mapping
Project (CHBMP) dataset.

Dataset: N=282 healthy adults (18-68 yrs), 120-channel EEG, 200 Hz, EDF format.
Protocol: 15 min EC → alternating EC/EO blocks → hyperventilation → recovery → photic stim.
We only use the resting-state portion (before hyperventilation).

Events (Spanish): "ojos cerrados" (value=65) = EC, "ojos abiertos" (value=66) = EO.
Hyperventilation (value=67) marks end of resting state.

Usage:
    PYTHONUNBUFFERED=1 /opt/anaconda3/bin/python scripts/run_chbmp_phi_replication.py
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
CHBMP_DIR = '/Volumes/T9/CHBMP/BIDS_dataset'
OUT_DIR = 'exports_chbmp'
MIN_CONDITION_DURATION = 10.0


def discover_subjects(data_dir):
    """Find all subjects with protmap EDF files."""
    pattern = os.path.join(data_dir, 'sub-CBM*', 'ses-V01', 'eeg',
                           'sub-CBM*_ses-V01_task-protmap_eeg.edf')
    files = sorted(glob.glob(pattern))
    subjects = []
    seen = set()
    for f in files:
        basename = os.path.basename(f)
        if basename.startswith('._'):
            continue
        # Extract sub-CBMxxxxx
        parts = f.split(os.sep)
        for p in parts:
            if p.startswith('sub-CBM'):
                if p not in seen:
                    subjects.append(p)
                    seen.add(p)
                break
    return subjects


def parse_events(data_dir, sub_id):
    """Parse events.tsv to get EC/EO segment boundaries.

    Returns dict with 'EC' and 'EO' lists of (onset, offset) tuples.
    Only includes segments before hyperventilation starts.
    """
    events_path = os.path.join(data_dir, sub_id, 'ses-V01', 'eeg',
                                f'{sub_id}_ses-V01_task-protmap_events.tsv')
    if not os.path.isfile(events_path):
        return None

    events = pd.read_csv(events_path, sep='\t')

    # Find when hyperventilation starts (value=67) — everything after is excluded
    hv_rows = events[events['value'] == 67]
    hv_onset = hv_rows['onset'].values[0] if len(hv_rows) > 0 else float('inf')

    # Filter to resting-state events only
    rest_events = events[events['onset'] < hv_onset].copy()

    # Get EC (value=65) and EO (value=66) onsets
    ec_onsets = rest_events[rest_events['value'] == 65]['onset'].values
    eo_onsets = rest_events[rest_events['value'] == 66]['onset'].values

    all_onsets = sorted(list(ec_onsets) + list(eo_onsets))

    segments = {'EO': [], 'EC': []}

    for onset in ec_onsets:
        later = [t for t in all_onsets if t > onset]
        offset = later[0] if later else hv_onset
        if offset - onset >= 2.0:  # skip trivially short segments
            segments['EC'].append((onset, offset))

    for onset in eo_onsets:
        later = [t for t in all_onsets if t > onset]
        offset = later[0] if later else hv_onset
        if offset - onset >= 2.0:
            segments['EO'].append((onset, offset))

    return segments


def make_loader(data_dir):
    """Create a loader function for CHBMP EDF files."""

    def load_chbmp_subject(sub_id, condition='combined'):
        edf_path = os.path.join(data_dir, sub_id, 'ses-V01', 'eeg',
                                 f'{sub_id}_ses-V01_task-protmap_eeg.edf')
        if not os.path.isfile(edf_path):
            raise FileNotFoundError(f"No EDF file for {sub_id}")

        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

        # Pick only EEG channels (exclude EOG: EOI, EOD; exclude ECG)
        eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False,
                                    ecg=False, misc=False, exclude='bads')
        raw.pick(eeg_picks)

        # Notch filter at 60 Hz + harmonics (Cuban power grid = 60 Hz)
        raw.notch_filter(freqs=[60], verbose=False)

        # Parse events to find resting-state segments
        segments = parse_events(data_dir, sub_id)

        if condition == 'combined':
            # Use all resting-state segments (EC + EO, before hyperventilation)
            if segments:
                all_segs = sorted(segments['EC'] + segments['EO'],
                                   key=lambda x: x[0])
                if len(all_segs) > 0:
                    raws = []
                    for onset, offset in all_segs:
                        try:
                            seg = raw.copy().crop(
                                tmin=onset,
                                tmax=min(offset, raw.times[-1]))
                            if len(seg.times) > 0:
                                raws.append(seg)
                        except Exception:
                            continue
                    if len(raws) > 0:
                        raw = mne.concatenate_raws(raws)
            else:
                # No events file — use first 1000s as fallback
                raw.crop(tmin=0, tmax=min(1000.0, raw.times[-1]))

        elif condition in ('EC', 'EO'):
            if segments and condition in segments and len(segments[condition]) > 0:
                raws = []
                for onset, offset in segments[condition]:
                    try:
                        seg = raw.copy().crop(
                            tmin=onset,
                            tmax=min(offset, raw.times[-1]))
                        if len(seg.times) > 0:
                            raws.append(seg)
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
                    raise ValueError(
                        f"No valid {condition} segments for {sub_id}")
            else:
                raise ValueError(
                    f"No event data for {condition} in {sub_id}")

        # Load metadata (age, sex)
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

    return load_chbmp_subject


def run_chbmp():
    """Run full phi replication protocol on CHBMP."""
    if not os.path.isdir(CHBMP_DIR):
        print(f"ERROR: Data directory not found: {CHBMP_DIR}")
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"  CHBMP — Phi-Lattice Replication")
    print(f"{'#'*70}")
    print(f"Data: {CHBMP_DIR}")
    print(f"Output: {OUT_DIR}")

    subjects = discover_subjects(CHBMP_DIR)
    print(f"Found {len(subjects)} subjects with protmap EEG data")

    if len(subjects) < 10:
        print("ERROR: Too few subjects!")
        return None

    loader_fn = make_loader(CHBMP_DIR)

    # Run 3 conditions: combined (primary), EO, EC
    condition_dfs = {}
    for condition in ['combined', 'EO', 'EC']:
        print(f"\n\n{'='*60}")
        print(f"  CHBMP — Condition: {condition}")
        print(f"{'='*60}")

        loader = lambda sub, cond=condition: loader_fn(sub, cond)
        dom_df = process_subjects_from_eeg(
            subjects, loader,
            out_dir=os.path.join(OUT_DIR, condition),
            label=f'CHBMP {condition}')

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
    print(f"  CHBMP — PRE-REGISTRATION SCORING")
    print(f"{'='*60}")

    dom_df = condition_dfs.get('combined', pd.DataFrame())
    if len(dom_df) > 0:
        stats_result = run_statistics(dom_df, 'CHBMP combined [OT]')
        enrichment_df = run_14position_enrichment(dom_df)

        if 'EO' in condition_dfs and 'EC' in condition_dfs:
            try:
                conv = test_theta_ec_convergence(
                    condition_dfs['EO'], condition_dfs['EC'])
                stats_result['theta_ec_p'] = conv.get('theta_ec_p', 1.0)
            except Exception:
                pass

        report = generate_preregistration_report(
            dom_df, enrichment_df, stats_result, label='CHBMP')

        report_df = pd.DataFrame([report])
        report_df.to_csv(os.path.join(OUT_DIR, 'preregistration_report.csv'),
                         index=False)

    print(f"\n\nDone with CHBMP! Results saved to {OUT_DIR}")
    return condition_dfs


if __name__ == '__main__':
    run_chbmp()
