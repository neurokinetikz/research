#!/usr/bin/env python3
"""Assemble Dryad data package for spectral differentiation paper.

Concatenates per-subject peak files, computes per-subject enrichment profiles,
and builds a demographics linkage table. All outputs go to dryad/.

Usage:
    python scripts/assemble_dryad.py --step peaks
    python scripts/assemble_dryad.py --step enrichment
    python scripts/assemble_dryad.py --step demographics
    python scripts/assemble_dryad.py --step all
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(BASE_DIR, 'lib'))
from phi_frequency_model import PHI, F0

OUT = os.path.join(BASE_DIR, 'dryad')
os.makedirs(OUT, exist_ok=True)

# ── Paper constants ──────────────────────────────────────────────────────
PHI_INV = 1.0 / PHI
MIN_POWER_PCT = 50
MIN_PEAKS_PER_BAND = 30

OCTAVE_BAND = {
    'n-1': 'theta', 'n+0': 'alpha', 'n+1': 'beta_low',
    'n+2': 'beta_high', 'n+3': 'gamma',
}
BAND_ORDER = ['theta', 'alpha', 'beta_low', 'beta_high', 'gamma']

# 12 Voronoi positions
POS_LIST = [
    ('boundary',    0.000),
    ('noble_6',     round(PHI_INV ** 6, 6)),
    ('noble_5',     round(PHI_INV ** 5, 6)),
    ('noble_4',     round(PHI_INV ** 4, 6)),
    ('noble_3',     round(PHI_INV ** 3, 6)),
    ('inv_noble_1', round(PHI_INV ** 2, 6)),
    ('attractor',   0.5),
    ('noble_1',     round(PHI_INV, 6)),
    ('inv_noble_3', round(1 - PHI_INV ** 3, 6)),
    ('inv_noble_4', round(1 - PHI_INV ** 4, 6)),
    ('inv_noble_5', round(1 - PHI_INV ** 5, 6)),
    ('inv_noble_6', round(1 - PHI_INV ** 6, 6)),
]
POS_NAMES = [p[0] for p in POS_LIST]
POS_VALS = np.array([p[1] for p in POS_LIST])
N_POS = len(POS_VALS)

# Voronoi bin edges
_VORONOI_EDGES = []
for i in range(N_POS):
    if i == 0:
        u_left = (POS_VALS[-1] + POS_VALS[0] + 1) / 2 % 1.0
        u_right = (POS_VALS[0] + POS_VALS[1]) / 2
    elif i == N_POS - 1:
        u_left = (POS_VALS[i - 1] + POS_VALS[i]) / 2
        u_right = (POS_VALS[i] + POS_VALS[0] + 1) / 2
    else:
        u_left = (POS_VALS[i - 1] + POS_VALS[i]) / 2
        u_right = (POS_VALS[i] + POS_VALS[i + 1]) / 2
    _VORONOI_EDGES.append((u_left, u_right))

# Hz-weighted bin fractions
HZ_FRACS = []
for i in range(N_POS):
    u_left, u_right = _VORONOI_EDGES[i]
    if i == 0:
        hz_frac = (PHI ** 1.0 - PHI ** u_left + PHI ** u_right - PHI ** 0.0) / (PHI - 1)
    else:
        hz_frac = (PHI ** u_right - PHI ** u_left) / (PHI - 1)
    HZ_FRACS.append(hz_frac)

# ── Dataset definitions (matching paper Table 1) ────────────────────────
# 9 EC datasets used in the paper
EC_DATASETS = {
    'eegmmidb': 'eegmmidb',
    'lemon': 'lemon',
    'dortmund': 'dortmund',
    # CHBMP excluded: CC BY-NC-SA license prevents CC0 redistribution
    # without explicit permission from Valdes-Sosa et al.
    # 'chbmp': 'chbmp',
    'hbn_R1': 'hbn_R1',
    'hbn_R2': 'hbn_R2',
    'hbn_R3': 'hbn_R3',
    'hbn_R4': 'hbn_R4',
    'hbn_R6': 'hbn_R6',
}

# Additional conditions for per-subject analyses
EXTRA_CONDITIONS = {
    'lemon_EO': 'lemon_EO',
    'dortmund_EC_post': 'dortmund_EC_post',
    'dortmund_EO_pre': 'dortmund_EO_pre',
    'dortmund_EO_post': 'dortmund_EO_post',
    'dortmund_EC_pre_ses2': 'dortmund_EC_pre_ses2',
    'dortmund_EC_post_ses2': 'dortmund_EC_post_ses2',
    'dortmund_EO_pre_ses2': 'dortmund_EO_pre_ses2',
    'dortmund_EO_post_ses2': 'dortmund_EO_post_ses2',
}

# Parent dataset for each subdirectory
PARENT_DATASET = {
    'eegmmidb': 'EEGMMIDB', 'lemon': 'LEMON', 'lemon_EO': 'LEMON',
    'dortmund': 'Dortmund', 'dortmund_EC_post': 'Dortmund',
    'dortmund_EO_pre': 'Dortmund', 'dortmund_EO_post': 'Dortmund',
    'dortmund_EC_pre_ses2': 'Dortmund', 'dortmund_EC_post_ses2': 'Dortmund',
    'dortmund_EO_pre_ses2': 'Dortmund', 'dortmund_EO_post_ses2': 'Dortmund',
    'chbmp': 'CHBMP',
    'hbn_R1': 'HBN', 'hbn_R2': 'HBN', 'hbn_R3': 'HBN',
    'hbn_R4': 'HBN', 'hbn_R6': 'HBN',
}

CONDITION_MAP = {
    'eegmmidb': 'EC', 'lemon': 'EC', 'lemon_EO': 'EO',
    'dortmund': 'EC_pre', 'dortmund_EC_post': 'EC_post',
    'dortmund_EO_pre': 'EO_pre', 'dortmund_EO_post': 'EO_post',
    'dortmund_EC_pre_ses2': 'EC_pre', 'dortmund_EC_post_ses2': 'EC_post',
    'dortmund_EO_pre_ses2': 'EO_pre', 'dortmund_EO_post_ses2': 'EO_post',
    'chbmp': 'EC',
    'hbn_R1': 'EC', 'hbn_R2': 'EC', 'hbn_R3': 'EC',
    'hbn_R4': 'EC', 'hbn_R6': 'EC',
}

SESSION_MAP = {
    'dortmund_EC_pre_ses2': 'ses-2', 'dortmund_EC_post_ses2': 'ses-2',
    'dortmund_EO_pre_ses2': 'ses-2', 'dortmund_EO_post_ses2': 'ses-2',
}

HBN_RELEASE = {
    'hbn_R1': 'R1', 'hbn_R2': 'R2', 'hbn_R3': 'R3',
    'hbn_R4': 'R4', 'hbn_R6': 'R6',
}


def lattice_coord(freqs):
    return (np.log(freqs / F0) / np.log(PHI)) % 1.0


def assign_voronoi(u_vals):
    u = np.asarray(u_vals, dtype=float) % 1.0
    dists = np.abs(u[:, None] - POS_VALS[None, :])
    dists = np.minimum(dists, 1 - dists)
    return np.argmin(dists, axis=1)


def subject_id_from_filename(fname):
    """Extract subject ID from peak filename."""
    base = os.path.basename(fname).replace('_peaks.csv', '')
    return base


# =========================================================================
# STEP 1: Concatenate peaks
# =========================================================================
def assemble_peaks(method='fooof'):
    if method == 'fooof':
        peak_base = os.path.join(BASE_DIR, 'exports_adaptive_v3')
        out_file = os.path.join(OUT, 'peaks_fooof_v3.csv')
    else:
        peak_base = os.path.join(BASE_DIR, 'exports_irasa_v4')
        out_file = os.path.join(OUT, 'peaks_irasa.csv')

    all_datasets = {**EC_DATASETS, **EXTRA_CONDITIONS}
    all_rows = []
    total_files = 0

    for ds_name, subdir in sorted(all_datasets.items()):
        ds_dir = os.path.join(peak_base, subdir)
        if not os.path.isdir(ds_dir):
            print(f"  SKIP {ds_name}: {ds_dir} not found")
            continue
        files = sorted(glob.glob(os.path.join(ds_dir, '*_peaks.csv')))
        print(f"  {ds_name}: {len(files)} files")
        total_files += len(files)

        for f in files:
            df = pd.read_csv(f)
            df['subject_id'] = subject_id_from_filename(f)
            df['subdirectory'] = ds_name
            df['dataset'] = PARENT_DATASET.get(ds_name, ds_name)
            df['condition'] = CONDITION_MAP.get(ds_name, 'unknown')
            df['session'] = SESSION_MAP.get(ds_name, 'ses-1')
            if ds_name in HBN_RELEASE:
                df['hbn_release'] = HBN_RELEASE[ds_name]
            # Map phi_octave to band name
            df['band'] = df['phi_octave'].map(OCTAVE_BAND)
            # Compute u-coordinate
            df['u'] = lattice_coord(df['freq'].values)
            all_rows.append(df)

    master = pd.concat(all_rows, ignore_index=True)

    # Select and order columns for Dryad
    cols = [
        'subject_id', 'dataset', 'subdirectory', 'condition', 'session',
        'channel', 'freq', 'power', 'bandwidth', 'r_squared',
        'band', 'phi_octave', 'phi_octave_n', 'u',
        'target_lo', 'target_hi', 'fit_lo', 'fit_hi',
        'nperseg', 'freq_res',
    ]
    if 'hbn_release' in master.columns:
        cols.insert(5, 'hbn_release')
    available = [c for c in cols if c in master.columns]
    master = master[available]

    master.to_csv(out_file, index=False, float_format='%.6f')
    n_ec = master[master['subdirectory'].isin(EC_DATASETS)].shape[0]
    print(f"\n  Wrote {out_file}")
    print(f"  Total rows: {len(master):,} ({total_files} files)")
    print(f"  EC-only rows (9 paper datasets): {n_ec:,}")
    return master


# =========================================================================
# STEP 2: Per-subject enrichment profiles
# =========================================================================
def assemble_enrichment(method='fooof'):
    if method == 'fooof':
        peak_base = os.path.join(BASE_DIR, 'exports_adaptive_v3')
        out_voronoi = os.path.join(OUT, 'enrichment_per_subject_fooof.csv')
        out_metrics = os.path.join(OUT, 'enrichment_metrics_fooof.csv')
    else:
        peak_base = os.path.join(BASE_DIR, 'exports_irasa_v4')
        out_voronoi = os.path.join(OUT, 'enrichment_per_subject_irasa.csv')
        out_metrics = os.path.join(OUT, 'enrichment_metrics_irasa.csv')

    all_datasets = {**EC_DATASETS, **EXTRA_CONDITIONS}
    voronoi_rows = []
    metric_rows = []

    for ds_name, subdir in sorted(all_datasets.items()):
        ds_dir = os.path.join(peak_base, subdir)
        if not os.path.isdir(ds_dir):
            continue
        files = sorted(glob.glob(os.path.join(ds_dir, '*_peaks.csv')))
        n_subjects = 0

        for f in files:
            df = pd.read_csv(f)
            df['band'] = df['phi_octave'].map(OCTAVE_BAND)
            df['u'] = lattice_coord(df['freq'].values)
            subj = subject_id_from_filename(f)

            for band in BAND_ORDER:
                bdf = df[df['band'] == band].copy()
                if len(bdf) == 0:
                    continue

                # Power filter: top 50%
                if 'power' in bdf.columns and len(bdf) > 1:
                    thresh = bdf['power'].median()
                    bdf = bdf[bdf['power'] >= thresh]

                if len(bdf) < MIN_PEAKS_PER_BAND:
                    continue

                # Voronoi assignment
                pos_idx = assign_voronoi(bdf['u'].values)
                n_total = len(bdf)
                for i in range(N_POS):
                    n_obs = np.sum(pos_idx == i)
                    n_exp = HZ_FRACS[i] * n_total
                    enr = (n_obs / n_exp - 1) * 100 if n_exp > 0 else np.nan
                    voronoi_rows.append({
                        'subject_id': subj,
                        'dataset': PARENT_DATASET.get(ds_name, ds_name),
                        'subdirectory': ds_name,
                        'condition': CONDITION_MAP.get(ds_name, 'unknown'),
                        'session': SESSION_MAP.get(ds_name, 'ses-1'),
                        'band': band,
                        'position': POS_NAMES[i],
                        'position_u': POS_VALS[i],
                        'n_observed': int(n_obs),
                        'n_expected': round(n_exp, 2),
                        'enrichment_pct': round(enr, 2) if not np.isnan(enr) else np.nan,
                        'n_peaks_band': n_total,
                    })

                # Derived metrics (6 per band)
                enr_by_pos = {}
                for i in range(N_POS):
                    n_obs = np.sum(pos_idx == i)
                    n_exp = HZ_FRACS[i] * n_total
                    enr_by_pos[POS_NAMES[i]] = (n_obs / n_exp - 1) * 100 if n_exp > 0 else 0

                mountain = enr_by_pos['noble_1'] - enr_by_pos['boundary']
                ushape = np.mean([enr_by_pos['boundary'], enr_by_pos['inv_noble_6']]) - enr_by_pos['attractor']
                peak_height = enr_by_pos['noble_1'] - enr_by_pos['attractor']
                ramp_depth = enr_by_pos['inv_noble_4'] - enr_by_pos['noble_4']
                lower_interior = np.mean([enr_by_pos[p] for p in ['noble_6', 'noble_5', 'noble_4', 'noble_3', 'inv_noble_1']])
                center_depletion = enr_by_pos['attractor'] - lower_interior
                upper_interior = np.mean([enr_by_pos[p] for p in ['inv_noble_3', 'inv_noble_4', 'inv_noble_5', 'inv_noble_6']])
                asymmetry = upper_interior - lower_interior

                metric_rows.append({
                    'subject_id': subj,
                    'dataset': PARENT_DATASET.get(ds_name, ds_name),
                    'subdirectory': ds_name,
                    'condition': CONDITION_MAP.get(ds_name, 'unknown'),
                    'session': SESSION_MAP.get(ds_name, 'ses-1'),
                    'band': band,
                    'n_peaks': n_total,
                    'mountain': round(mountain, 2),
                    'ushape': round(ushape, 2),
                    'peak_height': round(peak_height, 2),
                    'ramp_depth': round(ramp_depth, 2),
                    'center_depletion': round(center_depletion, 2),
                    'asymmetry': round(asymmetry, 2),
                })

            n_subjects += 1

        print(f"  {ds_name}: {n_subjects} subjects with ≥{MIN_PEAKS_PER_BAND} peaks/band")

    voronoi_df = pd.DataFrame(voronoi_rows)
    metrics_df = pd.DataFrame(metric_rows)

    voronoi_df.to_csv(out_voronoi, index=False)
    metrics_df.to_csv(out_metrics, index=False)
    print(f"\n  Wrote {out_voronoi} ({len(voronoi_df):,} rows)")
    print(f"  Wrote {out_metrics} ({len(metrics_df):,} rows)")
    return voronoi_df, metrics_df


# =========================================================================
# STEP 3: Demographics linkage table
# =========================================================================
def assemble_demographics():
    """Build subject demographics from each dataset's participants file."""
    rows = []

    # --- LEMON ---
    lemon_meta = '/Volumes/T9/lemon_data/behavioral/Behavioural_Data_MPILMBB_LEMON/META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv'
    if os.path.exists(lemon_meta):
        lm = pd.read_csv(lemon_meta)
        for _, r in lm.iterrows():
            sex_raw = r.get('Gender_ 1=female_2=male', np.nan)
            sex = 'F' if sex_raw == 1 else ('M' if sex_raw == 2 else np.nan)
            sid = str(r['ID'])
            if not sid.startswith('sub-'):
                sid = 'sub-' + sid
            rows.append({
                'subject_id': sid,
                'dataset': 'LEMON',
                'age': str(r['Age']) if pd.notna(r.get('Age')) else np.nan,
                'sex': sex,
                'handedness': r.get('Handedness', np.nan),
            })
        print(f"  LEMON: {len(lm)} subjects from META file")
    else:
        print(f"  LEMON: META file not found at {lemon_meta}")

    # --- Dortmund ---
    dort_tsv = '/Volumes/T9/dortmund_data/participants.tsv'
    if os.path.exists(dort_tsv):
        dm = pd.read_csv(dort_tsv, sep='\t')
        for _, r in dm.iterrows():
            sid = str(r['participant_id'])
            if not sid.startswith('sub-'):
                sid = 'sub-' + sid
            rows.append({
                'subject_id': sid,
                'dataset': 'Dortmund',
                'age': r.get('age', np.nan),
                'sex': r.get('sex', np.nan),
                'handedness': r.get('handedness', np.nan),
            })
        print(f"  Dortmund: {len(dm)} subjects from participants.tsv")
    else:
        print(f"  Dortmund: participants.tsv not found at {dort_tsv}")

    # --- HBN ---
    for release in ['R1', 'R2', 'R3', 'R4', 'R6']:
        hbn_tsv = f'/Volumes/T9/hbn_data/cmi_bids_{release}/participants.tsv'
        if os.path.exists(hbn_tsv):
            hm = pd.read_csv(hbn_tsv, sep='\t')
            for _, r in hm.iterrows():
                sid = str(r['participant_id'])
                if not sid.startswith('sub-'):
                    sid = 'sub-' + sid
                rows.append({
                    'subject_id': sid,
                    'dataset': 'HBN',
                    'hbn_release': release,
                    'age': r.get('age', np.nan),
                    'sex': r.get('sex', np.nan),
                    'handedness': r.get('ehq_total', np.nan),
                })
            print(f"  HBN {release}: {len(hm)} subjects")
        else:
            print(f"  HBN {release}: participants.tsv not found at {hbn_tsv}")

    # --- CHBMP excluded from Dryad (CC BY-NC-SA) ---
    print("  CHBMP: excluded (CC BY-NC-SA license; peaks available from CAN-BIND portal)")

    # --- EEGMMIDB (no demographics file; all "adult") ---
    eeg_dir = os.path.join(BASE_DIR, 'exports_adaptive_v3', 'eegmmidb')
    eeg_files = glob.glob(os.path.join(eeg_dir, '*_peaks.csv'))
    for f in eeg_files:
        rows.append({
            'subject_id': subject_id_from_filename(f),
            'dataset': 'EEGMMIDB',
            'age': np.nan,
            'sex': np.nan,
        })
    print(f"  EEGMMIDB: {len(eeg_files)} subjects (no demographics available)")

    demo = pd.DataFrame(rows)
    out_file = os.path.join(OUT, 'subjects_demographics.csv')
    demo.to_csv(out_file, index=False)
    print(f"\n  Wrote {out_file} ({len(demo)} rows)")
    return demo


# =========================================================================
# Main
# =========================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Assemble Dryad data package')
    parser.add_argument('--step', choices=['peaks', 'enrichment', 'demographics', 'all'],
                        default='all')
    args = parser.parse_args()

    if args.step in ('peaks', 'all'):
        print("=" * 60)
        print("STEP 1a: Concatenating FOOOF v3 peaks")
        print("=" * 60)
        assemble_peaks('fooof')

        print("\n" + "=" * 60)
        print("STEP 1b: Concatenating IRASA peaks")
        print("=" * 60)
        assemble_peaks('irasa')

    if args.step in ('enrichment', 'all'):
        print("\n" + "=" * 60)
        print("STEP 2a: Computing per-subject enrichment (FOOOF)")
        print("=" * 60)
        assemble_enrichment('fooof')

        print("\n" + "=" * 60)
        print("STEP 2b: Computing per-subject enrichment (IRASA)")
        print("=" * 60)
        assemble_enrichment('irasa')

    if args.step in ('demographics', 'all'):
        print("\n" + "=" * 60)
        print("STEP 3: Assembling demographics")
        print("=" * 60)
        assemble_demographics()

    print("\nDone. Files in:", OUT)
