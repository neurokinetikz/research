#!/usr/bin/env python3
"""
IRASA Per-Subject Trough Depths → Functional Correlations
==========================================================

Replicates the paper's per-subject trough depth computation (windowed
log-frequency counts) using IRASA peaks instead of FOOOF, then reruns
the psychopathology (HBN) and cognition (LEMON) correlations.

Uses EXACTLY the same depth metric and window parameters as
trough_depth_covariance.py (LOG_HALF_WINDOW=0.06, LOG_FLANK_OFFSET=0.15).

Usage:
    python scripts/irasa_trough_depth_functional.py

Outputs to: outputs/irasa_trough_functional/
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
IRASA_BASE = os.path.join(BASE_DIR, 'exports_irasa_v4')
FOOOF_BASE = os.path.join(BASE_DIR, 'exports_adaptive_v3')
OUT_DIR = os.path.join(BASE_DIR, 'outputs', 'irasa_trough_functional')
MIN_POWER_PCT = 50

# EXACTLY matching trough_depth_covariance.py parameters
KNOWN_TROUGHS_HZ = np.array([5.08, 7.81, 13.42, 25.30, 35.04])
SHORT_LABELS = ['δ/θ', 'θ/α', 'α/β', 'βL/βH', 'βH/γ']
LOG_HALF_WINDOW = 0.06
LOG_FLANK_OFFSET = 0.15

# Demographics
HBN_RELEASES = ['R1', 'R2', 'R3', 'R4', 'R6']
HBN_DEMO_TEMPLATE = '/Volumes/T9/hbn_data/cmi_bids_{release}/participants.tsv'
LEMON_DEMO = ('/Volumes/T9/lemon_data/behavioral/'
              'Behavioural_Data_MPILMBB_LEMON/'
              'META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv')
LEMON_AGE_MAP = {
    '20-25': 22.5, '25-30': 27.5, '30-35': 32.5, '35-40': 37.5,
    '55-60': 57.5, '60-65': 62.5, '65-70': 67.5, '70-75': 72.5, '75-80': 77.5,
}
LEMON_COG_DIR = '/Volumes/T9/lemon_data/behavioral/Behavioural_Data_MPILMBB_LEMON'


def per_subject_trough_depth(freqs, trough_hz):
    """Exact copy of paper's method from trough_depth_covariance.py."""
    log_freqs = np.log(freqs)
    log_trough = np.log(trough_hz)

    trough_mask = np.abs(log_freqs - log_trough) < LOG_HALF_WINDOW
    trough_count = trough_mask.sum()

    left_center = log_trough - LOG_FLANK_OFFSET
    left_mask = np.abs(log_freqs - left_center) < LOG_HALF_WINDOW
    left_count = left_mask.sum()

    right_center = log_trough + LOG_FLANK_OFFSET
    right_mask = np.abs(log_freqs - right_center) < LOG_HALF_WINDOW
    right_count = right_mask.sum()

    mean_flank = (left_count + right_count) / 2
    if mean_flank > 0:
        return trough_count / mean_flank
    return np.nan


def load_subjects_with_depths(base_dir, datasets):
    """Load per-subject peaks and compute trough depths."""
    rows = []
    for subdir, base_label in datasets.items():
        path = os.path.join(base_dir, subdir)
        files = sorted(glob.glob(os.path.join(path, '*_peaks.csv')))
        if not files:
            continue
        first = pd.read_csv(files[0], nrows=1)
        has_power = 'power' in first.columns
        cols = ['freq'] + (['power', 'phi_octave'] if has_power else ['phi_octave'])
        count = 0
        for f in files:
            subj_id = os.path.basename(f).replace('_peaks.csv', '')
            try:
                df = pd.read_csv(f, usecols=cols)
            except Exception:
                continue
            if has_power and MIN_POWER_PCT > 0:
                filtered = []
                for octave in df['phi_octave'].unique():
                    bp = df[df.phi_octave == octave]
                    if len(bp) == 0:
                        continue
                    thresh = bp['power'].quantile(MIN_POWER_PCT / 100)
                    filtered.append(bp[bp['power'] >= thresh])
                if filtered:
                    df = pd.concat(filtered, ignore_index=True)
                else:
                    continue
            freqs = df['freq'].values
            if len(freqs) < 100:
                continue
            row = {'subject': subj_id, 'dataset': base_label, 'n_peaks': len(freqs)}
            for trough_hz, label in zip(KNOWN_TROUGHS_HZ, SHORT_LABELS):
                row[f'depth_{label}'] = per_subject_trough_depth(freqs, trough_hz)
            rows.append(row)
            count += 1
        print(f"  {subdir}: {count} subjects")
    return pd.DataFrame(rows)


def load_hbn_demographics():
    """Load HBN age and psychopathology."""
    demo = {}
    for release in HBN_RELEASES:
        tsv = HBN_DEMO_TEMPLATE.format(release=release)
        if not os.path.exists(tsv):
            continue
        df = pd.read_csv(tsv, sep='\t')
        for _, row in df.iterrows():
            pid = row.get('participant_id', '')
            entry = {}
            for col in ['age', 'Age']:
                if col in df.columns and pd.notna(row.get(col)):
                    entry['age'] = float(row[col])
            for col in ['p_factor', 'attention', 'internalizing', 'externalizing']:
                if col in df.columns and pd.notna(row.get(col)):
                    entry[col] = float(row[col])
            if entry:
                demo[pid] = entry
    return demo


def load_lemon_cognitive():
    """Load LEMON cognitive test scores."""
    cog_files = {
        'LPS': 'Cognitive_Data_MPILMBB/LPS/LPS.csv',
        'RWT': 'Cognitive_Data_MPILMBB/RWT/RWT.csv',
        'TMT': 'Cognitive_Data_MPILMBB/TMT/TMT.csv',
        'TAP_Alert': 'Cognitive_Data_MPILMBB/TAP/TAP_Alertness.csv',
        'TAP_WM': 'Cognitive_Data_MPILMBB/TAP/TAP_WorkingMemory.csv',
        'WST': 'Cognitive_Data_MPILMBB/WST/WST.csv',
        'CVLT': 'Cognitive_Data_MPILMBB/CVLT/CVLT.csv',
    }
    cog = {}
    for test_name, rel_path in cog_files.items():
        full = os.path.join(LEMON_COG_DIR, rel_path)
        if not os.path.exists(full):
            continue
        try:
            df = pd.read_csv(full)
            id_col = [c for c in df.columns if 'ID' in c.upper() or 'participant' in c.lower()]
            if not id_col:
                continue
            id_col = id_col[0]
            # Take first numeric column as score
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not num_cols:
                continue
            score_col = num_cols[0]
            for _, row in df.iterrows():
                pid = str(row[id_col])
                score = row[score_col]
                if pd.notna(score):
                    if pid not in cog:
                        cog[pid] = {}
                    cog[pid][test_name] = float(score)
        except Exception:
            continue
    return cog


def run_correlations(df, variables, method_label):
    """Run Spearman correlations between trough depths and variables."""
    results = []
    depth_cols = [f'depth_{l}' for l in SHORT_LABELS]

    for var in variables:
        if var not in df.columns:
            continue
        for col, label in zip(depth_cols, SHORT_LABELS):
            valid = df.dropna(subset=[var, col])
            if len(valid) < 30:
                continue
            rho, p = stats.spearmanr(valid[var], valid[col])
            results.append({
                'variable': var, 'trough': label, 'method': method_label,
                'rho': rho, 'p': p, 'n': len(valid),
            })
    return pd.DataFrame(results)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("IRASA Per-Subject Trough Depths → Functional Correlations")
    print("=" * 70)

    # --- Load HBN data from both methods ---
    hbn_ds = {ds: 'hbn' for ds in ['hbn_R1', 'hbn_R2', 'hbn_R3', 'hbn_R4', 'hbn_R6']}

    print("\n--- Loading IRASA HBN peaks ---")
    irasa_hbn = load_subjects_with_depths(IRASA_BASE, hbn_ds)

    print("\n--- Loading FOOOF HBN peaks ---")
    fooof_hbn = load_subjects_with_depths(FOOOF_BASE, hbn_ds)

    # Merge demographics
    demo = load_hbn_demographics()
    for df_m in [irasa_hbn, fooof_hbn]:
        for col in ['age', 'externalizing', 'internalizing', 'p_factor', 'attention']:
            df_m[col] = df_m['subject'].map(lambda s: demo.get(s, {}).get(col, np.nan))

    print(f"\n  IRASA HBN: {len(irasa_hbn)} subjects")
    print(f"  FOOOF HBN: {len(fooof_hbn)} subjects")

    # --- Psychopathology correlations ---
    print("\n" + "=" * 70)
    print("HBN Psychopathology × Trough Depth")
    print("=" * 70)

    psych_vars = ['externalizing', 'internalizing', 'p_factor', 'attention']
    irasa_psych = run_correlations(irasa_hbn, psych_vars, 'irasa')
    fooof_psych = run_correlations(fooof_hbn, psych_vars, 'fooof')

    # Side-by-side comparison at α/β
    print(f"\n  --- α/β Trough × Psychopathology ---")
    print(f"  {'Variable':<16} {'FOOOF ρ':>10} {'p':>10} {'IRASA ρ':>10} {'p':>10} {'Sign':>6}")
    print("  " + "-" * 65)
    for var in psych_vars:
        f_row = fooof_psych[(fooof_psych.variable == var) & (fooof_psych.trough == 'α/β')]
        i_row = irasa_psych[(irasa_psych.variable == var) & (irasa_psych.trough == 'α/β')]
        if len(f_row) > 0 and len(i_row) > 0:
            fr, fp = f_row.iloc[0]['rho'], f_row.iloc[0]['p']
            ir, ip = i_row.iloc[0]['rho'], i_row.iloc[0]['p']
            sign = '✓' if np.sign(fr) == np.sign(ir) else '✗'
            f_sig = '*' if fp < 0.05 else ' '
            i_sig = '*' if ip < 0.05 else ' '
            print(f"  {var:<16} {fr:>+10.3f}{f_sig} {fp:>9.4f} {ir:>+10.3f}{i_sig} {ip:>9.4f} {sign:>6}")

    # Full table for all troughs × externalizing
    print(f"\n  --- Externalizing × All Troughs ---")
    print(f"  {'Trough':<10} {'FOOOF ρ':>10} {'p':>10} {'IRASA ρ':>10} {'p':>10} {'Sign':>6}")
    print("  " + "-" * 55)
    for label in SHORT_LABELS:
        f_row = fooof_psych[(fooof_psych.variable == 'externalizing') & (fooof_psych.trough == label)]
        i_row = irasa_psych[(irasa_psych.variable == 'externalizing') & (irasa_psych.trough == label)]
        if len(f_row) > 0 and len(i_row) > 0:
            fr, fp = f_row.iloc[0]['rho'], f_row.iloc[0]['p']
            ir, ip = i_row.iloc[0]['rho'], i_row.iloc[0]['p']
            sign = '✓' if np.sign(fr) == np.sign(ir) else '✗'
            f_sig = '*' if fp < 0.05 else ' '
            i_sig = '*' if ip < 0.05 else ' '
            print(f"  {label:<10} {fr:>+10.3f}{f_sig} {fp:>9.4f} {ir:>+10.3f}{i_sig} {ip:>9.4f} {sign:>6}")

    # Internalizing
    print(f"\n  --- Internalizing × All Troughs ---")
    print(f"  {'Trough':<10} {'FOOOF ρ':>10} {'p':>10} {'IRASA ρ':>10} {'p':>10} {'Sign':>6}")
    print("  " + "-" * 55)
    for label in SHORT_LABELS:
        f_row = fooof_psych[(fooof_psych.variable == 'internalizing') & (fooof_psych.trough == label)]
        i_row = irasa_psych[(irasa_psych.variable == 'internalizing') & (irasa_psych.trough == label)]
        if len(f_row) > 0 and len(i_row) > 0:
            fr, fp = f_row.iloc[0]['rho'], f_row.iloc[0]['p']
            ir, ip = i_row.iloc[0]['rho'], i_row.iloc[0]['p']
            sign = '✓' if np.sign(fr) == np.sign(ir) else '✗'
            f_sig = '*' if fp < 0.05 else ' '
            i_sig = '*' if ip < 0.05 else ' '
            print(f"  {label:<10} {fr:>+10.3f}{f_sig} {fp:>9.4f} {ir:>+10.3f}{i_sig} {ip:>9.4f} {sign:>6}")

    # --- LEMON Cognition ---
    print("\n" + "=" * 70)
    print("LEMON Cognition × Trough Depth")
    print("=" * 70)

    lemon_ds = {'lemon': 'lemon'}
    print("\n  Loading IRASA LEMON peaks...")
    irasa_lemon = load_subjects_with_depths(IRASA_BASE, lemon_ds)
    print("  Loading FOOOF LEMON peaks...")
    fooof_lemon = load_subjects_with_depths(FOOOF_BASE, lemon_ds)

    # Load cognitive scores
    cog = load_lemon_cognitive()
    cog_tests = list(set(t for scores in cog.values() for t in scores.keys()))

    for df_m in [irasa_lemon, fooof_lemon]:
        for test in cog_tests:
            df_m[test] = df_m['subject'].map(lambda s, t=test: cog.get(s, {}).get(t, np.nan))

    print(f"  IRASA LEMON: {len(irasa_lemon)} subjects")
    print(f"  FOOOF LEMON: {len(fooof_lemon)} subjects")
    print(f"  Cognitive tests found: {sorted(cog_tests)}")

    irasa_cog = run_correlations(irasa_lemon, cog_tests, 'irasa')
    fooof_cog = run_correlations(fooof_lemon, cog_tests, 'fooof')

    print(f"\n  --- α/β Trough × Cognition ---")
    print(f"  {'Test':<16} {'FOOOF ρ':>10} {'p':>10} {'IRASA ρ':>10} {'p':>10} {'Sign':>6}")
    print("  " + "-" * 65)
    for test in sorted(cog_tests):
        f_row = fooof_cog[(fooof_cog.variable == test) & (fooof_cog.trough == 'α/β')]
        i_row = irasa_cog[(irasa_cog.variable == test) & (irasa_cog.trough == 'α/β')]
        if len(f_row) > 0 and len(i_row) > 0:
            fr, fp = f_row.iloc[0]['rho'], f_row.iloc[0]['p']
            ir, ip = i_row.iloc[0]['rho'], i_row.iloc[0]['p']
            sign = '✓' if np.sign(fr) == np.sign(ir) else '✗'
            f_sig = '*' if fp < 0.05 else ' '
            i_sig = '*' if ip < 0.05 else ' '
            print(f"  {test:<16} {fr:>+10.3f}{f_sig} {fp:>9.4f} {ir:>+10.3f}{i_sig} {ip:>9.4f} {sign:>6}")

    # Save all
    all_results = pd.concat([irasa_psych, fooof_psych, irasa_cog, fooof_cog], ignore_index=True)
    all_results.to_csv(os.path.join(OUT_DIR, 'functional_correlations_both_methods.csv'), index=False)

    irasa_hbn.to_csv(os.path.join(OUT_DIR, 'irasa_hbn_per_subject_depths.csv'), index=False)
    fooof_hbn.to_csv(os.path.join(OUT_DIR, 'fooof_hbn_per_subject_depths.csv'), index=False)

    print(f"\nResults saved to {OUT_DIR}/")


if __name__ == '__main__':
    main()
