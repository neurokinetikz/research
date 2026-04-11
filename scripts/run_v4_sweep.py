#!/usr/bin/env python3
"""
v4 FOOOF Parameter Sweep
=========================

Runs extraction with different FOOOF parameter combinations on EEGMMIDB.
Then computes Hz-corrected enrichment at multiple power filter levels.
Outputs a comparison across all configurations.

Usage (single config):
    python scripts/run_v4_sweep.py --config 1

Usage (all configs):
    python scripts/run_v4_sweep.py --all

Usage (parallel, N workers):
    python scripts/run_v4_sweep.py --all --parallel 8
"""

import os
import sys
import time
import argparse
import warnings
import gc
from glob import glob
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from scipy.signal import welch
import mne

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
from lemon_utils import _get_peak_params, _get_r_squared

try:
    from specparam import SpectralModel
except ImportError:
    from fooof import FOOOF as SpectralModel

# =========================================================================
# CONSTANTS
# =========================================================================
PHI = (1 + np.sqrt(5)) / 2
F0 = 7.60
FILTER_LO = 1.0
R2_MIN = 0.70
PHI_INV = 1.0 / PHI

# Degree-7 positions for nperseg calculation
POS_14 = sorted(set([0.0, 0.5] +
                     [round(PHI_INV**k, 6) for k in range(1, 8)] +
                     [round(1 - PHI_INV**k, 6) for k in range(1, 8)]))
POS_14 = [p for p in POS_14 if 0 <= p < 1]
MIN_SEP = min(POS_14[i+1] - POS_14[i] for i in range(len(POS_14)-1))

# Data paths (VM layout)
EEGMMIDB_DIR = os.environ.get('EEGMMIDB_DIR', '/Volumes/T9/eegmmidb')
OUTPUT_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'exports_v4_sweep')

# 16 parameter configurations
CONFIGS = [
    # id, peak_threshold, min_peak_height, max_n_peaks, nperseg_mode, label
    (1,  0.001, 0.0001, 15, 'adaptive', 'v3 baseline'),
    (2,  0.001, 0.0001, 5,  'adaptive', 'v3 + old cap=5'),
    (3,  0.001, 0.0001, 20, 'adaptive', 'v3 + generous cap=20'),
    (4,  0.5,   0.0001, 15, 'adaptive', 'moderate relative'),
    (5,  1.0,   0.0001, 15, 'adaptive', 'strict relative'),
    (6,  2.0,   0.0001, 15, 'adaptive', 'FOOOF default relative'),
    (7,  0.001, 0.05,   15, 'adaptive', 'moderate absolute'),
    (8,  0.001, 0.10,   15, 'adaptive', 'strict absolute'),
    (9,  0.001, 0.20,   15, 'adaptive', 'very strict absolute'),
    (10, 1.0,   0.05,   15, 'adaptive', 'combined moderate'),
    (11, 1.0,   0.10,   15, 'adaptive', 'combined strict'),
    (12, 2.0,   0.10,   15, 'adaptive', 'FOOOF default + absolute'),
    (13, 0.5,   0.05,   10, 'adaptive', 'all moderate'),
    (14, 1.0,   0.05,   20, 'adaptive', 'strict thresh, generous cap'),
    (15, 0.001, 0.0001, 15, '2500',     'v3 + nperseg floor=2500'),
    (16, 1.0,   0.05,   15, '2500',     'combined moderate + floor=2500'),
    # --- ROUND 2: Zero in on max_n_peaks and min_peak_height ---
    # Fine-grained max_n_peaks (the dominant parameter)
    (17, 0.001, 0.0001, 7,  'adaptive', 'cap=7'),
    (18, 0.001, 0.0001, 8,  'adaptive', 'cap=8'),
    (19, 0.001, 0.0001, 9,  'adaptive', 'cap=9'),
    (20, 0.001, 0.0001, 10, 'adaptive', 'cap=10'),
    (21, 0.001, 0.0001, 11, 'adaptive', 'cap=11'),
    (22, 0.001, 0.0001, 12, 'adaptive', 'cap=12'),
    (23, 0.001, 0.0001, 13, 'adaptive', 'cap=13'),
    (24, 0.001, 0.0001, 14, 'adaptive', 'cap=14'),
    # min_peak_height as power filter substitute (no analysis-time filter needed?)
    (25, 0.001, 0.03,   15, 'adaptive', 'height=0.03 (10th pct)'),
    (26, 0.001, 0.08,   15, 'adaptive', 'height=0.08 (25th pct)'),
    (27, 0.001, 0.13,   15, 'adaptive', 'height=0.13 (50th pct)'),
    (28, 0.001, 0.15,   15, 'adaptive', 'height=0.15 (60th pct)'),
    # Interaction: optimal cap + height (does combining help?)
    (29, 0.001, 0.08,   12, 'adaptive', 'cap=12 + height=0.08'),
    (30, 0.001, 0.13,   12, 'adaptive', 'cap=12 + height=0.13'),
    (31, 0.001, 0.08,   10, 'adaptive', 'cap=10 + height=0.08'),
    (32, 0.001, 0.13,   10, 'adaptive', 'cap=10 + height=0.13'),
]

# Degree-6 positions for enrichment
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

# Hz-weighted Voronoi bins
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

HZ_FRACS = []
for i in range(N_POS):
    u_left, u_right = _VORONOI_EDGES[i]
    if i == 0:
        hz_frac = (PHI ** 1.0 - PHI ** u_left + PHI ** u_right - PHI ** 0.0) / (PHI - 1)
    else:
        hz_frac = (PHI ** u_right - PHI ** u_left) / (PHI - 1)
    HZ_FRACS.append(hz_frac)

BOUNDARY_HW = POS_VALS[1] / 2
BOUNDARY_LO_HZ_FRAC = (PHI ** BOUNDARY_HW - PHI ** 0.0) / (PHI - 1)
BOUNDARY_HI_HZ_FRAC = (PHI ** 1.0 - PHI ** (1 - BOUNDARY_HW)) / (PHI - 1)

OCTAVE_BAND = {'n-1': 'theta', 'n+0': 'alpha', 'n+1': 'beta_low',
               'n+2': 'beta_high', 'n+3': 'gamma'}
BAND_ORDER = ['theta', 'alpha', 'beta_low', 'beta_high', 'gamma']
BAND_HZ = {'theta': (4.70, 7.60), 'alpha': (7.60, 12.30),
            'beta_low': (12.30, 19.90), 'beta_high': (19.90, 32.19),
            'gamma': (32.19, 52.09)}

MERGE_THETA_ALPHA = True
FREQ_CEIL = 55.0
PAD_OCTAVES = 0.5


# =========================================================================
# EXTRACTION FUNCTIONS
# =========================================================================

def compute_adaptive_nperseg(f0, n, fs):
    lo = f0 * PHI ** n
    hi = f0 * PHI ** (n + 1)
    width = hi - lo
    min_sep_hz = MIN_SEP * width
    needed_res = min_sep_hz / 2
    nperseg = int(np.ceil(fs / needed_res))
    if nperseg % 2 != 0:
        nperseg += 1
    return nperseg


def build_bands(fs, nperseg_mode='adaptive'):
    bands = []
    for n in range(-4, 10):
        target_lo = F0 * PHI ** n
        target_hi = F0 * PHI ** (n + 1)
        if target_lo >= FREQ_CEIL or target_hi <= FILTER_LO:
            continue
        target_lo = max(target_lo, FILTER_LO)
        target_hi = min(target_hi, FREQ_CEIL)
        if target_hi - target_lo < 0.5:
            continue
        fit_lo = max(F0 * PHI ** (n - PAD_OCTAVES), FILTER_LO)
        fit_hi = min(F0 * PHI ** (n + 1 + PAD_OCTAVES), FREQ_CEIL)

        nperseg = compute_adaptive_nperseg(F0, n, fs)
        if nperseg_mode == '2500':
            nperseg = max(nperseg, 2500)
        nperseg = min(nperseg, int(60 * fs))

        bands.append({
            'name': f'n{n:+d}', 'n': n,
            'target_lo': target_lo, 'target_hi': target_hi,
            'fit_lo': fit_lo, 'fit_hi': fit_hi,
            'nperseg': nperseg, 'freq_res': fs / nperseg,
        })

    # Merge narrow bands
    merged = []
    i = 0
    while i < len(bands):
        b = bands[i].copy()
        if b['target_hi'] - b['target_lo'] < 1.5:
            while i + 1 < len(bands) and b['target_hi'] - b['target_lo'] < 1.5:
                i += 1
                b['target_hi'] = bands[i]['target_hi']
                b['fit_hi'] = bands[i]['fit_hi']
                b['nperseg'] = max(b['nperseg'], bands[i]['nperseg'])
                b['freq_res'] = min(b['freq_res'], bands[i]['freq_res'])
            b['name'] = f'{b["name"]}_merged'
        merged.append(b)
        i += 1

    # Merge theta+alpha
    if MERGE_THETA_ALPHA:
        theta = [b for b in merged if b['name'] == 'n-1']
        alpha = [b for b in merged if b['name'] == 'n+0']
        others = [b for b in merged if b['name'] not in ('n-1', 'n+0')]
        if theta and alpha:
            t, a = theta[0], alpha[0]
            m = {
                'name': 'n-1+n+0', 'n': -1,
                'target_lo': t['target_lo'], 'target_hi': a['target_hi'],
                'fit_lo': t['fit_lo'], 'fit_hi': a['fit_hi'],
                'nperseg': t['nperseg'], 'freq_res': t['freq_res'],
                '_split_at': F0, '_theta_name': 'n-1', '_alpha_name': 'n+0',
            }
            merged = [m] + others

    return merged


def extract_subject(args):
    """Extract one subject. args = (sub_id, config_id, peak_threshold, min_peak_height, max_n_peaks, nperseg_mode, out_dir)"""
    sub_id, config_id, peak_threshold, min_peak_height, max_n_peaks, nperseg_mode, out_dir = args

    out_path = os.path.join(out_dir, f'{sub_id}_peaks.csv')
    if os.path.exists(out_path):
        return sub_id, 'skipped', 0

    # Load EEGMMIDB subject
    raws = []
    for run in range(1, 15):
        edf = os.path.join(EEGMMIDB_DIR, f'{sub_id}', f'{sub_id}R{run:02d}.edf')
        if os.path.exists(edf):
            try:
                raw = mne.io.read_raw_edf(edf, preload=True, verbose=False)
                raws.append(raw)
            except Exception:
                continue
    if not raws:
        return sub_id, 'no_data', 0

    for i, raw in enumerate(raws):
        if abs(raw.info['sfreq'] - 160) > 0.1:
            raws[i] = raw.resample(160, verbose=False)
    raw = mne.concatenate_raws(raws) if len(raws) > 1 else raws[0]
    raw.filter(FILTER_LO, 59, verbose=False)
    fs = raw.info['sfreq']

    bands = build_bands(fs, nperseg_mode)
    all_peaks = []

    for band in bands:
        target_lo, target_hi = band['target_lo'], band['target_hi']
        fit_lo, fit_hi = band['fit_lo'], band['fit_hi']
        fit_width = fit_hi - fit_lo
        nperseg = band['nperseg']
        freq_res = band['freq_res']
        is_merged = '_split_at' in band

        max_peak_width = min(fit_width * 0.6, 12.0)
        peak_width_limits = [2 * freq_res, max_peak_width]

        fooof_params = {
            'peak_threshold': peak_threshold,
            'min_peak_height': min_peak_height,
            'max_n_peaks': max_n_peaks,
            'aperiodic_mode': 'fixed',
            'peak_width_limits': peak_width_limits,
        }

        for ch in raw.ch_names:
            try:
                data = raw.get_data(picks=[ch])[0]
            except Exception:
                continue
            if len(data) < nperseg:
                continue

            freqs, psd = welch(data, fs, nperseg=nperseg, noverlap=nperseg//2)
            sm = SpectralModel(**fooof_params)
            try:
                sm.fit(freqs, psd, [fit_lo, fit_hi])
            except Exception:
                continue

            r2 = _get_r_squared(sm)
            if np.isnan(r2) or r2 < R2_MIN:
                continue

            for row in _get_peak_params(sm):
                if target_lo <= row[0] < target_hi:
                    if is_merged:
                        octave_name = band['_theta_name'] if row[0] < band['_split_at'] else band['_alpha_name']
                    else:
                        octave_name = band['name']

                    all_peaks.append({
                        'freq': row[0], 'power': row[1], 'bandwidth': row[2],
                        'phi_octave': octave_name, 'r_squared': round(r2, 4),
                    })

    if all_peaks:
        df = pd.DataFrame(all_peaks)
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(out_path, index=False)
        return sub_id, 'ok', len(df)
    else:
        return sub_id, 'no_peaks', 0


def lattice_coord(freqs):
    return (np.log(freqs / F0) / np.log(PHI)) % 1.0


def compute_enrichment(peaks_df, min_power_pct=0):
    """Compute Hz-corrected enrichment with optional power filter."""
    if min_power_pct > 0 and 'power' in peaks_df.columns:
        filtered = []
        for octave in peaks_df['phi_octave'].unique():
            bp = peaks_df[peaks_df.phi_octave == octave]
            thresh = bp['power'].quantile(min_power_pct / 100)
            filtered.append(bp[bp['power'] >= thresh])
        peaks_df = pd.concat(filtered, ignore_index=True)

    rows = []
    for octave, band_name in OCTAVE_BAND.items():
        band_peaks = peaks_df[peaks_df.phi_octave == octave]
        freqs = band_peaks['freq'].values
        n = len(freqs)
        if n < 10:
            for p_name in POS_NAMES + ['boundary_hi']:
                rows.append({'band': band_name, 'position': p_name, 'enrichment_pct': np.nan})
            continue

        u = lattice_coord(freqs)
        dists = np.abs(u[:, None] - POS_VALS[None, :])
        dists = np.minimum(dists, 1 - dists)
        assignments = np.argmin(dists, axis=1)

        for i in range(N_POS):
            count = int((assignments == i).sum())
            expected = HZ_FRACS[i] * n
            e = round((count / expected - 1) * 100) if expected > 0 else 0
            rows.append({'band': band_name, 'position': POS_NAMES[i], 'enrichment_pct': e})

        lower = int((u < BOUNDARY_HW).sum())
        upper = int((u >= (1 - BOUNDARY_HW)).sum())
        # Replace boundary with split
        rows = [r for r in rows if not (r['band'] == band_name and r['position'] == 'boundary')]
        rows.append({'band': band_name, 'position': 'boundary',
                     'enrichment_pct': round((lower / (BOUNDARY_LO_HZ_FRAC * n) - 1) * 100) if BOUNDARY_LO_HZ_FRAC * n > 0 else 0})
        rows.append({'band': band_name, 'position': 'boundary_hi',
                     'enrichment_pct': round((upper / (BOUNDARY_HI_HZ_FRAC * n) - 1) * 100) if BOUNDARY_HI_HZ_FRAC * n > 0 else 0})

    return pd.DataFrame(rows)


# =========================================================================
# MAIN
# =========================================================================

def run_config(config_id):
    """Run extraction + analysis for one config."""
    cfg = [c for c in CONFIGS if c[0] == config_id]
    if not cfg:
        print(f"Config {config_id} not found")
        return
    cid, thresh, height, maxp, npmode, label = cfg[0]

    dir_name = f'config{cid:02d}_t{thresh}_h{height}_m{maxp}_{npmode}'
    out_dir = os.path.join(OUTPUT_BASE, dir_name)
    os.makedirs(out_dir, exist_ok=True)

    # Find subjects
    subs = sorted(set(
        d for d in os.listdir(EEGMMIDB_DIR)
        if d.startswith('S') and os.path.isdir(os.path.join(EEGMMIDB_DIR, d))))

    print(f"\nConfig {cid}: {label}")
    print(f"  threshold={thresh}, min_height={height}, max_peaks={maxp}, nperseg={npmode}")
    print(f"  Subjects: {len(subs)}, Output: {out_dir}")

    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for i, sub in enumerate(subs):
            result = extract_subject((sub, cid, thresh, height, maxp, npmode, out_dir))
            if (i + 1) % 20 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed * 60
                print(f"    [{i+1}/{len(subs)}] {rate:.1f} subjects/min")

    elapsed = time.time() - t0
    n_files = len(glob(os.path.join(out_dir, '*_peaks.csv')))
    print(f"  Done: {n_files} subjects in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Compute enrichment at multiple power thresholds
    files = sorted(glob(os.path.join(out_dir, '*_peaks.csv')))
    if not files:
        return

    all_peaks = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    n_total = len(all_peaks)
    print(f"  Total peaks: {n_total:,}")

    results = []
    for pwr in [0, 25, 50, 75]:
        enr = compute_enrichment(all_peaks, min_power_pct=pwr)
        for _, row in enr.iterrows():
            results.append({
                'config': cid, 'label': label,
                'threshold': thresh, 'min_height': height,
                'max_peaks': maxp, 'nperseg_mode': npmode,
                'power_filter': pwr,
                'band': row['band'], 'position': row['position'],
                'enrichment_pct': row['enrichment_pct'],
            })

    rdf = pd.DataFrame(results)
    rdf.to_csv(os.path.join(OUTPUT_BASE, f'config{cid:02d}_enrichment.csv'), index=False)

    # Print summary for alpha and beta-low
    for pwr in [0, 50]:
        print(f"\n  Power filter={pwr}%:")
        for band in ['alpha', 'beta_low']:
            bdf = rdf[(rdf['band'] == band) & (rdf['power_filter'] == pwr)]
            key_pos = {'alpha': ['boundary', 'attractor', 'noble_1', 'boundary_hi'],
                       'beta_low': ['boundary', 'noble_4', 'attractor', 'noble_1', 'inv_noble_4', 'inv_noble_6', 'boundary_hi']}
            vals = []
            for pos in key_pos[band]:
                r = bdf[bdf['position'] == pos]
                v = int(r.iloc[0]['enrichment_pct']) if not r.empty and not np.isnan(r.iloc[0]['enrichment_pct']) else 0
                vals.append(f'{pos}={v:+d}%')
            print(f"    {band}: {', '.join(vals)}")


def run_all(parallel=1):
    """Run all configs."""
    t0 = time.time()
    if parallel > 1:
        # Parallel: each config in a separate process
        with Pool(min(parallel, len(CONFIGS))) as pool:
            pool.map(run_config, [c[0] for c in CONFIGS])
    else:
        for cfg in CONFIGS:
            run_config(cfg[0])

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  ALL {len(CONFIGS)} CONFIGS COMPLETE in {elapsed/60:.1f} min")
    print(f"{'='*70}")

    # Combine all config enrichments
    all_csvs = sorted(glob(os.path.join(OUTPUT_BASE, 'config*_enrichment.csv')))
    if all_csvs:
        combined = pd.concat([pd.read_csv(f) for f in all_csvs], ignore_index=True)
        combined.to_csv(os.path.join(OUTPUT_BASE, 'all_configs_enrichment.csv'), index=False)
        print(f"  Combined CSV: {OUTPUT_BASE}/all_configs_enrichment.csv ({len(combined)} rows)")

        # Stability analysis
        print(f"\n  STABILITY ANALYSIS")
        for band in BAND_ORDER:
            for pwr in [0, 50]:
                profiles = []
                labels = []
                for cfg in CONFIGS:
                    cid = cfg[0]
                    bdf = combined[(combined['config'] == cid) &
                                   (combined['band'] == band) &
                                   (combined['power_filter'] == pwr)]
                    if len(bdf) < 5:
                        continue
                    # Sort by position to ensure alignment
                    bdf = bdf.sort_values('position')
                    vals = bdf['enrichment_pct'].values
                    if not np.any(np.isnan(vals)):
                        profiles.append(vals)
                        labels.append(f'c{cid}')

                if len(profiles) >= 2:
                    profiles = np.array(profiles)
                    # Mean pairwise correlation
                    from itertools import combinations
                    cors = []
                    for i, j in combinations(range(len(profiles)), 2):
                        r = np.corrcoef(profiles[i], profiles[j])[0, 1]
                        cors.append(r)
                    mean_r = np.mean(cors)
                    min_r = np.min(cors)

                    # Position-level CV
                    cv = np.std(profiles, axis=0) / (np.abs(np.mean(profiles, axis=0)) + 1)
                    mean_cv = np.mean(cv)

                    print(f"    {band:<12s} pwr={pwr}%: mean_r={mean_r:.3f} min_r={min_r:.3f} mean_CV={mean_cv:.3f} ({len(profiles)} configs)")


def main():
    parser = argparse.ArgumentParser(description='v4 FOOOF parameter sweep')
    parser.add_argument('--config', type=int, help='Run single config by ID (1-16)')
    parser.add_argument('--all', action='store_true', help='Run all configs')
    parser.add_argument('--parallel', type=int, default=1, help='Number of parallel workers')
    args = parser.parse_args()

    if args.config:
        run_config(args.config)
    elif args.all:
        run_all(parallel=args.parallel)
    else:
        print("Specify --config N or --all")
        print("\nConfigs:")
        for cfg in CONFIGS:
            print(f"  {cfg[0]:>2d}: {cfg[5]:<35s} thresh={cfg[1]}, height={cfg[2]}, maxp={cfg[3]}, nperseg={cfg[4]}")


if __name__ == '__main__':
    main()
