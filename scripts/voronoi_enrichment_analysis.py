#!/usr/bin/env python3
"""
Voronoi Enrichment Analysis
============================

Per-band, degree-6 Voronoi enrichment analysis across datasets.
Loads adaptive-resolution extraction results and computes enrichment
at 12 lattice positions (+ split boundary) within each phi-octave band.

Peaks are selected by phi_octave column from extraction (f0=7.83 bands),
then lattice coordinates are computed with f0=7.60 for enrichment.

Usage:
    # Single dataset
    python scripts/voronoi_enrichment_analysis.py --dataset eegmmidb

    # All datasets
    python scripts/voronoi_enrichment_analysis.py --all

    # Cross-dataset summary only
    python scripts/voronoi_enrichment_analysis.py --all --summary

    # Save CSV output
    python scripts/voronoi_enrichment_analysis.py --all --csv outputs/enrichment.csv
"""

import os
import sys
import argparse
import glob

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
from phi_frequency_model import PHI, F0

PHI_INV = 1.0 / PHI

# --- Degree-6 positions (12 unique on the circle) ---
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

# Hz-weighted Voronoi bin fractions (matches run_all_f0_760_analyses.py)
# Under a Hz-uniform null, expected_count = HZ_FRAC[i] * total_peaks
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

VORONOI_BINS = []
for i in range(N_POS):
    u_left, u_right = _VORONOI_EDGES[i]
    if i == 0:
        hz_frac = (PHI ** 1.0 - PHI ** u_left + PHI ** u_right - PHI ** 0.0) / (PHI - 1)
    else:
        hz_frac = (PHI ** u_right - PHI ** u_left) / (PHI - 1)
    VORONOI_BINS.append(hz_frac)

# Boundary half-width for split: half distance to noble_6
BOUNDARY_HW = POS_VALS[1] / 2  # ~0.02786
# Hz-weighted half-boundary fractions (matches run_all_f0_760_analyses.py)
BOUNDARY_LO_HZ_FRAC = (PHI ** BOUNDARY_HW - PHI ** 0.0) / (PHI - 1)
BOUNDARY_HI_HZ_FRAC = (PHI ** 1.0 - PHI ** (1 - BOUNDARY_HW)) / (PHI - 1)

# Phi-octave to band name mapping
OCTAVE_BAND = {
    'n-1': 'theta',
    'n+0': 'alpha',
    'n+1': 'beta_low',
    'n+2': 'beta_high',
    'n+3': 'gamma',
}

BAND_ORDER = ['theta', 'alpha', 'beta_low', 'beta_high', 'gamma']

# Band Hz ranges (f0=7.60) for display
BAND_HZ = {
    'theta':     (4.70, 7.60),
    'alpha':     (7.60, 12.30),
    'beta_low':  (12.30, 19.90),
    'beta_high': (19.90, 32.19),
    'gamma':     (32.19, 52.09),
}

# Known dataset directories
# Per-release HBN entries (used for cross-release analyses)
HBN_RELEASES = {
    f'hbn_R{i}': f'exports_adaptive_v4/hbn_R{i}' for i in range(1, 12)
}

# 6-dataset convention: HBN merged, TDBRAIN included
DATASETS_MERGED = {
    'eegmmidb': 'exports_adaptive_v4/eegmmidb',
    'lemon':    'exports_adaptive_v4/lemon',
    'dortmund': 'exports_adaptive_v4/dortmund',
    'chbmp':    'exports_adaptive_v4/chbmp',
    'hbn':      [f'exports_adaptive_v4/hbn_R{i}' for i in range(1, 12)],
    'tdbrain':  'exports_adaptive_v4/tdbrain',
}

# Per-release (legacy) — used with --per-release flag
DATASETS = {
    'eegmmidb': 'exports_adaptive_v4/eegmmidb',
    'lemon':    'exports_adaptive_v4/lemon',
    'dortmund': 'exports_adaptive_v4/dortmund',
    'chbmp':    'exports_adaptive_v4/chbmp',
    **HBN_RELEASES,
    'tdbrain':  'exports_adaptive_v4/tdbrain',
}


# =========================================================================
# CORE FUNCTIONS
# =========================================================================

def lattice_coord(freqs, f0=F0):
    """Map frequencies to lattice coordinate u = [log_phi(f/f0)] mod 1."""
    return (np.log(freqs / f0) / np.log(PHI)) % 1.0


def assign_voronoi(u_vals):
    """Assign each u to nearest position (circular distance)."""
    u = np.asarray(u_vals, dtype=float) % 1.0
    dists = np.abs(u[:, None] - POS_VALS[None, :])
    dists = np.minimum(dists, 1 - dists)
    return np.argmin(dists, axis=1)


def load_peaks(directory):
    """Load all peak CSVs from a dataset directory (or list of directories) with phi_octave column."""
    dirs = directory if isinstance(directory, list) else [directory]
    all_files = []
    for d in dirs:
        files = sorted(glob.glob(os.path.join(d, '*_peaks.csv')))
        all_files.extend(files)
    if not all_files:
        raise FileNotFoundError(f"No peak files found in {directory}")
    dfs = []
    for f in all_files:
        df = pd.read_csv(f, usecols=['freq', 'phi_octave'])
        dfs.append(df)
    peaks = pd.concat(dfs, ignore_index=True)
    return peaks, len(all_files)


def compute_enrichment(peaks_df, f0=F0):
    """
    Compute per-band Voronoi enrichment with split boundary.

    Peaks are selected by phi_octave column, lattice coordinates
    computed with f0.

    Returns DataFrame with columns:
        band, position, offset, hz, n_peaks, enrichment_pct
    """
    rows = []
    for octave, band_name in OCTAVE_BAND.items():
        band_peaks = peaks_df[peaks_df.phi_octave == octave]
        freqs = band_peaks['freq'].values
        n = len(freqs)
        f_lo, f_hi = BAND_HZ[band_name]

        if n < 10:
            for p_name in POS_NAMES + ['boundary_hi']:
                rows.append({
                    'band': band_name, 'position': p_name,
                    'offset': dict(POS_LIST).get(p_name, 1.0),
                    'hz': f_lo if p_name == 'boundary' else (f_hi if p_name == 'boundary_hi' else f_lo * PHI ** dict(POS_LIST).get(p_name, 0)),
                    'n_peaks': n, 'enrichment_pct': np.nan,
                })
            continue

        u = lattice_coord(freqs, f0)
        assignments = assign_voronoi(u)

        # Compute enrichment for all 12 positions
        enrichments = []
        counts = []
        for i in range(N_POS):
            count = int((assignments == i).sum())
            expected = VORONOI_BINS[i] * n
            e = round((count / expected - 1) * 100) if expected > 0 else 0
            enrichments.append(e)
            counts.append(count)

        # Split boundary: peaks near u=0 (lower edge) vs u=1 (upper edge)
        # Use Hz-weighted expected counts (matches run_all_f0_760_analyses.py)
        lower_count = int((u < BOUNDARY_HW).sum())
        upper_count = int((u >= (1 - BOUNDARY_HW)).sum())
        exp_lower = BOUNDARY_LO_HZ_FRAC * n
        exp_upper = BOUNDARY_HI_HZ_FRAC * n
        enr_lower = round((lower_count / exp_lower - 1) * 100) if exp_lower > 0 else 0
        enr_upper = round((upper_count / exp_upper - 1) * 100) if exp_upper > 0 else 0

        # Boundary (lower)
        rows.append({
            'band': band_name, 'position': 'boundary',
            'offset': 0.000, 'hz': f_lo,
            'n_peaks': lower_count, 'enrichment_pct': enr_lower,
        })
        # Interior positions (skip boundary at index 0)
        for i in range(1, N_POS):
            rows.append({
                'band': band_name, 'position': POS_NAMES[i],
                'offset': POS_VALS[i], 'hz': f_lo * PHI ** POS_VALS[i],
                'n_peaks': counts[i], 'enrichment_pct': enrichments[i],
            })
        # Boundary (upper)
        rows.append({
            'band': band_name, 'position': 'boundary_hi',
            'offset': 1.000, 'hz': f_hi,
            'n_peaks': upper_count, 'enrichment_pct': enr_upper,
        })

    return pd.DataFrame(rows)


def print_enrichment(df, dataset_name):
    """Print enrichment table for a single dataset."""
    print(f"\n{'='*70}")
    print(f"  {dataset_name} — Degree-6 Voronoi Enrichment (f₀={F0})")
    print(f"{'='*70}")

    for band in BAND_ORDER:
        bdf = df[df['band'] == band]
        if bdf.empty or bdf['enrichment_pct'].isna().all():
            print(f"\n  {band.upper()}: NO DATA")
            continue
        total = bdf['n_peaks'].sum()
        print(f"\n  {band.upper()} (N={total:,})")
        print(f"  {'Position':<16} {'u':>6} {'Hz':>8} {'Enrich':>8}")
        print(f"  {'-'*42}")
        for _, row in bdf.iterrows():
            pos_display = 'boundary (hi)' if row['position'] == 'boundary_hi' else row['position']
            enr = row['enrichment_pct']
            if np.isnan(enr):
                print(f"  {pos_display:<16} {row['offset']:>6.3f} {row['hz']:>8.2f}     N/A")
            else:
                print(f"  {pos_display:<16} {row['offset']:>6.3f} {row['hz']:>8.2f} {enr:>+7.0f}%")


def cross_dataset_summary(all_results):
    """Print cross-dataset comparison table matching impact report format."""
    ds_names = list(all_results.keys())
    short_names = {
        'eegmmidb': 'EEGM', 'lemon': 'LEM', 'dortmund': 'Dort',
        'chbmp': 'CHBMP', 'hbn': 'HBN', 'tdbrain': 'TDB',
        'hbn_R1': 'R1', 'hbn_R2': 'R2', 'hbn_R3': 'R3',
        'hbn_R4': 'R4', 'hbn_R5': 'R5', 'hbn_R6': 'R6',
        'hbn_R7': 'R7', 'hbn_R8': 'R8', 'hbn_R9': 'R9',
        'hbn_R10': 'R10', 'hbn_R11': 'R11',
    }

    all_positions = POS_NAMES + ['boundary_hi']
    pos_display = {p: p for p in all_positions}
    pos_display['boundary_hi'] = 'boundary (hi)'
    pos_offsets = dict(POS_LIST)
    pos_offsets['boundary_hi'] = 1.0

    print(f"\n{'='*80}")
    print(f"  CROSS-DATASET SUMMARY ({len(ds_names)} datasets, f₀={F0})")
    print(f"{'='*80}")

    band_consistency = {}

    for band in BAND_ORDER:
        f_lo, f_hi = BAND_HZ[band]
        band_n = {'theta': -1, 'alpha': 0, 'beta_low': 1, 'beta_high': 2, 'gamma': 3}[band]
        sign = '+' if band_n >= 0 else ''
        print(f"\n### {band.upper()} (n{sign}{band_n}, {f_lo:.2f}–{f_hi:.2f} Hz)")

        ds_short = [short_names.get(d, d) for d in ds_names]
        hdr = '| Position | u | Hz | ' + ' | '.join(ds_short) + ' | Mean | SD | |'
        sep = '|---|---|---|' + '---|' * len(ds_names) + '---|---|---|'
        print(hdr)
        print(sep)

        n_check = 0; n_tilde = 0; n_cross = 0

        for pos in all_positions:
            offset = pos_offsets[pos]
            hz = f_lo * PHI ** offset if offset < 1 else f_hi

            vals = []
            for ds in ds_names:
                df = all_results[ds]
                row = df[(df['band'] == band) & (df['position'] == pos)]
                if row.empty or (isinstance(row.iloc[0]['enrichment_pct'], float) and np.isnan(row.iloc[0]['enrichment_pct'])):
                    vals.append(None)
                else:
                    vals.append(int(row.iloc[0]['enrichment_pct']))

            valid = [v for v in vals if v is not None]
            if not valid:
                continue

            mean = np.mean(valid)
            sd = np.std(valid, ddof=0)

            # Consistency marker
            signs = [1 if v > 5 else (-1 if v < -5 else 0) for v in valid]
            pos_count = sum(1 for s in signs if s > 0)
            neg_count = sum(1 for s in signs if s < 0)
            n_ds = len(valid)

            if pos_count == n_ds or neg_count == n_ds:
                mark = '✓'; n_check += 1
            elif pos_count >= n_ds - 1 or neg_count >= n_ds - 1:
                mark = '~'; n_tilde += 1
            elif pos_count > 0 and neg_count > 0:
                mark = '✗'; n_cross += 1
            else:
                mark = '~'; n_tilde += 1

            val_strs = [f'{v:+d}%' if v is not None else '—' for v in vals]
            print(f"| {pos_display[pos]} | {offset:.3f} | {hz:.2f} | {' | '.join(val_strs)} | **{mean:+.0f}%** | {sd:.0f} | {mark} |")

        consistent = n_check + n_tilde
        print(f"\n✓={n_check} | ~={n_tilde} | ✗={n_cross} | **Consistent: {consistent}/13**")
        band_consistency[band] = (n_check, n_tilde, n_cross)

    shape_desc = {
        'theta':     'Boundary clustering at edges',
        'alpha':     'Mountain: Noble1/attractor peak, boundaries depleted',
        'beta_low':  'U-shape: boundaries enriched, center depleted',
        'beta_high': 'Weak ascending: inv_noble_4 enriched',
        'gamma':     'Ascending ramp: inv_noble_3/4/5/6 enriched',
    }
    print(f"\n### Consistency Summary\n")
    print("| Band | ✓ | ~ | ✗ | Consistent | Shape |")
    print("|------|---|---|---|-----------|-------|")
    for band in BAND_ORDER:
        nc, nt, nx = band_consistency[band]
        total = nc + nt
        print(f"| **{band.replace('_', '-').title()}** | {nc} | {nt} | {nx} | **{total}/13** | {shape_desc[band]} |")


def hbn_cross_release(all_results):
    """Print HBN cross-release consistency analysis."""
    releases = ['hbn_R1', 'hbn_R2', 'hbn_R3', 'hbn_R4', 'hbn_R6']
    available = [r for r in releases if r in all_results]
    if len(available) < 2:
        return

    short = {'hbn_R1': 'R1', 'hbn_R2': 'R2', 'hbn_R3': 'R3', 'hbn_R4': 'R4', 'hbn_R6': 'R6'}
    all_positions = POS_NAMES + ['boundary_hi']

    print(f"\n### Cross-Release Consistency ({', '.join(short[r] for r in available)})\n")
    print("| Band | ✓ | ✗ | Notable |")
    print("|------|---|---|---------|")

    for band in BAND_ORDER:
        n_check = 0; n_cross = 0
        best_sd = 999; best_pos = ''; best_vals = ''

        for pos in all_positions:
            vals = []
            for r in available:
                df = all_results[r]
                row = df[(df['band'] == band) & (df['position'] == pos)]
                if not row.empty and not np.isnan(row.iloc[0]['enrichment_pct']):
                    vals.append(int(row.iloc[0]['enrichment_pct']))

            if len(vals) < 2:
                continue

            signs = [1 if v > 5 else (-1 if v < -5 else 0) for v in vals]
            pos_count = sum(1 for s in signs if s > 0)
            neg_count = sum(1 for s in signs if s < 0)

            if pos_count > 0 and neg_count > 0:
                n_cross += 1
            else:
                n_check += 1

            sd = np.std(vals, ddof=0)
            if sd < best_sd and abs(np.mean(vals)) > 10:
                best_sd = sd
                best_pos = pos.replace('boundary_hi', 'boundary(hi)')
                best_vals = ', '.join(f'{v:+d}%' for v in vals)

        notable = f"{best_pos}: {best_vals} (SD={best_sd:.1f})" if best_pos else ''
        print(f"| {band.replace('_','-').title()} | **{n_check}/13** | {n_cross} | {notable} |")


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description='Voronoi enrichment analysis')
    parser.add_argument('--dataset', '-d', type=str, help='Dataset name (e.g., eegmmidb, lemon, hbn)')
    parser.add_argument('--dir', type=str, help='Custom directory with peak CSVs')
    parser.add_argument('--all', action='store_true', help='Run all known datasets (6 merged)')
    parser.add_argument('--per-release', action='store_true', help='Use per-release HBN entries instead of merged')
    parser.add_argument('--csv', type=str, help='Save results to CSV')
    parser.add_argument('--summary', action='store_true', help='Print cross-dataset summary only')
    parser.add_argument('--f0', type=float, default=F0, help=f'Fundamental frequency (default: {F0})')
    args = parser.parse_args()

    f0 = args.f0

    if args.dir:
        peaks, n_sub = load_peaks(args.dir)
        name = os.path.basename(args.dir.rstrip('/'))
        print(f"Loaded {name}: {n_sub} subjects, {len(peaks):,} peaks")
        df = compute_enrichment(peaks, f0)
        print_enrichment(df, name)
        if args.csv:
            df.to_csv(args.csv, index=False)
            print(f"\nSaved to {args.csv}")
        return

    if args.dataset:
        source = DATASETS if args.per_release else DATASETS_MERGED
        if args.dataset not in source:
            source = DATASETS  # fall back to per-release for specific release names
        datasets_to_run = {args.dataset: source[args.dataset]}
    elif args.all or args.summary:
        source = DATASETS if args.per_release else DATASETS_MERGED
        def _exists(v):
            if isinstance(v, list):
                return any(os.path.exists(d) for d in v)
            return os.path.exists(v)
        datasets_to_run = {k: v for k, v in source.items() if _exists(v)}
    else:
        parser.print_help()
        return

    all_results = {}
    all_dfs = []

    for name, directory in datasets_to_run.items():
        if isinstance(directory, list):
            if not any(os.path.exists(d) for d in directory):
                print(f"SKIP {name}: no directories found")
                continue
        elif not os.path.exists(directory):
            print(f"SKIP {name}: {directory} not found")
            continue
        peaks, n_sub = load_peaks(directory)
        print(f"Loaded {name}: {n_sub} subjects, {len(peaks):,} peaks")
        df = compute_enrichment(peaks, f0)
        df['dataset'] = name
        all_results[name] = df
        all_dfs.append(df)

        if not args.summary:
            print_enrichment(df, name)

    if len(all_results) > 1:
        cross_dataset_summary(all_results)
        hbn_cross_release(all_results)

    if args.csv and all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(args.csv, index=False)
        print(f"\nSaved combined results to {args.csv}")


if __name__ == '__main__':
    main()
