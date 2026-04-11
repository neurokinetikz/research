#!/usr/bin/env python3
"""
Compare Voronoi Enrichment: f0=7.83 extraction vs f0=7.60 extraction
=====================================================================

Runs degree-6 Voronoi enrichment (f0=7.60 coordinates) on EEGMMIDB
peaks extracted at both f0=7.83 and f0=7.60, producing a side-by-side
comparison table to assess the impact of the extraction f0 mismatch.

Usage:
    python scripts/compare_f0_enrichment.py
"""

import os
import sys
import glob

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
from phi_frequency_model import PHI

F0_ENRICHMENT = 7.60  # Enrichment coordinate system
PHI_INV = 1.0 / PHI

# Degree-6 positions
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

VORONOI_BINS = []
for i in range(N_POS):
    d_prev = (POS_VALS[i] - POS_VALS[(i - 1) % N_POS]) % 1.0
    d_next = (POS_VALS[(i + 1) % N_POS] - POS_VALS[i]) % 1.0
    VORONOI_BINS.append(d_prev / 2 + d_next / 2)

BOUNDARY_HW = POS_VALS[1] / 2

OCTAVE_BAND = {
    'n-1': 'theta', 'n+0': 'alpha', 'n+1': 'beta_low',
    'n+2': 'beta_high', 'n+3': 'gamma',
}
BAND_ORDER = ['theta', 'alpha', 'beta_low', 'beta_high', 'gamma']
BAND_HZ = {
    'theta':     (4.70, 7.60),
    'alpha':     (7.60, 12.30),
    'beta_low':  (12.30, 19.90),
    'beta_high': (19.90, 32.19),
    'gamma':     (32.19, 52.09),
}


def lattice_coord(freqs, f0=F0_ENRICHMENT):
    return (np.log(freqs / f0) / np.log(PHI)) % 1.0


def assign_voronoi(u_vals):
    u = np.asarray(u_vals, dtype=float) % 1.0
    dists = np.abs(u[:, None] - POS_VALS[None, :])
    dists = np.minimum(dists, 1 - dists)
    return np.argmin(dists, axis=1)


def load_peaks(directory):
    files = sorted(glob.glob(os.path.join(directory, '*_peaks.csv')))
    if not files:
        raise FileNotFoundError(f"No peak files found in {directory}")
    dfs = []
    for f in files:
        df = pd.read_csv(f, usecols=['freq', 'phi_octave'])
        dfs.append(df)
    peaks = pd.concat(dfs, ignore_index=True)
    return peaks, len(files)


def compute_enrichment(peaks_df, f0=F0_ENRICHMENT):
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

        enrichments = []
        counts = []
        for i in range(N_POS):
            count = int((assignments == i).sum())
            expected = VORONOI_BINS[i] * n
            e = round((count / expected - 1) * 100) if expected > 0 else 0
            enrichments.append(e)
            counts.append(count)

        lower_count = int((u < BOUNDARY_HW).sum())
        upper_count = int((u >= (1 - BOUNDARY_HW)).sum())
        exp_half = BOUNDARY_HW * n
        enr_lower = round((lower_count / exp_half - 1) * 100) if exp_half > 0 else 0
        enr_upper = round((upper_count / exp_half - 1) * 100) if exp_half > 0 else 0

        rows.append({
            'band': band_name, 'position': 'boundary',
            'offset': 0.000, 'hz': f_lo,
            'n_peaks': lower_count, 'enrichment_pct': enr_lower,
        })
        for i in range(1, N_POS):
            rows.append({
                'band': band_name, 'position': POS_NAMES[i],
                'offset': POS_VALS[i], 'hz': f_lo * PHI ** POS_VALS[i],
                'n_peaks': counts[i], 'enrichment_pct': enrichments[i],
            })
        rows.append({
            'band': band_name, 'position': 'boundary_hi',
            'offset': 1.000, 'hz': f_hi,
            'n_peaks': upper_count, 'enrichment_pct': enr_upper,
        })

    return pd.DataFrame(rows)


def main():
    base = os.path.join(os.path.dirname(__file__), '..')
    dir_783 = os.path.join(base, 'exports_adaptive', 'eegmmidb')
    dir_760 = os.path.join(base, 'exports_adaptive_f0_760', 'eegmmidb')

    if not os.path.isdir(dir_760):
        print(f"ERROR: {dir_760} does not exist. Run run_f0_760_extraction.py first.")
        sys.exit(1)

    print("Loading peaks...")
    peaks_783, n_783 = load_peaks(dir_783)
    peaks_760, n_760 = load_peaks(dir_760)
    print(f"  f0=7.83: {len(peaks_783):,} peaks from {n_783} subjects")
    print(f"  f0=7.60: {len(peaks_760):,} peaks from {n_760} subjects")

    # Show band boundary comparison
    print(f"\n{'='*70}")
    print(f"  BAND BOUNDARY COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Band':<12s}  {'f0=7.83 range':>20s}  {'f0=7.60 range':>20s}  {'Gap':>8s}")
    print(f"  {'-'*65}")
    for band in BAND_ORDER:
        f_lo_760, f_hi_760 = BAND_HZ[band]
        n = {'theta': -1, 'alpha': 0, 'beta_low': 1, 'beta_high': 2, 'gamma': 3}[band]
        f_lo_783 = 7.83 * PHI ** n
        f_hi_783 = 7.83 * PHI ** (n + 1)
        gap = abs(f_hi_783 - f_hi_760)
        print(f"  {band:<12s}  [{f_lo_783:6.2f}, {f_hi_783:6.2f}]  [{f_lo_760:6.2f}, {f_hi_760:6.2f}]  {gap:6.2f} Hz")

    # Peak count per band
    print(f"\n{'='*70}")
    print(f"  PEAK COUNTS PER BAND")
    print(f"{'='*70}")
    print(f"  {'Band':<12s}  {'f0=7.83':>10s}  {'f0=7.60':>10s}  {'Δ':>8s}  {'Δ%':>8s}")
    print(f"  {'-'*52}")
    for band in BAND_ORDER:
        octave = {v: k for k, v in OCTAVE_BAND.items()}[band]
        n783 = (peaks_783.phi_octave == octave).sum()
        n760 = (peaks_760.phi_octave == octave).sum()
        delta = n760 - n783
        pct = (delta / n783 * 100) if n783 > 0 else 0
        print(f"  {band:<12s}  {n783:>10,}  {n760:>10,}  {delta:>+8,}  {pct:>+7.1f}%")

    print(f"\nComputing enrichment...")
    enr_783 = compute_enrichment(peaks_783)
    enr_760 = compute_enrichment(peaks_760)

    # Side-by-side comparison
    all_positions = POS_NAMES + ['boundary_hi']
    pos_offsets = dict(POS_LIST)
    pos_offsets['boundary_hi'] = 1.0

    print(f"\n{'='*70}")
    print(f"  ENRICHMENT COMPARISON: f0=7.83 vs f0=7.60 extraction")
    print(f"  (both computed in f0=7.60 coordinate system)")
    print(f"{'='*70}")

    # Collect results for CSV output
    csv_rows = []

    for band in BAND_ORDER:
        f_lo, f_hi = BAND_HZ[band]
        band_n = {'theta': -1, 'alpha': 0, 'beta_low': 1, 'beta_high': 2, 'gamma': 3}[band]
        sign = '+' if band_n >= 0 else ''
        print(f"\n  {band.upper()} (n{sign}{band_n}, {f_lo:.2f}–{f_hi:.2f} Hz)")
        print(f"  {'Position':<16s} {'u':>6s} {'Hz':>7s} {'f0=7.83':>8s} {'f0=7.60':>8s} {'Δ':>6s}  {'Note'}")
        print(f"  {'-'*70}")

        for pos in all_positions:
            offset = pos_offsets[pos]
            hz = f_lo * PHI ** offset if offset < 1 else f_hi

            row_783 = enr_783[(enr_783['band'] == band) & (enr_783['position'] == pos)]
            row_760 = enr_760[(enr_760['band'] == band) & (enr_760['position'] == pos)]

            v783 = int(row_783.iloc[0]['enrichment_pct']) if (not row_783.empty and not np.isnan(row_783.iloc[0]['enrichment_pct'])) else None
            v760 = int(row_760.iloc[0]['enrichment_pct']) if (not row_760.empty and not np.isnan(row_760.iloc[0]['enrichment_pct'])) else None

            if v783 is None and v760 is None:
                continue

            s783 = f'{v783:+d}%' if v783 is not None else '—'
            s760 = f'{v760:+d}%' if v760 is not None else '—'

            delta = (v760 - v783) if (v783 is not None and v760 is not None) else None
            sdelta = f'{delta:+d}' if delta is not None else '—'

            # Flag sign changes and large deltas
            note = ''
            if v783 is not None and v760 is not None:
                if (v783 > 5 and v760 < -5) or (v783 < -5 and v760 > 5):
                    note = '*** SIGN FLIP ***'
                elif abs(delta) > 30:
                    note = '** LARGE Δ **'
                elif abs(delta) > 15:
                    note = '* notable *'

            pos_display = 'boundary (hi)' if pos == 'boundary_hi' else pos
            print(f"  {pos_display:<16s} {offset:>6.3f} {hz:>7.2f} {s783:>8s} {s760:>8s} {sdelta:>6s}  {note}")

            csv_rows.append({
                'band': band, 'position': pos, 'offset': offset, 'hz': round(hz, 2),
                'enr_f0_783': v783, 'enr_f0_760': v760, 'delta': delta,
            })

    # Summary
    df = pd.DataFrame(csv_rows)
    df_valid = df.dropna(subset=['delta'])

    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")

    sign_flips = df_valid[((df_valid['enr_f0_783'] > 5) & (df_valid['enr_f0_760'] < -5)) |
                           ((df_valid['enr_f0_783'] < -5) & (df_valid['enr_f0_760'] > 5))]
    print(f"  Sign flips (>±5%): {len(sign_flips)}")
    if len(sign_flips) > 0:
        for _, r in sign_flips.iterrows():
            print(f"    {r['band']:<12s} {r['position']:<16s} {r['enr_f0_783']:+d}% → {r['enr_f0_760']:+d}%")

    large = df_valid[df_valid['delta'].abs() > 30]
    print(f"\n  Large shifts (|Δ|>30): {len(large)}")
    if len(large) > 0:
        for _, r in large.iterrows():
            print(f"    {r['band']:<12s} {r['position']:<16s} {r['enr_f0_783']:+d}% → {r['enr_f0_760']:+d}% (Δ={r['delta']:+d})")

    # Per-band correlation
    print(f"\n  Per-band profile correlation (f0=7.83 vs f0=7.60):")
    for band in BAND_ORDER:
        bdf = df_valid[df_valid['band'] == band]
        if len(bdf) >= 3:
            r = np.corrcoef(bdf['enr_f0_783'], bdf['enr_f0_760'])[0, 1]
            mae = bdf['delta'].abs().mean()
            print(f"    {band:<12s}  r={r:.3f}  MAE={mae:.1f}pp")

    # Boundary-specific analysis
    print(f"\n  Boundary positions only (the critical test):")
    print(f"  {'Band':<12s} {'Pos':<16s} {'f0=7.83':>8s} {'f0=7.60':>8s} {'Δ':>6s}")
    print(f"  {'-'*50}")
    for _, r in df_valid[df_valid['position'].isin(['boundary', 'boundary_hi', 'noble_6', 'inv_noble_6'])].iterrows():
        print(f"  {r['band']:<12s} {r['position']:<16s} {r['enr_f0_783']:>+7d}% {r['enr_f0_760']:>+7d}% {r['delta']:>+5d}")

    # Save CSV
    out_csv = os.path.join(base, 'outputs', 'f0_extraction_comparison.csv')
    df.to_csv(out_csv, index=False)
    print(f"\n  Results saved to {out_csv}")


if __name__ == '__main__':
    main()
