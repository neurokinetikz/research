#!/usr/bin/env python3
"""
5-Year Longitudinal 2×2 Comparison
=====================================

Compares the full 2×2 (EC/EO × pre/post) enrichment pattern between
ses-1 (N≈608) and ses-2 (N=208, ~5 years later).

Tests:
  1. Profile correlation per condition × band (ses-1 vs ses-2)
  2. Eyes effect replication (does EC→EO pattern replicate across years?)
  3. Fatigue effect replication (does pre→post pattern replicate?)
  4. Cross-condition stability across all 8 conditions

Usage:
    python scripts/voronoi_longitudinal_2x2.py
"""

import os
import sys
import glob

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
from phi_frequency_model import PHI, F0

PHI_INV = 1.0 / PHI

POS_LIST = [
    ('boundary', 0.000), ('noble_6', round(PHI_INV**6, 6)),
    ('noble_5', round(PHI_INV**5, 6)), ('noble_4', round(PHI_INV**4, 6)),
    ('noble_3', round(PHI_INV**3, 6)), ('inv_noble_1', round(PHI_INV**2, 6)),
    ('attractor', 0.5), ('noble_1', round(PHI_INV, 6)),
    ('inv_noble_3', round(1 - PHI_INV**3, 6)), ('inv_noble_4', round(1 - PHI_INV**4, 6)),
    ('inv_noble_5', round(1 - PHI_INV**5, 6)), ('inv_noble_6', round(1 - PHI_INV**6, 6)),
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

OCTAVE_BAND = {'n-1': 'theta', 'n+0': 'alpha', 'n+1': 'beta_low',
               'n+2': 'beta_high', 'n+3': 'gamma'}
BAND_ORDER = ['theta', 'alpha', 'beta_low', 'beta_high', 'gamma']


def lattice_coord(freqs, f0=F0):
    return (np.log(freqs / f0) / np.log(PHI)) % 1.0


def analyze(directory):
    """Pooled enrichment for a dataset directory."""
    files = sorted(glob.glob(os.path.join(directory, '*_peaks.csv')))
    if not files:
        return None, 0, 0
    dfs = [pd.read_csv(f, usecols=['freq', 'phi_octave']) for f in files]
    peaks = pd.concat(dfs, ignore_index=True)
    results = {}
    for octave, band in OCTAVE_BAND.items():
        bp = peaks[peaks.phi_octave == octave]['freq'].values
        n = len(bp)
        if n < 10:
            continue
        u = lattice_coord(bp)
        dists = np.abs(u[:, None] - POS_VALS[None, :])
        dists = np.minimum(dists, 1 - dists)
        assignments = np.argmin(dists, axis=1)
        enrichments = {}
        for i in range(N_POS):
            count = (assignments == i).sum()
            expected = VORONOI_BINS[i] * n
            enrichments[POS_NAMES[i]] = round((count / expected - 1) * 100) if expected > 0 else 0
        lower = (u < BOUNDARY_HW).sum()
        upper = (u >= (1 - BOUNDARY_HW)).sum()
        exp_half = BOUNDARY_HW * n
        enrichments['lo'] = round((lower / exp_half - 1) * 100) if exp_half > 0 else 0
        enrichments['hi'] = round((upper / exp_half - 1) * 100) if exp_half > 0 else 0
        enrichments['n'] = n
        results[band] = enrichments
    return results, len(files), len(peaks)


def main():
    conds = {
        'S1 EC-pre': 'exports_adaptive/dortmund',
        'S1 EC-post': 'exports_adaptive/dortmund_EC_post',
        'S1 EO-pre': 'exports_adaptive/dortmund_EO_pre',
        'S1 EO-post': 'exports_adaptive/dortmund_EO_post',
        'S2 EC-pre': 'exports_adaptive/dortmund_EC_pre_ses2',
        'S2 EC-post': 'exports_adaptive/dortmund_EC_post_ses2',
        'S2 EO-pre': 'exports_adaptive/dortmund_EO_pre_ses2',
        'S2 EO-post': 'exports_adaptive/dortmund_EO_post_ses2',
    }

    data = {}
    for label, path in conds.items():
        if not os.path.exists(path):
            print(f"SKIP {label}: {path} not found")
            continue
        r, n, p = analyze(path)
        if r:
            data[label] = r
            print(f"{label}: {n} subjects, {p:,} peaks")

    all_pos = ['boundary'] + [n for n in POS_NAMES if n != 'boundary'] + ['boundary_hi']

    def get_val(d, band, pos):
        if pos == 'boundary':
            return d[band]['lo']
        elif pos == 'boundary_hi':
            return d[band]['hi']
        else:
            return d[band].get(pos, 0)

    # Profile correlations
    print(f"\nProfile correlation ses-1 vs ses-2:")
    for cond in ['EC-pre', 'EC-post', 'EO-pre', 'EO-post']:
        s1, s2 = f'S1 {cond}', f'S2 {cond}'
        if s1 not in data or s2 not in data:
            continue
        print(f"  {cond}:")
        for band in BAND_ORDER:
            if band not in data[s1] or band not in data[s2]:
                continue
            v1 = [get_val(data[s1], band, p) for p in all_pos]
            v2 = [get_val(data[s2], band, p) for p in all_pos]
            r, _ = stats.pearsonr(v1, v2)
            print(f"    {band:<12}: r = {r:.3f}")

    # Eyes effect replication
    print(f"\nEyes effect replication (EC→EO delta correlation S1↔S2):")
    for band in BAND_ORDER:
        s1_d = [get_val(data.get('S1 EO-pre', {}), band, p) - get_val(data.get('S1 EC-pre', {}), band, p)
                for p in all_pos
                if band in data.get('S1 EO-pre', {}) and band in data.get('S1 EC-pre', {})]
        s2_d = [get_val(data.get('S2 EO-pre', {}), band, p) - get_val(data.get('S2 EC-pre', {}), band, p)
                for p in all_pos
                if band in data.get('S2 EO-pre', {}) and band in data.get('S2 EC-pre', {})]
        if len(s1_d) == len(s2_d) and len(s1_d) > 3:
            r, _ = stats.pearsonr(s1_d, s2_d)
            print(f"  {band:<12}: r = {r:+.3f}")

    # Fatigue effect replication
    print(f"\nFatigue effect replication (pre→post delta correlation S1↔S2):")
    for band in BAND_ORDER:
        s1_d = [get_val(data.get('S1 EC-post', {}), band, p) - get_val(data.get('S1 EC-pre', {}), band, p)
                for p in all_pos
                if band in data.get('S1 EC-post', {}) and band in data.get('S1 EC-pre', {})]
        s2_d = [get_val(data.get('S2 EC-post', {}), band, p) - get_val(data.get('S2 EC-pre', {}), band, p)
                for p in all_pos
                if band in data.get('S2 EC-post', {}) and band in data.get('S2 EC-pre', {})]
        if len(s1_d) == len(s2_d) and len(s1_d) > 3:
            r, _ = stats.pearsonr(s1_d, s2_d)
            print(f"  {band:<12}: r = {r:+.3f}")

    # Cross-condition stability across all 8
    print(f"\nStability across ALL 8 conditions:")
    for band in BAND_ORDER:
        stable = 0
        for pos in all_pos:
            vals = []
            for c in conds:
                if c not in data or band not in data[c]:
                    continue
                vals.append(get_val(data[c], band, pos))
            if len(vals) < 8:
                continue
            signs = [1 if v > 10 else (-1 if v < -10 else 0) for v in vals]
            if all(s == signs[0] for s in signs) and signs[0] != 0:
                stable += 1
        print(f"  {band:<12}: {stable}/13")


if __name__ == '__main__':
    main()
