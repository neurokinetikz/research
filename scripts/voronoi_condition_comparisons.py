#!/usr/bin/env python3
"""
Voronoi Enrichment Condition Comparisons
=========================================

Runs all condition-based comparisons using per-subject Voronoi enrichment:
  1. EC vs EO (LEMON + Dortmund)
  2. Pre vs Post fatigue (Dortmund EC-pre vs EC-post)
  3. Adult vs Pediatric (4 adult datasets vs 5 HBN releases)
  4. Full Dortmund 2x2 (EC/EO × pre/post) — when all 4 conditions available

Usage:
    python scripts/voronoi_condition_comparisons.py --analysis ec_eo
    python scripts/voronoi_condition_comparisons.py --analysis fatigue
    python scripts/voronoi_condition_comparisons.py --analysis adult_pediatric
    python scripts/voronoi_condition_comparisons.py --analysis all
"""

import os
import sys
import argparse
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

DATASETS = {
    'eegmmidb': 'exports_adaptive/eegmmidb',
    'lemon': 'exports_adaptive/lemon',
    'lemon_EO': 'exports_adaptive/lemon_EO',
    'dortmund': 'exports_adaptive/dortmund',
    'dortmund_EO_pre': 'exports_adaptive/dortmund_EO_pre',
    'dortmund_EC_post': 'exports_adaptive/dortmund_EC_post',
    'dortmund_EO_post': 'exports_adaptive/dortmund_EO_post',
    'chbmp': 'exports_adaptive/chbmp',
    'hbn_R1': 'exports_adaptive/hbn_R1',
    'hbn_R2': 'exports_adaptive/hbn_R2',
    'hbn_R3': 'exports_adaptive/hbn_R3',
    'hbn_R4': 'exports_adaptive/hbn_R4',
    'hbn_R6': 'exports_adaptive/hbn_R6',
}


def lattice_coord(freqs, f0=F0):
    return (np.log(freqs / f0) / np.log(PHI)) % 1.0


def analyze_dataset(directory):
    """Compute pooled enrichment for a dataset. Returns dict of band -> position -> enrichment%."""
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
        pos_arr = POS_VALS
        dists = np.abs(u[:, None] - pos_arr[None, :])
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


def compare_conditions(data_a, data_b, label_a, label_b):
    """Print comparison table between two conditions."""
    all_pos = ['boundary'] + [n for n in POS_NAMES if n != 'boundary'] + ['boundary_hi']

    for band in BAND_ORDER:
        if band not in data_a or band not in data_b:
            continue
        print(f"\n  {band.upper()} (A N={data_a[band]['n']:,}  B N={data_b[band]['n']:,})")
        print(f"  {'Position':<16} {label_a:>7} {label_b:>7} {'Δ':>6} {'Dir'}")
        print(f"  {'-'*44}")
        for pos in all_pos:
            if pos == 'boundary':
                a_v, b_v = data_a[band]['lo'], data_b[band]['lo']
            elif pos == 'boundary_hi':
                a_v, b_v = data_a[band]['hi'], data_b[band]['hi']
            else:
                a_v = data_a[band].get(pos, 0)
                b_v = data_b[band].get(pos, 0)
            delta = b_v - a_v
            d = f'{label_b}↑' if delta > 10 else (f'{label_a}↑' if delta < -10 else '')
            pname = 'boundary (hi)' if pos == 'boundary_hi' else pos
            print(f"  {pname:<16} {a_v:>+5d}% {b_v:>+5d}% {delta:>+5d}  {d}")


def run_ec_eo():
    """EC vs EO comparison for LEMON and Dortmund."""
    print(f"\n{'='*70}")
    print(f"  EC vs EO COMPARISON")
    print(f"{'='*70}")

    pairs = [
        ('lemon', 'lemon_EO', 'LEMON'),
        ('dortmund', 'dortmund_EO_pre', 'Dortmund'),
    ]

    all_data = {}
    for ec_key, eo_key, label in pairs:
        ec_dir = DATASETS.get(ec_key)
        eo_dir = DATASETS.get(eo_key)
        if not ec_dir or not eo_dir:
            continue
        if not os.path.exists(ec_dir) or not os.path.exists(eo_dir):
            print(f"\n  SKIP {label}: missing {ec_dir} or {eo_dir}")
            continue

        ec, ec_n, ec_p = analyze_dataset(ec_dir)
        eo, eo_n, eo_p = analyze_dataset(eo_dir)
        print(f"\n  {label} EC: {ec_n} subjects, {ec_p:,} peaks")
        print(f"  {label} EO: {eo_n} subjects, {eo_p:,} peaks")
        compare_conditions(ec, eo, 'EC', 'EO')
        all_data[f'{label}_EC'] = ec
        all_data[f'{label}_EO'] = eo

    # Cross-dataset consistency
    if len(all_data) >= 4:
        print(f"\n{'='*70}")
        print(f"  CONSISTENTLY REPLICATED EC→EO SHIFTS")
        print(f"{'='*70}")
        all_pos = ['boundary'] + [n for n in POS_NAMES if n != 'boundary'] + ['boundary_hi']
        for band in BAND_ORDER:
            consistent = []
            for pos in all_pos:
                deltas = []
                for label in ['LEMON', 'Dortmund']:
                    ec = all_data.get(f'{label}_EC', {})
                    eo = all_data.get(f'{label}_EO', {})
                    if band not in ec or band not in eo:
                        continue
                    if pos == 'boundary':
                        d = eo[band]['lo'] - ec[band]['lo']
                    elif pos == 'boundary_hi':
                        d = eo[band]['hi'] - ec[band]['hi']
                    else:
                        d = eo[band].get(pos, 0) - ec[band].get(pos, 0)
                    deltas.append(d)
                if len(deltas) == 2 and all(abs(d) > 10 for d in deltas):
                    if (deltas[0] > 0) == (deltas[1] > 0):
                        pname = 'boundary(hi)' if pos == 'boundary_hi' else pos
                        consistent.append(f"{pname}(L:{deltas[0]:+d}, D:{deltas[1]:+d})")
            if consistent:
                print(f"\n  {band.upper()}: {', '.join(consistent)}")


def run_fatigue():
    """Dortmund pre vs post fatigue comparison."""
    print(f"\n{'='*70}")
    print(f"  DORTMUND PRE vs POST (FATIGUE EFFECT)")
    print(f"{'='*70}")

    pre_dir = DATASETS.get('dortmund')
    post_dir = DATASETS.get('dortmund_EC_post')
    if not pre_dir or not post_dir or not os.path.exists(post_dir):
        print("  EC-post not available")
        return

    pre, pre_n, pre_p = analyze_dataset(pre_dir)
    post, post_n, post_p = analyze_dataset(post_dir)
    print(f"  EC-pre: {pre_n} subjects, {pre_p:,} peaks")
    print(f"  EC-post: {post_n} subjects, {post_p:,} peaks")
    compare_conditions(pre, post, 'Pre', 'Post')

    # Compare fatigue vs eyes effect sizes
    eo_dir = DATASETS.get('dortmund_EO_pre')
    if eo_dir and os.path.exists(eo_dir):
        eo, _, _ = analyze_dataset(eo_dir)
        print(f"\n  Effect Size Comparison (max |Δ| per band):")
        print(f"  {'Band':<12} {'Fatigue':>10} {'Eyes':>10}")
        all_pos = ['boundary'] + [n for n in POS_NAMES if n != 'boundary'] + ['boundary_hi']
        for band in BAND_ORDER:
            if band not in pre or band not in post or band not in eo:
                continue
            max_fat = max_eye = 0
            for pos in all_pos:
                if pos == 'boundary':
                    k = 'lo'
                elif pos == 'boundary_hi':
                    k = 'hi'
                else:
                    k = pos
                fat = abs(post[band].get(k, 0) - pre[band].get(k, 0))
                eye = abs(eo[band].get(k, 0) - pre[band].get(k, 0))
                max_fat = max(max_fat, fat)
                max_eye = max(max_eye, eye)
            print(f"  {band:<12} {max_fat:>10} {max_eye:>10}")


def run_adult_pediatric():
    """Adult vs Pediatric comparison."""
    print(f"\n{'='*70}")
    print(f"  ADULT vs PEDIATRIC COMPARISON")
    print(f"{'='*70}")

    adult_keys = ['eegmmidb', 'lemon', 'dortmund', 'chbmp']
    ped_keys = ['hbn_R1', 'hbn_R2', 'hbn_R3', 'hbn_R4', 'hbn_R6']

    adult_data = {}
    ped_data = {}
    for key in adult_keys:
        d = DATASETS.get(key)
        if d and os.path.exists(d):
            r, n, _ = analyze_dataset(d)
            if r:
                adult_data[key] = r

    for key in ped_keys:
        d = DATASETS.get(key)
        if d and os.path.exists(d):
            r, n, _ = analyze_dataset(d)
            if r:
                ped_data[key] = r

    print(f"  Adult datasets: {len(adult_data)} ({', '.join(adult_data.keys())})")
    print(f"  Pediatric datasets: {len(ped_data)} ({', '.join(ped_data.keys())})")

    all_pos = ['boundary'] + [n for n in POS_NAMES if n != 'boundary'] + ['boundary_hi']

    for band in BAND_ORDER:
        print(f"\n  {band.upper()}")
        print(f"  {'Position':<16} {'Adult':>7} {'(SD)':>5} {'Ped':>7} {'(SD)':>5} {'Δ':>7} {'p':>8}")
        print(f"  {'-'*58}")
        for pos in all_pos:
            adult_vals = []
            ped_vals = []
            for ds in adult_data.values():
                if band not in ds:
                    continue
                if pos == 'boundary':
                    adult_vals.append(ds[band]['lo'])
                elif pos == 'boundary_hi':
                    adult_vals.append(ds[band]['hi'])
                else:
                    adult_vals.append(ds[band].get(pos, 0))
            for ds in ped_data.values():
                if band not in ds:
                    continue
                if pos == 'boundary':
                    ped_vals.append(ds[band]['lo'])
                elif pos == 'boundary_hi':
                    ped_vals.append(ds[band]['hi'])
                else:
                    ped_vals.append(ds[band].get(pos, 0))

            if len(adult_vals) < 2 or len(ped_vals) < 2:
                continue

            a_mean = np.mean(adult_vals)
            p_mean = np.mean(ped_vals)
            a_sd = np.std(adult_vals, ddof=1)
            p_sd = np.std(ped_vals, ddof=1)
            delta = p_mean - a_mean

            try:
                _, pval = stats.mannwhitneyu(adult_vals, ped_vals, alternative='two-sided')
            except Exception:
                pval = 1.0

            sig = '*' if pval < 0.05 else ''
            pname = 'boundary (hi)' if pos == 'boundary_hi' else pos
            print(f"  {pname:<16} {a_mean:>+6.0f}% {a_sd:>4.0f}  {p_mean:>+6.0f}% {p_sd:>4.0f}  {delta:>+6.0f}  {pval:>7.3f}{sig}")

    # Profile correlations
    print(f"\n  Adult-Pediatric profile correlations:")
    for band in BAND_ORDER:
        adult_profile = []
        ped_profile = []
        for pos in all_pos:
            a_vals = []
            p_vals = []
            for ds in adult_data.values():
                if band not in ds:
                    continue
                if pos == 'boundary':
                    a_vals.append(ds[band]['lo'])
                elif pos == 'boundary_hi':
                    a_vals.append(ds[band]['hi'])
                else:
                    a_vals.append(ds[band].get(pos, 0))
            for ds in ped_data.values():
                if band not in ds:
                    continue
                if pos == 'boundary':
                    p_vals.append(ds[band]['lo'])
                elif pos == 'boundary_hi':
                    p_vals.append(ds[band]['hi'])
                else:
                    p_vals.append(ds[band].get(pos, 0))
            if a_vals and p_vals:
                adult_profile.append(np.mean(a_vals))
                ped_profile.append(np.mean(p_vals))

        if len(adult_profile) > 3:
            r, p = stats.pearsonr(adult_profile, ped_profile)
            print(f"    {band:<12}: r={r:.3f} (p={p:.4f})")


def run_2x2():
    """Dortmund full 2×2: EC/EO × pre/post."""
    print(f"\n{'='*70}")
    print(f"  DORTMUND 2×2: EC/EO × Pre/Post")
    print(f"{'='*70}")

    cond_keys = {
        'EC-pre': 'dortmund',
        'EO-pre': 'dortmund_EO_pre',
        'EC-post': 'dortmund_EC_post',
        'EO-post': 'dortmund_EO_post',
    }

    data = {}
    for label, key in cond_keys.items():
        d = DATASETS.get(key)
        if not d or not os.path.exists(d):
            print(f"  SKIP {label}: {d} not found")
            return
        r, n, p = analyze_dataset(d)
        data[label] = r
        print(f"  {label}: {n} subjects, {p:,} peaks")

    all_pos = ['boundary'] + [n for n in POS_NAMES if n != 'boundary'] + ['boundary_hi']

    for band in BAND_ORDER:
        if any(band not in data[c] for c in cond_keys):
            continue
        print(f"\n  {band.upper()}")
        print(f"  {'Position':<16} {'EC-pre':>7} {'EC-post':>8} {'EO-pre':>7} {'EO-post':>8}"
              f" | {'Δeyes-pre':>10} {'Δeyes-post':>11} {'Δfat-EC':>8} {'Δfat-EO':>8}")
        print(f"  {'-'*100}")
        for pos in all_pos:
            vals = {}
            for c in cond_keys:
                if pos == 'boundary':
                    vals[c] = data[c][band]['lo']
                elif pos == 'boundary_hi':
                    vals[c] = data[c][band]['hi']
                else:
                    vals[c] = data[c][band].get(pos, 0)

            d_eyes_pre = vals['EO-pre'] - vals['EC-pre']
            d_eyes_post = vals['EO-post'] - vals['EC-post']
            d_fat_ec = vals['EC-post'] - vals['EC-pre']
            d_fat_eo = vals['EO-post'] - vals['EO-pre']

            pname = 'boundary (hi)' if pos == 'boundary_hi' else pos
            print(f"  {pname:<16} {vals['EC-pre']:>+5d}%  {vals['EC-post']:>+6d}%"
                  f"  {vals['EO-pre']:>+5d}%  {vals['EO-post']:>+6d}%"
                  f"  | {d_eyes_pre:>+9d}  {d_eyes_post:>+10d}"
                  f"  {d_fat_ec:>+7d}  {d_fat_eo:>+7d}")

    # Cross-condition stability
    print(f"\n  Cross-condition stability (all 4 agree on direction >±10%):")
    for band in BAND_ORDER:
        if any(band not in data[c] for c in cond_keys):
            continue
        stable = 0
        for pos in all_pos:
            vals = []
            for c in cond_keys:
                if pos == 'boundary':
                    vals.append(data[c][band]['lo'])
                elif pos == 'boundary_hi':
                    vals.append(data[c][band]['hi'])
                else:
                    vals.append(data[c][band].get(pos, 0))
            signs = [1 if v > 10 else (-1 if v < -10 else 0) for v in vals]
            if all(s == signs[0] for s in signs) and signs[0] != 0:
                stable += 1
        print(f"    {band:<12}: {stable}/13")


def main():
    parser = argparse.ArgumentParser(description='Voronoi condition comparisons')
    parser.add_argument('--analysis', type=str, default='all',
                        choices=['ec_eo', 'fatigue', 'adult_pediatric', 'dortmund_2x2', 'all'])
    args = parser.parse_args()

    if args.analysis in ('ec_eo', 'all'):
        run_ec_eo()
    if args.analysis in ('fatigue', 'all'):
        run_fatigue()
    if args.analysis in ('adult_pediatric', 'all'):
        run_adult_pediatric()
    if args.analysis in ('dortmund_2x2', 'all'):
        run_2x2()


if __name__ == '__main__':
    main()
