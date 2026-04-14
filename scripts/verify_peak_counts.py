#!/usr/bin/env python3
"""
Verify peak counts for the spectral differentiation paper.

Reports initial (all extracted) and final (R²-filtered + power-filtered)
peak counts per dataset and total, matching the numbers cited in the paper.

Usage:
    python scripts/verify_peak_counts.py
"""

import os
import sys
import glob
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
from phi_frequency_model import PHI, F0

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
PEAK_BASE = os.path.join(BASE_DIR, 'exports_adaptive_v3')
MIN_POWER_PCT = 50
R2_MIN = 0.70

EC_DATASETS = {
    'eegmmidb': 'eegmmidb', 'lemon': 'lemon', 'dortmund': 'dortmund',
    'chbmp': 'chbmp', 'hbn_R1': 'hbn_R1', 'hbn_R2': 'hbn_R2',
    'hbn_R3': 'hbn_R3', 'hbn_R4': 'hbn_R4', 'hbn_R6': 'hbn_R6',
}


def main():
    total_raw = 0
    total_r2 = 0
    total_final = 0
    total_subjects = 0

    print(f"Peak base: {PEAK_BASE}")
    print(f"R² threshold: {R2_MIN}")
    print(f"Power filter: top {MIN_POWER_PCT}%\n")

    print(f"{'Dataset':>12s}  {'Subjects':>8s}  {'Raw':>12s}  {'R²≥{:.2f}'.format(R2_MIN):>12s}  {'Final':>12s}")
    print("-" * 62)

    for name, subdir in EC_DATASETS.items():
        path = os.path.join(PEAK_BASE, subdir)
        files = sorted(glob.glob(os.path.join(path, '*_peaks.csv')))
        if not files:
            print(f"  {name}: no data")
            continue

        n_subjects = len(files)
        raw = 0
        r2_count = 0
        final = 0

        for f in files:
            df = pd.read_csv(f, usecols=['freq', 'power', 'phi_octave', 'r_squared'])
            raw += len(df)
            df = df[df['r_squared'] >= R2_MIN]
            r2_count += len(df)
            for octave in df['phi_octave'].unique():
                bp = df[df.phi_octave == octave]
                thresh = bp['power'].quantile(MIN_POWER_PCT / 100)
                final += len(bp[bp['power'] >= thresh])

        total_raw += raw
        total_r2 += r2_count
        total_final += final
        total_subjects += n_subjects

        print(f"{name:>12s}  {n_subjects:>8d}  {raw:>12,}  {r2_count:>12,}  {final:>12,}")

    print("-" * 62)
    print(f"{'TOTAL':>12s}  {total_subjects:>8d}  {total_raw:>12,}  {total_r2:>12,}  {total_final:>12,}")
    print(f"\nRetention: {total_final/total_raw*100:.1f}% of raw peaks")

    # Median R²
    import numpy as np
    all_r2 = []
    for name, subdir in EC_DATASETS.items():
        path = os.path.join(PEAK_BASE, subdir)
        for f in sorted(glob.glob(os.path.join(path, '*_peaks.csv'))):
            df = pd.read_csv(f, usecols=['r_squared'])
            df = df[df['r_squared'] >= R2_MIN]
            all_r2.extend(df['r_squared'].values)
    print(f"Median R² (retained peaks): {np.median(all_r2):.3f}")


if __name__ == '__main__':
    main()
