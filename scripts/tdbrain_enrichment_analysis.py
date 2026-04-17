#!/usr/bin/env python3
"""
TDBRAIN Enrichment-Based Analysis: ADHD vs MDD
================================================

The trough-depth metric fails at 26 channels / 2-minute recordings
(too few peaks per narrow window). This script uses the full Voronoi
enrichment profile instead -- pooling across all peaks within each band,
which is more robust to low peak counts.

Tests:
  1. Per-subject enrichment profiles → ADHD vs MDD comparison
  2. Key spectral differentiation features: alpha mountain height,
     beta-low ramp slope, center depletion
  3. α/β boundary enrichment (upper alpha depletion)
  4. Age prediction from enrichment features

Usage:
    python scripts/tdbrain_enrichment_analysis.py

Requires: exports_adaptive_v4/tdbrain/
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT_DIR = os.path.join(BASE_DIR, 'outputs', 'tdbrain_analysis')
PEAK_DIR = os.path.join(BASE_DIR, 'exports_adaptive_v4', 'tdbrain')
PARTICIPANTS_PATH = os.path.expanduser(
    '~/Desktop/TDBRAIN_participants_V2_data/TDBRAIN_participants_V2.tsv')

sys.path.insert(0, os.path.join(BASE_DIR, 'lib'))
from phi_frequency_model import PHI, F0

MIN_POWER_PCT = 50

BAND_OCTAVES = {'theta': -1, 'alpha': 0, 'beta_low': 1, 'beta_high': 2, 'gamma': 3}

# 8 key positions (degree-3 lattice)
POSITIONS = {
    'boundary': 0.000,
    'noble_3': 0.236,
    'noble_2': 0.382,
    'attractor': 0.500,
    'noble_1': 0.618,
    'inv_noble_3': 0.764,
    'inv_noble_4': 0.854,
    'boundary_hi': 1.000,
}


def lattice_coord(freq):
    """Convert frequency to phi-octave coordinate u ∈ [0,1)."""
    return (np.log(freq / F0) / np.log(PHI)) % 1.0


def compute_enrichment(freqs, band_octave, min_peaks=10):
    """Compute Voronoi enrichment at key positions for one band."""
    lo = F0 * PHI ** band_octave
    hi = F0 * PHI ** (band_octave + 1)
    band_freqs = freqs[(freqs >= lo) & (freqs < hi)]

    if len(band_freqs) < min_peaks:
        return {name: np.nan for name in POSITIONS}

    u = lattice_coord(band_freqs)
    n_total = len(u)

    # Voronoi assignment: each peak goes to nearest position
    pos_array = np.array(list(POSITIONS.values()))
    pos_names = list(POSITIONS.keys())

    enrichment = {}
    for name, pos_u in POSITIONS.items():
        # Circular distance to each position
        all_dists = np.abs(u[:, None] - pos_array[None, :])
        all_dists = np.minimum(all_dists, 1 - all_dists)
        assignments = np.argmin(all_dists, axis=1)
        pos_idx = pos_names.index(name)
        count = np.sum(assignments == pos_idx)

        # Hz-weighted expected count
        u_lo = (pos_u + pos_array[(pos_idx - 1) % len(pos_array)]) / 2
        u_hi = (pos_u + pos_array[(pos_idx + 1) % len(pos_array)]) / 2
        # Simplified: use equal expected
        expected = n_total / len(pos_array)
        if expected > 0:
            enrichment[name] = (count / expected - 1) * 100
        else:
            enrichment[name] = np.nan

    return enrichment


def compute_subject_features(sub_id):
    """Compute enrichment features for one subject."""
    peak_path = os.path.join(PEAK_DIR, f'{sub_id}_peaks.csv')
    if not os.path.exists(peak_path):
        return None

    try:
        df = pd.read_csv(peak_path, usecols=['freq', 'power', 'phi_octave'])
    except Exception:
        return None

    # Power filter
    filtered = []
    for octave in df['phi_octave'].unique():
        bp = df[df.phi_octave == octave]
        if len(bp) == 0:
            continue
        thresh = bp['power'].quantile(MIN_POWER_PCT / 100)
        filtered.append(bp[bp['power'] >= thresh])
    if not filtered:
        return None
    df = pd.concat(filtered, ignore_index=True)
    freqs = df['freq'].values

    if len(freqs) < 50:
        return None

    features = {'subject': sub_id, 'n_peaks': len(freqs)}

    # Per-band enrichment
    for band_name, octave in BAND_OCTAVES.items():
        enrichment = compute_enrichment(freqs, octave)
        for pos_name, val in enrichment.items():
            features[f'{band_name}_{pos_name}'] = val

        # Derived metrics
        vals = enrichment
        if not np.isnan(vals.get('attractor', np.nan)):
            # Mountain height: attractor enrichment (how peaked is the band center)
            features[f'{band_name}_mountain'] = vals.get('noble_1', 0) - vals.get('boundary', 0)
            # Ramp: upper vs lower enrichment
            features[f'{band_name}_ramp'] = vals.get('inv_noble_4', 0) - vals.get('noble_3', 0)
            # Center depletion: attractor vs mean of lower positions
            lower = np.nanmean([vals.get('boundary', 0), vals.get('noble_3', 0), vals.get('noble_2', 0)])
            features[f'{band_name}_center_depletion'] = vals.get('attractor', 0) - lower
            # Profile range
            all_vals = [v for v in vals.values() if not np.isnan(v)]
            if all_vals:
                features[f'{band_name}_range'] = max(all_vals) - min(all_vals)

    return features


def load_participants():
    """Load and classify TDBRAIN participants."""
    df = pd.read_csv(PARTICIPANTS_PATH, sep='\t')
    df['age_float'] = df['age'].str.replace(',', '.').astype(float)
    df['dx_group'] = 'OTHER'
    df.loc[df['indication'] == 'HEALTHY', 'dx_group'] = 'HEALTHY'
    df.loc[df['indication'].str.contains('ADHD', na=False) &
           ~df['indication'].str.contains('MDD', na=False), 'dx_group'] = 'ADHD'
    df.loc[df['indication'].str.contains('MDD', na=False) &
           ~df['indication'].str.contains('ADHD', na=False), 'dx_group'] = 'MDD'
    df.loc[df['indication'].str.contains('OCD', na=False), 'dx_group'] = 'OCD'
    df = df[df['DISC/REP'] == 'DISCOVERY'].copy()
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("TDBRAIN Enrichment Analysis: ADHD vs MDD")
    print("=" * 70)

    participants = load_participants()

    # Compute features
    print("\n--- Computing Enrichment Features ---")
    peak_files = sorted(glob.glob(os.path.join(PEAK_DIR, '*_peaks.csv')))
    rows = []
    for i, f in enumerate(peak_files):
        sub_id = os.path.basename(f).replace('_peaks.csv', '')
        features = compute_subject_features(sub_id)
        if features:
            rows.append(features)
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(peak_files)}")

    features_df = pd.DataFrame(rows)
    print(f"  {len(features_df)} subjects with features")

    # Merge
    merged = features_df.merge(
        participants[['participants_ID', 'age_float', 'gender', 'dx_group']],
        left_on='subject', right_on='participants_ID', how='inner')

    adults = merged[merged['age_float'] >= 18].copy()
    adhd = adults[adults.dx_group == 'ADHD']
    mdd = adults[adults.dx_group == 'MDD']
    healthy = adults[adults.dx_group == 'HEALTHY']

    print(f"\n  Adults: ADHD={len(adhd)}, MDD={len(mdd)}, HEALTHY={len(healthy)}")

    # ==============================================
    # Key spectral differentiation features
    # ==============================================
    print("\n" + "=" * 70)
    print("KEY SPECTRAL DIFFERENTIATION FEATURES: ADHD vs MDD")
    print("=" * 70)

    key_features = [
        # Alpha organization
        ('alpha_attractor', 'Alpha mountain (attractor enrichment)'),
        ('alpha_mountain', 'Alpha mountain height (noble_1 - boundary)'),
        ('alpha_range', 'Alpha profile range'),
        # Beta-low differentiation
        ('beta_low_ramp', 'Beta-low ramp (inv_noble_4 - noble_3)'),
        ('beta_low_center_depletion', 'Beta-low center depletion'),
        ('beta_low_range', 'Beta-low profile range'),
        # Alpha/beta boundary (upper alpha depletion)
        ('alpha_boundary_hi', 'Alpha upper boundary (α/β transition)'),
        ('alpha_inv_noble_4', 'Alpha inv_noble_4 (near α/β boundary)'),
        # Theta
        ('theta_ramp', 'Theta ramp'),
        ('theta_boundary_hi', 'Theta upper boundary (f₀ convergence)'),
        # Beta-high
        ('beta_high_boundary', 'Beta-high lower boundary (bridge)'),
    ]

    print(f"\n  {'Feature':<45} {'ADHD':>8} {'MDD':>8} {'d':>8} {'p':>8} {'Dir':>15}")
    print("  " + "-" * 95)

    results = []
    for feat_col, feat_name in key_features:
        a = adhd[feat_col].dropna()
        m = mdd[feat_col].dropna()
        if len(a) > 10 and len(m) > 10:
            u, p = stats.mannwhitneyu(a, m, alternative='two-sided')
            d = (a.mean() - m.mean()) / np.sqrt((a.std()**2 + m.std()**2) / 2)
            sig = '*' if p < 0.05 else ' '
            direction = 'ADHD higher' if a.median() > m.median() else 'MDD higher'
            print(f"  {feat_name:<45} {a.median():>8.1f} {m.median():>8.1f} {d:>+8.3f} {p:>7.4f}{sig} {direction:>15}")
            results.append({
                'feature': feat_col, 'description': feat_name,
                'adhd_median': a.median(), 'mdd_median': m.median(),
                'cohens_d': d, 'p_value': p, 'direction': direction,
            })

    # ==============================================
    # The paper's prediction at α/β boundary
    # ==============================================
    print("\n" + "=" * 70)
    print("α/β BOUNDARY: Paper's Inhibitory Prediction")
    print("=" * 70)

    # The α/β boundary is measured by depletion at alpha's upper edge
    # More depletion = deeper boundary = more inhibitory carving
    # Prediction: ADHD less depleted (weaker inhibition), MDD more depleted
    for col, desc in [('alpha_boundary_hi', 'Alpha boundary_hi (upper edge)'),
                       ('alpha_inv_noble_4', 'Alpha inv_noble_4'),
                       ('alpha_inv_noble_3', 'Alpha inv_noble_3')]:
        a = adhd[col].dropna()
        m = mdd[col].dropna()
        if len(a) > 10 and len(m) > 10:
            u, p = stats.mannwhitneyu(a, m, alternative='two-sided')
            d = (a.mean() - m.mean()) / np.sqrt((a.std()**2 + m.std()**2) / 2)
            print(f"  {desc}: ADHD={a.median():.1f}%, MDD={m.median():.1f}%, d={d:+.3f}, p={p:.4f}")
            # Prediction: ADHD should be LESS depleted (higher/less negative enrichment)
            if a.median() > m.median():
                print(f"    ✓ ADHD less depleted (weaker inhibition) — predicted")
            else:
                print(f"    ✗ MDD less depleted — opposite to prediction")

    # ==============================================
    # Age correlation of features (replication)
    # ==============================================
    print("\n" + "=" * 70)
    print("AGE CORRELATIONS (Replication of Paper's Developmental Findings)")
    print("=" * 70)

    print(f"\n  {'Feature':<45} {'ρ':>8} {'p':>10} {'N':>6}")
    print("  " + "-" * 75)

    for feat_col, feat_name in key_features:
        valid = adults.dropna(subset=[feat_col, 'age_float'])
        if len(valid) > 30:
            rho, p = stats.spearmanr(valid['age_float'], valid[feat_col])
            sig = '*' if p < 0.05 else ' '
            print(f"  {feat_name:<45} {rho:>+8.3f} {p:>9.4f}{sig} {len(valid):>6}")

    # ==============================================
    # Full profile comparison
    # ==============================================
    print("\n" + "=" * 70)
    print("FULL ENRICHMENT PROFILES: ADHD vs MDD by Band")
    print("=" * 70)

    for band_name in ['alpha', 'beta_low', 'theta', 'beta_high']:
        print(f"\n  --- {band_name} ---")
        print(f"  {'Position':<20} {'ADHD':>8} {'MDD':>8} {'d':>8} {'p':>8}")
        print(f"  " + "-" * 55)
        for pos_name in POSITIONS:
            col = f'{band_name}_{pos_name}'
            a = adhd[col].dropna()
            m = mdd[col].dropna()
            if len(a) > 10 and len(m) > 10:
                u, p = stats.mannwhitneyu(a, m, alternative='two-sided')
                d = (a.mean() - m.mean()) / np.sqrt((a.std()**2 + m.std()**2) / 2)
                sig = '*' if p < 0.05 else ' '
                print(f"  {pos_name:<20} {a.median():>8.1f} {m.median():>8.1f} {d:>+8.3f} {p:>7.4f}{sig}")

    # ==============================================
    # Three-way: HEALTHY vs ADHD vs MDD
    # ==============================================
    print("\n" + "=" * 70)
    print("THREE-WAY: Key Features by Diagnostic Group")
    print("=" * 70)

    for feat_col, feat_name in key_features[:6]:
        h = healthy[feat_col].dropna()
        a = adhd[feat_col].dropna()
        m = mdd[feat_col].dropna()
        if len(h) > 5 and len(a) > 10 and len(m) > 10:
            _, p_kw = stats.kruskal(h, a, m)
            print(f"  {feat_name}: HEALTHY={h.median():.1f}, ADHD={a.median():.1f}, "
                  f"MDD={m.median():.1f}, KW p={p_kw:.4f}")

    # Save
    pd.DataFrame(results).to_csv(os.path.join(OUT_DIR, 'tdbrain_enrichment_adhd_mdd.csv'), index=False)
    merged.to_csv(os.path.join(OUT_DIR, 'tdbrain_enrichment_features.csv'), index=False)

    print(f"\nResults saved to {OUT_DIR}/")


if __name__ == '__main__':
    main()
