#!/usr/bin/env python3
"""
P20 Analysis: Extended Gamma Enrichment (n+4 band, 53.67–75 Hz)
================================================================

Tests whether phi-lattice enrichment structure extends into the
previously unmeasured upper gamma region (n+4 phi-octave).

Uses EEGMMIDB overlap-trim extraction WITHOUT notch filter to preserve
clean spectrum through 75 Hz. 60 Hz line noise peaks are identified
and excluded by bandwidth signature (narrow peaks at 59-61 Hz).

Analyses:
  1. Lattice position enrichment in n+4 band (with 60 Hz exclusion)
  2. Comparison with n+3 band (established gamma enrichment)
  3. Per-position enrichment at all 14 degree-7 positions
  4. Statistical validation (permutation test vs uniform null)
  5. 60 Hz exclusion sensitivity analysis

Usage:
    python scripts/analyze_p20_extended_gamma.py
    python scripts/analyze_p20_extended_gamma.py --input-dir exports_eegmmidb/p20_no_notch_f07.83
"""

import os
import sys
import argparse
from glob import glob
import numpy as np
import pandas as pd
from scipy import stats

# =========================================================================
# CONSTANTS
# =========================================================================

PHI = (1 + np.sqrt(5)) / 2
F0 = 7.83

# 14 degree-7 positions (same as used in Papers 2 & 3)
POSITIONS_DEG7 = {
    'boundary':   0.000,
    'inv_noble7': 1 - 1/PHI**7,   # 0.0557
    'inv_noble6': 1 - 1/PHI**6,   # 0.0902
    'inv_noble5': 1 - 1/PHI**5,   # 0.1459
    'inv_noble4': 1 - 1/PHI**4,   # 0.2361
    'inv_noble3': 1 - 1/PHI**3,   # 0.3090
    'noble_2':    1/PHI**2,        # 0.3820
    'attractor':  0.500,
    'noble_1':    1/PHI,           # 0.6180
    'noble_3':    1 - 1/PHI**3,    # wait - this is same as inv_noble3
}

# Recalculate properly
def build_deg7_positions():
    inv = 1.0 / PHI
    pos = {}
    pos['boundary']   = 0.0
    pos['inv_noble7'] = inv**7           # 0.0557
    pos['inv_noble6'] = inv**6           # 0.0902
    pos['inv_noble5'] = inv**5           # 0.1459
    pos['inv_noble4'] = inv**4           # 0.2361
    pos['inv_noble3'] = inv**3           # 0.3820? No...
    # Standard degree-7 layout from the papers:
    # 0, inv^7, inv^6, inv^5, inv^4, inv^3, inv^2, 0.5, 1-inv^2, 1-inv^3, 1-inv^4, 1-inv^5, 1-inv^6, 1-inv^7
    pos_list = []
    for k in range(1, 8):
        pos_list.append((f'pos_{k}L', inv**k))
        pos_list.append((f'pos_{k}R', 1 - inv**k))
    pos_list.append(('boundary', 0.0))
    pos_list.append(('attractor', 0.5))
    # Deduplicate and sort
    all_pos = {}
    for name, val in pos_list:
        val = val % 1.0
        # Skip if too close to existing
        if all(abs(val - v) > 0.01 for v in all_pos.values()):
            all_pos[name] = val
    return dict(sorted(all_pos.items(), key=lambda x: x[1]))

# Use the standard 14-position layout
POSITIONS = {
    'boundary':    0.000,
    '7_noble':     (1/PHI)**7,       # 0.0557
    '6_noble':     (1/PHI)**6,       # 0.0902
    '5_noble':     (1/PHI)**5,       # 0.1459
    '4_noble':     (1/PHI)**4,       # 0.2361
    '3_noble':     (1/PHI)**3,       # 0.3820
    'noble_2':     (1/PHI)**2,       # 0.3820 -- same as 3_noble!
}

# Let me just hardcode the known values from the papers
POSITIONS_14 = {}
inv = 1.0 / PHI
for k in range(1, 8):
    POSITIONS_14[f'{k}L'] = round(inv**k, 6)
    POSITIONS_14[f'{k}R'] = round(1 - inv**k, 6)
# boundary (0.0) maps to 7R (1 - inv^7 ≈ 0.9443 ... no)
# Actually boundary = 0.0 and attractor = 0.5

# Simplify: just use the canonical positions
POS_14 = np.array(sorted(set([0.0, 0.5] + [inv**k for k in range(1,8)] + [1 - inv**k for k in range(1,8)])))
# Remove duplicates near 0 and 1
POS_14 = POS_14[(POS_14 >= 0) & (POS_14 < 1)]
POS_14 = np.unique(POS_14.round(6))

# Named positions
POS_NAMES = {}
POS_NAMES[0.0] = 'boundary'
POS_NAMES[0.5] = 'attractor'
for k in range(1, 8):
    v = round(inv**k, 6)
    if v not in POS_NAMES:
        POS_NAMES[v] = f'noble_{k}'
    v2 = round(1 - inv**k, 6)
    if v2 not in POS_NAMES:
        POS_NAMES[v2] = f'inv_noble_{k}'


def lattice_coord(freq, f0=F0):
    """Map frequency to phi-lattice coordinate u ∈ [0, 1)."""
    return (np.log(freq / f0) / np.log(PHI)) % 1.0


def nearest_position(u, positions=POS_14):
    """Find nearest position and distance (with wraparound)."""
    dists = np.minimum(np.abs(u - positions), 1 - np.abs(u - positions))
    idx = np.argmin(dists)
    return positions[idx], dists[idx]


def enrichment_at_positions(coords, positions=POS_14, half_width=0.04):
    """Compute enrichment (observed/expected - 1) at each position.

    For each position, count peaks within ±half_width and compare
    to uniform expectation.
    """
    n = len(coords)
    if n == 0:
        return {}

    results = {}
    for pos in positions:
        pos_name = POS_NAMES.get(round(pos, 6), f'{pos:.4f}')
        # Count peaks within ±half_width (with wraparound)
        dists = np.minimum(np.abs(coords - pos), 1 - np.abs(coords - pos))
        n_in_bin = (dists <= half_width).sum()
        # Expected under uniform: fraction of [0,1) covered × n
        expected = 2 * half_width * n
        enrichment = (n_in_bin / expected - 1) * 100 if expected > 0 else 0
        results[pos_name] = {
            'position': pos,
            'n_peaks': int(n_in_bin),
            'expected': expected,
            'enrichment_pct': enrichment,
        }
    return results


def structural_score(coords, positions=POS_14):
    """Mean alignment score: mean of min-distance to nearest position."""
    if len(coords) == 0:
        return np.nan
    dists = []
    for u in coords:
        _, d = nearest_position(u, positions)
        dists.append(d)
    return np.mean(dists)


# =========================================================================
# 60 Hz LINE NOISE FILTERING
# =========================================================================

def flag_line_noise(peaks_df, freq_center=60.0, freq_width=2.0, max_bw=0.5):
    """Flag peaks that are likely 60 Hz line noise.

    Criteria: frequency within ±freq_width of 60 Hz AND bandwidth < max_bw Hz.
    """
    is_near_60 = (peaks_df['freq'] >= freq_center - freq_width) & \
                 (peaks_df['freq'] <= freq_center + freq_width)
    is_narrow = peaks_df['bandwidth'] <= max_bw
    return is_near_60 & is_narrow


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description='P20: Extended gamma enrichment analysis')
    parser.add_argument('--input-dir', type=str,
                        default='exports_eegmmidb/p20_no_notch_f07.83',
                        help='Directory with per-subject peak CSVs')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--half-width', type=float, default=0.04,
                        help='Half-width for enrichment bins (default: 0.04)')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir or os.path.join(input_dir, 'p20_analysis')
    os.makedirs(output_dir, exist_ok=True)
    hw = args.half_width

    # --- LOAD ALL PEAKS ---
    files = sorted(glob(os.path.join(input_dir, 'S*_peaks.csv')))
    print(f"Loading peaks from {len(files)} subjects...")

    all_dfs = []
    for f in files:
        df = pd.read_csv(f)
        df['subject'] = os.path.basename(f)[:4]
        all_dfs.append(df)

    all_peaks = pd.concat(all_dfs, ignore_index=True)
    print(f"Total peaks: {len(all_peaks):,}")
    print(f"Frequency range: [{all_peaks.freq.min():.1f}, {all_peaks.freq.max():.1f}] Hz")
    print(f"Phi-octaves: {sorted(all_peaks.phi_octave.unique())}")
    print(f"Positions used: {len(POS_14)} degree-7 positions")

    # --- FLAG AND REMOVE 60 Hz LINE NOISE ---
    noise_mask = flag_line_noise(all_peaks)
    n_noise = noise_mask.sum()
    print(f"\n60 Hz line noise peaks flagged: {n_noise:,} "
          f"({n_noise/len(all_peaks)*100:.1f}%)")

    clean_peaks = all_peaks[~noise_mask].copy()
    print(f"Clean peaks: {len(clean_peaks):,}")

    # --- SEPARATE BANDS ---
    n3_peaks = clean_peaks[clean_peaks.phi_octave == 'n+3'].copy()
    n4_peaks = clean_peaks[clean_peaks.phi_octave == 'n+4'].copy()

    # Additional 60 Hz zone exclusion for n+4 (conservative: drop 58-62 Hz entirely)
    n4_conservative = n4_peaks[(n4_peaks.freq < 58) | (n4_peaks.freq > 62)].copy()

    print(f"\nn+3 band (33.17–53.67 Hz): {len(n3_peaks):,} peaks")
    print(f"n+4 band (53.67–75.00 Hz): {len(n4_peaks):,} peaks (after BW filter)")
    print(f"n+4 conservative (excl 58-62 Hz): {len(n4_conservative):,} peaks")

    # Also get noise peaks for diagnostics
    n4_all = all_peaks[all_peaks.phi_octave == 'n+4'].copy()
    n4_noise = n4_all[flag_line_noise(n4_all)]
    print(f"n+4 60 Hz noise peaks removed: {len(n4_noise):,}")

    # --- COMPUTE LATTICE COORDINATES ---
    n3_peaks['u'] = lattice_coord(n3_peaks['freq'].values)
    n4_peaks['u'] = lattice_coord(n4_peaks['freq'].values)
    n4_conservative['u'] = lattice_coord(n4_conservative['freq'].values)

    # ============================================================
    # ANALYSIS 1: POSITION ENRICHMENT COMPARISON (n+3 vs n+4)
    # ============================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 1: POSITION ENRICHMENT (n+3 vs n+4)")
    print(f"  Half-width = ±{hw} lattice units")
    print("=" * 70)

    enrich_n3 = enrichment_at_positions(n3_peaks['u'].values, half_width=hw)
    enrich_n4 = enrichment_at_positions(n4_peaks['u'].values, half_width=hw)
    enrich_n4c = enrichment_at_positions(n4_conservative['u'].values, half_width=hw)

    print(f"\n  {'Position':<16s} {'Lattice u':>10s} {'n+3 enrich%':>12s} "
          f"{'n+4 enrich%':>12s} {'n+4 cons%':>12s} {'Freq n+4':>10s}")
    print(f"  {'-'*16} {'-'*10} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")

    rows = []
    for pos in POS_14:
        pname = POS_NAMES.get(round(pos, 6), f'{pos:.4f}')
        e3 = enrich_n3.get(pname, {})
        e4 = enrich_n4.get(pname, {})
        e4c = enrich_n4c.get(pname, {})
        # Map position to frequency in n+4 band
        freq_n4 = F0 * PHI**(4 + pos)
        in_notch = " *60Hz*" if 58 <= freq_n4 <= 62 else ""

        print(f"  {pname:<16s} {pos:>10.4f} {e3.get('enrichment_pct', 0):>+11.1f}% "
              f"{e4.get('enrichment_pct', 0):>+11.1f}% "
              f"{e4c.get('enrichment_pct', 0):>+11.1f}% "
              f"{freq_n4:>9.1f} Hz{in_notch}")

        rows.append({
            'position_name': pname,
            'position_u': pos,
            'freq_in_n4': freq_n4,
            'in_60hz_zone': 58 <= freq_n4 <= 62,
            'n3_enrichment_pct': e3.get('enrichment_pct', 0),
            'n3_n_peaks': e3.get('n_peaks', 0),
            'n4_enrichment_pct': e4.get('enrichment_pct', 0),
            'n4_n_peaks': e4.get('n_peaks', 0),
            'n4_conservative_enrichment_pct': e4c.get('enrichment_pct', 0),
            'n4_conservative_n_peaks': e4c.get('n_peaks', 0),
        })

    enrich_df = pd.DataFrame(rows)
    enrich_df.to_csv(os.path.join(output_dir, 'position_enrichment_n3_vs_n4.csv'), index=False)

    # ============================================================
    # ANALYSIS 2: STRUCTURAL SCORE (ALL BANDS)
    # ============================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 2: STRUCTURAL ALIGNMENT SCORE — ALL BANDS")
    print("=" * 70)

    # Get all bands for comparison
    band_data = {}
    for octave in sorted(clean_peaks.phi_octave.unique()):
        bp = clean_peaks[clean_peaks.phi_octave == octave].copy()
        bp['u'] = lattice_coord(bp['freq'].values)
        band_data[octave] = bp

    # Add conservative n+4
    band_data['n+4_cons'] = n4_conservative

    np.random.seed(42)
    n_perm = 1_000

    print(f"\n  {'Band':<25s} {'N peaks':>8s} {'Freq range':>16s} {'SS':>8s} "
          f"{'Null mean':>10s} {'Cohen d':>9s} {'p':>12s}")
    print(f"  {'-'*25} {'-'*8} {'-'*16} {'-'*8} {'-'*10} {'-'*9} {'-'*12}")

    band_results = {}
    for octave in sorted(band_data.keys()):
        bp = band_data[octave]
        if len(bp) < 50:
            continue
        coords = bp['u'].values
        ss = structural_score(coords)
        # Use subsample for null to keep runtime manageable
        n_null_sample = min(len(bp), 5000)
        null_ss = np.array([structural_score(np.random.uniform(0, 1, n_null_sample))
                           for _ in range(n_perm)])
        p = (null_ss <= ss).mean()
        d = (null_ss.mean() - ss) / null_ss.std()
        freq_range = f"[{bp.freq.min():.1f}, {bp.freq.max():.1f}]"

        band_results[octave] = {
            'ss': ss, 'null_mean': float(null_ss.mean()),
            'd': d, 'p': p, 'n': len(bp),
        }

        marker = " <<<" if octave == 'n+4_cons' else ""
        print(f"  {octave:<25s} {len(bp):>8,} {freq_range:>16s} {ss:>8.4f} "
              f"{null_ss.mean():>10.4f} {d:>+9.3f} {p:>12.6f}{marker}")

    # Save for later use
    ss_n3 = band_results.get('n+3', {}).get('ss', np.nan)
    ss_n4c = band_results.get('n+4_cons', {}).get('ss', np.nan)
    d_n3 = band_results.get('n+3', {}).get('d', np.nan)
    d_n4c = band_results.get('n+4_cons', {}).get('d', np.nan)
    p_n3 = band_results.get('n+3', {}).get('p', np.nan)
    p_n4c = band_results.get('n+4_cons', {}).get('p', np.nan)

    # Band results CSV
    br_rows = [{'band': k, **v} for k, v in band_results.items()]
    pd.DataFrame(br_rows).to_csv(os.path.join(output_dir, 'structural_score_all_bands.csv'), index=False)

    # ============================================================
    # ANALYSIS 3: PER-SUBJECT CONSISTENCY
    # ============================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 3: PER-SUBJECT n+4 ALIGNMENT")
    print("=" * 70)

    subj_scores_n3 = []
    subj_scores_n4 = []
    for subj in sorted(n3_peaks.subject.unique()):
        s3 = n3_peaks[n3_peaks.subject == subj]
        s4 = n4_conservative[n4_conservative.subject == subj]
        if len(s3) >= 10:
            subj_scores_n3.append({'subject': subj, 'ss': structural_score(s3['u'].values), 'n': len(s3)})
        if len(s4) >= 10:
            subj_scores_n4.append({'subject': subj, 'ss': structural_score(s4['u'].values), 'n': len(s4)})

    ss3_df = pd.DataFrame(subj_scores_n3)
    ss4_df = pd.DataFrame(subj_scores_n4)

    print(f"\n  Subjects with ≥10 peaks:")
    print(f"    n+3: {len(ss3_df)}/{len(files)}")
    print(f"    n+4 (conservative): {len(ss4_df)}/{len(files)}")

    if len(ss4_df) > 0:
        # Compare per-subject SS to null
        null_expected = band_results.get('n+4_cons', {}).get('null_mean', 0.25)
        t_n4, p_t_n4 = stats.ttest_1samp(ss4_df['ss'], null_expected)
        d_per_subj = (null_expected - ss4_df['ss'].mean()) / ss4_df['ss'].std()

        print(f"\n  n+4 per-subject structural score:")
        print(f"    Mean SS: {ss4_df['ss'].mean():.4f}")
        print(f"    Null expected: {null_expected:.4f}")
        print(f"    t = {t_n4:.3f}, p = {p_t_n4:.2e}")
        print(f"    Cohen's d = {d_per_subj:.3f}")
        print(f"    Subjects with SS < null: {(ss4_df['ss'] < null_expected).sum()}/{len(ss4_df)} "
              f"({(ss4_df['ss'] < null_expected).mean()*100:.0f}%)")

    # ============================================================
    # ANALYSIS 4: 60 Hz SENSITIVITY
    # ============================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 4: 60 Hz EXCLUSION SENSITIVITY")
    print("=" * 70)

    # Try different exclusion zones
    zones = [
        ('No exclusion (BW filter only)', n4_peaks),
        ('Exclude 59-61 Hz', n4_peaks[(n4_peaks.freq < 59) | (n4_peaks.freq > 61)]),
        ('Exclude 58-62 Hz', n4_conservative),
        ('Exclude 57-63 Hz', n4_peaks[(n4_peaks.freq < 57) | (n4_peaks.freq > 63)]),
        ('Exclude 56-64 Hz', n4_peaks[(n4_peaks.freq < 56) | (n4_peaks.freq > 64)]),
    ]

    print(f"\n  {'Zone':<30s} {'N peaks':>8s} {'SS':>8s} {'Cohen d':>9s}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*9}")
    for label, subset in zones:
        if len(subset) == 0:
            continue
        coords = lattice_coord(subset['freq'].values)
        ss = structural_score(coords)
        null_ss = np.array([structural_score(np.random.uniform(0, 1, len(subset)))
                           for _ in range(1000)])
        d = (null_ss.mean() - ss) / null_ss.std()
        print(f"  {label:<30s} {len(subset):>8,} {ss:>8.4f} {d:>+9.3f}")

    # ============================================================
    # ANALYSIS 5: FREQUENCY HISTOGRAM
    # ============================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 5: n+4 PEAK FREQUENCY DISTRIBUTION")
    print("=" * 70)

    # 1 Hz bins
    print(f"\n  {'Bin':>12s} {'Count':>7s} {'Lattice pos near':>20s}")
    print(f"  {'-'*12} {'-'*7} {'-'*20}")
    for lo in range(54, 75):
        hi = lo + 1
        count = ((n4_peaks.freq >= lo) & (n4_peaks.freq < hi)).sum()
        # What lattice position is nearest?
        mid = lo + 0.5
        u = lattice_coord(mid)
        nearest_p, nearest_d = nearest_position(u)
        pname = POS_NAMES.get(round(nearest_p, 6), f'{nearest_p:.3f}')
        close = f"{pname} (d={nearest_d:.3f})" if nearest_d < 0.05 else ""
        bar = '#' * (count // 50)
        notch = " *60Hz*" if 58 <= lo < 62 else ""
        print(f"  [{lo:2d}, {hi:2d}):  {count:5d}  {close:<20s} {bar}{notch}")

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("P20 SUMMARY")
    print("=" * 70)

    print(f"""
  Dataset:     EEGMMIDB (109 subjects, 64 channels, 160 Hz, NO notch filter)
  Band:        n+4 phi-octave (53.67–75.00 Hz)
  Peaks:       {len(n4_conservative):,} (conservative, excluding 58-62 Hz)

  STRUCTURAL ALIGNMENT:
    n+3 (established gamma): SS = {ss_n3:.4f}, d = {d_n3:+.3f}, p = {p_n3:.6f}
    n+4 (high gamma):       SS = {ss_n4c:.4f}, d = {d_n4c:+.3f}, p = {p_n4c:.6f}

  VERDICT: {"ENRICHMENT EXTENDS to n+4" if d_n4c > 0.1 else "NO CLEAR ENRICHMENT in n+4" if d_n4c < -0.1 else "INCONCLUSIVE — effect near zero"}
""")

    # Also extract n+4 (non-conservative) results
    ss_n4 = band_results.get('n+4', {}).get('ss', np.nan)
    d_n4 = band_results.get('n+4', {}).get('d', np.nan)
    p_n4 = band_results.get('n+4', {}).get('p', np.nan)

    # Save all results
    summary = {
        'dataset': 'EEGMMIDB',
        'n_subjects': len(files),
        'n_peaks_n3': len(n3_peaks),
        'n_peaks_n4': len(n4_peaks),
        'n_peaks_n4_conservative': len(n4_conservative),
        'n_60hz_noise_removed': len(n4_noise),
        'ss_n3': ss_n3,
        'ss_n4': ss_n4,
        'ss_n4_conservative': ss_n4c,
        'd_n3': d_n3,
        'd_n4': d_n4,
        'd_n4_conservative': d_n4c,
        'p_n3': p_n3,
        'p_n4': p_n4,
        'p_n4_conservative': p_n4c,
    }
    pd.DataFrame([summary]).to_csv(os.path.join(output_dir, 'p20_summary.csv'), index=False)

    ss3_df.to_csv(os.path.join(output_dir, 'per_subject_ss_n3.csv'), index=False)
    if len(ss4_df) > 0:
        ss4_df.to_csv(os.path.join(output_dir, 'per_subject_ss_n4.csv'), index=False)

    print(f"Results saved to: {output_dir}/")


if __name__ == '__main__':
    main()
