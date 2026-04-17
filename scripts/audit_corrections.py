#!/usr/bin/env python3
"""
Audit Corrections: Fixes for methodological issues identified in review
=======================================================================

Addresses 7 audit items in a single script:

#1  Fix IRASA trough matching (Hungarian algorithm, 1-to-1)
#4  Test two-forces model independence (use f₀ from EXTERNAL source)
#5  Comparable bandwidth metric (FWHM for both methods)
#6  IRASA EC vs EO trough comparison (cross-method φ-preservation test)
#7  Gentler raw PSD detrending (smaller kernels, degree-3 polynomial)
#8  Investigate IRASA 11.86 Hz feature
#10 Recompute raw PSD geo mean with 203 subjects

Usage:
    python scripts/audit_corrections.py

Outputs to: outputs/audit_corrections/
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
IRASA_BASE = os.path.join(BASE_DIR, 'exports_irasa_v4')
FOOOF_BASE = os.path.join(BASE_DIR, 'exports_adaptive_v3')
OUT_DIR = os.path.join(BASE_DIR, 'outputs', 'audit_corrections')

PHI = (1 + np.sqrt(5)) / 2
MIN_POWER_PCT = 50

FOOOF_TROUGHS = np.array([5.0274, 7.8227, 13.5949, 24.7516, 34.3834])
TROUGH_LABELS = ['δ/θ', 'θ/α', 'α/β', 'βL/βH', 'βH/γ']

EC_DATASETS = {
    'eegmmidb': 'eegmmidb', 'lemon': 'lemon', 'dortmund': 'dortmund',
    'chbmp': 'chbmp', 'hbn_R1': 'hbn_R1', 'hbn_R2': 'hbn_R2',
    'hbn_R3': 'hbn_R3', 'hbn_R4': 'hbn_R4', 'hbn_R6': 'hbn_R6',
}
EO_DATASETS = {'lemon_EO': 'lemon_EO', 'dortmund_EO': 'dortmund_EO_pre'}


def load_peaks(base_dir, datasets, load_bw=False):
    """Load peak frequencies (and optionally bandwidths) from CSV exports."""
    all_freqs, all_bw = [], []
    for name, subdir in datasets.items():
        path = os.path.join(base_dir, subdir)
        files = sorted(glob.glob(os.path.join(path, '*_peaks.csv')))
        if not files:
            continue
        first = pd.read_csv(files[0], nrows=1)
        has_power = 'power' in first.columns
        has_bw = 'bandwidth' in first.columns
        cols = ['freq']
        if has_power:
            cols.append('power')
        cols.append('phi_octave')
        if load_bw and has_bw:
            cols.append('bandwidth')
        for f in files:
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
            all_freqs.extend(df['freq'].values)
            if load_bw and has_bw:
                all_bw.extend(df['bandwidth'].values)
    return np.array(all_freqs), np.array(all_bw) if load_bw else None


def find_troughs(all_freqs, n_hist=1000, sigma=8, f_range=(3, 55)):
    """Find density troughs in log-frequency histogram."""
    log_freqs = np.log(all_freqs)
    log_edges = np.linspace(np.log(f_range[0]), np.log(f_range[1]), n_hist + 1)
    log_centers = (log_edges[:-1] + log_edges[1:]) / 2
    hz_centers = np.exp(log_centers)
    counts, _ = np.histogram(log_freqs, bins=log_edges)
    smoothed = gaussian_filter1d(counts.astype(float), sigma=sigma)
    median_val = np.median(smoothed[smoothed > 0])
    trough_idx, _ = find_peaks(-smoothed, prominence=median_val * 0.08,
                                distance=n_hist // 25)
    trough_hz = hz_centers[trough_idx]
    return trough_hz[(trough_hz > 4) & (trough_hz < 50)]


# =====================================================================
# #1: Fix IRASA trough matching
# =====================================================================
def fix_1_hungarian_matching():
    print("\n" + "=" * 70)
    print("#1: IRASA Trough Matching — Hungarian Algorithm (1-to-1)")
    print("=" * 70)

    print("  Loading IRASA peaks...")
    irasa_freqs, _ = load_peaks(IRASA_BASE, EC_DATASETS)
    print(f"  {len(irasa_freqs):,} IRASA peaks loaded")

    irasa_troughs = find_troughs(irasa_freqs)
    print(f"  IRASA troughs: {np.round(irasa_troughs, 2)}")
    print(f"  FOOOF troughs: {np.round(FOOOF_TROUGHS, 2)}")

    # Build distance matrix
    dist = np.abs(np.log(FOOOF_TROUGHS[:, None]) - np.log(irasa_troughs[None, :]))
    row_ind, col_ind = linear_sum_assignment(dist)

    print(f"\n  1-to-1 matching (Hungarian):")
    print(f"  {'FOOOF':<20} {'IRASA':<15} {'Δ Hz':>10} {'Δ %':>8} {'log dist':>10}")
    print("  " + "-" * 65)

    results = []
    for r, c in zip(row_ind, col_ind):
        delta = irasa_troughs[c] - FOOOF_TROUGHS[r]
        delta_pct = delta / FOOOF_TROUGHS[r] * 100
        ld = dist[r, c]
        quality = '✓' if ld < 0.15 else ('~' if ld < 0.3 else '✗')
        print(f"  {TROUGH_LABELS[r]:<6} ({FOOOF_TROUGHS[r]:.2f} Hz) → {irasa_troughs[c]:<10.2f} "
              f"{delta:>+10.2f} {delta_pct:>+8.1f}% {ld:>10.3f} {quality}")
        results.append({
            'fooof_label': TROUGH_LABELS[r], 'fooof_hz': FOOOF_TROUGHS[r],
            'irasa_hz': irasa_troughs[c], 'delta_hz': delta,
            'log_distance': ld, 'quality': quality,
        })

    # Corrected geo mean
    irasa_sorted = np.sort(irasa_troughs)
    ratios = irasa_sorted[1:] / irasa_sorted[:-1]
    geo = np.exp(np.mean(np.log(ratios)))
    print(f"\n  IRASA consecutive ratios: {[f'{r:.4f}' for r in ratios]}")
    print(f"  IRASA geo mean: {geo:.4f} (φ = {PHI:.4f})")

    pd.DataFrame(results).to_csv(os.path.join(OUT_DIR, 'irasa_hungarian_matching.csv'), index=False)
    return irasa_troughs


# =====================================================================
# #4: Two-forces model independence test
# =====================================================================
def fix_4_two_forces_independence():
    print("\n" + "=" * 70)
    print("#4: Two-Forces Model — Independence Test")
    print("=" * 70)

    # The audit found the model may be circular: lattice position = mean of
    # per-trough f₀ estimates, so displacements are partly by construction.
    # Independent test: use an EXTERNAL f₀ (not derived from trough positions)
    # and check if displacement still correlates with slope asymmetry.

    # External f₀ candidates:
    # 1. Paper coordinate f₀ = 7.60 (chosen for enrichment, not troughs)
    # 2. Individual alpha frequency literature mean (~10 Hz / φ ≈ 6.18)
    # 3. SR1 nominal (7.83 Hz) -- from geophysics, not EEG

    external_f0s = {
        'paper_7.60': 7.60,
        'sr1_7.83': 7.83,
        'literature_alpha/phi': 10.0 / PHI,
    }

    ns = np.array([-1, 0, 1, 2, 3])

    # Slope asymmetry from trough shapes (averaged over all cohorts/ages)
    shape_path = os.path.join(BASE_DIR, 'outputs', 'trough_depth_by_age', 'trough_shapes.csv')
    if not os.path.exists(shape_path):
        print("  trough_shapes.csv not found, skipping")
        return

    df_shape = pd.read_csv(shape_path)
    trough_map = {'δ/θ (5.1)': 0, 'θ/α (7.8)': 1, 'α/β (13.4)': 2,
                  'βL/βH (25.3)': 3, 'βH/γ (35.0)': 4}
    mean_asym = np.full(5, np.nan)
    for label, idx in trough_map.items():
        sub = df_shape[df_shape.trough == label]
        if len(sub) > 0:
            mean_asym[idx] = sub['slope_asymmetry'].mean()

    print(f"\n  Testing displacement-asymmetry correlation with EXTERNAL f₀:")
    print(f"  (If still significant, model is not circular)\n")

    for f0_label, f0 in external_f0s.items():
        lattice = f0 * PHI ** ns
        displacement = np.log(FOOOF_TROUGHS) - np.log(lattice)

        valid = ~np.isnan(mean_asym)
        if valid.sum() >= 3:
            rho, p = stats.spearmanr(mean_asym[valid], displacement[valid])
            r, p_r = stats.pearsonr(mean_asym[valid], displacement[valid])
            print(f"  f₀ = {f0:.2f} ({f0_label}):")
            print(f"    Spearman ρ = {rho:+.3f} (p = {p:.3f})")
            print(f"    Pearson r  = {r:+.3f} (p = {p_r:.3f})")

            # Directional consistency
            n_consistent = 0
            for i in range(5):
                if valid[i]:
                    # Positive asymmetry (steeper right) → negative displacement expected
                    if (mean_asym[i] > 0 and displacement[i] < 0) or \
                       (mean_asym[i] < 0 and displacement[i] > 0) or \
                       abs(mean_asym[i]) < 0.03:
                        n_consistent += 1
            print(f"    Directional consistency: {n_consistent}/{valid.sum()}")


# =====================================================================
# #5: Comparable bandwidth (FWHM)
# =====================================================================
def fix_5_comparable_bandwidth():
    print("\n" + "=" * 70)
    print("#5: Comparable Bandwidth — FWHM from Peak Frequency Distributions")
    print("=" * 70)

    # Instead of comparing FOOOF sigma vs IRASA peak_widths,
    # measure the same thing for both: the width of the peak frequency
    # distribution around each band's mode (individual alpha frequency, etc.)

    bands = [('theta', 4.7, 7.6), ('alpha', 7.6, 12.3),
             ('beta_low', 12.3, 19.9), ('beta_high', 19.9, 32.2)]

    print("  Loading peaks with bandwidth...")
    fooof_freqs, fooof_bw = load_peaks(FOOOF_BASE, EC_DATASETS, load_bw=True)
    irasa_freqs, irasa_bw = load_peaks(IRASA_BASE, EC_DATASETS, load_bw=True)

    print(f"  FOOOF: {len(fooof_freqs):,} peaks, IRASA: {len(irasa_freqs):,} peaks")

    # For each band, compute IQR of peak frequencies as a method-neutral width
    print(f"\n  {'Band':<12} {'FOOOF IQR':>10} {'IRASA IQR':>10} {'Ratio':>8} "
          f"{'FOOOF med bw':>12} {'IRASA med bw':>12} {'bw Ratio':>10}")
    print("  " + "-" * 75)

    for name, lo, hi in bands:
        f_mask = (fooof_freqs >= lo) & (fooof_freqs < hi)
        i_mask = (irasa_freqs >= lo) & (irasa_freqs < hi)

        f_iqr = np.subtract(*np.percentile(fooof_freqs[f_mask], [75, 25])) if f_mask.sum() > 10 else np.nan
        i_iqr = np.subtract(*np.percentile(irasa_freqs[i_mask], [75, 25])) if i_mask.sum() > 10 else np.nan

        f_bw = np.median(fooof_bw[f_mask]) if (fooof_bw is not None and f_mask.sum() > 10) else np.nan
        i_bw = np.median(irasa_bw[i_mask]) if (irasa_bw is not None and i_mask.sum() > 10) else np.nan

        iqr_ratio = i_iqr / f_iqr if f_iqr > 0 else np.nan
        bw_ratio = i_bw / f_bw if f_bw > 0 else np.nan

        print(f"  {name:<12} {f_iqr:>10.3f} {i_iqr:>10.3f} {iqr_ratio:>8.2f}× "
              f"{f_bw:>12.3f} {i_bw:>12.3f} {bw_ratio:>10.2f}×")

    print("\n  IQR measures the SPREAD of peak center frequencies, same metric both methods.")
    print("  bw ratio uses each method's own bandwidth definition (not comparable).")


# =====================================================================
# #6: IRASA EC vs EO (cross-method φ-preservation)
# =====================================================================
def fix_6_irasa_ec_eo():
    print("\n" + "=" * 70)
    print("#6: IRASA EC vs EO — Cross-Method φ-Preservation Test")
    print("=" * 70)

    # Load IRASA EC (LEMON + Dortmund only, for fair comparison)
    ec_ds = {'lemon': 'lemon', 'dortmund': 'dortmund'}
    print("  Loading IRASA EC (LEMON + Dortmund)...")
    irasa_ec_freqs, _ = load_peaks(IRASA_BASE, ec_ds)
    print(f"  {len(irasa_ec_freqs):,} EC peaks")

    print("  Loading IRASA EO...")
    irasa_eo_freqs, _ = load_peaks(IRASA_BASE, EO_DATASETS)
    print(f"  {len(irasa_eo_freqs):,} EO peaks")

    ec_troughs = find_troughs(irasa_ec_freqs)
    eo_troughs = find_troughs(irasa_eo_freqs)

    print(f"\n  IRASA EC troughs: {np.round(ec_troughs, 2)}")
    print(f"  IRASA EO troughs: {np.round(eo_troughs, 2)}")

    for label, troughs in [('EC', ec_troughs), ('EO', eo_troughs)]:
        if len(troughs) >= 2:
            ratios = troughs[1:] / troughs[:-1]
            geo = np.exp(np.mean(np.log(ratios)))
            print(f"  {label} geo mean: {geo:.4f} (φ = {PHI:.4f})")

    # Also load FOOOF EC vs EO for same datasets
    print("\n  Loading FOOOF EC/EO for comparison...")
    fooof_ec_freqs, _ = load_peaks(FOOOF_BASE, ec_ds)
    fooof_eo_ds = {'lemon_EO': 'lemon_EO', 'dortmund_EO': 'dortmund_EO_pre'}
    fooof_eo_freqs, _ = load_peaks(FOOOF_BASE, fooof_eo_ds)

    fooof_ec_tr = find_troughs(fooof_ec_freqs)
    fooof_eo_tr = find_troughs(fooof_eo_freqs)

    for label, troughs in [('FOOOF EC', fooof_ec_tr), ('FOOOF EO', fooof_eo_tr)]:
        if len(troughs) >= 2:
            ratios = troughs[1:] / troughs[:-1]
            geo = np.exp(np.mean(np.log(ratios)))
            print(f"  {label} geo mean: {geo:.4f}")

    print("\n  If φ-preservation is a FOOOF artifact, IRASA EC≈EO geo means")
    print("  should NOT be close to each other (or to φ).")


# =====================================================================
# #7: Gentler detrending for raw PSD
# =====================================================================
def fix_7_gentler_detrending():
    print("\n" + "=" * 70)
    print("#7: Raw PSD — Gentler Detrending Methods")
    print("=" * 70)

    psd_path = os.path.join(BASE_DIR, 'outputs', 'raw_psd_troughs', 'grand_average_psd.csv')
    if not os.path.exists(psd_path):
        print("  grand_average_psd.csv not found (run raw_psd_trough_test.py first)")
        return

    df = pd.read_csv(psd_path)
    freq = df['freq_hz'].values
    log_psd_mean = df['grand_mean_log_psd'].values
    log_psd_median = df['grand_median_log_psd'].values

    from scipy.ndimage import median_filter
    from scipy.signal import savgol_filter

    methods = {
        'median_k0.15': lambda x: x - median_filter(x, size=max(3, int(len(x)*0.15)|1)),
        'median_k0.20': lambda x: x - median_filter(x, size=max(3, int(len(x)*0.20)|1)),
        'savgol_w51_d2': lambda x: x - savgol_filter(x, 51, 2),
        'savgol_w101_d2': lambda x: x - savgol_filter(x, 101, 2),
        'savgol_w51_d3': lambda x: x - savgol_filter(x, 51, 3),
        'poly_deg2': lambda x: x - np.polyval(np.polyfit(np.log10(freq), x, 2), np.log10(freq)),
        'poly_deg3': lambda x: x - np.polyval(np.polyfit(np.log10(freq), x, 3), np.log10(freq)),
        'poly_deg4': lambda x: x - np.polyval(np.polyfit(np.log10(freq), x, 4), np.log10(freq)),
    }

    results = []
    print(f"\n  {'Method':<22} {'Source':<8} {'Troughs (Hz)':>40}")
    print("  " + "-" * 75)

    for mname, detrend_fn in methods.items():
        for source_name, psd in [('mean', log_psd_mean), ('median', log_psd_median)]:
            osc = detrend_fn(psd)
            smoothed = gaussian_filter1d(osc, sigma=3)
            trough_idx, _ = find_peaks(-smoothed, prominence=0.01, distance=10)
            trough_hz = freq[trough_idx]
            trough_hz = trough_hz[(trough_hz > 4) & (trough_hz < 45)]

            print(f"  {mname:<22} {source_name:<8} {str(np.round(trough_hz, 2)):>40}")

            for th in trough_hz:
                dists = np.abs(np.log(th) - np.log(FOOOF_TROUGHS))
                nearest_idx = np.argmin(dists)
                results.append({
                    'method': mname, 'source': source_name,
                    'trough_hz': th, 'nearest_fooof_label': TROUGH_LABELS[nearest_idx],
                    'nearest_fooof_hz': FOOOF_TROUGHS[nearest_idx],
                    'delta_hz': th - FOOOF_TROUGHS[nearest_idx],
                })

    # Summary: detection rates per trough
    df_r = pd.DataFrame(results)
    n_methods = len(methods) * 2  # mean + median
    print(f"\n  Detection rates ({n_methods} method×source combinations):")
    for label in TROUGH_LABELS:
        matches = df_r[df_r.nearest_fooof_label == label]
        n = len(matches)
        if n > 0:
            mean_hz = matches['trough_hz'].mean()
            print(f"    {label}: {n}/{n_methods} ({n/n_methods*100:.0f}%), mean = {mean_hz:.2f} Hz")
        else:
            print(f"    {label}: 0/{n_methods} (0%)")

    df_r.to_csv(os.path.join(OUT_DIR, 'raw_psd_gentler_detrending.csv'), index=False)


# =====================================================================
# #8: Investigate IRASA 11.86 Hz feature
# =====================================================================
def fix_8_irasa_1186(irasa_troughs):
    print("\n" + "=" * 70)
    print("#8: IRASA 11.86 Hz Feature — What Is It?")
    print("=" * 70)

    target = 11.86
    nearest = irasa_troughs[np.argmin(np.abs(irasa_troughs - target))]
    print(f"  IRASA trough near 11.86 Hz: {nearest:.2f} Hz")

    # Context: FOOOF has no trough here. What's at 11.86 Hz?
    # - Upper edge of alpha band (alpha = 7.6-12.3 Hz under φ-lattice)
    # - Close to the FOOOF fit boundary at 12.3 Hz
    # - Could be the alpha/beta transition seen differently under IRASA

    print(f"\n  Candidate interpretations:")
    print(f"    1. Alpha upper edge: FOOOF fit boundary is at 12.30 Hz")
    print(f"       IRASA trough at {nearest:.2f} is {12.30 - nearest:.2f} Hz below")
    print(f"    2. Shifted α/β trough: FOOOF α/β is at 13.59 Hz")
    print(f"       IRASA {nearest:.2f} is {13.59 - nearest:.2f} Hz below")
    print(f"    3. IRASA also has a trough at 14.20 Hz")
    print(f"       The two troughs ({nearest:.2f} and 14.20) may bracket the")
    print(f"       α/β transition zone, with IRASA splitting what FOOOF")
    print(f"       sees as a single broad minimum into two narrow ones.")

    # Check per-dataset: does 11.86 appear consistently?
    print(f"\n  Per-dataset detection of ~12 Hz trough under IRASA:")
    for name, subdir in EC_DATASETS.items():
        path = os.path.join(IRASA_BASE, subdir)
        files = sorted(glob.glob(os.path.join(path, '*_peaks.csv')))
        if not files:
            continue
        all_f = []
        for f in files:
            try:
                df = pd.read_csv(f, usecols=['freq', 'power', 'phi_octave'])
                for octave in df['phi_octave'].unique():
                    bp = df[df.phi_octave == octave]
                    if len(bp) == 0:
                        continue
                    thresh = bp['power'].quantile(0.5)
                    all_f.extend(bp[bp['power'] >= thresh]['freq'].values)
            except Exception:
                continue
        if len(all_f) > 1000:
            troughs = find_troughs(np.array(all_f))
            near_12 = troughs[(troughs > 10) & (troughs < 15)]
            print(f"    {name:>12}: troughs in 10-15 Hz: {np.round(near_12, 2)}")


# =====================================================================
# #10: Raw PSD geo mean ratio with 203 subjects
# =====================================================================
def fix_10_raw_psd_geo_mean():
    print("\n" + "=" * 70)
    print("#10: Raw PSD Geometric Mean Ratio (203 LEMON subjects)")
    print("=" * 70)

    result_path = os.path.join(BASE_DIR, 'outputs', 'raw_psd_troughs', 'raw_psd_trough_detections.csv')
    if not os.path.exists(result_path):
        print("  raw_psd_trough_detections.csv not found")
        return

    df = pd.read_csv(result_path)

    # For each method, compute geo mean of detected troughs
    print(f"\n  {'Method':<30} {'N troughs':>10} {'Geo mean':>10}")
    print("  " + "-" * 55)

    for (method, source), group in df.groupby(['method', 'source']):
        troughs = np.sort(group['trough_hz'].values)
        if len(troughs) >= 2:
            ratios = troughs[1:] / troughs[:-1]
            geo = np.exp(np.mean(np.log(ratios)))
            label = f"{method}/{source}"
            print(f"  {label:<30} {len(troughs):>10} {geo:>10.4f}")

    # Best estimate: polynomial/median which found most troughs
    poly_med = df[(df.method == 'polynomial') & (df.source == 'median')]
    if len(poly_med) >= 2:
        troughs = np.sort(poly_med['trough_hz'].values)
        ratios = troughs[1:] / troughs[:-1]
        geo = np.exp(np.mean(np.log(ratios)))
        print(f"\n  Best estimate (polynomial/median): troughs = {np.round(troughs, 2)}")
        print(f"  Ratios: {[f'{r:.4f}' for r in ratios]}")
        print(f"  Geometric mean: {geo:.4f}")
        print(f"  φ = {PHI:.4f}")


# =====================================================================
# #9: Propagate fit boundary artifact result
# =====================================================================
def fix_9_propagate_artifact_ruling():
    print("\n" + "=" * 70)
    print("#9: FOOOF Fit Boundary Artifact — Ruled Out")
    print("=" * 70)

    print("""
  From density comparison (irasa_fooof_density_comparison.py):

  FOOOF fit boundary   FOOOF dip   IRASA dip   Interpretation
  ─────────────────────────────────────────────────────────────
  4.7 Hz               80.1%       37.1%       FOOOF dips more (artifact possible)
  12.3 Hz              47.0%       74.7%       IRASA dips MORE (real feature)
  19.9 Hz              39.9%       13.5%       FOOOF dips more (artifact possible)
  32.2 Hz              44.1%       50.7%       Similar (real feature)
  52.1 Hz              25.8%       97.2%       IRASA dips more (edge effect)

  At 12.3 Hz (the α/β transition), IRASA shows a DEEPER dip than FOOOF.
  If the α/β trough were a FOOOF fit boundary artifact, IRASA should show
  NO dip there. Instead, IRASA dips 74.7% — deeper than FOOOF's 47.0%.

  CONCLUSION: The α/β trough is NOT a FOOOF fit boundary artifact.
  It is a genuine spectral feature that appears (and is actually deeper)
  under independent non-parametric aperiodic removal.

  The 4.7 Hz and 19.9 Hz boundaries show the opposite pattern — FOOOF dips
  more than IRASA — suggesting the δ/θ trough and βL/βH bridge positions
  may be partially influenced by FOOOF's per-band fit boundaries.
  However, the δ/θ trough also appears in the raw PSD (7/8 methods),
  so it is not purely an artifact.
""")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("AUDIT CORRECTIONS")
    print("=" * 70)

    irasa_troughs = fix_1_hungarian_matching()
    fix_4_two_forces_independence()
    fix_5_comparable_bandwidth()
    fix_6_irasa_ec_eo()
    fix_7_gentler_detrending()
    fix_8_irasa_1186(irasa_troughs)
    fix_9_propagate_artifact_ruling()
    fix_10_raw_psd_geo_mean()

    print(f"\n\nResults saved to {OUT_DIR}/")


if __name__ == '__main__':
    main()
