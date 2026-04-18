#!/usr/bin/env python3
"""
SIE Window Enrichment — Follow-up Analyses V3
==============================================

  (D) Trait-baseline check for alpha decrease with n_events. Correlate
      baseline alpha_asymmetry (and top alpha metrics) with n_events per
      subject, pooled across datasets. If baseline alpha is already more
      asymmetric in many-event subjects, that explains the decreased
      ignition delta (less room to move).

  (E) Partialed conservation test. Regress per-band Σ ignition effect on
      per-subject total n_peaks ignition effect; test whether residual Σ
      is still significantly nonzero. Closes the "aggregate SNR artifact"
      door on the cross-band transfer.
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy import stats

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs'))

BANDS = ['theta', 'alpha', 'beta_low', 'beta_high', 'gamma']
POSITIONS = ['boundary', 'noble_6', 'noble_5', 'noble_4', 'noble_3',
             'inv_noble_1', 'attractor', 'noble_1',
             'inv_noble_3', 'inv_noble_4', 'inv_noble_5', 'inv_noble_6']


def load_all():
    files = sorted(glob.glob(os.path.join(OUT_DIR, 'sie_window_enrichment_*.csv')))
    dfs = {}
    for f in files:
        base = os.path.basename(f).replace('sie_window_enrichment_', '').replace('.csv', '')
        if base in ('stats', 'pooled_summary', 'pooled_replication'):
            continue
        df = pd.read_csv(f)
        if 'status' in df.columns:
            df = df[df['status'] == 'ok'].copy()
        df['dataset'] = base
        dfs[base] = df
    return dfs


def col(prefix, band, x):
    return f'{prefix}_{band}_{x}'


# =========================================================================
# (D) TRAIT-BASELINE CHECK
# =========================================================================

def trait_baseline_check(dfs):
    """Correlate |pre_band_metric| with n_events per subject (pooled)."""
    metrics = [
        ('alpha', 'asymmetry'), ('alpha', 'noble_5'), ('alpha', 'mountain'),
        ('alpha', 'center_depletion'),
        ('theta', 'asymmetry'), ('theta', 'inv_noble_5'),
        ('beta_low', 'attractor'),
    ]
    rows = []
    for band, key in metrics:
        all_baseline = []
        all_n_events = []
        for name, df in dfs.items():
            pc = col('pre', band, key)
            if pc not in df.columns or 'n_events' not in df.columns:
                continue
            for b, n in zip(df[pc].values, df['n_events'].values):
                if pd.notna(b) and pd.notna(n):
                    all_baseline.append(b); all_n_events.append(n)
        baseline = np.array(all_baseline); n_ev = np.array(all_n_events)
        if len(baseline) < 50:
            continue
        # Raw correlation (signed)
        r_raw, p_raw = stats.spearmanr(n_ev, baseline)
        # Absolute-magnitude correlation (baseline more "extreme" with more events?)
        r_abs, p_abs = stats.spearmanr(n_ev, np.abs(baseline))
        rows.append({
            'metric': f'{band}_{key}', 'n': len(baseline),
            'rho_signed': r_raw, 'p_signed': p_raw,
            'rho_abs': r_abs, 'p_abs': p_abs,
            'mean_baseline_all': baseline.mean(),
            'mean_abs_baseline_all': np.abs(baseline).mean(),
        })
    return pd.DataFrame(rows)


# =========================================================================
# (E) PARTIALED CONSERVATION TEST
# =========================================================================

def band_sum_effect(df, band):
    """Sum of ignition effects across 12 positions in one band, per subject."""
    s = np.zeros(len(df))
    valid = np.ones(len(df), dtype=bool)
    for pos in POSITIONS:
        pc = col('pre', band, pos); ic = col('ignition', band, pos); poc = col('post', band, pos)
        if not all(c in df.columns for c in [pc, ic, poc]):
            continue
        eff = (df[ic] - (df[pc] + df[poc]) / 2.0).values
        s = s + np.nan_to_num(eff, nan=0.0)
        valid = valid & np.isfinite(eff)
    return s, valid


def n_peaks_total_effect(df):
    """Sum of n_peaks ignition effects across bands, per subject (aggregate detector ↑)."""
    s = np.zeros(len(df))
    for b in BANDS:
        pc = col('pre', b, 'n_peaks'); ic = col('ignition', b, 'n_peaks'); poc = col('post', b, 'n_peaks')
        if not all(c in df.columns for c in [pc, ic, poc]):
            continue
        eff = (df[ic] - (df[pc] + df[poc]) / 2.0).values
        s = s + np.nan_to_num(eff, nan=0.0)
    return s


def partial_conservation(dfs):
    """For each band, ask: is Σ independent of n_peaks_effect?

    Two checks:
    1. Intercept of Σ ~ n_peaks regression (with its SE) — if intercept is
       significantly nonzero and slope is near zero, Σ is independent of
       n_peaks and the effect stands on its own.
    2. Subgroup check: for subjects with |Δn_peaks| < 1 (essentially no
       aggregate detector change), does Σ still differ from zero?
    """
    rows = []
    all_rows = []
    for name, df in dfs.items():
        np_eff = n_peaks_total_effect(df)
        for b in BANDS:
            band_s, valid = band_sum_effect(df, b)
            for i in range(len(df)):
                if valid[i]:
                    all_rows.append({'dataset': name, 'band': b, 'band_sum': band_s[i],
                                     'n_peaks_effect': np_eff[i]})
    all_df = pd.DataFrame(all_rows)

    for b in BANDS:
        sub = all_df[all_df['band'] == b].dropna()
        if len(sub) < 50:
            continue
        x = sub['n_peaks_effect'].values
        y = sub['band_sum'].values
        # Raw test (unconditional)
        t_raw, p_raw = stats.ttest_1samp(y, 0)
        # Correlation between Σ and n_peaks
        r_sn, p_sn = stats.spearmanr(x, y)
        # Linear regression intercept + slope, with SE on intercept
        if np.std(x) > 0:
            slope, intercept, r_val, p_slope, se_slope = stats.linregress(x, y)
            # Intercept SE: need to compute manually
            n = len(x); sx2 = np.sum(x ** 2); sx = np.sum(x)
            x_bar = x.mean()
            y_pred = intercept + slope * x
            resid = y - y_pred
            mse = np.sum(resid ** 2) / (n - 2)
            se_intercept = np.sqrt(mse * sx2 / (n * np.sum((x - x_bar) ** 2)))
            t_intercept = intercept / se_intercept if se_intercept > 0 else 0
            p_intercept = 2 * (1 - stats.t.cdf(abs(t_intercept), df=n - 2))
        else:
            slope = 0; intercept = y.mean(); p_slope = 1
            se_intercept = np.std(y) / np.sqrt(len(y))
            t_intercept = intercept / se_intercept if se_intercept > 0 else 0
            p_intercept = 2 * (1 - stats.norm.cdf(abs(t_intercept)))
        # Subgroup: |Δn_peaks| < 1 (essentially no detector change)
        subgroup = np.abs(x) < 1.0
        if subgroup.sum() >= 20:
            y_sub = y[subgroup]
            t_sub, p_sub = stats.ttest_1samp(y_sub, 0)
            mean_sub = y_sub.mean()
            n_sub = len(y_sub)
        else:
            mean_sub = np.nan; t_sub = np.nan; p_sub = np.nan; n_sub = 0
        rows.append({
            'band': b, 'n_subjects': len(sub),
            'raw_mean_sum': y.mean(), 'raw_p': p_raw,
            'spearman_sigma_npeaks': r_sn, 'p_spearman': p_sn,
            'slope_on_n_peaks': slope, 'p_slope': p_slope,
            'intercept': intercept, 'p_intercept': p_intercept,
            'subgroup_n': n_sub, 'subgroup_mean': mean_sub, 'subgroup_p': p_sub,
        })
    return pd.DataFrame(rows)


# =========================================================================
# MAIN
# =========================================================================

def main():
    print('Loading...')
    dfs = load_all()
    print(f'  {len(dfs)} datasets, total N = {sum(len(d) for d in dfs.values())}')

    print('\n=== (D) Trait-baseline check ===')
    print('    (does baseline become more extreme / different with more events?)')
    baseline = trait_baseline_check(dfs)
    baseline.to_csv(os.path.join(OUT_DIR, 'sie_wenr_followup_v3_trait_baseline.csv'), index=False)
    for _, r in baseline.iterrows():
        sig_abs = '**' if r['p_abs'] < 0.01 else ('*' if r['p_abs'] < 0.05 else '')
        print(f"  {r['metric']:30s}  N={r['n']:4d}  "
              f"rho(n_ev, baseline)={r['rho_signed']:+.3f} p={r['p_signed']:.2e}  "
              f"rho(n_ev, |baseline|)={r['rho_abs']:+.3f} p={r['p_abs']:.2e} {sig_abs}")

    print('\n=== (E) Partialed conservation test ===')
    part = partial_conservation(dfs)
    part.to_csv(os.path.join(OUT_DIR, 'sie_wenr_followup_v3_partialed_conservation.csv'),
                index=False)
    for _, r in part.iterrows():
        print(f"\n  {r['band']}:")
        print(f"    Raw Σ = {r['raw_mean_sum']:+.3f}  (p={r['raw_p']:.2e})")
        print(f"    Σ ↔ n_peaks correlation: ρ={r['spearman_sigma_npeaks']:+.3f}  p={r['p_spearman']:.2e}")
        print(f"    Regression Σ ~ n_peaks:  intercept={r['intercept']:+.3f} (p={r['p_intercept']:.2e})  "
              f"slope={r['slope_on_n_peaks']:+.4f} (p={r['p_slope']:.2e})")
        if pd.notna(r['subgroup_mean']):
            print(f"    Subgroup |Δn_peaks|<1 (N={r['subgroup_n']}): Σ={r['subgroup_mean']:+.3f}  "
                  f"p={r['subgroup_p']:.2e}")


if __name__ == '__main__':
    main()
