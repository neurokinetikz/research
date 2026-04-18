#!/usr/bin/env python3
"""
SIE Window Enrichment — Follow-up Analyses (6 checks)
======================================================

Addresses reviewer-anticipated concerns about the pooled N=1196 result:

  (1) Effect size × dataset size — is d=0.2 the "true" value or was d=0.3-0.4
      winner's curse from the original N=460?
  (2) Conservation test — does Σ (per-bin ignition_effect) ≈ 0 within each
      band? (conservative flow) or Σ > 0 (flow + source)?
  (3) Beta_low attractor position — does the attractor bin that lights up
      sit at a Schumann harmonic or on the φ-lattice?
  (4) Dose-response — does the ignition effect strengthen with
      events-per-subject? Rules out subject-trait confound.
  (5) EC vs EO ignition signature — does the structural restructuring hold
      across arousal states? Extends the state-independence claim.
  (6) Peak generation vs redistribution — sum of positive deltas vs
      negative deltas within a band.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy import stats

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs'))

BANDS = ['theta', 'alpha', 'beta_low', 'beta_high', 'gamma']
POSITIONS = ['boundary', 'noble_6', 'noble_5', 'noble_4', 'noble_3',
             'inv_noble_1', 'attractor', 'noble_1',
             'inv_noble_3', 'inv_noble_4', 'inv_noble_5', 'inv_noble_6']
SHAPE_METRICS = ['mountain', 'ushape', 'peak_height', 'ramp_depth',
                 'center_depletion', 'asymmetry', 'n_peaks']

PHI = (1 + 5 ** 0.5) / 2
F0 = 7.6
SCHUMANN_HZ = [7.83, 14.3, 20.8, 27.3, 33.8, 39.0]


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


def col(prefix, band, pos_or_shape):
    return f'{prefix}_{band}_{pos_or_shape}'


def ignition_effect(df, prefix_metric_triplet):
    """For one metric, return per-subject ignition − mean(pre, post)."""
    pre_col, ign_col, post_col = prefix_metric_triplet
    if not all(c in df.columns for c in [pre_col, ign_col, post_col]):
        return None
    e = df[ign_col] - (df[pre_col] + df[post_col]) / 2.0
    return e.dropna().values


def subject_effects(df, band, pos_or_shape):
    """Per-subject ignition effect for one (band, pos/shape) metric."""
    pc = col('pre', band, pos_or_shape)
    ic = col('ignition', band, pos_or_shape)
    pc2 = col('post', band, pos_or_shape)
    return ignition_effect(df, (pc, ic, pc2))


# =========================================================================
# (1) EFFECT SIZE vs DATASET SIZE
# =========================================================================

def check_effect_size_vs_n(dfs, top_metrics):
    """For each top metric, compute per-dataset d and regress against N."""
    rows = []
    for m in top_metrics:
        band, *rest = m.split('_', 1)
        if band in ('beta', 'gamma'):
            parts = m.split('_')
            band = f'{parts[0]}_{parts[1]}'
            key = '_'.join(parts[2:])
        else:
            band = m.split('_')[0]
            key = '_'.join(m.split('_')[1:])
        for name, df in dfs.items():
            e = subject_effects(df, band, key)
            if e is None or len(e) < 5:
                continue
            n = len(e)
            sd = e.std(ddof=1)
            d = e.mean() / sd if sd > 0 else 0.0
            mean_events = df['n_events'].mean() if 'n_events' in df.columns else np.nan
            rows.append({
                'metric': m, 'dataset': name, 'N': n, 'd': d,
                'mean_events_per_subject': mean_events,
            })
    out = pd.DataFrame(rows)

    # Per-metric correlation d ~ log(N), d ~ events
    summary = []
    for m in out['metric'].unique():
        sub = out[out['metric'] == m]
        if len(sub) < 5:
            continue
        r_n, p_n = stats.spearmanr(sub['N'], np.abs(sub['d']))
        r_e, p_e = stats.spearmanr(sub['mean_events_per_subject'], np.abs(sub['d']))
        # "Big" datasets: N >= 100
        big = sub[sub['N'] >= 100]
        small = sub[sub['N'] < 50]
        summary.append({
            'metric': m,
            'n_datasets': len(sub),
            'mean_d_big_datasets': big['d'].mean() if len(big) else np.nan,
            'mean_d_small_datasets': small['d'].mean() if len(small) else np.nan,
            'spearman_N_vs_d': r_n, 'p_N_vs_d': p_n,
            'spearman_events_vs_d': r_e, 'p_events_vs_d': p_e,
        })
    return out, pd.DataFrame(summary)


# =========================================================================
# (2) CONSERVATION TEST
# =========================================================================

def conservation_test(dfs, band):
    """Σ ignition_effect across 12 positions in one band, per subject."""
    # For each subject, sum per-bin effects within the band
    per_sub_sums = []
    for name, df in dfs.items():
        sub_sums = np.zeros(len(df))
        for pos in POSITIONS:
            e_col_pre = col('pre', band, pos)
            e_col_ign = col('ignition', band, pos)
            e_col_post = col('post', band, pos)
            if e_col_ign not in df.columns:
                continue
            eff = df[e_col_ign] - (df[e_col_pre] + df[e_col_post]) / 2.0
            sub_sums = sub_sums + eff.fillna(0).values
        # Exclude subjects with all-NaN
        for s in sub_sums:
            if np.isfinite(s):
                per_sub_sums.append({'dataset': name, 'band_sum': s})
    d = pd.DataFrame(per_sub_sums)
    result = {'band': band, 'n_subjects': len(d),
              'mean_sum': d['band_sum'].mean(),
              'median_sum': d['band_sum'].median(),
              'se_sum': d['band_sum'].sem(),
              't': d['band_sum'].mean() / d['band_sum'].sem() if d['band_sum'].sem() > 0 else 0}
    result['p'] = 2 * (1 - stats.t.cdf(abs(result['t']), df=result['n_subjects'] - 1))
    return result


# =========================================================================
# (3) BETA_LOW ATTRACTOR FREQUENCY
# =========================================================================

def beta_low_attractor_freq():
    """Compute the Hz of attractor position inside the beta_low octave.

    The φ-lattice wraps every octave (factor φ). Starting from f0=7.6, octave k
    covers f ∈ [f0·φ^k, f0·φ^(k+1)]. Attractor is at u=0.5 within the octave.
    Find which octave's attractor falls in beta_low (15-22 Hz).
    """
    for k in range(-2, 5):
        u_attractor = k + 0.5
        f = F0 * PHI ** u_attractor
        if 15 <= f <= 22:
            return {'octave_k': k, 'attractor_freq': f,
                    'nearest_schumann': min(SCHUMANN_HZ, key=lambda s: abs(s - f)),
                    'dist_to_nearest_schumann_hz': min(abs(f - s) for s in SCHUMANN_HZ)}
    return None


# =========================================================================
# (4) DOSE-RESPONSE
# =========================================================================

def dose_response(dfs, top_metrics):
    """For each top metric, compute per-subject effect and regress on n_events."""
    rows = []
    for m in top_metrics:
        parts = m.split('_')
        if parts[0] in ('beta', 'gamma'):
            band = f'{parts[0]}_{parts[1]}'
            key = '_'.join(parts[2:])
        else:
            band = parts[0]
            key = '_'.join(parts[1:])
        # Collect all subjects across datasets
        all_rows = []
        for name, df in dfs.items():
            pc = col('pre', band, key); ic = col('ignition', band, key)
            pc2 = col('post', band, key)
            if not all(c in df.columns for c in [pc, ic, pc2, 'n_events']):
                continue
            eff = df[ic] - (df[pc] + df[pc2]) / 2.0
            sub_df = pd.DataFrame({
                'effect': eff, 'n_events': df['n_events'],
                'dataset': name,
            }).dropna()
            all_rows.append(sub_df)
        if not all_rows:
            continue
        pooled = pd.concat(all_rows, ignore_index=True)
        if len(pooled) < 20:
            continue
        # Spearman correlation
        r, p = stats.spearmanr(pooled['n_events'], pooled['effect'])
        # Direction of expected effect: positive if mean effect > 0, else negative
        # Dose-response means |effect| increases with n_events
        r_abs, p_abs = stats.spearmanr(pooled['n_events'], pooled['effect'].abs())
        # Binned means
        bins = pooled.groupby('n_events')['effect'].agg(['mean', 'count']).reset_index()
        rows.append({
            'metric': m, 'n_subjects_all': len(pooled),
            'spearman_signed': r, 'p_signed': p,
            'spearman_abs': r_abs, 'p_abs': p_abs,
            'n_events_bins': bins.to_dict('records'),
        })
    return pd.DataFrame(rows)


# =========================================================================
# (5) EC vs EO IGNITION SIGNATURE
# =========================================================================

EC_EO_PAIRS = [
    ('lemon', 'lemon_EO'),
    ('tdbrain', 'tdbrain_EO'),
    ('dortmund', 'dortmund_EO-pre'),
    ('dortmund_EC-post', 'dortmund_EO-post'),
]


def ec_eo_comparison(dfs):
    """For each EC/EO pair, compute per-metric d and correlate EC vs EO pattern."""
    results = []
    all_metrics = []
    # Enumerate metrics
    for b in BANDS:
        for p in POSITIONS:
            all_metrics.append((b, p))
        for s in SHAPE_METRICS:
            all_metrics.append((b, s))

    for ec_name, eo_name in EC_EO_PAIRS:
        if ec_name not in dfs or eo_name not in dfs:
            continue
        ec_df, eo_df = dfs[ec_name], dfs[eo_name]
        ec_ds, eo_ds = [], []
        for band, key in all_metrics:
            ec_e = subject_effects(ec_df, band, key)
            eo_e = subject_effects(eo_df, band, key)
            if ec_e is None or eo_e is None or len(ec_e) < 5 or len(eo_e) < 5:
                continue
            ec_d = ec_e.mean() / ec_e.std(ddof=1) if ec_e.std(ddof=1) > 0 else 0
            eo_d = eo_e.mean() / eo_e.std(ddof=1) if eo_e.std(ddof=1) > 0 else 0
            ec_ds.append(ec_d)
            eo_ds.append(eo_d)
        ec_ds, eo_ds = np.array(ec_ds), np.array(eo_ds)
        if len(ec_ds) > 10:
            r, p = stats.pearsonr(ec_ds, eo_ds)
            results.append({
                'pair': f'{ec_name} vs {eo_name}',
                'n_metrics': len(ec_ds),
                'corr_ec_eo_d_patterns': r,
                'p': p,
                'sign_agreement': (np.sign(ec_ds) == np.sign(eo_ds)).mean(),
            })
    return pd.DataFrame(results)


# =========================================================================
# (6) PEAK GENERATION vs REDISTRIBUTION
# =========================================================================

def peak_generation_test(dfs):
    """Per band: positive-sum vs negative-sum across positions.

    Pure redistribution: |Σ+| = |Σ-|
    Generation:          |Σ+| > |Σ-|
    """
    rows = []
    for band in BANDS:
        pos_sum_per_sub = []
        neg_sum_per_sub = []
        total_sum_per_sub = []
        for name, df in dfs.items():
            for _, s in df.iterrows():
                pos_tot = 0.0
                neg_tot = 0.0
                tot = 0.0
                valid = True
                for pos in POSITIONS:
                    pc = col('pre', band, pos)
                    ic = col('ignition', band, pos)
                    pc2 = col('post', band, pos)
                    if not all(c in s.index for c in [pc, ic, pc2]):
                        valid = False; break
                    try:
                        eff = s[ic] - (s[pc] + s[pc2]) / 2.0
                    except Exception:
                        valid = False; break
                    if pd.isna(eff):
                        continue
                    tot += eff
                    if eff > 0:
                        pos_tot += eff
                    else:
                        neg_tot += eff
                if valid:
                    pos_sum_per_sub.append(pos_tot)
                    neg_sum_per_sub.append(neg_tot)
                    total_sum_per_sub.append(tot)
        ps = np.array(pos_sum_per_sub); ns = np.array(neg_sum_per_sub); ts = np.array(total_sum_per_sub)
        if len(ps) < 10:
            continue
        # Net (signed total) — if ≠ 0, there's a source term
        t_net, p_net = stats.ttest_1samp(ts, 0)
        rows.append({
            'band': band, 'n_subjects': len(ps),
            'mean_pos_sum': ps.mean(), 'mean_neg_sum': ns.mean(),
            'ratio_pos_to_abs_neg': ps.mean() / abs(ns.mean()) if ns.mean() != 0 else np.inf,
            'mean_net': ts.mean(),
            't_net': t_net, 'p_net': p_net,
        })
    return pd.DataFrame(rows)


# =========================================================================
# MAIN
# =========================================================================

def main():
    print('Loading datasets...')
    dfs = load_all()
    print(f'  {len(dfs)} datasets, total N = {sum(len(d) for d in dfs.values())}')

    top_metrics = [
        'alpha_asymmetry', 'theta_asymmetry', 'alpha_center_depletion',
        'alpha_noble_5', 'theta_inv_noble_5', 'theta_inv_noble_6',
        'theta_ushape', 'theta_mountain', 'alpha_mountain',
        'beta_low_ushape', 'beta_low_attractor', 'n_peaks',
    ]

    print('\n=== (1) Effect size × dataset size ===')
    per_ds, summary_n = check_effect_size_vs_n(dfs, top_metrics)
    per_ds.to_csv(os.path.join(OUT_DIR, 'sie_wenr_followup_d_by_dataset.csv'), index=False)
    summary_n.to_csv(os.path.join(OUT_DIR, 'sie_wenr_followup_d_vs_n.csv'), index=False)
    for _, r in summary_n.iterrows():
        print(f"  {r['metric']:30s}  big-d={r['mean_d_big_datasets']:+.3f}  "
              f"small-d={r['mean_d_small_datasets']:+.3f}  "
              f"rho_N={r['spearman_N_vs_d']:+.2f} (p={r['p_N_vs_d']:.2f})")

    print('\n=== (2) Conservation test (Σ per-bin effect within band) ===')
    cons_rows = [conservation_test(dfs, b) for b in BANDS]
    cons_df = pd.DataFrame(cons_rows)
    cons_df.to_csv(os.path.join(OUT_DIR, 'sie_wenr_followup_conservation.csv'), index=False)
    for _, r in cons_df.iterrows():
        sig = '***' if r['p'] < 0.001 else ('**' if r['p'] < 0.01 else ('*' if r['p'] < 0.05 else ''))
        print(f"  {r['band']:12s}  Σ_mean={r['mean_sum']:+.4f}  t={r['t']:+.2f}  p={r['p']:.2e} {sig}")

    print('\n=== (3) Beta_low attractor frequency ===')
    beta_info = beta_low_attractor_freq()
    print(f"  {beta_info}")

    print('\n=== (4) Dose-response (effect × n_events) ===')
    dose = dose_response(dfs, top_metrics)
    dose_out = dose[['metric', 'n_subjects_all', 'spearman_signed', 'p_signed',
                     'spearman_abs', 'p_abs']].copy()
    dose_out.to_csv(os.path.join(OUT_DIR, 'sie_wenr_followup_dose_response.csv'), index=False)
    for _, r in dose_out.iterrows():
        sig_abs = '***' if r['p_abs'] < 0.001 else ('**' if r['p_abs'] < 0.01 else ('*' if r['p_abs'] < 0.05 else ''))
        print(f"  {r['metric']:30s}  N={r['n_subjects_all']}  rho(|effect|)={r['spearman_abs']:+.3f} p={r['p_abs']:.2e} {sig_abs}")

    print('\n=== (5) EC vs EO ignition signature correlation ===')
    ec_eo = ec_eo_comparison(dfs)
    ec_eo.to_csv(os.path.join(OUT_DIR, 'sie_wenr_followup_ec_eo.csv'), index=False)
    for _, r in ec_eo.iterrows():
        print(f"  {r['pair']:40s}  r_d_pattern={r['corr_ec_eo_d_patterns']:+.3f}  "
              f"p={r['p']:.1e}  sign_agree={r['sign_agreement']:.2f}")

    print('\n=== (6) Peak generation vs redistribution (per band) ===')
    gen = peak_generation_test(dfs)
    gen.to_csv(os.path.join(OUT_DIR, 'sie_wenr_followup_peak_gen.csv'), index=False)
    for _, r in gen.iterrows():
        sig = '***' if r['p_net'] < 0.001 else ('**' if r['p_net'] < 0.01 else ('*' if r['p_net'] < 0.05 else ''))
        print(f"  {r['band']:12s}  Σ+={r['mean_pos_sum']:+.3f}  Σ-={r['mean_neg_sum']:+.3f}  "
              f"net={r['mean_net']:+.4f}  p={r['p_net']:.2e} {sig}")


if __name__ == '__main__':
    main()
