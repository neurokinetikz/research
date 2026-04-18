#!/usr/bin/env python3
"""
SIE Window Enrichment — Pre-Submission Checks (3 checks)
=========================================================

Three pre-submission checks that close the remaining reviewer-attack surfaces:

  (1) n_events × recording duration confound. Correlate n_events with
      recording duration per subject. If n_events ∝ duration, the
      dose-response analysis is partly about "longer recording = more
      averaging" rather than "more SIEs = more signal."

  (2) Within-subject theta-alpha dissociation. Per-subject ignition theta
      effect (mean |effect| across top theta metrics) vs per-subject alpha
      effect (same across top alpha metrics). Correlate. Low r → two
      dissociable processes; high r → one output with multiple dimensions.

  (3) Inter-event interval × alpha regression-to-mean. Compute median IEI
      per subject from events CSV. Stratify alpha_asymmetry ignition
      effect by IEI (bunched vs spread). If the alpha decrease-with-events
      disappears in widely-spaced-event subjects, regression-to-mean is the
      explanation.
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy import stats

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs'))
EXPORTS_SIE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'exports_sie'))


def load_window_enrichment():
    """Load all window enrichment CSVs with dataset tag."""
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


def subject_effect(df, band, key):
    pc = col('pre', band, key); ic = col('ignition', band, key); poc = col('post', band, key)
    if not all(c in df.columns for c in [pc, ic, poc]):
        return None
    return (df[ic] - (df[pc] + df[poc]) / 2.0).values


# =========================================================================
# DATASET DIRECTORY MAP (window enrichment file name -> exports_sie dir)
# =========================================================================
DATASET_DIR_MAP = {
    'chbmp': 'chbmp',
    'eegmmidb': 'eegmmidb',
    'hbn_R1': 'hbn_R1', 'hbn_R2': 'hbn_R2', 'hbn_R3': 'hbn_R3',
    'hbn_R4': 'hbn_R4', 'hbn_R6': 'hbn_R6',
    'lemon': 'lemon',
    'lemon_EO': 'lemon_EO',
    'tdbrain': 'tdbrain',
    'tdbrain_EO': 'tdbrain_EO',
    'dortmund': 'dortmund',
    'dortmund_EO-pre': 'dortmund_EO_pre',
    'dortmund_EC-post': 'dortmund_EC_post',
    'dortmund_EO-post': 'dortmund_EO_post',
}


# =========================================================================
# (1) n_events × recording duration
# =========================================================================

def load_extraction_summary(dataset_key):
    """Load extraction_summary.csv for a dataset."""
    dir_name = DATASET_DIR_MAP.get(dataset_key)
    if not dir_name:
        return None
    path = os.path.join(EXPORTS_SIE, dir_name, 'extraction_summary.csv')
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def check_n_events_vs_duration(wenr_dfs):
    """Per dataset, correlate n_events with recording duration_sec."""
    rows = []
    for name, wdf in wenr_dfs.items():
        sdf = load_extraction_summary(name)
        if sdf is None:
            continue
        # Merge on subject_id
        merged = wdf[['subject_id', 'n_events']].merge(
            sdf[['subject_id', 'duration_sec']], on='subject_id', how='inner'
        )
        merged = merged.dropna()
        if len(merged) < 10:
            continue
        r, p = stats.spearmanr(merged['duration_sec'], merged['n_events'])
        rows.append({
            'dataset': name, 'n': len(merged),
            'spearman_dur_events': r, 'p': p,
            'median_duration_sec': merged['duration_sec'].median(),
            'median_n_events': merged['n_events'].median(),
            'events_per_100s_median': (merged['n_events'] / merged['duration_sec'] * 100).median(),
        })
    df_out = pd.DataFrame(rows)

    # Pooled across datasets
    all_dur = []
    all_nev = []
    for name, wdf in wenr_dfs.items():
        sdf = load_extraction_summary(name)
        if sdf is None:
            continue
        merged = wdf[['subject_id', 'n_events']].merge(
            sdf[['subject_id', 'duration_sec']], on='subject_id', how='inner'
        ).dropna()
        all_dur.extend(merged['duration_sec'].tolist())
        all_nev.extend(merged['n_events'].tolist())
    if len(all_dur) > 20:
        r_pool, p_pool = stats.spearmanr(all_dur, all_nev)
    else:
        r_pool, p_pool = np.nan, np.nan
    return df_out, {'n_pooled': len(all_dur), 'rho_pooled': r_pool, 'p_pooled': p_pool}


# =========================================================================
# (2) Within-subject theta-alpha dissociation
# =========================================================================

TOP_THETA_METRICS = [
    ('theta', 'asymmetry'),    # +
    ('theta', 'inv_noble_5'),  # +
    ('theta', 'inv_noble_6'),  # +
    ('theta', 'ushape'),       # +
]
TOP_ALPHA_METRICS = [
    ('alpha', 'asymmetry'),          # -
    ('alpha', 'center_depletion'),   # -
    ('alpha', 'noble_5'),            # +
    ('alpha', 'inv_noble_3'),        # -
]


def _sign_for_expected_direction(band, key):
    """Return +1 if the FDR-surviving direction is positive, -1 if negative."""
    if (band, key) in [('alpha', 'asymmetry'), ('alpha', 'center_depletion'),
                       ('alpha', 'inv_noble_3'), ('alpha', 'inv_noble_5'),
                       ('alpha', 'mountain'), ('theta', 'mountain'),
                       ('beta_low', 'ushape'), ('theta', 'noble_1'),
                       ('theta', 'inv_noble_3'), ('alpha', 'ramp_depth')]:
        return -1.0
    return 1.0


def signed_composite_effect(df, metrics):
    """Per subject: signed composite of ignition effects, with each metric
    multiplied by its expected-direction sign so that a unified 'bigger =
    more ignition response' scalar is produced."""
    components = []
    for band, key in metrics:
        e = subject_effect(df, band, key)
        if e is None:
            continue
        sign = _sign_for_expected_direction(band, key)
        # z-score within dataset to put on common scale
        e_arr = np.array(e)
        if np.std(e_arr[np.isfinite(e_arr)]) == 0:
            continue
        e_z = (e_arr - np.nanmean(e_arr)) / np.nanstd(e_arr)
        components.append(sign * e_z)
    if not components:
        return None
    # Mean z-score across metrics per subject
    mat = np.array(components)
    return np.nanmean(mat, axis=0)


def check_theta_alpha_dissociation(wenr_dfs):
    """Per subject: correlate theta-composite vs alpha-composite."""
    all_theta = []
    all_alpha = []
    per_ds = []
    for name, df in wenr_dfs.items():
        theta_comp = signed_composite_effect(df, TOP_THETA_METRICS)
        alpha_comp = signed_composite_effect(df, TOP_ALPHA_METRICS)
        if theta_comp is None or alpha_comp is None:
            continue
        valid = np.isfinite(theta_comp) & np.isfinite(alpha_comp)
        if valid.sum() < 10:
            continue
        tc = theta_comp[valid]; ac = alpha_comp[valid]
        r, p = stats.spearmanr(tc, ac)
        per_ds.append({'dataset': name, 'n': valid.sum(),
                       'spearman_theta_vs_alpha': r, 'p': p})
        all_theta.extend(tc.tolist())
        all_alpha.extend(ac.tolist())
    pool_r, pool_p = stats.spearmanr(all_theta, all_alpha)
    return pd.DataFrame(per_ds), {'n_pooled': len(all_theta),
                                   'rho_pooled': pool_r, 'p_pooled': pool_p}


# =========================================================================
# (3) Inter-event interval × alpha regression-to-mean
# =========================================================================

def load_subject_events(dataset_key, subject_id):
    """Load events CSV for a subject; return t0_net array."""
    dir_name = DATASET_DIR_MAP.get(dataset_key)
    if not dir_name:
        return None
    path = os.path.join(EXPORTS_SIE, dir_name, f'{subject_id}_sie_events.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if 't0_net' not in df.columns:
        return None
    return df['t0_net'].values


def check_iei_alpha_regression(wenr_dfs):
    """Per subject: median inter-event interval; correlate with |alpha_asymmetry effect|."""
    rows = []
    for name, wdf in wenr_dfs.items():
        e_alpha = subject_effect(wdf, 'alpha', 'asymmetry')
        if e_alpha is None:
            continue
        for i, sub_id in enumerate(wdf['subject_id'].values):
            t0s = load_subject_events(name, sub_id)
            if t0s is None or len(t0s) < 2:
                continue
            ieis = np.diff(np.sort(t0s))
            if len(ieis) == 0:
                continue
            med_iei = float(np.median(ieis))
            rows.append({
                'dataset': name, 'subject_id': sub_id,
                'n_events': len(t0s),
                'median_iei_sec': med_iei,
                'alpha_asymmetry_effect': e_alpha[i] if i < len(e_alpha) else np.nan,
            })
    all_df = pd.DataFrame(rows).dropna()
    # Overall correlations
    if len(all_df) < 20:
        return all_df, {}
    r_raw, p_raw = stats.spearmanr(all_df['median_iei_sec'],
                                    all_df['alpha_asymmetry_effect'])
    r_abs, p_abs = stats.spearmanr(all_df['median_iei_sec'],
                                    all_df['alpha_asymmetry_effect'].abs())
    # Bunched vs spread split at median
    median_iei = all_df['median_iei_sec'].median()
    bunched = all_df[all_df['median_iei_sec'] < median_iei]
    spread = all_df[all_df['median_iei_sec'] >= median_iei]
    # Compute mean |effect| in each bin
    bunched_abs = bunched['alpha_asymmetry_effect'].abs().mean()
    spread_abs = spread['alpha_asymmetry_effect'].abs().mean()
    t_split, p_split = stats.ttest_ind(
        bunched['alpha_asymmetry_effect'].abs(),
        spread['alpha_asymmetry_effect'].abs(),
    )
    # Also check n_events × alpha effect WITHIN the spread (widely-spaced) subset:
    # if regression-to-mean is driving the dose-response pattern, the pattern
    # should be WEAKER in spread-event subjects.
    if len(spread) > 20:
        r_spread_n, p_spread_n = stats.spearmanr(
            spread['n_events'], spread['alpha_asymmetry_effect'].abs(),
        )
    else:
        r_spread_n, p_spread_n = np.nan, np.nan
    if len(bunched) > 20:
        r_bunched_n, p_bunched_n = stats.spearmanr(
            bunched['n_events'], bunched['alpha_asymmetry_effect'].abs(),
        )
    else:
        r_bunched_n, p_bunched_n = np.nan, np.nan
    summary = {
        'n': len(all_df),
        'median_iei': median_iei,
        'rho_iei_vs_alpha_raw': r_raw, 'p_raw': p_raw,
        'rho_iei_vs_alpha_abs': r_abs, 'p_abs': p_abs,
        'bunched_mean_abs_effect': bunched_abs,
        'spread_mean_abs_effect': spread_abs,
        'bunched_vs_spread_t': t_split, 'p_bunched_vs_spread': p_split,
        'rho_n_events_vs_alpha_abs_in_spread': r_spread_n, 'p_spread_subset': p_spread_n,
        'rho_n_events_vs_alpha_abs_in_bunched': r_bunched_n, 'p_bunched_subset': p_bunched_n,
    }
    return all_df, summary


# =========================================================================
# MAIN
# =========================================================================

def main():
    print('Loading window enrichment data...')
    dfs = load_window_enrichment()
    print(f'  {len(dfs)} datasets, total N = {sum(len(d) for d in dfs.values())}')

    print('\n=== (1) n_events vs recording duration ===')
    per_ds, pool = check_n_events_vs_duration(dfs)
    per_ds.to_csv(os.path.join(OUT_DIR, 'sie_wenr_presub_duration.csv'), index=False)
    print(f"  Pooled: N={pool['n_pooled']}  rho={pool['rho_pooled']:+.3f}  p={pool['p_pooled']:.2e}")
    print(f"  {'dataset':30s}  {'N':>4s}  {'rho':>8s}  {'p':>10s}  {'med_dur':>8s}  "
          f"{'med_nev':>8s}  {'ev/100s':>8s}")
    for _, r in per_ds.iterrows():
        sig = '*' if r['p'] < 0.05 else ''
        print(f"  {r['dataset']:30s}  {r['n']:>4d}  "
              f"{r['spearman_dur_events']:>+8.3f}  {r['p']:>10.2e}  "
              f"{r['median_duration_sec']:>8.1f}  {r['median_n_events']:>8.0f}  "
              f"{r['events_per_100s_median']:>8.3f} {sig}")

    print('\n=== (2) Within-subject theta-alpha dissociation ===')
    per_ds2, pool2 = check_theta_alpha_dissociation(dfs)
    per_ds2.to_csv(os.path.join(OUT_DIR, 'sie_wenr_presub_theta_alpha.csv'), index=False)
    print(f"  Pooled: N={pool2['n_pooled']}  rho={pool2['rho_pooled']:+.3f}  p={pool2['p_pooled']:.2e}")
    print(f"  {'dataset':30s}  {'N':>4s}  {'rho':>8s}  {'p':>10s}")
    for _, r in per_ds2.iterrows():
        sig = '*' if r['p'] < 0.05 else ''
        print(f"  {r['dataset']:30s}  {r['n']:>4d}  "
              f"{r['spearman_theta_vs_alpha']:>+8.3f}  {r['p']:>10.2e} {sig}")

    print('\n=== (3) Inter-event interval × alpha regression-to-mean ===')
    all_iei, sum3 = check_iei_alpha_regression(dfs)
    if sum3:
        all_iei.to_csv(os.path.join(OUT_DIR, 'sie_wenr_presub_iei.csv'), index=False)
        print(f"  N = {sum3['n']}, median IEI = {sum3['median_iei']:.1f} s")
        print(f"  Spearman median_IEI vs signed alpha_asymmetry effect: "
              f"rho={sum3['rho_iei_vs_alpha_raw']:+.3f}, p={sum3['p_raw']:.2e}")
        print(f"  Spearman median_IEI vs |alpha_asymmetry effect|:     "
              f"rho={sum3['rho_iei_vs_alpha_abs']:+.3f}, p={sum3['p_abs']:.2e}")
        print(f"  Bunched (IEI < median) mean |effect| = {sum3['bunched_mean_abs_effect']:.3f}")
        print(f"  Spread  (IEI ≥ median) mean |effect| = {sum3['spread_mean_abs_effect']:.3f}")
        print(f"  t-test bunched vs spread: t={sum3['bunched_vs_spread_t']:+.2f}  "
              f"p={sum3['p_bunched_vs_spread']:.2e}")
        print(f"  In bunched subset: rho(n_events, |alpha effect|) = "
              f"{sum3['rho_n_events_vs_alpha_abs_in_bunched']:+.3f} p={sum3['p_bunched_subset']:.2e}")
        print(f"  In spread  subset: rho(n_events, |alpha effect|) = "
              f"{sum3['rho_n_events_vs_alpha_abs_in_spread']:+.3f} p={sum3['p_spread_subset']:.2e}")


if __name__ == '__main__':
    main()
