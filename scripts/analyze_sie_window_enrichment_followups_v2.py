#!/usr/bin/env python3
"""
SIE Window Enrichment — Follow-up Analyses V2 (3 additional checks)
====================================================================

  (A) n_peaks as SNR artifact — does n_peaks ↑ correlate with the
      redistribution effects (asymmetry/noble_5/inv_noble_5)? If yes, the
      "cross-band transfer" may partly reflect improved detector sensitivity
      during high-coherence epochs; if no, redistribution is separate.

  (B) Dose-response curve shape — bin mean |effect| by n_events (2, 3, 4,
      5, 6+). Plateau above ~3 events supports event-linkage; monotone
      decrease from n=2 supports floor-artifact interpretation.

  (C) EC vs EO on FDR-surviving metrics only (r across ~20 top hits, not
      all 95). Should give r > 0.8 if core signature is genuinely
      state-invariant.
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


def subject_effect(df, band, key):
    pc = col('pre', band, key); ic = col('ignition', band, key); poc = col('post', band, key)
    if not all(c in df.columns for c in [pc, ic, poc]):
        return None
    return (df[ic] - (df[pc] + df[poc]) / 2.0).values


# =========================================================================
# (A) n_peaks vs redistribution correlation
# =========================================================================

def check_n_peaks_as_snr(dfs):
    """Per subject, compute (a) total n_peaks ignition effect (sum across bands)
    and (b) magnitude of redistribution effect (|alpha_asymmetry_effect| +
    |theta_asymmetry_effect|). Correlate."""
    rows = []
    for name, df in dfs.items():
        # Total n_peaks ignition effect (sum across bands if available)
        n_peaks_effect = np.zeros(len(df))
        for b in BANDS:
            e = subject_effect(df, b, 'n_peaks')
            if e is not None:
                n_peaks_effect = n_peaks_effect + np.where(np.isfinite(e), e, 0)
        # Redistribution magnitude
        alpha_asym_eff = subject_effect(df, 'alpha', 'asymmetry')
        theta_asym_eff = subject_effect(df, 'theta', 'asymmetry')
        if alpha_asym_eff is None or theta_asym_eff is None:
            continue
        redist_mag = np.abs(alpha_asym_eff) + np.abs(theta_asym_eff)
        valid = np.isfinite(n_peaks_effect) & np.isfinite(redist_mag)
        if valid.sum() < 10:
            continue
        r, p = stats.spearmanr(n_peaks_effect[valid], redist_mag[valid])
        rows.append({'dataset': name, 'n_subjects': valid.sum(),
                     'spearman_r_n_peaks_vs_redist': r, 'p': p})
    df_out = pd.DataFrame(rows)
    # Overall: pool all subjects
    all_nps = []
    all_rm = []
    for name, df in dfs.items():
        nps = np.zeros(len(df))
        for b in BANDS:
            e = subject_effect(df, b, 'n_peaks')
            if e is not None:
                nps = nps + np.where(np.isfinite(e), e, 0)
        aa = subject_effect(df, 'alpha', 'asymmetry')
        ta = subject_effect(df, 'theta', 'asymmetry')
        if aa is None or ta is None:
            continue
        rm = np.abs(aa) + np.abs(ta)
        ok = np.isfinite(nps) & np.isfinite(rm)
        all_nps.append(nps[ok]); all_rm.append(rm[ok])
    all_nps = np.concatenate(all_nps); all_rm = np.concatenate(all_rm)
    r_pool, p_pool = stats.spearmanr(all_nps, all_rm)
    return df_out, {'pooled_n': len(all_nps), 'pooled_rho': r_pool, 'pooled_p': p_pool}


# =========================================================================
# (B) Dose-response curve shape
# =========================================================================

def dose_response_binned(dfs, metric_bands_keys):
    """For each metric, bin subjects by n_events and plot mean |effect|."""
    results = []
    for (band, key) in metric_bands_keys:
        all_rows = []
        for name, df in dfs.items():
            e = subject_effect(df, band, key)
            if e is None or 'n_events' not in df.columns:
                continue
            sub_df = pd.DataFrame({'effect': e, 'n_events': df['n_events'].values,
                                   'dataset': name}).dropna()
            all_rows.append(sub_df)
        if not all_rows:
            continue
        pooled = pd.concat(all_rows, ignore_index=True)
        pooled['abs_effect'] = pooled['effect'].abs()
        pooled['n_events_bin'] = pooled['n_events'].apply(
            lambda n: 2 if n == 2 else (3 if n == 3 else (4 if n == 4 else
                         (5 if n == 5 else (6 if n <= 7 else 8))))
        )
        agg = pooled.groupby('n_events_bin').agg(
            mean_abs_effect=('abs_effect', 'mean'),
            sem_abs_effect=('abs_effect', 'sem'),
            n=('abs_effect', 'count')
        ).reset_index()
        results.append({'band': band, 'key': key,
                        'bins': agg.to_dict('records')})
    return results


# =========================================================================
# (C) EC vs EO on FDR survivors only
# =========================================================================

FDR_SURVIVORS = [
    ('alpha', 'asymmetry'),
    ('theta', 'asymmetry'),
    ('alpha', 'center_depletion'),
    ('alpha', 'noble_5'),
    ('theta', 'inv_noble_5'),
    ('beta_low', 'ushape'),
    ('theta', 'inv_noble_6'),
    ('theta', 'ushape'),
    ('alpha', 'inv_noble_3'),
    ('alpha', 'noble_4'),
    ('theta', 'mountain'),
    ('beta_low', 'attractor'),
    ('theta', 'ramp_depth'),
    ('alpha', 'inv_noble_5'),
    ('alpha', 'ramp_depth'),
    ('theta', 'inv_noble_3'),
    ('theta', 'noble_1'),
    ('alpha', 'mountain'),
]

EC_EO_PAIRS = [
    ('lemon', 'lemon_EO'),
    ('tdbrain', 'tdbrain_EO'),
    ('dortmund', 'dortmund_EO-pre'),
    ('dortmund_EC-post', 'dortmund_EO-post'),
]


def ec_eo_survivors_only(dfs):
    """Correlate per-metric d across EC and EO datasets, using only FDR-surviving metrics."""
    rows = []
    for ec, eo in EC_EO_PAIRS:
        if ec not in dfs or eo not in dfs:
            continue
        ec_d = []; eo_d = []
        for band, key in FDR_SURVIVORS:
            e_ec = subject_effect(dfs[ec], band, key)
            e_eo = subject_effect(dfs[eo], band, key)
            if e_ec is None or e_eo is None or len(e_ec) < 5 or len(e_eo) < 5:
                continue
            d_ec = e_ec.mean() / e_ec.std(ddof=1) if e_ec.std(ddof=1) > 0 else 0
            d_eo = e_eo.mean() / e_eo.std(ddof=1) if e_eo.std(ddof=1) > 0 else 0
            ec_d.append(d_ec); eo_d.append(d_eo)
        ec_d = np.array(ec_d); eo_d = np.array(eo_d)
        if len(ec_d) >= 5:
            r, p = stats.pearsonr(ec_d, eo_d)
            rows.append({
                'pair': f'{ec} vs {eo}',
                'n_metrics': len(ec_d),
                'corr': r, 'p': p,
                'sign_agreement': (np.sign(ec_d) == np.sign(eo_d)).mean(),
            })
    return pd.DataFrame(rows)


# =========================================================================
# MAIN
# =========================================================================

def main():
    print('Loading...')
    dfs = load_all()
    print(f'  {len(dfs)} datasets, total N = {sum(len(d) for d in dfs.values())}')

    print('\n=== (A) n_peaks ↑ vs redistribution magnitude ===')
    per_ds, pool = check_n_peaks_as_snr(dfs)
    print(f"  Pooled: N={pool['pooled_n']}  rho={pool['pooled_rho']:+.3f}  p={pool['pooled_p']:.2e}")
    print('  Per dataset:')
    for _, r in per_ds.iterrows():
        sig = '*' if r['p'] < 0.05 else ''
        print(f"    {r['dataset']:25s}  N={r['n_subjects']:3d}  "
              f"rho={r['spearman_r_n_peaks_vs_redist']:+.3f}  p={r['p']:.2e} {sig}")

    print('\n=== (B) Dose-response curve shape (|effect| by n_events bin) ===')
    results = dose_response_binned(dfs, [
        ('alpha', 'asymmetry'), ('theta', 'asymmetry'),
        ('alpha', 'noble_5'), ('theta', 'inv_noble_5'),
        ('beta_low', 'attractor'), ('theta', 'mountain'),
    ])
    for r in results:
        print(f"\n  {r['band']}_{r['key']}:")
        print(f"    {'n_events':>10s}  {'|effect|':>10s}  {'SEM':>8s}  {'N':>5s}")
        for b in r['bins']:
            bin_label = f"{b['n_events_bin']}" if b['n_events_bin'] < 8 else '6+'
            print(f"    {bin_label:>10s}  {b['mean_abs_effect']:>10.4f}  "
                  f"{b['sem_abs_effect']:>8.4f}  {b['n']:>5d}")

    print('\n=== (C) EC vs EO on FDR survivors only ===')
    ec_eo = ec_eo_survivors_only(dfs)
    ec_eo.to_csv(os.path.join(OUT_DIR, 'sie_wenr_followup_v2_ec_eo_survivors.csv'), index=False)
    for _, r in ec_eo.iterrows():
        print(f"  {r['pair']:45s}  N_metrics={r['n_metrics']}  r={r['corr']:+.3f}  "
              f"p={r['p']:.1e}  sign_agree={r['sign_agreement']:.2f}")


if __name__ == '__main__':
    main()
