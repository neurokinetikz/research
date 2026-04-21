#!/usr/bin/env python3
"""
A5f — Subject-level ICC for each ratio, across three event sources:
current, composite, random.

Uses the already-extracted ratios from:
  - composite_vs_current_ratios.csv (A5b)
  - random_window_ratios.csv (A5e)

Two-level MixedLM per paper's canonical specification:
    ratio ~ 1 + (1|subject_id)
ICC = σ²_subject / (σ²_subject + σ²_residual)

Reports per-ratio ICC for each source, with CIs from parametric bootstrap
(resample subjects, refit, 500 iterations).
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import warnings

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'composite_detector')
warnings.filterwarnings('ignore')

PHI = (1 + 5 ** 0.5) / 2
RATIO_TARGETS = {
    'sr3/sr1': PHI ** 2,
    'sr5/sr1': PHI ** 3,
    'sr5/sr3': PHI ** 1,
    'sr6/sr4': PHI ** 1,
}


def fit_icc(df, ratio_col):
    """Fit MixedLM, return ICC or NaN on failure."""
    df = df[['subject_id', ratio_col]].dropna()
    df = df.rename(columns={ratio_col: 'y'})
    if df['subject_id'].nunique() < 5 or len(df) < 20:
        return np.nan, np.nan, np.nan
    try:
        md = smf.mixedlm("y ~ 1", df, groups=df['subject_id'])
        mdf = md.fit(reml=True, method='lbfgs')
        var_sub = float(mdf.cov_re.iloc[0, 0])
        var_res = float(mdf.scale)
        icc = var_sub / (var_sub + var_res) if (var_sub + var_res) > 0 else 0.0
        return icc, var_sub, var_res
    except Exception:
        return np.nan, np.nan, np.nan


def bootstrap_icc(df, ratio_col, n_boot=300, seed=0):
    """Subject-level cluster bootstrap CI on ICC."""
    rng = np.random.default_rng(seed)
    subs = df['subject_id'].unique()
    iccs = []
    for _ in range(n_boot):
        picked = rng.choice(subs, size=len(subs), replace=True)
        # rebuild resampled df
        parts = []
        for i, s in enumerate(picked):
            sub_df = df[df['subject_id'] == s].copy()
            sub_df['subject_id'] = f'{s}_b{i}'
            parts.append(sub_df)
        df_b = pd.concat(parts, ignore_index=True)
        icc, _, _ = fit_icc(df_b, ratio_col)
        if np.isfinite(icc):
            iccs.append(icc)
    if not iccs:
        return np.nan, np.nan
    return float(np.percentile(iccs, 2.5)), float(np.percentile(iccs, 97.5))


def main():
    det = pd.read_csv(os.path.join(OUT_DIR, 'composite_vs_current_ratios.csv'))
    rnd = pd.read_csv(os.path.join(OUT_DIR, 'random_window_ratios.csv'))
    df_all = pd.concat([det, rnd], ignore_index=True)

    rows = []
    print(f"\n{'ratio':10s} {'source':10s} {'n_sub':>6s} {'n_evt':>6s}  "
          f"{'ICC':>7s}  [95% CI]")
    for name in RATIO_TARGETS:
        for src in ['current', 'composite', 'random']:
            sub = df_all[df_all['source'] == src][['subject_id', name]].dropna()
            n_sub = sub['subject_id'].nunique()
            n_evt = len(sub)
            icc, var_s, var_r = fit_icc(sub, name)
            lo, hi = bootstrap_icc(sub, name, n_boot=200)
            rows.append({
                'ratio': name, 'source': src, 'n_subjects': n_sub,
                'n_events': n_evt, 'var_subject': var_s, 'var_residual': var_r,
                'ICC': icc, 'ICC_lo': lo, 'ICC_hi': hi,
            })
            print(f"{name:10s} {src:10s} {n_sub:>6d} {n_evt:>6d}  "
                  f"{icc:>7.3f}  [{lo:.3f}, {hi:.3f}]")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, 'icc_3way_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Figure: grouped bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(RATIO_TARGETS))
    width = 0.27
    colors = {'current': 'steelblue', 'composite': 'coral', 'random': 'gray'}
    for i, src in enumerate(['current', 'composite', 'random']):
        sub = df[df['source'] == src].set_index('ratio').reindex(RATIO_TARGETS.keys())
        vals = sub['ICC'].values
        lo = sub['ICC_lo'].values
        hi = sub['ICC_hi'].values
        err_lo = np.clip(vals - lo, 0, None)
        err_hi = np.clip(hi - vals, 0, None)
        ax.bar(x + (i - 1) * width, vals, width=width, color=colors[src], label=src,
               edgecolor='k', lw=0.3,
               yerr=[err_lo, err_hi], capsize=3, ecolor='k')
        for xi, v in zip(x + (i - 1) * width, vals):
            ax.text(xi, v + 0.005, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(list(RATIO_TARGETS.keys()))
    ax.set_ylabel('Subject-level ICC (two-level MixedLM)')
    ax.set_title('A5f — Subject ICC for harmonic ratios across event sources\n'
                  'LEMON EC · 123 subjects · ~710 events per source')
    ax.legend(title='event source')
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, 'icc_3way.png')
    plt.savefig(fig_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}")


if __name__ == '__main__':
    main()
