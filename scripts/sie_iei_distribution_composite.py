#!/usr/bin/env python3
"""
B2b re-run on composite v2 detector.

Composite v2 events are the direct detector output (local maxima of
S(t) = cbrt(zE·zR·zP·zM) ≥ 1.5, min-ISI 2 s, edge-masked 5 s). Unlike envelope
Stage 1, there is no separate window-merge rule — composite events are the
terminal unit.

This script computes IEIs = diff(t0_net) per subject, per-subject CV, burstiness
fraction (CV > 1), and pooled IEI distribution. Compares to:
  - B2 envelope post-merge: CV 0.40 (ARTIFACT per B2b)
  - B2b envelope raw-crossing: CV 0.89 (Poisson-like)
  - Poisson expectation: CV ≈ 1

Cohort-parameterized.

Usage:
    python scripts/sie_iei_distribution_composite.py --cohort lemon
    python scripts/sie_iei_distribution_composite.py --cohort lemon_EO
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, optimize

ROOT = os.path.join(os.path.dirname(__file__), '..')


def cohort_events_dir(cohort):
    return os.path.join(ROOT, 'exports_sie', f'{cohort}_composite')


def fit_exponential(ieis):
    lam = 1.0 / np.mean(ieis)
    ks_stat, ks_p = stats.kstest(ieis, 'expon', args=(0, 1.0 / lam))
    return lam, ks_stat, ks_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', default='lemon')
    args = ap.parse_args()

    events_dir = cohort_events_dir(args.cohort)
    out_dir = os.path.join(ROOT, 'outputs', 'schumann', 'images',
                            'iei', f'{args.cohort}_composite')
    os.makedirs(out_dir, exist_ok=True)

    summary = pd.read_csv(os.path.join(events_dir, 'extraction_summary.csv'))
    ok = summary[(summary['status'] == 'ok') & (summary['n_events'] >= 3)]
    all_ieis = []
    per_subject_rows = []

    for _, r in ok.iterrows():
        sub = r['subject_id']
        ev_path = os.path.join(events_dir, f'{sub}_sie_events.csv')
        if not os.path.isfile(ev_path):
            continue
        ev = pd.read_csv(ev_path).dropna(subset=['t0_net']).sort_values('t0_net')
        t = ev['t0_net'].astype(float).values
        if len(t) < 3:
            continue
        iei = np.diff(t)
        iei = iei[iei > 0]
        if len(iei) < 2:
            continue
        all_ieis.append(iei)
        per_subject_rows.append({
            'subject_id': sub,
            'n_events': int(len(t)),
            'n_iei': int(len(iei)),
            'iei_mean': float(np.mean(iei)),
            'iei_median': float(np.median(iei)),
            'iei_std': float(np.std(iei, ddof=1)),
            'iei_cv': float(np.std(iei, ddof=1) / np.mean(iei)),
            'iei_min': float(np.min(iei)),
            'iei_max': float(np.max(iei)),
        })

    sub_df = pd.DataFrame(per_subject_rows)
    sub_df.to_csv(os.path.join(out_dir, 'per_subject_iei_stats.csv'), index=False)

    pooled = np.concatenate(all_ieis)
    pooled_cv = float(np.std(pooled, ddof=1) / np.mean(pooled))

    print(f"=== {args.cohort} composite · IEI summary ===")
    print(f"  n_subjects = {len(sub_df)}   pooled n_iei = {len(pooled)}")
    print(f"  Per-subject CV:         median {sub_df['iei_cv'].median():.3f}   "
          f"IQR [{sub_df['iei_cv'].quantile(.25):.3f}, {sub_df['iei_cv'].quantile(.75):.3f}]")
    print(f"  Per-subject mean IEI:   median {sub_df['iei_mean'].median():.2f} s")
    print(f"  Per-subject median IEI: median {sub_df['iei_median'].median():.2f} s")
    print(f"  % subjects CV > 1 (bursty): {(sub_df['iei_cv'] > 1).mean()*100:.1f}%")
    print(f"  % subjects CV < 0.5 (sub-Poisson): {(sub_df['iei_cv'] < 0.5).mean()*100:.1f}%")
    print(f"  Pooled IEI: mean {np.mean(pooled):.2f}s  std {np.std(pooled, ddof=1):.2f}s  "
          f"median {np.median(pooled):.2f}s  CV {pooled_cv:.3f}")

    # KS test vs exponential (Poisson expectation)
    lam, ks_stat, ks_p = fit_exponential(pooled)
    print(f"\n  Exponential fit: λ = {lam:.4f}  (mean 1/λ = {1/lam:.2f} s)")
    print(f"  KS test vs exponential: stat={ks_stat:.3f} p={ks_p:.3g}")

    # Compare to envelope baselines
    print(f"\n=== Comparison to envelope baselines ===")
    print(f"  B2  envelope post-merge: CV 0.40 (ARTIFACT per B2b)")
    print(f"  B2b envelope raw-crossing: CV 0.89, 37% bursty (Poisson-like)")
    print(f"  B2b Poisson-rate-matched + merge sim: CV 0.43")
    print(f"  Poisson expected: CV ≈ 1.0")
    print(f"  --> {args.cohort} composite CV = {pooled_cv:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.hist(sub_df['iei_cv'], bins=30, color='slategray', edgecolor='k',
             lw=0.3, alpha=0.85)
    ax.axvline(sub_df['iei_cv'].median(), color='firebrick', ls='--', lw=1.5,
                label=f"median {sub_df['iei_cv'].median():.2f}")
    ax.axvline(1.0, color='green', ls=':', lw=1.5, label='Poisson CV=1')
    ax.axvline(0.40, color='orange', ls=':', lw=1.5, label='B2 envelope post-merge')
    ax.axvline(0.89, color='blue', ls=':', lw=1.5, label='B2b envelope raw')
    ax.set_xlabel('per-subject IEI CV')
    ax.set_ylabel('subjects')
    ax.set_title(f'Per-subject CV · n={len(sub_df)}')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    bins = np.linspace(0, min(np.percentile(pooled, 99), 180), 80)
    ax.hist(pooled, bins=bins, color='steelblue', edgecolor='k',
             lw=0.3, alpha=0.85, density=True, label='empirical')
    x = np.linspace(0.1, bins[-1], 400)
    ax.plot(x, lam * np.exp(-lam * x), 'r-', lw=1.5,
             label=f'exp(λ={lam:.3f})')
    ax.set_xlabel('IEI (s)')
    ax.set_ylabel('density')
    ax.set_title(f'Pooled IEI distribution · CV={pooled_cv:.2f}')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    # Log-log survival curve
    sorted_iei = np.sort(pooled)[::-1]
    surv = np.arange(1, len(sorted_iei) + 1) / len(sorted_iei)
    ax.loglog(sorted_iei, surv, color='steelblue', lw=1.5, label='empirical')
    ax.loglog(x, np.exp(-lam * x), 'r-', lw=1.2, label=f'exp(λ={lam:.3f})')
    ax.set_xlabel('IEI (s)')
    ax.set_ylabel('P(IEI > x)')
    ax.set_title('Survival function (log-log)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')

    plt.suptitle(f'B2b · IEI distribution · {args.cohort} composite v2',
                 y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'iei_distribution.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_dir}/iei_distribution.png")
    print(f"Saved: {out_dir}/per_subject_iei_stats.csv")


if __name__ == '__main__':
    main()
