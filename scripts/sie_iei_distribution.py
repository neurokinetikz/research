#!/usr/bin/env python3
"""
B2 — Inter-event interval (IEI) distribution.

Tests whether ignition events occur as Poisson-like independent triggers or
show bursty/power-law/critical statistics.

For each subject with ≥3 events:
  - IEIs from consecutive t0_net values
  - Coefficient of variation (CV = std/mean)
    Poisson CV ≈ 1, bursty CV > 1, regular CV < 1

Pool IEIs across subjects:
  - Fit exponential, power-law, and stretched-exponential
  - Compare by AIC (lowest wins)
  - Report KS test against exponential

Also per-subject CV distribution; median and IQR.

Lightweight — reads event CSVs only.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, optimize

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'iei')
EVENTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'exports_sie', 'lemon')
os.makedirs(OUT_DIR, exist_ok=True)


def fit_models(ieis):
    """Fit exponential, power-law (truncated), stretched-exp. Return dict."""
    ieis = np.asarray(ieis)
    ieis = ieis[np.isfinite(ieis) & (ieis > 0)]
    n = len(ieis)
    out = {'n': n}

    # Exponential: lambda = 1/mean
    lam = 1.0 / np.mean(ieis)
    logL_exp = n * np.log(lam) - lam * np.sum(ieis)
    k_exp = 1
    out['exp'] = {'lam': lam, 'logL': logL_exp, 'AIC': 2 * k_exp - 2 * logL_exp}

    # Power-law (truncated) p(x) = (alpha-1)/xmin * (x/xmin)^(-alpha), x >= xmin
    xmin = float(np.percentile(ieis, 5))  # conservative lower bound
    sel = ieis[ieis >= xmin]
    if len(sel) > 10:
        alpha = 1.0 + len(sel) / np.sum(np.log(sel / xmin))
        logL_pl = (len(sel) * np.log((alpha - 1) / xmin)
                   - alpha * np.sum(np.log(sel / xmin)))
        # For fair AIC, compare using only the tail — but we want whole-distribution
        # comparison. Use full fit instead: shifted Pareto via MLE on all events.
        # Using tail-based is a limited comparison; report both.
        k_pl = 1
        out['power_law_tail'] = {'alpha': alpha, 'xmin': xmin,
                                   'n_tail': len(sel), 'logL': logL_pl,
                                   'AIC': 2 * k_pl - 2 * logL_pl}

    # Stretched exponential: p(x) = (beta/tau) * (x/tau)^(beta-1) * exp(-(x/tau)^beta)
    def neg_ll_stretched(params):
        beta, tau = params
        if beta <= 0 or tau <= 0:
            return np.inf
        ll = np.sum(np.log(beta / tau) + (beta - 1) * np.log(ieis / tau)
                     - (ieis / tau) ** beta)
        return -ll
    try:
        res = optimize.minimize(neg_ll_stretched, [1.0, np.mean(ieis)],
                                  method='Nelder-Mead')
        beta, tau = res.x
        logL_st = -res.fun
        k_st = 2
        out['stretched'] = {'beta': beta, 'tau': tau, 'logL': logL_st,
                             'AIC': 2 * k_st - 2 * logL_st}
    except Exception:
        out['stretched'] = None

    # KS test vs exponential with the fitted lambda
    try:
        ks_stat, ks_p = stats.kstest(ieis, lambda x: 1 - np.exp(-lam * x))
        out['ks_vs_exp'] = {'stat': float(ks_stat), 'p': float(ks_p)}
    except Exception:
        out['ks_vs_exp'] = None

    return out


def main():
    summary = pd.read_csv(os.path.join(EVENTS_DIR, 'extraction_summary.csv'))
    ok = summary[(summary['status'] == 'ok') & (summary['n_events'] >= 3)]

    per_subject_rows = []
    pooled_iei = []

    for _, r in ok.iterrows():
        path = os.path.join(EVENTS_DIR, f'{r["subject_id"]}_sie_events.csv')
        if not os.path.isfile(path):
            continue
        events = pd.read_csv(path).dropna(subset=['t0_net']).sort_values('t0_net')
        if len(events) < 3:
            continue
        t0 = events['t0_net'].values
        ieis = np.diff(t0)
        if len(ieis) < 2:
            continue
        mean_iei = float(np.mean(ieis))
        std_iei = float(np.std(ieis))
        cv = std_iei / mean_iei if mean_iei > 0 else np.nan
        per_subject_rows.append({
            'subject_id': r['subject_id'],
            'n_events': int(len(events)),
            'n_iei': int(len(ieis)),
            'mean_iei_s': mean_iei,
            'std_iei_s': std_iei,
            'CV': cv,
            'median_iei_s': float(np.median(ieis)),
        })
        pooled_iei.extend(ieis.tolist())

    df_sub = pd.DataFrame(per_subject_rows)
    df_sub.to_csv(os.path.join(OUT_DIR, 'per_subject_iei_stats.csv'), index=False)

    print(f"Subjects with ≥2 IEIs: {len(df_sub)}")
    print(f"Pooled IEIs: {len(pooled_iei)}")
    print(f"\nPer-subject CV (Poisson = 1):")
    print(f"  median {df_sub['CV'].median():.3f}  "
          f"IQR [{df_sub['CV'].quantile(.25):.3f}, {df_sub['CV'].quantile(.75):.3f}]")
    print(f"  % CV > 1 (bursty): {(df_sub['CV'] > 1).mean()*100:.1f}%")
    print(f"\nPer-subject mean IEI: median {df_sub['mean_iei_s'].median():.2f}s  "
          f"IQR [{df_sub['mean_iei_s'].quantile(.25):.2f}, "
          f"{df_sub['mean_iei_s'].quantile(.75):.2f}]s")

    # Fit models on pooled IEIs
    fits = fit_models(pooled_iei)
    print(f"\n=== Pooled IEI model fits ===")
    for name, v in fits.items():
        if name == 'n':
            print(f"  n = {v}")
            continue
        if v is None:
            print(f"  {name}: (fit failed)")
            continue
        items = ', '.join(f'{k}={val:.4g}' if isinstance(val, float) else f'{k}={val}'
                          for k, val in v.items())
        print(f"  {name}: {items}")

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.hist(df_sub['CV'], bins=30, color='steelblue', edgecolor='k', lw=0.3)
    ax.axvline(1.0, color='red', ls='--', lw=1, label='Poisson (CV=1)')
    ax.axvline(df_sub['CV'].median(), color='k', ls='-', lw=1,
                label=f"median = {df_sub['CV'].median():.2f}")
    ax.set_xlabel('coefficient of variation (CV)')
    ax.set_ylabel('subjects')
    ax.set_title('Per-subject CV distribution')
    ax.legend(fontsize=9)

    ax = axes[1]
    pooled = np.array(pooled_iei)
    ax.hist(pooled[pooled < 60], bins=50, color='seagreen', edgecolor='k',
             lw=0.3, alpha=0.85, density=True)
    lam = fits['exp']['lam']
    xs = np.linspace(0.01, 60, 400)
    ax.plot(xs, lam * np.exp(-lam * xs), color='red', lw=1.5,
             label=f'exp fit (λ={lam:.3f})')
    ax.set_xlabel('IEI (s, clipped ≤60)')
    ax.set_ylabel('density')
    ax.set_title('Pooled IEI histogram')
    ax.legend()

    ax = axes[2]
    # Log-log survival function
    sorted_ieis = np.sort(pooled)[::-1]
    surv = np.arange(1, len(sorted_ieis) + 1) / len(sorted_ieis)
    ax.loglog(sorted_ieis, surv, 'o', markersize=2, color='seagreen',
               label='empirical')
    # Exponential survival for reference
    xs = np.logspace(-1, np.log10(sorted_ieis.max()), 200)
    ax.loglog(xs, np.exp(-lam * xs), 'r--', lw=1.5, label='exp')
    ax.set_xlabel('IEI (s, log)')
    ax.set_ylabel('P(IEI > x)')
    ax.set_title('Survival function (log-log)')
    ax.legend()

    plt.suptitle(f'B2 — IEI distribution · {len(df_sub)} subjects · '
                 f'{len(pooled_iei)} pooled intervals', y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'iei_distribution.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {OUT_DIR}/iei_distribution.png")


if __name__ == '__main__':
    main()
