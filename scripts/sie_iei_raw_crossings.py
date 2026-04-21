#!/usr/bin/env python3
"""
B2b — Re-extract IEIs from raw envelope threshold crossings (before merge).

Reconstructs Stage 1 exactly:
  - Envelope at 7.83 ± 0.6 Hz (narrowband mean-channel, Hilbert envelope)
  - z-scored against whole session
  - Rising crossings of z >= 3.0, with min_isi = 2.0 s
These are the RAW trigger times — no window creation, no merge.

Compare to:
  - The merged/windowed t0_net IEIs (B2) — sub-Poisson CV ≈ 0.40
  - Expected Poisson with same rate + merge rule applied (simulation)

Outputs:
  - Per-subject raw IEI stats
  - Pooled raw IEI distribution (histogram, log-log survival)
  - Simulation: generate same number of Poisson events per subject, apply the
    merge rule (eliminate points within 30 s), measure CV of the surviving set.
    Compare empirical raw CV to simulated merged-Poisson CV.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import zscore
from multiprocessing import Pool
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.run_sie_extraction import load_lemon
from scripts.sie_perionset_triple_average import bandpass

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'iei')
EVENTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'exports_sie', 'lemon')
os.makedirs(OUT_DIR, exist_ok=True)
warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

F0 = 7.83
HALF_BW = 0.6
Z_THRESH = 3.0
MIN_ISI_SEC = 2.0
# Effective merge-rule minimum: window_sec + merge_gap_sec = 20 + 10 = 30 s
MERGE_EFFECTIVE_MIN = 30.0


def extract_raw_crossings(sub_id, events_path=None):
    """Return list of envelope-threshold crossing times (raw, unmerged)."""
    try:
        raw = load_lemon(sub_id, condition='EC')
    except Exception:
        return None
    if raw is None:
        return None
    fs = raw.info['sfreq']
    X = raw.get_data() * 1e6
    y = X.mean(axis=0)
    yb = bandpass(y, fs, F0 - HALF_BW, F0 + HALF_BW)
    env = np.abs(signal.hilbert(yb))
    z = zscore(env, nan_policy='omit')
    mask = z >= Z_THRESH
    on_idx = np.where(np.diff(mask.astype(int)) == 1)[0] + 1
    last_t = -np.inf
    onsets = []
    for i in on_idx:
        t = i / fs
        if t - last_t >= MIN_ISI_SEC:
            onsets.append(t); last_t = t
    return {
        'subject_id': sub_id,
        'n_raw_onsets': len(onsets),
        'duration_s': float(raw.n_times / fs),
        'onsets': np.array(onsets),
    }


def process_subject(args):
    return extract_raw_crossings(*args)


def simulated_merged_poisson(n_onsets_target, duration_s, seed=0):
    """Simulate a Poisson process with rate matching n_onsets/duration,
    then remove crossings within 30 s of the previous surviving one (greedy)
    to match the merge rule. Return IEIs of surviving points."""
    rng = np.random.default_rng(seed)
    # Generate 5× the target to account for elimination
    n_gen = int(n_onsets_target * 5 + 30)
    lam = n_onsets_target / duration_s
    # Poisson process: exponential IEIs
    ieis = rng.exponential(1.0/lam, n_gen)
    times = np.cumsum(ieis)
    times = times[times < duration_s]
    # Merge rule: enforce ≥30 s gap
    last_t = -np.inf
    surviving = []
    for t in times:
        if t - last_t >= MERGE_EFFECTIVE_MIN:
            surviving.append(t); last_t = t
    surv_ieis = np.diff(np.array(surviving))
    return surv_ieis


def main():
    summary = pd.read_csv(os.path.join(EVENTS_DIR, 'extraction_summary.csv'))
    ok = summary[summary['status'] == 'ok']
    tasks = [(r['subject_id'], None) for _, r in ok.iterrows()]
    print(f"Subjects: {len(tasks)}")

    with Pool(8) as pool:
        results = pool.map(process_subject, tasks)
    results = [r for r in results if r is not None]

    # Pool raw IEIs + per-subject stats
    per_subject_rows = []
    raw_pooled = []
    for r in results:
        ons = r['onsets']
        if len(ons) < 3:
            continue
        ieis = np.diff(ons)
        cv = float(np.std(ieis) / np.mean(ieis)) if np.mean(ieis) > 0 else np.nan
        per_subject_rows.append({
            'subject_id': r['subject_id'],
            'n_raw_onsets': len(ons),
            'n_iei': len(ieis),
            'mean_iei_s': float(np.mean(ieis)),
            'std_iei_s':  float(np.std(ieis)),
            'CV': cv,
            'median_iei_s': float(np.median(ieis)),
            'rate_per_min': 60.0 * len(ons) / r['duration_s'],
            'duration_s': r['duration_s'],
        })
        raw_pooled.extend(ieis.tolist())

    df_raw = pd.DataFrame(per_subject_rows)
    df_raw.to_csv(os.path.join(OUT_DIR, 'per_subject_raw_iei_stats.csv'), index=False)
    print(f"Subjects with ≥2 raw IEIs: {len(df_raw)}")
    print(f"Raw onsets per subject: median {df_raw['n_raw_onsets'].median():.0f}, "
          f"IQR [{df_raw['n_raw_onsets'].quantile(.25):.0f}, "
          f"{df_raw['n_raw_onsets'].quantile(.75):.0f}]")
    print(f"Raw rate/min: median {df_raw['rate_per_min'].median():.2f}")

    print(f"\n=== RAW envelope-crossing IEI ===")
    print(f"  Per-subject CV: median {df_raw['CV'].median():.3f}  "
          f"IQR [{df_raw['CV'].quantile(.25):.3f}, {df_raw['CV'].quantile(.75):.3f}]")
    print(f"  % CV > 1 (bursty): {(df_raw['CV'] > 1).mean() * 100:.1f}%")
    print(f"  Mean IEI: median {df_raw['mean_iei_s'].median():.2f}s  "
          f"IQR [{df_raw['mean_iei_s'].quantile(.25):.2f}, "
          f"{df_raw['mean_iei_s'].quantile(.75):.2f}]s")

    # Simulation: Poisson + merge rule
    sim_cvs = []
    rng_seed = 0
    for _, r in df_raw.iterrows():
        surv_ieis = simulated_merged_poisson(
            int(r['n_raw_onsets']), float(r['duration_s']), seed=rng_seed)
        rng_seed += 1
        if len(surv_ieis) >= 2:
            sim_cvs.append(float(np.std(surv_ieis) / np.mean(surv_ieis)))
    sim_cvs = np.array(sim_cvs)
    print(f"\n=== Simulated Poisson + 30-s merge rule ===")
    print(f"  Per-subject CV: median {np.median(sim_cvs):.3f}  "
          f"IQR [{np.percentile(sim_cvs,25):.3f}, {np.percentile(sim_cvs,75):.3f}]")
    print(f"  (If empirical CV < simulated, the regularity is above-and-beyond "
           "what the merge rule alone would produce.)")

    # Also load the original B2 t0_net-based CV numbers
    try:
        b2_df = pd.read_csv(os.path.join(OUT_DIR, 'per_subject_iei_stats.csv'))
        b2_cv = b2_df['CV'].median()
    except Exception:
        b2_cv = None

    # Fig: compare 3 CV distributions
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.hist(df_raw['CV'], bins=30, color='seagreen', edgecolor='k', lw=0.3,
             alpha=0.8, label=f'RAW crossings (median {df_raw["CV"].median():.2f})')
    ax.hist(sim_cvs, bins=30, color='gray', edgecolor='k', lw=0.3, alpha=0.5,
             label=f'Poisson+merge sim (median {np.median(sim_cvs):.2f})')
    if b2_cv is not None:
        ax.axvline(b2_cv, color='steelblue', ls='--', lw=1.5,
                    label=f'B2 t0_net (median {b2_cv:.2f})')
    ax.axvline(1.0, color='red', ls='--', lw=0.8, label='Poisson CV=1')
    ax.set_xlabel('coefficient of variation (CV)')
    ax.set_ylabel('subjects')
    ax.set_title('Per-subject CV distribution (3-way)')
    ax.legend(fontsize=8)

    ax = axes[1]
    pooled = np.array(raw_pooled)
    ax.hist(pooled[pooled < 60], bins=60, color='seagreen', edgecolor='k',
             lw=0.3, alpha=0.85, density=True)
    lam = 1 / np.mean(pooled)
    xs = np.linspace(0.01, 60, 400)
    ax.plot(xs, lam * np.exp(-lam * xs), color='red', lw=1.5,
             label=f'exp fit (λ={lam:.3f})')
    ax.set_xlabel('raw IEI (s, clipped ≤60)')
    ax.set_ylabel('density')
    ax.set_title(f'Pooled raw IEI (n={len(pooled)})')
    ax.legend()

    ax = axes[2]
    sorted_ieis = np.sort(pooled)[::-1]
    surv = np.arange(1, len(sorted_ieis)+1) / len(sorted_ieis)
    ax.loglog(sorted_ieis, surv, 'o', markersize=2, color='seagreen',
               label='empirical raw')
    xs = np.logspace(-1, np.log10(sorted_ieis.max()), 200)
    ax.loglog(xs, np.exp(-lam * xs), 'r--', lw=1.5, label='exp')
    ax.set_xlabel('IEI (s, log)')
    ax.set_ylabel('P(IEI > x)')
    ax.set_title('Survival function (log-log)')
    ax.legend()

    plt.suptitle(f'B2b — Raw envelope-crossing IEIs · {len(df_raw)} subjects · '
                 f'{len(raw_pooled)} pooled intervals', y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'iei_raw_vs_merged.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {OUT_DIR}/iei_raw_vs_merged.png")


if __name__ == '__main__':
    main()
