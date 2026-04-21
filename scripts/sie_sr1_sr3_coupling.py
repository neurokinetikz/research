#!/usr/bin/env python3
"""
B34 — SR1 × SR3 coupling analysis (envelope correlation + PAC).

Tests whether the two narrowband peaks at 7.82 Hz (SR1) and 19.95 Hz (SR3)
are coupled during ignition events. Since their ratio is 2.55 (non-integer),
direct PLV is uninformative. We use:

  1. Envelope cross-correlation: |hilbert(bp_SR1)| vs |hilbert(bp_SR3)|,
     with lag search. Peak correlation + lag at peak indicate whether and how
     the two amplitudes track each other.

  2. Tort PAC modulation index: does SR1 phase modulate SR3 amplitude? Bin
     SR1 phase into 18 equal bins, compute mean SR3 amplitude per bin,
     normalize. Larger MI → stronger phase→amplitude coupling.

  3. Null control: matched non-event windows drawn from the same recording
     (≥ 30 s from any event).

Per subject in LEMON (first pass): compute event-mean and control-mean of
each coupling measure. Report per-subject and cohort Wilcoxon.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import wilcoxon, mannwhitneyu
from multiprocessing import Pool
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.run_sie_extraction import load_lemon

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'coupling')
EVENTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'exports_sie', 'lemon')
os.makedirs(OUT_DIR, exist_ok=True)
warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

# Bands
SR1_BAND = (7.0, 8.2)
SR3_BAND = (19.5, 20.4)

# Event windows
EVENT_WIN = 6.0         # 6-s window around t0+1s (so [t0-2, t0+4])
EVENT_LAG = 1.0
MIN_GAP_FROM_EVENT = 30.0

LAGS_S = np.arange(-1.5, 1.5 + 0.05, 0.05)
N_PHASE_BINS = 18


def bandpass(x, fs, lo, hi, order=4):
    b, a = signal.butter(order, [lo, hi], btype='band', fs=fs)
    return signal.filtfilt(b, a, x, axis=-1)


def envelope_xcorr(env1, env2, fs, lags_s):
    """Cross-correlation of two envelopes at specified lags.
    Positive lag → env2 is shifted forward relative to env1 (env2 lags env1).
    """
    env1 = (env1 - env1.mean()) / (env1.std() + 1e-12)
    env2 = (env2 - env2.mean()) / (env2.std() + 1e-12)
    out = np.zeros(len(lags_s))
    n = len(env1)
    for i, lag_s in enumerate(lags_s):
        lag_samp = int(round(lag_s * fs))
        if lag_samp >= 0:
            a = env1[:n - lag_samp]
            b = env2[lag_samp:]
        else:
            a = env1[-lag_samp:]
            b = env2[:n + lag_samp]
        if len(a) < 10:
            out[i] = np.nan; continue
        out[i] = np.corrcoef(a, b)[0, 1]
    return out


def tort_mi(phase, amp, n_bins=N_PHASE_BINS):
    """Tort modulation index: KL divergence of amplitude distribution across
    phase bins from uniform. Higher → stronger phase-amplitude coupling."""
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_idx = np.digitize(phase, edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    mean_amp = np.array([np.nanmean(amp[bin_idx == k]) if np.sum(bin_idx == k) > 0 else np.nan
                          for k in range(n_bins)])
    if np.any(np.isnan(mean_amp)) or np.sum(mean_amp) <= 0:
        return np.nan
    p = mean_amp / np.sum(mean_amp)
    kl = np.nansum(p * np.log(p * n_bins + 1e-12))
    return float(kl / np.log(n_bins))   # normalize to [0, 1]


def sample_control_times(t_events, n_target, t_end, seed=0):
    rng = np.random.default_rng(seed)
    lo = EVENT_WIN / 2 + 1
    hi = t_end - EVENT_WIN / 2 - 1
    out, tries = [], 0
    while len(out) < n_target and tries < n_target * 200:
        t = rng.uniform(lo, hi)
        if len(t_events) == 0 or np.min(np.abs(t - t_events)) >= MIN_GAP_FROM_EVENT:
            out.append(t)
        tries += 1
    return np.array(out)


def process_subject(args):
    sub_id, events_path = args
    try:
        events = pd.read_csv(events_path).dropna(subset=['t0_net'])
    except Exception:
        return None
    if len(events) < 3:
        return None
    try:
        raw = load_lemon(sub_id, condition='EC')
    except Exception:
        return None
    if raw is None:
        return None
    fs = raw.info['sfreq']
    X = raw.get_data() * 1e6
    y = X.mean(axis=0)
    t_end = raw.times[-1]

    # Bandpass SR1 and SR3
    bp1 = bandpass(y, fs, *SR1_BAND)
    bp3 = bandpass(y, fs, *SR3_BAND)
    # Hilbert
    z1 = signal.hilbert(bp1)
    z3 = signal.hilbert(bp3)
    env1 = np.abs(z1)
    env3 = np.abs(z3)
    phase1 = np.angle(z1)

    t_events = events['t0_net'].astype(float).values
    t_controls = sample_control_times(t_events, len(t_events), t_end,
                                        seed=abs(hash(sub_id)) % (2**31))

    def score_window(t_center):
        i0 = int(round((t_center - EVENT_WIN / 2) * fs))
        i1 = i0 + int(round(EVENT_WIN * fs))
        if i0 < 0 or i1 > len(y):
            return None
        env1_w = env1[i0:i1]
        env3_w = env3[i0:i1]
        phase1_w = phase1[i0:i1]
        xc = envelope_xcorr(env1_w, env3_w, fs, LAGS_S)
        pk_idx = np.nanargmax(xc)
        pk_r = xc[pk_idx]
        pk_lag = LAGS_S[pk_idx]
        mi = tort_mi(phase1_w, env3_w)
        return {'r_peak': pk_r, 'lag_peak_s': pk_lag, 'r_at_zero': xc[len(LAGS_S)//2],
                'mi': mi}

    ev_rows, ct_rows = [], []
    for t0 in t_events:
        s = score_window(t0 + EVENT_LAG)
        if s: ev_rows.append(s)
    for tc in t_controls:
        s = score_window(tc)
        if s: ct_rows.append(s)

    if len(ev_rows) < 2 or len(ct_rows) < 2:
        return None
    ev = pd.DataFrame(ev_rows)
    ct = pd.DataFrame(ct_rows)
    return {
        'subject_id': sub_id,
        'n_events': len(ev),
        'n_controls': len(ct),
        'ev_r_peak_mean': float(ev['r_peak'].mean()),
        'ev_lag_peak_s_median': float(ev['lag_peak_s'].median()),
        'ev_r_at_zero_mean': float(ev['r_at_zero'].mean()),
        'ev_mi_mean': float(ev['mi'].mean()),
        'ct_r_peak_mean': float(ct['r_peak'].mean()),
        'ct_lag_peak_s_median': float(ct['lag_peak_s'].median()),
        'ct_r_at_zero_mean': float(ct['r_at_zero'].mean()),
        'ct_mi_mean': float(ct['mi'].mean()),
    }


def main():
    summary = pd.read_csv(os.path.join(EVENTS_DIR, 'extraction_summary.csv'))
    ok = summary[(summary['status'] == 'ok') & (summary['n_events'] >= 3)]
    tasks = []
    for _, r in ok.iterrows():
        ep = os.path.join(EVENTS_DIR, f'{r["subject_id"]}_sie_events.csv')
        if os.path.isfile(ep):
            tasks.append((r['subject_id'], ep))
    print(f"Subjects: {len(tasks)}")
    n_workers = int(os.environ.get('SIE_WORKERS', min(4, os.cpu_count() or 4)))
    print(f"Workers: {n_workers}")

    with Pool(n_workers) as pool:
        results = pool.map(process_subject, tasks)
    df = pd.DataFrame([r for r in results if r is not None])
    df.to_csv(os.path.join(OUT_DIR, 'sr1_sr3_coupling.csv'), index=False)
    print(f"Successful: {len(df)}")

    # Wilcoxon paired: event vs control per subject
    print(f"\n=== Paired Wilcoxon event vs control (per subject) ===")
    for metric in ['r_peak', 'r_at_zero', 'mi']:
        ev = df[f'ev_{metric}_mean']
        ct = df[f'ct_{metric}_mean']
        d = ev - ct
        if (d != 0).sum() < 5:
            continue
        s, p = wilcoxon(d.dropna())
        print(f"  {metric:10s}  event {ev.mean():+.4f} ± {ev.std():.4f}   "
              f"control {ct.mean():+.4f} ± {ct.std():.4f}   "
              f"Δ {d.mean():+.4f}   p = {p:.3g}")

    print(f"\n=== Lag at peak envelope correlation (per subject median) ===")
    print(f"  event lag: median {df['ev_lag_peak_s_median'].median():.2f}s  "
          f"IQR [{df['ev_lag_peak_s_median'].quantile(.25):.2f}, {df['ev_lag_peak_s_median'].quantile(.75):.2f}]")
    print(f"  control lag: median {df['ct_lag_peak_s_median'].median():.2f}s")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, metric, label in [
        (axes[0], 'r_peak', 'peak envelope correlation'),
        (axes[1], 'r_at_zero', 'envelope correlation @ lag 0'),
        (axes[2], 'mi', 'Tort PAC MI (SR1 phase → SR3 amp)'),
    ]:
        ev = df[f'ev_{metric}_mean']
        ct = df[f'ct_{metric}_mean']
        lo_x = min(ev.min(), ct.min())
        hi_x = max(ev.max(), ct.max())
        ax.hist(ct, bins=25, range=(lo_x, hi_x), color='gray', alpha=0.6,
                 label=f'control (mean {ct.mean():.4f})')
        ax.hist(ev, bins=25, range=(lo_x, hi_x), color='firebrick', alpha=0.6,
                 label=f'event (mean {ev.mean():.4f})')
        ax.axvline(ct.mean(), color='gray', ls='--', lw=1)
        ax.axvline(ev.mean(), color='firebrick', ls='--', lw=1)
        ax.set_xlabel(label)
        ax.set_ylabel('subjects')
        s, p = wilcoxon((ev - ct).dropna()) if (ev != ct).sum() > 5 else (np.nan, np.nan)
        ax.set_title(f'{label}\nevent − control Δ = {(ev-ct).mean():+.4f}  Wilcoxon p = {p:.3g}')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle('B34 — SR1 × SR3 coupling: envelope correlation + PAC',
                 y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'sr1_sr3_coupling.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {OUT_DIR}/sr1_sr3_coupling.png")


if __name__ == '__main__':
    main()
