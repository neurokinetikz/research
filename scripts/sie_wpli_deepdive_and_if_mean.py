#!/usr/bin/env python3
"""
A10 — wPLI deep-dive + central IF (not just dispersion).

Part A: central instantaneous frequency per event per phase.
  For each event, compute mean IF across channels within each A7 phase
  (baseline, preparatory desync, nadir, ignition rise, peak, decay).
  Report the distribution of mean IFs per phase — what frequency is the
  signal locked to at each moment?

  Also report IF_mean(t) as a new peri-onset stream.

Part B: wPLI deep-dive.
  1. Imaginary coherence (ICoh) as a cross-check — another VC-robust measure.
     If wPLI and ICoh both peak pre-nadir, that's strong evidence against VC.
  2. Pairwise wPLI split by scalp distance (near vs far channel pairs).
     Near pairs are more susceptible to residual VC effects. If the pre-nadir
     peak is driven by near pairs only, it's VC-residual. If far pairs
     contribute equally or more, it's genuine distributed coupling.

  Standard 10-20 montage positions are used for distance computation.

Output: 2 figures (IF analysis, wPLI analysis) + CSVs.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from multiprocessing import Pool
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.run_sie_extraction import load_lemon
from scripts.sie_perionset_triple_average import bootstrap_ci, bandpass
from scripts.sie_perionset_multistream import (
    F0, HALF_BW, R_BAND, PRE_SEC, POST_SEC, PAD_SEC, STEP_SEC, WIN_SEC,
    TGRID, find_nadir, inst_freq_channel,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'multistream')
EVENTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'exports_sie', 'lemon')
warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

# A7 phase boundaries (relative to nadir, t=0)
# Re-shifted from the t0_net-aligned A7 boundaries by +1.3 s.
PHASES = [
    ('baseline',            -6.0, -0.8),
    ('preparatory_desync',  -0.8, -0.2),
    ('nadir',               -0.2,  0.5),
    ('ignition_rise',        0.5,  2.5),
    ('peak',                 2.3,  2.7),
    ('decay',                2.7,  6.9),
]


# -------------------------------------------------------------------------
# Part A helpers
# -------------------------------------------------------------------------
def central_if(X_uV, fs):
    """Time-resolved mean-across-channels instantaneous frequency."""
    Xb = bandpass(X_uV, fs, R_BAND[0], R_BAND[1])
    IF = np.array([inst_freq_channel(Xb[ci], fs) for ci in range(Xb.shape[0])])
    # clip extreme values (Hilbert edge artifacts, etc.)
    IF = np.clip(IF, 5.0, 12.0)
    return IF  # (n_ch, n_samples)


# -------------------------------------------------------------------------
# Part B helpers
# -------------------------------------------------------------------------
def pair_distances(ch_names, fs=None):
    """Return (n_pairs, n_ch, n_ch) pairwise Euclidean distances on 10-20 sphere.
    Uses mne standard_1020 montage coords."""
    from mne.channels import make_standard_montage
    mon = make_standard_montage('standard_1020')
    pos_dict = {name: pos for name, pos in
                 zip(mon.ch_names, mon.get_positions()['ch_pos'].values())}
    # handle alias channels (T3/T4/T5/T6 etc.)
    aliases = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}
    n = len(ch_names)
    D = np.full((n, n), np.nan)
    pos = {}
    for c in ch_names:
        key = aliases.get(c, c)
        if key in pos_dict:
            pos[c] = pos_dict[key]
    for i, ci in enumerate(ch_names):
        for j, cj in enumerate(ch_names):
            if ci in pos and cj in pos:
                D[i, j] = np.linalg.norm(pos[ci] - pos[cj])
    return D  # meters; NaN if unknown channel


def wpli_window(ph_i, ph_j):
    """wPLI between two channels in a window."""
    cdiff = np.exp(1j * (ph_i - ph_j))
    imag = np.imag(cdiff)
    num = np.abs(np.mean(imag))
    den = np.mean(np.abs(imag))
    return (num / den) if den > 1e-12 else 0.0


def icoh_window(x_i, x_j, fs, f0=F0):
    """Imaginary coherence at f0 between two signals."""
    try:
        f_c, Cxy = signal.coherence(x_i, x_j, fs=fs, nperseg=len(x_i))
        # But coherence returns magnitude — need cross-spectrum
        f_c, Pxy = signal.csd(x_i, x_j, fs=fs, nperseg=len(x_i))
        f_c, Pxx = signal.welch(x_i, fs=fs, nperseg=len(x_i))
        f_c, Pyy = signal.welch(x_j, fs=fs, nperseg=len(x_i))
        k = int(np.argmin(np.abs(f_c - f0)))
        return float(abs(np.imag(Pxy[k])) / np.sqrt(Pxx[k] * Pyy[k] + 1e-30))
    except Exception:
        return np.nan


# -------------------------------------------------------------------------
# Per-subject worker
# -------------------------------------------------------------------------
def process_subject(args):
    sub_id, events_path = args
    try:
        events = pd.read_csv(events_path).dropna(subset=['t0_net'])
    except Exception:
        return None
    if len(events) == 0:
        return None
    try:
        raw = load_lemon(sub_id, condition='EC')
    except Exception:
        return None
    if raw is None:
        return None
    fs = raw.info['sfreq']
    X_all = raw.get_data() * 1e6
    t_end_rec = raw.times[-1]
    ch_names = raw.ch_names

    # Distance matrix for this subject's channels
    D = pair_distances(ch_names)
    # Separate near (<0.10 m) and far (>0.15 m) pairs on unit sphere scale.
    # The scale depends on MNE's head_size; at default 0.095 m radius,
    # near = <0.10 means adjacent channels; far = >0.15 cross-hemisphere.
    # We'll just compute median and use <median / >median split.
    offdiag = D[np.triu_indices_from(D, k=1)]
    offdiag = offdiag[np.isfinite(offdiag)]
    if len(offdiag) < 10:
        return None
    d_med = np.nanmedian(offdiag)

    # Accumulators
    if_mean_rows = []      # mean IF (across channels) at each time → then nadir-realigned
    if_mean_per_phase = {ph[0]: [] for ph in PHASES}  # per-event phase means (scalar)
    # wPLI/ICoh grand streams: near vs far pair means, peri-nadir
    wpli_near_rows, wpli_far_rows = [], []
    icoh_near_rows, icoh_far_rows = [], []

    # Pre-compute pair index arrays
    n_ch = len(ch_names)
    triu_i, triu_j = np.triu_indices(n_ch, k=1)
    pair_D = D[triu_i, triu_j]
    near_mask = pair_D < d_med
    far_mask  = pair_D >= d_med
    near_pairs = list(zip(triu_i[near_mask], triu_j[near_mask]))
    far_pairs  = list(zip(triu_i[far_mask], triu_j[far_mask]))
    # Limit pairs for speed
    import random
    rng = random.Random(42)
    if len(near_pairs) > 50: near_pairs = rng.sample(near_pairs, 50)
    if len(far_pairs)  > 50: far_pairs  = rng.sample(far_pairs,  50)

    nwin = int(round(WIN_SEC * fs))
    nstep = int(round(STEP_SEC * fs))

    for _, ev in events.iterrows():
        t0 = float(ev['t0_net'])
        lo = t0 - PRE_SEC - PAD_SEC
        hi = t0 + POST_SEC + PAD_SEC
        if lo < 0 or hi > t_end_rec:
            continue
        i0 = int(round(lo * fs)); i1 = int(round(hi * fs))
        X_seg = X_all[:, i0:i1]
        if X_seg.shape[1] < int(round((hi - lo) * fs * 0.95)):
            continue

        try:
            # Part A: central IF over time
            IF = central_if(X_seg, fs)
            # Part B: bandpass and phase per channel in R_BAND
            Xb = bandpass(X_seg, fs, R_BAND[0], R_BAND[1])
            ph = np.angle(signal.hilbert(Xb, axis=-1))
        except Exception:
            continue

        # Per-time-window aggregation
        centers = []
        if_mean_win = []
        wpli_near_win, wpli_far_win = [], []
        icoh_near_win, icoh_far_win = [], []
        for i in range(0, X_seg.shape[1] - nwin + 1, nstep):
            if_mean_win.append(float(np.nanmean(IF[:, i:i+nwin])))
            # wPLI by pair group
            wpli_n = [wpli_window(ph[a, i:i+nwin], ph[b, i:i+nwin])
                      for a, b in near_pairs]
            wpli_f = [wpli_window(ph[a, i:i+nwin], ph[b, i:i+nwin])
                      for a, b in far_pairs]
            wpli_near_win.append(float(np.mean(wpli_n)) if wpli_n else np.nan)
            wpli_far_win.append(float(np.mean(wpli_f)) if wpli_f else np.nan)
            # ICoh by pair group
            icoh_n = [icoh_window(Xb[a, i:i+nwin], Xb[b, i:i+nwin], fs)
                      for a, b in near_pairs]
            icoh_f = [icoh_window(Xb[a, i:i+nwin], Xb[b, i:i+nwin], fs)
                      for a, b in far_pairs]
            icoh_near_win.append(float(np.nanmean(icoh_n)) if icoh_n else np.nan)
            icoh_far_win.append(float(np.nanmean(icoh_f)) if icoh_f else np.nan)
            centers.append((i + nwin/2) / fs)

        centers = np.array(centers)
        if_mean_win = np.array(if_mean_win)
        wpli_near_win = np.array(wpli_near_win)
        wpli_far_win = np.array(wpli_far_win)
        icoh_near_win = np.array(icoh_near_win)
        icoh_far_win = np.array(icoh_far_win)

        # Nadir alignment — we need the 4 primary streams to compute nadir.
        # Fast proxy: use IF mean inverse (minimum IF is near nadir) — but
        # that's not quite right. Compute envelope/R/PLV/MSC here for nadir.
        # For simplicity, use the already-computed ones from multistream-style logic:
        from scripts.sie_perionset_multistream import compute_all_streams
        try:
            S = compute_all_streams(X_seg, fs)
        except Exception:
            continue
        rel = S['centers'] - PAD_SEC - PRE_SEC
        nadir = find_nadir(rel, S['env'], S['R'], S['PLV'], S['MSC'])
        if not np.isfinite(nadir):
            continue
        rel_my = centers - PAD_SEC - PRE_SEC - nadir  # relative to nadir

        if_mean_rows.append(np.interp(TGRID, rel_my, if_mean_win, left=np.nan, right=np.nan))
        wpli_near_rows.append(np.interp(TGRID, rel_my, wpli_near_win, left=np.nan, right=np.nan))
        wpli_far_rows.append(np.interp(TGRID, rel_my, wpli_far_win,   left=np.nan, right=np.nan))
        icoh_near_rows.append(np.interp(TGRID, rel_my, icoh_near_win, left=np.nan, right=np.nan))
        icoh_far_rows.append(np.interp(TGRID, rel_my, icoh_far_win,   left=np.nan, right=np.nan))

        # Per-phase IF means
        for phname, lo_ph, hi_ph in PHASES:
            mask = (rel_my >= lo_ph) & (rel_my <= hi_ph)
            if mask.any():
                if_mean_per_phase[phname].append(float(np.nanmean(if_mean_win[mask])))

    if not if_mean_rows:
        return None

    return {
        'subject_id': sub_id,
        'n_events': len(if_mean_rows),
        'IF_mean': np.nanmean(np.array(if_mean_rows), axis=0),
        'wPLI_near': np.nanmean(np.array(wpli_near_rows), axis=0),
        'wPLI_far':  np.nanmean(np.array(wpli_far_rows),  axis=0),
        'ICoh_near': np.nanmean(np.array(icoh_near_rows), axis=0),
        'ICoh_far':  np.nanmean(np.array(icoh_far_rows),  axis=0),
        'IF_per_phase_means': {ph: np.mean(if_mean_per_phase[ph])
                                 if if_mean_per_phase[ph] else np.nan
                                 for ph, _, _ in PHASES},
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    summary = pd.read_csv(os.path.join(EVENTS_DIR, 'extraction_summary.csv'))
    ok = summary[(summary['status'] == 'ok') & (summary['n_events'] >= 3)]
    tasks = []
    for _, r in ok.iterrows():
        events_path = os.path.join(EVENTS_DIR, f'{r["subject_id"]}_sie_events.csv')
        if os.path.isfile(events_path):
            tasks.append((r['subject_id'], events_path))
    print(f"Subjects: {len(tasks)}")

    import os as _os
    n_workers = int(_os.environ.get('SIE_WORKERS', min(30, _os.cpu_count() or 8)))
    print(f"  workers: {n_workers}")
    with Pool(n_workers) as pool:
        results = pool.map(process_subject, tasks)
    results = [r for r in results if r is not None]
    print(f"Successful: {len(results)}")

    # ---- Part A: IF mean ----
    IFmean_arr = np.array([r['IF_mean'] for r in results])
    IFm_m, IFm_lo, IFm_hi = bootstrap_ci(IFmean_arr, n_boot=500)

    # Per-phase IF means
    phase_rows = []
    for ph, _, _ in PHASES:
        vals = np.array([r['IF_per_phase_means'][ph] for r in results])
        vals = vals[np.isfinite(vals)]
        phase_rows.append({
            'phase': ph,
            'n_subj': len(vals),
            'IF_mean_Hz': float(np.mean(vals)),
            'IF_std_Hz':  float(np.std(vals)),
            'IF_25%':     float(np.percentile(vals, 25)),
            'IF_75%':     float(np.percentile(vals, 75)),
        })
    phase_df = pd.DataFrame(phase_rows)
    phase_df.to_csv(os.path.join(OUT_DIR, 'central_IF_per_phase.csv'), index=False)
    print("\nCentral IF per phase (Hz):")
    print(phase_df.to_string(index=False))

    # ---- Part B: wPLI/ICoh deep-dive ----
    streams = ['wPLI_near', 'wPLI_far', 'ICoh_near', 'ICoh_far']
    bootstats = {}
    for s in streams:
        arr = np.array([r[s] for r in results])
        m, lo, hi = bootstrap_ci(arr, n_boot=500)
        bootstats[s] = {'arr': arr, 'mean': m, 'lo': lo, 'hi': hi}
        pk = TGRID[np.argmax(m)]
        tr = TGRID[np.argmin(m)]
        print(f"  {s:12s}: peak {pk:+.2f}s ({m[np.argmax(m)]:.4f}), "
              f"trough {tr:+.2f}s ({m[np.argmin(m)]:.4f})")

    # Save combined CSV
    out_csv = {'t_rel': TGRID, 'IF_mean_Hz': IFm_m,
               'IF_mean_ci_lo': IFm_lo, 'IF_mean_ci_hi': IFm_hi}
    for s in streams:
        out_csv[f'{s}_mean'] = bootstats[s]['mean']
        out_csv[f'{s}_lo']   = bootstats[s]['lo']
        out_csv[f'{s}_hi']   = bootstats[s]['hi']
    pd.DataFrame(out_csv).to_csv(
        os.path.join(OUT_DIR, 'wpli_deepdive_and_IF.csv'), index=False)

    # ---- Figures ----
    # IF plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax = axes[0]
    ax.fill_between(TGRID, IFm_lo, IFm_hi, color='navy', alpha=0.25)
    ax.plot(TGRID, IFm_m, color='navy', lw=2, label='mean IF across channels')
    ax.axhline(F0, color='red', ls='--', lw=0.8, label=f'F₀ = {F0} Hz')
    ax.axvline(0, color='k', ls='--', lw=0.6)
    ax.set_ylabel('instantaneous frequency (Hz)')
    ax.set_title(f'A10 — Central IF at f₀ band peri-onset ({len(results)} subj)')
    ax.legend(fontsize=9)

    ax = axes[1]
    xs = np.arange(len(PHASES))
    ax.bar(xs, phase_df['IF_mean_Hz'], yerr=phase_df['IF_std_Hz']/np.sqrt(phase_df['n_subj']),
           capsize=4, color='slategray', edgecolor='k')
    ax.axhline(F0, color='red', ls='--', lw=0.8, label=f'F₀ = {F0} Hz')
    ax.set_xticks(xs)
    ax.set_xticklabels(phase_df['phase'], rotation=20, ha='right')
    ax.set_ylabel('mean IF ± SEM across subjects (Hz)')
    ax.set_title('Central IF by phase')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'central_IF_analysis.png'),
                dpi=120, bbox_inches='tight')
    plt.close()

    # wPLI deep-dive
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    specs = [
        ('wPLI_near', axes[0, 0], 'near pairs', 'orangered'),
        ('wPLI_far',  axes[0, 1], 'far pairs',  'steelblue'),
        ('ICoh_near', axes[1, 0], 'near pairs', 'orangered'),
        ('ICoh_far',  axes[1, 1], 'far pairs',  'steelblue'),
    ]
    for s, ax, sub, color in specs:
        d = bootstats[s]
        ax.fill_between(TGRID, d['lo'], d['hi'], color=color, alpha=0.25)
        ax.plot(TGRID, d['mean'], color=color, lw=2)
        ax.axvline(0, color='k', ls='--', lw=0.6)
        peak_t = TGRID[np.argmax(d['mean'])]
        ax.axvline(peak_t, color='red', ls=':', lw=0.5)
        prefix = 'wPLI' if 'wPLI' in s else 'ICoh'
        ax.set_title(f'{prefix} — {sub}  (peak {peak_t:+.2f}s)')
        ax.set_ylabel(s)
    for ax in axes[-1]:
        ax.set_xlabel('time relative to nadir (s)')
    fig.suptitle(f'A10 — VC-robust coupling, near vs far pair split\n'
                 f'{len(results)} subjects · pair-distance split at median',
                 fontsize=12, y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'wpli_deepdive.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved figures to {OUT_DIR}")


if __name__ == '__main__':
    main()
