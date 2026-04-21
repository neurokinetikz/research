#!/usr/bin/env python3
"""
A6b + A5g — Redefine onset as joint dip minimum (env + R + PLV + MSC),
then run narrow 4-s peri-onset FOOOF comparing three event sources.

Part 1 (A6b): add MSC as a fourth stream; compute joint-dip-minimum onset
for each event using all four streams.

Part 2 (A5g): for each event, run FOOOF on a narrow 4-s window centered on
the joint-dip onset, comparing against:
  - current event (4s centered on t0_net)
  - random 4s window

Compute ratio precision (MAE vs φⁿ) for each.

MSC stream: mean magnitude-squared coherence across channels vs median
reference, at f0 (7.83 Hz), with 1-s sliding window.
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
from scripts.sie_perionset_triple_average import (
    bandpass, PRE_SEC, POST_SEC, PAD_SEC,
)
from scripts.sie_composite_vs_current_precision import (
    harmonics_in_window, RATIO_TARGETS, CAP_PER_SUB,
)
from lib.mne_to_ignition import mne_raw_to_ignition_df

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'perionset')
OUT_DIR_FOOOF = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                              'images', 'composite_detector')
EVENTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'exports_sie', 'lemon')

warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

F0 = 7.83
HALF_BW = 0.6
R_BAND = (7.2, 8.4)
STEP_SEC = 0.1
WIN_SEC = 1.0

# Narrow-window FOOOF parameters
NARROW_WIN_SEC = 4.0
NARROW_HALF = NARROW_WIN_SEC / 2

DIP_WINDOW = (-3.0, 0.4)


def compute_streams_4way(X_uV, fs):
    """Return time grid + envelope z, R, PLV, MSC (all at 10 Hz step)."""
    # Envelope at F0
    y = X_uV.mean(axis=0)
    yb = bandpass(y, fs, F0 - HALF_BW, F0 + HALF_BW)
    env = np.abs(signal.hilbert(yb))
    env_z_full = zscore(env, nan_policy='omit')

    # Phase streams for R, PLV, and for MSC we use bandpassed data
    Xb = bandpass(X_uV, fs, R_BAND[0], R_BAND[1])
    ph = np.angle(signal.hilbert(Xb, axis=-1))
    ref = np.median(Xb, axis=0)  # median-reference (bandpassed)
    ph_ref = np.angle(signal.hilbert(ref))
    dphi = ph - ph_ref[None, :]

    # MSC must use RAW (unfiltered) signal vs raw median reference — filtering
    # all channels through the same passband before coherence saturates MSC at 1.
    ref_raw = np.median(X_uV, axis=0)

    nwin = int(round(WIN_SEC * fs))
    nstep = int(round(STEP_SEC * fs))
    nperseg_msc = max(int(round(0.5 * fs)), 32)  # multiple segs per 1-s window
    centers = []
    env_vals, Rv, Pv, Mv = [], [], [], []
    for i in range(0, X_uV.shape[1] - nwin + 1, nstep):
        seg_ph = ph[:, i:i+nwin]
        R_t = np.abs(np.mean(np.exp(1j * seg_ph), axis=0))
        Rv.append(float(np.mean(R_t)))
        pseg = dphi[:, i:i+nwin]
        plv = np.abs(np.mean(np.exp(1j * pseg), axis=1))
        Pv.append(float(np.mean(plv)))
        env_vals.append(float(np.mean(env_z_full[i:i+nwin])))
        # MSC at F0 on raw signals
        ref_seg_raw = ref_raw[i:i+nwin]
        msc_per_ch = []
        for ci in range(X_uV.shape[0]):
            try:
                f_c, Cxy = signal.coherence(X_uV[ci, i:i+nwin], ref_seg_raw, fs=fs,
                                              nperseg=min(nperseg_msc, nwin))
                k = int(np.argmin(np.abs(f_c - F0)))
                msc_per_ch.append(Cxy[k])
            except Exception:
                pass
        Mv.append(float(np.mean(msc_per_ch)) if msc_per_ch else np.nan)
        centers.append((i + nwin/2) / fs)
    return (np.array(centers), np.array(env_vals),
            np.array(Rv), np.array(Pv), np.array(Mv))


def find_joint_dip(t_rel, zE, zR, zP, zM):
    """Joint-dip-minimum onset: argmin of zE+zR+zP+zM in DIP_WINDOW."""
    mask = (t_rel >= DIP_WINDOW[0]) & (t_rel <= DIP_WINDOW[1])
    if not mask.any():
        return np.nan
    # z-score each stream against pre-dip baseline (-5 to -3 s)
    base_mask = (t_rel >= -5.0) & (t_rel < -3.0)
    def zc(x):
        if base_mask.sum() < 3:
            return x - np.nanmean(x)
        mu = np.nanmean(x[base_mask]); sd = np.nanstd(x[base_mask])
        if not np.isfinite(sd) or sd < 1e-9: sd = 1.0
        return (x - mu) / sd
    s = zc(zE) + zc(zR) + zc(zP) + zc(zM)
    s_m = np.where(mask, s, np.inf)
    idx = int(np.nanargmin(s_m))
    return float(t_rel[idx])


def process_subject(args):
    sub_id, events_path, seed = args
    try:
        events = pd.read_csv(events_path).dropna(subset=['t0_net'])
    except Exception:
        return None
    if len(events) < 5:
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

    df, eeg_channels = mne_raw_to_ignition_df(raw)

    rng = np.random.default_rng(seed)
    event_list = events.head(CAP_PER_SUB).copy()

    # Step 1: compute joint-dip onset for each event
    onsets = []
    for _, ev in event_list.iterrows():
        t0 = float(ev['t0_net'])
        lo = t0 - PRE_SEC - PAD_SEC
        hi = t0 + POST_SEC + PAD_SEC
        if lo < 0 or hi > t_end_rec:
            onsets.append(np.nan)
            continue
        i0 = int(round(lo * fs)); i1 = int(round(hi * fs))
        X_seg = X_all[:, i0:i1]
        if X_seg.shape[1] < int(round((hi - lo) * fs * 0.95)):
            onsets.append(np.nan)
            continue
        try:
            t_rel_offset, env, R, P, M = compute_streams_4way(X_seg, fs)
            rel = t_rel_offset - PAD_SEC - PRE_SEC
            dip = find_joint_dip(rel, env, R, P, M)
            onsets.append(t0 + dip if np.isfinite(dip) else np.nan)
        except Exception:
            onsets.append(np.nan)
    event_list['t_joint_dip'] = onsets

    # Step 2: for each event, run FOOOF in 3 conditions (4-s windows)
    rows = []
    recording_duration = t_end_rec
    for _, ev in event_list.iterrows():
        t0 = float(ev['t0_net'])
        tj = float(ev['t_joint_dip']) if np.isfinite(ev['t_joint_dip']) else np.nan

        # Sources
        sources = {
            'current_4s': t0,
            'joint_dip_4s': tj,
            'random_4s': rng.uniform(NARROW_HALF + 1.0, recording_duration - NARROW_HALF - 1.0),
        }
        for src, t_c in sources.items():
            if not np.isfinite(t_c):
                continue
            lo_t = t_c - NARROW_HALF
            hi_t = t_c + NARROW_HALF
            if lo_t < 0 or hi_t > recording_duration:
                continue
            h = harmonics_in_window(df, eeg_channels, fs, t_c)
            if h is None:
                continue
            rec = {'subject_id': sub_id, 'source': src, 't_center': t_c, **h}
            for name in RATIO_TARGETS:
                num, den = name.split('/')
                try:
                    v = rec[num] / rec[den]
                except Exception:
                    v = np.nan
                if not np.isfinite(v): v = np.nan
                rec[name] = v
            rows.append(rec)
    return {
        'sub_id': sub_id,
        'onset_table': event_list[['t0_net', 't_joint_dip']].to_dict('records'),
        'ratio_rows': rows,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(OUT_DIR_FOOOF, exist_ok=True)

    summary = pd.read_csv(os.path.join(EVENTS_DIR, 'extraction_summary.csv'))
    ok = summary[(summary['status'] == 'ok') & (summary['n_events'] >= 5)]
    tasks = []
    for i, (_, r) in enumerate(ok.iterrows()):
        events_path = os.path.join(EVENTS_DIR, f'{r["subject_id"]}_sie_events.csv')
        if os.path.isfile(events_path):
            tasks.append((r['subject_id'], events_path, i + 1))
    print(f"Subjects: {len(tasks)}, narrow window {NARROW_WIN_SEC}s, "
          f"cap {CAP_PER_SUB} events/subject")

    with Pool(8) as pool:
        results = pool.map(process_subject, tasks)
    results = [r for r in results if r is not None]

    # Aggregate joint-dip onsets
    onset_df = pd.DataFrame([
        {'subject_id': r['sub_id'], **row}
        for r in results for row in r['onset_table']
        if np.isfinite(row.get('t_joint_dip', np.nan))
    ])
    onset_df['dip_offset'] = onset_df['t_joint_dip'] - onset_df['t0_net']
    print(f"\nJoint-dip onset stats (vs t0_net, n={len(onset_df)}):")
    print(f"  median offset: {onset_df['dip_offset'].median():+.3f} s")
    print(f"  IQR: [{onset_df['dip_offset'].quantile(0.25):+.3f}, "
          f"{onset_df['dip_offset'].quantile(0.75):+.3f}] s")

    # Aggregate FOOOF ratios
    ratio_rows = []
    for r in results:
        ratio_rows.extend(r['ratio_rows'])
    df_r = pd.DataFrame(ratio_rows)
    df_r.to_csv(os.path.join(OUT_DIR_FOOOF, 'narrow_fooof_ratios.csv'), index=False)
    print(f"\nFOOOF events: {len(df_r)}")

    # Ratio precision summary
    print(f"\n=== Narrow 4-s FOOOF ratio precision ===\n")
    summ_rows = []
    for name, target in RATIO_TARGETS.items():
        for src in ['current_4s', 'joint_dip_4s', 'random_4s']:
            sub = df_r[df_r['source'] == src][name].dropna()
            if len(sub) == 0:
                continue
            mae = float(np.mean(np.abs(sub - target)))
            mean = float(np.mean(sub))
            std = float(np.std(sub))
            summ_rows.append({'ratio': name, 'source': src, 'target': target,
                              'n': int(len(sub)), 'mean': mean, 'std': std,
                              'MAE_vs_phi': mae, 'bias': mean - target})
    summ = pd.DataFrame(summ_rows)
    print(summ.to_string(index=False))
    summ.to_csv(os.path.join(OUT_DIR_FOOOF, 'narrow_fooof_precision_summary.csv'),
                 index=False)

    # --- Figure 1: joint-dip onset distribution ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(onset_df['dip_offset'], bins=60, color='teal', edgecolor='k', lw=0.3)
    ax.axvline(0, color='k', ls='--', lw=0.6, label='t₀_net')
    ax.axvline(onset_df['dip_offset'].median(), color='red', lw=1.2,
               label=f"median {onset_df['dip_offset'].median():+.2f} s")
    ax.set_xlabel('joint-dip onset − t₀_net (s)')
    ax.set_ylabel('events')
    ax.set_title(f'A6b — Joint-dip onset (env+R+PLV+MSC) vs t₀_net\n'
                  f'{onset_df["subject_id"].nunique()} subjects · {len(onset_df)} events')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'joint_dip_onset_distribution.png'),
                dpi=120, bbox_inches='tight')
    plt.close()

    # --- Figure 2: narrow FOOOF ratio precision 3-way ---
    colors = {'current_4s': 'steelblue', 'joint_dip_4s': 'teal',
              'random_4s': 'gray'}
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ci, (name, target) in enumerate(RATIO_TARGETS.items()):
        ax = axes[ci]
        src_names, maes = [], []
        for src in ['current_4s', 'joint_dip_4s', 'random_4s']:
            vals = df_r[df_r['source'] == src][name].dropna()
            if len(vals) == 0: continue
            src_names.append(src)
            maes.append(float(np.mean(np.abs(vals - target))))
        ax.bar(src_names, maes, color=[colors[s] for s in src_names])
        ax.set_ylabel(f'MAE vs φⁿ = {target:.3f}')
        ax.set_title(name)
        for si, (s, m) in enumerate(zip(src_names, maes)):
            ax.text(si, m, f'{m:.3f}', ha='center', va='bottom', fontsize=9)
        ax.tick_params(axis='x', rotation=15)
    plt.suptitle(f'A5g — Narrow 4-s FOOOF ratio precision\n'
                 f'LEMON EC · {df_r["subject_id"].nunique()} subjects',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR_FOOOF, 'narrow_fooof_precision.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved figures to:")
    print(f"  {OUT_DIR}/joint_dip_onset_distribution.png")
    print(f"  {OUT_DIR_FOOOF}/narrow_fooof_precision.png")


if __name__ == '__main__':
    main()
