#!/usr/bin/env python3
"""
B47 — Cross-cohort replication of the posterior-vs-anterior SR1 contrast.

The surviving individual-level claim from B45/B46 is:
  "Event-locked SR1 engagement is posterior-dominant within-subject."

Test: does this hold on HBN R4 and TDBRAIN using the same event-locked
posterior-vs-anterior SR1 ratio pipeline?

Uses ALL events for each subject (template_rho is LEMON-only; Q4-restriction
caveats noted in report). If the dominance replicates on all-event
aggregates, it is strong.
"""
from __future__ import annotations
import os
import sys
import glob as _glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import wilcoxon, mannwhitneyu
from multiprocessing import Pool
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.run_sie_extraction import load_lemon, load_hbn, load_tdbrain

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'coupling')
EVENTS_ROOT = os.path.join(os.path.dirname(__file__), '..', 'exports_sie')
HBN_DATA = '/Volumes/T9/hbn_data'
os.makedirs(OUT_DIR, exist_ok=True)
warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

SR1_BAND = (7.0, 8.3)
EV_WIN_SEC = 4.0
EV_NFFT_MULT = 16
EV_LAG_S = 1.0


def is_posterior(ch):
    n = ch.upper()
    if n.startswith('FP'):
        return False
    return (any(n.startswith(p) for p in ('O', 'PO', 'P', 'TP')) or
            n in ('T7', 'T8', 'T5', 'T6'))


def is_anterior(ch):
    n = ch.upper()
    if is_posterior(ch):
        return False
    return any(n.startswith(p) for p in ('F', 'AF', 'FP'))


def region_picks_from_raw(raw):
    """Return (posterior_idx, anterior_idx). First try 10-20 label prefixes;
    if none match (e.g. EGI E-numbered HBN channels), fall back to spatial
    selection using the standard montage y-coordinate (anteroposterior axis,
    posterior = y < 0, anterior = y > 0 in RAS head coords)."""
    ch_names = raw.ch_names
    post_idx = [i for i, ch in enumerate(ch_names) if is_posterior(ch)]
    ant_idx = [i for i, ch in enumerate(ch_names) if is_anterior(ch)]
    if len(post_idx) >= 5 and len(ant_idx) >= 5:
        return post_idx, ant_idx
    # Fallback — apply montage and use y-coordinate
    for mname in ('GSN-HydroCel-129', 'GSN-HydroCel-128', 'standard_1020'):
        try:
            montage = mne.channels.make_standard_montage(mname)
            raw_copy = raw.copy().set_montage(montage, match_case=False,
                                               on_missing='ignore',
                                               verbose=False)
            pos = raw_copy.get_montage().get_positions()['ch_pos']
            ys = np.array([pos[ch][1] if ch in pos and np.all(np.isfinite(pos[ch]))
                           else np.nan for ch in ch_names])
            if np.isnan(ys).all():
                continue
            # Exclude channels without position; compute median y as midline
            valid = ~np.isnan(ys)
            if valid.sum() < 20:
                continue
            mid = np.nanmedian(ys[valid])
            # Posterior = y < mid - small margin; anterior = y > mid + margin
            margin = 0.02
            post = [i for i in range(len(ch_names))
                    if valid[i] and ys[i] < mid - margin]
            ant = [i for i in range(len(ch_names))
                   if valid[i] and ys[i] > mid + margin]
            if len(post) >= 5 and len(ant) >= 5:
                return post, ant
        except Exception:
            continue
    return post_idx, ant_idx


def welch_one(seg, fs, nfft):
    seg = seg - np.mean(seg)
    w = signal.windows.hann(len(seg))
    wp = np.sum(w ** 2)
    X = np.fft.rfft(seg * w, nfft)
    psd = (np.abs(X) ** 2) / (fs * wp)
    psd[1:-1] *= 2.0
    return psd


def region_ratio(raw, events, picks_idx):
    if len(picks_idx) < 3 or raw is None:
        return np.nan
    fs = raw.info['sfreq']
    if fs < 40:
        return np.nan
    X = raw.get_data() * 1e6
    sig1d = X[picks_idx].mean(axis=0)
    nper = int(round(EV_WIN_SEC * fs))
    nhop = int(round(1.0 * fs))
    nfft = nper * EV_NFFT_MULT
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    mask = (freqs >= SR1_BAND[0]) & (freqs <= SR1_BAND[1])

    base_rows = []
    for i in range(0, len(sig1d) - nper + 1, nhop):
        base_rows.append(welch_one(sig1d[i:i+nper], fs, nfft)[mask])
    if len(base_rows) < 10:
        return np.nan
    baseline = np.nanmedian(np.array(base_rows), axis=0)

    ev_rows = []
    for _, ev in events.iterrows():
        tc = float(ev['t0_net']) + EV_LAG_S
        i0 = int(round((tc - EV_WIN_SEC / 2) * fs))
        i1 = i0 + nper
        if i0 < 0 or i1 > len(sig1d):
            continue
        ev_rows.append(welch_one(sig1d[i0:i1], fs, nfft)[mask])
    if not ev_rows:
        return np.nan
    ev_avg = np.nanmean(np.array(ev_rows), axis=0)
    ratio = (ev_avg + 1e-20) / (baseline + 1e-20)
    return float(ratio[int(np.argmax(ratio))])


QUALITY_TOP_Q = float(os.environ.get('QUALITY_TOP_Q', '0.75'))
QUALITY_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs',
                            'schumann', 'images', 'quality')
_QUALITY_CACHE = {}


def _load_quality(cohort):
    if cohort in _QUALITY_CACHE:
        return _QUALITY_CACHE[cohort]
    if cohort == 'lemon':
        p = os.path.join(QUALITY_DIR, 'per_event_quality.csv')
    else:
        p = os.path.join(QUALITY_DIR, f'per_event_quality_{cohort}.csv')
    q = pd.read_csv(p) if os.path.isfile(p) else None
    if q is not None and 'template_rho' in q.columns:
        q = q[['subject_id', 't0_net', 'template_rho']].copy()
        q['t0_round'] = q['t0_net'].round(3)
        q = q.dropna(subset=['template_rho'])
    _QUALITY_CACHE[cohort] = q
    return q


def process_subject(args):
    cohort, sub_id, events_path, load_arg = args
    try:
        events = pd.read_csv(events_path).dropna(subset=['t0_net'])
    except Exception:
        return None
    if len(events) < 3:
        return None
    quality = _load_quality(cohort)
    if quality is not None:
        qsub = quality[quality['subject_id'] == sub_id]
        if len(qsub) >= 4:
            thr = qsub['template_rho'].quantile(QUALITY_TOP_Q)
            q4_times = set(qsub.loc[qsub['template_rho'] >= thr,
                                     't0_round'].values)
            events['t0_round'] = events['t0_net'].round(3)
            events = events[events['t0_round'].isin(q4_times)]
    if len(events) < 1:
        return None
    try:
        if cohort.startswith('hbn'):
            raw = load_hbn(load_arg)
        elif cohort == 'tdbrain':
            raw = load_tdbrain(sub_id, condition='EC')
        elif cohort == 'lemon':
            raw = load_lemon(sub_id, condition='EC')
        else:
            return None
    except Exception:
        return None
    if raw is None:
        return None
    post_idx, ant_idx = region_picks_from_raw(raw)
    if len(post_idx) < 5 or len(ant_idx) < 5:
        return None
    post_r = region_ratio(raw, events, post_idx)
    ant_r = region_ratio(raw, events, ant_idx)
    if np.isnan(post_r) or np.isnan(ant_r):
        return None
    return {
        'cohort': cohort,
        'subject_id': sub_id,
        'sr1_ratio_posterior': post_r,
        'sr1_ratio_anterior': ant_r,
        'sr1_contrast': post_r - ant_r,
        'n_events': len(events),
        'n_post_chs': len(post_idx),
        'n_ant_chs': len(ant_idx),
    }


def build_tasks_lemon():
    summary_csv = os.path.join(EVENTS_ROOT, 'lemon', 'extraction_summary.csv')
    summary = pd.read_csv(summary_csv)
    ok = summary[(summary['status']=='ok') & (summary['n_events']>=3)]
    tasks = []
    for _, r in ok.iterrows():
        sub = r['subject_id']
        ev = os.path.join(EVENTS_ROOT, 'lemon', f'{sub}_sie_events.csv')
        if os.path.isfile(ev):
            tasks.append(('lemon', sub, ev, None))
    return tasks


def build_tasks_hbn_release(release):
    """Build tasks for a single HBN release (R1/R2/R3/R4/R6)."""
    events_key = f'hbn_{release}'
    summary_csv = os.path.join(EVENTS_ROOT, events_key,
                                'extraction_summary.csv')
    if not os.path.isfile(summary_csv):
        return []
    summary = pd.read_csv(summary_csv)
    ok = summary[(summary['status']=='ok') & (summary['n_events']>=3)]
    release_dir = os.path.join(HBN_DATA, f'cmi_bids_{release}')
    tasks = []
    for _, r in ok.iterrows():
        sub = r['subject_id']
        ev = os.path.join(EVENTS_ROOT, events_key, f'{sub}_sie_events.csv')
        if not os.path.isfile(ev):
            continue
        set_path = os.path.join(release_dir, sub, 'eeg',
                                f'{sub}_task-RestingState_eeg.set')
        if not os.path.isfile(set_path):
            cand = _glob.glob(os.path.join(release_dir, sub, 'eeg',
                                           f'{sub}_task-RestingState_eeg.set'))
            if not cand:
                continue
            set_path = cand[0]
        # Cohort tag encodes release for downstream
        tasks.append((f'hbn_{release}', sub, ev, set_path))
    return tasks


def build_tasks_hbn_r4():
    return build_tasks_hbn_release('R4')


def build_tasks_hbn_all():
    tasks = []
    for rel in ('R1', 'R2', 'R3', 'R4', 'R6'):
        tasks += build_tasks_hbn_release(rel)
    return tasks


def build_tasks_tdbrain():
    summary_csv = os.path.join(EVENTS_ROOT, 'tdbrain', 'extraction_summary.csv')
    summary = pd.read_csv(summary_csv)
    ok = summary[(summary['status']=='ok') & (summary['n_events']>=3)]
    tasks = []
    for _, r in ok.iterrows():
        sub = r['subject_id']
        ev = os.path.join(EVENTS_ROOT, 'tdbrain', f'{sub}_sie_events.csv')
        if os.path.isfile(ev):
            tasks.append(('tdbrain', sub, ev, None))
    return tasks


def main():
    tasks = []
    tasks += build_tasks_lemon()
    tasks += build_tasks_hbn_all()
    tasks += build_tasks_tdbrain()
    print(f"Total subject-cohort tasks: {len(tasks)}")

    n_workers = int(os.environ.get('SIE_WORKERS', min(4, os.cpu_count() or 4)))
    with Pool(n_workers) as pool:
        results = pool.map(process_subject, tasks)
    results = [r for r in results if r is not None]
    df = pd.DataFrame(results)
    out_csv = os.path.join(OUT_DIR, 'posterior_sr1_crosscohort.csv')
    df.to_csv(out_csv, index=False)
    print(f"Successful rows: {len(df)}")
    if len(df) == 0:
        print("No results — aborting.")
        return

    # Add a 'hbn_all' pooled category for reporting
    df_hbn_all = df[df['cohort'].str.startswith('hbn')].copy()
    df_hbn_all['cohort'] = 'hbn_all'
    df = pd.concat([df, df_hbn_all], ignore_index=True)
    df.to_csv(out_csv, index=False)

    # ===== per-cohort stats =====
    cohorts = ['lemon', 'hbn_R1', 'hbn_R2', 'hbn_R3', 'hbn_R4',
               'hbn_R6', 'hbn_all', 'tdbrain']
    print(f"\n{'cohort':<10}{'n':>5}{'post med':>12}{'ant med':>10}"
          f"{'contrast':>11}{'% p>a':>8}{'Wilcoxon p':>14}")
    print('-' * 72)
    rows = []
    for c in cohorts:
        sub = df[df['cohort'] == c]
        if len(sub) == 0:
            continue
        contrast = sub['sr1_contrast'].values
        pm = np.median(sub['sr1_ratio_posterior'])
        am = np.median(sub['sr1_ratio_anterior'])
        cm = np.median(contrast)
        pct = (contrast > 0).mean() * 100
        p = wilcoxon(contrast).pvalue if len(contrast) > 10 else np.nan
        rows.append((c, len(sub), pm, am, cm, pct, p))
        print(f"{c:<10}{len(sub):>5}{pm:>12.3f}{am:>10.3f}"
              f"{cm:>+11.3f}{pct:>7.0f}%{p:>14.2g}")

    # ===== figure =====
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    col = {'lemon': '#8c1a1a', 'hbn_all': '#1a9641', 'tdbrain': '#2b5fb8'}
    titles = {'lemon': 'LEMON', 'hbn_all': 'HBN (R1+R2+R3+R4+R6)',
              'tdbrain': 'TDBRAIN'}
    main_cohorts = ['lemon', 'hbn_all', 'tdbrain']

    for ax, c in zip(axes, main_cohorts):
        sub = df[df['cohort'] == c]
        if len(sub) == 0:
            ax.set_axis_off(); continue
        post = sub['sr1_ratio_posterior'].values
        ant = sub['sr1_ratio_anterior'].values
        contrast = post - ant
        pct = (contrast > 0).mean() * 100
        p = wilcoxon(contrast).pvalue if len(contrast) > 10 else np.nan
        ax.scatter(ant, post, s=24, alpha=0.55, color=col[c],
                   edgecolor='k', lw=0.3)
        lim = [min(ant.min(), post.min()) - 0.2,
               max(ant.max(), post.max()) + 0.2]
        ax.plot(lim, lim, 'k--', lw=1, alpha=0.6)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel('anterior SR1 ratio (×)')
        ax.set_ylabel('posterior SR1 ratio (×)')
        ax.set_title(f'{titles[c]}\nn={len(sub)}  post>ant {pct:.0f}%  '
                     f'Wilcoxon p={p:.1g}',
                     loc='left', fontsize=10, fontweight='bold')
        ax.grid(alpha=0.3)

    fig.suptitle('B47 — Cross-cohort posterior-vs-anterior SR1 dominance '
                 '(per-subject)', fontsize=12, y=1.02)
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, 'posterior_sr1_crosscohort.png')
    plt.savefig(out_png, dpi=180, bbox_inches='tight')
    plt.savefig(out_png.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_png}")


if __name__ == '__main__':
    main()
