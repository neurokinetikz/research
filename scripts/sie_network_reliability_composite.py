#!/usr/bin/env python3
"""
B45 re-run on composite v2 detector.

Within-subject reliability of network-identity topographies. Per composite-
event subject, event-locked topographies for Q4 and Q1 events at SR1, β16,
SR3. Three tests:

  1. Subject-to-group similarity (leave-one-out): ρ between subject Q4 topo
     and LOO group Q4 mean. High median + high % > 0.5 → per-subject
     reliable.
  2. Within-subject ρ(Q4, Q1) per band: is the cohort-level Q4×Q1 anti-
     correlation (if any) present individually?
  3. β16 hemispheric lateralization per subject.

Envelope B45: SR1 subject-to-group ρ +0.52 (55% > 0.5) — RELIABLE; β16
−0.05 (1% > 0.5) — NOT reliable; SR3 +0.03 (6%) — marginal. The β16 Q4-vs-
Q1 network-identity claim (B43) was retracted as not individually reliable.

Cohort-parameterized.

Usage:
    python scripts/sie_network_reliability_composite.py --cohort lemon
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import spearmanr, wilcoxon
from multiprocessing import Pool
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.run_sie_extraction import (
    load_lemon, load_dortmund, load_srm, load_tdbrain,
    load_chbmp, load_hbn_by_subject,
)

warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

EV_WIN_SEC = 4.0
EV_NFFT_MULT = 16
EV_LAG_S = 1.0
BANDS = {'SR1': (7.0, 8.3), 'β16': (14.5, 17.5), 'SR3': (19.5, 20.4)}

ROOT = os.path.join(os.path.dirname(__file__), '..')


def cohort_config(cohort):
    events = os.path.join(ROOT, 'exports_sie', f'{cohort}_composite')
    qual = os.path.join(ROOT, 'outputs', 'schumann', 'images', 'quality',
                        f'per_event_quality_{cohort}_composite.csv')
    if cohort == 'lemon':
        return load_lemon, {'condition': 'EC'}, events, qual
    if cohort == 'lemon_EO':
        return load_lemon, {'condition': 'EO'}, events, qual
    if cohort == 'tdbrain':
        return load_tdbrain, {'condition': 'EC'}, events, qual
    if cohort == 'tdbrain_EO':
        return load_tdbrain, {'condition': 'EO'}, events, qual
    if cohort == 'srm':
        return load_srm, {}, events, qual
    if cohort.startswith('dortmund_'):
        return load_dortmund, {}, events, qual
    if cohort == 'dortmund':
        return load_dortmund, {}, events, qual
    if cohort == 'chbmp':
        return load_chbmp, {}, events, qual
    if cohort.startswith('hbn_'):
        return load_hbn_by_subject, {'release': cohort.split('_', 1)[1]}, events, qual
    raise ValueError(f"unsupported cohort {cohort!r}")


def welch_one(seg, fs, nfft):
    seg = seg - np.mean(seg)
    w = signal.windows.hann(len(seg))
    wp = np.sum(w ** 2)
    X = np.fft.rfft(seg * w, nfft)
    psd = (np.abs(X) ** 2) / (fs * wp)
    psd[1:-1] *= 2.0
    return psd


_LOADER = None
_LOADER_KW = None


def _init_worker(loader_name, loader_kw):
    global _LOADER, _LOADER_KW
    _LOADER_KW = loader_kw
    _LOADER = {
        'load_lemon': load_lemon,
        'load_tdbrain': load_tdbrain,
        'load_srm': load_srm,
        'load_dortmund': load_dortmund,
        'load_chbmp': load_chbmp,
        'load_hbn_by_subject': load_hbn_by_subject,
    }[loader_name]


def process_subject(args):
    sub_id, events_path, quality_csv = args
    try:
        events = pd.read_csv(events_path).dropna(subset=['t0_net'])
    except Exception:
        return None
    try:
        qual = pd.read_csv(quality_csv).dropna(subset=['template_rho']).copy()
        qual['rho_q'] = pd.qcut(qual['template_rho'], 4,
                                 labels=['Q1','Q2','Q3','Q4'])
        q_sub = qual[qual['subject_id'] == sub_id][['t0_net','rho_q']].copy()
        q_sub['t0_round'] = q_sub['t0_net'].round(3)
        events['t0_round'] = events['t0_net'].round(3)
        events = events.merge(q_sub[['t0_round','rho_q']], on='t0_round', how='left')
        events = events.dropna(subset=['rho_q'])
    except Exception:
        return None
    q4 = events[events['rho_q']=='Q4']
    q1 = events[events['rho_q']=='Q1']
    if len(q4) < 1 or len(q1) < 1:
        return None
    try:
        raw = _LOADER(sub_id, **_LOADER_KW)
    except Exception:
        return None
    if raw is None:
        return None
    fs = raw.info['sfreq']
    if fs < 40:
        return None
    ch_names = raw.ch_names
    n_ch = len(ch_names)
    X = raw.get_data() * 1e6
    nperseg = int(round(EV_WIN_SEC * fs))
    nhop = int(round(1.0 * fs))
    nfft = nperseg * EV_NFFT_MULT
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    masks = {k: (freqs>=lo)&(freqs<=hi) for k,(lo,hi) in BANDS.items()}

    base_cnt = 0
    base = np.zeros((n_ch, len(freqs)))
    for i in range(0, X.shape[1] - nperseg + 1, nhop):
        for c in range(n_ch):
            base[c] += welch_one(X[c, i:i+nperseg], fs, nfft)
        base_cnt += 1
    if base_cnt < 10:
        return None
    base /= base_cnt

    def ev_psd(sub_events):
        ev = np.zeros((n_ch, len(freqs))); cnt = 0
        for _, e in sub_events.iterrows():
            tc = float(e['t0_net']) + EV_LAG_S
            i0 = int(round((tc - EV_WIN_SEC/2)*fs)); i1 = i0 + nperseg
            if i0 < 0 or i1 > X.shape[1]: continue
            for c in range(n_ch):
                ev[c] += welch_one(X[c,i0:i1], fs, nfft)
            cnt += 1
        return (ev/cnt) if cnt > 0 else None

    q4_psd = ev_psd(q4); q1_psd = ev_psd(q1)
    if q4_psd is None or q1_psd is None:
        return None

    def topo(ev):
        out = {}
        for name, m in masks.items():
            r = np.zeros(n_ch)
            for c in range(n_ch):
                r[c] = np.nanmax(ev[c,m]) / (np.nanmean(base[c,m]) + 1e-20)
            out[name] = r
        return out

    return {'subject_id': sub_id,
            'ch_names': ch_names,
            'q4_topo': topo(q4_psd),
            'q1_topo': topo(q1_psd)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', default='lemon')
    ap.add_argument('--workers', type=int,
                    default=int(os.environ.get('SIE_WORKERS',
                                                min(4, os.cpu_count() or 4))))
    args = ap.parse_args()

    loader, loader_kw, events_dir, quality_csv = cohort_config(args.cohort)
    out_dir = os.path.join(ROOT, 'outputs', 'schumann', 'images',
                            'coupling', f'{args.cohort}_composite')
    os.makedirs(out_dir, exist_ok=True)

    summary = pd.read_csv(os.path.join(events_dir, 'extraction_summary.csv'))
    ok = summary[(summary['status']=='ok') & (summary['n_events']>=3)]
    tasks = []
    for _, r in ok.iterrows():
        ep = os.path.join(events_dir, f'{r["subject_id"]}_sie_events.csv')
        if os.path.isfile(ep):
            tasks.append((r['subject_id'], ep, quality_csv))
    print(f"Cohort: {args.cohort} composite · subjects: {len(tasks)}")
    print(f"Workers: {args.workers}")
    with Pool(args.workers, initializer=_init_worker,
              initargs=(loader.__name__, loader_kw)) as pool:
        results = pool.map(process_subject, tasks)
    results = [r for r in results if r is not None]
    print(f"Subjects with both Q4 and Q1 events: {len(results)}")

    ch_counts = {}
    for r in results:
        for ch in r['ch_names']:
            ch_counts[ch] = ch_counts.get(ch, 0) + 1
    thr = int(0.8 * len(results))
    common_chs = [ch for ch, n in ch_counts.items() if n >= thr]
    print(f"Common channels: {len(common_chs)}")

    per_sub = {'q4': {b: {} for b in BANDS}, 'q1': {b: {} for b in BANDS}}
    for r in results:
        for qkey, topokey in [('q4', 'q4_topo'), ('q1', 'q1_topo')]:
            for b in BANDS:
                vec = np.array([r[topokey][b][r['ch_names'].index(ch)]
                                 if ch in r['ch_names'] else np.nan
                                 for ch in common_chs])
                per_sub[qkey][b][r['subject_id']] = vec

    print(f"\n{'='*64}")
    print(f"{args.cohort} composite · Test 1: Subject-to-group similarity (LOO)")
    print(f"{'='*64}")
    reliability = {'q4': {b: [] for b in BANDS}, 'q1': {b: [] for b in BANDS}}
    for qkey in ['q4', 'q1']:
        for b in BANDS:
            sub_ids = list(per_sub[qkey][b].keys())
            all_vecs = np.array([per_sub[qkey][b][s] for s in sub_ids])
            for i, s in enumerate(sub_ids):
                others = np.delete(all_vecs, i, axis=0)
                group_mean = np.nanmean(others, axis=0)
                v = all_vecs[i]
                good = np.isfinite(v) & np.isfinite(group_mean)
                if good.sum() < 10:
                    continue
                rho, _ = spearmanr(v[good], group_mean[good])
                reliability[qkey][b].append(rho)

    for qkey in ['q4', 'q1']:
        print(f"\n  {qkey.upper()}:")
        for b in BANDS:
            r = np.array(reliability[qkey][b])
            r = r[np.isfinite(r)]
            if len(r) == 0:
                continue
            print(f"    {b:6s}  ρ(subj, LOO group): median {np.median(r):+.3f}  "
                  f"IQR [{np.percentile(r,25):+.3f}, {np.percentile(r,75):+.3f}]  "
                  f"pct > 0: {(r>0).mean()*100:.0f}%  "
                  f"pct > 0.5: {(r>0.5).mean()*100:.0f}%  n={len(r)}")

    print(f"\n{'='*64}")
    print(f"{args.cohort} composite · Test 2: Within-subject ρ(Q4, Q1)")
    print(f"{'='*64}")
    within_rho = {b: [] for b in BANDS}
    for r in results:
        s = r['subject_id']
        for b in BANDS:
            q4v = per_sub['q4'][b][s]
            q1v = per_sub['q1'][b][s]
            good = np.isfinite(q4v) & np.isfinite(q1v)
            if good.sum() < 10:
                continue
            rho, _ = spearmanr(q4v[good], q1v[good])
            within_rho[b].append(rho)
    for b in BANDS:
        r = np.array(within_rho[b])
        r = r[np.isfinite(r)]
        s_stat, p = wilcoxon(r) if len(r)>=10 and np.any(r!=0) else (np.nan, np.nan)
        print(f"  {b:6s}  within-subj ρ(Q4,Q1): median {np.median(r):+.3f}  "
              f"mean {np.mean(r):+.3f}  "
              f"pct > 0: {(r>0).mean()*100:.0f}%  "
              f"Wilcoxon p = {p:.3g}  n={len(r)}")

    print(f"\n{'='*64}")
    print(f"{args.cohort} composite · Test 3: β16 hemispheric lateralization")
    print(f"{'='*64}")
    left_chs = [ch for ch in common_chs if ch[-1].isdigit() and int(ch[-1])%2==1]
    right_chs = [ch for ch in common_chs if ch[-1].isdigit() and int(ch[-1])%2==0]
    left_idx = [common_chs.index(c) for c in left_chs]
    right_idx = [common_chs.index(c) for c in right_chs]
    print(f"  left channels ({len(left_chs)}): {left_chs[:10]}...")
    print(f"  right channels ({len(right_chs)}): {right_chs[:10]}...")

    lat = {'q4': [], 'q1': []}
    for r in results:
        s = r['subject_id']
        for qkey in ['q4', 'q1']:
            v = per_sub[qkey]['β16'][s]
            L = np.nanmean(v[left_idx]); R = np.nanmean(v[right_idx])
            if np.isfinite(L) and np.isfinite(R) and (L+R) > 0:
                lat[qkey].append((L - R) / (L + R))
    for qkey in ['q4', 'q1']:
        a = np.array(lat[qkey])
        print(f"  β16 lateralization {qkey.upper()}: median "
              f"{np.median(a):+.3f}  pct left-dominant: {(a>0).mean()*100:.0f}%")

    rows = []
    for qkey in ['q4','q1']:
        for b in BANDS:
            for i, rho in enumerate(reliability[qkey][b]):
                rows.append({'test': 'reliability', 'quartile': qkey, 'band': b,
                              'rho': rho})
    for b in BANDS:
        for rho in within_rho[b]:
            rows.append({'test': 'within_q4q1', 'quartile': None, 'band': b,
                          'rho': rho})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'network_reliability.csv'),
                               index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax = axes[0]
    for b, c in zip(BANDS, ['#2166ac', '#d73027', '#1a9641']):
        r = np.array(reliability['q4'][b]); r = r[np.isfinite(r)]
        ax.hist(r, bins=np.linspace(-1,1,25), alpha=0.45, color=c,
                label=f'{b} (med {np.median(r):+.2f})')
    ax.axvline(0, color='k', lw=0.5)
    ax.set_xlabel('ρ(subject Q4, LOO group Q4)')
    ax.set_ylabel('subjects')
    ax.set_title('Q4 subject-to-group reliability')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    for b, c in zip(BANDS, ['#2166ac', '#d73027', '#1a9641']):
        r = np.array(within_rho[b]); r = r[np.isfinite(r)]
        ax.hist(r, bins=np.linspace(-1,1,25), alpha=0.45, color=c,
                label=f'{b} (med {np.median(r):+.2f})')
    ax.axvline(0, color='k', lw=0.5)
    ax.set_xlabel('within-subject ρ(Q4, Q1)')
    ax.set_ylabel('subjects')
    ax.set_title('Within-subject Q4 × Q1 topographic correlation')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[2]
    for qkey, color, lbl in [('q4', '#d73027', 'Q4'),
                              ('q1', '#4575b4', 'Q1')]:
        a = np.array(lat[qkey])
        ax.hist(a, bins=np.linspace(-0.5, 0.5, 25), alpha=0.55, color=color,
                label=f'{lbl} (med {np.median(a):+.2f})')
    ax.axvline(0, color='k', lw=0.5)
    ax.set_xlabel('β16 lateralization  (L − R) / (L + R)\npositive = left-dominant')
    ax.set_ylabel('subjects')
    ax.set_title('β16 hemispheric lateralization per subject')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.suptitle(f'B45 composite — Within-subject reliability of network-identity '
                 f'({args.cohort}, n={len(results)})',
                 y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'network_reliability.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_dir}/network_reliability.png")


if __name__ == '__main__':
    main()
