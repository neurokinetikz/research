#!/usr/bin/env python3
"""
B61v3 — Composite v2 canonical-event yield test.

Right question: which detector yields MORE canonical events per subject?
  Envelope + Q4 post-filter: ~1-2 canonical events per LEMON subject
  Composite at various thresholds: TBD

For each composite event, compute its template_rho against the SAME LEMON
grand-average template used in Q4 analysis. Then count events with
template_rho > 0.3 (proxy for "canonical") per subject.

Also report: mean composite S at canonical vs non-canonical events,
and whether high-S composite events are enriched for high template_rho
(i.e., composite score itself tracks canonicality).
"""
from __future__ import annotations
import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import signal
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.run_sie_extraction import load_lemon
from scripts.sie_composite_detector_v2 import (compute_streams, composite_S,
                                                 detect_events,
                                                 refine_onset_nadir)
from scripts.sie_perionset_triple_average import bandpass

warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'quality')
EVENTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'exports_sie',
                           'lemon')
os.makedirs(OUT_DIR, exist_ok=True)

THRESHOLDS = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
F0 = 7.83
HALF_BW = 0.6
PRE_SEC = 10.0
POST_SEC = 10.0
STEP_SEC = 0.1
WIN_SEC = 1.0
TGRID = np.arange(-PRE_SEC, POST_SEC + STEP_SEC/2, STEP_SEC)
# m_core indices — integer to avoid float equality issues
_M_CORE_IDX = np.where((TGRID >= -5.0 - 1e-6) & (TGRID <= 5.0 + 1e-6))[0]

# Load LEMON envelope template (from existing trajectories file)
TRAJ_LEMON = os.path.join(OUT_DIR, 'trajectories_lemon.npz')


def robust_z(x):
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * 1.4826
    if not np.isfinite(mad) or mad < 1e-9:
        return x - med
    return (x - med) / mad


def env_stream(raw):
    fs = raw.info['sfreq']
    X = raw.get_data() * 1e6
    y = X.mean(axis=0)
    yb = bandpass(y, fs, F0 - HALF_BW, F0 + HALF_BW)
    env = np.abs(signal.hilbert(yb))
    nwin = int(round(WIN_SEC * fs))
    nstep = int(round(STEP_SEC * fs))
    env_vals = []
    centers = []
    for i in range(0, X.shape[1] - nwin + 1, nstep):
        env_vals.append(float(np.mean(env[i:i+nwin])))
        centers.append((i + nwin/2) / fs)
    t = np.array(centers)
    zE_full = robust_z(np.array(env_vals))
    return t, zE_full, raw.times[-1]


def score_rho(t_full, zE_full, t0, template_core):
    """template_rho of event at t0 against template_core."""
    if t0 - PRE_SEC < 2 or t0 + POST_SEC > t_full[-1] - 2:
        return np.nan
    sel = (t_full >= t0 - PRE_SEC) & (t_full <= t0 + POST_SEC)
    if sel.sum() < int(PRE_SEC + POST_SEC) * 5:
        return np.nan
    t_rel = t_full[sel] - t0
    zE_seg = zE_full[sel]
    env_i = np.interp(TGRID, t_rel, zE_seg, left=np.nan, right=np.nan)
    ev = env_i[_M_CORE_IDX]
    if len(ev) != len(template_core) or np.any(~np.isfinite(ev)):
        return np.nan
    ev_c = ev - np.nanmean(ev)
    denom = np.sqrt(np.nansum(ev_c**2) * np.nansum(template_core**2))
    if denom <= 0: return np.nan
    return float(np.nansum(ev_c * template_core) / denom)


# Build LEMON template from saved trajectories
d = np.load(TRAJ_LEMON, allow_pickle=True)
trajs = d['trajs'].astype(np.float64)
# Use the SAME TGRID index to ensure length match in score_rho
template_full = np.nanmean(trajs, axis=0)
if len(template_full) != len(TGRID):
    # Fallback: interp to TGRID
    saved_tgrid = d['tgrid'].astype(np.float64)
    template_full = np.interp(TGRID, saved_tgrid, template_full)
tmpl_raw = template_full[_M_CORE_IDX]
TEMPLATE_CORE = tmpl_raw - np.nanmean(tmpl_raw)
print(f"Template shape: {TEMPLATE_CORE.shape}")


def process_one(sub_id):
    try:
        env_events = pd.read_csv(os.path.join(EVENTS_DIR,
                                               f'{sub_id}_sie_events.csv'))
    except Exception:
        return None
    env_events = env_events.dropna(subset=['t0_net'])
    try:
        raw = load_lemon(sub_id, condition='EC')
    except Exception:
        return None
    if raw is None:
        return None
    X = raw.get_data() * 1e6
    fs = raw.info['sfreq']
    try:
        t, env, R, P, M = compute_streams(X, fs)
        S = composite_S(env, R, P, M)
    except Exception as e:
        return None
    # envelope stream for rho scoring
    try:
        tE, zE_full, _ = env_stream(raw)
    except Exception:
        return None

    # Score envelope events
    env_rhos = []
    for _, ev in env_events.iterrows():
        env_rhos.append(score_rho(tE, zE_full, float(ev['t0_net']),
                                   TEMPLATE_CORE))
    env_rhos = np.array(env_rhos)

    rows = []
    for thr in THRESHOLDS:
        t_detects, S_vals = detect_events(t, S, threshold=thr)
        if len(t_detects) == 0:
            rows.append({
                'subject_id': sub_id, 'threshold': thr,
                'n_comp': 0,
                'n_canonical_comp': 0,
                'median_rho_comp': np.nan,
                'mean_S_comp': np.nan,
                'n_env': len(env_rhos),
                'n_canonical_env_Q4': int(np.sum(np.isfinite(env_rhos) &
                                                    (env_rhos > 0.3))),
                'median_rho_env': (float(np.nanmedian(env_rhos))
                                    if np.any(np.isfinite(env_rhos))
                                    else np.nan),
            })
            continue
        # Refine onsets, score template_rho on envelope signal
        comp_rhos = []
        for td in t_detects:
            t_onset = refine_onset_nadir(t, env, R, P, M, td)
            comp_rhos.append(score_rho(tE, zE_full, t_onset, TEMPLATE_CORE))
        comp_rhos = np.array(comp_rhos)
        rows.append({
            'subject_id': sub_id, 'threshold': thr,
            'n_comp': len(comp_rhos),
            'n_canonical_comp': int(np.sum(np.isfinite(comp_rhos) &
                                             (comp_rhos > 0.3))),
            'median_rho_comp': float(np.nanmedian(comp_rhos)),
            'mean_S_comp': float(np.mean(S_vals)),
            'n_env': len(env_rhos),
            'n_canonical_env_Q4': int(np.sum(np.isfinite(env_rhos) &
                                                (env_rhos > 0.3))),
            'median_rho_env': (float(np.nanmedian(env_rhos))
                                if np.any(np.isfinite(env_rhos))
                                else np.nan),
        })
    return rows


def main():
    summary = pd.read_csv(os.path.join(EVENTS_DIR, 'extraction_summary.csv'))
    ok = summary[(summary['status']=='ok') & (summary['n_events']>=4)]
    rng = np.random.default_rng(42)
    n = int(os.environ.get('N_SUBJECTS', 15))
    subs = rng.choice(ok['subject_id'].values,
                       size=min(n, len(ok)), replace=False).tolist()
    print(f"Canonical-yield test: {len(subs)} LEMON subjects × "
          f"{len(THRESHOLDS)} thresholds")
    print(f"Canonical criterion: template_rho > 0.3")

    n_workers = int(os.environ.get('SIE_WORKERS',
                                    min(4, os.cpu_count() or 4)))
    with Pool(n_workers) as pool:
        out_lists = pool.map(process_one, subs)
    df = pd.DataFrame([r for rs in out_lists if rs for r in rs])
    out_csv = os.path.join(OUT_DIR, 'composite_v2_canonical_yield.csv')
    df.to_csv(out_csv, index=False)

    print(f"\n=== Canonical-event yield (median across subjects) ===")
    print(f"{'thr':>6}{'n_comp':>8}{'comp>0.3':>11}{'rho_median':>12}"
          f"{'mean_S':>9} | "
          f"{'n_env':>6}{'env>0.3':>9}{'env_rho_med':>12}")
    print('-' * 90)
    env_q4_med = df['n_canonical_env_Q4'].median()
    env_n_med = df['n_env'].median()
    env_rho_med = df['median_rho_env'].median()
    for thr, sub in df.groupby('threshold'):
        n_comp = sub['n_comp'].median()
        n_can = sub['n_canonical_comp'].median()
        rho_m = sub['median_rho_comp'].median()
        mS = sub['mean_S_comp'].median()
        print(f"{thr:>6.2f}{n_comp:>8.1f}{n_can:>11.1f}"
              f"{rho_m:>+12.3f}{mS:>9.2f}  "
              f"{env_n_med:>5.0f}{env_q4_med:>8.0f}{env_rho_med:>+12.3f}")
    print(f"\nSaved: {out_csv}")


if __name__ == '__main__':
    main()
