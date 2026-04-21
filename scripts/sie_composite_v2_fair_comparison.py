#!/usr/bin/env python3
"""
B61v4 — Fair composite v2 vs envelope canonical-yield test.

Two fairness fixes over B61v3:
1. Build a TEMPLATE-FREE canonicality metric: composite S(t₀) itself.
   Score each event (envelope or composite) by the composite S at its
   detection time. High S = all four streams co-elevated.
2. ALSO build a SYMMETRIC template: average envelope trajectories over a
   balanced pool of envelope-detected AND composite-detected events.
   Score both detector outputs against that shared template.

For each subject:
  - Envelope events: score each by (a) symmetric template_ρ and (b) S(t₀)
  - Composite events at a chosen threshold (default 1.5): same scoring
  - Report per-detector: median template_ρ, median S(t₀), canonical yield
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

COMP_THRESHOLD = 1.5
F0 = 7.83
HALF_BW = 0.6
PRE_SEC = 10.0
POST_SEC = 10.0
STEP_SEC = 0.1
WIN_SEC = 1.0
TGRID = np.arange(-PRE_SEC, POST_SEC + STEP_SEC/2, STEP_SEC)
_M_CORE_IDX = np.where((TGRID >= -5.0 - 1e-6) & (TGRID <= 5.0 + 1e-6))[0]


def robust_z(x):
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * 1.4826
    if not np.isfinite(mad) or mad < 1e-9: return x - med
    return (x - med) / mad


def env_stream(raw):
    fs = raw.info['sfreq']
    X = raw.get_data() * 1e6
    y = X.mean(axis=0)
    yb = bandpass(y, fs, F0 - HALF_BW, F0 + HALF_BW)
    env = np.abs(signal.hilbert(yb))
    nwin = int(round(WIN_SEC * fs))
    nstep = int(round(STEP_SEC * fs))
    vals, centers = [], []
    for i in range(0, X.shape[1] - nwin + 1, nstep):
        vals.append(float(np.mean(env[i:i+nwin])))
        centers.append((i + nwin/2) / fs)
    return np.array(centers), robust_z(np.array(vals)), raw.times[-1]


def extract_traj(t_full, zE_full, t0):
    if t0 - PRE_SEC < 2 or t0 + POST_SEC > t_full[-1] - 2:
        return None
    sel = (t_full >= t0 - PRE_SEC) & (t_full <= t0 + POST_SEC)
    if sel.sum() < int(PRE_SEC + POST_SEC) * 5: return None
    # use endpoint extrapolation (no left/right NaN) to avoid float-edge
    # boundary issues when t0 isn't exactly on the step grid
    env_i = np.interp(TGRID, t_full[sel] - t0, zE_full[sel])
    return env_i


def score_rho(traj, template_core):
    ev = traj[_M_CORE_IDX]
    if len(ev) != len(template_core): return np.nan
    ev_c = ev - np.nanmean(ev)
    denom = np.sqrt(np.nansum(ev_c**2) * np.nansum(template_core**2))
    if denom <= 0: return np.nan
    return float(np.nansum(ev_c * template_core) / denom)


def S_at(t_comp_stream, S_stream, t0):
    idx = int(np.argmin(np.abs(t_comp_stream - t0)))
    return float(S_stream[idx])


def S_peak_near(t_comp_stream, S_stream, t0, win=3.0):
    """Peak S in [t0, t0+win]. Fair counterpart to composite's S_at_detect."""
    mask = (t_comp_stream >= t0) & (t_comp_stream <= t0 + win)
    if not mask.any():
        return np.nan
    return float(np.max(S_stream[mask]))


def process_subject(sub_id):
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
    if raw is None: return None
    X = raw.get_data() * 1e6
    fs = raw.info['sfreq']
    try:
        t, env, R, P, M = compute_streams(X, fs)
        S = composite_S(env, R, P, M)
        tE, zE_full, _ = env_stream(raw)
    except Exception:
        return None

    # Envelope event trajectories + S values (use peak S in [t0, t0+3])
    env_trajs, env_t0s, env_S_vals = [], [], []
    for _, ev in env_events.iterrows():
        t0 = float(ev['t0_net'])
        tr = extract_traj(tE, zE_full, t0)
        if tr is None: continue
        env_trajs.append(tr)
        env_t0s.append(t0)
        env_S_vals.append(S_peak_near(t, S, t0))

    # Composite event onsets
    t_detects, S_peaks = detect_events(t, S, threshold=COMP_THRESHOLD)
    comp_trajs, comp_t0s, comp_S_vals = [], [], []
    for td, sp in zip(t_detects, S_peaks):
        t_on = refine_onset_nadir(t, env, R, P, M, td)
        # Use t_on (nadir) as the anchor for trajectory extraction (matches
        # how envelope template in B6 was built), but record S at t_detect
        # (the S peak) for the template-free quality measure.
        tr = extract_traj(tE, zE_full, t_on)
        if tr is None: continue
        comp_trajs.append(tr)
        comp_t0s.append(t_on)
        comp_S_vals.append(float(sp))  # S at detection (= peak S near event)

    return {
        'subject_id': sub_id,
        'env_trajs': np.array(env_trajs),
        'env_t0s': np.array(env_t0s),
        'env_S_vals': np.array(env_S_vals),
        'comp_trajs': np.array(comp_trajs),
        'comp_t0s': np.array(comp_t0s),
        'comp_S_vals': np.array(comp_S_vals),
    }


def main():
    summary = pd.read_csv(os.path.join(EVENTS_DIR, 'extraction_summary.csv'))
    ok = summary[(summary['status']=='ok') & (summary['n_events']>=4)]
    rng = np.random.default_rng(42)
    n = int(os.environ.get('N_SUBJECTS', 15))
    subs = rng.choice(ok['subject_id'].values,
                       size=min(n, len(ok)), replace=False).tolist()
    print(f"Fair comparison on {len(subs)} LEMON subjects, composite S≥{COMP_THRESHOLD}")
    n_workers = int(os.environ.get('SIE_WORKERS',
                                    min(4, os.cpu_count() or 4)))
    with Pool(n_workers) as pool:
        results = pool.map(process_subject, subs)
    results = [r for r in results if r is not None]

    # ===== Build SYMMETRIC template: equal-sample from both detectors =====
    all_env_trajs = [r['env_trajs'] for r in results if len(r['env_trajs'])]
    all_comp_trajs = [r['comp_trajs'] for r in results if len(r['comp_trajs'])]
    env_pool = np.vstack(all_env_trajs) if all_env_trajs else np.empty((0,len(TGRID)))
    comp_pool = np.vstack(all_comp_trajs) if all_comp_trajs else np.empty((0,len(TGRID)))
    print(f"Pool sizes: envelope={len(env_pool)}, composite={len(comp_pool)}")

    # Balanced pool (equal per detector)
    n_per = min(len(env_pool), len(comp_pool))
    idx_env = rng.choice(len(env_pool), size=n_per, replace=False)
    idx_comp = rng.choice(len(comp_pool), size=n_per, replace=False)
    sym_pool = np.vstack([env_pool[idx_env], comp_pool[idx_comp]])
    sym_template = np.nanmean(sym_pool, axis=0)
    sym_tmpl_core = (sym_template[_M_CORE_IDX]
                      - np.nanmean(sym_template[_M_CORE_IDX]))
    print(f"Symmetric template built from {len(sym_pool)} events "
          f"({n_per} envelope + {n_per} composite)")

    # Also: envelope-only template (for sanity)
    env_template = np.nanmean(env_pool, axis=0)
    env_tmpl_core = (env_template[_M_CORE_IDX]
                      - np.nanmean(env_template[_M_CORE_IDX]))
    comp_template = np.nanmean(comp_pool, axis=0)
    comp_tmpl_core = (comp_template[_M_CORE_IDX]
                       - np.nanmean(comp_template[_M_CORE_IDX]))

    # ===== Score every event by each template + S(t₀) =====
    def score_all(trajs, core):
        return np.array([score_rho(tr, core) for tr in trajs])

    rows = []
    for r in results:
        e_t, c_t = r['env_trajs'], r['comp_trajs']
        e_S, c_S = r['env_S_vals'], r['comp_S_vals']
        # Symmetric template scoring
        rhos_env_sym = score_all(e_t, sym_tmpl_core) if len(e_t) else np.array([])
        rhos_comp_sym = score_all(c_t, sym_tmpl_core) if len(c_t) else np.array([])
        # Envelope template scoring
        rhos_env_env = score_all(e_t, env_tmpl_core) if len(e_t) else np.array([])
        rhos_comp_env = score_all(c_t, env_tmpl_core) if len(c_t) else np.array([])
        # Composite template scoring
        rhos_env_comp = score_all(e_t, comp_tmpl_core) if len(e_t) else np.array([])
        rhos_comp_comp = score_all(c_t, comp_tmpl_core) if len(c_t) else np.array([])
        rows.append({
            'subject_id': r['subject_id'],
            'n_env': len(e_t), 'n_comp': len(c_t),
            # S values
            'median_S_env': (float(np.nanmedian(e_S)) if len(e_S) else np.nan),
            'median_S_comp': (float(np.nanmedian(c_S)) if len(c_S) else np.nan),
            # Symmetric template
            'rho_env_sym_median': (float(np.nanmedian(rhos_env_sym))
                                     if len(rhos_env_sym) else np.nan),
            'rho_comp_sym_median': (float(np.nanmedian(rhos_comp_sym))
                                      if len(rhos_comp_sym) else np.nan),
            'n_env_canonical_sym': int(np.sum(np.isfinite(rhos_env_sym) &
                                                (rhos_env_sym > 0.3))),
            'n_comp_canonical_sym': int(np.sum(np.isfinite(rhos_comp_sym) &
                                                 (rhos_comp_sym > 0.3))),
            # env template (biases env)
            'rho_env_envTpl_median': (float(np.nanmedian(rhos_env_env))
                                       if len(rhos_env_env) else np.nan),
            'rho_comp_envTpl_median': (float(np.nanmedian(rhos_comp_env))
                                        if len(rhos_comp_env) else np.nan),
            # comp template (biases comp)
            'rho_env_compTpl_median': (float(np.nanmedian(rhos_env_comp))
                                        if len(rhos_env_comp) else np.nan),
            'rho_comp_compTpl_median': (float(np.nanmedian(rhos_comp_comp))
                                         if len(rhos_comp_comp) else np.nan),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, 'composite_v2_fair_comparison.csv'),
               index=False)

    print(f"\n=== Template-free: composite S(t₀) at event onsets ===")
    print(f"  envelope events:    median S = {df['median_S_env'].median():.3f}")
    print(f"  composite events:   median S = {df['median_S_comp'].median():.3f}")

    print(f"\n=== Symmetric template (balanced env + comp pool) ===")
    print(f"  envelope events:    median ρ = {df['rho_env_sym_median'].median():+.3f}")
    print(f"  composite events:   median ρ = {df['rho_comp_sym_median'].median():+.3f}")
    print(f"  envelope canonical (ρ>0.3, median): {df['n_env_canonical_sym'].median():.1f} / subject")
    print(f"  composite canonical (ρ>0.3, median): {df['n_comp_canonical_sym'].median():.1f} / subject")

    print(f"\n=== Envelope-only template (biased toward envelope) ===")
    print(f"  envelope events:    median ρ = {df['rho_env_envTpl_median'].median():+.3f}")
    print(f"  composite events:   median ρ = {df['rho_comp_envTpl_median'].median():+.3f}")

    print(f"\n=== Composite-only template (biased toward composite) ===")
    print(f"  envelope events:    median ρ = {df['rho_env_compTpl_median'].median():+.3f}")
    print(f"  composite events:   median ρ = {df['rho_comp_compTpl_median'].median():+.3f}")

    print(f"\n=== Totals ===")
    print(f"  envelope total events/subject (median): {df['n_env'].median():.0f}")
    print(f"  composite total events/subject (median): {df['n_comp'].median():.0f}")


if __name__ == '__main__':
    main()
