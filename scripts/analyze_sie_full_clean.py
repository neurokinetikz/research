#!/usr/bin/env python3
"""
SIE Replication Analysis -- Full Clean Run
============================================
Comprehensive analysis with MSC artifact filter (mean MSC >= 0.9 excluded).
Combines Phase 1 (replication), Phase 2 (age/EC-EO/test-retest),
Phase 3 (correlates), PLV analysis, and ratio unpredictability scan.

Usage:
    python scripts/analyze_sie_full_clean.py
"""

import os
import sys
import warnings
from glob import glob

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr, pearsonr

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

SIE_BASE = os.path.join(os.path.dirname(__file__), '..', 'exports_sie')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

PHI = (1 + np.sqrt(5)) / 2
F0 = 7.6

LABELS = ['sr1', 'sr1.5', 'sr2', 'sr2o', 'sr2.5', 'sr3', 'sr4', 'sr5', 'sr6']
N_VALUES = [0, 0.5, 1, None, 1.5, 2, 2.5, 3, 3.5]
PREDICTED = {lbl: F0 * PHI ** n if n is not None else 13.75
             for lbl, n in zip(LABELS, N_VALUES)}

DATASET_NAMES = {
    'eegmmidb': 'EEGMMIDB', 'chbmp': 'CHBMP',
    'lemon': 'LEMON EC', 'lemon_EO': 'LEMON EO',
    'dortmund': 'Dortmund EC-pre s1', 'dortmund_EO_pre': 'Dortmund EO-pre s1',
    'dortmund_EC_post': 'Dortmund EC-post s1', 'dortmund_EO_post': 'Dortmund EO-post s1',
    'dortmund_EC_pre_ses2': 'Dortmund EC-pre s2', 'dortmund_EO_pre_ses2': 'Dortmund EO-pre s2',
    'dortmund_EC_post_ses2': 'Dortmund EC-post s2', 'dortmund_EO_post_ses2': 'Dortmund EO-post s2',
    'hbn_R1': 'HBN R1', 'hbn_R2': 'HBN R2', 'hbn_R3': 'HBN R3',
    'hbn_R4': 'HBN R4', 'hbn_R6': 'HBN R6',
    'tdbrain': 'TDBRAIN EC', 'tdbrain_EO': 'TDBRAIN EO',
}


def load_events():
    dfs = []
    for dirname in sorted(os.listdir(SIE_BASE)):
        dirpath = os.path.join(SIE_BASE, dirname)
        if not os.path.isdir(dirpath):
            continue
        for f in glob(os.path.join(dirpath, '*_sie_events.csv')):
            try:
                df = pd.read_csv(f)
                if len(df) > 0:
                    df['_source_dir'] = dirname
                    dfs.append(df)
            except Exception:
                continue
    events = pd.concat(dfs, ignore_index=True)
    for col in events.columns:
        if col not in ['subject_id', 'dataset', 'condition', 'session_name',
                       '_source_dir', 'ignition_freqs']:
            events[col] = pd.to_numeric(events[col], errors='coerce')
    return events


def load_summaries():
    dfs = []
    for dirname in sorted(os.listdir(SIE_BASE)):
        dirpath = os.path.join(SIE_BASE, dirname)
        if not os.path.isdir(dirpath):
            continue
        for f in glob(os.path.join(dirpath, '*_sie_summary.csv')):
            try:
                df = pd.read_csv(f)
                if len(df) > 0:
                    df['_source_dir'] = dirname
                    dfs.append(df)
            except Exception:
                continue
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def load_extraction_summaries():
    dfs = []
    for dirname in sorted(os.listdir(SIE_BASE)):
        dirpath = os.path.join(SIE_BASE, dirname)
        f = os.path.join(dirpath, 'extraction_summary.csv')
        if os.path.isfile(f):
            try:
                df = pd.read_csv(f)
                df['_source_dir'] = dirname
                dfs.append(df)
            except Exception:
                continue
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def apply_msc_filter(events):
    """Remove events with mean MSC >= 0.9 across all 9 harmonics (artifact filter)."""
    msc_cols = [f'msc_{lbl}_v' for lbl in LABELS]
    msc_mean = events[msc_cols].mean(axis=1)
    mask = msc_mean < 0.9
    n_before = len(events)
    events_clean = events[mask].copy().reset_index(drop=True)
    n_removed = n_before - len(events_clean)
    print(f"  MSC artifact filter: removed {n_removed:,} events ({100*n_removed/n_before:.1f}%)")
    print(f"  Retained: {len(events_clean):,} events")
    return events_clean


# =====================================================================
# ANALYSES
# =====================================================================

def dataset_overview(events, summaries, extraction):
    print("\n" + "=" * 80)
    print("DATASET OVERVIEW (after MSC filter)")
    print("=" * 80)
    rows = []
    for dirname in sorted(DATASET_NAMES.keys()):
        e = events[events['_source_dir'] == dirname]
        x = extraction[extraction['_source_dir'] == dirname]
        n_ok = len(x[x['status'] == 'ok']) if 'status' in x.columns and len(x) > 0 else 0
        n_events = len(e)
        s = summaries[summaries['_source_dir'] == dirname]
        dur = s['recording_duration_sec'].dropna().mean() if 'recording_duration_sec' in s.columns else np.nan
        rate = n_events / max(n_ok, 1) / max(dur / 60, 1e-9) if np.isfinite(dur) and n_ok > 0 else np.nan
        rows.append({
            'Dataset': DATASET_NAMES.get(dirname, dirname),
            'Subjects': n_ok, 'Events': n_events,
            'Evts/subj': round(n_events / max(n_ok, 1), 2),
            'Rate/min': round(rate, 2) if np.isfinite(rate) else '?',
            'Dur(s)': round(dur, 0) if np.isfinite(dur) else '?',
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print(f"\nTOTAL: {df['Subjects'].sum()} subjects, {df['Events'].sum()} events")
    return df


def harmonic_frequencies(events):
    print("\n" + "=" * 80)
    print("HARMONIC FREQUENCIES")
    print("=" * 80)
    print(f"{'Harm':>6s} {'n':>5s} {'Pred':>8s} {'Measured':>16s} {'Err%':>7s} {'N':>7s}")
    for lbl, n in zip(LABELS, N_VALUES):
        v = events[lbl].dropna()
        if len(v) == 0:
            continue
        pred = PREDICTED[lbl]
        err = 100 * (v.mean() - pred) / pred
        print(f"{lbl.upper():>6s} {str(n) if n is not None else '---':>5s} {pred:>8.2f} "
              f"{v.mean():>7.2f} +/- {v.std():.2f} {err:>+6.1f}% {len(v):>7,}")
    phi_lbls = [l for l, n in zip(LABELS, N_VALUES) if n is not None]
    mae = np.mean([abs(100 * (events[l].dropna().mean() - PREDICTED[l]) / PREDICTED[l])
                    for l in phi_lbls if len(events[l].dropna()) > 0])
    print(f"\nMean absolute error (phi-n): {mae:.2f}%")


def ratio_precision(events):
    print("\n" + "=" * 80)
    print("RATIO PRECISION")
    print("=" * 80)
    tests = [('sr3/sr1', PHI**2, 'phi2'), ('sr5/sr1', PHI**3, 'phi3'),
             ('sr5/sr3', PHI, 'phi'), ('sr6/sr4', PHI, 'phi')]
    errs = []
    for col, pred, lbl in tests:
        v = events[col].dropna()
        v = v[(v > 0) & (v < 20)]
        if len(v) == 0:
            continue
        ci = stats.t.interval(0.95, len(v) - 1, loc=v.mean(), scale=stats.sem(v))
        err = 100 * (v.mean() - pred) / pred
        errs.append(abs(err))
        contains = 'YES' if ci[0] <= pred <= ci[1] else 'no'
        print(f"  {col} = {lbl}: {v.mean():.4f} ({err:+.2f}%) CI=[{ci[0]:.4f},{ci[1]:.4f}] "
              f"contains phi: {contains}  N={len(v):,}")
    print(f"\n  Mean absolute ratio error: {np.mean(errs):.2f}%")


def independence_convergence(events):
    print("\n" + "=" * 80)
    print("INDEPENDENCE-CONVERGENCE PARADOX")
    print("=" * 80)
    for a, b, label in [('sr1', 'sr3', 'SR1 vs SR3'), ('sr1', 'sr5', 'SR1 vs SR5'),
                         ('sr3', 'sr5', 'SR3 vs SR5')]:
        va, vb = events[a], events[b]
        mask = va.notna() & vb.notna()
        if mask.sum() < 10:
            continue
        r, p = pearsonr(va[mask], vb[mask])
        print(f"  {label}: r = {r:+.4f}, p = {p:.4f}, N = {mask.sum():,}")


def cross_dataset_anova(events):
    print("\n" + "=" * 80)
    print("CROSS-DATASET CONSISTENCY (ANOVA)")
    print("=" * 80)
    for lbl in ['sr1', 'sr3', 'sr5']:
        groups = []
        for dirname in sorted(DATASET_NAMES.keys()):
            v = events[events['_source_dir'] == dirname][lbl].dropna()
            if len(v) >= 10:
                groups.append(v.values)
        if len(groups) >= 2:
            F, p = stats.f_oneway(*groups)
            grand = np.concatenate(groups).mean()
            ss_b = sum(len(g) * (g.mean() - grand)**2 for g in groups)
            ss_t = sum(((g - grand)**2).sum() for g in groups)
            eta2 = ss_b / ss_t if ss_t > 0 else 0
            print(f"  {lbl.upper()}: F = {F:.2f}, p = {p:.2e}, eta2 = {eta2:.4f}")


def ec_vs_eo(events, summaries):
    print("\n" + "=" * 80)
    print("EC vs EO")
    print("=" * 80)
    # Recompute per-subject event counts from clean events
    pairs = [('lemon', 'lemon_EO', 'LEMON'), ('dortmund', 'dortmund_EO_pre', 'Dortmund pre s1'),
             ('dortmund_EC_post', 'dortmund_EO_post', 'Dortmund post s1'),
             ('tdbrain', 'tdbrain_EO', 'TDBRAIN')]
    for ec_dir, eo_dir, label in pairs:
        ec_counts = events[events['_source_dir'] == ec_dir].groupby('subject_id').size()
        eo_counts = events[events['_source_dir'] == eo_dir].groupby('subject_id').size()
        # Get durations from summaries
        ec_dur = summaries[summaries['_source_dir'] == ec_dir].set_index('subject_id')['recording_duration_sec']
        eo_dur = summaries[summaries['_source_dir'] == eo_dir].set_index('subject_id')['recording_duration_sec']
        shared = ec_counts.index.intersection(eo_counts.index).intersection(ec_dur.index).intersection(eo_dur.index)
        if len(shared) < 10:
            print(f"  {label}: <10 paired, skipping")
            continue
        ec_rate = ec_counts.loc[shared] / (ec_dur.loc[shared] / 60)
        eo_rate = eo_counts.loc[shared] / (eo_dur.loc[shared] / 60)
        t, p = stats.ttest_rel(ec_rate, eo_rate)
        d = (ec_rate.mean() - eo_rate.mean()) / np.sqrt((ec_rate.std()**2 + eo_rate.std()**2) / 2)
        print(f"  {label} (N={len(shared)}): EC={ec_rate.mean():.3f} EO={eo_rate.mean():.3f} "
              f"t={t:.3f} p={p:.4f} d={d:.3f}")


def test_retest(events, summaries):
    print("\n" + "=" * 80)
    print("TEST-RETEST (Dortmund ses-1 vs ses-2)")
    print("=" * 80)
    for s1_dir, s2_dir, label in [('dortmund', 'dortmund_EC_pre_ses2', 'EC-pre'),
                                    ('dortmund_EO_pre', 'dortmund_EO_pre_ses2', 'EO-pre')]:
        s1 = events[events['_source_dir'] == s1_dir].groupby('subject_id').size()
        s2 = events[events['_source_dir'] == s2_dir].groupby('subject_id').size()
        shared = s1.index.intersection(s2.index)
        if len(shared) < 10:
            print(f"  {label}: <10 paired, skipping")
            continue
        r1, r2 = s1.loc[shared].values.astype(float), s2.loc[shared].values.astype(float)
        r, p = pearsonr(r1, r2)
        n, k = len(shared), 2
        data = np.column_stack([r1, r2])
        rm = data.mean(axis=1)
        gm = data.mean()
        ms_r = k * np.sum((rm - gm)**2) / (n - 1)
        ms_e = np.sum((data - rm[:, None])**2) / ((n - 1) * (k - 1))
        icc = (ms_r - ms_e) / (ms_r + (k - 1) * ms_e) if (ms_r + (k - 1) * ms_e) > 0 else 0
        print(f"  {label} (N={len(shared)}): ses1={r1.mean():.2f} ses2={r2.mean():.2f} "
              f"r={r:.3f} ICC={icc:.3f}")


def age_trajectories(events, summaries):
    print("\n" + "=" * 80)
    print("AGE TRAJECTORIES")
    print("=" * 80)

    # HBN
    hbn_dfs = []
    for release in ['R1', 'R2', 'R3', 'R4', 'R6']:
        path = f'/Volumes/T9/hbn_data/cmi_bids_{release}/participants.tsv'
        if os.path.isfile(path):
            df = pd.read_csv(path, sep='\t')
            hbn_dfs.append(df)
    if hbn_dfs:
        hbn_demo = pd.concat(hbn_dfs, ignore_index=True)
        hbn_demo = hbn_demo.rename(columns={'participant_id': 'subject_id'})
        hbn_demo['age'] = pd.to_numeric(hbn_demo['age'], errors='coerce')

        # Build per-subject metrics from clean events
        hbn_events = events[events['_source_dir'].str.startswith('hbn_')]
        hbn_subj = hbn_events.groupby('subject_id').agg(
            n_events=('sr1_z_max', 'count'),
            mean_z=('sr1_z_max', 'mean'),
            mean_HSI=('HSI', 'mean'),
            mean_sr_score=('sr_score', 'mean'),
        ).reset_index()
        hbn_subj = hbn_subj.merge(hbn_demo[['subject_id', 'age']].drop_duplicates(),
                                   on='subject_id', how='left')
        has_age = hbn_subj['age'].notna()
        print(f"\n  HBN (ages 5-21, N={has_age.sum()}):")
        for m in ['n_events', 'mean_z', 'mean_HSI', 'mean_sr_score']:
            vals = hbn_subj[has_age][[m, 'age']].dropna()
            if len(vals) >= 20:
                rho, p = spearmanr(vals['age'], vals[m])
                sig = '*' if p < 0.05 else ''
                print(f"    {m:>20s}: rho={rho:+.3f} p={p:.2e} {sig}")

    # TDBRAIN
    tdb_path = os.path.expanduser('~/Desktop/TDBRAIN_participants_V2_data/TDBRAIN_participants_V2.tsv')
    if os.path.isfile(tdb_path):
        tdb_demo = pd.read_csv(tdb_path, sep='\t')
        tdb_demo = tdb_demo.rename(columns={'participants_ID': 'subject_id'})
        tdb_demo['age'] = tdb_demo['age'].astype(str).str.replace(',', '.', regex=False)
        tdb_demo['age'] = pd.to_numeric(tdb_demo['age'], errors='coerce')

        tdb_events = events[events['_source_dir'] == 'tdbrain']
        tdb_subj = tdb_events.groupby('subject_id').agg(
            n_events=('sr1_z_max', 'count'),
            mean_z=('sr1_z_max', 'mean'),
            mean_HSI=('HSI', 'mean'),
            mean_sr_score=('sr_score', 'mean'),
        ).reset_index()
        tdb_subj = tdb_subj.merge(tdb_demo[['subject_id', 'age']].drop_duplicates(),
                                   on='subject_id', how='left')
        has_age = tdb_subj['age'].notna()
        print(f"\n  TDBRAIN (ages 5-88, N={has_age.sum()}):")
        for m in ['n_events', 'mean_z', 'mean_HSI', 'mean_sr_score']:
            vals = tdb_subj[has_age][[m, 'age']].dropna()
            if len(vals) >= 20:
                rho, p = spearmanr(vals['age'], vals[m])
                sig = '*' if p < 0.05 else ''
                print(f"    {m:>20s}: rho={rho:+.3f} p={p:.2e} {sig}")


def cognitive_correlates(events, summaries):
    print("\n" + "=" * 80)
    print("COGNITIVE CORRELATES (LEMON)")
    print("=" * 80)
    try:
        from lib.lemon_utils import load_demographics, load_cognitive_data, build_master_table, COG_TESTS
    except ImportError:
        print("  lemon_utils not available")
        return

    master = build_master_table(load_demographics(), load_cognitive_data())
    lemon_events = events[events['_source_dir'] == 'lemon']
    lemon_subj = lemon_events.groupby('subject_id').agg(
        n_events=('sr1_z_max', 'count'),
        mean_z=('sr1_z_max', 'mean'),
        mean_HSI=('HSI', 'mean'),
        mean_sr_score=('sr_score', 'mean'),
    ).reset_index()
    lemon_subj = lemon_subj.merge(master, on='subject_id', how='left')
    print(f"  N={len(lemon_subj)}")

    sie_metrics = ['n_events', 'mean_z', 'mean_HSI', 'mean_sr_score']
    cog_tests = list(COG_TESTS.keys())

    n_sig, n_total = 0, 0
    for sm in sie_metrics:
        for ct in cog_tests:
            col = f'log_{ct}' if f'log_{ct}' in lemon_subj.columns else ct
            if col not in lemon_subj.columns:
                continue
            pair = lemon_subj[[sm, col, 'age_midpoint']].dropna()
            if len(pair) < 20:
                continue
            rho, p = spearmanr(pair[sm], pair[col])
            n_total += 1
            if p < 0.05:
                n_sig += 1
                # Age-partial
                from scipy.stats import linregress
                res_s = pair[sm] - linregress(pair['age_midpoint'], pair[sm]).slope * pair['age_midpoint']
                res_c = pair[col] - linregress(pair['age_midpoint'], pair[col]).slope * pair['age_midpoint']
                rho_p, p_p = spearmanr(res_s, res_c)
                surv = '*' if p_p < 0.05 else ''
                print(f"  {sm} x {ct}: rho={rho:+.3f} p={p:.3e}  partial: rho={rho_p:+.3f} p={p_p:.3e} {surv}")

    print(f"\n  {n_sig}/{n_total} nominally significant (p<0.05)")


def psychopathology(events):
    print("\n" + "=" * 80)
    print("PSYCHOPATHOLOGY (HBN)")
    print("=" * 80)
    hbn_dfs = []
    for release in ['R1', 'R2', 'R3', 'R4', 'R6']:
        path = f'/Volumes/T9/hbn_data/cmi_bids_{release}/participants.tsv'
        if os.path.isfile(path):
            df = pd.read_csv(path, sep='\t')
            hbn_dfs.append(df)
    if not hbn_dfs:
        print("  HBN demographics not available")
        return
    demo = pd.concat(hbn_dfs, ignore_index=True)
    demo = demo.rename(columns={'participant_id': 'subject_id'})
    for col in ['age', 'p_factor', 'attention', 'internalizing', 'externalizing']:
        demo[col] = pd.to_numeric(demo.get(col, pd.Series()), errors='coerce')

    hbn_events = events[events['_source_dir'].str.startswith('hbn_')]
    hbn_subj = hbn_events.groupby('subject_id').agg(
        n_events=('sr1_z_max', 'count'), mean_z=('sr1_z_max', 'mean'),
        mean_HSI=('HSI', 'mean'), mean_sr_score=('sr_score', 'mean'),
    ).reset_index()
    hbn_subj = hbn_subj.merge(demo[['subject_id', 'age', 'p_factor', 'attention',
                                     'internalizing', 'externalizing']].drop_duplicates(),
                               on='subject_id', how='left')
    n_sig = 0
    n_total = 0
    for sm in ['n_events', 'mean_z', 'mean_HSI', 'mean_sr_score']:
        for pc in ['p_factor', 'attention', 'internalizing', 'externalizing']:
            trip = hbn_subj[[sm, pc, 'age']].dropna()
            if len(trip) < 20:
                continue
            from scipy.stats import linregress
            res_s = trip[sm] - linregress(trip['age'], trip[sm]).slope * trip['age']
            res_p = trip[pc] - linregress(trip['age'], trip[pc]).slope * trip['age']
            rho, p = spearmanr(res_s, res_p)
            n_total += 1
            if p < 0.05:
                n_sig += 1
                print(f"  {sm} x {pc}: rho={rho:+.3f} p={p:.3e} N={len(trip)}")
    print(f"\n  {n_sig}/{n_total} significant after age-partialing")


def clinical_associations(events):
    print("\n" + "=" * 80)
    print("CLINICAL ASSOCIATIONS (TDBRAIN)")
    print("=" * 80)
    tdb_path = os.path.expanduser('~/Desktop/TDBRAIN_participants_V2_data/TDBRAIN_participants_V2.tsv')
    if not os.path.isfile(tdb_path):
        print("  TDBRAIN demographics not available")
        return
    demo = pd.read_csv(tdb_path, sep='\t')
    demo = demo.rename(columns={'participants_ID': 'subject_id'})

    tdb_events = events[events['_source_dir'] == 'tdbrain']
    tdb_subj = tdb_events.groupby('subject_id').agg(
        n_events=('sr1_z_max', 'count'), mean_z=('sr1_z_max', 'mean'),
        mean_HSI=('HSI', 'mean'), mean_sr_score=('sr_score', 'mean'),
    ).reset_index()
    tdb_subj = tdb_subj.merge(demo[['subject_id', 'indication']].drop_duplicates(),
                               on='subject_id', how='left')

    dx_groups = ['HEALTHY', 'ADHD', 'MDD', 'OCD', 'SMC']
    for m in ['n_events', 'mean_sr_score', 'mean_HSI']:
        print(f"\n  {m}:")
        group_data = {}
        for dx in dx_groups:
            sub = tdb_subj[tdb_subj['indication'].str.contains(dx, case=False, na=False)]
            vals = sub[m].dropna()
            if len(vals) >= 5:
                group_data[dx] = vals.values
                print(f"    {dx:>10s}: {vals.mean():.3f} +/- {vals.std():.3f} (N={len(vals)})")
        groups = [v for v in group_data.values() if len(v) >= 10]
        if len(groups) >= 2:
            F, p = stats.f_oneway(*groups)
            print(f"    ANOVA: F={F:.2f} p={p:.4e}")


def plv_analysis(events):
    print("\n" + "=" * 80)
    print("PLV vs RATIO PRECISION (aggregate)")
    print("=" * 80)
    plv = events['plv_sr1_pm5']
    print(f"  {'PLV Q':>8s} {'range':>16s} {'N':>6s} {'4-ratio err':>12s} {'sr5/sr3':>10s}")
    for q_lo, q_hi, label in [(0, 0.2, 'Q1'), (0.2, 0.4, 'Q2'), (0.4, 0.6, 'Q3'),
                                (0.6, 0.8, 'Q4'), (0.8, 1.0, 'Q5')]:
        lo, hi = plv.quantile(q_lo), plv.quantile(q_hi)
        sub = events[(plv >= lo) & (plv < hi if q_hi < 1.0 else plv <= hi)]
        errs = []
        for col, pred in [('sr5/sr3', PHI), ('sr3/sr1', PHI**2),
                           ('sr5/sr1', PHI**3), ('sr6/sr4', PHI)]:
            v = sub[col].dropna()
            v = v[(v > 0) & (v < 20)]
            if len(v) > 0:
                errs.append(abs(100 * (v.mean() - pred) / pred))
        r53 = sub['sr5/sr3'].dropna()
        r53 = r53[(r53 > 0) & (r53 < 20)]
        print(f"  {label:>8s} [{lo:.3f},{hi:.3f}] {len(sub):>6,} {np.mean(errs):>11.2f}% {r53.mean():>10.4f}")

    # Multi-harmonic PLV lock
    plv_cols = [f'plv_{lbl}_pm5' for lbl in LABELS]
    events['n_high_plv'] = (events[plv_cols] > 0.7).sum(axis=1)
    print(f"\n  Multi-harmonic phase-lock (PLV > 0.7):")
    print(f"  {'#harm':>6s} {'N':>6s} {'sr5/sr3':>10s} {'err from phi':>12s}")
    for n in [0, 3, 6, 9]:
        sub = events[events['n_high_plv'] == n]
        if len(sub) < 30:
            continue
        r53 = sub['sr5/sr3'].dropna()
        r53 = r53[(r53 > 0) & (r53 < 20)]
        if len(r53) > 0:
            err = 100 * (r53.mean() - PHI) / PHI
            print(f"  {n:>6d} {len(sub):>6,} {r53.mean():>10.4f} {err:>+11.2f}%")


def ratio_unpredictability(events):
    print("\n" + "=" * 80)
    print("EVENT-LEVEL RATIO UNPREDICTABILITY SCAN")
    print("=" * 80)
    for col, pred in [('sr5/sr3', PHI), ('sr3/sr1', PHI**2),
                       ('sr5/sr1', PHI**3), ('sr6/sr4', PHI)]:
        v = events[col]
        events[f'err_{col.replace("/", "_")}'] = ((v - pred) / pred * 100).abs()
    events['mean_ratio_err'] = events[['err_sr5_sr3', 'err_sr3_sr1',
                                        'err_sr5_sr1', 'err_sr6_sr4']].mean(axis=1)

    skip = {'subject_id', 'dataset', 'condition', 'session_name', '_source_dir',
            'ignition_freqs', 'err_sr5_sr3', 'err_sr3_sr1', 'err_sr5_sr1',
            'err_sr6_sr4', 'mean_ratio_err', 'sr3/sr1', 'sr5/sr1', 'sr5/sr3',
            'sr6/sr4', 'n_high_plv'}
    test_cols = [c for c in events.columns if c not in skip
                 and events[c].dtype in ['float64', 'int64', 'float32']]

    results = []
    for col in test_cols:
        pair = events[[col, 'mean_ratio_err']].dropna()
        pair = pair[pair['mean_ratio_err'] < 50]
        if len(pair) < 100:
            continue
        rho, p = spearmanr(pair[col], pair['mean_ratio_err'])
        results.append((col, rho, p, len(pair)))

    results.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"  Top 15 correlates with mean |ratio error|:")
    print(f"  {'Dimension':>25s} {'rho':>10s} {'p':>12s}")
    for col, rho, p, n in results[:15]:
        print(f"  {col:>25s} {rho:>+10.4f} {p:>12.2e}")
    print(f"\n  Max |rho| = {max(abs(r) for _, r, _, _ in results):.4f}")


def core_distribution(events):
    print("\n" + "=" * 80)
    print("CORE DISTRIBUTION (z = 3-5)")
    print("=" * 80)
    z = events['sr1_z_max']
    core = events[(z >= 3) & (z < 5)]
    print(f"  N = {len(core):,} ({100*len(core)/len(events):.1f}% of clean events)")

    print(f"\n  Ratio precision:")
    errs = []
    for col, pred, name in [('sr5/sr3', PHI, 'phi'), ('sr3/sr1', PHI**2, 'phi2'),
                              ('sr5/sr1', PHI**3, 'phi3'), ('sr6/sr4', PHI, 'phi')]:
        v = core[col].dropna()
        v = v[(v > 0) & (v < 20)]
        if len(v) > 0:
            err = 100 * (v.mean() - pred) / pred
            errs.append(abs(err))
            print(f"    {col} = {name}: {v.mean():.4f} ({err:+.2f}%) N={len(v):,}")
    print(f"    Mean absolute ratio error: {np.mean(errs):.2f}%")


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("Loading data...")
    events_raw = load_events()
    summaries = load_summaries()
    extraction = load_extraction_summaries()
    print(f"  Raw events: {len(events_raw):,}")

    events = apply_msc_filter(events_raw)
    print(f"  Unique subjects with events: {events['subject_id'].nunique():,}")
    print(f"  Datasets: {events['_source_dir'].nunique()}")

    dataset_overview(events, summaries, extraction)
    harmonic_frequencies(events)
    ratio_precision(events)
    core_distribution(events)
    independence_convergence(events)
    cross_dataset_anova(events)
    plv_analysis(events)
    ratio_unpredictability(events)
    ec_vs_eo(events, summaries)
    test_retest(events, summaries)
    age_trajectories(events, summaries)
    cognitive_correlates(events, summaries)
    psychopathology(events)
    clinical_associations(events)

    print("\n" + "=" * 80)
    print("ALL ANALYSES COMPLETE (MSC-filtered)")
    print("=" * 80)


if __name__ == '__main__':
    main()
