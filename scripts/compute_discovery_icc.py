#!/usr/bin/env python3
"""
Compute three-level ICC on the discovery cohort under the canonical
MixedLM specification that Methods 2.4 commits to.

Aggregates per-event CSVs from all 6 discovery datasets (pulled back from
gs://eeg-extraction-data/results/exports_sie/) and fits a linear
mixed-effects model with random intercepts at subject and session levels:

    ratio ~ 1 + (1 | subject) + (1 | session:subject)

ICC_subject = var_subject / (var_subject + var_session + var_residual)

Reports per-ratio (SR3/SR1, SR5/SR1, SR5/SR3) subject ICC plus full
variance-decomposition percentages — the canonical numbers that will
replace the PDF draft's 8-12 / 39-45 / 45-49 in Section 3.1 P3.

Usage:
    # 1. Pull exports from GCS (skip if already local)
    gcloud storage cp -r gs://eeg-extraction-data/results/exports_sie exports_sie_discovery

    # 2. Run aggregation + ICC
    python scripts/compute_discovery_icc.py exports_sie_discovery
"""

import os
import sys
import glob
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings('ignore')

DISCOVERY_DATASETS = ['muse', 'epoc_self', 'insight_self', 'physf', 'mpeng', 'vep']


def load_events(exports_base):
    """Load per-event CSVs from all discovery-dataset subfolders."""
    frames = []
    for ds in DISCOVERY_DATASETS:
        ds_dir = os.path.join(exports_base, ds)
        if not os.path.isdir(ds_dir):
            print(f"  {ds}: directory not found, skipping")
            continue
        files = sorted(glob.glob(os.path.join(ds_dir, 'sub-*_sie_events.csv'))
                       + glob.glob(os.path.join(ds_dir, '*_sie_events.csv')))
        for f in files:
            try:
                df = pd.read_csv(f, low_memory=False)
                if df.empty or 'sr1' not in df.columns:
                    continue
                df['dataset'] = ds
                frames.append(df)
            except Exception as e:
                print(f"    skip {os.path.basename(f)}: {e}")
        print(f"  {ds}: {len(files)} subjects, "
              f"{sum(len(f_) for f_ in frames if len(f_) and f_['dataset'].iloc[0] == ds)} events loaded")
    if not frames:
        raise SystemExit("No events loaded")
    return pd.concat(frames, ignore_index=True)


def compute_ratios(df):
    """Ensure per-event ratio columns are numeric and non-NaN where possible."""
    for col in ('sr1', 'sr3', 'sr5', 'sr6'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df['r_sr3_sr1'] = df['sr3'] / df['sr1']
    df['r_sr5_sr1'] = df['sr5'] / df['sr1']
    df['r_sr5_sr3'] = df['sr5'] / df['sr3']
    if 'sr6' in df.columns and 'sr4' in df.columns:
        df['sr4'] = pd.to_numeric(df['sr4'], errors='coerce')
        df['r_sr6_sr4'] = df['sr6'] / df['sr4']
    return df


def reconstruct_subjects(df):
    """Collapse extraction-level subject_ids into biological subjects.

    Extraction outputs one subject_id per recording file, but multiple files
    often come from the same human (Michael's 37 sessions across 3 devices;
    PhySF's flow/no_flow pairs). This function reconstructs:
      - true_subject: biological subject identity
      - session: within-subject repeat identifier
    """
    df = df.copy()

    def _subject_session(row):
        ds = row['dataset']
        sid = str(row['subject_id'])
        if ds in ('muse', 'epoc_self', 'insight_self'):
            return 'michael', f'{ds}_{sid}'
        if ds == 'physf':
            # physf_s10_flow → biological 's10', session 'flow'
            # physf_s10_no_flow → biological 's10', session 'no_flow'
            stem = sid.replace('physf_', '')
            for cond in ('no_flow', 'flow'):
                if stem.endswith(f'_{cond}'):
                    return stem[:-(len(cond) + 1)], cond
            return stem, 'unknown'
        if ds == 'mpeng':
            stem = sid.replace('mpeng_', '')
            return stem, 'task_concat'
        if ds == 'vep':
            stem = sid.replace('vep_', '')
            return stem, 'perception_concat'
        return sid, sid

    pairs = df.apply(_subject_session, axis=1)
    df['true_subject'] = pairs.str[0]
    df['biological_session'] = pairs.str[1]
    return df


def fit_icc(df, ratio_col, label):
    """Three-level MixedLM: ratio ~ 1 + (1|true_subject) + (1|session:subject).

    Uses reconstructed biological subject identity + within-subject session.
    """
    d = df[[ratio_col, 'true_subject', 'biological_session', 'dataset']].copy()
    d = d.rename(columns={'biological_session': 'session'})
    d = d.dropna(subset=[ratio_col, 'true_subject', 'session'])
    d = d[np.isfinite(d[ratio_col])]
    if len(d) < 30:
        print(f"  {label}: skipped (N={len(d)} too small)")
        return None
    try:
        md = smf.mixedlm(f'{ratio_col} ~ 1', data=d, groups=d['true_subject'],
                         re_formula='~1',
                         vc_formula={'session': '0 + C(session)'})
        mdf = md.fit(method='lbfgs', reml=True)
        v_subj = float(mdf.cov_re.iloc[0, 0])
        v_sess = float(list(mdf.vcomp)[0]) if len(mdf.vcomp) > 0 else 0.0
        v_resid = float(mdf.scale)
        total = v_subj + v_sess + v_resid
        if total <= 0:
            print(f"  {label}: zero total variance")
            return None
        return {
            'ratio': label,
            'n_events': len(d),
            'n_subjects': d['true_subject'].nunique(),
            'n_sessions': d['session'].nunique(),
            'mean': float(d[ratio_col].mean()),
            'icc_subject': v_subj / total,
            'pct_subject': 100 * v_subj / total,
            'pct_session': 100 * v_sess / total,
            'pct_residual': 100 * v_resid / total,
            'var_subject': v_subj,
            'var_session': v_sess,
            'var_residual': v_resid,
        }
    except Exception as e:
        print(f"  {label}: FAIL — {type(e).__name__}: {e}")
        return None


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else 'exports_sie_discovery'
    if not os.path.isdir(base):
        raise SystemExit(f"Directory not found: {base}")

    print(f"Loading events from {base}...")
    df = load_events(base)
    df = compute_ratios(df)
    df = reconstruct_subjects(df)

    print(f"\nTotal: {len(df)} events, "
          f"{df['true_subject'].nunique()} biological subjects "
          f"({df['subject_id'].nunique()} extraction-level subject_ids), "
          f"{df['dataset'].nunique()} datasets")
    print(f"\nPer-dataset event counts:")
    print(df.groupby('dataset')['subject_id'].agg(['nunique', 'count']).to_string())

    ratios = [
        ('r_sr3_sr1', 'SR3/SR1', 2.6180),  # φ²
        ('r_sr5_sr1', 'SR5/SR1', 4.2361),  # φ³
        ('r_sr5_sr3', 'SR5/SR3', 1.6180),  # φ
    ]
    if 'r_sr6_sr4' in df.columns:
        ratios.append(('r_sr6_sr4', 'SR6/SR4', 1.6180))  # φ

    print(f"\nThree-level MixedLM ICC (canonical specification):")
    print(f"  ratio ~ 1 + (1 | subject) + (1 | session:subject)")
    print(f"  REML, random intercepts at subject and session\n")
    print(f"{'Ratio':<10} {'N events':>10} {'N subj':>8} {'Mean':>9} "
          f"{'φⁿ pred':>9} {'ICC_subj':>9} {'%subj':>7} {'%sess':>7} {'%resid':>7}")
    print("-" * 85)
    rows = []
    for col, label, pred in ratios:
        res = fit_icc(df, col, label)
        if res is None:
            continue
        res['phi_n_pred'] = pred
        rows.append(res)
        print(f"{label:<10} {res['n_events']:>10} {res['n_subjects']:>8} "
              f"{res['mean']:>9.4f} {pred:>9.4f} {res['icc_subject']:>9.4f} "
              f"{res['pct_subject']:>6.1f}% {res['pct_session']:>6.1f}% "
              f"{res['pct_residual']:>6.1f}%")

    if rows:
        out = pd.DataFrame(rows)
        out_path = os.path.join('outputs', 'discovery_icc_mixedlm.csv')
        os.makedirs('outputs', exist_ok=True)
        out.to_csv(out_path, index=False)
        print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
