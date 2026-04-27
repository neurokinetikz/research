#!/usr/bin/env python3
"""LEMON cognitive battery × Q4 SR1 source-space region ratios.

Correlates per-subject cognitive test scores with per-subject source ratios in
key regions (PCC, PHG, ACC, IFG, lingual) for LEMON EC and LEMON EO.

Uses LEMON cognitive battery (CVLT, LPS, RWT, TAP-Alertness, TAP-Incompatibility,
TAP-Working Memory, TMT, WST) from /Volumes/T9/lemon_data/behavioral/.

Outputs:
  outputs/2026-04-24-lemon-cognitive-source/
    lemon_cognitive_per_subject.csv
    lemon_cognitive_spearman.csv
    lemon_cognitive_heatmap.png
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import mne

ROOT = Path(__file__).parent.parent
SOURCE_DIR = ROOT / 'outputs' / 'schumann' / 'images' / 'source'
LEMON_BEH = Path('/Volumes/T9/lemon_data/behavioral/Behavioural_Data_MPILMBB_LEMON/Cognitive_Test_Battery_LEMON')
LEMON_META = Path('/Volumes/T9/lemon_data/behavioral/Behavioural_Data_MPILMBB_LEMON/META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv')
OUT = ROOT / 'outputs' / '2026-04-24-lemon-cognitive-source'
OUT.mkdir(exist_ok=True)


def _get_label_masks():
    subjects_dir = os.path.dirname(mne.datasets.fetch_fsaverage(verbose=False))
    labels = mne.read_labels_from_annot('fsaverage', 'aparc',
                                         subjects_dir=subjects_dir, verbose=False)
    return {L.name: L for L in labels}


def _per_subject_region_ratios(cohort_composite, regions_hemis, masks):
    stc_dir = SOURCE_DIR / cohort_composite / 'stcs'
    if not stc_dir.is_dir():
        return pd.DataFrame()
    rows = []
    for f in sorted(stc_dir.glob('*_Q4_SR1_ratio-lh.stc')):
        sub_id = f.name.replace('_Q4_SR1_ratio-lh.stc', '')
        try:
            stc = mne.read_source_estimate(str(f).replace('-lh.stc', ''), 'fsaverage')
        except Exception:
            continue
        row = {'subject_id': sub_id}
        for region, hemi in regions_hemis:
            lbl = masks.get(f'{region}-{hemi}')
            if lbl is None:
                continue
            try:
                row[f'{region}-{hemi}'] = float(np.mean(stc.in_label(lbl).data[:, 0]))
            except Exception:
                pass
        rows.append(row)
    return pd.DataFrame(rows)


def _load_cognitive_battery():
    """Load all LEMON cognitive CSVs, merge by ID."""
    tests = {
        'CVLT': 'CVLT /CVLT.csv',
        'LPS': 'LPS/LPS.csv',
        'RWT': 'RWT/RWT.csv',
        'TAP_A': 'TAP_Alertness/TAP-Alertness.csv',
        'TAP_I': 'TAP_Incompatibility/TAP-Incompatibility.csv',
        'TAP_WM': 'TAP_Working_Memory/TAP-Working Memory.csv',
        'TMT': 'TMT/TMT.csv',
        'WST': 'WST/WST.csv',
    }
    merged = None
    for key, path in tests.items():
        fp = LEMON_BEH / path
        if not fp.exists():
            print(f'  missing: {fp}')
            continue
        df = pd.read_csv(fp)
        # LEMON IDs in these files are plain (no sub- prefix). Normalize:
        df['subject_id'] = df['ID'].astype(str).apply(
            lambda x: x if x.startswith('sub-') else f'sub-{x}')
        df = df.drop(columns=['ID'])
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on='subject_id', how='outer')
    return merged


def _load_lemon_meta():
    m = pd.read_csv(LEMON_META)
    m = m.rename(columns={'ID': 'subject_id_raw'})
    m['subject_id'] = m['subject_id_raw'].astype(str).apply(
        lambda x: x if x.startswith('sub-') else f'sub-{x}')
    # age binned
    def _age_mid(v):
        if pd.isna(v):
            return np.nan
        s = str(v)
        if '-' in s:
            try:
                lo, hi = s.split('-')
                return (float(lo) + float(hi)) / 2
            except Exception:
                return np.nan
        try:
            return float(s)
        except Exception:
            return np.nan
    m['age'] = m['Age'].apply(_age_mid)
    m['gender'] = m['Gender_ 1=female_2=male']
    return m[['subject_id', 'age', 'gender']]


def main():
    masks = _get_label_masks()
    regions_hemis = [
        ('posteriorcingulate', 'rh'),
        ('posteriorcingulate', 'lh'),
        ('parahippocampal', 'rh'),
        ('parahippocampal', 'lh'),
        ('caudalanteriorcingulate', 'lh'),
        ('parsopercularis', 'rh'),
        ('parsopercularis', 'lh'),
        ('lingual', 'lh'),
        ('precuneus', 'rh'),
        ('bankssts', 'lh'),
    ]

    # ---- Load source ratios ----
    print('Loading LEMON EC source ratios...')
    ec = _per_subject_region_ratios('lemon_composite', regions_hemis, masks)
    ec['condition'] = 'EC'
    print(f'  n={len(ec)}')
    print('Loading LEMON EO source ratios...')
    eo = _per_subject_region_ratios('lemon_EO_composite', regions_hemis, masks)
    eo['condition'] = 'EO'
    print(f'  n={len(eo)}')
    ratios = pd.concat([ec, eo], ignore_index=True)

    # ---- Load cognitive + demographics ----
    print('Loading cognitive battery...')
    cog = _load_cognitive_battery()
    print(f'  {len(cog)} subjects, {len(cog.columns)-1} cognitive variables')
    meta = _load_lemon_meta()
    cog = cog.merge(meta, on='subject_id', how='left')

    # ---- Merge ----
    merged = ratios.merge(cog, on='subject_id', how='inner')
    print(f'\nMerged n={len(merged)} subject-conditions '
          f'(EC={sum(merged.condition=="EC")}, EO={sum(merged.condition=="EO")})')
    merged.to_csv(OUT / 'lemon_cognitive_per_subject.csv', index=False)

    region_names = [f'{r}-{h}' for r, h in regions_hemis if f'{r}-{h}' in merged.columns]
    cognitive_cols = [c for c in merged.columns
                      if c.startswith(('CVLT_', 'LPS_', 'RWT_', 'TAP_A_',
                                        'TAP_I_', 'TAP_WM_', 'TMT_', 'WST_'))]
    print(f'  {len(region_names)} regions × {len(cognitive_cols)} cognitive variables = {len(region_names)*len(cognitive_cols)} tests per condition')

    # ---- Spearman per condition ----
    print('\n=== Running correlations per condition ===')
    all_results = []
    for cond in ['EC', 'EO']:
        sub = merged[merged['condition'] == cond]
        for reg in region_names:
            for cvar in cognitive_cols:
                v = sub[[reg, cvar, 'age']].dropna()
                # Filter extreme outliers in ratio
                v = v[(v[reg] > 0.1) & (v[reg] < 10)]
                if len(v) < 30:
                    continue
                # Convert cognitive to numeric if not
                v[cvar] = pd.to_numeric(v[cvar], errors='coerce')
                v = v.dropna(subset=[cvar])
                if len(v) < 30:
                    continue
                rho, p = spearmanr(v[reg], v[cvar])
                # Partial controlling for age
                df_vals = v[[reg, cvar, 'age']].dropna()
                if len(df_vals) >= 20:
                    ranks = df_vals.rank()
                    from numpy.linalg import lstsq
                    z = ranks['age'].values.reshape(-1, 1)
                    z_aug = np.hstack([z, np.ones_like(z)])
                    bx, *_ = lstsq(z_aug, ranks[reg].values, rcond=None)
                    by, *_ = lstsq(z_aug, ranks[cvar].values, rcond=None)
                    x_res = ranks[reg].values - z_aug @ bx
                    y_res = ranks[cvar].values - z_aug @ by
                    rho_p, p_p = spearmanr(x_res, y_res)
                else:
                    rho_p, p_p = np.nan, np.nan
                all_results.append({
                    'condition': cond,
                    'region_hemi': reg,
                    'cognitive_var': cvar,
                    'n': len(v),
                    'spearman_rho': rho,
                    'spearman_p': p,
                    'partial_rho_age_ctrl': rho_p,
                    'partial_p_age_ctrl': p_p,
                })
    df = pd.DataFrame(all_results)
    df.to_csv(OUT / 'lemon_cognitive_spearman.csv', index=False)

    print(f'Total tests: {len(df)}')
    nsig = (df['spearman_p'] < 0.05).sum()
    print(f'Nominally significant (p<0.05): {nsig} ({nsig/len(df)*100:.1f}%)')
    # FDR (Benjamini-Hochberg manual)
    def _bh_fdr(ps, alpha=0.05):
        ps = np.asarray(ps)
        m = len(ps)
        order = np.argsort(ps)
        ranks = np.arange(1, m + 1)
        p_sorted = ps[order]
        q = p_sorted * m / ranks
        # enforce monotonicity
        q = np.minimum.accumulate(q[::-1])[::-1]
        q_unsorted = np.empty_like(q)
        q_unsorted[order] = q
        reject = q_unsorted < alpha
        return reject, q_unsorted

    for cond in ['EC', 'EO']:
        sub = df[df['condition']==cond].copy()
        if len(sub) == 0:
            continue
        reject, p_fdr = _bh_fdr(sub['spearman_p'].values)
        sub = sub.copy()
        sub['fdr_reject'] = reject
        sub['p_fdr'] = p_fdr
        nsig_fdr = reject.sum()
        print(f'  {cond}: {nsig_fdr}/{len(sub)} FDR-significant (q<0.05)')
        if nsig_fdr > 0:
            top = sub[reject].sort_values('spearman_p')
            print(f'  Top FDR-surviving:')
            for _, row in top.head(10).iterrows():
                print(f"    {row['region_hemi']:25s} × {row['cognitive_var']:12s}  "
                      f"n={row['n']:3d}  ρ={row['spearman_rho']:+.3f}  p={row['spearman_p']:.3g}  q={row['p_fdr']:.3g}")

    # ---- Top uncorrected ----
    print('\n=== Top 20 uncorrected results (sorted by |ρ|) ===')
    df['abs_rho'] = df['spearman_rho'].abs()
    top = df.sort_values('abs_rho', ascending=False).head(20)
    for _, row in top.iterrows():
        sig = '***' if row['spearman_p']<0.001 else '**' if row['spearman_p']<0.01 else '*' if row['spearman_p']<0.05 else ''
        print(f"  [{row['condition']}] {row['region_hemi']:25s} × {row['cognitive_var']:12s}  "
              f"n={row['n']:3d}  ρ={row['spearman_rho']:+.3f} p={row['spearman_p']:.3g} {sig}")

    # ---- Heatmap: mean |ρ| per region per cognitive domain ----
    def _domain(cvar):
        for pref in ['CVLT_', 'LPS_', 'RWT_', 'TAP_A_', 'TAP_I_', 'TAP_WM_', 'TMT_', 'WST_']:
            if cvar.startswith(pref):
                return pref.rstrip('_')
        return cvar
    df['domain'] = df['cognitive_var'].apply(_domain)
    # pivot: condition × region × domain, mean |ρ|
    pivot_ec = df[df['condition']=='EC'].pivot_table(
        values='abs_rho', index='region_hemi', columns='domain', aggfunc='mean')
    pivot_eo = df[df['condition']=='EO'].pivot_table(
        values='abs_rho', index='region_hemi', columns='domain', aggfunc='mean')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, pivot, title in [(axes[0], pivot_ec, 'LEMON EC'), (axes[1], pivot_eo, 'LEMON EO')]:
        im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.3)
        ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, rotation=30, ha='right')
        ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
        ax.set_title(f'{title}: Mean |ρ| per region × cognitive domain')
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                ax.text(j, i, f'{pivot.values[i,j]:.2f}', ha='center', va='center', fontsize=7)
        plt.colorbar(im, ax=ax, shrink=0.7)
    plt.tight_layout()
    plt.savefig(OUT / 'lemon_cognitive_heatmap.png', dpi=120, bbox_inches='tight')
    plt.close()
    print(f'\nSaved {OUT / "lemon_cognitive_heatmap.png"}')


if __name__ == '__main__':
    main()
