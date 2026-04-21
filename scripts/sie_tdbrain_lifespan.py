#!/usr/bin/env python3
"""
B54 — TDBRAIN lifespan × sex analysis + combined lifespan view.

Extends B53's developmental sex finding (HBN 5-21, young girls drive effect)
into adulthood/aging using TDBRAIN (5-89 years).

Tests:
  1. TDBRAIN age × posterior-anterior SR1 contrast (continuous)
  2. TDBRAIN sex effect (same pattern as HBN?)
  3. TDBRAIN sex × age interaction
  4. Combined LEMON + HBN + TDBRAIN lifespan plot
  5. Exploratory: TDBRAIN clinical indication × contrast
  6. Exploratory: TDBRAIN sessSeason × contrast (Schumann seasonal amplitude)
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, wilcoxon, mannwhitneyu, kruskal

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'coupling')
B47_CSV = os.path.join(OUT_DIR, 'posterior_sr1_crosscohort.csv')
TDBRAIN_META = os.path.expanduser(
    '~/Desktop/TDBRAIN_participants_V2_data/TDBRAIN_participants_V2.tsv')
LEMON_META = ('/Volumes/T9/lemon_data/behavioral/Behavioural_Data_MPILMBB_LEMON/'
              'META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv')
HBN_BASE = '/Volumes/T9/hbn_data'

df = pd.read_csv(B47_CSV)
print(f"Cross-cohort B47 rows: {len(df)}")

# ===== TDBRAIN =====
td = df[df['cohort'] == 'tdbrain'].copy()
meta = pd.read_csv(TDBRAIN_META, sep='\t')
meta.columns = [c.strip() for c in meta.columns]
meta = meta.rename(columns={'participants_ID': 'subject_id'})

# Age stored as European comma-decimal "51,59" → 51.59
meta['age'] = (meta['age'].astype(str).str.replace(',', '.', regex=False)
                           .replace('nan', np.nan))
meta['age'] = pd.to_numeric(meta['age'], errors='coerce')
meta['gender'] = pd.to_numeric(meta['gender'], errors='coerce')
meta['sex'] = meta['gender'].map({0.0: 'F', 1.0: 'M'})

# Drop rows where meta fields are placeholders
keep_cols = ['subject_id', 'age', 'sex', 'indication', 'sessSeason',
              'sessTime', 'nrSessions', 'NEOFFI_OpennessForExperiences']
# NEO-FFI Openness derivation — LEMON uses that column name but TDBRAIN uses
# raw q1-60; skip for simplicity here
keep_cols = [c for c in keep_cols if c in meta.columns]
td_m = td.merge(meta[keep_cols], on='subject_id', how='left')
td_m = td_m.dropna(subset=['age', 'sex'])
print(f"TDBRAIN merged with age+sex: {len(td_m)} (from {len(td)} B47 rows)")

if len(td_m) >= 10:
    print(f"  Age: min {td_m['age'].min():.1f}  max {td_m['age'].max():.1f}  "
          f"mean {td_m['age'].mean():.1f}")
    print(f"  Sex: F={(td_m['sex']=='F').sum()}  M={(td_m['sex']=='M').sum()}")

    # Continuous age
    print(f"\n=== TDBRAIN continuous age ===")
    valid = td_m[['age', 'sr1_contrast']].dropna()
    rho, p = spearmanr(valid['age'], valid['sr1_contrast'])
    print(f"  age × contrast: n={len(valid)}  ρ={rho:+.3f}  p={p:.2g}")

    # Sex effect
    print(f"\n=== TDBRAIN sex effect ===")
    for s in ['F', 'M']:
        sub = td_m[td_m['sex'] == s]['sr1_contrast'].dropna()
        if len(sub) < 3: continue
        print(f"  {s}: n={len(sub)}  median={np.median(sub):+.3f}  "
              f"% post>ant={(sub>0).mean()*100:.0f}%")
    try:
        f = td_m[td_m['sex']=='F']['sr1_contrast'].dropna()
        m = td_m[td_m['sex']=='M']['sr1_contrast'].dropna()
        u, p = mannwhitneyu(f, m)
        print(f"  Mann-Whitney F vs M: p={p:.2g}")
    except Exception:
        pass

    # Sex × age-decade
    print(f"\n=== TDBRAIN sex × age-decade ===")
    td_m['age_dec'] = pd.cut(td_m['age'],
                               bins=[0, 20, 40, 60, 100],
                               labels=['<20', '20-39', '40-59', '60+'],
                               right=False)
    for dec in ['<20', '20-39', '40-59', '60+']:
        for s in ['F', 'M']:
            sub = td_m[(td_m['age_dec']==dec) & (td_m['sex']==s)]
            if len(sub) < 3: continue
            cm = np.median(sub['sr1_contrast'])
            print(f"  {dec:<7} {s}  n={len(sub):3d}  contrast={cm:+.3f}")

    # Indication (clinical)
    if 'indication' in td_m.columns:
        print(f"\n=== TDBRAIN clinical indication (n>=5) ===")
        for ind, sub in td_m.groupby('indication'):
            sub = sub.dropna(subset=['sr1_contrast'])
            if len(sub) < 5: continue
            print(f"  {str(ind):<40s}  n={len(sub):3d}  "
                  f"contrast median={np.median(sub['sr1_contrast']):+.3f}")

    # Session season (Schumann seasonal)
    if 'sessSeason' in td_m.columns:
        print(f"\n=== TDBRAIN sessSeason × contrast ===")
        valid_s = td_m.dropna(subset=['sessSeason', 'sr1_contrast'])
        valid_s = valid_s[valid_s['sessSeason'].isin(
            ['Winter','Spring','Summer','Fall'])]
        if len(valid_s) >= 10:
            groups = []
            labels = []
            for season in ['Winter', 'Spring', 'Summer', 'Fall']:
                sub = valid_s[valid_s['sessSeason'] == season]['sr1_contrast']
                if len(sub) < 3: continue
                labels.append(season); groups.append(sub.values)
                print(f"  {season:<7}: n={len(sub):3d}  "
                      f"median={np.median(sub):+.3f}")
            if len(groups) >= 2:
                try:
                    stat, p = kruskal(*groups)
                    print(f"  Kruskal-Wallis: p={p:.2g}")
                except Exception:
                    pass

# ===== LEMON sex & age =====
print(f"\n========== LEMON sex × age ==========")
le = df[df['cohort'] == 'lemon'].copy()
try:
    lm = pd.read_csv(LEMON_META)
    lm.columns = [c.strip() for c in lm.columns]
    sex_col = [c for c in lm.columns if 'Gender' in c][0]
    lm['sex'] = lm[sex_col].map({1: 'F', 2: 'M', '1': 'F', '2': 'M'})
    # Age is binned in LEMON (e.g. "20-25"). Extract midpoint.
    def midpoint(a):
        if pd.isna(a): return np.nan
        a = str(a).strip()
        if '-' in a:
            try:
                lo, hi = [float(x) for x in a.split('-')]
                return (lo + hi) / 2
            except Exception:
                return np.nan
        try:
            return float(a)
        except Exception:
            return np.nan
    lm['age_mid'] = lm['Age'].apply(midpoint)
    le_m = le.merge(lm[['ID', 'sex', 'age_mid']], left_on='subject_id',
                    right_on='ID', how='left').dropna(subset=['sex', 'age_mid'])
    print(f"LEMON merged: {len(le_m)}")
    for s in ['F', 'M']:
        sub = le_m[le_m['sex']==s]['sr1_contrast'].dropna()
        print(f"  {s}: n={len(sub)}  median={np.median(sub):+.3f}")
    # Age-decade in LEMON
    le_m['age_dec'] = pd.cut(le_m['age_mid'],
                              bins=[0, 30, 40, 50, 60, 100],
                              labels=['<30', '30-39', '40-49', '50-59', '60+'],
                              right=False)
    print(f"\nLEMON sex × age-decade:")
    for dec in ['<30', '30-39', '40-49', '50-59', '60+']:
        for s in ['F', 'M']:
            sub = le_m[(le_m['age_dec']==dec) & (le_m['sex']==s)]
            if len(sub) < 3: continue
            cm = np.median(sub['sr1_contrast'])
            print(f"  {dec:<7} {s}  n={len(sub):3d}  contrast={cm:+.3f}")
except Exception as e:
    print(f"LEMON merge failed: {e}")
    le_m = pd.DataFrame()

# ===== Combined lifespan dataset =====
print(f"\n========== Combined lifespan ==========")
# Build a stacked DF with (cohort, age, sex, contrast)
pieces = []
if len(le_m):
    pieces.append(pd.DataFrame({'cohort': 'LEMON', 'age': le_m['age_mid'],
                                 'sex': le_m['sex'],
                                 'contrast': le_m['sr1_contrast']}))
# HBN
hbn = df[df['cohort'].str.startswith('hbn') & (df['cohort']!='hbn_all')].copy()
meta_list = []
for rel in ['R1','R2','R3','R4','R6']:
    p = os.path.join(HBN_BASE, f'cmi_bids_{rel}', 'participants.tsv')
    if os.path.isfile(p):
        mm = pd.read_csv(p, sep='\t').rename(columns={'participant_id':'subject_id'})
        meta_list.append(mm)
h_meta = pd.concat(meta_list, ignore_index=True)
hbn_m = hbn.merge(h_meta[['subject_id','age','sex']], on='subject_id',
                   how='left').dropna(subset=['age','sex'])
hbn_m['age'] = pd.to_numeric(hbn_m['age'], errors='coerce')
hbn_m = hbn_m.dropna(subset=['age'])
pieces.append(pd.DataFrame({'cohort': 'HBN', 'age': hbn_m['age'],
                             'sex': hbn_m['sex'],
                             'contrast': hbn_m['sr1_contrast']}))
if len(td_m):
    pieces.append(pd.DataFrame({'cohort': 'TDBRAIN', 'age': td_m['age'],
                                 'sex': td_m['sex'],
                                 'contrast': td_m['sr1_contrast']}))
combined = pd.concat(pieces, ignore_index=True).dropna()
print(f"Combined n: {len(combined)}  "
      f"(LEMON={len(combined[combined['cohort']=='LEMON'])}  "
      f"HBN={len(combined[combined['cohort']=='HBN'])}  "
      f"TDBRAIN={len(combined[combined['cohort']=='TDBRAIN'])})")
combined.to_csv(os.path.join(OUT_DIR, 'lifespan_sex_contrast.csv'),
                 index=False)

# Correlation age × contrast separately by sex over full lifespan
for s in ['F', 'M']:
    sub = combined[combined['sex']==s].dropna()
    rho, p = spearmanr(sub['age'], sub['contrast'])
    print(f"  {s} lifespan age×contrast: n={len(sub)}  ρ={rho:+.3f}  p={p:.2g}")

# ===== FIGURE =====
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# A — Lifespan scatter colored by sex, faceted (or combined)
ax = axes[0, 0]
for s, c in [('F', '#d73027'), ('M', '#4575b4')]:
    sub = combined[combined['sex']==s]
    ax.scatter(sub['age'], np.clip(sub['contrast'], -20, 30),
                s=18, alpha=0.5, color=c, edgecolor='k', lw=0.2,
                label=f'{s}  n={len(sub)}')
ax.axhline(0, color='k', lw=0.8)
ax.set_xlabel('age (years)'); ax.set_ylabel('posterior − anterior SR1 contrast (clipped)')
ax.set_title('A — Lifespan × sex (all cohorts)', loc='left',
              fontweight='bold', fontsize=11)
ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(-10, 20)

# B — Binned lifespan: decade × sex boxplots
ax = axes[0, 1]
combined['age_dec'] = pd.cut(combined['age'],
                               bins=[0, 10, 20, 30, 50, 70, 100],
                               labels=['<10','10-19','20-29','30-49',
                                       '50-69','70+'], right=False)
positions = []
box_data = []
box_colors = []
box_labels = []
x = 0
for dec in ['<10','10-19','20-29','30-49','50-69','70+']:
    for s, c in [('F', '#d73027'), ('M', '#4575b4')]:
        sub = combined[(combined['age_dec']==dec)&(combined['sex']==s)]
        if len(sub) < 3:
            continue
        box_data.append(sub['contrast'].values)
        positions.append(x)
        box_colors.append(c)
        box_labels.append(f'{dec}\n{s} n={len(sub)}')
        x += 1
    x += 0.5
bp = ax.boxplot(box_data, positions=positions, widths=0.8,
                 patch_artist=True, showfliers=False,
                 tick_labels=box_labels)
for patch, c in zip(bp['boxes'], box_colors):
    patch.set_facecolor(c); patch.set_alpha(0.6)
ax.axhline(0, color='k', lw=0.8)
ax.set_ylabel('posterior − anterior SR1 contrast')
ax.set_title('B — Lifespan decade × sex', loc='left',
              fontweight='bold', fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.tick_params(axis='x', labelsize=7)
ax.set_ylim(-10, 20)

# C — TDBRAIN sessSeason effect
ax = axes[1, 0]
if len(td_m) and 'sessSeason' in td_m.columns:
    seasons = ['Winter', 'Spring', 'Summer', 'Fall']
    season_data = []; season_labels = []
    for season in seasons:
        sub = td_m[td_m['sessSeason']==season]['sr1_contrast'].dropna()
        if len(sub) < 2: continue
        season_data.append(sub.values)
        season_labels.append(f'{season}\nn={len(sub)}')
    if season_data:
        bp = ax.boxplot(season_data, tick_labels=season_labels,
                         patch_artist=True, showfliers=False)
        for patch, c in zip(bp['boxes'],
                             ['#4575b4','#91bfdb','#fdae61','#d73027']):
            patch.set_facecolor(c); patch.set_alpha(0.6)
        ax.axhline(0, color='k', lw=0.8)
        ax.set_ylabel('posterior − anterior SR1 contrast')
        ax.set_title('C — TDBRAIN sessSeason (Schumann seasonal amplitude)',
                      loc='left', fontweight='bold', fontsize=11)
        ax.grid(axis='y', alpha=0.3)

# D — Per-cohort sex summary bars
ax = axes[1, 1]
cohort_order = ['HBN', 'LEMON', 'TDBRAIN']
bar_data = []; bar_labels = []; bar_colors = []
for coh in cohort_order:
    for s, c in [('F', '#d73027'), ('M', '#4575b4')]:
        sub = combined[(combined['cohort']==coh)&(combined['sex']==s)]
        if len(sub) < 3:
            continue
        med = np.median(sub['contrast'])
        bar_data.append(med)
        bar_labels.append(f'{coh}\n{s} n={len(sub)}')
        bar_colors.append(c)
ax.bar(range(len(bar_data)), bar_data, color=bar_colors, edgecolor='k', lw=0.3,
        alpha=0.7)
ax.set_xticks(range(len(bar_labels)))
ax.set_xticklabels(bar_labels, fontsize=9)
ax.axhline(0, color='k', lw=0.8)
ax.set_ylabel('median posterior − anterior SR1 contrast')
ax.set_title('D — Cohort × sex summary', loc='left',
              fontweight='bold', fontsize=11)
ax.grid(axis='y', alpha=0.3)

fig.suptitle('B54 — Lifespan × sex pattern of posterior-α SIE dominance',
              fontsize=13, y=1.01)
fig.tight_layout()
out_png = os.path.join(OUT_DIR, 'lifespan_sex_sr1.png')
plt.savefig(out_png, dpi=160, bbox_inches='tight')
plt.savefig(out_png.replace('.png', '.pdf'), bbox_inches='tight')
plt.close()
print(f"\nSaved: {out_png}")
