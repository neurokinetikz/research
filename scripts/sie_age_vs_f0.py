#!/usr/bin/env python3
"""
B59 — Does age explain the between-cohort f0 difference (B57)?

B57: per-subject median f0 differs across cohorts (KW p = 4e-6):
  LEMON 7.77, HBN R1-R6 7.67-7.75, TDBRAIN 7.81.
Alpha peak frequency is known to shift with age (lower in children,
peak in adolescence, declining in older adults). Pool all cohorts,
test continuous age × f0. If age explains most of the between-cohort
variance, cohort becomes non-significant after age-adjustment.
"""
from __future__ import annotations
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, kruskal
from scipy import stats

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'coupling')
EVENTS_ROOT = os.path.join(os.path.dirname(__file__), '..', 'exports_sie')
HBN_BASE = '/Volumes/T9/hbn_data'
LEMON_META = ('/Volumes/T9/lemon_data/behavioral/'
               'Behavioural_Data_MPILMBB_LEMON/'
               'META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv')
TDBRAIN_META = os.path.expanduser(
    '~/Desktop/TDBRAIN_participants_V2_data/TDBRAIN_participants_V2.tsv')

# ===== per-subject f0 medians (same as B57) =====
events_per_sub = []
for f in glob.glob(os.path.join(EVENTS_ROOT, 'lemon', 'sub-*_sie_events.csv')):
    d = pd.read_csv(f)
    if 'sr1' not in d.columns: continue
    d = d.dropna(subset=['sr1']); d = d[(d['sr1']>=7)&(d['sr1']<=8.3)]
    if len(d):
        sid = os.path.basename(f).replace('_sie_events.csv','')
        events_per_sub.append({'cohort':'LEMON','subject_id':sid,
                                'f_median':d['sr1'].median(),'n':len(d)})
for rel in ['R1','R2','R3','R4','R6']:
    d_dir = os.path.join(EVENTS_ROOT, f'hbn_{rel}')
    if not os.path.isdir(d_dir): continue
    for f in glob.glob(os.path.join(d_dir, 'sub-*_sie_events.csv')):
        d = pd.read_csv(f)
        if 'sr1' not in d.columns: continue
        d = d.dropna(subset=['sr1']); d = d[(d['sr1']>=7)&(d['sr1']<=8.3)]
        if len(d):
            sid = os.path.basename(f).replace('_sie_events.csv','')
            events_per_sub.append({'cohort':f'HBN_{rel}','subject_id':sid,
                                    'f_median':d['sr1'].median(),'n':len(d)})
for f in glob.glob(os.path.join(EVENTS_ROOT, 'tdbrain', 'sub-*_sie_events.csv')):
    d = pd.read_csv(f)
    if 'sr1' not in d.columns: continue
    d = d.dropna(subset=['sr1']); d = d[(d['sr1']>=7)&(d['sr1']<=8.3)]
    if len(d):
        sid = os.path.basename(f).replace('_sie_events.csv','')
        events_per_sub.append({'cohort':'TDBRAIN','subject_id':sid,
                                'f_median':d['sr1'].median(),'n':len(d)})
df = pd.DataFrame(events_per_sub)
print(f"Per-subject rows: {len(df)}")

# ===== merge ages =====
# LEMON
lm = pd.read_csv(LEMON_META)
lm.columns = [c.strip() for c in lm.columns]
def midpoint(a):
    if pd.isna(a): return np.nan
    a = str(a).strip()
    if '-' in a:
        try: return np.mean([float(x) for x in a.split('-')])
        except: return np.nan
    try: return float(a)
    except: return np.nan
lm['age'] = lm['Age'].apply(midpoint)
df = df.merge(lm[['ID','age']], left_on='subject_id', right_on='ID',
               how='left').drop(columns=['ID'],errors='ignore')

# HBN releases
h_meta = []
for rel in ['R1','R2','R3','R4','R6']:
    p = os.path.join(HBN_BASE, f'cmi_bids_{rel}', 'participants.tsv')
    if os.path.isfile(p):
        m = pd.read_csv(p, sep='\t').rename(
            columns={'participant_id':'subject_id'})
        h_meta.append(m[['subject_id','age']])
h_meta = pd.concat(h_meta, ignore_index=True)
h_meta['age'] = pd.to_numeric(h_meta['age'], errors='coerce')
# Fill HBN age into df
hbn_mask = df['cohort'].str.startswith('HBN_')
df_hbn_ages = df[hbn_mask].drop(columns=['age'],errors='ignore').merge(
    h_meta, on='subject_id', how='left')
# Replace rows
df = pd.concat([df[~hbn_mask], df_hbn_ages], ignore_index=True)

# TDBRAIN
td_m = pd.read_csv(TDBRAIN_META, sep='\t')
td_m.columns = [c.strip() for c in td_m.columns]
td_m = td_m.rename(columns={'participants_ID':'subject_id'})
td_m['age'] = (td_m['age'].astype(str).str.replace(',','.',regex=False)
                          .replace('nan',np.nan))
td_m['age'] = pd.to_numeric(td_m['age'], errors='coerce')
td_mask = df['cohort']=='TDBRAIN'
df_td_ages = df[td_mask].drop(columns=['age'],errors='ignore').merge(
    td_m[['subject_id','age']], on='subject_id', how='left')
df = pd.concat([df[~td_mask], df_td_ages], ignore_index=True)

df_valid = df.dropna(subset=['age','f_median']).copy()
print(f"With age: {len(df_valid)}  ({df_valid['cohort'].value_counts().to_dict()})")
df_valid.to_csv(os.path.join(OUT_DIR, 'b59_age_f0_per_subject.csv'),
                 index=False)

# ===== CONTINUOUS AGE × F0 (all pooled) =====
print(f"\n=== Continuous age × f_median (all cohorts pooled) ===")
rho_sp, p_sp = spearmanr(df_valid['age'], df_valid['f_median'])
r_p, p_p = pearsonr(df_valid['age'], df_valid['f_median'])
slope, intercept = np.polyfit(df_valid['age'], df_valid['f_median'], 1)
print(f"  Spearman ρ = {rho_sp:+.3f}  p = {p_sp:.2g}")
print(f"  Pearson r  = {r_p:+.3f}  p = {p_p:.2g}")
print(f"  OLS slope  = {slope:+.4f} Hz/year  intercept = {intercept:.2f}")

# Per-cohort
print(f"\n=== Age × f_median per cohort ===")
print(f"{'cohort':<10}{'n':>5}{'ρ':>8}{'p':>10}{'slope':>12}{'mean_age':>10}")
for coh, sub in df_valid.groupby('cohort'):
    if len(sub) < 10: continue
    rho, pp = spearmanr(sub['age'], sub['f_median'])
    slp, _ = np.polyfit(sub['age'], sub['f_median'], 1)
    print(f"{coh:<10}{len(sub):>5}{rho:>+8.3f}{pp:>10.2g}{slp:>+12.4f}"
          f"{sub['age'].mean():>10.1f}")

# ===== AGE-ADJUSTED COHORT TEST =====
# Residualize f_median on age (linear), then Kruskal-Wallis on residuals
# across cohorts. If cohort effect survives, there's a real cohort factor
# beyond age.
age_ols_slope, age_ols_int = np.polyfit(df_valid['age'], df_valid['f_median'], 1)
df_valid['f_residual'] = (df_valid['f_median']
                            - (age_ols_slope * df_valid['age'] + age_ols_int))

# Unadjusted KW
cohorts_with_n = [c for c,g in df_valid.groupby('cohort') if len(g)>=10]
groups_raw = [df_valid[df_valid['cohort']==c]['f_median'].values
               for c in cohorts_with_n]
stat_raw, p_raw = kruskal(*groups_raw)
print(f"\n=== KW across cohorts on f_median (raw) ===")
print(f"  H = {stat_raw:.2f}  p = {p_raw:.3g}")

groups_res = [df_valid[df_valid['cohort']==c]['f_residual'].values
               for c in cohorts_with_n]
stat_res, p_res = kruskal(*groups_res)
print(f"\n=== KW across cohorts on age-residualized f_median ===")
print(f"  H = {stat_res:.2f}  p = {p_res:.3g}")
print(f"\nAge-adjustment effect:")
print(f"  H dropped from {stat_raw:.1f} → {stat_res:.1f}  "
      f"(reduction: {(1-stat_res/stat_raw)*100:.0f}%)")
print(f"  p changed from {p_raw:.2g} → {p_res:.2g}")

# Per-cohort mean f_median before/after age-adjust
print(f"\n=== Per-cohort mean f_median (raw vs age-adjusted) ===")
print(f"{'cohort':<10}{'n':>5}{'raw mean':>12}{'adj mean':>12}{'diff':>10}"
      f"{'mean age':>10}")
for coh in cohorts_with_n:
    sub = df_valid[df_valid['cohort']==coh]
    print(f"{coh:<10}{len(sub):>5}{sub['f_median'].mean():>12.4f}"
          f"{sub['f_residual'].mean():>+12.4f}"
          f"{sub['f_residual'].mean() - (sub['f_median'].mean() - df_valid['f_median'].mean()):>+10.4f}"
          f"{sub['age'].mean():>10.1f}")

# ===== FIGURE =====
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
cohort_colors = {
    'LEMON':'#8c1a1a', 'HBN_R1':'#fdae61', 'HBN_R2':'#fdae61',
    'HBN_R3':'#fdae61', 'HBN_R4':'#fdae61', 'HBN_R6':'#fdae61',
    'TDBRAIN':'#2b5fb8',
}

# A — age × f_median scatter, all cohorts
ax = axes[0]
for coh in df_valid['cohort'].unique():
    sub = df_valid[df_valid['cohort']==coh]
    ax.scatter(sub['age'], sub['f_median'], s=14, alpha=0.5,
                color=cohort_colors.get(coh, 'gray'),
                edgecolor='k', lw=0.1,
                label=coh if coh in ['LEMON','TDBRAIN','HBN_R4'] else None)
xs = np.array(sorted(df_valid['age']))
ax.plot(xs, age_ols_slope * xs + age_ols_int, color='black', lw=1.5,
         label=f'OLS slope {age_ols_slope:+.4f} Hz/yr')
ax.axhline(7.83, color='red', ls=':', label='Schumann SR1')
ax.set_xlabel('age (years)'); ax.set_ylabel('per-subject f_median (Hz)')
ax.set_title(f'A — Age × f_median pooled\n'
              f'n={len(df_valid)}  ρ={rho_sp:+.2f}  p={p_sp:.2g}',
              loc='left', fontweight='bold', fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# B — per-cohort age × f_median (faceted)
ax = axes[1]
for coh, color in [('LEMON', '#8c1a1a'), ('HBN', '#1a9641'),
                    ('TDBRAIN', '#2b5fb8')]:
    if coh == 'HBN':
        sub = df_valid[df_valid['cohort'].str.startswith('HBN')]
    else:
        sub = df_valid[df_valid['cohort']==coh]
    if len(sub) < 10: continue
    ax.scatter(sub['age'], sub['f_median'], s=16, alpha=0.5,
                color=color, edgecolor='k', lw=0.1, label=coh)
    slp, inc = np.polyfit(sub['age'], sub['f_median'], 1)
    xs = np.array(sorted(sub['age']))
    ax.plot(xs, slp*xs + inc, color=color, lw=2, alpha=0.9)
ax.axhline(7.83, color='red', ls=':', label='SR1')
ax.set_xlabel('age'); ax.set_ylabel('f_median (Hz)')
ax.set_title('B — Per-cohort trends (super-cohorts)', loc='left',
              fontweight='bold', fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# C — cohort means: raw vs age-adjusted residual
ax = axes[2]
cohort_order = ['HBN_R1','HBN_R2','HBN_R3','HBN_R4','HBN_R6',
                 'LEMON','TDBRAIN']
cohort_order = [c for c in cohort_order if c in cohorts_with_n]
x = np.arange(len(cohort_order))
raw_means = [df_valid[df_valid['cohort']==c]['f_median'].mean()
             - df_valid['f_median'].mean()
             for c in cohort_order]
adj_means = [df_valid[df_valid['cohort']==c]['f_residual'].mean()
             for c in cohort_order]
w = 0.38
ax.bar(x-w/2, raw_means, w, label=f'raw (KW p={p_raw:.2g})',
        color='#d73027', alpha=0.75, edgecolor='k', lw=0.3)
ax.bar(x+w/2, adj_means, w, label=f'age-adjusted (KW p={p_res:.2g})',
        color='#2b5fb8', alpha=0.75, edgecolor='k', lw=0.3)
ax.axhline(0, color='k', lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(cohort_order, rotation=35, ha='right',
                                      fontsize=9)
ax.set_ylabel('f_median deviation from grand mean (Hz)')
ax.set_title(f'C — Cohort means: raw vs age-adjusted\n'
              f'H reduction: {(1-stat_res/stat_raw)*100:.0f}%',
              loc='left', fontweight='bold', fontsize=10)
ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

fig.suptitle('B59 — Age × f0 test: does age explain between-cohort '
              'f_median differences?', fontsize=12, y=1.02)
fig.tight_layout()
out_png = os.path.join(OUT_DIR, 'b59_age_vs_f0.png')
plt.savefig(out_png, dpi=160, bbox_inches='tight')
plt.savefig(out_png.replace('.png', '.pdf'), bbox_inches='tight')
plt.close()
print(f"\nSaved: {out_png}")
