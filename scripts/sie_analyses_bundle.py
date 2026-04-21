#!/usr/bin/env python3
"""
B56-B58 bundle — three local analyses.

B56: LEMON sex × cognitive stratified analysis (extended battery).
     Tests whether B51's null correlations become significant within-sex,
     with more cognitive measures than B51.

B57: Between-cohort f0 comparison.
     Tests whether per-subject median f0 differs by cohort (LEMON vs HBN
     releases vs TDBRAIN), which would suggest cohort-level environmental
     drivers (e.g., recording-site geomagnetism) or methodological factors.

B58: Global template_rho canonicality ranking.
     Pools events across all cohorts, globally ranks by template_rho, and
     examines composition of top 10%/25% (which cohort/subject/age/sex
     populates the tail of globally-canonical events).
"""
from __future__ import annotations
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, mannwhitneyu, kruskal, wilcoxon

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'coupling')
QUALITY_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs',
                            'schumann', 'images', 'quality')
EVENTS_ROOT = os.path.join(os.path.dirname(__file__), '..', 'exports_sie')
B47_CSV = os.path.join(OUT_DIR, 'posterior_sr1_crosscohort.csv')

# =========================================================================
# B56 — LEMON sex × cognitive stratified
# =========================================================================
print("=" * 70)
print("B56 — LEMON sex × cognitive stratified analysis")
print("=" * 70)

LEMON_BASE = ('/Volumes/T9/lemon_data/behavioral/'
               'Behavioural_Data_MPILMBB_LEMON')
COG = os.path.join(LEMON_BASE, 'Cognitive_Test_Battery_LEMON')
EMO = os.path.join(LEMON_BASE, 'Emotion_and_Personality_Test_Battery_LEMON')
META_PATH = os.path.join(LEMON_BASE,
                          'META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv')

summary = pd.read_csv(os.path.join(EVENTS_ROOT, 'lemon',
                                    'extraction_summary.csv'))
ok = summary[summary['status']=='ok'].copy()
ok['total_rate_per_min'] = ok['n_events'] / (ok['duration_sec']/60)
qual = pd.read_csv(os.path.join(QUALITY_DIR,
                                 'per_event_quality.csv')).dropna(subset=['template_rho']).copy()
qual['rho_q'] = pd.qcut(qual['template_rho'], 4,
                          labels=['Q1','Q2','Q3','Q4'])
q4_counts = qual.groupby('subject_id').apply(
    lambda d: (d['rho_q']=='Q4').sum()).rename('q4_n').reset_index()
rate = ok.merge(q4_counts, on='subject_id', how='left')
rate['q4_rate_per_min'] = rate['q4_n'] / (rate['duration_sec']/60)

# Load extended cognitive battery
nyc = pd.read_csv(os.path.join(EMO, 'NYC_Q_lemon.csv'))
nyc.columns = [c.strip() for c in nyc.columns]
nyc = nyc[[c for c in nyc.columns if not c.startswith('Unnamed')]]
items = [c for c in nyc.columns if c.startswith('NYC-Q_lemon_')]
nyc[items] = nyc[items].apply(pd.to_numeric, errors='coerce')
nyc['nyc_content'] = nyc[[f'NYC-Q_lemon_{i}' for i in range(1,24)]].mean(axis=1)
nyc['nyc_selfref'] = nyc[[f'NYC-Q_lemon_{i}' for i in [3,4,7,11,12,18,22]]].mean(axis=1)
neo = pd.read_csv(os.path.join(EMO, 'NEO_FFI.csv'))
neo.columns = [c.strip() for c in neo.columns]
tap = pd.read_csv(os.path.join(COG, 'TAP_Alertness', 'TAP-Alertness.csv'))
tap.columns = [c.strip() for c in tap.columns]
tap['TAP_A_5'] = pd.to_numeric(tap['TAP_A_5'], errors='coerce')
tap['TAP_A_15'] = pd.to_numeric(tap['TAP_A_15'], errors='coerce')
tap_wm = pd.read_csv(os.path.join(COG, 'TAP_Working_Memory',
                                   'TAP-Working Memory.csv'))
tap_wm.columns = [c.strip() for c in tap_wm.columns]
tap_wm['TAP_WM_2'] = pd.to_numeric(tap_wm['TAP_WM_2'], errors='coerce')  # median RT
tap_wm['TAP_WM_7'] = pd.to_numeric(tap_wm['TAP_WM_7'], errors='coerce')  # errors
tmt = pd.read_csv(os.path.join(COG, 'TMT', 'TMT.csv'))
tmt.columns = [c.strip() for c in tmt.columns]
tmt['TMT_1'] = pd.to_numeric(tmt['TMT_1'], errors='coerce')
tmt['TMT_5'] = pd.to_numeric(tmt['TMT_5'], errors='coerce')
lps = pd.read_csv(os.path.join(COG, 'LPS', 'LPS.csv'))
lps.columns = [c.strip() for c in lps.columns]
lps['LPS_1'] = pd.to_numeric(lps['LPS_1'], errors='coerce')

# Meta for sex
meta = pd.read_csv(META_PATH)
meta.columns = [c.strip() for c in meta.columns]
sex_col = [c for c in meta.columns if 'Gender' in c][0]
meta['sex'] = meta[sex_col].map({1:'F', 2:'M', '1':'F', '2':'M'})

# Merge all
df = rate[['subject_id','duration_sec','n_events','q4_n',
            'total_rate_per_min','q4_rate_per_min']].copy()
for src, cols in [(nyc, ['ID','nyc_content','nyc_selfref']),
                   (neo, ['ID','NEOFFI_Neuroticism','NEOFFI_Extraversion',
                           'NEOFFI_OpennessForExperiences',
                           'NEOFFI_Agreeableness','NEOFFI_Conscientiousness']),
                   (tap, ['ID','TAP_A_5','TAP_A_15']),
                   (tap_wm, ['ID','TAP_WM_2','TAP_WM_7']),
                   (tmt, ['ID','TMT_1','TMT_5']),
                   (lps, ['ID','LPS_1'])]:
    cols_exist = [c for c in cols if c in src.columns]
    df = df.merge(src[cols_exist], left_on='subject_id', right_on='ID',
                   how='left').drop(columns=['ID'], errors='ignore')
df = df.merge(meta[['ID','sex']], left_on='subject_id', right_on='ID',
               how='left').drop(columns=['ID'], errors='ignore')
df = df.dropna(subset=['sex'])
print(f"Merged: {len(df)}  F={(df['sex']=='F').sum()}  M={(df['sex']=='M').sum()}")

targets = ['nyc_content','nyc_selfref',
           'NEOFFI_Neuroticism','NEOFFI_Extraversion',
           'NEOFFI_OpennessForExperiences',
           'NEOFFI_Agreeableness','NEOFFI_Conscientiousness',
           'TAP_A_5','TAP_A_15','TAP_WM_2','TAP_WM_7',
           'TMT_1','TMT_5','LPS_1']

print(f"\n=== Q4 rate × cognitive, stratified by sex ===")
print(f"{'target':<32}{'all ρ':>10}{'F ρ':>10}{'M ρ':>10}{'F p':>8}{'M p':>8}")
print("-" * 80)
rows = []
for t in targets:
    if t not in df.columns: continue
    for label, sub in [('all', df), ('F', df[df['sex']=='F']),
                        ('M', df[df['sex']=='M'])]:
        valid = sub[['q4_rate_per_min', t]].dropna()
        if len(valid) < 10: continue
        r,p = spearmanr(valid['q4_rate_per_min'], valid[t])
        rows.append({'target':t, 'group':label, 'n':len(valid),
                     'rho':r, 'p':p})
    # summary line
    d_all = next((x for x in rows if x['target']==t and x['group']=='all'), None)
    d_f = next((x for x in rows if x['target']==t and x['group']=='F'), None)
    d_m = next((x for x in rows if x['target']==t and x['group']=='M'), None)
    if d_all and d_f and d_m:
        print(f"{t:<32}{d_all['rho']:>+10.3f}{d_f['rho']:>+10.3f}"
              f"{d_m['rho']:>+10.3f}{d_f['p']:>8.2g}{d_m['p']:>8.2g}")
pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'b56_sex_cognitive.csv'),
                           index=False)

# =========================================================================
# B57 — Between-cohort f0 comparison
# =========================================================================
print("\n" + "=" * 70)
print("B57 — Between-cohort f0 comparison")
print("=" * 70)

cohort_events = []
# LEMON
for f in glob.glob(os.path.join(EVENTS_ROOT, 'lemon', 'sub-*_sie_events.csv')):
    d = pd.read_csv(f)
    if 'sr1' in d.columns:
        d = d.dropna(subset=['sr1'])
        d = d[(d['sr1']>=7)&(d['sr1']<=8.3)]
        if len(d):
            sid = os.path.basename(f).replace('_sie_events.csv','')
            cohort_events.append(pd.DataFrame(
                {'cohort':'LEMON','subject_id':sid,'sr1':d['sr1'].values}))
# HBN releases
for rel in ['R1','R2','R3','R4','R6']:
    d_dir = os.path.join(EVENTS_ROOT, f'hbn_{rel}')
    if not os.path.isdir(d_dir): continue
    for f in glob.glob(os.path.join(d_dir, 'sub-*_sie_events.csv')):
        d = pd.read_csv(f)
        if 'sr1' in d.columns:
            d = d.dropna(subset=['sr1'])
            d = d[(d['sr1']>=7)&(d['sr1']<=8.3)]
            if len(d):
                sid = os.path.basename(f).replace('_sie_events.csv','')
                cohort_events.append(pd.DataFrame(
                    {'cohort': f'HBN_{rel}', 'subject_id':sid,
                     'sr1':d['sr1'].values}))
# TDBRAIN
for f in glob.glob(os.path.join(EVENTS_ROOT, 'tdbrain', 'sub-*_sie_events.csv')):
    d = pd.read_csv(f)
    if 'sr1' in d.columns:
        d = d.dropna(subset=['sr1'])
        d = d[(d['sr1']>=7)&(d['sr1']<=8.3)]
        if len(d):
            sid = os.path.basename(f).replace('_sie_events.csv','')
            cohort_events.append(pd.DataFrame(
                {'cohort':'TDBRAIN','subject_id':sid,'sr1':d['sr1'].values}))
all_ev_cohort = pd.concat(cohort_events, ignore_index=True)
per_sub_cohort = all_ev_cohort.groupby(['cohort','subject_id']).agg(
    f_median=('sr1','median'), n=('sr1','size')).reset_index()

print(f"\n{'cohort':<12}{'subjects':>10}{'events':>10}{'f_median':>12}"
      f"{'IQR':>18}")
print("-" * 62)
ordered = ['LEMON', 'HBN_R1', 'HBN_R2', 'HBN_R3', 'HBN_R4', 'HBN_R6',
           'TDBRAIN']
for coh in ordered:
    sub = per_sub_cohort[per_sub_cohort['cohort']==coh]
    ev = all_ev_cohort[all_ev_cohort['cohort']==coh]
    if len(sub) == 0: continue
    f_med = sub['f_median'].median()
    q25, q75 = sub['f_median'].quantile([.25,.75])
    print(f"{coh:<12}{len(sub):>10}{len(ev):>10}{f_med:>12.3f}"
          f"   [{q25:.3f}, {q75:.3f}]")
# Kruskal-Wallis across cohorts
groups = [per_sub_cohort[per_sub_cohort['cohort']==c]['f_median'].values
          for c in ordered
          if (per_sub_cohort['cohort']==c).sum() >= 10]
kw_p = np.nan
if len(groups) >= 3:
    stat, kw_p = kruskal(*groups)
    print(f"\nKruskal-Wallis per-subject f_median across cohorts: "
          f"H={stat:.2f}  p={kw_p:.2g}")
per_sub_cohort.to_csv(os.path.join(OUT_DIR, 'b57_per_subject_f0_by_cohort.csv'),
                       index=False)

# =========================================================================
# B58 — Global template_rho canonicality ranking
# =========================================================================
print("\n" + "=" * 70)
print("B58 — Global template_rho canonicality ranking across cohorts")
print("=" * 70)

quality_files = {
    'LEMON': 'per_event_quality.csv',
    'HBN_R1': 'per_event_quality_hbn_R1.csv',
    'HBN_R2': 'per_event_quality_hbn_R2.csv',
    'HBN_R3': 'per_event_quality_hbn_R3.csv',
    'HBN_R4': 'per_event_quality_hbn.csv',  # B47 named it hbn (R4 only)
    'HBN_R6': 'per_event_quality_hbn_R6.csv',
    'TDBRAIN': 'per_event_quality_tdbrain.csv',
}
all_quality = []
for coh, fn in quality_files.items():
    p = os.path.join(QUALITY_DIR, fn)
    if not os.path.isfile(p): continue
    q = pd.read_csv(p).dropna(subset=['template_rho'])
    q = q[['subject_id', 't0_net', 'template_rho']].copy()
    q['cohort'] = coh
    all_quality.append(q)
gq = pd.concat(all_quality, ignore_index=True)
print(f"\nTotal events pooled: {len(gq)}  subjects: "
      f"{gq['subject_id'].nunique()}")
print(f"Cohort counts:")
for coh, sub in gq.groupby('cohort'):
    print(f"  {coh:<10}  events={len(sub):>5d}  "
          f"subjects={sub['subject_id'].nunique():>4d}  "
          f"template_rho median={sub['template_rho'].median():+.3f}")

# Global ranking
gq['global_rank'] = gq['template_rho'].rank(ascending=False, pct=True)
gq['global_q4'] = (gq['global_rank'] <= 0.25)
gq['global_top10'] = (gq['global_rank'] <= 0.10)

print(f"\n=== Composition of top 25% globally-canonical events ===")
top25 = gq[gq['global_q4']]
print(f"n = {len(top25)}")
comp = top25.groupby('cohort').size() / len(top25) * 100
total_comp = gq.groupby('cohort').size() / len(gq) * 100
for coh in comp.index:
    print(f"  {coh:<10}  top25% share: {comp[coh]:>5.1f}%  "
          f"(overall share: {total_comp[coh]:>5.1f}%)  "
          f"over-representation: "
          f"{comp[coh]/total_comp[coh]:.2f}x")

print(f"\n=== Composition of top 10% globally-canonical events ===")
top10 = gq[gq['global_top10']]
comp10 = top10.groupby('cohort').size() / len(top10) * 100
for coh in comp10.index:
    print(f"  {coh:<10}  top10% share: {comp10[coh]:>5.1f}%  "
          f"(overall share: {total_comp[coh]:>5.1f}%)  "
          f"over-representation: "
          f"{comp10[coh]/total_comp[coh]:.2f}x")

gq.to_csv(os.path.join(OUT_DIR, 'b58_global_canonicality.csv'), index=False)

# =========================================================================
# FIGURE
# =========================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# A — B56 sex-stratified forest-like plot (Spearman rho per cognitive test)
ax = axes[0, 0]
b56 = pd.DataFrame(rows)
key_tests = ['nyc_content','nyc_selfref','NEOFFI_OpennessForExperiences',
              'NEOFFI_Neuroticism','NEOFFI_Extraversion',
              'TAP_A_5','TMT_5','LPS_1']
y_pos = np.arange(len(key_tests))
for lab, color, shift in [('F', '#d73027', -0.15),
                            ('M', '#4575b4', +0.15)]:
    xs = []; ys = []
    for i, t in enumerate(key_tests):
        d = b56[(b56['target']==t) & (b56['group']==lab)]
        if len(d) == 0: continue
        xs.append(d.iloc[0]['rho']); ys.append(i + shift)
    ax.scatter(xs, ys, s=50, color=color, alpha=0.7, edgecolor='k', lw=0.3,
                label=lab)
ax.axvline(0, color='k', lw=0.8)
ax.set_yticks(y_pos); ax.set_yticklabels(key_tests, fontsize=8)
ax.set_xlabel('Spearman ρ (Q4 rate × cognitive)')
ax.set_title('A — B56 LEMON sex × cognitive (Spearman ρ)',
              loc='left', fontweight='bold', fontsize=11)
ax.legend(); ax.grid(axis='x', alpha=0.3)

# B — B57 per-cohort f_median distribution
ax = axes[0, 1]
cohorts_plot = [c for c in ordered
                 if (per_sub_cohort['cohort']==c).sum() >= 10]
data = [per_sub_cohort[per_sub_cohort['cohort']==c]['f_median'].values
        for c in cohorts_plot]
labels_plot = [f'{c}\nn={len(d)}' for c,d in zip(cohorts_plot, data)]
bp = ax.boxplot(data, tick_labels=labels_plot, patch_artist=True,
                 showfliers=False)
palette = ['#8c1a1a','#fdae61','#d73027','#1a9641','#91bfdb','#4575b4',
           '#2b5fb8']
for patch, c in zip(bp['boxes'], palette[:len(bp['boxes'])]):
    patch.set_facecolor(c); patch.set_alpha(0.6)
ax.axhline(7.83, color='red', ls=':', label='Schumann SR1 7.83')
ax.set_ylabel('per-subject median f0 (Hz)')
ax.set_title(f'B — B57 between-cohort f0 · KW p={kw_p:.2g}',
              loc='left', fontweight='bold', fontsize=11)
ax.tick_params(axis='x', labelsize=8); ax.grid(axis='y', alpha=0.3)
ax.legend()

# C — B58 global canonicality composition
ax = axes[1, 0]
cohorts_g = sorted(gq['cohort'].unique())
all_share = np.array([total_comp.get(c, 0) for c in cohorts_g])
top25_share = np.array([comp.get(c, 0) for c in cohorts_g])
top10_share = np.array([comp10.get(c, 0) for c in cohorts_g])
x_pos = np.arange(len(cohorts_g))
w = 0.25
ax.bar(x_pos - w, all_share, w, label='all events', color='gray', alpha=0.7)
ax.bar(x_pos, top25_share, w, label='top 25%', color='#fdae61', alpha=0.8)
ax.bar(x_pos + w, top10_share, w, label='top 10%', color='#d73027', alpha=0.8)
ax.set_xticks(x_pos); ax.set_xticklabels(cohorts_g, rotation=35, ha='right',
                                          fontsize=8)
ax.set_ylabel('% of events in set')
ax.set_title('C — B58 cohort composition: all / top-25% / top-10% by '
              'template_ρ',
              loc='left', fontweight='bold', fontsize=11)
ax.legend(); ax.grid(axis='y', alpha=0.3)

# D — B58 template_rho distributions per cohort
ax = axes[1, 1]
for coh in cohorts_g:
    sub = gq[gq['cohort']==coh]['template_rho']
    ax.hist(sub, bins=40, alpha=0.5, label=f'{coh} n={len(sub)}',
             density=True)
ax.set_xlabel('template_ρ')
ax.set_ylabel('density')
ax.set_title('D — template_ρ distribution per cohort',
              loc='left', fontweight='bold', fontsize=11)
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

fig.suptitle('B56/B57/B58 — LEMON sex×cog + between-cohort f0 + '
              'global canonicality',
              fontsize=12, y=1.01)
fig.tight_layout()
out_png = os.path.join(OUT_DIR, 'b56_58_bundle.png')
plt.savefig(out_png, dpi=160, bbox_inches='tight')
plt.savefig(out_png.replace('.png', '.pdf'), bbox_inches='tight')
plt.close()
print(f"\nSaved: {out_png}")
