#!/usr/bin/env python3
"""
B52 — Age-stratified HBN R4: does posterior > anterior SR1 contrast
       emerge developmentally?

Background. B47 found a robust within-LEMON posterior-vs-anterior SR1
contrast at template_ρ Q4 events (+1.17, p = 1.7e-5) but a weak/null
HBN R4 result (+0.35, p = 0.78). HBN spans ages 5-21; posterior α
matures with age (peak ~10-12 years according to developmental
literature). The null may reflect developmental heterogeneity.

Test. Merge existing HBN B47 per-subject contrast values with the HBN
R4 participants.tsv age field. Test:
  1. Continuous: Spearman ρ(age, sr1_contrast)
  2. Stratified: split HBN subjects into age bins and compare per-bin
     posterior-vs-anterior Wilcoxon
  3. Age-controlled: partial correlations of contrast with sex and
     pathology factors

Output: per-age-bin summary, scatter, bin-wise box plots.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, wilcoxon, mannwhitneyu

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'coupling')
HBN_BASE = '/Volumes/T9/hbn_data'
B47_CSV = os.path.join(OUT_DIR, 'posterior_sr1_crosscohort.csv')

# Load B47 per-subject values (include all HBN releases, excluding 'hbn_all')
df = pd.read_csv(B47_CSV)
hbn = df[df['cohort'].str.startswith('hbn') & (df['cohort'] != 'hbn_all')].copy()
hbn['release'] = hbn['cohort'].str.replace('hbn_', '')
print(f"HBN rows in B47 (across releases): {len(hbn)}")
print(hbn['release'].value_counts().to_string())

# Load participants metadata from each release
meta_list = []
for rel in ['R1', 'R2', 'R3', 'R4', 'R6']:
    p = os.path.join(HBN_BASE, f'cmi_bids_{rel}', 'participants.tsv')
    if os.path.isfile(p):
        m = pd.read_csv(p, sep='\t')
        m['release'] = rel
        meta_list.append(m)
meta = pd.concat(meta_list, ignore_index=True)
meta = meta.rename(columns={'participant_id': 'subject_id'})
print(f"HBN participants (all releases): {len(meta)}")

# Merge
m = hbn.merge(meta[['subject_id', 'age', 'sex', 'p_factor', 'attention',
                     'internalizing', 'externalizing', 'ehq_total']],
              on='subject_id', how='left')
m['age'] = pd.to_numeric(m['age'], errors='coerce')
m = m.dropna(subset=['age'])
print(f"Merged with age: {len(m)}")
print(f"Age: min {m['age'].min():.1f}, max {m['age'].max():.1f}, "
      f"mean {m['age'].mean():.1f}, median {m['age'].median():.1f}")

# ===== Continuous age tests =====
print(f"\n=== Continuous age correlations (HBN R4) ===")
for y in ['sr1_ratio_posterior', 'sr1_ratio_anterior', 'sr1_contrast']:
    valid = m[['age', y]].dropna()
    rho, p = spearmanr(valid['age'], valid[y])
    print(f"  age × {y:22s}  n={len(valid)}  ρ={rho:+.3f}  p={p:.3g}")

# ===== Age-stratified analysis =====
# Three bins: 5-9 (early childhood α still developing), 10-13
# (alpha peak-freq stabilizing), 14-21 (adult-like alpha)
bins = [(5, 9.99), (10, 13.99), (14, 21.99)]
bin_labels = ['5-9 yrs', '10-13 yrs', '14-21 yrs']
m['age_bin'] = pd.cut(m['age'],
                       bins=[0, 10, 14, 30],
                       labels=['5-9', '10-13', '14-21'],
                       right=False)
print(f"\n=== Age-stratified posterior-vs-anterior SR1 ===")
print(f"{'bin':<10}{'n':>5}{'post med':>12}{'ant med':>10}"
      f"{'contrast':>11}{'% p>a':>8}{'Wilcoxon p':>13}")
print('-' * 72)
strat_rows = []
for bin_name in ['5-9', '10-13', '14-21']:
    sub = m[m['age_bin'] == bin_name]
    if len(sub) < 5:
        continue
    post = sub['sr1_ratio_posterior'].values
    ant = sub['sr1_ratio_anterior'].values
    contrast = sub['sr1_contrast'].values
    pm = np.median(post); am = np.median(ant); cm = np.median(contrast)
    pct = (contrast > 0).mean() * 100
    try:
        p = wilcoxon(contrast).pvalue
    except ValueError:
        p = np.nan
    strat_rows.append({'age_bin': bin_name, 'n': len(sub),
                       'post_median': pm, 'ant_median': am,
                       'contrast_median': cm,
                       'pct_post_gt_ant': pct, 'Wilcoxon_p': p,
                       'mean_age': sub['age'].mean()})
    print(f"{bin_name:<10}{len(sub):>5}{pm:>12.3f}{am:>10.3f}"
          f"{cm:>+11.3f}{pct:>7.0f}%{p:>13.2g}")

pd.DataFrame(strat_rows).to_csv(
    os.path.join(OUT_DIR, 'hbn_age_stratified_sr1.csv'), index=False)

# Clip outliers for visualization (some HBN subjects have baseline-near-zero
# blow-up); still include them in stats
clip_hi = 30.0
m['post_clip'] = np.clip(m['sr1_ratio_posterior'], 0, clip_hi)
m['ant_clip'] = np.clip(m['sr1_ratio_anterior'], 0, clip_hi)
m['contrast_clip'] = np.clip(m['sr1_contrast'], -clip_hi, clip_hi)

# ===== FIGURE =====
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# A. age vs posterior-anterior contrast
ax = axes[0]
valid = m[['age', 'contrast_clip']].dropna()
rho_c, p_c = spearmanr(m.dropna(subset=['age', 'sr1_contrast'])['age'],
                        m.dropna(subset=['age', 'sr1_contrast'])['sr1_contrast'])
ax.scatter(valid['age'], valid['contrast_clip'], s=28, alpha=0.55,
           color='steelblue', edgecolor='k', lw=0.3)
ax.axhline(0, color='k', lw=0.8)
# OLS trendline
sort = valid.sort_values('age')
slope, intercept = np.polyfit(sort['age'], sort['contrast_clip'], 1)
ax.plot(sort['age'], slope * sort['age'] + intercept, color='red', lw=1.5,
         label=f'OLS slope {slope:+.2f}')
ax.set_xlabel('age (years)')
ax.set_ylabel('posterior − anterior SR1 ratio (clipped ±30)')
ax.set_title(f'A — HBN R4 age × posterior-anterior SR1 contrast\n'
              f'n={len(valid)}  ρ={rho_c:+.2f}  p={p_c:.2g}',
              loc='left', fontweight='bold', fontsize=10)
ax.legend()
ax.grid(alpha=0.3)

# B. Box plot of contrast by age bin × sex
ax = axes[1]
bin_labels_plot = []
bin_data = []
colors = []
pal = {'F': '#d73027', 'M': '#4575b4'}
for bin_name in ['5-9', '10-13', '14-21']:
    for s in ['F', 'M']:
        sub = m[(m['age_bin']==bin_name) & (m['sex']==s)]['sr1_contrast'].dropna().values
        if len(sub) < 5:
            continue
        bin_data.append(sub)
        bin_labels_plot.append(f'{bin_name}\n{s} n={len(sub)}')
        colors.append(pal[s])
bp = ax.boxplot(bin_data, labels=bin_labels_plot,
                patch_artist=True, showfliers=False)
for patch, c in zip(bp['boxes'], colors):
    patch.set_facecolor(c); patch.set_alpha(0.6)
ax.axhline(0, color='k', lw=0.8)
ax.set_xlabel('age bin × sex')
ax.set_ylabel('posterior − anterior SR1 ratio')
ax.set_title('B — HBN age × sex (F red, M blue)', loc='left',
              fontweight='bold', fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(-10, 20)
ax.tick_params(axis='x', labelsize=8)

# C. posterior vs anterior scatter, colored by age
ax = axes[2]
valid = m[['age', 'post_clip', 'ant_clip']].dropna()
sc = ax.scatter(valid['ant_clip'], valid['post_clip'],
                 c=valid['age'], cmap='viridis',
                 s=28, alpha=0.75, edgecolor='k', lw=0.3)
lim = max(valid['ant_clip'].max(), valid['post_clip'].max()) + 1
ax.plot([0, lim], [0, lim], 'k--', lw=1, alpha=0.6)
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel('anterior SR1 ratio (×, clipped)')
ax.set_ylabel('posterior SR1 ratio (×, clipped)')
ax.set_title('C — Post vs ant, colored by age', loc='left',
              fontweight='bold', fontsize=10)
plt.colorbar(sc, ax=ax, label='age (years)')
ax.grid(alpha=0.3)

fig.suptitle('B52 — HBN R4 age-stratified posterior-vs-anterior SR1 contrast',
              fontsize=12, y=1.02)
fig.tight_layout()
out_png = os.path.join(OUT_DIR, 'hbn_age_stratified_sr1.png')
plt.savefig(out_png, dpi=160, bbox_inches='tight')
plt.savefig(out_png.replace('.png', '.pdf'), bbox_inches='tight')
plt.close()
print(f"\nSaved: {out_png}")

# ===== Additional: check if sex effects =====
print(f"\n=== Sex effects (posterior-anterior contrast) ===")
for s in ['F', 'M']:
    sub = m[m['sex'] == s]['sr1_contrast'].dropna()
    if len(sub) < 5: continue
    print(f"  {s}: n={len(sub)}  median={np.median(sub):+.3f}  "
          f"% post>ant={(sub>0).mean()*100:.0f}%")
try:
    f_data = m[m['sex']=='F']['sr1_contrast'].dropna()
    m_data = m[m['sex']=='M']['sr1_contrast'].dropna()
    u, p = mannwhitneyu(f_data, m_data)
    print(f"  Mann-Whitney F vs M: p={p:.2g}")
except Exception:
    pass
