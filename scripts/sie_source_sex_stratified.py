#!/usr/bin/env python3
"""
B60 — Source-space sex-stratified comparison.

Produce two group-average maps:
  (i)  LEMON adult males — Q4 SIE posterior-α generator in adult males
  (ii) HBN 5-9 yr females — Q4 SIE posterior-α generator in young girls

Compare: Spearman ρ between the two label-rank orderings; visual
comparison of top-region lists. If maps are similar → the posterior-
temporoparietal network is conserved across development and sex. If
maps differ → same scalp contrast reflects different anatomy in
different populations.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from scipy.stats import spearmanr

mne.set_log_level('ERROR')

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'source')
STC_DIR_LEMON = os.path.join(OUT_DIR, 'stcs')
STC_DIR_HBN = os.path.join(OUT_DIR, 'stcs_hbn_girls_5_9')

_fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
SUBJECTS_DIR = os.path.dirname(_fs_dir)

LEMON_META = ('/Volumes/T9/lemon_data/behavioral/'
               'Behavioural_Data_MPILMBB_LEMON/'
               'META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv')


def load_group_stcs(stc_dir):
    files = sorted(f for f in os.listdir(stc_dir) if f.endswith('-lh.stc'))
    data = []
    ref_stc = None
    sub_ids = []
    for f in files:
        base = f[:-len('-lh.stc')]
        sid = base.split('_Q4_')[0]
        path = os.path.join(stc_dir, base)
        try:
            stc = mne.read_source_estimate(path, 'fsaverage')
        except Exception:
            continue
        data.append(stc.data.squeeze())
        if ref_stc is None: ref_stc = stc
        sub_ids.append(sid)
    if not data:
        return None, None, None
    return np.array(data), ref_stc, sub_ids


def group_map(stcs_data, ref_stc):
    grand = np.median(stcs_data, axis=0)
    group = ref_stc.copy()
    group.data = grand[:, np.newaxis]
    return group


def label_rank(stc, labels):
    rows = []
    for lbl in labels:
        if lbl.name.startswith('unknown-'): continue
        idx_vertices = lbl.get_vertices_used()
        hemi_data = stc.lh_data if lbl.hemi == 'lh' else stc.rh_data
        vert_vno = stc.lh_vertno if lbl.hemi == 'lh' else stc.rh_vertno
        match = np.isin(vert_vno, idx_vertices)
        if match.sum() == 0: continue
        rows.append({'label': lbl.name,
                     'simple': lbl.name.replace('-lh','').replace('-rh',''),
                     'hemi': lbl.hemi,
                     'ratio_median': float(np.nanmedian(hemi_data[match, 0])),
                     'ratio_mean': float(np.nanmean(hemi_data[match, 0])),
                     'n_vert': int(match.sum())})
    return pd.DataFrame(rows).sort_values('ratio_median', ascending=False)


# ===== LEMON sex-stratified =====
print("=" * 70)
print("Loading LEMON STCs...")
lemon_data, lemon_ref, lemon_ids = load_group_stcs(STC_DIR_LEMON)
print(f"LEMON STCs loaded: {len(lemon_ids)}")

# Read LEMON META for sex
meta = pd.read_csv(LEMON_META)
meta.columns = [c.strip() for c in meta.columns]
sex_col = [c for c in meta.columns if 'Gender' in c][0]
meta['sex'] = meta[sex_col].map({1:'F', 2:'M', '1':'F', '2':'M'})
sex_map = dict(zip(meta['ID'], meta['sex']))
lemon_sex = np.array([sex_map.get(s, None) for s in lemon_ids])
male_mask = (lemon_sex == 'M')
female_mask = (lemon_sex == 'F')
print(f"  Male LEMON: n={male_mask.sum()}")
print(f"  Female LEMON: n={female_mask.sum()}")

lemon_male_map = group_map(lemon_data[male_mask], lemon_ref) if male_mask.sum() >= 5 else None
lemon_female_map = group_map(lemon_data[female_mask], lemon_ref) if female_mask.sum() >= 5 else None

# ===== HBN 5-9 yr girls =====
print(f"\nLoading HBN 5-9 yr female STCs...")
if os.path.isdir(STC_DIR_HBN):
    hbn_data, hbn_ref, hbn_ids = load_group_stcs(STC_DIR_HBN)
    if hbn_data is not None:
        print(f"  HBN girls 5-9: n={len(hbn_ids)}")
        hbn_girls_map = group_map(hbn_data, hbn_ref)
    else:
        hbn_girls_map = None
        print("  HBN girls 5-9: no STCs")
else:
    hbn_girls_map = None
    print(f"  HBN girls dir not found: {STC_DIR_HBN}")

# ===== label rankings =====
labels = mne.read_labels_from_annot('fsaverage', 'aparc',
                                     subjects_dir=SUBJECTS_DIR,
                                     verbose=False)

groups = {}
if lemon_male_map is not None:
    groups['LEMON_Male'] = label_rank(lemon_male_map, labels)
if lemon_female_map is not None:
    groups['LEMON_Female'] = label_rank(lemon_female_map, labels)
if hbn_girls_map is not None:
    groups['HBN_Girls_5_9'] = label_rank(hbn_girls_map, labels)

# Save rankings
all_out = []
for group_name, df in groups.items():
    df_out = df.copy()
    df_out['group'] = group_name
    all_out.append(df_out)
if all_out:
    pd.concat(all_out).to_csv(
        os.path.join(OUT_DIR, 'B60_sex_stratified_label_rankings.csv'),
        index=False)

# ===== pairwise Spearman across label rankings =====
print(f"\n=== Pairwise label-rank Spearman ρ ===")
for g1, d1 in groups.items():
    for g2, d2 in groups.items():
        if g1 >= g2: continue
        # Align by label name
        merged = d1.set_index('label').join(
            d2.set_index('label'), how='inner', lsuffix='_1', rsuffix='_2')
        r, p = spearmanr(merged['ratio_median_1'], merged['ratio_median_2'])
        print(f"  {g1} vs {g2}: ρ={r:+.3f}  p={p:.2g}  (n_labels={len(merged)})")

# ===== top 15 per group =====
print(f"\n=== Top 15 labels per group ===")
for g, df in groups.items():
    print(f"\n--- {g} ---")
    print(df.head(15).to_string(index=False))

# ===== FIGURE: horizontal bar chart side-by-side =====
def lobe_color(label_simple):
    LOBE = {
        'parahippocampal':'#8c1a1a','temporalpole':'#8c1a1a',
        'entorhinal':'#8c1a1a',
        'fusiform':'#d73027','inferiortemporal':'#d73027',
        'middletemporal':'#d73027','superiortemporal':'#d73027',
        'transversetemporal':'#d73027','bankssts':'#d73027',
        'precuneus':'#1a9641','superiorparietal':'#1a9641',
        'inferiorparietal':'#1a9641','supramarginal':'#1a9641',
        'postcentral':'#1a9641',
        'lingual':'#2b5fb8','cuneus':'#2b5fb8','lateraloccipital':'#2b5fb8',
        'pericalcarine':'#2b5fb8',
        'insula':'#d7b5d8','isthmuscingulate':'#fdae61',
        'posteriorcingulate':'#fdae61','caudalanteriorcingulate':'#fdae61',
        'rostralanteriorcingulate':'#fdae61',
        'precentral':'#756bb1','paracentral':'#756bb1',
        'superiorfrontal':'#756bb1','rostralmiddlefrontal':'#756bb1',
        'caudalmiddlefrontal':'#756bb1',
        'parsopercularis':'#756bb1','parstriangularis':'#756bb1',
        'parsorbitalis':'#756bb1',
        'lateralorbitofrontal':'#756bb1','medialorbitofrontal':'#756bb1',
        'frontalpole':'#756bb1',
    }
    return LOBE.get(label_simple, '#aaaaaa')


n_groups = len(groups)
fig, axes = plt.subplots(1, n_groups, figsize=(5*n_groups, 14),
                          sharex=False)
if n_groups == 1:
    axes = [axes]

for ax, (group_name, df) in zip(axes, groups.items()):
    df_s = df.copy().reset_index(drop=True)
    y_pos = range(len(df_s))
    colors = [lobe_color(s) for s in df_s['simple']]
    ax.barh(y_pos, df_s['ratio_median'], color=colors, edgecolor='k',
             lw=0.3)
    ax.axvline(1.0, color='k', lw=1, alpha=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{s}-{h}" for s,h in
                        zip(df_s['simple'], df_s['hemi'])], fontsize=6)
    ax.set_xlabel('SR1 ratio (event/baseline)')
    ax.set_title(group_name)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

fig.suptitle('B60 — Source-space SR1 ratio rankings by sex × age group',
              fontsize=12, y=1.01)
fig.tight_layout()
out_png = os.path.join(OUT_DIR, 'B60_sex_stratified_rankings.png')
plt.savefig(out_png, dpi=150, bbox_inches='tight')
plt.savefig(out_png.replace('.png','.pdf'), bbox_inches='tight')
plt.close()
print(f"\nSaved: {out_png}")
