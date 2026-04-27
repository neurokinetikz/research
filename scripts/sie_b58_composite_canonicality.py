#!/usr/bin/env python3
"""B58-equivalent shared-template canonicality ranking across composite-
extracted cohorts.

Loads trajectory .npz files from multiple composite cohorts, builds a
balanced shared template, re-scores each event against the shared
template, and reports top-quartile composition per cohort.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'quality')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs',
                            'schumann', 'images', 'coupling')
os.makedirs(RESULTS_DIR, exist_ok=True)

COHORTS = [
    'lemon_composite', 'lemon_EO_composite',
    'dortmund_composite', 'dortmund_EC_post_composite',
    'dortmund_EO_pre_composite', 'dortmund_EO_post_composite',
]

all_trajs = []
meta_rows = []
tgrid = None
for coh in COHORTS:
    p = os.path.join(OUT_DIR, f'trajectories_{coh}.npz')
    if not os.path.isfile(p):
        print(f"[skip] {coh}: no traj file")
        continue
    d = np.load(p, allow_pickle=True)
    trajs = d['trajs'].astype(np.float64)
    tgrid = d['tgrid'].astype(np.float64)
    sub_ids = d['subject_id']
    t0s = d['t0_net']
    n = trajs.shape[0]
    print(f"[load] {coh}: {n} events")
    all_trajs.append(trajs)
    for i in range(n):
        meta_rows.append({'cohort': coh, 'subject_id': sub_ids[i],
                           't0_net': t0s[i], 'traj_idx': len(meta_rows)})
full_trajs = np.vstack(all_trajs)
meta = pd.DataFrame(meta_rows)
print(f"\nTotal pooled events: {len(meta)}")

# Balanced pool (≤300 per cohort)
rng = np.random.default_rng(42)
sampled_idx = []
for coh, grp in meta.groupby('cohort'):
    idx = grp['traj_idx'].values
    if len(idx) <= 300:
        sampled_idx.extend(idx.tolist())
    else:
        sampled_idx.extend(rng.choice(idx, size=300, replace=False).tolist())
sampled_idx = np.array(sampled_idx)
print(f"Balanced pool: {len(sampled_idx)} events")

# Shared template on core window
m_core = (tgrid >= -5) & (tgrid <= +5)
template = np.nanmean(full_trajs[sampled_idx], axis=0)
tmpl_core = template[m_core] - np.nanmean(template[m_core])

# Re-score all events against shared template
rhos = []
for i in range(full_trajs.shape[0]):
    ev = full_trajs[i, m_core]
    if np.any(~np.isfinite(ev)):
        rhos.append(np.nan); continue
    ev_c = ev - np.nanmean(ev)
    denom = np.sqrt(np.nansum(ev_c**2) * np.nansum(tmpl_core**2))
    rhos.append(float(np.nansum(ev_c * tmpl_core) / denom)
                 if denom > 0 else np.nan)
meta['template_rho_shared'] = rhos

# Ranking
meta['global_rank'] = meta['template_rho_shared'].rank(ascending=False,
                                                         pct=True)
meta['top25'] = meta['global_rank'] <= 0.25
meta['top10'] = meta['global_rank'] <= 0.10

total_share = meta.groupby('cohort').size() / len(meta) * 100
top25_share = meta[meta['top25']].groupby('cohort').size() / meta['top25'].sum() * 100
top10_share = meta[meta['top10']].groupby('cohort').size() / meta['top10'].sum() * 100

print(f"\n{'cohort':<42}{'n':>6}{'overall%':>10}{'top25%':>10}"
      f"{'top10%':>10}{'over-rep(10%)':>16}")
print('-'*95)
for coh in COHORTS:
    sub = meta[meta['cohort'] == coh]
    if len(sub) == 0: continue
    ov = total_share.get(coh, 0)
    t25 = top25_share.get(coh, 0)
    t10 = top10_share.get(coh, 0)
    or10 = t10 / ov if ov > 0 else 0
    print(f"{coh:<42}{len(sub):>6}{ov:>9.1f}%{t25:>9.1f}%{t10:>9.1f}%"
          f"{or10:>15.2f}x")

print(f"\n=== Per-cohort shared-template median ===")
for coh in COHORTS:
    sub = meta[meta['cohort']==coh]
    if len(sub) == 0: continue
    print(f"  {coh:<42}  median ρ = {sub['template_rho_shared'].median():+.3f}")

meta.to_csv(os.path.join(RESULTS_DIR,
                          'b58_composite_shared_template.csv'), index=False)
print(f"\nSaved: b58_composite_shared_template.csv")

# Figure
fig, ax = plt.subplots(figsize=(10, 6))
cohort_order = [c for c in COHORTS if c in meta['cohort'].values]
x = np.arange(len(cohort_order))
w = 0.25
all_s = np.array([total_share.get(c, 0) for c in cohort_order])
t25_s = np.array([top25_share.get(c, 0) for c in cohort_order])
t10_s = np.array([top10_share.get(c, 0) for c in cohort_order])
ax.bar(x - w, all_s, w, label='all events', color='gray', alpha=0.7)
ax.bar(x, t25_s, w, label='top 25% (shared)', color='#fdae61', alpha=0.8)
ax.bar(x + w, t10_s, w, label='top 10% (shared)', color='#d73027', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([c.replace('_composite','') for c in cohort_order],
                    rotation=35, ha='right', fontsize=9)
ax.set_ylabel('% of events')
ax.set_title('B58 composite cohorts — shared-template canonicality',
              loc='left', fontweight='bold')
ax.legend(); ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
out_png = os.path.join(RESULTS_DIR, 'b58_composite_canonicality.png')
plt.savefig(out_png, dpi=160, bbox_inches='tight')
plt.close()
print(f"Saved: {out_png}")
