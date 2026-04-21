#!/usr/bin/env python3
"""
B58v2 — Shared-template canonicality ranking across cohorts.

Loads per-cohort trajectories_{cohort}.npz (saved by
sie_template_rho_crosscohort.py --save-trajectories), builds a SHARED
grand-average envelope template from a balanced pool (equal-per-cohort
sampling), re-scores each event's template_ρ against the shared template,
and re-examines cross-cohort canonicality composition.

This is the fair cross-cohort canonicality ranking promised in B58's
caveat.
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
COHORTS = ['lemon', 'hbn_R1', 'hbn_R2', 'hbn_R3', 'hbn_R4', 'hbn_R6',
           'tdbrain']

# ===== load trajectories =====
all_trajs = []
all_meta = []
tgrid = None
for coh in COHORTS:
    p = os.path.join(OUT_DIR, f'trajectories_{coh}.npz')
    if not os.path.isfile(p):
        print(f"[skip] {coh}: no trajectory file")
        continue
    d = np.load(p, allow_pickle=True)
    trajs = d['trajs'].astype(np.float64)
    tgrid = d['tgrid'].astype(np.float64)
    sub_ids = d['subject_id']
    t0s = d['t0_net']
    n = trajs.shape[0]
    print(f"[load] {coh}: {n} events, trajs shape {trajs.shape}")
    all_trajs.append(trajs)
    all_meta.append(pd.DataFrame({'cohort': coh,
                                   'subject_id': sub_ids,
                                   't0_net': t0s,
                                   'traj_idx': np.arange(n)}))
full_trajs = np.vstack(all_trajs)
meta = pd.concat(all_meta, ignore_index=True)
meta['traj_idx'] = np.arange(len(meta))
print(f"\nTotal pooled events: {len(meta)}")

# ===== build shared template (balanced sampling to avoid
# LEMON/HBN dominance) =====
# Target: ≤300 events per cohort (smallest cohort gets all its events)
rng = np.random.default_rng(42)
sampled_idx = []
for coh, grp in meta.groupby('cohort'):
    idx = grp['traj_idx'].values
    if len(idx) <= 300:
        sampled_idx.extend(idx.tolist())
    else:
        sampled_idx.extend(rng.choice(idx, size=300, replace=False).tolist())
sampled_idx = np.array(sampled_idx)
print(f"Balanced pool for template: {len(sampled_idx)} events "
      f"({len(set(meta.loc[sampled_idx, 'cohort']))} cohorts)")

# Compute shared template on core window [-5, +5]
m_core = (tgrid >= -5) & (tgrid <= +5)
template_full = np.nanmean(full_trajs[sampled_idx], axis=0)
tmpl_core = template_full[m_core]
tmpl_core = tmpl_core - np.nanmean(tmpl_core)

# ===== re-score all events against shared template =====
rhos_shared = []
for i in range(full_trajs.shape[0]):
    ev = full_trajs[i, m_core]
    if np.any(~np.isfinite(ev)):
        rhos_shared.append(np.nan); continue
    ev_c = ev - np.nanmean(ev)
    denom = np.sqrt(np.nansum(ev_c**2) * np.nansum(tmpl_core**2))
    rhos_shared.append(float(np.nansum(ev_c * tmpl_core) / denom)
                        if denom > 0 else np.nan)
meta['template_rho_shared'] = rhos_shared

# Load existing within-cohort template_ρ for comparison
within_rhos = []
for coh in COHORTS:
    if coh == 'lemon':
        p = os.path.join(OUT_DIR, 'per_event_quality.csv')
    else:
        p = os.path.join(OUT_DIR, f'per_event_quality_{coh}.csv')
    if not os.path.isfile(p):
        continue
    q = pd.read_csv(p).dropna(subset=['template_rho'])
    q['cohort'] = coh
    within_rhos.append(q[['cohort','subject_id','t0_net','template_rho']])
within_df = pd.concat(within_rhos, ignore_index=True)
within_df['t0_round'] = within_df['t0_net'].round(3)
meta['t0_round'] = meta['t0_net'].round(3)
meta = meta.merge(within_df[['cohort','subject_id','t0_round','template_rho']],
                   on=['cohort','subject_id','t0_round'],
                   how='left')
meta.rename(columns={'template_rho':'template_rho_within'}, inplace=True)

# ===== comparison & ranking =====
print(f"\n=== Within-cohort template_ρ vs shared template_ρ ===")
print(f"{'cohort':<10}{'n':>6}{'within med':>14}{'shared med':>14}"
      f"{'delta':>10}")
print('-'*60)
for coh in COHORTS:
    sub = meta[meta['cohort']==coh]
    if len(sub)==0: continue
    w = sub['template_rho_within'].median()
    s = sub['template_rho_shared'].median()
    print(f"{coh:<10}{len(sub):>6}{w:>+14.3f}{s:>+14.3f}{s-w:>+10.3f}")

# Global ranking by shared template_ρ
meta['global_rank_shared'] = meta['template_rho_shared'].rank(
    ascending=False, pct=True)
meta['top25_shared'] = (meta['global_rank_shared'] <= 0.25)
meta['top10_shared'] = (meta['global_rank_shared'] <= 0.10)

total_share = meta.groupby('cohort').size() / len(meta) * 100
print(f"\n=== Shared-template top 25% composition ===")
top25_share = (meta[meta['top25_shared']].groupby('cohort').size() /
                meta['top25_shared'].sum() * 100)
for coh in meta['cohort'].unique():
    ov = total_share.get(coh, 0)
    t25 = top25_share.get(coh, 0)
    print(f"  {coh:<10}  top25% share: {t25:>5.1f}%  "
          f"(overall: {ov:>5.1f}%)  over-rep: {t25/ov:.2f}x")

print(f"\n=== Shared-template top 10% composition ===")
top10_share = (meta[meta['top10_shared']].groupby('cohort').size() /
                meta['top10_shared'].sum() * 100)
for coh in meta['cohort'].unique():
    ov = total_share.get(coh, 0)
    t10 = top10_share.get(coh, 0)
    print(f"  {coh:<10}  top10% share: {t10:>5.1f}%  "
          f"(overall: {ov:>5.1f}%)  over-rep: {t10/ov:.2f}x")

# Save
out_csv = os.path.join(RESULTS_DIR, 'b58v2_shared_template_canonicality.csv')
meta.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}")

# ===== FIGURE =====
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# A — within vs shared template_ρ scatter per cohort
ax = axes[0]
colors = ['#8c1a1a','#fdae61','#d73027','#f1a340','#1a9641','#91bfdb','#2b5fb8']
for i, coh in enumerate(COHORTS):
    sub = meta[meta['cohort']==coh].dropna(subset=['template_rho_within',
                                                     'template_rho_shared'])
    if len(sub) == 0: continue
    ax.scatter(sub['template_rho_within'], sub['template_rho_shared'],
                s=8, alpha=0.4, color=colors[i], label=coh)
ax.plot([-1, 1], [-1, 1], 'k--', lw=0.8, alpha=0.5)
ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
ax.set_xlabel('within-cohort template_ρ'); ax.set_ylabel('shared-template ρ')
ax.set_title('A — within vs shared template_ρ',
              loc='left', fontweight='bold', fontsize=11)
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)

# B — cohort composition bars
ax = axes[1]
cohorts_g = [c for c in COHORTS if c in meta['cohort'].values]
x_pos = np.arange(len(cohorts_g))
w = 0.25
all_s = np.array([total_share.get(c, 0) for c in cohorts_g])
t25_s = np.array([top25_share.get(c, 0) for c in cohorts_g])
t10_s = np.array([top10_share.get(c, 0) for c in cohorts_g])
ax.bar(x_pos - w, all_s, w, label='all events', color='gray', alpha=0.7)
ax.bar(x_pos, t25_s, w, label='top 25%', color='#fdae61', alpha=0.8)
ax.bar(x_pos + w, t10_s, w, label='top 10%', color='#d73027', alpha=0.8)
ax.set_xticks(x_pos); ax.set_xticklabels(cohorts_g, rotation=35, ha='right',
                                          fontsize=9)
ax.set_ylabel('% of set')
ax.set_title('B — Shared-template cohort composition',
              loc='left', fontweight='bold', fontsize=11)
ax.legend(); ax.grid(axis='y', alpha=0.3)

# C — distribution shapes
ax = axes[2]
for i, coh in enumerate(cohorts_g):
    sub = meta[meta['cohort']==coh]['template_rho_shared'].dropna()
    ax.hist(sub, bins=40, alpha=0.4, density=True, color=colors[i],
             label=f'{coh} n={len(sub)}')
ax.set_xlabel('shared-template ρ'); ax.set_ylabel('density')
ax.set_title('C — shared template_ρ distribution per cohort',
              loc='left', fontweight='bold', fontsize=11)
ax.legend(fontsize=7); ax.grid(alpha=0.3)

fig.suptitle('B58v2 — Shared-template canonicality ranking '
              f'(balanced pool n={len(sampled_idx)}, '
              f'total {len(meta)} events)',
              fontsize=12, y=1.02)
fig.tight_layout()
out_png = os.path.join(RESULTS_DIR, 'b58v2_shared_template.png')
plt.savefig(out_png, dpi=160, bbox_inches='tight')
plt.savefig(out_png.replace('.png', '.pdf'), bbox_inches='tight')
plt.close()
print(f"Saved: {out_png}")
