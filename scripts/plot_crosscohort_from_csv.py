#!/usr/bin/env python3
"""Regenerate B47 cross-cohort figure from CSV with clipped axis + correct labels."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'coupling')
df = pd.read_csv(os.path.join(OUT_DIR, 'posterior_sr1_crosscohort.csv'))

cohorts = ['lemon', 'hbn', 'tdbrain']
col = {'lemon': '#8c1a1a', 'hbn': '#1a9641', 'tdbrain': '#2b5fb8'}
titles = {'lemon': 'LEMON · template_ρ Q4',
          'hbn': 'HBN R4 · template_ρ Q4',
          'tdbrain': 'TDBRAIN · template_ρ Q4'}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, c in zip(axes, cohorts):
    sub = df[df['cohort'] == c]
    if len(sub) == 0:
        ax.set_axis_off(); continue
    post = sub['sr1_ratio_posterior'].values
    ant = sub['sr1_ratio_anterior'].values
    # Fixed axis for visual comparability (outliers are clipped, counted)
    lo = 0.0
    hi = 30.0
    contrast = post - ant
    pct = (contrast > 0).mean() * 100
    p = wilcoxon(contrast).pvalue if len(contrast) > 10 else np.nan

    # Clip points for plotting (still count for stats)
    ant_p = np.clip(ant, lo, hi)
    post_p = np.clip(post, lo, hi)
    clipped = ((post > hi) | (ant > hi)).sum()
    ax.scatter(ant_p, post_p, s=24, alpha=0.55, color=col[c],
                edgecolor='k', lw=0.3)
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.6)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel('anterior SR1 ratio (×)')
    ax.set_ylabel('posterior SR1 ratio (×)')
    n_filt = sub['n_events'].sum()
    extra = f' · {clipped} clipped' if clipped else ''
    ax.set_title(f'{titles[c]}\nn={len(sub)} subj · '
                  f'{int(n_filt)} evts · post>ant {pct:.0f}% · '
                  f'p={p:.1g}{extra}',
                  loc='left', fontsize=10, fontweight='bold')
    ax.grid(alpha=0.3)

fig.suptitle('B47 — Cross-cohort posterior-vs-anterior SR1 dominance '
              '(template_ρ Q4 events per subject)',
              fontsize=12, y=1.02)
fig.tight_layout()
out_png = os.path.join(OUT_DIR, 'posterior_sr1_crosscohort.png')
plt.savefig(out_png, dpi=180, bbox_inches='tight')
plt.savefig(out_png.replace('.png', '.pdf'), bbox_inches='tight')
plt.close()
print(f"Saved: {out_png}")
