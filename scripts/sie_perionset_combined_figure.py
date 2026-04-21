#!/usr/bin/env python3
"""
Combined peri-onset summary figure: t0_net-alignment vs computed-onset alignment,
all three streams on shared axes, with annotated peak times.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'perionset')

real_csv = os.path.join(OUT_DIR, 'perionset_triple_average.csv')
null_csv = os.path.join(OUT_DIR, 'perionset_null_random.csv')
comp_csv = os.path.join(OUT_DIR, 'perionset_computed_onset.csv')

real = pd.read_csv(real_csv)
null = pd.read_csv(null_csv)
comp = pd.read_csv(comp_csv)

fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex='col')
specs = [
    ('envelope z (7.83 ± 0.6 Hz)', 'env', 'darkorange'),
    ('Kuramoto R(t) in 7.2–8.4 Hz', 'R', 'seagreen'),
    ('mean PLV to median', 'P', 'purple'),
]

for row, (label, key, color) in enumerate(specs):
    # Left: t0_net alignment (real vs null)
    ax = axes[row, 0]
    ax.fill_between(null['t_rel'], null[f'{key}_ci_lo'], null[f'{key}_ci_hi'],
                    color='gray', alpha=0.30, label='random-onset null')
    ax.plot(null['t_rel'], null[f'{key}_mean'], color='black', lw=1, ls='--')
    ax.fill_between(real['t_rel'], real[f'{key}_ci_lo'], real[f'{key}_ci_hi'],
                    color=color, alpha=0.30, label='real events')
    ax.plot(real['t_rel'], real[f'{key}_mean'], color=color, lw=2)
    peak_t = real['t_rel'].iloc[real[f'{key}_mean'].idxmax()]
    peak_v = real[f'{key}_mean'].max()
    ax.axvline(0, color='k', ls='--', lw=0.5)
    ax.axvline(peak_t, color='red', ls=':', lw=0.8)
    ax.annotate(f'peak {peak_t:+.2f} s', xy=(peak_t, peak_v),
                xytext=(peak_t + 0.5, peak_v), fontsize=9, color='red')
    ax.set_ylabel(label)
    if row == 0:
        ax.set_title('Aligned on t₀_net (Stage 3 refined onset)\n192 subj · 914 events')
        ax.legend(loc='upper right', fontsize=8)

    # Right: computed-onset alignment
    ax = axes[row, 1]
    ax.fill_between(comp['t_rel'], comp[f'{key}_ci_lo'], comp[f'{key}_ci_hi'],
                    color=color, alpha=0.30, label='real events (computed onset)')
    ax.plot(comp['t_rel'], comp[f'{key}_mean'], color=color, lw=2)
    peak_t2 = comp['t_rel'].iloc[comp[f'{key}_mean'].idxmax()]
    peak_v2 = comp[f'{key}_mean'].max()
    ax.axvline(0, color='k', ls='--', lw=0.5)
    ax.axvline(peak_t2, color='red', ls=':', lw=0.8)
    ax.annotate(f'peak {peak_t2:+.2f} s', xy=(peak_t2, peak_v2),
                xytext=(peak_t2 + 0.5, peak_v2), fontsize=9, color='red')
    if row == 0:
        ax.set_title('Aligned on computed onset (composite S(t))\n192 subj · 906 events')
        ax.legend(loc='upper right', fontsize=8)

axes[-1, 0].set_xlabel('time relative to t₀_net (s)')
axes[-1, 1].set_xlabel('time relative to computed onset (s)')

fig.suptitle('Peri-onset triple average — effect of onset definition on peak sharpness',
             fontsize=13, y=1.00)
plt.tight_layout()
fig_path = os.path.join(OUT_DIR, 'perionset_summary.png')
plt.savefig(fig_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {fig_path}")
