#!/usr/bin/env python3
"""Four-panel figure: peri-event four-stream architecture across f0 detectors.

Shows that the canonical six-phase peri-event architecture is essentially
band-independent: f0 = 7.6, 8.6, 12.0 Hz detectors all produce events
with the same desynchronization -> nadir -> rebound trajectory.

Reads:
  outputs/schumann/images/psd_timelapse/lemon_composite/
    off_frequency_peri_event_streams.csv

Output:
  papers/schumann_canonical/images/fig_off_frequency_peri_event.png
"""
from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt

REPO = os.path.join(os.path.dirname(__file__), '..')
SRC = os.path.join(REPO, 'outputs/schumann/images/psd_timelapse/lemon_composite/'
                   'off_frequency_peri_event_streams.csv')
OUT_DIR = os.path.join(REPO, 'papers/schumann_canonical/images')
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    df = pd.read_csv(SRC)
    streams = ['env', 'R', 'PLV', 'MSC']
    titles = {
        'env': 'Envelope amplitude',
        'R':   'Kuramoto $R$ (global phase coherence)',
        'PLV': 'PLV (phase-locking to median ref)',
        'MSC': 'MSC (magnitude-squared coherence)',
    }
    f0_list = sorted(df['f0'].unique())
    colors = {7.6: '#1f77b4', 8.6: '#2ca02c', 12.0: '#d62728'}

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharex=True)
    for j, sname in enumerate(streams):
        ax = axes[j]
        for f0 in f0_list:
            sel = df[(df['f0'] == f0) & (df['stream'] == sname)].sort_values('t_rel')
            label = f'$f_0 = {f0}$ Hz'
            if f0 == 7.6:
                label += ' (canonical)'
            ax.plot(sel['t_rel'], sel['z'], color=colors.get(f0, 'gray'),
                    lw=1.8, label=label)
        ax.axvline(0, color='black', lw=0.5, ls=':', alpha=0.6)
        ax.axhline(0, color='black', lw=0.5, ls=':', alpha=0.6)
        ax.set_xlabel('Time relative to event onset (s)', fontsize=11)
        ax.set_ylabel('z (vs pre-event baseline)' if j == 0 else '', fontsize=11)
        ax.set_title(titles[sname], fontsize=12, loc='left')
        ax.set_xlim(-3, 3)
        ax.grid(True, alpha=0.3)
        if j == 3:
            ax.legend(loc='lower right', fontsize=10)

    fig.suptitle('Peri-event architecture is band-independent: '
                 'the canonical six-phase trajectory replicates across '
                 '$f_0 \\in \\{7.6, 8.6, 12.0\\}$~Hz detectors '
                 '(LEMON EC, $N = 50$ pilot)', fontsize=13, y=1.04)
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, 'fig_off_frequency_peri_event.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
