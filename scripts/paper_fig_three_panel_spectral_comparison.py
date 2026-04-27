#!/usr/bin/env python3
"""
Three-panel spectral comparison figure for the schumann_canonical paper.

Panel A: standing aggregate 1/f-corrected residual (B26 result; existing CSV)
Panel B: event-locked log-ratio (B27 result; existing CSV)
Panel C: random-window log-ratio null (newly computed)

The hypothesis the paper makes is that SR1 (~7.8 Hz) elevates only in panel B,
not in A or C. This figure tests it visually.

For LEMON composite: panels A and B come from
  outputs/schumann/images/psd_timelapse/lemon_composite/aggregate_psd_B26_B27.csv
panel C comes from
  outputs/schumann/images/psd_timelapse/lemon_composite/random_window_null_psd.csv

Output: papers/schumann_canonical/images/fig_three_panel_spectral_comparison.png
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO = os.path.join(os.path.dirname(__file__), '..')
PSD_DIR = os.path.join(REPO, 'outputs/schumann/images/psd_timelapse/lemon_composite')
OUT_DIR = os.path.join(REPO, 'papers/schumann_canonical/images')
SCOPE = os.environ.get('SCOPE', 'all')  # 'all' or 'sw'
_TAG = '' if SCOPE == 'all' else f'_{SCOPE}'
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    ab_csv = os.path.join(PSD_DIR, f'aggregate_psd_B26_B27{_TAG}.csv')
    c_csv = os.path.join(PSD_DIR, 'random_window_null_psd.csv')

    if not os.path.exists(ab_csv):
        print(f"Missing: {ab_csv}")
        sys.exit(1)
    if not os.path.exists(c_csv):
        print(f"Missing: {c_csv}")
        sys.exit(1)

    df_ab = pd.read_csv(ab_csv)
    df_c = pd.read_csv(c_csv)

    # Restrict to alpha-region zoom: 4 to 13 Hz
    f_lo, f_hi = 4.0, 13.0
    mask_ab = (df_ab['freq_hz'] >= f_lo) & (df_ab['freq_hz'] <= f_hi)
    mask_c = (df_c['freq_hz'] >= f_lo) & (df_c['freq_hz'] <= f_hi)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=False)

    # Common visual: SR1 marker at 7.83 Hz; alpha range shading at 8-13
    sr1 = 7.83

    # Panel A: standing aggregate 1/f-corrected residual
    ax = axes[0]
    f = df_ab.loc[mask_ab, 'freq_hz'].values
    y = df_ab.loc[mask_ab, 'panel_A_log_resid_grand'].values
    lo = df_ab.loc[mask_ab, 'panel_A_lo'].values
    hi = df_ab.loc[mask_ab, 'panel_A_hi'].values
    ax.fill_between(f, lo, hi, alpha=0.25, color='#888888', label='95% CI')
    ax.plot(f, y, color='#444444', lw=2, label='Standing aggregate')
    ax.axvline(sr1, color='#cc0000', ls='--', lw=1.5, alpha=0.7, label=f'SR1 = {sr1} Hz')
    ax.axhline(0, color='black', ls=':', lw=0.5, alpha=0.5)
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel(r'$\log_{10}$(power / 1/f baseline)', fontsize=11)
    ax.set_title(r'(A) Standing aggregate $1/f$-corrected residual' +
                 '\n(no event-detection)', fontsize=12, loc='left')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(f_lo, f_hi)
    ax.grid(True, alpha=0.3)

    # Panel B: event-locked log-ratio (event/baseline)
    ax = axes[1]
    f = df_ab.loc[mask_ab, 'freq_hz'].values
    y = df_ab.loc[mask_ab, 'panel_B_log_ratio_grand'].values
    lo = df_ab.loc[mask_ab, 'panel_B_lo'].values
    hi = df_ab.loc[mask_ab, 'panel_B_hi'].values
    ax.fill_between(f, lo, hi, alpha=0.25, color='#1f77b4', label='95% CI')
    ax.plot(f, y, color='#1f77b4', lw=2, label='Event-locked')
    ax.axvline(sr1, color='#cc0000', ls='--', lw=1.5, alpha=0.7, label=f'SR1 = {sr1} Hz')
    ax.axhline(0, color='black', ls=':', lw=0.5, alpha=0.5)
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel(r'$\log_{10}$(event / baseline)', fontsize=11)
    ax.set_title(r'(B) Event-locked $\log_{10}$ ratio' +
                 '\n(canonical Q4 events)', fontsize=12, loc='left')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(f_lo, f_hi)
    ax.grid(True, alpha=0.3)

    # Panel C: random-window log-ratio null
    ax = axes[2]
    f = df_c.loc[mask_c, 'freq_hz'].values
    y = df_c.loc[mask_c, 'panel_C_random_window_log_ratio_grand'].values
    lo = df_c.loc[mask_c, 'panel_C_lo'].values
    hi = df_c.loc[mask_c, 'panel_C_hi'].values
    ax.fill_between(f, lo, hi, alpha=0.25, color='#7f7f7f', label='95% CI')
    ax.plot(f, y, color='#7f7f7f', lw=2, label='Random-window null')
    ax.axvline(sr1, color='#cc0000', ls='--', lw=1.5, alpha=0.7, label=f'SR1 = {sr1} Hz')
    ax.axhline(0, color='black', ls=':', lw=0.5, alpha=0.5)
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel(r'$\log_{10}$(random window / baseline)', fontsize=11)
    n_subj = int(os.environ.get('N_SUBJ', '50'))
    ax.set_title(r'(C) Random-window null $\log_{10}$ ratio' +
                 f'\n(matched count, ≥20 s from events; N={n_subj})',
                 fontsize=12, loc='left')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(f_lo, f_hi)
    ax.grid(True, alpha=0.3)

    # Force common y-axis for visual comparability of B and C
    y_b_min = df_ab.loc[mask_ab, 'panel_B_lo'].min()
    y_b_max = df_ab.loc[mask_ab, 'panel_B_hi'].max()
    y_c_min = df_c.loc[mask_c, 'panel_C_lo'].min()
    y_c_max = df_c.loc[mask_c, 'panel_C_hi'].max()
    y_min = min(y_b_min, y_c_min) - 0.05
    y_max = max(y_b_max, y_c_max) + 0.05
    axes[1].set_ylim(y_min, y_max)
    axes[2].set_ylim(y_min, y_max)

    fig.suptitle('SR1 spectral peak is event-conditional: standing vs random-window vs event-locked aggregates (LEMON EC, $N \\approx 192$ for Panels A/B; $N = 50$ pilot for Panel C)',
                 fontsize=13, y=1.02)
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, f'fig_three_panel_spectral_comparison{_TAG}.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Wrote {out_path}")

    # Print SR1-region values for each panel
    sr1_lo, sr1_hi = 7.5, 8.2
    for label, df, col in [
        ('A standing residual', df_ab, 'panel_A_log_resid_grand'),
        ('B event-locked', df_ab, 'panel_B_log_ratio_grand'),
        ('C random-window', df_c, 'panel_C_random_window_log_ratio_grand'),
    ]:
        if 'freq_hz' not in df.columns:
            continue
        m = (df['freq_hz'] >= sr1_lo) & (df['freq_hz'] <= sr1_hi)
        if m.any():
            peak_val = df.loc[m, col].max()
            peak_idx = df.loc[m, col].idxmax()
            peak_freq = df.loc[peak_idx, 'freq_hz']
            print(f"  Panel {label}: peak in [{sr1_lo}, {sr1_hi}] Hz = "
                  f"{peak_val:.4f} at {peak_freq:.2f} Hz")


if __name__ == "__main__":
    main()
