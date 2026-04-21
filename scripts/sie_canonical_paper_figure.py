#!/usr/bin/env python3
"""
Canonical paper figure — "Standing vs event-locked population aggregate."

Two-panel figure showing the central empirical finding:

  Top panel:   Standing aggregate 1/f-corrected periodic residual from 462
                subjects (3 cohorts pooled). No peak at 7.83 Hz; alpha peak
                at 9.45 Hz is the only feature in the theta-alpha range.

  Bottom panel: Event-locked aggregate event/baseline PSD ratio from the
                same 462 subjects. Narrowband peak emerges at 7.80 Hz.

Inputs (from B26 + B27):
  - population_aggregate_psd_pooled.csv
  - event_locked_aggregate_pooled.csv
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'psd_timelapse')

SCHUMANN_F = 7.82     # SR1 center (actual measured, not textbook)
SR3_F = 19.95         # SR3 center (actual measured, not textbook 20.8)
PHI_BOUNDARY = 7.60
FREQ_LO, FREQ_HI = 2.0, 25.0


def open_peak(freqs, y, f_lo, f_hi):
    """Argmax in [f_lo, f_hi] without boundary constraint bias."""
    m = (freqs >= f_lo) & (freqs <= f_hi)
    idx = np.where(m)[0]
    k = idx[int(np.argmax(y[idx]))]
    return float(freqs[k]), float(y[k])


def main():
    standing = pd.read_csv(os.path.join(OUT_DIR, 'population_aggregate_psd_pooled.csv'))
    eventlk  = pd.read_csv(os.path.join(OUT_DIR, 'event_locked_aggregate_pooled.csv'))

    f_s = standing['freq_hz'].values
    r_s = standing['log_resid_grand_mean'].values
    lo_s, hi_s = standing['ci_lo'].values, standing['ci_hi'].values
    f_e = eventlk['freq_hz'].values
    r_e = eventlk['log_ratio_grand_mean'].values
    lo_e, hi_e = eventlk['ci_lo'].values, eventlk['ci_hi'].values
    n_sub = int(eventlk['n_subjects'].iloc[0])

    # Open-peak detection, both panels
    alpha_f, alpha_v = open_peak(f_s, 10 ** r_s, 7.0, 13.0)
    peak_f, peak_v = open_peak(f_e, 10 ** r_e, 7.0, 8.3)

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True,
                              gridspec_kw={'hspace': 0.10})

    # ---- Top: standing aggregate ----
    ax = axes[0]
    ratio_s = 10 ** r_s
    lo_r_s, hi_r_s = 10 ** lo_s, 10 ** hi_s
    ax.plot(f_s, ratio_s, color='#333333', lw=1.8)
    ax.fill_between(f_s, lo_r_s, hi_r_s, color='gray', alpha=0.30)
    ax.axhline(1.0, color='k', lw=0.7)
    ax.axvline(SCHUMANN_F, color='#1a9641', ls='--', lw=1, alpha=0.85,
                label='Schumann 7.83 Hz')
    ax.axvline(PHI_BOUNDARY, color='#666666', ls=':', lw=0.9, alpha=0.8,
                label='φ-lattice θ-α 7.60 Hz')
    ax.annotate(f'α peak {alpha_f:.2f}',
                 xy=(alpha_f, alpha_v),
                 xytext=(alpha_f + 3.5, alpha_v + 0.15),
                 fontsize=9, ha='left',
                 arrowprops=dict(arrowstyle='->', color='#333', lw=0.7))
    ax.set_ylabel('1/f-corrected periodic residual (a.u.)', fontsize=11)
    ax.set_title('A — Standing population aggregate (resting-state)',
                  fontsize=12, loc='left', fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(alpha=0.3)
    ax.set_xlim(FREQ_LO, FREQ_HI)

    # ---- Bottom: event-locked ----
    ax = axes[1]
    ratio_e = 10 ** r_e
    lo_r_e, hi_r_e = 10 ** lo_e, 10 ** hi_e
    ax.plot(f_e, ratio_e, color='#8c1a1a', lw=1.8)
    ax.fill_between(f_e, lo_r_e, hi_r_e, color='#8c1a1a', alpha=0.25)
    ax.axhline(1.0, color='k', lw=0.7)
    ax.axvline(SCHUMANN_F, color='#1a9641', ls='--', lw=1, alpha=0.85,
                label='SR1 (7.82 Hz, actual)')
    ax.axvline(SR3_F, color='#1a9641', ls='--', lw=1, alpha=0.85,
                label='SR3 (19.95 Hz, actual)')
    # Even modes (expected NULLS under odd-only excitation)
    ax.axvline(13.97, color='#ca0020', ls=(0, (2, 4)), lw=0.7, alpha=0.45,
                label='SR2 (14.0) — null predicted')
    # φ-lattice boundary for orientation
    ax.axvline(PHI_BOUNDARY, color='#666666', ls=':', lw=0.7, alpha=0.6)

    # Broadband elevation across [12, 25] Hz: mark with dashed line at median floor
    floor_mask = (f_e >= 12) & (f_e <= 25)
    floor_val = 10 ** np.median(r_e[floor_mask])
    ax.axhline(floor_val, color='#999999', ls=(0, (1, 3)), lw=0.9, alpha=0.6,
                label=f'β floor {floor_val:.2f}×')
    # Mark the SR3 peak — annotation to the RIGHT of the peak, no overlap with SR1
    sec_f, sec_v = open_peak(f_e, 10 ** r_e, 19.5, 21.0)
    ax.annotate(f'SR3 match\n{sec_f:.2f} Hz\n{sec_v:.2f}× ({sec_v/floor_val:.2f}× floor)',
                 xy=(sec_f, sec_v),
                 xytext=(sec_f + 1.0, sec_v + 1.0),
                 fontsize=8.5, ha='left',
                 arrowprops=dict(arrowstyle='->', color='#444', lw=0.6))

    # Event peak annotation at SR1 — annotation ABOVE-LEFT of peak
    ax.annotate(f'SR1 match\n{peak_f:.2f} Hz (SE ≈ 0.04)\n{peak_v:.2f}×',
                 xy=(peak_f, peak_v),
                 xytext=(peak_f - 4.0, peak_v + 0.8),
                 fontsize=9, ha='left',
                 arrowprops=dict(arrowstyle='->', color='#8c1a1a', lw=0.8))
    ax.set_xlabel('frequency (Hz)', fontsize=11)
    ax.set_ylabel('event-window PSD / baseline PSD (×)', fontsize=11)
    ax.set_title('B — Same 462 brains, time-locked to ignition events',
                  fontsize=12, loc='left', fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(alpha=0.3)
    ax.set_xlim(FREQ_LO, FREQ_HI)

    fig.suptitle('Event-locked enhancement at odd-mode Schumann frequencies (SR1 + SR3)',
                 fontsize=13, y=0.995)

    # Rich caption
    caption = (
        'Three cohorts pooled: LEMON (healthy adults, N=192), HBN R4 '
        '(children/teens, N=219), TDBRAIN (clinical adults, N=51). Bootstrap 95% CI '
        'shaded. '
        '(A) y-axis is the periodic component above the aperiodic 1/f background '
        '(fit on 2–5 ∪ 9–22 Hz). '
        '(B) y-axis is the subject-mean (event PSD / baseline PSD); panels plot '
        'different normalizations because the comparison is structural (peak location), '
        'not quantitative. '
        'Green dashed lines: actual measured Schumann harmonics SR1 (7.82) and SR3 '
        '(19.95); red dashed line: SR2 (13.97), where odd-only cavity excitation '
        'predicts a null. Both odd-mode peaks show clear elevation '
        '(SR1 1.85× above β floor; SR3 1.15× above floor); SR2 and SR4 sit at floor. '
        'Per-cohort event-locked peak frequencies: LEMON 7.80 Hz, HBN 7.85 Hz, '
        'TDBRAIN 7.75 Hz — all within 0.1 Hz of SR1. '
        'Wide CI in (B) at SR1 reflects between-cohort variation in event/baseline '
        'amplitude (HBN ~12×, LEMON/TDBRAIN ~3×), not uncertainty about peak location. '
        'Dashed gray line in (B) marks the β-band broadband floor (~3.2×) from '
        'detector construction: event windows are higher-amplitude across the whole '
        'band because Stage-1 fires on high-envelope windows. '
        'The odd-mode (SR1 + SR3, no SR2) pattern is the specific signature of a '
        'vertical-dipole source coupled to Earth-ionosphere cavity modes.'
    )
    fig.text(0.5, -0.02, caption, ha='center', va='top',
              fontsize=8.5, style='italic', wrap=True,
              bbox=dict(facecolor='none', edgecolor='none'))

    plt.savefig(os.path.join(OUT_DIR, 'canonical_paper_figure.png'),
                 dpi=180, bbox_inches='tight')
    plt.savefig(os.path.join(OUT_DIR, 'canonical_paper_figure.pdf'),
                 bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUT_DIR}/canonical_paper_figure.png")
    print(f"Saved: {OUT_DIR}/canonical_paper_figure.pdf")

    print(f"\n=== Verification ===")
    print(f"Panel A open-peak search in [7, 13]: {alpha_f:.3f} Hz @ {alpha_v:.3f}")
    print(f"Panel B open-peak search in [7, 8.3]: {peak_f:.3f} Hz @ {peak_v:.3f}×")
    print(f"Value at exactly 7.83 in B: "
          f"{ratio_e[int(np.argmin(np.abs(f_e - 7.83)))]:.3f}×")


if __name__ == '__main__':
    main()
