#!/usr/bin/env python3
"""
Figure 3 redesign under composite v2 detector.

Four-panel figure showcasing Arc 7 composite-specific findings that diverge
from or strengthen the envelope story:

  A. Posterior co-localization (§47): three topomaps SR1, β16, SR3.
     Highlights the SR1 × SR3 r sign-flip (envelope −0.44 → composite +0.48)
     and SR3's move from central-left (envelope) to purely posterior.

  B. Per-subject reliability by band (§48): histograms of subject-to-group
     LOO ρ for Q4 SR1, β16, SR3. SR1 median +0.46 (45% > 0.5); β16/SR3 not
     reliable. Core individual-subject result.

  C. Posterior IAF × SR1 peak (§49): scatter showing near-zero Spearman
     (ρ = −0.004). SR1 peak = 7.86 Hz, within 0.03 Hz of Schumann. Strongest
     form of IAF-independence.

  D. SR1 × SR3 envelope coupling (§50): event vs control peak correlation
     bars. Δr = +0.064, p = 0.002 — shared-envelope coupling at events,
     not directed (lag Δ null).

Pulls from pre-computed CSVs in outputs/schumann/images/coupling/lemon_composite/.

Usage:
    python scripts/sie_paper_figure3_composite.py --cohort lemon
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.run_sie_extraction import (
    load_lemon, load_dortmund, load_srm, load_tdbrain,
)

warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

ROOT = os.path.join(os.path.dirname(__file__), '..')

BAND_LABELS = {'SR1': 'SR1  (7-8 Hz)',
               'β16': 'β16  (14.5-17.5 Hz)',
               'SR3': 'SR3  (19.5-20.4 Hz)'}


def cohort_loader(cohort):
    if cohort == 'lemon':
        return load_lemon, {'condition': 'EC'}
    if cohort == 'lemon_EO':
        return load_lemon, {'condition': 'EO'}
    if cohort == 'tdbrain':
        return load_tdbrain, {'condition': 'EC'}
    raise ValueError(f"unsupported cohort {cohort!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', default='lemon')
    args = ap.parse_args()

    data_dir = os.path.join(ROOT, 'outputs', 'schumann', 'images',
                             'coupling', f'{args.cohort}_composite')
    out_path = os.path.join(data_dir, 'paper_figure3_composite.png')

    # Load inputs
    topo_df = pd.read_csv(os.path.join(data_dir, '16hz_topography.csv'))
    rel_df = pd.read_csv(os.path.join(data_dir, 'network_reliability.csv'))
    iaf_df = pd.read_csv(os.path.join(data_dir, 'posterior_sr1_tightened.csv'))
    dc_df = pd.read_csv(os.path.join(data_dir, 'directed_coupling.csv'))

    # Build figure
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 4, figure=fig,
                            height_ratios=[1.0, 1.0],
                            hspace=0.42, wspace=0.32)

    # ------------------------------------------------------------------
    # Panel A: Three topomaps (SR1, β16, SR3)
    # ------------------------------------------------------------------
    loader, loader_kw = cohort_loader(args.cohort)
    # Use a reference subject for channel montage
    # Find first subject from the extraction summary
    events_dir = os.path.join(ROOT, 'exports_sie', f'{args.cohort}_composite')
    summary = pd.read_csv(os.path.join(events_dir, 'extraction_summary.csv'))
    ok = summary[summary['status'] == 'ok']
    ref_sub = None
    for sid in ok['subject_id']:
        try:
            raw_ref = loader(sid, **loader_kw)
            if raw_ref is not None:
                ref_sub = sid
                break
        except Exception:
            continue
    if ref_sub is None:
        raise RuntimeError("could not load any reference subject for montage")
    fs_ref = raw_ref.info['sfreq']

    # Get common channels from the CSV (all channels present in grand-average)
    common_chs = sorted(topo_df[topo_df['band'] == 'SR1']['channel'].unique().tolist())
    montage = mne.channels.make_standard_montage('standard_1020')
    info_common = mne.create_info(ch_names=common_chs, sfreq=fs_ref,
                                    ch_types='eeg')
    info_common.set_montage(montage, match_case=False, on_missing='ignore')

    band_map_axes = []
    # Determine vmin/vmax separately per band (each has different scale)
    for i, band in enumerate(['SR1', 'β16', 'SR3']):
        ax = fig.add_subplot(gs[0, i])
        band_map_axes.append(ax)
        sub = topo_df[topo_df['band'] == band]
        data = np.array([sub[sub['channel'] == c]['ratio'].values[0]
                          if not sub[sub['channel'] == c].empty else np.nan
                          for c in common_chs])
        im, _ = mne.viz.plot_topomap(data, info_common, axes=ax,
                                      show=False, cmap='viridis',
                                      contours=6)
        peak_ch = common_chs[int(np.nanargmax(data))]
        ax.set_title(f'{BAND_LABELS[band]}\n'
                     f'range {data.min():.2f}-{data.max():.2f}×\n'
                     f'peak @ {peak_ch}',
                     fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.75, fraction=0.046)

    # ------------------------------------------------------------------
    # Panel B: reliability histograms
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 3])
    band_colors = {'SR1': '#2166ac', 'β16': '#d73027', 'SR3': '#1a9641'}
    for b in ['SR1', 'β16', 'SR3']:
        r = rel_df[(rel_df['test'] == 'reliability') &
                    (rel_df['quartile'] == 'q4') &
                    (rel_df['band'] == b)]['rho'].dropna().values
        med = np.median(r)
        pct05 = (r > 0.5).mean() * 100
        ax.hist(r, bins=np.linspace(-1, 1, 25), alpha=0.45,
                color=band_colors[b], edgecolor='k', lw=0.2,
                label=f'{b}: med {med:+.2f}, {pct05:.0f}% > 0.5')
    ax.axvline(0, color='k', lw=0.5)
    ax.axvline(0.5, color='green', ls=':', lw=0.7)
    ax.set_xlabel('subject-to-group ρ  (Q4 topography vs LOO)')
    ax.set_ylabel('subjects')
    ax.set_title('Per-subject reliability by band')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.3)

    # ------------------------------------------------------------------
    # Panel C: Posterior IAF × SR1 peak
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, :2])
    x = iaf_df['iaf_hz_posterior'].values
    y = iaf_df['sr1_peak_hz_posterior'].values
    rho, p = spearmanr(x, y)
    slope, intercept = np.polyfit(x, y, 1)
    ax.scatter(x, y, s=35, alpha=0.55, color='steelblue', edgecolor='k', lw=0.3)
    rng = np.array([x.min() - 0.3, x.max() + 0.3])
    ax.plot(rng, rng, 'k--', lw=1.0, label='IAF-lock:  y = x')
    ax.axhline(7.83, color='#1a9641', ls=':', lw=1.2,
                label='Schumann f₁ = 7.83 Hz')
    ax.plot(rng, slope * rng + intercept, color='red', lw=1.6,
            label=f'OLS  y = {slope:+.3f}·IAF {intercept:+.2f}')
    ax.set_xlabel('IAF  (posterior-mean, Hz)', fontsize=11)
    ax.set_ylabel('SR1 peak at events  (posterior-mean, Hz)', fontsize=11)
    ax.set_title(f'Posterior IAF-independence   Spearman ρ = {rho:+.3f}  p = {p:.2g}  '
                 f'n = {len(x)}   (envelope B46 ρ = −0.115)',
                 fontsize=11)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3)

    # ------------------------------------------------------------------
    # Panel D: SR1 × SR3 envelope coupling strength — event vs control
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 2:])
    pair_labels = ['SR1 × β16', 'SR1 × SR3', 'β16 × SR3']
    pair_keys = ['sr1_b16', 'sr1_sr3', 'b16_sr3']
    ev_rpk = []
    ct_rpk = []
    p_rpk = []
    for pk in pair_keys:
        row = dc_df[(dc_df['pair'] == pk) & (dc_df['metric'] == 'rpk')]
        ev_rpk.append(float(row['event_mean'].values[0]))
        ct_rpk.append(float(row['control_mean'].values[0]))
        p_rpk.append(float(row['p_wilcoxon'].values[0]))
    xpos = np.arange(len(pair_labels))
    w = 0.38
    ax.bar(xpos - w/2, ev_rpk, w, color='firebrick', alpha=0.85,
            edgecolor='k', lw=0.3, label='event')
    ax.bar(xpos + w/2, ct_rpk, w, color='gray', alpha=0.75,
            edgecolor='k', lw=0.3, label='control')
    for i, (ev, ct, pv) in enumerate(zip(ev_rpk, ct_rpk, p_rpk)):
        delta = ev - ct
        marker = '*' if pv < 0.01 else ('+' if pv < 0.05 else '')
        ymax = max(ev, ct) + 0.04
        ax.text(xpos[i], ymax,
                f'Δ = {delta:+.3f}\np = {pv:.3g} {marker}',
                ha='center', va='bottom', fontsize=9)
    ax.set_xticks(xpos)
    ax.set_xticklabels(pair_labels)
    ax.set_ylabel('peak envelope correlation  r')
    ax.set_ylim(0, max(ev_rpk + ct_rpk) * 1.25)
    ax.set_title('Envelope coupling strength  event vs control   '
                 '(no lag Δ significant)', fontsize=11)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle(
        f'Figure 3 (composite v2) — Arc 7 under composite detector   ({args.cohort} Q4)',
        y=0.995, fontsize=13
    )
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
