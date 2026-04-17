#!/usr/bin/env python3
"""
Regional Enrichment Analysis — Per-Channel and Per-Region Spectral Differentiation
====================================================================================

Computes Voronoi enrichment broken down by EEG channel and brain region, using
the existing v3 peaks CSVs (which already contain a 'channel' column).

Analyses:
1. Per-channel enrichment pooled across subjects → identifies which electrodes
   show strongest spectral differentiation at each lattice position
2. ROI-level enrichment (frontal, central, parietal, temporal, occipital) with
   cross-dataset replication
3. Topographic maps of enrichment strength per band×position
4. Regional EC vs EO comparison (LEMON + Dortmund)
5. Regional age trajectories (HBN developmental, Dortmund adult)

Datasets: LEMON (62ch), Dortmund (64ch), EEGMMIDB (64ch), CHBMP (62-120ch),
HBN (128ch). Sparse-montage datasets (Emotiv 14ch) are excluded.

Usage:
    python scripts/voronoi_regional_enrichment.py --step all
    python scripts/voronoi_regional_enrichment.py --step channel
    python scripts/voronoi_regional_enrichment.py --step region
    python scripts/voronoi_regional_enrichment.py --step topomap
    python scripts/voronoi_regional_enrichment.py --step eceo
    python scripts/voronoi_regional_enrichment.py --step age
"""

import os
import sys
import argparse
import glob
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
from phi_frequency_model import PHI, F0

# =========================================================================
# CONSTANTS (mirrors run_all_f0_760_analyses.py)
# =========================================================================

PHI_INV = 1.0 / PHI
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
PEAK_BASE = os.path.join(BASE_DIR, 'exports_adaptive_v4')
OUT_DIR = os.path.join(BASE_DIR, 'outputs', 'regional_enrichment')

MIN_POWER_PCT = 50

POS_LIST = [
    ('boundary',    0.000),
    ('noble_6',     round(PHI_INV ** 6, 6)),
    ('noble_5',     round(PHI_INV ** 5, 6)),
    ('noble_4',     round(PHI_INV ** 4, 6)),
    ('noble_3',     round(PHI_INV ** 3, 6)),
    ('inv_noble_1', round(PHI_INV ** 2, 6)),
    ('attractor',   0.5),
    ('noble_1',     round(PHI_INV, 6)),
    ('inv_noble_3', round(1 - PHI_INV ** 3, 6)),
    ('inv_noble_4', round(1 - PHI_INV ** 4, 6)),
    ('inv_noble_5', round(1 - PHI_INV ** 5, 6)),
    ('inv_noble_6', round(1 - PHI_INV ** 6, 6)),
]
POS_NAMES = [p[0] for p in POS_LIST]
POS_VALS = np.array([p[1] for p in POS_LIST])
N_POS = len(POS_VALS)

# Hz-weighted Voronoi bin fractions (same as run_all_f0_760_analyses.py)
_VORONOI_EDGES = []
for i in range(N_POS):
    if i == 0:
        u_left = (POS_VALS[-1] + POS_VALS[0] + 1) / 2 % 1.0
        u_right = (POS_VALS[0] + POS_VALS[1]) / 2
    elif i == N_POS - 1:
        u_left = (POS_VALS[i - 1] + POS_VALS[i]) / 2
        u_right = (POS_VALS[i] + POS_VALS[0] + 1) / 2
    else:
        u_left = (POS_VALS[i - 1] + POS_VALS[i]) / 2
        u_right = (POS_VALS[i] + POS_VALS[i + 1]) / 2
    _VORONOI_EDGES.append((u_left, u_right))

HZ_FRACS = []
for i in range(N_POS):
    u_left, u_right = _VORONOI_EDGES[i]
    if i == 0:
        hz_frac = (PHI ** 1.0 - PHI ** u_left + PHI ** u_right - PHI ** 0.0) / (PHI - 1)
    else:
        hz_frac = (PHI ** u_right - PHI ** u_left) / (PHI - 1)
    HZ_FRACS.append(hz_frac)

OCTAVE_BAND = {
    'n-1': 'theta', 'n+0': 'alpha', 'n+1': 'beta_low',
    'n+2': 'beta_high', 'n+3': 'gamma',
}
BAND_ORDER = ['theta', 'alpha', 'beta_low', 'beta_high', 'gamma']

# =========================================================================
# CHANNEL → ROI MAPPING
# =========================================================================
# Canonical 10-20 / 10-10 region assignments. Handles naming variants across
# datasets (EEGMMIDB trailing dots, CHBMP -REF suffix, HBN E-numbers).

ROI_MAP_1020 = {
    # Frontal
    'Fp1': 'frontal', 'Fp2': 'frontal', 'Fpz': 'frontal', 'FpZ': 'frontal',
    'AF3': 'frontal', 'AF4': 'frontal', 'AF7': 'frontal', 'AF8': 'frontal', 'AFz': 'frontal',
    'F1': 'frontal', 'F2': 'frontal', 'F3': 'frontal', 'F4': 'frontal',
    'F5': 'frontal', 'F6': 'frontal', 'F7': 'frontal', 'F8': 'frontal', 'Fz': 'frontal',
    # Frontocentral
    'FC1': 'central', 'FC2': 'central', 'FC3': 'central', 'FC4': 'central',
    'FC5': 'central', 'FC6': 'central', 'FCz': 'central', 'FCZ': 'central',
    'FT7': 'temporal', 'FT8': 'temporal', 'FT9': 'temporal', 'FT10': 'temporal',
    # Central
    'C1': 'central', 'C2': 'central', 'C3': 'central', 'C4': 'central',
    'C5': 'central', 'C6': 'central', 'Cz': 'central', 'CZ': 'central',
    # Temporal
    'T7': 'temporal', 'T8': 'temporal', 'T9': 'temporal', 'T10': 'temporal',
    'TP7': 'temporal', 'TP8': 'temporal', 'TP9': 'temporal', 'TP10': 'temporal',
    # Centroparietal
    'CP1': 'parietal', 'CP2': 'parietal', 'CP3': 'parietal', 'CP4': 'parietal',
    'CP5': 'parietal', 'CP6': 'parietal', 'CPz': 'parietal', 'CPZ': 'parietal',
    # Parietal
    'P1': 'parietal', 'P2': 'parietal', 'P3': 'parietal', 'P4': 'parietal',
    'P5': 'parietal', 'P6': 'parietal', 'P7': 'parietal', 'P8': 'parietal',
    'Pz': 'parietal', 'PZ': 'parietal',
    # Parieto-occipital
    'PO3': 'occipital', 'PO4': 'occipital', 'PO5': 'occipital', 'PO6': 'occipital',
    'PO7': 'occipital', 'PO8': 'occipital', 'PO9': 'occipital', 'PO10': 'occipital',
    'POz': 'occipital', 'POZ': 'occipital',
    # Occipital
    'O1': 'occipital', 'O2': 'occipital', 'Oz': 'occipital', 'OZ': 'occipital',
    'Iz': 'occipital',
}

ROI_ORDER = ['frontal', 'central', 'temporal', 'parietal', 'occipital']

# HBN EGI 128-channel → ROI mapping (approximate, based on GSN 200 layout)
# Groups: frontal (E1-E36 approx), central (E37-E55), temporal (E56-E67,E96-E103),
# parietal (E68-E82,E87-E95), occipital (E83-E86,E104-E128 approx)
HBN_ROI = {}
_frontal = [1,2,3,4,8,9,10,11,14,15,16,17,18,19,21,22,23,24,25,26,27,28,32,33,122,123,124,125,126,127,128]
_central = [5,6,7,12,13,20,29,30,31,34,35,36,37,42,54,55,79,80,87,93,104,105,106,111,112,118]
_temporal = [38,39,40,43,44,45,46,47,48,49,50,51,56,57,63,64,95,96,97,98,99,100,101,102,103,107,108,113,114,115,116,117,119,120,121]
_parietal = [41,52,53,58,59,60,61,62,65,66,67,71,72,77,78,84,85,86,88,89,90,91,92,94]
_occipital = [68,69,70,73,74,75,76,81,82,83]
for _nums, _roi in [(_frontal, 'frontal'), (_central, 'central'), (_temporal, 'temporal'),
                     (_parietal, 'parietal'), (_occipital, 'occipital')]:
    for n in _nums:
        HBN_ROI[f'E{n}'] = _roi


def normalize_channel(ch):
    """Normalize channel name to canonical 10-20 form."""
    ch = ch.strip()
    # EEGMMIDB: trailing dots (e.g., 'Fc5.', 'C3..')
    ch = ch.rstrip('.')
    # CHBMP: '-REF' suffix and spaces (e.g., 'C1 -REF', 'AF3-REF')
    ch = ch.replace('-REF', '').replace(' ', '')
    # Capitalize first letter, rest lower (e.g., 'Fc5' → 'FC5' needs special handling)
    # Standard 10-20: uppercase region letters, lowercase z, digits
    return ch


def channel_to_roi(ch, dataset=None):
    """Map a channel name to a brain region."""
    norm = normalize_channel(ch)
    # HBN uses E-number naming
    if norm in HBN_ROI:
        return HBN_ROI[norm]
    # Try direct lookup
    if norm in ROI_MAP_1020:
        return ROI_MAP_1020[norm]
    # Try case-insensitive match
    norm_upper = norm.upper()
    for key, roi in ROI_MAP_1020.items():
        if key.upper() == norm_upper:
            return roi
    # CHBMP numeric channels (from high-density montage) — skip
    return None


# =========================================================================
# CORE FUNCTIONS
# =========================================================================

def lattice_coord(freqs, f0=F0):
    return (np.log(freqs / f0) / np.log(PHI)) % 1.0


def assign_voronoi(u_vals):
    u = np.asarray(u_vals, dtype=float) % 1.0
    dists = np.abs(u[:, None] - POS_VALS[None, :])
    dists = np.minimum(dists, 1 - dists)
    return np.argmin(dists, axis=1)


def compute_enrichment(freqs):
    """Compute enrichment at each position from an array of frequencies.
    Returns dict of position_name → enrichment_pct, or None if too few peaks."""
    n = len(freqs)
    if n < 10:
        return None
    u = lattice_coord(freqs)
    assignments = assign_voronoi(u)
    result = {}
    for i, pname in enumerate(POS_NAMES):
        count = int((assignments == i).sum())
        expected = HZ_FRACS[i] * n
        result[pname] = (count / expected - 1) * 100 if expected > 0 else 0
    result['n_peaks'] = n
    return result


def load_peaks_with_channels(directory, min_power_pct=None):
    """Load peaks from CSVs, keeping channel column."""
    files = sorted(glob.glob(os.path.join(directory, '*_peaks.csv')))
    if not files:
        return None, 0
    first = pd.read_csv(files[0], nrows=1)
    has_power = 'power' in first.columns
    has_channel = 'channel' in first.columns
    if not has_channel:
        return None, 0
    if min_power_pct is None:
        min_power_pct = MIN_POWER_PCT
    cols = ['freq', 'phi_octave', 'channel'] + (['power'] if has_power else [])
    dfs = []
    for f in files:
        df = pd.read_csv(f, usecols=cols)
        sub_id = os.path.basename(f).replace('_peaks.csv', '')
        df['subject'] = sub_id
        dfs.append(df)
    peaks = pd.concat(dfs, ignore_index=True)
    if has_power and min_power_pct > 0:
        filtered = []
        for octave in peaks['phi_octave'].unique():
            bp = peaks[peaks.phi_octave == octave]
            if len(bp) >= 2:
                thresh = bp['power'].quantile(min_power_pct / 100)
                filtered.append(bp[bp['power'] >= thresh])
            else:
                filtered.append(bp)
        peaks = pd.concat(filtered, ignore_index=True)
    return peaks, len(files)


# Datasets with good spatial coverage
REGIONAL_DATASETS = {
    'lemon': 'lemon',
    'eegmmidb': 'eegmmidb',
    'dortmund': 'dortmund',
    'chbmp': 'chbmp',
    'hbn_R1': 'hbn_R1',
    'hbn_R2': 'hbn_R2',
    'hbn_R3': 'hbn_R3',
    'hbn_R4': 'hbn_R4',
    'hbn_R6': 'hbn_R6',
}


# =========================================================================
# STEP 1: PER-CHANNEL ENRICHMENT
# =========================================================================

def run_channel_enrichment():
    print("=" * 70)
    print("  STEP 1: Per-Channel Enrichment")
    print("=" * 70)

    os.makedirs(OUT_DIR, exist_ok=True)
    all_channel_rows = []

    for ds_name, subdir in REGIONAL_DATASETS.items():
        path = os.path.join(PEAK_BASE, subdir)
        peaks, n_sub = load_peaks_with_channels(path)
        if peaks is None:
            print(f"  {ds_name}: NO DATA or no channel column")
            continue

        channels = sorted(peaks['channel'].unique())
        print(f"  {ds_name}: {n_sub} subjects, {len(channels)} channels, {len(peaks):,} peaks")

        for band_octave, band_name in OCTAVE_BAND.items():
            band_peaks = peaks[peaks.phi_octave == band_octave]
            for ch in channels:
                ch_freqs = band_peaks[band_peaks.channel == ch]['freq'].values
                enr = compute_enrichment(ch_freqs)
                if enr is None:
                    continue
                roi = channel_to_roi(ch, ds_name)
                row = {'dataset': ds_name, 'band': band_name, 'channel': ch,
                       'roi': roi, 'n_peaks': enr.pop('n_peaks')}
                row.update(enr)
                all_channel_rows.append(row)

    if not all_channel_rows:
        print("  No channel data found!")
        return

    df = pd.DataFrame(all_channel_rows)
    df.to_csv(os.path.join(OUT_DIR, 'per_channel_enrichment.csv'), index=False)
    print(f"\n  Saved: {OUT_DIR}/per_channel_enrichment.csv")

    # Summary: top/bottom channels per band for key positions
    print("\n  TOP CHANNELS BY ENRICHMENT (pooled across datasets with 10-20 names)")
    df_named = df[df['roi'].notna()].copy()
    for band in BAND_ORDER:
        bd = df_named[df_named.band == band]
        if bd.empty:
            continue
        for pos in ['noble_1', 'attractor', 'boundary']:
            if pos not in bd.columns:
                continue
            top5 = bd.nlargest(5, pos)[['dataset', 'channel', 'roi', pos, 'n_peaks']]
            print(f"\n  {band} / {pos} — top 5:")
            for _, r in top5.iterrows():
                print(f"    {r['dataset']:>8s} {r['channel']:>6s} ({r['roi']:>10s}): {r[pos]:>+6.0f}%  (n={r['n_peaks']})")

    return df


# =========================================================================
# STEP 2: ROI-LEVEL ENRICHMENT
# =========================================================================

def run_region_enrichment():
    print("\n" + "=" * 70)
    print("  STEP 2: ROI-Level Enrichment")
    print("=" * 70)

    os.makedirs(OUT_DIR, exist_ok=True)
    all_roi_rows = []

    for ds_name, subdir in REGIONAL_DATASETS.items():
        path = os.path.join(PEAK_BASE, subdir)
        peaks, n_sub = load_peaks_with_channels(path)
        if peaks is None:
            continue

        # Assign ROI
        peaks['roi'] = peaks['channel'].apply(lambda c: channel_to_roi(c, ds_name))
        peaks_roi = peaks[peaks['roi'].notna()]
        n_mapped = peaks_roi['channel'].nunique()
        n_total = peaks['channel'].nunique()
        print(f"  {ds_name}: {n_mapped}/{n_total} channels mapped to ROIs, "
              f"{len(peaks_roi):,}/{len(peaks):,} peaks")

        for band_octave, band_name in OCTAVE_BAND.items():
            band_peaks = peaks_roi[peaks_roi.phi_octave == band_octave]
            for roi in ROI_ORDER:
                roi_freqs = band_peaks[band_peaks.roi == roi]['freq'].values
                enr = compute_enrichment(roi_freqs)
                if enr is None:
                    continue
                row = {'dataset': ds_name, 'band': band_name, 'roi': roi,
                       'n_peaks': enr.pop('n_peaks')}
                row.update(enr)
                all_roi_rows.append(row)

    if not all_roi_rows:
        print("  No ROI data!")
        return

    df = pd.DataFrame(all_roi_rows)
    df.to_csv(os.path.join(OUT_DIR, 'roi_enrichment.csv'), index=False)
    print(f"\n  Saved: {OUT_DIR}/roi_enrichment.csv")

    # Cross-dataset summary table
    print("\n  ROI ENRICHMENT SUMMARY (% enrichment at key positions)")
    for band in BAND_ORDER:
        bd = df[df.band == band]
        if bd.empty:
            continue
        print(f"\n  {band.upper()}")
        # Aggregate across datasets: mean enrichment per ROI
        print(f"  {'ROI':<12s}", end='')
        key_pos = ['boundary', 'noble_1', 'attractor', 'inv_noble_3', 'noble_3']
        for pos in key_pos:
            print(f" {pos:>14s}", end='')
        print(f" {'n_datasets':>10s}")

        for roi in ROI_ORDER:
            rd = bd[bd.roi == roi]
            if rd.empty:
                continue
            print(f"  {roi:<12s}", end='')
            for pos in key_pos:
                vals = rd[pos].dropna().values
                if len(vals) > 0:
                    print(f" {np.mean(vals):>+13.0f}%", end='')
                else:
                    print(f" {'—':>14s}", end='')
            print(f" {len(rd):>10d}")

    # Test: does enrichment differ significantly by region?
    print("\n  KRUSKAL-WALLIS: does enrichment differ by ROI? (per band × position)")
    sig_results = []
    for band in BAND_ORDER:
        bd = df[df.band == band]
        for pos in POS_NAMES:
            groups = [bd[bd.roi == roi][pos].dropna().values for roi in ROI_ORDER]
            groups = [g for g in groups if len(g) >= 3]
            if len(groups) < 3:
                continue
            h_stat, p_val = stats.kruskal(*groups)
            if p_val < 0.05:
                sig_results.append({'band': band, 'position': pos, 'H': h_stat, 'p': p_val})

    if sig_results:
        sig_df = pd.DataFrame(sig_results).sort_values('p')
        print(f"  {len(sig_df)} significant (p<0.05) out of {len(BAND_ORDER) * N_POS} tests:")
        for _, r in sig_df.head(15).iterrows():
            print(f"    {r['band']:>10s} / {r['position']:<14s}  H={r['H']:.1f}  p={r['p']:.4f}")
    else:
        print("  None significant at p<0.05")

    return df


# =========================================================================
# STEP 3: TOPOGRAPHIC MAPS
# =========================================================================

def run_topomap():
    print("\n" + "=" * 70)
    print("  STEP 3: Topographic Maps")
    print("=" * 70)

    try:
        import mne
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  MNE or matplotlib not available — skipping topomaps")
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    # Use LEMON and Dortmund (standard 10-10 names, good coverage)
    for ds_name in ['lemon', 'dortmund', 'eegmmidb']:
        path = os.path.join(PEAK_BASE, ds_name)
        peaks, n_sub = load_peaks_with_channels(path)
        if peaks is None:
            continue

        # Normalize channel names for MNE montage lookup
        raw_channels = sorted(peaks['channel'].unique())
        ch_map = {}
        for ch in raw_channels:
            norm = normalize_channel(ch)
            ch_map[ch] = norm

        peaks['ch_norm'] = peaks['channel'].map(ch_map)

        # Build MNE info with standard montage
        unique_norm = sorted(set(ch_map.values()))
        try:
            montage = mne.channels.make_standard_montage('standard_1020')
            valid_chs = [ch for ch in unique_norm if ch.lower() in
                         [m.lower() for m in montage.ch_names]]
            if len(valid_chs) < 10:
                # Try standard_1005
                montage = mne.channels.make_standard_montage('standard_1005')
                valid_chs = [ch for ch in unique_norm if ch.lower() in
                             [m.lower() for m in montage.ch_names]]
        except Exception:
            print(f"  {ds_name}: could not create montage — skipping")
            continue

        if len(valid_chs) < 10:
            print(f"  {ds_name}: only {len(valid_chs)} channels matched montage — skipping")
            continue

        # Match case to montage
        montage_lower = {m.lower(): m for m in montage.ch_names}
        matched_chs = [montage_lower[ch.lower()] for ch in valid_chs]

        info = mne.create_info(ch_names=matched_chs, sfreq=256, ch_types='eeg')
        info.set_montage(montage)

        # Build norm→montage mapping
        norm_to_montage = {}
        for ch in valid_chs:
            norm_to_montage[ch] = montage_lower[ch.lower()]

        print(f"  {ds_name}: {len(matched_chs)} channels for topomap")

        # Compute per-channel enrichment for this dataset
        key_positions = ['noble_1', 'attractor', 'boundary']
        for band_octave, band_name in OCTAVE_BAND.items():
            band_peaks = peaks[peaks.phi_octave == band_octave]

            fig, axes = plt.subplots(1, len(key_positions), figsize=(4 * len(key_positions), 3.5))
            if len(key_positions) == 1:
                axes = [axes]
            has_data = False

            for ax, pos in zip(axes, key_positions):
                values = []
                ch_order = []
                for ch_norm, ch_montage in norm_to_montage.items():
                    # Get all raw channel names mapping to this norm
                    raw_chs = [k for k, v in ch_map.items() if v == ch_norm]
                    ch_freqs = band_peaks[band_peaks.channel.isin(raw_chs)]['freq'].values
                    enr = compute_enrichment(ch_freqs)
                    if enr is not None:
                        values.append(enr[pos])
                        ch_order.append(ch_montage)

                if len(values) < 5:
                    ax.set_title(f'{pos}\n(insufficient data)')
                    ax.axis('off')
                    continue

                has_data = True
                # Create data array in channel order
                ch_idx = {ch: i for i, ch in enumerate(matched_chs)}
                data = np.full(len(matched_chs), np.nan)
                for ch, val in zip(ch_order, values):
                    if ch in ch_idx:
                        data[ch_idx[ch]] = val

                # Mask NaN channels
                mask = ~np.isnan(data)
                if mask.sum() < 5:
                    ax.set_title(f'{pos}\n(insufficient data)')
                    ax.axis('off')
                    continue

                # For topomap, fill NaN with 0 for display
                data_plot = np.where(np.isnan(data), 0, data)
                vlim = max(abs(np.nanmin(data[mask])), abs(np.nanmax(data[mask])))
                vlim = max(vlim, 10)  # minimum scale

                mne.viz.plot_topomap(
                    data_plot, info, axes=ax, show=False,
                    cmap='RdBu_r', vlim=(-vlim, vlim),
                    contours=0)
                ax.set_title(f'{pos}')

            if has_data:
                fig.suptitle(f'{ds_name.upper()} — {band_name} enrichment (%)', fontsize=12, y=1.02)
                fig.tight_layout()
                fname = os.path.join(OUT_DIR, f'topomap_{ds_name}_{band_name}.png')
                fig.savefig(fname, dpi=150, bbox_inches='tight')
                print(f"    Saved: {fname}")
            plt.close(fig)

    print("  Done.")


# =========================================================================
# STEP 4: REGIONAL EC vs EO
# =========================================================================

def run_eceo():
    print("\n" + "=" * 70)
    print("  STEP 4: Regional EC vs EO Comparison")
    print("=" * 70)

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []

    for ds_name, ec_subdir, eo_subdir in [
        ('lemon', 'lemon', 'lemon_EO'),
        ('dortmund', 'dortmund', 'dortmund_EO_pre'),
    ]:
        ec_path = os.path.join(PEAK_BASE, ec_subdir)
        eo_path = os.path.join(PEAK_BASE, eo_subdir)
        ec_peaks, n_ec = load_peaks_with_channels(ec_path)
        eo_peaks, n_eo = load_peaks_with_channels(eo_path)
        if ec_peaks is None or eo_peaks is None:
            print(f"  {ds_name}: missing EC or EO data")
            continue

        print(f"  {ds_name}: EC={n_ec} subjects, EO={n_eo} subjects")

        for condition, peaks in [('EC', ec_peaks), ('EO', eo_peaks)]:
            peaks['roi'] = peaks['channel'].apply(lambda c: channel_to_roi(c, ds_name))
            peaks_roi = peaks[peaks['roi'].notna()]
            for band_octave, band_name in OCTAVE_BAND.items():
                band_peaks = peaks_roi[peaks_roi.phi_octave == band_octave]
                for roi in ROI_ORDER:
                    roi_freqs = band_peaks[band_peaks.roi == roi]['freq'].values
                    enr = compute_enrichment(roi_freqs)
                    if enr is None:
                        continue
                    row = {'dataset': ds_name, 'condition': condition,
                           'band': band_name, 'roi': roi,
                           'n_peaks': enr.pop('n_peaks')}
                    row.update(enr)
                    rows.append(row)

    if not rows:
        print("  No EC/EO data!")
        return

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, 'eceo_regional_enrichment.csv'), index=False)
    print(f"\n  Saved: {OUT_DIR}/eceo_regional_enrichment.csv")

    # Print EC→EO shift per region
    print("\n  EC → EO ENRICHMENT SHIFT (key positions)")
    for ds_name in ['lemon', 'dortmund']:
        dd = df[df.dataset == ds_name]
        if dd.empty:
            continue
        print(f"\n  {ds_name.upper()}")
        for band in BAND_ORDER:
            bd = dd[dd.band == band]
            if bd.empty:
                continue
            has_shift = False
            for pos in ['noble_1', 'attractor', 'boundary']:
                shifts = []
                for roi in ROI_ORDER:
                    ec_val = bd[(bd.condition == 'EC') & (bd.roi == roi)]
                    eo_val = bd[(bd.condition == 'EO') & (bd.roi == roi)]
                    if not ec_val.empty and not eo_val.empty:
                        shift = eo_val.iloc[0][pos] - ec_val.iloc[0][pos]
                        shifts.append((roi, shift))
                if shifts:
                    if not has_shift:
                        print(f"\n    {band}:")
                        has_shift = True
                    line = f"      {pos:<14s} "
                    for roi, s in shifts:
                        line += f" {roi[:5]:>5s}:{s:>+5.0f}"
                    print(line)


# =========================================================================
# STEP 5: REGIONAL AGE TRAJECTORIES
# =========================================================================

def run_age():
    print("\n" + "=" * 70)
    print("  STEP 5: Regional Age Trajectories")
    print("=" * 70)

    os.makedirs(OUT_DIR, exist_ok=True)
    all_rows = []

    # HBN developmental (per-subject, per-ROI enrichment × age)
    hbn_releases = ['hbn_R1', 'hbn_R2', 'hbn_R3', 'hbn_R4', 'hbn_R5', 'hbn_R6',
                     'hbn_R7', 'hbn_R8', 'hbn_R9', 'hbn_R10', 'hbn_R11']
    hbn_demo_path = '/Volumes/T9/hbn_data'

    for release in hbn_releases:
        path = os.path.join(PEAK_BASE, release)
        peaks, n_sub = load_peaks_with_channels(path)
        if peaks is None:
            continue

        peaks['roi'] = peaks['channel'].apply(lambda c: channel_to_roi(c, release))
        peaks_roi = peaks[peaks['roi'].notna()]

        # Load demographics (path: cmi_bids_R1/participants.tsv)
        release_num = release.split('_')[1]
        demo_file = os.path.join(hbn_demo_path, f'cmi_bids_{release_num}', 'participants.tsv')
        if not os.path.exists(demo_file):
            print(f"  {release}: no participants.tsv at {demo_file}")
            continue

        demo = pd.read_csv(demo_file, sep='\t')
        demo['subject'] = demo['participant_id']  # keep sub- prefix to match peaks
        demo = demo[['subject', 'age']].dropna()

        subjects = sorted(peaks_roi['subject'].unique())
        print(f"  {release}: {len(subjects)} subjects, computing per-subject per-ROI enrichment...")

        for sub in subjects:
            sub_peaks = peaks_roi[peaks_roi.subject == sub]
            age_row = demo[demo.subject == sub]
            if age_row.empty:
                continue
            age = age_row.iloc[0]['age']

            for roi in ROI_ORDER:
                roi_peaks = sub_peaks[sub_peaks.roi == roi]
                for band_octave, band_name in OCTAVE_BAND.items():
                    bp = roi_peaks[roi_peaks.phi_octave == band_octave]
                    enr = compute_enrichment(bp['freq'].values)
                    if enr is None:
                        continue
                    row = {'dataset': release, 'subject': sub, 'age': age,
                           'roi': roi, 'band': band_name,
                           'n_peaks': enr.pop('n_peaks')}
                    row.update(enr)
                    all_rows.append(row)

    # Dortmund adult aging
    dort_path = os.path.join(PEAK_BASE, 'dortmund')
    dort_peaks, n_dort = load_peaks_with_channels(dort_path)
    dort_demo_path = '/Volumes/T9/dortmund_data/participants.tsv'

    if dort_peaks is not None and os.path.exists(dort_demo_path):
        dort_peaks['roi'] = dort_peaks['channel'].apply(lambda c: channel_to_roi(c, 'dortmund'))
        dort_peaks_roi = dort_peaks[dort_peaks['roi'].notna()]
        demo = pd.read_csv(dort_demo_path, sep='\t')
        demo['subject'] = demo['participant_id']  # keep sub- prefix to match peaks
        demo = demo[['subject', 'age']].dropna()

        subjects = sorted(dort_peaks_roi['subject'].unique())
        print(f"  dortmund: {len(subjects)} subjects, computing per-subject per-ROI enrichment...")

        for sub in subjects:
            sub_peaks = dort_peaks_roi[dort_peaks_roi.subject == sub]
            age_row = demo[demo.subject == sub]
            if age_row.empty:
                continue
            age = age_row.iloc[0]['age']

            for roi in ROI_ORDER:
                roi_peaks = sub_peaks[sub_peaks.roi == roi]
                for band_octave, band_name in OCTAVE_BAND.items():
                    bp = roi_peaks[roi_peaks.phi_octave == band_octave]
                    enr = compute_enrichment(bp['freq'].values)
                    if enr is None:
                        continue
                    row = {'dataset': 'dortmund', 'subject': sub, 'age': age,
                           'roi': roi, 'band': band_name,
                           'n_peaks': enr.pop('n_peaks')}
                    row.update(enr)
                    all_rows.append(row)

    if not all_rows:
        print("  No age data!")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(OUT_DIR, 'age_regional_enrichment.csv'), index=False)
    print(f"\n  Saved: {OUT_DIR}/age_regional_enrichment.csv")

    # Spearman age correlations per ROI × band × position
    print("\n  AGE × ENRICHMENT CORRELATIONS BY REGION (Spearman, FDR<0.05)")
    sig_results = []
    for dataset in df['dataset'].unique():
        dd = df[df.dataset == dataset]
        for band in BAND_ORDER:
            bd = dd[dd.band == band]
            for roi in ROI_ORDER:
                rd = bd[bd.roi == roi]
                if len(rd) < 20:
                    continue
                for pos in POS_NAMES:
                    vals = rd[[pos, 'age']].dropna()
                    if len(vals) < 20:
                        continue
                    rho, p = stats.spearmanr(vals['age'], vals[pos])
                    sig_results.append({
                        'dataset': dataset, 'band': band, 'roi': roi,
                        'position': pos, 'rho': rho, 'p': p, 'n': len(vals)})

    if sig_results:
        sig_df = pd.DataFrame(sig_results)
        # FDR correction
        from statsmodels.stats.multitest import multipletests
        reject, pvals_corrected, _, _ = multipletests(sig_df['p'].values, method='fdr_bh')
        sig_df['p_fdr'] = pvals_corrected
        sig_df['significant'] = reject

        sig_df.to_csv(os.path.join(OUT_DIR, 'age_regional_correlations.csv'), index=False)

        n_sig = sig_df['significant'].sum()
        print(f"  {n_sig} FDR-significant out of {len(sig_df)} tests")

        if n_sig > 0:
            top = sig_df[sig_df.significant].sort_values('p_fdr').head(20)
            for _, r in top.iterrows():
                print(f"    {r['dataset']:>8s} {r['band']:>10s} {r['roi']:>10s} "
                      f"{r['position']:<14s} rho={r['rho']:+.3f} p_fdr={r['p_fdr']:.4f} n={r['n']}")

        # Regional specificity: which ROIs have the most significant age effects?
        print("\n  REGIONAL SPECIFICITY: FDR-significant age effects by ROI")
        for roi in ROI_ORDER:
            n_roi = sig_df[(sig_df.roi == roi) & sig_df.significant].shape[0]
            n_total = sig_df[sig_df.roi == roi].shape[0]
            print(f"    {roi:<12s}: {n_roi}/{n_total} significant")
    else:
        print("  No correlations computed")


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description='Regional Enrichment Analysis')
    parser.add_argument('--step', default='all',
                        choices=['all', 'channel', 'region', 'topomap', 'eceo', 'age'],
                        help='Which analysis step to run')
    args = parser.parse_args()

    steps = {
        'channel': run_channel_enrichment,
        'region': run_region_enrichment,
        'topomap': run_topomap,
        'eceo': run_eceo,
        'age': run_age,
    }

    if args.step == 'all':
        for name, func in steps.items():
            func()
    else:
        steps[args.step]()

    print("\n  All requested steps complete.")


if __name__ == '__main__':
    main()
