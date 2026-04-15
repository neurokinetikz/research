"""
MNE-to-Ignition Bridge Layer
=============================
Adapts MNE Raw objects (from dataset loaders) to the pandas DataFrame format
that detect_ignitions_session() expects, and provides batch-mode wrappers.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import mne

from lib.detect_ignition import detect_ignitions_session

# Standard 10-20 subset (19 channels) present across all research datasets.
# Used to ensure cross-dataset comparability.
STANDARD_1020 = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T7', 'C3', 'Cz', 'C4', 'T8',
    'P7', 'P3', 'Pz', 'P4', 'P8',
    'O1', 'O2',
]

# Some datasets use older 10-20 names; map to modern equivalents
CHANNEL_ALIASES = {
    'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8',
}

# Detection parameters matching the notebook exactly
LABELS      = ['sr1', 'sr1.5', 'sr2', 'sr2o', 'sr2.5', 'sr3', 'sr4', 'sr5', 'sr6']
CANON       = [7.6,   10,      12,    13.75,  15.5,     20,    25,    32,    40]
HALF_BW     = [0.6,   0.618,   0.7,   0.75,   0.8,      1,     2,     2.5,   3]
FREQ_RANGES = [[5,15],[5,15],[7,17],[12,22],[12,22],[15,25],[20,30],[30,40],[35,45]]


def mne_raw_to_ignition_df(raw: mne.io.BaseRaw,
                           channel_subset: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Convert MNE Raw to DataFrame format expected by detect_ignitions_session().

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Loaded and preprocessed MNE Raw object.
    channel_subset : list of str, optional
        Channel names to include. If None, uses STANDARD_1020 subset
        (intersected with available channels). Pass 'all' for all channels.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with 'Timestamp' column and 'EEG.X' prefixed channel columns.
    eeg_channels : list of str
        List of 'EEG.X' channel names used.
    """
    ch_names = raw.ch_names

    # Build alias map for this dataset
    alias_map = {}
    for old, new in CHANNEL_ALIASES.items():
        if old in ch_names and new not in ch_names:
            alias_map[old] = new

    if channel_subset is None:
        # Use standard 10-20 subset
        keep = []
        for ch in STANDARD_1020:
            if ch in ch_names:
                keep.append(ch)
            else:
                # Check aliases
                for old, new in alias_map.items():
                    if new == ch and old in ch_names:
                        keep.append(old)
                        break
        if len(keep) < 5:
            # Fall back to all EEG channels if too few match
            keep = ch_names
    elif channel_subset == 'all':
        keep = ch_names
    else:
        keep = [ch for ch in channel_subset if ch in ch_names]

    # Extract data
    raw_subset = raw.copy().pick(keep)
    data = raw_subset.get_data()  # (n_channels, n_samples), in Volts
    times = raw_subset.times

    # Build DataFrame with EEG.X naming convention (use dict + concat to avoid fragmentation warning)
    cols = {'Timestamp': times}
    eeg_channels = []
    for i, ch in enumerate(raw_subset.ch_names):
        display_name = alias_map.get(ch, ch)
        col_name = f'EEG.{display_name}'
        cols[col_name] = data[i] * 1e6  # Volts to µV
        eeg_channels.append(col_name)
    df = pd.DataFrame(cols)

    return df, eeg_channels


def detect_ignitions_mne(raw: mne.io.BaseRaw,
                         session_name: str = 'session',
                         out_dir: str = '/tmp/sie_scratch',
                         channel_subset: list[str] | None = None,
                         **kwargs) -> tuple[dict, list]:
    """Run SIE detection on an MNE Raw object.

    Wraps detect_ignitions_session() with the notebook's exact parameters.

    Returns
    -------
    result : dict
        The result dict from detect_ignitions_session() containing 'events',
        'summary', etc.
    ign_windows : list of tuple
        Ignition window time intervals.
    """
    df, eeg_channels = mne_raw_to_ignition_df(raw, channel_subset=channel_subset)

    # Merge notebook defaults with any overrides
    params = dict(
        eeg_channels=eeg_channels,
        center_hz=CANON[0],
        harmonics_hz=CANON,
        half_bw_hz=HALF_BW,
        labels=LABELS,
        smooth_sec=0.003,  # 0.01 at 128Hz gives n_smooth=1 (no-op); at 250Hz it gives n_smooth=2 and np.hanning(2)=[0,0] -> NaN. Use 0.003 to stay at n_smooth=1 at any fs.
        z_thresh=3,
        min_isi_sec=2.0,
        window_sec=20,
        merge_gap_sec=10,
        R_band=(CANON[0] - HALF_BW[0], CANON[0] + HALF_BW[0]),
        sr_reference='auto-SSD',
        seed_method='latency',
        pel_band=(32, 64),
        harmonic_method='fooof_hybrid',
        nperseg_sec=100.0,
        fooof_freq_ranges=FREQ_RANGES,
        fooof_max_n_peaks=10,
        fooof_peak_threshold=0.01,
        fooof_min_peak_height=0.01,
        fooof_peak_width_limits=(0.1, 4),
        fooof_match_method='power',
        make_passport=False,
        show=False,
        verbose=False,
        session_name=session_name,
        out_dir=out_dir,
    )
    params.update(kwargs)

    result, ign_windows = detect_ignitions_session(df, **params)
    return result, ign_windows


def summarize_session(result: dict, sub_id: str, dataset: str,
                      condition: str, n_channels: int, fs: float,
                      duration_sec: float) -> dict:
    """Compress detection result into a flat per-subject summary row.

    Parameters
    ----------
    result : dict
        Output from detect_ignitions_session().
    sub_id, dataset, condition : str
        Metadata fields.
    n_channels : int
        Number of EEG channels used.
    fs : float
        Sampling rate in Hz.
    duration_sec : float
        Total recording duration in seconds.

    Returns
    -------
    summary : dict
        Flat dict of scalar metrics suitable for CSV export.
    """
    events = result.get('events', pd.DataFrame())
    n_events = len(events) if events is not None and not events.empty else 0

    row = {
        'subject_id': sub_id,
        'dataset': dataset,
        'condition': condition,
        'n_events': n_events,
        'event_rate_per_min': n_events / max(duration_sec / 60, 1e-9),
        'recording_duration_sec': round(duration_sec, 1),
        'n_channels': n_channels,
        'fs_hz': fs,
    }

    if n_events == 0:
        # Fill metrics with NaN
        for key in ['coverage_pct', 'median_duration_s', 'median_sr1_z_max',
                     'median_HSI', 'median_zR_max', 'median_sr_score',
                     'median_sr_score_canonical', 'mean_base_est_hz',
                     'std_base_est_hz', 'dominant_seed_roi']:
            row[key] = np.nan
        row['dominant_seed_roi'] = ''
        return row

    # Numeric helper
    def _med(col):
        if col in events.columns:
            vals = pd.to_numeric(events[col], errors='coerce')
            return float(vals.median()) if vals.notna().any() else np.nan
        return np.nan

    def _mean(col):
        if col in events.columns:
            vals = pd.to_numeric(events[col], errors='coerce')
            return float(vals.mean()) if vals.notna().any() else np.nan
        return np.nan

    def _std(col):
        if col in events.columns:
            vals = pd.to_numeric(events[col], errors='coerce')
            return float(vals.std()) if vals.notna().any() else np.nan
        return np.nan

    row['coverage_pct'] = float(result.get('summary', {}).get('coverage_pct', np.nan))
    row['median_duration_s'] = _med('duration_s')
    row['median_sr1_z_max'] = _med('sr1_z_max')
    row['median_HSI'] = _med('HSI')
    row['median_zR_max'] = _med('zR_max')
    row['median_sr_score'] = _med('sr_score')
    row['median_sr_score_canonical'] = _med('sr_score_canonical')
    row['mean_base_est_hz'] = _mean('base_est_hz')
    row['std_base_est_hz'] = _std('base_est_hz')

    # Dominant seed ROI
    if 'seed_roi' in events.columns:
        mode = events['seed_roi'].mode()
        row['dominant_seed_roi'] = str(mode.iloc[0]) if len(mode) > 0 else ''
    else:
        row['dominant_seed_roi'] = ''

    return row
