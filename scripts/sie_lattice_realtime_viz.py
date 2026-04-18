#!/usr/bin/env python3
"""
SIE Lattice Real-Time Visualization Prototype
=============================================

Per-channel PSD grid with Schumann / φ-lattice reference lines, built for
comparing spectral peaks across time windows (pre / ignition / post) or
across animated sliding windows. Matches the visual style of per-channel
theta-band PSD plots with a vertical lattice line.

Two output modes:

    --mode static  (default)
        One figure per event. Per-channel grid of band-focused PSDs with
        three overlaid traces: pre (30-10 s before t₀), ignition (±5 s
        around t₀), post (10-30 s after t₀). Red vertical line at lattice
        position inside the band.

    --mode animation
        One MP4 per event. Same per-channel grid, but the PSD traces
        scroll through sliding windows across the recording. Scrubs in
        ~real-time so you can watch peaks migrate toward lattice lines.

Usage:
    # Static per-event overlays for theta band
    python scripts/sie_lattice_realtime_viz.py \\
        --dataset lemon --subject sub-010004 --condition EC \\
        --band theta --mode static

    # Animation (requires ffmpeg)
    python scripts/sie_lattice_realtime_viz.py \\
        --dataset lemon --subject sub-010004 --condition EC \\
        --band theta --mode animation --out outputs/sie_anim.mp4
"""

from __future__ import annotations
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import welch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.run_sie_extraction import (
    load_eegmmidb, load_lemon, load_dortmund, load_chbmp,
)
from lib.mne_to_ignition import CANON, LABELS

warnings.filterwarnings('ignore')
import mne
from mne.time_frequency import psd_array_multitaper
mne.set_log_level('ERROR')

# =========================================================================
# LATTICE POSITIONS
# =========================================================================
# True Schumann Resonance harmonics (Earth-ionosphere cavity)
SCHUMANN_HZ = {
    'SR1': 7.83, 'SR2': 14.3, 'SR3': 20.8, 'SR4': 27.3, 'SR5': 33.8, 'SR6': 39.0,
}

# Internal CANON grid (mix of Schumann + interstitial φⁿ positions)
CANON_BY_LABEL = dict(zip([l.upper() for l in LABELS], CANON))

# Band definitions — (fmin, fmax, lattice positions inside, primary Schumann label)
BANDS = {
    'theta':       (4.0,  9.0,  [('SR1', 7.83), ('sr1_canon', 7.6)]),
    'alpha':       (8.0, 14.0,  [('SR1.5', 10.0), ('SR2', 12.0)]),
    'theta_alpha': (4.0, 13.0,  [('SR1', 7.83)]),
    'schumann2':   (12.0, 16.0, [('SR2o_canon', 13.75), ('SR2', 14.3), ('SR2.5', 15.5)]),
    'beta_low':    (16.0, 24.0, [('SR3_canon', 20.0), ('SR3', 20.8)]),
    'beta_mid':    (22.0, 32.0, [('SR4_canon', 25.0), ('SR4', 27.3), ('SR5_canon', 32.0)]),
    'gamma':   (30.0, 42.0, [('SR5', 33.8), ('SR6_canon', 40.0), ('SR6', 39.0)]),
}

# ============================================================================
# NOBLE / INVERSE-NOBLE φ-LATTICE POSITIONS
# ============================================================================
# u-space offsets (peak position within a φ-octave, in [0, 1)).
# noble_k = φ⁻ᵏ; inv_noble_k = 1 − φ⁻ᵏ (mirror around attractor at 0.5)
# Same-degree noble and inv_noble share a color so visual pairs are obvious.
PHI_FLOAT  = (1 + 5 ** 0.5) / 2
PHI_INV    = 1 / PHI_FLOAT
F0_LATTICE = 7.6  # matches the SIE pipeline

POSITION_OFFSETS = {
    'boundary':    0.0,
    'noble_5':     PHI_INV ** 5,      'inv_noble_5': 1 - PHI_INV ** 5,
    'noble_4':     PHI_INV ** 4,      'inv_noble_4': 1 - PHI_INV ** 4,
    'noble_3':     PHI_INV ** 3,      'inv_noble_3': 1 - PHI_INV ** 3,
    'inv_noble_1': PHI_INV ** 2,  # = 1 − PHI_INV (mirror of noble_1 about attractor)
    'noble_1':     PHI_INV,
    'attractor':   0.5,
}
DEGREE_MAP = {
    'boundary': 0,  'attractor': -1,
    'noble_1': 1,   'inv_noble_1': 1,
    'noble_3': 3,   'inv_noble_3': 3,
    'noble_4': 4,   'inv_noble_4': 4,
    'noble_5': 5,   'inv_noble_5': 5,
    'noble_6': 6,   'inv_noble_6': 6,
}
DEGREE_COLORS = {
    0:  '#000000',  # boundary — black (always partially lit; darkens further on peak)
    -1: '#228b22',  # attractor — forest green
    1:  '#d4af37',  # gold
    3:  '#4682b4',  # steel blue
    4:  '#6a0dad',  # purple (swapped from attractor)
    5:  '#ff8c00',  # orange
    6:  '#ff1493',  # deep pink
}

# Per-position alpha (base / lit / max). Boundary positions are always
# partially visible and darken further when a supra-threshold peak lands
# in them, but with a lower ceiling than other bins so they don't dominate.
BASE_ALPHA_DEFAULT  = 0.08
LIT_ALPHA_DEFAULT   = 0.50
MAX_ALPHA_DEFAULT   = 0.90
BASE_ALPHA_BOUNDARY = 0.22
LIT_ALPHA_BOUNDARY  = 0.35
MAX_ALPHA_BOUNDARY  = 0.50


def _voronoi_edges_u():
    """Voronoi edges in u-space (circular midpoints between adjacent positions).

    Returns
    -------
    sorted_items : list of (label, u) in ascending u order
    left_edge_u  : list of left Voronoi edges (in u-space; can be < 0 for boundary)
    right_edge_u : list of right Voronoi edges (can be > 1)
    """
    items = sorted(POSITION_OFFSETS.items(), key=lambda x: x[1])
    u_vals = np.array([u for _, u in items])
    prev_u = np.roll(u_vals, 1).copy()
    prev_u[0] -= 1.0          # circular wrap on the left
    next_u = np.roll(u_vals, -1).copy()
    next_u[-1] += 1.0         # circular wrap on the right
    left_edge  = (prev_u + u_vals) / 2.0
    right_edge = (u_vals + next_u) / 2.0
    return items, left_edge, right_edge


def positions_in_band(fmin: float, fmax: float, f0: float = F0_LATTICE):
    """All noble/inv-noble Voronoi bins overlapping [fmin, fmax].

    Each bin extends from the midpoint to the previous position in u-space
    to the midpoint to the next position (circular). Bins tile the lattice
    without gaps or overlap, which matches how the enrichment pipeline
    assigns peaks to positions (nearest position in u-space).

    Returns list of dicts: {label, degree, f_center, f_low, f_high}.
    """
    items, left_edge, right_edge = _voronoi_edges_u()
    out = []
    oct_lo = int(np.floor(np.log(fmin / f0) / np.log(PHI_FLOAT))) - 1
    oct_hi = int(np.ceil(np.log(fmax / f0) / np.log(PHI_FLOAT))) + 1
    for k in range(oct_lo, oct_hi + 1):
        for (lbl, u), le, re in zip(items, left_edge, right_edge):
            u_ctr = k + u
            u_lo  = k + le
            u_hi  = k + re
            f_ctr = f0 * PHI_FLOAT ** u_ctr
            f_lo  = f0 * PHI_FLOAT ** u_lo
            f_hi  = f0 * PHI_FLOAT ** u_hi
            if f_hi < fmin or f_lo > fmax:
                continue
            out.append({'label': lbl, 'degree': DEGREE_MAP[lbl],
                        'f_center': f_ctr, 'f_low': f_lo, 'f_high': f_hi})
    return out


def window_power_z(psd: np.ndarray, freqs: np.ndarray,
                   f_low: float, f_high: float) -> float:
    """Peak power inside window expressed as MAD-SDs above local baseline."""
    in_win = (freqs >= f_low) & (freqs <= f_high)
    out_win = ~in_win
    if not in_win.any() or not out_win.any():
        return 0.0
    peak = float(psd[in_win].max())
    base_med = float(np.median(psd[out_win]))
    base_mad = float(np.median(np.abs(psd[out_win] - base_med)))
    base_sd = base_mad * 1.4826 + 1e-12
    return (peak - base_med) / base_sd

# Frontal / midline channel preference order (matches the reference images)
PREFERRED_CHANNELS = [
    'AF3', 'AF4', 'F7', 'F8', 'F3', 'F4',
    'Fz', 'C3', 'C4', 'Cz', 'P3', 'P4',
]


def load_recording(dataset: str, subject: str, condition: str = 'EC'):
    if dataset == 'lemon':
        return load_lemon(subject, condition=condition)
    if dataset == 'eegmmidb':
        sub_num = int(subject.replace('S', '').replace('sub-', ''))
        return load_eegmmidb(sub_num)
    if dataset == 'dortmund':
        task_map = {'EC-pre': ('EyesClosed', 'pre'), 'EO-pre': ('EyesOpen', 'pre'),
                    'EC-post': ('EyesClosed', 'post'), 'EO-post': ('EyesOpen', 'post')}
        task, acq = task_map.get(condition, ('EyesClosed', 'pre'))
        return load_dortmund(subject, task=task, acq=acq)
    if dataset == 'chbmp':
        return load_chbmp(subject)
    raise ValueError(f'Unknown dataset: {dataset}')


def pick_channels(raw, n_max: int = 6) -> list[str]:
    """Pick up to n_max channels in preferred order from what's available."""
    avail = raw.ch_names
    picked = [ch for ch in PREFERRED_CHANNELS if ch in avail]
    if len(picked) < n_max:
        for ch in avail:
            if ch not in picked:
                picked.append(ch)
            if len(picked) >= n_max:
                break
    return picked[:n_max]


def window_psd(data: np.ndarray, fs: float, band: tuple[float, float],
               bandwidth: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Multitaper PSD for a single-channel window, restricted to band.

    bandwidth : full-width in Hz of the multitaper smoothing kernel.
        Smaller = sharper peaks, higher variance. 0.5 Hz gives ~0.25 Hz
        half-bandwidth, tighter than any Welch choice with segments.

    Frequency grid is computed via zero-padding to nfft = next power of 2
    above len(data), which gives a smoother interpolated curve (visual
    only — does not add real resolution).
    """
    n = len(data)
    nfft = int(2 ** np.ceil(np.log2(max(n, 256))) * 2)  # zero-pad ×2 for smooth curves
    psd, f = psd_array_multitaper(
        data[np.newaxis, :], sfreq=fs, fmin=band[0], fmax=band[1],
        bandwidth=bandwidth, adaptive=False, normalization='full',
        n_jobs=1, verbose=False, low_bias=True,
    )
    m = (f >= band[0]) & (f <= band[1])
    return f[m], psd[0][m]


def window_psd_welch(data: np.ndarray, fs: float, band: tuple[float, float],
                     nperseg_sec: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD (fallback/coarser). Kept for comparison."""
    nperseg = max(min(len(data), int(nperseg_sec * fs)), 32)
    f, p = welch(data, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    m = (f >= band[0]) & (f <= band[1])
    return f[m], p[m]


def plot_event_static(raw, ev, channels, band_key: str, out_path: str,
                      subject: str, condition: str, event_num: int):
    """Static per-event overlay: pre (gray), ignition (red), post (blue)."""
    fs = raw.info['sfreq']
    fmin, fmax, lattice = BANDS[band_key]
    t0 = float(ev['t0_net'])

    # Window boundaries (seconds)
    pre_lo, pre_hi = t0 - 30, t0 - 10
    ign_lo, ign_hi = t0 - 5,  t0 + 5
    post_lo, post_hi = t0 + 10, t0 + 30

    # Clip to recording duration
    total_s = raw.n_times / fs
    if pre_lo < 0 or post_hi > total_s:
        print(f'  Event {event_num} near recording edge, skipping')
        return False

    def get_segment(lo, hi):
        s0 = int(lo * fs)
        s1 = int(hi * fs)
        return raw.get_data(picks=channels)[:, s0:s1] * 1e6

    seg_pre = get_segment(pre_lo, pre_hi)
    seg_ign = get_segment(ign_lo, ign_hi)
    seg_post = get_segment(post_lo, post_hi)

    n_ch = len(channels)
    ncols = 2
    nrows = (n_ch + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 2.6 * nrows), sharex=True)
    axes = np.atleast_2d(axes).flatten()

    for i, ch in enumerate(channels):
        ax = axes[i]
        ch_idx = channels.index(ch)
        f_p, p_p = window_psd(seg_pre[ch_idx], fs, (fmin, fmax))
        f_i, p_i = window_psd(seg_ign[ch_idx], fs, (fmin, fmax))
        f_o, p_o = window_psd(seg_post[ch_idx], fs, (fmin, fmax))

        ax.plot(f_p, p_p, color='gray',     lw=1.2, alpha=0.75, label='pre (−30→−10 s)')
        ax.plot(f_i, p_i, color='crimson',  lw=1.8, alpha=0.95, label='ignition (±5 s)')
        ax.plot(f_o, p_o, color='steelblue',lw=1.2, alpha=0.75, label='post (+10→+30 s)')

        # Lattice reference lines
        for lbl, f_lat in lattice:
            if fmin <= f_lat <= fmax:
                is_schumann = not lbl.endswith('_canon')
                ax.axvline(f_lat, color='red' if is_schumann else 'cyan',
                           lw=1.0 if is_schumann else 0.8,
                           ls='-' if is_schumann else ':',
                           alpha=0.85 if is_schumann else 0.65,
                           zorder=5)
                ax.text(f_lat, ax.get_ylim()[1] * 0.98, lbl.replace('_canon', ''),
                        rotation=90, fontsize=6.5, color='red' if is_schumann else 'darkcyan',
                        va='top', ha='right', alpha=0.9)

        ax.set_title(f'{ch} — {band_key}', fontsize=10)
        ax.set_xlabel('Frequency (Hz)', fontsize=9)
        ax.set_ylabel('PSD (µV²/Hz)', fontsize=9)
        ax.grid(alpha=0.2)
        if i == 0:
            ax.legend(fontsize=7, framealpha=0.9, loc='upper right')

    # Hide any unused axes
    for j in range(n_ch, len(axes)):
        axes[j].axis('off')

    # Title with event info
    det_harmonics = ', '.join(f'{l.upper()}={ev[l]:.2f}' for l in LABELS
                              if pd.notna(ev.get(l)) and fmin <= float(ev[l]) <= fmax)
    fig.suptitle(f'{subject} ({condition}) — Event {event_num} at t₀={t0:.1f}s, {band_key} band\n'
                 f'detected in band: {det_harmonics or "(no detected harmonics in band)"}',
                 fontsize=11, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')
    return True


def plot_animation(raw, events_df: pd.DataFrame, channels, band_key: str,
                   out_path: str, subject: str, condition: str,
                   window_sec: float = 4.0, hop_sec: float = 0.25, fps: int = 8):
    """Animated sliding-window PSD grid."""
    try:
        import matplotlib
        matplotlib.rcParams['animation.writer'] = 'ffmpeg'
    except Exception:
        pass

    fs = raw.info['sfreq']
    fmin, fmax, lattice = BANDS[band_key]
    n_win = int(window_sec * fs)
    n_hop = int(hop_sec * fs)
    data = raw.get_data(picks=channels) * 1e6
    n_samples = data.shape[1]
    starts = np.arange(0, n_samples - n_win + 1, n_hop)
    win_times = (starts + n_win / 2) / fs

    # Precompute all PSDs so animation playback is fast
    print(f'  Precomputing {len(starts)} × {len(channels)} PSDs...')
    all_psds = []
    freqs = None
    for s in starts:
        ch_psds = []
        for ch_idx in range(len(channels)):
            f, p = window_psd(data[ch_idx, s:s + n_win], fs, (fmin, fmax))
            if freqs is None:
                freqs = f
            ch_psds.append(p)
        all_psds.append(np.array(ch_psds))
    all_psds = np.array(all_psds)  # (T, n_ch, F)
    # Per-channel y-axis max so ignition spikes aren't cropped
    ymax_per_ch = all_psds.max(axis=(0, 2)) * 1.05

    from matplotlib.ticker import MultipleLocator, AutoMinorLocator
    band_span = fmax - fmin
    major_step = 0.5 if band_span <= 6 else (1.0 if band_span <= 12 else 2.0)
    minor_div = 5  # minor ticks between majors

    n_ch = len(channels)
    ncols = 2
    nrows = (n_ch + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.8 * nrows),
                             sharex=True, constrained_layout=False)
    axes = np.atleast_2d(axes).flatten()

    # Precompute noble / inv-noble position windows inside the band
    positions = positions_in_band(fmin, fmax)
    print(f'  Partitioning band into {len(positions)} noble/inv-noble windows')

    lines = []
    spans = []   # spans[channel_idx] = list of axvspan patches aligned with `positions`
    centers = [] # centers[channel_idx] = list of axvline handles (thin, always visible)

    def alphas_for(p):
        if p['label'] == 'boundary':
            return BASE_ALPHA_BOUNDARY, LIT_ALPHA_BOUNDARY, MAX_ALPHA_BOUNDARY
        return BASE_ALPHA_DEFAULT, LIT_ALPHA_DEFAULT, MAX_ALPHA_DEFAULT

    # Unique Voronoi edges across all positions (for bin separators)
    edge_freqs = sorted({p['f_low'] for p in positions}
                        | {p['f_high'] for p in positions})
    edge_freqs = [f for f in edge_freqs if fmin <= f <= fmax]

    for i, ch in enumerate(channels):
        ax = axes[i]
        # Position shading (one axvspan per Voronoi bin, colored by degree pair)
        ch_spans = []
        ch_centers = []
        for p in positions:
            color = DEGREE_COLORS[p['degree']]
            base_a, _, _ = alphas_for(p)
            sp = ax.axvspan(p['f_low'], p['f_high'], color=color,
                            alpha=base_a, zorder=1)
            ch_spans.append(sp)
            cl = ax.axvline(p['f_center'], color=color, lw=0.6, alpha=0.35,
                            zorder=2)
            ch_centers.append(cl)
        # Bin separators: subtle vertical lines at each Voronoi edge
        for ef in edge_freqs:
            ax.axvline(ef, color='black', lw=0.5, alpha=0.25,
                       ls='-', zorder=3)
        spans.append(ch_spans)
        centers.append(ch_centers)

        # Schumann reference line kept as thick red overlay
        for lbl, f_lat in lattice:
            if fmin <= f_lat <= fmax:
                is_sch = not lbl.endswith('_canon')
                if is_sch:
                    ax.axvline(f_lat, color='red', lw=1.0, ls='-',
                               alpha=0.6, zorder=4)

        ln, = ax.plot(freqs, all_psds[0, i], color='black', lw=1.4, zorder=6)
        ax.set_ylim(0, ymax_per_ch[i])
        ax.set_xlim(fmin, fmax)
        ax.set_title(f'{ch} — {band_key}', fontsize=10, pad=6)
        is_bottom_row = (i // ncols) == nrows - 1 or i + ncols >= n_ch
        if is_bottom_row:
            ax.set_xlabel('Frequency (Hz)', fontsize=9)
        ax.set_ylabel('PSD (µV²/Hz)', fontsize=9)
        # Log-φ x-axis: each φ-octave takes equal display width so theta
        # and alpha octaves are visually comparable.
        ax.set_xscale('function', functions=(
            lambda x: np.log(np.clip(x, 1e-6, None) / F0_LATTICE) / np.log(PHI_FLOAT),
            lambda u: F0_LATTICE * PHI_FLOAT ** u,
        ))
        # Tick labels still in Hz; positions nonuniform on display
        hz_major = np.arange(np.ceil(fmin), np.floor(fmax) + 1, 1.0)
        ax.set_xticks(hz_major)
        ax.set_xticklabels([f'{int(h)}' for h in hz_major])
        ax.grid(which='major', alpha=0.25)
        ax.tick_params(axis='x', which='major', labelsize=8, labelbottom=True)
        lines.append(ln)
    for j in range(n_ch, len(axes)):
        axes[j].axis('off')
    fig.subplots_adjust(hspace=0.55, wspace=0.18, top=0.91, bottom=0.10,
                        left=0.07, right=0.98)

    # Legend: one swatch per degree that appears in the band
    from matplotlib.patches import Patch
    degrees_present = sorted({p['degree'] for p in positions}, key=lambda d: (d == -1, d))
    degree_label = {0: 'boundary', -1: 'attractor (½)', 1: 'k=1 (n₁/inv)',
                    3: 'k=3 (n₃/inv)', 4: 'k=4', 5: 'k=5', 6: 'k=6'}
    handles = [Patch(color=DEGREE_COLORS[d], alpha=0.45, label=degree_label.get(d, f'k={d}'))
               for d in degrees_present]
    handles.append(plt.Line2D([0], [0], color='red', lw=1.2, label='Schumann'))
    fig.legend(handles=handles, loc='lower center', ncol=len(handles),
               fontsize=8, frameon=True, bbox_to_anchor=(0.5, 0.00))

    # Title with progress/time indicator — updated per frame
    title = fig.suptitle('', fontsize=11, y=0.995)

    # Event t0 set for "is active ignition" marker
    event_t0s = events_df['t0_net'].values
    event_ranges = list(zip(events_df['t_start'].values, events_df['t_end'].values))

    def update(frame):
        t = win_times[frame]
        active = any(lo <= t <= hi for lo, hi in event_ranges)
        for i in range(n_ch):
            psd_i = all_psds[frame, i]
            lines[i].set_ydata(psd_i)
            # Alpha varies continuously with peak power (z-score vs local
            # baseline). Below threshold_sd = base alpha; above threshold
            # the alpha ramps from lit_alpha (at threshold) toward max_alpha
            # (at z = 6). Boundary bins use darker baseline + ceiling.
            threshold_sd = 2.5
            z_cap = 6.0
            for p_idx, p in enumerate(positions):
                z = window_power_z(psd_i, freqs, p['f_low'], p['f_high'])
                base_a, lit_a, max_a = alphas_for(p)
                if z < threshold_sd:
                    alpha = base_a
                else:
                    ramp = min((z - threshold_sd) / (z_cap - threshold_sd), 1.0)
                    alpha = lit_a + (max_a - lit_a) * ramp
                spans[i][p_idx].set_alpha(alpha)
                centers[i][p_idx].set_alpha(0.85 if z >= threshold_sd else 0.35)
        status = 'IGNITION' if active else 'baseline'
        title.set_text(f'{subject} ({condition}) — {band_key} band | t={t:.1f}s [{status}]')
        return lines + [title]

    print(f'  Rendering {len(starts)} frames at {fps} fps...')
    anim = FuncAnimation(fig, update, frames=len(starts), interval=1000 / fps, blit=False)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    if out_path.endswith('.mp4'):
        anim.save(out_path, writer='ffmpeg', fps=fps, dpi=100)
    elif out_path.endswith('.gif'):
        anim.save(out_path, writer='pillow', fps=fps, dpi=80)
    else:
        out_path = out_path + '.mp4'
        anim.save(out_path, writer='ffmpeg', fps=fps, dpi=100)
    plt.close(fig)
    print(f'  Saved: {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='lemon')
    ap.add_argument('--subject', default='sub-010004')
    ap.add_argument('--condition', default='EC')
    ap.add_argument('--band', default='theta', choices=list(BANDS.keys()))
    ap.add_argument('--mode', default='static', choices=['static', 'animation'])
    ap.add_argument('--max-events', type=int, default=3,
                    help='Max number of events to render (static mode)')
    ap.add_argument('--n-channels', type=int, default=6)
    ap.add_argument('--events-csv', default=None)
    ap.add_argument('--out', default=None, help='Output path (file or directory)')
    # Animation-specific
    ap.add_argument('--window', type=float, default=8.0,
                    help='Sliding window (s). Longer = sharper frequency resolution '
                         '(8 s → ~0.125 Hz with multitaper bandwidth=0.5).')
    ap.add_argument('--hop', type=float, default=0.5)
    ap.add_argument('--fps', type=int, default=8)
    ap.add_argument('--bandwidth', type=float, default=0.5,
                    help='Multitaper full bandwidth in Hz (default 0.5 = tight peaks)')
    args = ap.parse_args()

    # Find events CSV
    if args.events_csv:
        events_path = args.events_csv
    else:
        suffix = '' if args.condition in ('', 'EC') else f'_{args.condition}'
        events_path = f'exports_sie/{args.dataset}{suffix}/{args.subject}_sie_events.csv'

    if not os.path.exists(events_path):
        print(f'ERROR: events CSV not found: {events_path}')
        sys.exit(1)

    events_df = pd.read_csv(events_path)
    print(f'Loaded {len(events_df)} events from {events_path}')

    print(f'Loading {args.dataset} {args.subject} ({args.condition})...')
    raw = load_recording(args.dataset, args.subject, args.condition)
    if raw is None:
        print('ERROR: raw load failed.')
        sys.exit(1)
    print(f'  {raw.info["sfreq"]} Hz, {len(raw.ch_names)} ch, {raw.n_times / raw.info["sfreq"]:.1f}s')

    channels = pick_channels(raw, n_max=args.n_channels)
    print(f'  Channels: {channels}')

    tag = f'{args.subject}_{args.condition}_{args.band}'

    if args.mode == 'static':
        if len(events_df) == 0:
            print('No events to visualize.')
            sys.exit(0)
        out_dir = args.out or f'outputs/sie_viz_{args.dataset}'
        os.makedirs(out_dir, exist_ok=True) if not out_dir.endswith('.png') else None
        n_to_render = min(len(events_df), args.max_events)
        for i in range(n_to_render):
            ev = events_df.iloc[i]
            if args.out and args.out.endswith('.png') and n_to_render == 1:
                out_path = args.out
            else:
                base = args.out if args.out and not args.out.endswith('.png') else f'outputs/sie_viz_{args.dataset}'
                out_path = f'{base}/{tag}_event{i+1}.png'
            plot_event_static(raw, ev, channels, args.band, out_path,
                              args.subject, args.condition, i + 1)

    elif args.mode == 'animation':
        out_path = args.out or f'outputs/sie_anim_{args.dataset}_{tag}.mp4'
        plot_animation(raw, events_df, channels, args.band, out_path,
                       args.subject, args.condition,
                       window_sec=args.window, hop_sec=args.hop, fps=args.fps)


if __name__ == '__main__':
    main()
