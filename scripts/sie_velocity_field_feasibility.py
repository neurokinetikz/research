#!/usr/bin/env python3
"""
SIE Velocity-Field Feasibility Test (Paper 3 scoping)
======================================================

Prototype of kinematic analysis on φ-lattice Voronoi occupancy:

    sliding-window multitaper PSD
    → per-frame per-position power-weighted occupancy O_p(t)
    → event-triggered averaging within subject (ERP-style)
    → Gaussian smoothing (σ = 3 s)
    → time derivative dŌ_p/dt (velocity field)
    → baseline vs ignition contrast
    → temporal-shuffle null control

Runs on 1-2 LEMON EO subjects with many SIE events. Three-level structure
(single-event qualitative; per-subject averaged statistical; across-subject
population) — at N = 1-2 subjects this is single-subject feasibility only.

Requires T9 data disk mounted at /Volumes/T9.

Outputs:
    outputs/sie_velocity_{subject}.png
        — per-position trajectory Ō_p(t), event-triggered averaged
        — velocity dŌ_p/dt with shuffle null overlay
        — per-position baseline vs ignition contrast

Usage:
    python scripts/sie_velocity_field_feasibility.py \\
        --subject sub-010146 --condition EO
    python scripts/sie_velocity_field_feasibility.py \\
        --subject sub-010050 --condition EO --out outputs/velocity_10050.png
"""

from __future__ import annotations
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.run_sie_extraction import load_lemon
from scripts.sie_lattice_realtime_viz import (
    positions_in_band, window_psd, F0_LATTICE, PHI_FLOAT, DEGREE_COLORS,
    BANDS,
)

warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')


# =========================================================================
# OCCUPANCY TRAJECTORY
# =========================================================================

def compute_occupancy_trajectory(raw: mne.io.BaseRaw,
                                  fmin: float = 4.0, fmax: float = 13.0,
                                  window_sec: float = 8.0, hop_sec: float = 0.5,
                                  bandwidth: float = 0.5,
                                  channel_subset: list[str] | None = None):
    """Sliding-window multitaper PSD → per-position occupancy per frame.

    Returns
    -------
    times : (T,) array of window centers in seconds
    positions : list of position dicts (from positions_in_band)
    occupancy : (n_positions, T) array — mean power in each Voronoi bin per frame
    """
    fs = raw.info['sfreq']
    if channel_subset:
        raw_sub = raw.copy().pick(channel_subset)
    else:
        raw_sub = raw.copy()
    data = raw_sub.get_data() * 1e6  # V to µV
    n_samples = data.shape[1]
    n_win = int(window_sec * fs)
    n_hop = int(hop_sec * fs)
    starts = np.arange(0, n_samples - n_win + 1, n_hop)
    times = (starts + n_win / 2) / fs

    positions = positions_in_band(fmin, fmax)
    n_pos = len(positions)
    # Average across channels per frame (grand-averaged reference)
    data_mean = data.mean(axis=0)

    occupancy = np.zeros((n_pos, len(starts)), dtype=np.float32)
    for j, s in enumerate(starts):
        seg = data_mean[s:s + n_win]
        f, psd = window_psd(seg, fs, (fmin, fmax), bandwidth=bandwidth)
        # Power-weighted occupancy: mean PSD within each Voronoi bin
        for i, p in enumerate(positions):
            mask = (f >= p['f_low']) & (f <= p['f_high'])
            if mask.any():
                occupancy[i, j] = float(psd[mask].mean())
    print(f'  Occupancy trajectory: {n_pos} positions × {len(starts)} frames')
    return times, positions, occupancy


# =========================================================================
# EVENT-TRIGGERED AVERAGING
# =========================================================================

def event_triggered_average(times: np.ndarray, trajectory: np.ndarray,
                             event_t0s: np.ndarray,
                             pre_sec: float = 20.0, post_sec: float = 20.0,
                             hop_sec: float = 0.5):
    """ERP-style event-triggered averaging of trajectory.

    Returns
    -------
    lag_times : (L,) array of lag seconds (− = pre, + = post)
    eta_mean : (n_positions, L) subject-level averaged trajectory
    n_events_used : int (may be < len(event_t0s) due to edge effects)
    """
    n_pos = trajectory.shape[0]
    n_pre = int(pre_sec / hop_sec)
    n_post = int(post_sec / hop_sec)
    lag_len = n_pre + n_post + 1
    lag_times = np.arange(-n_pre, n_post + 1) * hop_sec

    segments = []
    for t0 in event_t0s:
        # Find closest time index to t0
        j0 = int(np.argmin(np.abs(times - t0)))
        lo = j0 - n_pre
        hi = j0 + n_post + 1
        if lo < 0 or hi > trajectory.shape[1]:
            continue
        segments.append(trajectory[:, lo:hi])
    if not segments:
        return lag_times, np.zeros((n_pos, lag_len)), 0
    eta = np.stack(segments, axis=0)  # (n_events, n_pos, lag_len)
    return lag_times, eta.mean(axis=0), len(segments)


# =========================================================================
# VELOCITY ESTIMATION
# =========================================================================

def compute_velocity(eta: np.ndarray, sigma_sec: float, hop_sec: float):
    """Gaussian-smooth each position's trajectory, then differentiate.

    Returns
    -------
    eta_smooth : (n_pos, L) smoothed trajectory
    velocity : (n_pos, L) time derivative (units/s)
    """
    sigma_frames = sigma_sec / hop_sec
    eta_smooth = gaussian_filter1d(eta, sigma=sigma_frames, axis=1, mode='nearest')
    velocity = np.gradient(eta_smooth, hop_sec, axis=1)
    return eta_smooth, velocity


# =========================================================================
# TEMPORAL-SHUFFLE NULL
# =========================================================================

def shuffle_null(trajectory: np.ndarray, event_t0s: np.ndarray, times: np.ndarray,
                 hop_sec: float, sigma_sec: float, n_shuffles: int = 50):
    """Shuffle time order of occupancy vectors; compute velocity from event-
    triggered average; return the distribution of shuffled velocities.

    If baseline (far from t0) velocities in the real data are structured
    but in the shuffled data they look identical to ignition velocities, the
    structure was methodological. If shuffled velocities are noise, real
    structure is dynamical."""
    rng = np.random.default_rng(42)
    n_pos, T = trajectory.shape
    shuffled_velocities = []
    for _ in range(n_shuffles):
        perm = rng.permutation(T)
        shuf_traj = trajectory[:, perm]
        _, eta_s, _ = event_triggered_average(times, shuf_traj, event_t0s,
                                               hop_sec=hop_sec)
        _, v_s = compute_velocity(eta_s, sigma_sec, hop_sec)
        shuffled_velocities.append(v_s)
    return np.stack(shuffled_velocities, axis=0)  # (n_shuffles, n_pos, L)


# =========================================================================
# PLOTTING
# =========================================================================

def plot_feasibility(subject, condition, lag_times, positions,
                     eta_smooth, velocity, shuffled_vel, n_events_used,
                     out_path):
    """4-panel figure: trajectory, velocity, shuffle-null envelope, baseline vs
    ignition contrast."""
    n_pos = len(positions)
    pre_mask = lag_times <= -5
    ign_mask = (lag_times >= -5) & (lag_times <= 5)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax1, ax2, ax3, ax4 = axes.flatten()

    # --- PANEL 1: trajectory (smoothed) ---
    for i, p in enumerate(positions):
        color = DEGREE_COLORS[p['degree']]
        ax1.plot(lag_times, eta_smooth[i], color=color, lw=0.8, alpha=0.7,
                 label=p['label'] if i < 5 else None)
    ax1.axvline(0, color='magenta', lw=1.0, alpha=0.7)
    ax1.set_xlabel('Lag from t₀ (s)')
    ax1.set_ylabel('Occupancy (µV²/Hz)')
    ax1.set_title(f'Event-triggered trajectory\n(N={n_events_used} events, σ={3}s smooth)')
    ax1.grid(alpha=0.2)

    # --- PANEL 2: velocity field ---
    for i, p in enumerate(positions):
        color = DEGREE_COLORS[p['degree']]
        ax2.plot(lag_times, velocity[i], color=color, lw=0.8, alpha=0.7)
    ax2.axvline(0, color='magenta', lw=1.0, alpha=0.7)
    ax2.axhline(0, color='k', lw=0.5, alpha=0.3)
    ax2.set_xlabel('Lag from t₀ (s)')
    ax2.set_ylabel('dO/dt (µV²/Hz · s⁻¹)')
    ax2.set_title('Velocity field')
    ax2.grid(alpha=0.2)

    # --- PANEL 3: shuffle null envelope on velocity ---
    # For each position, plot real velocity against shuffle 5–95% envelope
    pct_lo = np.percentile(shuffled_vel, 5, axis=0)
    pct_hi = np.percentile(shuffled_vel, 95, axis=0)
    # Mean squared velocity magnitude across positions, real vs null
    real_mag = np.sqrt(np.mean(velocity ** 2, axis=0))
    null_mags = np.sqrt(np.mean(shuffled_vel ** 2, axis=1))  # (n_shuffles, L)
    null_mag_lo = np.percentile(null_mags, 5, axis=0)
    null_mag_hi = np.percentile(null_mags, 95, axis=0)
    null_mag_med = np.median(null_mags, axis=0)
    ax3.fill_between(lag_times, null_mag_lo, null_mag_hi, color='gray',
                     alpha=0.3, label='Shuffle null 5-95%')
    ax3.plot(lag_times, null_mag_med, color='gray', lw=1.0, ls='--',
             label='Shuffle null median')
    ax3.plot(lag_times, real_mag, color='crimson', lw=1.5, label='Observed')
    ax3.axvline(0, color='magenta', lw=1.0, alpha=0.7)
    ax3.set_xlabel('Lag from t₀ (s)')
    ax3.set_ylabel('Velocity magnitude (RMS across positions)')
    ax3.set_title('Observed velocity magnitude vs shuffle null')
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.2)

    # --- PANEL 4: baseline vs ignition velocity per position ---
    base_v = np.mean(velocity[:, pre_mask], axis=1)  # per-position mean pre-velocity
    ign_v = np.mean(velocity[:, ign_mask], axis=1)
    # Also shuffle null for comparison
    null_base = np.mean(shuffled_vel[:, :, pre_mask], axis=2)  # (n_shuffles, n_pos)
    null_ign = np.mean(shuffled_vel[:, :, ign_mask], axis=2)
    pos_labels = [p['label'] for p in positions]
    x = np.arange(n_pos)
    # Plot baseline and ignition as two bars
    width = 0.4
    ax4.bar(x - width / 2, base_v, width, label='baseline (t < −5 s)',
            color='steelblue', alpha=0.7)
    ax4.bar(x + width / 2, ign_v, width, label='ignition (±5 s)',
            color='crimson', alpha=0.7)
    # Null bands
    null_base_lo = np.percentile(null_base, 5, axis=0)
    null_base_hi = np.percentile(null_base, 95, axis=0)
    null_ign_lo = np.percentile(null_ign, 5, axis=0)
    null_ign_hi = np.percentile(null_ign, 95, axis=0)
    for i in range(n_pos):
        ax4.plot([x[i] - width / 2] * 2, [null_base_lo[i], null_base_hi[i]],
                 color='k', lw=0.8, alpha=0.5)
        ax4.plot([x[i] + width / 2] * 2, [null_ign_lo[i], null_ign_hi[i]],
                 color='k', lw=0.8, alpha=0.5)
    ax4.axhline(0, color='k', lw=0.5, alpha=0.3)
    ax4.set_xticks(x)
    ax4.set_xticklabels(pos_labels, rotation=65, fontsize=7, ha='right')
    ax4.set_ylabel('Mean dO/dt over window')
    ax4.set_title('Baseline vs ignition velocity, per position\n(black bars = shuffle null 5-95%)')
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.2, axis='y')

    fig.suptitle(f'{subject} ({condition}) — velocity-field feasibility '
                 f'(N={n_events_used} events)', fontsize=12, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'>>> Saved: {out_path}')


# =========================================================================
# MAIN
# =========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subject', default='sub-010146')
    ap.add_argument('--condition', default='EO')
    ap.add_argument('--band', default='theta_alpha')
    ap.add_argument('--window', type=float, default=8.0)
    ap.add_argument('--hop', type=float, default=0.5)
    ap.add_argument('--bandwidth', type=float, default=0.5)
    ap.add_argument('--sigma', type=float, default=3.0, help='Gaussian smoothing σ (s)')
    ap.add_argument('--pre', type=float, default=20.0)
    ap.add_argument('--post', type=float, default=20.0)
    ap.add_argument('--n-shuffles', type=int, default=50)
    ap.add_argument('--data-dir', default=None,
                    help='Override LEMON data dir (default: /Volumes/T9/...). '
                         'For GCS-pulled local copies, try data_local/lemon_EO')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    fmin, fmax, _ = BANDS.get(args.band, (4.0, 13.0, []))

    # Load events
    suffix = '' if args.condition == 'EC' else f'_{args.condition}'
    events_path = f'exports_sie/lemon{suffix}/{args.subject}_sie_events.csv'
    if not os.path.exists(events_path):
        print(f'ERROR: events CSV not found: {events_path}')
        sys.exit(1)
    events_df = pd.read_csv(events_path)
    event_t0s = events_df['t0_net'].values
    print(f'Loaded {len(event_t0s)} events for {args.subject}')

    # Load raw
    print(f'Loading LEMON {args.subject} ({args.condition})...')
    if args.data_dir:
        raw = load_lemon(args.subject, data_dir=args.data_dir,
                         condition=args.condition)
    else:
        raw = load_lemon(args.subject, condition=args.condition)
    if raw is None:
        print('ERROR: raw load failed (LEMON data accessible? /Volumes/T9 mounted?)')
        sys.exit(1)
    print(f'  fs = {raw.info["sfreq"]} Hz, {len(raw.ch_names)} ch, '
          f'{raw.n_times / raw.info["sfreq"]:.1f} s')

    # Compute occupancy trajectory
    print('Computing occupancy trajectory...')
    times, positions, trajectory = compute_occupancy_trajectory(
        raw, fmin=fmin, fmax=fmax, window_sec=args.window, hop_sec=args.hop,
        bandwidth=args.bandwidth,
    )

    # Nyquist sanity check on hop vs window
    if args.hop * 2 > args.window:
        print(f'  WARNING: hop {args.hop}s > window/2 ({args.window/2}s) — samples correlated')

    # Event-triggered average
    print(f'Event-triggered averaging (±{args.pre}/{args.post}s)...')
    lag_times, eta, n_used = event_triggered_average(
        times, trajectory, event_t0s, pre_sec=args.pre, post_sec=args.post,
        hop_sec=args.hop,
    )
    print(f'  Used {n_used}/{len(event_t0s)} events (edges clipped)')

    if n_used < 3:
        print('ERROR: too few usable events after edge-clipping')
        sys.exit(1)

    # Velocity
    print(f'Smoothing (σ = {args.sigma}s) + differentiating...')
    eta_smooth, velocity = compute_velocity(eta, args.sigma, args.hop)

    # Shuffle null
    print(f'Running {args.n_shuffles} temporal shuffles...')
    shuffled_vel = shuffle_null(trajectory, event_t0s, times, args.hop,
                                 args.sigma, n_shuffles=args.n_shuffles)

    # Plot
    out = args.out or f'outputs/sie_velocity_{args.subject}_{args.condition}.png'
    plot_feasibility(args.subject, args.condition, lag_times, positions,
                     eta_smooth, velocity, shuffled_vel, n_used, out)

    # Summary stats
    pre_mask = lag_times <= -5
    ign_mask = (lag_times >= -5) & (lag_times <= 5)
    real_pre_rms = np.sqrt(np.mean(velocity[:, pre_mask] ** 2))
    real_ign_rms = np.sqrt(np.mean(velocity[:, ign_mask] ** 2))
    null_pre_rms = np.sqrt(np.mean(shuffled_vel[:, :, pre_mask] ** 2, axis=(1, 2)))
    null_ign_rms = np.sqrt(np.mean(shuffled_vel[:, :, ign_mask] ** 2, axis=(1, 2)))
    print()
    print(f'Real velocity RMS — baseline (t ≤ −5 s): {real_pre_rms:.4f}')
    print(f'Real velocity RMS — ignition (±5 s):   {real_ign_rms:.4f}')
    print(f'Ignition / baseline ratio:             {real_ign_rms / real_pre_rms:.2f}')
    print(f'Null baseline RMS 5-95%: [{np.percentile(null_pre_rms, 5):.4f}, '
          f'{np.percentile(null_pre_rms, 95):.4f}]')
    print(f'Null ignition RMS 5-95%: [{np.percentile(null_ign_rms, 5):.4f}, '
          f'{np.percentile(null_ign_rms, 95):.4f}]')


if __name__ == '__main__':
    main()
