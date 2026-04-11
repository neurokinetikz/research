#!/usr/bin/env python3
"""
Ratio Lattice Enrichment — Per Band Pair
=========================================

For each pair of bands (e.g., alpha/theta, gamma/beta_high), compute the
dominant-peak ratio per subject, map to phi-octave coordinate u = log_φ(ratio)
mod 1, and test enrichment at all 14 degree-7 positions.

Key question: Do specific band-pair ratios cluster near specific lattice
positions (nobles, inverse nobles, attractor, boundary)?
"""

import os, sys, time, itertools
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from phi_replication import PHI, F0, BANDS, BASES, POSITIONS_14, lattice_coord

lines = []
def P(s=''):
    print(s, flush=True)
    lines.append(s)


# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

DATASETS = {
    'EEGMMIDB_EC': 'exports_eegmmidb/replication/combined/per_subject_dominant_peaks.csv',
    'LEMON_EC':    'exports_lemon/replication/EC/per_subject_dominant_peaks.csv',
    'LEMON_EO':    'exports_lemon/replication/EO/per_subject_dominant_peaks.csv',
    'HBN_EC':      'exports_hbn/EC/per_subject_dominant_peaks.csv',
    'Dortmund_EC': '/Volumes/T9/dortmund_data/lattice_results_replication_v2/EyesClosed_pre/per_subject_dominant_peaks.csv',
}

BAND_ORDER = ['delta', 'theta', 'alpha', 'beta_low', 'beta_high', 'gamma']


def positions_degree_7(base, min_sep=0.02):
    inv = 1.0 / base
    pos = {'boundary': 0.0, 'attractor': 0.5}
    for k in range(1, 8):
        val = inv ** k
        if min_sep <= val <= 1 - min_sep and abs(val - 0.5) >= min_sep:
            if all(abs(val - v) > min_sep for v in pos.values()):
                pos[f'noble_{k}'] = val
        ival = 1 - inv ** k
        if min_sep <= ival <= 1 - min_sep and abs(ival - 0.5) >= min_sep:
            if all(abs(ival - v) > min_sep for v in pos.values()):
                pos[f'inv_noble_{k}'] = ival
    return pos


def assign_nearest(u_values, pos_vals):
    """Assign each u to nearest position (circular)."""
    u = np.asarray(u_values)
    dists = np.abs(u[:, None] - pos_vals[None, :])
    dists = np.minimum(dists, 1 - dists)
    return np.argmin(dists, axis=1)


def compute_expected_fracs(pos_vals, n_grid=100_000):
    u_grid = np.linspace(0, 1, n_grid, endpoint=False)
    exp_assign = assign_nearest(u_grid, pos_vals)
    exp_counts = np.bincount(exp_assign, minlength=len(pos_vals))
    return exp_counts / n_grid


def circ_dist(a, b):
    d = abs(a - b) % 1.0
    return min(d, 1 - d)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    t0 = time.time()

    P("=" * 80)
    P("  RATIO LATTICE ENRICHMENT — PER BAND PAIR")
    P("=" * 80)
    P()

    # Load datasets
    all_dfs = {}
    for name, path in DATASETS.items():
        if os.path.isfile(path):
            df = pd.read_csv(path)
            all_dfs[name] = df
            P(f"  {name:<14s}  N={len(df):>4d}")
        else:
            P(f"  {name:<14s}  NOT FOUND")
    P()

    # Pool all subjects
    pooled = pd.concat(all_dfs.values(), ignore_index=True)
    N_total = len(pooled)
    P(f"  Pooled: N={N_total}")

    # ──────────────────────────────────────────────────────────────
    # PART 1: Per band-pair ratio → lattice coordinate
    # ──────────────────────────────────────────────────────────────
    P()
    P("=" * 80)
    P("  PART 1: BAND-PAIR RATIOS → PHI-LATTICE POSITIONS")
    P("=" * 80)
    P()

    pairs = list(itertools.combinations(BAND_ORDER, 2))
    positions = POSITIONS_14
    pos_names = list(positions.keys())
    pos_vals = np.array(list(positions.values()))
    expected_fracs = compute_expected_fracs(pos_vals)

    # Sort positions by value for display
    sorted_idx = np.argsort(pos_vals)
    sorted_names = [pos_names[i] for i in sorted_idx]
    sorted_vals = pos_vals[sorted_idx]

    # Build header
    P(f"  For each band pair (high/low), map ratio to u = log_φ(ratio) mod 1")
    P(f"  Then find nearest of {len(pos_names)} degree-7 positions")
    P()

    # Collect per-pair stats
    pair_results = {}

    for lo_band, hi_band in pairs:
        lo_freq = pooled[f'{lo_band}_freq'].values
        hi_freq = pooled[f'{hi_band}_freq'].values

        valid = np.isfinite(lo_freq) & np.isfinite(hi_freq) & (lo_freq > 0) & (hi_freq > 0)
        lo_f = lo_freq[valid]
        hi_f = hi_freq[valid]

        ratios = hi_f / lo_f
        # Ensure ratio > 1 (high / low)
        ratios = np.where(ratios < 1, 1.0 / ratios, ratios)

        # Map to lattice coordinate
        u_vals = np.array([lattice_coord(r, f0=1.0, base=PHI) for r in ratios])

        # Assign to nearest position
        assignments = assign_nearest(u_vals, pos_vals)
        obs_counts = np.bincount(assignments, minlength=len(pos_vals))
        obs_fracs = obs_counts / len(u_vals)

        # Enrichment
        enrichments = {}
        for i, name in enumerate(pos_names):
            if expected_fracs[i] > 0:
                enrichments[name] = (obs_fracs[i] / expected_fracs[i] - 1) * 100
            else:
                enrichments[name] = 0.0

        # Mean u and nearest position
        mean_u = np.mean(u_vals)
        median_u = np.median(u_vals)
        mean_ratio = np.mean(ratios)
        median_ratio = np.median(ratios)

        # Modal position (most common assignment)
        modal_pos_idx = np.argmax(obs_counts)
        modal_pos = pos_names[modal_pos_idx]
        modal_frac = obs_fracs[modal_pos_idx]

        # Distance to nearest position from median u
        dists_to_pos = np.array([circ_dist(median_u, v) for v in pos_vals])
        nearest_idx = np.argmin(dists_to_pos)
        nearest_pos = pos_names[nearest_idx]
        nearest_dist = dists_to_pos[nearest_idx]

        pair_results[(lo_band, hi_band)] = {
            'n': int(valid.sum()),
            'mean_ratio': mean_ratio,
            'median_ratio': median_ratio,
            'mean_u': mean_u,
            'median_u': median_u,
            'nearest_pos': nearest_pos,
            'nearest_dist': nearest_dist,
            'modal_pos': modal_pos,
            'modal_frac': modal_frac,
            'enrichments': enrichments,
            'u_vals': u_vals,
        }

    # Print summary table
    P(f"  {'Band pair':<22s}  {'N':>5s}  {'med ratio':>9s}  {'med u':>6s}  {'nearest pos':>14s}  {'dist':>5s}  {'modal pos':>14s}  {'modal%':>6s}")
    P(f"  {'─'*90}")

    for (lo, hi), r in pair_results.items():
        label = f"{hi}/{lo}"
        P(f"  {label:<22s}  {r['n']:>5d}  {r['median_ratio']:>9.3f}  {r['median_u']:>6.3f}  {r['nearest_pos']:>14s}  {r['nearest_dist']:>5.3f}  {r['modal_pos']:>14s}  {r['modal_frac']:>5.1%}")

    # ──────────────────────────────────────────────────────────────
    # PART 2: Full enrichment profiles per band pair
    # ──────────────────────────────────────────────────────────────
    P()
    P("=" * 80)
    P("  PART 2: ENRICHMENT PROFILES PER BAND PAIR (phi, 14 positions)")
    P("=" * 80)
    P()

    # Select most informative pairs (skip delta which is noisy)
    key_pairs = [
        ('theta', 'alpha'),
        ('theta', 'gamma'),
        ('alpha', 'beta_low'),
        ('alpha', 'beta_high'),
        ('alpha', 'gamma'),
        ('beta_low', 'gamma'),
        ('beta_high', 'gamma'),
        ('theta', 'beta_low'),
        ('theta', 'beta_high'),
        ('delta', 'theta'),
        ('delta', 'alpha'),
    ]

    # Header
    header = f"  {'Position':<14s}  {'u':>5s}"
    for lo, hi in key_pairs:
        label = f"{hi}/{lo}"
        if len(label) > 8:
            label = label[:8]
        header += f"  {label:>8s}"
    P(header)
    P(f"  {'─' * (22 + 10 * len(key_pairs))}")

    for idx in sorted_idx:
        name = pos_names[idx]
        val = pos_vals[idx]
        row = f"  {name:<14s}  {val:>5.3f}"
        for lo, hi in key_pairs:
            e = pair_results[(lo, hi)]['enrichments'][name]
            row += f"  {e:>+7.1f}%"
        P(row)

    # ──────────────────────────────────────────────────────────────
    # PART 3: Per-dataset replication of band-pair lattice positions
    # ──────────────────────────────────────────────────────────────
    P()
    P("=" * 80)
    P("  PART 3: PER-DATASET BAND-PAIR MEDIAN u (replication check)")
    P("=" * 80)
    P()

    focus_pairs = [('theta', 'alpha'), ('theta', 'gamma'), ('alpha', 'gamma'),
                   ('alpha', 'beta_high'), ('delta', 'theta')]

    header = f"  {'Pair':<16s}"
    for ds_name in all_dfs:
        short = ds_name[:10]
        header += f"  {short:>10s}"
    header += f"  {'nearest':>14s}"
    P(header)
    P(f"  {'─' * (18 + 12 * len(all_dfs) + 16)}")

    for lo, hi in focus_pairs:
        label = f"{hi}/{lo}"
        row = f"  {label:<16s}"
        all_med_u = []
        for ds_name, df in all_dfs.items():
            lo_f = df[f'{lo}_freq'].values
            hi_f = df[f'{hi}_freq'].values
            valid = np.isfinite(lo_f) & np.isfinite(hi_f) & (lo_f > 0) & (hi_f > 0)
            ratios = hi_f[valid] / lo_f[valid]
            ratios = np.where(ratios < 1, 1.0 / ratios, ratios)
            u = np.array([lattice_coord(r, f0=1.0, base=PHI) for r in ratios])
            med_u = np.median(u)
            row += f"  {med_u:>10.3f}"
            all_med_u.append(med_u)

        # Nearest position for the pooled median
        pooled_med = pair_results[(lo, hi)]['median_u']
        nearest = pair_results[(lo, hi)]['nearest_pos']
        row += f"  {nearest:>14s}"
        P(row)

    # ──────────────────────────────────────────────────────────────
    # PART 4: Cross-base comparison — do band-pair ratios prefer phi?
    # ──────────────────────────────────────────────────────────────
    P()
    P("=" * 80)
    P("  PART 4: CROSS-BASE COMPARISON (band-pair ratios)")
    P("=" * 80)
    P()
    P("  For each band pair, compute mean lattice distance across 9 bases.")
    P("  Lower mean_d = ratios closer to lattice positions.")
    P()

    # For each pair, compute mean_d at each base
    base_items = list(BASES.items())

    header = f"  {'Pair':<16s}"
    for bname, _ in base_items:
        header += f"  {bname:>6s}"
    header += f"  {'best':>6s}"
    P(header)
    P(f"  {'─' * (18 + 8 * len(base_items) + 8)}")

    pair_base_ds = {}
    for lo, hi in key_pairs:
        u_data = pair_results[(lo, hi)]['u_vals']
        ratios_data = pair_results[(lo, hi)]['median_ratio']
        label = f"{hi}/{lo}"

        # Recompute from raw ratios for each base
        lo_freq = pooled[f'{lo}_freq'].values
        hi_freq = pooled[f'{hi}_freq'].values
        valid = np.isfinite(lo_freq) & np.isfinite(hi_freq) & (lo_freq > 0) & (hi_freq > 0)
        ratios = hi_freq[valid] / lo_freq[valid]
        ratios = np.where(ratios < 1, 1.0 / ratios, ratios)

        row = f"  {label:<16s}"
        best_base = None
        best_d = 1.0
        ds_for_pair = {}

        for bname, bval in base_items:
            bpos = positions_degree_7(bval) if bname != 'phi' else POSITIONS_14
            bpos_vals = np.array(list(bpos.values()))

            # Compute u for this base
            log_base = np.log(bval)
            u_base = (np.log(ratios) / log_base) % 1.0

            # Mean min distance to nearest position
            assignments = assign_nearest(u_base, bpos_vals)
            dists = np.minimum(
                np.abs(u_base - bpos_vals[assignments]),
                1 - np.abs(u_base - bpos_vals[assignments])
            )
            mean_d = np.mean(dists)
            ds_for_pair[bname] = mean_d

            if mean_d < best_d:
                best_d = mean_d
                best_base = bname

            row += f"  {mean_d:>6.4f}"

        row += f"  {best_base:>6s}"
        P(row)
        pair_base_ds[(lo, hi)] = ds_for_pair

    # Count how many pairs each base wins
    P()
    base_wins = {bname: 0 for bname, _ in base_items}
    for (lo, hi), ds in pair_base_ds.items():
        winner = min(ds, key=ds.get)
        base_wins[winner] += 1

    P(f"  Base wins (out of {len(key_pairs)} pairs):")
    for bname in sorted(base_wins, key=base_wins.get, reverse=True):
        if base_wins[bname] > 0:
            P(f"    {bname:>6s}: {base_wins[bname]}")

    # ──────────────────────────────────────────────────────────────
    # PART 5: Concentration test — are ratios more concentrated
    #         at lattice positions than uniform?
    # ──────────────────────────────────────────────────────────────
    P()
    P("=" * 80)
    P("  PART 5: CONCENTRATION TEST (observed vs uniform mean_d)")
    P("=" * 80)
    P()

    # Null: uniform u → expected mean_d
    rng = np.random.RandomState(42)
    null_u = rng.uniform(0, 1, 100_000)
    null_assign = assign_nearest(null_u, pos_vals)
    null_dists = np.minimum(
        np.abs(null_u - pos_vals[null_assign]),
        1 - np.abs(null_u - pos_vals[null_assign])
    )
    null_mean_d = np.mean(null_dists)

    P(f"  Uniform null mean_d (14 positions): {null_mean_d:.4f}")
    P()
    P(f"  {'Pair':<16s}  {'N':>5s}  {'mean_d':>7s}  {'null':>7s}  {'Δ':>7s}  {'Cohen d':>8s}  {'direction':>10s}")
    P(f"  {'─' * 75}")

    for lo, hi in key_pairs:
        u_vals = pair_results[(lo, hi)]['u_vals']
        n = len(u_vals)
        assignments = assign_nearest(u_vals, pos_vals)
        dists = np.minimum(
            np.abs(u_vals - pos_vals[assignments]),
            1 - np.abs(u_vals - pos_vals[assignments])
        )
        obs_mean_d = np.mean(dists)
        obs_sd = np.std(dists)
        delta = obs_mean_d - null_mean_d
        cohen_d = delta / obs_sd if obs_sd > 0 else 0
        direction = "CLOSER" if delta < 0 else "FARTHER"
        label = f"{hi}/{lo}"
        P(f"  {label:<16s}  {n:>5d}  {obs_mean_d:>7.4f}  {null_mean_d:>7.4f}  {delta:>+7.4f}  {cohen_d:>+8.3f}  {direction:>10s}")

    # ──────────────────────────────────────────────────────────────
    # PART 6: Permutation test — are ratios closer to phi lattice
    #         than shuffled (band-label permutation)?
    # ──────────────────────────────────────────────────────────────
    P()
    P("=" * 80)
    P("  PART 6: PERMUTATION TEST (shuffle band labels within subjects)")
    P("=" * 80)
    P()
    P("  Shuffle which frequency belongs to which band, recompute ratios.")
    P("  This preserves per-subject frequency sets but breaks band identity.")
    P()

    N_PERM = 5000
    focus_test = [('theta', 'alpha'), ('theta', 'gamma'), ('alpha', 'gamma')]

    for lo, hi in focus_test:
        label = f"{hi}/{lo}"
        lo_freq = pooled[f'{lo}_freq'].values
        hi_freq = pooled[f'{hi}_freq'].values
        valid = np.isfinite(lo_freq) & np.isfinite(hi_freq) & (lo_freq > 0) & (hi_freq > 0)
        lo_f = lo_freq[valid]
        hi_f = hi_freq[valid]
        n = int(valid.sum())

        ratios = hi_f / lo_f
        ratios = np.where(ratios < 1, 1.0 / ratios, ratios)
        u_obs = (np.log(ratios) / np.log(PHI)) % 1.0

        obs_assign = assign_nearest(u_obs, pos_vals)
        obs_dists = np.minimum(
            np.abs(u_obs - pos_vals[obs_assign]),
            1 - np.abs(u_obs - pos_vals[obs_assign])
        )
        obs_mean_d = np.mean(obs_dists)

        # Collect all 6 band frequencies per subject for shuffling
        all_band_freqs = np.column_stack([
            pooled[f'{b}_freq'].values for b in BAND_ORDER
        ])
        all_band_freqs = all_band_freqs[valid]  # same subjects

        rng = np.random.RandomState(42)
        null_ds = np.empty(N_PERM)
        lo_idx = BAND_ORDER.index(lo)
        hi_idx = BAND_ORDER.index(hi)

        for pi in range(N_PERM):
            # Shuffle band labels within each subject
            shuf = all_band_freqs.copy()
            for s in range(len(shuf)):
                rng.shuffle(shuf[s])
            s_lo = shuf[:, lo_idx]
            s_hi = shuf[:, hi_idx]
            s_ratios = s_hi / s_lo
            s_ratios = np.where(s_ratios < 1, 1.0 / s_ratios, s_ratios)
            # Filter invalid
            s_valid = np.isfinite(s_ratios) & (s_ratios > 0)
            s_u = (np.log(s_ratios[s_valid]) / np.log(PHI)) % 1.0
            s_assign = assign_nearest(s_u, pos_vals)
            s_dists = np.minimum(
                np.abs(s_u - pos_vals[s_assign]),
                1 - np.abs(s_u - pos_vals[s_assign])
            )
            null_ds[pi] = np.mean(s_dists)

        z = (np.mean(null_ds) - obs_mean_d) / np.std(null_ds) if np.std(null_ds) > 0 else 0
        p_val = np.mean(null_ds <= obs_mean_d)

        P(f"  {label}:")
        P(f"    Observed mean_d: {obs_mean_d:.4f}")
        P(f"    Null mean(mean_d): {np.mean(null_ds):.4f} ± {np.std(null_ds):.4f}")
        P(f"    z = {z:+.2f}, p(obs ≤ null) = {p_val:.4f}")
        P(f"    Verdict: {'CLOSER than shuffled' if z > 1.96 else 'NOT significantly closer'}")
        P()

    elapsed = time.time() - t0
    P(f"\nDone in {elapsed:.0f}s")

    with open('ratio_lattice_per_band_results.txt', 'w') as f:
        f.write('\n'.join(lines))
    print(f"\nSaved to ratio_lattice_per_band_results.txt")
