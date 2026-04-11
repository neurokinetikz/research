#!/usr/bin/env python3
"""
EC vs EO Phi-Lattice Position Comparison
==========================================

For each dataset with paired EC/EO conditions:
1. Match subjects across conditions
2. Compare per-band lattice coordinates (u) and distances (d)
3. Test whether position shifts are consistent across datasets
4. Test whether alignment (mean_d) changes with eye state
"""

import os, sys, time
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from phi_replication import PHI, F0, BANDS, POSITIONS_14, lattice_coord, min_lattice_dist

lines = []
def P(s=''):
    print(s, flush=True)
    lines.append(s)


DATASETS = {
    'LEMON': {
        'EC': 'exports_lemon/replication/EC/per_subject_dominant_peaks.csv',
        'EO': 'exports_lemon/replication/EO/per_subject_dominant_peaks.csv',
    },
    'EEGMMIDB': {
        'EC': 'exports_eegmmidb/replication/EC/per_subject_dominant_peaks.csv',
        'EO': 'exports_eegmmidb/replication/EO/per_subject_dominant_peaks.csv',
    },
    'HBN': {
        'EC': 'exports_hbn/EC/per_subject_dominant_peaks.csv',
        'EO': 'exports_hbn/EO/per_subject_dominant_peaks.csv',
    },
    'HBN_R2': {
        'EC': 'exports_hbn_R2/EC/per_subject_dominant_peaks.csv',
        'EO': 'exports_hbn_R2/EO/per_subject_dominant_peaks.csv',
    },
    'Dortmund_pre': {
        'EC': '/Volumes/T9/dortmund_data/lattice_results_replication_v2/EyesClosed_pre/per_subject_dominant_peaks.csv',
        'EO': '/Volumes/T9/dortmund_data/lattice_results_replication_v2/EyesOpen_pre/per_subject_dominant_peaks.csv',
    },
    'Dortmund_post': {
        'EC': '/Volumes/T9/dortmund_data/lattice_results_replication_v2/EyesClosed_post/per_subject_dominant_peaks.csv',
        'EO': '/Volumes/T9/dortmund_data/lattice_results_replication_v2/EyesOpen_post/per_subject_dominant_peaks.csv',
    },
}

BAND_ORDER = ['delta', 'theta', 'alpha', 'beta_low', 'beta_high', 'gamma']


def circ_dist_signed(u_eo, u_ec):
    """Signed circular distance: positive = EO shifted toward higher u."""
    d = (u_eo - u_ec) % 1.0
    return np.where(d > 0.5, d - 1.0, d)


def circ_mean(angles):
    """Circular mean on [0,1)."""
    rad = angles * 2 * np.pi
    return (np.arctan2(np.sin(rad).mean(), np.cos(rad).mean()) / (2 * np.pi)) % 1.0


if __name__ == '__main__':
    t0 = time.time()

    P("=" * 90)
    P("  EC vs EO PHI-LATTICE POSITION COMPARISON")
    P("=" * 90)
    P()

    # ──────────────────────────────────────────────────────────
    # LOAD & PAIR SUBJECTS
    # ──────────────────────────────────────────────────────────

    paired_data = {}  # dataset → merged DataFrame

    for ds_name, paths in DATASETS.items():
        ec_path = paths['EC']
        eo_path = paths['EO']
        if not (os.path.isfile(ec_path) and os.path.isfile(eo_path)):
            P(f"  {ds_name}: SKIPPED (files not found)")
            continue

        ec = pd.read_csv(ec_path)
        eo = pd.read_csv(eo_path)

        # Merge on subject
        merged = ec.merge(eo, on='subject', suffixes=('_ec', '_eo'))

        # Check for duplicate data (EC == EO)
        dup_frac = (merged['alpha_freq_ec'] == merged['alpha_freq_eo']).mean()
        if dup_frac > 0.9:
            P(f"  {ds_name}: EXCLUDED — EC/EO data {dup_frac:.0%} identical (extraction error)")
            continue

        P(f"  {ds_name}: EC={len(ec)}, EO={len(eo)}, paired={len(merged)}")
        paired_data[ds_name] = merged

    P()

    # ──────────────────────────────────────────────────────────
    # PART 1: OVERALL ALIGNMENT (mean_d) EC vs EO
    # ──────────────────────────────────────────────────────────

    P("=" * 90)
    P("  PART 1: OVERALL ALIGNMENT (mean_d) — EC vs EO")
    P("=" * 90)
    P()
    P(f"  {'Dataset':<16s}  {'N':>4s}  {'EC mean_d':>9s}  {'EO mean_d':>9s}  {'Δ(EO-EC)':>9s}  {'p(paired)':>10s}  {'Cohen d':>8s}  {'direction':>12s}")
    P(f"  {'─' * 90}")

    for ds_name, df in paired_data.items():
        ec_d = df['mean_d_ec'].dropna().values
        eo_d = df['mean_d_eo'].dropna().values

        # Paired subjects (same index)
        valid = np.isfinite(df['mean_d_ec'].values) & np.isfinite(df['mean_d_eo'].values)
        ec_v = df.loc[valid, 'mean_d_ec'].values
        eo_v = df.loc[valid, 'mean_d_eo'].values
        n = int(valid.sum())

        delta = eo_v - ec_v
        mean_delta = np.mean(delta)
        t_stat, p_val = stats.ttest_rel(eo_v, ec_v)
        sd_delta = np.std(delta)
        cohen_d = mean_delta / sd_delta if sd_delta > 0 else 0
        direction = "EC tighter" if mean_delta > 0 else "EO tighter"

        P(f"  {ds_name:<16s}  {n:>4d}  {np.mean(ec_v):>9.4f}  {np.mean(eo_v):>9.4f}  {mean_delta:>+9.4f}  {p_val:>10.2e}  {cohen_d:>+8.3f}  {direction:>12s}")

    # ──────────────────────────────────────────────────────────
    # PART 2: PER-BAND FREQUENCY SHIFTS (EC→EO)
    # ──────────────────────────────────────────────────────────

    P()
    P("=" * 90)
    P("  PART 2: PER-BAND PEAK FREQUENCY SHIFTS (EO − EC, Hz)")
    P("=" * 90)
    P()

    header = f"  {'Band':<12s}"
    for ds_name in paired_data:
        short = ds_name[:10]
        header += f"  {short:>10s}"
    header += f"  {'consensus':>10s}"
    P(header)
    P(f"  {'─' * (14 + 12 * (len(paired_data) + 1))}")

    band_shift_signs = {b: [] for b in BAND_ORDER}

    for band in BAND_ORDER:
        row = f"  {band:<12s}"
        shifts = []
        for ds_name, df in paired_data.items():
            ec_f = df[f'{band}_freq_ec'].values
            eo_f = df[f'{band}_freq_eo'].values
            valid = np.isfinite(ec_f) & np.isfinite(eo_f) & (ec_f > 0) & (eo_f > 0)
            if valid.sum() < 10:
                row += f"  {'—':>10s}"
                continue
            delta_f = eo_f[valid] - ec_f[valid]
            mean_shift = np.mean(delta_f)
            _, p = stats.ttest_rel(eo_f[valid], ec_f[valid])
            sig = '*' if p < 0.05 else ''
            sig = '**' if p < 0.01 else sig
            sig = '***' if p < 0.001 else sig
            row += f"  {mean_shift:>+8.2f}{sig:>2s}"
            shifts.append(mean_shift)
            band_shift_signs[band].append(np.sign(mean_shift))

        # Consensus
        if shifts:
            n_pos = sum(1 for s in shifts if s > 0)
            n_neg = sum(1 for s in shifts if s < 0)
            consensus = f"{n_pos}↑/{n_neg}↓"
        else:
            consensus = "—"
        row += f"  {consensus:>10s}"
        P(row)

    P()
    P("  Significance: * p<.05  ** p<.01  *** p<.001 (paired t-test)")

    # ──────────────────────────────────────────────────────────
    # PART 3: PER-BAND LATTICE COORDINATE (u) SHIFTS
    # ──────────────────────────────────────────────────────────

    P()
    P("=" * 90)
    P("  PART 3: PER-BAND LATTICE COORDINATE SHIFTS (Δu = EO − EC, circular)")
    P("=" * 90)
    P()

    header = f"  {'Band':<12s}"
    for ds_name in paired_data:
        short = ds_name[:10]
        header += f"  {short:>10s}"
    header += f"  {'consensus':>10s}"
    P(header)
    P(f"  {'─' * (14 + 12 * (len(paired_data) + 1))}")

    band_u_shifts = {b: {} for b in BAND_ORDER}

    for band in BAND_ORDER:
        row = f"  {band:<12s}"
        all_shifts = []
        for ds_name, df in paired_data.items():
            ec_u = df[f'{band}_u_ec'].values
            eo_u = df[f'{band}_u_eo'].values
            valid = np.isfinite(ec_u) & np.isfinite(eo_u)
            if valid.sum() < 10:
                row += f"  {'—':>10s}"
                continue
            delta_u = circ_dist_signed(eo_u[valid], ec_u[valid])
            mean_shift = np.mean(delta_u)
            # Wilcoxon signed-rank (non-parametric, better for circular)
            _, p = stats.wilcoxon(delta_u)
            sig = '*' if p < 0.05 else ''
            sig = '**' if p < 0.01 else sig
            sig = '***' if p < 0.001 else sig
            row += f"  {mean_shift:>+7.3f}{sig:>3s}"
            all_shifts.append(mean_shift)
            band_u_shifts[band][ds_name] = mean_shift

        if all_shifts:
            n_pos = sum(1 for s in all_shifts if s > 0)
            n_neg = sum(1 for s in all_shifts if s < 0)
            consensus = f"{n_pos}↑/{n_neg}↓"
        else:
            consensus = "—"
        row += f"  {consensus:>10s}"
        P(row)

    # ──────────────────────────────────────────────────────────
    # PART 4: PER-BAND DISTANCE-TO-LATTICE (d) SHIFTS
    # ──────────────────────────────────────────────────────────

    P()
    P("=" * 90)
    P("  PART 4: PER-BAND LATTICE DISTANCE SHIFTS (Δd = EO − EC)")
    P("=" * 90)
    P()
    P("  Positive = EO farther from lattice position (less aligned)")
    P()

    header = f"  {'Band':<12s}"
    for ds_name in paired_data:
        short = ds_name[:10]
        header += f"  {short:>10s}"
    header += f"  {'consensus':>10s}"
    P(header)
    P(f"  {'─' * (14 + 12 * (len(paired_data) + 1))}")

    band_d_shifts = {b: {} for b in BAND_ORDER}

    for band in BAND_ORDER:
        row = f"  {band:<12s}"
        all_shifts = []
        for ds_name, df in paired_data.items():
            ec_d = df[f'{band}_d_ec'].values
            eo_d = df[f'{band}_d_eo'].values
            valid = np.isfinite(ec_d) & np.isfinite(eo_d)
            if valid.sum() < 10:
                row += f"  {'—':>10s}"
                continue
            delta_d = eo_d[valid] - ec_d[valid]
            mean_shift = np.mean(delta_d)
            _, p = stats.wilcoxon(delta_d)
            sig = '*' if p < 0.05 else ''
            sig = '**' if p < 0.01 else sig
            sig = '***' if p < 0.001 else sig
            row += f"  {mean_shift:>+7.4f}{sig:>3s}"
            all_shifts.append(mean_shift)
            band_d_shifts[band][ds_name] = mean_shift

        if all_shifts:
            n_pos = sum(1 for s in all_shifts if s > 0)
            n_neg = sum(1 for s in all_shifts if s < 0)
            consensus = f"{n_pos}↑/{n_neg}↓"
        else:
            consensus = "—"
        row += f"  {consensus:>10s}"
        P(row)

    # ──────────────────────────────────────────────────────────
    # PART 5: NEAREST POSITION TRANSITIONS (EC → EO)
    # ──────────────────────────────────────────────────────────

    P()
    P("=" * 90)
    P("  PART 5: NEAREST POSITION TRANSITIONS (EC → EO)")
    P("=" * 90)
    P()
    P("  For each band, what fraction of subjects change their nearest")
    P("  lattice position between EC and EO?")
    P()

    pos_names_sorted = sorted(POSITIONS_14.keys(), key=lambda k: POSITIONS_14[k])

    for band in BAND_ORDER:
        P(f"  {band.upper()}:")

        # Collect transitions across datasets
        all_ec_pos = []
        all_eo_pos = []
        all_ds_labels = []

        for ds_name, df in paired_data.items():
            ec_col = f'{band}_nearest_ec'
            eo_col = f'{band}_nearest_eo'
            if ec_col not in df.columns or eo_col not in df.columns:
                continue
            valid = df[ec_col].notna() & df[eo_col].notna()
            ec_pos = df.loc[valid, ec_col].values
            eo_pos = df.loc[valid, eo_col].values
            all_ec_pos.extend(ec_pos)
            all_eo_pos.extend(eo_pos)
            all_ds_labels.extend([ds_name] * int(valid.sum()))

        all_ec_pos = np.array(all_ec_pos)
        all_eo_pos = np.array(all_eo_pos)

        changed = all_ec_pos != all_eo_pos
        P(f"    Total subjects: {len(all_ec_pos)}, changed position: {changed.sum()} ({np.mean(changed):.1%})")

        # Transition matrix: top 5 most common transitions
        if changed.sum() > 0:
            from collections import Counter
            transitions = Counter(zip(all_ec_pos[changed], all_eo_pos[changed]))
            top5 = transitions.most_common(5)
            P(f"    Top transitions (EC → EO):")
            for (from_pos, to_pos), count in top5:
                pct = count / len(all_ec_pos) * 100
                P(f"      {from_pos:>14s} → {to_pos:<14s}  N={count:>4d} ({pct:.1f}%)")

        # EC and EO position distributions
        ec_counts = Counter(all_ec_pos)
        eo_counts = Counter(all_eo_pos)
        positions_seen = sorted(set(list(ec_counts.keys()) + list(eo_counts.keys())),
                                key=lambda k: POSITIONS_14.get(k, 0.5))

        changes_line = "    Position distribution:  "
        for pos in positions_seen:
            ec_pct = ec_counts.get(pos, 0) / len(all_ec_pos) * 100
            eo_pct = eo_counts.get(pos, 0) / len(all_eo_pos) * 100
            delta_pct = eo_pct - ec_pct
            if abs(delta_pct) > 1.0:
                changes_line += f"{pos}({delta_pct:+.0f}%) "
        P(changes_line)
        P()

    # ──────────────────────────────────────────────────────────
    # PART 5B: PER-BAND 14-POSITION ENRICHMENT: EC vs EO
    # ──────────────────────────────────────────────────────────

    P("=" * 90)
    P("  PART 5B: PER-BAND 14-POSITION ENRICHMENT — EC vs EO")
    P("=" * 90)
    P()
    P("  For each band, compute what fraction of subjects have their")
    P("  nearest position = X, separately for EC and EO.")
    P("  Enrichment = (obs_frac / uniform_frac - 1) × 100%")
    P()

    # Compute uniform expected fractions (Voronoi cell sizes)
    pos_sorted = sorted(POSITIONS_14.items(), key=lambda x: x[1])
    pos_names_list = [p[0] for p in pos_sorted]
    pos_vals_arr = np.array([p[1] for p in pos_sorted])
    n_pos = len(pos_vals_arr)

    # Voronoi cell sizes on circle
    def voronoi_fracs(pos_vals):
        """Expected fraction for each position under uniform u."""
        n = len(pos_vals)
        sv = np.sort(pos_vals)
        fracs = np.zeros(n)
        for i in range(n):
            left = sv[(i - 1) % n]
            right = sv[(i + 1) % n]
            # Half-distance to left neighbor + half-distance to right neighbor
            dl = ((sv[i] - left) % 1.0) / 2.0
            dr = ((right - sv[i]) % 1.0) / 2.0
            fracs[i] = dl + dr
        return fracs

    uniform_fracs = voronoi_fracs(pos_vals_arr)

    def assign_to_nearest_pos(u_vals, pos_vals):
        """Assign each u to index of nearest position (circular)."""
        u = np.asarray(u_vals, dtype=float)
        dists = np.abs(u[:, None] - pos_vals[None, :])
        dists = np.minimum(dists, 1 - dists)
        return np.argmin(dists, axis=1)

    # Store EC/EO enrichment per band per dataset for concordance
    ec_eo_enrichment = {}  # (band, dataset) → {'ec': [...], 'eo': [...]}

    for band in BAND_ORDER:
        P(f"  ── {band.upper()} ──")
        P()

        # Per-dataset table
        ds_list = list(paired_data.keys())

        # Collect per-dataset EC and EO enrichments
        ec_enrichments_all = []  # list of arrays, one per dataset
        eo_enrichments_all = []
        ds_labels = []

        for ds_name, df in paired_data.items():
            ec_u_col = f'{band}_u_ec'
            eo_u_col = f'{band}_u_eo'
            if ec_u_col not in df.columns:
                continue

            valid_ec = df[ec_u_col].notna()
            valid_eo = df[eo_u_col].notna()

            ec_u = df.loc[valid_ec, ec_u_col].values
            eo_u = df.loc[valid_eo, eo_u_col].values

            # Assign to positions
            ec_assign = assign_to_nearest_pos(ec_u, pos_vals_arr)
            eo_assign = assign_to_nearest_pos(eo_u, pos_vals_arr)

            ec_counts = np.bincount(ec_assign, minlength=n_pos).astype(float)
            eo_counts = np.bincount(eo_assign, minlength=n_pos).astype(float)

            ec_fracs = ec_counts / len(ec_u)
            eo_fracs = eo_counts / len(eo_u)

            ec_enrich = np.where(uniform_fracs > 0,
                                 (ec_fracs / uniform_fracs - 1) * 100, 0)
            eo_enrich = np.where(uniform_fracs > 0,
                                 (eo_fracs / uniform_fracs - 1) * 100, 0)

            ec_enrichments_all.append(ec_enrich)
            eo_enrichments_all.append(eo_enrich)
            ds_labels.append(ds_name)

            ec_eo_enrichment[(band, ds_name)] = {
                'ec': ec_enrich, 'eo': eo_enrich
            }

        if not ds_labels:
            P(f"    No data")
            P()
            continue

        # Print pooled enrichment (average across datasets)
        ec_pooled = np.mean(ec_enrichments_all, axis=0)
        eo_pooled = np.mean(eo_enrichments_all, axis=0)
        delta_pooled = eo_pooled - ec_pooled

        # Sign concordance across datasets for each position
        n_ds = len(ds_labels)
        delta_per_ds = [eo_enrichments_all[i] - ec_enrichments_all[i]
                        for i in range(n_ds)]
        delta_matrix = np.array(delta_per_ds)  # (n_ds, n_pos)

        header = f"    {'Position':<14s}  {'u':>5s}  {'EC':>7s}  {'EO':>7s}  {'Δ(EO-EC)':>8s}  {'sign':>6s}"
        P(header)
        P(f"    {'─' * 55}")

        sig_positions = 0
        for i in range(n_pos):
            name = pos_names_list[i]
            u_val = pos_vals_arr[i]
            ec_e = ec_pooled[i]
            eo_e = eo_pooled[i]
            delta_e = delta_pooled[i]

            # Sign concordance
            signs = np.sign(delta_matrix[:, i])
            n_pos_sign = int(np.sum(signs > 0))
            n_neg_sign = int(np.sum(signs < 0))
            max_agree = max(n_pos_sign, n_neg_sign)
            sign_str = f"{max_agree}/{n_ds}"
            if max_agree == n_ds:
                sign_str += "!"
                sig_positions += 1

            P(f"    {name:<14s}  {u_val:>5.3f}  {ec_e:>+6.1f}%  {eo_e:>+6.1f}%  {delta_e:>+7.1f}%  {sign_str:>6s}")

        P(f"    Unanimous positions: {sig_positions}/{n_pos}")
        P()

    # Cross-band summary: which positions show unanimous EC→EO shifts?
    P("  ── CROSS-BAND SUMMARY: UNANIMOUS EC→EO SHIFTS ──")
    P()
    P(f"  Positions where ALL {len(list(paired_data.keys()))} datasets agree on direction of Δ enrichment:")
    P()

    n_ds = len(list(paired_data.keys()))
    header = f"    {'Position':<14s}  {'u':>5s}"
    for band in BAND_ORDER:
        header += f"  {band[:6]:>7s}"
    P(header)
    P(f"    {'─' * (22 + 9 * len(BAND_ORDER))}")

    for i in range(n_pos):
        name = pos_names_list[i]
        u_val = pos_vals_arr[i]
        row = f"    {name:<14s}  {u_val:>5.3f}"
        for band in BAND_ORDER:
            # Collect deltas across datasets
            deltas = []
            for ds_name in paired_data:
                key = (band, ds_name)
                if key in ec_eo_enrichment:
                    d = ec_eo_enrichment[key]['eo'][i] - ec_eo_enrichment[key]['ec'][i]
                    deltas.append(d)
            if deltas:
                mean_d = np.mean(deltas)
                signs = np.sign(deltas)
                n_agree = max(int(np.sum(signs > 0)), int(np.sum(signs < 0)))
                unanimous = "!" if n_agree == len(deltas) else " "
                row += f"  {mean_d:>+6.1f}%{unanimous}"
            else:
                row += f"  {'—':>8s}"
        P(row)

    # Count fully unanimous cells (all datasets agree for a given band × position)
    unanimous_cells = 0
    total_cells = 0
    for band in BAND_ORDER:
        for i in range(n_pos):
            deltas = []
            for ds_name in paired_data:
                key = (band, ds_name)
                if key in ec_eo_enrichment:
                    d = ec_eo_enrichment[key]['eo'][i] - ec_eo_enrichment[key]['ec'][i]
                    deltas.append(d)
            if len(deltas) >= 3:
                total_cells += 1
                signs = np.sign(deltas)
                n_agree = max(int(np.sum(signs > 0)), int(np.sum(signs < 0)))
                if n_agree == len(deltas):
                    unanimous_cells += 1

    P()
    P(f"  Unanimous cells (band × position): {unanimous_cells}/{total_cells}")
    P(f"  Expected by chance (p=0.5^{n_ds}): {total_cells * 0.5**n_ds:.0f}/{total_cells}")
    binomial_p = 1 - stats.binom.cdf(unanimous_cells - 1, total_cells, 2 * 0.5**n_ds)
    P(f"  Binomial p(≥{unanimous_cells}): {binomial_p:.4e}")
    P()

    # ──────────────────────────────────────────────────────────
    # PART 6: CROSS-DATASET CONCORDANCE OF EC→EO SHIFTS
    # ──────────────────────────────────────────────────────────

    P("=" * 90)
    P("  PART 6: CROSS-DATASET CONCORDANCE OF EC→EO SHIFTS")
    P("=" * 90)
    P()
    P("  Do the same bands shift in the same direction across datasets?")
    P()

    # For each band, collect the Δu across datasets
    ds_names = list(paired_data.keys())

    P(f"  Δu (lattice coordinate shift, EO - EC):")
    P()
    header = f"  {'Band':<12s}"
    for ds in ds_names:
        header += f"  {ds[:8]:>8s}"
    header += f"  {'sign concordance':>18s}"
    P(header)
    P(f"  {'─' * (14 + 10 * len(ds_names) + 20)}")

    concordance_count = 0
    total_bands = 0

    for band in BAND_ORDER:
        row = f"  {band:<12s}"
        signs = []
        for ds in ds_names:
            val = band_u_shifts[band].get(ds, np.nan)
            if np.isfinite(val):
                row += f"  {val:>+8.3f}"
                signs.append(np.sign(val))
            else:
                row += f"  {'—':>8s}"

        if len(signs) >= 3:
            n_pos = sum(1 for s in signs if s > 0)
            n_neg = sum(1 for s in signs if s < 0)
            max_agree = max(n_pos, n_neg)
            concordance = f"{max_agree}/{len(signs)}"
            if max_agree == len(signs):
                concordance += " UNANIMOUS"
                concordance_count += 1
            elif max_agree >= len(signs) - 1:
                concordance += " strong"
                concordance_count += 1
            total_bands += 1
        else:
            concordance = "—"
        row += f"  {concordance:>18s}"
        P(row)

    P()
    P(f"  Δd (distance shift, EO - EC):")
    P()
    header = f"  {'Band':<12s}"
    for ds in ds_names:
        header += f"  {ds[:8]:>8s}"
    header += f"  {'sign concordance':>18s}"
    P(header)
    P(f"  {'─' * (14 + 10 * len(ds_names) + 20)}")

    for band in BAND_ORDER:
        row = f"  {band:<12s}"
        signs = []
        for ds in ds_names:
            val = band_d_shifts[band].get(ds, np.nan)
            if np.isfinite(val):
                row += f"  {val:>+8.4f}"
                signs.append(np.sign(val))
            else:
                row += f"  {'—':>8s}"

        if len(signs) >= 3:
            n_pos = sum(1 for s in signs if s > 0)
            n_neg = sum(1 for s in signs if s < 0)
            max_agree = max(n_pos, n_neg)
            concordance = f"{max_agree}/{len(signs)}"
            if max_agree == len(signs):
                concordance += " UNANIMOUS"
            elif max_agree >= len(signs) - 1:
                concordance += " strong"
        else:
            concordance = "—"
        row += f"  {concordance:>18s}"
        P(row)

    # ──────────────────────────────────────────────────────────
    # PART 7: EFFECT SIZE SUMMARY
    # ──────────────────────────────────────────────────────────

    P()
    P("=" * 90)
    P("  PART 7: SUMMARY — PAIRED COHEN'S d FOR KEY METRICS")
    P("=" * 90)
    P()

    header = f"  {'Metric':<20s}"
    for ds in ds_names:
        header += f"  {ds[:8]:>8s}"
    P(header)
    P(f"  {'─' * (22 + 10 * len(ds_names))}")

    # mean_d
    row = f"  {'mean_d (EO-EC)':<20s}"
    for ds_name, df in paired_data.items():
        valid = np.isfinite(df['mean_d_ec'].values) & np.isfinite(df['mean_d_eo'].values)
        ec_v = df.loc[valid, 'mean_d_ec'].values
        eo_v = df.loc[valid, 'mean_d_eo'].values
        delta = eo_v - ec_v
        d = np.mean(delta) / np.std(delta) if np.std(delta) > 0 else 0
        row += f"  {d:>+8.3f}"
    P(row)

    # Per-band d
    for band in BAND_ORDER:
        row = f"  {band + '_d (EO-EC)':<20s}"
        for ds_name, df in paired_data.items():
            ec_d = df[f'{band}_d_ec'].values
            eo_d = df[f'{band}_d_eo'].values
            valid = np.isfinite(ec_d) & np.isfinite(eo_d)
            if valid.sum() < 10:
                row += f"  {'—':>8s}"
                continue
            delta = eo_d[valid] - ec_d[valid]
            d = np.mean(delta) / np.std(delta) if np.std(delta) > 0 else 0
            row += f"  {d:>+8.3f}"
        P(row)

    # Per-band freq
    P()
    row = f"  {'mean_d freq Cohen d':<20s}"
    P(row)
    for band in BAND_ORDER:
        row = f"  {band + '_freq (EO-EC)':<20s}"
        for ds_name, df in paired_data.items():
            ec_f = df[f'{band}_freq_ec'].values
            eo_f = df[f'{band}_freq_eo'].values
            valid = np.isfinite(ec_f) & np.isfinite(eo_f) & (ec_f > 0) & (eo_f > 0)
            if valid.sum() < 10:
                row += f"  {'—':>8s}"
                continue
            delta = eo_f[valid] - ec_f[valid]
            d = np.mean(delta) / np.std(delta) if np.std(delta) > 0 else 0
            row += f"  {d:>+8.3f}"
        P(row)

    elapsed = time.time() - t0
    P(f"\nDone in {elapsed:.0f}s")

    with open('ec_eo_lattice_comparison_results.txt', 'w') as f:
        f.write('\n'.join(lines))
    print(f"\nSaved to ec_eo_lattice_comparison_results.txt")
