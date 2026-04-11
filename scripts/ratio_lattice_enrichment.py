#!/usr/bin/env python3
"""
Ratio Lattice Enrichment — 14-Position Fine-Structure Test
═══════════════════════════════════════════════════════════

Maps pairwise frequency ratios to phi-octave lattice coordinates and
tests enrichment at ALL degree-7 positions (14 for phi).

Key question: When ratios are mapped to lattice coordinates, does the
14-position enrichment profile show structured, replicable patterns?
And does phi's profile replicate better than other bases' profiles?

Also stratifies by octave distance (how many phi-octaves apart the
two peaks are) to test whether cross-octave ratio structure matches
the within-band position structure.
"""

import os
import sys
import glob
import time
import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
from phi_replication import PHI, F0, BASES, POSITIONS_14

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEDUP_TOL = 0.15
RATIO_RANGE = (1.05, 4.0)

lines = []
def P(s=''):
    print(s, flush=True)
    lines.append(s)


# ═══════════════════════════════════════════════════════════════
# DATA LOADING & RATIO COMPUTATION (reused from pairwise_ratio_test)
# ═══════════════════════════════════════════════════════════════

def load_peaks(peaks_dir):
    """Load all per-subject peak CSVs, deduplicate OT edge peaks."""
    files = sorted(glob.glob(os.path.join(peaks_dir, '*_peaks.csv')))
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        sub = os.path.basename(f).replace('_peaks.csv', '')
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if len(df) == 0:
            continue
        df['subject'] = sub
        df['channel'] = df['channel'].str.rstrip('.')
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    peaks = pd.concat(dfs, ignore_index=True)
    peaks['freq_bin'] = (peaks['freq'] / DEDUP_TOL).round() * DEDUP_TOL
    peaks = (peaks.sort_values('power', ascending=False)
             .drop_duplicates(subset=['subject', 'channel', 'freq_bin'], keep='first')
             .drop(columns=['freq_bin'])
             .sort_values(['subject', 'channel', 'freq']))
    return peaks


def compute_ratios(peaks_df, ratio_range=RATIO_RANGE):
    """Compute all within-channel pairwise ratios f_high / f_low."""
    all_ratios = []
    for (sub, ch), group in peaks_df.groupby(['subject', 'channel']):
        freqs = np.sort(group['freq'].values)
        n = len(freqs)
        if n < 2:
            continue
        for i in range(n):
            for j in range(i + 1, n):
                r = freqs[j] / freqs[i]
                if ratio_range[0] <= r <= ratio_range[1]:
                    all_ratios.append(r)
    return np.array(all_ratios)


# ═══════════════════════════════════════════════════════════════
# LATTICE POSITION GENERATION
# ═══════════════════════════════════════════════════════════════

def positions_degree_7(base, min_sep=0.02):
    """Generate degree-7 lattice positions for any base.
    Same construction as phi's POSITIONS_14: powers of 1/base
    and their complements, filtered for uniqueness."""
    inv = 1.0 / base
    pos = {'boundary': 0.0, 'attractor': 0.5}

    for k in range(1, 8):
        val = inv ** k
        name = f'noble_{k}'
        if min_sep <= val <= 1 - min_sep and abs(val - 0.5) >= min_sep:
            if all(abs(val - v) > min_sep for v in pos.values()):
                pos[name] = val

        ival = 1 - inv ** k
        iname = f'inv_noble_{k}'
        if min_sep <= ival <= 1 - min_sep and abs(ival - 0.5) >= min_sep:
            if all(abs(ival - v) > min_sep for v in pos.values()):
                pos[iname] = ival

    return pos


# ═══════════════════════════════════════════════════════════════
# ENRICHMENT COMPUTATION
# ═══════════════════════════════════════════════════════════════

def assign_nearest(u_values, pos_vals, chunk_size=1_000_000):
    """Assign each u to nearest position (circular), chunked for memory."""
    assignments = np.empty(len(u_values), dtype=np.int32)
    for start in range(0, len(u_values), chunk_size):
        end = min(start + chunk_size, len(u_values))
        chunk = u_values[start:end]
        dists = np.abs(chunk[:, None] - pos_vals[None, :])
        dists = np.minimum(dists, 1 - dists)
        assignments[start:end] = np.argmin(dists, axis=1)
    return assignments


def compute_expected_fracs(pos_vals, n_grid=100_000):
    """Expected assignment fractions under uniform distribution."""
    u_grid = np.linspace(0, 1, n_grid, endpoint=False)
    exp_assign = assign_nearest(u_grid, pos_vals)
    exp_counts = np.bincount(exp_assign, minlength=len(pos_vals))
    return exp_counts / n_grid


def enrichment_at_positions(u_values, positions_dict, expected_fracs=None):
    """Enrichment at each position: (obs_frac / exp_frac - 1) * 100%.

    If expected_fracs is provided (precomputed), skips uniform grid computation.
    """
    pos_names = list(positions_dict.keys())
    pos_vals = np.array(list(positions_dict.values()))
    n_pos = len(pos_vals)

    # Observed
    obs_assign = assign_nearest(u_values, pos_vals)
    obs_counts = np.bincount(obs_assign, minlength=n_pos)
    obs_fracs = obs_counts / len(u_values)

    # Expected
    if expected_fracs is None:
        expected_fracs = compute_expected_fracs(pos_vals)

    # Enrichment
    enrichments = {}
    for i, name in enumerate(pos_names):
        if expected_fracs[i] > 0:
            enrichments[name] = (obs_fracs[i] / expected_fracs[i] - 1) * 100
        else:
            enrichments[name] = 0.0

    return enrichments


def phase_rotation_null(u_values, positions_dict, expected_fracs,
                        n_perm=5000, n_bins=5000):
    """Phase-rotation null: shift positions by random offset, compute enrichment.

    This controls for the broad shape of the u distribution.
    Under the null, enrichment is driven by the u distribution shape alone.
    The corrected enrichment = observed - mean(null) reveals genuine
    lattice-specific fine structure.

    Uses histogram representation for speed.
    """
    pos_names = list(positions_dict.keys())
    pos_vals = np.array(list(positions_dict.values()))
    n_pos = len(pos_vals)
    N = len(u_values)

    # Bin the data
    bin_edges = np.linspace(0, 1, n_bins + 1)
    counts, _ = np.histogram(u_values, bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    null_enrichments = np.zeros((n_perm, n_pos))

    for p in range(n_perm):
        delta = np.random.uniform(0, 1)
        shifted_pos = (pos_vals + delta) % 1.0

        # Assign bins to shifted positions
        dists = np.abs(bin_centers[:, None] - shifted_pos[None, :])
        dists = np.minimum(dists, 1 - dists)
        bin_assign = np.argmin(dists, axis=1)

        # Observed fracs (data is fixed, positions are shifted)
        obs_counts_p = np.bincount(
            bin_assign, weights=counts.astype(float), minlength=n_pos)
        obs_fracs_p = obs_counts_p / N

        # Enrichment vs uniform expectation (cell sizes unchanged)
        for j in range(n_pos):
            if expected_fracs[j] > 0:
                null_enrichments[p, j] = (
                    obs_fracs_p[j] / expected_fracs[j] - 1) * 100
            else:
                null_enrichments[p, j] = 0.0

    return null_enrichments


def kendalls_w(enrichment_matrix):
    """Kendall's W coefficient of concordance.
    enrichment_matrix: shape (n_datasets, n_positions)."""
    k, n = enrichment_matrix.shape
    if k < 2 or n < 2:
        return np.nan, np.nan

    # Rank within each dataset (row)
    ranked = np.array([rankdata(row) for row in enrichment_matrix])
    col_sums = ranked.sum(axis=0)
    mean_sum = col_sums.mean()
    S = np.sum((col_sums - mean_sum) ** 2)
    W = 12 * S / (k ** 2 * (n ** 3 - n))

    # Chi-squared test
    chi2 = k * (n - 1) * W
    from scipy.stats import chi2 as chi2_dist
    p = 1 - chi2_dist.cdf(chi2, df=n - 1)

    return W, p


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    P("═" * 72)
    P("  RATIO LATTICE ENRICHMENT — 14-POSITION FINE-STRUCTURE TEST")
    P("═" * 72)
    P()
    P("Maps pairwise ratios → lattice coordinates → enrichment at ALL")
    P("degree-7 positions. Tests whether ratio structure replicates")
    P("across datasets and whether phi's profile is uniquely structured.")
    P()

    # ── Load datasets ──
    DATASETS = {}
    for name, relpath in [
        ('EEGMMIDB_EC', 'exports_eegmmidb/replication/EC/per_subject_peaks'),
        ('LEMON_EC',    'exports_lemon/replication/EC/per_subject_peaks'),
        ('LEMON_EO',    'exports_lemon/replication/EO/per_subject_peaks'),
        ('HBN_EC',      'exports_hbn/EC/per_subject_peaks'),
    ]:
        full = os.path.join(BASE_DIR, relpath)
        if os.path.isdir(full):
            DATASETS[name] = full
    dort = ('/Volumes/T9/dortmund_data/lattice_results_replication_v2/'
            'EyesClosed_pre/per_subject_peaks')
    if os.path.isdir(dort):
        DATASETS['Dortmund_EC'] = dort

    P("─" * 72)
    P("DATA LOADING & RATIO COMPUTATION")
    P("─" * 72)

    dataset_ratios = {}
    for dname, dpath in DATASETS.items():
        t1 = time.time()
        peaks = load_peaks(dpath)
        if len(peaks) == 0:
            continue
        ratios = compute_ratios(peaks)
        dataset_ratios[dname] = ratios
        n_sub = peaks['subject'].nunique()
        P(f"  {dname:<15} N={n_sub:>4}  "
          f"{len(ratios):>10,} ratios  [{time.time()-t1:.1f}s]")

    all_ratios = np.concatenate(list(dataset_ratios.values()))
    P(f"\n  TOTAL: {len(all_ratios):,} pairwise ratios "
      f"across {len(dataset_ratios)} datasets")

    # ── Generate degree-7 positions for all bases ──
    P(f"\n{'─' * 72}")
    P("DEGREE-7 POSITIONS PER BASE")
    P("─" * 72)

    base_positions = {}
    base_expected = {}
    for bname, bval in BASES.items():
        if bname == 'phi':
            pos = dict(POSITIONS_14)
        else:
            pos = positions_degree_7(bval)
        base_positions[bname] = pos
        pos_vals = np.array(list(pos.values()))
        base_expected[bname] = compute_expected_fracs(pos_vals)
        vals_str = ', '.join(f'{v:.3f}' for v in sorted(pos.values()))
        P(f"  {bname:<8} {len(pos):>2} positions: {vals_str}")

    # ══════════════════════════════════════════════════════════════
    # PART 1: PHI RATIO-ENRICHMENT TABLE (14 positions, pooled)
    # ══════════════════════════════════════════════════════════════

    P(f"\n{'═' * 72}")
    P("  PART 1: PHI RATIO-ENRICHMENT (14 positions, pooled)")
    P(f"{'═' * 72}")

    u_phi_all = (np.log(all_ratios) / np.log(PHI)) % 1.0
    phi_enrich_pooled = enrichment_at_positions(
        u_phi_all, POSITIONS_14, base_expected['phi'])

    sorted_pos = sorted(POSITIONS_14.items(), key=lambda x: x[1])
    pos_names_sorted = [name for name, _ in sorted_pos]

    P(f"\n  {'Position':<15} {'u':>6} {'Enrich%':>9}")
    P(f"  {'─' * 34}")
    for name, u_val in sorted_pos:
        e = phi_enrich_pooled[name]
        n_chars = min(40, int(abs(e) / 2))
        bar = ('+' * n_chars) if e > 0 else ('-' * n_chars)
        P(f"  {name:<15} {u_val:>6.3f} {e:>+8.1f}%  {bar}")

    mean_abs = np.mean([abs(v) for v in phi_enrich_pooled.values()])
    P(f"\n  Mean |enrichment|: {mean_abs:.1f}%")

    # ── Phase-rotation correction ──
    P(f"\n  Phase-rotation null (5000 permutations)...")
    t1 = time.time()
    null_enrichments = phase_rotation_null(
        u_phi_all, POSITIONS_14, base_expected['phi'])
    null_mean = np.mean(null_enrichments, axis=0)
    null_std = np.std(null_enrichments, axis=0)
    P(f"  Done [{time.time()-t1:.1f}s]")

    P(f"\n  {'Position':<15} {'u':>6} {'Raw':>8} {'Null':>8} "
      f"{'Corr':>8} {'z':>7}")
    P(f"  {'─' * 57}")
    corrected_enrichments = {}
    for j, (name, u_val) in enumerate(sorted_pos):
        raw = phi_enrich_pooled[name]
        corr = raw - null_mean[j]
        z = corr / null_std[j] if null_std[j] > 0 else 0
        corrected_enrichments[name] = corr
        sig = '**' if abs(z) > 2.58 else '*' if abs(z) > 1.96 else ''
        P(f"  {name:<15} {u_val:>6.3f} {raw:>+7.1f}% {null_mean[j]:>+7.1f}% "
          f"{corr:>+7.1f}% {z:>+6.2f} {sig}")

    mean_abs_corr = np.mean([abs(v) for v in corrected_enrichments.values()])
    n_sig = sum(1 for j, (name, _) in enumerate(sorted_pos)
                if abs(phi_enrich_pooled[name] - null_mean[j]) /
                max(null_std[j], 1e-6) > 1.96)
    P(f"\n  Corrected mean |enrichment|: {mean_abs_corr:.1f}%")
    P(f"  Positions significant at p<0.05: {n_sig}/{len(sorted_pos)}")

    # ── Phase-rotation for ALL bases ──
    P(f"\n  Computing phase-rotation corrected concordance for all bases...")
    t1 = time.time()
    base_corr_mean_abs = {}
    for bname, bval in BASES.items():
        positions = base_positions[bname]
        pos_vals_b = np.array(list(positions.values()))
        u_b = (np.log(all_ratios) / np.log(bval)) % 1.0
        enrich_b = enrichment_at_positions(u_b, positions, base_expected[bname])

        null_b = phase_rotation_null(
            u_b, positions, base_expected[bname], n_perm=2000)
        null_mean_b = np.mean(null_b, axis=0)
        null_std_b = np.std(null_b, axis=0)

        # Corrected enrichments
        pos_names_b = list(positions.keys())
        corr_vals = []
        n_sig_b = 0
        for j, pn in enumerate(pos_names_b):
            raw = enrich_b[pn]
            corr = raw - null_mean_b[j]
            corr_vals.append(abs(corr))
            if null_std_b[j] > 0 and abs(corr) / null_std_b[j] > 1.96:
                n_sig_b += 1

        base_corr_mean_abs[bname] = {
            'mean_abs_corr': np.mean(corr_vals),
            'n_sig': n_sig_b,
            'n_pos': len(pos_names_b),
        }

    P(f"  Done [{time.time()-t1:.1f}s]")

    corr_ranked = sorted(base_corr_mean_abs.items(),
                         key=lambda x: x[1]['mean_abs_corr'], reverse=True)
    P(f"\n  {'Base':<8} {'Corr|E|':>8} {'Sig/Pos':>8} {'Rank':>5}")
    P(f"  {'─' * 32}")
    for rank, (bname, bc) in enumerate(corr_ranked, 1):
        marker = ' <--' if bname == 'phi' else ''
        P(f"  {bname:<8} {bc['mean_abs_corr']:>7.2f}% "
          f"{bc['n_sig']:>3}/{bc['n_pos']:<3}  {rank:>4}/9{marker}")

    # ══════════════════════════════════════════════════════════════
    # PART 2: PER-DATASET ENRICHMENT PROFILES (phi)
    # ══════════════════════════════════════════════════════════════

    P(f"\n{'═' * 72}")
    P("  PART 2: PHI ENRICHMENT PROFILES PER DATASET")
    P(f"{'═' * 72}")

    n_ds = len(dataset_ratios)
    n_pos = len(pos_names_sorted)
    phi_enrichment_matrix = np.zeros((n_ds, n_pos))

    # Precompute per-dataset enrichments
    per_ds_enrich = {}
    for i, (dname, d_ratios) in enumerate(dataset_ratios.items()):
        u_d = (np.log(d_ratios) / np.log(PHI)) % 1.0
        enrich = enrichment_at_positions(
            u_d, POSITIONS_14, base_expected['phi'])
        per_ds_enrich[dname] = enrich
        for j, pname in enumerate(pos_names_sorted):
            phi_enrichment_matrix[i, j] = enrich[pname]

    # Short dataset names for table header
    ds_short = {d: d[:9] for d in dataset_ratios}

    header = f"  {'Position':<13}"
    for dname in dataset_ratios:
        header += f" {ds_short[dname]:>9}"
    P(f"\n{header}")
    sep = f"  {'─' * 13}" + f" {'─' * 9}" * n_ds
    P(sep)

    for j, (pname, u_val) in enumerate(sorted_pos):
        row = f"  {pname:<13}"
        for dname in dataset_ratios:
            row += f" {per_ds_enrich[dname][pname]:>+8.1f}%"
        P(row)

    phi_W, phi_p = kendalls_w(phi_enrichment_matrix)
    P(f"\n  Kendall's W (phi, {n_pos} positions, {n_ds} datasets): "
      f"{phi_W:.4f}  (p={phi_p:.2e})")

    # Sign consistency: how many positions have same sign across all datasets?
    sign_consistent = 0
    for j in range(n_pos):
        col = phi_enrichment_matrix[:, j]
        if np.all(col > 0) or np.all(col < 0):
            sign_consistent += 1
    P(f"  Sign-consistent positions: {sign_consistent}/{n_pos}")

    # ══════════════════════════════════════════════════════════════
    # PART 3: CROSS-BASE CONCORDANCE COMPARISON (degree-7)
    # ══════════════════════════════════════════════════════════════

    P(f"\n{'═' * 72}")
    P("  PART 3: CROSS-BASE ENRICHMENT CONCORDANCE (degree-7)")
    P(f"{'═' * 72}")
    P(f"\n  Which base's ratio-enrichment profile replicates best?")

    base_concordances = {}

    for bname, bval in BASES.items():
        t1 = time.time()
        positions = base_positions[bname]
        pos_names_b = list(positions.keys())
        n_pos_b = len(pos_names_b)

        enrichment_matrix = np.zeros((n_ds, n_pos_b))
        expected = base_expected[bname]

        for i, (dname, d_ratios) in enumerate(dataset_ratios.items()):
            u_d = (np.log(d_ratios) / np.log(bval)) % 1.0
            enrich = enrichment_at_positions(u_d, positions, expected)
            for j, pn in enumerate(pos_names_b):
                enrichment_matrix[i, j] = enrich[pn]

        W, p = kendalls_w(enrichment_matrix)
        mean_abs_e = np.mean(np.abs(enrichment_matrix))

        # Sign consistency
        n_sign = 0
        for j in range(n_pos_b):
            col = enrichment_matrix[:, j]
            if np.all(col > 0) or np.all(col < 0):
                n_sign += 1

        base_concordances[bname] = {
            'W': W, 'p': p,
            'n_positions': n_pos_b,
            'mean_abs_enrich': mean_abs_e,
            'sign_consistent': n_sign,
        }

    ranked = sorted(base_concordances.items(),
                    key=lambda x: x[1]['W'], reverse=True)

    P(f"\n  {'Base':<8} {'W':>8} {'p-value':>10} {'#Pos':>5} "
      f"{'Mean|E|':>8} {'SignCon':>8} {'Rank':>5}")
    P(f"  {'─' * 56}")
    for rank, (bname, bc) in enumerate(ranked, 1):
        marker = ' <--' if bname == 'phi' else ''
        P(f"  {bname:<8} {bc['W']:>8.4f} {bc['p']:>10.2e} "
          f"{bc['n_positions']:>5} {bc['mean_abs_enrich']:>7.1f}% "
          f"{bc['sign_consistent']:>4}/{bc['n_positions']:<3} "
          f"{rank:>4}/9{marker}")

    phi_rank = [i for i, (b, _) in enumerate(ranked, 1) if b == 'phi'][0]
    P(f"\n  Phi concordance rank: {phi_rank}/9")

    # ══════════════════════════════════════════════════════════════
    # PART 4: OCTAVE-DISTANCE STRATIFICATION (phi)
    # ══════════════════════════════════════════════════════════════

    P(f"\n{'═' * 72}")
    P("  PART 4: OCTAVE-DISTANCE STRATIFICATION (phi lattice)")
    P(f"{'═' * 72}")
    P(f"\n  n_oct = floor(log_phi(ratio))")
    P(f"  n=0: same octave (ratio in [1, {PHI:.3f}))")
    P(f"  n=1: adjacent octaves (ratio in [{PHI:.3f}, {PHI**2:.3f}))")
    P(f"  n=2: 2 octaves apart (ratio in [{PHI**2:.3f}, {PHI**3:.3f}))")

    log_r = np.log(all_ratios) / np.log(PHI)
    n_oct = np.floor(log_r).astype(int)
    u_frac = log_r % 1.0

    for oct_dist in [0, 1, 2]:
        mask = n_oct == oct_dist
        n_in = np.sum(mask)
        if n_in < 1000:
            P(f"\n  n_oct={oct_dist}: {n_in} ratios (insufficient)")
            continue

        u_oct = u_frac[mask]
        oct_enrich = enrichment_at_positions(
            u_oct, POSITIONS_14, base_expected['phi'])

        r_lo = PHI ** oct_dist
        r_hi = PHI ** (oct_dist + 1)
        P(f"\n  n_oct={oct_dist}: {n_in:,} ratios "
          f"(ratio [{r_lo:.3f}, {r_hi:.3f}))")
        P(f"    {'Position':<15} {'u':>6} {'Enrich%':>9}")
        P(f"    {'─' * 34}")
        for name, u_val in sorted_pos:
            e = oct_enrich[name]
            P(f"    {name:<15} {u_val:>6.3f} {e:>+8.1f}%")

    # Per-octave cross-dataset concordance
    P(f"\n  Cross-dataset concordance by octave distance:")
    for oct_dist in [0, 1, 2]:
        mask_all = n_oct == oct_dist
        if np.sum(mask_all) < 1000:
            continue

        oct_matrix = np.zeros((n_ds, n_pos))
        offset = 0
        for i, (dname, d_ratios) in enumerate(dataset_ratios.items()):
            d_log = np.log(d_ratios) / np.log(PHI)
            d_noct = np.floor(d_log).astype(int)
            d_mask = d_noct == oct_dist
            if np.sum(d_mask) < 100:
                oct_matrix[i, :] = np.nan
                continue
            d_u = d_log[d_mask] % 1.0
            enrich = enrichment_at_positions(
                d_u, POSITIONS_14, base_expected['phi'])
            for j, pn in enumerate(pos_names_sorted):
                oct_matrix[i, j] = enrich[pn]

        # Remove datasets with NaN
        valid = ~np.any(np.isnan(oct_matrix), axis=1)
        if np.sum(valid) >= 2:
            W_oct, p_oct = kendalls_w(oct_matrix[valid])
            P(f"    n_oct={oct_dist}: W={W_oct:.4f} (p={p_oct:.2e}), "
              f"{np.sum(mask_all):,} ratios")
        else:
            P(f"    n_oct={oct_dist}: insufficient datasets")

    # ══════════════════════════════════════════════════════════════
    # PART 5: POSITION-MATCHED COMPARISON (phi vs competitors)
    # ══════════════════════════════════════════════════════════════

    P(f"\n{'═' * 72}")
    P("  PART 5: ENRICHMENT PROFILE SHAPE COMPARISON")
    P(f"{'═' * 72}")
    P(f"\n  Does the phi ratio-enrichment profile match the known")
    P(f"  frequency-position enrichment profile?")
    P(f"\n  Known from frequency analysis:")
    P(f"    boundary: enriched (theta convergence on f0)")
    P(f"    attractor: enriched (alpha/IAF near attractor)")
    P(f"    noble_1: enriched (IAF cluster)")
    P(f"    noble_2-5: gradient (decreasing enrichment)")

    # Compare ratio enrichment to frequency enrichment pattern
    P(f"\n  Ratio enrichment at key positions (pooled):")
    key_positions = [
        ('boundary', 'Expected: enriched'),
        ('noble_2', 'Expected: enriched'),
        ('attractor', 'Expected: enriched'),
        ('noble_1', 'Expected: enriched'),
        ('inv_noble_3', 'Expected: variable'),
        ('inv_noble_5', 'Expected: variable'),
    ]
    for pname, expected in key_positions:
        e = phi_enrich_pooled[pname]
        P(f"    {pname:<15} {e:>+8.1f}%  ({expected})")

    # ══════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════

    P(f"\n{'═' * 72}")
    P("  SUMMARY")
    P(f"{'═' * 72}")

    P(f"\n  Phi 14-position ratio enrichment:")
    P(f"    Mean |enrichment|:     {mean_abs:.1f}%")
    P(f"    Cross-dataset W:      {phi_W:.4f} (p={phi_p:.2e})")
    P(f"    Sign-consistent:      {sign_consistent}/{n_pos} positions")
    P(f"    Concordance rank:     {phi_rank}/9")

    top_base, top_bc = ranked[0]
    P(f"\n  Best replicating base: {top_base} "
      f"(W={top_bc['W']:.4f}, {top_bc['n_positions']} positions)")

    if phi_rank <= 3:
        P(f"\n  RESULT: Phi ratio-enrichment profile shows structured,")
        P(f"  replicable patterns — ranks top 3 for concordance.")
    elif phi_W > 0.3 and phi_p < 0.05:
        P(f"\n  RESULT: Phi ratio-enrichment profile shows significant")
        P(f"  concordance but is not the top-ranked base.")
    else:
        P(f"\n  RESULT: Phi ratio-enrichment profile does not show")
        P(f"  robust replication across datasets.")

    elapsed = time.time() - t0
    P(f"\n  Total time: {elapsed:.1f}s")

    outfile = os.path.join(BASE_DIR, 'ratio_lattice_enrichment_results.txt')
    with open(outfile, 'w') as f:
        f.write('\n'.join(lines))
    P(f"  Results written to {outfile}")


if __name__ == '__main__':
    np.random.seed(42)
    main()
