#!/usr/bin/env python
"""
Shape Replication Inference
============================

Formal inferential framework for the cross-dataset enrichment architecture.
Tests whether the SHAPE (rank ordering of position enrichments within each band)
replicates across independent datasets, not just individual cells.

Framework:
  1. Per-band Kendall's W (concordance) with permutation p-value
  2. Pairwise Spearman rank correlations between datasets
  3. Leave-one-out shape prediction accuracy
  4. Permutation null for global predictability count
  5. Independent-datasets-only (4 EC) replication of all tests

Usage:
    python scripts/shape_replication_inference.py
"""
import os, re, sys, time, io
import numpy as np
import pandas as pd
from scipy import stats

os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'lib')
from phi_replication import BASES, F0, PHI, lattice_coord, KDE_BANDWIDTH

# ═══════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════
POSITIONS_14 = [
    'boundary', 'noble_7', 'noble_6', 'noble_5', 'noble_4',
    'noble_3', 'noble_2', 'attractor', 'noble_1',
    'inv_noble_3', 'inv_noble_4', 'inv_noble_5', 'inv_noble_6', 'inv_noble_7',
]

PHI_OCTAVE_BANDS = ['phi_-2', 'phi_-1', 'phi_0', 'phi_1', 'phi_2', 'phi_3', 'phi_4']
CONV_BANDS = ['delta', 'theta', 'alpha', 'beta_low', 'beta_high', 'gamma']

ALL_DATASETS_PHI = ['EEGMMIDB EC', 'Dortmund EC', 'Dortmund EO', 'HBN EC', 'LEMON EC', 'LEMON EO']
EC_ONLY = ['EEGMMIDB EC', 'Dortmund EC', 'HBN EC', 'LEMON EC']

ALL_DATASETS_CONV = [
    'EEGMMIDB EC', 'Dortmund EC-pre', 'Dortmund EO-pre', 'HBN EC', 'LEMON EC', 'LEMON EO',
]
EC_ONLY_CONV = ['EEGMMIDB EC', 'Dortmund EC-pre', 'HBN EC', 'LEMON EC']

CONV_TO_UNIFIED = {
    'EEGMMIDB EC': 'EEGMMIDB EC',
    'Dortmund EC-pre': 'Dortmund EC',
    'Dortmund EO-pre': 'Dortmund EO',
    'HBN EC': 'HBN EC',
    'LEMON EC': 'LEMON EC',
    'LEMON EO': 'LEMON EO',
}


# ═══════════════════════════════════════════════════════════════════
# PARSERS (copied from cross_dataset_position_consistency.py)
# ═══════════════════════════════════════════════════════════════════
def parse_phi_octave_tables(filepath):
    with open(filepath) as f:
        text = f.read()
    blocks = re.split(r'={50,}\n\s+(.+?)\s+\(N=\d+\)\n={50,}', text)
    data = {}
    for i in range(1, len(blocks), 2):
        ds_name = blocks[i].strip()
        block = blocks[i + 1]
        ds_data = {}
        for line in block.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('Position') or line.startswith('-') or line.startswith('n_peaks'):
                continue
            # Skip sub-header lines (Hz ranges)
            if line[0].isdigit() or line[0] == ' ':
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            pos = parts[0]
            if pos not in POSITIONS_14:
                continue
            vals = {}
            for j, band in enumerate(PHI_OCTAVE_BANDS):
                idx = 2 + j  # skip pos, u
                if idx < len(parts):
                    raw = parts[idx].replace('%', '').replace('+', '')
                    if raw == '---':
                        vals[band] = np.nan
                    else:
                        try:
                            vals[band] = float(raw)
                        except ValueError:
                            vals[band] = np.nan
                else:
                    vals[band] = np.nan
            ds_data[pos] = vals
        if ds_data:
            data[ds_name] = ds_data
    return data


def parse_conventional_tables(filepath):
    with open(filepath) as f:
        text = f.read()
    pattern = r'### ([^\n]+?)\s+\(N=\d+\)\s*\n```\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    data = {}
    for ds_name, block in matches:
        ds_name = ds_name.strip()
        ds_data = {}
        for line in block.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('Position') or line.startswith('position') or line.startswith('-'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            pos = parts[0].strip()
            if pos not in POSITIONS_14:
                continue
            vals = {}
            band_cols = ['delta', 'theta', 'alpha', 'beta_low', 'beta_high', 'gamma']
            for bi, band in enumerate(band_cols):
                idx = 2 + bi  # skip pos, u
                if idx < len(parts):
                    raw = parts[idx].replace('%', '').replace('+', '')
                    if raw == '---':
                        vals[band] = np.nan
                    else:
                        try:
                            vals[band] = float(raw)
                        except ValueError:
                            vals[band] = np.nan
                else:
                    vals[band] = np.nan
            ds_data[pos] = vals
        if ds_data:
            data[ds_name] = ds_data
    return data


# ═══════════════════════════════════════════════════════════════════
# BUILD ENRICHMENT MATRIX
# ═══════════════════════════════════════════════════════════════════
def build_matrix(data, datasets, bands, positions=POSITIONS_14):
    """Build 3D array: datasets × positions × bands.

    Returns: matrix (n_datasets, n_positions, n_bands), with NaN for missing.
    """
    n_ds = len(datasets)
    n_pos = len(positions)
    n_bands = len(bands)
    M = np.full((n_ds, n_pos, n_bands), np.nan)

    for di, ds in enumerate(datasets):
        if ds not in data:
            continue
        for pi, pos in enumerate(positions):
            if pos not in data[ds]:
                continue
            for bi, band in enumerate(bands):
                val = data[ds][pos].get(band, np.nan)
                M[di, pi, bi] = val

    return M


# ═══════════════════════════════════════════════════════════════════
# KENDALL'S W (Concordance)
# ═══════════════════════════════════════════════════════════════════
def kendall_w(rankings):
    """Compute Kendall's W for k raters ranking n items.

    rankings: array of shape (k, n) — each row is one rater's ranking.
    Returns: W, chi2, p_value
    """
    k, n = rankings.shape
    # Rank each row
    ranked = np.zeros_like(rankings, dtype=float)
    for i in range(k):
        ranked[i] = stats.rankdata(rankings[i])

    # Sum of ranks per item
    R = ranked.sum(axis=0)
    R_mean = R.mean()
    S = np.sum((R - R_mean) ** 2)

    W = 12 * S / (k**2 * (n**3 - n))
    chi2 = k * (n - 1) * W
    p = 1.0 - stats.chi2.cdf(chi2, n - 1)
    return W, chi2, p


def kendall_w_permutation(M_band, n_perm=10000, seed=42):
    """Permutation test for Kendall's W.

    M_band: (n_datasets, n_positions) enrichment values for one band.
    Permutes position labels independently within each dataset.
    """
    # Remove datasets/positions with all NaN
    valid_mask = ~np.all(np.isnan(M_band), axis=1)
    M = M_band[valid_mask]
    if M.shape[0] < 3:
        return np.nan, np.nan, np.nan, 0

    # Remove positions with any NaN
    pos_mask = ~np.any(np.isnan(M), axis=0)
    M = M[:, pos_mask]
    if M.shape[1] < 3:
        return np.nan, np.nan, np.nan, 0

    k, n = M.shape
    obs_W, obs_chi2, obs_p_chi2 = kendall_w(M)

    rng = np.random.RandomState(seed)
    null_Ws = np.empty(n_perm)
    for pi in range(n_perm):
        M_perm = M.copy()
        for i in range(k):
            rng.shuffle(M_perm[i])
        null_Ws[pi], _, _ = kendall_w(M_perm)

    p_perm = np.mean(null_Ws >= obs_W)
    return obs_W, obs_chi2, p_perm, n


# ═══════════════════════════════════════════════════════════════════
# PAIRWISE SPEARMAN CORRELATIONS
# ═══════════════════════════════════════════════════════════════════
def pairwise_spearman(M_band, dataset_names):
    """Pairwise Spearman rank correlations between datasets for one band.

    M_band: (n_datasets, n_positions)
    Returns list of (ds_i, ds_j, rho, p)
    """
    n_ds = M_band.shape[0]
    results = []
    for i in range(n_ds):
        for j in range(i + 1, n_ds):
            vi = M_band[i]
            vj = M_band[j]
            # Shared non-NaN positions
            mask = ~(np.isnan(vi) | np.isnan(vj))
            if mask.sum() < 5:
                continue
            rho, p = stats.spearmanr(vi[mask], vj[mask])
            results.append((dataset_names[i], dataset_names[j], rho, p))
    return results


# ═══════════════════════════════════════════════════════════════════
# LEAVE-ONE-OUT SIGN PREDICTION
# ═══════════════════════════════════════════════════════════════════
def leave_one_out_prediction(M_band, dataset_names):
    """For each dataset, predict sign of each cell from mean of other datasets.

    Returns: accuracy, n_predictions
    """
    n_ds, n_pos = M_band.shape
    correct = 0
    total = 0

    for left_out in range(n_ds):
        others = [i for i in range(n_ds) if i != left_out]
        for p in range(n_pos):
            val = M_band[left_out, p]
            if np.isnan(val):
                continue
            other_vals = [M_band[i, p] for i in others if not np.isnan(M_band[i, p])]
            if len(other_vals) < 2:
                continue
            predicted_sign = np.sign(np.mean(other_vals))
            actual_sign = np.sign(val)
            if predicted_sign == actual_sign:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else np.nan
    return accuracy, total


# ═══════════════════════════════════════════════════════════════════
# GLOBAL PREDICTABILITY PERMUTATION TEST
# ═══════════════════════════════════════════════════════════════════
def global_predictability_test(M, bands, n_perm=10000, seed=42):
    """Permutation test for the total number of predictable cells.

    Null: for each band, independently permute position labels within each dataset.
    Count how many cells show ≥(N-1)/N same-sign agreement.

    M: (n_datasets, n_positions, n_bands)
    """
    n_ds, n_pos, n_bands_dim = M.shape

    def count_predictable(M_input):
        count = 0
        for bi in range(n_bands_dim):
            for pi in range(n_pos):
                vals = M_input[:, pi, bi]
                valid = vals[~np.isnan(vals)]
                if len(valid) < 3:
                    continue
                n_pos_sign = np.sum(valid > 0)
                n_neg_sign = np.sum(valid < 0)
                n_total = len(valid)
                if max(n_pos_sign, n_neg_sign) >= n_total - 1 and abs(np.mean(valid)) > 10:
                    count += 1
        return count

    obs_count = count_predictable(M)

    rng = np.random.RandomState(seed)
    null_counts = np.empty(n_perm)
    for pi in range(n_perm):
        M_perm = M.copy()
        for bi in range(n_bands_dim):
            for di in range(n_ds):
                # Shuffle position labels within this dataset × band
                col = M_perm[di, :, bi].copy()
                valid_mask = ~np.isnan(col)
                valid_vals = col[valid_mask]
                rng.shuffle(valid_vals)
                col[valid_mask] = valid_vals
                M_perm[di, :, bi] = col
        null_counts[pi] = count_predictable(M_perm)

    p = np.mean(null_counts >= obs_count)
    null_mean = np.mean(null_counts)
    null_sd = np.std(null_counts)
    z = (obs_count - null_mean) / null_sd if null_sd > 0 else 0

    return obs_count, p, null_mean, null_sd, z, null_counts


# ═══════════════════════════════════════════════════════════════════
# CROSS-DATASET MATRIX CORRELATION
# ═══════════════════════════════════════════════════════════════════
def matrix_correlation(M, dataset_names):
    """Pairwise correlations of full enrichment vectors (all bands flattened).

    M: (n_datasets, n_positions, n_bands)
    """
    n_ds = M.shape[0]
    results = []
    for i in range(n_ds):
        for j in range(i + 1, n_ds):
            vi = M[i].flatten()
            vj = M[j].flatten()
            mask = ~(np.isnan(vi) | np.isnan(vj))
            if mask.sum() < 10:
                continue
            rho, p = stats.spearmanr(vi[mask], vj[mask])
            results.append((dataset_names[i], dataset_names[j], rho, p))
    return results


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    t0 = time.time()
    output = io.StringIO()

    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
        def flush(self):
            for s in self.streams:
                s.flush()

    tee = Tee(sys.stdout, output)
    old_stdout = sys.stdout
    sys.stdout = tee

    print("Shape Replication Inference")
    print("=" * 90)

    # ── Parse data ──
    phi_data = parse_phi_octave_tables('phi_octave_enrichment_tables.txt')
    print(f"\nPhi-octave datasets found: {list(phi_data.keys())}")

    conv_data = parse_conventional_tables('replication_pipeline.md')
    print(f"Conventional datasets found: {list(conv_data.keys())}")

    # ══════════════════════════════════════════════════════════════
    # PART 1: PHI-OCTAVE BANDS
    # ══════════════════════════════════════════════════════════════
    for ds_set_name, ds_list in [('All 6 conditions', ALL_DATASETS_PHI),
                                  ('4 EC only (independent)', EC_ONLY)]:
        available = [ds for ds in ds_list if ds in phi_data]
        M = build_matrix(phi_data, available, PHI_OCTAVE_BANDS)

        print(f"\n\n{'#'*90}")
        print(f"  PHI-OCTAVE BANDS — {ds_set_name} ({len(available)} datasets)")
        print(f"{'#'*90}")

        # ── Test 1: Per-band Kendall's W ──
        print(f"\n{'='*90}")
        print(f"  TEST 1: PER-BAND KENDALL'S W (CONCORDANCE)")
        print(f"{'='*90}")
        print(f"  W=1.0 means perfect agreement on rank ordering of positions.")
        print(f"  W=0.0 means no agreement (random).\n")
        print(f"  {'Band':<10s}  {'N_pos':>5s}  {'W':>6s}  {'chi2':>8s}  {'p_perm':>10s}  interpretation")
        print(f"  {'-'*70}")

        w_values = []
        for bi, band in enumerate(PHI_OCTAVE_BANDS):
            M_band = M[:, :, bi]
            W, chi2, p_perm, n_eff = kendall_w_permutation(M_band, n_perm=10000)
            if np.isnan(W):
                print(f"  {band:<10s}  {'—':>5s}  {'—':>6s}  {'—':>8s}  {'—':>10s}  insufficient data")
                continue
            interp = 'strong' if W > 0.7 else ('moderate' if W > 0.4 else ('weak' if W > 0.2 else 'none'))
            sig = '***' if p_perm < 0.001 else ('**' if p_perm < 0.01 else ('*' if p_perm < 0.05 else 'ns'))
            print(f"  {band:<10s}  {n_eff:>5d}  {W:>6.3f}  {chi2:>8.1f}  {p_perm:>10.4f}  {interp} {sig}")
            w_values.append((band, W, p_perm))

        if w_values:
            mean_W = np.mean([w for _, w, _ in w_values])
            sig_bands = sum(1 for _, _, p in w_values if p < 0.05)
            print(f"\n  Mean W = {mean_W:.3f}, {sig_bands}/{len(w_values)} bands significant at p<0.05")

        # ── Test 2: Pairwise Spearman per band ──
        print(f"\n{'='*90}")
        print(f"  TEST 2: PAIRWISE SPEARMAN RANK CORRELATIONS")
        print(f"{'='*90}")
        print(f"  How similar is each pair of datasets' enrichment profile?\n")

        for bi, band in enumerate(PHI_OCTAVE_BANDS):
            M_band = M[:, :, bi]
            pairs = pairwise_spearman(M_band, available)
            if not pairs:
                continue

            rhos = [rho for _, _, rho, _ in pairs]
            mean_rho = np.mean(rhos)
            n_sig = sum(1 for _, _, _, p in pairs if p < 0.05)
            print(f"  {band:<10s}: mean ρ = {mean_rho:+.3f}, {n_sig}/{len(pairs)} pairs significant")

            # Show EC-only pairs for independent datasets
            ec_pairs = [(a, b, rho, p) for a, b, rho, p in pairs
                        if a in EC_ONLY and b in EC_ONLY]
            if ec_pairs:
                ec_mean = np.mean([rho for _, _, rho, _ in ec_pairs])
                ec_sig = sum(1 for _, _, _, p in ec_pairs if p < 0.05)
                print(f"  {'':10s}  EC-only: mean ρ = {ec_mean:+.3f}, {ec_sig}/{len(ec_pairs)} sig")

        # ── Test 3: Leave-one-out sign prediction ──
        print(f"\n{'='*90}")
        print(f"  TEST 3: LEAVE-ONE-OUT SIGN PREDICTION")
        print(f"{'='*90}")
        print(f"  Can one dataset's enrichment sign be predicted from the others?\n")
        print(f"  {'Band':<10s}  {'accuracy':>8s}  {'N_pred':>7s}  {'p(binom)':>10s}  interp")
        print(f"  {'-'*55}")

        total_correct = 0
        total_predictions = 0
        for bi, band in enumerate(PHI_OCTAVE_BANDS):
            M_band = M[:, :, bi]
            acc, n_pred = leave_one_out_prediction(M_band, available)
            if np.isnan(acc) or n_pred == 0:
                continue
            correct = int(round(acc * n_pred))
            p_binom = 1 - stats.binom.cdf(correct - 1, n_pred, 0.5)
            sig = '***' if p_binom < 0.001 else ('**' if p_binom < 0.01 else ('*' if p_binom < 0.05 else 'ns'))
            print(f"  {band:<10s}  {acc:>7.1%}  {n_pred:>7d}  {p_binom:>10.2e}  {sig}")
            total_correct += correct
            total_predictions += n_pred

        if total_predictions > 0:
            overall_acc = total_correct / total_predictions
            p_overall = 1 - stats.binom.cdf(total_correct - 1, total_predictions, 0.5)
            print(f"\n  Overall: {overall_acc:.1%} ({total_correct}/{total_predictions}), p = {p_overall:.2e}")

        # ── Test 4: Global predictability permutation test ──
        print(f"\n{'='*90}")
        print(f"  TEST 4: GLOBAL PREDICTABILITY PERMUTATION TEST")
        print(f"{'='*90}")
        print(f"  Null: shuffle position labels within each dataset × band.\n")

        obs_count, p_global, null_mean, null_sd, z_global, null_dist = \
            global_predictability_test(M, PHI_OCTAVE_BANDS, n_perm=10000)
        print(f"  Observed predictable cells: {obs_count}")
        print(f"  Null distribution: mean={null_mean:.1f}, SD={null_sd:.1f}")
        print(f"  z = {z_global:+.2f}, p = {p_global:.4f}")
        pcts = np.percentile(null_dist, [50, 90, 95, 99, 99.9])
        print(f"  Null percentiles: 50th={pcts[0]:.0f}, 90th={pcts[1]:.0f}, 95th={pcts[2]:.0f}, 99th={pcts[3]:.0f}, 99.9th={pcts[4]:.0f}")

        # ── Test 5: Matrix-level correlation ──
        print(f"\n{'='*90}")
        print(f"  TEST 5: FULL MATRIX CORRELATION (ALL BANDS)")
        print(f"{'='*90}")
        print(f"  Spearman correlation of full 14×7 enrichment vectors between datasets.\n")

        pairs = matrix_correlation(M, available)
        if pairs:
            print(f"  {'Dataset A':<16s}  {'Dataset B':<16s}  {'ρ':>6s}  {'p':>10s}")
            print(f"  {'-'*55}")
            for a, b, rho, p in sorted(pairs, key=lambda x: -x[2]):
                sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
                print(f"  {a:<16s}  {b:<16s}  {rho:+.3f}  {p:>9.2e} {sig}")

            rhos = [rho for _, _, rho, _ in pairs]
            print(f"\n  Mean ρ = {np.mean(rhos):+.3f} (range: {min(rhos):+.3f} to {max(rhos):+.3f})")

            ec_pairs = [(a, b, rho, p) for a, b, rho, p in pairs
                        if a in EC_ONLY and b in EC_ONLY]
            if ec_pairs:
                ec_rhos = [rho for _, _, rho, _ in ec_pairs]
                print(f"  EC-only mean ρ = {np.mean(ec_rhos):+.3f} ({len(ec_pairs)} pairs)")

    # ══════════════════════════════════════════════════════════════
    # PART 2: CONVENTIONAL BANDS
    # ══════════════════════════════════════════════════════════════
    conv_data_unified = {}
    for ds_name, ds_data in conv_data.items():
        unified = CONV_TO_UNIFIED.get(ds_name, ds_name)
        conv_data_unified[unified] = ds_data

    for ds_set_name, ds_list_raw in [('All 6 conditions', ALL_DATASETS_CONV)]:
        ds_list = [CONV_TO_UNIFIED.get(d, d) for d in ds_list_raw]
        available = [ds for ds in ds_list if ds in conv_data_unified]
        M = build_matrix(conv_data_unified, available, CONV_BANDS)

        print(f"\n\n{'#'*90}")
        print(f"  CONVENTIONAL BANDS — {ds_set_name} ({len(available)} datasets)")
        print(f"{'#'*90}")

        # Kendall's W
        print(f"\n{'='*90}")
        print(f"  PER-BAND KENDALL'S W")
        print(f"{'='*90}\n")
        print(f"  {'Band':<10s}  {'N_pos':>5s}  {'W':>6s}  {'p_perm':>10s}")
        print(f"  {'-'*40}")
        for bi, band in enumerate(CONV_BANDS):
            M_band = M[:, :, bi]
            W, chi2, p_perm, n_eff = kendall_w_permutation(M_band, n_perm=10000)
            if np.isnan(W):
                print(f"  {band:<10s}  insufficient data")
                continue
            print(f"  {band:<10s}  {n_eff:>5d}  {W:>6.3f}  {p_perm:>10.4f}")

        # Leave-one-out
        print(f"\n  LEAVE-ONE-OUT SIGN PREDICTION:")
        print(f"  {'Band':<10s}  {'accuracy':>8s}  {'N_pred':>7s}  {'p(binom)':>10s}")
        print(f"  {'-'*45}")
        total_c = 0
        total_n = 0
        for bi, band in enumerate(CONV_BANDS):
            M_band = M[:, :, bi]
            acc, n_pred = leave_one_out_prediction(M_band, available)
            if np.isnan(acc) or n_pred == 0:
                continue
            correct = int(round(acc * n_pred))
            p_binom = 1 - stats.binom.cdf(correct - 1, n_pred, 0.5)
            print(f"  {band:<10s}  {acc:>7.1%}  {n_pred:>7d}  {p_binom:>10.2e}")
            total_c += correct
            total_n += n_pred
        if total_n > 0:
            print(f"\n  Overall: {total_c/total_n:.1%} ({total_c}/{total_n}), p = {1-stats.binom.cdf(total_c-1, total_n, 0.5):.2e}")

        # Global test
        print(f"\n  GLOBAL PREDICTABILITY:")
        obs, p_g, nm, ns, z_g, _ = global_predictability_test(M, CONV_BANDS, n_perm=10000)
        print(f"  Observed={obs}, null mean={nm:.1f}±{ns:.1f}, z={z_g:+.2f}, p={p_g:.4f}")

    # ══════════════════════════════════════════════════════════════
    # PART 3: CROSS-BASE SHAPE COMPARISON
    # ══════════════════════════════════════════════════════════════
    # The critical test: does phi's shape replicate BETTER than other bases?
    # Computes enrichment from raw per-subject peaks for all 9 bases,
    # then compares concordance (Kendall's W) and predictability.

    print(f"\n\n{'#'*90}")
    print(f"  PART 3: CROSS-BASE SHAPE COMPARISON")
    print(f"  (Does phi's enrichment shape replicate better than other bases?)")
    print(f"{'#'*90}")

    # ── Load per-subject peak CSVs ──
    DATASET_CSVS = {
        'EEGMMIDB EC': 'exports_eegmmidb/replication/combined/per_subject_dominant_peaks.csv',
        'LEMON EC': 'exports_lemon/replication/EC/per_subject_dominant_peaks.csv',
        'LEMON EO': 'exports_lemon/replication/EO/per_subject_dominant_peaks.csv',
        'HBN EC': 'exports_hbn/EC/per_subject_dominant_peaks.csv',
        'Dortmund EC': '/Volumes/T9/dortmund_data/lattice_results_replication_v2/EyesClosed_pre/per_subject_dominant_peaks.csv',
        'Dortmund EO': '/Volumes/T9/dortmund_data/lattice_results_replication_v2/EyesOpen_pre/per_subject_dominant_peaks.csv',
    }
    BAND_LIST = ['delta', 'theta', 'alpha', 'beta_low', 'beta_high', 'gamma']

    peak_data = {}
    for ds_name, path in DATASET_CSVS.items():
        if os.path.exists(path):
            peak_data[ds_name] = pd.read_csv(path)

    datasets_avail = [ds for ds in DATASET_CSVS if ds in peak_data]
    print(f"\n  Loaded {len(datasets_avail)} datasets: {datasets_avail}")
    for ds in datasets_avail:
        print(f"    {ds}: N={len(peak_data[ds])}")

    # ── Position generator for any base at degree 7 ──
    def positions_degree_n(base, max_k=7, min_sep=0.02):
        pos = {'boundary': 0.0, 'attractor': 0.5}
        inv = 1.0 / base
        for k in range(1, max_k + 1):
            val = (inv ** k) % 1.0
            inv_val = (1 - inv ** k) % 1.0

            def _can_add(v):
                if v < min_sep or v > 1 - min_sep:
                    return False
                if abs(v - 0.5) < min_sep:
                    return False
                return all(abs(v - ex) > min_sep for ex in pos.values())

            if _can_add(val):
                pos[f'noble_{k}'] = val
            if _can_add(inv_val):
                pos[f'inv_noble_{k}'] = inv_val
        return pos

    # ── Uniform expected density for KDE ──
    bw = KDE_BANDWIDTH  # 0.03
    uniform_expected = 2 * bw * np.sqrt(np.pi / 2)  # exact for narrow bw on [0,1) circle

    # ── Compute enrichment for each base ──
    print(f"\n  Computing enrichment profiles for 9 bases (degree-7 positions)...")

    base_results = {}
    for base_name, base_val in BASES.items():
        positions = positions_degree_n(base_val, max_k=7)
        pos_names = sorted(positions.keys(), key=lambda n: positions[n])
        pos_vals = [positions[n] for n in pos_names]
        n_pos = len(pos_names)

        # Build enrichment matrix: datasets × positions × bands
        M = np.full((len(datasets_avail), n_pos, len(BAND_LIST)), np.nan)

        for di, ds_name in enumerate(datasets_avail):
            df = peak_data[ds_name]
            for bi, band in enumerate(BAND_LIST):
                freq_col = f'{band}_freq'
                if freq_col not in df.columns:
                    continue
                freqs = df[freq_col].dropna().values
                if len(freqs) < 10:
                    continue
                # Lattice coordinates for this base
                us = np.array([lattice_coord(f, F0, base_val) for f in freqs])
                us = us[~np.isnan(us)]
                if len(us) < 10:
                    continue
                # KDE density at each position
                for pi, pval in enumerate(pos_vals):
                    dists = np.abs(us - pval)
                    dists = np.minimum(dists, 1 - dists)
                    density = np.exp(-0.5 * (dists / bw)**2).mean()
                    M[di, pi, bi] = (density / uniform_expected - 1) * 100

        # Kendall's W per band
        w_values = []
        for bi in range(len(BAND_LIST)):
            M_band = M[:, :, bi]
            W, chi2, p_perm, n_eff = kendall_w_permutation(M_band, n_perm=2000, seed=42)
            if not np.isnan(W):
                w_values.append(W)
        mean_W = np.mean(w_values) if w_values else np.nan

        # Global predictability
        obs_count, p_global, null_mean, null_sd, z_global, _ = \
            global_predictability_test(M, BAND_LIST, n_perm=2000, seed=42)

        # Leave-one-out accuracy
        total_correct = 0
        total_preds = 0
        for bi in range(len(BAND_LIST)):
            M_band = M[:, :, bi]
            acc, n_pred = leave_one_out_prediction(M_band, datasets_avail)
            if not np.isnan(acc) and n_pred > 0:
                total_correct += int(round(acc * n_pred))
                total_preds += n_pred
        loo_acc = total_correct / total_preds if total_preds > 0 else np.nan

        # Full matrix Spearman (mean pairwise ρ)
        pairs = matrix_correlation(M, datasets_avail)
        mean_rho = np.mean([rho for _, _, rho, _ in pairs]) if pairs else np.nan

        base_results[base_name] = {
            'n_positions': n_pos,
            'mean_W': mean_W,
            'w_per_band': w_values,
            'obs_predictable': obs_count,
            'null_mean': null_mean,
            'null_sd': null_sd,
            'z_predictable': z_global,
            'loo_accuracy': loo_acc,
            'mean_rho': mean_rho,
        }
        print(f"    {base_name:>5s}: {n_pos} positions, mean W={mean_W:.3f}, "
              f"predictable={obs_count}, z={z_global:+.2f}, LOO={loo_acc:.1%}")

    # ── Summary table sorted by mean W ──
    print(f"\n{'='*90}")
    print(f"  CROSS-BASE SHAPE COMPARISON — SUMMARY")
    print(f"{'='*90}")
    print(f"  Conventional bands (6), degree-7 positions, {len(datasets_avail)} datasets\n")
    print(f"  {'Base':<8s}  {'N_pos':>5s}  {'Mean W':>7s}  {'Rank':>4s}  "
          f"{'LOO':>6s}  {'Predict':>7s}  {'z':>6s}  {'Mean ρ':>7s}")
    print(f"  {'-'*65}")

    sorted_bases = sorted(base_results, key=lambda b: -base_results[b]['mean_W'])
    for rank, base_name in enumerate(sorted_bases, 1):
        r = base_results[base_name]
        marker = '  ◄' if base_name == 'phi' else ''
        print(f"  {base_name:<8s}  {r['n_positions']:>5d}  {r['mean_W']:>7.3f}  "
              f"{'#'+str(rank):>4s}  {r['loo_accuracy']:>5.1%}  "
              f"{r['obs_predictable']:>7d}  {r['z_predictable']:>+5.2f}  "
              f"{r['mean_rho']:>+6.3f}{marker}")

    phi_rank = sorted_bases.index('phi') + 1
    print(f"\n  Phi shape-replication rank: #{phi_rank}/9")

    # ── Per-band W comparison (phi vs best non-phi) ──
    print(f"\n{'='*90}")
    print(f"  PER-BAND KENDALL'S W — ALL 9 BASES")
    print(f"{'='*90}\n")

    # Collect per-band W for all bases
    print(f"  {'Band':<10s}", end='')
    for bn in sorted_bases:
        print(f"  {bn:>7s}", end='')
    print()
    print(f"  {'-'*(10 + 9*9)}")

    # Re-run per-band W for each base (use cached M matrices)
    # Need to recompute — store band-level W during main loop
    # Actually, we stored w_per_band but not labeled. Let me recompute cleanly.
    band_w_table = {}
    for base_name, base_val in BASES.items():
        positions = positions_degree_n(base_val, max_k=7)
        pos_names = sorted(positions.keys(), key=lambda n: positions[n])
        pos_vals = [positions[n] for n in pos_names]
        n_pos = len(pos_names)

        M = np.full((len(datasets_avail), n_pos, len(BAND_LIST)), np.nan)
        for di, ds_name in enumerate(datasets_avail):
            df = peak_data[ds_name]
            for bi, band in enumerate(BAND_LIST):
                freq_col = f'{band}_freq'
                if freq_col not in df.columns:
                    continue
                freqs = df[freq_col].dropna().values
                if len(freqs) < 10:
                    continue
                us = np.array([lattice_coord(f, F0, base_val) for f in freqs])
                us = us[~np.isnan(us)]
                if len(us) < 10:
                    continue
                for pi, pval in enumerate(pos_vals):
                    dists = np.abs(us - pval)
                    dists = np.minimum(dists, 1 - dists)
                    density = np.exp(-0.5 * (dists / bw)**2).mean()
                    M[di, pi, bi] = (density / uniform_expected - 1) * 100

        for bi, band in enumerate(BAND_LIST):
            M_band = M[:, :, bi]
            W, _, p_perm, n_eff = kendall_w_permutation(M_band, n_perm=2000, seed=42)
            if band not in band_w_table:
                band_w_table[band] = {}
            band_w_table[band][base_name] = W

    for band in BAND_LIST:
        print(f"  {band:<10s}", end='')
        ws = band_w_table.get(band, {})
        best_val = max((w for w in ws.values() if not np.isnan(w)), default=0)
        for bn in sorted_bases:
            w = ws.get(bn, np.nan)
            marker = '*' if (not np.isnan(w) and abs(w - best_val) < 0.001) else ' '
            if np.isnan(w):
                print(f"  {'—':>7s}", end='')
            else:
                print(f"  {w:>6.3f}{marker}", end='')
        print()

    print(f"\n  * = highest W for that band")

    # ── Verdict ──
    print(f"\n{'='*90}")
    print(f"  VERDICT")
    print(f"{'='*90}")
    phi_W = base_results['phi']['mean_W']
    best_non_phi = max((r['mean_W'] for bn, r in base_results.items() if bn != 'phi'),
                       default=0)
    best_non_phi_name = [bn for bn, r in base_results.items()
                         if bn != 'phi' and r['mean_W'] == best_non_phi][0]
    print(f"  Phi mean W = {phi_W:.3f} (rank #{phi_rank}/9)")
    print(f"  Best non-phi = {best_non_phi_name} (W = {best_non_phi:.3f})")
    print(f"  Difference = {phi_W - best_non_phi:+.3f}")
    if phi_rank == 1:
        print(f"\n  Phi's enrichment shape replicates BETTER than all other bases.")
        print(f"  The shape consistency is phi-specific, not just a consequence of peak stability.")
    elif phi_rank <= 3:
        print(f"\n  Phi ranks #{phi_rank}/9 — competitive but not uniquely best.")
        print(f"  Shape consistency is partially phi-specific.")
    else:
        print(f"\n  Phi ranks #{phi_rank}/9 — shape consistency is NOT phi-specific.")
        print(f"  Any lattice on stable peaks would produce similar concordance.")

    elapsed = time.time() - t0
    print(f"\n\nDone in {elapsed:.0f}s")

    sys.stdout = old_stdout
    with open('shape_replication_inference.txt', 'w') as f:
        f.write(output.getvalue())
    print(f"Results saved to shape_replication_inference.txt")
