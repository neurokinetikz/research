#!/usr/bin/env python
"""
Per-band structure test for logarithmic EEG band structure.

Steps 2-4: Phi-only analysis at phi's 14 degree-7 positions — the primary result.
Step 5: Log-frequency spectral analysis (parameter-free):
  Detects periodicity in the peak frequency distribution via autocorrelation
  in log-frequency space. No assumed base, f₀, positions, or boundaries.
  If periodicity exists, its period P maps to base β = e^P.

Usage:
    python scripts/run_band_structure_test.py
"""
import sys, os, glob, time
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from phi_replication import (
    F0, PHI, POSITIONS_14,
    BASES,
    lattice_coord, circ_dist,
    positions_for_base,
)

# ═══════════════════════════════════════════════════════════════════
# PHI-OCTAVE BANDS
# ═══════════════════════════════════════════════════════════════════
EXTRACT_LO, EXTRACT_HI = 1.0, 45.0

PHI_BANDS = {}
for n in range(-2, 5):
    lo = F0 * PHI ** (n - 1)
    hi = F0 * PHI ** n
    eff_lo = max(lo, EXTRACT_LO)
    eff_hi = min(hi, EXTRACT_HI)
    if eff_hi <= eff_lo:
        continue
    coverage = (eff_hi - eff_lo) / (hi - lo)
    PHI_BANDS[f'phi_{n}'] = {
        'lo': lo, 'hi': hi,
        'eff_lo': eff_lo, 'eff_hi': eff_hi,
        'coverage': coverage,
    }

# Primary analysis: 5 phi-octave bands with full coverage
PRIMARY_BANDS = ['phi_-1', 'phi_0', 'phi_1', 'phi_2', 'phi_3']

# Standard EEG bands (theory-neutral, same for ALL bases in cross-base comparison)
STANDARD_BANDS = [
    {'label': 'delta', 'eff_lo':  1.0, 'eff_hi':  4.0},
    {'label': 'theta', 'eff_lo':  4.0, 'eff_hi':  8.0},
    {'label': 'alpha', 'eff_lo':  8.0, 'eff_hi': 13.0},
    {'label': 'beta',  'eff_lo': 13.0, 'eff_hi': 30.0},
    {'label': 'gamma', 'eff_lo': 30.0, 'eff_hi': 45.0},
]
STD_BAND_LABELS = [b['label'] for b in STANDARD_BANDS]

# ═══════════════════════════════════════════════════════════════════
# DATASETS
# ═══════════════════════════════════════════════════════════════════
DATASETS = {
    'EEGMMIDB EC': {
        'peaks_dir': 'exports_eegmmidb/replication/combined/per_subject_peaks',
    },
    'Dortmund EC': {
        'peaks_dir': '/Volumes/T9/dortmund_data/lattice_results_replication_v2/EyesClosed_pre/per_subject_peaks',
    },
    'Dortmund EO': {
        'peaks_dir': '/Volumes/T9/dortmund_data/lattice_results_replication_v2/EyesOpen_pre/per_subject_peaks',
    },
    'HBN EC': {
        'peaks_dir': 'exports_hbn/combined/per_subject_peaks',
    },
    'LEMON EC': {
        'peaks_dir': 'exports_lemon/replication/EC/per_subject_peaks',
    },
    'LEMON EO': {
        'peaks_dir': 'exports_lemon/replication/EO/per_subject_peaks',
    },
}

# ═══════════════════════════════════════════════════════════════════
# KDE ENRICHMENT (generic — works with any position set)
# ═══════════════════════════════════════════════════════════════════
KDE_BW = 0.03

def kde_density(u_values, pos, bw=KDE_BW):
    """Gaussian KDE density at a lattice position."""
    dists = np.array([circ_dist(u, pos) for u in u_values])
    return np.exp(-0.5 * (dists / bw) ** 2).mean()

def uniform_expected_density(pos, bw=KDE_BW, n_samples=100000):
    """Expected density under uniform distribution on [0,1)."""
    u_uniform = np.linspace(0, 1, n_samples)
    dists = np.minimum(np.abs(u_uniform - pos), 1 - np.abs(u_uniform - pos))
    return np.exp(-0.5 * (dists / bw) ** 2).mean()

def compute_null_densities(positions):
    """Precompute null densities for a set of positions."""
    return {pname: uniform_expected_density(pval) for pname, pval in positions.items()}

def compute_enrichment_profile(u_values, positions, null_densities):
    """Compute enrichment % at each position for an array of u-values."""
    profile = np.empty(len(positions))
    for i, (pname, pval) in enumerate(positions.items()):
        if len(u_values) < 5:
            profile[i] = np.nan
        else:
            obs = kde_density(u_values, pval)
            null = null_densities[pname]
            profile[i] = ((obs / null) - 1) * 100 if null > 0 else 0
    return profile

# ═══════════════════════════════════════════════════════════════════
# DOMINANT PEAK EXTRACTION (returns frequencies, not u-values)
# ═══════════════════════════════════════════════════════════════════
def extract_band_freqs(peak_files, bands_list):
    """Extract dominant peak FREQUENCIES per band from raw peak CSVs.

    Returns: dict[band_label] -> np.array of frequencies (one per subject)
    """
    band_freqs = {b: [] for b in bands_list}

    for pf in peak_files:
        df = pd.read_csv(pf)
        if len(df) == 0:
            continue

        for blabel in bands_list:
            binfo = PHI_BANDS[blabel]
            eff_lo, eff_hi = binfo['eff_lo'], binfo['eff_hi']
            bp = df[(df['freq'] >= eff_lo) & (df['freq'] < eff_hi)]
            if len(bp) > 0:
                idx = bp['power'].idxmax()
                band_freqs[blabel].append(bp.loc[idx, 'freq'])

    return {b: np.array(fs) for b, fs in band_freqs.items()}

def extract_band_freqs_generic(peak_files, bands):
    """Extract dominant peak FREQUENCIES per band from raw peak CSVs.

    Args:
        peak_files: list of CSV paths
        bands: list of dicts with 'label', 'eff_lo', 'eff_hi'

    Returns: dict[band_label] -> np.array of frequencies (one per subject)
    """
    band_freqs = {b['label']: [] for b in bands}
    for pf in peak_files:
        df = pd.read_csv(pf)
        if len(df) == 0:
            continue
        for binfo in bands:
            eff_lo, eff_hi = binfo['eff_lo'], binfo['eff_hi']
            bp = df[(df['freq'] >= eff_lo) & (df['freq'] < eff_hi)]
            if len(bp) > 0:
                idx = bp['power'].idxmax()
                band_freqs[binfo['label']].append(bp.loc[idx, 'freq'])
    return {b: np.array(fs) for b, fs in band_freqs.items()}


def freqs_to_u(freqs, base):
    """Convert frequency array to u-values using a specific base."""
    return np.array([lattice_coord(f, f0=F0, base=base) for f in freqs])


# ═══════════════════════════════════════════════════════════════════
# DEGREE-7 POSITIONS (for cross-base comparison)
# ═══════════════════════════════════════════════════════════════════
def positions_for_base_deg7(base):
    """Generate degree-7 lattice positions for any base.

    boundary + attractor + noble_k/inv_noble_k for k=1..7,
    filtered by MIN_SEP=0.02 circular distance.
    """
    MIN_SEP = 0.02
    inv = 1.0 / base
    pos = {'boundary': 0.0, 'attractor': 0.5}

    def _try_add(name, val):
        if val < MIN_SEP or val > 1 - MIN_SEP:
            return
        if abs(val - 0.5) < MIN_SEP:
            return
        if all(abs(val - v) > MIN_SEP for v in pos.values()):
            pos[name] = val

    for k in range(1, 8):
        _try_add(f'noble_{k}', inv ** k)
        _try_add(f'inv_noble_{k}', 1 - inv ** k)

    return dict(sorted(pos.items(), key=lambda x: x[1]))



# ═══════════════════════════════════════════════════════════════════
# VECTORIZED STACKED ENRICHMENT (fast path for permutations)
# ═══════════════════════════════════════════════════════════════════
def compute_stacked_fast(band_us, pos_vals, null_vals, band_labels):
    """Compute stacked enrichment vector from per-band u-values.

    Args:
        band_us: dict {band_label: np.array of u-values}
        pos_vals: np.array of position values (P,)
        null_vals: np.array of null densities (P,)
        band_labels: list of band labels in stacking order

    Returns: 1D array (n_bands * n_pos)
    """
    profiles = []
    for bl in band_labels:
        us = band_us.get(bl, np.array([]))
        if len(us) >= 5:
            d = np.abs(us[:, None] - pos_vals[None, :])
            dists = np.minimum(d, 1 - d)
            kde = np.exp(-0.5 * (dists / KDE_BW) ** 2).mean(axis=0)
            profiles.append(((kde / null_vals) - 1) * 100)
        else:
            profiles.append(np.full(len(pos_vals), np.nan))
    return np.concatenate(profiles)


def compute_stacked_perm(band_ns, pos_vals, null_vals,
                         band_labels, rng):
    """Compute stacked enrichment with uniform-random u-values.

    Args:
        band_ns: dict {band_label: int (count of u-values)}
        pos_vals, null_vals: arrays from precomputation
        band_labels: stacking order
        rng: np.random.RandomState
    """
    profiles = []
    for bl in band_labels:
        n = band_ns.get(bl, 0)
        if n >= 5:
            us = rng.uniform(0, 1, n)
            d = np.abs(us[:, None] - pos_vals[None, :])
            dists = np.minimum(d, 1 - d)
            kde = np.exp(-0.5 * (dists / KDE_BW) ** 2).mean(axis=0)
            profiles.append(((kde / null_vals) - 1) * 100)
        else:
            profiles.append(np.full(len(pos_vals), np.nan))
    return np.concatenate(profiles)



# ═══════════════════════════════════════════════════════════════════
# BAND BOUNDARY FIT (mathematical — no data needed, kept for reference)
# ═══════════════════════════════════════════════════════════════════
CANONICAL_EDGES = [1.0, 4.0, 8.0, 13.0, 30.0, 45.0]


def compute_boundary_fit(base, f0=F0, canonical_edges=None,
                         include_attractors=False):
    """How well do β-octave boundaries match canonical EEG band edges?

    Args:
        include_attractors: if True, also include half-octave (attractor)
            positions f0·β^(n+0.5) as candidate band edges.

    Returns dict with 'rmse', 'edges' (per-edge match detail),
    'n_octaves' (count of β-octaves in [1,45]), 'boundaries'.
    """
    if canonical_edges is None:
        canonical_edges = CANONICAL_EDGES
    # Boundaries: f0·β^n
    boundaries = sorted(
        f0 * base ** n
        for n in range(-20, 21)
        if 0.5 <= f0 * base ** n <= 50.0)
    if include_attractors:
        # Attractors: f0·β^(n+0.5) — half-octave midpoints
        attractors = sorted(
            f0 * base ** (n + 0.5)
            for n in range(-20, 21)
            if 0.5 <= f0 * base ** (n + 0.5) <= 50.0)
        candidates = sorted(set(boundaries + attractors))
    else:
        candidates = boundaries
    log_dists = []
    matched = []
    for ce in canonical_edges:
        dists = [abs(np.log(b / ce)) for b in candidates]
        best_idx = int(np.argmin(dists))
        log_dists.append(dists[best_idx])
        matched.append((ce, candidates[best_idx], dists[best_idx]))
    rmse = np.sqrt(np.mean(np.array(log_dists) ** 2))
    return {'rmse': rmse, 'edges': matched,
            'n_octaves': len(boundaries) - 1, 'boundaries': boundaries,
            'n_candidates': len(candidates)}


# ═══════════════════════════════════════════════════════════════════
# LOG-FREQUENCY SPECTRAL ANALYSIS (parameter-free period detection)
# ═══════════════════════════════════════════════════════════════════
LOG_DENSITY_BW = 0.05  # KDE bandwidth in log-frequency space
LOG_GRID_N = 2000      # Grid resolution

# Known base periods in log-frequency space
KNOWN_BASES = {
    '√2':  (np.sqrt(2), np.log(np.sqrt(2))),  # 0.347
    '3/2': (1.5, np.log(1.5)),                  # 0.405
    'φ':   (PHI, np.log(PHI)),                  # 0.481
    '2':   (2.0, np.log(2.0)),                  # 0.693
    'e':   (np.e, np.log(np.e)),                # 1.000
    'π':   (np.pi, np.log(np.pi)),              # 1.145
}


def compute_log_density(freqs, x_grid, bw=LOG_DENSITY_BW):
    """KDE of peak frequencies in log-space.

    Args:
        freqs: array of frequencies (Hz, NOT log-transformed)
        x_grid: grid of log-frequency values to evaluate density on
        bw: Gaussian kernel bandwidth in log-frequency units
    """
    log_f = np.log(freqs[freqs > 0])
    # Vectorized Gaussian KDE
    diffs = x_grid[:, None] - log_f[None, :]  # (grid, freqs)
    density = np.exp(-0.5 * (diffs / bw) ** 2).sum(axis=1)
    density /= (len(log_f) * bw * np.sqrt(2 * np.pi))
    return density


def compute_acf(density):
    """Autocorrelation via FFT, normalized so ACF(0) = 1."""
    d = density - density.mean()
    n = len(d)
    f = np.fft.fft(d, n=2 * n)
    acf = np.fft.ifft(f * np.conj(f)).real[:n]
    if acf[0] > 0:
        acf /= acf[0]
    return acf


def find_acf_peaks(acf, lag_axis, min_lag=0.2, max_lag=1.5):
    """Find local maxima in the ACF within the specified lag range."""
    mask = (lag_axis >= min_lag) & (lag_axis <= max_lag)
    acf_win = acf[mask]
    lag_win = lag_axis[mask]
    # Simple peak detection: points higher than both neighbors
    peaks = []
    for i in range(1, len(acf_win) - 1):
        if acf_win[i] > acf_win[i - 1] and acf_win[i] > acf_win[i + 1]:
            peaks.append((lag_win[i], acf_win[i]))
    peaks.sort(key=lambda x: -x[1])  # sort by height
    return peaks


# ═══════════════════════════════════════════════════════════════════
# V-TEST (Circular mean direction test)
# ═══════════════════════════════════════════════════════════════════
def v_test(u_values, predicted_mode):
    """V-test for mean direction at a predicted mode on [0,1) circle."""
    n = len(u_values)
    if n < 5:
        return np.nan, np.nan
    angles = 2 * np.pi * u_values
    mu0 = 2 * np.pi * predicted_mode
    C = np.cos(angles).mean()
    S = np.sin(angles).mean()
    R_bar = np.sqrt(C**2 + S**2)
    theta_bar = np.arctan2(S, C)
    V = R_bar * np.cos(theta_bar - mu0)
    z = np.sqrt(2 * n) * V
    p = 1 - stats.norm.cdf(z)
    return V, p

# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    t0 = time.time()
    np.random.seed(42)
    N_PERM = 5000

    print("Per-Band Structure Test: Phi-Octave Template Correlation")
    print("=" * 100)
    print(f"f₀ = {F0} Hz, base = φ = {PHI:.6f}")
    print(f"KDE bandwidth = {KDE_BW}")
    print(f"Primary bands: {PRIMARY_BANDS}")
    print(f"Permutations: {N_PERM}")

    # ── Phi 14-position setup (for Steps 2-4: phi-only analysis) ──
    print(f"\nPhi degree-7 positions (for phi-only analysis): {len(POSITIONS_14)} positions")
    print(f"  [{', '.join(f'{v:.3f}' for v in POSITIONS_14.values())}]")

    # ── Step 1: Load peak frequencies from all datasets ──
    print("\n" + "=" * 100)
    print("STEP 1: Loading dominant peak frequencies from all datasets")
    print("=" * 100)

    all_band_freqs = {}   # {ds_name: {band: freq array}}
    all_peak_files = {}

    for ds_name, ds_info in DATASETS.items():
        peaks_dir = ds_info['peaks_dir']
        peak_files = sorted(glob.glob(os.path.join(peaks_dir, '*_peaks.csv')))

        if not peak_files:
            print(f"  *** {ds_name}: no peak files at {peaks_dir}")
            continue

        all_peak_files[ds_name] = peak_files
        band_freqs = extract_band_freqs(peak_files, PRIMARY_BANDS)
        all_band_freqs[ds_name] = band_freqs
        print(f"  {ds_name}: {len(peak_files)} subjects, "
              f"peaks/band: {', '.join(f'{b}={len(band_freqs[b])}' for b in PRIMARY_BANDS)}")

    ds_names = list(all_band_freqs.keys())

    # ── Step 2: Phi-only template correlation (using phi's 14 positions) ──
    print("\n" + "=" * 100)
    print("STEP 2: Phi template correlation (LOO, phi's 14 positions)")
    print("=" * 100)

    phi_positions = POSITIONS_14
    phi_nulls = compute_null_densities(phi_positions)

    # Compute phi enrichment profiles for all datasets
    phi_profiles = {}  # {ds: {band: profile}}
    phi_band_us = {}   # {ds: {band: u-values}}
    for ds in ds_names:
        phi_profiles[ds] = {}
        phi_band_us[ds] = {}
        for blabel in PRIMARY_BANDS:
            us = freqs_to_u(all_band_freqs[ds][blabel], PHI)
            phi_band_us[ds][blabel] = us
            phi_profiles[ds][blabel] = compute_enrichment_profile(us, phi_positions, phi_nulls)

    # LOO template r
    hdr = f"{'Band':<10s}"
    for ds in ds_names:
        hdr += f"  {ds:>14s}"
    hdr += f"  {'mean_r':>8s}"
    print(hdr)
    print("-" * len(hdr))

    for blabel in PRIMARY_BANDS:
        row = f"{blabel:<10s}"
        rs = []
        for i, ds in enumerate(ds_names):
            other = [phi_profiles[d][blabel] for j, d in enumerate(ds_names) if j != i]
            template = np.nanmean(other, axis=0)
            observed = phi_profiles[ds][blabel]
            valid = ~np.isnan(observed) & ~np.isnan(template)
            if valid.sum() < 5:
                row += f"  {'N/A':>14s}"
                continue
            r, _ = stats.pearsonr(observed[valid], template[valid])
            rs.append(r)
            row += f"  {r:>+14.3f}"
        mean_r = np.mean(rs) if rs else np.nan
        row += f"  {mean_r:>+8.3f}"
        print(row)

    # ── Step 3: Permutation p-values (phi only) ──
    print("\n" + "=" * 100)
    print("STEP 3: Permutation p-values (phi, 5K shuffles)")
    print("=" * 100)

    hdr = f"{'Band':<10s}"
    for ds in ds_names:
        hdr += f"  {ds:>14s}"
    print(hdr)
    print("-" * len(hdr))

    perm_results = {}

    for blabel in PRIMARY_BANDS:
        row = f"{blabel:<10s}"
        for i, ds in enumerate(ds_names):
            us = phi_band_us[ds][blabel]
            if len(us) < 10:
                row += f"  {'N/A':>14s}"
                continue

            other = [phi_profiles[d][blabel] for j, d in enumerate(ds_names) if j != i]
            template = np.nanmean(other, axis=0)
            observed = phi_profiles[ds][blabel]
            valid = ~np.isnan(observed) & ~np.isnan(template)
            if valid.sum() < 5:
                row += f"  {'N/A':>14s}"
                continue

            obs_r, _ = stats.pearsonr(observed[valid], template[valid])

            null_rs = np.empty(N_PERM)
            rng = np.random.RandomState(42)
            for p_i in range(N_PERM):
                shuffled_u = rng.uniform(0, 1, len(us))
                perm_profile = compute_enrichment_profile(shuffled_u, phi_positions, phi_nulls)
                perm_valid = ~np.isnan(perm_profile) & ~np.isnan(template)
                if perm_valid.sum() >= 5:
                    null_rs[p_i], _ = stats.pearsonr(perm_profile[perm_valid], template[perm_valid])
                else:
                    null_rs[p_i] = 0.0

            p_perm = (np.sum(null_rs >= obs_r) + 1) / (N_PERM + 1)
            perm_results[(blabel, ds)] = (obs_r, p_perm)
            row += f"  {p_perm:>14.4f}"
        print(row)

    # ── Step 4: Fisher-combined p ──
    print("\n" + "=" * 100)
    print("STEP 4: Aggregate scores per dataset (phi)")
    print("=" * 100)

    print(f"\n{'Dataset':<18s}  {'mean_r':>8s}  {'Fisher_chi2':>11s}  {'Fisher_p':>10s}  "
          f"{'n_sig':>5s}")
    print("-" * 70)

    for ds in ds_names:
        rs = []
        ps = []
        for blabel in PRIMARY_BANDS:
            key = (blabel, ds)
            if key in perm_results:
                obs_r, p = perm_results[key]
                rs.append(obs_r)
                ps.append(max(p, 1e-10))

        if len(ps) >= 2:
            chi2 = -2 * np.sum(np.log(ps))
            fisher_p = 1 - stats.chi2.cdf(chi2, df=2 * len(ps))
            n_sig = sum(1 for p in ps if p < 0.05)
            mean_r = np.mean(rs)
            print(f"{ds:<18s}  {mean_r:>+8.3f}  {chi2:>11.1f}  {fisher_p:>10.2e}  "
                  f"{n_sig:>5d}/5")
        else:
            print(f"{ds:<18s}  {'N/A':>8s}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 5: LOG-FREQUENCY SPECTRAL ANALYSIS (parameter-free)
    # Detect periodicity in peak density via autocorrelation in
    # log-frequency space. No assumed base, f₀, positions, or boundaries.
    # If periodicity exists, its period P maps to base β = e^P.
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("STEP 5: Log-Frequency Spectral Analysis (parameter-free)")
    print("=" * 100)
    print("Detect periodicity in peak density via autocorrelation.")
    print("No assumed base, f₀, positions, or boundaries.")
    print(f"ACF peak at lag τ → periodicity with base β = e^τ\n")

    # ── 5a: Extract standard-band peaks ──
    print(f"{'─' * 100}")
    print("5a: Standard-band peak extraction")
    print(f"{'─' * 100}")
    print("Standard bands: " + ", ".join(
        "{} [{:.0f}-{:.0f}]".format(b['label'], b['eff_lo'], b['eff_hi'])
        for b in STANDARD_BANDS))

    std_band_freqs = {}
    for ds in ds_names:
        std_band_freqs[ds] = extract_band_freqs_generic(
            all_peak_files[ds], STANDARD_BANDS)
        counts = ", ".join(
            "{}={}".format(bl, len(std_band_freqs[ds][bl]))
            for bl in STD_BAND_LABELS)
        print(f"  {ds}: {counts}")

    # ── 5b: Phi's 14-position enrichment (for Step 6) ──
    phi_pos_arr = np.array(list(POSITIONS_14.values()))
    phi_null_dict = compute_null_densities(POSITIONS_14)
    phi_null_arr = np.array(
        [phi_null_dict[k] for k in POSITIONS_14.keys()])
    phi_std_profiles = {}  # {ds: {band: enrichment at 14 pos}}

    for ds in ds_names:
        bus = {}
        for bl in STD_BAND_LABELS:
            freqs = std_band_freqs[ds][bl]
            if len(freqs) >= 5:
                bus[bl] = freqs_to_u(freqs, PHI)
            else:
                bus[bl] = np.array([])
        sv = compute_stacked_fast(
            bus, phi_pos_arr, phi_null_arr, STD_BAND_LABELS)
        phi_std_profiles[ds] = {}
        n14 = len(phi_pos_arr)
        for bi, bl in enumerate(STD_BAND_LABELS):
            phi_std_profiles[ds][bl] = sv[bi * n14:(bi + 1) * n14]

    # ── 5c: Per-dataset density in log-frequency space ──
    print(f"\n{'─' * 100}")
    print("5c: Log-frequency density (per dataset)")
    print(f"{'─' * 100}")

    x_lo, x_hi = np.log(1.0), np.log(45.0)
    x_grid = np.linspace(x_lo, x_hi, LOG_GRID_N)
    dx = x_grid[1] - x_grid[0]

    ds_densities = {}   # {ds: density array}
    ds_all_freqs = {}   # {ds: pooled freq array}

    for ds in ds_names:
        # Pool all dominant peak frequencies across bands
        all_freqs = np.concatenate(
            [std_band_freqs[ds][bl] for bl in STD_BAND_LABELS
             if len(std_band_freqs[ds][bl]) > 0])
        ds_all_freqs[ds] = all_freqs
        density = compute_log_density(all_freqs, x_grid)
        ds_densities[ds] = density
        print(f"  {ds}: {len(all_freqs)} peaks, "
              f"density range [{density.min():.3f}, {density.max():.3f}]")

    # ── 5d: Autocorrelation analysis ──
    print(f"\n{'─' * 100}")
    print("5d: Autocorrelation of log-frequency density")
    print(f"{'─' * 100}")

    lag_axis = np.arange(LOG_GRID_N) * dx  # in log-frequency units
    ds_acfs = {}  # {ds: acf array}

    print(f"\nKnown base periods (lag = ln(base)):")
    for bname, (bval, bperiod) in KNOWN_BASES.items():
        print(f"  {bname:<4s}: ln({bval:.4f}) = {bperiod:.4f}")

    print(f"\nACF at known base periods:")
    hdr = f"  {'Base':<4s}  {'lag':>6s}"
    for ds in ds_names:
        hdr += f"  {ds[:12]:>12s}"
    hdr += f"  {'mean':>8s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for bname, (bval, bperiod) in KNOWN_BASES.items():
        row = f"  {bname:<4s}  {bperiod:>6.3f}"
        vals = []
        for ds in ds_names:
            if ds not in ds_acfs:
                ds_acfs[ds] = compute_acf(ds_densities[ds])
            acf = ds_acfs[ds]
            # Interpolate ACF at exact lag
            i_lag = int(round(bperiod / dx))
            if i_lag < len(acf):
                acf_val = acf[i_lag]
            else:
                acf_val = np.nan
            vals.append(acf_val)
            row += f"  {acf_val:>+12.4f}"
        row += f"  {np.nanmean(vals):>+8.4f}"
        print(row)

    # ── 5e: ACF peak detection ──
    print(f"\n{'─' * 100}")
    print("5e: ACF peak detection (strongest peaks in [0.2, 1.5] range)")
    print(f"{'─' * 100}")

    detected_periods = []
    for ds in ds_names:
        acf = ds_acfs[ds]
        peaks = find_acf_peaks(acf, lag_axis, min_lag=0.2, max_lag=1.5)
        print(f"\n  {ds}:")
        if peaks:
            for pi, (lag, height) in enumerate(peaks[:5]):
                base_val = np.exp(lag)
                # Find closest known base
                closest = min(KNOWN_BASES.items(),
                              key=lambda x: abs(x[1][1] - lag))
                dist = abs(closest[1][1] - lag)
                near_str = (f" (near {closest[0]}, Δ={dist:.3f})"
                            if dist < 0.05 else "")
                tag = " <-- strongest" if pi == 0 else ""
                print(f"    peak {pi+1}: lag={lag:.4f}, "
                      f"ACF={height:+.4f}, "
                      f"β=e^{lag:.3f}={base_val:.3f}"
                      f"{near_str}{tag}")
            detected_periods.append(peaks[0][0])
        else:
            print(f"    No peaks found")
            detected_periods.append(np.nan)

    if detected_periods:
        mean_period = np.nanmean(detected_periods)
        std_period = np.nanstd(detected_periods)
        mean_base = np.exp(mean_period)
        closest = min(KNOWN_BASES.items(),
                      key=lambda x: abs(x[1][1] - mean_period))
        print(f"\n  Consensus detected period: "
              f"{mean_period:.4f} ± {std_period:.4f}")
        print(f"  → base β = e^{mean_period:.3f} = {mean_base:.4f}")
        print(f"  Nearest known base: {closest[0]} "
              f"(ln={closest[1][1]:.4f}, Δ={abs(closest[1][1]-mean_period):.4f})")

    # ── 5f: Power spectrum ──
    print(f"\n{'─' * 100}")
    print("5f: Power spectrum of log-frequency density")
    print(f"{'─' * 100}")

    ds_power = {}
    for ds in ds_names:
        d = ds_densities[ds] - ds_densities[ds].mean()
        F = np.fft.rfft(d)
        power = np.abs(F) ** 2
        ds_power[ds] = power

    freq_axis = np.fft.rfftfreq(LOG_GRID_N, d=dx)
    # Convert to period and base
    # freq = cycles per log-Hz unit
    # period = 1/freq = log-Hz units per cycle
    # base = e^period

    # Find spectral peaks (avoid DC at freq=0)
    print(f"\nSpectral peaks (top 3 per dataset, excluding DC):")
    for ds in ds_names:
        power = ds_power[ds]
        # Skip DC (first bin) and very low frequencies
        min_freq_idx = max(1, int(1.0 / (1.5 * dx * LOG_GRID_N / 2)))
        max_freq_idx = min(len(power), int(1.0 / (0.2 * dx)))
        p_win = power[min_freq_idx:max_freq_idx]
        f_win = freq_axis[min_freq_idx:max_freq_idx]
        # Top 3 by power
        top_idx = np.argsort(-p_win)[:3]
        print(f"  {ds}:")
        for ti in top_idx:
            freq = f_win[ti]
            period = 1.0 / freq if freq > 0 else np.inf
            base = np.exp(period) if np.isfinite(period) else np.inf
            closest = min(KNOWN_BASES.items(),
                          key=lambda x: abs(x[1][1] - period))
            dist = abs(closest[1][1] - period)
            near_str = (f" (near {closest[0]})"
                        if dist < 0.05 else "")
            print(f"    freq={freq:.3f} cyc/ln-Hz, "
                  f"period={period:.4f}, "
                  f"β={base:.3f}, "
                  f"power={p_win[ti]:.1f}{near_str}")

    # ── 5g: LOO reproducibility of ACF ──
    print(f"\n{'─' * 100}")
    print("5g: LOO reproducibility of ACF shape")
    print(f"{'─' * 100}")

    # Use ACF in [0.2, 2.0] range for template comparison
    acf_lo = int(0.2 / dx)
    acf_hi = min(int(2.0 / dx), LOG_GRID_N)
    acf_vectors = {ds: ds_acfs[ds][acf_lo:acf_hi] for ds in ds_names}

    loo_rs = []
    for i, ds in enumerate(ds_names):
        other = [acf_vectors[d]
                 for j, d in enumerate(ds_names) if j != i]
        template = np.mean(other, axis=0)
        r, _ = stats.pearsonr(acf_vectors[ds], template)
        loo_rs.append(r)
        print(f"  {ds}: LOO r = {r:+.4f}")
    mean_loo_r = np.mean(loo_rs)
    print(f"  Mean LOO r = {mean_loo_r:+.4f}")

    # ── 5h: Phase extraction (f₀ estimation) ──
    print(f"\n{'─' * 100}")
    print("5h: Phase extraction (f₀ estimation at detected period)")
    print(f"{'─' * 100}")

    if detected_periods and not np.isnan(mean_period):
        # For each dataset, fit a cosine at the detected period
        # density ~ A*cos(2π*x/P + φ) + baseline
        # Phase φ → f₀ = e^(-φ*P/(2π)) (mod P)
        print(f"  Detected period: {mean_period:.4f} "
              f"(base {mean_base:.4f})")
        for ds in ds_names:
            d = ds_densities[ds] - ds_densities[ds].mean()
            # Fourier component at frequency 1/P
            freq_target = 1.0 / mean_period
            # Find nearest FFT bin
            freq_idx = int(round(freq_target / (freq_axis[1]
                           if len(freq_axis) > 1 else 1)))
            if 0 < freq_idx < len(ds_power[ds]):
                F = np.fft.rfft(d)
                phase = np.angle(F[freq_idx])
                # The density peaks at x where cos(2π*ν*x + phase) is max
                # i.e., x_peak = -phase / (2π*ν)
                x_offset = -phase / (2 * np.pi * freq_target)
                # Wrap to [0, P)
                x_offset = x_offset % mean_period
                f0_detected = np.exp(x_offset)
                print(f"  {ds}: phase={phase:+.3f} rad, "
                      f"offset={x_offset:.4f} → "
                      f"f₀={f0_detected:.2f} Hz")

    # ── 5i: Visualization ──
    print(f"\n{'─' * 100}")
    print("5i: Saving spectral analysis visualization")
    print(f"{'─' * 100}")

    colors = plt.cm.tab10(np.linspace(0, 1, len(ds_names)))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Log-Frequency Spectral Analysis: '
                 'Parameter-Free Period Detection',
                 fontsize=13, fontweight='bold')

    # Panel 1: Densities in log-frequency space
    ax = axes[0, 0]
    for ci, ds in enumerate(ds_names):
        # Normalize for overlay
        d = ds_densities[ds]
        d_norm = d / d.max()
        ax.plot(x_grid, d_norm, color=colors[ci],
                label=ds[:12], linewidth=1.2, alpha=0.8)
    # Mark band centers
    for bl, binfo in zip(STD_BAND_LABELS, STANDARD_BANDS):
        x_mid = np.log(np.sqrt(binfo['eff_lo'] * binfo['eff_hi']))
        ax.axvline(x_mid, color='gray', alpha=0.3, linewidth=0.5)
        ax.text(x_mid, 1.05, bl, fontsize=7, ha='center',
                transform=ax.get_xaxis_transform())
    ax.set_xlabel('ln(frequency)')
    ax.set_ylabel('normalized density')
    ax.set_title('Peak density in log-frequency space')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_xlim(x_lo, x_hi)
    # Add Hz scale on top
    ax2 = ax.twiny()
    hz_ticks = [1, 2, 4, 8, 13, 20, 30, 45]
    ax2.set_xlim(x_lo, x_hi)
    ax2.set_xticks([np.log(f) for f in hz_ticks])
    ax2.set_xticklabels([str(f) for f in hz_ticks], fontsize=7)
    ax2.set_xlabel('Hz', fontsize=8)

    # Panel 2: ACF with known base periods marked
    ax = axes[0, 1]
    for ci, ds in enumerate(ds_names):
        acf = ds_acfs[ds]
        mask = lag_axis <= 2.0
        ax.plot(lag_axis[mask], acf[mask], color=colors[ci],
                label=ds[:12], linewidth=1.2, alpha=0.8)
    # Mark known base periods
    for bname, (bval, bperiod) in KNOWN_BASES.items():
        ax.axvline(bperiod, color='red', alpha=0.5, linewidth=1,
                   linestyle='--')
        ax.text(bperiod, 1.02, bname, fontsize=7, ha='center',
                color='red', transform=ax.get_xaxis_transform())
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xlabel('lag τ (log-frequency units)')
    ax.set_ylabel('ACF')
    ax.set_title('Autocorrelation (ACF peak → periodicity)')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_xlim(0, 2.0)

    # Panel 3: Power spectrum
    ax = axes[1, 0]
    for ci, ds in enumerate(ds_names):
        power = ds_power[ds]
        # Plot up to period ~0.2 (freq ~5)
        max_idx = min(len(power),
                      int(1.0 / (0.15 * dx)) + 1)
        # Skip DC
        ax.plot(freq_axis[1:max_idx], power[1:max_idx],
                color=colors[ci], label=ds[:12],
                linewidth=1.2, alpha=0.8)
    # Mark known base frequencies
    for bname, (bval, bperiod) in KNOWN_BASES.items():
        if bperiod > 0.15:
            bf = 1.0 / bperiod
            ax.axvline(bf, color='red', alpha=0.5, linewidth=1,
                       linestyle='--')
            ax.text(bf, 0.98, bname, fontsize=7, ha='center',
                    color='red', transform=ax.get_xaxis_transform())
    ax.set_xlabel('frequency (cycles per ln-Hz)')
    ax.set_ylabel('power')
    ax.set_title('Power spectrum')
    ax.legend(fontsize=7, loc='upper right')

    # Panel 4: ACF value at known base periods (bar chart)
    ax = axes[1, 1]
    base_names = list(KNOWN_BASES.keys())
    x_pos = np.arange(len(base_names))
    bar_width = 0.8 / len(ds_names)
    for ci, ds in enumerate(ds_names):
        acf = ds_acfs[ds]
        acf_vals = []
        for bname, (bval, bperiod) in KNOWN_BASES.items():
            i_lag = int(round(bperiod / dx))
            if i_lag < len(acf):
                acf_vals.append(acf[i_lag])
            else:
                acf_vals.append(0)
        ax.bar(x_pos + ci * bar_width, acf_vals, bar_width,
               color=colors[ci], label=ds[:12], alpha=0.8)
    ax.set_xticks(x_pos + bar_width * (len(ds_names) - 1) / 2)
    ax.set_xticklabels(
        [f"{bn}\n(ln={KNOWN_BASES[bn][1]:.3f})" for bn in base_names],
        fontsize=7)
    ax.set_ylabel('ACF value')
    ax.set_title('ACF at known base periods')
    ax.legend(fontsize=7, loc='upper right')
    ax.axhline(0, color='gray', linewidth=0.5)

    fig.tight_layout()
    fig_path = 'band_structure_test_spectral.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fig_path}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 6: PHI ENRICHMENT PREDICTION TABLE
    # Cross-dataset mean enrichment at 14 positions × 5 standard bands
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("STEP 6: Phi enrichment prediction table "
          "(14 positions × 5 standard bands)")
    print("=" * 100)
    print("Cross-dataset mean enrichment (%). "
          "This is the template that replicates.\n")

    pos_names = list(POSITIONS_14.keys())

    hdr = f"{'Position':<14s} {'u':>6s}"
    for bl in STD_BAND_LABELS:
        hdr += f"  {bl:>8s}"
    print(hdr)
    print("-" * len(hdr))

    for pi, pname in enumerate(pos_names):
        pval = list(POSITIONS_14.values())[pi]
        row = f"{pname:<14s} {pval:>6.3f}"
        for bl in STD_BAND_LABELS:
            vals = [phi_std_profiles[ds][bl][pi]
                    for ds in ds_names
                    if not np.isnan(phi_std_profiles[ds][bl][pi])]
            if vals:
                me = np.mean(vals)
                row += f"  {me:>+8.1f}"
            else:
                row += f"  {'N/A':>8s}"
        print(row)

    # ── Step 7: V-test at predicted modes (phi only) ──
    print("\n" + "=" * 100)
    print("STEP 7: V-test at predicted modes "
          "(phi, phi-octave bands, supplementary)")
    print("=" * 100)

    V_PREDICTIONS = {
        'phi_-1': 0.000,
        'phi_0':  0.000,
        'phi_1':  0.500,
        'phi_2':  0.000,
        'phi_3':  0.000,
    }

    hdr = f"{'Band':<10s} {'mode':>5s}"
    for ds in ds_names:
        hdr += f"  {ds:>14s}"
    print(hdr)
    print("-" * len(hdr))

    for blabel in PRIMARY_BANDS:
        predicted = V_PREDICTIONS[blabel]
        row = f"{blabel:<10s} {predicted:>5.3f}"
        for ds in ds_names:
            us = phi_band_us[ds][blabel]
            V, p = v_test(us, predicted)
            if not np.isnan(V):
                sig = '*' if p < 0.05 else ' '
                row += f"  V={V:+.3f} p={p:.3f}{sig}"
            else:
                row += f"  {'N/A':>14s}"
        print(row)

    # ── Step 8: Segregation check ──
    print("\n" + "=" * 100)
    print("STEP 8: Segregation check "
          "(inter-template correlations, phi-octave bands)")
    print("=" * 100)

    global_templates = {}
    for blabel in PRIMARY_BANDS:
        profiles = [phi_profiles[ds][blabel] for ds in ds_names]
        global_templates[blabel] = np.nanmean(profiles, axis=0)

    print(f"\n{'':>10s}", end="")
    for b2 in PRIMARY_BANDS:
        print(f"  {b2:>8s}", end="")
    print()

    for b1 in PRIMARY_BANDS:
        t1 = global_templates[b1]
        row = f"{b1:>10s}"
        for b2 in PRIMARY_BANDS:
            t2 = global_templates[b2]
            valid = ~np.isnan(t1) & ~np.isnan(t2)
            if valid.sum() >= 5:
                r, _ = stats.pearsonr(t1[valid], t2[valid])
                row += f"  {r:>+8.3f}"
            else:
                row += f"  {'N/A':>8s}"
        print(row)

    # ══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    print("\n1. PHI-ONLY (Steps 2-4): LOO template r at "
          "phi's 14 positions, phi-octave bands")
    hdr = f"   {'Dataset':<16s}"
    for b in PRIMARY_BANDS:
        hdr += f"  {b:>8s}"
    hdr += f"  {'Fisher p':>10s}"
    print(hdr)
    print("   " + "-" * (len(hdr) - 3))

    for ds in ds_names:
        row = f"   {ds:<16s}"
        ps = []
        for blabel in PRIMARY_BANDS:
            key = (blabel, ds)
            if key in perm_results:
                obs_r, p = perm_results[key]
                sig = '**' if p < 0.01 else (
                    '*' if p < 0.05 else ' ')
                row += f"  {obs_r:>+7.3f}{sig}"
                ps.append(max(p, 1e-10))
            else:
                row += f"  {'N/A':>8s}"
        if len(ps) >= 2:
            chi2 = -2 * np.sum(np.log(ps))
            fp = 1 - stats.chi2.cdf(chi2, df=2 * len(ps))
            row += f"  {fp:>10.2e}"
        print(row)

    print(f"\n2. LOG-FREQUENCY SPECTRAL ANALYSIS (Step 5): "
          f"parameter-free period detection")
    if detected_periods and not np.isnan(mean_period):
        print(f"   Detected period: {mean_period:.4f} ± "
              f"{std_period:.4f} (base = {mean_base:.4f})")
        closest = min(KNOWN_BASES.items(),
                      key=lambda x: abs(x[1][1] - mean_period))
        print(f"   Nearest known base: {closest[0]} "
              f"(Δ={abs(closest[1][1]-mean_period):.4f})")
    print(f"   ACF LOO r = {mean_loo_r:+.4f}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
