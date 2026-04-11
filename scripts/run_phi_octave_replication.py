#!/usr/bin/env python
"""
Phi-octave band replication: Cohen's d and cross-base z-rank.

Reads existing raw peak CSVs (from OT extraction) and recomputes
replication statistics using phi-octave-aligned bands instead of
conventional EEG bands. Every phi-octave band spans the full [0,1)
lattice circle, eliminating coverage gaps.

Usage:
    python scripts/run_phi_octave_replication.py
"""
import sys, os, glob, time
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from phi_replication import (
    F0, PHI, BANDS,
    POSITIONS_DEG2, PHI_POSITIONS, POSITIONS_14,
    BASES,
    lattice_coord, min_lattice_dist, nearest_position_name,
    positions_for_base,
)

# ═══════════════════════════════════════════════════════════════════
# PHI-OCTAVE BANDS
# ═══════════════════════════════════════════════════════════════════
EXTRACT_CEIL = 45.0

PHI_BANDS = {}
for n in range(-2, 5):
    lo = F0 * PHI ** (n - 1)
    hi = F0 * PHI ** n
    eff_lo = max(lo, 1.0)
    eff_hi = min(hi, EXTRACT_CEIL)
    if eff_hi <= eff_lo:
        continue
    coverage = (eff_hi - eff_lo) / (hi - lo)
    PHI_BANDS[f'phi_{n}'] = {
        'lo': lo, 'hi': hi,
        'eff_lo': eff_lo, 'eff_hi': eff_hi,
        'coverage': coverage,
    }

# Primary = fully covered bands only
PRIMARY_BANDS = {k: v for k, v in PHI_BANDS.items() if v['coverage'] > 0.99}
ALL_BANDS = PHI_BANDS

# ═══════════════════════════════════════════════════════════════════
# DATASETS
# ═══════════════════════════════════════════════════════════════════
DATASETS = {
    'EEGMMIDB EC': {
        'peaks_dir': 'exports_eegmmidb/replication/combined/per_subject_peaks',
        'conv_csv': 'exports_eegmmidb/replication/EC/summary_statistics.csv',
    },
    'Dortmund EC': {
        'peaks_dir': '/Volumes/T9/dortmund_data/lattice_results_replication_v2/EyesClosed_pre/per_subject_peaks',
        'conv_csv': '/Volumes/T9/dortmund_data/lattice_results_replication_v2/EyesClosed_pre/summary_statistics.csv',
    },
    'Dortmund EO': {
        'peaks_dir': '/Volumes/T9/dortmund_data/lattice_results_replication_v2/EyesOpen_pre/per_subject_peaks',
        'conv_csv': '/Volumes/T9/dortmund_data/lattice_results_replication_v2/EyesOpen_pre/summary_statistics.csv',
    },
    'HBN EC': {
        'peaks_dir': 'exports_hbn/combined/per_subject_peaks',
        'conv_csv': 'exports_hbn/EC/summary_statistics.csv',
    },
    'LEMON EC': {
        'peaks_dir': 'exports_lemon/replication/EC/per_subject_peaks',
        'conv_csv': 'exports_lemon/replication/EC/summary_statistics.csv',
    },
    'LEMON EO': {
        'peaks_dir': 'exports_lemon/replication/EO/per_subject_peaks',
        'conv_csv': 'exports_lemon/replication/EO/summary_statistics.csv',
    },
}


def select_dominant_peaks(peak_files, bands_dict):
    """Select strongest peak per band for each subject.

    Returns DataFrame with columns: subject, {band}_{freq|power|u|d|nearest}, mean_d, n_bands
    """
    rows = []
    for pf in peak_files:
        df = pd.read_csv(pf)
        if len(df) == 0:
            continue

        sub = os.path.basename(pf).replace('_peaks.csv', '')
        row = {'subject': sub}
        n_bands = 0

        for bname, binfo in bands_dict.items():
            eff_lo, eff_hi = binfo['eff_lo'], binfo['eff_hi']
            bp = df[(df['freq'] >= eff_lo) & (df['freq'] < eff_hi)]
            if len(bp) > 0:
                idx = bp['power'].idxmax()
                freq = bp.loc[idx, 'power']  # power for sorting
                freq = bp.loc[idx, 'freq']
                u = lattice_coord(freq, f0=F0, base=PHI)
                d = min_lattice_dist(u, POSITIONS_DEG2)
                nearest = nearest_position_name(u, POSITIONS_DEG2)

                row[f'{bname}_freq'] = freq
                row[f'{bname}_power'] = bp.loc[idx, 'power']
                row[f'{bname}_u'] = u
                row[f'{bname}_d'] = d
                row[f'{bname}_nearest'] = nearest
                n_bands += 1
            else:
                row[f'{bname}_freq'] = np.nan
                row[f'{bname}_power'] = np.nan
                row[f'{bname}_u'] = np.nan
                row[f'{bname}_d'] = np.nan
                row[f'{bname}_nearest'] = 'none'

        row['n_bands'] = n_bands
        d_vals = [row[f'{b}_d'] for b in bands_dict
                  if not np.isnan(row.get(f'{b}_d', np.nan))]
        row['mean_d'] = np.mean(d_vals) if d_vals else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def run_stats(dom_df, bands_dict, label=''):
    """Compute Cohen's d and cross-base z-rank. Mirrors phi_replication.run_statistics()."""
    n_bands_expected = len(bands_dict)

    # Filter to complete-band subjects
    valid = dom_df[dom_df['n_bands'] == n_bands_expected].copy()
    if len(valid) < 30:
        valid = dom_df[dom_df['n_bands'] >= n_bands_expected - 1].copy()

    if len(valid) < 10:
        print(f"  {label}: Too few subjects ({len(valid)})")
        return None

    analysis_df = valid

    # ── Per-subject Cohen's d (POSITIONS_DEG2, 4 positions) ──
    mean_ds = analysis_df['mean_d'].values
    obs_mean = mean_ds.mean()
    obs_sd = mean_ds.std()

    null_expected = np.mean([min_lattice_dist(np.random.uniform(0, 1), POSITIONS_DEG2)
                             for _ in range(100_000)])
    cohen_d = (null_expected - obs_mean) / obs_sd if obs_sd > 0 else 0.0
    t_stat, p_ttest = stats.ttest_1samp(mean_ds, null_expected)
    try:
        _, p_wilcox = stats.wilcoxon(mean_ds - null_expected, alternative='less')
    except Exception:
        p_wilcox = np.nan

    # ── Cross-base z-rank (degree-3, 5K permutations, seed=42) ──
    band_freqs = {}
    for bname in bands_dict:
        vals = analysis_df[f'{bname}_freq'].dropna().values
        band_freqs[bname] = vals

    base_results = {}
    for base_name, base_val in BASES.items():
        positions = positions_for_base(base_val, degree=3)
        seg_ds = []
        for _, srow in analysis_df.iterrows():
            band_ds = []
            for bname in bands_dict:
                freq = srow[f'{bname}_freq']
                if np.isnan(freq):
                    continue
                u = lattice_coord(freq, f0=F0, base=base_val)
                d = min_lattice_dist(u, positions)
                band_ds.append(d)
            if band_ds:
                seg_ds.append(np.mean(band_ds))
        seg_ds = np.array(seg_ds)

        rng = np.random.RandomState(42)
        n_perm = 5000
        null_means = np.empty(n_perm)
        for perm_i in range(n_perm):
            perm_ds = []
            for bname, freqs in band_freqs.items():
                if len(freqs) == 0:
                    continue
                shuffled = rng.uniform(0, 1, len(freqs))
                dists = np.array([min_lattice_dist(u, positions) for u in shuffled])
                perm_ds.append(dists.mean())
            null_means[perm_i] = np.mean(perm_ds) if perm_ds else np.nan

        null_mean = np.nanmean(null_means)
        null_sd = np.nanstd(null_means)
        z_score = (null_mean - seg_ds.mean()) / null_sd if null_sd > 0 else 0.0

        base_results[base_name] = {
            'mean_d': seg_ds.mean(), 'z_score': z_score,
            'n_positions': len(positions), 'values': seg_ds,
        }

    ranking_z = sorted(base_results.items(), key=lambda x: -x[1]['z_score'])
    ranking_raw = sorted(base_results.items(), key=lambda x: x[1]['mean_d'])
    phi_rank_z = next(i+1 for i, (name, _) in enumerate(ranking_z) if name == 'phi')
    phi_rank_raw = next(i+1 for i, (name, _) in enumerate(ranking_raw) if name == 'phi')

    # Top-4 z-scores
    top4 = [(name, br['z_score']) for name, br in ranking_z[:4]]

    return {
        'label': label,
        'n_valid': len(analysis_df),
        'n_bands': n_bands_expected,
        'obs_mean_d': obs_mean,
        'null_expected': null_expected,
        'cohen_d': cohen_d,
        'p_ttest': p_ttest,
        'p_wilcox': p_wilcox,
        'phi_rank_z': phi_rank_z,
        'phi_rank_raw': phi_rank_raw,
        'top4_z': top4,
        'base_results': base_results,
    }


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    t0 = time.time()

    print("Phi-Octave Band Replication")
    print("=" * 80)
    print(f"\nf₀ = {F0} Hz, base = φ = {PHI:.6f}")
    print(f"\nPhi-octave bands:")
    for bname, binfo in PHI_BANDS.items():
        flag = " [PRIMARY]" if binfo['coverage'] > 0.99 else f" [{binfo['coverage']:.0%} coverage]"
        print(f"  {bname:>8s}: {binfo['eff_lo']:6.2f} – {binfo['eff_hi']:6.2f} Hz{flag}")

    print(f"\nPrimary analysis: {len(PRIMARY_BANDS)} fully-covered bands")
    print(f"Conventional:     {len(BANDS)} bands")

    # ── Run each dataset ──
    all_results = []

    for ds_name, ds_info in DATASETS.items():
        peaks_dir = ds_info['peaks_dir']
        peak_files = sorted(glob.glob(os.path.join(peaks_dir, '*_peaks.csv')))

        if not peak_files:
            print(f"\n*** {ds_name}: no peak files at {peaks_dir}")
            continue

        print(f"\n{'='*80}")
        print(f"  {ds_name}  ({len(peak_files)} subjects)")
        print(f"{'='*80}")

        # Select dominant peaks with phi-octave bands (primary)
        dom_primary = select_dominant_peaks(peak_files, PRIMARY_BANDS)
        res_primary = run_stats(dom_primary, PRIMARY_BANDS, label=f'{ds_name} [phi-oct primary]')

        # Select dominant peaks with ALL phi-octave bands (including partial)
        dom_all = select_dominant_peaks(peak_files, ALL_BANDS)
        res_all = run_stats(dom_all, ALL_BANDS, label=f'{ds_name} [phi-oct all]')

        # Load conventional results for comparison
        conv_csv = ds_info['conv_csv']
        conv_d = conv_rank = conv_rank_raw = '?'
        if os.path.isfile(conv_csv):
            cdf = pd.read_csv(conv_csv)
            if len(cdf) > 0:
                conv_d = f"+{cdf['cohen_d'].iloc[0]:.3f}"
                conv_rank = f"{int(cdf['phi_rank'].iloc[0])}/9"
                conv_rank_raw = f"{int(cdf['phi_rank_raw'].iloc[0])}/9"

        # Print results
        if res_primary:
            r = res_primary
            print(f"\n  PRIMARY ({r['n_bands']} phi-octave bands, N={r['n_valid']}):")
            print(f"    Cohen's d  = +{r['cohen_d']:.3f}  (obs_mean_d={r['obs_mean_d']:.4f}, null={r['null_expected']:.4f})")
            print(f"    p_ttest    = {r['p_ttest']:.2e}")
            print(f"    phi_rank_z = {r['phi_rank_z']}/9  (raw: {r['phi_rank_raw']}/9)")
            print(f"    top-4 z:     {', '.join(f'{n}:{z:+.1f}' for n,z in r['top4_z'])}")

        if res_all:
            r = res_all
            print(f"\n  ALL BANDS ({r['n_bands']} phi-octave bands, N={r['n_valid']}):")
            print(f"    Cohen's d  = +{r['cohen_d']:.3f}")
            print(f"    phi_rank_z = {r['phi_rank_z']}/9  (raw: {r['phi_rank_raw']}/9)")

        print(f"\n  CONVENTIONAL ({len(BANDS)} bands):")
        print(f"    Cohen's d  = {conv_d}")
        print(f"    phi_rank_z = {conv_rank}  (raw: {conv_rank_raw})")

        all_results.append({
            'dataset': ds_name,
            'primary': res_primary,
            'all_bands': res_all,
            'conv_d': conv_d,
            'conv_rank': conv_rank,
        })

    # ── Summary table ──
    print(f"\n\n{'='*110}")
    print("SUMMARY: Phi-Octave vs Conventional Bands")
    print(f"{'='*110}")
    print(f"{'Dataset':<18s}  {'N':>4s}  {'d(conv)':>8s} {'rank(conv)':>10s}  "
          f"{'d(φ-oct)':>8s} {'rank(φ-oct)':>11s}  "
          f"{'d(all7)':>8s} {'rank(all7)':>10s}  top-4 z-scores")
    print("-" * 110)

    for r in all_results:
        ds = r['dataset']
        rp = r['primary']
        ra = r['all_bands']
        n = rp['n_valid'] if rp else '?'
        d_p = f"+{rp['cohen_d']:.3f}" if rp else '?'
        rk_p = f"{rp['phi_rank_z']}/9" if rp else '?'
        d_a = f"+{ra['cohen_d']:.3f}" if ra else '?'
        rk_a = f"{ra['phi_rank_z']}/9" if ra else '?'
        top4 = ', '.join(f'{nm}:{z:+.1f}' for nm, z in rp['top4_z']) if rp else ''
        print(f"{ds:<18s}  {n:>4}  {r['conv_d']:>8s} {r['conv_rank']:>10s}  "
              f"{d_p:>8s} {rk_p:>11s}  "
              f"{d_a:>8s} {rk_a:>10s}  {top4}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
