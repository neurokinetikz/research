#!/usr/bin/env python
"""
Decisive Tests — 1σ threshold variant
======================================

Same 5 tests as decisive_phi_tests.py but using a HARD THRESHOLD:
- A peak is "aligned" only if min_lattice_dist < 0.03 (1σ of KDE bandwidth)
- Primary metric: hit rate (fraction of bands with d < threshold)
- Also reports results at threshold = 0.06 (2σ) for comparison

Usage:
    python scripts/decisive_phi_tests_1sd.py
"""
import sys, os, time, io
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from phi_replication import (
    F0, PHI, BANDS, BASES,
    POSITIONS_DEG2, PHI_POSITIONS, POSITIONS_14,
    lattice_coord, circ_dist, min_lattice_dist, positions_for_base,
)

DATASETS = {
    'EEGMMIDB EC': 'exports_eegmmidb/replication/combined/per_subject_dominant_peaks.csv',
    'LEMON EC':    'exports_lemon/replication/EC/per_subject_dominant_peaks.csv',
    'LEMON EO':    'exports_lemon/replication/EO/per_subject_dominant_peaks.csv',
    'HBN EC':      'exports_hbn/EC/per_subject_dominant_peaks.csv',
    'Dortmund EC': '/Volumes/T9/dortmund_data/lattice_results_replication_v2/EyesClosed_pre/per_subject_dominant_peaks.csv',
    'Dortmund EO': '/Volumes/T9/dortmund_data/lattice_results_replication_v2/EyesOpen_pre/per_subject_dominant_peaks.csv',
}

ALL_BANDS = list(BANDS.keys())
NON_ALPHA_BANDS = [b for b in ALL_BANDS if b != 'alpha']
THETA_GAMMA_ONLY = ['theta', 'gamma']
N_PERM = 5000
RNG_SEED = 42

THRESHOLD_1SD = 0.03   # 1σ of KDE bandwidth
THRESHOLD_2SD = 0.06   # 2σ of KDE bandwidth


def load_datasets():
    loaded = {}
    for name, path in DATASETS.items():
        if os.path.isfile(path):
            df = pd.read_csv(path)
            loaded[name] = df
            print(f"  {name}: N={len(df)}")
        else:
            print(f"  {name}: NOT FOUND")
    return loaded


def compute_hit_rate(row, bands, threshold, f0=F0, base=PHI, positions=POSITIONS_DEG2):
    """Fraction of bands where peak is within threshold of a position."""
    hits = 0
    total = 0
    for b in bands:
        freq = row.get(f'{b}_freq', np.nan)
        if pd.isna(freq) or freq <= 0:
            continue
        u = lattice_coord(freq, f0=f0, base=base)
        d = min_lattice_dist(u, positions)
        total += 1
        if d < threshold:
            hits += 1
    return hits / total if total > 0 else np.nan


def compute_mean_d(row, bands, f0=F0, base=PHI, positions=POSITIONS_DEG2):
    """Mean lattice distance for specified bands."""
    ds = []
    for b in bands:
        freq = row.get(f'{b}_freq', np.nan)
        if pd.isna(freq) or freq <= 0:
            continue
        u = lattice_coord(freq, f0=f0, base=base)
        d = min_lattice_dist(u, positions)
        ds.append(d)
    return np.mean(ds) if ds else np.nan


def null_hit_rate(positions, threshold, n_samples=100_000, seed=42):
    """Expected hit rate under uniform null."""
    rng = np.random.RandomState(seed)
    us = rng.uniform(0, 1, n_samples)
    hits = sum(1 for u in us if min_lattice_dist(u, positions) < threshold)
    return hits / n_samples


def cross_base_threshold(df, bands, threshold, degree=3, n_perm=N_PERM, seed=RNG_SEED):
    """Cross-base comparison using hit-rate metric."""
    subject_freqs = []
    for _, row in df.iterrows():
        freqs = {}
        for b in bands:
            freq = row.get(f'{b}_freq', np.nan)
            if not pd.isna(freq) and freq > 0:
                freqs[b] = freq
        if freqs:
            subject_freqs.append(freqs)

    if len(subject_freqs) < 10:
        return {}

    band_freq_arrays = {}
    for b in bands:
        vals = [sf[b] for sf in subject_freqs if b in sf]
        if vals:
            band_freq_arrays[b] = np.array(vals)

    base_results = {}
    for base_name, base_val in BASES.items():
        positions = positions_for_base(base_val, degree=degree)

        # Observed: per-subject hit rate
        obs_hits = []
        for sf in subject_freqs:
            hits = 0
            total = 0
            for b, freq in sf.items():
                u = lattice_coord(freq, f0=F0, base=base_val)
                d = min_lattice_dist(u, positions)
                total += 1
                if d < threshold:
                    hits += 1
            obs_hits.append(hits / total if total > 0 else 0)
        obs_hits = np.array(obs_hits)
        obs_mean = obs_hits.mean()

        # Permutation null
        rng = np.random.RandomState(seed)
        null_means = np.empty(n_perm)
        for pi in range(n_perm):
            perm_hits = []
            for b, freqs in band_freq_arrays.items():
                shuffled_u = rng.uniform(0, 1, len(freqs))
                n_hit = sum(1 for u in shuffled_u if min_lattice_dist(u, positions) < threshold)
                perm_hits.append(n_hit / len(freqs))
            null_means[pi] = np.mean(perm_hits) if perm_hits else 0

        null_mean = np.nanmean(null_means)
        null_sd = np.nanstd(null_means)
        z_score = (obs_mean - null_mean) / null_sd if null_sd > 0 else 0.0

        base_results[base_name] = {
            'hit_rate': obs_mean,
            'z_score': z_score,
            'n_positions': len(positions),
        }

    return base_results


def phi_rank(base_results):
    ranking = sorted(base_results.items(), key=lambda x: -x[1]['z_score'])
    for i, (name, _) in enumerate(ranking):
        if name == 'phi':
            return i + 1
    return -1


# ═══════════════════════════════════════════════════════════════════
# TEST 1: Alpha-Excluded Cross-Base (threshold variant)
# ═══════════════════════════════════════════════════════════════════
def test1(datasets, threshold):
    print(f"\n{'='*90}")
    print(f"TEST 1: ALPHA-EXCLUDED CROSS-BASE (threshold={threshold})")
    print(f"{'='*90}")

    results = {}
    for ds_name, df in datasets.items():
        print(f"\n  {ds_name} (N={len(df)}):")

        for label, bands in [('All 6 bands', ALL_BANDS),
                              ('No-alpha (5b)', NON_ALPHA_BANDS),
                              ('Theta+gamma', THETA_GAMMA_ONLY)]:
            br = cross_base_threshold(df, bands, threshold, degree=3)
            rank = phi_rank(br)
            phi_hr = br['phi']['hit_rate'] if 'phi' in br else 0
            phi_z = br['phi']['z_score'] if 'phi' in br else 0
            ranking = sorted(br.items(), key=lambda x: -x[1]['z_score'])[:3]
            top3 = ', '.join(f"{n}:{r['z_score']:+.1f}" for n, r in ranking)
            print(f"    {label:<18s}: phi rank {rank}/9  (hit={phi_hr:.1%}, z={phi_z:+.1f})  top3: {top3}")

        results[ds_name] = rank  # last one is theta+gamma
    return results


# ═══════════════════════════════════════════════════════════════════
# TEST 2: IAF-Stratified (threshold variant)
# ═══════════════════════════════════════════════════════════════════
def test2(datasets, threshold):
    print(f"\n{'='*90}")
    print(f"TEST 2: IAF-STRATIFIED NON-ALPHA (threshold={threshold})")
    print(f"{'='*90}")

    all_rows = []
    for ds_name, df in datasets.items():
        df_copy = df.copy()
        df_copy['dataset'] = ds_name
        all_rows.append(df_copy)
    pooled = pd.concat(all_rows, ignore_index=True)
    pooled['iaf'] = pooled['alpha_freq']
    valid = pooled[pooled['iaf'].notna() & (pooled['iaf'] > 0)].copy()

    null_hr = null_hit_rate(POSITIONS_DEG2, threshold)

    hit_rates = []
    for _, row in valid.iterrows():
        hr = compute_hit_rate(row, NON_ALPHA_BANDS, threshold)
        hit_rates.append(hr)
    valid = valid.copy()
    valid['hit_rate'] = hit_rates
    valid = valid[valid['hit_rate'].notna()].copy()

    print(f"  N={len(valid)}, null hit rate (uniform): {null_hr:.1%}")

    valid['iaf_quartile'] = pd.qcut(valid['iaf'], 4, labels=['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)'])

    print(f"\n  {'Quartile':<14s}  {'IAF range':>16s}  {'N':>4s}  {'hit_rate':>9s}  {'vs null':>8s}  {'p-value':>10s}")
    print(f"  {'-'*70}")

    for q in ['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)']:
        qdf = valid[valid['iaf_quartile'] == q]
        iaf_lo, iaf_hi = qdf['iaf'].min(), qdf['iaf'].max()
        hr_arr = qdf['hit_rate'].values
        obs_hr = hr_arr.mean()
        # Binomial-like test: is observed hit rate > null?
        _, p_val = stats.ttest_1samp(hr_arr, null_hr)
        excess = obs_hr - null_hr
        print(f"  {q:<14s}  {iaf_lo:6.1f}–{iaf_hi:5.1f} Hz  {len(qdf):>4d}  {obs_hr:>8.1%}  {excess:>+7.1%}  {p_val:.2e}")

    r, p = stats.pearsonr(valid['iaf'], valid['hit_rate'])
    print(f"\n  r(non-alpha hit_rate, IAF) = {r:+.3f}, p = {p:.3e}")
    return r, p


# ═══════════════════════════════════════════════════════════════════
# TEST 3: f₀ Sweep (threshold variant)
# ═══════════════════════════════════════════════════════════════════
def test3(datasets, threshold):
    print(f"\n{'='*90}")
    print(f"TEST 3: PER-SUBJECT f₀ SWEEP (threshold={threshold})")
    print(f"{'='*90}")

    f0_range = np.arange(6.0, 10.05, 0.05)
    positions = POSITIONS_DEG2

    test_ds = {k: v for k, v in datasets.items() if k in ['EEGMMIDB EC', 'LEMON EC', 'HBN EC']}
    if not test_ds:
        test_ds = dict(list(datasets.items())[:2])

    all_opt_f0 = []
    all_opt_f0_na = []
    all_iaf = []

    for ds_name, df in test_ds.items():
        print(f"\n  {ds_name} (N={len(df)}):")
        opt_f0s = []
        opt_f0s_na = []

        for _, row in df.iterrows():
            freqs_all = {}
            freqs_na = {}
            for b in ALL_BANDS:
                freq = row.get(f'{b}_freq', np.nan)
                if not pd.isna(freq) and freq > 0:
                    freqs_all[b] = freq
                    if b != 'alpha':
                        freqs_na[b] = freq
            if len(freqs_all) < 3:
                continue

            # Sweep: maximize hit rate (not minimize distance)
            best_hr = -1
            best_f0 = 7.83
            best_hr_na = -1
            best_f0_na = 7.83

            for f0 in f0_range:
                hits = sum(1 for freq in freqs_all.values()
                           if min_lattice_dist(lattice_coord(freq, f0=f0, base=PHI), positions) < threshold)
                hr = hits / len(freqs_all)
                if hr > best_hr or (hr == best_hr and abs(f0 - 7.83) < abs(best_f0 - 7.83)):
                    best_hr = hr
                    best_f0 = f0

                if freqs_na:
                    hits_na = sum(1 for freq in freqs_na.values()
                                  if min_lattice_dist(lattice_coord(freq, f0=f0, base=PHI), positions) < threshold)
                    hr_na = hits_na / len(freqs_na)
                    if hr_na > best_hr_na or (hr_na == best_hr_na and abs(f0 - 7.83) < abs(best_f0_na - 7.83)):
                        best_hr_na = hr_na
                        best_f0_na = f0

            opt_f0s.append(best_f0)
            opt_f0s_na.append(best_f0_na)

        opt_f0s = np.array(opt_f0s)
        opt_f0s_na = np.array(opt_f0s_na)
        print(f"    Optimal f₀ (all):       mean={np.mean(opt_f0s):.2f}, median={np.median(opt_f0s):.2f}")
        print(f"    Optimal f₀ (non-alpha): mean={np.mean(opt_f0s_na):.2f}, median={np.median(opt_f0s_na):.2f}")

        all_opt_f0.extend(opt_f0s)
        all_opt_f0_na.extend(opt_f0s_na)
        iaf = df['alpha_freq'].dropna().values
        all_iaf.extend(iaf[:len(opt_f0s)])

    all_opt_f0 = np.array(all_opt_f0)
    all_opt_f0_na = np.array(all_opt_f0_na)
    all_iaf = np.array(all_iaf)

    print(f"\n  POOLED (N={len(all_opt_f0)}):")
    bins = np.arange(6.0, 10.2, 0.2)
    hist = np.histogram(all_opt_f0, bins=bins)[0]
    hist_na = np.histogram(all_opt_f0_na, bins=bins)[0]

    print(f"  Optimal f₀ distribution (all bands):")
    for i in range(len(hist)):
        bar = '#' * (hist[i] * 50 // max(hist.max(), 1))
        mark = ' <-- 7.83' if bins[i] <= 7.83 < bins[i+1] else ''
        print(f"    {bins[i]:5.1f}-{bins[i+1]:5.1f}: {hist[i]:>4d} {bar}{mark}")

    print(f"\n  Optimal f₀ distribution (non-alpha):")
    for i in range(len(hist_na)):
        bar = '#' * (hist_na[i] * 50 // max(hist_na.max(), 1))
        mark = ' <-- 7.83' if bins[i] <= 7.83 < bins[i+1] else ''
        print(f"    {bins[i]:5.1f}-{bins[i+1]:5.1f}: {hist_na[i]:>4d} {bar}{mark}")

    valid_mask = np.isfinite(all_iaf)
    if valid_mask.sum() > 10:
        r, p = stats.pearsonr(all_iaf[valid_mask], all_opt_f0[valid_mask])
        r_na, p_na = stats.pearsonr(all_iaf[valid_mask], all_opt_f0_na[valid_mask])
        print(f"\n  r(f₀*, IAF): r={r:+.3f}, p={p:.3e}")
        print(f"  r(f₀*_na, IAF): r={r_na:+.3f}, p={p_na:.3e}")

    return all_opt_f0, all_opt_f0_na


# ═══════════════════════════════════════════════════════════════════
# TEST 5: Generative Null (threshold variant)
# ═══════════════════════════════════════════════════════════════════
def test5(datasets, threshold):
    print(f"\n{'='*90}")
    print(f"TEST 5: GENERATIVE NULL (threshold={threshold})")
    print(f"{'='*90}")

    # Empirical distributions from data
    real_dists = {}
    for b in ALL_BANDS:
        all_f = []
        for ds_name, df in datasets.items():
            vals = df[f'{b}_freq'].dropna().values
            vals = vals[vals > 0]
            all_f.extend(vals)
        all_f = np.array(all_f)
        if len(all_f) > 0:
            real_dists[b] = (np.mean(all_f), np.std(all_f))

    n_synth = 5000
    rng = np.random.RandomState(42)
    null_hr = null_hit_rate(POSITIONS_DEG2, threshold)

    # Synthetic subjects
    synth_hit_rates = []
    for _ in range(n_synth):
        hits = 0
        total = 0
        for b, (mu, sd) in real_dists.items():
            freq = rng.normal(mu, sd)
            lo, hi = BANDS[b]
            freq = np.clip(freq, lo + 0.1, hi - 0.1)
            u = lattice_coord(freq, f0=F0, base=PHI)
            d = min_lattice_dist(u, POSITIONS_DEG2)
            total += 1
            if d < threshold:
                hits += 1
        synth_hit_rates.append(hits / total if total > 0 else 0)
    synth_hit_rates = np.array(synth_hit_rates)

    # Real subjects
    real_hit_rates = []
    for ds_name, df in datasets.items():
        for _, row in df.iterrows():
            hr = compute_hit_rate(row, ALL_BANDS, threshold)
            if not np.isnan(hr):
                real_hit_rates.append(hr)
    real_hit_rates = np.array(real_hit_rates)

    print(f"\n  Null hit rate (uniform): {null_hr:.1%}")
    print(f"\n  {'Source':<25s}  {'hit_rate':>9s}  {'SD':>6s}  {'excess':>8s}")
    print(f"  {'-'*55}")
    print(f"  {'Real subjects':<25s}  {np.mean(real_hit_rates):>8.1%}  {np.std(real_hit_rates):.4f}  {np.mean(real_hit_rates)-null_hr:>+7.1%}")
    print(f"  {'Synthetic (empirical)':<25s}  {np.mean(synth_hit_rates):>8.1%}  {np.std(synth_hit_rates):.4f}  {np.mean(synth_hit_rates)-null_hr:>+7.1%}")

    u_stat, p_mw = stats.mannwhitneyu(real_hit_rates, synth_hit_rates, alternative='greater')
    print(f"\n  Mann-Whitney (real > synthetic): U={u_stat:.0f}, p={p_mw:.3e}")

    # Cross-base for synthetic
    synth_rows = []
    rng2 = np.random.RandomState(99)
    for i in range(n_synth):
        row = {'subject': f'synth_{i}'}
        for b, (mu, sd) in real_dists.items():
            freq = rng2.normal(mu, sd)
            lo, hi = BANDS[b]
            freq = np.clip(freq, lo + 0.1, hi - 0.1)
            row[f'{b}_freq'] = freq
        synth_rows.append(row)
    synth_df = pd.DataFrame(synth_rows)

    br_synth = cross_base_threshold(synth_df, ALL_BANDS, threshold, degree=3, n_perm=2000, seed=99)
    rank = phi_rank(br_synth)
    ranking = sorted(br_synth.items(), key=lambda x: -x[1]['z_score'])
    print(f"\n  Synthetic cross-base (phi rank: {rank}/9):")
    for name, r in ranking:
        mark = ' <--' if name == 'phi' else ''
        print(f"    {name:>6s}: z={r['z_score']:+.2f}, hit={r['hit_rate']:.1%}{mark}")

    return {
        'real_hr': np.mean(real_hit_rates),
        'synth_hr': np.mean(synth_hit_rates),
        'null_hr': null_hr,
        'synth_rank': rank,
    }


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

    print("Decisive Tests — 1σ vs 2σ Threshold Comparison")
    print("=" * 90)
    print(f"f₀ = {F0} Hz, base = φ = {PHI:.6f}")
    print(f"1σ threshold = {THRESHOLD_1SD}, 2σ threshold = {THRESHOLD_2SD}")
    print(f"\nDegree-2 positions: {list(POSITIONS_DEG2.keys())}")
    print(f"  {', '.join(f'{k}={v:.3f}' for k, v in POSITIONS_DEG2.items())}")

    # Coverage: what fraction of [0,1) is within threshold of a deg-2 position?
    null_1sd = null_hit_rate(POSITIONS_DEG2, THRESHOLD_1SD)
    null_2sd = null_hit_rate(POSITIONS_DEG2, THRESHOLD_2SD)
    print(f"\n  Null hit rate at 1σ: {null_1sd:.1%} (fraction of [0,1) within {THRESHOLD_1SD} of any position)")
    print(f"  Null hit rate at 2σ: {null_2sd:.1%} (fraction of [0,1) within {THRESHOLD_2SD} of any position)")

    print(f"\nLoading datasets:")
    datasets = load_datasets()

    for threshold, label in [(THRESHOLD_1SD, '1σ'), (THRESHOLD_2SD, '2σ')]:
        print(f"\n\n{'#'*90}")
        print(f"  RUNNING ALL TESTS AT {label} THRESHOLD = {threshold}")
        print(f"{'#'*90}")

        test1(datasets, threshold)
        test2(datasets, threshold)
        test3(datasets, threshold)
        test5(datasets, threshold)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")

    sys.stdout = old_stdout
    with open('decisive_phi_tests_1sd_results.txt', 'w') as f:
        f.write(output.getvalue())
    print(f"Results saved to decisive_phi_tests_1sd_results.txt")
