#!/usr/bin/env python3
"""
Paper 4 — Pairwise Frequency Ratio Test
Anti-Mode-Locking Signature in Resting-State EEG
═══════════════════════════════════════════════════

Theory (Pletzer 2014): Neural oscillators use phi-spaced frequencies
to prevent destructive synchronization (mode-locking). If true:

  ENRICHMENT at: φ^n ratios (1.272, 1.618, 2.618)
  DEPLETION  at: simple rationals (5/4, 4/3, 3/2, 2/1)

This tests a qualitatively different prediction than position alignment.
No other base predicts BOTH enrichment at its own multiples AND depletion
at simple rationals simultaneously.

Data: All FOOOF-detected spectral peaks per electrode (~15-20 peaks/ch)
from per_subject_peaks/ directories (NOT dominant-only).
"""

import os
import sys
import glob
import time
import numpy as np
import pandas as pd
from scipy import stats, signal
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
from phi_replication import (
    PHI, F0, BASES, lattice_coord, min_lattice_dist,
    positions_for_base, circ_dist, POSITIONS_DEG2
)

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset paths — per_subject_peaks directories
DATASET_PATHS = [
    ('EEGMMIDB_EC', 'exports_eegmmidb/replication/EC/per_subject_peaks'),
    ('LEMON_EC',    'exports_lemon/replication/EC/per_subject_peaks'),
    ('LEMON_EO',    'exports_lemon/replication/EO/per_subject_peaks'),
    ('HBN_EC',      'exports_hbn/EC/per_subject_peaks'),
]
DORTMUND_PATH = '/Volumes/T9/dortmund_data/lattice_results_replication_v2/EyesClosed_pre/per_subject_peaks'

# Target ratios
PHI_TARGETS = {
    'φ^½':  PHI**0.5,       # 1.2720
    'φ':    PHI,             # 1.6180
    'φ²':   PHI**2,          # 2.6180
}
RATIONAL_TARGETS = {
    '5/4':  5/4,             # 1.2500
    '4/3':  4/3,             # 1.3333
    '3/2':  3/2,             # 1.5000
    '8/5':  8/5,             # 1.6000
    '5/3':  5/3,             # 1.6667
    '7/4':  7/4,             # 1.7500
    '2/1':  2.0,             # 2.0000
    '3/1':  3.0,             # 3.0000
}
ALL_TARGETS = {**PHI_TARGETS, **RATIONAL_TARGETS}

RATIO_RANGE = (1.05, 4.0)   # Focus range
RATIO_WINDOW = 0.015         # ±window around target for counting
DEDUP_TOL = 0.15             # Hz, merge phi-octave edge duplicates
N_BOOTSTRAP = 2000
N_NULL_PAIRS = 5_000_000     # Independent-pair null size
SG_WINDOW_RATIO = 0.25       # Savitzky-Golay window in log-ratio units
HIST_BIN_WIDTH = 0.003       # Log-ratio bin width

lines = []
def P(s=''):
    print(s, flush=True)
    lines.append(s)


# ═══════════════════════════════════════════════════════════════
# PART 1: DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_peaks(peaks_dir):
    """Load all per-subject peak CSVs, deduplicate OT edge peaks."""
    files = sorted(glob.glob(os.path.join(peaks_dir, '*_peaks.csv')))
    if not files:
        return pd.DataFrame(), 0, 0

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
        return pd.DataFrame(), 0, 0

    peaks = pd.concat(dfs, ignore_index=True)
    n_before = len(peaks)

    # Deduplicate: within (subject, channel), merge peaks within DEDUP_TOL Hz
    # Keep the one with higher power
    peaks['freq_bin'] = (peaks['freq'] / DEDUP_TOL).round() * DEDUP_TOL
    peaks = (peaks.sort_values('power', ascending=False)
             .drop_duplicates(subset=['subject', 'channel', 'freq_bin'], keep='first')
             .drop(columns=['freq_bin'])
             .sort_values(['subject', 'channel', 'freq']))
    n_after = len(peaks)

    return peaks, n_before, n_after


# ═══════════════════════════════════════════════════════════════
# PART 2: PAIRWISE RATIO COMPUTATION
# ═══════════════════════════════════════════════════════════════

def compute_ratios(peaks_df, ratio_range=RATIO_RANGE):
    """Compute all within-channel pairwise ratios f_high / f_low."""
    ratio_list = []
    subject_list = []

    for (sub, ch), group in peaks_df.groupby(['subject', 'channel']):
        freqs = np.sort(group['freq'].values)
        n = len(freqs)
        if n < 2:
            continue
        for i in range(n):
            for j in range(i + 1, n):
                r = freqs[j] / freqs[i]
                if ratio_range[0] <= r <= ratio_range[1]:
                    ratio_list.append(r)
                    subject_list.append(sub)

    return np.array(ratio_list), np.array(subject_list)


# ═══════════════════════════════════════════════════════════════
# PART 3: NULL MODELS
# ═══════════════════════════════════════════════════════════════

def independent_pair_null(all_freqs, n_pairs=N_NULL_PAIRS, ratio_range=RATIO_RANGE):
    """Null: independently sample two frequencies from pooled distribution.
    Tests whether within-channel co-occurrence creates ratio structure."""
    f1 = np.random.choice(all_freqs, size=n_pairs)
    f2 = np.random.choice(all_freqs, size=n_pairs)
    ratios = np.maximum(f1, f2) / np.minimum(f1, f2)
    mask = (ratios >= ratio_range[0]) & (ratios <= ratio_range[1])
    return ratios[mask]


def sg_baseline(log_ratios, n_bins, bin_edges):
    """Savitzky-Golay smooth of log-ratio histogram as baseline.
    Tests for fine structure beyond the broad ratio distribution."""
    counts, _ = np.histogram(log_ratios, bins=bin_edges)

    # SG window: broad enough to capture gross trend, miss fine structure
    sg_bins = max(15, int(SG_WINDOW_RATIO / HIST_BIN_WIDTH) | 1)
    if sg_bins % 2 == 0:
        sg_bins += 1
    if sg_bins > len(counts) - 2:
        sg_bins = len(counts) - 2
        if sg_bins % 2 == 0:
            sg_bins -= 1

    baseline = signal.savgol_filter(counts.astype(float), sg_bins, 3)
    baseline = np.maximum(baseline, 1.0)  # Avoid division by zero
    return counts, baseline


# ═══════════════════════════════════════════════════════════════
# PART 4: ENRICHMENT ANALYSIS
# ═══════════════════════════════════════════════════════════════

def enrichment_vs_null(ratios, null_ratios, targets, window=RATIO_WINDOW):
    """Enrichment at each target: (obs_density / null_density - 1) × 100%."""
    n_obs = len(ratios)
    n_null = len(null_ratios)
    results = {}

    for name, target in targets.items():
        obs_count = np.sum(np.abs(ratios - target) < window)
        null_count = np.sum(np.abs(null_ratios - target) < window)

        obs_frac = obs_count / n_obs
        null_frac = null_count / n_null

        if null_frac > 0:
            enrichment = (obs_frac / null_frac - 1) * 100
        else:
            enrichment = np.nan

        results[name] = {
            'target': target,
            'obs_count': obs_count,
            'null_frac': null_frac,
            'enrichment_pct': enrichment,
        }

    return results


def enrichment_vs_sg(ratios, targets, window=RATIO_WINDOW):
    """Enrichment at each target relative to SG-smoothed baseline.
    Tests fine structure: peaks/troughs at specific ratio values."""
    log_r = np.log(ratios)
    lr_min, lr_max = np.log(RATIO_RANGE[0]), np.log(RATIO_RANGE[1])
    n_bins = int((lr_max - lr_min) / HIST_BIN_WIDTH) + 1
    bin_edges = np.linspace(lr_min, lr_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    counts, baseline = sg_baseline(log_r, n_bins, bin_edges)

    results = {}
    for name, target in targets.items():
        lt = np.log(target)
        idx = np.argmin(np.abs(bin_centers - lt))

        # Average over ±window in ratio space → ±window/target in log space
        log_window = window / target
        idx_lo = np.searchsorted(bin_centers, lt - log_window)
        idx_hi = np.searchsorted(bin_centers, lt + log_window)
        if idx_hi <= idx_lo:
            idx_lo, idx_hi = max(0, idx - 1), min(len(counts), idx + 2)

        obs_density = np.mean(counts[idx_lo:idx_hi]) if idx_hi > idx_lo else counts[idx]
        base_density = np.mean(baseline[idx_lo:idx_hi]) if idx_hi > idx_lo else baseline[idx]

        if base_density > 0:
            enrichment = (obs_density / base_density - 1) * 100
        else:
            enrichment = np.nan

        results[name] = {
            'target': target,
            'obs_density': obs_density,
            'base_density': base_density,
            'enrichment_pct': enrichment,
        }

    return results, bin_centers, counts, baseline


# ═══════════════════════════════════════════════════════════════
# PART 5: BOOTSTRAP
# ═══════════════════════════════════════════════════════════════

def bootstrap_enrichment_sg(ratios, subjects, targets,
                             n_boot=N_BOOTSTRAP, window=RATIO_WINDOW):
    """Bootstrap over subjects for SG-enrichment CIs.
    Precomputes per-subject log-ratio histograms for speed."""
    unique_subs = np.unique(subjects)
    n_sub = len(unique_subs)

    # Fixed histogram bins
    lr_min, lr_max = np.log(RATIO_RANGE[0]), np.log(RATIO_RANGE[1])
    n_bins = int((lr_max - lr_min) / HIST_BIN_WIDTH) + 1
    bin_edges = np.linspace(lr_min, lr_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # SG parameters
    sg_bins = max(15, int(SG_WINDOW_RATIO / HIST_BIN_WIDTH) | 1)
    if sg_bins % 2 == 0:
        sg_bins += 1
    if sg_bins > n_bins - 2:
        sg_bins = n_bins - 2
        if sg_bins % 2 == 0:
            sg_bins -= 1

    # Precompute per-subject histograms
    sub_hists = {}
    for s in unique_subs:
        s_ratios = ratios[subjects == s]
        s_log = np.log(s_ratios)
        h, _ = np.histogram(s_log, bins=bin_edges)
        sub_hists[s] = h

    # Precompute target bin ranges
    target_ranges = {}
    for name, target in targets.items():
        lt = np.log(target)
        log_window = window / target
        idx_lo = np.searchsorted(bin_centers, lt - log_window)
        idx_hi = np.searchsorted(bin_centers, lt + log_window)
        if idx_hi <= idx_lo:
            idx = np.argmin(np.abs(bin_centers - lt))
            idx_lo, idx_hi = max(0, idx - 1), min(n_bins, idx + 2)
        target_ranges[name] = (idx_lo, idx_hi)

    # Bootstrap
    boot_results = {name: np.zeros(n_boot) for name in targets}

    for b in range(n_boot):
        boot_subs = np.random.choice(unique_subs, size=n_sub, replace=True)
        boot_hist = np.zeros(n_bins)
        for s in boot_subs:
            boot_hist += sub_hists[s]

        baseline = signal.savgol_filter(boot_hist.astype(float), sg_bins, 3)
        baseline = np.maximum(baseline, 1.0)

        for name in targets:
            lo, hi = target_ranges[name]
            obs = np.mean(boot_hist[lo:hi]) if hi > lo else boot_hist[lo]
            base = np.mean(baseline[lo:hi]) if hi > lo else baseline[lo]
            boot_results[name][b] = (obs / base - 1) * 100 if base > 0 else np.nan

    stats_out = {}
    for name in targets:
        vals = boot_results[name]
        vals = vals[np.isfinite(vals)]
        if len(vals) > 0:
            stats_out[name] = {
                'mean': np.mean(vals),
                'ci_lo': np.percentile(vals, 2.5),
                'ci_hi': np.percentile(vals, 97.5),
            }
        else:
            stats_out[name] = {'mean': np.nan, 'ci_lo': np.nan, 'ci_hi': np.nan}

    return stats_out


# ═══════════════════════════════════════════════════════════════
# PART 6: CROSS-BASE LATTICE ALIGNMENT OF RATIOS
# ═══════════════════════════════════════════════════════════════

def cross_base_ratio_alignment(ratios, degree=2, max_ratios=500_000):
    """Which base's lattice best captures pairwise ratio structure?"""
    if len(ratios) > max_ratios:
        idx = np.random.choice(len(ratios), size=max_ratios, replace=False)
        r_sample = ratios[idx]
    else:
        r_sample = ratios

    results = {}
    for bname, bval in BASES.items():
        positions = positions_for_base(bval, degree=degree)
        pos_vals = np.array(list(positions.values()))

        # Vectorized lattice coordinate and distance
        u = (np.log(r_sample) / np.log(bval)) % 1.0
        dists = np.abs(u[:, None] - pos_vals[None, :])
        dists = np.minimum(dists, 1 - dists)
        d_vals = np.min(dists, axis=1)

        mean_d = np.mean(d_vals)

        # Uniform null
        u_grid = np.linspace(0, 1, 10000, endpoint=False)
        dists_grid = np.abs(u_grid[:, None] - pos_vals[None, :])
        dists_grid = np.minimum(dists_grid, 1 - dists_grid)
        null_d = np.mean(np.min(dists_grid, axis=1))

        cohens_d = (null_d - mean_d) / np.std(d_vals) if np.std(d_vals) > 0 else 0

        results[bname] = {
            'mean_d': mean_d,
            'null_d': null_d,
            'cohens_d': cohens_d,
            'n_positions': len(pos_vals),
        }

    return results


# ═══════════════════════════════════════════════════════════════
# PART 7: ANTI-MODE-LOCKING SCORE PER BASE
# ═══════════════════════════════════════════════════════════════

def per_base_aml_score(ratios, null_ratios, window=RATIO_WINDOW):
    """For each base, test enrichment at its own powers vs depletion at competitors.

    Only phi predicts avoidance of simple rationals. Test whether phi's
    AML score (own_enrichment - competitor_enrichment) is uniquely positive.
    """
    simple_rationals = [5/4, 4/3, 3/2, 2/1, 3/1]
    n_obs = len(ratios)
    n_null = len(null_ratios)

    def _enrich(target):
        obs = np.sum(np.abs(ratios - target) < window) / n_obs
        null = np.sum(np.abs(null_ratios - target) < window) / n_null
        return (obs / null - 1) * 100 if null > 0 else np.nan

    results = {}
    for bname, bval in BASES.items():
        # Own powers in ratio range
        own_targets = []
        for k in range(-3, 4):
            r = bval ** k
            if RATIO_RANGE[0] <= r <= RATIO_RANGE[1]:
                own_targets.append(r)
            r_inv = bval ** (-k)
            if RATIO_RANGE[0] <= r_inv <= RATIO_RANGE[1] and r_inv not in own_targets:
                own_targets.append(r_inv)

        # Remove near-duplicates
        own_targets = sorted(set(round(t, 6) for t in own_targets))

        if not own_targets:
            continue

        own_enrichments = [_enrich(t) for t in own_targets]
        rat_enrichments = [_enrich(t) for t in simple_rationals]

        own_mean = np.nanmean(own_enrichments)
        rat_mean = np.nanmean(rat_enrichments)
        aml = own_mean - rat_mean

        results[bname] = {
            'own_targets': own_targets,
            'own_mean_enrich': own_mean,
            'rat_mean_enrich': rat_mean,
            'aml_score': aml,
        }

    return results


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    P("═" * 72)
    P("  PAIRWISE FREQUENCY RATIO TEST — ANTI-MODE-LOCKING SIGNATURE")
    P("═" * 72)
    P()
    P("Prediction: Enrichment at φ^n ratios, depletion at p/q rationals")
    P(f"Ratio window: ±{RATIO_WINDOW}  |  Focus range: {RATIO_RANGE}")
    P(f"Bootstrap: {N_BOOTSTRAP}  |  Null pairs: {N_NULL_PAIRS:,}")
    P()

    # ── Load all datasets ──
    P("─" * 72)
    P("DATA LOADING")
    P("─" * 72)

    DATASETS = {}
    for name, relpath in DATASET_PATHS:
        full = os.path.join(BASE_DIR, relpath)
        if os.path.isdir(full):
            DATASETS[name] = full
    if os.path.isdir(DORTMUND_PATH):
        DATASETS['Dortmund_EC'] = DORTMUND_PATH

    all_peaks = {}
    for dname, dpath in DATASETS.items():
        t1 = time.time()
        peaks, n_before, n_after = load_peaks(dpath)
        if len(peaks) == 0:
            P(f"  {dname}: NO DATA")
            continue
        all_peaks[dname] = peaks
        n_sub = peaks['subject'].nunique()
        n_ch = peaks.groupby('subject')['channel'].nunique().mean()
        ppch = len(peaks) / (n_sub * n_ch)
        P(f"  {dname:<15} {n_sub:>4} subjects  {len(peaks):>9,} peaks  "
          f"({ppch:.1f}/ch)  dedup -{n_before-n_after:,}  [{time.time()-t1:.1f}s]")

    pooled = pd.concat(all_peaks.values(), ignore_index=True)
    total_sub = pooled['subject'].nunique()
    P(f"\n  TOTAL: {total_sub} subjects, {len(pooled):,} peaks")

    # ── Compute pairwise ratios ──
    P(f"\n{'─' * 72}")
    P("COMPUTING PAIRWISE RATIOS")
    P("─" * 72)

    t1 = time.time()
    ratios, subjects = compute_ratios(pooled)
    P(f"  Within-channel ratios in [{RATIO_RANGE[0]}, {RATIO_RANGE[1]}]: {len(ratios):,}")
    P(f"  Unique subjects: {len(np.unique(subjects))}")
    P(f"  Ratios/subject: {len(ratios)/len(np.unique(subjects)):,.0f} (mean)")
    P(f"  Time: {time.time()-t1:.1f}s")

    if len(ratios) < 100:
        P("\nINSUFFICIENT DATA — aborting")
        return

    # ── Null distributions ──
    P(f"\n{'─' * 72}")
    P("NULL DISTRIBUTIONS")
    P("─" * 72)

    all_freqs = pooled['freq'].values
    t1 = time.time()
    null_ratios = independent_pair_null(all_freqs)
    P(f"  Independent-pair null: {len(null_ratios):,} ratios [{time.time()-t1:.1f}s]")

    # ══════════════════════════════════════════════════════════════
    # PART A: ENRICHMENT AT TARGET RATIOS
    # ══════════════════════════════════════════════════════════════

    P(f"\n{'═' * 72}")
    P("  PART A: ENRICHMENT AT TARGET RATIOS")
    P(f"{'═' * 72}")

    # A1: vs independent-pair null
    P(f"\n  A1: vs Independent-Pair Null (tests co-occurrence structure)")
    P(f"  {'─' * 66}")
    ipn_results = enrichment_vs_null(ratios, null_ratios, ALL_TARGETS)

    P(f"  {'Ratio':<8} {'Value':>7} {'ObsCount':>9} {'Enrich%':>9}")
    P(f"  {'═' * 8} PHI FAMILY (predicted enrichment) {'═' * 8}")
    for name in PHI_TARGETS:
        r = ipn_results[name]
        P(f"  {name:<8} {r['target']:>7.4f} {r['obs_count']:>9,} {r['enrichment_pct']:>+9.1f}%")
    P(f"  {'═' * 8} SIMPLE RATIONALS (predicted depletion) {'═' * 5}")
    for name in RATIONAL_TARGETS:
        r = ipn_results[name]
        P(f"  {name:<8} {r['target']:>7.4f} {r['obs_count']:>9,} {r['enrichment_pct']:>+9.1f}%")

    phi_ipn = np.mean([ipn_results[n]['enrichment_pct'] for n in PHI_TARGETS])
    rat_ipn = np.mean([ipn_results[n]['enrichment_pct'] for n in RATIONAL_TARGETS])
    P(f"\n  Mean phi enrichment (IPN):       {phi_ipn:>+7.1f}%")
    P(f"  Mean rational enrichment (IPN):   {rat_ipn:>+7.1f}%")
    P(f"  AML score (phi - rational):       {phi_ipn - rat_ipn:>+7.1f}%")

    # A2: vs Savitzky-Golay baseline
    P(f"\n  A2: vs Savitzky-Golay Baseline (tests fine structure)")
    P(f"  {'─' * 66}")
    sg_results, bin_centers, counts, baseline = enrichment_vs_sg(ratios, ALL_TARGETS)

    # Bootstrap CIs for SG enrichment
    P(f"  Computing {N_BOOTSTRAP} bootstrap CIs over subjects...")
    t1 = time.time()
    boot_stats = bootstrap_enrichment_sg(ratios, subjects, ALL_TARGETS)
    P(f"  Bootstrap done [{time.time()-t1:.1f}s]")

    P(f"\n  {'Ratio':<8} {'Value':>7} {'SG-Enrich%':>11} {'95% CI':>22} {'Sig':>4}")
    P(f"  {'═' * 8} PHI FAMILY (predicted enrichment) {'═' * 8}")
    for name in PHI_TARGETS:
        r = sg_results[name]
        b = boot_stats[name]
        sig = '**' if (b['ci_lo'] > 0 or b['ci_hi'] < 0) else ''
        P(f"  {name:<8} {r['target']:>7.4f} {r['enrichment_pct']:>+10.1f}% "
          f"[{b['ci_lo']:>+7.1f}%, {b['ci_hi']:>+7.1f}%]  {sig}")
    P(f"  {'═' * 8} SIMPLE RATIONALS (predicted depletion) {'═' * 5}")
    for name in RATIONAL_TARGETS:
        r = sg_results[name]
        b = boot_stats[name]
        sig = '**' if (b['ci_lo'] > 0 or b['ci_hi'] < 0) else ''
        P(f"  {name:<8} {r['target']:>7.4f} {r['enrichment_pct']:>+10.1f}% "
          f"[{b['ci_lo']:>+7.1f}%, {b['ci_hi']:>+7.1f}%]  {sig}")

    phi_sg = np.mean([sg_results[n]['enrichment_pct'] for n in PHI_TARGETS])
    rat_sg = np.mean([sg_results[n]['enrichment_pct'] for n in RATIONAL_TARGETS])
    P(f"\n  Mean phi enrichment (SG):         {phi_sg:>+7.1f}%")
    P(f"  Mean rational enrichment (SG):     {rat_sg:>+7.1f}%")
    P(f"  AML score (phi - rational):        {phi_sg - rat_sg:>+7.1f}%")

    # Combined significance
    phi_boots = np.array([boot_stats[n]['mean'] for n in PHI_TARGETS])
    rat_boots = np.array([boot_stats[n]['mean'] for n in RATIONAL_TARGETS])
    all_phi = np.mean(phi_boots)
    all_rat = np.mean(rat_boots)

    # ══════════════════════════════════════════════════════════════
    # PART B: CROSS-BASE LATTICE ALIGNMENT OF RATIOS
    # ══════════════════════════════════════════════════════════════

    P(f"\n{'═' * 72}")
    P("  PART B: CROSS-BASE LATTICE ALIGNMENT OF RATIOS")
    P(f"{'═' * 72}")
    P(f"\n  Which base's lattice best captures ratio structure? (degree-2)")

    t1 = time.time()
    base_results = cross_base_ratio_alignment(ratios, degree=2)
    P(f"  Computed in {time.time()-t1:.1f}s")

    ranked = sorted(base_results.items(), key=lambda x: x[1]['cohens_d'], reverse=True)
    P(f"\n  {'Base':<8} {'mean_d':>8} {'null_d':>8} {'Cohen d':>9} {'#Pos':>5} {'Rank':>5}")
    P(f"  {'─' * 45}")
    for rank, (bname, br) in enumerate(ranked, 1):
        marker = ' <--' if bname == 'phi' else ''
        P(f"  {bname:<8} {br['mean_d']:>8.5f} {br['null_d']:>8.5f} "
          f"{br['cohens_d']:>+8.4f} {br['n_positions']:>5} {rank:>4}/9{marker}")

    phi_rank = [i for i, (b, _) in enumerate(ranked, 1) if b == 'phi'][0]
    P(f"\n  Phi rank for ratio alignment: {phi_rank}/9")

    # ══════════════════════════════════════════════════════════════
    # PART C: PER-BASE ANTI-MODE-LOCKING SCORE
    # ══════════════════════════════════════════════════════════════

    P(f"\n{'═' * 72}")
    P("  PART C: PER-BASE ANTI-MODE-LOCKING SCORE")
    P(f"{'═' * 72}")
    P(f"\n  AML = enrichment at own powers - enrichment at simple rationals")
    P(f"  Only phi predicts BOTH own-enrichment AND rational-depletion")

    t1 = time.time()
    aml_results = per_base_aml_score(ratios, null_ratios)
    P(f"  Computed in {time.time()-t1:.1f}s")

    aml_ranked = sorted(aml_results.items(), key=lambda x: x[1]['aml_score'], reverse=True)
    P(f"\n  {'Base':<8} {'Own-Enrich':>11} {'Rat-Enrich':>11} {'AML':>8} {'#Targets':>9} {'Rank':>5}")
    P(f"  {'─' * 56}")
    for rank, (bname, ar) in enumerate(aml_ranked, 1):
        marker = ' <--' if bname == 'phi' else ''
        P(f"  {bname:<8} {ar['own_mean_enrich']:>+10.1f}% {ar['rat_mean_enrich']:>+10.1f}% "
          f"{ar['aml_score']:>+7.1f}% {len(ar['own_targets']):>8} {rank:>4}/9{marker}")

    phi_aml_rank = [i for i, (b, _) in enumerate(aml_ranked, 1) if b == 'phi'][0]
    P(f"\n  Phi AML rank: {phi_aml_rank}/9")

    # ══════════════════════════════════════════════════════════════
    # PART D: PER-DATASET REPLICATION
    # ══════════════════════════════════════════════════════════════

    P(f"\n{'═' * 72}")
    P("  PART D: PER-DATASET REPLICATION")
    P(f"{'═' * 72}")

    for dname, peaks_df in all_peaks.items():
        t1 = time.time()
        d_ratios, d_subjects = compute_ratios(peaks_df)
        if len(d_ratios) < 100:
            P(f"\n  {dname}: insufficient ratios ({len(d_ratios)})")
            continue

        d_freqs = peaks_df['freq'].values
        d_null = independent_pair_null(d_freqs, n_pairs=2_000_000)
        d_ipn = enrichment_vs_null(d_ratios, d_null, ALL_TARGETS)
        d_sg, _, _, _ = enrichment_vs_sg(d_ratios, ALL_TARGETS)

        # Cross-base
        d_base = cross_base_ratio_alignment(d_ratios, degree=2)
        d_ranked = sorted(d_base.items(), key=lambda x: x[1]['cohens_d'], reverse=True)
        d_phi_rank = [i for i, (b, _) in enumerate(d_ranked, 1) if b == 'phi'][0]

        n_sub = peaks_df['subject'].nunique()
        P(f"\n  {dname} (N={n_sub}, {len(d_ratios):,} ratios) [{time.time()-t1:.1f}s]")

        # Key targets
        P(f"    {'Ratio':<8} {'IPN':>9} {'SG':>9}")
        for name in ['φ', '3/2', '2/1', 'φ²']:
            ipn_e = d_ipn[name]['enrichment_pct']
            sg_e = d_sg[name]['enrichment_pct']
            P(f"    {name:<8} {ipn_e:>+8.1f}% {sg_e:>+8.1f}%")

        phi_ipn_d = np.mean([d_ipn[n]['enrichment_pct'] for n in PHI_TARGETS])
        rat_ipn_d = np.mean([d_ipn[n]['enrichment_pct'] for n in RATIONAL_TARGETS])
        phi_sg_d = np.mean([d_sg[n]['enrichment_pct'] for n in PHI_TARGETS])
        rat_sg_d = np.mean([d_sg[n]['enrichment_pct'] for n in RATIONAL_TARGETS])

        P(f"    AML (IPN): {phi_ipn_d - rat_ipn_d:>+7.1f}%  |  AML (SG): {phi_sg_d - rat_sg_d:>+7.1f}%")
        P(f"    Phi ratio-lattice rank: {d_phi_rank}/9")

    # ══════════════════════════════════════════════════════════════
    # PART E: WINDOW SENSITIVITY (is enrichment sharp or broad?)
    # ══════════════════════════════════════════════════════════════

    P(f"\n{'═' * 72}")
    P("  PART E: WINDOW SENSITIVITY (sharpness of ratio clustering)")
    P(f"{'═' * 72}")
    P(f"\n  If enrichment increases with narrower window → genuine ratio-locking")
    P(f"  If decreases → broad bump, not specific to exact ratio value")

    key_targets = {'φ': PHI, '3/2': 1.5, '2/1': 2.0}
    windows = [0.030, 0.020, 0.015, 0.010, 0.005]

    header = f"  {'Window':>8}"
    for name in key_targets:
        header += f"  {name:>10}"
    P(header)
    sep = f"  {'─' * 8}"
    for _ in key_targets:
        sep += f"  {'─' * 10}"
    P(sep)

    for w in windows:
        row = f"  ±{w:.3f} "
        for name, target in key_targets.items():
            obs = np.sum(np.abs(ratios - target) < w) / len(ratios)
            null = np.sum(np.abs(null_ratios - target) < w) / len(null_ratios)
            enrich = (obs / null - 1) * 100 if null > 0 else 0
            row += f"  {enrich:>+9.1f}%"
        P(row)

    # ══════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════

    P(f"\n{'═' * 72}")
    P("  SUMMARY VERDICT")
    P(f"{'═' * 72}")

    # Determine outcome
    phi_enriched = phi_sg > 0 and boot_stats['φ']['ci_lo'] > 0
    rats_depleted = rat_sg < 0
    aml_positive = phi_sg - rat_sg > 0
    phi_lattice_top3 = phi_rank <= 3
    phi_aml_top3 = phi_aml_rank <= 3

    P(f"\n  Anti-mode-locking predictions:")
    P(f"    φ enrichment (SG):        {phi_sg:>+7.1f}%  {'CONFIRMED' if phi_enriched else 'NOT CONFIRMED'}")
    P(f"    Rational depletion (SG):  {rat_sg:>+7.1f}%  {'CONFIRMED' if rats_depleted else 'NOT CONFIRMED'}")
    P(f"    AML score > 0:            {phi_sg-rat_sg:>+7.1f}%  {'YES' if aml_positive else 'NO'}")
    P(f"    φ ratio-lattice rank:     {phi_rank}/9    {'TOP 3' if phi_lattice_top3 else 'NOT TOP 3'}")
    P(f"    φ AML rank:               {phi_aml_rank}/9    {'TOP 3' if phi_aml_top3 else 'NOT TOP 3'}")

    if phi_enriched and rats_depleted and phi_aml_top3:
        P(f"\n  VERDICT: Anti-mode-locking signature DETECTED")
        P(f"  φ-ratio enrichment with rational depletion — dynamics signature")
    elif aml_positive and phi_aml_top3:
        P(f"\n  VERDICT: Partial anti-mode-locking signature")
        P(f"  AML score positive but not all components significant")
    else:
        P(f"\n  VERDICT: Anti-mode-locking signature NOT detected at this resolution")
        P(f"  Ratio structure does not show phi-specific enrichment/depletion pattern")

    elapsed = time.time() - t0
    P(f"\n  Total time: {elapsed:.1f}s")

    # Write results file
    outfile = os.path.join(BASE_DIR, 'pairwise_ratio_test_results.txt')
    with open(outfile, 'w') as f:
        f.write('\n'.join(lines))
    P(f"  Results written to {outfile}")


if __name__ == '__main__':
    np.random.seed(42)
    main()
