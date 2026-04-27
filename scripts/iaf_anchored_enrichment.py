#!/usr/bin/env python3
"""
IAF-Anchored Spectral Differentiation
======================================

Recomputes Voronoi enrichment using per-subject individualised coordinate
systems where each subject's IAF is aligned to the alpha attractor position.

For each subject:
    f0_i = IAF_i / sqrt(phi)     (so IAF maps to u = 0.5 in the alpha octave)
    band_n = floor(log(f / f0_i) / log(phi))   (reassign each peak's octave)
    u = (log(f / f0_i) / log(phi)) mod 1       (within-octave coordinate)

Then compares against the population-anchored enrichment (f0 = 7.60 Hz) on
the same peaks, with the same power filter, same Voronoi positions, same
biomarker metrics -- isolating the effect of boundary alignment.

Key tests:
    1. Does cognitive correlation (beta-low x LPS) strengthen under IAF-anchor?
    2. Does developmental trajectory (age correlations) weaken under IAF-anchor?
       (If enrichment tracks IAF maturation, anchoring removes that component.)
    3. Do enrichment profiles become simpler (higher polynomial R^2)?

Usage:
    python scripts/iaf_anchored_enrichment.py [--dataset lemon|hbn]
"""

import argparse
import glob
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'lib'))
from phi_frequency_model import PHI, F0

PHI_INV = 1.0 / PHI
SQRT_PHI = np.sqrt(PHI)  # 1.2720196495...

PEAK_BASE = os.path.join(BASE_DIR, 'exports_adaptive_v4')
OUT_DIR = os.path.join(BASE_DIR, 'outputs', 'iaf_anchored')
os.makedirs(OUT_DIR, exist_ok=True)

MIN_POWER_PCT = 50
MIN_PEAKS_PER_BAND = 30

# Band mapping by phi-octave integer n (relative to f0 = f0_i or F0)
BAND_BY_N = {-1: 'theta', 0: 'alpha', 1: 'beta_low', 2: 'beta_high', 3: 'gamma'}

# Voronoi position layout (12 positions per octave, same as paper)
POS_LIST = [
    ('boundary',    0.000),
    ('noble_6',     round(PHI_INV ** 6, 6)),
    ('noble_5',     round(PHI_INV ** 5, 6)),
    ('noble_4',     round(PHI_INV ** 4, 6)),
    ('noble_3',     round(PHI_INV ** 3, 6)),
    ('inv_noble_1', round(PHI_INV ** 2, 6)),
    ('attractor',   0.5),
    ('noble_1',     round(PHI_INV, 6)),
    ('inv_noble_3', round(1 - PHI_INV ** 3, 6)),
    ('inv_noble_4', round(1 - PHI_INV ** 4, 6)),
    ('inv_noble_5', round(1 - PHI_INV ** 5, 6)),
    ('inv_noble_6', round(1 - PHI_INV ** 6, 6)),
]
POS_NAMES = [p[0] for p in POS_LIST]
POS_VALS = np.array([p[1] for p in POS_LIST])
N_POS = len(POS_VALS)

# Voronoi edges (u-space) for each position
_VORONOI_EDGES = []
for i in range(N_POS):
    if i == 0:
        u_left = (POS_VALS[-1] + POS_VALS[0] + 1) / 2 % 1.0
        u_right = (POS_VALS[0] + POS_VALS[1]) / 2
    elif i == N_POS - 1:
        u_left = (POS_VALS[i - 1] + POS_VALS[i]) / 2
        u_right = (POS_VALS[i] + POS_VALS[0] + 1) / 2
    else:
        u_left = (POS_VALS[i - 1] + POS_VALS[i]) / 2
        u_right = (POS_VALS[i] + POS_VALS[i + 1]) / 2
    _VORONOI_EDGES.append((u_left, u_right))

# Hz-weighted null fractions per bin (accounts for log-Hz compression)
HZ_FRACS = []
for i in range(N_POS):
    u_left, u_right = _VORONOI_EDGES[i]
    if i == 0:
        hz_frac = (PHI ** 1.0 - PHI ** u_left + PHI ** u_right - PHI ** 0.0) / (PHI - 1)
    else:
        hz_frac = (PHI ** u_right - PHI ** u_left) / (PHI - 1)
    HZ_FRACS.append(hz_frac)
HZ_FRACS = np.array(HZ_FRACS)


def assign_bands(freqs, f0):
    """Assign each freq to integer octave n = floor(log_phi(f/f0))."""
    freqs = np.asarray(freqs, dtype=float)
    valid = freqs > 0
    n = np.full(freqs.shape, -999, dtype=int)
    n[valid] = np.floor(np.log(freqs[valid] / f0) / np.log(PHI)).astype(int)
    return n


def lattice_u(freqs, f0):
    """Within-octave fractional coordinate, u in [0, 1)."""
    return (np.log(freqs / f0) / np.log(PHI)) % 1.0


def assign_voronoi(u_vals):
    u = np.asarray(u_vals, dtype=float) % 1.0
    # Circular distance to each position
    dists = np.abs(u[:, None] - POS_VALS[None, :])
    dists = np.minimum(dists, 1 - dists)
    return np.argmin(dists, axis=1)


def compute_enrichment_from_freqs(freqs, f0):
    """Compute Voronoi enrichment for a set of peak freqs under anchor f0.

    Returns dict mapping band -> {position: enrichment_pct, ...derived_metrics}
    """
    n_int = assign_bands(freqs, f0)
    u = lattice_u(freqs, f0)
    assignments = assign_voronoi(u)

    out = {}
    for n_val, band in BAND_BY_N.items():
        mask = (n_int == n_val)
        band_n = int(mask.sum())
        if band_n < MIN_PEAKS_PER_BAND:
            for pname in POS_NAMES:
                out[f'{band}_{pname}'] = np.nan
            for metric in ['mountain', 'ushape', 'peak_height',
                           'ramp_depth', 'center_depletion', 'asymmetry']:
                out[f'{band}_{metric}'] = np.nan
            out[f'{band}_n_peaks'] = band_n
            continue

        band_assign = assignments[mask]
        for i, pname in enumerate(POS_NAMES):
            count = int((band_assign == i).sum())
            expected = HZ_FRACS[i] * band_n
            out[f'{band}_{pname}'] = (count / expected - 1) * 100 if expected > 0 else 0

        # Derived metrics (same as v3 paper)
        out[f'{band}_mountain'] = out[f'{band}_noble_1'] - out[f'{band}_boundary']
        out[f'{band}_peak_height'] = out[f'{band}_noble_1'] - out[f'{band}_attractor']
        out[f'{band}_ushape'] = (out[f'{band}_boundary'] + out[f'{band}_inv_noble_6']) / 2 - out[f'{band}_attractor']
        out[f'{band}_ramp_depth'] = out[f'{band}_inv_noble_4'] - out[f'{band}_noble_4']
        center = np.mean([out[f'{band}_{p}'] for p in ['noble_5', 'noble_4', 'noble_3']])
        out[f'{band}_center_depletion'] = out[f'{band}_attractor'] - center
        upper = np.mean([out[f'{band}_{p}'] for p in ['inv_noble_3', 'inv_noble_4', 'inv_noble_5']])
        lower = np.mean([out[f'{band}_{p}'] for p in ['noble_5', 'noble_4', 'noble_3']])
        out[f'{band}_asymmetry'] = upper - lower
        out[f'{band}_n_peaks'] = band_n

    return out


def compute_iaf(peaks_df, f0_pop=F0):
    """IAF as power-weighted centroid of alpha-range peaks (7.5-13 Hz)."""
    freqs = peaks_df['freq'].values
    if 'power' not in peaks_df.columns:
        return np.nan
    powers = peaks_df['power'].values
    mask = (freqs >= 7.5) & (freqs <= 13.0) & (powers > 0)
    if mask.sum() < 3:
        return np.nan
    return float(np.average(freqs[mask], weights=powers[mask]))


def power_filter_by_band(df, n_col, min_pct=MIN_POWER_PCT):
    """Keep top (100-min_pct)% of peaks by power within each octave."""
    if min_pct <= 0 or 'power' not in df.columns:
        return df
    out_parts = []
    for n_val in df[n_col].unique():
        sub = df[df[n_col] == n_val]
        if len(sub) == 0:
            continue
        thresh = sub['power'].quantile(min_pct / 100)
        out_parts.append(sub[sub['power'] >= thresh])
    return pd.concat(out_parts, ignore_index=True) if out_parts else df.iloc[0:0]


def process_subject(peaks_df, f0_pop=F0):
    """Return {subject_row_dict} with IAF, population and IAF-anchored enrichment."""
    iaf = compute_iaf(peaks_df, f0_pop=f0_pop)
    if not np.isfinite(iaf):
        return None
    f0_i = iaf / SQRT_PHI  # IAF -> u = 0.5 in the alpha octave

    row = {'iaf': iaf, 'f0_i': f0_i, 'n_peaks_total': len(peaks_df)}

    # Population-anchored (reassign bands with F0, then power-filter, then enrich)
    freqs_all = peaks_df['freq'].values
    n_pop = assign_bands(freqs_all, f0_pop)
    df_pop = peaks_df.assign(_n_pop=n_pop)
    df_pop = power_filter_by_band(df_pop, '_n_pop')
    pop_enr = compute_enrichment_from_freqs(df_pop['freq'].values, f0_pop)
    for k, v in pop_enr.items():
        row[f'pop_{k}'] = v

    # IAF-anchored
    n_iaf = assign_bands(freqs_all, f0_i)
    df_iaf = peaks_df.assign(_n_iaf=n_iaf)
    df_iaf = power_filter_by_band(df_iaf, '_n_iaf')
    iaf_enr = compute_enrichment_from_freqs(df_iaf['freq'].values, f0_i)
    for k, v in iaf_enr.items():
        row[f'iaf_{k}'] = v

    return row


def load_dataset(dataset):
    """Load per-subject peak CSVs for a dataset."""
    peak_dir = os.path.join(PEAK_BASE, dataset)
    files = sorted(glob.glob(os.path.join(peak_dir, '*_peaks.csv')))
    print(f"Loading {len(files)} subjects from {dataset}...")
    rows = []
    for f in files:
        subj = os.path.basename(f).replace('_peaks.csv', '')
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"  skip {subj}: {e}")
            continue
        if len(df) == 0 or 'freq' not in df.columns:
            continue
        result = process_subject(df)
        if result is None:
            continue
        result['subject'] = subj
        rows.append(result)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cognitive / age analysis (LEMON-specific)
# ---------------------------------------------------------------------------

COG_DIR = '/Volumes/T9/lemon_data/behavioral/Behavioural_Data_MPILMBB_LEMON/Cognitive_Test_Battery_LEMON'
META_PATH = '/Volumes/T9/lemon_data/behavioral/Behavioural_Data_MPILMBB_LEMON/META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv'
COG_TESTS = {
    'LPS': ('LPS/LPS.csv', 'LPS_1'),
    'TAP_Incompat': ('TAP_Incompatibility/TAP-Incompatibility.csv', 'TAP_I_1'),
    'RWT': ('RWT/RWT.csv', 'RWT_1'),
    'TMT': ('TMT/TMT.csv', 'TMT_1'),
}


def load_lemon_behavioural(subjects_df):
    """Merge LEMON cognition + age into the subject dataframe."""
    for test, (filename, col) in COG_TESTS.items():
        path = os.path.join(COG_DIR, filename)
        if not os.path.exists(path):
            continue
        cog = pd.read_csv(path)
        cog[col] = pd.to_numeric(cog[col], errors='coerce')
        subjects_df[f'cog_{test}'] = subjects_df['subject'].map(dict(zip(cog['ID'], cog[col])))
    if os.path.exists(META_PATH):
        meta = pd.read_csv(META_PATH)
        def age_mid(s):
            try:
                lo, hi = s.split('-')
                return (float(lo) + float(hi)) / 2
            except Exception:
                return np.nan
        meta['age_mid'] = meta['Age'].apply(age_mid)
        subjects_df['age'] = subjects_df['subject'].map(dict(zip(meta['ID'], meta['age_mid'])))
    return subjects_df


def partial_spearman(x, y, z):
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if valid.sum() < 20:
        return np.nan, np.nan
    x, y, z = x[valid], y[valid], z[valid]
    x_r = x - np.polyval(np.polyfit(z, x, 1), z)
    y_r = y - np.polyval(np.polyfit(z, y, 1), z)
    rho, p = stats.spearmanr(x_r, y_r)
    return float(rho), float(p)


def feature_base_names(cols, prefix):
    """Return sorted list of feature-suffixes present under a prefix."""
    n = len(prefix)
    feats = [c[n:] for c in cols
             if c.startswith(prefix)
             and not c.endswith('_n_peaks')]
    return sorted(set(feats))


def run_comparison(subjects, out_csv):
    """For each feature, correlate both pop-anchored and IAF-anchored versions
    with cognition (raw + age-partialed) and with age."""
    results = []
    feats = feature_base_names(subjects.columns, 'pop_')

    for feat in feats:
        pop_col = f'pop_{feat}'
        iaf_col = f'iaf_{feat}'
        if iaf_col not in subjects.columns:
            continue

        x_pop = subjects[pop_col].values
        x_iaf = subjects[iaf_col].values
        age = subjects['age'].values if 'age' in subjects.columns else np.full(len(subjects), np.nan)

        # Age correlation
        rho_age_pop, p_age_pop = stats.spearmanr(x_pop, age, nan_policy='omit')
        rho_age_iaf, p_age_iaf = stats.spearmanr(x_iaf, age, nan_policy='omit')

        row = {
            'feature': feat,
            'n_pop': np.isfinite(x_pop).sum(),
            'n_iaf': np.isfinite(x_iaf).sum(),
            'rho_age_pop': round(rho_age_pop, 4) if np.isfinite(rho_age_pop) else np.nan,
            'p_age_pop':   round(p_age_pop, 5) if np.isfinite(p_age_pop) else np.nan,
            'rho_age_iaf': round(rho_age_iaf, 4) if np.isfinite(rho_age_iaf) else np.nan,
            'p_age_iaf':   round(p_age_iaf, 5) if np.isfinite(p_age_iaf) else np.nan,
        }

        # Cognitive correlations
        for test in COG_TESTS:
            cog_col = f'cog_{test}'
            if cog_col not in subjects.columns:
                continue
            y = subjects[cog_col].values

            rho_raw_pop, p_raw_pop = stats.spearmanr(x_pop, y, nan_policy='omit')
            rho_raw_iaf, p_raw_iaf = stats.spearmanr(x_iaf, y, nan_policy='omit')
            rho_ap_pop, p_ap_pop = partial_spearman(x_pop, y, age)
            rho_ap_iaf, p_ap_iaf = partial_spearman(x_iaf, y, age)

            row[f'{test}_rho_pop']       = round(rho_raw_pop, 4) if np.isfinite(rho_raw_pop) else np.nan
            row[f'{test}_p_pop']         = round(p_raw_pop, 5) if np.isfinite(p_raw_pop) else np.nan
            row[f'{test}_rho_iaf']       = round(rho_raw_iaf, 4) if np.isfinite(rho_raw_iaf) else np.nan
            row[f'{test}_p_iaf']         = round(p_raw_iaf, 5) if np.isfinite(p_raw_iaf) else np.nan
            row[f'{test}_rho_pop_agept'] = round(rho_ap_pop, 4) if np.isfinite(rho_ap_pop) else np.nan
            row[f'{test}_rho_iaf_agept'] = round(rho_ap_iaf, 4) if np.isfinite(rho_ap_iaf) else np.nan

        results.append(row)

    rdf = pd.DataFrame(results)
    rdf.to_csv(out_csv, index=False)
    return rdf


def summarize(rdf, subjects, out_md):
    """Write a human-readable summary comparing pop vs IAF anchoring."""
    lines = []
    lines.append("# IAF-Anchored Spectral Differentiation: Pop vs IAF anchor comparison\n")
    lines.append(f"Dataset N (with IAF): **{len(subjects)}**  ")
    if 'age' in subjects.columns:
        age_valid = subjects['age'].dropna()
        lines.append(f"With age: **{len(age_valid)}** (range {age_valid.min():.0f}-{age_valid.max():.0f})  ")
    lines.append(f"IAF: **{subjects['iaf'].mean():.2f} +/- {subjects['iaf'].std():.2f} Hz**  ")
    lines.append(f"Per-subject f0_i = IAF_i / sqrt(phi): **{subjects['f0_i'].mean():.2f} +/- {subjects['f0_i'].std():.2f} Hz**  ")
    lines.append(f"Population anchor: f0 = {F0:.2f} Hz\n")

    # ------ 1. Anchor cognitive test: beta-low x LPS ------
    lines.append("## 1. Anchor cognitive test: beta-low center_depletion x LPS\n")
    anchor = rdf[rdf.feature == 'beta_low_center_depletion']
    if len(anchor) > 0:
        r = anchor.iloc[0]
        lines.append(f"| Anchor | Raw rho | Raw p | Age-partialed rho |")
        lines.append(f"|--------|--------:|------:|------------------:|")
        lines.append(f"| Population (f0 = 7.60) | {r.get('LPS_rho_pop', float('nan')):+.3f} | {r.get('LPS_p_pop', float('nan')):.4f} | {r.get('LPS_rho_pop_agept', float('nan')):+.3f} |")
        lines.append(f"| IAF-anchored           | {r.get('LPS_rho_iaf', float('nan')):+.3f} | {r.get('LPS_p_iaf', float('nan')):.4f} | {r.get('LPS_rho_iaf_agept', float('nan')):+.3f} |")
    lines.append("")

    # ------ 2. Top 10 cognitive features either way ------
    lines.append("## 2. Top 10 cognitive features (by |rho|, LPS)\n")
    if 'LPS_rho_pop' in rdf.columns:
        top = rdf.assign(abs_pop=rdf['LPS_rho_pop'].abs()).nlargest(10, 'abs_pop')
        lines.append("| Feature | Pop rho | IAF rho | Pop age-p | IAF age-p |")
        lines.append("|---------|--------:|--------:|----------:|----------:|")
        for _, r in top.iterrows():
            lines.append(
                f"| {r['feature']} | "
                f"{r['LPS_rho_pop']:+.3f} | "
                f"{r['LPS_rho_iaf']:+.3f} | "
                f"{r['LPS_rho_pop_agept']:+.3f} | "
                f"{r['LPS_rho_iaf_agept']:+.3f} |"
            )
    lines.append("")

    # ------ 3. Age correlation summary ------
    lines.append("## 3. Developmental trajectory: age correlations\n")
    if 'rho_age_pop' in rdf.columns:
        valid = rdf.dropna(subset=['rho_age_pop', 'rho_age_iaf']).copy()
        valid['abs_pop'] = valid['rho_age_pop'].abs()
        valid['abs_iaf'] = valid['rho_age_iaf'].abs()
        lines.append(f"Features with finite age-rho: **{len(valid)}** of {len(rdf)}  ")
        lines.append(f"Mean |rho_age| pop-anchored: **{valid['abs_pop'].mean():.3f}**  ")
        lines.append(f"Mean |rho_age| IAF-anchored: **{valid['abs_iaf'].mean():.3f}**  ")
        # How many survive p < 0.05 under each?
        surv_pop = (valid['p_age_pop'] < 0.05).sum()
        surv_iaf = (valid['p_age_iaf'] < 0.05).sum()
        lines.append(f"Features with p_age < 0.05: pop={surv_pop}, IAF={surv_iaf}  \n")
        # Top 10 age features pop
        lines.append("### Top 10 age-correlated features (pop-anchored), both anchors shown\n")
        top = valid.nlargest(10, 'abs_pop')
        lines.append("| Feature | Pop rho_age | IAF rho_age | Attenuation |")
        lines.append("|---------|------------:|------------:|------------:|")
        for _, r in top.iterrows():
            att = 1.0 - (abs(r['rho_age_iaf']) / abs(r['rho_age_pop'])) if abs(r['rho_age_pop']) > 1e-6 else np.nan
            att_str = f"{att:+.1%}" if np.isfinite(att) else "n/a"
            lines.append(
                f"| {r['feature']} | "
                f"{r['rho_age_pop']:+.3f} | "
                f"{r['rho_age_iaf']:+.3f} | "
                f"{att_str} |"
            )
    lines.append("")

    # ------ 4. Interpretation ------
    lines.append("## 4. Interpretation guide\n")
    lines.append("- **If IAF-anchored rho > Pop-anchored rho for LPS**: boundary alignment "
                 "noise was hiding cognitive signal. IAF-anchoring improves measurement.\n")
    lines.append("- **If IAF-anchored rho_age << Pop-anchored rho_age**: much of the developmental "
                 "trajectory was tracking IAF passage through fixed lattice positions, not "
                 "within-band reorganisation. Supports the v3 paper's coupling caveat.\n")
    lines.append("- **If alpha enrichment flattens under IAF-anchor**: the alpha mountain was "
                 "partly an artifact of IAF variance across subjects being projected onto a "
                 "fixed lattice. IAF-anchoring centres everyone's IAF at u = 0.5 by construction.\n")

    with open(out_md, 'w') as f:
        f.write('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='lemon',
                    choices=['lemon', 'lemon_EO', 'dortmund', 'eegmmidb', 'chbmp', 'tdbrain',
                             'hbn_R1', 'hbn_R2', 'hbn_R3', 'hbn_R4', 'hbn_R5',
                             'hbn_R6', 'hbn_R7', 'hbn_R8', 'hbn_R9', 'hbn_R10', 'hbn_R11'])
    args = ap.parse_args()

    subjects = load_dataset(args.dataset)
    print(f"  Subjects with valid IAF: {len(subjects)}")
    if len(subjects) == 0:
        print("  No subjects. Exiting.")
        return
    print(f"  IAF: {subjects['iaf'].mean():.2f} +/- {subjects['iaf'].std():.2f} Hz "
          f"(range {subjects['iaf'].min():.2f}-{subjects['iaf'].max():.2f})")
    print(f"  f0_i: {subjects['f0_i'].mean():.2f} +/- {subjects['f0_i'].std():.2f} Hz")

    # Save per-subject enrichment (both anchors)
    subj_csv = os.path.join(OUT_DIR, f'{args.dataset}_per_subject.csv')
    subjects.to_csv(subj_csv, index=False)
    print(f"  -> {subj_csv}")

    if args.dataset == 'lemon':
        subjects = load_lemon_behavioural(subjects)
        rdf = run_comparison(
            subjects,
            out_csv=os.path.join(OUT_DIR, f'{args.dataset}_comparison.csv'),
        )
        summarize(rdf, subjects,
                  out_md=os.path.join(OUT_DIR, f'{args.dataset}_summary.md'))
        print(f"  -> {os.path.join(OUT_DIR, args.dataset + '_comparison.csv')}")
        print(f"  -> {os.path.join(OUT_DIR, args.dataset + '_summary.md')}")

        # Terminal summary of the anchor test
        anchor = rdf[rdf.feature == 'beta_low_center_depletion']
        if len(anchor) > 0:
            r = anchor.iloc[0]
            print(f"\n  LPS x beta_low_center_depletion:")
            print(f"    Pop-anchored raw: rho = {r.get('LPS_rho_pop', float('nan')):+.3f}, "
                  f"age-partialed: rho = {r.get('LPS_rho_pop_agept', float('nan')):+.3f}")
            print(f"    IAF-anchored raw: rho = {r.get('LPS_rho_iaf', float('nan')):+.3f}, "
                  f"age-partialed: rho = {r.get('LPS_rho_iaf_agept', float('nan')):+.3f}")


if __name__ == '__main__':
    main()
