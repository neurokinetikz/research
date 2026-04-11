#!/usr/bin/env python
"""
Cross-dataset position consistency analysis for phi-octave enrichment.

Parses enrichment tables from phi_octave_enrichment_tables.txt and
replication_pipeline.md, then identifies which position × band cells
show consistent enrichment/depletion across all datasets.

Usage:
    python scripts/cross_dataset_position_consistency.py
"""
import os, re, sys
import numpy as np
from scipy.stats import binomtest

os.chdir(os.path.join(os.path.dirname(__file__), '..'))

# ═══════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════
POSITIONS_14 = [
    'boundary', 'noble_7', 'noble_6', 'noble_5', 'noble_4',
    'noble_3', 'noble_2', 'attractor', 'noble_1',
    'inv_noble_3', 'inv_noble_4', 'inv_noble_5', 'inv_noble_6', 'inv_noble_7',
]

POS_U = {
    'boundary': 0.000, 'noble_7': 0.034, 'noble_6': 0.056,
    'noble_5': 0.090, 'noble_4': 0.146, 'noble_3': 0.236,
    'noble_2': 0.382, 'attractor': 0.500, 'noble_1': 0.618,
    'inv_noble_3': 0.764, 'inv_noble_4': 0.854, 'inv_noble_5': 0.910,
    'inv_noble_6': 0.944, 'inv_noble_7': 0.966,
}

PHI_OCTAVE_BANDS = ['phi_-2', 'phi_-1', 'phi_0', 'phi_1', 'phi_2', 'phi_3', 'phi_4']
CONV_BANDS = ['delta', 'theta', 'alpha', 'beta_low', 'beta_high', 'gamma']

EC_DATASETS = ['EEGMMIDB EC', 'Dortmund EC', 'HBN EC', 'LEMON EC']
ALL_DATASETS_PHI = ['EEGMMIDB EC', 'Dortmund EC', 'Dortmund EO', 'HBN EC', 'LEMON EC', 'LEMON EO']
ALL_DATASETS_CONV = [
    'EEGMMIDB EC', 'Dortmund EC-pre', 'LEMON EC', 'HBN EC',
    'Dortmund EO-pre', 'LEMON EO',
]

# Mapping for dataset name normalization between files
CONV_TO_UNIFIED = {
    'EEGMMIDB EC': 'EEGMMIDB EC',
    'Dortmund EC-pre': 'Dortmund EC',
    'Dortmund EO-pre': 'Dortmund EO',
    'HBN EC': 'HBN EC',
    'LEMON EC': 'LEMON EC',
    'LEMON EO': 'LEMON EO',
}

EC_UNIFIED = ['EEGMMIDB EC', 'Dortmund EC', 'HBN EC', 'LEMON EC']


# ═══════════════════════════════════════════════════════════════════
# PARSERS
# ═══════════════════════════════════════════════════════════════════

def parse_phi_octave_tables(filepath):
    """Parse phi_octave_enrichment_tables.txt.

    Returns dict: dataset_name -> {position -> {band -> enrichment_pct}}
    """
    with open(filepath) as f:
        text = f.read()

    # Split into dataset blocks
    blocks = re.split(r'={50,}\n\s+(.+?)\s+\(N=\d+\)\n={50,}', text)
    # blocks[0] = header, then alternating: dataset_name, block_content

    data = {}
    for i in range(1, len(blocks), 2):
        ds_name = blocks[i].strip()
        block = blocks[i + 1]

        ds_data = {}
        for line in block.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('Position') or line.startswith('-') or line.startswith('n_peaks'):
                continue
            # Match position lines: "boundary       0.000       -25%       +30%  ..."
            m = re.match(r'(\S+)\s+([\d.]+)\s+(.*)', line)
            if not m:
                continue
            pos = m.group(1)
            if pos not in POS_U:
                continue
            vals_str = m.group(3)
            # Parse each value (+25%, -10%, ---)
            vals = re.findall(r'([+-]?\d+)%|---', vals_str)
            pos_data = {}
            for j, band in enumerate(PHI_OCTAVE_BANDS):
                if j < len(vals):
                    v = vals[j]
                    if isinstance(v, tuple):
                        v = v[0] if v[0] else None
                    if v and v != '---' and v != '':
                        pos_data[band] = float(v)
                    else:
                        pos_data[band] = np.nan
                else:
                    pos_data[band] = np.nan
            ds_data[pos] = pos_data
        data[ds_name] = ds_data

    return data


def parse_conventional_band_tables(filepath):
    """Parse per-band position enrichment from replication_pipeline.md.

    Returns dict: dataset_name -> {position -> {band -> enrichment_pct}}
    """
    with open(filepath) as f:
        text = f.read()

    # Find all code blocks after dataset headers
    # Pattern: "### DatasetName (N=xxx)\n```\n...table...\n```"
    # Use [^\n]+? for name to prevent crossing newlines (DOTALL needed for block)
    pattern = r'### ([^\n]+?)\s+\(N=\d+\)\s*\n```\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)

    data = {}
    for ds_name, block in matches:
        ds_name = ds_name.strip()
        ds_data = {}
        for line in block.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('Position') or line.startswith('-'):
                continue
            m = re.match(r'(\S+)\s+([\d.]+)\s+(.*)', line)
            if not m:
                continue
            pos = m.group(1)
            if pos not in POS_U:
                continue
            vals_str = m.group(3)
            vals = re.findall(r'([+-]?\d+)%|---', vals_str)
            pos_data = {}
            for j, band in enumerate(CONV_BANDS):
                if j < len(vals):
                    v = vals[j]
                    if isinstance(v, tuple):
                        v = v[0] if v[0] else None
                    if v and v != '---' and v != '':
                        pos_data[band] = float(v)
                    else:
                        pos_data[band] = np.nan
                else:
                    pos_data[band] = np.nan
            ds_data[pos] = pos_data
        data[ds_name] = ds_data

    return data


# ═══════════════════════════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def compute_consistency(data, datasets, bands, positions=POSITIONS_14):
    """Compute cross-dataset consistency for each position × band.

    Returns dict: (position, band) -> {
        'values': list of enrichment values,
        'n_avail': int,
        'n_pos': int, 'n_neg': int,
        'mean': float, 'median': float, 'sd': float,
        'min': float, 'max': float,
        'sign_consistency': float,
        'sign_p': float (binomial test),
        'classification': str
    }
    """
    results = {}
    for pos in positions:
        for band in bands:
            vals = []
            for ds in datasets:
                if ds not in data:
                    continue
                if pos not in data[ds]:
                    continue
                v = data[ds][pos].get(band, np.nan)
                if not np.isnan(v):
                    vals.append(v)

            n = len(vals)
            if n == 0:
                results[(pos, band)] = {
                    'values': [], 'n_avail': 0,
                    'n_pos': 0, 'n_neg': 0,
                    'mean': np.nan, 'median': np.nan, 'sd': np.nan,
                    'min': np.nan, 'max': np.nan,
                    'sign_consistency': np.nan, 'sign_p': np.nan,
                    'classification': 'no_data',
                }
                continue

            n_pos = sum(1 for v in vals if v > 0)
            n_neg = sum(1 for v in vals if v < 0)
            n_zero = n - n_pos - n_neg
            k = max(n_pos, n_neg)

            # Binomial test: is the dominant sign more frequent than chance?
            if n > 0 and k > 0:
                bt = binomtest(k, n, 0.5, alternative='greater')
                sign_p = bt.pvalue
            else:
                sign_p = 1.0

            sign_consistency = k / n if n > 0 else 0.0
            mean_val = np.mean(vals)
            median_val = np.median(vals)
            sd_val = np.std(vals, ddof=1) if n > 1 else 0.0

            # Classification
            if n_pos == n and n >= 4 and sign_p < 0.05 and mean_val > 20:
                cls = '+++'
            elif n_neg == n and n >= 4 and sign_p < 0.05 and mean_val < -20:
                cls = '---'
            elif n_pos >= n - 1 and n >= 4 and mean_val > 10:
                cls = '++'
            elif n_neg >= n - 1 and n >= 4 and mean_val < -10:
                cls = '--'
            elif n_pos >= n - 1 and n >= 4 and mean_val > 0:
                cls = '+'
            elif n_neg >= n - 1 and n >= 4 and mean_val < 0:
                cls = '-'
            else:
                cls = '.'

            results[(pos, band)] = {
                'values': vals,
                'n_avail': n,
                'n_pos': n_pos, 'n_neg': n_neg,
                'mean': mean_val, 'median': median_val, 'sd': sd_val,
                'min': min(vals), 'max': max(vals),
                'sign_consistency': sign_consistency,
                'sign_p': sign_p,
                'classification': cls,
            }

    return results


# ═══════════════════════════════════════════════════════════════════
# OUTPUT FORMATTING
# ═══════════════════════════════════════════════════════════════════

def print_full_matrix(results, bands, label, out):
    """Table 1: Full position × band matrix with classification."""
    out.write(f"\n{'='*120}\n")
    out.write(f"  {label}\n")
    out.write(f"{'='*120}\n")

    # Header
    hdr = f"{'Position':<15s} {'u':>5s}"
    for b in bands:
        hdr += f"  {b:>12s}"
    out.write(hdr + '\n')
    out.write('-' * len(hdr) + '\n')

    for pos in POSITIONS_14:
        row = f"{pos:<15s} {POS_U[pos]:>5.3f}"
        for band in bands:
            r = results.get((pos, band))
            if r is None or r['n_avail'] == 0:
                row += f"  {'---':>12s}"
            else:
                cls = r['classification']
                mean = r['mean']
                row += f"  {mean:>+6.0f}% {cls:>4s}"
        out.write(row + '\n')


def print_predictable_by_band(results, bands, label, out):
    """Table 2: Predictable positions per band."""
    out.write(f"\n{'='*120}\n")
    out.write(f"  {label} — Predictable Positions by Band\n")
    out.write(f"{'='*120}\n")

    for band in bands:
        enriched = []
        depleted = []
        for pos in POSITIONS_14:
            r = results.get((pos, band))
            if r is None or r['n_avail'] == 0:
                continue
            cls = r['classification']
            if cls in ('+++', '++', '+'):
                enriched.append((pos, r))
            elif cls in ('---', '--', '-'):
                depleted.append((pos, r))

        enriched.sort(key=lambda x: -x[1]['mean'])
        depleted.sort(key=lambda x: x[1]['mean'])

        out.write(f"\n  {band}\n")
        out.write(f"  {'─'*80}\n")

        if enriched:
            out.write(f"    ENRICHED:\n")
            for pos, r in enriched:
                sign_str = f"{r['n_pos']}/{r['n_avail']}"
                rng = f"[{r['min']:+.0f}%, {r['max']:+.0f}%]"
                out.write(f"      {pos:<15s}  mean={r['mean']:>+6.0f}%  {r['classification']:>3s}  "
                          f"sign={sign_str}  p={r['sign_p']:.3f}  range={rng}\n")
        if depleted:
            out.write(f"    DEPLETED:\n")
            for pos, r in depleted:
                sign_str = f"{r['n_neg']}/{r['n_avail']}"
                rng = f"[{r['min']:+.0f}%, {r['max']:+.0f}%]"
                out.write(f"      {pos:<15s}  mean={r['mean']:>+6.0f}%  {r['classification']:>3s}  "
                          f"sign={sign_str}  p={r['sign_p']:.3f}  range={rng}\n")
        if not enriched and not depleted:
            out.write(f"    (no predictable positions)\n")


def print_position_profiles(results, bands, label, out):
    """Table 3: Position profiles — fingerprint per position."""
    out.write(f"\n{'='*120}\n")
    out.write(f"  {label} — Position Profiles\n")
    out.write(f"{'='*120}\n")

    for pos in POSITIONS_14:
        enriched_bands = []
        depleted_bands = []
        inconsistent_bands = []
        for band in bands:
            r = results.get((pos, band))
            if r is None or r['n_avail'] == 0:
                continue
            cls = r['classification']
            if cls in ('+++', '++', '+'):
                enriched_bands.append((band, r['mean'], cls))
            elif cls in ('---', '--', '-'):
                depleted_bands.append((band, r['mean'], cls))
            else:
                inconsistent_bands.append((band, r['mean']))

        out.write(f"\n  {pos} (u={POS_U[pos]:.3f})\n")
        if enriched_bands:
            parts = [f"{b} ({m:+.0f}% {c})" for b, m, c in enriched_bands]
            out.write(f"    Enriched in:  {', '.join(parts)}\n")
        if depleted_bands:
            parts = [f"{b} ({m:+.0f}% {c})" for b, m, c in depleted_bands]
            out.write(f"    Depleted in:  {', '.join(parts)}\n")
        if inconsistent_bands:
            parts = [f"{b} ({m:+.0f}%)" for b, m in inconsistent_bands]
            out.write(f"    Inconsistent: {', '.join(parts)}\n")
        if not enriched_bands and not depleted_bands and not inconsistent_bands:
            out.write(f"    (no data)\n")


def print_summary_narrative(results_all6, results_ec4, bands, label, out):
    """Table 5: Summary narrative."""
    out.write(f"\n{'='*120}\n")
    out.write(f"  {label} — Summary\n")
    out.write(f"{'='*120}\n")

    # Count predictable cells
    strong_e = [(p, b) for (p, b), r in results_all6.items() if r['classification'] == '+++']
    strong_d = [(p, b) for (p, b), r in results_all6.items() if r['classification'] == '---']
    pred_e = [(p, b) for (p, b), r in results_all6.items() if r['classification'] in ('+++', '++')]
    pred_d = [(p, b) for (p, b), r in results_all6.items() if r['classification'] in ('---', '--')]
    weak_e = [(p, b) for (p, b), r in results_all6.items() if r['classification'] == '+']
    weak_d = [(p, b) for (p, b), r in results_all6.items() if r['classification'] == '-']
    incon = [(p, b) for (p, b), r in results_all6.items() if r['classification'] == '.']
    no_data = [(p, b) for (p, b), r in results_all6.items() if r['classification'] == 'no_data']

    total = len(results_all6)
    out.write(f"\n  Total cells: {total} ({len(POSITIONS_14)} positions × {len(bands)} bands)\n")
    out.write(f"  Strongly enriched (+++): {len(strong_e)}\n")
    out.write(f"  Strongly depleted (---): {len(strong_d)}\n")
    out.write(f"  Predictable enriched (++ or +++): {len(pred_e)}\n")
    out.write(f"  Predictable depleted (-- or ---): {len(pred_d)}\n")
    out.write(f"  Weakly predictable (+/-): {len(weak_e) + len(weak_d)}\n")
    out.write(f"  Inconsistent (.): {len(incon)}\n")
    out.write(f"  No data: {len(no_data)}\n")

    # Strongest signals
    all_pred = [(p, b, results_all6[(p, b)]) for p, b in pred_e + pred_d]
    all_pred.sort(key=lambda x: -abs(x[2]['mean']))

    out.write(f"\n  Top 10 strongest predictable signals (by |mean enrichment|):\n")
    for p, b, r in all_pred[:10]:
        sign_str = f"{r['n_pos']}+/{r['n_neg']}-" if r['mean'] > 0 else f"{r['n_neg']}-/{r['n_pos']}+"
        out.write(f"    {p:<15s} × {b:<10s}  mean={r['mean']:>+7.0f}%  {r['classification']:>3s}  "
                  f"({sign_str} of {r['n_avail']})  range=[{r['min']:+.0f}%, {r['max']:+.0f}%]\n")

    # Per-band summary
    out.write(f"\n  Per-band predictability:\n")
    for band in bands:
        n_pred = sum(1 for (p, b), r in results_all6.items()
                     if b == band and r['classification'] in ('+++', '++', '--', '---'))
        n_weak = sum(1 for (p, b), r in results_all6.items()
                     if b == band and r['classification'] in ('+', '-'))
        n_incon = sum(1 for (p, b), r in results_all6.items()
                      if b == band and r['classification'] == '.')
        n_avail = sum(1 for (p, b), r in results_all6.items()
                      if b == band and r['n_avail'] > 0)
        out.write(f"    {band:<10s}: {n_pred} predictable, {n_weak} weak, "
                  f"{n_incon} inconsistent (of {n_avail} cells)\n")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import io

    # ── Parse data ──
    phi_data = parse_phi_octave_tables('phi_octave_enrichment_tables.txt')
    conv_data = parse_conventional_band_tables('replication_pipeline.md')

    print(f"Parsed phi-octave data: {list(phi_data.keys())}")
    print(f"Parsed conventional data: {list(conv_data.keys())}")

    # ── Phi-octave analysis ──
    phi_all6 = compute_consistency(phi_data, ALL_DATASETS_PHI, PHI_OCTAVE_BANDS)
    phi_ec4 = compute_consistency(phi_data, EC_DATASETS, PHI_OCTAVE_BANDS)

    # ── Conventional-band analysis ──
    # Normalize names
    conv_data_unified = {}
    for ds_name, ds_data in conv_data.items():
        unified = CONV_TO_UNIFIED.get(ds_name, ds_name)
        conv_data_unified[unified] = ds_data

    CONV_ALL6 = ['EEGMMIDB EC', 'Dortmund EC', 'Dortmund EO',
                 'HBN EC', 'LEMON EC', 'LEMON EO']
    conv_all6 = compute_consistency(conv_data_unified, CONV_ALL6, CONV_BANDS)
    conv_ec4 = compute_consistency(conv_data_unified, EC_UNIFIED, CONV_BANDS)

    # ── Output ──
    out = io.StringIO()

    out.write("CROSS-DATASET POSITION CONSISTENCY ANALYSIS\n")
    out.write("=" * 120 + '\n')
    out.write(f"Classification: +++ = strongly enriched (all same sign, p<0.05, |mean|>20%)\n")
    out.write(f"                ---  = strongly depleted\n")
    out.write(f"                ++/-- = predictable (≥N-1 same sign, |mean|>10%)\n")
    out.write(f"                +/-  = weakly predictable (≥N-1 same sign, 0<|mean|≤10%)\n")
    out.write(f"                .    = inconsistent\n")

    # ═══════ PHI-OCTAVE BANDS ═══════
    out.write(f"\n\n{'#'*120}\n")
    out.write(f"  PART 1: PHI-OCTAVE BANDS (phi_-2 through phi_4)\n")
    out.write(f"{'#'*120}\n")

    print_full_matrix(phi_all6, PHI_OCTAVE_BANDS, "All 6 Conditions — Phi-Octave Bands", out)
    print_full_matrix(phi_ec4, PHI_OCTAVE_BANDS, "4 EC Only (Independent Datasets) — Phi-Octave Bands", out)
    print_predictable_by_band(phi_all6, PHI_OCTAVE_BANDS, "All 6 Conditions — Phi-Octave Bands", out)
    print_position_profiles(phi_all6, PHI_OCTAVE_BANDS, "All 6 Conditions — Phi-Octave Bands", out)
    print_summary_narrative(phi_all6, phi_ec4, PHI_OCTAVE_BANDS, "Phi-Octave Bands", out)

    # ═══════ CONVENTIONAL BANDS ═══════
    out.write(f"\n\n{'#'*120}\n")
    out.write(f"  PART 2: CONVENTIONAL BANDS (delta through gamma)\n")
    out.write(f"{'#'*120}\n")

    print_full_matrix(conv_all6, CONV_BANDS, "All 6 Conditions — Conventional Bands", out)
    print_full_matrix(conv_ec4, CONV_BANDS, "4 EC Only (Independent Datasets) — Conventional Bands", out)
    print_predictable_by_band(conv_all6, CONV_BANDS, "All 6 Conditions — Conventional Bands", out)
    print_position_profiles(conv_all6, CONV_BANDS, "All 6 Conditions — Conventional Bands", out)
    print_summary_narrative(conv_all6, conv_ec4, CONV_BANDS, "Conventional Bands", out)

    # ═══════ Write output ═══════
    text = out.getvalue()
    print(text)

    with open('cross_dataset_position_consistency.txt', 'w') as f:
        f.write(text)

    print(f"\nOutput written to cross_dataset_position_consistency.txt")
