#!/usr/bin/env python3
"""
Combine all phi-lattice enrichment and summary data into a single queryable CSV.

Outputs:
  master_enrichment.csv          — long-format: one row per (dataset, condition, method, band, position)
  master_summary.csv             — one row per (dataset, condition, method, base) from summary_statistics
  master_enrichment_wide.csv     — wide-format: one row per (dataset, condition, method, position), band columns

Usage:
  python scripts/combine_enrichment_master.py
"""

import os, re, sys
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path("/Users/neurokinetikz/Code/schumann")
OUT_DIR = ROOT / "exports_combined"
OUT_DIR.mkdir(exist_ok=True)

BANDS = ["delta", "theta", "alpha", "beta_low", "beta_high", "gamma"]

# ─── File registry ───────────────────────────────────────────────────────────
# Each entry: (path_pattern, dataset, condition, method)
# method: "OT" = overlap-trim (primary), "standard" = sensitivity_standard

def find_enrichment_files():
    """Discover all 14position_enrichment CSVs and tag with metadata."""
    entries = []

    # ── Local exports ──
    local_datasets = {
        "exports_eegmmidb/replication": ("EEGMMIDB", None),
        "exports_lemon/replication":    ("LEMON", None),
        "exports_chbmp":                ("CHBMP", None),
        "exports_hbn":                  ("HBN_R1", None),
        "exports_hbn_R2":               ("HBN_R2", None),
        "exports_hbn_R3":               ("HBN_R3", None),
        "exports_hbn_R4":               ("HBN_R4", None),
        "exports_hbn_R6":               ("HBN_R6", None),
    }

    for rel_dir, (dataset, _) in local_datasets.items():
        base = ROOT / rel_dir
        if not base.exists():
            continue
        for cond_dir in sorted(base.iterdir()):
            if not cond_dir.is_dir():
                continue
            cond = cond_dir.name  # EC, EO, combined
            # Primary (OT)
            f = cond_dir / "14position_enrichment.csv"
            if f.exists():
                entries.append((f, dataset, cond, "OT"))
            # Sensitivity standard
            f2 = cond_dir / "sensitivity_standard" / "14position_enrichment.csv"
            if f2.exists():
                entries.append((f2, dataset, cond, "standard"))

    # ── Dortmund (external drive) ──
    dortmund_base = Path("/Volumes/T9/dortmund_data/lattice_results_replication_v2")
    if dortmund_base.exists():
        cond_map = {
            "EyesClosed_pre": "EC_pre",
            "EyesClosed_post": "EC_post",
            "EyesOpen_pre": "EO_pre",
            "EyesOpen_post": "EO_post",
        }
        for subdir, cond in cond_map.items():
            f = dortmund_base / subdir / "14position_enrichment.csv"
            if f.exists():
                entries.append((f, "Dortmund", cond, "OT"))
            f2 = dortmund_base / subdir / "sensitivity_standard" / "14position_enrichment.csv"
            if f2.exists():
                entries.append((f2, "Dortmund", cond, "standard"))

    # ── 6-band exports ──
    sixband = ROOT / "exports_6band"
    if sixband.exists():
        for ds_dir in sorted(sixband.iterdir()):
            if not ds_dir.is_dir():
                continue
            for f in ds_dir.glob("14position_enrichment*.csv"):
                entries.append((f, ds_dir.name, "6band", "OT"))

    return entries


def find_summary_files():
    """Discover all summary_statistics CSVs."""
    entries = []

    local_datasets = {
        "exports_eegmmidb/replication": "EEGMMIDB",
        "exports_lemon/replication":    "LEMON",
        "exports_chbmp":                "CHBMP",
        "exports_hbn":                  "HBN_R1",
        "exports_hbn_R2":               "HBN_R2",
        "exports_hbn_R3":               "HBN_R3",
        "exports_hbn_R4":               "HBN_R4",
        "exports_hbn_R6":               "HBN_R6",
    }

    for rel_dir, dataset in local_datasets.items():
        base = ROOT / rel_dir
        if not base.exists():
            continue
        for cond_dir in sorted(base.iterdir()):
            if not cond_dir.is_dir():
                continue
            cond = cond_dir.name
            f = cond_dir / "summary_statistics.csv"
            if f.exists():
                entries.append((f, dataset, cond, "OT"))
            f2 = cond_dir / "sensitivity_standard" / "summary_statistics.csv"
            if f2.exists():
                entries.append((f2, dataset, cond, "standard"))

    dortmund_base = Path("/Volumes/T9/dortmund_data/lattice_results_replication_v2")
    if dortmund_base.exists():
        cond_map = {
            "EyesClosed_pre": "EC_pre",
            "EyesClosed_post": "EC_post",
            "EyesOpen_pre": "EO_pre",
            "EyesOpen_post": "EO_post",
        }
        for subdir, cond in cond_map.items():
            f = dortmund_base / subdir / "summary_statistics.csv"
            if f.exists():
                entries.append((f, "Dortmund", cond, "OT"))
            f2 = dortmund_base / subdir / "sensitivity_standard" / "summary_statistics.csv"
            if f2.exists():
                entries.append((f2, "Dortmund", cond, "standard"))

    return entries


def find_band_position_files():
    """Discover band_position_enrichment CSVs (FOOOF-based)."""
    entries = []
    bp_base = ROOT / "exports_peak_distribution"
    if bp_base.exists():
        for ds_dir in sorted(bp_base.iterdir()):
            if not ds_dir.is_dir():
                continue
            f = ds_dir / "band_position_enrichment.csv"
            if f.exists():
                entries.append((f, ds_dir.name))
    return entries


# ─── Parse & normalize 14position_enrichment ─────────────────────────────────

def parse_enrichment_wide(filepath):
    """Read a 14position_enrichment CSV, return normalized wide DataFrame."""
    df = pd.read_csv(filepath)

    # Normalize column names — the 6band files have swapped order (band_z before band_enrich)
    # Ensure we have: position, u, enrichment_pct, z_score, p_value, and per-band columns
    out_cols = ["position", "u", "enrichment_pct", "z_score", "p_value"]

    # Optional columns
    for c in ["observed_density", "null_mean"]:
        if c in df.columns:
            out_cols.append(c)

    # Per-band columns
    for band in BANDS:
        ecol = f"{band}_enrich"
        zcol = f"{band}_z"
        if ecol in df.columns:
            out_cols.append(ecol)
        if zcol in df.columns:
            out_cols.append(zcol)

    return df[[c for c in out_cols if c in df.columns]]


def melt_enrichment_to_long(df_wide, dataset, condition, method):
    """Convert wide 14position_enrichment to long format: one row per band x position."""
    rows = []
    for _, row in df_wide.iterrows():
        # Overall (all-band) row
        rows.append({
            "dataset": dataset,
            "condition": condition,
            "method": method,
            "position": row["position"],
            "u": row["u"],
            "band": "all",
            "enrichment_pct": row.get("enrichment_pct", np.nan),
            "z_score": row.get("z_score", np.nan),
            "p_value": row.get("p_value", np.nan),
        })
        # Per-band rows
        for band in BANDS:
            ecol = f"{band}_enrich"
            zcol = f"{band}_z"
            if ecol in row.index:
                rows.append({
                    "dataset": dataset,
                    "condition": condition,
                    "method": method,
                    "position": row["position"],
                    "u": row["u"],
                    "band": band,
                    "enrichment_pct": row.get(ecol, np.nan),
                    "z_score": row.get(zcol, np.nan),
                    "p_value": np.nan,  # per-band p not in standard files
                })
    return pd.DataFrame(rows)


# ─── Position metadata ────────────────────────────────────────────────────────

POSITION_META = {
    "boundary":    {"degree": 1, "type": "boundary",    "mirror": "boundary"},
    "noble_7":     {"degree": 7, "type": "noble",       "mirror": "inv_noble_7"},
    "noble_6":     {"degree": 6, "type": "noble",       "mirror": "inv_noble_6"},
    "noble_5":     {"degree": 5, "type": "noble",       "mirror": "inv_noble_5"},
    "noble_4":     {"degree": 4, "type": "noble",       "mirror": "inv_noble_4"},
    "noble_3":     {"degree": 3, "type": "noble",       "mirror": "inv_noble_3"},
    "noble_2":     {"degree": 2, "type": "noble",       "mirror": "inv_noble_2"},
    "attractor":   {"degree": 1, "type": "attractor",   "mirror": "attractor"},
    "noble_1":     {"degree": 1, "type": "noble",       "mirror": "inv_noble_1"},
    "inv_noble_3": {"degree": 3, "type": "inv_noble",   "mirror": "noble_3"},
    "inv_noble_4": {"degree": 4, "type": "inv_noble",   "mirror": "noble_4"},
    "inv_noble_5": {"degree": 5, "type": "inv_noble",   "mirror": "noble_5"},
    "inv_noble_6": {"degree": 6, "type": "inv_noble",   "mirror": "noble_6"},
    "inv_noble_7": {"degree": 7, "type": "inv_noble",   "mirror": "noble_7"},
}

# Dataset metadata
DATASET_META = {
    "EEGMMIDB":  {"population": "adult",     "N_approx": 109, "fs_hz": 160, "n_channels": 64},
    "LEMON":     {"population": "adult",     "N_approx": 202, "fs_hz": 250, "n_channels": 62},
    "Dortmund":  {"population": "adult",     "N_approx": 608, "fs_hz": 250, "n_channels": 64},
    "CHBMP":     {"population": "adult",     "N_approx": 249, "fs_hz": 200, "n_channels": 120},
    "HBN_R1":    {"population": "pediatric", "N_approx": 136, "fs_hz": 500, "n_channels": 128},
    "HBN_R2":    {"population": "pediatric", "N_approx": 150, "fs_hz": 500, "n_channels": 128},
    "HBN_R3":    {"population": "pediatric", "N_approx": 182, "fs_hz": 500, "n_channels": 128},
    "HBN_R4":    {"population": "pediatric", "N_approx": 319, "fs_hz": 500, "n_channels": 128},
    "HBN_R6":    {"population": "pediatric", "N_approx": 135, "fs_hz": 500, "n_channels": 128},
}

# Condition normalization
def normalize_condition(cond):
    """Map raw condition strings to standardized eye-state labels."""
    cond_lower = cond.lower()
    if cond_lower in ("ec", "eyesclosed_pre", "ec_pre"):
        return "EC"
    elif cond_lower in ("eo", "eyesopen_pre", "eo_pre"):
        return "EO"
    elif cond_lower in ("eyesclosed_post", "ec_post"):
        return "EC_post"
    elif cond_lower in ("eyesopen_post", "eo_post"):
        return "EO_post"
    elif cond_lower == "combined":
        return "combined"
    elif cond_lower == "6band":
        return "6band"
    return cond


def add_position_metadata(df):
    """Add degree, type, mirror columns from position name."""
    df["position_degree"] = df["position"].map(lambda p: POSITION_META.get(p, {}).get("degree", np.nan))
    df["position_type"] = df["position"].map(lambda p: POSITION_META.get(p, {}).get("type", "unknown"))
    df["mirror_position"] = df["position"].map(lambda p: POSITION_META.get(p, {}).get("mirror", ""))
    return df


def add_dataset_metadata(df):
    """Add population, N, fs, n_channels from dataset name."""
    for col, key in [("population", "population"), ("N_approx", "N_approx"),
                     ("fs_hz", "fs_hz"), ("n_channels", "n_channels")]:
        df[col] = df["dataset"].map(lambda d: DATASET_META.get(d, {}).get(key, np.nan))
    return df


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Combining all phi-lattice enrichment data")
    print("=" * 70)

    # ── 1. 14position_enrichment → long format ──
    enrich_files = find_enrichment_files()
    print(f"\nFound {len(enrich_files)} 14position_enrichment files")

    all_long = []
    all_wide = []
    for filepath, dataset, cond, method in enrich_files:
        try:
            df_w = parse_enrichment_wide(filepath)
            df_l = melt_enrichment_to_long(df_w, dataset, normalize_condition(cond), method)
            all_long.append(df_l)

            # Wide version with metadata
            df_w2 = df_w.copy()
            df_w2.insert(0, "dataset", dataset)
            df_w2.insert(1, "condition", normalize_condition(cond))
            df_w2.insert(2, "method", method)
            all_wide.append(df_w2)

            print(f"  OK: {dataset} / {cond} / {method} ({len(df_w)} positions)")
        except Exception as e:
            print(f"  FAIL: {filepath}: {e}")

    if all_long:
        master_long = pd.concat(all_long, ignore_index=True)
        master_long = add_position_metadata(master_long)
        master_long = add_dataset_metadata(master_long)

        # Add derived columns for easy querying
        master_long["is_significant"] = master_long["z_score"].abs() >= 1.96
        master_long["direction"] = np.where(master_long["enrichment_pct"] > 0, "enriched",
                                   np.where(master_long["enrichment_pct"] < 0, "depleted", "neutral"))
        master_long["abs_enrichment"] = master_long["enrichment_pct"].abs()

        # Eye state column (EC/EO/combined, ignoring pre/post)
        master_long["eye_state"] = master_long["condition"].map(
            lambda c: "EC" if "EC" in c else ("EO" if "EO" in c else c))

        # HBN aggregate flag
        master_long["is_hbn"] = master_long["dataset"].str.startswith("HBN")
        master_long["is_adult"] = master_long["population"] == "adult"

        out_long = OUT_DIR / "master_enrichment.csv"
        master_long.to_csv(out_long, index=False)
        print(f"\n  -> {out_long}")
        print(f"     {len(master_long)} rows = {master_long['dataset'].nunique()} datasets "
              f"x {master_long['condition'].nunique()} conditions "
              f"x {master_long['band'].nunique()} bands "
              f"x {master_long['position'].nunique()} positions")

        # Wide version
        master_wide = pd.concat(all_wide, ignore_index=True)
        master_wide = add_position_metadata(master_wide)
        master_wide = add_dataset_metadata(master_wide)
        out_wide = OUT_DIR / "master_enrichment_wide.csv"
        master_wide.to_csv(out_wide, index=False)
        print(f"  -> {out_wide} ({len(master_wide)} rows)")

    # ── 2. summary_statistics ──
    summary_files = find_summary_files()
    print(f"\nFound {len(summary_files)} summary_statistics files")

    all_summary = []
    for filepath, dataset, cond, method in summary_files:
        try:
            df = pd.read_csv(filepath)
            df.insert(0, "dataset", dataset)
            df.insert(1, "condition", normalize_condition(cond))
            df.insert(2, "method", method)
            all_summary.append(df)
            print(f"  OK: {dataset} / {cond} / {method}")
        except Exception as e:
            print(f"  FAIL: {filepath}: {e}")

    if all_summary:
        master_summary = pd.concat(all_summary, ignore_index=True)
        master_summary = add_dataset_metadata(master_summary)
        out_summary = OUT_DIR / "master_summary.csv"
        master_summary.to_csv(out_summary, index=False)
        print(f"\n  -> {out_summary} ({len(master_summary)} rows)")

    # ── 3. band_position_enrichment (FOOOF) ──
    bp_files = find_band_position_files()
    print(f"\nFound {len(bp_files)} band_position_enrichment files")

    all_bp = []
    for filepath, variant in bp_files:
        try:
            df = pd.read_csv(filepath)
            df.insert(0, "variant", variant)
            all_bp.append(df)
            print(f"  OK: {variant} ({len(df)} rows)")
        except Exception as e:
            print(f"  FAIL: {filepath}: {e}")

    if all_bp:
        master_bp = pd.concat(all_bp, ignore_index=True)
        out_bp = OUT_DIR / "master_band_position_fooof.csv"
        master_bp.to_csv(out_bp, index=False)
        print(f"\n  -> {out_bp} ({len(master_bp)} rows)")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("DONE. All outputs in:", OUT_DIR)
    print("=" * 70)

    if all_long:
        print("\n── Quick stats for master_enrichment.csv ──")
        ml = master_long[master_long["band"] != "all"]  # per-band only
        print(f"  Total per-band rows: {len(ml)}")
        print(f"  Datasets: {sorted(ml['dataset'].unique())}")
        print(f"  Conditions: {sorted(ml['condition'].unique())}")
        print(f"  Methods: {sorted(ml['method'].unique())}")
        print(f"  Bands: {sorted(ml['band'].unique())}")
        print(f"  Positions: {sorted(ml['position'].unique())}")

        print("\n── Example queries (pandas) ──")
        print("""
  import pandas as pd
  df = pd.read_csv("exports_combined/master_enrichment.csv")

  # All adult EC data, OT method, per-band
  q = df[(df.is_adult) & (df.eye_state=="EC") & (df.method=="OT") & (df.band!="all")]

  # Theta at boundary across all datasets
  q = df[(df.band=="theta") & (df.position=="boundary") & (df.method=="OT")]

  # Noble vs inv_noble comparison for gamma
  q = df[(df.band=="gamma") & (df.position_type.isin(["noble","inv_noble"])) & (df.method=="OT")]

  # Mean enrichment by position_type and band (adult EC only)
  pivot = q.groupby(["band","position_type"])["enrichment_pct"].mean().unstack()

  # Heatmap: band x position, averaged across adult datasets
  adult_ec = df[(df.is_adult) & (df.eye_state=="EC") & (df.method=="OT") & (df.band!="all")]
  heatmap = adult_ec.groupby(["band","position"])["enrichment_pct"].mean().unstack()

  # Which positions are significantly enriched across ALL adult datasets?
  sig = df[(df.is_adult) & (df.method=="OT") & (df.band!="all") & (df.is_significant)]
  sig.groupby(["band","position"])["dataset"].nunique()

  # Compare noble push-zone (nobles 3-5) depletion by band
  push = df[(df.position.isin(["noble_3","noble_4","noble_5"])) & (df.method=="OT") & (df.band!="all")]
  push.groupby(["band","dataset"])["enrichment_pct"].mean()

  # Mirror symmetry: noble vs inverse noble at same degree
  nobles = df[(df.position_type=="noble") & (df.position_degree>=3)]
  inv_nobles = df[(df.position_type=="inv_noble") & (df.position_degree>=3)]
  # merge on (dataset, condition, method, band, position_degree)
""")


if __name__ == "__main__":
    main()
