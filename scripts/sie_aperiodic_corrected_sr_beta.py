#!/usr/bin/env python3
"""
Aperiodic-Corrected SR×β Cross-Subject Coupling (Supplementary Table S3)
==========================================================================

Tests whether the flagship pediatric SR×β coupling finding (3-5× stronger
in pediatric than adult cohorts; ρ = +0.534 to +0.819 across HBN R1-R11)
reflects genuine periodic-band coupling or age-tracking 1/f slope confound.

For each subject in HBN R7, HBN R11, and LEMON EC:
1. Load raw EEG (BrainVision or BIDS .set), preprocess (HP 1 Hz, notch,
   resample 250 Hz).
2. For each composite-v2 SIE event, extract a 4-s window centered on
   t0_net.
3. Compute posterior-channel-mean Welch PSD.
4. Compute three amplitude estimates per event:
   (a) RAW: peak power in 7.0-8.2 Hz (SR1) and 15-22 Hz (β) of Welch PSD
   (b) FOOOF-corrected: peak power in 7.0-8.2 Hz and 15-22 Hz of FOOOF
       periodic-only spectrum (after specparam aperiodic removal)
   (c) IRASA-corrected: peak power in 7.0-8.2 Hz and 15-22 Hz of IRASA
       periodic-only spectrum (P_osc)
5. Per-subject weighted mean across events (weights = max(rho_template, 0)).
6. Cross-subject Spearman ρ between SR1 and β amplitudes (one ρ per cohort
   × method).

Tests whether pediatric ρ remains substantially elevated over adult after
aperiodic correction. If yes, the developmental gradient reflects genuine
periodic-band coupling. If pediatric ρ collapses toward adult, the
gradient is driven by 1/f slope.

Usage (on VM):
    bash scripts/vm_run.sh scripts/sie_aperiodic_corrected_sr_beta.py [--cohort COHORT]

Usage (local, if cohort raw data accessible):
    /opt/anaconda3/envs/brainwaves/bin/python3 \
        scripts/sie_aperiodic_corrected_sr_beta.py --cohort lemon_EC

Outputs:
    outputs/schumann/2026-04-27-aperiodic-corrected-sr-beta/
        - per_event_amplitudes_<cohort>.csv
        - per_subject_amplitudes_<cohort>.csv
        - cross_cohort_summary.csv
        - cross_cohort_summary.txt
"""

import os
import sys
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.stats import spearmanr
import mne

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "lib"))
from shape_vs_resonance import irasa_psd  # noqa: E402

try:
    from specparam import SpectralModel as FOOOF
    HAS_SPECPARAM = True
except ImportError:
    try:
        from fooof import FOOOF
        HAS_SPECPARAM = False
    except ImportError:
        FOOOF = None
        HAS_SPECPARAM = False

# --- Cohort / data path configuration ----------------------------------------
# On the VM, /Volumes/T9 is symlinked to /mnt/eeg-data/T9; on local Mac the
# path is /Volumes/T9. Both work via the same Path here.
T9 = Path("/Volumes/T9")
if not T9.exists():
    T9 = Path("/mnt/eeg-data/T9")

COHORTS = {
    "lemon_EC": {
        "raw_root": T9 / "lemon_data" / "eeg_raw" / "EEG_MPILMBB_LEMON" / "EEG_Raw_BIDS_ID",
        "raw_pattern": "{sid}/RSEEG/{sid}.vhdr",
        "raw_format": "brainvision",
        "events_dir": BASE_DIR / "exports_sie" / "lemon_composite",
        "events_filter": ("condition", "EC"),
        "raw_eeg_filter_func": None,
    },
    "hbn_R7": {
        "raw_root": T9 / "hbn_data" / "cmi_bids_R7",
        "raw_pattern": "{sid}/eeg/{sid}_task-RestingState_eeg.set",
        "raw_format": "set",
        "events_dir": BASE_DIR / "exports_sie" / "hbn_R7_composite",
        "events_filter": None,
        "raw_eeg_filter_func": None,
    },
    "hbn_R11": {
        "raw_root": T9 / "hbn_data" / "cmi_bids_R11",
        "raw_pattern": "{sid}/eeg/{sid}_task-RestingState_eeg.set",
        "raw_format": "set",
        "events_dir": BASE_DIR / "exports_sie" / "hbn_R11_composite",
        "events_filter": None,
        "raw_eeg_filter_func": None,
    },
}

OUT_DIR = BASE_DIR / "outputs" / "schumann" / "2026-04-27-aperiodic-corrected-sr-beta"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Posterior-channel selection
POSTERIOR_CHANNELS_LEMON = ["O1", "O2", "Oz", "PO3", "PO4", "POz", "PO7", "PO8",
                             "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "Pz"]
# HBN uses EGI 128-ch; map to occipitoparietal
POSTERIOR_CHANNELS_HBN = ["E70", "E75", "E83", "E76", "E71", "E74", "E62", "E72",
                          "E73", "E69", "E68", "E67", "E61", "E66", "E65", "E60",
                          "E81", "E82"]  # approximate posterior cluster

WINDOW_SEC = 4.0
HALF_WIN = WINDOW_SEC / 2

SR1_BAND = (7.0, 8.2)
BETA_BAND = (15.0, 22.0)


def load_raw(sid, cfg):
    """Load raw EEG for subject sid using cohort cfg; return MNE Raw or None."""
    raw_path = cfg["raw_root"] / cfg["raw_pattern"].format(sid=sid)
    if not raw_path.exists():
        return None
    try:
        if cfg["raw_format"] == "brainvision":
            raw = mne.io.read_raw_brainvision(raw_path, preload=True, verbose="error")
        elif cfg["raw_format"] == "set":
            raw = mne.io.read_raw_eeglab(raw_path, preload=True, verbose="error")
        else:
            return None
    except Exception as exc:
        print(f"  [{sid}] read_raw failed: {exc}")
        return None
    raw.filter(l_freq=1.0, h_freq=None, fir_design="firwin", verbose="error")
    if 50.0 < raw.info["sfreq"] / 2:
        raw.notch_filter(freqs=[50.0], verbose="error")
    if raw.info["sfreq"] > 250:
        raw.resample(250.0, verbose="error")
    return raw


def posterior_signal(raw, t_start, t_end, channels):
    avail = [c for c in channels if c in raw.ch_names]
    if not avail:
        return None, None
    s_start = int(t_start * raw.info["sfreq"])
    s_end = int(t_end * raw.info["sfreq"])
    if s_end > raw.n_times or s_start < 0:
        return None, None
    data, _ = raw[avail, s_start:s_end]
    return data.mean(axis=0), raw.info["sfreq"]


def band_peak_power(f, P, lo, hi):
    """Peak power within frequency band [lo, hi] Hz of spectrum (f, P)."""
    band_mask = (f >= lo) & (f <= hi)
    if not np.any(band_mask):
        return np.nan
    P_band = P[band_mask]
    if np.all(np.isnan(P_band)) or np.nanmax(P_band) <= 0:
        return np.nan
    return float(np.nanmax(P_band))


def fooof_periodic_amp(f, P, lo, hi, fit_lo=2.0, fit_hi=40.0):
    """Run FOOOF/specparam on (f, P), return peak power in band [lo, hi] of periodic-only."""
    if FOOOF is None:
        return np.nan
    fit_mask = (f >= fit_lo) & (f <= fit_hi) & (P > 0)
    if fit_mask.sum() < 20:
        return np.nan
    try:
        if HAS_SPECPARAM:
            fm = FOOOF(peak_width_limits=(0.5, 6.0), max_n_peaks=8,
                       min_peak_height=0.01, peak_threshold=2.0)
        else:
            fm = FOOOF(peak_width_limits=(0.5, 6.0), max_n_peaks=8,
                       min_peak_height=0.01, peak_threshold=2.0, verbose=False)
        fm.fit(f[fit_mask], P[fit_mask], freq_range=(fit_lo, fit_hi))
        # Periodic-only spectrum: total - aperiodic on log scale
        # Use fooofed_spectrum_ minus _ap_fit
        try:
            log_periodic = fm.fooofed_spectrum_ - fm._ap_fit
        except AttributeError:
            log_periodic = fm.modeled_spectrum_ - fm.aperiodic_fit_
        f_fit = fm.freqs
        # Linear power scale for peak picking; clip negative values
        periodic = np.clip(10 ** log_periodic - 1, 0, None)
        return band_peak_power(f_fit, periodic, lo, hi)
    except Exception as exc:
        return np.nan


def irasa_periodic_amp(sig, fs, lo, hi):
    """Run IRASA on signal sig at fs; return peak power in band [lo, hi] of P_osc."""
    try:
        f, P, P_frac, P_osc = irasa_psd(sig, fs=fs, fmax=40.0, nperseg=int(2 * fs))
        return band_peak_power(f, P_osc, lo, hi)
    except Exception as exc:
        return np.nan


def list_cohort_subjects(cohort_name):
    cfg = COHORTS[cohort_name]
    csvs = sorted(cfg["events_dir"].glob("sub-*_sie_events.csv"))
    out = []
    for path in csvs:
        sid = path.stem.replace("_sie_events", "")
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if cfg["events_filter"]:
            col, val = cfg["events_filter"]
            if col not in df.columns:
                continue
            df = df[df[col] == val]
        if len(df) >= 3:
            out.append((sid, df))
    return out, cfg


def process_cohort(cohort_name):
    print(f"\n=== Processing cohort: {cohort_name} ===")
    subjects, cfg = list_cohort_subjects(cohort_name)
    print(f"  {len(subjects)} subjects with ≥3 events")
    if cohort_name == "lemon_EC":
        channels = POSTERIOR_CHANNELS_LEMON
    else:
        channels = POSTERIOR_CHANNELS_HBN

    per_event_rows = []
    per_subject_rows = []

    for i, (sid, events_df) in enumerate(subjects):
        if i % 20 == 0:
            print(f"  [{i+1:>4}/{len(subjects)}] {sid}")
        raw = load_raw(sid, cfg)
        if raw is None:
            continue

        sub_events = []
        for _, ev in events_df.iterrows():
            t0 = ev.get("t0_net", np.nan)
            if not np.isfinite(t0):
                continue
            sig, fs = posterior_signal(raw, t0 - HALF_WIN, t0 + HALF_WIN, channels)
            if sig is None:
                continue

            # Welch PSD (raw)
            nperseg = int(2 * fs)
            f, P = scipy_signal.welch(sig, fs=fs, window="hann", nperseg=nperseg,
                                       noverlap=nperseg // 2, scaling="density")
            sr1_raw = band_peak_power(f, P, *SR1_BAND)
            beta_raw = band_peak_power(f, P, *BETA_BAND)

            # FOOOF-corrected
            sr1_fooof = fooof_periodic_amp(f, P, *SR1_BAND)
            beta_fooof = fooof_periodic_amp(f, P, *BETA_BAND)

            # IRASA-corrected
            sr1_irasa = irasa_periodic_amp(sig, fs, *SR1_BAND)
            beta_irasa = irasa_periodic_amp(sig, fs, *BETA_BAND)

            # Template-correlation weight (proxy via sr1_z_max if not in events)
            w = float(ev.get("rho_template", ev.get("sr1_z_max", 0.0)) or 0.0)
            w = max(w, 0.0)

            ev_row = {
                "subject_id": sid,
                "t0_net": t0,
                "weight": w,
                "sr1_raw": sr1_raw, "beta_raw": beta_raw,
                "sr1_fooof": sr1_fooof, "beta_fooof": beta_fooof,
                "sr1_irasa": sr1_irasa, "beta_irasa": beta_irasa,
            }
            per_event_rows.append(ev_row)
            sub_events.append(ev_row)

        if not sub_events:
            continue
        # Per-subject weighted mean amplitudes
        sub_df = pd.DataFrame(sub_events)
        wts = sub_df["weight"].to_numpy()
        if wts.sum() == 0:
            wts = np.ones(len(sub_df))
        wts = wts / wts.sum()

        sub_summary = {"subject_id": sid, "n_events": len(sub_df)}
        for col in ["sr1_raw", "beta_raw", "sr1_fooof", "beta_fooof", "sr1_irasa", "beta_irasa"]:
            vals = sub_df[col].to_numpy()
            mask = np.isfinite(vals)
            if mask.sum() == 0:
                sub_summary[col] = np.nan
                continue
            sub_summary[col] = float(np.average(np.log(vals[mask] + 1e-30),
                                                 weights=wts[mask]))
        per_subject_rows.append(sub_summary)

    pd.DataFrame(per_event_rows).to_csv(
        OUT_DIR / f"per_event_amplitudes_{cohort_name}.csv", index=False)
    sub_df = pd.DataFrame(per_subject_rows)
    sub_df.to_csv(OUT_DIR / f"per_subject_amplitudes_{cohort_name}.csv", index=False)

    if len(sub_df) < 5:
        print(f"  [{cohort_name}] insufficient subjects ({len(sub_df)})")
        return None

    rhos = {}
    for method in ["raw", "fooof", "irasa"]:
        x = sub_df[f"sr1_{method}"].to_numpy()
        y = sub_df[f"beta_{method}"].to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() >= 5:
            rho, p = spearmanr(x[mask], y[mask])
            rhos[method] = (float(rho), float(p), int(mask.sum()))
        else:
            rhos[method] = (np.nan, np.nan, int(mask.sum()))
    return cohort_name, rhos, len(sub_df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=list(COHORTS.keys()) + ["all"], default="all")
    args = ap.parse_args()

    cohorts = list(COHORTS.keys()) if args.cohort == "all" else [args.cohort]

    results = []
    for c in cohorts:
        try:
            r = process_cohort(c)
            if r is not None:
                results.append(r)
        except Exception as exc:
            print(f"[{c}] failed: {exc}")
            import traceback
            traceback.print_exc()

    summary_rows = []
    for cohort_name, rhos, n in results:
        for method, (rho, p, n_used) in rhos.items():
            summary_rows.append({
                "cohort": cohort_name, "method": method,
                "rho": rho, "p_value": p, "n_subjects": n_used,
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "cross_cohort_summary.csv", index=False)

    # Pretty-print summary
    txt = "Aperiodic-Corrected SR×β Cross-Subject Coupling\n"
    txt += "=" * 50 + "\n\n"
    for cohort_name, rhos, n in results:
        txt += f"Cohort: {cohort_name} (N = {n})\n"
        for method in ["raw", "fooof", "irasa"]:
            rho, p, n_used = rhos[method]
            txt += f"  {method:>6} ρ = {rho:+.3f} (p = {p:.2e}, n = {n_used})\n"
        txt += "\n"
    txt += "Interpretation:\n"
    txt += "- If pediatric ρ remains substantially elevated over adult under FOOOF\n"
    txt += "  and IRASA aperiodic correction, the developmental gradient reflects\n"
    txt += "  genuine periodic-band coupling, not 1/f slope.\n"
    txt += "- If pediatric ρ collapses toward adult under aperiodic correction,\n"
    txt += "  the gradient is driven by aperiodic curvature.\n"
    print("\n" + txt)
    with open(OUT_DIR / "cross_cohort_summary.txt", "w") as f:
        f.write(txt)
    print(f"Results: {OUT_DIR}")


if __name__ == "__main__":
    main()
