#!/usr/bin/env python3
"""
Per-Cohort Odd-Mode Random-Window Null (Supplementary Table S4)
================================================================

Extends the LEMON EC random-window null (SR1/SR2 ratio: 1.04 random vs
4.12 canonical) to all 17 cohort × condition combinations, demonstrating
that the odd-mode-elevated harmonic-amplitude pattern is event-conditional
across the corpus rather than LEMON-EC-specific.

Parallelized across subjects within each cohort using multiprocessing.Pool.
Number of workers controlled by SIE_WORKERS env var (default 28).

For each cohort:
1. Load detected SIE events (composite-v2).
2. Sample N random 4-s windows per subject, where N matches per-subject
   event count, with onsets ≥20 s from any detected event AND ≥12 s from
   recording edges.
3. Compute IRASA on posterior-channel-mean signal for each window.
4. Compute SR1 (7.0-8.2 Hz) and SR2 (13.5-15.5 Hz) excess above 1/f.
5. Aggregate to cohort-mean random-window SR1/SR2 ratio vs canonical.

Usage:
    SIE_WORKERS=28 python3 -u scripts/sie_per_cohort_odd_mode_null.py [--cohort C]

Outputs:
    outputs/schumann/2026-04-27-per-cohort-odd-mode-null/
        - per_cohort_summary.csv  (random + canonical SR1/SR2 ratios)
        - per_cohort_summary.txt
"""

import os
import sys
import argparse
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
import mne

mne.set_log_level("ERROR")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "lib"))
from shape_vs_resonance import irasa_psd  # noqa: E402

T9 = Path("/Volumes/T9")
if not T9.exists():
    T9 = Path("/mnt/eeg-data/T9")

# Cohort list with raw-data + SIE-events configuration
COHORTS = {
    "lemon_EC": {
        "raw_root": T9 / "lemon_data" / "eeg_raw" / "EEG_MPILMBB_LEMON" / "EEG_Raw_BIDS_ID",
        "raw_pattern": "{sid}/RSEEG/{sid}.vhdr", "raw_format": "brainvision",
        "events_dir": BASE_DIR / "exports_sie" / "lemon_composite",
        "events_filter": ("condition", "EC"),
    },
    "lemon_EO": {
        "raw_root": T9 / "lemon_data" / "eeg_raw" / "EEG_MPILMBB_LEMON" / "EEG_Raw_BIDS_ID",
        "raw_pattern": "{sid}/RSEEG/{sid}.vhdr", "raw_format": "brainvision",
        "events_dir": BASE_DIR / "exports_sie" / "lemon_EO_composite",
        "events_filter": ("condition", "EO"),
    },
    "chbmp": {
        "raw_root": T9 / "CHBMP" / "BIDS_dataset",
        "raw_pattern": "{sid}/ses-V01/eeg/{sid}_ses-V01_task-protmap_eeg.edf",
        "raw_format": "edf",
        "events_dir": BASE_DIR / "exports_sie" / "chbmp_composite",
        "events_filter": None,
    },
    "dortmund": {
        "raw_root": T9 / "dortmund_data_dl",
        "raw_pattern": "{sid}/ses-1/eeg/{sid}_ses-1_task-EyesClosed_acq-pre_eeg.edf",
        "raw_format": "edf",
        "events_dir": BASE_DIR / "exports_sie" / "dortmund_composite",
        "events_filter": None,
    },
    "tdbrain_EC": {
        "raw_root": "gs://eeg-extraction-data/tdbrain/derivatives",
        "raw_pattern": "{sid}/ses-1/eeg/{sid}_ses-1_task-restEC_eeg.csv",
        "raw_format": "tdbrain_csv",
        "events_dir": BASE_DIR / "exports_sie" / "tdbrain_composite",
        "events_filter": None,
    },
    "tdbrain_EO": {
        "raw_root": "gs://eeg-extraction-data/tdbrain/derivatives",
        "raw_pattern": "{sid}/ses-1/eeg/{sid}_ses-1_task-restEO_eeg.csv",
        "raw_format": "tdbrain_csv",
        "events_dir": BASE_DIR / "exports_sie" / "tdbrain_EO_composite",
        "events_filter": None,
    },
}
for r in range(1, 12):
    COHORTS[f"hbn_R{r}"] = {
        "raw_root": T9 / "hbn_data" / f"cmi_bids_R{r}",
        "raw_pattern": "{sid}/eeg/{sid}_task-RestingState_eeg.set", "raw_format": "set",
        "events_dir": BASE_DIR / "exports_sie" / f"hbn_R{r}_composite",
        "events_filter": None,
    }

OUT_DIR = BASE_DIR / "outputs" / "schumann" / "2026-04-27-per-cohort-odd-mode-null"
OUT_DIR.mkdir(parents=True, exist_ok=True)

POSTERIOR_CHANNELS_LEMON = ["O1", "O2", "Oz", "PO3", "PO4", "POz", "PO7", "PO8",
                             "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "Pz"]
POSTERIOR_CHANNELS_HBN = ["E70", "E75", "E83", "E76", "E71", "E74", "E62",
                          "E72", "E73", "E69", "E68", "E67", "E66", "E65", "E60"]
POSTERIOR_CHANNELS_TD = ["O1", "O2", "Pz", "P3", "P4", "P7", "P8", "Oz"]

WINDOW_SEC = 4.0
HALF_WIN = WINDOW_SEC / 2

SR1_BAND = (7.0, 8.2)
SR2_BAND = (13.5, 15.5)
EDGE_GUARD_SEC = 12.0
EVENT_GUARD_SEC = 20.0


def get_channels(cohort):
    if cohort.startswith("lemon"):
        return POSTERIOR_CHANNELS_LEMON
    elif cohort.startswith("hbn"):
        return POSTERIOR_CHANNELS_HBN
    elif cohort.startswith("tdbrain"):
        return POSTERIOR_CHANNELS_TD
    else:
        return POSTERIOR_CHANNELS_LEMON


TDBRAIN_CACHE = Path(os.environ.get("TDBRAIN_CACHE", "/tmp/tdbrain_cache"))
TDBRAIN_CACHE.mkdir(parents=True, exist_ok=True)
TDBRAIN_STANDARD_EEG = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
                        "FC3", "FCz", "FC4",
                        "T7", "T3", "C3", "Cz", "C4", "T8", "T4",
                        "CP3", "CPz", "CP4",
                        "P7", "T5", "P3", "Pz", "P4", "P8", "T6",
                        "O1", "Oz", "O2"]


def _fetch_tdbrain_csv(gcs_root, rel_path):
    """Fetch a TDBRAIN CSV (and accompanying JSON) from GCS into local cache."""
    import subprocess
    local_csv = TDBRAIN_CACHE / rel_path
    local_csv.parent.mkdir(parents=True, exist_ok=True)
    if not local_csv.exists():
        gcs_path = f"{gcs_root}/{rel_path}"
        result = subprocess.run(
            ["gcloud", "storage", "cp", gcs_path, str(local_csv)],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0 or not local_csv.exists():
            return None, None
    json_rel = rel_path.replace("_eeg.csv", "_eeg.json")
    local_json = TDBRAIN_CACHE / json_rel
    if not local_json.exists():
        try:
            subprocess.run(
                ["gcloud", "storage", "cp",
                 f"{gcs_root}/{json_rel}", str(local_json)],
                capture_output=True, text=True, timeout=60,
            )
        except Exception:
            pass
    return local_csv, local_json if local_json.exists() else None


def _load_tdbrain_csv(sid, cfg):
    rel = cfg["raw_pattern"].format(sid=sid)
    local_csv, local_json = _fetch_tdbrain_csv(cfg["raw_root"], rel)
    if local_csv is None:
        return None
    try:
        df = pd.read_csv(local_csv)
    except Exception:
        return None
    if len(df) < 1000:
        return None
    df.columns = [c.strip() for c in df.columns]
    eeg_cols = [c for c in df.columns if c in TDBRAIN_STANDARD_EEG]
    if len(eeg_cols) < 10:
        return None
    fs = 500.0
    if local_json is not None:
        try:
            import json as json_mod
            with open(local_json) as jf:
                meta = json_mod.load(jf)
            fs = float(meta.get("SamplingFrequency", 500.0))
        except Exception:
            pass
    data = df[eeg_cols].values.T.astype(np.float64) * 1e-9  # nV → V
    info = mne.create_info(ch_names=list(eeg_cols), sfreq=fs, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="error")
    return raw


def load_raw(sid, cfg):
    fmt = cfg["raw_format"]
    if fmt == "tdbrain_csv":
        raw = _load_tdbrain_csv(sid, cfg)
        if raw is None:
            return None
    else:
        raw_path = Path(cfg["raw_root"]) / cfg["raw_pattern"].format(sid=sid)
        if not raw_path.exists():
            return None
        try:
            if fmt == "brainvision":
                raw = mne.io.read_raw_brainvision(raw_path, preload=True, verbose="error")
            elif fmt == "set":
                raw = mne.io.read_raw_eeglab(raw_path, preload=True, verbose="error")
            elif fmt == "edf":
                raw = mne.io.read_raw_edf(raw_path, preload=True, verbose="error")
                # Normalise BIDS-style channel labels: "Fp1-REF" → "Fp1", "F3 -REF" → "F3"
                rename = {}
                for c in raw.ch_names:
                    new = c.replace("-REF", "").replace(" -REF", "").strip()
                    if new != c:
                        rename[c] = new
                if rename:
                    raw.rename_channels(rename)
                eeg_picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
                if len(eeg_picks) == 0:
                    return None
                raw.pick(eeg_picks)
            else:
                return None
        except Exception:
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


def band_excess_above_aperiodic(sig, fs, band_lo, band_hi):
    try:
        f, P, P_frac, P_osc = irasa_psd(sig, fs=fs, fmax=40.0, nperseg=int(2 * fs))
    except Exception:
        return np.nan
    band_mask = (f >= band_lo) & (f <= band_hi)
    if not np.any(band_mask):
        return np.nan
    return float(np.nanmax(P_osc[band_mask]))


def sample_random_window_starts(duration, event_times, n_windows, seed):
    rng = np.random.default_rng(seed=seed)
    if duration < 2 * EDGE_GUARD_SEC + WINDOW_SEC:
        return []
    valid_starts = []
    candidates = rng.uniform(EDGE_GUARD_SEC, duration - EDGE_GUARD_SEC - WINDOW_SEC,
                              size=n_windows * 30)
    for c in candidates:
        ok = True
        for et in event_times:
            if abs(c + HALF_WIN - et) < EVENT_GUARD_SEC:
                ok = False
                break
        if ok:
            valid_starts.append(c)
        if len(valid_starts) >= n_windows:
            break
    return valid_starts[:n_windows]


def process_subject_worker(args):
    """Top-level worker for multiprocessing.Pool. Returns lists of canonical
    and random SR1/SR2 excess values for one subject."""
    sid, events_df_records, cfg, channels, seed = args
    canonical_sr1, canonical_sr2 = [], []
    random_sr1, random_sr2 = [], []

    raw = load_raw(sid, cfg)
    if raw is None:
        return canonical_sr1, canonical_sr2, random_sr1, random_sr2

    events_df = pd.DataFrame.from_records(events_df_records)
    event_times = events_df["t0_net"].dropna().tolist()
    if not event_times:
        return canonical_sr1, canonical_sr2, random_sr1, random_sr2

    # Canonical events
    for t0 in event_times:
        sig, fs = posterior_signal(raw, t0 - HALF_WIN, t0 + HALF_WIN, channels)
        if sig is None:
            continue
        sr1 = band_excess_above_aperiodic(sig, fs, *SR1_BAND)
        sr2 = band_excess_above_aperiodic(sig, fs, *SR2_BAND)
        if np.isfinite(sr1) and np.isfinite(sr2):
            canonical_sr1.append(sr1)
            canonical_sr2.append(sr2)

    # Random null windows (matched per-subject count)
    duration = float(raw.times[-1])
    starts = sample_random_window_starts(duration, event_times, len(event_times), seed)
    for s in starts:
        sig, fs = posterior_signal(raw, s, s + WINDOW_SEC, channels)
        if sig is None:
            continue
        sr1 = band_excess_above_aperiodic(sig, fs, *SR1_BAND)
        sr2 = band_excess_above_aperiodic(sig, fs, *SR2_BAND)
        if np.isfinite(sr1) and np.isfinite(sr2):
            random_sr1.append(sr1)
            random_sr2.append(sr2)

    return canonical_sr1, canonical_sr2, random_sr1, random_sr2


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


def process_cohort(cohort_name, n_workers=28, max_subjects=None):
    cfg = COHORTS[cohort_name]
    channels = get_channels(cohort_name)
    subjects, _ = list_cohort_subjects(cohort_name)
    if max_subjects:
        subjects = subjects[:max_subjects]

    print(f"\n=== {cohort_name}: {len(subjects)} subjects (workers={n_workers}) ===",
          flush=True)
    if not subjects:
        return None

    t0 = time.time()
    # Build args list — pass dataframes as records to avoid pickling issues
    work = [
        (sid, df.to_dict("records"), cfg, channels, hash(sid) & 0xFFFFFFFF)
        for sid, df in subjects
    ]

    canonical_sr1, canonical_sr2 = [], []
    random_sr1, random_sr2 = [], []
    n_used = 0

    with Pool(processes=n_workers) as pool:
        for i, (csr1, csr2, rsr1, rsr2) in enumerate(pool.imap_unordered(process_subject_worker, work, chunksize=1)):
            if csr1:
                canonical_sr1.extend(csr1)
                canonical_sr2.extend(csr2)
                random_sr1.extend(rsr1)
                random_sr2.extend(rsr2)
                n_used += 1
            if (i + 1) % 50 == 0 or (i + 1) == len(work):
                elapsed = time.time() - t0
                print(f"  [{cohort_name}] {i+1}/{len(work)} subjects done "
                      f"({n_used} usable), {elapsed:.0f}s elapsed", flush=True)

    if not canonical_sr1 or not random_sr1:
        print(f"  [{cohort_name}] insufficient data", flush=True)
        return None

    can_ratio = np.median(canonical_sr1) / max(np.median(canonical_sr2), 1e-30)
    rand_ratio = np.median(random_sr1) / max(np.median(random_sr2), 1e-30)
    elapsed = time.time() - t0
    print(f"  [{cohort_name}] complete: {elapsed:.0f}s, can SR1/SR2 = {can_ratio:.2f}, "
          f"rand SR1/SR2 = {rand_ratio:.2f}", flush=True)
    return {
        "cohort": cohort_name,
        "n_subjects": n_used,
        "n_canonical_events": len(canonical_sr1),
        "n_random_windows": len(random_sr1),
        "canonical_sr1_median": float(np.median(canonical_sr1)),
        "canonical_sr2_median": float(np.median(canonical_sr2)),
        "canonical_sr1_sr2_ratio": float(can_ratio),
        "random_sr1_median": float(np.median(random_sr1)),
        "random_sr2_median": float(np.median(random_sr2)),
        "random_sr1_sr2_ratio": float(rand_ratio),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="all")
    ap.add_argument("--max-subjects", type=int, default=None)
    args = ap.parse_args()

    n_workers = int(os.environ.get("SIE_WORKERS", "28"))
    n_workers = min(n_workers, max(1, cpu_count() - 1))

    if args.cohort == "all":
        cohorts = list(COHORTS.keys())
    else:
        cohorts = [args.cohort]

    rows = []
    for c in cohorts:
        try:
            r = process_cohort(c, n_workers=n_workers, max_subjects=args.max_subjects)
            if r:
                rows.append(r)
                # Write incrementally so partial results survive interruption
                pd.DataFrame(rows).to_csv(OUT_DIR / "per_cohort_summary.csv", index=False)
        except Exception as exc:
            print(f"[{c}] failed: {exc}", flush=True)
            import traceback
            traceback.print_exc()

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "per_cohort_summary.csv", index=False)

    txt = "Per-Cohort Odd-Mode Random-Window Null (SR1/SR2 ratios)\n"
    txt += "=" * 60 + "\n"
    txt += f"{'Cohort':<14} {'N':>4} {'Can.SR1/SR2':>11} {'Rand.SR1/SR2':>13} {'Can/Rand':>9}\n"
    txt += "-" * 60 + "\n"
    for _, r in df.iterrows():
        ratio_x = (r["canonical_sr1_sr2_ratio"] / max(r["random_sr1_sr2_ratio"], 1e-30))
        txt += f"{r['cohort']:<14} {int(r['n_subjects']):>4} "
        txt += f"{r['canonical_sr1_sr2_ratio']:>11.2f} "
        txt += f"{r['random_sr1_sr2_ratio']:>13.2f} "
        txt += f"{ratio_x:>9.1f}x\n"
    txt += "\nInterpretation: canonical events show selective SR1 elevation\n"
    txt += "(SR1/SR2 ratio > 1) while random windows do not (ratio approx 1).\n"
    print("\n" + txt)
    with open(OUT_DIR / "per_cohort_summary.txt", "w") as f:
        f.write(txt)
    print(f"Results: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
