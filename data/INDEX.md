# External Datasets Index

All datasets are stored on `/Volumes/T9`. Replication datasets (Papers 2 & 3) are at the drive root. Paper 1 datasets are in `/Volumes/T9/Code/data/`.

---

# Paper 1 Datasets (SIE Discovery)

All recorded with consumer-grade EEG devices. Stored as CSV in `/Volumes/T9/Code/data/`.

## PhySF — Physiological Sense of Flow

| Property | Value |
|----------|-------|
| **Path** | `/Volumes/T9/Code/data/PhySF/` |
| **Size** | 2.3 GB |
| **Subjects** | 25 |
| **Files** | 46 CSV (flow + no-flow conditions per subject) |
| **Channels** | 14 EEG (Emotiv EPOC X) |
| **Sampling Rate** | 128 Hz (2048 Hz internal) |
| **Hardware Bandwidth** | 0.16–43 Hz (built-in 5th-order Sinc filter) |
| **Notch Filter** | Digital 50 Hz and 60 Hz (hardware) |
| **Effective Nyquist** | 43 Hz (hardware-limited, not 64 Hz) |
| **FOOOF Range** | [1, 50] Hz |
| **Additional Sensors** | Empatica E4 wristband (5 channels), RespiBan professional (4 channels) |

**EEG Channels:** AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4

**Notes:**
- Multi-device simultaneous recording (EEG + wristband + respiration)
- README with full device and channel documentation included
- 23 total columns per CSV (14 EEG + 5 Empatica + 4 RespiBan)
- EPOC X hardware rolls off above 43 Hz; FOOOF range [1, 50] extends slightly past true bandwidth

---

## VEP — Visual Evoked Potential

| Property | Value |
|----------|-------|
| **Path** | `/Volumes/T9/Code/data/vep/` |
| **Size** | 138 MB |
| **Subjects** | 32 |
| **Files** | 64 CSV (2 sessions per subject) |
| **Channels** | 14 EEG (Emotiv EPOC X) |
| **Sampling Rate** | 128 Hz (2048 Hz internal) |
| **Hardware Bandwidth** | 0.16–43 Hz (built-in 5th-order Sinc filter) |
| **Notch Filter** | Digital 50 Hz and 60 Hz (hardware) |
| **FOOOF Range** | [1, 50] Hz |

**Notes:**
- 96 columns per CSV (14 raw EEG + counter/timestamp + 80 power spectral features)
- 4 session labels: A, C, F, P

---

## MPENG — Motor/Physiological Engagement

| Property | Value |
|----------|-------|
| **Path** | `/Volumes/T9/Code/data/mpeng/` |
| **Size** | 2.4 GB |
| **Files** | 900 CSV |
| **Channels** | 14 EEG (Emotiv EPOC X) |
| **Sampling Rate** | 128 Hz (2048 Hz internal) |
| **Hardware Bandwidth** | 0.16–43 Hz (built-in 5th-order Sinc filter) |
| **Notch Filter** | Digital 50 Hz and 60 Hz (hardware) |
| **FOOOF Range** | [1, 50] Hz |

**Notes:**
- 183 columns per CSV (EEG + quality metrics + performance metrics + facial expressions + PSD)
- Filename encodes ID, trial, and 4-digit condition codes

---

## Emotions — EEG Emotions Dataset

| Property | Value |
|----------|-------|
| **Path** | `/Volumes/T9/Code/data/emotions/` |
| **Size** | 7.0 GB |
| **Subjects** | 88 |
| **Sessions** | 2,343 |
| **Files** | 4,686 (CSV + TXT pairs per session) |
| **Channels** | 14 EEG (Emotiv EPOC X) |
| **Sampling Rate** | 128 Hz (2048 Hz internal) |
| **Hardware Bandwidth** | 0.16–43 Hz (built-in 5th-order Sinc filter) |
| **Notch Filter** | Digital 50 Hz and 60 Hz (hardware) |
| **FOOOF Range** | [1, 50] Hz (analysis capped at 48 Hz to avoid notch artifacts) |

**Notes:**
- TXT files contain raw signal values; CSV files contain processed EEG channel data
- Used as primary replication dataset in Paper 1 (Figure 2)
- Filename pattern: `Subject_Trial.csv` (e.g., `10_1.0.csv`)

---

## Brain Invaders — BCI P300 Paradigm

| Property | Value |
|----------|-------|
| **Path** | `/Volumes/T9/Code/data/brain_invaders/` |
| **Size** | 2.8 GB |
| **Subjects** | 64 |
| **Files** | 64 CSV |
| **Channels** | 16 EEG (active dry electrodes) |
| **Sampling Rate** | 512 Hz |
| **Nyquist** | 256 Hz |
| **Notch Filter** | Not documented; analysis removes peaks within ±5 Hz of [50, 100, 150, 200] Hz |
| **FOOOF Range** | [1, 250] Hz |

**EEG Channels:** Fp1, Fp2, F5, AFz, F6, T7, Cz, T8, P7, P3, Pz, P4, P8, O1, Oz, O2

**Notes:**
- BCI game paradigm with Event and Target columns
- Channel layout documented in `Header.csv`
- Widest usable frequency range of any Paper 1 dataset

---

## Arithmetic — Mental Arithmetic EEG

| Property | Value |
|----------|-------|
| **Path** | `/Volumes/T9/Code/data/arithmetic/` |
| **Size** | 7.6 GB |
| **Files** | 159 EDF (raw) + 159 CSV (converted) |
| **Channels** | 19 EEG (10-20 system) |
| **Sampling Rate** | 128 Hz |
| **Hardware Filter** | 30 Hz high-pass cutoff, 50 Hz notch (Neurocom EEG 23-channel, Ukraine) |
| **Notch Filter** | 50 Hz (hardware, European power line) |
| **FOOOF Range** | [1, 50] Hz |

**EEG Channels:** Fp1, Fp2, F7, F3, Fz, F4, F8, T7, C3, Cz, C4, T8, P7, P3, Pz, P4, P8, O1, O2

**Notes:**
- Two experiments: Experiment 1 (63 files), Experiment 2 (96 files)
- 4 conditions: A (arithmetic), M (meditation), B (baseline), R (rest)
- Both raw EDF and converted CSV available
- ICA artifact removal applied during preprocessing
- Source: PhysioNet (Zyma et al., 2019)

---

## Muse — Portable EEG Recordings

| Property | Value |
|----------|-------|
| **Path** | `/Volumes/T9/Code/data/muse/` |
| **Size** | 4.0 GB |
| **Devices** | 2 (Muse-357D, Muse-461E) |
| **Files** | 17 CSV |
| **Channels** | 4 EEG |
| **Sampling Rate** | 256 Hz |
| **Notch Filter** | Configurable 50/60 Hz (default 60 Hz US/Canada, switchable in app) |
| **FOOOF Range** | [1, 50] Hz |

**Notes:**
- 106 columns (raw EEG, filtered EEG, accelerometer, gyroscope, power bands, facial movements, heart rate)
- Consumer-grade portable EEG headband
- Notch filter applied via Mind Monitor app, not hardware

---

## Insight — Emotiv Insight Recordings

| Property | Value |
|----------|-------|
| **Path** | `/Volumes/T9/Code/data/insight/` |
| **Size** | 197 MB |
| **Subjects** | 1 (device A3D2076D) |
| **Files** | 12 CSV |
| **Channels** | 14 EEG (Emotiv INSIGHT2) |
| **Sampling Rate** | 128 Hz (2048 Hz internal) |
| **Hardware Bandwidth** | 0.5–45 Hz (built-in 5th-order Sinc filter) |
| **Notch Filter** | Digital 50 Hz and 60 Hz (hardware) |
| **FOOOF Range** | [1, 50] Hz |

**Notes:**
- 96 columns (EEG + motion + performance metrics + power bands)
- Reduced-channel Emotiv device; metadata embedded in CSV header
- Slightly wider bandwidth than EPOC X (45 Hz vs 43 Hz)

---

## EPOC — Emotiv EPOC X Recordings

| Property | Value |
|----------|-------|
| **Path** | `/Volumes/T9/Code/data/epoc/` |
| **Size** | 374 MB |
| **Devices** | 4 EPOC X headsets |
| **Files** | 8 CSV |
| **Channels** | 14 EEG (Emotiv EPOC X) |
| **Sampling Rate** | 128 Hz (2048 Hz internal) |
| **Hardware Bandwidth** | 0.16–43 Hz (built-in 5th-order Sinc filter) |
| **Notch Filter** | Digital 50 Hz and 60 Hz (hardware) |
| **FOOOF Range** | [1, 50] Hz |

**Notes:**
- 151 columns (EEG + motion + markers + quality + power + performance)
- Personal/lab recordings from multiple EPOC X devices

---

## ArEEG — Arabic Voice Command EEG

| Property | Value |
|----------|-------|
| **Path** | `/Volumes/T9/Code/data/ArEEG/` |
| **Size** | 228 MB |
| **Devices** | 18 EPOC X headsets |
| **Files** | 359 CSV |
| **Channels** | 14 EEG (Emotiv EPOC X) |
| **Sampling Rate** | 128 Hz (2048 Hz internal) |
| **Hardware Bandwidth** | 0.16–43 Hz (built-in 5th-order Sinc filter) |
| **Notch Filter** | Digital 50 Hz and 60 Hz (hardware) |
| **FOOOF Range** | [1, 50] Hz |

**Notes:**
- Arabic voice command paradigm (yes/no, left/right, medicine, bathroom, etc.)
- 67 columns (EEG + motion + quality metrics)
- Data collected across 18 different headsets

---

## Bonn — Bonn Epilepsy Dataset

| Property | Value |
|----------|-------|
| **Path** | `/Volumes/T9/bonn_data/` |
| **Size** | 64 MB |
| **Segments** | 500 (5 conditions x 100 each) |
| **Channels** | 1 per file (single-channel recordings) |
| **Samples** | 4,097 per file (23.6 seconds at 173.61 Hz) |
| **Format** | Plain TXT |
| **Hardware Filter** | Bandpass 0.53–40 Hz (12-bit A/D) |
| **Notch Filter** | None needed — 40 Hz low-pass eliminates power line frequencies |
| **FOOOF Range** | [1, 45] Hz (limited by 40 Hz hardware cutoff) |

**Conditions:**
| Set | Description | Recording Type |
|-----|-------------|----------------|
| Z | Eyes open, healthy volunteers | Scalp |
| O | Eyes closed, healthy volunteers | Scalp |
| N | Hippocampal formation, seizure-free | Intracranial |
| F | Epileptogenic zone, seizure-free | Intracranial |
| S | Seizure activity | Intracranial |

**Notes:**
- Classic benchmark dataset for seizure detection
- Reference: Andrzejak et al., Physical Review E, 2001 (PDF included)
- Analysis outputs in `lattice_results/`: `bonn_dominant_peaks_all.csv`, `bonn_summary.csv`
- Artifact-free segments; no additional preprocessing required

---

# Paper 1 Summary

| Dataset | Size | Subjects | Ch | Hz | Bandwidth | Notch | FOOOF Range |
|---------|------|----------|-----|-----|-----------|-------|-------------|
| PhySF | 2.3 GB | 25 | 14 | 128 | 0.16–43 Hz | 50/60 Hz (hw) | [1, 50] |
| VEP | 138 MB | 32 | 14 | 128 | 0.16–43 Hz | 50/60 Hz (hw) | [1, 50] |
| MPENG | 2.4 GB | ~900 files | 14 | 128 | 0.16–43 Hz | 50/60 Hz (hw) | [1, 50] |
| Emotions | 7.0 GB | 88 | 14 | 128 | 0.16–43 Hz | 50/60 Hz (hw) | [1, 48] |
| Brain Invaders | 2.8 GB | 64 | 16 | 512 | ~256 Hz | Unknown | [1, 250] |
| Arithmetic | 7.6 GB | ~32 | 19 | 128 | 0.53–40 Hz | 50 Hz (hw) | [1, 50] |
| Muse | 4.0 GB | 2 devices | 4 | 256 | ~128 Hz | 50/60 Hz (app) | [1, 50] |
| Insight | 197 MB | 1 | 14 | 128 | 0.5–45 Hz | 50/60 Hz (hw) | [1, 50] |
| EPOC | 374 MB | 4 devices | 14 | 128 | 0.16–43 Hz | 50/60 Hz (hw) | [1, 50] |
| ArEEG | 228 MB | 18 devices | 14 | 128 | 0.16–43 Hz | 50/60 Hz (hw) | [1, 50] |
| Bonn | 64 MB | 500 seg | 1 | 173.6 | 0.53–40 Hz | None (LP@40) | [1, 45] |
| **Total** | **~27 GB** | | | | | | |

All EPOC X datasets have a **hardware-imposed 43 Hz ceiling** — the 128 Hz sampling rate (64 Hz Nyquist) is misleading; the device's Sinc filter rolls off above 43 Hz. Brain Invaders is the only Paper 1 dataset with meaningful high-frequency content.

---

# Replication Datasets (Papers 2 & 3)

Large public datasets stored at `/Volumes/T9/` root. All research-grade, high-density EEG.

---

## CHBMP — Cuban Human Brain Mapping Project

| Property | Value |
|----------|-------|
| **Path** | `/Volumes/T9/CHBMP/BIDS_dataset/` |
| **Size** | 27 GB |
| **Subjects** | 282 healthy adults (ages 18–68, mean 31.9 ± 9.3) |
| **Channels** | 62–120 EEG (10-5 system) + 2 EOG + 1 ECG (varies by subject) |
| **Sampling Rate** | 200 Hz |
| **Nyquist** | 100 Hz |
| **Notch Filter** | 60 Hz (hardware, documented in channels.tsv) |
| **Pipeline Notch** | 60 Hz applied in analysis script |
| **FOOOF Range** | [1, 45] Hz |
| **Format** | EDF (BIDS v1.2.1) |
| **Duration** | ~25 min per subject (15 min EC, 5 min EO, 3 min hyperventilation, 3 min recovery) |
| **Reference** | Common average |
| **Conditions** | Eyes closed, eyes open, hyperventilation, recovery |
| **Other Modalities** | T1w MRI (1.5T), DWI (2 runs), fieldmaps |
| **Demographics** | Age, gender, education, ethnicity, handedness (via LORIS portal) |
| **Cognition** | MMSE, WAIS III, reaction time tests |
| **DOI** | https://doi.org/10.5281/zenodo.3945385 |
| **License** | CC BY-NC-SA |
| **Institution** | Cuban Center for Neuroscience, Havana |
| **Collection Period** | 2004–2008 |

**Notes:**
- Channel count varies across subjects: 62ch (142 subjects), 120ch (60 subjects), 121ch (18 subjects), 58–59ch (25 subjects)
- Events marked in Spanish: "ojos cerrados" (eyes closed), "ojos abiertos" (eyes open)
- Stratified random sample from La Lisa municipality, Havana
- BIDS structure: `sub-CBM{XXXXX}/ses-V01/{anat,dwi,eeg,fmap}/`
- 3 subject IDs missing from sequence (00039, 00143, 00251)

**Citation:** Valdes-Sosa et al., Cuban Human Brain Mapping Project (EEG, MRI, and Cognition dataset)

---

## Dortmund — Dortmund Vital Study

| Property | Value |
|----------|-------|
| **Path** | `/Volumes/T9/dortmund_data/` |
| **Full EDFs** | `/Volumes/T9/dortmund_data_dl/` (75 GB) |
| **Size** | 3 GB (BIDS) / 75 GB (raw EDFs) |
| **Subjects** | 608 baseline (376F/232M), 208 longitudinal follow-up (~5 years) |
| **Age Range** | 20–70 years |
| **Channels** | 64 EEG (actiCAP 64, 10-20 system) |
| **Sampling Rate** | 1000 Hz (downsampled to 250 Hz in analysis) |
| **Nyquist** | 500 Hz (native), 125 Hz (downsampled) |
| **Power Line** | 50 Hz (Germany) |
| **Hardware Notch** | Not documented |
| **Pipeline Notch** | [50, 100] Hz applied in analysis script |
| **Online Low-pass** | 250 Hz |
| **FOOOF Range** | [1, 45] Hz |
| **Freq Resolution** | 0.25 Hz (nperseg=1000, 4s windows) |
| **Format** | EDF (BIDS v1.9.0) |
| **Duration** | ~184 seconds per condition |
| **Reference** | FCz (online), ground AFz |
| **Hardware** | BrainProducts BrainAmp DC |
| **Conditions** | 4 per session: EyesClosed pre, EyesClosed post, EyesOpen pre, EyesOpen post |
| **Sessions** | ses-1 (baseline, all 608), ses-2 (5-year follow-up, 208) |
| **Handedness** | ~93% right-handed |
| **DOI** | https://doi.org/10.18112/openneuro.ds005385.v1.0.3 |
| **License** | CC0 |
| **Institution** | Leibniz Research Centre for Working Environment (IfADo), Dortmund |
| **Clinical Trial** | NCT05155397 |

**Notes:**
- "Pre" and "post" refer to before/after a cognitive test battery within the same session
- Impedance kept below 10 kOhm
- 50 Hz notch hole sits in the critical gamma inverse noble region — problematic for P20 extended-range analysis despite excellent native sampling rate
- `dortmund_data/` has trimmed BIDS exports; `dortmund_data_dl/` has full raw EDFs
- `dortmund_test/` contains a single test subject (sub-001)
- `participants.tsv` columns: participant_id, sex, age, handedness, session1, late_ses1, session2, late_ses2

**Analysis outputs** (in `dortmund_data/`):
- `lattice_results/` — dominant peaks and summary stats for all 4 conditions
- `lattice_results_longitudinal/` — ses-2 peaks + longitudinal comparison
- `lattice_results_ot/` — overlap-trim analysis with degree-3 cross-base subfolder
- `lattice_results_replication/` and `lattice_results_replication_v2/` — unified protocol replication with enrichment, cross-base, sensitivity analyses

**Citation:** Gajewski et al., 2022 (doi: 10.2196/32352)

---

## EEGMMIDB — EEG Motor Movement/Imagery Database

| Property | Value |
|----------|-------|
| **Path** | `/Volumes/T9/eegmmidb/` |
| **Size** | 3.6 GB |
| **Subjects** | 110 (S001–S109, continuous numbering) |
| **Channels** | 64 EEG + 1 annotation (extended 10-20 / Sharbrough montage) |
| **Sampling Rate** | 160 Hz (native, no resampling) |
| **Nyquist** | 80 Hz |
| **Power Line** | 60 Hz (USA) |
| **Hardware Notch** | Not documented in EDF headers |
| **Pipeline Notch** | 60 Hz applied in analysis script |
| **FOOOF Range** | [1, 75] Hz |
| **Freq Resolution** | 0.125 Hz (nperseg=1280, 8s windows) |
| **Format** | EDF + `.edf.event` annotation files |
| **Runs per Subject** | 14 |
| **Files** | 1,540 EDF files total (14 × 110) |
| **Source** | PhysioNet |

**14-Run Protocol:**

| Run | Duration | Task |
|-----|----------|------|
| R01 | ~61s | Baseline: Eyes Open |
| R02 | ~61s | Baseline: Eyes Closed |
| R03, R07, R11 | ~120s | Motor Execution: Both Fists |
| R04, R08, R12 | ~120s | Motor Execution: Both Feet |
| R05, R09, R13 | ~120s | Motor Imagery: Both Fists |
| R06, R10, R14 | ~120s | Motor Imagery: Both Feet |

**Channel List (64 EEG):**
Fc5, Fc3, Fc1, Fcz, Fc2, Fc4, Fc6, C5, C3, C1, Cz, C2, C4, C6, Cp5, Cp3, Cp1, Cpz, Cp2, Cp4, Cp6, Fp1, Fpz, Fp2, Af7, Af3, Afz, Af4, Af8, F7, F5, F3, F1, Fz, F2, F4, F6, F8, Ft7, Ft8, T7, T8, T9, T10, Tp7, Tp8, P7, P5, P3, P1, Pz, P2, P4, P6, P8, Po7, Po3, Poz, Po4, Po8, O1, Oz, O2, Iz

**Notes:**
- Event annotations contain T0 (rest), T1 (left fist / both fists), T2 (right fist / both feet)
- Event durations ~4.1–4.2 seconds each
- Montage diagram available as `64_channel_sharbrough.pdf`
- Motor execution and imagery tasks each repeated 3 times (runs 3–14)

---

## HBN — Healthy Brain Network

| Property | Value |
|----------|-------|
| **Path** | `/Volumes/T9/hbn_data/` |
| **Size** | 686 GB |
| **Subjects** | 929 total across 5 releases |
| **Population** | Children and adolescents (ages 5.2–21.7) |
| **Channels** | 129 (128 EEG + 1 ref, EGI HydroCel GSN 128) |
| **Sampling Rate** | 500 Hz (downsampled to 250 Hz in analysis) |
| **Nyquist** | 250 Hz (native), 125 Hz (downsampled) |
| **Acquisition Bandpass** | 0.1–100 Hz (EGI online filter) |
| **Pipeline Notch** | [60, 120] Hz applied in analysis script |
| **FOOOF Range** | [1, 45] Hz (current); extendable to [1, 100+] Hz for P20 |
| **Freq Resolution** | 0.244 Hz (nperseg=1024, 4.1s windows) |
| **Format** | EEGLAB SET/FDT (BIDS) |
| **Reference** | Cz |
| **Power Line** | 60 Hz (USA) |
| **Hardware** | Magstim EGI 128-channel GSN 200 v2.1 |
| **Institution** | Child Mind Institute, New York |
| **DOI** | https://doi.org/10.1101/2024.10.03.615261 |

**Releases:**

| Release | Subjects | .set Files |
|---------|----------|------------|
| R1 | 136 | 1,500 |
| R2 | 150 | 2,740 |
| R3 | 184 | 3,624 |
| R4 | 324 | 6,684 |
| R6 | 135 | 2,454 |
| **Total** | **929** | **17,002** |

**Tasks (up to 13 per subject):**
- RestingState (eyes-open/eyes-closed alternating)
- DespicableMe, DiaryOfAWimpyKid, FunwithFractals, ThePresent (passive video viewing)
- contrastChangeDetection (3 runs)
- surroundSupp (2 runs)
- seqLearning6target, seqLearning8target
- symbolSearch

**Demographics (participants.tsv):**
- sex, age, ehq_total (handedness)
- Psychopathology: p_factor, attention, internalizing, externalizing (CBCL-derived)
- Per-task availability flags
- commercial_use, full_pheno flags

**Notes:**
- Channel naming: E1–E129
- HED (Hierarchical Event Descriptors) annotations included
- R5 not present in this download
- Largest dataset by far; rich phenotyping makes it suitable for brain-behavior analyses
- P20 candidate: 60 Hz notch hole requires analyzing [45–58] and [62–100] Hz separately, skipping notch zone

---

## LEMON — Leipzig Study for Mind-Body-Emotion Interactions

| Property | Value |
|----------|-------|
| **Path** | `/Volumes/T9/lemon_data/` |
| **Size** | 75 GB |
| **Subjects** | 220 (raw) / 205 (preprocessed) / 227 (behavioral) |
| **Channels** | 62 (59 EEG + VEOG + 2 reference-related) |
| **Sampling Rate** | 2500 Hz (raw), downsampled to 250 Hz (preprocessed) |
| **Nyquist** | 1250 Hz (raw), 125 Hz (preprocessed) |
| **Acquisition Bandpass** | 0.015–1000 Hz (raw recording) |
| **Hardware Notch** | **Off** (confirmed in vhdr headers) |
| **Pipeline Notch** | **None applied at any stage** |
| **Preprocessed Filter** | 1–45 Hz bandpass (8th-order Butterworth), ICA-cleaned |
| **FOOOF Range** | [1, 85] Hz |
| **Freq Resolution** | 0.244 Hz (nperseg=1024, 4.1s windows) |
| **Format** | BrainVision .vhdr/.eeg/.vmrk (raw), EEGLAB SET/FDT (preprocessed) |
| **Conditions** | Eyes Closed (EC), Eyes Open (EO) |
| **Hardware** | BrainVision Recorder Professional v1.20 |

**Directory Structure:**

| Subfolder | Contents | Subjects |
|-----------|----------|----------|
| `eeg_raw/` | BrainVision files (RSEEG/) | 220 |
| `eeg_preprocessed/` | EEGLAB SET/FDT pairs (EC + EO per subject) | 205 |
| `eeg_localizer/` | Electrode positions (MATLAB .mat, Brainstorm format) | 145 |
| `behavioral/` | Demographics + cognitive + emotion batteries | 227 |

**Channel List (62 channels):**
Fp1, Fp2, F7, F3, Fz, F4, F8, FC5, FC1, FC2, FC6, T7, C3, Cz, C4, T8, VEOG, CP5, CP1, CP2, CP6, AFz, P7, P3, Pz, P4, P8, PO9, O1, Oz, O2, PO10, AF7, AF3, AF4, AF8, F5, F1, F2, F6, FT7, FC3, FC4, FT8, C5, C1, C2, C6, TP7, CP3, CPz, CP4, TP8, P5, P1, P2, P6, PO7, PO3, POz, PO4, PO8

**Cognitive Test Battery (8 tests):**
- CVLT (California Verbal Learning Test)
- LPS (Leistungsprüfsystem — performance testing)
- RWT (Regensburger Wortflüssigkeits-Test — verbal fluency)
- TAP Alertness, TAP Incompatibility, TAP Working Memory (Test of Attentional Performance)
- TMT (Trail Making Test)
- WST (Wortschatz Test — vocabulary)

**Emotion & Personality Battery (41 instruments):**
- Personality: NEO-FFI, BISBAS, UPPS (impulsivity)
- Emotion regulation: ERQ, CERQ, COPE
- Affect/mood: MDBF (Days 1–3), FEV
- Emotional intelligence: TEIQue-SF
- Stress/anxiety: PSQ, STAI-G-X2, TICS
- Anger: STAXI
- Social support: MSPSS, F-SozU K-22
- Quality of life: NYC-Q
- Other: MARS, TAS (alexithymia), FTP, LOT-R (optimism), YFAS (food addiction)

**Medical Data:**
- Anthropometry, blood pressure, blood sample biomarkers

**Demographics (META file):**
- ID, gender, age (5-year bins), handedness, education, drug screening, smoking status, SKID diagnosis (DSM-IV), Hamilton scale, BSL-23, AUDIT, alcohol units, family alcohol history, relationship status

**Notes:**
- Preprocessed data has 205 of 220 raw subjects (15 excluded during QC)
- Raw EEG acquisition bandpass: 0.015–1000 Hz at 2500 Hz; preprocessed: 1–45 Hz at 250 Hz
- Richest behavioral phenotyping of all datasets — ideal for brain-behavior correlation analyses
- **Best P20 candidate**: no notch filter at any stage means no spectral holes; cleanest spectrum for extended-range analysis up to 100+ Hz
- 50 Hz line noise (Germany) may be present in raw data but was likely captured by ICA in preprocessed version

---

# Replication Summary

| Dataset | Size | Subjects | Ch | Hz | Notch | FOOOF Range | Spectral Holes |
|---------|------|----------|-----|------|-------|-------------|----------------|
| CHBMP | 27 GB | 282 | 62–120 | 200 | 60 Hz (hw) | [1, 45] | 60 Hz |
| Dortmund | 75 GB | 608 (+208) | 64 | 1000→250 | [50, 100] (sw) | [1, 45] | 50, 100 Hz |
| EEGMMIDB | 3.6 GB | 110 | 64 | 160 | 60 (sw) | [1, 75] | 60 Hz |
| HBN | 686 GB | 929 | 129 | 500→250 | [60, 120] (sw) | [1, 45] | 60, 120 Hz |
| LEMON | 75 GB | 220 | 62 | 2500→250 | **None** | [1, 85] | **None** |
| **Total** | **~867 GB** | **~2,149** | | | | | |

hw = hardware notch applied during acquisition; sw = software notch applied in analysis pipeline.

**P20 extended-range priority:**
1. **LEMON** — no spectral holes, cleanest path to [1, 100+] Hz
2. **HBN** — 60 Hz hole avoidable by analyzing [45–58] + [62–100] Hz; massive N=929
3. **Dortmund** — 50 Hz hole directly in gamma inverse noble region; less suitable despite 1000 Hz native rate
