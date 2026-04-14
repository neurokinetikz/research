# v3 Enrichment Reanalysis: Executive Summary

**Date:** 2026-04-12
**Scope:** Complete reanalysis of the φ-lattice enrichment framework across 9 EEG datasets, 2,097 subjects, 17 recording conditions, correcting three systematic biases in the original analysis pipeline
**Output:** 22 analysis steps, 28 CSV outputs, 4 comprehensive reports

---

## What We Did

We audited the enrichment analysis pipeline from raw EEG through FOOOF peak extraction through Voronoi enrichment through individual-differences correlations. We identified three methodological biases, corrected them, re-extracted all data, and re-ran every analysis. We then validated the corrected extraction parameters through a 48-configuration parameter sweep on GCP.

### Three Corrections Applied

**1. f₀ alignment (extraction at 7.83 → 7.60).** The original extraction assigned peaks to frequency bands using f₀=7.83, but enrichment coordinates used f₀=7.60. This 3% mismatch caused peaks near band boundaries to wrap to the wrong position via modular arithmetic. At the beta-low lower boundary, ~20 Hz peaks wrapped into the 12.30 Hz bin, creating a spurious +101% "boundary enrichment." Corrected value: -33%. Five sign flips at boundary/noble_6 positions across theta, beta-low, and gamma.

**2. Hz-weighted null hypothesis.** The Voronoi bins assumed uniform peak distribution in log-frequency (u-space), but FOOOF detects peaks in linear Hz. The upper end of each phi-octave covers 1.618× more Hz than the lower end, creating a systematic 48 pp ascending ramp from bottom to top of every band. After Hz-correction, beta-high's apparent "ascending ramp" vanished entirely and revealed a genuine descending pattern (boundary +86%, boundary_hi -20%).

**3. Merged theta+alpha FOOOF.** Per-band FOOOF fitting created a 15× step function in peak density at f₀=7.60 Hz (the theta/alpha boundary). The alpha aperiodic model detected many peaks near 7.6-8.3 Hz that the theta model absorbed into its aperiodic slope. Merging theta and alpha into a single FOOOF fit (4.70-12.30 Hz) eliminated this artifact. The merge barely affects alpha (profile r=0.913 vs separate) but substantially changes theta (r=0.200).

### Additional Methodological Improvements

- **max_n_peaks raised from 3-15 to 15 uniformly.** The old per-band caps (3 for theta, 6 for alpha) discarded the majority of detected peaks, biasing enrichment toward dominant-frequency positions. With cap=15, all genuine peaks are retained.
- **50 Hz notch filter** on European datasets (LEMON, Dortmund). Removed mains-frequency contamination that inflated gamma inv_noble_5 by +384% in Dortmund.
- **Bandwidth floor lowered** from 0.5 Hz to 2×freq_res. Allows FOOOF to fit narrow peaks at their true width, making bandwidth usable as a quality metric for the first time.
- **R² saved per peak** for post-hoc quality filtering. Median R²=0.977; enrichment stable across thresholds from 0.70 to 0.95.
- **Power filtering at analysis time.** Top 50% of peaks per band by power, removing low-amplitude noise peaks that dilute the signal. The single most important analysis parameter.
- **Interior-only derived metrics** (peak_height, ramp_depth, center_depletion, asymmetry). These avoid boundary positions entirely, providing robust individual-differences measures unaffected by boundary artifacts.

### Parameter Validation

48-configuration parameter sweep on EEGMMIDB (GCP, 32 cores, ~30 minutes total):
- **peak_threshold (0.001 to 2.0):** No effect on enrichment after power filtering. Exception: 2.0 kills gamma peaks (only 7% survive the 2 SD threshold), making gamma unusable.
- **min_peak_height (0.0 to 0.20):** No effect up to 0.13. Acts as a power filter above 0.15.
- **max_n_peaks (5 to 20):** The only parameter that matters. Inflection at cap=12 (matching 12 Voronoi bins). Cap=15 chosen for production because per-subject enrichment needs more peaks for stable estimates.
- **nperseg_floor:** No effect (adaptive vs 2500).

**Conclusion:** v3 config (thresh=0.001, cap=15, power filter=50%) is optimal. Permissive extraction + analysis-time filtering outperforms strict extraction across all metrics.

---

## What We Found

### The Enrichment Landscape: Two Regimes

**Regime 1: Alpha Mountain.** Alpha is the only band where the central octave is enriched. Attractor +43%, Noble1 +30%, inv_noble_1 +39%. This is the empirical realization of IAF clustering -- the dominant ~10 Hz rhythm concentrates at the attractor and Noble1 (φ⁻¹ position). The mountain broadens asymmetrically upward during development (alpha_inv_noble_4 × age: rho=+0.354, the strongest developmental feature) and narrows during aging (Dortmund alpha_inv_noble_3 × age: rho=-0.276). Under EC, the mountain sharpens (+16% EO → +43% EC at attractor).

**Regime 2: Edge Ramp.** In theta, beta-low, and gamma, peaks cluster toward the upper edge of each phi-octave. Beta-low: noble_4 -56% ramping to inv_noble_6 +87%. Theta: boundary -52% ramping to boundary_hi +126% (the f₀ convergence). Gamma: inv_noble_3 +22% to inv_noble_4 +42%. Beta-high descends (boundary +86% to boundary_hi -20%) -- the same edge-ramp seen from the other side of the ~20 Hz attractor. Alpha shows the inverse at every position -- where alpha is enriched, everything else is depleted, and vice versa.

### Cross-Boundary Architecture

| Boundary | Hz | Type | Enrichment |
|---|---|---|---|
| **theta/alpha** | **7.60 (f₀)** | **Cliff** | +126% below, -35% above (Δ=-161 pp) |
| alpha/beta-low | 12.30 (f₀×φ) | Void | -74% below, -33% above (spectral desert) |
| **beta-low/beta-high** | **19.90 (f₀×φ²)** | **Bridge** | +78% below, +86% above (the only cross-band enrichment) |
| beta-high/gamma | 32.19 (f₀×φ³) | Weak | -20% below, -4% above |

The ~20 Hz bridge is the single strongest cross-band feature. Peaks converge from both sides at f₀×φ², consistent with Fibonacci three-wave resonance (f(0)+f(1)=f(2)) or biophysical convergence of M-current, corticomuscular coherence, and PMBR time constants at this frequency.

### Cognitive Correlates

**31 FDR-significant enrichment × cognitive correlations** (up from 4 in the original report). The signal spans 4 cognitive tests (LPS logical reasoning, TAP incompatibility, RWT verbal fluency, TMT trail making) and 4 frequency bands (beta-low, theta, gamma, alpha).

Top findings:
- beta_low_center_depletion × LPS: rho=-0.273 (steeper ramp = better reasoning)
- theta_ushape × TAP_Incompat: rho=+0.267 (more theta differentiation = faster executive control)
- gamma_inv_noble_4 × RWT: rho=+0.260 (steeper gamma ramp = more verbal fluency)

The cognitive signal replicates under EO (25 FDR survivors), disproving the old claim that it was EC-specific. After partialing out age, the top LPS correlation attenuates from -0.273 to -0.153 but survives (p=0.030) -- about half the variance is age-shared, half is genuine cognitive association.

**0 FDR across 11,970 personality tests.** Enrichment predicts cognitive performance but is psychometrically silent for personality traits.

### Developmental and Aging Trajectories

**HBN (N=927, ages 5-21): 60 FDR-significant age correlations.** The alpha mountain broadens upward (alpha_inv_noble_4: rho=+0.354), beta-low center clears (noble_3: rho=-0.194), and gamma ramp sharpens modestly (noble_3: rho=-0.207). All effects are continuous developmental trajectories replicated across 5 independent HBN releases (cross-release rho correlation: mean r=0.787).

**Dortmund (N=608, ages 20-70): 41 FDR-significant age correlations.** The same features show OPPOSITE directions: alpha_inv_noble_3 rho=-0.276 (mountain narrows), beta_low_attractor rho=+0.311 (center fills in).

**Inverted-U lifespan trajectory.** 31 features are FDR-significant in at least 2 of 3 datasets; 28 show opposite directions between development and aging. Spectral differentiation peaks in early adulthood (~20 years) and de-differentiates thereafter.

### Psychopathology (HBN, N=906)

- **Externalizing: 18 FDR survivors.** Flatter enrichment profiles = more externalizing behavior (gamma_inv_noble_4: rho=-0.159, beta_low_inv_noble_1: rho=+0.127). Less spectral differentiation across multiple bands.
- **Internalizing: 7 FDR survivors.** Dissociable pattern: MORE upper-alpha enrichment (alpha_inv_noble_4: rho=+0.126), not less. Opposite direction from externalizing in alpha.
- **p_factor and attention: 0 FDR.** The spectral differentiation link is specific to externalizing, not general psychopathology.

### Cross-Band Coupling

**alpha_noble_1 × beta_low_ushape: rho=+0.412 (HBN), +0.344 (Dortmund).** Individuals with taller alpha mountains have deeper beta-low ramps. This is a new finding -- the old report's alpha_boundary × beta_low_attractor coupling (rho=-0.41) was a boundary artifact. The interior-only coupling is equally strong, replicates across datasets, and is individually stable within session (coupling product r=+0.393, p<1e-22).

93 FDR-significant cross-band pairs in HBN (N=927). Beta-low and alpha are the most coupled bands. Beta-low and gamma are independent (rho≈0).

### Reliability

**5-year test-retest (Dortmund ses-1 vs ses-2, N=208):**

| Band | Median ICC |
|---|---|
| Beta-low | **+0.604** |
| Beta-high | +0.507 |
| Alpha | +0.454 |
| Theta | +0.382 |
| Gamma | +0.250 |
| **Overall** | **+0.421** |

For comparison, Paper 3's dominant-peak ICC was -0.25 to -0.36. Per-subject Voronoi enrichment is a stable individual trait. beta_low_ushape has ICC=**+0.746** -- the most stable metric across 5 years.

**Group profiles:** r > 0.96 across 5 years in all bands. The EC→EO delta pattern is itself stable (r=0.875-0.990 between sessions). Age does not predict 5-year change (0 FDR across 90 tests).

**HBN cross-release:** Alpha shows 13/13 position agreement across 5 independent releases. inv_noble_1 has SD=1.0 across releases (N=136-322 each) -- remarkable reproducibility.

### State Effects

Theta is the most state-sensitive band (boundary_hi: EC +128%, EO +48%, Δ=80 pp in Dortmund). This is f₀ convergence -- EC drives theta peaks toward 7.60 Hz. Gamma is state-invariant (max Δ=14 pp). Beta-low ramp attenuates under EO (shape preserved, contrast reduced). Alpha mountain sharpens under EC (attractor +41% EC vs +36% EO in Dortmund).

State sensitivity is age-independent (0 FDR across 90 tests in both LEMON and Dortmund).

### Nulls

- Personality/emotion: 0 FDR (11,970 tests)
- Medical/metabolic: 0 FDR (~2,500 tests)
- Handedness: 0 FDR (HBN continuous + dichotomized, Dortmund)
- Sex × age interaction: 0 FDR (HBN + Dortmund)
- State × age interaction: 0 FDR (LEMON + Dortmund)
- Age predicts 5-year change: 0 FDR (90 tests)

---

## What Changed From the Original Report

### Artifacts Removed

| Original Claim | Corrected | Cause |
|---|---|---|
| Beta-low boundary +101% | **-33%** | f₀ mismatch wrapping |
| Theta boundary +47% | **-55%** | f₀ mismatch wrapping |
| Beta-low "U-shape" | **Ascending ramp** | Lower boundary was artifact |
| Beta-high ascending ramp | **Descending ramp** | Hz-correction removed bias |
| Gamma inv_noble_5 +61% | **+14%** | 50 Hz line noise |
| Cross-band α×βL rho=-0.41 | **rho=+0.001** | Boundary artifact |
| Alpha Noble1 +25% | **+30%** (corrected for cap bias) | Old cap=6 inflated Noble1 |

### Findings Strengthened

| Finding | Original | v3 Corrected |
|---|---|---|
| Cognitive FDR | 4 | **31** |
| HBN age FDR | 43/66 | **60/90** |
| Externalizing FDR | 10 | **18** |
| Internalizing FDR | 4 | **7** |
| Alpha attractor | +24% | **+43%** |
| Theta consistency | 7/13 | **12/13** |
| Test-retest ICC | +0.42 | **+0.42** (confirmed) |

### New Discoveries

1. **31 cognitive FDR survivors spanning 4 tests and 4 bands** (theta, alpha, beta-low, gamma all predict cognition)
2. **EO cognitive replication** (25 FDR -- cognitive signal is NOT EC-specific)
3. **Beta-high descending ramp** (boundary +86%, peaks cluster at ~20 Hz from above)
4. **~20 Hz bridge** -- the only boundary enriched from both sides
5. **alpha × beta-low interior coupling** (rho=+0.412, replicates across datasets)
6. **Externalizing = spectral de-differentiation** (18 FDR, dissociable from internalizing)
7. **Theta 12/13 consistency** (was 7/13 -- ascending ramp to f₀ is a robust finding)
8. **Interior-only metrics** outperform boundary-based metrics for individual differences
9. **beta_low_ushape ICC=+0.746** -- most stable enrichment metric across 5 years
10. **Per-release alpha consistency 13/13 with SD=1.0** -- the most reproducible cross-sample finding

---

## Methodology Summary

### v3 Extraction Pipeline

```
Raw EEG → Bandpass 1-59 Hz → 50 Hz notch (European) → Resample 250 Hz
→ Merged theta+alpha FOOOF (4.70-12.30 Hz, single aperiodic)
→ Per-band FOOOF (beta-low through gamma)
→ peak_threshold=0.001, max_n_peaks=15, R²≥0.70
→ Bandwidth floor = 2×freq_res
→ Save: freq, power, bandwidth, phi_octave, r_squared per peak
→ Analysis: top-50% power filter per band
→ Hz-weighted Voronoi enrichment at 12 degree-6 positions
```

### Datasets

| Dataset | N | Channels | Population | Conditions |
|---|---|---|---|---|
| EEGMMIDB | 109 | 64 | Adult (US) | 14 runs pooled |
| LEMON | 203 | 59 | Adult (Germany) | EC, EO |
| Dortmund | 608 | 64 | Adult (Germany) | EC/EO × pre/post × ses-1/ses-2 |
| CHBMP | 250 | 62-120 | Adult (Cuba) | EC |
| HBN R1-R6 | 927 | 128 | Pediatric (US) | Mixed resting |

### Analysis Steps (22 total)

1. Pooled enrichment (9 EC datasets)
2. Cognitive correlations (LEMON EC, 720 tests)
3. HBN developmental trajectory (age, sex, psychopathology)
4. Dortmund adult aging
5. EC vs EO comparison (LEMON + Dortmund)
6. Dortmund 2×2 (EC/EO × pre/post)
7. Personality (LEMON, 11,970 tests)
8. 5-year test-retest reliability (Dortmund ses-1 vs ses-2)
9. Old vs new comparison
10. HBN cross-release consistency
11. EO cognitive replication
12. Adult vs pediatric comparison
13. HBN per-release age replication
14. Lifespan trajectory (HBN + Dortmund + LEMON)
15. Cross-band coupling (3 datasets)
16. Within-session reliability (EC-pre vs EC-post)
17. Medical/metabolic markers (LEMON)
18. Handedness (HBN + Dortmund)
19. Sex × age interaction
20. State × age interaction
21. Power sensitivity sweep
22. Report comparison (old vs new)

### GCP Infrastructure

- Custom VM image with full Python environment
- 32-core c2d-standard machines, parallel per-subject extraction
- Direct copy from GCS (gcsfuse dropped subjects)
- Spawn → extract → push → delete in ~5 min per dataset
- 48-config parameter sweep in ~30 min
- All raw data and results archived on GCS

---

## Files

### Reports
- [2026-04-12-v3-enrichment-detailed-analysis.md](2026-04-12-v3-enrichment-detailed-analysis.md) -- Complete analysis (757 lines)
- [2026-04-12-v3-results-and-preregistration-verification.md](2026-04-12-v3-results-and-preregistration-verification.md) -- Pre-registration verification
- [2026-04-10-enrichment-reanalysis-audit.md](2026-04-10-enrichment-reanalysis-audit.md) -- Methodological audit
- [2026-04-10-v3-preregistration.md](2026-04-10-v3-preregistration.md) -- Pre-registration (151 predictions, 10 falsification criteria)
- [2026-04-11-v4-plan.md](2026-04-11-v4-plan.md) -- v4 parameter optimization plan

### Scripts
- `scripts/run_f0_760_extraction.py` -- v3 extraction (all parameters)
- `scripts/run_all_f0_760_analyses.py` -- 22-step analysis suite
- `scripts/run_v4_sweep.py` -- 48-config parameter sweep
- `scripts/gcp_run.sh` -- GCP VM orchestration

### Data
- `exports_adaptive_v3/` -- v3 peak CSVs (17 conditions, local)
- `gs://eeg-extraction-data/` -- All raw data + results (GCS)
- `outputs/f0_760_reanalysis/` -- 28 analysis CSV outputs
