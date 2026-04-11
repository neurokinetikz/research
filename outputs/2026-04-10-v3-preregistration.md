# Pre-Registration: v3 Enrichment Reanalysis Predictions

**Date:** 2026-04-10
**Status:** Written before v3 extraction completes
**v3 changes from v2:** Lowered bandwidth floor (0.5 Hz → 2×freq_res), 50 Hz notch filter on LEMON and Dortmund
**Analysis changes from v2 run:** Added interior-only derived metrics (peak_height, ramp_depth, center_depletion, asymmetry), power sensitivity sweep

---

## Predictions: v3 vs v2 Enrichment (9 EC datasets, top-50% power, Hz-corrected)

### What should NOT change (bandwidth floor and notch don't affect these)

| Finding | v2 value | v3 prediction | Rationale |
|---|---|---|---|
| Beta-low ascending ramp shape | 12/13 consistent | 12/13 | Interior positions unaffected by either change |
| Beta-low noble_4 | -55% | -55% ± 5 | Far from 50 Hz, interior position |
| Beta-low inv_noble_6 | +91% | +91% ± 5 | Far from 50 Hz, interior position |
| Alpha attractor | +42% | +42% ± 5 | Far from 50 Hz, interior position |
| Alpha Noble1 | +34% | +34% ± 5 | Same |
| Theta profile | 6/13 consistent | 6/13 | No 50 Hz in theta range |
| Beta-high profile | 2/13 consistent | 2/13 | 50 Hz above beta-high range |
| Alpha profile | 1/13 consistent | 1/13 | Same |

### What SHOULD change

**Gamma inv_noble_5 (49.87 Hz) in LEMON and Dortmund:**

| Dataset | v2 value | v3 prediction | Rationale |
|---|---|---|---|
| Dortmund | +135% (power-filtered) | +20% to +40% | Remove 50 Hz spike (+384% raw density). Genuine neural signal at ~50 Hz is modest. |
| LEMON | +53% (power-filtered) | +20% to +40% | Remove +185% spike. Should converge toward US dataset values. |
| EEGMMIDB | +35% | +35% ± 3 | No notch applied, unchanged |
| HBN (all) | +30-50% | +30-50% ± 3 | No notch applied, unchanged |
| CHBMP | -14% | -14% ± 3 | 60 Hz mains, no notch needed |
| **9-ds mean** | **+74%** | **+30% to +45%** | European datasets will drop substantially |

**Gamma inv_noble_4 (48.6 Hz) in LEMON and Dortmund:**

| Dataset | v2 value | v3 prediction | Rationale |
|---|---|---|---|
| Dortmund | +34% | +25% to +40% | Slight improvement -- 50 Hz spike leakage no longer distorts aperiodic fit |
| LEMON | +5% | +10% to +25% | Same -- aperiodic fit cleaner, may detect more genuine peaks |
| 9-ds mean | +22% | +20% to +30% | Small change |

**Gamma inv_noble_6 (50.7 Hz) in LEMON and Dortmund:**

| v2 9-ds mean | v3 prediction | Rationale |
|---|---|---|
| +18% | +15% to +25% | Slight improvement from cleaner aperiodic near 50 Hz |

**Gamma boundary (32.19 Hz) and lower nobles:**
No change expected. 50 Hz notch doesn't affect 32-48 Hz.

**Per-dataset gamma predictions at notch-affected positions:**

| Position | EEGM (v3) | LEM (v3) | Dort (v3) | CHBMP (v3) | HBN mean (v3) |
|---|---|---|---|---|---|
| inv_noble_3 (46.5 Hz) | +56 ± 5 | +13 ± 5 | -10 ± 10 | +32 ± 5 | +10 ± 5 |
| inv_noble_4 (48.6 Hz) | +44 ± 5 | +5 to +30 | +15 to +40 | -69 ± 5 | +30 ± 5 |
| inv_noble_5 (49.9 Hz) | +35 ± 5 | +20 to +40 | +20 to +40 | -91 ± 5 | +35 ± 5 |
| inv_noble_6 (50.7 Hz) | -4 ± 5 | -75 to -30 | -10 to +30 | -87 ± 5 | +55 ± 5 |

**Peak count predictions per dataset (v2 → v3):**

| Dataset | v2 peaks | v3 prediction | Change source |
|---|---|---|---|
| EEGMMIDB | 266K | 275-310K (+5-15%) | Bandwidth floor only |
| LEMON | 468K | 490-540K (+5-15%) | Bandwidth floor + notch |
| Dortmund | 1,190K | 1,240-1,370K (+5-15%) | Bandwidth floor + notch (large dataset) |
| CHBMP | 742K | 775-855K (+5-15%) | Bandwidth floor only |
| HBN (all 5) | ~2,000K | 2,100-2,300K (+5-15%) | Bandwidth floor only |

**LEMON subject count:** v3 LEMON EC shows 167 subjects (v2 had 203). The difference may be due to the notch filter causing some subjects to fail R² thresholds, or the bandwidth floor change affecting peak detection. LEMON EO should have ~202 (no notch for EO since it's the same preprocessing). **Update:** need to verify -- v2 LEMON EC had 203 but v3 shows 167, which matches the old f₀=7.83 count. The extra 36 subjects in v2 may have been from the broader band boundaries capturing subjects at the theta/alpha edge. If v3 notch removes peaks near 50 Hz that were keeping some subjects above the 30-peak minimum, those subjects could drop out. But this would only affect gamma band. More likely: the LEMON EC extraction ran before the notch fix and found only 167 subjects that had data files. Need to check if this is a data availability issue vs extraction parameter issue.

**Bandwidth floor effect (all datasets, all bands):**

The lowered bandwidth floor (0.5 Hz → 0.06-0.42 Hz) allows FOOOF to fit narrower peaks. Predictions:

| Effect | Prediction | Rationale |
|---|---|---|
| Total peak count | +5% to +15% increase | Narrower peaks fit where 0.5 Hz minimum previously prevented detection |
| Alpha peak count | Largest increase (+10-20%) | Alpha peaks are genuinely narrow (~0.2-0.3 Hz true width) |
| Theta peak count | Modest increase (+5-10%) | Theta peaks are broader |
| Beta-low peak count | Moderate increase (+5-15%) | Mix of narrow and broad peaks |
| Beta-high peak count | Moderate increase (+5-10%) | Same |
| Gamma peak count | Minimal change (+0-5%) | Gamma 2×freq_res = 0.42 Hz, close to old 0.5 Hz floor |
| Enrichment magnitudes | ±5 pp at most positions | More peaks but distributed similarly |
| Alpha mountain | +34% ± 8 | Narrow alpha peaks now better detected -- could sharpen or dilute depending on noise |
| Power distribution | Lower-power peaks shift to narrower bandwidth | The 0.5 Hz floor was forcing FOOOF to widen small peaks; now they can be narrow + low-power |
| Bandwidth as quality metric | Now usable | Distribution should spread from 0.06 to 3+ Hz instead of all floored at 0.5 |

**v2 vs v3 bandwidth distribution comparison:**

| Band | v2 median BW | v2 range | v3 predicted median | v3 predicted range |
|---|---|---|---|---|
| Theta | 0.50 Hz (floored) | 0.50-0.50 | 0.15-0.30 Hz | 0.06-2.0 Hz |
| Alpha | 0.50 Hz (floored) | 0.50-0.50 | 0.20-0.40 Hz | 0.06-3.0 Hz |
| Beta-low | 0.50 Hz (floored) | 0.50-0.73 | 0.30-0.50 Hz | 0.16-4.0 Hz |
| Beta-high | 0.50 Hz (floored) | 0.50-0.73 | 0.40-0.70 Hz | 0.26-6.0 Hz |
| Gamma | 0.50 Hz (floored) | 0.50-0.56 | 0.45-0.80 Hz | 0.42-8.0 Hz |

---

## Predictions: Individual Differences (v3, top-50% power)

### Cognitive (LEMON EC × LPS)

| Finding | v2 value | v3 prediction | Rationale |
|---|---|---|---|
| FDR survivors | 4 | 3-5 | Bandwidth change is secondary; power filter dominates |
| beta_low_mountain × LPS | rho=-0.264, FDR=0.033 | rho=-0.25 ± 0.03 | Boundary-based metric; stable |
| beta_low_attractor × LPS | rho=-0.260, FDR=0.033 | rho=-0.25 ± 0.03 | Interior position; stable |
| beta_low_ramp_depth × LPS | Not in v2 | rho ≈ -0.20 to -0.28 | Interior-only metric; should show similar signal without boundary noise |
| beta_low_asymmetry × LPS | Not in v2 | rho ≈ -0.20 to -0.28 | Same prediction -- steeper ramp = better cognition |
| alpha_peak_height × LPS | Not in v2 | rho ≈ +0.10 to +0.20 | Weaker than beta-low; alpha mountain may not predict LPS |

### Age Trajectories

| Finding | v2 value | v3 prediction | Rationale |
|---|---|---|---|
| HBN FDR survivors | 40/70 | 38-45 | Now 90 features (12 pos + 6 derived × 5 bands); some NaN-excluded. Bandwidth change secondary |
| Dortmund FDR survivors | 45/70 | 43-50 | Same |
| LEMON FDR survivors | 5/70 | 4-7 | Same |
| Lifespan jointly significant | 28 | 26-30 | Same |
| Lifespan opposite direction | 23/28 | 21-25 | Same |
| beta_low_ushape rho (HBN) | +0.169 | +0.16 ± 0.02 | Robust interior positions |
| beta_low_ramp_depth rho (HBN) | Not in v2 | +0.15 to +0.20 | Interior metric should replicate ushape finding |

### EO Cognitive Replication (LEMON EO × LPS)

| Finding | v2 EC value | v3 EO prediction | Rationale |
|---|---|---|---|
| FDR survivors | 4 (EC) | 0-2 (EO) | EO weaker than EC; old report had 0 FDR for EO |
| beta_low_mountain × LPS | rho=-0.264 (EC) | rho ≈ -0.15 to -0.22 (EO) | Same direction, ~20-30% weaker |
| beta_low_attractor × LPS | rho=-0.260 (EC) | rho ≈ -0.15 to -0.22 (EO) | Same |
| beta_low_ramp_depth × LPS | Not tested yet | rho ≈ -0.12 to -0.20 (EO) | Interior metric; should track mountain |
| Direction match EC→EO | — | All same direction | Consistent with v2 |

### Other Individual Differences

| Finding | v2 value | v3 prediction | Rationale |
|---|---|---|---|
| Personality FDR | 0 | 0 | Never significant; 8,778+ tests, ratio ≈ 1.1× chance |
| Personality top rho | ~0.17 | < 0.20 | Noise ceiling |
| Medical FDR | 0 | 0 | Never significant; ~2,500 tests |
| Medical top rho | ~0.15 | < 0.20 | Underpowered for theta |
| Handedness continuous FDR | 0 | 0 | Never significant |
| Handedness dichotomized FDR | 0 | 0 | Never significant |
| Sex × age interaction (HBN) | 0 | 0 | Fisher z-test; no sex differences in trajectory |
| Sex × age interaction (Dort) | 0 | 0 | Same |
| State × age interaction (LEMON) | 0 | 0 | EC→EO delta independent of age |
| State × age interaction (Dort) | 0 | 0 | Same |
| Externalizing FDR | 4 | 3-6 | Modest effect; bandwidth change secondary |
| Externalizing top rho | ~0.14 | 0.12-0.16 | beta_low_noble_3 or ramp_depth |
| Internalizing FDR | 0 | 0-2 | Was 4 in old report, 0 in v2; borderline |
| Test-retest ICC | +0.32 | +0.30 to +0.35 | Bandwidth change may help or hurt slightly |
| Within-session ICC | +0.30 | +0.28 to +0.33 | Same |
| Cross-band coupling α_bnd×βL_att | rho=+0.07 | rho ≈ -0.05 to +0.10 | Artifact is gone; boundary-based |
| Cross-band coupling α_n1×βL_ramp_depth | Not in v2 | rho ≈ +0.10 to +0.25 | Interior-only; may reveal genuine coupling |

---

## Predictions: Power Sensitivity Sweep

### Alpha Noble1 across thresholds (Dortmund)

| Threshold | v2 value | v3 prediction |
|---|---|---|
| 0% (all peaks) | +7% | +5% to +10% |
| 25% | +12% | +10% to +15% |
| 50% | +18% | +16% to +22% |
| 75% | +31% | +28% to +35% |

**Prediction:** Signal strengthens monotonically with threshold. No plateau. v3 may show slightly different absolute values if bandwidth floor change affects which peaks are kept by the power filter (narrow-but-tall peaks now detected that were previously widened to 0.5 Hz and given lower power).

### Beta-low ramp depth across thresholds (Dortmund)

| Threshold | v2 inv_noble_6 | v3 prediction |
|---|---|---|
| 0% | +43% | +40% to +48% |
| 25% | +61% | +58% to +65% |
| 50% | +76% | +72% to +82% |
| 75% | +82% | +78% to +88% |

**Prediction:** Ramp steepens with threshold but plateaus earlier than alpha (beta-low signal is stronger, less noise dilution). v3 values should be within ±5 pp of v2.

### LPS cognitive correlation across thresholds

| Threshold | v2 beta_low_mountain rho | v3 prediction |
|---|---|---|
| 0% | ~-0.15 | ~-0.15 |
| 25% | ~-0.20 | ~-0.20 |
| 50% | -0.264 | -0.25 ± 0.03 |
| 75% | ~-0.28 | ~-0.28 |

**Prediction:** Signal strengthens with threshold but diminishing returns above 50%. The 50% threshold is near-optimal -- higher thresholds reduce sample size faster than they improve signal.

---

## Predictions: Bandwidth Distribution (new in v3)

With the 0.5 Hz floor removed, v3 should show a meaningful bandwidth distribution for the first time:

| Band | Predicted median bandwidth | Predicted range |
|---|---|---|
| Theta | 0.15-0.30 Hz | 0.06-2.0 Hz |
| Alpha | 0.20-0.40 Hz | 0.06-3.0 Hz |
| Beta-low | 0.30-0.50 Hz | 0.16-4.0 Hz |
| Beta-high | 0.40-0.70 Hz | 0.26-6.0 Hz |
| Gamma | 0.45-0.80 Hz | 0.42-8.0 Hz |

**Key test:** If bandwidth is now informative, filtering on bandwidth + power should outperform power-only filtering. Specifically: narrow+tall peaks (bandwidth < median AND power > median) should show stronger enrichment at Noble1/attractor than tall-only peaks (power > median regardless of bandwidth).

If bandwidth filtering DOESN'T add value beyond power, then the old 0.5 Hz floor wasn't actually hiding useful information -- it was just clipping a dimension that doesn't discriminate signal from noise.

---

## Predictions: EC/EO Comparison (LEMON + Dortmund)

v3 changes should not affect EC/EO patterns except at gamma inv_noble_5 in LEMON/Dortmund.

### State-Sensitivity by Band

| Band | v2 pattern | v3 prediction | Rationale |
|---|---|---|---|
| Theta | Most state-sensitive (max Δ 40-70 pp) | Same | Below 50 Hz, no notch effect |
| Alpha | Moderate (max Δ 20-30 pp) | Same | Same |
| Beta-low | Moderate (max Δ 20-35 pp), ramp attenuates under EO | Same | Same |
| Beta-high | Low (max Δ < 15 pp) | Same | Same |
| Gamma | Near-zero (max Δ < 10 pp) | Same except inv_noble_5 for LEMON/Dort | Notch removes 50 Hz; EC/EO delta at that position should shrink |

### Key EC→EO Shifts (should replicate from v2)

| Shift | v2 observation | v3 prediction |
|---|---|---|
| Theta boundary_hi EC > EO | +66% EC vs ~+5% EO | Same ± 5 |
| Alpha Noble1 stable across states | +34% EC, ~+30% EO | Same ± 5 |
| Beta-low ramp attenuates under EO | inv_noble_6 ~20 pp lower in EO | Same ± 5 |
| Gamma state-invariant | Max Δ < 10 pp | Same |

---

## Predictions: Dortmund 2×2 (EC/EO × pre/post)

| Band | v2 observation | v3 prediction |
|---|---|---|
| Beta-low | 13/13 stable across all 4 conditions | 12-13/13 |
| Gamma | 11/13 stable | 10-12/13 (inv_noble_5 may shift with notch) |
| Alpha | 7/13 stable | 6-8/13 |
| Theta | 4/13 stable | 3-5/13 |
| Beta-high | 0/13 stable (all too weak) | 0-2/13 |

### Fatigue Effect (pre vs post)

| Finding | v3 prediction | Rationale |
|---|---|---|
| Theta boundary_hi strengthens with fatigue | EC-pre ~+68%, EC-post ~+78% (Δ≈+10 pp) | Same direction as v2 |
| Alpha mountain contracts with fatigue | Noble1 EC-pre ~+41%, EC-post ~+35% (Δ≈-6 pp) | Upper flank narrows |
| Beta-low ramp fatigue-invariant | inv_noble_6: EC-pre ~+76%, EC-post ~+76% (Δ≈0) | Structural, not state-dependent |
| Gamma ramp fatigue-invariant | inv_noble_3: Δ < 5 pp across conditions | Same |

### Fatigue × Eyes Interaction

| Band | Max interaction |Δ| | v3 prediction | Rationale |
|---|---|---|---|
| Theta | ~47 pp (boundary) | 30-50 pp | Theta is the only band with notable interaction |
| Alpha | ~14 pp | 10-18 pp | Weak |
| Beta-low | ~13 pp | 8-15 pp | Structural band; weak interactions |
| Beta-high | ~12 pp | 5-15 pp | Weak |
| Gamma | ~9 pp | 5-12 pp | State-invariant |

---

## Predictions: HBN Cross-Release Consistency

| Band | v2 prediction | Rationale |
|---|---|---|
| Beta-low | 12-13/13 agreement across R1-R6 | Most robust band; no 50 Hz effect |
| Alpha | 5-8/13 | Weak signal; high inter-release variability |
| Theta | 4-7/13 | Similar to v2 |
| Gamma | 8-11/13 | HBN is US (60 Hz), no notch. Should be clean. |
| Beta-high | 2-5/13 | Weak signal |

### Cross-Release SD at Key Positions

| Position × Band | v2 prediction (HBN only) | Rationale |
|---|---|---|
| Alpha Noble1 SD | < 5 pp | Consistent across releases in v2 |
| Beta-low inv_noble_4 SD | < 8 pp | Consistent |
| Gamma inv_noble_5 SD (HBN) | < 5 pp | No notch in HBN; should be very consistent |
| Theta boundary_hi SD | 10-20 pp | Most variable position |

---

## Predictions: Adult vs Pediatric

| Band | v2 profile r (adult vs ped) | v3 prediction | Rationale |
|---|---|---|---|
| Beta-low | ~0.95+ | 0.93-0.98 | Same ramp shape; adults more extreme |
| Theta | ~0.85-0.95 | 0.83-0.95 | Similar shape, adults stronger |
| Alpha | ~0.60-0.80 | 0.55-0.80 | Children narrower mountain |
| Gamma | ~0.50-0.70 | 0.50-0.75 | Cleaner with notch on adult European datasets |
| Beta-high | ~0.60-0.80 | 0.55-0.80 | Weak signal in both |

### Key Adult-Pediatric Differences

| Finding | v3 prediction |
|---|---|
| Alpha mountain narrower in children | Confirmed (ped attractor > adult attractor relative to flanks) |
| Beta-low ramp weaker in children | Confirmed (ped inv_noble_6 ~50-70% of adult value) |
| Gamma ramp similar across development | Confirmed at inv_noble_3; inv_noble_5 now cleaner for adults |

---

## Predictions: Per-Band Developmental Patterns (HBN age rhos)

| Band | Key features | v2 rho | v3 prediction | Rationale |
|---|---|---|---|---|
| Alpha | inv_noble_3 (broadening) | +0.191 | +0.18 ± 0.03 | Interior; stable |
| Alpha | inv_noble_4 | +0.15-0.20 | +0.15 ± 0.03 | Same |
| Alpha | boundary (relaxation) | +0.10-0.15 | +0.10 ± 0.05 | Boundary position; noisier |
| Beta-low | noble_3 (center clearing) | -0.165 | -0.16 ± 0.03 | Interior; stable |
| Beta-low | ushape (deepening) | +0.169 | +0.16 ± 0.02 | Composite metric; stable |
| Beta-low | ramp_depth (new) | Not in v2 | +0.14 to +0.20 | Should track ushape |
| Beta-low | asymmetry (new) | Not in v2 | +0.14 to +0.20 | Should track ramp_depth |
| Gamma | noble_3 (lower depletion) | -0.208 | -0.20 ± 0.03 | Interior; no 50 Hz |
| Gamma | inv_noble_4 | +0.17 | +0.16 ± 0.03 | Same |
| Theta | boundary (-0.14) | -0.14 ± 0.05 | -0.14 ± 0.05 | Boundary; noisier |

### Per-Band FDR Summary (HBN age)

| Band | v2 FDR | v3 prediction | Note |
|---|---|---|---|
| Alpha | ~11/18 | 9-14 | 18 features per band (12 positions + 6 derived) |
| Beta-low | ~10/18 | 9-13 | Same |
| Beta-high | ~7/18 | 5-10 | Same |
| Gamma | ~8/18 | 7-11 | Notch doesn't affect HBN |
| Theta | ~4/18 | 3-6 | Same |

---

## Predictions: Per-Band Adult Aging (Dortmund age rhos)

| Band | Key feature | v2 rho direction | v3 prediction |
|---|---|---|---|
| Alpha | inv_noble_3 (narrowing) | negative | Same sign, ±0.03 |
| Beta-low | inv_noble_1 (center filling) | positive | Same sign, ±0.03 |
| Beta-low | ramp_depth (new) | Not in v2 | Negative (ramp flattens with age) |
| Beta-low | asymmetry (new) | Not in v2 | Negative (same) |
| Gamma | inv_noble_5 (weakening) | negative | Magnitude may change (notch cleans Dortmund) |
| Theta | attractor | negative | Same sign, ±0.05 |

### Dortmund FDR by Band (Dortmund age)

| Band | v2 FDR | v3 prediction | Note |
|---|---|---|---|
| Gamma | ~10/18 | 8-13 | Notch may change gamma features in Dortmund |
| Beta-low | ~9/18 | 8-12 | 18 features per band |
| Alpha | ~9/18 | 8-12 | Same |
| Beta-high | ~8/18 | 6-11 | Same |
| Theta | ~4/18 | 3-6 | Same |

---

## Predictions: Per-Release Replication (HBN)

| Metric | v2 value | v3 prediction | Rationale |
|---|---|---|---|
| Cross-release rho correlation | 0.68-0.82 | 0.65-0.85 | No HBN changes |
| Features FDR in ALL 5 releases | 2 (alpha inv_noble_3, inv_noble_4) | 2 ± 1 | Same |
| Features FDR in ≥3/5 releases | ~10 | 8-12 | Same |
| Theta per-release replication | 0 features in all 5 | 0-1 | Weakest band |

---

## Predictions: Sex Differences (HBN)

| Finding | v2 value | v3 prediction | Rationale |
|---|---|---|---|
| FDR survivors | ~13 | 10-16 | Modest effects; bandwidth change secondary |
| Largest d | ~0.24 (beta-high noble_1) | 0.20-0.28 | Same direction |
| Males more lower-noble in beta-high | Present | Same | No 50 Hz effect on beta-high |
| Alpha sex differences | Minimal | Minimal | Same |

---

## Predictions: Psychopathology (HBN)

| Dimension | v2 FDR | v3 prediction | Rationale |
|---|---|---|---|
| p_factor | 0 | 0 | Never significant |
| Attention | 0 | 0 | Never significant |
| Externalizing | 4 | 3-6 | Modest; bandwidth secondary |
| Internalizing | 0 | 0-2 | Borderline |

### Externalizing Key Features

| Feature | v2 rho | v3 prediction | Rationale |
|---|---|---|---|
| beta_low_noble_3 | +0.140 | +0.12 to +0.16 | Interior; stable |
| gamma_inv_noble_4 | -0.137 | -0.10 to -0.16 | May change with notch (Dort/LEMON gamma) |
| gamma_mountain | +0.137 | +0.10 to +0.16 | Same |
| beta_low_inv_noble_1 | +0.129 | +0.10 to +0.15 | Interior; stable |
| beta_low_ramp_depth (new) | Not in v2 | -0.10 to -0.16 | Less differentiated = more externalizing |
| beta_low_asymmetry (new) | Not in v2 | -0.10 to -0.16 | Same prediction |

### Internalizing Key Features (if any survive)

| Feature | v2 value | v3 prediction |
|---|---|---|
| gamma_noble_4 | rho=-0.125 | -0.10 to -0.14 |
| beta_high_noble_3 | rho=+0.108 | +0.08 to +0.12 |

---

## Predictions: Reliability Per Band

### 5-Year Test-Retest (Dortmund ses-1 vs ses-2)

| Band | v2 median ICC | v3 prediction | Rationale |
|---|---|---|---|
| Beta-low | ~0.40 | 0.38-0.44 | Strongest; most stable band |
| Beta-high | ~0.38 | 0.35-0.42 | Same |
| Gamma | ~0.30 | 0.28-0.35 | May change slightly at inv_noble_5 with notch |
| Alpha | ~0.28 | 0.25-0.32 | Bandwidth floor may affect narrow alpha peaks |
| Theta | ~0.22 | 0.18-0.26 | Weakest; most extraction-dependent |
| **Overall** | **+0.32** | **+0.29 to +0.35** | |

**Key per-feature ICC predictions (5-year):**

| Feature | v2 ICC | v3 prediction | Rationale |
|---|---|---|---|
| beta_low_inv_noble_4 | ~0.45 | 0.40-0.50 | Strong interior position |
| beta_low_attractor | ~0.40 | 0.35-0.45 | Interior; stable |
| beta_low_ramp_depth (new) | Not in v2 | 0.35-0.50 | Interior-only; may be more stable than boundary-based metrics |
| alpha_noble_1 | ~0.30 | 0.25-0.35 | Bandwidth floor change may affect |
| alpha_peak_height (new) | Not in v2 | 0.25-0.35 | Interior-only version of mountain |
| gamma_inv_noble_3 | ~0.30 | 0.28-0.35 | No notch effect at 47.9 Hz |
| theta_attractor | ~0.20 | 0.15-0.25 | Weakest |

### Within-Session (EC-pre vs EC-post)

| Band | v2 median ICC | v3 prediction | Rationale |
|---|---|---|---|
| Beta-low | ~0.38 | 0.36-0.44 | Most reliable |
| Beta-high | ~0.36 | 0.34-0.42 | Same |
| Gamma | ~0.28 | 0.26-0.35 | Same |
| Alpha | ~0.26 | 0.24-0.32 | Same |
| Theta | ~0.20 | 0.16-0.26 | Least reliable |
| **Overall** | **+0.30** | **+0.28 to +0.33** | |

### Cross-Condition (EC vs EO)

| Band | v3 prediction | Rationale |
|---|---|---|
| Beta-low | 0.30-0.40 | Ramp shape consistent across states |
| Beta-high | 0.28-0.38 | Same |
| Gamma | 0.22-0.32 | Same |
| Alpha | 0.20-0.30 | Mountain broadens under EO; less stable |
| Theta | 0.10-0.22 | Most state-sensitive; lowest cross-condition stability |
| **Overall** | **+0.25 to +0.32** | Lower than within-session (different states) |

### Group Profile Stability (ses-1 vs ses-2)

| Band | v3 prediction | Rationale |
|---|---|---|
| Beta-low | r > 0.98 | Nearly identical profiles across 5 years |
| Gamma | r > 0.95 (cleaner with notch for Dortmund) | Same |
| Beta-high | r > 0.90 | Same |
| Alpha | r > 0.85 | Same |
| Theta | r > 0.70 | Most variable |

### Age Does NOT Predict 5-Year Change

| Finding | v3 prediction | Rationale |
|---|---|---|
| FDR survivors (age × ses2-ses1 change) | 0 | Same as v2; individual change is not age-dependent |

---

## Predictions: Cross-Band Coupling

### Population-Level (per dataset)

**Boundary-based pairs (expected weak/absent):**

| Pair | v2 LEMON | v2 Dort | v2 HBN | v3 prediction | Rationale |
|---|---|---|---|---|---|
| alpha_boundary × beta_low_attractor | +0.07 | ~0 | ~0 | -0.05 to +0.10 all | Artifact gone |
| alpha_boundary × beta_low_boundary | ~0 | ~0 | ~0 | -0.05 to +0.10 all | Both boundary positions; noisy |

**Interior-based pairs (expected genuine coupling):**

| Pair | v3 LEMON prediction | v3 Dort prediction | v3 HBN prediction | Rationale |
|---|---|---|---|---|
| alpha_noble_1 × beta_low_ushape | +0.10 to +0.30 | +0.10 to +0.25 | +0.10 to +0.25 | IAF-linked; should replicate |
| alpha_noble_1 × beta_low_ramp_depth | +0.10 to +0.30 | +0.10 to +0.25 | +0.10 to +0.25 | Interior version of above |
| alpha_peak_height × beta_low_asymmetry | +0.10 to +0.25 | +0.10 to +0.20 | +0.10 to +0.20 | Shared E/I balance? |
| alpha_attractor × beta_low_attractor | -0.10 to -0.25 | -0.10 to -0.20 | -0.05 to -0.15 | Inverse: more alpha center = less beta-low center |
| beta_low_ushape × gamma_inv_noble_3 | -0.05 to +0.10 | -0.05 to +0.10 | -0.05 to +0.10 | Likely independent |
| beta_low_ramp_depth × gamma_ramp_depth | -0.05 to +0.10 | -0.05 to +0.10 | -0.05 to +0.10 | Likely independent |

**Cross-band FDR survivors prediction:**

| Dataset | v3 prediction | Rationale |
|---|---|---|
| LEMON (N≈200) | 5-15 FDR survivors | Small N limits power; interior pairs strongest |
| Dortmund (N=608) | 15-40 FDR survivors | Larger N detects weaker effects |
| HBN (N≈900) | 20-50 FDR survivors | Largest N; pediatric may differ from adult |

### Coupling Stability (Dortmund EC-pre vs EC-post)

| Pair | v3 prediction | Rationale |
|---|---|---|
| alpha_boundary × beta_low_attractor | r(pre,post) ≈ +0.05 to +0.15 | Weak coupling; low stability |
| alpha_noble_1 × beta_low_ushape | r(pre,post) ≈ +0.15 to +0.30 | If genuine, should be stable |
| alpha_noble_1 × beta_low_ramp_depth | r(pre,post) ≈ +0.15 to +0.30 | Same prediction |

### Coupling Stability (Dortmund ses-1 vs ses-2, 5 years)

| Pair | v3 prediction | Rationale |
|---|---|---|
| alpha_boundary × beta_low_attractor | r(ses1,ses2) ≈ +0.00 to +0.10 | Boundary noise; not stable |
| alpha_noble_1 × beta_low_ushape | r(ses1,ses2) ≈ +0.10 to +0.25 | If trait-like, stable across years |
| alpha_noble_1 × beta_low_ramp_depth | r(ses1,ses2) ≈ +0.10 to +0.25 | Same |

### Cross-Dataset Replication of Coupling

| Metric | v3 prediction | Rationale |
|---|---|---|
| LEMON-Dort rho correlation (all pairs) | r ≈ 0.40-0.60 | Two adult datasets; similar populations |
| LEMON-HBN rho correlation | r ≈ 0.20-0.40 | Adult vs pediatric; weaker agreement |
| Dort-HBN rho correlation | r ≈ 0.25-0.45 | Same |

---

## Predictions: Three-Dataset Validation (HBN + LEMON + Dortmund)

| Comparison | v2 value | v3 prediction |
|---|---|---|
| LEMON vs Dortmund age rhos | r ≈ +0.80 | +0.75 to +0.85 |
| LEMON vs HBN age rhos | r ≈ -0.49 | -0.55 to -0.40 |
| HBN vs Dortmund age rhos | r ≈ -0.55 | -0.60 to -0.45 |
| Features FDR in ALL 3 datasets | ~4 (all beta-low) | 3-5 (all beta-low) |

### Cognition × Age × Enrichment Triangle

| Feature | LPS direction | Age direction | v3 prediction |
|---|---|---|---|
| beta_low_ramp_depth (new) | Negative (steeper = better) | Inverted-U (peaks ~20) | Opposite directions confirmed |
| beta_low_asymmetry (new) | Negative | Inverted-U | Same |
| beta_low_attractor | Negative | U-shape (minimized ~20) | Same |

---

## What Would Falsify the Main Findings

1. **Beta-low ramp disappears at any power threshold.** If the ascending ramp at inv_noble_3 through inv_noble_6 goes to zero or reverses sign at any threshold from 0% to 75%, it's not a robust finding.

2. **Alpha Noble1 goes negative at 50% threshold.** If the mountain disappears even with power filtering, the FOOOF peak detection is not capturing alpha reliably.

3. **Cognitive LPS correlations reverse sign.** If beta_low_ramp_depth × LPS is positive (steeper ramp = worse performance), the interpretation inverts.

4. **9-dataset consistency drops below 8/13 for beta-low.** If correcting the 50 Hz artifact introduces new inconsistencies in beta-low, something else is wrong.

5. **Gamma inv_noble_5 INCREASES after 50 Hz notch in European datasets.** This would mean the notch is creating artifacts rather than removing them.

6. **Interior-only metrics (ramp_depth, asymmetry) show weaker cognitive correlations than boundary-based metrics (mountain, ushape).** If boundary noise was ADDING signal rather than obscuring it, the interior metrics would perform worse.

7. **New bandwidth dimension shows NO discriminative value.** If narrow+tall peaks produce identical enrichment to tall-only peaks, the bandwidth floor change was irrelevant.

8. **HBN cross-release consistency WORSENS in v3.** Since no HBN extraction parameters changed, any consistency change would indicate an analysis-level bug.

9. **Lifespan opposite-direction count drops below 15/28.** The inverted-U trajectory is a central finding; if more than half the jointly-significant features become same-direction, the developmental vs aging contrast collapses.

10. **Per-band reliability ordering changes.** Beta-low should be the most reliable band and theta the least. If this ordering reverses, the extraction changes are affecting band-specific noise in unexpected ways.

---

## Verification Plan

After v3 extraction completes:

1. Update `NEW_PEAK_BASE` in `run_all_f0_760_analyses.py` from `exports_adaptive_v2` to `exports_adaptive_v3`
2. Run `python scripts/run_all_f0_760_analyses.py --step all --min-power-pct 50`
3. Check LEMON subject count (167 vs 203 discrepancy)
4. Check bandwidth distribution (first usable bandwidth data)
5. Check gamma inv_noble_5 in LEMON and Dortmund (50 Hz notch effect)
6. Check each prediction against actual values -- score accuracy
7. Investigate any large deviations (predictions off by > 2× range)
8. Run power sensitivity sweep (`--step power_sensitivity`)
9. Test bandwidth filtering value: compare enrichment with power-only vs power+bandwidth filtering
10. Update audit report with v3 results, prediction accuracy, and bandwidth findings
11. Document any new issues discovered for v4 planning
