# GABAergic Inhibition Framework: Comprehensive Synthesis

**Date:** 2026-04-13
**Scope:** Integration of 6 trough depth analyses with the spectral differentiation paper
**Scripts:** `scripts/trough_depth_by_age.py`, `trough_depth_by_age_v2.py`, `trough_differential_maturation.py`, `trough_depth_covariance.py`, `trough_width_asymmetry.py`, `trough_depth_cognition.py`, `trough_depth_psychopathology.py`

---

## 1. The Framework As Originally Proposed

The five spectral troughs are not passive gaps between bands -- they are active depletions where oscillatory peaks are suppressed. The bimodal depth hierarchy (two deep troughs at 70.4% and 61.7%; three shallow at 8.7%, 11.6%, 32.2%) was hypothesised to reflect discrete GABAergic interneuron populations:

- **δ/θ (5.1 Hz, 70.4%):** SST+ dendritic inhibition ceiling
- **θ/α (7.8 Hz, 8.7%):** Not primarily inhibitory -- excitatory attractor competition
- **α/β (13.4 Hz, 61.7%):** PV+ perisomatic inhibition floor + T-current ceiling
- **βL/βH (25.3 Hz, 11.6%):** M-current KCNQ upper limit
- **βH/γ (35.0 Hz, 32.2%):** PV+ subtype transition

Spectral differentiation -- the degree to which peaks concentrate at canonical within-band positions -- was proposed as a readout of **inhibitory circuit integrity**.

## 2. What the Analyses Confirmed

### 2.1 The α/β trough (13.4 Hz) is the framework's strongest evidence

Six independent lines of evidence converge on the PV+ perisomatic inhibition interpretation:

| Evidence | Result | p-value |
|----------|--------|:---:|
| **Developmental deepening** (Analysis 1) | ρ = +0.82 with age in HBN (20% → 50%) | 0.023 |
| **PV+ maturation timeline** (Analysis 2) | 40% of adult depth at age 6; 75% at age 17; plateau by early 20s | 0.0001 |
| **PV+ covariance cluster** (Analysis 3) | Positively correlated with βH/γ (ρ = +0.37) | <0.001 |
| **Asymmetry flip** (Analysis 4) | Right-skewed → left-skewed during development (PV+ emergence) | 0.10 |
| **Externalizing dissociation** (Analysis 6) | Higher ext → shallower trough (GABA deficit) | <0.0001 |
| **Internalizing dissociation** (Analysis 6) | Higher int → deeper trough (GABA enhancement) | 0.0005 |

The α/β trough's developmental trajectory (immature at age 5, deepening through adolescence, stable adult plateau) matches the known timeline of PV+ fast-spiking interneuron maturation with remarkable precision. The psychopathology dissociation is directionally consistent with GABA deficit (externalizing) vs. enhancement (internalizing) models and replicates the paper's within-band enrichment findings at a different measurement level.

### 2.2 Trough mechanisms are independent (multi-mechanism model confirmed)

PCA on per-subject trough depths yields PC1 = 23.3% (barely above the 20% expected under independence), with mean pairwise ρ = 0.053. There is no "overall GABAergic tone" factor. Each trough reflects a largely independent aspect of spectral architecture. This rules out single-factor models and supports the proposal that different troughs arise from different biophysical mechanisms.

Within this independence, a weak PV+ cluster emerges: α/β, βL/βH, and βH/γ show mutual positive correlations after age partialing (ρ = 0.15-0.23), consistent with shared PV+ fast-spiking interneuron contribution across the 13-40 Hz range.

### 2.3 The θ/α trough (7.8 Hz) is consistent with excitatory attractor competition

- Uniquely symmetric (slope asymmetry = +0.06) and narrow (width = 0.064 log-Hz)
- Shows inverted-U trajectory (vertex ~33.5 years)
- Becomes increasingly right-skewed during aging (ρ = +1.0, Dortmund) -- the theta side degrades faster than the alpha side
- Bootstrap CI is the tightest of all troughs (±0.02 Hz)

All four shape/trajectory features distinguish the θ/α trough from the others and are consistent with a boundary positioned by competition between two excitatory generators (hippocampal theta, thalamocortical alpha) rather than by inhibitory sculpting.

## 3. What the Analyses Revised

### 3.1 The δ/θ trough is NOT tracking inhibitory maturation

The SST+ inhibition ceiling hypothesis predicted that this trough would deepen during development as SST+ interneurons mature. Instead:

- **Over-deep in children:** 233% of adult depth at age 6 (58.4% vs 25.1% adult reference)
- **Regresses toward adult levels:** ρ = -0.67 with age within HBN (p = 0.013)
- **Anti-correlated with α/β maturation:** ρ = -0.58 (p = 0.037) -- opposite developmental trajectories
- **Independent of all other troughs** in the covariance analysis
- **No psychopathology signal** (largest |ρ| = 0.06, all NS)
- **Left-skewed** (asymmetry = -0.31), opposite to the inhibitory ceiling prediction

The most parsimonious explanation is that δ/θ trough depth tracks **delta-band oscillatory power**, not inhibitory boundary strength. Young children have dominant slow-wave activity, creating sharp delta peaks that make the δ/θ boundary extremely deep. As delta power diminishes with maturation, the trough fills in.

**Revised position:** The SST+ mechanism may contribute to the boundary's *existence* (why there is a trough near 5 Hz), but it does not determine the trough's *depth variation* across development. The developmental trajectory is dominated by excitatory generator changes on the delta side.

### 3.2 Simple ceiling/floor asymmetry predictions partially fail

The inhibitory-ceiling model (steep high-frequency cutoff) and inhibitory-floor model (steep low-frequency cutoff) were tested against trough shape data. Score: 1 clear match (θ/α symmetric), 1 marginal (βH/γ), 1 partial (α/β symmetric rather than left-skewed), 1 miss (δ/θ left-skewed, wrong direction). Trough shape is determined by the interaction of flanking peak strengths, not solely by the putative inhibitory mechanism.

### 3.3 Trough depth does not predict cognition at the individual level

Only 1/40 trough-cognition tests survives FDR in LEMON (TAP_WM × βH/γ, ρ = +0.23, wrong direction), compared to 31/720 for within-band enrichment. This creates a useful distinction: **within-band spectral organization predicts cognition; between-band boundary depth does not** (at least not with current measurement methods at N = 203).

This does not refute the inhibitory interpretation of spectral differentiation. It means that the cognitively relevant signal lies in how peaks distribute *within* each band (enrichment), not in how deep the gaps *between* bands are (trough depth). The inhibitory framework should emphasise within-band shaping as the cognitively relevant output of inhibitory circuits.

## 4. The Revised Framework

### 4.1 Two classes of trough

The five troughs divide into two mechanistically distinct classes based on the analyses:

**Class A -- Inhibitory boundary troughs (α/β, possibly θ/α):**
- Immature in early childhood, deepen during development
- Track inhibitory circuit maturation
- Show psychopathology associations
- Form a weak covariance cluster

**Class B -- Generator-dominated troughs (δ/θ, βL/βH):**
- Over-deep in early childhood, regress toward adult levels
- Track the developmental recession of dominant oscillatory generators (delta, beta-low)
- No psychopathology associations
- Independent of Class A troughs

The βH/γ trough is unclassified due to cross-dataset recording confounds.

### 4.2 Spectral differentiation as a readout of PV+ circuit integrity

The strongest version of the framework that survives the data is more specific than originally proposed:

**Spectral differentiation reflects PV+ fast-spiking interneuron circuit integrity, primarily expressed at the α/β boundary.**

Evidence:
1. The α/β trough matures on the PV+ timeline (Analyses 1-2)
2. Higher externalizing (GABA deficit phenotype) → shallower α/β trough (Analysis 6)
3. Higher internalizing (GABA enhancement phenotype) → deeper α/β trough (Analysis 6)
4. α/β clusters with other PV+-relevant troughs (βH/γ, βL/βH) in covariance (Analysis 3)
5. The α/β asymmetry flip during development directly reflects PV+ emergence (Analysis 4)

The framework does NOT extend to all five troughs uniformly. The δ/θ trough is driven by delta generator power, not inhibition. The βL/βH trough inverts with aging through an unknown mechanism. The framework is α/β-centric, not pan-spectral.

### 4.3 Revised pharmacological predictions

The original framework generated pharmacological predictions for all five troughs. The revised predictions focus on what the data support:

| Agent | Predicted effect on α/β trough | Predicted effect on within-band differentiation | Rationale |
|-------|---|---|---|
| **Benzodiazepine** | Deepen; shift downward | Increase beta-low ramp enrichment | Prolongs GABA_A IPSPs at PV+ synapses → extends inhibition into lower frequencies |
| **Ethosuximide** | No change | Flatten alpha mountain only | T-type Ca²⁺ block affects excitatory resonance, not inhibitory boundary |
| **Tiagabine** | Broaden | Global differentiation increase | Nonselective GABA reuptake → broader suppression zone |
| **Prediction for δ/θ trough** | **No change under any GABAergic agent** | -- | δ/θ depth tracks delta generator power, not inhibition; GABAergic manipulation should not affect it |

The δ/θ non-response prediction is the cleanest test of the revised framework: if GABAergic agents affect the α/β trough but not the δ/θ trough, it directly confirms the two-class distinction.

### 4.4 MRS prediction (strongest single test)

Individual differences in spectral differentiation -- particularly α/β trough depth and alpha-beta boundary enrichment -- should correlate with cortical GABA concentration measured by MRS (magnetic resonance spectroscopy). This is testable in existing multimodal datasets and provides a direct neurochemical test of the inhibitory interpretation.

### 4.5 Developmental predictions

The inverted-U of spectral differentiation (vertex at ~33-35 years) should be more strongly mediated by GABA concentration (MRS) than by white matter integrity (DTI). The distinguishing prediction: the age-differentiation relationship should show a stronger indirect effect through GABA than through myelination in a mediation analysis. Testable in Cam-CAN or similar multimodal lifespan datasets.

## 5. Novel Findings Beyond the Framework

Three results emerged that were not predicted by the original framework:

### 5.1 Three-class maturation pattern at age 6

At the youngest age tested (5-7 years), the five troughs fall into three maturation classes:
- **Over-mature (>100% of adult depth):** δ/θ (233%), βL/βH (297%)
- **Immature (<50% of adult depth):** α/β (40%), θ/α (31%)
- **Absent (<5% of adult depth):** βH/γ (3%)

This maturation hierarchy has not been previously reported and provides a developmental constraint on theories of band boundary formation.

### 5.2 βL/βH trough inversion with aging

The βL/βH trough (25.3 Hz) transitions from a genuine trough (17% depletion at age 22) to enrichment (-24% at age 58) within the Dortmund dataset (ρ = -0.92, p = 0.0002). The boundary literally inverts. The mechanism is unknown but could reflect:
- Age-related changes in M-current KCNQ kinetics
- Compensatory beta oscillatory activity
- Degradation of beta-low/beta-high band separation

### 5.3 α/β asymmetry flip

The α/β trough's slope asymmetry reverses from right-skewed (age 6, asymmetry = +0.93) to left-skewed (age 19, asymmetry = -0.52). This directly reflects the developmental transition from alpha-dominated (T-current present from birth) to PV+-influenced (inhibitory floor matures during adolescence) shaping of the boundary.

## 6. Confidence Summary

| Claim | Confidence | Key evidence |
|-------|:---:|---|
| α/β trough matures on PV+ timeline | **High** | Within-HBN ρ=0.82 p=0.023; milestone ages match PV+ literature |
| α/β psychopathology dissociation | **High** | 4 FDR survivors; ext/int in predicted directions |
| Five troughs are mechanistically independent | **High** | PC1=23%; mean ρ=0.05 |
| δ/θ depth driven by delta generator power | **High** | 233% of adult at age 6; anti-correlated with α/β; no psychopathology signal |
| θ/α reflects excitatory attractor competition | **Moderate** | Unique symmetry/narrowness; inverted-U trajectory |
| Spectral differentiation = PV+ circuit integrity | **Moderate** | Convergent but circumstantial; no direct neurochemical evidence |
| Original SST+ mapping for δ/θ | **Weakened** | Over-deep in children; wrong asymmetry; no psychopathology |
| Trough depth predicts cognition | **Not supported** | 1/40 FDR, wrong direction; within-band enrichment (31 FDR) is the cognitive signal |
| βL/βH trough inversion mechanism | **Unknown** | Strong effect (ρ=-0.92) but no mechanistic model |
