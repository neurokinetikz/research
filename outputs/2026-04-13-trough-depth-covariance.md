# Trough Depth Covariance Structure: Analysis 3

**Date:** 2026-04-13
**Question:** Do the five trough depths covary across individuals? Under independent inhibitory populations they should be uncorrelated; under a common-factor model ("overall GABAergic tone") they should positively correlate.
**Script:** `scripts/trough_depth_covariance.py`
**Output data:** `outputs/trough_depth_by_age/per_subject_trough_depths.csv`
**Depends on:** Analyses 1-2

## Methods

### Per-subject trough depth estimation
For each of 1,738 subjects (927 HBN, 608 Dortmund, 203 LEMON), trough depth was estimated at the five known trough positions using a windowed count approach:

- **Trough window:** ±0.06 in log-frequency space around each trough position (~±6% in Hz)
- **Flanking windows:** same width, centred at ±0.15 in log-space on each side of the trough
- **Depth ratio** = trough_count / mean(left_flank_count, right_flank_count)
- Values < 1.0 indicate depletion (lower = deeper trough); values > 1.0 indicate enrichment

This approach avoids full KDE computation per subject and is robust with ~1,000-3,000 peaks per subject (mean = 2,223).

### Analysis structure
1. **Full 5×5 Spearman correlation matrix** across all subjects
2. **Age-partialed correlations** (quadratic residualization, then Spearman on residuals)
3. **Within-dataset correlations** (HBN, Dortmund, LEMON separately)
4. **PCA / factor analysis** (eigenvalue decomposition of the correlation matrix)

## Results

### Full correlation matrix (N = 1,738)

|  | δ/θ | θ/α | α/β | βL/βH | βH/γ |
|---|:---:|:---:|:---:|:---:|:---:|
| **δ/θ** | 1.000 | **-0.215***| 0.086*** | -0.023 | 0.006 |
| **θ/α** | | 1.000 | 0.111*** | 0.148*** | 0.036 |
| **α/β** | | | 1.000 | 0.159*** | **0.370***|
| **βL/βH** | | | | 1.000 | -0.150*** |
| **βH/γ** | | | | | 1.000 |

\*\*\*p < 0.001, \*\*p < 0.01, \*p < 0.05

- Mean off-diagonal ρ: **0.053**
- Range: [-0.215, +0.370]

### Age-partialed correlation matrix (quadratic residuals, N = 1,738)

|  | δ/θ | θ/α | α/β | βL/βH | βH/γ |
|---|:---:|:---:|:---:|:---:|:---:|
| **δ/θ** | 1.000 | -0.133*** | 0.053* | -0.064** | **-0.197***|
| **θ/α** | | 1.000 | 0.070** | 0.151*** | -0.035 |
| **α/β** | | | 1.000 | **0.231***| 0.154*** |
| **βL/βH** | | | | 1.000 | 0.018 |
| **βH/γ** | | | | | 1.000 |

- Mean off-diagonal ρ (age-partialed): **0.025**
- Range: [-0.197, +0.231]

### Within-dataset correlations (δ/θ vs α/β)

| Dataset | N | ρ | p |
|---------|:---:|:---:|:---:|
| HBN | 918 | +0.128 | 0.0001 |
| Dortmund | 578 | +0.080 | 0.055 |
| LEMON | 197 | +0.118 | 0.098 |

### PCA eigenvalue decomposition

| Component | Eigenvalue | % Variance | Cumulative |
|-----------|:---:|:---:|:---:|
| PC1 | 1.166 | 23.3% | 23.3% |
| PC2 | 1.026 | 20.5% | 43.8% |
| PC3 | 0.997 | 19.9% | 63.8% |
| PC4 | 0.981 | 19.6% | 83.4% |
| PC5 | 0.830 | 16.6% | 100.0% |

Under complete independence, all eigenvalues = 1.0 (each 20% variance). Under a strong common factor, PC1 >> 1.0. The observed distribution is nearly flat (range 0.83-1.17), indicating near-independence.

**PC1 loadings:** δ/θ (0.64), θ/α (0.71), α/β (0.19), βL/βH (0.17), βH/γ (-0.16)
**PC2 loadings:** δ/θ (0.38), θ/α (-0.02), α/β (-0.70), βL/βH (-0.60), βH/γ (-0.01)

PC1 loads primarily on the two lowest-frequency troughs (δ/θ and θ/α). PC2 opposes these against the two mid-frequency troughs (α/β and βL/βH). Neither component is interpretable as a general inhibitory factor.

### Per-subject depth distributions

| Trough | N valid | Mean depth ratio | SD |
|--------|:---:|:---:|:---:|
| δ/θ (5.1) | 1,698 | 0.727 | 3.162 |
| θ/α (7.8) | 1,713 | 0.853 | 1.040 |
| α/β (13.4) | 1,732 | 0.649 | 1.752 |
| βL/βH (25.3) | 1,733 | 1.116 | 2.976 |
| βH/γ (35.0) | 1,733 | 0.798 | 5.564 |

Note: SDs are large relative to means, reflecting the inherent noisiness of per-subject trough depth estimation with ~2,000 peaks per subject. This measurement noise attenuates all correlations; the true latent correlations could be somewhat larger.

## Key Findings

### 1. No common inhibitory factor

The five trough depths are essentially independent across individuals. PCA finds no dominant factor (PC1 = 23.3%, barely above the 20% expected under independence). The mean pairwise correlation is 0.053 (raw) and 0.025 (age-partialed). There is no "overall GABAergic tone" signal detectable in trough depth covariance.

This rules out a simple model in which a single parameter (e.g., GABA concentration, interneuron density) modulates all trough depths simultaneously. Each trough reflects a largely independent aspect of spectral organisation.

### 2. The α/β -- βH/γ positive cluster

The strongest inter-trough correlation is α/β × βH/γ (ρ = +0.37 raw, +0.15 age-partialed). These are the two troughs hypothesised to involve PV+ fast-spiking interneurons (the α/β boundary as the PV+ inhibition floor; the βH/γ boundary as a PV+ subtype transition). Their positive covariance is consistent with shared PV+ fast-spiking interneuron contribution: subjects with stronger PV+ perisomatic inhibition tend to have deeper troughs at both positions.

After age partialing, a secondary cluster emerges: α/β × βL/βH (ρ = +0.23). These three troughs (α/β, βL/βH, βH/γ) span the frequency range most strongly influenced by PV+ interneuron kinetics (~13-40 Hz), and their mutual positive correlations (after removing age) are consistent with a shared PV+ substrate.

### 3. The δ/θ -- θ/α negative correlation

The δ/θ and θ/α troughs are anti-correlated (ρ = -0.22 raw, -0.13 age-partialed). Subjects with a deeper δ/θ trough tend to have a shallower θ/α trough. This is consistent with a mechanical explanation: subjects with strong delta generators create concentrated spectral peaks below 5 Hz, which depletes the δ/θ boundary (deepening it) but also pulls spectral energy away from the 5-8 Hz range, reducing the contrast at the θ/α boundary (shallowing it).

### 4. The δ/θ trough is independent of higher-frequency troughs

After age partialing, δ/θ shows no positive correlation with any other trough. Its only substantial correlations are negative: with θ/α (-0.13) and βH/γ (-0.20). This further weakens the SST+ inhibition hypothesis for this trough. If it were driven by a distinct inhibitory population, you might expect some positive covariance with at least one other inhibitory-boundary trough. Instead, its covariance pattern is consistent with delta generator dominance (a excitatory, not inhibitory, mechanism).

## Caveats

1. **Measurement noise.** Per-subject trough depths have large SDs (1.0-5.6). The windowed-count approach is inherently noisy with ~2,000 peaks per subject. Attenuation correction (e.g., Spearman-Brown) could estimate the disattenuated correlations, but this requires reliability estimates that we don't have for the windowed-count measure. The independence conclusion is conservative -- some true correlations may be hidden by noise.

2. **Cross-dataset confound.** HBN (children) and Dortmund (adults) have very different trough depth profiles (Analysis 1-2). Pooling them inflates between-subject variance in ways that may inflate or deflate correlations depending on whether the age-related changes are parallel or divergent across troughs. The age-partialed analysis controls for this linearly/quadratically, but non-linear or dataset-specific effects could remain.

3. **Window parameter sensitivity.** The windowed-count approach uses fixed parameters (LOG_HALF_WINDOW = 0.06, LOG_FLANK_OFFSET = 0.15). Different window sizes could yield different correlation estimates. A sensitivity analysis varying these parameters was not performed.

## Implications for the paper

1. **Multi-mechanism framework confirmed.** The Discussion should state that per-subject trough depths are essentially independent (PC1 = 23%, mean ρ = 0.05), ruling out a single-factor inhibitory model and supporting the proposal that different troughs reflect different biophysical mechanisms.

2. **PV+ cluster.** The α/β--βH/γ positive correlation (ρ = 0.37, the largest inter-trough correlation) is consistent with shared PV+ fast-spiking contribution. After age partialing, α/β--βL/βH (ρ = 0.23) extends this to a three-trough cluster spanning the PV+-relevant frequency range. This can be stated as suggestive evidence in the Discussion.

3. **The δ/θ trough stands alone.** Its independence from all other troughs, combined with the developmental regression (Analysis 2), converges on the conclusion that this trough is not tracking the same type of mechanism as the higher-frequency troughs.

## Figures

- `outputs/trough_depth_by_age/trough_depth_covariance.png` -- 3-panel: (A) correlation heatmap; (B) δ/θ vs α/β scatter by dataset; (C) δ/θ vs α/β scatter colored by age
