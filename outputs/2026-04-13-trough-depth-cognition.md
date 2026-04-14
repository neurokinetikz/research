# Per-Subject Trough Depth vs Cognition: Analysis 5

**Date:** 2026-04-13
**Question:** Does individual variation in trough depth predict cognitive performance (LEMON dataset)?
**Script:** `scripts/trough_depth_cognition.py`
**Output data:** `outputs/trough_depth_by_age/trough_depth_cognition.csv`
**Depends on:** Analyses 1-4

## Methods

Per-subject trough depths (windowed count ratio, as in Analysis 3) computed for 203 LEMON subjects. Correlated with 8 cognitive tests (LPS reasoning, RWT verbal fluency, TMT trail-making, CVLT verbal memory, WST vocabulary, TAP Alertness, TAP Incompatibility, TAP Working Memory). Spearman correlations, Benjamini-Hochberg FDR correction at q < 0.05 across 40 tests (5 troughs × 8 tests). Age-partialed analysis: quadratic residualization of both depth and cognitive scores on age, then Spearman on residuals.

## Results

### Raw correlations: 1/40 FDR survivor

| Survivor | ρ | p | Direction |
|----------|:---:|:---:|---|
| TAP_WM × βH/γ | +0.230 | 0.001 | Shallower trough → better WM |

The direction is **opposite** to the inhibitory prediction (deeper trough = stronger inhibition = better cognition). A shallower βH/γ trough predicts better working memory.

Near-survivors (p < 0.02, not FDR-significant): LPS × βL/βH (ρ = -0.171), TMT × α/β (ρ = -0.170), RWT × θ/α (ρ = -0.171), LPS × α/β (ρ = +0.162).

### Age-partialed: 2/40 FDR survivors (likely artifactual)

| Survivor | ρ (raw) | ρ (age-partialed) | p |
|----------|:---:|:---:|:---:|
| CVLT × δ/θ | +0.145 | **-0.555** | <0.0001 |
| CVLT × βH/γ | +0.111 | **+0.439** | <0.0001 |

**These results are flagged as likely artifactual.** The CVLT × δ/θ correlation reverses direction and quadruples in magnitude after age-partialing. LEMON has a bimodal age distribution (young 20-40, old 55-80, gap 40-55). Quadratic residualization on a bimodal distribution can create spurious correlations when the two age groups differ systematically in both the predictor and outcome. The CVLT × δ/θ result should not be interpreted as a genuine age-independent cognitive association.

### Effect size comparison

| Measure | Peak |ρ| (raw) | Peak |ρ| (age-partialed) |
|---------|:---:|:---:|
| **Trough depth** (this analysis) | 0.230 | 0.555 (artifactual) |
| **Within-band enrichment** (paper) | 0.270 | 0.150 |
| **FDR survivors** (trough depth) | 1/40 | 2/40 (artifactual) |
| **FDR survivors** (enrichment) | 31/720 | -- |

## Interpretation

### Trough depth does not robustly predict cognition

The contrast with the paper's within-band enrichment results is stark: 31 FDR survivors for enrichment vs. 1 for trough depth, in the same dataset with the same subjects. Trough depth at the individual-subject level is either:

1. **Too noisy to detect cognitive correlations.** Per-subject depth ratios have large SDs (1.0-5.6 from Analysis 3), attenuating true correlations. With N = 203 and this level of noise, the detectable effect size threshold is high.

2. **Genuinely unrelated to cognition.** If trough depth reflects structural properties of the spectral landscape (generator strength, equipment sensitivity) rather than functional properties of neural circuits, it would not predict cognition.

3. **Capturing a different signal.** Trough depth measures boundary sharpness; enrichment measures within-band organization. Both could reflect inhibitory integrity but at different levels. The within-band signal may be more cognitively relevant because it reflects the precision of oscillatory tuning (which directly affects communication efficiency), while boundary depth reflects the gross separation between bands (which may be necessary but not sufficient for efficient communication).

### Implications for the inhibitory framework

The cognitive null weakens but does not refute the inhibitory interpretation of spectral differentiation. The paper's core cognitive finding (31 FDR survivors for within-band enrichment features) is real and method-independent. But extending it to trough depth -- "if differentiation = inhibitory integrity, then boundaries should predict cognition too" -- does not hold at the individual level with current measurement methods.

This creates a useful distinction for the Discussion: **within-band spectral organization predicts cognition; between-band boundary depth does not** (at least not with the windowed-count approach at N = 203). The inhibitory framework should emphasise within-band shaping as the cognitively relevant output of inhibitory circuits, not boundary depth per se.

## Caveats

1. **Measurement noise.** The windowed-count approach is a coarse per-subject depth estimator. A KDE-based approach with optimized parameters, or a model-based approach fitting the trough shape, might yield more reliable per-subject estimates and reveal buried signal.

2. **Sample size.** N = 203 with 40 tests provides ~80% power to detect |ρ| ≥ 0.20 at α = 0.05 per test, but after FDR correction the effective threshold is higher. True associations weaker than |ρ| ≈ 0.25 are likely missed.

3. **LEMON age distribution.** The bimodal distribution (no subjects aged 40-55) makes age-partialing unreliable. Raw correlations are more trustworthy for this dataset.

## Figures

None generated (no strong effects to visualize).
