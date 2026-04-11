# Paper 4 Outline: Per-Band Phi-Lattice Architecture

**Working title:** "Band-Specific Enrichment Reveals Three Organizational Regimes in the Golden Ratio Frequency Lattice"

**Alternate titles:**
- "Fibonacci Coupling Gateways: Per-Band Spectral Organization Across 2,061 Subjects"
- "Beyond Aggregate Enrichment: Band-Specific Architecture of the Phi-Lattice"

---

## Positioning in the Series

| Paper | Question | Answer |
|-------|----------|--------|
| 1 (Frontiers) | Does phi-lattice organization exist? | Yes (aggregate) |
| 2 (EEGMMIDB) | Is phi uniquely best? | Conditionally (anchor-dependent) |
| 3 (LEMON) | Are aggregate metrics reliable? | No (extraction-dominated) |
| **4 (This)** | **What does per-band analysis reveal?** | **Three distinct regimes** |

Paper 4 resolves what Papers 1-3 could only hint at: the aggregate enrichment values were weighted averages across bands with fundamentally different organizational principles. Per-band analysis with proper normalization reveals the actual structure.

---

## Core Contribution

The aggregate phi-lattice analysis reported in Papers 1-3 conflated three distinct organizational regimes operating within the same coordinate system. Using adaptive-resolution spectral extraction and Voronoi density normalization across 9 datasets (2,061 subjects, ~5 million peaks), we resolve:

1. **Alpha mountain** — Noble1/attractor enrichment, boundary depletion (anti-mode-locking)
2. **Beta-low U-shape** — boundary enrichment, center depletion (Fibonacci coupling gateways)
3. **Gamma ascending ramp** — inverse noble enrichment (upward frequency cascade)

Each regime is independently replicated. The phi-lattice is MORE structured per-band than in aggregate — but the structure is different from what aggregate analysis suggested.

---

## Outline

### Abstract (~250 words)
- Papers 1-3 reported aggregate enrichment (boundary -18%, Noble1 +39%)
- We show these are weighted averages across bands with opposing patterns
- Adaptive-resolution extraction + Voronoi bins across 9 datasets, 2,061 subjects
- Three regimes discovered, each independently replicated
- Beta-low U-shape is the strongest signal (13/13 consistent, boundaries +101%/+74%)
- Alpha mountain confirmed as robust (Noble1 +25%, SD=5)
- Gamma +144.8% at Noble1 was aggregate artifact — real signal is ascending ramp at inverse nobles
- Fibonacci additive coupling (f(n) = f(n-1) + f(n-2)) provides mechanistic explanation for boundary enrichment in 4/5 bands
- Alpha's inverted pattern consistent with intrinsic thalamo-cortical resonance overriding coupling
- Cross-release consistency: zero conflicts across 5 HBN pediatric releases (927 subjects)
- Adult-pediatric comparison reveals developmental narrowing of alpha mountain

### 1. Introduction
- Papers 1-3 established aggregate phi-lattice organization but with known fragility
- Paper 3 showed aggregate metrics are extraction-dependent
- Key unresolved question: what does per-band enrichment look like with proper normalization?
- The aggregate boundary depletion (-18%) was a weighted average — per-band analysis needed
- This paper introduces adaptive-resolution extraction and Voronoi binning for the first fair cross-band comparison
- Preview: three distinct regimes, Fibonacci coupling as mechanism

### 2. Methods

#### 2.1 Datasets
- 9 datasets, 2,061 subjects total
- 4 adult: EEGMMIDB (N=109, 64ch, motor), LEMON (N=167, 62ch, rest), Dortmund (N=608, 64ch, rest+cognitive), CHBMP (N=250, 62-120ch, rest)
- 5 pediatric: HBN R1 (136), R2 (150), R3 (184), R4 (322), R6 (135) — ages 5-21, 129ch
- Diversity: 4 countries, 4 EEG systems, ages 5-77, resting-state eyes-closed
- Table of dataset properties

#### 2.2 Adaptive-Resolution Overlap-Trim Extraction
- Problem: fixed nperseg gives different frequency resolution across bands (theta 3 Hz wide vs gamma 20 Hz wide)
- Solution: band-adaptive nperseg ensuring equal position discriminability
- Table: band / nperseg / window / freq_res (theta 31.4s/0.032 Hz through gamma 4.6s/0.218 Hz)
- Overlap-trim: half-octave padding on each side, trim to target band after FOOOF
- f₀=7.83 for extraction (band boundary placement), f₀=7.60 for enrichment analysis
- FOOOF parameters, peak detection, quality filtering

#### 2.3 Voronoi Enrichment Analysis
- Problem: fixed-width bins (±0.05 lu) give different expected counts for positions at different spacings
- Solution: Voronoi bins — each position's bin extends to midpoint of neighbors on [0,1) circle
- 12 degree-6 positions (maximum resolvable at adaptive spectral resolution)
- Split boundary: separate lower (band bottom) and upper (band top) enrichment
- Enrichment = (observed_fraction / expected_fraction - 1) × 100%
- Band assignment by phi_octave column from extraction (not Hz range)
- Reproducible: `python scripts/voronoi_enrichment_analysis.py --all --summary`

#### 2.4 Consistency Classification
- ✓ = all 9 datasets agree beyond ±5%
- ~ = 8/9 agree or weak consensus
- ✗ = sign conflict between datasets
- Cross-release consistency: HBN R1/R2/R3/R4/R6 compared separately

#### 2.5 Statistical Analysis
- Phase-rotation permutation tests (within each band)
- Bootstrap confidence intervals
- Adult (4 datasets) vs pediatric (5 datasets) comparison: Mann-Whitney U
- Pearson/Spearman profile correlations across bands

### 3. Results

#### 3.1 Dataset Overview
- Total peaks by dataset and band
- Extraction success rates
- Spectral resolution achieved per band

#### 3.2 Three Organizational Regimes
- Overview figure: 5-band × 13-position heatmap with 9-dataset means
- The three shapes are visually obvious and statistically distinct

#### 3.3 Alpha Mountain
- Noble1 +25% (SD=5), attractor +24% (SD=9), boundary -37% (SD=14)
- 10/13 positions consistent across 9 datasets
- The "classic" phi-lattice pattern — boundary depleted, Noble1/attractor enriched
- Monotonic gradient from boundary through noble positions to peak at Noble1
- Cross-dataset Noble1 range: +17% to +31% — remarkably tight
- **This is the real core of the aggregate +39% Noble1 finding from Paper 1**

#### 3.4 Beta-Low U-Shape
- 13/13 consistent — PERFECT unanimity across 9 datasets
- Boundary enrichment: lower +101% (SD=20), upper +74% (SD=25)
- Center depletion: noble_5 -59%, noble_4 -59%, noble_3 -57%
- Noble1 sits at zero-crossing (+2%)
- Inverse nobles enriched: inv_noble_3 +31%, inv_noble_4 +56%, inv_noble_5 +65%, inv_noble_6 +89%
- **Strongest and most consistent signal of any band**
- Table of all 9 datasets showing perfect consistency

#### 3.5 Gamma Ascending Ramp
- inv_noble_3 +27% (SD=7), inv_noble_4 +36% (SD=13), inv_noble_5 +61% (SD=37)
- Noble1 near null (+1%)
- Lower positions depleted (noble_3 -30%, inv_noble_1 -28%)
- 7/13 consistent overall, BUT 13/13 across 5 HBN releases
- Gamma conflicts driven by adult dataset divergence (CHBMP inversion, LEMON boundary discrepancy)
- The +144.8% aggregate Noble1 from Paper 1 was a cross-band density artifact

#### 3.6 Theta Boundary Clustering
- Boundary enriched: lower +47%, upper +38%
- Noble_6 enriched (+34%) — adjacent to boundary
- Rest of octave mixed/depleted
- 7/13 consistent — weakest overall but boundary signal clear
- Consistent with Paper 3's theta f₀ convergence finding

#### 3.7 Beta-High Weak Ascending
- Weakest overall pattern (8/13 consistent)
- Mild ascending: inv_noble_4 +12%, inv_noble_6 +19%, boundary_hi +17%
- Center mildly depleted: attractor -12%, noble_1 -7%
- Transitional between beta-low U-shape and gamma ramp

#### 3.8 Cross-Release Consistency (HBN)
- 5 independent releases, 927 subjects, ages 5-21
- Alpha: 13/13 (Noble5 SD=1.8!)
- Beta-low: 13/13 (inv_noble_1 SD=1.0!)
- Beta-high: 13/13
- Gamma: 13/13
- Theta: 9/13
- **Zero conflicts in alpha, beta-low, beta-high, gamma across 5 releases**
- This establishes per-band enrichment as a developmental invariant

#### 3.9 Adult vs Pediatric Comparison
- Profile correlations: r=0.64 (gamma) to r=0.97 (beta-low)
- Alpha: narrower, taller mountain in children (9/13 positions p<0.05)
  - Boundary depletion stronger in children (-46% vs -26%)
  - Attractor enrichment stronger (+31% vs +16%)
  - Sharper falloff: inv_noble_3 drops to -3% ped vs +20% adult
- Beta-low: same U-shape; adults more extreme (±40 pp)
- Gamma: clean ramp in children (zero boundary conflicts); adults disrupted at boundary/noble_6
- Interpretation: developmental narrowing of alpha bandwidth; less muscle artifact in children

#### 3.10 Null Position Analysis
- Noble5/Noble4 (u≈0.09/0.15) universally depleted (-14% to -59%) — "no-man's land"
- Noble3 (u≈0.24) depleted in 3/5 bands
- These are the positions between boundary and mid-octave — consistently avoided
- The phi-lattice has structural dead zones as well as enrichment zones

#### 3.11 Position Symmetry
- Inverse nobles consistently MORE enriched than their symmetric partners
- Asymmetry is systematic: peaks drift upward within each phi-octave
- Largest in beta-low (+44 to +124 pp difference) and gamma (+29 to +85 pp)

### 4. Discussion

#### 4.1 Resolving the Aggregate Enrichment Values
- Paper 1's boundary -18% = alpha's -37% + beta-low's +101% weighted by peak density
- Paper 1's Noble1 +39% = alpha's +25% inflated by gamma cross-band density effects
- Paper 1's gamma +144.8% = aggregate artifact, not within-band Noble1 dominance
- The aggregate was a blurry photograph of three sharp images

#### 4.2 Fibonacci Coupling as Mechanism
- Every boundary is a Fibonacci sum: f(n+2) = f(n+1) + f(n) (exact, 0% error)
- Boundary enrichment in 4/5 bands = three-wave resonance at coupling gateways
- Beta-low shows strongest coupling (sits between alpha and theta — the two strongest oscillators)
- Alpha resists coupling due to intrinsic thalamo-cortical resonance at ~10 Hz = Noble1
- The phi-lattice prediction of boundary DEPLETION was wrong for most bands
- Correct interpretation: boundaries are Fibonacci coupling GATEWAYS (attractive), not depletion zones (repulsive)
- Anti-mode-locking (Pletzer/Kramer) applies specifically to alpha, not universally

#### 4.3 Why Alpha Is Different
- T-type Ca²⁺ channel resonance at ~100ms → ~10 Hz → Noble1 position
- Strongest intrinsic oscillator overrides Fibonacci coupling tendency
- Alpha boundary depletion = anti-mode-locking as originally predicted
- But this is the exception, not the rule — other bands show coupling, not avoidance

#### 4.4 The Ascending Ramp and Upward Cascade
- Inverse nobles consistently enriched more than regular nobles across bands
- Peaks accumulate toward upper boundary of each phi-octave
- Consistent with nonlinear harmonic generation / spectral energy cascade
- Gamma ramp most pronounced — extends into high-gamma (>52 Hz) territory

#### 4.5 Implications for Band Definitions
- Phi-octave boundaries land at biophysically meaningful frequencies
- 7.60 Hz (f₀): HCN channel resonance, theta/alpha transition
- 12.30 Hz: alpha/beta transition, M-current onset
- 19.90 Hz: KCNQ channel resonance, mu/beta boundary
- 32.19 Hz: beta/gamma transition
- The phi-lattice provides principled band definitions that match biophysical time constants

#### 4.6 Developmental Stability
- Zero conflicts across 5 HBN releases confirms per-band enrichment as species-level architecture
- Alpha narrowing in children consistent with known IAF maturation trajectory
- Gamma cleanup in children consistent with reduced muscle artifact / scalp conductance differences
- Beta-low U-shape is the most developmentally invariant pattern

#### 4.7 Limitations
1. Extraction used f₀=7.83 for band boundaries but enrichment computed at f₀=7.60 — slight mismatch
2. Eyes-closed resting state only (EO comparison in progress)
3. FOOOF-based — IRASA/eBOSC validation needed (Paper 3's highest priority)
4. Cross-base comparison not re-run with Voronoi (relative rankings may differ)
5. Single-channel analysis — GED per-band not performed
6. CHBMP gamma anomaly unexplained (dataset-specific, isolated to gamma)
7. Degree-6 is maximum resolvable — finer lattice structure may exist at higher resolution

#### 4.8 Future Directions
1. EC vs EO comparison with adaptive extraction (in progress)
2. Cross-base structural specificity with Voronoi per-band normalization
3. IRASA/eBOSC alternative aperiodic separation
4. Per-subject per-band enrichment (rather than pooled)
5. High-gamma (52-85 Hz) extension for datasets without notch holes (LEMON, HBN)
6. Source-space analysis: which cortical generators produce which regime?
7. Clinical datasets: do neurological conditions alter band-specific enrichment patterns?

### 5. Conclusions
- Aggregate phi-lattice enrichment conflated three distinct per-band regimes
- Alpha follows the original prediction (anti-mode-locking at Noble1)
- Beta-low reveals Fibonacci coupling gateways (boundary enrichment, 13/13 consistent)
- Gamma shows ascending ramp at inverse nobles (not Noble1 dominance)
- The phi-lattice coordinate system describes ALL three regimes — what differs is which positions are occupied per band
- Fibonacci additive coupling provides mechanistic explanation for boundary enrichment
- The phi-lattice is more structured, not less, when viewed per-band — but the structure is richer than "boundary depleted, Noble1 enriched"

---

## Figures (Planned)

1. **Methods overview** — Adaptive nperseg diagram, Voronoi bin illustration, split boundary schematic
2. **Three-regime heatmap** — 5 bands × 13 positions, 9-dataset means, with band shapes annotated
3. **Alpha mountain** — Enrichment profile with 9-dataset error bars, Noble1 convergence inset
4. **Beta-low U-shape** — The cleanest figure, 9 overlaid dataset profiles showing perfect unanimity
5. **Gamma ramp** — HBN 5-release overlay showing zero-conflict consistency
6. **Cross-release consistency** — SD heatmap showing sub-2% variation across 927 subjects
7. **Adult vs pediatric** — Alpha mountain narrowing, gamma ramp cleanup
8. **Fibonacci coupling schematic** — f(n) = f(n-1) + f(n-2) at boundaries, with enrichment arrows
9. **Position symmetry** — Noble vs inverse noble asymmetry across bands

## Tables (Planned)

1. Dataset properties (9 datasets, demographics, hardware)
2. Adaptive extraction parameters (band / nperseg / freq_res)
3. Complete 5-band × 13-position × 9-dataset enrichment table (the master table)
4. Consistency summary (✓/~/✗ counts per band)
5. Cross-release consistency (HBN R1-R6, SD values)
6. Adult vs pediatric comparison (means, Δ, p-values)
7. Fibonacci coupling scorecard

---

## Target Journal
- **Frontiers in Computational Neuroscience** (companion to Paper 1)
- **NeuroImage** (methodology-focused, LEMON paper may go here)
- **eLife** (if the Fibonacci coupling story is strong enough)
- **PNAS** (if the three-regime discovery + 2,061-subject replication + Fibonacci mechanism is compelling)

## Estimated Length
- ~8,000-10,000 words (shorter than Papers 2-3; the story is cleaner)
- 9 figures, 7 tables
- Data and code: `voronoi_enrichment_analysis.py --all --summary` reproduces all primary results
