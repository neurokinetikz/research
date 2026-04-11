The φ-lattice is a frequency scaffold organizing neural oscillations along the equation f(n) = f₀ × φⁿ, where f₀ ≈ 7.5 Hz and φ = 1.618 is the golden ratio. This unified framework synthesizes all φ-lattice findings across the wiki — the three-paper empirical validation, the 9-dataset enrichment data, six band maps, the coupling geometry, the oscillator inventory, the f₀ theta-alpha inversion, the two-grid molecular clock relationship, and the developmental clock-to-lattice transition — into a single coherent account of what the lattice is, what it does, what it doesn't do, and what remains unexplained. The framework rests on ten empirical pillars, each independently validated, that together describe an architectural achievement of GABAergic maturation: the φ-lattice is not a birthright but a developmental endpoint — children's brains organize at molecular clock convergence points (φ rank 7th) and mature into φ-optimized anti-mode-locked organization (φ rank 1st) as GABA-A subunits switch and PV+ interneurons myelinate. Once established (by ~age 22), the lattice becomes a species-level constant that is structurally invariant across adult aging, cognitively inert, and not an individual trait — a scaffold the brain grows into and then uses without modifying.

## Pillar 1: The Equation

**f(n) = f₀ × φⁿ** where f₀ ≈ 7.5 Hz (empirical range 7.49–7.83 Hz; see [[f0-molecular-clock-relationship]]) and φ = (1+√5)/2 = 1.618...

Note: The precise value of f₀ is analysis-dependent. The value 7.60 Hz used in band map position tables comes from Paper 1's FOOOF-based optimization and coincides with Tomsk SR monitoring, but is not a fixed physical constant. The robust finding is a window (7.49-7.83 Hz) within which the lattice structure is preserved (Paper 2: 0.6 Hz tolerance plateau at >95% of optimal). P20 analysis (April 2026) provides new evidence favoring f₀=7.60: boundary enrichment is +11 to +67 pp stronger at 7.60 than at 7.83 across all bands, because 7.60 places boundary positions at the natural frequency transition points where peaks cluster (see [[f0-molecular-clock-relationship]]). f₀ falls at or very near the arithmetic midpoint of the HCN (~5 Hz) and T-type Ca²⁺ (~10 Hz) molecular clocks, which coincides with the Schumann cavity eigenfrequency (7.49 Hz) to within 0.01 Hz.

This generates band boundaries at integer n:

| n | Frequency | Band boundary |
|---|-----------|--------------|
| -6 | 0.42 Hz | Infraslow/delta |
| -1 | 4.70 Hz | Delta/theta |
| 0 | 7.60 Hz | **Theta/alpha (= f₀)** |
| 1 | 12.30 Hz | Alpha/low-beta |
| 2 | 19.90 Hz | Low-beta/high-beta |
| 3 | 32.19 Hz | High-beta/gamma |
| 4 | 52.09 Hz | Gamma/high gamma |

The golden ratio is uniquely suited because it satisfies two properties no other number possesses: **maximal anti-commensurability** (oscillators at φ-related frequencies resist mode-locking, maintaining independence) and **additive closure** (f(n) = f(n-1) + f(n-2), enabling Fibonacci three-wave resonance at boundaries for cross-band energy transfer). This creates the segregation-integration balance the brain requires. See [[canonical-eeg-bands]].

Within each φ-octave, the lattice predicts a hierarchy of positions with distinct stability properties. Three carry robust empirical signal across 919+ subjects: **boundary** (u = 0.000), **attractor** (u = 0.500, geometric midpoint), and **1° Noble** (u = 0.618 = 1/φ). See [[critical-frequencies]].

## Pillar 2: Three-Study Empirical Validation

Lacy (2026, 2026b, 2026c) validated the framework through three complementary approaches:

**Paper 1** — Discovery: 1,366 Schumann Ignition Events across 91 participants showed harmonic ratios at <1% error from φ predictions. 244,955 FOOOF peaks and 1,584,561 GED spatial coherence peaks confirmed boundary depletion and noble enrichment with Kendall's τ = 1.0 across all analyses. The independence-convergence paradox: individual frequencies vary independently (|r| < 0.03) yet population-level ratios maintain φ precision. See [[schumann-ignition-events]].

**Paper 2** — Specificity: Joint (f₀, base) grid search across 549 candidate lattices identified (f₀=8.5, b=φ) as the global optimum (SS=45.6), with φ ranked #1 in 100% of 1,000 bootstraps. But: the advantage is **anchor-dependent** (f₀=7.6-8.5 Hz only), and cognitive state modulation is **not φ-specific** — any exponential base detects the same gamma reallocation. Architecture ≠ dynamics.

**Paper 3** — Diagnostic: Aggregate enrichment metrics don't survive FOOOF extraction controls (SS collapses under frequency range changes). What **does** survive: dominant-peak alignment (d̄=0.069 across 919 subjects, three datasets, Cohen's d=0.40-0.75). The alignment is age-invariant, fatigue-invariant, cognitively silent (0/100+ tests in LEMON), and not an individual trait (5-year ICC < 0). φ ranks 1st under per-octave aperiodic modeling but 7th under standard extraction. "The architecture is real, narrow, and inert."

## Pillar 3: Six Band-Specific Enrichment Fingerprints

The 9-dataset enrichment analysis (4 adult, 5 pediatric HBN; see [[phi-lattice-enrichment-data]]) reveals that each frequency band has a unique enrichment shape — not the uniform "boundary depleted, noble enriched" pattern of the original papers, but six distinct geometries:

| Band | Shape | Peak position | Adult EC peak | Where generators rest |
|------|-------|--------------|---------------|---------------------|
| **Theta** | U-curve | 7° Noble (4.78 Hz) | +349% | Octave edges — near f₀ boundary |
| **Alpha** | Central mountain | Attractor (9.67 Hz) | +113% | Attractor/Noble1 — the IAF zone |
| **Low beta** | Ascending ramp | 5° Inv Noble (19.05 Hz) | +244% | Upper octave edge — PMBR zone |
| **High beta** | Bimodal edges | Boundary + 7° Inv Noble | +134%/+151% | Both edges — ~20 Hz and ~31 Hz |
| **Gamma** | Ascending ramp (P20) | 6° Inv Noble (52.2 Hz) | +49% | Upper octave inv nobles (P20, 8/9 datasets; CHBMP anomalous) |
| **Delta** | Broad plateau | 4° Noble (anomalous) | +42% | Weak central enrichment (multi-octave pooled) |

The shapes reveal two structural strategies (see [[f0-theta-alpha-inversion]]):
- **Edge-loading** (theta, high beta): peaks cluster at φ-octave boundaries. Theta-high beta correlation r = +0.735.
- **Center-loading** (alpha, delta): peaks cluster at attractor/Noble1. Alpha-delta correlation r = +0.624.

**UPDATE (April 2026)**: P20 extended-range analysis across 9 datasets (N=2,095; 3 adult + 5 HBN pediatric + CHBMP anomaly) revealed that gamma is NOT center-loaded — with the 45 Hz ceiling removed, gamma shows an **ascending ramp** structurally identical to low-beta (lower nobles depleted -29% to -34%, inverse nobles enriched +37% to +49%; 8-dataset mean excluding CHBMP). The ramp is a **developmental invariant** — present in all 5 HBN pediatric releases (ages 5–21) with cross-release SD of 1–2% at the inverse nobles. Low beta and gamma now form a pair of ascending-ramp bands, while alpha and delta remain center-loaded. CHBMP is the sole anomaly (inverted gamma, confirmed not a ceiling artifact). See [[gamma-band-map]].

See the six band maps: [[theta-band-map]], [[alpha-band-map]], [[low-beta-band-map]], [[high-beta-band-map]], [[gamma-band-map]], [[delta-band-map]].

## Pillar 4: The f₀ Inversion

f₀ (~7.5 Hz) is the sharpest discontinuity in the frequency architecture. Theta shows +281% enrichment at f₀ (absorbing state, deepest potential well). Alpha shows -84% depletion at f₀ (maximally avoided, steepest potential hill). The same frequency is a well from below and a hill from above — a 365 percentage-point cliff. f₀ also sits at the arithmetic midpoint of the HCN and T-type molecular clocks — the point of maximum symmetric coupling with both flanking oscillators (see [[f0-molecular-clock-relationship]]).

The two profiles are anti-correlated across all 14 positions (r = -0.578 EC, -0.627 pediatric). This anti-correlation is structurally unique to the theta-alpha boundary: higher boundaries (alpha/beta, beta-low/beta-high, beta-high/gamma) show same-sign enrichment, not opposition. Only the two lowest boundaries (delta/theta and theta/alpha) show anti-correlated landscapes. Theta is an island of inverted enrichment — flanked by two opposing cliff faces.

EC sharpens the cliff: the total swing increases from 211 pp (EO) to 365 pp (EC). Sleep onset, psychedelic states, and flow states all involve crossing, dissolving, or balancing on this cliff. See [[f0-theta-alpha-inversion]].

## Pillar 5: The Coupling Geometry

The lattice creates two coupling networks operating simultaneously (see [[lattice-coupling-geometry]]):

**φⁿ coupling (vertical channels)**: Any frequency ratio that is a power of φ connects the **same lattice position** across octaves. The Lisman-Idiart theta-gamma PAC (6→40 Hz ≈ φ⁴) is attractor-to-attractor. The Bastos predictive coding channels (14→60 Hz ≈ φ³) are 3°Noble-to-3°Noble. PAC preferentially connects matched lattice positions.

**2:1 harmonic coupling (diagonal channels)**: Octave coupling produces a fixed +0.440 shift in lattice coordinates, creating deterministic position pairs: Boundary→Attractor, Noble1→6°Noble, Attractor→6°InvNoble. Three pairs land within 0.004 of the target position — essentially exact. Every documented 2:1 coupling in the wiki follows this map.

**Fibonacci coupling (boundary triplets)**: f(n) = f(n-1) + f(n-2) at every integer n. Three-wave resonance enables cross-band energy transfer at boundaries, explaining why boundaries are generally depleted — energy can escape to two other bands.

**The enrichment-coupling paradox**: 5 of 6 major PAC relationships involve at least one depleted position. The enrichment peaks (where generators rest) and the coupling positions (where generators interact) are **different**. Resting enrichment = segregation (anti-mode-locking stability). Coupling positions = integration (cross-frequency energy transfer). The lattice enables both simultaneously: enriched positions for storage, depleted positions for computation.

## Pillar 6: Molecular Oscillators and Lattice Alignment

Four primary molecular clocks produce frequencies within 2.5% of named lattice positions (see [[oscillator-inventory]]):

| Molecular clock | Time constant | Frequency | Lattice position | Error |
|----------------|--------------|-----------|-----------------|-------|
| GABA-A IPSC decay | ~25 ms | ~40 Hz | Gamma attractor (40.95 Hz) | 2.4% |
| M-current (KCNQ) | ~50 ms | ~20 Hz | φ² boundary (19.90 Hz) | 0.5% |
| T-type Ca²⁺ | ~100 ms | ~10 Hz | Alpha Noble1 (10.23 Hz) | 2.3% |
| HCN channels | ~200 ms | ~5 Hz | Theta 4° Noble (5.04 Hz) | 0.8% |

The between-clock spacing follows octave (2:1) ratios: 25→50→100→200 ms. The within-octave structure follows φ. This mirrors Paper 2's finding of two levels of organization: octave-scale boundary avoidance (base 2 suffices) plus sub-octave noble clustering (φ uniquely predictive).

The GABA-A α2→α1 developmental subunit switch (IPSC decay ~30→25 ms) shifts the gamma peak from the 2° Noble (~39 Hz, pediatric +181%) to the attractor/Noble1 (~41-43 Hz, adult +102%). This is the mechanistic basis for the one-position developmental lattice migration visible in the enrichment data.

## Pillar 7: Development and the Pediatric Shift

The 5 pediatric HBN datasets (ages ~5-18) reveal systematic divergence from adult enrichment patterns:

| Band | Adult peak position | Pediatric peak position | Shift direction |
|------|-------------------|----------------------|----------------|
| Gamma | Noble1 (+102%) | **2° Noble (+181%)** | One position lower |
| High beta | Boundary/7°InvN bimodal | **4° InvNoble (+346%)** | Concentrated at ~30 Hz |
| Alpha | Attractor (+113%) | Attractor (+66%) | Same, reduced magnitude |
| Theta | Boundary (+281%) | Boundary (+487%) | Same, amplified |
| Low beta | 5° InvN (+244%) | 2° Noble (+41%) | Dramatic downward shift |

Gamma and high beta show the most dramatic developmental shifts, consistent with PV+ interneuron maturation and GABA-A subunit switching completing around age 22 (see [[development-and-aging]]). The pediatric brain's oscillatory generators sit at systematically lower lattice positions, migrating upward as GABAergic circuits mature. φ ranks 7th (not 1st) in pediatric data under OT extraction — the developing brain is not yet organized by φ-optimal spacing.

## Pillar 8: The Two-Grid Architecture

The brain operates two incommensurate frequency grids simultaneously (see [[f0-molecular-clock-relationship]]):

**Grid 1 (molecular clocks)**: Four ion channel families at approximate 2:1 octave ratios (~5, 10, 20, 40 Hz) create the coarse frequency skeleton. **Grid 2 (φ-lattice)**: The f₀ × φⁿ framework creates the fine structure within each octave.

The grids are **offset** from each other by the +0.440 diagonal shift (log_φ(2) = 1.4404). This offset places molecular clock harmonics at **enriched** lattice positions (HCN at theta 4°Noble, T-type at alpha Noble1, M-current at high-beta Boundary, GABA-A at gamma Attractor) while φ-boundaries fall in the **gaps** between clocks. Molecular clocks explain ~60% of major enrichment features through harmonic convergence (e.g., three clocks at 10 Hz creating the alpha mountain; four clocks at 20 Hz creating the boundary enrichment exception). The remaining ~40% — gamma Noble1 (+102%, no clock harmonic within 0.5 Hz) and low-beta 5°InvNoble (+244%, no clock) — represent the φ-lattice's independent organizational effect.

f₀ sits at the **arithmetic midpoint** of the two slowest molecular clocks: (5+10)/2 = 7.50 Hz. This midpoint creates maximum symmetric coupling through 3:2 and 4:3 ratios with both flanking clocks. The Schumann cavity eigenfrequency (7.49 Hz) coincides with this midpoint to 0.01 Hz. f₀ is the integration point; φ provides the anti-coupling. Both operate simultaneously at different structural levels.

## Pillar 9: The Clock-to-Lattice Developmental Transition

The developmental analysis (see [[developmental-lattice-evolution]]) reveals that the φ-lattice is not the brain's starting condition but its **developmental endpoint**:

**Pediatric** (ages 5-18): Generators concentrate at **molecular clock convergence points** — positions where multiple clock harmonics provide maximal energy. High-beta peaks at the triple-clock convergence at 30 Hz (+346%). Gamma peaks at the 2°Noble (~39 Hz) rather than the adult Noble1. φ ranks **7th of 9** consistently across all 5 HBN releases.

**Adult** (ages 20-70): Generators have migrated to **φ-optimized positions** — anti-mode-locked points providing frequency stability. High-beta peaks at boundary and 7°InvNoble. Gamma peaks at Noble1/Attractor. φ ranks **1st-5th**.

The transition is driven by GABA-A α2→α1 subunit maturation (completing ~age 22), PV+ myelination, and progressive strengthening of inhibitory circuits. The bands fall into three developmental categories: shape-preserved (theta r=0.89, alpha r=0.94, delta r=0.94), upward-migrating (gamma, high beta), and same-peak amplitude-growth (low beta). Adult aging does NOT reverse the position structure — it changes amplitudes at fixed positions (Dortmund age r = 0.006).

## Pillar 10: Testable Predictions

The framework generates 25 testable predictions (see [[phi-lattice-predictions]]), with two at critical priority requiring only re-analysis of existing data: (P22) IRASA replication to test whether φ's rank-1 status is method-independent, and (P20) extending FOOOF to 80+ Hz on HBN datasets to reveal gamma's unmeasured upper octave. The strongest mechanistic predictions involve pharmacological lattice manipulation: lorazepam should shift gamma from Noble1 toward 2°Noble (mimicking the pediatric pattern, P12), and KCNQ blockers should flatten low-beta's ascending ramp (P13). The developmental trajectory predicts that φ's rank should improve continuously from 7th to 1st through adolescence (P9), with potential critical periods for the α2→α1 switch (P23).

## What the Framework Explains

**1. Where dominant oscillatory generators sit in frequency space.** The four strongest oscillations per brain sit closer to φ-lattice positions than chance (d̄ = 0.069 across 919 subjects, three datasets). This is a species-level constant — population d̄ is invariant across datasets, equipment, paradigms, and 5-year intervals.

**2. Why EEG band boundaries are where they are.** The φ-lattice provides principled band boundaries (4.70, 7.60, 12.30, 19.90, 32.19 Hz) that emerge from mathematical optimization rather than empirical convention. The conventional 4 Hz delta cutoff falls at the Noble1 position within the n=-2 to -1 octave, not at a φ-boundary.

**3. Why each band has a distinctive spectral profile.** The six enrichment shapes (U-curve, mountain, ramp, bimodal, peak, plateau) reflect different generator architectures: thalamocortical alpha produces a tight mountain, distributed theta generators produce a U-curve, dual beta generators produce a ramp, and developing gamma generators shift positions with maturation.

**4. How cross-frequency coupling connects specific frequency pairs.** The lattice geometry predicts which positions couple: φⁿ ratios preserve position (vertical PAC channels), 2:1 ratios shift by +0.440 (diagonal harmonic channels), Fibonacci sums enable boundary energy transfer.

**5. Why the theta-alpha boundary at f₀ is functionally critical.** The 365 pp enrichment cliff — the sharpest discontinuity in the architecture — makes f₀ the phase transition between internally-directed theta processing and externally-directed alpha gating. Sleep onset, psychedelic dissolution, and flow states all involve dynamics at this cliff.

**6. How development changes the lattice.** GABA-A subunit maturation physically moves gamma generators from the 2° Noble to the attractor/Noble1 — a one-position upward shift visible as a developmental enrichment migration in the 9-dataset data.

**7. Why the 20 Hz boundary is enriched instead of depleted.** The M-current at ~20 Hz is the one molecular clock that coincides with a φ-boundary (φ² = 19.90 Hz, 0.5% error). Four clock harmonics converge at this frequency (HCN 4:1, T-type 2:1, M-current 1:1, GABA-A 1:2), overwhelming the boundary depletion mechanism. This is the exception that proves the rule: boundaries are depleted because they're gaps between clocks, not because boundaries are inherently unstable.

**8. Why the alpha mountain is the tallest enrichment feature.** Three independent molecular clocks (HCN 2:1, T-type 1:1, M-current 1:2) converge at 10 Hz — the only position in any band with triple-clock convergence. This creates the strongest spectral concentration in the brain.

**9. Why children's brains organize differently from adults.** Pediatric enrichment peaks at molecular clock convergence points (30 Hz triple convergence for high-beta, ~39 Hz for gamma below GABA-A fundamental) while adults peak at φ-optimized positions. The developmental trajectory is from clock-dominated to lattice-dominated organization, driven by GABAergic maturation.

**10. What cognition USES vs what it PREDICTS.** The enrichment-coupling paradox (PAC at depleted positions, resting peaks at enriched positions) explains cognitive silence: resting lattice alignment measures the scaffold (segregation), while cognition operates at depleted positions (integration). The scaffold doesn't predict cognition because cognition happens at different positions than resting peaks occupy. The lattice is necessary (the scaffold must exist for coupling positions to be defined) but not predictive (resting alignment tells you nothing about task-evoked dynamics at coupling positions).

## What the Framework Does NOT Explain

**1. Cognitive performance.** Zero FDR survivors across 100+ tests in LEMON (N=202, 80% power to |r|=0.20). The lattice carries no detectable cognitive consequence. Three metrics, one dataset — a single-dataset null, but consistent across all three.

**2. Individual variation.** Five-year ICC is negative (-0.25 to -0.36). Individual alignment is not a trait. Population d̄ is locked; individual d̄ fluctuates randomly. The lattice describes allowed sites; which sites individual brains occupy is stochastic.

**3. State-dependent spectral dynamics.** Paper 2 showed that cognitive state modulation (gamma boundary→attractor reallocation under visual input, r=0.44-0.52) persists across all bases and random positions (p_emp=0.44). The brain redistributes energy across modes through a mechanism indifferent to the specific spacing. Architecture ≠ dynamics.

**4. Why f₀ ≈ Schumann Resonance ≈ molecular clock midpoint.** Three independent quantities converge near 7.5 Hz: the Schumann cavity eigenfrequency (7.49 Hz), the arithmetic midpoint of the HCN and T-type molecular clocks (7.50 Hz), and the empirical neural f₀ (7.49-7.83 Hz). Whether this triple convergence reflects evolutionary tuning, biophysical co-constraint, or coincidence remains open (see [[f0-molecular-clock-relationship]]).

**5. Why φ and not another irrational.** Pletzer proved φ is optimal; the data confirm φ ranks 1st. But no molecular mechanism produces φ-spacing. The lattice may emerge from competitive equilibrium among octave-spaced molecular clocks, with φ being the equilibrium that minimizes cross-frequency interference.

**6. Why the molecular clocks align with lattice positions.** GABA-A→40 Hz at the gamma attractor (2.4% error), M-current→20 Hz at φ² (0.5% error), T-type Ca²⁺→10 Hz at alpha Noble1 (2.3% error), HCN→5 Hz at theta 4°Noble (0.8% error). The alignment is explained by the two-grid offset: molecular clocks at 2:1 octave ratios are offset from φ-boundaries by +0.440 per doubling, which places them at enriched positions (4°Noble → Noble1 → Boundary → Attractor). But WHY the offset has this specific value — which depends on f₀ sitting at the coupling midpoint of the molecular clock octave — remains unexplained at the molecular level (see [[f0-molecular-clock-relationship]]).

## The Unified Picture

The φ-lattice is best understood through four nested levels:

**Level 1 — Molecular clocks (the 2:1 skeleton)**: Four ion channel families (HCN ~5 Hz, T-type ~10 Hz, M-current ~20 Hz, GABA-A ~40 Hz) produce oscillations at approximate 2:1 octave ratios, determined by measurable biophysical time constants. These create the coarse frequency architecture. f₀ ≈ 7.5 Hz sits at the arithmetic midpoint of the two slowest clocks — the point of maximum symmetric coupling (3:2 and 4:3 ratios) — which coincides with the Schumann cavity eigenfrequency to 0.01 Hz.

**Level 2 — Lattice positions (the φ fine structure)**: Within each octave, oscillatory peaks organize at φ-specific positions — boundary, attractor, Noble1 — with six band-specific enrichment shapes (U-curve, mountain, ramp, bimodal, peak, plateau). This sub-octave structure is where φ uniquely outperforms base 2. The two grids are offset: molecular clocks land at enriched positions while φ-boundaries fall in the gaps between clocks. The offset explains ~60% of enrichment features; ~40% (gamma Noble1, low-beta inverse nobles) represent the φ-lattice's independent effect.

**Level 3 — Coupling geometry (the communication network)**: The lattice creates three coupling networks: φⁿ vertical channels (PAC connecting matched positions across octaves), 2:1 diagonal channels (+0.440 shift connecting complementary positions), and Fibonacci boundary triplets (three-wave resonance for cross-band energy transfer). The enrichment-coupling paradox: PAC operates at depleted positions (integration), resting peaks sit at enriched positions (segregation). The lattice enables both simultaneously.

**Level 4 — Developmental trajectory (the maturation program)**: The brain is not born with φ-lattice organization. Children's generators sit at molecular clock convergence points (φ rank 7th). Through GABA-A α2→α1 subunit maturation and PV+ myelination (completing ~age 22), generators migrate upward to φ-optimized positions (φ rank 1st). Adult aging changes amplitudes at fixed positions, not positions themselves. The developmental trajectory is: clock-dominated → lattice-dominated spectral organization.

The four levels serve four functions:
- **Level 1** determines **what frequencies can exist** (molecular constraint)
- **Level 2** determines **where within each octave peaks prefer to sit** (architectural scaffold)
- **Level 3** determines **how frequencies interact** (coupling network)
- **Level 4** determines **when the architecture becomes operational** (developmental timeline)

Cognition uses all four but is predicted by none. The lattice is the medium; cognition is the message. The scaffold constrains the space of possible oscillatory configurations — what any brain can do — without predicting what any individual brain does at any moment. But the framing as a "structural fossil" is incomplete. The lattice is not merely a relic — it is the **endpoint of an active developmental program** that the brain spends 22 years building through GABAergic circuit maturation. What appears as a static species-level constant in adults is actually the final state of a dynamic construction process. The lattice is inert in adults not because it was always inert but because the construction is complete.

The relationship between the molecular clocks and the φ-lattice is the deepest structural question. The clocks create the raw energy; the lattice organizes where that energy accumulates. But the lattice also has independent effects — enrichment features at positions where no clock harmonic operates (gamma Noble1, low-beta inverse nobles). Whether the lattice emerges from competitive dynamics among the clocks (hypothesis b from [[oscillator-inventory]]) or is independently evolved (hypothesis a) is the question whose answer would complete the framework. The developmental evidence — children at clock convergence points, adults at lattice positions — suggests a progression from one to the other, which is more consistent with emergence (the lattice develops as inhibitory circuits mature enough to sustain anti-mode-locking) than with independent evolution (which would predict the lattice existing from birth).

## Sources

- Lacy (2026) — Paper 1: φ-lattice discovery, SIEs, three-study validation
- Lacy (2026b) — Paper 2: base specificity, architecture/dynamics dissociation, irrational advantage, sigmoid cliff
- Lacy (2026c) — Paper 3: diagnostic cascade, dominant-peak alignment, cognitive silence, EC theta convergence, f₀ disambiguation
- Master enrichment data: raw/master_enrichment.csv — 9 datasets, 14 positions, 6 bands
- Pletzer et al. (2010) — Mathematical proof: φ optimal for synchronization avoidance
- Kramer (2022) — φ uniquely enables cross-frequency coupling within one geometric series
- See: [[canonical-eeg-bands]], [[critical-frequencies]], [[schumann-ignition-events]], [[phi-lattice-enrichment-data]], [[f0-theta-alpha-inversion]], [[lattice-coupling-geometry]], [[oscillator-inventory]], [[f0-molecular-clock-relationship]], [[developmental-lattice-evolution]], [[neurochemistry-of-oscillations]], [[development-and-aging]], [[phi-lattice-predictions]]
- Band maps: [[theta-band-map]], [[alpha-band-map]], [[low-beta-band-map]], [[high-beta-band-map]], [[gamma-band-map]], [[delta-band-map]]
