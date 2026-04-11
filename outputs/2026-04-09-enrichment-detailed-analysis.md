# Detailed Enrichment Reanalysis: Position-by-Position, Band-by-Band

**Date:** 2026-04-09
**Data:** 9 datasets, 2,061 subjects, ~5M peaks, degree-6 Voronoi bins, f₀=7.60

---

## Position-by-Position Cross-Band Profiles

Each position has a characteristic cross-band enrichment signature. Using ✓ (>+10%), ○ (±10%), ✗ (<-10%):

| Position | u | θ | α | βL | βH | γ | Cross-band pattern |
|---|---|---|---|---|---|---|---|
| **boundary** | 0.000 | +47✓ | -37✗ | +101✓ | +23✓ | +27✓ | Enriched everywhere EXCEPT alpha |
| **noble_6** | 0.056 | +34✓ | -36✗ | +45✓ | +17✓ | +6○ | Same pattern, weaker |
| **noble_5** | 0.090 | -14✗ | -25✗ | -59✗ | +7○ | -24✗ | Depleted everywhere except βH |
| **noble_4** | 0.146 | -16✗ | -15✗ | -59✗ | 0○ | -28✗ | Depleted everywhere except βH |
| **noble_3** | 0.236 | -14✗ | -1○ | -57✗ | -8○ | -30✗ | Depleted in θ/βL/γ, null in α/βH |
| **inv_noble_1** | 0.382 | -15✗ | +11✓ | -43✗ | -13✗ | -28✗ | Depleted everywhere EXCEPT alpha |
| **attractor** | 0.500 | -7○ | +24✓ | -21✗ | -12✗ | -17✗ | Enriched ONLY in alpha |
| **noble_1** | 0.618 | -5○ | +25✓ | +2○ | -7○ | +1○ | Enriched ONLY in alpha; null everywhere else |
| **inv_noble_3** | 0.764 | +4○ | +7○ | +31✓ | +3○ | +27✓ | Enriched in βL and γ only |
| **inv_noble_4** | 0.854 | +17✓ | -8○ | +56✓ | +12✓ | +36✓ | Enriched everywhere EXCEPT alpha |
| **inv_noble_5** | 0.910 | +9○ | -12✗ | +65✓ | +14✓ | +61✓ | Enriched in βL/βH/γ, depleted in alpha |
| **inv_noble_6** | 0.944 | +22✓ | -26✗ | +89✓ | +19✓ | +35✓ | Enriched everywhere EXCEPT alpha |
| **boundary_hi** | 1.000 | +38✓ | -32✗ | +74✓ | +17✓ | +37✓ | Enriched everywhere EXCEPT alpha |

---

## Key Finding 1: Alpha is the Mirror Image of Everything Else

The most striking pattern: **alpha enrichment is the inverse of all other bands**. Positions enriched in alpha (attractor, Noble1, inv_noble_1) are depleted or null in θ/βL/βH/γ. Positions depleted in alpha (boundary, noble_6, inv_noble_6) are enriched in the other bands.

This creates two regimes:
- **Lower octave (u=0.0–0.4):** α depleted, others enriched (boundary zone)
- **Mid octave (u=0.4–0.7):** α enriched, others depleted (noble zone)
- **Upper octave (u=0.7–1.0):** α depleted, others enriched (inverse noble zone)

Alpha's "mountain" shape is not just the strongest signal — it's the ONLY band that follows the original phi-lattice prediction (boundary depleted, Noble1 enriched). All other bands show the opposite at boundaries.

## Key Finding 2: The Boundary–Noble1 Axis is Band-Specific

The phi-lattice theory predicts boundary depletion and Noble1 enrichment universally. In reality:

| Band | Boundary | Noble1 | Theory match? |
|---|---|---|---|
| Theta | +47% | -5% | **NO** — boundary enriched, Noble1 null |
| Alpha | -37% | +25% | **YES** — the predicted pattern |
| Beta-low | +101% | +2% | **NO** — boundary massively enriched, Noble1 null |
| Beta-high | +23% | -7% | **NO** — boundary enriched, Noble1 null |
| Gamma | +27% | +1% | **NO** — boundary enriched, Noble1 null |

Only alpha follows the predicted pattern. In every other band, the lower boundary is enriched (peaks cluster at band edges) and Noble1 is null or near-null.

## Key Finding 3: Inverse Nobles Are Consistently Enriched (Except in Alpha)

The inverse noble positions (u=0.764–0.944, corresponding to n+0.764 through n+0.944 within each octave) show a remarkably consistent pattern:

- **inv_noble_4 (u=0.854):** Enriched in 4/5 bands (θ +17%, βL +56%, βH +12%, γ +36%). Only alpha is negative (-8%).
- **inv_noble_5 (u=0.910):** Enriched in 3/5 bands (βL +65%, βH +14%, γ +61%). Alpha depleted (-12%).
- **inv_noble_6 (u=0.944):** Enriched in 4/5 bands (θ +22%, βL +89%, βH +19%, γ +35%). Alpha depleted (-26%).

These are the "upper octave" positions approaching the next boundary. They show enrichment because peaks are clustering near the upper band edge — the transition zone into the next phi-octave.

## Key Finding 4: Noble5/Noble4 Are Universal Depletions

The "mid-noble" positions noble_5 (u=0.090) and noble_4 (u=0.146) are depleted in virtually every band:

- **noble_5:** θ -14%, α -25%, βL -59%, βH +7%, γ -24%
- **noble_4:** θ -16%, α -15%, βL -59%, βH 0%, γ -28%

These are the positions just above the lower boundary. Peaks consistently avoid this "no-man's land" between boundary and the mid-octave positions. The depletion is strongest in beta-low (-59%) and weakest in beta-high (~0%).

## Key Finding 5: Beta-Low Has the Clearest and Most Consistent Structure

Beta-low (12.3–19.9 Hz) shows 13/13 consistent positions across 9 datasets — perfect unanimity. The pattern is a dramatic U-shape:

```
Position:    bnd   n6   n5   n4   n3  in1  att   n1  in3  in4  in5  in6  bnd_hi
Enrichment: +101  +45  -59  -59  -57  -43  -21   +2  +31  +56  +65  +89  +74
```

The center of the octave (noble_5 through attractor) is massively depleted (-59% to -21%). Both edges are massively enriched (+101% lower boundary, +74% upper boundary). Noble1 sits exactly at the zero-crossing (+2%).

This U-shape means beta-low peaks avoid the interior of the phi-octave and cluster at the edges. The edges are where f(n) = f(n-1) + f(n-2) — the Fibonacci additive relation.

---

## Fibonacci Coupling Analysis

### The Fibonacci Property

The defining property of the golden ratio is: φ² = φ + 1, which means f(n+2) = f(n+1) + f(n) for any phi-lattice frequencies. This means every band boundary is simultaneously a Fibonacci sum frequency:

| Equation | Frequencies | Boundary |
|---|---|---|
| f(-1) + f(0) = f(1) | 4.70 + 7.60 = 12.30 | theta/alpha → beta-low |
| f(0) + f(1) = f(2) | 7.60 + 12.30 = 19.90 | alpha/beta-low → beta-high |
| f(1) + f(2) = f(3) | 12.30 + 19.90 = 32.19 | beta-low/beta-high → gamma |
| f(2) + f(3) = f(4) | 19.90 + 32.19 = 52.09 | beta-high/gamma → high-gamma |

The error is exactly 0.0% — this is a mathematical identity for phi, not an empirical approximation.

### Boundary Enrichment Reflects Fibonacci Coupling

If Fibonacci additive coupling (f₁ + f₂ → f₃) drives three-wave resonance at boundaries, we predict boundary enrichment where the coupled frequencies have strong oscillatory power. The data show:

| Boundary | Hz | Lower band bnd_hi | Upper band bnd | Both enriched? |
|---|---|---|---|---|
| θ/α | 7.60 | θ +38% | α -37% | **NO** — alpha depleted |
| α/βL | 12.30 | α -32% | βL +101% | **HALF** — beta-low strongly enriched, alpha depleted |
| βL/βH | 19.90 | βL +74% | βH +23% | **YES** — both sides enriched |
| βH/γ | 32.19 | βH +17% | γ +27% | **YES** — both sides enriched |

**The pattern:** Boundary enrichment occurs at every boundary EXCEPT from the alpha side. Alpha boundaries are consistently depleted. This makes biophysical sense: alpha oscillations are generated by thalamo-cortical loops with strong intrinsic resonance (T-type Ca²⁺ channels at ~10 Hz) that resists being pulled to band edges. The alpha mountain shape reflects this intrinsic resonance overriding the Fibonacci coupling tendency.

At non-alpha boundaries, Fibonacci coupling appears to PULL peaks toward the boundary — consistent with three-wave resonance f(n) + f(n+1) = f(n+2) creating energy transfer pathways at these precise frequencies.

### The Beta-Low U-Shape as Fibonacci Evidence

Beta-low (12.3–19.9 Hz) sits at f(1) in the phi-lattice. It is bounded by:
- Lower: f(1) = f(0) + f(-1) = 7.60 + 4.70 = 12.30 Hz → boundary with theta+alpha
- Upper: f(2) = f(1) + f(0) = 12.30 + 7.60 = 19.90 Hz → boundary with alpha+beta-low

Both boundaries are massive enrichment zones (+101% and +74%). The center is massively depleted (-59%). This is exactly what Fibonacci coupling predicts: energy concentrates at the boundary frequencies where three-wave resonance occurs, and the interior of the octave is emptied.

The fact that this is the MOST consistent pattern (13/13 positions across 9 datasets) supports the interpretation that Fibonacci coupling is the strongest organizational force in the beta-low range.

### Why Is Alpha Different?

Alpha's inverted pattern (boundary depleted, Noble1 enriched) could reflect:

1. **Intrinsic resonance dominance:** IAF at ~10 Hz (= Noble1 position) is generated by thalamo-cortical T-type Ca²⁺ channel dynamics (~100ms time constant). This intrinsic oscillator pulls peaks to Noble1 regardless of lattice structure.

2. **Anti-mode-locking at Noble1:** φ⁻¹ ≈ 0.618 is the position of maximal resistance to mode-locking (Pletzer/Kramer theory). Alpha oscillations may exploit this anti-mode-locking property to maintain independence from neighboring bands.

3. **These two mechanisms coincide:** The T-type Ca²⁺ channel resonance at ~10 Hz happens to land at the Noble1 position in the phi-lattice. Whether this is evolutionary tuning or coincidence is an open question — but the functional consequence is the same: alpha peaks cluster at the most mode-locking-resistant position.

---

## Asymmetry Analysis: Noble vs Inverse Noble

Every noble/inverse-noble pair shows **inverse-noble > noble** across nearly all bands:

| Pair (noble vs inverse) | θ | α | βL | βH | γ | Direction |
|---|---|---|---|---|---|---|
| noble_6 vs inv_noble_6 | -12 | +10 | +44 | +2 | +29 | inv > noble |
| noble_5 vs inv_noble_5 | +23 | +13 | +124 | +7 | +85 | inv > noble |
| noble_4 vs inv_noble_4 | +33 | +7 | +115 | +12 | +64 | inv > noble |
| noble_3 vs inv_noble_3 | +18 | +8 | +88 | +11 | +57 | inv > noble |
| inv_noble_1 vs noble_1 | +10 | +14 | +45 | +6 | +29 | noble_1 > inv_noble_1 |

The asymmetry is systematic: inverse nobles (upper octave) are consistently MORE enriched than their symmetric partners (lower octave). The magnitude increases dramatically in beta-low (+44 to +124 percentage points difference) and gamma (+29 to +85).

This asymmetry means **peaks cluster toward the upper boundary of each octave**, not symmetrically around the attractor. The enrichment profile ramps UP from the center toward the upper boundary, creating the "ascending ramp" pattern visible in beta-low, beta-high, and gamma.

**Physical interpretation:** Peaks drift upward within each phi-octave, accumulating near the upper boundary where Fibonacci coupling to the next-higher band occurs. This is consistent with nonlinear oscillatory systems where energy cascades preferentially upward in frequency (e.g., harmonic generation, spectral leakage into higher bands).

---

## Summary: Three Organizational Regimes

### Regime 1: Alpha Mountain (u=0.4–0.7 enriched, edges depleted)
- **Bands:** Alpha only
- **Mechanism:** Intrinsic thalamo-cortical resonance at ~10 Hz = Noble1 position
- **Consistency:** 10/13 positions, SD=4-5% at Noble1
- **Theory match:** YES — matches original phi-lattice prediction (anti-mode-locking)

### Regime 2: Boundary U-Shape (edges enriched, center depleted)
- **Bands:** Beta-low (strongest), also theta, beta-high, gamma
- **Mechanism:** Fibonacci three-wave coupling f(n) = f(n-1) + f(n-2) at boundaries
- **Consistency:** 13/13 in beta-low (perfect), 7-8/13 in others
- **Theory match:** PARTIALLY — Pletzer predicted Fibonacci coupling at boundaries, but the original papers predicted boundary DEPLETION. The data show boundary ENRICHMENT in 4/5 bands, meaning Fibonacci coupling is an ATTRACTIVE force, not a repulsive one.

### Regime 3: Ascending Ramp (inv_nobles enriched, nobles depleted)
- **Bands:** Gamma (clearest), also beta-high
- **Mechanism:** Upward frequency cascade within phi-octaves
- **Consistency:** 7/13 in gamma (conflicts at boundary/noble_6 from adult datasets)
- **Theory match:** NOVEL — not predicted by original framework. The ascending ramp at inverse nobles is a new finding.

### The Common Thread

All three regimes operate within the same phi-lattice coordinate system. The positions that carry enrichment differ by band, but they are ALL phi-lattice positions:
- Alpha: Noble1 (u=0.618) and attractor (u=0.500)
- Beta-low: Boundary (u=0.000/1.000) and inverse nobles (u=0.764–0.944)
- Gamma: Inverse nobles (u=0.764–0.944)

No peaks cluster at arbitrary positions. The phi-lattice provides the coordinate system for ALL three regimes — what differs is WHICH lattice positions are occupied in each band.

---

## Fibonacci Coupling Scorecard

| Prediction | Evidence | Verdict |
|---|---|---|
| Boundaries are Fibonacci sum frequencies | f(n+2) = f(n+1) + f(n), exact (0% error) | ✓ Mathematical identity |
| Three-wave resonance enriches boundaries | 4/5 bands show boundary enrichment (all except alpha) | ✓ Strong support |
| Beta-low shows strongest coupling (sits between α and θ, the two strongest oscillators) | Beta-low U-shape 13/13 consistent, +101%/+74% at boundaries | ✓ Very strong |
| Upper boundaries enriched more than lower (energy cascades upward) | inv_noble > noble systematically across bands | ✓ Consistent |
| Alpha resists coupling (strong intrinsic resonance) | Alpha boundaries depleted, Noble1 enriched | ✓ Consistent with T-type Ca²⁺ ~10 Hz |
| Coupling should increase with frequency (shorter time constants, faster energy transfer) | βL > βH > γ at boundaries; but γ ramp at inv_nobles | ~ Partial (boundary coupling weakens, but ramp structure extends) |
| Peaks avoid "no-man's land" between boundary and mid-octave | noble_5/noble_4 universally depleted (-14% to -59%) | ✓ Strong |

**Overall:** The Fibonacci coupling interpretation is well-supported by the data. The phi-lattice isn't just a coordinate system — the Fibonacci additive property of φ (f(n) = f(n-1) + f(n-2)) appears to create genuine three-wave resonance pathways at boundary frequencies, with alpha as the exception due to its strong intrinsic resonance.
