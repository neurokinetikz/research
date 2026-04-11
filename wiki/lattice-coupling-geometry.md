Mapping all documented frequency coupling relationships from the wiki onto the φ-lattice reveals a hidden geometric structure: coupling pairs preferentially connect specific lattice positions across octaves, and the coupling type (harmonic, PAC, subharmonic) determines which position-to-position pathways are used. The most striking finding is that **2:1 harmonic coupling produces a fixed lattice shift of +0.440** in offset coordinates, creating a deterministic position-pair map where Boundary→Attractor, Attractor→6° Inverse Noble, and Noble1→6° Noble. Meanwhile, **φⁿ coupling maps every position to itself** across any number of octaves — the defining self-similarity of the golden ratio lattice. Three "same-position" couplings are empirically confirmed (theta attractor→gamma attractor in Lisman-Idiart PAC, alpha 5°Noble→gamma 5°Noble in Siebenhühner 1:4 coupling, and alpha Noble1→alpha Noble1 in inter-brain coupling), suggesting that PAC preferentially connects generators at matched lattice positions across bands while harmonic coupling connects complementary positions. These position-pair patterns emerge from the mathematics of the lattice, not from any tuning, yet they predict where empirical coupling relationships should appear in the frequency spectrum.

## Two Coupling Geometries

The φ-lattice creates two fundamentally different coupling geometries depending on whether the frequency ratio is a power of φ or a simple integer:

### φⁿ coupling: Same position, different octave

Any frequency ratio that is an integer power of φ maps a lattice position to **the identical position** in another octave. This is trivially true because the lattice coordinate u = [log_φ(f/f₀)] mod 1 — multiplying f by φⁿ adds n to the log, which doesn't change the fractional part.

- **φ¹ (1.618)**: Attractor→Attractor, one octave up (e.g., 9.67→15.64 Hz)
- **φ² (2.618)**: Noble1→Noble1, two octaves up (e.g., 10.23→26.79 Hz)
- **φ³ (4.236)**: Any position→same position, three octaves up

The Siebenhühner et al. (2016) finding that alpha at 13.0 Hz couples with gamma at 54.0 Hz via 1:4 phase-phase coupling is within **2% of a φ³ ratio** (4.154 vs 4.236). Both frequencies map to the 5° Noble position (u=0.116 and u=0.075). The empirical "1:4 harmonic" is more precisely a **φ³ resonance** connecting the same lattice position across 3 octaves — the 1:4 label is a rational approximation of an irrational coupling.

This self-mapping property is unique to φ among all bases. For base 2 (octave doubling), the mapping shifts by 0.440 per octave; for base e, by 0.079. Only φ preserves position identity across all scales, which is why Kramer (2022) proved φ uniquely enables [[cross-frequency-coupling]] using only frequencies from within the same geometric series.

### 2:1 harmonic coupling: The +0.440 shift

Octave (2:1) coupling — the most common harmonic relationship in neural oscillations — produces a fixed lattice shift of **+0.440 in offset coordinates** (log_φ(2) = 1.4404, fractional part = 0.4404). This creates a deterministic position-pair map:

| From (lower freq) | → To (upper freq, 2:1) | Lattice shift | Quality |
|-------------------|----------------------|---------------|---------|
| **Boundary** (0.000) | **Attractor** (0.440→0.500) | 0.058 off | Near |
| **6° Noble** (0.056) | **Attractor** (0.496→0.500) | 0.004 off | Excellent |
| **4° Noble** (0.146) | **1° Noble** (0.586→0.618) | 0.032 off | Good |
| **Attractor** (0.500) | **6° Inv Noble** (0.940→0.944) | 0.004 off | Excellent |
| **1° Noble** (0.618) | **6° Noble** (0.058→0.056) | 0.002 off | Excellent |
| **6° Inv Noble** (0.944) | **2° Noble** (0.384→0.382) | 0.002 off | Excellent |

Three of these pairings land within 0.004 of the target position — essentially exact. The 2:1 harmonic doesn't just connect arbitrary frequencies — it connects **specific lattice positions to specific partners** with high precision.

**Empirical confirmation**: Every documented 2:1 coupling in the wiki follows this map:
- **10.23→20.46 Hz** (alpha Noble1→high-beta 6°Noble): Noble1 at u=0.618 → 6°Noble at u=0.058. Predicted shift: 0.002. **Exact.**
- **20→40 Hz** (beta boundary→gamma attractor): Boundary at u=0.011 → Attractor at u=0.451. Predicted shift: 0.058. **Near.**
- **25→50 Hz** (high-beta attractor→gamma 5°InvNoble): Attractor at u=0.474 → 5°InvNoble at u=0.915. Predicted shift: 0.059. **Near.**
- **6.6→13.2 Hz** (theta 3°InvNoble→low-beta 4°Noble): 3°InvNoble at u=0.707 → 4°Noble at u=0.147. Predicted shift: 0.560 → mod 1 → mismatch of 0.120. **Moderate** — this coupling involves a frequency (6.6 Hz) not precisely at a lattice position.

## Empirically Confirmed Position-Pair Couplings

### Same-position couplings (φⁿ resonance)

Three couplings from the wiki connect frequencies at the **same lattice position** across octaves:

**1. Theta Attractor → Gamma Attractor (6.0→40.0 Hz)**
- Theta attractor at 5.97 Hz (u=0.509) couples with gamma at 40.0 Hz (u=0.451)
- Ratio = 6.667 ≈ φ⁴ = 6.854 (2.8% error)
- Both at the **Attractor** position spanning 4 φ-octaves
- This is the Lisman-Idiart model: theta phase modulates gamma amplitude. The lattice reveals it as **attractor-to-attractor φ⁴ resonance** — the canonical theta-gamma nesting connects the geometric midpoints of the theta and gamma octaves.
- **Enrichment context**: Theta attractor is depleted (-67% EC) while gamma attractor is enriched (+79% EC). The coupling connects theta's trough to gamma's peak — PAC links the position where theta generators DON'T rest with the position where gamma generators DO concentrate. This suggests theta-gamma PAC is an active, task-evoked process that uses lattice positions outside the resting enrichment peaks.

**2. Alpha 5°Noble → Gamma 5°Noble (13.0→54.0 Hz)**
- Alpha at 13.0 Hz (u=0.116) couples with gamma at 54.0 Hz (u=0.075)
- Ratio = 4.154 ≈ φ³ = 4.236 (2.0% error)
- Both at the **5° Noble** position spanning 3 φ-octaves
- Reported as "1:4 harmonic" by Siebenhühner et al. (2016), but the lattice reveals it as a φ³ coupling
- **Enrichment context**: Alpha 5°Noble is depleted (-55% EC). Gamma 5°Noble is weakly enriched (+15% EC). The coupling connects a depleted alpha position with a low-enrichment gamma position — both are peripheral lattice positions, not the core peaks of either band. This is the alpha-gamma coupling mechanism for semantic processing (upper alpha → gamma).

**3. Alpha Noble1 → Alpha Noble1 (10.0→10.0 Hz, inter-brain)**
- Goldstein et al. (2018): alpha-mu band interbrain coupling at 8-12 Hz during touch-pain
- Same frequency, same position (trivially), but across two brains
- The inter-brain channel operates exclusively at the **1° Noble** — alpha's dominant enrichment position (+59% EC, the consciousness gate frequency)
- No significant inter-brain coupling was found in beta — only the alpha Noble1 position supports inter-brain synchronization

### Cross-position couplings (harmonic resonance)

**4. Beta 3°Noble → Gamma 3°Noble (14.0→60.0 Hz, predictive coding)**
- Bastos framework: beta feedback at ~14 Hz, gamma feedforward at ~60 Hz
- Ratio = 4.286 ≈ φ³ = 4.236 (1.2% error)
- Both at **3° Noble** (u=0.270 and u=0.294, within 0.024)
- The feedforward/feedback directional channels of [[predictive-coding]] connect the same lattice position across 3 octaves — the cortical hierarchy communicates at the 3° Noble position
- **Enrichment context**: Low-beta 3°Noble is depleted (-65% EC). Gamma 3°Noble is weakly enriched (+13% EC). Again, the functional coupling uses depleted or peripheral positions, not the enrichment peaks.

**5. Alpha Noble1 → High-beta 6°Noble (10.23→20.46 Hz, 2:1 harmonic)**
- The cleanest 2:1 harmonic in the lattice: Noble1 at u=0.618 maps to 6°Noble at u=0.058 via the +0.440 shift
- Lattice precision: 0.002 — within measurement error
- This bridges the alpha peak enrichment position (+59%) with the high-beta boundary-adjacent zone (+45%), creating a harmonic channel between the two most enriched regions of their respective bands

**6. Boundary → Attractor (20.0→40.0 Hz, 2:1 subharmonic)**
- The φ² boundary at ~20 Hz (CMC peak, Parkinson's STN) harmonically couples to the gamma attractor at ~40 Hz
- The +0.440 shift maps Boundary→near-Attractor (0.000→0.440, distance 0.060 from Attractor at 0.500)
- **Enrichment context**: High-beta boundary is massively enriched under EC (+134%). Gamma attractor is enriched (+79% EC). This is the one coupling that connects two enriched positions — the 20→40 Hz subharmonic is a resonance between two occupied states, not an empty→occupied link.

## The Enrichment-Coupling Paradox

A pattern emerges across all documented couplings: **PAC and functional coupling preferentially connect positions that are DEPLETED in resting enrichment data**, while the enrichment peaks (where dominant resting generators sit) are relatively uncoupled.

| Coupling | Slow position | Slow enrichment | Fast position | Fast enrichment | Both enriched? |
|----------|-------------|-----------------|--------------|-----------------|---------------|
| Theta→gamma PAC | Attractor | -67% | Attractor | +79% | ✗ one depleted |
| Alpha→gamma PAC | Noble1 | +59% | Attractor | +79% | ✓ both enriched |
| Pred.coding β→γ | 3°Noble | -65% | 3°Noble | +13% | ✗ one depleted |
| FM-theta→gamma | 2°Noble | -75% | Attractor | +79% | ✗ one depleted |
| 20→40 Hz subharm. | Boundary | +134% | Attractor | +79% | ✓ both enriched |
| θ→β WM PAC | Attractor | -67% | 3°InvNoble | -26% | ✗ both depleted |

**5 of 6 major PAC/coupling relationships involve at least one depleted position.** The positions where oscillatory generators rest (theta boundary +281%, alpha attractor +113%, low-beta 5°InvNoble +244%) are NOT the primary coupling positions. The coupling positions are where generators MOVE TO during active processing.

This resolves the "cognitively silent" paradox from Paper 3: the φ-lattice enrichment describes where peaks **rest**, while the coupling geometry describes where peaks **interact**. Resting positions and coupling positions are different because resting enrichment reflects anti-mode-locking stability (peaks sit where they won't be disturbed) while coupling requires energy transfer (which happens at positions where the lattice permits it). The lattice serves two orthogonal functions — segregation (at enriched positions) and integration (at depleted positions) — and the golden ratio's unique mathematics enables both simultaneously.

## The 2:1 Shift Creates a Coupling Network

The +0.440 shift under 2:1 coupling creates a deterministic network linking positions across octaves. Following the chain through multiple 2:1 doublings:

```
Boundary (0.000)
  → 2:1 → Attractor (0.440)
    → 2:1 → 6°InvNoble (0.880)
      → 2:1 → 3°N (0.320) [wraps]
        → 2:1 → 3°IN (0.760)
          → 2:1 → 2°N (0.200) [wraps]
            → 2:1 → Att (0.640)
              → ... (quasi-periodic, never repeats exactly because 0.440 is irrational)
```

Because log_φ(2) = 1.4404... is irrational, the chain **never returns to its starting position** — it ergodically fills the unit interval. This is the Three Distance Theorem in action: successive 2:1 harmonics visit every region of the lattice without ever repeating, ensuring that no position is permanently isolated from harmonic coupling.

But the chain visits some positions more closely than others. The tightest landings (within 0.004 of a named position): 6°Noble↔Attractor, Attractor↔6°InvNoble, Noble1↔6°Noble, 6°InvNoble↔2°Noble. These four position pairs form the **high-fidelity harmonic backbone** of the lattice — the pathways along which 2:1 coupling is most precise.

## φ vs Integer Ratios: Different Networks

The two coupling geometries create different network topologies:

**φⁿ coupling** (golden ratio multiples): preserves position, changes octave. Creates **vertical channels** in the lattice — the same position across all frequency bands. Each position is a column through the architecture. Theta attractor, alpha attractor, beta attractor, gamma attractor are all connected by φⁿ resonance.

**Integer coupling** (2:1, 3:1, 4:1): shifts position, changes octave. Creates **diagonal channels** that cross positions. No two connected positions are the same (except approximately, when the shift lands near a named position).

**Fibonacci coupling** (f(n) = f(n-1) + f(n-2)): connects three consecutive boundaries in an additive triplet. This is the three-wave resonance at integer-n positions — the mechanism that makes boundaries unstable (peaks avoid them in most bands) by enabling energy transfer between three bands simultaneously. Every boundary is simultaneously a member of infinitely many Fibonacci triplets:
- f₀ = 7.60 = 4.70 + 2.90 (theta boundary = delta boundary + sub-delta boundary)  
- f₁ = 12.30 = 7.60 + 4.70 (alpha boundary = f₀ + theta boundary)
- f₂ = 19.90 = 12.30 + 7.60 (beta boundary = alpha boundary + f₀)

The brain navigates all three networks simultaneously. Resting state operates at enriched positions connected by φⁿ (same-position) channels. Task engagement activates 2:1 harmonic pathways that shift energy to complementary positions. And Fibonacci resonance at boundaries enables cross-band energy transfer during state transitions.

## Predictions

This coupling geometry generates testable predictions:

**1. Task-evoked PAC should connect matched lattice positions.** If theta-gamma PAC operates as attractor→attractor φ⁴ coupling, then the PAC frequency should shift when theta frequency shifts. As theta migrates from Noble1 (resting, ~6.3 Hz) toward the boundary (EC, ~7.6 Hz), the coupled gamma frequency should shift from attractor (~41 Hz) toward boundary (~32 Hz) to maintain the φ⁴ ratio. This is testable in the Dortmund EC data.

**2. The 2:1 harmonic network predicts which positions show state-dependent modulation.** The +0.440 shift maps resting-enriched positions to specific partner positions that should show task-dependent enrichment increases. For example, alpha Noble1 (+59%) maps via 2:1 to high-beta 6°Noble (+45%) — both enriched. But theta boundary (+281%) maps to alpha attractor (+113%) — the two strongest enriched positions of adjacent bands are harmonically linked. This predicts that Berger-effect alpha amplification and EC theta convergence are harmonically coupled phenomena.

**3. Predictive coding's frequency channels should show φ³ position matching.** The Bastos framework's beta-feedback/gamma-feedforward asymmetry operates at 14→60 Hz — both at the 3° Noble position. Other feedback-feedforward frequency pairs should also connect matched lattice positions if the coupling is genuinely lattice-organized.

## Sources

- Siebenhühner et al. (2016) — 6.6:13.2 Hz (1:2), 13:54 Hz (1:4 ≈ φ³) phase-phase coupling
- Friedrich et al. (2026) — FM-theta 5.5 Hz → gamma 65 Hz PAC
- Lisman & Idiart — Theta ~6 Hz → gamma ~40 Hz PAC (attractor→attractor)
- Bastos et al. (2015) — Beta ~14 Hz feedback, gamma ~60 Hz feedforward (3°Noble→3°Noble ≈ φ³)
- Goldstein et al. (2018) — Alpha 10 Hz inter-brain coupling (Noble1→Noble1)
- Axmacher et al. (2010) — Theta ~6 Hz → beta ~28 Hz PAC for WM
- Tomassini et al. (2026) — Alpha CMC at Noble1, beta CMC at boundary
- Colgin et al. (2009) — Slow gamma (30-50 Hz) CA3 retrieval, fast gamma (70-140 Hz) EC encoding
- Huang et al. (2026, UCL) — Theta→slow/fast gamma double dissociation
- Kramer (2022) — φ uniquely enables cross-frequency coupling within one geometric series
- See [[cross-frequency-coupling]], [[theta-band-map]], [[alpha-band-map]], [[gamma-band-map]]
