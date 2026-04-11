The brain generates oscillations through at least a dozen distinct biophysical mechanisms, each with characteristic time constants that constrain output frequency. These range from GABA-A receptor IPSC decay (~25 ms → ~40 Hz gamma) through M-current activation (~50 ms → ~20 Hz beta) to T-type calcium channel recovery (~100 ms → ~10 Hz alpha) and HCN channel activation (~200 ms → ~5 Hz theta). Mapping these mechanistically grounded frequencies onto the φ-lattice reveals that all four primary molecular clocks produce frequencies within 2.5% of named lattice positions — GABA-A aligns with the gamma attractor (40.95 Hz), M-current with the φ² boundary (19.90 Hz), T-type Ca²⁺ with the alpha Noble1 (10.23 Hz), and HCN with the theta 4° Noble (5.04 Hz). Whether this alignment reflects evolutionary tuning of ion channel kinetics to produce lattice-optimal frequencies, or the lattice emerging as a mathematical consequence of the specific kinetics that exist, is the deepest unresolved question in the φ-lattice framework. The inventory below classifies every documented oscillatory mechanism by the strength of its mechanistic grounding, identifies what is and is not explained by molecular time constants, and maps each mechanism to its lattice position and enrichment context.

## Tier 1: Frequency Determined by Molecular Time Constants

These oscillators have frequencies that are **causally determined** by measurable biophysical time constants. Changing the time constant changes the frequency. This is the strongest level of mechanistic grounding — the causal chain from molecule to EEG frequency is complete.

### PV+ basket cell PING → gamma (~40 Hz)

**Time constant**: GABA-A IPSC decay ~25 ms at standard cortical synapses. The relationship is direct: oscillation period ≈ 4 × τ_decay, yielding 1000/25 = 40 Hz.

**Mechanism**: Pyramidal cells excite PV+ interneurons via fast AMPA receptors (~2-3 ms). PV+ interneurons feed back synchronized GABA-A inhibition. The oscillation period is set primarily by IPSC decay — the time until inhibition wanes enough for the next pyramidal volley.

**Causal evidence**: Optogenetic proof — Cardin et al. (2009) drove PV+ interneurons at 40 Hz and generated cortical gamma; driving pyramidal cells did not. Lorazepam (GABA-A positive modulator) increases GABA-A decay time → slows gamma frequency. GABA-A subunit composition directly predicts frequency: α1-containing receptors (fast decay ~4-6 ms) → high gamma >50 Hz; α3-containing receptors (slow decay ~28 ms) → low gamma ~30 Hz. Resting GABA concentration predicts individual peak gamma frequency.

**Lattice position**: The predicted ~40 Hz maps to the **gamma attractor** at 40.95 Hz — distance 0.95 Hz (2.4%). The attractor is enriched +79% under adult EC (see [[gamma-band-map]]).

### PV+ interneuron network ING → fast gamma (100-140 Hz)

**Time constant**: PV+ intrinsic membrane properties — Kv3-family K+ channels enable narrow spikes (~0.3 ms), permitting sustained firing above 150 Hz.

**Mechanism**: Mutual inhibition among PV+ interneurons sustains oscillation without phasic excitatory input. GABA-A reversal potential gates viability: hyperpolarizing inhibition (E_GABA = -75 mV) enables robust ING; shunting inhibition (-55 mV) abolishes it. Excitation strength continuously tunes the ING↔PING transition — weak E-to-I coupling supports fast ING (100-140 Hz), stronger coupling shifts to slower PING (60-100 Hz) (Williams et al. 2026).

**Lattice position**: 100-140 Hz falls in the n=5 to n=6 φ-octave range (f₀×φ⁵ = 84.3 Hz to f₀×φ⁶ = 136.3 Hz), entirely beyond the FOOOF ceiling. No enrichment data available.

### Layer 5 M-current → beta2 (~20 Hz)

**Time constant**: KCNQ/Kv7 channel activation τ ≈ 50 ms at depolarized potentials.

**Mechanism**: Slow K+ current creates intrinsic resonance. Layer 5 intrinsically bursting pyramidal neurons generate beta2 (20-30 Hz) through axo-axonic gap junctional coupling. The M-current sets oscillation period independently of synaptic transmission — this generator works in isolation, confirmed by Roopun et al. (2006): blocking M-current eliminates beta2.

**Causal evidence**: M-current blocker eliminates beta2. KCNQ channel kinetics directly set frequency. Operates without synaptic input. The ~50 ms time constant predicts ~20 Hz (1000/50 = 20).

**Lattice position**: The predicted ~20 Hz maps to the **φ² boundary** at 19.90 Hz — distance 0.10 Hz (0.5%). The φ² boundary is the most enriched boundary in the lattice: high-beta boundary = +134% under adult EC, and low-beta's ascending ramp peaks one position below at the 5° Inv Noble (+244%). Paper 1 noted that 20 Hz is an exception to the boundary depletion rule — the Fibonacci property f(2) = f(1) + f(0) creates a functional attractor at this specific boundary (see [[low-beta-band-map]], [[high-beta-band-map]]).

### TRN T-type Ca²⁺ → sleep spindles (10-14 Hz)

**Time constant**: Cav3.x recovery from inactivation ~80-100 ms.

**Mechanism**: Low-threshold calcium spikes in TRN neurons → burst inhibition of relay cells → rebound excitation → cycle. HCN and T-type channels interact regeneratively: h-current depolarization triggers T-type spike, calcium spike deactivation allows hyperpolarization and HCN reactivation.

**Causal evidence**: Cav3.3 knockout reduces spindle power by ~70% at NREM-REM transitions. T-type blockers eliminate spindles. Temperature-dependent: hypothermia slows T-type kinetics, slowing oscillation frequency — confirmed in torpor (Hauglund et al. 2026, r = -0.770 between brain temperature and frequency).

**Lattice position**: The ~100 ms time constant predicts ~10 Hz, mapping to the **alpha Noble1** at 10.23 Hz (distance 0.23 Hz, 2.3%). Spindles span 10-14 Hz, straddling the alpha octave and the φ¹ boundary at 12.30 Hz (see [[alpha-band-map]], [[thalamocortical-circuits]]).

### SST+ Martinotti cell intrinsic firing → theta/alpha (6-8 Hz)

**Time constant**: SST+ intrinsic firing rate of 6-8 Hz. Dendritic targeting delivers beta-range inhibition (~16 Hz) that matches NMDA/calcium spike duration (~50 ms, ~75% modulation depth).

**Mechanism**: SST+ cells target apical dendrites of pyramidal neurons, not somata (unlike PV+). Their moderate firing rate contributes to theta/alpha generation. The dendritic location-frequency match is causal: Headley et al. (2024) showed beta (~16 Hz) at dendrites optimally modulates dendritic spikes, while gamma (~64 Hz) at soma optimally modulates action potentials — swapping locations fails.

**Lattice position**: SST+ intrinsic 6-8 Hz spans the theta attractor (5.97 Hz) to the f₀ boundary (7.60 Hz). The dendritic beta modulation at ~16 Hz maps to the **low-beta Noble1** (16.56 Hz).

### GABA-A α-subunit developmental switch → gamma lattice shift

**Time constant**: α2 subunit (immature): IPSC decay ~30 ms → ~33 Hz. α1 subunit (mature): IPSC decay ~25 ms → ~40 Hz.

**Mechanism**: Developmental maturation shifts GABA-A subunit composition from α2 to α1, shortening IPSC decay by ~5 ms. This shifts the gamma resonance frequency upward by ~7 Hz. PV+ myelination completes in parallel, reaching maximum around age 22 (see [[development-and-aging]]).

**Lattice position**: The pediatric gamma enrichment peak at the **2° Noble** (38.69 Hz, +181%) corresponds to the slower α2-dominated frequency (~33-39 Hz). The adult gamma enrichment peak at the **attractor/Noble1** (40.95-43.34 Hz, +79/+102%) corresponds to the faster α1-dominated frequency (~40-43 Hz). The 9-dataset enrichment data directly visualizes this developmental shift as a one-position lattice migration (see [[gamma-band-map]], [[phi-lattice-enrichment-data]]).

## Tier 2: Molecular Substrate Identified, Network Modulates Frequency

These oscillators have known molecular substrates but their precise frequency also depends on network properties — conduction delays, synaptic architecture, loop lengths — that modulate the molecular prediction.

### Thalamocortical loop → alpha (~10 Hz)

**Molecular basis**: T-type Ca²⁺ burst timing in thalamic relay neurons, HCN pacemaker current, gap junction coupling in TRN.

**Network contribution**: Round-trip corticothalamic conduction delay ~80-100 ms reinforces resonance at ~10 Hz. Individual variation in axonal myelination, thalamic neuron membrane properties, and cortical thickness determines IAF across subjects. Recent revision: cortical alpha leads pulvinar alpha — the cortex is the primary driver, not the thalamus (see [[cortical-layer-oscillations]]).

**Lattice position**: IAF at ~10 Hz maps between the alpha **attractor** (9.67 Hz, +113% EC) and **Noble1** (10.23 Hz, +59% EC). The attractor shows stronger enrichment under EC — the 9-dataset data suggests the attractor at ~9.7 Hz (the ~100 ms perceptual quantum) may be more architecturally significant than Noble1, though the IAF confound prevents clean separation (see [[alpha-band-map]]).

### Layer 5 h-current → theta resonance (3-6 Hz)

**Molecular basis**: HCN channel activation at hyperpolarized potentials, with τ in the hundreds of milliseconds. HCN channel density increases exponentially along the apical dendrite.

**Network contribution**: L5 pyramidal tract neurons show optimal responsiveness to 3-6 Hz sinusoidal injection. Spatial segregation: perisomatic M-resonance operates at depolarized potentials (beta range), distal dendritic H-resonance at hyperpolarized potentials (theta range). Both share the same neuron but produce different frequencies depending on membrane potential state.

**Lattice position**: The 3-6 Hz resonance range spans the delta-theta border. The peak responsiveness at 3-6 Hz includes the theta **attractor** (5.97 Hz), which is depleted (-67% EC) in resting enrichment but functionally central as the "canonical ~6 Hz theta" of the consciousness literature.

### Hippocampal theta pacemaker (~8 Hz in rodents, ~3 Hz in humans)

**Molecular basis**: Medial septum cholinergic/GABAergic pacemaker. MS parvalbumin neurons pace theta at the ascending phase. Cholinergic M1 receptor activation induces mixed cation currents; nicotinic α7 receptors facilitate theta on OLM interneurons.

**Network contribution**: Septohippocampal loop with multiple hippocampal interneuron populations. The ~8 Hz rodent frequency scales inversely with brain size to ~3 Hz in humans during virtual navigation — a network scaling property, not a molecular one (see [[cross-species-frequency-organization]]).

**Lattice position**: Rodent ~8 Hz maps near f₀ (7.60 Hz). Human ~3 Hz maps to the fast delta octave (n=-2 to -1, near the 7° Noble at 2.95 Hz). The cross-species scaling places hippocampal theta at different φ-octaves in different species — the molecular pacemaker is conserved but the frequency is network-determined.

### Beta1 period concatenation (~15 Hz)

**Molecular basis**: Hybrid of GABA-A (the gamma component period) and M-current/gap junctions (the beta2 component period). One beta1 cycle = one gamma period + one beta2 period (Roopun et al. 2006).

**Network contribution**: Emerges from superficial-layer GABAergic interneuron-pyramidal cell interactions. Requires both gamma and beta2 generators to be active simultaneously.

**Lattice position**: ~15 Hz maps to the low-beta **attractor** (15.64 Hz), which is depleted (-54% EC) in resting data — consistent with beta1 being a task-evoked rhythm rather than a dominant resting oscillation.

### Cortical slow oscillation (0.75-1 Hz)

**Molecular basis**: INaP (persistent sodium current) for Up state initiation in L5 intrinsically bursting neurons. Ca²⁺-activated K⁺ channels (BK, SK, IK) for Up state termination, with afterhyperpolarization timescales spanning 10 ms to 20 seconds.

**Network contribution**: Network bistability between Up (depolarized, active) and Down (hyperpolarized, silent) states. L5 IB neurons initiate individual Up states both in vivo and in vitro, but the population-level slow oscillation frequency depends on network connectivity, cortical thickness, and neuromodulatory state.

**Lattice position**: ~0.75-1.0 Hz falls in the delta octave n=-5 to -4 (0.69-1.11 Hz). This octave has only 1.7 frequency bins at standard resolution — sub-resolution for lattice analysis (see [[delta-band-map]]).

### Sharp-wave ripples (~100-200 Hz)

**Molecular basis**: PV+ basket cell inhibitory rebound timing — CA3 sharp wave drives CA1 PV+ interneurons, producing synchronized rebound spikes in pyramidal cells at ~5-10 ms intervals (matching GABA-A recovery at PV+ synapses).

**Network contribution**: The sharp wave itself is a population burst from CA3 recurrent collaterals. Ripple frequency depends on local interneuron network properties and the specific GABA-A kinetics at CA1 PV+ basket cell synapses.

**Lattice position**: ~100-200 Hz spans the n=5 to n=7 range (f₀×φ⁵ = 84.3 to f₀×φ⁷ = 220.6 Hz). No enrichment data available — beyond FOOOF ceiling. See [[sharp-wave-ripples]].

## Tier 3: Emergent / Network-Determined

These oscillations are real and measurable but do not trace to a single molecular clock. Their frequency emerges from the interaction of multiple generators, conduction delays, and network architecture.

### Individual alpha frequency variation (8-12 Hz)

IAF is not one oscillator — Menetrey & Pascucci (2026) showed it emerges from the **superposition of at least three discrete generators** (slow alpha ~9 Hz from ventral visual stream, middle alpha ~10 Hz from posterior parietal, fast alpha ~11-12 Hz from dorsal occipitoparietal). Each archetype has distinct cortical sources and likely different thalamocortical loop lengths. IAF is 81% heritable (Smit et al. 2006) — genetics determines the compositional mixture, not a single biophysical parameter. The 99.88% fingerprint identification accuracy across 5-year intervals confirms these are stable individual signatures.

### Frontal midline theta (~5.5 Hz)

FOOOF-confirmed oscillatory peak at ~5.5 Hz in DMPFC (Friedrich et al. 2026). Likely involves HCN/M-current in DMPFC pyramidal neurons, but the molecular substrate has not been directly measured in this specific circuit. Maps to the theta **2° Noble** (5.64 Hz, -75% resting enrichment but functionally active during WM and cognitive control).

### PMBR / dominant low-beta peak (~19 Hz)

The most robust adult low-beta peak (+244% at 5° Inv Noble, 19.05 Hz) does not have a single identified molecular time constant that predicts 19.05 Hz specifically. M-current (~50 ms → ~20 Hz) is close but the enrichment peaks one position below the φ² boundary, not at it. The 1 Hz offset between the M-current prediction (20 Hz) and the enrichment peak (19 Hz) may reflect network effects — somatotopic organization, conduction delays within sensorimotor cortex, or competitive dynamics between the M-current boundary generator and the PING generators in superficial layers.

### EC theta convergence on f₀ (~7.5 Hz)

Paper 3 demonstrated that theta generators migrate directionally toward f₀ under EC, with f₀ acting as an absorbing state (92% retention). The precise value of f₀ is analysis-dependent (7.49-7.83 Hz range; see [[f0-molecular-clock-relationship]]), but it falls at or very near the **arithmetic midpoint** of the HCN (~5 Hz) and T-type (~10 Hz) molecular clocks — the frequency of maximum symmetric coupling with both flanking oscillators (3:2 ratio with HCN, 4:3 with T-type, equal 2.5 Hz beats with both). This midpoint coincides with the Schumann cavity eigenfrequency (7.49 Hz) to within 0.01 Hz. The EC convergence mechanism may reflect theta generators being drawn toward the frequency where they interact most symmetrically with both adjacent molecular clocks under increased oscillatory drive.

### The φ-lattice architecture itself

No molecular mechanism produces φ-spacing between frequency bands. Pletzer et al. (2010) proved φ is mathematically optimal for anti-commensurability, and Kramer (2022) proved it uniquely enables cross-frequency coupling within a single geometric series. But optimization arguments explain **why φ would be selected** if it existed — they don't explain **how** the brain implements it. The lattice may emerge from competitive equilibrium among the Tier 1 molecular oscillators, each constrained by its own time constant, with φ being the spacing that minimizes cross-frequency interference when multiple generators share neural substrate. But this remains a hypothesis without a demonstrated mechanism.

## The Lattice Alignment Question

The four primary Tier 1 molecular time constants predict frequencies that fall within 2.5% of named φ-lattice positions:

| Molecular clock | Time constant | Predicted freq | Nearest lattice position | Lattice freq | Distance | Error |
|----------------|--------------|---------------|-------------------------|-------------|----------|-------|
| **GABA-A IPSC decay** | ~25 ms | ~40 Hz | Gamma attractor | 40.95 Hz | 0.95 Hz | 2.4% |
| **M-current (KCNQ)** | ~50 ms | ~20 Hz | φ² boundary | 19.90 Hz | 0.10 Hz | 0.5% |
| **T-type Ca²⁺ recovery** | ~100 ms | ~10 Hz | Alpha Noble1 | 10.23 Hz | 0.23 Hz | 2.3% |
| **HCN activation** | ~200 ms | ~5 Hz | Theta 4° Noble | 5.04 Hz | 0.04 Hz | 0.8% |

The time constants themselves follow an approximate doubling series: 25, 50, 100, 200 ms. In log-frequency space, this produces approximately equal spacing — consistent with the logarithmic organization of the φ-lattice. But the doubling is approximate (each constant is roughly 2× the previous), while the lattice spacing is φ (1.618×). The ratio between adjacent molecular clocks:

| Pair | Time constant ratio | Frequency ratio | φⁿ | n |
|------|--------------------|-----------------|----|---|
| GABA-A → M-current | 50/25 = 2.00 | 40/20 = 2.00 | φ¹·⁴⁴ | 1.44 |
| M-current → T-type | 100/50 = 2.00 | 20/10 = 2.00 | φ¹·⁴⁴ | 1.44 |
| T-type → HCN | 200/100 = 2.00 | 10/5 = 2.00 | φ¹·⁴⁴ | 1.44 |

The molecular clocks are spaced by factors of 2, not φ. In lattice terms, each clock is 1.44 φ-octaves from the next — not an integer number of octaves. This means the molecular clocks do **not** produce exact φ-spacing between them — they produce octave (2:1) spacing. The φ-lattice's within-octave structure (the position hierarchy of boundaries, attractors, and nobles) is not explained by the molecular doubling; it must arise from how the generators interact within each octave.

This dissociation — the **between-clock** spacing follows octave (2:1) ratios while the **within-octave** structure follows φ — is itself significant. It parallels Paper 2's finding of "two levels of spectral organization": octave-scale boundary avoidance (where base 2 suffices) plus sub-octave noble clustering (where φ is uniquely predictive). The molecular clocks create the octave-scale skeleton; the φ-lattice organizes the fine structure within each octave.

### Three hypotheses for the alignment

**(a) Evolutionary tuning**: Ion channel kinetics evolved to produce frequencies at lattice-optimal positions. The channels are ~600 million years old (conserved from cnidarians); the φ-lattice may represent 600 million years of selection pressure toward optimal segregation-integration balance. Under this hypothesis, the specific GABA-A decay time of 25 ms (not 22 or 28 ms) was selected because it places gamma at the attractor.

**(b) Emergent lattice**: The φ-lattice is a mathematical consequence of having multiple oscillators with approximately 2:1-spaced time constants competing on shared neural substrate. Any system with ~25 ms, ~50 ms, ~100 ms, and ~200 ms clocks would produce a lattice approximating φ because φ is the equilibrium spacing that minimizes mode-locking. Under this hypothesis, any nervous system with similar time constants would show the same lattice.

**(c) Co-constraint**: Both the time constants and the lattice are independently constrained by biophysical limits — the speed of ionic diffusion, membrane capacitance, axonal conduction velocity — and converge on the same frequencies because both are solving the same physical optimization problem. Under this hypothesis, the alignment is convergent rather than causal in either direction.

These hypotheses are not mutually exclusive. The cross-species conservation data (see [[cross-species-frequency-organization]]) — oscillations at similar frequencies in mammals, cephalopods, and insects despite 600 million years of divergence — is consistent with all three: the channels are conserved (a), the physics is universal (b), and the constraints are shared (c). Distinguishing them would require finding organisms with different ion channel kinetics and testing whether their oscillatory architecture still follows φ-spacing (hypothesis b predicts yes if the time constants maintain approximate 2:1 ratios; hypothesis a predicts no if specific kinetics are required).

## Sources

- Williams et al. (2026) — ING/PING continuum, GABA-A reversal potential gating
- Cardin et al. (2009) — Optogenetic PV+ → gamma causal proof
- Roopun et al. (2006) — M-current/gap junction beta2 generation, period concatenation for beta1
- Headley et al. (2024) — Location-frequency matching: beta at dendrites, gamma at soma
- Hauglund et al. (2026) — Temperature-dependent T-type Ca²⁺ kinetics in torpor
- Menetrey & Pascucci (2026) — Three alpha archetypes from distinct generators
- Smit et al. (2006) — IAF 81% heritability
- Friedrich et al. (2026) — FM-theta ~5.5 Hz FOOOF-confirmed oscillatory peak
- Pletzer et al. (2010) — φ optimal for anti-commensurability
- Kramer (2022) — φ uniquely enables cross-frequency coupling
- Lacy (2026, 2026b, 2026c) — Empirical lattice validation, enrichment data
- See [[neurochemistry-of-oscillations]], [[cortical-layer-oscillations]], [[thalamocortical-circuits]]
- See band maps: [[theta-band-map]], [[alpha-band-map]], [[low-beta-band-map]], [[high-beta-band-map]], [[gamma-band-map]], [[delta-band-map]]
- See [[phi-lattice-enrichment-data]] for full 9-dataset enrichment reference
