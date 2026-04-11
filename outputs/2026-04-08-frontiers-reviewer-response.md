# Response to Reviewers

**Manuscript ID:** 1786996
**Title:** Golden Ratio Architecture of Human Neural Oscillations
**Journal:** Frontiers in Computational Neuroscience

Dear Editor and Reviewers,

I thank both reviewers for their thoughtful and constructive feedback. The revision addresses each comment through targeted improvements to the manuscript. In addition to the changes requested by reviewers, two substantive additions strengthen the paper:

1. **Independent replication dataset (EEGEmotions-27):** An independent dataset of 612,990 FOOOF-detected peaks across 2,342 sessions recorded during emotion induction paradigms has been added to Study 2 as a replication analysis. This dataset, which shares no participants, recording sites, or cognitive contexts with the primary dataset, confirms all primary findings in aggregate cross-band analysis (Kendall's tau = 1.0, gamma showing strongest aggregate adherence, session-level consistency).

2. **Companion manuscripts:** Two companion manuscripts provide extended validation and are available as preprints for reviewer reference:
   - Lacy, M. (2026). *Golden Ratio EEG Architecture Emerges Near Schumann Frequencies but Does Not Constrain State Modulation.* Zenodo. https://doi.org/10.5281/zenodo.18765773
   - Lacy, M. (2026). *Dominant Peaks, Fragile Metrics: Separating Architectural Signal from Methodological Artifact in Phi-Lattice EEG Analysis.* Zenodo. https://doi.org/10.5281/zenodo.18856274

   The first (Lacy, 2026a) analyzes structural specificity of the EEGMMIDB dataset (N=109, phi ranks 1st among 9 exponential bases). The second (Lacy, 2026b) provides a diagnostic cascade across the LEMON (N=202) and Dortmund Vital Study (N=608, plus N=208 longitudinal) datasets, demonstrating that dominant-peak alignment replicates robustly across three independent datasets. Key results from both are summarized in a new Discussion subsection ("Independent Replication and Extension," Section 6.3) without expanding the scope of the present paper.

   References:
   - **(a)** Lacy, M. (2026a). *Golden Ratio EEG Architecture Emerges Near Schumann Frequencies but Does Not Constrain State Modulation.* Zenodo. https://doi.org/10.5281/zenodo.18765773
   - **(b)** Lacy, M. (2026b). *Dominant Peaks, Fragile Metrics: Separating Architectural Signal from Methodological Artifact in Phi-Lattice EEG Analysis.* Zenodo. https://doi.org/10.5281/zenodo.18856274

Below I address each comment point by point. Reviewer text is quoted in italics; responses follow in plain text.

---

## REVIEWER 1

### General Comments

#### R1-A1: Statistical robustness and FOOOF dependence

> *"The ms. would benefit from additional analyses demonstrating that the observed effects cannot be explained by known properties of EEG spectra, such as band clustering, spectral smoothing, or biases introduced by peak detection algorithms. In particular, the dependence on FOOOF-based peak extraction requires further discussion... consider... alternative approaches (e.g., simple local maxima detection or different parameter settings)."*

**Response:**

I appreciate this important concern. Three lines of evidence now address it:

1. **Methodological triangulation (new paragraph in Discussion Section 6.3):** In subsequent analyses (Lacy, 2026a), Generalized Eigendecomposition (GED) -- a multi-channel spatial coherence method fundamentally different from FOOOF -- was applied to 1,584,561 peaks across 3,261 sessions. GED reproduced the identical position-type hierarchy (Kendall's tau = 1.0), confirming that the boundary-attractor-noble ordering is not an artifact of FOOOF's peak detection algorithm. Additionally, structural specificity analysis in the same companion manuscript tested nine exponential bases -- each evaluated at their own natural lattice positions -- and phi achieved the highest structural score (SS = 45.6), ranked 1st in 100% of 1,000 bootstraps.

2. **Parameter sensitivity (Discussion Section 6.3):** A companion analysis (Lacy, 2026b) systematically varied FOOOF parameters including frequency range and aperiodic fitting approach. While aggregate enrichment scores are sensitive to frequency range (sign reversal at [1, 85] Hz) and to cross-band pooling methodology (see Limitation 5), dominant-peak alignment replicated across three independent datasets (LEMON N=202, Dortmund N=608, EEGMMIDB N=109; Cohen's d = 0.40 in all three).

3. **Future validation (Limitations Section 6.8):** IRASA and eBOSC aperiodic separation methods have been identified as priority targets for further methodological validation.

---

#### R1-A2: "Schumann Ignition Events" terminology

> *"The framing of 'Schumann Ignition Events' may suggest a causal relationship that is NOT supported by the presented data. The authors may wish to consider alternative terminology or emphasize more clearly that the name refers to frequency correspondence rather than demonstrated coupling."*

**Response:**

The term "Schumann Ignition Events" reflects the initial discovery of transient ignitions -- brief episodes of elevated amplitude, coherence, and phase-locking at frequencies near the Schumann Resonance fundamental (~7.8 Hz). The name references these observed characteristics -- the "ignition" of large-amplitude, highly coherent oscillations at SR frequencies -- rather than demonstrated electromagnetic coupling between neural and geophysical systems.

**Changes made:** An explicit clarification paragraph has been added where SIEs are first defined (Section 2.1.1): "The term 'Schumann Ignition Events' reflects the initial discovery of transient ignitions -- brief episodes of elevated amplitude, coherence, and phase-locking at frequencies near Earth's Schumann Resonance fundamental (~7.8 Hz). The name references these observed characteristics rather than demonstrated electromagnetic coupling between neural and geophysical systems." A parallel caveat has been added to Discussion Section 6.4 (Relationship to Schumann Resonance).

---

#### R1-A3: Single-meditator discovery phase

> *"The discovery phase of the study relies heavily on recordings from a single experienced meditator... The manuscript should more clearly distinguish between hypothesis-generating observations and confirmatory analyses."*

**Response:**

This distinction has been sharpened throughout the revision:

1. Study 1 is now explicitly labeled as "hypothesis-generating" in Section 2.1.1: "The initial discovery and characterization constitutes an exploratory, hypothesis-generating analysis. The core discovery originated from longitudinal recordings of a single experienced meditator, with subsequent validation extending to 91 participants across five cognitive contexts."

2. A new transition paragraph between Studies 1 and 2 explicitly marks the shift: "The discovery of phi-ratio precision in SIE harmonics motivated the confirmatory analysis that follows. Critically, Study 2 employs unconstrained FOOOF spectral parameterization across the full frequency range -- unlike Study 1's targeted search within Schumann Resonance frequency windows -- providing an independent, hypothesis-testing evaluation of the phi-lattice framework."

3. The Integration section (Section 5.1.3, Methodological Triangulation) now notes that Study 1's constrained search windows partially predetermine outcomes, and that "Study 1's primary contributions are: (1) discovery and characterization of SIE phenomenology, (2) the independence-convergence paradox, and (3) motivation for the phi-lattice hypothesis."

---

#### R1-A4: Separating empirical findings from speculative mechanisms

> *"The discussion outlines plausible ideas involving oscillator coupling, receptor kinetics, and cross-frequency communication, but these mechanisms are not directly tested... clearly separate empirical findings and speculative interpretation."*

**Response:**

The Discussion has been restructured:

1. Section 6.2 has been renamed "Theoretical Implications (Speculative)" with an opening disclaimer: "The following mechanistic interpretations are not directly tested in the present study. They represent theoretical hypotheses motivated by the empirical findings and intended to guide future investigation."

2. A new paragraph notes that companion work (Lacy, 2026a) found that state-dependent spectral modulation in the EEGMMIDB dataset was base-indifferent -- spectral shifts during motor tasks occurred equivalently across all tested exponential bases, not specifically at phi-lattice positions -- illustrating how mechanistic predictions can be empirically evaluated and, in this case, falsified.

---

### Specific Comments

#### R1-B1: SIE pipeline threshold justification

> *"It would be helpful to provide a clearer justification for key threshold choices (e.g., z-score thresholds, synchronization criteria, and composite scoring parameters). Sensitivity analyses demonstrating that the results are robust to these parameter choices must be carried out."*

**Response:**

Justifications for key thresholds have been added to Section 2.1.5:

- **Envelope z-score >= 3.0:** This standard threshold (3 SD above session mean) identifies events with unambiguous power elevation while maintaining a detection rate of ~2 events/session (range 1-27), consistent with the transient nature of the phenomena.

- **Kuramoto R(t) >= 0.6:** At R = 0.6, the probability of chance phase alignment across N >= 4 independent channels is < 0.01 by circular statistics.

- **Composite SR-Score weights:** The phi-derived weights (SR1: 0.618, SR3: 0.326, SR5: 0.146) give greatest emphasis to the fundamental frequency, where synchronization is strongest and most reliable.

Regarding formal sensitivity analysis across threshold ranges: the detection pipeline was applied with consistent parameters across six independent datasets spanning five cognitive contexts and three EEG devices, all yielding convergent harmonic frequency estimates (Table 3). This cross-dataset consistency provides indirect evidence of robustness to recording-specific factors. A systematic parameter sweep across z-score and Kuramoto thresholds is identified as an important direction for future work.

---

#### R1-B2: Individual alpha frequency variability

> *"Individual differences in alpha frequency are well documented and can vary across subjects. The manuscript briefly acknowledges this but does not fully address how such variability might influence the apparent alignment with phi-lattice positions."*

**Response:**

The revised Discussion addresses this substantially:

1. **Alpha shows modest aggregate phi-adherence:** Among all frequency bands in the aggregate cross-band analysis, alpha exhibits modest enrichment at noble positions (+4.2% vs. +144.8% for gamma), consistent with IAF variability (~1 Hz population range) spreading peaks across lattice positions. (Note: these aggregate figures may mask within-band position structure; see Limitation 5.)

2. **Companion analysis (Lacy, 2026b):** In the LEMON dataset (N=202), per-subject correction for individual f_0 reversed an apparent age-related decline in lattice alignment. However, dominant-peak alignment is robust to IAF variation (Cohen's d = 0.40, age-invariant: r = +0.054, p = 0.45).

3. **Theta convergence to f_0:** Analysis from the Dortmund Vital Study (N=608) showed that theta peak frequencies under eyes-closed conditions converge to f_0 = 7.8 Hz (near the Schumann fundamental), NOT IAF/2. This rules out models in which the phi-lattice is an artifact of IAF-derived harmonic relationships.

---

#### R1-B3: How could population-level constraints arise from realistic networks?

> *"How such population-level constraints could arise from realistic neural oscillator networks? Do the authors suggest a specific model?"*

**Response:**

This is an excellent question that the present empirical study cannot fully answer. However, four considerations now narrow the space of viable models:

1. **Population-level encoding:** The independence-convergence paradox (Section 2.3.3) demonstrates that individual harmonic frequencies vary independently while their ratios remain precise -- indicating population-level distributional constraints.

2. **Species-level constant, not individual trait:** Five-year longitudinal data (Dortmund Vital Study, N=208; Lacy, 2026b) shows population-level phi-alignment invariant to the third decimal (group d-bar = 0.031-0.033), while individual alignment shows near-zero test-retest reliability (all ICC < 0). Any viable model must produce stable emergent statistics from individually unstable dynamics.

3. **Biophysical time constants:** Four ion channel families create a coarse 2:1 frequency skeleton: HCN (~5 Hz), T-type Ca²⁺ (~10 Hz), M-current KCNQ (~20 Hz), and GABA-A (~40 Hz). The fundamental frequency f₀ = 7.6 Hz falls at the 3:2 harmonic of 5 Hz and the 3:4 harmonic of 10 Hz -- the point where the resonance of both flanking oscillators overlaps maximally. This harmonic midpoint (7.5 Hz) independently converges with the Schumann cavity eigenfrequency (7.49 Hz) and the empirical neural optimum (7.60 Hz) to within 0.11 Hz. Population-level phi constraints may thus arise not from coupling but from independent biophysical optimization: each generator is constrained by local ion channel kinetics to frequencies that happen to approximate phi-ratio relationships.

4. **Mathematical foundation:** Pletzer et al. (2010) proved that phi uniquely prevents spurious synchronization while enabling controlled cross-frequency coupling via Fibonacci additivity.

A dedicated computational modeling study is identified as an important direction for future work (Section 6.7, Population-Level Model; Section 6.9, Future Directions).

---

#### R1-B4: Schematic figure of the phi-lattice

> *"The manuscript would benefit from a schematic figure summarizing the proposed mathematical framework of the phi-lattice, including boundary positions, attractor positions, and their relation to canonical EEG bands."*

**Response:**

A new schematic figure has been added to Section 3.2 (The Core Equation). It displays the f(n) = f_0 x phi^n frequency positions from delta through gamma on a logarithmic frequency axis, with position types distinguished by symbol shape and color: boundaries (integer n, red squares), attractors (half-integer n, blue circles), and noble positions (n + 0.618, gold diamonds). Shaded vertical spans indicate phi-derived EEG band definitions. Specific frequencies are labeled for all position types.

---

#### R1-B5: Missing cross-frequency coupling literature

> *"The ms. may appear insufficiently grounded in the extensive literature on cross-frequency coupling and oscillatory scaling... [5 specific references]"*

**Response:**

All five recommended references have been added and integrated into a new Discussion subsection (Section 6.5, "Relationship to Cross-Frequency Coupling Literature"):

1. **Jensen & Colgin (2007):** CFC as a fundamental coordination mechanism across timescales.
2. **Aru et al. (2015):** Methodological challenges in measuring CFC; phi-lattice generates specific testable predictions distinguishing genuine from artifactual coupling.
3. **Abe et al. (2024):** Quadratic phase coupling detection via cross-bicoherence -- directly testable prediction from Fibonacci additivity.
4. **von Stein & Sarnthein (2000):** Different frequencies for different spatial scales -- consistent with phi-lattice role in maintaining independent processing across scales.
5. **Roopun et al. (2008):** Period concatenation between gamma and beta rhythms -- biophysical mechanism for inter-band coupling at Fibonacci-related positions.

---

## REVIEWER 2

#### R2-1: Use of "we" with single author

> *"Is it technically correct to use 'we' if there is one single author?"*

**Response:** The manuscript has been revised to use passive voice throughout ("was analyzed," "were detected") or first person singular ("I"). All instances of "we" have been replaced.

---

#### R2-2: "A systematic review" -- reference needed

> *"Page 5: 'A systematic review of band definitions across published EEG studies reveals substantial inconsistency:' which systematic review? Reference needed."*

**Response:** The phrasing has been revised to: "A survey of band definitions across published EEG studies (Table 1) reveals substantial inconsistency." A source note has been added to Table 1's caption: "Band definition ranges compiled by the author from representative published studies spanning 1999-2024, including Klimesch (1999), Buzsaki (2006), Cohen (2017), and Herrmann et al. (2016)."

---

#### R2-3: Table 1 not mentioned in text

> *"Table 1 is not mentioned in the text, as it is commonly done in scientific manuscripts."*

**Response:** An in-text reference has been added: "A survey of band definitions across published EEG studies (Table 1) reveals substantial inconsistency."

---

#### R2-4: Missing explicit research question and hypothesis

> *"In this introduction section, I was expecting an explicit statement of the research question of the manuscript, as well as maybe a research hypothesis."*

**Response:** The following has been added at the end of Section 1.4 (Study Overview):

"The present study addresses two primary research questions: (1) Are the harmonic frequencies observed during transient high-coherence events organized according to golden ratio (phi = 1.618) scaling? (2) Does this phi-ratio organization extend beyond transient events to govern the continuous spectral architecture of EEG? Based on the mathematical properties of the golden ratio -- maximal anti-commensurability and Fibonacci additivity -- the hypothesis is that EEG frequency organization follows f(n) = f_0 x phi^n, generating specific predictions for where spectral peaks should cluster (attractor and noble positions) and where they should be depleted (boundary positions)."

---

#### R2-5: Figure 1 panel labels and text references

> *"Figure 1. Which panel is (A), (B), etc, labels missing. Vertical axis legend should be 'Frequency (Hz).' Time-frequency spectrogram might be unnecessary? Figure 1 not mentioned in the text, same as Figure 2."*

**Response:**

1. Panel labels (A)-(F) have been added to Figure 1 and (G)-(L) to Figure 2
2. The spectrogram panel (A) vertical axis displays frequency in Hz; the caption now clarifies the frequency range "(2--25 Hz)"
3. "Time-frequency spectrogram" has been shortened to "Spectrogram" in the caption
4. In-text references have been added: Figure 1 is now referenced in Section 2.2.3 (Synchronization Metrics) and Figure 2 in Section 2.2.4 (Temporal Dynamics)

---

#### R2-6: Tables 5 and 6 formatting

> *"Table 5 and 6 treated as text in the manuscript."*

**Response:** Tables 5 and 6 are formatted as proper LaTeX tables in the source. This appears to have been a rendering artifact in the review PDF. The tables have been verified and are correctly formatted with \begin{table} environments, column headers, and booktabs formatting.

---

## Summary of Changes

| Change | Section(s) | Reviewer |
|--------|-----------|----------|
| EEGEmotions-27 independent replication added | Study 2 (throughout) | New |
| Aggregate enrichment labels and caveats | Abstract, enrichment/band tables, lattice schematic figure, Secs 4.2.2/4.2.3/6.2.1/6.2.3/6.2.6/7 | New |
| FOOOF Limitation 5 strengthened (cross-band density) | 6.8 | New |
| Per-band enrichment normalization future direction | 6.9 | New |
| Section 6.2.1 title change | 6.2.1 | New |
| SIE terminology disclaimer | 2.1.1, 6.4 | R1-A2 |
| Discovery vs. confirmation distinction | 2.1.1, transition before Sec. 3, 5.1.3 | R1-A3 |
| Mechanistic discussion labeled speculative | 6.2 (renamed) | R1-A4 |
| SIE pipeline threshold justification | 2.1.5 | R1-B1 |
| Alternative scaling factor comparison | 4.3.1 | R1-A1 |
| IAF variability discussion | 6.2.6, 6.3 | R1-B2 |
| Population-level constraint discussion | 6.7 | R1-B3 |
| Schematic phi-lattice figure | New figure in 3.2 | R1-B4 |
| Five CFC references added | 6.5, Bibliography | R1-B5 |
| FOOOF sensitivity discussion | 6.8 (Limitations) | R1-A1 |
| GED triangulation summary | 6.3 | R1-A1 |
| Independent replication summary | 6.3 (new subsection) | R1-A1, A3 |
| "we" -> passive/first person | Throughout | R2-1 |
| "Systematic review" reworded + Table 1 sourced | 1.2, Table 1 caption | R2-2, R2-3 |
| Table 1 in-text reference | 1.2 | R2-3 |
| Explicit research question/hypothesis | 1.4 | R2-4 |
| Figure panel labels and in-text references | Figs 1-2, Secs 2.2.3, 2.2.4 | R2-5 |
| Tables 5-6 formatting verified | 2.3.2-2.3.3 | R2-6 |

I believe these changes address all reviewer concerns while maintaining the scope and structure of the original submission. The addition of the EEGEmotions-27 replication dataset and the summary of companion manuscript results substantially strengthen the empirical foundation. Additionally, session-level consistency metrics have been updated following re-analysis with corrected peak filtering criteria (primary dataset: 87.4\%, Cohen's d = 1.19; previously reported as 83.3\%, d = 0.89). Aggregate enrichment values throughout the manuscript have been explicitly labeled as cross-band aggregates, and Limitation 5 (FOOOF dependence) has been strengthened to note that aggregate figures may mask band-specific enrichment patterns — an important methodological consideration that does not undermine the core findings but appropriately qualifies the precision of specific numerical claims. I am grateful for the reviewers' careful reading and constructive suggestions, which have substantially improved the manuscript.

Sincerely,
Michael Lacy
