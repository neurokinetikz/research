# Golden Ratio Architecture of Human Neural Oscillations

## From Transient Discovery to Continuous Substrate

Michael Lacy

michael.lacy@gmail.com

January 13, 2026

---

*Submitted to Journal:* Frontiers in Computational Neuroscience

*Article type:* Original Research Article

*Manuscript ID:* 1786996

*Received on:* 03 Feb 2026

---

## Scope Statement

This manuscript reports the discovery and validation of golden ratio (φⁿ) frequency organization in human EEG, directly addressing fundamental questions in neural oscillation research. Through analysis of 1,366 transient high-coherence events and 244,955 spectral peaks across 91 participants, we demonstrate that neural oscillations follow a mathematically principled architecture governed by f(n) = f₀ × φⁿ, with harmonic ratios deviating less than 1% from theoretical predictions. The work fits squarely within Frontiers in Neuroscience's scope by: (1) introducing novel EEG analysis methodology combining FOOOF spectral parameterization with lattice coordinate analysis; (2) providing the first mathematically derived framework for EEG frequency band definitions, resolving long-standing inconsistencies in the field; (3) presenting a testable theoretical model (substrate-ignition) linking transient synchronization events to continuous spectral organization; and (4) generating specific, falsifiable predictions for cross-frequency coupling, band-specific dynamics, and cross-species organization. The findings have immediate implications for understanding neural computation and potential clinical applications in disorders characterized by oscillatory abnormalities.

## Conflict of interest statement

**The authors declare that the research was conducted in the absence of any commercial or financial relationships that could be construed as a potential conflict of interest**

## Credit Author Statement

**Michael Lacy**: Conceptualization, Data curation, Formal Analysis, Funding acquisition, Investigation, Methodology, Project administration, Resources, Software, Supervision, Validation, Visualization, Writing -- original draft, Writing -- review & editing.

## Keywords

EEG, Frequency bands, golden ratio, Neural oscillations, phase synchronization, Schumann resonance, Spectral architecture, Transient Dynamics

## Abstract

Word count: 335

We report the discovery and systematic validation of golden ratio (φ = 1.618) frequency organization in human EEG, revealing a mathematical architecture that spans both transient high-coherence states and continuous spectral structure. Analysis of 1,366 Schumann Ignition Events (SIEs)---transient episodes of multi-band network synchronization detected across 91 participants, 661 recording sessions, and three consumer EEG devices---revealed that harmonic frequencies at SR1 (~7.6 Hz), SR3 (~20 Hz), and SR5 (~32 Hz) maintain precise golden ratio relationships, with the SR3/SR1 ratio deviating only +0.16% from φ² and SR5/SR3 deviating +0.79% from φ, yielding mean ratio error below 1%. Critically, individual harmonic frequencies varied completely independently across events (all pairwise |r| < 0.03), yet ratio precision was preserved---an "independence-convergence paradox" indicating that φⁿ constraints are encoded in population-level distributions rather than event-level coupling. This discovery prompted the hypothesis that φⁿ organization governs continuous EEG architecture, formalized as f(n) = f₀ × φⁿ where f₀ = 7.60 Hz (from independent geophysical Schumann Resonance monitoring), generating specific predictions: spectral peak depletion at integer n positions (boundaries), enrichment at half-integer positions (attractors), and maximal enrichment at n + 0.618 (noble positions). Spectral parameterization of 244,955 oscillatory peaks across 968 sessions confirmed all predictions: boundaries showed −18% depletion, attractors showed +21% enrichment, and noble positions showed +39% enrichment, with the gamma band exhibiting dramatically stronger adherence (+144.8% at noble positions) consistent with functional requirements for precise phase relationships in temporal binding. Among six scaling factors tested, only φ exhibited the theoretically predicted position-type ordering. Convergent f₀ estimates from independent neural (SIE mean: 7.63 Hz) and geophysical (Schumann Resonance: 7.60 Hz) sources support a "substrate-ignition" model wherein the φⁿ lattice exists continuously as an architectural constraint, with SIEs representing transient amplification of this substrate. These findings establish φⁿ architecture as a fundamental organizing principle of human neural oscillations, providing the first mathematically principled basis for EEG frequency band definitions and suggesting that the golden ratio's unique mathematical properties---maximal anti-commensurability enabling frequency segregation, and Fibonacci additivity enabling cross-frequency integration---may represent an optimal solution to the competing demands of neural computation.

## Funding statement

The author(s) declare that no financial support was received for the research and/or publication of this article.

## Ethics statements

### Studies involving animal subjects

Generated Statement: No animal studies are presented in this manuscript.

### Studies involving human subjects

Generated Statement: Ethical approval was not required for the studies involving humans because Author analyzed his own longitudinal EEG recordings.. The studies were conducted in accordance with the local legislation and institutional requirements. Written informed consent for participation was not required from the participants or the participants' legal guardians/next of kin in accordance with the national legislation and institutional requirements because Author analyzed his own longitudinal EEG recordings. Publicly available datasets from published research used as validation sets..

### Inclusion of identifiable human data

Generated Statement: No potentially identifiable images or data are presented in this study.

### Data availability statement

Generated Statement: The original contributions presented in the study are included in the article/supplementary material, further inquiries can be directed to the corresponding author/s.

### Generative AI disclosure

The author(s) verify and take full responsibility for the use of generative AI in the preparation of this manuscript. Generative AI was used

In accordance with policy on AI-assisted tools, we disclose the following: AI tools (Claude Opus 4.5, Anthropic) were used for coding assistance in developing the analysis pipeline and for manuscript preparation. Overleaf was used for paper creation. ChatGPT (OpenAI), Google Gemini, and Grok (xAI) were used as manuscript reviewers for consistency and clarity. All scientific hypotheses, experimental design, data collection, interpretation of results, and intellectual contributions are the authors' own work. AI-generated code was reviewed, tested, and validated by the author prior to use. All code is publicly available in github repository (https://github.com/neurokinetikz/schumann).

---

## 1 Introduction

### 1.1 Transient Synchronization Phenomena in EEG

Neural oscillations are fundamental to brain function, coordinating activity across distributed networks to support perception, cognition, and behavior (Buzsaki, 2006; Fries, 2015). These rhythmic fluctuations in neuronal population activity occur across characteristic frequency bands---from slow delta oscillations (< 4 Hz) through high gamma (> 80 Hz)---each associated with distinct functional roles (Cohen, 2017; Engel & Fries, 2010; Klimesch, 1999). Although substantial research has characterized sustained oscillatory states and their behavioral correlates, transient synchronization phenomena---brief episodes of enhanced coordination across brain regions---have received comparatively less systematic investigation despite their potential significance for understanding rapid neural state transitions and information integration (Canolty & Knight, 2010; Siegel et al., 2012).

Well-characterized examples of transient oscillatory events include sleep spindles (Fernandez & Luthi, 2020), hippocampal sharp-wave ripples (Buzsaki, 2015), and task-related gamma bursts (Fries et al., 2001). These phenomena share common features: discrete temporal boundaries, elevated power relative to baseline, and enhanced inter-regional synchronization. Such transient events may carry functional significance distinct from sustained oscillations, potentially serving as temporal windows for memory consolidation, inter-regional communication, or state transitions (J. Lisman & Buzsaki, 2008).

The overlap between Earth's Schumann Resonance frequencies (fundamental ~7.83 Hz with harmonics at ~14.3, 20.8, 27.3, and 33.8 Hz) and canonical EEG bands has prompted investigation into potential brain-environment relationships (Cherry, 2002; Konig et al., 1974; Persinger, 2008; Saroka et al., 2016). However, prior studies have mainly employed correlational designs examining continuous SR-EEG relationships rather than systematically detecting and characterizing discrete events. The temporal structure, duration, and network dynamics of individual SR-frequency episodes have remained poorly characterized.

During exploratory analysis of EEG recordings from meditation sessions collected between 2019 and 2024, we repeatedly observed transient periods of elevated power spectral density at approximately 7.6--7.8 Hz across multiple brain regions. These events involved coordinated power increases at multiple frequencies, accompanied by marked elevation in inter-channel coherence and phase synchronization---suggesting network-wide coordination rather than isolated oscillatory bursts. This discovery prompted systematic investigation, revealing an unexpected mathematical structure in the frequency relationships that forms the basis of the present paper.

### 1.2 The Problem of EEG Frequency Band Definitions

Since Hans Berger's discovery of the human alpha rhythm in 1929, electroencephalography has been organized around a taxonomy of frequency bands: delta (< 4 Hz), theta (4--8 Hz), alpha (8--13 Hz), beta (13--30 Hz), and gamma (> 30 Hz). These bands emerged from empirical observation and clinical utility rather than theoretical derivation.

A systematic review of band definitions across published EEG studies reveals substantial inconsistency:

**Table 1: Variability in EEG Band Definitions Across Published Literature**

| Band | Lower Bound Range | Upper Bound Range | Variation |
|------|-------------------|-------------------|-----------|
| Delta | 0.1--1.0 Hz | 3.0--4.0 Hz | 1.0 Hz |
| Theta | 3.0--5.0 Hz | 7.0--8.0 Hz | 3.0 Hz |
| Alpha | 7.0--9.0 Hz | 12.0--14.0 Hz | 4.0 Hz |
| Beta | 12.0--15.0 Hz | 25.0--35.0 Hz | 13.0 Hz |
| Gamma | 25.0--40.0 Hz | 80.0--150.0 Hz | 85.0 Hz |

This variability suggests that current band definitions may not reflect natural categories in neural organization. If frequency bands represented genuine biological discontinuities, we would expect convergence on boundary positions. The observed divergence implies either (a) bands are arbitrary conveniences, or (b) the true boundaries exist but have not been identified.

As will be demonstrated in this paper, the discovery of golden ratio frequency ratios in transient high-coherence events provides a principled mathematical framework that may resolve this long-standing ambiguity.

### 1.3 Schumann Resonance as Potential Reference

The Earth's ionospheric cavity supports a set of global electromagnetic resonances, termed Schumann Resonances (SR), arising from electromagnetic waves circling the Earth-ionosphere waveguide driven primarily by global lightning activity (Nickolaenko & Hayakawa, 2014; Schumann, 1952). For a simplified cavity model, the fundamental frequency is:

> f₁^approx = c / (2πr) = 299,792,458 / (2π × 6,371,000) = 7.49 Hz     (1)

where c is the speed of light and r is Earth's radius.

Long-term geophysical monitoring provides direct measurement of SR parameters. The Tomsk Space Observing System (Russia), operating continuously since 1994, reports the fundamental centered at 7.6 ± 0.2 Hz with diurnal and seasonal variations of ±0.3 Hz. The "canonical" value of 7.83 Hz frequently cited in literature represents a theoretical approximation; empirical monitoring consistently yields lower values near 7.6 Hz.

The proximity of the SR fundamental (~7.6 Hz) to the theta-alpha boundary in human EEG has been noted in previous work. However, investigations have been limited by lack of a theoretical framework specifying *what pattern* to expect if neural frequencies were organized relative to SR. The present study arose from the unexpected discovery of precise mathematical relationships in the harmonic structure of transient neural events---relationships that suggested a framework based on the golden ratio rather than simple integer harmonics.

### 1.4 Study Overview: From Discovery to Validation

This paper follows the actual trajectory of discovery:

**Phase 1 (Discovery):** During analysis of meditation EEG recordings, we detected transient high-coherence events showing multi-band synchronization at specific frequencies falling within the Schumann Resonance range. We termed these "Schumann Ignition Events" (SIEs). Initial characterization revealed 1,366 events across 91 subjects and five cognitive contexts, with robust network synchronization signatures.

**Phase 2 (Pattern Recognition):** Upon examining the harmonic frequencies detected in SIE events---particularly the relationships between SR1 (~7.6 Hz), SR3 (~20 Hz), and SR5 (~32 Hz)---we discovered that their ratios approximated powers of the golden ratio (φ = 1.618) with less than 1% error. This was unexpected; we had anticipated integer harmonic relationships if any mathematical structure existed.

**Phase 3 (Hypothesis Generation):** The discovery of φⁿ ratios in transient events raised a fundamental question: Is this organization specific to high-coherence states, or does it reflect a deeper architectural principle governing all EEG frequency organization? We formalized the hypothesis as f(n) = f₀ × φⁿ and derived specific predictions for how spectral peaks should be distributed.

**Phase 4 (Validation):** Systematic analysis of 244,955 oscillatory peaks across 968 EEG sessions confirmed the predicted boundary-attractor structure: depletion at integer φⁿ positions, enrichment at half-integer positions, and maximal enrichment at "noble" positions (n + 0.618).

**Phase 5 (Synthesis):** Convergent evidence from both temporal scales---transient events and continuous spectral architecture---supports a "substrate-ignition" model in which the φⁿ lattice exists continuously, with SIEs representing transient amplification of this underlying structure.

The following sections present these phases in sequence, beginning with the full methodological and empirical characterization of SIE discovery (Section 2), followed by the theoretical framework that emerged from that discovery (Section 3), the validation study testing predictions in continuous EEG (Section 4), and the integration of both lines of evidence (Section 5).

## 2 Study 1: Discovery and Characterization of Schumann Ignition Events

### 2.1 Methods: SIE Detection and Analysis

#### 2.1.1 Overview

This study employed a two-phase design: (1) discovery and initial characterization of Schumann Ignition Events (SIEs) in longitudinal meditation EEG recordings, and (2) validation and extension in independent datasets collected during cognitive tasks. All analyses were performed in Python 3.9+ using NumPy, SciPy, MNE-Python (Gramfort et al., 2013; Harris et al., 2020; Virtanen et al., 2020), and the FOOOF library (Donoghue et al., 2020).

#### 2.1.2 Participants and Recordings

**Discovery Dataset: Longitudinal Meditation Recordings.** The initial discovery and characterization of SIEs was conducted using EEG recordings from a single experienced meditation practitioner (male, age range 44--48 years, >10 years meditation experience). Recordings were collected during eyes-closed meditation sessions spanning November 2019 to March 2024, comprising 34 sessions with a total recording duration of approximately 7.5 hours. Three consumer-grade EEG devices were used across sessions to enable cross-device validation:

- **Muse (2016 model; InteraXon Inc.)**: 4 channels (TP9, AF7, AF8, TP10; dry electrodes), 256 Hz sampling rate, FPz reference, 17 sessions
- **Emotiv EPOC X (Emotiv Inc.)**: 14 channels (AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4; saline-soaked felt pads), 128 Hz sampling rate, CMS/DRL reference, 5 sessions
- **Emotiv Insight (Emotiv Inc.)**: 5 channels (AF3, AF4, T7, T8, Pz; semi-dry polymer sensors), 128 Hz sampling rate, CMS/DRL reference, 12 sessions

**Validation Datasets.** To test generalization beyond meditation, we analyzed three publicly available datasets:

- **PhySF** (Albuquerque et al., 2024): N = 26 participants performing mathematical tasks designed to induce flow states, recorded with Emotiv EPOC X
- **MultiPENG** (Albuquerque et al., 2023): N = 36 participants playing video games across varying difficulty levels
- **VEP** (Faisal et al., 2023): N = 29 participants viewing natural object images (Emotiv EPOC X)

**Table 2: Combined Dataset Characteristics**

| Dataset | Participants | Sessions | SIE Events | Device(s) | Context |
|---------|-------------|----------|------------|-----------|---------|
| Meditation (Files) | 1 | 5 | 38 | EPOC X | Eyes-closed meditation |
| Meditation (Muse) | 1 | 17 | 165 | Muse | Eyes-closed meditation |
| Meditation (Insight) | 1 | 12 | 57 | Insight | Eyes-closed meditation |
| PhySF | 25 | 46 | 447 | EPOC X | Cognitive flow/non-flow tasks |
| MultiPENG | 36 | 533 | 604 | EPOC X | Video game engagement |
| VEP | 29 | 48 | 55 | EPOC X | Passive visual perception |
| **Combined** | **91** | **661** | **1,366** | **3 types** | **5 contexts** |

#### 2.1.3 EEG Preprocessing

Raw EEG signals were high-pass filtered at 1 Hz using a 4th-order zero-phase Butterworth filter (Butterworth, 1930) to remove slow drifts while preserving oscillatory activity in the Schumann Resonance frequency range. All electrodes were positioned according to the international 10-20 system (Jasper, 1958).

#### 2.1.4 Schumann Resonance Harmonic Detection

**Canonical Frequencies.** We derived canonical frequencies and search bandwidths from long-term magnetometer data recorded by the Space Observing System in Tomsk, Russia: SR1 = 7.6 Hz (±0.6 Hz), SR2o = 13.75 Hz (±0.75 Hz), SR3 = 20 Hz (±1.0 Hz), SR4 = 25 Hz (±2.0 Hz), and SR5 = 32 Hz (±2.5 Hz).

**Spectral Parameterization Using FOOOF.** To separate oscillatory (periodic) activity from aperiodic (1/f) background noise, we employed FOOOF spectral parameterization (Donoghue et al., 2020):

> log₁₀(PSD(f)) = L(f) + Σᵢ Gᵢ(f)     (2)

where L(f) = b − χ × log₁₀(f) represents the aperiodic component and Gᵢ(f) represents Gaussian periodic peaks. FOOOF parameters: frequency range 1--50 Hz, maximum 20 peaks, minimum peak height 0.0001 log units, peak width limits 0.2--20.0 Hz.

#### 2.1.5 Seven-Stage Detection Pipeline

We developed a seven-stage pipeline to systematically identify, refine, and validate candidate SIE events:

**Stage 1: Envelope Thresholding.** A composite SR envelope was computed by averaging all EEG channels, bandpass filtering at the fundamental frequency (f₀ ± 0.6 Hz), computing the Hilbert envelope, and z-score normalizing. Candidate onsets were identified where z(t) ≥ 3.0.

**Stage 2: FOOOF Harmonic Refinement.** For each candidate window, FOOOF was applied to extract session-specific harmonic frequencies within SR search windows.

**Stage 3: Kuramoto Phase Synchronization.** The precise ignition onset t₀ was refined using the Kuramoto order parameter:

> R(t) = |1/N × Σₖ₌₁ᴺ e^(iφₖ(t))|     (3)

where φₖ(t) is the instantaneous phase of channel k at SR1 frequency. Ignition onset was defined as the time of maximum dR/dt where R(t) ≥ 0.6.

**Stage 4: Coherence Characterization.** Magnitude-squared coherence (MSC) and phase-locking value (PLV) (Lachaux et al., 1999) were computed for each harmonic:

> PLV = 1/T |Σₜ₌₁ᵀ e^(iΔφₓᵧ(t))|     (4)

**Stage 5: Harmonic Organization.** The Harmonic Stack Index (HSI) quantified the ratio of overtone power to fundamental power: HSI = P_overtones / P_f.

**Stage 6: Quality Metrics.** Spectral slope (β) and Frequency Specificity Index (FSI = P_SR / P_total) were computed.

**Stage 7: Composite Scoring.** A composite SR-Score was computed using the three observed Schumann harmonics (SR1, SR3, SR5):

> SR-Score = (Z_weighted^0.7 × MSC_weighted^1.2 × PLV_weighted) / (1 + HSI)     (5)

with φ-derived weights: SR1 (0.618), SR3 (0.326), SR5 (0.146).

#### 2.1.6 Null Control Analyses

Six null control analyses validated that SIE characteristics represent genuine neural organization:

**Null Control A (Phase Randomization):** For each SIE, phases were randomized while preserving amplitude spectra (100 surrogates per event), testing whether synchronization reflects genuine phase coupling.

**Null Control B (Random Temporal Windows):** 1,366 random 3-second baseline windows were sampled from periods >10 seconds from any SIE, testing whether SR organization is event-specific.

**Null Control C (Uniform Random Triplets):** 10,000 random frequency triplets sampled uniformly from physiologically plausible ranges tested whether φ precision arises by chance.

**Null Control D (Peak-Based Random Triplets):** 10,000 triplets sampled from actual FOOOF peaks (N = 244,955 peaks from 661 sessions) provided a more stringent test controlling for EEG spectral structure.

**Null Control E (Shuffled Bootstrap):** SR1, SR3, SR5 frequencies were independently permuted across events (1,000 iterations), testing whether ratio precision requires within-event coordination.

**Null Control F (Distributional Null Model):** Synthetic datasets generated from observed frequency distributions tested population-level encoding.

### 2.2 Results: SIE Characteristics and Harmonic Structure

#### 2.2.1 Dataset Overview and Event Detection

Application of the seven-stage detection pipeline yielded 1,366 discrete events across 661 recording sessions from 91 unique participants (Table 2). Events were detected across all six datasets and five cognitive contexts, with overall mean detection rate of 2.1 ± 3.3 events per session (range: 1--27). Event duration averaged 26.9 ± 21.3 seconds (median: 20.0 seconds), consistent with the transient nature of SIE phenomena and characteristic timescales of transient oscillatory events (Siegel et al., 2012).

PhySF cognitive flow recordings and meditation recordings yielded the highest per-session detection rates (9.7 and 7.6 events/session respectively), with gaming and visual perception tasks showing lower rates (~1 event/session). This pattern may reflect differential engagement of SR-aligned neural mechanisms across cognitive states.

#### 2.2.2 Harmonic Frequency Definitions

Spectral analysis of detected SIEs revealed nine distinct harmonic frequencies spanning the 7--41 Hz range (Table 3). The fundamental frequency (SR1) centers near 7.6 Hz at the theta/alpha boundary, with higher harmonics extending through alpha, beta, and gamma bands.

**Table 3: Harmonic Frequency Definitions from SIE Analysis**

| Harmonic | Measured (Hz) | CV (%) | N | EEG Band |
|----------|---------------|--------|------|----------|
| SR1 | 7.63 ± 0.33 | 4.35 | 1121 | θ/α Boundary |
| SR1.5 | 9.96 ± 0.35 | 3.56 | 1139 | α Attractor |
| SR2 | 11.98 ± 0.40 | 3.37 | 1050 | α/β Boundary |
| SR2o | 13.76 ± 0.44 | 3.17 | 1150 | Low β |
| SR2.5 | 15.50 ± 0.46 | 2.99 | 1188 | β Attractor |
| SR3 | 19.99 ± 0.57 | 2.84 | 1250 | β Boundary |
| SR4 | 24.98 ± 1.17 | 4.69 | 1359 | High β Attractor |
| SR5 | 32.57 ± 1.23 | 3.77 | 1364 | β/γ Boundary |
| SR6 | 40.11 ± 1.61 | 4.01 | 1365 | γ Attractor |

CV = coefficient of variation. N = events with detectable harmonic.

**Figure 1:** Exemplar SIE event spectral and temporal dynamics. (A) Time-frequency spectrogram showing harmonic activation at φⁿ frequencies. (B) Harmonic piano-roll showing z-score elevation across all six harmonics. (C) Envelope traces overlaid with phase-locking value, demonstrating coherence-first dynamics. (D) Harmonic Stack Index showing fundamental dominance (decreasing HSI) preceding ignition.

#### 2.2.3 Synchronization Metrics

SIE events exhibited robust network synchronization. At the fundamental frequency SR1, 81.5% of events showed phase-locking values exceeding 0.6, and 60.5% showed magnitude-squared coherence above 0.5. Both MSC (0.552) and PLV (0.689) showed gradients decreasing from SR1 to SR6, with SR1 exhibiting highest synchronization---confirming whole-brain coordination during SIE events consistent with communication through coherence (Fries, 2015).

Power spectral elevation at SR harmonics was quantified using z-scores relative to session baselines. The fundamental frequency SR1 showed strongest elevation (z = 4.76 ± 2.95), with systematic decrease at higher harmonics, confirming multi-band spectral enhancement during SIE events.

#### 2.2.4 Temporal Dynamics: The Coherence-First Signature

Individual SIE events exhibited a stereotyped six-phase temporal evolution:

1. **Baseline:** Low-to-moderate global synchronization (R ≈ 0.4--0.6)
2. **Coherence Rise:** Rapid increase in phase alignment preceding amplitude changes, with R typically rising to > 0.7 over 1--3 seconds
3. **Plateau:** Brief stabilization (0.5--2 seconds) at elevated coherence (R > 0.8) prior to amplitude ignition
4. **Ignition:** Coincident peak in both coherence and power, with simultaneous z-score elevation across multiple harmonics
5. **Propagation:** Sustained high-coherence state (5--15 seconds) with gradual spread of synchronization
6. **Decay:** Gradual return to baseline over 3--8 seconds

The coherence-first signature---phase alignment preceding amplitude elevation by 2--3 seconds---is consistent with network-level coordination rather than local oscillatory bursts (Varela et al., 2001).

**Figure 2:** Exemplar SIE event network dynamics. (G) Harmonic envelope comparison between baseline and ignition states. (H) EEG-SR coherence by electrode and harmonic. (I) State-space trajectory showing coherence vs. power. (J) Kuramoto order parameter time series. (K) Transfer entropy matrix. (L) Cross-scale theta-gamma correlation dynamics.

#### 2.2.5 Cross-Device and Cross-Context Validation

One-way ANOVA tested whether harmonic frequencies differed across devices and contexts:

**Device Independence:** SR1 and SR3 showed no significant device differences (p > 0.25), supporting biological rather than device-specific origins. SR5 exhibited a modest device effect (F = 5.74, p = 0.003), with Muse devices detecting slightly higher frequencies (32.88 Hz vs. 32.53--32.54 Hz for Emotiv devices)---likely reflecting differential hardware filtering at higher frequencies.

**Context Independence:** One-way ANOVA across five cognitive contexts (meditation, flow, non-flow, gaming, visual) revealed no significant frequency differences for any primary harmonic (all p > 0.15). The φⁿ architecture is preserved across contexts spanning eyes-closed meditation, cognitively demanding flow states, gaming engagement, and passive visual perception.

### 2.3 The Emergent Pattern: Golden Ratio Frequency Ratios

#### 2.3.1 The Discovery

Upon examining the nine harmonic frequencies detected across events (Table 3), a striking pattern emerged: the measured frequencies align closely with a geometric series f(n) = f₀ × φⁿ where φ = 1.618034 (the golden ratio). Using an empirically-derived fundamental frequency f₀ = 7.6 Hz, measured frequencies matched theoretical predictions across integer, half-integer, and quarter-integer values of n.

**Table 4: Predicted vs. Measured Frequencies Using f(n) = 7.6 × φⁿ**

| n | φⁿ | Predicted Frequency (Hz) | Harmonic |
|---|-----|--------------------------|----------|
| 0 | 1.000 | 7.60 | SR1 (fundamental) |
| 2 | 2.618 | 19.90 | SR3 |
| 3 | 4.236 | 32.19 | SR5 |

#### 2.3.2 Golden Ratio Precision in Harmonic Ratios

The φⁿ architecture predicts specific ratios between harmonics that are independent of f₀ choice, providing the most stringent test of golden ratio organization:

**Table 5: Harmonic Ratio Precision: Measured vs. φⁿ Predictions**

| Ratio | Predicted | Measured | Error (%) | 95% CI |
|-------|-----------|----------|-----------|--------|
| SR3/SR1 = φ² | 2.6180 | 2.6223 ± 0.134 | +0.16 | [2.614, 2.630] |
| SR5/SR1 = φ³ | 4.2361 | 4.2766 ± 0.249 | +0.96 | [4.262, 4.291] |
| SR5/SR3 = φ | 1.6180 | 1.6309 ± 0.078 | +0.79 | [1.627, 1.635] |
| SR6/SR4 = φ | 1.6180 | 1.6094 ± 0.100 | −0.53 | [1.604, 1.615] |

**Mean absolute ratio error: 0.61%.** All four ratios deviated less than 1% from predicted φⁿ values. Bootstrap analysis (N = 10,000 iterations) confirmed that the 95% confidence interval for SR3/SR1 (2.614--2.630) includes the predicted φ² value (2.618).

#### 2.3.3 The Independence-Convergence Paradox

A critical question emerged: Do the three primary harmonics (SR1, SR3, SR5) vary together proportionally, or independently? If frequencies were proportionally coupled, we would expect strong positive correlations across events. Instead, pairwise Pearson correlations revealed **complete independence**:

**Table 6: Frequency Correlations Across Events: The Independence-Convergence Paradox**

| Pair | Pearson r | p-value | Interpretation |
|------|-----------|---------|----------------|
| SR1 vs SR3 | +0.030 | 0.33 | No correlation |
| SR1 vs SR5 | −0.002 | 0.94 | No correlation |
| SR3 vs SR5 | −0.020 | 0.47 | No correlation |

All correlations were statistically non-significant (all |r| < 0.03, all p > 0.3), demonstrating that harmonic frequencies vary completely independently across events. An event with high SR1 frequency is no more likely to have high SR3 or SR5 frequencies than expected by chance.

> **Key Insight**
>
> **The Paradox:** Individual harmonic frequencies vary completely independently across events, yet their **ratios** maintain < 1% precision. This pattern cannot arise from coupled oscillators (which would show frequency covariation) nor from independent rhythms (which would degrade ratio precision).
>
> **The Resolution:** φⁿ relationships are encoded in the **population-level marginal distributions** of each harmonic rather than through event-level coordination. Each oscillatory generator is independently constrained to a frequency distribution whose mean satisfies φⁿ relationships relative to other harmonics.

#### 2.3.4 Null Control Validation

The peak-based null control (Null Control D) provided the most stringent test. Comparing SIE frequency triplets against 10,000 random triplets sampled from actual FOOOF peaks (N = 244,955 peaks):

- Observed mean φ-error: 4.37% ± 2.00%
- Random peak triplet mean error: 9.05% ± 4.15%
- Cohen's d: 1.44 (large effect)
- p-value: < 0.0001 (permutation test)

SIE events show **significantly better φ-precision** than random triplets sampled from actual EEG spectral peaks. The highly significant result (p < 0.0001) with large effect size (d = 1.44) indicates that SIE detection preferentially identifies frequency combinations exhibiting exceptional φⁿ organization (Figure 3).

**Figure 3:** Peak-based null control results. (A) Histogram comparing φ-error distributions for SIE events (red) versus random triplets sampled from actual FOOOF peaks (gray). SIE events cluster at lower φ-error values. (B) Box plot comparison showing significantly lower φ-error for SIE events. (C) Cumulative distribution functions demonstrating separation between distributions. (D) Percentile comparison across the error distribution.

#### 2.3.5 The Emergent Question

The discovery of precise φⁿ ratio organization in transient high-coherence events raised a fundamental question that motivated the second phase of this research:

*Is φⁿ organization specific to transient SIE events, or does it reflect a deeper architectural principle governing **all** EEG frequency organization?*

If the latter, we would expect the φⁿ boundary-attractor structure to be visible in the aggregate distribution of oscillatory peaks across the full spectrum, not only during SIE events. The theoretical framework and validation study presented in the following sections test this prediction.

## 3 Theoretical Framework: The φⁿ Hypothesis

The discovery of precise golden ratio relationships in SIE harmonic frequencies (Section 2.3) raised a fundamental question: Does this organization reflect a deeper architectural principle governing all EEG frequency organization? We formalized the pattern into a testable theoretical framework.

### 3.1 Golden Ratio Mathematics and Neural Relevance

#### 3.1.1 Definition and Properties

The golden ratio is defined as the positive solution to x² = x + 1:

> φ = (1 + √5) / 2 = 1.6180339887...     (6)

The golden ratio possesses two unique mathematical properties relevant to coupled oscillator systems:

**Property 1: Maximal Anti-Commensurability.** The golden ratio has the continued fraction representation φ = [1; 1, 1, 1, ...], which converges more slowly than any other irrational. This means the rational approximations to φ (the Fibonacci ratios 1/1, 2/1, 3/2, 5/3, 8/5...) are the *worst possible* among all irrationals of comparable magnitude.

*Implication:* Coupled oscillators at frequency ratio φ maximally resist mode-locking. The phase relationship never settles into a stable pattern because φ cannot be well-approximated by any simple ratio. This enables **segregation**: oscillators at different φⁿ positions maintain independence rather than collapsing into synchrony.

**Property 2: Fibonacci Additivity.** The golden ratio uniquely satisfies:

> φⁿ = φⁿ⁻¹ + φⁿ⁻² for all n ∈ ℤ     (7)

No other positive real number satisfies this property.

*Implication:* At frequency f(n) = f₀ × φⁿ, the three frequencies f(n), f(n−1), and f(n−2) satisfy an exact sum relationship, enabling three-wave resonant energy transfer. This creates "gateways" for cross-frequency communication at specific positions, enabling **integration**.

> **Key Insight**
>
> **Why φ May Be Unique for Neural Organization:** No other scaling factor provides both properties. Integer ratios (2:1, 3:2) enable coupling but produce mode-locking; other irrationals (√2, e) resist mode-locking but lack Fibonacci coupling. If neural systems require both independent processing within bands and communication between bands, φⁿ organization may represent the unique optimal solution---a "segregation-integration balance" that no other architecture achieves.

### 3.2 The Core Equation: f(n) = f₀ × φⁿ

Based on the SIE discovery, we formalized the hypothesis that EEG frequencies are organized according to:

> **Key Equation**
>
> f(n) = f₀ × φⁿ     (8)
>
> where:
> - f₀ = 7.60 Hz (measured Schumann Resonance fundamental from Tomsk Observatory)
> - φ = 1.6180339887... (golden ratio)
> - n ∈ ℝ (continuous position index)

**Convergent f₀ Estimates.** The fundamental frequency f₀ = 7.60 Hz derives from two independent sources:

- **Geophysical:** Multi-year monitoring at Tomsk Space Observing System reports SR fundamental at 7.6 ± 0.2 Hz
- **Neural (SIE):** Mean SR1 frequency across 1,121 SIE events was 7.63 ± 0.33 Hz

The agreement within 0.03 Hz (0.4%) between independent neural and geophysical estimates provides mutual validation.

**Key Conceptual Claim.** The hypothesis concerns φⁿ *ratio architecture*, not absolute frequency locking. Both Schumann Resonance (±0.3 Hz diurnal variation) and neural oscillations (individual differences, state fluctuations) are naturally variable. The claim is that **ratio relationships** (boundary depletion, attractor enrichment) persist across this natural variability---ratios, not frequencies, are the deep invariant.

### 3.3 Predictions for Continuous Spectral Organization

#### 3.3.1 Position Classification

The framework classifies positions by their n value:

**Table 7: Position Classification and Predicted Behavior**

| Position Type | n Value | Mathematical Property | Predicted Behavior |
|---------------|---------|----------------------|-------------------|
| **Boundary** | Integer (k) | Fibonacci sum point | **Depletion** (unstable) |
| 2° Noble | k + 0.382 | φ⁻² from next boundary | Modest enrichment |
| **Attractor** | k + 0.5 | Log-midpoint | **Enrichment** (stable) |
| 1° Noble | k + 0.618 | φ⁻¹ from next boundary | **Strong enrichment** |

**Why half-integer positions are stable:** At a boundary (integer n), the Fibonacci property f(n) = f(n − 1) + f(n − 2) enables three-wave energy transfer, destabilizing sustained oscillations. Half-integer positions are maximally distant from these coupling points---oscillations "hide" from cross-frequency energy redistribution.

#### 3.3.2 A Priori Frequency Landmarks

**Table 8: A Priori Frequency Predictions (Generated Before Data Analysis)**

| n | φⁿ | f(n) Hz | Type | Predicted Peak Count |
|---|-----|---------|------|---------------------|
| −1 | 0.618 | 4.70 | Boundary | LOW (δ/θ border) |
| −0.5 | 0.786 | 5.97 | Attractor | ELEVATED (θ center) |
| 0 | 1.000 | 7.60 | Boundary | LOW (θ/α border) |
| **+0.5** | 1.272 | **9.67** | **Attractor** | **HIGH (α peak)** |
| +1 | 1.618 | 12.30 | Boundary | LOW (α/β border) |
| +1.5 | 2.058 | 15.64 | Attractor | ELEVATED (low β) |
| +2 | 2.618 | 19.90 | Boundary | LOW or ELEVATED* |
| +2.5 | 3.330 | 25.31 | Attractor | ELEVATED (high β) |
| **+3** | 4.236 | **32.19** | **Boundary** | **LOW (β/γ trough)** |
| **+3.5** | 5.388 | **40.95** | **Attractor** | **HIGH (γ peak)** |
| +4 | 6.854 | 52.09 | Boundary | LOW (high γ border) |

\*The φ² position may show resonance enhancement due to Fibonacci three-wave coupling

#### 3.3.3 Predicted Band-Specific Heterogeneity

While the φⁿ architecture is predicted across all frequency bands, we expect **frequency-dependent expression strength**:

- **Gamma band (φ³--φ⁴; 32--52 Hz):** Should show the *strongest* φⁿ alignment. Gamma oscillations subserve temporal binding and feature integration, requiring precise phase relationships within ~25 ms windows. The GABA-A receptor decay constant (τ ≈ 25 ms) creates a natural oscillatory bottleneck near 40 Hz. Strict φⁿ compliance ensures that distributed gamma oscillators remain independent rather than collapsing into trivial synchrony.

- **Alpha band (φ⁰--φ¹; 7.6--12.3 Hz):** Should show *moderate* alignment, potentially obscured by individual alpha frequency variability (±1 Hz across subjects). Alpha functions primarily as a gating mechanism, which may tolerate broader frequency flexibility.

- **Theta/Delta bands (< φ⁰):** Should show the *weakest* alignment. These slower oscillations support memory consolidation and homeostatic functions that integrate information over longer timescales, permitting greater frequency variability.

This gradient predicts that φⁿ architecture manifests as a **universal position hierarchy** (1° Noble > Attractor > 2° Noble) with **frequency-dependent magnitude** (gamma ≫ alpha > theta > delta).

## 4 Study 2: Testing the φⁿ Hypothesis in Continuous EEG

The theoretical framework derived from SIE discovery (Section 3) generated specific predictions for continuous spectral organization. If φⁿ architecture is fundamental rather than event-specific, aggregate peak distributions should show boundary depletion, attractor enrichment, and the predicted noble position hierarchy.

### 4.1 Methods: Spectral Parameterization and Analysis

#### 4.1.1 Extended Dataset

The analysis utilized the same recording sessions as Study 1, extended to include all available EEG data regardless of SIE detection. The complete dataset comprised 968 sessions, approximately 96 subjects, and 34.2 hours of recording across three device types and four cognitive contexts.

#### 4.1.2 FOOOF Spectral Parameterization

We employed FOOOF (Fitting Oscillations and One-Over-F) (Donoghue et al., 2020) for spectral parameterization. FOOOF separates the power spectrum into:

> P(f) = L(f) + Σᵢ Gᵢ(f)     (9)

where L(f) = b − log(f^χ) is the aperiodic component and Gᵢ(f) are Gaussian peaks with center frequency μᵢ, amplitude aᵢ, and bandwidth σᵢ.

FOOOF parameters: frequency range 1--50 Hz, peak width limits 0.2--20.0 Hz, maximum 20 peaks, minimum peak height 0.0001 log units. Peaks were retained if power > 0.1 log units above aperiodic fit, bandwidth 1--8 Hz, and frequency within 1--48 Hz.

#### 4.1.3 Lattice Coordinate Analysis

To analyze peaks independent of band-specific effects, we computed the lattice coordinate for each peak:

> u = [log_φ(f/f₀)] mod 1     (10)

This maps all frequencies to the unit interval [0, 1), where u = 0 corresponds to boundaries, u = 0.5 to attractors, and u = 0.618 to primary noble positions.

### 4.2 Results: Position-Type Enrichment and Band Structure

#### 4.2.1 Peak Detection Results

FOOOF parameterization yielded **244,955 oscillatory peaks** across 968 sessions, with a median peak frequency of 21.2 Hz and mean of 18.5 peaks per channel (Figure 4).

**Figure 4:** Distribution of 244,955 spectral peaks detected via FOOOF across 968 sessions. Vertical lines indicate φⁿ frequency predictions: boundaries at integer n (solid orange), attractors at half-integer n (dashed red), 1° nobles at n + 0.618 (dotted teal), and 2° nobles at n + 0.382 (dashed olive). Shaded regions demarcate traditional EEG bands.

#### 4.2.2 Position-Type Enrichment

The aggregate peak distribution confirmed all theoretical predictions:

**Table 9: Position-Type Enrichment: Predicted vs. Observed**

| Position Type | n Value | Predicted | Observed | 95% CI |
|---------------|---------|-----------|----------|--------|
| Boundary | Integer | Depletion | −18.0% | [−19.1%, −17.0%] |
| 2° Noble | k + 0.382 | Modest enrichment | +1.9% | [+0.6%, +3.2%] |
| Attractor | k + 0.5 | Enrichment | +21.0% | [+19.7%, +22.4%] |
| 1° Noble | k + 0.618 | Strong enrichment | +39.0% | [+37.7%, +40.4%] |

The observed ordering (Boundary < 2° Noble < Attractor < 1° Noble) exactly matches theoretical predictions. Kendall's τ = 1.0 (perfect agreement), p = 0.042 (Figure 5).

**Figure 5:** Position-type enrichment in the φⁿ framework with 95% bootstrap confidence intervals (n = 243,704 peaks). Bars show percent deviation from uniform expectation. Boundaries (integer n): −18.0% [CI: −19.1% to −17.0%]. 1° nobles (n + 0.618): +39.0% [CI: +37.7% to +40.4%]. Attractors (n + 0.5): +21.0% [CI: +19.7% to +22.4%]. The monotonic ordering (Boundary < 2° Noble < Attractor < 1° Noble) confirms theoretical predictions.

#### 4.2.3 Alignment with Predicted Landmarks

Key spectral landmarks aligned precisely with φⁿ predictions:

- **Alpha peak:** Observed 9.8 Hz, predicted φ^0.5 = 9.67 Hz (error: +0.13 Hz)
- **Beta-gamma trough:** Observed 32.4 Hz, predicted φ³ = 32.19 Hz (error: +0.21 Hz)
- **Gamma recovery:** Observed 41.2 Hz, predicted φ^3.5 = 40.95 Hz (error: +0.25 Hz)

Mean absolute prediction error: 0.17 ± 0.09 Hz across all positions.

#### 4.2.4 Band-Specific Heterogeneity

Stratified analysis by φⁿ-defined bands revealed frequency-dependent enrichment strength, confirming the predicted gradient:

**Table 10: Band-Specific 1° Noble Enrichment**

| Band | n Range | Peaks | 1° Noble Enrichment |
|------|---------|-------|---------------------|
| Delta | φ⁻² to φ⁻¹ | 3,140 | −12.1% |
| Theta | φ⁻¹ to φ⁰ | 15,315 | +2.2% |
| Alpha | φ⁰ to φ¹ | 39,189 | +4.2% |
| Low Beta | φ¹ to φ² | 54,709 | +3.9% |
| High Beta | φ² to φ³ | 71,641 | +8.8% |
| **Gamma** | φ³ to φ⁴ | 58,774 | **+144.8%** |

**Gamma dominance:** The gamma band showed dramatically stronger φⁿ adherence (+144.8% at 1° noble) than any other band (Figure 6). This is mechanistically required: gamma oscillations subserve temporal binding and feature integration within ~25 ms windows, demanding precise phase relationships (Bartos et al., 2007). Lower-frequency bands serve functions tolerating greater flexibility.

**Figure 6:** Band-stratified lattice coordinate analysis. Six panels show histograms of the fractional lattice position u = [log_φ(f/f₀)] mod 1 within each φⁿ-defined frequency band. Vertical lines mark theoretically significant positions. Gamma shows the strongest 1° noble enrichment (+145%), while alpha shows weak effects due to individual alpha frequency variability.

### 4.3 Validation: Cross-Device and Cross-Context Consistency

#### 4.3.1 Alternative Scaling Factor Comparison

To test whether φ is uniquely optimal, we compared five alternative scaling factors (e, π, √2, 2, 1.5) at f₀ = 7.6 Hz.

**Critical finding:** φ is the *only* scaling factor exhibiting both (1) the highest alignment score, and (2) the theoretically predicted ordering (1° Noble > Attractor > 2° Noble > Boundary). Bootstrap comparison confirms φ significantly outperforms all alternatives (p < 0.001; Figure 7).

**Figure 7:** Alternative scaling factor comparison at f₀ = 7.6 Hz. Bar height indicates overall alignment. Green bar indicates correct theoretical ordering (1° Noble > Attractor > 2° Noble > Boundary); orange bars indicate incorrect ordering. Only φ achieves both the highest alignment score and the correct theoretical ordering.

#### 4.3.2 Session-Level Consistency

Across 968 sessions:

- **83.3%** showed attractor enrichment > boundary enrichment
- Session-level Cohen's d = 0.89 for attractor enrichment

#### 4.3.3 f₀ Sensitivity Analysis

Varying f₀ from 7.0--8.5 Hz revealed a **0.6 Hz tolerance plateau** (7.3--7.9 Hz) where alignment remains >95% of optimal. This plateau encompasses:

- Theoretical SR: c/2πr = 7.49 Hz
- Empirical SR (Tomsk): 7.6 ± 0.2 Hz
- Neural SIE mean SR1: 7.63 ± 0.33 Hz

The tolerance plateau confirms that **ratios, not absolute frequencies**, are the preserved quantity---consistent with two inherently variable systems maintaining ratio architecture (Figure 8).

**Figure 8:** f₀ sensitivity analysis. **Top:** Comprehensive alignment metric vs. base frequency, showing optimal at 7.60 Hz (green dashed) coinciding with geophysical Schumann measurements. The canonical literature value (7.83 Hz) falls within the > 95% plateau (shaded blue). **Bottom:** Position-type enrichment breakdown showing stable boundary depletion and attractor enrichment across the plateau.

#### 4.3.4 Permutation Testing

Two complementary tests validated the findings:

**Uniform Frequency Test:** p < 0.0001 (0/10,000 permutations exceeded observed). EEG peak frequencies show significant φⁿ structure, not uniform distribution.

**Phase-Shift Test:** p = 0.21. The 0.6 Hz tolerance plateau explains this: multiple grid positions within the natural variability range achieve comparable alignment, consistent with ratio preservation (Figure 9).

**Figure 9:** Dual permutation tests for φⁿ alignment. **Left:** Phase-shift test shows high variance in null distribution (p = 0.21). **Right:** Uniform frequency test shows highly significant result (p < 0.0001), demonstrating that observed peak frequencies are not uniformly distributed but show significant φⁿ organization.

## 5 Integration: The Substrate-Ignition Model

The convergence of findings from Study 1 (SIE discovery) and Study 2 (spectral validation) reveals a unified model of neural frequency organization.

### 5.1 Convergent Evidence from Both Studies

#### 5.1.1 Independent f₀ Convergence

The most striking validation comes from the independent convergence of fundamental frequency estimates:

**Table 11: Convergent f₀ Estimates Across Independent Sources**

| Source | f₀ Estimate | Basis | Independence |
|--------|-------------|-------|-------------|
| Geophysical (Tomsk) | 7.60 ± 0.20 Hz | SR monitoring | External |
| Neural (SIE SR1) | 7.63 ± 0.33 Hz | 1,121 events | Study 1 |
| Spectral (optimal) | 7.60 Hz | 244,955 peaks | Study 2 |

The agreement within 0.03 Hz (0.4%) between independent geophysical and neural estimates provides strong mutual validation. Neither study was informed by the other during f₀ determination.

#### 5.1.2 Precision Gradient: Transient vs. Continuous

Study 1 revealed exceptionally high precision in SIE harmonic ratios (< 1% error from φⁿ predictions), while Study 2 showed more distributed organization across 244,955 peaks (±0.2 Hz tolerance). This precision gradient is expected: transient high-coherence states represent amplified expression of the continuous φⁿ substrate.

### 5.2 The Independence-Convergence Paradox

The most theoretically significant finding is the **independence-convergence paradox** revealed in Study 1:

- **Independence:** Harmonic frequencies (SR1, SR3, SR5) vary completely independently across events (all pairwise |r| < 0.03)
- **Convergence:** Despite independence, harmonic *ratios* maintain < 1% deviation from φⁿ predictions

**Resolution:** The φⁿ relationships are encoded in the **population-level marginal distributions** of each harmonic rather than through event-level coordination. Each oscillatory generator is independently constrained to a frequency distribution whose mean satisfies φⁿ relationships relative to other harmonics.

**Evidence from shuffled bootstrap:** When SR1, SR3, SR5 frequencies were independently permuted across events (destroying within-event pairings but preserving marginal distributions), ratio precision was unchanged. This demonstrates that φⁿ constraints are encoded in the distributions themselves, not in event-level coordination.

**Biological mechanism:** What biophysical process could produce population-level constraints among independently varying frequencies? One possibility is that each frequency band is constrained by local biophysical properties---membrane time constants, network architecture, receptor kinetics---that independently converge on φⁿ values. The GABA-A receptor decay time constant (~25 ms) directly determines gamma oscillation frequency (Bartos et al., 2007). If these timescales have been evolutionarily tuned to produce compatible frequency relationships, independent variation within each system could still satisfy population-level φⁿ constraints.

### 5.3 SIEs as Amplification of Continuous φⁿ Architecture

The findings support a **substrate-ignition model**:

> **Key Insight**
>
> **The Substrate-Ignition Model:**
>
> 1. A continuous φⁿ frequency lattice exists as the substrate of neural oscillatory organization
> 2. This substrate is continuously present, organizing spectral peak distributions (Study 2: 244,955 peaks)
> 3. SIE events represent **transient amplification** of this substrate, not de novo generation
> 4. During SIEs, the normally distributed peaks "snap" to tighter φⁿ compliance with higher coherence

**Evidence supporting the model:**

- Baseline windows (non-SIE periods) show the same frequency ratios with reduced coherence
- The φⁿ lattice is visible in aggregate spectral structure (Study 2) even without event detection
- SIEs show higher precision (0.61% ratio error) than aggregate measures (±0.2 Hz tolerance)
- The coherence-first temporal signature (phase alignment before amplitude) suggests network coordination recruits existing architecture

This model explains both the ubiquity of φⁿ organization (present in all spectral data) and the precision of transient events (amplified expression of the substrate).

## 6 Discussion

### 6.1 Summary of Findings

This study provides the first systematic evidence for golden ratio (φⁿ) organization of human EEG frequency architecture, discovered through a two-phase investigation:

**Phase 1 (Study 1): Discovery of SIEs.** Analysis of meditation EEG revealed transient high-coherence neural states (Schumann Ignition Events) exhibiting precise frequency organization. Across 1,366 events, 91 participants, five cognitive contexts, and three EEG devices, SIEs showed:

- Multi-band synchronization at specific harmonic frequencies (SR1, SR3, SR5)
- Harmonic ratios deviating < 1% from φⁿ predictions
- Complete independence of individual frequencies (all |r| < 0.03) despite ratio precision
- Stereotyped temporal dynamics with coherence preceding amplitude ("ignition" signature)

**Phase 2 (Study 2): Validation in continuous EEG.** Testing whether φⁿ organization extends beyond transient events, analysis of 244,955 FOOOF-detected spectral peaks confirmed:

- Boundary positions (integer n): −18% depletion
- Attractor positions (half-integer n): +21% enrichment
- Noble positions (n + 0.618): +39% enrichment
- Gamma showing strongest adherence (+144.8% at 1° noble)

The **substrate-ignition model** integrates these findings: a continuous φⁿ lattice underlies neural frequency organization, with SIEs representing transient amplification of this architecture.

### 6.2 Mechanistic Interpretations

#### 6.2.1 Why Gamma Shows Strongest Adherence

The dramatic gamma enrichment at φⁿ positions (+144.8% vs. < 10% for other bands) is not anomalous but mechanistically required. Gamma oscillations subserve temporal binding, feature integration, and conscious perception---functions demanding precise phase relationships within ~25 ms windows (Canolty & Knight, 2010; J. E. Lisman & Jensen, 2013). The GABA-A receptor decay constant (τ ≈ 25 ms) creates a natural oscillatory bottleneck near 40 Hz (Bartos et al., 2007). Strict φⁿ compliance ensures gamma oscillators remain independent (anti-mode-locking) while enabling cross-frequency integration (Fibonacci coupling).

Lower-frequency bands serve functions---gating, memory consolidation, homeostatic regulation---that tolerate or benefit from frequency flexibility. The φⁿ architecture is universal, but expression strength reflects functional requirements.

#### 6.2.2 Two Synchronization Subsystems

Analysis of cross-frequency amplitude correlations revealed two partially dissociable subsystems:

- **Theta-anchored system (SR1):** Moderate coupling to higher harmonics (r = 0.32--0.54), associated with hippocampal-cortical communication and working memory (Hyman et al., 2005)
- **Beta-gamma complex (SR3--SR6):** Tightly coupled (all r > 0.6), associated with sensorimotor integration and long-range cortical coordination (Engel & Fries, 2010)

This architecture may enable independent recruitment of theta-based and beta-gamma coordination during SIE ignition, with full integration occurring through φⁿ constraints.

### 6.3 Relationship to Schumann Resonance

The term "Schumann Ignition Events" references the correspondence between neural frequencies and Earth's Schumann Resonances (~7.83 Hz fundamental). Several aspects warrant careful interpretation:

**Correspondence, not demonstrated coupling.** The measured SR fundamental (7.83 Hz) provides the poorest fit to neural frequencies (3.28% error). The empirical f₀ = 7.6 Hz from Tomsk monitoring and neural SIE data provides better fit (1.45% error). Without concurrent magnetometer recording, we cannot assess whether SIEs correlate with actual SR activity.

**Three hypotheses:**

1. **Evolutionary tuning:** Neural frequencies evolved to match planetary electromagnetic environment
2. **Biophysical convergence:** Independent optimization to similar frequencies due to shared physical constraints (characteristic length scales, resonance conditions)
3. **Direct coupling:** Real-time electromagnetic interaction between brain and ionosphere

**Parsimony favors biophysical convergence.** Given the extremely weak SR field strengths (~picoTesla) and the absence of plausible coupling mechanisms, convergent constraints provide a more parsimonious explanation than direct electromagnetic interaction.

### 6.4 Implications for EEG Band Definitions

The φⁿ framework provides a principled basis for EEG band definitions:

**Table 12: Proposed φⁿ-Based EEG Band Definitions**

| Band | Boundaries | Center (Attractor) | 1° Noble |
|------|-----------|-------------------|----------|
| Delta | φ⁻²--φ⁻¹ (2.9--4.7 Hz) | φ⁻¹·⁵ (3.7 Hz) | φ⁻¹·³⁸ (3.9 Hz) |
| Theta | φ⁻¹--φ⁰ (4.7--7.6 Hz) | φ⁻⁰·⁵ (6.0 Hz) | φ⁻⁰·³⁸ (6.2 Hz) |
| Alpha | φ⁰--φ¹ (7.6--12.3 Hz) | φ⁰·⁵ (9.7 Hz) | φ⁰·⁶² (10.1 Hz) |
| Low Beta | φ¹--φ² (12.3--19.9 Hz) | φ¹·⁵ (15.6 Hz) | φ¹·⁶² (16.3 Hz) |
| High Beta | φ²--φ³ (19.9--32.2 Hz) | φ²·⁵ (25.3 Hz) | φ²·⁶² (26.4 Hz) |
| Gamma | φ³--φ⁴ (32.2--52.1 Hz) | φ³·⁵ (41.0 Hz) | φ³·⁶² (42.7 Hz) |

This framework explains why canonical frequencies (alpha ~10 Hz, gamma ~40 Hz) emerge at half-integer φⁿ positions, and why band boundaries show transition zones rather than discrete cutoffs.

### 6.5 Limitations

1. **Consumer-grade EEG:** Limited spatial resolution (4--14 channels) constrains source localization. However, cross-device consistency argues against device-specific artifacts.

2. **No concurrent magnetometer:** Cannot test SR coupling hypotheses without simultaneous SR field measurement.

3. **Population demographics:** Participants skewed toward tech-savvy users of consumer EEG; clinical populations may show different patterns.

4. **FOOOF limitations:** Spectral parameterization is less reliable at low frequencies (delta/theta) where 1/f noise dominates.

5. **Descriptive, not mechanistic:** We document φⁿ organization but cannot explain why neural oscillations would organize according to golden ratio mathematics.

6. **No behavioral correlates:** We did not measure whether SIE occurrence or φⁿ precision predicts cognitive performance.

7. **Single-subject discovery:** The initial discovery dataset relied on one experienced meditator, though validation extended to 91 participants.

### 6.6 Future Directions

1. **Concurrent EEG-magnetometer recording:** Direct test of SR coupling hypotheses requires simultaneous measurement of brain activity and Schumann Resonance field strength.

2. **High-density EEG with source localization:** Clarify spatial organization of φⁿ generators and frontal hub topology suggested by transfer entropy analysis.

3. **Cross-species validation:** Test whether φⁿ architecture is evolutionarily conserved across species with different brain sizes. Preliminary evidence suggests similar frequency organization in mammals despite 17,000-fold brain volume variation (Buzsaki et al., 2013).

4. **Developmental trajectory:** Does φⁿ organization emerge with neural maturation, or is it present from early development?

5. **Clinical biomarkers:** Do disorders with oscillatory abnormalities (schizophrenia, Parkinson's, ADHD) show disrupted φⁿ organization? Could restoration serve as treatment target?

6. **Entrainment studies:** Test whether tACS at φⁿ frequencies produces stronger entrainment or superior cognitive effects compared to non-φⁿ frequencies (Herrmann et al., 2016).

7. **Behavioral correlates:** Examine whether SIE precision or aggregate φⁿ alignment predicts cognitive performance, creativity, or meditative depth.

## 7 Conclusions

This study makes four principal contributions to our understanding of neural oscillatory organization:

**First, we establish Schumann Ignition Events as a reproducible neural phenomenon.** Across 1,366 events, 91 participants, five cognitive contexts, and three consumer EEG devices, SIEs showed consistent spectral, temporal, and network characteristics. The coherence-first temporal signature and distinct state-space trajectories suggest that SIEs represent a qualitatively distinct neural regime rather than extreme values of continuous oscillatory dynamics.

**Second, we discover φⁿ ratio organization in transient high-coherence states.** Analysis of harmonic ratios (SR1, SR3, SR5) revealed < 1% deviation from golden ratio predictions, with the independence-convergence paradox revealing that ratio constraints operate at the population level rather than through event-level frequency coupling.

**Third, we validate φⁿ architecture in continuous spectral organization.** Systematic analysis of 244,955 spectral peaks confirms the predicted boundary-attractor structure: boundaries depleted (−18%), attractors enriched (+21%), noble positions maximally enriched (+39%). This provides the first mathematically principled basis for EEG frequency band definitions.

**Fourth, we propose the substrate-ignition model as an integrative framework.** The φⁿ lattice exists continuously as an architectural constraint; SIEs represent transient amplification of this substrate. This model explains both the precision of transient events and the robustness of aggregate spectral organization.

The core finding can be summarized in a single equation:

> f(n) = f₀ × φⁿ where f₀ = 7.60 Hz, φ = 1.6180339...     (11)

Whether the correspondence between neural f₀ and Earth's Schumann Resonance reflects evolutionary optimization, biophysical convergence, or coincidence cannot be determined without concurrent geomagnetic measurement. What the present findings establish is that human neural oscillations exhibit precise φⁿ ratio organization---a mathematical architecture that may represent a universal solution to the segregation-integration balance required for complex neural computation.

## References

Albuquerque, I., Tiwari, A., Parent, M., Cassani, R., Kachmar, J.-F., Bherer, L., Bhattacharya, J., Bhattacharya, S., Bhattacharya, S., Bhattacharya, S., Bhattacharya, S., Bhattacharya, S., & Falk, T. H. (2024). PhySF: A multimodal dataset for physiological states of flow. *Scientific Data, 11*, 88.

Albuquerque, I., Tiwari, A., Parent, M., Cassani, R., Kachmar, J.-F., Bherer, L., Bhattacharya, J., & Falk, T. H. (2023). MultiPENG: A multimodal dataset for player engagement analysis in video games. *Scientific Data, 10*, 423.

Bartos, M., Vida, I., & Jonas, P. (2007). Synaptic mechanisms of synchronized gamma oscillations in inhibitory interneuron networks. *Nature Reviews Neuroscience, 8*, 45--56.

Butterworth, S. (1930). On the theory of filter amplifiers. *Wireless Engineer, 7*, 536--541.

Buzsaki, G. (2006). Rhythms of the brain [Comprehensive treatment of neural oscillations and their functional roles]. *Oxford University Press*.

Buzsaki, G. (2015). Hippocampal sharp wave-ripple: A cognitive biomarker for episodic memory and planning. *Hippocampus, 25*, 1073--1188.

Buzsaki, G., Logothetis, N., & Singer, W. (2013). Scaling brain size, keeping timing: Evolutionary preservation of brain rhythms. *Neuron, 80*, 751--764.

Canolty, R. T., & Knight, R. T. (2010). The functional role of cross-frequency coupling. *Trends in Cognitive Sciences, 14*, 506--515.

Cherry, N. (2002). Schumann resonances, a plausible biophysical mechanism for the human health effects of solar/geomagnetic activity. *Natural Hazards, 26*, 279--331.

Cohen, M. X. (2017). Where does EEG come from and what does it mean? *Trends in Neurosciences, 40*, 208--218.

Donoghue, T., Haller, M., Peterson, E. J., Varma, P., Sebastian, P., Gao, R., Noto, T., Lara, A. H., Wallis, J. D., Knight, R. T., Shestyuk, A., & Voytek, B. (2020). Parameterizing neural power spectra into periodic and aperiodic components. *Nature Neuroscience, 23*, 1655--1665.

Engel, A. K., & Fries, P. (2010). Beta-band oscillations--signalling the status quo? *Current Opinion in Neurobiology, 20*, 156--165.

Faisal, M., Chowdhury, M. E. H., Khandakar, A., & Tahir, A. M. (2023). EEG dataset for natural image recognition through visual stimuli. *Data in Brief, 47*, 108991.

Fernandez, L. M. J., & Luthi, A. (2020). Sleep spindles: Mechanisms and functions. *Physiological Reviews, 100*, 805--868.

Fries, P. (2015). Rhythms for cognition: Communication through coherence. *Neuron, 88*, 220--235.

Fries, P., Reynolds, J. H., Rorie, A. E., & Desimone, R. (2001). Modulation of oscillatory neuronal synchronization by selective visual attention. *Science, 291*, 1560--1563.

Gramfort, A., Luessi, M., Larson, E., et al. (2013). MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience, 7*, 267.

Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). Array programming with NumPy. *Nature, 585*, 357--362.

Herrmann, C. S., Struber, D., Helfrich, R. F., & Engel, A. K. (2016). EEG oscillations: From correlation to causality. *International Journal of Psychophysiology, 103*, 12--21.

Hyman, J. M., Zilli, E. A., Paley, A. M., & Hasselmo, M. E. (2005). Medial prefrontal cortex cells show dynamic modulation with the hippocampal theta rhythm dependent on behavior. *Hippocampus, 15*, 739--749.

Jasper, H. H. (1958). The ten-twenty electrode system of the International Federation. *Electroencephalography and Clinical Neurophysiology, 10*, 371--375.

Klimesch, W. (1999). EEG alpha and theta oscillations reflect cognitive and memory performance: A review and analysis. *Brain Research Reviews, 29*, 169--195.

Konig, H. L., Krueger, A. P., Lang, S., & Sonning, W. (1974). Biologic effects of environmental electromagnetism.

Lachaux, J.-P., Rodriguez, E., Martinerie, J., & Varela, F. J. (1999). Measuring phase synchrony in brain signals. *Human Brain Mapping, 8*, 194--208.

Lisman, J., & Buzsaki, G. (2008). A neural coding scheme formed by the combined function of gamma and theta oscillations. *Schizophrenia Bulletin, 34*, 974--980.

Lisman, J. E., & Jensen, O. (2013). The theta-gamma neural code. *Neuron, 77*, 1002--1016.

Nickolaenko, A. P., & Hayakawa, M. (2014). *Schumann resonance for tyros*. Springer.

Persinger, M. A. (2008). On the possible representation of the electromagnetic equivalents of all human memory within the earth's magnetic field: Implications for theoretical biology. *Theoretical Biology Insights, 1*, 3--11.

Saroka, K. S., Vares, D. E., & Persinger, M. A. (2016). Similar spectral power densities within the schumann resonance and a large population of quantitative electroencephalographic profiles: Supportive evidence for Koenig and Pobachenko. *PLoS ONE, 11*, e0146595.

Schumann, W. O. (1952). Uber die strahlungslosen eigenschwingungen einer leitenden kugel, die von einer luftschicht und einer ionospharenhulle umgeben ist. *Zeitschrift fur Naturforschung A, 7*, 149--154.

Siegel, M., Donner, T. H., & Engel, A. K. (2012). Spectral fingerprints of large-scale neuronal interactions. *Nature Reviews Neuroscience, 13*, 121--134.

Varela, F., Lachaux, J.-P., Rodriguez, E., & Martinerie, J. (2001). The brainweb: Phase synchronization and large-scale integration. *Nature Reviews Neuroscience, 2*, 229--239.

Virtanen, P., Gommers, R., Oliphant, T. E., et al. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods, 17*, 261--272.
