# Aperiodic (1/f) Dynamics

Aperiodic dynamics refer to the broadband, non-oscillatory component of neural power spectra that follows a power-law decay (1/f^n), where power decreases as frequency increases. This "1/f background" is not neural noise but reflects the aggregate temporal statistics of population-level neural activity, including the balance between excitatory and inhibitory synaptic currents. Separating aperiodic from periodic (oscillatory) components is now recognized as methodologically essential: without this decomposition, apparent changes in band-specific power -- such as an "alpha power decrease" -- may actually reflect a shift in the aperiodic slope rather than any change in true oscillatory activity. The aperiodic exponent has emerged as a proxy for excitation/inhibition (E/I) balance, linking this seemingly technical spectral feature to fundamental questions about cortical state, consciousness, and pathology. A systematic review of 177 clinical reports across 38 disorders found that 32 reported significant aperiodic effects related to diagnosis or treatment (Donoghue, 2025), cementing the aperiodic component as one of the most actively investigated features in clinical electrophysiology. Yet the field also confronts mounting evidence that the aperiodic exponent is frequency-range dependent, not a universal E/I readout, and sensitive to methodological choices that can determine whether genuine neural signals are detected or buried in fitting residuals.

## The Aperiodic Power Spectrum

When neural signals are transformed into frequency-domain power spectra, the result is a combination of two components: (1) narrowband oscillatory peaks (periodic) riding atop (2) a broadband power-law decay (aperiodic). The aperiodic component is characterized by two parameters: the **offset**, which reflects the overall level of broadband power and correlates with population-level neural spiking rates (Manning et al., 2009; Miller et al., 2012), and the **exponent** (or slope), which describes the steepness of the spectral decay. A steeper exponent indicates greater dominance of low-frequency power relative to high-frequency power, while a flatter exponent indicates a more even distribution of power across frequencies.

## FOOOF / Specparam Methodology

The standard tool for separating periodic from aperiodic components is FOOOF (Fitting Oscillations & One-Over-F), now officially renamed **specparam** (spectral parameterization) in its 2.0 release (Donoghue et al., 2020). The algorithm fits an aperiodic function (typically L = b - log(k + f^x), where b is offset, x is exponent, and k is the knee parameter) to the power spectrum, then identifies oscillatory peaks as deviations above this aperiodic fit. Typical parameter settings across the literature include a frequency range of 1-48 Hz, fixed aperiodic mode (no knee), peak width limits of 0.5-12 Hz, and a peak threshold of 2 standard deviations. Studies consistently report excellent model fits (R² of 0.93-0.99) across diverse datasets and recording modalities including EEG and MEG (Masina et al., 2025; Stein et al., 2026; Subramani et al., 2026; Balaji et al., 2026; Griffiths, 2026). See [[canonical-eeg-bands]] for how this decomposition reframes the definition of frequency band power.

The specparam 2.0 release represents a full code rewrite with several important new capabilities. **SPRiNT** (Spectral Parameterization Resolved in Time) enables time-resolved decomposition, tracking how aperiodic and periodic features fluctuate across task epochs or [[sleep-oscillations|sleep stages]] (Wilson et al., 2025). The new version also introduces **data-driven model selection** to reduce reliance on subjective hyperparameter choices, and expanded support for the **knee parameter** when fitting broader frequency ranges where the aperiodic component bends rather than following a single power law. These updates are particularly important for intracranial recordings and MEG, where broader bandwidth data routinely exhibit spectral knees.

### Alternative Decomposition Methods

Specparam is not the only approach to aperiodic-periodic separation. **IRASA** (Irregular Resampling Auto-Spectral Analysis) separates the two components by exploiting the scale-free property of aperiodic signals: resampling the time series at non-integer factors displaces oscillatory peaks but preserves the 1/f structure, allowing the aperiodic component to be isolated by median-averaging across resampled spectra. IRASA outputs the separated components without parametric modeling, which offers flexibility but also means it does not directly provide exponent or offset values. Comparative evaluations show that IRASA consistently underestimates periodic components in multi-peak contexts, while specparam and modified specparam variants demonstrate higher accuracy (Ostlund et al., 2022). A modified specparam approach introducing a customized loss function that penalizes localized spectral dips differently from peaks has been shown to produce more stable parameter estimates and reduced mean squared error.

### Reliability and Reproducibility

Test-retest reliability of specparam-derived aperiodic parameters has been systematically evaluated. Regression-based methods including specparam provide aperiodic parameter estimates of high internal consistency under most circumstances, supported by good-to-excellent odd-even reliability values (Hartmann et al., 2024). Intra-class correlations (ICC) for the aperiodic exponent in young adults are good to excellent over 30-minute intervals (ICC 0.78-0.93), and remain good over 90-minute and 30-day intervals (average ICC 0.64-0.73). A five-year longitudinal follow-up found that parameterized resting-state EEG measures show remarkable long-term stability, supporting their use as trait-like individual difference measures (Mierau et al., 2026). However, the user-defined maximum number of oscillatory peaks in specparam introduces analytic flexibility: variability in the number of detected peaks may increase sensitivity to noise and reduce the reliability of aperiodic parameter estimates, motivating the data-driven model selection features in the 2.0 release.

## Why Separation Matters: The Masina Warning

The critical importance of aperiodic-periodic separation was demonstrated by Masina et al. (2025), who applied 40 Hz tACS to sensorimotor cortex and found that raw power analysis showed a significant alpha power decrease (F(1,1) = 8.66, p = 0.003). Without FOOOF, this would be interpreted as oscillatory alpha suppression. However, spectral parameterization revealed that tACS selectively modulated the aperiodic **offset** (F(1,2) = 4.59, p = 0.033) -- a broadband downward shift -- while periodic [[alpha-oscillations]] were entirely unaffected. The apparent "alpha reduction" was an artifact of the broadband offset decrease affecting all frequencies, including the alpha band. This finding has profound implications for interpreting all prior studies of [[neurofeedback-and-stimulation]] that relied on raw band power.

## The Aperiodic Exponent as E/I Balance Proxy

The aperiodic exponent has been established as a non-invasive proxy for excitation/inhibition balance (Voytek et al., 2015; Donoghue et al., 2020). A **steeper** (more negative) exponent reflects greater inhibitory dominance, while a **flatter** exponent reflects increased cortical excitability and a shift toward excitation. This relationship has been validated through computational modeling and pharmacological manipulations. Stein et al. (2026) leveraged this relationship by designing transcranial Endogenous Current Stimulation (tECS), which reduces the aperiodic exponent by 50% to selectively shift E/I balance toward excitation while preserving oscillatory structure. Their pilot data showed a mean exponent reduction of 9.71% over five days of stimulation, demonstrating that the aperiodic component can be causally manipulated. See [[neurochemistry-of-oscillations]] for the synaptic basis of E/I balance.

### Pharmacological Validation and Its Limits

The E/I proxy interpretation has received partial pharmacological support. GABA_A receptor positive allosteric modulators reliably increase the aperiodic exponent, consistent with the prediction that enhanced cortical inhibition steepens the spectral slope (Koponen et al., 2024). However, the same study found that ketamine (an NMDA receptor antagonist) and GABA_A antagonists at subconvulsive doses did not follow the predicted pattern, leading the authors to conclude that the aperiodic exponent does not yield a universally reliable marker of cortical E/I ratio (Koponen et al., 2024). Similarly, in freely-moving rats, MK-801 (NMDAR antagonist) and diazepam (GABA_A modulator) produced region-dependent and compound-dependent changes to both periodic and aperiodic EEG features that did not map cleanly onto a single E/I axis (Colombo et al., 2023). These findings suggest that while the aperiodic exponent tracks GABAergic tone reasonably well, it may be less sensitive to glutamatergic manipulations, and its interpretation as a simple E/I ratio is an oversimplification of a multidimensional synaptic landscape.

### Frequency-Range Dependence

A further complication is that the aperiodic exponent depends on the frequency range over which it is estimated. Gyorgy et al. (2025) analyzed resting-state intracortical recordings from 62 patients using both specparam and IRASA, systematically demonstrating a strong positive dependency between frequency and aperiodic exponent: low frequencies display flatter spectra than high frequencies, and no frequency range was found in which the aperiodic exponent was fully stable. This means that exponent values from studies using different fitting ranges (e.g., 1-30 Hz vs. 2-50 Hz vs. 1-100 Hz) are not directly comparable, a methodological caveat that complicates cross-study synthesis and meta-analysis. The per-octave modeling approach described below (Lacy, 2026c) represents one strategy for addressing this limitation.

## Aperiodic Changes Under Psychedelics

Subramani et al. (2026) provided comprehensive evidence that LSD flattens the aperiodic slope across cortex, with strongest effects in posterior cingulate cortex and visual cortex (p < 10^-5). This flattening reflects a shift toward excitation, consistent with the known 5-HT2A-mediated reduction of inhibitory constraints under psychedelics. Critically, some spectral effects previously attributed to oscillatory changes were revealed as purely aperiodic: in the posterior cingulate cortex, robust LSD-placebo differences visible in raw spectra completely disappeared after FOOOF parameterization, indicating the effects were entirely aperiodic rather than oscillatory. The aperiodic flattening also correlated with increased neural signal complexity (Lempel-Ziv complexity and Higuchi Fractal Dimension), suggesting a link between E/I balance, spectral slope, and the entropy increases characteristic of [[psychedelic-states]].

## Aperiodic Changes Across States and Conditions

The aperiodic exponent varies systematically with cognitive state, development, and pathology. Tasks that engage [[gamma-oscillations]], such as auditory steady-state responses, flatten the spectrum compared to rest (Masina et al., 2025). Aging is associated with flattening of the 1/f slope, reflecting age-related shifts in E/I balance toward excitation. Neuropsychiatric conditions including autism, ADHD, depression, and schizophrenia have been linked to altered aperiodic features, positioning the exponent as a potential transdiagnostic biomarker for [[clinical-applications]]. The [[individual-alpha-frequency]] literature has also been reframed by aperiodic considerations: Balaji et al. (2026) used FOOOF to separate periodic alpha peaks from the 1/f background before measuring moment-to-moment peak frequency variability, a step necessary to ensure that apparent frequency shifts reflect genuine oscillatory dynamics rather than aperiodic slope changes.

## Aperiodic Dynamics in Development and Aging

The aperiodic exponent follows a characteristic **inverted-U trajectory** across the human lifespan. During infancy and early childhood (2-44 months), the aperiodic slope steepens -- that is, the exponent increases -- reflecting progressive maturation of inhibitory circuits, with females exhibiting greater increases relative to males, particularly at ages 3 and 5 years (Tröndle et al., 2024). McSweeney et al. (2023) confirmed quadratic age-related effects in 502 children aged 4-11 years: both offset and exponent increased from age 4 to approximately age 7, then decreased from 7 to 11, suggesting a developmental inflection point in mid-childhood. A large international lifespan study (N = 1,563, ages 5-95) revealed that aperiodic activity follows a **monotonically decreasing** trajectory when plotted on a logarithmic age scale, mirroring fundamental biomarkers of biological aging such as DNA methylation and telomere length (Kosciessa et al., 2025). Both aperiodic offset and exponent rapidly decrease from early life until approximately age 18, then continue a slower decline through old age. This contrasts with periodic (oscillatory) components, which follow a growth-then-decline trajectory aligning with GABAergic function and [[thalamocortical-circuits|myelination]].

In older adults, the flatter aperiodic slope has functional consequences. Education modulates the relationship between aperiodic components and cognitive performance in aging: older adults with higher educational attainment show a weaker association between exponent flattening and cognitive decline, suggesting that cognitive reserve may buffer the functional impact of age-related E/I shifts (Cesnaite et al., 2024). Task-related aperiodic dynamics also change with age: during verbal working memory, older adults show less task-related aperiodic modulation compared to younger adults, consistent with a neural inefficiency account of cognitive aging (Waschke et al., 2025). See [[development-and-aging]] for the broader developmental context of oscillatory architecture.

## Clinical Applications of Aperiodic Activity

### ADHD

Children with ADHD consistently show **flatter aperiodic spectral slopes** at rest compared to typically developing controls, reflecting a shift toward cortical excitation (Robertson et al., 2019; Ostlund et al., 2021). Longitudinal evidence suggests that children with ADHD have delayed maturation of the aperiodic exponent from ages 6-12, with the developmental trajectory lagging behind neurotypical peers (Karalunas et al., 2024). The aperiodic slope also predicts ADHD-related cognitive differences: flatter slopes are associated with attenuated event-related potential amplitudes and poorer sustained attention performance. Stimulant medication (methylphenidate) steepens the aperiodic slope, partially normalizing E/I balance, though the effect varies by dose and brain region.

### Schizophrenia

Adults with schizophrenia show a trend toward shallower (flatter) aperiodic exponents compared to healthy controls (Racz et al., 2021), consistent with disrupted E/I balance. Anti-seizure medications, which are sometimes used as adjunct treatments, have distinct effects on the aperiodic exponent depending on their mechanism of action -- GABAergic agents steepen the slope while sodium channel blockers show more variable effects (Barkmeier et al., 2025).

### Depression

In major depressive disorder (MDD), a reduced aperiodic exponent has been observed compared to controls, with the aperiodic offset correlating with depression severity. A particularly promising finding is that aperiodic slope changes robustly track symptom severity changes in treatment-resistant depression: flatter (less negative) slopes correlate with reduced depression severity, particularly over the ventromedial prefrontal cortex, offering a potential objective biomarker for treatment response monitoring (Vetter et al., 2024).

### Autism

Prior studies in autism report mixed findings on aperiodic activity: some report flatter slopes consistent with E/I imbalance toward excitation, while others find no significant group differences (Donoghue, 2025). This inconsistency may reflect the heterogeneity of the autism spectrum, methodological variability in fitting ranges and recording conditions, and the possibility that task-related aperiodic dynamics differ from resting-state measures.

### Alzheimer's Disease

Changes in the aperiodic exponent may emerge in the **preclinical and prodromal stages** of Alzheimer's disease, potentially preceding overt cognitive decline. Klyuzhin et al. (2026) used high-density EEG with tau PET imaging in 64 individuals across the AD continuum and found region-specific aperiodic exponent changes in amyloid-positive but cognitively unimpaired individuals, with the exponent associated with regional tau load. These findings support the utility of aperiodic EEG as a noninvasive approach for detecting early neural alterations in AD. However, conflicting evidence exists: Watanabe et al. (2023) found that aperiodic features did not differ between AD patients and healthy controls, arguing that resting-state EEG signatures of AD are driven by periodic rather than aperiodic changes. The discrepancy may reflect differences between preclinical vs. clinical-stage disease, or between global vs. region-specific analyses.

### Epilepsy and Glioma

In temporal lobe epilepsy (TLE), patients show a larger (steeper) aperiodic exponent compared to healthy controls, suggesting a shift toward inhibition, with the exponent correlating with memory functioning and cortical expression of genes involved in E/I regulation (Kramer et al., 2024). Paradoxically, the aperiodic exponent increases progressively in the minutes preceding seizures, reflecting a widespread shift toward cortical inhibition before the network tips into a seizure (van den Berg et al., 2025). In glioma patients, tumor infiltration induces a **flattening** of the aperiodic slope, indicating a shift toward excitation dominance that varies by tumor subtype, providing a novel signature of glioma-induced E/I dysregulation (Zijlmans et al., 2025).

### Systematic Review: The State of the Evidence

Donoghue (2025) conducted the first comprehensive systematic review of aperiodic activity in clinical electrophysiology, following PRISMA guidelines across 177 reports spanning 38 psychiatric and neurological disorders. The review found that while aperiodic activity is commonly reported to relate to clinical diagnoses (32 of 38 disorders showing significant effects), there is considerable variation in the consistency of results across disorders. Heterogeneity of patient groups, disease aetiologies, treatment status, and methodological differences in fitting ranges and recording conditions emerge as recurring themes. The review concluded that further work is needed before aperiodic activity can be established as a reliable clinical biomarker, particularly given the unresolved questions about the biological validity of the E/I interpretation.

## Aperiodic Dynamics in Consciousness and Anesthesia

The aperiodic exponent carries substantial information about the level of [[consciousness-and-anesthesia|consciousness]]. During propofol-induced loss of consciousness, the spectral exponent increases dramatically (steepening), with an area under the curve (AUC) of 0.98 for discriminating conscious from unconscious states (Colombo et al., 2025). This steepening reflects a global shift toward inhibitory dominance as consciousness fades. Crucially, propofol and volatile anesthetics (sevoflurane, desflurane) produce distinct aperiodic profiles despite appearing similar on conventional depth-of-anesthesia monitors, suggesting that aperiodic parameterization could improve clinical monitoring of anesthetic depth.

Schaworonkow and Bhatt (2024) provided a biophysical grounding for these observations using forward modeling of scalp potentials. Their simulations showed that propofol-induced broadband EEG changes quantitatively match the drug's known effects on GABA_A receptor kinetics -- specifically, prolongation of synaptic decay time constants. After correcting for these aperiodic confounds, they found that [[delta-oscillations|delta power]] uniquely increased within seconds of individuals losing consciousness, a finding masked in conventional spectral analyses by the broadband shift. This work demonstrates that what appears as a "delta power increase" under anesthesia is partially a steepening of the aperiodic component, with only a residual true oscillatory delta enhancement.

In disorders of consciousness, the aperiodic exponent distinguishes clinical states. Patients in **unresponsive wakefulness syndrome** (UWS, formerly vegetative state) show significantly higher (steeper) aperiodic exponents compared to patients in **minimally conscious state** (MCS), reflecting deeper inhibitory dominance (Bédard et al., 2023). Combining periodic and aperiodic parameters improves diagnostic accuracy for distinguishing UWS from MCS compared to periodic features alone, and the aperiodic exponent may be superior to oscillatory measures in assessing level of consciousness following traumatic brain injury and stroke. During [[sleep-oscillations|sleep]], time-resolved aperiodic analyses track the transition across sleep stages, with the exponent steepening from wakefulness through NREM sleep (Wilson et al., 2025).

## Biophysical and Computational Models of 1/f Generation

The mechanistic origins of the aperiodic 1/f spectrum have been clarified by recent computational work. Schaworonkow and Bhatt (2024) developed a biophysically grounded theory showing that aperiodic neural activity can generate detectable scalp potentials when populations of neurons fire asynchronously with sufficient spatiotemporal correlation. Their model demonstrates that the aperiodic spectral trend is shaped not only by the temporal statistics of neural firing but also by the biophysics of synaptic currents -- specifically, the time constants of excitatory (AMPA, NMDA) and inhibitory (GABA_A, GABA_B) receptor-mediated currents. Changes in these synaptic properties shift the aperiodic spectrum in predictable ways: prolonging inhibitory decay time constants (as GABA_A agonists do) steepens the slope, while increasing excitatory drive flattens it. Importantly, the model also shows that rhythmic EEG signals are profoundly corrupted by shifts in synapse properties, meaning that pharmacological or pathological changes to synaptic function can masquerade as oscillatory effects if the aperiodic component is not properly accounted for.

The relationship between aperiodic dynamics and neural synchrony has also been modeled computationally. Coupled oscillator models demonstrate that ubiquitous 1/f noise enhances phase synchrony more effectively than spectrally flat (white) noise, with a competitive synergy between noise intensity and the 1/f spectral exponent: increasing the exponent while decreasing noise intensity leads to enhanced synchrony, peaking at a specific parameter regime (Kluger et al., 2025). This suggests that the brain's 1/f background is not merely a byproduct of network activity but may actively facilitate the [[cross-frequency-coupling|coordination of oscillatory dynamics]].

A striking recent demonstration of the functional significance of 1/f dynamics comes from the ripple detection literature. Approximately 77% of putative awake ripples detected in the human medial temporal lobe -- including hippocampus -- reflect false positives within the 1/f noise floor (Nagahama et al., 2026). Task-related 1/f modulations generate spurious ripple detections because conventional detection algorithms do not account for state-dependent changes in the aperiodic background. This finding has major implications for the [[sharp-wave-ripples]] literature, as many reported task-related "ripple" effects may actually index modulations of the aperiodic component rather than genuine high-frequency oscillatory events.

## Criticality and Scale-Free Dynamics

The 1/f power spectrum is one of several signatures linking neural dynamics to **criticality** -- the hypothesis that the brain operates near a phase transition between ordered and disordered states, optimizing information processing capacity. Power-law distributions in neural avalanche sizes, long-range temporal correlations, and the 1/f spectral structure all emerge naturally in systems poised at criticality. A meta-analysis of 140 datasets published between 2003 and 2024 documented accelerating evidence for neural criticality, with 33 papers in 2024 alone and a cumulative 320 papers reporting experimental support (Shew and Bhatt, 2025). The criticality framework predicts that the aperiodic exponent should cluster near specific values (typically around 1.0-1.5 for EEG) corresponding to the critical point, and deviations from criticality -- toward either subcritical (steeper) or supercritical (flatter) regimes -- should impair information processing.

However, the criticality interpretation is not without controversy. Power-law statistics can arise from non-critical processes, weakening the inferential power of 1/f spectra alone as evidence for criticality. Recent work has sought to address this by examining additional signatures beyond power laws, including the relationship between correlation length and system size, and the specific scaling exponents predicted by different universality classes. The observation that [[psychedelic-states|psychedelics]] flatten the aperiodic slope -- pushing the brain toward a more supercritical, disordered regime -- while [[consciousness-and-anesthesia|anesthesia]] steepens it -- pushing toward a more subcritical, ordered regime -- provides converging evidence that the aperiodic exponent tracks position along the order-disorder continuum that criticality theory describes.

## Per-Octave Aperiodic Modeling and Oscillatory Claims

Lacy (2026c) provides a concrete example of how aperiodic separation methodology determines the detectability of oscillatory structural claims. Standard whole-range FOOOF (e.g., 1–45 Hz) poorly approximates the full aperiodic spectrum due to the spectral knee, leading to systematic misestimation of peak frequencies and amplitudes. Under standard extraction, the φ-lattice structural advantage is undetectable (φ ranks 7th of 9 bases). Only per-octave aperiodic modeling -- fitting the aperiodic component separately within each frequency octave -- recovers the signal, elevating φ to rank 1st. This demonstrates that the choice of aperiodic fitting granularity is not merely a technical detail but can determine whether a genuine structural signal is detected or buried in aperiodic misfit residuals.

## Implications for Interpreting Oscillatory Research

The recognition of aperiodic dynamics forces a reappraisal of many claims about "power changes" in specific frequency bands. Any study reporting increased or decreased power in theta, alpha, beta, or gamma bands without aperiodic-periodic separation may be conflating oscillatory and non-oscillatory effects. This applies to research on [[consciousness-and-anesthesia]], where broadband spectral changes accompany state transitions, and to [[cross-frequency-coupling]] analyses, where apparent coupling could be inflated by shared aperiodic trends. Griffiths (2026) exemplifies best practice by using FOOOF to verify that alpha-rhythmic memory reactivation reflects genuine narrowband oscillatory activity rather than broadband fluctuations. Going forward, aperiodic parameterization should be considered a prerequisite for any spectral analysis claiming to identify oscillatory effects.

## Related Topics

- [[canonical-eeg-bands]] -- Aperiodic separation redefines how band power is measured
- [[alpha-oscillations]] -- Many "alpha power" findings may reflect aperiodic shifts
- [[gamma-oscillations]] -- Gamma-band tasks flatten the aperiodic slope
- [[neurofeedback-and-stimulation]] -- tACS effects may be aperiodic rather than oscillatory
- [[psychedelic-states]] -- LSD flattens 1/f slope as marker of increased excitability
- [[individual-alpha-frequency]] -- FOOOF required to isolate true periodic peak frequency
- [[clinical-applications]] -- Aperiodic exponent as transdiagnostic biomarker
- [[neurochemistry-of-oscillations]] -- E/I balance as the biological basis of spectral slope
- [[consciousness-and-anesthesia]] -- Aperiodic exponent tracks consciousness level with AUC 0.98
- [[development-and-aging]] -- Lifespan trajectory of aperiodic flattening mirrors biological aging
- [[delta-oscillations]] -- True oscillatory delta masked by aperiodic confounds under anesthesia
- [[sharp-wave-ripples]] -- ~77% of detected awake ripples may be 1/f noise floor artifacts
- [[sleep-oscillations]] -- Time-resolved aperiodic tracking across sleep stages
- [[cross-frequency-coupling]] -- 1/f noise may actively facilitate oscillatory coordination
- [[thalamocortical-circuits]] -- Aperiodic lifespan trajectory maps onto myelination
- [[critical-frequencies]] -- Criticality framework links 1/f to optimal information processing

## Sources

- Donoghue et al. (2020) -- FOOOF / specparam methodology
- Manning et al. (2009); Miller et al. (2012) -- Aperiodic offset and neural spiking rates
- Masina et al. (2025) -- tACS aperiodic offset effect, the Masina Warning
- Subramani et al. (2026) -- LSD aperiodic flattening
- Stein et al. (2026) -- tECS and aperiodic exponent manipulation
- Lacy (2026c) -- Dominant Peaks, Fragile Metrics: Separating Architectural Signal from Methodological Artifact in Phi-Lattice EEG Analysis (per-octave aperiodic modeling as critical determinant of φ-lattice detectability)
- Donoghue (2025) -- Systematic review of aperiodic neural activity across 177 clinical reports and 38 disorders (European Journal of Neuroscience)
- Schaworonkow and Bhatt (2024) -- Biophysical model of aperiodic EEG generation; synaptic time constants shape 1/f slope; propofol validation (Nature Communications)
- Koponen et al. (2024) -- Pharmacological validation showing GABA_A agonists steepen exponent but other drugs do not follow E/I predictions (Journal of Neurophysiology)
- Colombo et al. (2023) -- Region-dependent aperiodic effects of GABAergic and glutamatergic modulation in rats (eNeuro)
- Colombo et al. (2025) -- Aperiodic exponent reflects hypnotic level of anaesthesia, AUC 0.98 (British Journal of Anaesthesia)
- Gyorgy et al. (2025) -- Aperiodic exponent is frequency-range dependent; intracortical recordings from 62 patients (IEEE)
- McSweeney et al. (2023) -- Age-related trends in aperiodic EEG and alpha during early-to-middle childhood, N=502 (NeuroImage)
- Kosciessa et al. (2025) -- Lifespan aperiodic and periodic EEG trajectories, N=1,563, ages 5-95 (bioRxiv)
- Tröndle et al. (2024) -- Developmental trajectories of aperiodic components in children 2-44 months; sex differences (Nature Communications)
- Cesnaite et al. (2024) -- Education modulates aperiodic-cognition relationship in aging (Scientific Reports)
- Waschke et al. (2025) -- Task-related aperiodic dynamics and neural inefficiency in aging (bioRxiv)
- Hartmann et al. (2024) -- Test-retest reliability of specparam aperiodic estimates (Cerebral Cortex)
- Mierau et al. (2026) -- Five-year longitudinal stability of parameterized resting-state EEG (bioRxiv)
- Ostlund et al. (2022) -- Comparison of IRASA, specparam, and modified specparam on simulated signals
- Wilson et al. (2025) -- Time-resolved aperiodic analyses tracking sleep dynamics (Communications Psychology)
- Robertson et al. (2019); Ostlund et al. (2021) -- Aperiodic slope differences in ADHD
- Karalunas et al. (2024) -- ADHD-related aperiodic maturation delay from ages 6-12 (Clinical Neurophysiology)
- Vetter et al. (2024) -- Aperiodic slope tracks symptom severity in treatment-resistant depression
- Klyuzhin et al. (2026) -- Early aperiodic EEG changes in preclinical and prodromal Alzheimer's disease; tau PET association (Alzheimer's Research & Therapy)
- Watanabe et al. (2023) -- Resting-state EEG signatures of AD driven by periodic not aperiodic changes (Neurobiology of Disease)
- Kramer et al. (2024) -- Aperiodic exponent in temporal lobe epilepsy relates to memory and gene expression
- van den Berg et al. (2025) -- Dynamic E/I balance changes preceding seizures (BMC Medicine)
- Zijlmans et al. (2025) -- Aperiodic dynamics as signature of glioma-induced E/I dysregulation (bioRxiv)
- Bédard et al. (2023) -- Aperiodic activity and response to anesthesia in disorders of consciousness (NeuroImage)
- Nagahama et al. (2026) -- Aperiodic 1/f noise drives ~77% of putative awake ripple detections in humans (Nature Communications)
- Kluger et al. (2025) -- Computational model: 1/f noise enhances neural synchrony (bioRxiv)
- Racz et al. (2021) -- Aperiodic dynamics in clinical populations
- Shew and Bhatt (2025) -- Meta-analysis of 140 criticality datasets; accelerating evidence (Neuron)
- Voytek et al. (2015) -- Age-related changes in aperiodic neural activity
- Barkmeier et al. (2025) -- Anti-seizure medication effects on aperiodic EEG activity (bioRxiv)
