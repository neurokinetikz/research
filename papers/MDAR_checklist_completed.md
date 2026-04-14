# MDAR Checklist -- Completed for Spectral Differentiation Manuscript

## Materials

| Item | Where provided | N/A |
|------|---------------|-----|
| **Newly created materials** | No new physical materials were created. Analysis code and extracted peak datasets are described in "Data and Code Availability" section. | |
| **Antibodies** | | N/A |
| **DNA and RNA sequences** | | N/A |
| **Cell materials** | | N/A |
| **Experimental animals** | | N/A |
| **Plants and microbes** | | N/A |
| **Human research participants** | Age: reported per dataset in Methods, Table 1 (ranges 5--21 for HBN, 20--77 for LEMON, 18--70 for Dortmund, adult for EEGMMIDB and CHBMP). Sex: tested as a covariate (Sex x age interaction: 0 FDR survivors; Results Part III, Section "Informative Nulls"). Sex-disaggregated demographics were not reported per dataset because this is a secondary analysis of publicly released, de-identified data and not all source datasets provide individual-level sex breakdowns in their public releases. Gender and ethnicity were not collected by the present study and are not uniformly available across the five source datasets. | |

## Design

| Item | Where provided | N/A |
|------|---------------|-----|
| **Study protocol** | This study was not pre-registered. It is a secondary analysis of five publicly available EEG datasets. The analysis pipeline and statistical approach are described in Methods, Sections 4.1--4.8. | |
| **Laboratory protocol** | | N/A -- computational study. All analysis code is publicly available at https://github.com/neurokinetikz/research (MIT License), as stated in "Data and Code Availability." |
| **Experimental study design** | | |
| -- Sample size determination | Sample sizes were determined by the availability of qualifying publicly released EEG datasets, not by a priori power analysis. The five datasets (total N = 2,097) were selected to maximize demographic and methodological diversity. A post hoc power discussion is provided in Discussion, Section "Limitations" (paragraph 2): at N = 203 (LEMON cognitive analyses), two-tailed Spearman correlation has ~80% power to detect |rho| >= 0.20 at alpha = 0.05 for a single test; FDR correction raises the effective detection threshold. | |
| -- Randomization | Not applicable. This is an observational secondary analysis of existing recordings; no experimental manipulation or group allocation was performed. | |
| -- Blinding | Not applicable. All analyses are computational on pre-existing de-identified datasets with no experimental conditions requiring blinding. | |
| -- Inclusion/exclusion criteria | Methods, Section 4.2 ("EEG Preprocessing and Spectral Peak Extraction"): Peaks retained only if FOOOF model R-squared >= 0.70 (median R-squared = 0.931). Power filtering retained top 50% of peaks by power within each band per subject (Section 4.2.1). Individual-differences analyses required >= 30 peaks per band per subject after filtering (Section 4.8). HBN Release 5 excluded due to artifact contamination (Table 1). Initial 9,132,221 peaks reduced to 4,572,489 after quality and power filtering. | |
| **Sample definition and in-laboratory replication** | Replication is addressed at multiple levels: (1) Cross-dataset: all enrichment analyses computed independently on 9 datasets and consistency reported (Results Part II, e.g., 12/13 position agreement for theta). (2) Cross-release: 5 independent HBN data releases analyzed separately (Results Part IV, Section "Cross-Release Consistency"; mean inter-release r = 0.787). (3) Cross-method: full pipeline replicated under IRASA as an independent aperiodic decomposition method (Results Part V). (4) Cross-state: eyes-closed vs. eyes-open comparison (Results Part III, cognitive section; Part IV, state sensitivity). Each subject contributes multiple peaks (technical replicates from channels and epochs); subjects are the independent biological units of analysis. The subject-level bootstrap (Results Part I, Section "Subject-Level Bootstrap") explicitly accounts for this nested structure. | |
| **Ethics** | Methods, Section "Ethics Statement": This study is a secondary analysis of publicly available, de-identified EEG datasets. All original studies obtained institutional ethics approval and informed consent from participants (or their guardians for HBN minors) prior to data collection and public release. No additional ethics approval was required for the present analyses. Original ethics approvals are documented in the cited source publications: EEGMMIDB (Goldberger et al., 2000), LEMON (Babayan et al., 2019), HBN (Alexander et al., 2017), Dortmund (Vossen et al., 2015). | |
| **DURC** | | N/A |

## Analysis

| Item | Where provided | N/A |
|------|---------------|-----|
| **Attrition** | Methods, Section 4.2: Initial extraction yielded 9,132,221 peaks; after quality filtering (R-squared >= 0.70) and power filtering (top 50% per band per subject), 4,572,489 peaks were retained (24% excluded). HBN Release 5 excluded entirely due to artifact contamination. Exclusion criteria were established before analysis based on the v3 pipeline corrections (Methods, Section 4.2.2 "Three Methodological Corrections"). No individual subjects were excluded post hoc; all subjects with at least one valid EC recording contributed peaks. For individual-differences analyses, a minimum of 30 peaks per band per subject was required (Methods, Section 4.8). | |
| **Statistics** | Methods, Sections 4.3--4.8 describe and justify all statistical tests: (1) BIC for model comparison with explicit formula and penalty for parameters (Section 4.3, Tests 1--4). (2) Permutation test against Poisson surrogates for aperiodic null (Section 4.3, Test 6; N = 200 surrogates). (3) Subject-level stratified bootstrap for CIs and p-values (Section 4.3, Test 7; 1,000 iterations, N = 2,097 subjects). (4) Spearman rank correlations for all individual-differences associations, justified by non-normality of enrichment distributions (Section 4.8). (5) Benjamini-Hochberg FDR at q < 0.05, applied within each analysis family with exact test counts reported (e.g., 720 tests for cognition, 90 for age, 11,970 for personality). (6) ICC(2,1) two-way mixed absolute agreement for test-retest reliability (Section 4.8). (7) Steiger's z-test for comparing dependent correlations (Results Part III, psychopathology). (8) One-sample t-tests on log-transformed ratios for per-dataset ratio comparisons (Results Part I). Effect sizes reported as Spearman rho, R-squared, ICC, BIC differences, and percentage-point enrichment values throughout. Exact p-values reported wherever possible; p < 0.0001 used only when below computational precision. | |
| **Data availability** | "Data and Code Availability" section: All five source EEG datasets are publicly available with URLs and citations provided (EEGMMIDB via PhysioNet, LEMON via NITRC, HBN via NITRC, Dortmund via Leibniz Centre portal, CHBMP via CAN-BIND portal). Newly created extracted peak CSVs (4.57 million peaks) are archived on Google Cloud Storage. | |
| **Code availability** | "Data and Code Availability" section: All analysis code publicly available at https://github.com/neurokinetikz/research under the MIT License. Software stack specified in Methods, Section 4.9 ("Computational Infrastructure"): Python 3.10, NumPy 1.26, SciPy 1.11, pandas 2.1, specparam 1.0. | |

## Reporting

| Item | Where provided | N/A |
|------|---------------|-----|
| **Adherence to community standards** | This MDAR checklist is provided as a supplementary file. No discipline-specific reporting guidelines (CONSORT, PRISMA, ARRIVE, STRANGE) apply, as this is not a clinical trial, systematic review, or animal study. The study follows ICMJE recommendations for authorship (Author Contributions section, CRediT taxonomy) and competing interests disclosure (Competing Interests section). Use of AI tools is disclosed in Methods, Section "Use of Artificial Intelligence Tools," per eLife policy. | |
