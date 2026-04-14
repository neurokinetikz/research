# Scripts Index

## Paper Figures

| Script | Description |
|--------|-------------|
| create_nature_fig1.py | Figure 1: SIE discovery characterization with exemplar spectrograms, harmonic ratios, and convergence |
| create_nature_fig2.py | Figure 2: Continuous spectral architecture with peak distributions and enrichment |
| create_nature_fig3.py | Figure 3: Band-stratified analysis and gamma dominance with lattice histograms |
| create_nature_fig4.py | Figure 4: Multi-channel GED convergence with FOOOF vs GED comparison |
| create_nature_fig5.py | Figure 5: Substrate-ignition model with lattice schematic and f0 convergence |
| create_nature_fig6.py | Figure 6: Structural specificity with bootstrap gap and cross-resolution robustness |
| generate_lemon_paper_figures.py | All figures for LEMON phi-lattice paper from pre-computed CSV exports |
| generate_paper3_dortmund_figures.py | Figures for Paper 3 revision incorporating Dortmund replication data |
| generate_paper_statistics.py | Computes all statistics for SIE analysis paper from structured CSV for LaTeX |
| power_analysis_paper.py | Sensitivity power analysis and achieved power for all key statistical tests |
| paper3_step0_freeze_numbers.py | Freezes numbers for Paper 3 by computing measurement-quality gradient |
| e8_poster.py | Conference poster figure visualizing E8 energy flow simulation results |

## Phi-Lattice Analysis

| Script | Description |
|--------|-------------|
| phi_statistical_validation.py | Position enrichment, permutation testing, f0 sensitivity, and alternative scaling comparisons |
| phi_landmark_model_comparison.py | Distinguishes discrete phi-anchored structure from smooth non-uniformity via likelihood ratio tests |
| phi_octave_histograms.py | Log-phi histograms of dominant peaks per phi-octave band with degree-7 position lines |
| structural_phi_specificity.py | Tests whether structural enrichment pattern is unique to phi or appears under any exponential base |
| per_position_alignment.py | Tests whether individual lattice positions carry significant alignment signal |
| per_position_diagnostic.py | Investigates boundary offset, spectral gaps, and lattice asymmetry puzzles |
| per_session_phi_ratios.py | Per-session harmonic ratio precision with three-level variance decomposition and ICC |
| per_subject_phi_ratios.py | Analyzes per-subject phi-ratio precision to show relationships hold within individuals |
| comprehensive_phi_comparison.py | Compares gedBounds boundaries against all phi-position types across datasets |
| band_position_enrichment.py | Enrichment at 12 lattice positions within 5 frequency bands with heatmap visualization |
| voronoi_enrichment_analysis.py | Per-band degree-6 Voronoi enrichment across 9 datasets with split boundary, cross-dataset summary, and HBN cross-release consistency. Reproducible via `--all --summary`. |
| voronoi_condition_comparisons.py | EC vs EO, pre vs post fatigue, and adult vs pediatric enrichment comparisons across datasets. Usage: `--analysis all`. |
| voronoi_lifespan_trajectory.py | Per-subject enrichment × age across HBN (5-21), Dortmund (20-70), and LEMON (20-77). Computes Spearman correlations with FDR, compares developmental vs aging trajectories. Usage: `--dataset all`. |
| per_subject_voronoi_cognitive.py | Per-subject Voronoi enrichment × LEMON cognitive battery (8 tests). Correlates enrichment at each position × band with cognitive scores. Also computes age × enrichment. |
| per_subject_voronoi_hbn_age.py | Per-subject enrichment × age/sex/psychopathology for HBN (N=927). Includes p_factor, attention, internalizing, externalizing correlations. |
| per_subject_voronoi_personality.py | Per-subject enrichment × LEMON emotion/personality battery (133 subscales from 23 instruments). 0 FDR survivors across 8,778 tests. |
| voronoi_state_sensitivity_age.py | Tests whether EC→EO enrichment change magnitude/direction correlates with age. Matched within-subject pairs for LEMON and Dortmund. Result: 0 FDR survivors — state sensitivity is age-independent. |
| voronoi_test_retest_reliability.py | Within-session test-retest reliability of per-subject enrichment. Dortmund EC-pre vs EC-post (ICC=+0.40), EO-pre vs EO-post (ICC=+0.35), EC vs EO cross-condition (ICC=+0.35). |
| voronoi_cross_band_coupling.py | Cross-band coupling: do subjects with stronger alpha mountain have deeper beta-low U-shape? Tests all pairwise cross-band correlations across LEMON, Dortmund, and HBN with replication analysis. |
| voronoi_sex_age_interaction.py | Sex × age interaction: does the inverted-U trajectory differ between M and F? Fisher z-test on sex-stratified age rhos. Result: 0 FDR survivors in both HBN and Dortmund. |
| voronoi_longitudinal.py | 5-year longitudinal test-retest (Dortmund ses-1 vs ses-2, N=208). ICC, group stability, and baseline age × 5-year change. |
| voronoi_medical_handedness.py | Medical/metabolic (LEMON BMI, BP, blood biomarkers) and handedness (HBN EHQ, Dortmund) × enrichment. Both null (0 FDR survivors). |
| voronoi_hbn_per_release_age.py | Tests whether developmental age correlations replicate independently within each HBN release. Cross-release rho correlations r=0.68-0.82. Alpha inv_noble_3/4 significant in ALL 5 releases. |
| voronoi_longitudinal_2x2.py | 5-year 2×2 replication: full EC/EO × pre/post pattern ses-1 vs ses-2 (N=208). Beta-low r=0.99 profile correlation, 13/13 stable across all 8 conditions. |
| ratio_lattice_enrichment.py | Maps pairwise frequency ratios to phi-octave lattice and tests 14-position enrichment |
| ratio_lattice_per_band.py | Dominant-peak ratios per band-pair mapped to phi-octave coordinates |
| combine_enrichment_master.py | Combines all phi-lattice enrichment data into queryable master CSVs |
| cross_dataset_position_consistency.py | Identifies position-band cells showing consistent enrichment across datasets |
| pairwise_ratio_test.py | Tests whether phi-spaced frequency ratios are enriched while rationals are depleted |
| shape_replication_inference.py | Tests whether enrichment shape replicates across datasets via Kendall's W |

## v3 Enrichment Reanalysis

| Script | Description |
|--------|-------------|
| run_f0_760_extraction.py | v3 peak extraction pipeline: merged θ+α FOOOF or IRASA (--method flag), cap=12, bandwidth floor=2×freq_res, 50 Hz notch on European datasets, R²/quality saved per peak. All datasets. |
| run_all_f0_760_analyses.py | 22-step analysis suite on v3 extraction: pooled enrichment, cognitive correlations, HBN developmental trajectory, Dortmund aging, EC/EO, personality, test-retest, cross-band coupling, and more. |
| run_v4_sweep.py | 48-configuration FOOOF parameter sweep on EEGMMIDB (peak_threshold, min_peak_height, max_n_peaks, nperseg). Validates v3 config as optimal. |
| gcp_run.sh | GCP VM orchestration: spawn, extract, push results, delete. Used for all v3 extractions. |
| boundary_sweep.py | Sweeps 2D (f0, ratio) parameter space to evaluate coordinate systems for EEG band definitions. Computes boundary sharpness, profile simplicity, band independence, cross-dataset consistency, and enrichment contrast. Also runs per-boundary slide analysis. |
| within_band_coordinates.py | Tests within-band coordinate structure: scaling comparison (log vs linear vs ERB vs mel), landmark capture (phi vs equal vs rational vs random positions), feature alignment (permutation test), periodicity (autocorrelation), and noble vs rational enrichment. |

## f0 Optimization & Sensitivity

| Script | Description |
|--------|-------------|
| optimize_f0.py | Finds optimal f0 maximizing alignment between GED peaks and phi-positions via KDE scoring |
| explain_f0_shift.py | Visualizes how changing f0 affects which frequencies land at special lattice positions |
| generate_f0_ranking_simple.py | Intuitive bar charts showing f0 ranking validation at key f0 values |
| generate_f0_ranking_validation.py | Shows f0=7.6 Hz satisfies theoretical enrichment ranking across lattice positions |
| generate_f0_sensitivity_figure.py | Publication-quality f0 sensitivity figure showing alignment score across f0 range |
| theta_target_disambiguation.py | Tests whether EC theta convergence tracks fixed f0=7.83 Hz or IAF-derived subharmonic |

## Golden Ratio Analysis

| Script | Description |
|--------|-------------|
| golden_ratio_analysis.py | Detects FOOOF peaks and analyzes pairwise frequency ratios for golden ratio proximity |
| golden_ratio_emotions.py | Golden ratio relationships in EMOTIONS dataset (256 Hz sampling) |
| golden_ratio_per_file_histograms.py | Individual histograms of FOOOF-detected peaks for each EEG file |
| eeg_phi.py | Tests golden ratio organization hypothesis in EEG spectral peaks using phase-rotation null control |

## Null Controls

| Script | Description |
|--------|-------------|
| null_control_1_unconstrained.py | Tests if phi-ratios appear in random spectral peaks without frequency constraints |
| null_control_2_constrained.py | Tests if SIE events show better phi-convergence than random triplets in same SR ranges |
| null_control_2_distributional.py | Tests if phi-convergence emerges from independent sampling of marginal distributions |
| null_control_3_phase_randomization.py | Tests if SR-brain coupling reflects genuine temporal phase relationships vs spectral artifacts |
| null_control_3_shuffled_data.py | Tests if phi-ratios depend on specific SR1-SR3-SR5 pairings by shuffling values |
| null_control_4_event_vs_random.py | Tests if detected SIE events have significantly better metrics than random windows |
| null_control_4_hybrid.py | Fast validation of SIE event quality vs random windows using pre-computed results |
| null_control_4_random_windows.py | Tests if ignition events show better sr_score than random 20s windows |
| null_control_4_random_windows_v2.py | Improved version using detection results for random window metric extraction |
| null_control_5_blind_clustering.py | Tests if FOOOF peaks naturally cluster around SR frequencies without pre-specification |
| null_control_5_pairwise.py | Blind DBSCAN clustering on paired-subject FOOOF peaks to test SR frequency privilege |
| null_control_5_per_subject.py | Per-subject blind clustering analysis with collective summary |
| null_control_7_peak_based.py | Tests if SIE events show better phi-convergence than random triplets from actual FOOOF peaks |
| null_control_examples.py | Demonstrates different file selection configurations for null control tests |
| run_null_control.py | Quick runner for null control test with predefined dataset selections |
| validate_null_control_2.py | Comprehensive validation of distributional null control model |
| notebook_null_control_helper.py | Simplified aliases for running null control tests in Jupyter notebooks |

## Replication Pipelines (by Dataset)

### LEMON
| Script | Description |
|--------|-------------|
| run_lemon_phi_replication.py | Unified protocol phi-lattice replication on LEMON preprocessed EC/EO recordings |
| run_lemon_phioctave_extraction.py | Per-phi-octave FOOOF eliminating cross-band redistribution artifact |
| run_lemon_phioctave_overlap_trim.py | Overlap-trim FOOOF solving aperiodic fit tension |
| run_lemon_raw_extraction.py | FOOOF extraction on raw .vhdr files with minimal preprocessing to test ICA effects |
| run_lemon_reextract_peaks.py | Re-extracts FOOOF peaks at [1,85] Hz from preprocessed data |
| run_lemon_amplitude_weighted.py | Weighted compliance metrics with age/cognition analyses |
| run_lemon_base_specificity.py | Tests phi-lattice effects across alternative exponential bases |
| run_lemon_phi_cognition.py | Tests golden-ratio lattice precision predicting adult cognitive performance |
| run_lemon_sensitivity_la.py | Sensitivity analyses for max_n_peaks=40 and eyes-closed conditions |
| run_lemon_structural_specificity.py | Full structural phi-specificity pipeline (Paper 2 analysis) |
| run_continuous_reanalysis.py | Recomputes compliance using Gaussian kernel density for continuous alignment |

### EEGMMIDB
| Script | Description |
|--------|-------------|
| run_eegmmidb_phi_replication.py | Locked phi-lattice protocol on EEGMMIDB (109 subjects, 14 runs each) |
| run_eegmmidb_dominant_peak.py | Dominant-peak analysis testing phi-lattice alignment across canonical bands |
| run_eegmmidb_full_pipeline.py | Complete FOOOF + GED pipeline with standard output formats |
| run_eegmmidb_global_fooof.py | FOOOF peaks without phi-octave bias for baseline comparison |
| run_eegmmidb_phioctave_overlap_trim.py | Overlap-trim per-phi-octave FOOOF solving aperiodic fit edge artifacts |
| run_eegmmidb_shuffle_prominence.py | Structural score robustness via within-band shuffle and prominence decomposition |
| compute_iaf_eegmmidb.py | Individual Alpha Frequency distribution across EEGMMIDB subjects |
| compute_iaf_eegmmidb_v2.py | IAF with higher frequency resolution, peak validation, and center-of-gravity method |
| extract_eegmmidb_peaks.py | Peak extraction using median-filter, FOOOF with critic params, and custom params |

### Dortmund
| Script | Description |
|--------|-------------|
| run_dortmund_phi_replication.py | Re-extracts from raw EDF using locked unified protocol |
| run_dortmund_dominant_peak.py | Replicates LEMON analysis on Dortmund Vital Study (608 subjects, 64 channels, 4 conditions) |
| run_dortmund_longitudinal.py | 5-year longitudinal follow-up (208 subjects) assessing within-subject stability |
| run_dortmund_overlap_trim.py | Overlap-trim FOOOF solving 1/f knee contamination, predicting EC delta sign-flip |

### HBN
| Script | Description |
|--------|-------------|
| run_hbn_phi_replication.py | Locked protocol on HBN Release 1 (136 subjects, 500 to 250 Hz) |
| run_hbn_release_phi_replication.py | Per-release phi replication runner for specified HBN data releases |

### CHBMP
| Script | Description |
|--------|-------------|
| run_chbmp_phi_replication.py | Locked protocol on Cuban Human Brain Mapping Project (282 subjects, 120-channel, EC/EO) |

### Bonn
| Script | Description |
|--------|-------------|
| run_bonn_dominant_peak.py | Phi-lattice dominant-peak analysis on Bonn epilepsy dataset (5 sets, 100 segments) |

### Brain Invaders
| Script | Description |
|--------|-------------|
| analyze_brain_invaders.py | Converts Brain Invaders EEG to pipeline format, runs FOOOF, generates golden ratio chart |
| analyze_brain_invaders_phi_lattice.py | Unified phi-lattice analysis with enrichment, band stratification, and permutation tests |
| run_continuous_ged_brain_invaders.py | Continuous GED peak detection (64 subjects, 16-channel, 512 Hz) |

### Emotions
| Script | Description |
|--------|-------------|
| analyze_emotions_phi_lattice.py | Unified phi-lattice analysis with enrichment, band stratification, and consistency metrics |
| run_continuous_ged_emotions.py | Continuous GED peak detection (2,343 sessions, 88 subjects, 14-channel, 256 Hz) |

### PhySF / VEP / MPENG
| Script | Description |
|--------|-------------|
| run_physf_baseline_analysis.py | GED validation with ignition vs baseline comparison on PhySF |
| run_vep_baseline_analysis.py | GED validation with ignition vs baseline comparison on VEP |
| run_continuous_ged_mpeng.py | Continuous GED peak detection on MPENG (900 files, Emotiv EPOC) |
| run_continuous_ged_full.py | Continuous GED sweep with per-band normalization on full PhySF |

## GED & Boundary Detection

| Script | Description |
|--------|-------------|
| run_ged_validation.py | Batch runner for phi-architecture validation using GED on SIE sessions |
| run_gedbounds_from_peaks.py | Detects empirical frequency band boundaries from ~1.58M GED peaks |
| run_true_gedbounds.py | Cohen (2021) gedBounds algorithm for covariance-based frequency boundary detection |
| run_true_continuous_ged.py | TRUE continuous GED with no band gaps; detects peaks on full profile, assigns bands post-hoc |
| run_all_datasets_true_continuous.py | Continuous GED on PhySF, Emotions, and MPENG with per-band normalization |
| run_phi_validation.py | CLI script for GED-based phi-frequency model validation on EPOC sessions |

## Band & Spectral Analysis

| Script | Description |
|--------|-------------|
| analyze_aggregate_enrichment.py | Validates combined GED peak distribution across PhySF/MPENG/Emotions against phi-lattice |
| analyze_alpha_paradox.py | Investigates prominent alpha peak vs weak enrichment at 1-degree Noble position |
| analyze_band_heterogeneity.py | Quantifies inter-band differences in phi-alignment to test gamma specificity |
| analyze_bandwidth_normalized.py | Tests whether gamma alignment is an artifact of bandwidth differences |
| analyze_nyquist_effects.py | Analyzes frequency resolution limits and Nyquist boundary truncation on gamma results |
| analyze_secondary_structure.py | Detects local peaks at phi-positions in band distributions via residual analysis |
| analyze_inverse_nobles.py | Extends phi-lattice analysis to include inverse noble positions with symmetric pairs |
| run_6band_beta_analysis.py | Adds beta sub-bands to standard 4-band dominant-peak analysis |
| run_band_structure_test.py | Phi-only analysis at 14 degree-7 positions and parameter-free log-frequency periodicity |
| run_phi_octave_replication.py | Replication statistics using phi-octave-aligned bands instead of conventional bands |
| mode_shift_analysis.py | Per-condition bootstrap CIs on KDE mode location with permutation tests |
| run_ratio_specificity.py | All four ratio-specificity analyses (D1-D4) on FOOOF/GED peak datasets |

## Dissociation & Cognition

| Script | Description |
|--------|-------------|
| noble_boundary_dissociation.py | Tests double dissociation where noble/boundary enrichment shifts across cognitive states |
| dissociation_validation.py | Validates noble-boundary dissociation with FDR correction and null distribution diagnostics |
| dissociation_validation_fast.py | Optimized dissociation validation using pre-built session-frequency indices |
| decisive_phi_tests.py | Five tests to discriminate SR vs geometric-accident explanations of phi-lattice alignment |
| decisive_phi_tests_1sd.py | Variant using hard threshold (1 sigma = 0.03) for peak alignment determination |
| ec_eo_lattice_comparison.py | Compares phi-lattice positions between eyes-closed and eyes-open conditions |

## E8 & Consciousness Modeling

| Script | Description |
|--------|-------------|
| e8_canonical_attractors.py | E8 Lie algebra geometry framework connecting golden ratio dynamics to EEG bands and SR |
| e8_consciousness_simulation.py | Simulates consciousness as coherent E8 activation using Kuramoto oscillators and phi-scaling |
| e8_energy_flow.py | Bidirectional energy flow between attractors and boundaries in E8 frequency space |
| e8_schumann_coupling.py | Combines E8 lattice model with Schumann Resonance harmonics for brain-field coupling |
| run_e8_analysis.py | Extended E8 simulation including phi-power sweeps and coupling strength sweeps |

## Ignition Detection (SIE)

| Script | Description |
|--------|-------------|
| analyze_ignition_custom.py | Detects ignition events based on SR band power fluctuations using custom detection logic |
| batch_analyze_sessions.py | Runs all configured EEG files, uses Claude API for chart analysis, generates session summaries |
| aggregate_non_sr_peaks.py | Aggregates non-SR peaks from session CSVs for cross-session clustering analysis |

## Visualization & Figure Generation

| Script | Description |
|--------|-------------|
| create_aggregate_chart.py | Aggregate GED peak distribution chart combining all datasets in log-phi style |
| create_clean_modes_chart.py | Clean modes versus predictions chart with all position types styled |
| generate_band_position_heatmap.py | Band-by-position z-score heatmap showing enrichment/depletion |
| generate_band_stratified_analysis.py | 6-panel figure showing lattice coordinate distributions within each phi-band |
| generate_session_consistency_figure.py | 8-panel session-level consistency figure with histograms and enrichment |
| generate_primary_session_consistency.py | Session-level consistency chart matching Figure 13 style |
| regenerate_charts.py | Peak distribution charts with phi-overlay bands and legend positioning |
| regenerate_combined_figures.py | GED validation combined figures from existing CSV data |
| regenerate_combined_with_continuous.py | Merges continuous GED peaks with existing analysis for complete visualization |
| regenerate_lattice.py | Lattice coordinate distribution as two-panel histogram + circular density plot |
| regenerate_nc4_figure.py | NC4 figure with fixed Panel B visualization and statistical annotations |
| visualize_prediction_errors.py | gedBounds prediction error analysis with 4 figures |
| visualize_sr_harmonics.py | Publication-quality visualizations of SR harmonic detection results |

## Schumann Resonance Exploratory Analyses

| Script | Description |
|--------|-------------|
| schumann_alignment_test.py | Permutation test for trough–SR mode overlap significance (4/5 troughs in SR ranges); includes sensitivity analysis on SR range widths, restricted T2-T5 test, and φ-constrained null |
| schumann_depth_correlation.py | Correlates trough depletion depth with SR mode amplitude; tests aligned subset (T2/SR1, T3/SR2, T5/SR5) and all pairs; uses age-binned depths and trough shape metrics |
| schumann_frequency_precision.py | Quantifies precision of trough–SR frequency alignment: absolute distance, z-scores, CI overlap, log-space distances; highlights T2–SR1 as strongest candidate (Δ = 0.01 Hz) |
| schumann_bridge_sr3.py | Analyzes bridge enrichment at ~20 Hz (f₀×φ²) vs SR3 (19.5–21.5 Hz); per-dataset bridge enrichment patterns; frequency comparison with motor control frequencies |
| schumann_developmental_trajectories.py | Compares developmental trajectories of SR-aligned (T2,T3,T5) vs non-aligned (T1) troughs; per-subject age correlations (N=1738) and age-binned analysis |
| schumann_phi_packing.py | Constrained optimization: fix avoidance zones at SR frequencies, optimize band placement to maximize desynchronization; tests whether φ emerges as packing ratio |
| schumann_alternative_nulls.py | Specificity test: compares trough alignment with SR vs chromatic scale, Bark, ERB, harmonic series, equal-log-spaced, and φ-lattice reference systems |

## Statistical Validation & Bootstrap

| Script | Description |
|--------|-------------|
| bootstrap_gap_analysis.py | Statistical significance of gap between irrational and rational structural scores |
| audit_crossbase_fairness.py | Tests whether phi's rank-1 status is artifact of asymmetric position counting |
| reanalyze_deg3.py | Re-analyzes replication CSVs with current degree-3 cross-base code |
| reanalyze_replication_csvs.py | Re-runs unified protocol statistics on pre-extracted CSVs without FOOOF re-extraction |
| run_critic_d9_on_our_data.py | Applies critic's exact D9 analysis functions to our peaks to isolate data vs framework differences |

## Data Conversion & Utilities

| Script | Description |
|--------|-------------|
| convert_arithmetic_edf_to_csv.py | Converts arithmetic EEG EDF files to pipeline-compatible CSV |
| convert_emotions_to_csv.py | Converts Emotions dataset tab-separated txt files to pipeline-compatible CSV |
| dataset_durations.py | Calculates session durations for all datasets with summary statistics |
| gen_docs.py | Generates documentation for Python function signatures using AST parsing |
| investigate_sampling_rate_artifact.py | Tests whether 128 Hz sampling rate creates integer Hz artifacts in boundaries |
| check_sr_indexing.py | Debugs which harmonic index maps to which Schumann Resonance frequency |

## Diagnostic & Debugging

| Script | Description |
|--------|-------------|
| debug_actual_values.py | Prints actual parameter values and checks FOOOF harmonic window configurations |
| debug_fooof_wrapper.py | Debugging wrapper for match_peaks_to_canonical to trace peak-matching issues |
| diagnose_zscore_discrepancy.py | Investigates why Hilbert envelope z-scores decrease with phase randomization |

## Tests & Demos

| Script | Description |
|--------|-------------|
| demo_advanced_matching.py | Demonstrates advanced FOOOF peak matching with multiple methods |
| demo_fooof_sensitivity.py | Shows how to adjust FOOOF sensitivity via max_n_peaks parameter |
| demo_freq_ranges_usage.py | Demonstrates freq_ranges parameter for grouped FOOOF windows |
| demo_peak_labels.py | FOOOF peak detection with frequency labels on visualization |
| test_bandwidth_array.py | Verifies per-frequency bandwidth array handling in sr_signature_panel |
| test_bw_array.py | Tests that bw_hz can be scalar or array-valued |
| test_bw_fix.py | Tests bandwidth parameter handling in various functions |
| test_canonical_fix.py | Tests per-event FOOOF uses canonical values, not session-detected ones |
| test_compare_with_freq_ranges.py | Verifies compare_psd_fooof and plot_fooof_fit_with_harmonics |
| test_comprehensive_metric.py | Tests comprehensive alignment metric including all position types |
| test_debug_wrapper.py | Tests debug wrapper with exact user configuration |
| test_fix_directly.py | Direct test demonstrating the per-event FOOOF canonical fix |
| test_fooof_compat.py | FOOOF/SpecParam compatibility test for specparam vs legacy fooof |
| test_fooof_harmonics.py | Demonstrates fooof_harmonics module with example usage patterns |
| test_fooof_hybrid.py | Tests FOOOF hybrid mode combining session-level and per-event harmonics |
| test_fooof_integration.py | Demonstrates all three harmonic detection methods |
| test_fooof_nan_display.py | Verifies max_n_peaks correctly displays NaN for undetected peaks |
| test_freq_ranges.py | Validates manual freq_ranges parameter for grouped FOOOF windows |
| test_halfband_debug.py | Verifies which halfband values are being used |
| test_index_mismatch.py | Diagnoses potential index mismatch between CANON and _half_bw arrays |
| test_match_method.py | Tests fooof_match_method parameter (distance, power, average) |
| test_max_n_peaks.py | Verifies fooof_max_n_peaks parameter works correctly |
| test_max_n_peaks_detailed.py | Detailed test of fooof_max_n_peaks parameter behavior |
| test_max_n_peaks_integration.py | Integration test of fooof_max_n_peaks in detect_ignitions_session |
| test_nperseg_sec.py | Verifies nperseg_sec parameter controls spectral resolution |
| test_null_control.py | Quick test of null control functions with a single file |
| test_per_harmonic_fooof.py | Tests per-harmonic FOOOF fitting with separate fits per canonical frequency |
| test_seed_scoring.py | Validates new composite seed scoring approach |
| test_user_exact_issue.py | Reproduces user's exact issue detecting 34.27 Hz in [30,34] window |
| test_window_8.py | Demonstrates correct way to render a single ignition window |
| test_window_clipping.py | Tests automatic window clipping when windows extend beyond recording bounds |
