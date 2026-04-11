# lib/ Index

## Phi-Frequency Architecture

| File | Description |
|------|-------------|
| phi_frequency_model.py | Complete φⁿ frequency prediction framework with 8 position types per phi-octave |
| phi_replication.py | Unified phi-lattice replication protocol with dominant-peak and prediction scoring |
| phi_validation_pipeline.py | Batch processing and publication table generators for GED phi-frequency validation |
| continuous_compliance.py | Gaussian KDE for continuous phi-lattice structural scoring metrics |
| ratio_specificity.py | Golden ratio specificity testing with phase-rotation nulls against 11 other ratios |
| peak_distribution_analysis.py | Unbiased PSD peak detection and comparison against phi-frequency predictions |

## GED (Generalized Eigendecomposition)

| File | Description |
|------|-------------|
| ged_band_analysis.py | Band-constrained GED analysis with position-type contrasts per EEG band |
| ged_bounds.py | GED frequency optimization testing attractor-boundary framework for harmonics |
| ged_bounds_clustering.py | GED peak boundary detection using kernel density estimation and clustering |
| ged_phi_analysis.py | GED-based validation of golden ratio (φⁿ) frequency predictions |
| ged_validation_pipeline.py | Integration layer for GED-based φⁿ architecture hypothesis validation |
| ged_poster_figure.py | 4-panel poster visualization of boundary vs attractor comparison metrics |
| true_gedbounds.py | Cohen-based covariance frequency boundary detection for spatial structure changes |

## Schumann Resonance & Harmonics

| File | Description |
|------|-------------|
| schumann_coherence.py | Per-channel Welch MSC at Schumann harmonics with wavelet coherence time series |
| harmonic_coherence.py | Magnitude-squared coherence computation at target frequencies with windowing |
| harmonic_groups.py | SR harmonic group analysis with adaptive parameters and null threshold building |
| harmonic_locking.py | Harmonic phase-locking metrics (H-PLI, XH-PLI, SubH-PLI) for SR fundamentals |
| harmonic_resonance.py | Spectral harmonicity and spatial mode analysis for Schumann harmonics |
| harmonics.py | Morlet wavelet spike detector for transient Schumann harmonic activity |
| fooof_harmonics.py | FOOOF/SpecParam-based Schumann harmonic detection from periodic peaks |
| resonant_modes.py | Spatial harmonic mode projection testing Schumann band concentration and resonance |
| shape_vs_resonance.py | Distinguishes waveform morphology from true multi-mode resonance via IRASA and bicoherence |
| spatial_source_harmonics.py | H-PLI topographies, source localization, and harmonic network graphs at Schumann frequencies |
| detect_ignition.py | Ignition/expansion event detection via Schumann harmonics with robust SR envelope tracking |
| ignition_rebound.py | Band power plotting and topographic mapping for ignition vs rebound conditions |
| non_sr_clustering.py | Collection and clustering of non-Schumann resonance peaks from FOOOF analysis |

## Cross-Frequency Coupling

| File | Description |
|------|-------------|
| cross_frequency.py | Phase-amplitude coupling and harmonically-locked cross-frequency interactions at Schumann frequencies |
| cross_frequency_harmonics.py | CF-PLV, PAC, and cross-frequency directionality tied to Schumann harmonic ladder |
| cross_frequency_region_coupling.py | Cross-channel PAC comodulograms and n:m phase locking across channel pairs |
| pac_multiplexing.py | Phase-amplitude coupling (theta/alpha→gamma) dynamics and Schumann field correlation |
| temporal_holography.py | Trial binning by reference phase at event onset for multiplexed spectral/PAC analysis |
| directionality_harmonics.py | Cross-system directional coupling (PDC/GC) and phase metrics across harmonic frequencies |

## Directed Connectivity & Information Flow

| File | Description |
|------|-------------|
| directed_connectivity.py | DLPFC↔sensory top-down connectivity using beamforming, dPLI, and Granger causality |
| directional_coupling.py | Right-DLPFC→sensory directed connectivity via dPLI and Granger causality with band analysis |
| causal_routing.py | Directed connectivity measures (DTF, PDC) from VAR models on band-limited EEG |
| information_flow.py | Frequency-domain Granger, PDC, DTF, and transfer entropy directionality analysis |
| frequency_domain_coupling.py | Multi-taper MSC and wavelet coherence for frequency-domain coupling |

## Network & Graph Analysis

| File | Description |
|------|-------------|
| network_graph_hubs.py | Graph metrics (small-world, clustering, modularity, hubs) per band with null models |
| network_geometry.py | EEG network geometry using manifold embedding (PCA, t-SNE, UMAP) across frequency bands |
| network_coupling.py | EEG-Schumann coupling via PLV graphs and ROI-level coherence with surrogate controls |
| connectome.py | Connectome harmonics via eigendecomposition of normalized graph Laplacian matrices |
| connectome_harmonics.py | EEG projection onto spatial harmonic modes measuring conscious state expansion |
| surface_cuts.py | Multi-seed min-cut capacities between brain subsystems with degree-preserving surrogates |
| entanglement_entropy.py | Multi-channel integration metrics (TC, O-info, LZc) and PLV-based graph complexity |
| entanglement_geometry.py | Plotting utilities for min-cut coherence and integration metrics across frequency bands |

## Nonlinear Dynamics & Criticality

| File | Description |
|------|-------------|
| chaos_metrics.py | Nonlinear dynamics validation using RQA, Lyapunov exponents, and correlation dimension |
| criticality.py | Criticality signatures via 1/f exponent, DFA, and avalanche statistics in band-limited EEG |
| attractor_geometry.py | Topological data analysis of attractor geometry via delay embedding and persistent homology |
| attractor_topology.py | Phase-space attractor reconstruction with embedding, Lyapunov exponents, and recurrence analysis |
| multiscale_entropy_and_fractal_scaling.py | Multiscale entropy and detrended fluctuation analysis with surrogate null bands |

## State Dynamics & Metastability

| File | Description |
|------|-------------|
| hidden_markov.py | Event-related and HMM-based state analysis on EEG spectrograms with permutation tests |
| microstate_segmentation.py | EEG microstate clustering and segmentation with temporal dynamics validation |
| dynamic_connectivity_metastability.py | Sliding-window PLV/coherence with Kuramoto synchrony and k-means state clustering |
| temporal_dynamics.py | Lead/lag quantification among Schumann families via envelope cross-correlation |

## Geometry & Manifold Embedding

| File | Description |
|------|-------------|
| emergent_geometry.py | Manifold embedding (Isomap/UMAP) of phase-distance matrices for emergent geometric structure |
| informational_geometry.py | State manifold embedding (PCA/Isomap/UMAP) with geometry validation metrics |
| toroidal_phase.py | Toroidal phase structure (S1×S1) analysis from two bands with torus metrics |

## Spectral & Time-Frequency Analysis

| File | Description |
|------|-------------|
| median_filter_peaks.py | Welch PSD + median-filter peak extraction for spectral analysis comparison |
| psd_waterfall.py | Grand waterfall/heatmap visualization from accumulated ignition-window PSD rows |
| synchrosqueeze.py | Ridge-sharp time-frequency analysis for SR fundamentals using synchrosqueezed CWT |
| wavelet_coherence.py | Wavelet coherence time-frequency analysis with ridge tracking and cluster permutation testing |

## Utilities & Infrastructure

| File | Description |
|------|-------------|
| utilities.py | Generic signal processing utilities: filters, PSD, entropy metrics, and file I/O |
| lemon_utils.py | LEMON-specific utility functions for phi-lattice cognition analysis pipeline |
| session_metadata.py | Parses subject, device, context, and dataset identifiers from EEG file paths |
| extra.py | Miscellaneous utility code with alternative timecourse analysis implementations |
| test.py | Testing module with visualization utilities, color schemes, and spectral helpers |
