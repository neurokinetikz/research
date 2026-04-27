#!/usr/bin/env python3
"""
Manifest of composite v2 analysis scripts to run per cohort, in order.

Each entry: (script_filename, section_tag). The orchestrator runs these
sequentially on a dedicated VM per cohort, all with --cohort <cohort>
and SIE_WORKERS=28 (Pool size for Arc-scripts that use multiprocessing).

Outputs land in outputs/schumann/images/coupling/<cohort>_composite/ on
the VM and are pushed to GCS bucket for local retrieval.

Section tags map to §N in the LEMON master report
(outputs/2026-04-21-composite-v2-extraction-report.md) for cross-reference
when generating per-cohort reports.
"""
from __future__ import annotations


# Arc 2 — peri-event mechanism battery
ARC2 = [
    ('sie_perionset_triple_average_composite.py',   'A3 (§23)'),
    ('sie_dip_rebound_analysis_composite.py',       'A6 (§23b)'),
    ('sie_ignition_phase_segmentation_composite.py','A7 (§23c)'),
    ('sie_iei_distribution_composite.py',           'B2b (§24)'),
    ('sie_mechanism_battery_composite_v2.py',       'B5/B8/B10 (§25b)'),
    ('sie_phase_reset_null_composite.py',           'B10 null (§25)'),
    ('sie_b8_propagation_composite.py',             'B8 (§25c)'),
    ('sie_perionset_multistream_composite.py',      'A9 (§26)'),
    ('sie_wpli_near_far_composite.py',              'A10-B (§27)'),
    ('sie_if_corrections_composite.py',             'A10 (§28)'),
    ('sie_single_event_inspection_composite.py',    'B1 (§29)'),
]

# Arc 3 — event quality axes
ARC3 = [
    ('sie_perionset_by_quality_composite.py',       'B7 (§30)'),
    ('sie_timing_consistency_axis_composite.py',    'B16 (§31)'),
    ('sie_event_quality_axes_composite.py',         'B6 (§32)'),
    # sie_b15_shift_by_rho_composite is lemon-only (EC vs EO compare) -- excluded
    ('sie_event_quality_literal_composite.py',      'B6-lit (§33a)'),
    ('sie_sr_recentered_morphology_composite.py',   'B15-1 (§33b)'),
]

# Arc 4 — frequency-domain SR-band signature
ARC4 = [
    ('sie_sr_band_event_boost_composite.py',         'B11-B12 (§13)'),
    ('sie_sr_band_1f_normalized_composite.py',       'B13 (§14)'),
    ('sie_sr_peri_event_timecourse_composite.py',    'B14 (§15)'),
    ('sie_phi_lattice_trajectory_composite.py',      'B18 (§16)'),
    ('sie_sr_zoom_peak_composite.py',                'B19 (§17)'),
    ('sie_iaf_coupling_composite.py',                'B20/B20b (§18)'),
    ('sie_within_subject_event_peaks_composite.py',  'B22 (§19)'),
    ('sie_event_peak_covariates_composite.py',       'B23 (§20)'),
    ('sie_subject_spectral_diff_vs_ignition_composite.py', 'B24 (§21)'),
    ('sie_lattice_anchoring_composite.py',           'B25 (§22)'),
]

# Arc 5 — canonical aggregate figures
ARC5 = [
    ('sie_aggregate_psd_composite.py',               'B26+B27 (§34)'),
    ('sie_beta_peak_iaf_coupling_composite.py',      'B28 (§35)'),
    ('sie_beta_peak_covariates_composite.py',        'B29 (§36)'),
    ('sie_peri_event_sr_vs_beta_composite.py',       'B31 (§37)'),
    ('sie_a6b_msc_codip_composite.py',               'A6b (§38)'),
]

# Arc 6 — Schumann cavity interpretation
ARC6 = [
    ('sie_odd_mode_by_quartile_composite.py',        'B33+B40 (§39)'),
    ('sie_frontal_pac_hsi_composite.py',             'B38 (§40)'),
    ('sie_sr1_sr3_coupling_composite.py',            'B34 (§41)'),
    ('sie_pac_comodulogram_composite.py',            'B35 (§42)'),
    ('sie_pac_time_resolved_composite.py',           'B36 (§43)'),
    ('sie_bicoherence_fibonacci_composite.py',       'B37 (§44)'),
    ('sie_hsi_variants_composite.py',                'B39 (§45)'),
]

# Arc 7 — topography & reliability audit
ARC7 = [
    ('sie_16hz_harmonic_test_composite.py',          'B41 (§46)'),
    ('sie_16hz_topography_composite.py',             'B42 (§47)'),
    ('sie_network_reliability_composite.py',         'B45 (§48)'),
    ('sie_posterior_sr1_tightened_composite.py',     'B46 (§49)'),
    ('sie_reliability_and_directed_composite.py',    'B44 (§50)'),
]


# Full pipeline — 43 scripts across Arcs 2-7
ANALYSIS_SCRIPTS = ARC2 + ARC3 + ARC4 + ARC5 + ARC6 + ARC7


# 16 target cohorts for cross-cohort sweep (LEMON EC is done locally)
COHORTS = [
    'lemon_EO',
    'tdbrain',
    'tdbrain_EO',
    'chbmp',
    'dortmund',
    'hbn_R1',
    'hbn_R2',
    'hbn_R3',
    'hbn_R4',
    'hbn_R5',
    'hbn_R6',
    'hbn_R7',
    'hbn_R8',
    'hbn_R9',
    'hbn_R10',
    'hbn_R11',
]


if __name__ == '__main__':
    print(f"Arcs 2-7: {len(ANALYSIS_SCRIPTS)} scripts")
    print(f"Cohorts:  {len(COHORTS)}")
    print(f"Total jobs: {len(COHORTS) * len(ANALYSIS_SCRIPTS)}")
    for i, (script, tag) in enumerate(ANALYSIS_SCRIPTS, 1):
        print(f"  {i:2d}. {script:<52} {tag}")
