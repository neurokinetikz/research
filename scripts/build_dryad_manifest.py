#!/usr/bin/env python3
"""
Build MANIFEST.csv for the dryad/ directory.

For each file (excluding README.md and MANIFEST.csv itself):
    - relative path
    - size in bytes
    - SHA256 hex digest
    - brief description (populated from a hand-curated mapping)
"""

import csv
import hashlib
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRYAD = os.path.join(BASE, 'dryad')
OUT = os.path.join(DRYAD, 'MANIFEST.csv')

# Hand-curated descriptions. Anything not listed here gets a default
# categorical description.
DESCRIPTIONS = {
    # Peaks
    'peaks_fooof_v3.csv.gz': 'FOOOF/specparam-extracted peaks, all redistributed datasets',
    'peaks_irasa.csv.gz': 'IRASA-extracted peaks, method-independence replication',
    # Per-subject enrichment
    'enrichment_per_subject_fooof.csv': 'Per-subject enrichment profiles (population anchor, FOOOF)',
    'enrichment_per_subject_irasa.csv': 'Per-subject enrichment profiles (population anchor, IRASA)',
    'enrichment_metrics_fooof.csv': 'Derived enrichment metrics per subject (FOOOF)',
    'enrichment_metrics_irasa.csv': 'Derived enrichment metrics per subject (IRASA)',
    'enrichment_per_subject_iaf_anchored.csv.gz': 'Part V: Per-subject enrichment under population AND IAF anchor, 16 strata concatenated',
    'iaf_estimates.csv': 'Part V: Per-subject IAF estimates and anchor f0_i',
    # Demographics
    'subjects_demographics.csv': 'Subject ID, dataset, age, sex, handedness, recording condition',
    # Part I (coordinate system)
    'log_scaling_bandwidth_stability.csv': 'Part I: Trough position stability across 30 KDE bandwidths',
    'log_scaling_model_comparison.csv': 'Part I: Geometric series BIC comparison (7 named ratios + linear + free)',
    'log_scaling_per_dataset_ratios.csv': 'Part I: Per-dataset inter-trough ratios (44 strata ratios)',
    'bootstrap_geo_means.csv': 'Part I: 1000 subject-level bootstrap geometric mean ratios',
    'bootstrap_trough_position_cis.csv': 'Part I: Bootstrap CIs per trough position',
    'boundary_named_systems.csv': 'Part I: Profile simplicity by named coordinate system',
    'boundary_slide.csv': 'Part I: Boundary slide analysis (single-boundary ±25% sweep)',
    'boundary_sweep_results.csv': 'Part I: 36x36 boundary sweep grid results',
    'phi_lattice_positions_reference.csv': 'Phi-lattice Voronoi landmark positions reference',
    'per_subject_trough_depths.csv': 'Part I/II: Per-subject trough depth estimates',
    # Within-band
    'within_band_test1_scaling.csv': 'Part I: Within-band scaling comparison (log/linear/ERB/mel/sqrt)',
    'within_band_test2_landmark.csv': 'Part I: Landmark capture test vs random positions',
    'within_band_test3_alignment.csv': 'Part I: Feature alignment permutation test',
    'within_band_test4_periodicity.csv': 'Part I: Within-band enrichment periodicity autocorrelation',
    'within_band_test5_noble_rational.csv': 'Part I: Noble-number vs simple-rational enrichment permutation',
    # Part II / Part III correlations
    'correlations_age_dortmund.csv': 'Part III: Dortmund age x 90-feature FDR',
    'correlations_age_hbn.csv': 'Part III: HBN pool age x 90-feature FDR',
    'correlations_age_lemon.csv': 'Part III: LEMON age x 90-feature FDR',
    'correlations_age_tdbrain.csv': 'TDBRAIN: age x 90-feature FDR (FOOOF)',
    'correlations_age_tdbrain_irasa.csv': 'TDBRAIN: age x 90-feature FDR (IRASA)',
    'correlations_attention_hbn.csv': 'Part III: HBN CBCL attention x 90-feature FDR',
    'correlations_cognitive_ec.csv': 'Part III: LEMON cognitive x 90-feature FDR (eyes-closed, 720 tests)',
    'correlations_cognitive_eo.csv': 'Part III: LEMON cognitive x 90-feature FDR (eyes-open replication)',
    'correlations_cognitive_iaf_partialed.csv': 'Part III: LEMON cognitive with IAF partialing',
    'correlations_cognitive_tdbrain.csv': 'TDBRAIN: cognitive x 90-feature FDR',
    'correlations_externalizing_hbn.csv': 'Part III: HBN CBCL externalizing x 90-feature FDR',
    'correlations_handedness_hbn.csv': 'Part III: HBN handedness x 90-feature FDR (null control)',
    'correlations_internalizing_hbn.csv': 'Part III: HBN CBCL internalizing x 90-feature FDR',
    'correlations_p_factor_hbn.csv': 'Part III: HBN CBCL p-factor x 90-feature FDR',
    'correlations_personality_lemon.csv': 'Part III: LEMON personality x 90-feature FDR (null control)',
    'correlations_personality_tdbrain.csv': 'TDBRAIN: NEO-FFI personality x 90-feature FDR (null)',
    'cross_band_coupling_dortmund.csv': 'Part III: Cross-band alpha-beta_L coupling in Dortmund',
    'cross_band_coupling_hbn.csv': 'Part III: Cross-band alpha-beta_L coupling in HBN',
    'cross_band_coupling_lemon.csv': 'Part III: Cross-band alpha-beta_L coupling in LEMON',
    'cross_band_coupling_tdbrain.csv': 'TDBRAIN: Cross-band alpha-beta_L coupling',
    'ec_eo_state_comparison.csv': 'Part III: Eyes-closed vs eyes-open state comparison (LEMON + Dortmund)',
    'lifespan_trajectory.csv': 'Part III: Lifespan trajectory points (all datasets combined)',
    'adult_vs_pediatric.csv': 'Part III: Adult-vs-pediatric signed-comparison table',
    'reliability_cross_condition.csv': 'Part III: Cross-condition reliability',
    'reliability_longitudinal_icc.csv': 'Part III: Dortmund 5-year longitudinal ICC(2,1)',
    'reliability_within_session.csv': 'Part III: Within-session reliability',
    'hbn_cross_release_consistency.csv': 'Part III: HBN R1-R11 cross-release consistency',
    'hbn_per_release_age_correlations.csv': 'Part III: Per-HBN-release age correlations',
    'hbn_sex_differences.csv': 'Part III: HBN sex differences in enrichment',
    # Part IV
    'enrichment_comparison_full.csv': 'Part IV: FOOOF vs IRASA enrichment side-by-side',
    'enrichment_10dataset_comparison.csv': 'Historical: 10-dataset enrichment comparison (includes TDBRAIN)',
    'enrichment_9dataset_comparison.csv': 'Historical: 9-dataset enrichment comparison (pre-TDBRAIN)',
    'irasa_subsample_fdr_distribution.csv': 'Part IV: FOOOF-subsampled-to-IRASA-yield FDR survivor distribution',
    'irasa_subsample_summary.csv': 'Part IV: FOOOF vs IRASA peak yield summary',
    'power_sensitivity_analysis.csv': 'Part IV: Power-threshold sensitivity analysis',
    # TDBRAIN
    'tdbrain_adhd_vs_healthy.csv': 'TDBRAIN: ADHD vs healthy control comparison',
    'tdbrain_adhd_vs_mdd.csv': 'TDBRAIN: ADHD vs MDD comparison',
    'tdbrain_mdd_vs_healthy.csv': 'TDBRAIN: MDD vs healthy control comparison',
    'tdbrain_age_controlled_diagnostic.csv': 'TDBRAIN: Age-controlled diagnostic group comparisons',
    'tdbrain_fooof_irasa_enrichment.csv': 'TDBRAIN: FOOOF vs IRASA enrichment side-by-side',
    'tdbrain_power_sensitivity.csv': 'TDBRAIN: Power-threshold sensitivity',
    'tdbrain_regional_trough_depths.csv': 'TDBRAIN: Regional trough depth by ROI',
    'tdbrain_sex_differences.csv': 'TDBRAIN: Sex differences in enrichment',
    'tdbrain_test_retest_fooof.csv': 'TDBRAIN: Test-retest ICC (FOOOF, ~2min recordings)',
    'tdbrain_test_retest_irasa.csv': 'TDBRAIN: Test-retest ICC (IRASA)',
    'tdbrain_trough_depth_by_age.csv': 'TDBRAIN: Trough depth by age bin (includes late-life)',
    # Part V
    'iaf_anchored_fdr_cognitive_lemon.csv': 'Part V: LEMON cognitive FDR under both anchors (360 tests)',
    'iaf_anchored_fdr_age_hbn.csv': 'Part V: HBN developmental FDR under both anchors (90 features)',
    'iaf_anchored_fdr_age_dortmund.csv': 'Part V: Dortmund adult aging FDR under both anchors',
    'iaf_anchored_power_matched_subsamples.csv': 'Part V: 100 HBN subsamples at N=516 FDR counts',
    'iaf_anchored_power_matched_summary.csv': 'Part V: Power-matched control summary statistics',
    'iaf_anchored_tertile_stratified.csv': 'Part V: Within-IAF-tertile correlations for 5 target features',
    'iaf_anchored_trimmed_pool.csv': 'Part V: 5%-tail-trimmed pool vs full pool for 5 target features',
    'iaf_anchored_reassignment_detail.csv': 'Part V: Per-subject band reassignment counts (construct validation)',
    'iaf_anchored_sanity_check.txt': 'Part V: Plain-text sanity-check report',
}


def sha256_of(path, bufsize=1024 * 1024):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(bufsize)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main():
    rows = []
    skip = {'README.md', 'MANIFEST.csv'}
    for name in sorted(os.listdir(DRYAD)):
        if name in skip or name.startswith('.'):
            continue
        path = os.path.join(DRYAD, name)
        if not os.path.isfile(path):
            continue
        size = os.path.getsize(path)
        digest = sha256_of(path)
        desc = DESCRIPTIONS.get(name, 'See README for category')
        rows.append({
            'file': name,
            'size_bytes': size,
            'size_human': human_size(size),
            'sha256': digest,
            'description': desc,
        })
    with open(OUT, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['file', 'size_bytes', 'size_human', 'sha256', 'description'])
        writer.writeheader()
        writer.writerows(rows)
    total_bytes = sum(r['size_bytes'] for r in rows)
    print(f"Wrote {OUT}: {len(rows)} files, {human_size(total_bytes)} total")
    # Count undescribed
    undescribed = [r['file'] for r in rows
                   if r['description'] == 'See README for category']
    if undescribed:
        print(f"WARNING: {len(undescribed)} files without explicit description:")
        for f in undescribed:
            print(f"  - {f}")


def human_size(n):
    for u in ('B', 'KB', 'MB', 'GB'):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


if __name__ == '__main__':
    main()
