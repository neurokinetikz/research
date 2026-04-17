#!/usr/bin/env python3
"""
TDBRAIN Challenge: Diagnostic and Age Predictions
===================================================

Generates predictions for the TDBRAIN Challenge using spectral
differentiation features from the φ-lattice framework.

Submissions:
  1. ADHD vs MDD diagnostic prediction (balanced accuracy > 60% target)
  2. Age prediction (R² target: beat 28.5% from current leaderboard)

Strategy: Zero-shot -- features are defined by the φ-lattice coordinate
system (not learned from TDBRAIN data). Per-subject enrichment profiles
computed from FOOOF peaks, then simple classifier/regressor applied.

For diagnostic prediction: train on DISCOVERY set, predict REPLICATION set.
For age prediction: train on DISCOVERY set, predict REPLICATION set.

Usage:
    python scripts/tdbrain_challenge_predictions.py

Requires: TDBRAIN peaks extracted to exports_adaptive_v4/tdbrain/
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT_DIR = os.path.join(BASE_DIR, 'outputs', 'tdbrain_challenge')
PEAK_DIR = os.path.join(BASE_DIR, 'exports_adaptive_v4', 'tdbrain')
PARTICIPANTS_PATH = os.path.expanduser(
    '~/Desktop/TDBRAIN_participants_V2_data/TDBRAIN_participants_V2.tsv')

sys.path.insert(0, os.path.join(BASE_DIR, 'lib'))
from phi_frequency_model import PHI, F0, POSITION_OFFSETS, BANDS

# Trough depth parameters (paper's method)
KNOWN_TROUGHS_HZ = np.array([5.08, 7.81, 13.42, 25.30, 35.04])
TROUGH_LABELS = ['δ/θ', 'θ/α', 'α/β', 'βL/βH', 'βH/γ']
LOG_HALF_WINDOW = 0.06
LOG_FLANK_OFFSET = 0.15
MIN_POWER_PCT = 50

# Voronoi enrichment positions (12 per octave)
POSITION_U = np.array(sorted(POSITION_OFFSETS.values()))
POSITION_NAMES = sorted(POSITION_OFFSETS.keys(), key=lambda k: POSITION_OFFSETS[k])

# Band definitions
BAND_NAMES = ['theta', 'alpha', 'beta_low', 'beta_high', 'gamma']
BAND_OCTAVES = {'theta': -1, 'alpha': 0, 'beta_low': 1, 'beta_high': 2, 'gamma': 3}


def lattice_coord(freq):
    """Convert frequency to phi-octave coordinate u ∈ [0,1)."""
    return (np.log(freq / F0) / np.log(PHI)) % 1.0


def per_subject_trough_depth(freqs, trough_hz):
    """Paper's windowed log-frequency count method."""
    log_freqs = np.log(freqs)
    log_trough = np.log(trough_hz)
    trough_count = np.sum(np.abs(log_freqs - log_trough) < LOG_HALF_WINDOW)
    left_count = np.sum(np.abs(log_freqs - (log_trough - LOG_FLANK_OFFSET)) < LOG_HALF_WINDOW)
    right_count = np.sum(np.abs(log_freqs - (log_trough + LOG_FLANK_OFFSET)) < LOG_HALF_WINDOW)
    mean_flank = (left_count + right_count) / 2
    return trough_count / mean_flank if mean_flank > 0 else np.nan


def compute_voronoi_enrichment(freqs, band_octave):
    """Compute 12-position Voronoi enrichment for one band in one subject."""
    # Get frequencies in this band
    lo = F0 * PHI ** band_octave
    hi = F0 * PHI ** (band_octave + 1)
    band_freqs = freqs[(freqs >= lo) & (freqs < hi)]

    if len(band_freqs) < 15:
        return {name: np.nan for name in POSITION_NAMES}

    # Compute u-coordinates
    u = lattice_coord(band_freqs)

    # Voronoi bin edges (midpoints between adjacent positions, circular)
    sorted_pos = np.sort(POSITION_U)
    bin_edges = []
    for i in range(len(sorted_pos)):
        mid = (sorted_pos[i] + sorted_pos[(i + 1) % len(sorted_pos)]) / 2
        if i == len(sorted_pos) - 1:
            mid = (sorted_pos[i] + sorted_pos[0] + 1) / 2
            if mid >= 1:
                mid -= 1
        bin_edges.append(mid)
    bin_edges = np.sort(bin_edges)

    # Count peaks in each Voronoi cell
    n_total = len(u)
    enrichment = {}
    for name, pos_u in zip(POSITION_NAMES, POSITION_U):
        # Find nearest bin edges
        dists = np.abs(u - pos_u)
        dists = np.minimum(dists, 1 - dists)  # circular
        # Simple: count peaks closer to this position than any other
        all_dists = np.abs(u[:, None] - POSITION_U[None, :])
        all_dists = np.minimum(all_dists, 1 - all_dists)
        assignments = np.argmin(all_dists, axis=1)
        pos_idx = list(POSITION_U).index(pos_u)
        count = np.sum(assignments == pos_idx)
        expected = n_total / len(POSITION_U)
        if expected > 0:
            enrichment[name] = (count / expected - 1) * 100
        else:
            enrichment[name] = np.nan

    return enrichment


def extract_features(sub_id):
    """Extract full feature vector for one subject."""
    peak_path = os.path.join(PEAK_DIR, f'{sub_id}_peaks.csv')
    if not os.path.exists(peak_path):
        return None

    try:
        df = pd.read_csv(peak_path, usecols=['freq', 'power', 'phi_octave'])
    except Exception:
        return None

    # Power filter
    filtered = []
    for octave in df['phi_octave'].unique():
        bp = df[df.phi_octave == octave]
        if len(bp) == 0:
            continue
        thresh = bp['power'].quantile(MIN_POWER_PCT / 100)
        filtered.append(bp[bp['power'] >= thresh])
    if not filtered:
        return None
    df = pd.concat(filtered, ignore_index=True)
    freqs = df['freq'].values

    if len(freqs) < 100:
        return None

    features = {'subject': sub_id, 'n_peaks': len(freqs)}

    # 1. Trough depths (5 features)
    for trough_hz, label in zip(KNOWN_TROUGHS_HZ, TROUGH_LABELS):
        features[f'trough_{label}'] = per_subject_trough_depth(freqs, trough_hz)

    # 2. Per-band Voronoi enrichment (5 bands × 12 positions = 60 features)
    for band_name, octave in BAND_OCTAVES.items():
        enrichment = compute_voronoi_enrichment(freqs, octave)
        for pos_name, val in enrichment.items():
            features[f'{band_name}_{pos_name}'] = val

    # 3. Derived spectral differentiation metrics per band
    for band_name, octave in BAND_OCTAVES.items():
        enrichment = compute_voronoi_enrichment(freqs, octave)
        vals = [v for v in enrichment.values() if not np.isnan(v)]
        if len(vals) > 2:
            features[f'{band_name}_sd_range'] = max(vals) - min(vals)
            features[f'{band_name}_sd_std'] = np.std(vals)
            # Mountain height (attractor enrichment)
            features[f'{band_name}_mountain'] = enrichment.get('attractor', np.nan)
            # Ramp (noble_1 - boundary)
            features[f'{band_name}_ramp'] = (
                enrichment.get('noble_1', 0) - enrichment.get('boundary', 0))

    return features


def load_participants():
    """Load and classify TDBRAIN participants."""
    df = pd.read_csv(PARTICIPANTS_PATH, sep='\t')
    df['age_float'] = df['age'].str.replace(',', '.').astype(float)

    df['dx_group'] = 'OTHER'
    df.loc[df['indication'] == 'HEALTHY', 'dx_group'] = 'HEALTHY'
    df.loc[df['indication'].str.contains('ADHD', na=False) &
           ~df['indication'].str.contains('MDD', na=False), 'dx_group'] = 'ADHD'
    df.loc[df['indication'].str.contains('MDD', na=False) &
           ~df['indication'].str.contains('ADHD', na=False), 'dx_group'] = 'MDD'

    return df


def train_diagnostic_model(train_df, target_col='dx_group', pos_class='ADHD', neg_class='MDD'):
    """Train a simple logistic regression for ADHD vs MDD."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score

    subset = train_df[train_df[target_col].isin([pos_class, neg_class])].copy()
    y = (subset[target_col] == pos_class).astype(int).values

    feature_cols = [c for c in subset.columns
                    if c.startswith('trough_') or c.startswith('theta_') or
                    c.startswith('alpha_') or c.startswith('beta_') or
                    c.startswith('gamma_')]
    X = subset[feature_cols].fillna(0).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(C=0.1, max_iter=1000, class_weight='balanced')
    model.fit(X_scaled, y)

    # Training accuracy
    y_pred = model.predict(X_scaled)
    train_ba = balanced_accuracy_score(y, y_pred)

    # Cross-validation
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='balanced_accuracy')

    return model, scaler, feature_cols, train_ba, cv_scores


def train_age_model(train_df):
    """Train ridge regression for age prediction."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score

    valid = train_df.dropna(subset=['age_float']).copy()
    y = valid['age_float'].values

    feature_cols = [c for c in valid.columns
                    if c.startswith('trough_') or c.startswith('theta_') or
                    c.startswith('alpha_') or c.startswith('beta_') or
                    c.startswith('gamma_')]
    X = valid[feature_cols].fillna(0).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Ridge(alpha=10.0)
    model.fit(X_scaled, y)

    # Training R²
    train_r2 = model.score(X_scaled, y)

    # Cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')

    return model, scaler, feature_cols, train_r2, cv_scores


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("TDBRAIN Challenge: Diagnostic and Age Predictions")
    print("=" * 70)

    # Load participants
    participants = load_participants()

    # Extract features for all subjects with peaks
    print("\n--- Extracting Features ---")
    peak_files = sorted(glob.glob(os.path.join(PEAK_DIR, '*_peaks.csv')))
    if not peak_files:
        print(f"  No peak files found in {PEAK_DIR}")
        print(f"  Run extraction first: python scripts/run_f0_760_extraction.py --dataset tdbrain --parallel 28")
        return

    print(f"  {len(peak_files)} subjects with peaks")

    rows = []
    for i, f in enumerate(peak_files):
        sub_id = os.path.basename(f).replace('_peaks.csv', '')
        features = extract_features(sub_id)
        if features:
            rows.append(features)
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(peak_files)} processed")

    features_df = pd.DataFrame(rows)
    print(f"  {len(features_df)} subjects with valid features")

    # Merge with participants
    merged = features_df.merge(
        participants[['participants_ID', 'age_float', 'gender', 'dx_group', 'DISC/REP']],
        left_on='subject', right_on='participants_ID', how='inner')

    discovery = merged[merged['DISC/REP'] == 'DISCOVERY'].copy()
    replication = merged[merged['DISC/REP'] == 'REPLICATION'].copy()

    print(f"\n  Discovery: {len(discovery)} subjects")
    print(f"  Replication: {len(replication)} subjects")
    print(f"  Discovery groups: {discovery['dx_group'].value_counts().to_dict()}")

    # =============================================
    # 1. DIAGNOSTIC: ADHD vs MDD
    # =============================================
    print("\n" + "=" * 70)
    print("1. ADHD vs MDD Diagnostic Prediction")
    print("=" * 70)

    # Adults only for training
    disc_adults = discovery[discovery['age_float'] >= 18]
    print(f"  Adult discovery: {len(disc_adults)}")
    print(f"  ADHD adults: {len(disc_adults[disc_adults.dx_group == 'ADHD'])}")
    print(f"  MDD adults: {len(disc_adults[disc_adults.dx_group == 'MDD'])}")

    try:
        model_dx, scaler_dx, feat_cols_dx, train_ba, cv_scores = train_diagnostic_model(disc_adults)
        print(f"\n  Training balanced accuracy: {train_ba:.3f}")
        print(f"  5-fold CV balanced accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"  CV range: [{cv_scores.min():.3f}, {cv_scores.max():.3f}]")

        target = 0.60
        if cv_scores.mean() >= target:
            print(f"  ✓ Exceeds {target:.0%} target")
        else:
            print(f"  ✗ Below {target:.0%} target (need more features or better model)")

        # Feature importance
        coefs = pd.Series(model_dx.coef_[0], index=feat_cols_dx)
        top_pos = coefs.nlargest(5)
        top_neg = coefs.nsmallest(5)
        print(f"\n  Top features favoring ADHD:")
        for name, val in top_pos.items():
            print(f"    {name}: {val:+.3f}")
        print(f"  Top features favoring MDD:")
        for name, val in top_neg.items():
            print(f"    {name}: {val:+.3f}")

        # Predict on replication set
        rep_adhd_mdd = replication[replication['dx_group'].isin(['ADHD', 'MDD', 'OTHER'])]
        if len(rep_adhd_mdd) > 0:
            X_rep = rep_adhd_mdd[feat_cols_dx].fillna(0).values
            X_rep_scaled = scaler_dx.transform(X_rep)
            rep_adhd_mdd = rep_adhd_mdd.copy()
            rep_adhd_mdd['predicted_dx'] = ['ADHD' if p == 1 else 'MDD'
                                             for p in model_dx.predict(X_rep_scaled)]
            rep_adhd_mdd['predicted_proba_adhd'] = model_dx.predict_proba(X_rep_scaled)[:, 1]

            print(f"\n  Replication predictions: {len(rep_adhd_mdd)} subjects")
            print(f"  Predicted ADHD: {(rep_adhd_mdd['predicted_dx'] == 'ADHD').sum()}")
            print(f"  Predicted MDD: {(rep_adhd_mdd['predicted_dx'] == 'MDD').sum()}")

            rep_adhd_mdd[['subject', 'predicted_dx', 'predicted_proba_adhd']].to_csv(
                os.path.join(OUT_DIR, 'replication_dx_predictions.csv'), index=False)

    except Exception as e:
        print(f"  Error in diagnostic model: {e}")

    # =============================================
    # 2. AGE PREDICTION
    # =============================================
    print("\n" + "=" * 70)
    print("2. Age Prediction")
    print("=" * 70)

    try:
        model_age, scaler_age, feat_cols_age, train_r2, cv_r2 = train_age_model(discovery)
        print(f"  Training R²: {train_r2:.3f}")
        print(f"  5-fold CV R²: {cv_r2.mean():.3f} ± {cv_r2.std():.3f}")
        print(f"  CV range: [{cv_r2.min():.3f}, {cv_r2.max():.3f}]")

        target_r2 = 0.285  # current leaderboard best
        if cv_r2.mean() >= target_r2:
            print(f"  ✓ Exceeds leaderboard best (R²={target_r2})")
        else:
            print(f"  ~ Below leaderboard best (R²={target_r2})")

        # Also compute Pearson r for comparison with leaderboard
        valid = discovery.dropna(subset=['age_float'])
        X_train = valid[feat_cols_age].fillna(0).values
        y_pred_train = model_age.predict(scaler_age.transform(X_train))
        r_train, p_train = stats.pearsonr(valid['age_float'], y_pred_train)
        print(f"  Training Pearson r: {r_train:.3f}")

        # Predict on replication
        rep_valid = replication.dropna(subset=['age_float'])
        if len(rep_valid) > 0:
            X_rep = rep_valid[feat_cols_age].fillna(0).values
            rep_valid = rep_valid.copy()
            rep_valid['predicted_age'] = model_age.predict(scaler_age.transform(X_rep))

            print(f"\n  Replication age predictions: {len(rep_valid)} subjects")
            rep_valid[['subject', 'predicted_age']].to_csv(
                os.path.join(OUT_DIR, 'replication_age_predictions.csv'), index=False)

    except Exception as e:
        print(f"  Error in age model: {e}")

    # =============================================
    # 3. TROUGH-ONLY BASELINE (α/β depth as single feature)
    # =============================================
    print("\n" + "=" * 70)
    print("3. Single-Feature Baseline: α/β Trough Depth Only")
    print("=" * 70)

    adhd_adults = disc_adults[disc_adults.dx_group == 'ADHD']['trough_α/β'].dropna()
    mdd_adults = disc_adults[disc_adults.dx_group == 'MDD']['trough_α/β'].dropna()

    if len(adhd_adults) > 5 and len(mdd_adults) > 5:
        # Simple threshold classifier
        threshold = (adhd_adults.median() + mdd_adults.median()) / 2

        # Apply to all ADHD + MDD adults
        subset = disc_adults[disc_adults.dx_group.isin(['ADHD', 'MDD'])].dropna(subset=['trough_α/β'])
        y_true = (subset['dx_group'] == 'ADHD').astype(int)
        y_pred = (subset['trough_α/β'] > threshold).astype(int)

        from sklearn.metrics import balanced_accuracy_score
        ba = balanced_accuracy_score(y_true, y_pred)
        print(f"  Threshold: {threshold:.3f}")
        print(f"  Balanced accuracy (single feature): {ba:.3f}")
        print(f"  ADHD median depth: {adhd_adults.median():.3f}")
        print(f"  MDD median depth: {mdd_adults.median():.3f}")
        print(f"  Mann-Whitney p: {stats.mannwhitneyu(adhd_adults, mdd_adults).pvalue:.4f}")

    # =============================================
    # 4. FORMAT FOR TDBRAIN REPLICATION TEMPLATE
    # =============================================
    print("\n" + "=" * 70)
    print("4. Format Predictions for TDBRAIN Replication Template")
    print("=" * 70)

    template_path = os.path.expanduser(
        '~/Desktop/TDBRAIN_participants_V2_data/TDBRAIN_replication_template_V2.xlsx')
    if os.path.exists(template_path):
        template = pd.read_excel(template_path)
        rep_ids = template['participants_ID'].values

        # Match replication IDs to our features
        # Template uses numeric IDs; our subjects use sub-{ID}
        rep_features = []
        for pid in rep_ids:
            sub_id = f'sub-{pid}'
            match = features_df[features_df['subject'] == sub_id]
            if len(match) > 0:
                rep_features.append(match.iloc[0])

        print(f"  Template has {len(rep_ids)} rows (120 subjects, some with 2 sessions)")
        print(f"  Matched {len(rep_features)} to extracted features")

        if len(rep_features) > 0 and 'model_dx' in dir():
            # Fill in indication predictions
            rep_feat_df = pd.DataFrame(rep_features)
            X_rep = rep_feat_df[feat_cols_dx].fillna(0).values
            X_rep_scaled = scaler_dx.transform(X_rep)

            predictions = model_dx.predict(X_rep_scaled)
            probas = model_dx.predict_proba(X_rep_scaled)

            # Build prediction lookup
            pred_lookup = {}
            for feat_row, pred, proba in zip(rep_features, predictions, probas):
                sub_id = feat_row['subject']
                pid = int(sub_id.replace('sub-', ''))
                pred_lookup[pid] = {
                    'indication': 'ADHD' if pred == 1 else 'MDD',
                    'proba_adhd': proba[1],
                }

            # Fill template
            output = template.copy()
            for idx, row in output.iterrows():
                pid = row['participants_ID']
                if pid in pred_lookup:
                    output.at[idx, 'indication'] = pred_lookup[pid]['indication']

            # Also predict age if model available
            if 'model_age' in dir():
                for idx, row in output.iterrows():
                    pid = row['participants_ID']
                    sub_id = f'sub-{pid}'
                    match = features_df[features_df['subject'] == sub_id]
                    if len(match) > 0:
                        X_age = match[feat_cols_age].fillna(0).values
                        pred_age = model_age.predict(scaler_age.transform(X_age))[0]
                        # Age is already visible, so this is just validation
                        output.at[idx, 'age_predicted'] = round(pred_age, 1)

            output_path = os.path.join(OUT_DIR, 'TDBRAIN_replication_predictions.xlsx')
            output.to_excel(output_path, index=False)
            print(f"  Saved predictions to {output_path}")
            print(f"  Submit to: brainbank@brainclinics.com")
        else:
            print("  No model trained or no features matched -- skipping template fill")
    else:
        print(f"  Template not found at {template_path}")

    # Save all features
    merged.to_csv(os.path.join(OUT_DIR, 'tdbrain_all_features.csv'), index=False)
    print(f"\nAll results saved to {OUT_DIR}/")


if __name__ == '__main__':
    main()
