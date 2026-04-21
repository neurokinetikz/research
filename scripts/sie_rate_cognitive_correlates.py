#!/usr/bin/env python3
"""
B51 — SIE rate × cognitive phenotype correlates (LEMON EC).

Direct test of B49's DMN-engagement interpretation: if Q4 canonical SIEs
are spontaneous DMN / internal-mentation events, then subjects with higher
SIE rate should show:
  (+) higher NYC-Q mind-wandering / self-generated-thought scores
  (+) higher NEO-FFI Openness
  (-) slower TAP-Alertness (worse sustained attention)
  (-) slower TMT (worse executive / visual-spatial search)

All measures are from the LEMON behavioral battery (Babayan et al. 2019).
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

BASE = '/Volumes/T9/lemon_data/behavioral/Behavioural_Data_MPILMBB_LEMON'
COG = os.path.join(BASE, 'Cognitive_Test_Battery_LEMON')
EMO = os.path.join(BASE, 'Emotion_and_Personality_Test_Battery_LEMON')
META = os.path.join(BASE, 'META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv')

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'coupling')
EVENTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'exports_sie', 'lemon')
QUALITY_CSV = os.path.join(os.path.dirname(__file__), '..', 'outputs',
                            'schumann', 'images', 'quality',
                            'per_event_quality.csv')

# ==== Per-subject SIE rate (total + Q4) ====
summary = pd.read_csv(os.path.join(EVENTS_DIR, 'extraction_summary.csv'))
ok = summary[summary['status'] == 'ok'].copy()
ok['total_rate_per_min'] = ok['n_events'] / (ok['duration_sec'] / 60)

# Per-subject Q4 rate from quality CSV
qual = pd.read_csv(QUALITY_CSV).dropna(subset=['template_rho']).copy()
qual['rho_q'] = pd.qcut(qual['template_rho'], 4, labels=['Q1','Q2','Q3','Q4'])
q4_counts = qual.groupby('subject_id').apply(
    lambda df: (df['rho_q'] == 'Q4').sum()).rename('q4_n').reset_index()
rate = ok.merge(q4_counts, left_on='subject_id', right_on='subject_id',
                how='left')
rate['q4_rate_per_min'] = rate['q4_n'] / (rate['duration_sec'] / 60)

# ==== Cognitive/behavioral batteries ====
nyc = pd.read_csv(os.path.join(EMO, 'NYC_Q_lemon.csv'))
nyc.columns = [c.strip() for c in nyc.columns]
# Drop unnamed trailing columns
nyc = nyc[[c for c in nyc.columns if not c.startswith('Unnamed')]]
# Ensure numeric
nyc_items = [c for c in nyc.columns if c.startswith('NYC-Q_lemon_')]
nyc[nyc_items] = nyc[nyc_items].apply(pd.to_numeric, errors='coerce')
# Key subscores
# Content: items 1-23, Form: items 24-31
nyc['nyc_content_mean'] = nyc[[f'NYC-Q_lemon_{i}' for i in range(1, 24)]].mean(axis=1)
nyc['nyc_form_mean'] = nyc[[f'NYC-Q_lemon_{i}' for i in range(24, 32)]].mean(axis=1)
# Self-referential autobiographical: family, friends, past events, past interactions,
# important-to-me, personal worries (items 3, 4, 7, 11, 12, 18, 22)
selfref_items = [3, 4, 7, 11, 12, 18, 22]
nyc['nyc_selfreferential'] = nyc[[f'NYC-Q_lemon_{i}' for i in selfref_items]].mean(axis=1)
# Narrative coherence (form): purpose, consistent narrative (28, 29) minus fragmented (31)
nyc['nyc_narrative'] = (nyc[['NYC-Q_lemon_28', 'NYC-Q_lemon_29']].mean(axis=1)
                        - nyc['NYC-Q_lemon_31'])

# NEO-FFI
neo = pd.read_csv(os.path.join(EMO, 'NEO_FFI.csv'))
neo.columns = [c.strip() for c in neo.columns]

# TAP-Alertness
tap = pd.read_csv(os.path.join(COG, 'TAP_Alertness', 'TAP-Alertness.csv'))
tap.columns = [c.strip() for c in tap.columns]
tap['TAP_A_5'] = pd.to_numeric(tap['TAP_A_5'], errors='coerce')  # mean RT no-signal
tap['TAP_A_15'] = pd.to_numeric(tap['TAP_A_15'], errors='coerce')  # phasic alertness

# TMT
tmt = pd.read_csv(os.path.join(COG, 'TMT', 'TMT.csv'))
tmt.columns = [c.strip() for c in tmt.columns]
tmt['TMT_1'] = pd.to_numeric(tmt['TMT_1'], errors='coerce')  # Trail A seconds
tmt['TMT_5'] = pd.to_numeric(tmt['TMT_5'], errors='coerce')  # Trail B seconds
tmt['tmt_b_minus_a'] = tmt['TMT_5'] - tmt['TMT_1']

# Meta (for covariate age)
meta = pd.read_csv(META)
meta.columns = [c.strip() for c in meta.columns]

# Merge
df = rate[['subject_id', 'duration_sec', 'n_events', 'q4_n',
            'total_rate_per_min', 'q4_rate_per_min']].copy()
df = df.merge(nyc[['ID', 'nyc_content_mean', 'nyc_form_mean',
                    'nyc_selfreferential', 'nyc_narrative']],
              left_on='subject_id', right_on='ID', how='left')
df = df.merge(neo, left_on='subject_id', right_on='ID', how='left')
df = df.merge(tap[['ID', 'TAP_A_5', 'TAP_A_15']], left_on='subject_id',
              right_on='ID', how='left', suffixes=('', '_tap'))
df = df.merge(tmt[['ID', 'TMT_1', 'TMT_5', 'tmt_b_minus_a']],
              left_on='subject_id', right_on='ID', how='left',
              suffixes=('', '_tmt'))
# Pull age if present
age_col = next((c for c in meta.columns if c.lower() in
                ('age', 'age_range', 'age_yrs', 'age_category')), None)
if age_col:
    df = df.merge(meta[['ID', age_col]], left_on='subject_id',
                  right_on='ID', how='left', suffixes=('', '_meta'))

df = df.drop(columns=[c for c in df.columns if c == 'ID'])
df.to_csv(os.path.join(OUT_DIR, 'sie_rate_cognitive_merged.csv'), index=False)
print(f"Merged subjects: {len(df)}")
print(f"SIE rate (per min): median {df['total_rate_per_min'].median():.3f}")
print(f"Q4 rate (per min): median {df['q4_rate_per_min'].median():.3f}")

# ==== Correlations ====
predictions = [
    ('q4_rate_per_min',     'nyc_content_mean',
     '+', 'NYC-Q content (mind-wandering)'),
    ('q4_rate_per_min',     'nyc_selfreferential',
     '+', 'NYC-Q self-referential'),
    ('q4_rate_per_min',     'nyc_form_mean',
     '+', 'NYC-Q form (narrative richness)'),
    ('q4_rate_per_min',     'nyc_narrative',
     '+', 'NYC-Q narrative coherence'),
    ('q4_rate_per_min',     'NEOFFI_OpennessForExperiences',
     '+', 'NEO-FFI Openness'),
    ('q4_rate_per_min',     'NEOFFI_Neuroticism',
     '?', 'NEO-FFI Neuroticism (exploratory)'),
    ('q4_rate_per_min',     'TAP_A_5',
     '+', 'TAP-Alertness RT (higher = slower)'),
    ('q4_rate_per_min',     'TAP_A_15',
     '?', 'TAP-Alertness phasic benefit'),
    ('q4_rate_per_min',     'TMT_1',
     '+', 'TMT-A seconds (higher = slower)'),
    ('q4_rate_per_min',     'TMT_5',
     '+', 'TMT-B seconds (higher = slower)'),
    # Same predictions but for total rate
    ('total_rate_per_min',  'nyc_content_mean',
     '+', 'NYC-Q content (all-event rate)'),
    ('total_rate_per_min',  'nyc_selfreferential',
     '+', 'NYC-Q self-referential (all-event rate)'),
]

def partial_spearman(x, y, z):
    """Partial Spearman correlation of x with y controlling for z."""
    from scipy.stats import rankdata
    valid = pd.concat([x, y, z], axis=1).dropna()
    if len(valid) < 5:
        return np.nan, np.nan, 0
    rx = rankdata(valid.iloc[:, 0]); ry = rankdata(valid.iloc[:, 1])
    rz = rankdata(valid.iloc[:, 2])
    # Regress x and y on z, correlate residuals
    bx, ax = np.polyfit(rz, rx, 1); by, ay = np.polyfit(rz, ry, 1)
    res_x = rx - (bx * rz + ax); res_y = ry - (by * rz + ay)
    r, p = spearmanr(res_x, res_y)
    return r, p, len(valid)


rows = []
print(f"\n{'Predictor':<25}{'Target':<40}{'dir':<5}"
      f"{'full ρ':>8}{'full p':>8}"
      f"{'age-ctl ρ':>11}{'age-ctl p':>10}"
      f"{'n>=2Q4 ρ':>10}{'n>=2Q4 p':>10}")
print('-' * 130)
sub_q4 = df[df['q4_n'] >= 2].copy()
age_col = 'Age' if 'Age' in df.columns else None
for x, y, expected, label in predictions:
    valid = df[[x, y]].dropna()
    n_full = len(valid)
    if n_full < 5:
        continue
    rho_full, p_full = spearmanr(valid[x], valid[y])
    # Partial out age
    if age_col and age_col in df.columns:
        rho_age, p_age, n_age = partial_spearman(df[x], df[y], df[age_col])
    else:
        rho_age = p_age = np.nan
    # Subset ≥2 Q4 events
    valid_sub = sub_q4[[x, y]].dropna()
    if len(valid_sub) >= 5:
        rho_sub, p_sub = spearmanr(valid_sub[x], valid_sub[y])
    else:
        rho_sub, p_sub = np.nan, np.nan
    rows.append({'predictor': x, 'target': y, 'label': label,
                 'expected_direction': expected, 'n_full': n_full,
                 'rho_full': rho_full, 'p_full': p_full,
                 'rho_age_controlled': rho_age, 'p_age_controlled': p_age,
                 'n_q4ge2': len(valid_sub),
                 'rho_q4ge2': rho_sub, 'p_q4ge2': p_sub})
    print(f"{x:<25}{y:<40}{expected:<5}"
          f"{rho_full:>+8.2f}{p_full:>8.2g}"
          f"{rho_age:>+11.2f}{p_age:>10.2g}"
          f"{rho_sub:>+10.2f}{p_sub:>10.2g}")

pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'sie_rate_cognitive_corrs.csv'),
                           index=False)
print(f"\nQ4-rate zero-inflation: {(df['q4_n']==0).mean()*100:.0f}% subjects "
      f"have 0 Q4 events; {(df['q4_n']<2).mean()*100:.0f}% have <2.")
print(f"n subjects with >=2 Q4 events: {len(sub_q4)}")

# ==== FIGURE ====
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
plot_tests = [
    ('q4_rate_per_min', 'nyc_content_mean',
     'Q4 SIE rate (/min)', 'NYC-Q content mean (1-9)',
     'A · Mind-wandering (content)'),
    ('q4_rate_per_min', 'nyc_selfreferential',
     'Q4 SIE rate (/min)', 'NYC-Q self-referential items mean',
     'B · Self-referential thinking'),
    ('q4_rate_per_min', 'NEOFFI_OpennessForExperiences',
     'Q4 SIE rate (/min)', 'NEO-FFI Openness',
     'C · Openness to experience'),
    ('q4_rate_per_min', 'TAP_A_5',
     'Q4 SIE rate (/min)', 'TAP-Alertness RT no-signal (ms, higher = slower)',
     'D · Sustained alertness RT'),
    ('q4_rate_per_min', 'TMT_1',
     'Q4 SIE rate (/min)', 'TMT-A time (s, higher = slower)',
     'E · Trail Making A'),
    ('q4_rate_per_min', 'TMT_5',
     'Q4 SIE rate (/min)', 'TMT-B time (s, higher = slower)',
     'F · Trail Making B'),
]
for ax, (x, y, xl, yl, title) in zip(axes, plot_tests):
    valid = df[[x, y]].dropna()
    if len(valid) < 5:
        ax.set_axis_off(); continue
    rho, p = spearmanr(valid[x], valid[y])
    ax.scatter(valid[x], valid[y], s=24, alpha=0.6, color='steelblue',
               edgecolor='k', lw=0.3)
    # OLS line
    xs = np.array(sorted(valid[x]))
    slope, intercept = np.polyfit(valid[x], valid[y], 1)
    ax.plot(xs, slope * xs + intercept, color='red', lw=1.5)
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    color = '#8c1a1a' if p < 0.05 else '#666'
    ax.set_title(f'{title}\nn={len(valid)}  ρ={rho:+.2f}  p={p:.2g}',
                  loc='left', fontweight='bold', fontsize=10, color=color)
    ax.grid(alpha=0.3)

fig.suptitle('B51 — Q4 SIE rate vs LEMON cognitive/behavioral measures — '
              'testing the DMN-engagement hypothesis',
              fontsize=12, y=1.01)
fig.tight_layout()
out_png = os.path.join(OUT_DIR, 'sie_rate_cognitive_correlates.png')
plt.savefig(out_png, dpi=160, bbox_inches='tight')
plt.savefig(out_png.replace('.png', '.pdf'), bbox_inches='tight')
plt.close()
print(f"\nSaved: {out_png}")
