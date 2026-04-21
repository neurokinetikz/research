#!/usr/bin/env python3
"""
B55 — Within-session f0 trend + between-subject f0 distribution.

Tests the visual observation that ignition f0 frequency (within the 7-8.2 Hz
Schumann SR1 band) appears to:
  (a) vary between subjects/sessions (high / med / low f0)
  (b) drift smoothly WITHIN a session (not discontinuous jumps)

If the within-session drift is real and monotonic, it's consistent with an
external driver (e.g., ionospheric cavity-size variation that shifts Schumann
resonance frequency on minute-to-hour timescales).

Data: per-event sr1 frequency + t0_net (seconds into session) from the
events CSVs. LEMON cohort (largest n, cleanest).
"""
from __future__ import annotations
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'coupling')
EVENTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'exports_sie',
                           'lemon')

# Aggregate per-event data
rows = []
for f in sorted(glob.glob(os.path.join(EVENTS_DIR, 'sub-*_sie_events.csv'))):
    df = pd.read_csv(f)
    if 'sr1' not in df.columns or 't0_net' not in df.columns:
        continue
    df = df.dropna(subset=['sr1', 't0_net'])
    # Keep events within SR1 band
    df = df[(df['sr1'] >= 7.0) & (df['sr1'] <= 8.3)]
    if len(df) == 0:
        continue
    sub_id = os.path.basename(f).replace('_sie_events.csv', '')
    df = df.sort_values('t0_net').reset_index(drop=True)
    df['subject_id'] = sub_id
    df['event_order'] = np.arange(len(df))
    df['session_dur_sec'] = df['t0_net'].max() - df['t0_net'].min()
    rows.append(df[['subject_id', 't0_net', 'sr1', 'event_order',
                     'session_dur_sec']])
all_ev = pd.concat(rows, ignore_index=True)
print(f"Total events: {len(all_ev)}  subjects: {all_ev['subject_id'].nunique()}")

# ===== Per-subject within-session slope: f0 vs time-in-session =====
print(f"\n=== Within-session f0 drift (subjects with >=4 events) ===")
per_sub = []
for sid, sub in all_ev.groupby('subject_id'):
    if len(sub) < 4:
        continue
    # Pearson (linear fit of f0 on t0)
    slope, intercept = np.polyfit(sub['t0_net'], sub['sr1'], 1)
    # Spearman for monotonic trend
    rho, p = spearmanr(sub['t0_net'], sub['sr1'])
    # Range of f0 within this session
    f_range = sub['sr1'].max() - sub['sr1'].min()
    per_sub.append({'subject_id': sid,
                     'n_events': len(sub),
                     'f_mean_hz': sub['sr1'].mean(),
                     'f_std_hz': sub['sr1'].std(),
                     'f_range_hz': f_range,
                     'linear_slope_hz_per_sec': slope,
                     'spearman_rho': rho,
                     'spearman_p': p})
ps = pd.DataFrame(per_sub)
ps.to_csv(os.path.join(OUT_DIR, 'f0_within_session_trend.csv'), index=False)

print(f"n subjects tested: {len(ps)}")
print(f"\nPer-subject f0 variation:")
print(f"  f_mean distribution: median {ps['f_mean_hz'].median():.3f} Hz  "
      f"IQR [{ps['f_mean_hz'].quantile(.25):.3f}, {ps['f_mean_hz'].quantile(.75):.3f}]")
print(f"  f_range within session: median {ps['f_range_hz'].median():.3f} Hz  "
      f"max {ps['f_range_hz'].max():.3f} Hz")
print(f"  f_std within session: median {ps['f_std_hz'].median():.3f} Hz")

# Per-subject slope distribution
pos_slope = (ps['linear_slope_hz_per_sec'] > 0).sum()
neg_slope = (ps['linear_slope_hz_per_sec'] < 0).sum()
sig_ps = ps[ps['spearman_p'] < 0.05]
print(f"\n  Slope directions: {pos_slope} pos, {neg_slope} neg "
      f"(pct pos = {pos_slope/len(ps)*100:.0f}%)")
print(f"  Subjects with p<0.05 monotonic trend: {len(sig_ps)} "
      f"(of {len(ps)}, {len(sig_ps)/len(ps)*100:.0f}%)")
print(f"    of which {(sig_ps['spearman_rho']>0).sum()} positive, "
      f"{(sig_ps['spearman_rho']<0).sum()} negative")

# Test: is % positive different from 50% (null=random)?
try:
    from scipy.stats import binomtest
    bt = binomtest(pos_slope, len(ps), p=0.5)
    print(f"  Binomial test for pos-direction bias: p={bt.pvalue:.2g}")
except Exception as e:
    print(f"  Binomial test failed: {e}")

# ===== Is within-session f0 smoother than chance? =====
# Compare observed within-session SD to SD of same-size random draws from the
# full cohort f0 distribution (shuffled null). If smooth drift is real, the
# within-session SD will be smaller than the null.
print(f"\n=== Within-session smoothness vs iid-null ===")
cohort_f = all_ev['sr1'].values
rng = np.random.default_rng(42)
observed_sd = []
null_sd = []
for sid, sub in all_ev.groupby('subject_id'):
    n = len(sub)
    if n < 4:
        continue
    observed_sd.append(sub['sr1'].std())
    # Draw N iid samples from cohort distribution
    null_samples = rng.choice(cohort_f, size=n, replace=False)
    null_sd.append(np.std(null_samples))
observed_sd = np.array(observed_sd)
null_sd = np.array(null_sd)
print(f"  Observed within-session SD: median {np.median(observed_sd):.3f}")
print(f"  Null iid-draw SD (same-n): median {np.median(null_sd):.3f}")
from scipy.stats import wilcoxon as _wilcoxon
try:
    _, p = _wilcoxon(observed_sd, null_sd)
    print(f"  Wilcoxon observed < null (drift smoothness): p={p:.3g}")
except Exception:
    pass
# Fraction with observed SD < null SD
pct_smoother = (observed_sd < null_sd).mean() * 100
print(f"  {pct_smoother:.0f}% of subjects have smaller-than-null within-session SD")

# ===== FIGURE =====
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# A — per-subject f0 mean distribution (between-subject variation)
ax = axes[0, 0]
ax.hist(ps['f_mean_hz'], bins=30, color='#2b5fb8', edgecolor='k', lw=0.3,
         alpha=0.8)
ax.axvline(7.83, color='red', ls='--', lw=1.5, label='Schumann SR1 7.83')
ax.set_xlabel('per-subject mean f0 (Hz)')
ax.set_ylabel('subjects')
ax.set_title(f'A — Between-subject f0 variation\n'
              f'n={len(ps)}  median {ps["f_mean_hz"].median():.2f} Hz  '
              f'std {ps["f_mean_hz"].std():.3f}',
              loc='left', fontweight='bold', fontsize=11)
ax.legend(); ax.grid(alpha=0.3)

# B — within-session f0 range distribution
ax = axes[0, 1]
ax.hist(ps['f_range_hz'], bins=30, color='#8c1a1a', edgecolor='k', lw=0.3,
         alpha=0.8)
ax.set_xlabel('f0 range within session (Hz)')
ax.set_ylabel('subjects')
ax.set_title(f'B — Within-session f0 variability\n'
              f'median range {ps["f_range_hz"].median():.3f} Hz  '
              f'(SR1 band width = 1.3 Hz)',
              loc='left', fontweight='bold', fontsize=11)
ax.grid(alpha=0.3)

# C — within-session slope distribution
ax = axes[1, 0]
ax.hist(ps['linear_slope_hz_per_sec'] * 60, bins=40, color='#1a9641',
         edgecolor='k', lw=0.3, alpha=0.8)
ax.axvline(0, color='k', lw=1)
median_slope = ps['linear_slope_hz_per_sec'].median() * 60
ax.axvline(median_slope, color='red', ls='--',
            label=f'median {median_slope:+.3f} Hz/min')
ax.set_xlabel('within-session f0 slope (Hz/minute)')
ax.set_ylabel('subjects')
pos_pct = pos_slope / len(ps) * 100
ax.set_title(f'C — Within-session slope\n'
              f'{pos_pct:.0f}% positive · {len(sig_ps)}/{len(ps)} '
              f'p<0.05 monotonic',
              loc='left', fontweight='bold', fontsize=11)
ax.legend(); ax.grid(alpha=0.3)

# D — examples of three subjects: smooth drift vs discontinuity
ax = axes[1, 1]
# Pick 6 subjects with ≥6 events sorted by |slope|
candidates = all_ev.groupby('subject_id').size()
cand_ids = candidates[candidates >= 6].index.tolist()
# Top-6 by |slope|
slopes_abs = ps.set_index('subject_id').loc[
    [c for c in cand_ids if c in ps['subject_id'].values],
    'linear_slope_hz_per_sec'].abs().sort_values(ascending=False)
sample_ids = slopes_abs.head(6).index.tolist()
cmap = plt.get_cmap('viridis')
for i, sid in enumerate(sample_ids):
    sub = all_ev[all_ev['subject_id'] == sid].sort_values('t0_net')
    slope, intercept = np.polyfit(sub['t0_net'], sub['sr1'], 1)
    ax.plot(sub['t0_net'] / 60, sub['sr1'], 'o-', color=cmap(i/6),
            alpha=0.7, lw=1, markersize=5,
            label=f'{sid[-6:]}  slope={slope*60:+.2f} Hz/min')
ax.set_xlabel('time in session (min)')
ax.set_ylabel('f0 (Hz)')
ax.axhline(7.83, color='red', ls=':', lw=1, alpha=0.5)
ax.set_title('D — 6 example subjects (highest |slope|)',
              loc='left', fontweight='bold', fontsize=11)
ax.legend(fontsize=7, loc='upper right')
ax.grid(alpha=0.3)

fig.suptitle('B55 — f0 within-session trend + between-subject variation '
              '(LEMON EC, SR1 band 7.0-8.3 Hz)',
              fontsize=12, y=1.02)
fig.tight_layout()
out_png = os.path.join(OUT_DIR, 'f0_within_session_trend.png')
plt.savefig(out_png, dpi=160, bbox_inches='tight')
plt.savefig(out_png.replace('.png', '.pdf'), bbox_inches='tight')
plt.close()
print(f"\nSaved: {out_png}")
