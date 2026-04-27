#!/usr/bin/env python3
"""Within-subject analyses of composite v2 events on the three self-recorded
cohorts (muse, insight_self, epoc_self). Single subject = user, across
channel counts 4 → 5 → 14. Also benchmarks against LEMON/Dortmund group norms.

Runs five analyses and writes a synthesis report:
  1. Test-retest reliability (session-to-session variability per device)
  2. Channel-count sensitivity (4 → 5 → 14 → 64 → 128)
  3. User vs group norms (percentile ranks)
  4. State / context from session name patterns (if any)
  5. Synthesis
"""
from __future__ import annotations
import os, glob
import numpy as np
import pandas as pd

EXPORTS = 'exports_sie'
OUT_DIR = 'outputs/schumann/images/self_within_subject'
os.makedirs(OUT_DIR, exist_ok=True)

SELF_COHORTS = {
    'muse':         ('muse_composite',         4),
    'insight_self': ('insight_self_composite', 5),
    'epoc_self':    ('epoc_self_composite',   14),
}

GROUP_COHORTS = {
    'lemon_EC':        ('lemon_composite',           64),
    'lemon_EO':        ('lemon_EO_composite',        64),
    'dortmund_EC_pre': ('dortmund_EC_pre_ses2_composite', 31),
    'dortmund_EO_pre': ('dortmund_EO_pre_composite',      31),
    'chbmp':           ('chbmp_composite',           20),
    'srm':             ('srm_composite',             64),
    'hbn_R1':          ('hbn_R1_composite',         128),
    'hbn_R6':          ('hbn_R6_composite',         128),
}

FEATURE_COLS = ['sr_score', 'sr_score_canonical', 'HSI', 'HSI_canonical',
                'FSI', 'FSI_canonical', 'msc_sr1_v', 'plv_sr1_pm5',
                'sr1', 'sr1_z_max']


def load_cohort_events(cohort_dir: str) -> pd.DataFrame:
    """Concatenate all per-session event files for a cohort."""
    ev_files = sorted(glob.glob(f'{EXPORTS}/{cohort_dir}/*_sie_events.csv'))
    frames = []
    for p in ev_files:
        try:
            df = pd.read_csv(p)
            if len(df) == 0: continue
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_cohort_summary(cohort_dir: str) -> pd.DataFrame:
    p = f'{EXPORTS}/{cohort_dir}/extraction_summary.csv'
    if os.path.isfile(p):
        return pd.read_csv(p)
    return pd.DataFrame()


# =========================================================================
# #1. Test-retest reliability (within-session consistency)
# =========================================================================
def analysis_1_reliability():
    print("\n" + "="*78)
    print("[1/5] Test-retest reliability")
    print("="*78)
    rows = []
    for name, (d, nch) in SELF_COHORTS.items():
        s = load_cohort_summary(d)
        if s.empty: continue
        ok = s[s['status']=='ok']
        n_sessions = len(ok)
        ev_rate_per_min = ok['n_events'] / (ok['duration_sec']/60.0)
        rows.append({
            'cohort': name,
            'n_ch': nch,
            'n_sessions': n_sessions,
            'total_events': int(ok['n_events'].sum()),
            'dur_med_min': float(ok['duration_sec'].median()/60),
            'ev_per_session_med': int(ok['n_events'].median()),
            'ev_per_session_iqr': f"{ok['n_events'].quantile(0.25):.0f}-{ok['n_events'].quantile(0.75):.0f}",
            'ev_per_min_med': float(ev_rate_per_min.median()),
            'ev_per_min_cv': float(ev_rate_per_min.std()/ev_rate_per_min.mean()) if ev_rate_per_min.mean()>0 else np.nan,
        })
    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT_DIR}/analysis1_reliability.csv', index=False)
    print(df.to_string(index=False))
    return df


# =========================================================================
# #2. Channel-count sensitivity
# =========================================================================
def analysis_2_channels():
    print("\n" + "="*78)
    print("[2/5] Channel-count sensitivity (within subject)")
    print("="*78)
    rows = []
    for name, (d, nch) in SELF_COHORTS.items():
        ev = load_cohort_events(d)
        if ev.empty: continue
        row = {'cohort': name, 'n_ch': nch, 'n_events': len(ev)}
        for col in ['sr_score', 'sr_score_canonical', 'HSI_canonical',
                    'msc_sr1_v', 'plv_sr1_pm5', 'sr1_z_max']:
            if col in ev.columns:
                vals = pd.to_numeric(ev[col], errors='coerce').dropna()
                row[f'{col}_med'] = float(vals.median()) if len(vals) else np.nan
                row[f'{col}_mad'] = float(np.median(np.abs(vals-vals.median()))) if len(vals) else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT_DIR}/analysis2_channels.csv', index=False)
    keys = ['cohort','n_ch','n_events','sr_score_med','HSI_canonical_med',
            'msc_sr1_v_med','plv_sr1_pm5_med']
    print(df[[k for k in keys if k in df.columns]].to_string(index=False))
    return df


# =========================================================================
# #3. User vs group norms (percentile rank)
# =========================================================================
def analysis_3_vs_group():
    print("\n" + "="*78)
    print("[3/5] User vs group norms — percentile ranks")
    print("="*78)
    group_ev = []
    for name, (d, _) in GROUP_COHORTS.items():
        ev = load_cohort_events(d)
        if ev.empty: continue
        ev = ev.copy(); ev['_group_cohort'] = name
        group_ev.append(ev)
    group_df = pd.concat(group_ev, ignore_index=True) if group_ev else pd.DataFrame()
    self_cohorts_combined = []
    for name, (d, _) in SELF_COHORTS.items():
        ev = load_cohort_events(d)
        if ev.empty: continue
        ev = ev.copy(); ev['_self_cohort'] = name
        self_cohorts_combined.append(ev)
    user_df = pd.concat(self_cohorts_combined, ignore_index=True) if self_cohorts_combined else pd.DataFrame()

    metrics = ['sr_score', 'sr_score_canonical', 'HSI_canonical',
                'msc_sr1_v', 'plv_sr1_pm5', 'sr1_z_max']
    rows = []
    for m in metrics:
        if m not in group_df.columns or m not in user_df.columns: continue
        gvals = pd.to_numeric(group_df[m], errors='coerce').dropna().values
        uvals = pd.to_numeric(user_df[m], errors='coerce').dropna().values
        if len(gvals)==0 or len(uvals)==0: continue
        u_med = float(np.median(uvals))
        # Percentile of user median within group distribution
        pct = float(100 * np.mean(gvals <= u_med))
        rows.append({
            'feature': m,
            'group_n_events': len(gvals),
            'user_n_events': len(uvals),
            'group_median': float(np.median(gvals)),
            'user_median': u_med,
            'user_pct_rank_in_group': pct,
        })
    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT_DIR}/analysis3_vs_group.csv', index=False)
    print(df.to_string(index=False))
    # Per-cohort breakdown
    rows2 = []
    for m in metrics:
        if m not in group_df.columns: continue
        for self_name in SELF_COHORTS:
            u = user_df[user_df['_self_cohort']==self_name]
            uvals = pd.to_numeric(u.get(m), errors='coerce').dropna().values
            if len(uvals)==0: continue
            rows2.append({
                'feature': m,
                'self_cohort': self_name,
                'user_median': float(np.median(uvals)),
                'user_pct_rank_vs_group': float(100*np.mean(
                    pd.to_numeric(group_df[m], errors='coerce').dropna().values
                    <= float(np.median(uvals))))
            })
    df2 = pd.DataFrame(rows2)
    df2.to_csv(f'{OUT_DIR}/analysis3_per_cohort.csv', index=False)
    return df, df2


# =========================================================================
# #4. State / context hints from session names
# =========================================================================
def analysis_4_state():
    print("\n" + "="*78)
    print("[4/5] State / context hints from session-name tokens")
    print("="*78)
    tokens_pattern = {
        'meditation': ['medit','calm','relax'],
        'focus':      ['focus','work','flow','task'],
        'QA':         ['quality','qa','assess'],
        'test':       ['test','trial'],
    }
    rows = []
    for name, (d, _) in SELF_COHORTS.items():
        s = load_cohort_summary(d)
        if s.empty: continue
        for _, r in s.iterrows():
            sid = str(r['subject_id']).lower()
            label = 'unlabeled'
            for k, toks in tokens_pattern.items():
                if any(t in sid for t in toks):
                    label = k; break
            rows.append({
                'cohort': name, 'session_id': r['subject_id'],
                'label': label, 'n_events': r['n_events'] if r['status']=='ok' else 0,
                'duration_sec': r['duration_sec'] if r['status']=='ok' else 0,
            })
    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT_DIR}/analysis4_state.csv', index=False)
    if len(df):
        by_label = df.groupby(['cohort','label']).agg(
            n_sessions=('session_id','count'),
            med_events=('n_events','median'),
            med_duration=('duration_sec','median')
        ).reset_index()
        print(by_label.to_string(index=False))
    return df


# =========================================================================
# #5. Synthesis report
# =========================================================================
def analysis_5_synthesis(d1, d2, d3a, d3b, d4):
    print("\n" + "="*78)
    print("[5/5] Synthesis")
    print("="*78)

    total_user_sessions = int(d1['n_sessions'].sum()) if not d1.empty else 0
    total_user_events = int(d1['total_events'].sum()) if not d1.empty else 0

    report = []
    report.append(f"# Composite v2 — within-subject analyses (single subject = user)")
    report.append(f"")
    report.append(f"**Data scope:** {total_user_sessions} sessions, {total_user_events} composite v2 events across 3 consumer-grade devices (4–14 channels).")
    report.append(f"")

    # #1
    report.append(f"## 1. Test-retest reliability\n")
    report.append(d1.to_string(index=False) if hasattr(d1,'to_markdown') else d1.to_string(index=False))
    report.append(f"\n**Reading:** `ev_per_min_cv` is the coefficient-of-variation of event-rate across sessions — lower = more reproducible. Muse has {d1.loc[d1.cohort=='muse','ev_per_min_cv'].iloc[0] if (d1.cohort=='muse').any() else 'N/A':.2f}, "
                  f"insight {d1.loc[d1.cohort=='insight_self','ev_per_min_cv'].iloc[0] if (d1.cohort=='insight_self').any() else 'N/A':.2f}, "
                  f"epoc {d1.loc[d1.cohort=='epoc_self','ev_per_min_cv'].iloc[0] if (d1.cohort=='epoc_self').any() else 'N/A':.2f}. "
                  f"CV ≈ 0.3 is roughly 'good test-retest' for biomarker studies.")

    # #2
    report.append(f"\n## 2. Channel-count sensitivity\n")
    cols2 = [c for c in ['cohort','n_ch','n_events','sr_score_med','HSI_canonical_med','msc_sr1_v_med','plv_sr1_pm5_med'] if c in d2.columns]
    report.append(d2[cols2].to_string(index=False) if hasattr(d2,'to_markdown') else d2[cols2].to_string(index=False))
    report.append(f"\n**Reading:** With 4 channels (muse) MSC and PLV have very few channel pairs — expect noisier estimates. With 14 channels (epoc) the streams should be closer to research-grade. If medians don't shift much across 4→5→14, that's strong evidence the composite detector is robust to low channel counts on *this* subject.")

    # #3
    report.append(f"\n## 3. User vs group norms\n")
    report.append(d3a.to_string(index=False) if hasattr(d3a,'to_markdown') else d3a.to_string(index=False))
    report.append(f"\n**Reading:** `user_pct_rank_in_group` shows where user's median event feature sits in the pooled research-grade group distribution (percentile). 50 = typical; <20 or >80 = noteworthy outlier.")
    report.append(f"\n### Per-device breakdown\n")
    if not d3b.empty:
        pivot = d3b.pivot(index='feature', columns='self_cohort', values='user_pct_rank_vs_group').round(1)
        report.append(pivot.to_string())

    # #4
    report.append(f"\n## 4. State / context hints from session names\n")
    if not d4.empty:
        bl = d4.groupby(['cohort','label']).agg(
            n_sessions=('session_id','count'),
            med_events=('n_events','median')
        ).reset_index()
        report.append(bl.to_string(index=False) if hasattr(bl,'to_markdown') else bl.to_string(index=False))
        unlabeled_frac = float((d4['label']=='unlabeled').mean())
        report.append(f"\n{unlabeled_frac*100:.0f}% of sessions have no state token in filename. If you want within-subject state analyses, add a metadata mapping.")

    # #5
    report.append(f"\n## 5. Publishable angle\n")
    report.append(f"The 4→5→14 channel ladder on one brain is unique — most reliability/device-comparison studies have ≤2 sessions per subject. With {total_user_sessions} sessions × 3 devices, per-feature ICC can be estimated with meaningful CIs. If features are stable across channel counts (Analysis #2), the framing is: **composite v2 detector is robust to consumer-grade hardware on a single subject**. If they drift, the framing is: **minimum channel count for canonical composite events is N ≥ X**.")

    text = '\n'.join(report)
    out_md = f'{OUT_DIR}/synthesis_report.md'
    with open(out_md, 'w') as f:
        f.write(text)
    print(f"\nSaved: {out_md}")


def main():
    d1 = analysis_1_reliability()
    d2 = analysis_2_channels()
    d3a, d3b = analysis_3_vs_group()
    d4 = analysis_4_state()
    analysis_5_synthesis(d1, d2, d3a, d3b, d4)
    print(f"\nAll outputs in: {OUT_DIR}/")


if __name__ == '__main__':
    main()
