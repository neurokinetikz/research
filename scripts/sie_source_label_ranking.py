#!/usr/bin/env python3
"""Read group_Q4_SR1_ratio STC and produce Desikan-Killiany label ranking + brain figure."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'source')
mne.set_log_level('ERROR')

_fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
SUBJECTS_DIR = os.path.dirname(_fs_dir)

stc_base = os.path.join(OUT_DIR, 'group_Q4_SR1_ratio')
stc = mne.read_source_estimate(stc_base, 'fsaverage')
print(f"Loaded group STC: lh={stc.lh_data.shape} rh={stc.rh_data.shape}")
print(f"Ratio: median {np.nanmedian(stc.data):.3f} p90 {np.nanpercentile(stc.data,90):.3f} "
      f"p99 {np.nanpercentile(stc.data,99):.3f} max {np.nanmax(stc.data):.3f}")

labels = mne.read_labels_from_annot('fsaverage', 'aparc',
                                     subjects_dir=SUBJECTS_DIR,
                                     verbose=False)
rows = []
for lbl in labels:
    if lbl.name.endswith('unknown-lh') or lbl.name.endswith('unknown-rh'):
        continue
    idx = lbl.get_vertices_used()
    hemi_data = stc.lh_data if lbl.hemi == 'lh' else stc.rh_data
    vert_vno = (stc.lh_vertno if lbl.hemi == 'lh' else stc.rh_vertno)
    match = np.isin(vert_vno, idx)
    if match.sum() == 0:
        continue
    r_mean = float(np.nanmean(hemi_data[match, 0]))
    r_median = float(np.nanmedian(hemi_data[match, 0]))
    rows.append({'label': lbl.name.replace('-lh','').replace('-rh',''),
                 'hemi': lbl.hemi,
                 'n_vertices': int(match.sum()),
                 'ratio_mean': r_mean,
                 'ratio_median': r_median})
df = pd.DataFrame(rows).sort_values('ratio_median', ascending=False)
csv_path = os.path.join(OUT_DIR, 'Q4_SR1_label_ranking.csv')
df.to_csv(csv_path, index=False)
print(f"\nTop 15 labels by median ratio:")
print(df.head(15).to_string(index=False))
print(f"\nBottom 5 labels by median ratio:")
print(df.tail(5).to_string(index=False))
print(f"\nSaved: {csv_path}")

# Brain render — save PNG via matplotlib backend (no PyVista/OpenGL)
try:
    brain = stc.plot(subject='fsaverage', subjects_dir=SUBJECTS_DIR,
                      hemi='split', views=['lat','med'],
                      clim=dict(kind='value', lims=[1.0, 1.15, 1.3]),
                      time_viewer=False, show_traces=False,
                      background='white', size=(1200, 600),
                      backend='matplotlib')
    if isinstance(brain, plt.Figure):
        brain.suptitle(f'B49 — SR1 ratio (Q4 event/baseline), n={134} subjects',
                        fontsize=11)
        brain.savefig(os.path.join(OUT_DIR, 'group_Q4_SR1_ratio_brain.png'),
                       dpi=120, bbox_inches='tight')
        plt.close(brain)
        print(f"Saved brain figure.")
except Exception as e:
    print(f"Brain render failed: {e}")
