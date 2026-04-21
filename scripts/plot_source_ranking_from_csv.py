#!/usr/bin/env python3
"""Plot the Desikan-Killiany label ranking from Q4_SR1_label_ranking.csv."""
from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'source')
df = pd.read_csv(os.path.join(OUT_DIR, 'Q4_SR1_label_ranking.csv'))
df['full'] = df['label'] + '-' + df['hemi']
df = df.sort_values('ratio_median', ascending=False).reset_index(drop=True)

# Colour by lobe (manual mapping)
LOBE_MAP = {
    'parahippocampal': 'medial_temporal',
    'fusiform': 'temporal', 'inferiortemporal': 'temporal',
    'middletemporal': 'temporal', 'superiortemporal': 'temporal',
    'transversetemporal': 'temporal', 'bankssts': 'temporal',
    'temporalpole': 'medial_temporal', 'entorhinal': 'medial_temporal',
    'precuneus': 'parietal', 'superiorparietal': 'parietal',
    'inferiorparietal': 'parietal', 'supramarginal': 'parietal',
    'postcentral': 'parietal',
    'lingual': 'occipital', 'cuneus': 'occipital', 'lateraloccipital': 'occipital',
    'pericalcarine': 'occipital',
    'insula': 'insula', 'isthmuscingulate': 'cingulate',
    'posteriorcingulate': 'cingulate', 'caudalanteriorcingulate': 'cingulate',
    'rostralanteriorcingulate': 'cingulate',
    'precentral': 'frontal', 'paracentral': 'frontal',
    'superiorfrontal': 'frontal', 'rostralmiddlefrontal': 'frontal',
    'caudalmiddlefrontal': 'frontal',
    'parsopercularis': 'frontal', 'parstriangularis': 'frontal',
    'parsorbitalis': 'frontal',
    'lateralorbitofrontal': 'frontal', 'medialorbitofrontal': 'frontal',
    'frontalpole': 'frontal',
}
LOBE_COLOR = {
    'occipital': '#2b5fb8', 'parietal': '#1a9641',
    'temporal': '#d73027', 'medial_temporal': '#8c1a1a',
    'cingulate': '#fdae61', 'insula': '#d7b5d8',
    'frontal': '#756bb1',
}
df['lobe'] = df['label'].map(LOBE_MAP).fillna('other')
df['color'] = df['lobe'].map(LOBE_COLOR).fillna('#999999')

fig, ax = plt.subplots(figsize=(11, 14))
pos = range(len(df))
ax.barh(pos, df['ratio_median'], color=df['color'], edgecolor='k', lw=0.3)
ax.axvline(1.0, color='k', lw=1, alpha=0.6)
ax.set_yticks(pos)
ax.set_yticklabels(df['full'], fontsize=7)
ax.set_xlabel('median SR1 source-power ratio (Q4 event / baseline)')
ax.set_xlim(1.0, df['ratio_median'].max() * 1.02)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

# Legend
from matplotlib.patches import Patch
handles = [Patch(color=c, label=l) for l, c in LOBE_COLOR.items()]
ax.legend(handles=handles, loc='lower right', fontsize=9)

fig.suptitle('B49 — Source-space SR1 ratio (Q4 event/baseline) by '
              'Desikan-Killiany label · LEMON EC, n=134 subjects · '
              'fsaverage sLORETA',
              fontsize=11)
fig.tight_layout()
out_png = os.path.join(OUT_DIR, 'Q4_SR1_label_ranking.png')
plt.savefig(out_png, dpi=140, bbox_inches='tight')
plt.savefig(out_png.replace('.png', '.pdf'), bbox_inches='tight')
plt.close()
print(f"Saved: {out_png}")
