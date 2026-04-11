"""
φⁿ frequency lattice schematic — clean design.
Each band is bounded by integer n positions. The three position types
(boundary, attractor, noble) repeat within every band.
"""

import numpy as np
import matplotlib.pyplot as plt

phi = 1.6180339887
f0 = 7.60

fig, ax = plt.subplots(figsize=(14, 5.5))

# ── Band definitions: each spans from φⁿ to φⁿ⁺¹ ──
bands = [
    (r'$\delta$',       -2, -1, '#E8E8E8'),
    (r'$\theta$',       -1,  0, '#CADCEE'),
    (r'$\alpha$',        0,  1, '#C4E3CB'),
    (r'Low $\beta$',     1,  2, '#F5DEBB'),
    (r'High $\beta$',    2,  3, '#F0C4C4'),
    (r'$\gamma$',        3,  4, '#D5CCE6'),
]

# Shaded bands
for name, n_lo, n_hi, color in bands:
    flo, fhi = f0 * phi**n_lo, f0 * phi**n_hi
    ax.axvspan(flo, fhi, color=color, alpha=0.5, zorder=0)

# ── Boundary lines + labels (integer n) ──
for n in range(-2, 5):
    fb = f0 * phi**n
    if fb < 2.0 or fb > 65:
        continue
    ax.axvline(fb, color='#C0392B', ls='-', lw=1.2, alpha=0.6, zorder=2)
    # Frequency + n label at top
    ax.text(fb, 1.02, f'{fb:.1f} Hz', ha='center', va='bottom',
            fontsize=8, color='#922B21', fontweight='bold',
            transform=ax.get_xaxis_transform())
    ax.text(fb, 1.06, f'n = {n}', ha='center', va='bottom',
            fontsize=7.5, color='#666666', fontstyle='italic',
            transform=ax.get_xaxis_transform())

# Band name labels (centered in each band)
for name, n_lo, n_hi, color in bands:
    flo, fhi = f0 * phi**n_lo, f0 * phi**n_hi
    if flo < 2.0:
        flo = 2.5
    geo = np.sqrt(flo * fhi)
    ax.text(geo, 0.82, name, ha='center', va='center',
            fontsize=14, fontweight='bold', color='#333333',
            transform=ax.get_xaxis_transform())

# ── Position markers within each band ──
# Three rows: boundary (top), attractor (mid), noble (bottom)
y_b, y_a, y_n = 0.16, 0.38, 0.60

for n in range(-1, 5):
    fb = f0 * phi**n
    if 2.5 < fb < 60:
        ax.plot(fb, y_b, 's', color='#C0392B', markersize=11,
                markeredgecolor='#711F17', markeredgewidth=0.8, zorder=5,
                transform=ax.get_xaxis_transform())
        ax.text(fb * 1.04, y_b, f'{fb:.1f}', ha='left', va='center',
                fontsize=7, color='#922B21', fontweight='bold',
                transform=ax.get_xaxis_transform())

for n in range(-1, 4):
    fa = f0 * phi**(n + 0.5)
    fn = f0 * phi**(n + 0.618)
    if 2.5 < fa < 60:
        ax.axvline(fa, color='#2E86C1', alpha=0.15, ls='--', lw=0.7, zorder=1)
        ax.plot(fa, y_a, 'o', color='#2E86C1', markersize=10,
                markeredgecolor='#1A4971', markeredgewidth=0.8, zorder=5,
                transform=ax.get_xaxis_transform())
        ax.text(fa * 1.04, y_a, f'{fa:.1f}', ha='left', va='center',
                fontsize=7, color='#1A5276', fontweight='bold',
                transform=ax.get_xaxis_transform())
    if 2.5 < fn < 60:
        ax.axvline(fn, color='#D4AC0D', alpha=0.15, ls='--', lw=0.7, zorder=1)
        ax.plot(fn, y_n, 'D', color='#D4AC0D', markersize=10,
                markeredgecolor='#6D5A07', markeredgewidth=0.8, zorder=5,
                transform=ax.get_xaxis_transform())
        ax.text(fn * 1.04, y_n, f'{fn:.1f}', ha='left', va='center',
                fontsize=7, color='#7D6608', fontweight='bold',
                transform=ax.get_xaxis_transform())

# ── Row labels on left ──
ax.text(0.01, y_b, 'Boundary (integer n)\nPeak depletion: -18%',
        ha='left', va='center', fontsize=8.5, color='#922B21',
        fontweight='bold', transform=ax.transAxes)
ax.text(0.01, y_a, 'Attractor (n + 0.5)\nPeak enrichment: +21%',
        ha='left', va='center', fontsize=8.5, color='#1A5276',
        fontweight='bold', transform=ax.transAxes)
ax.text(0.01, y_n, '1° Noble (n + 0.618)\nMax enrichment: +39%',
        ha='left', va='center', fontsize=8.5, color='#7D6608',
        fontweight='bold', transform=ax.transAxes)

# ── Equation box, upper right ──
ax.text(0.99, 0.98,
        r'$f(n) = f_0 \times \varphi^{\,n}$' + '\n'
        r'$f_0 = 7.60\ \mathrm{Hz}$' + '\n'
        r'$\varphi = 1.618...$',
        transform=ax.transAxes, ha='right', va='top',
        fontsize=11,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor='#888888', alpha=0.95))

# ── Axes ──
ax.set_xscale('log')
ax.set_xlim(2.5, 60)
ax.set_ylim(0, 1)
ax.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
ax.set_yticks([])
ax.spines[['top', 'right', 'left']].set_visible(False)

ticks = [3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60]
ax.set_xticks(ticks)
ax.set_xticklabels([str(t) for t in ticks], fontsize=9)
ax.tick_params(axis='x', which='minor', bottom=False)

ax.set_title(r'The $\varphi^{\,n}$ Frequency Lattice',
             fontsize=14, fontweight='bold', pad=35)

plt.savefig('/Users/neurokinetikz/Code/research/papers/images/phi_lattice_schematic.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/Users/neurokinetikz/Code/research/papers/images/phi_lattice_schematic.pdf',
            bbox_inches='tight', facecolor='white')
print("Saved phi_lattice_schematic.png and .pdf")
plt.close()
