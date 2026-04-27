#!/usr/bin/env python3
"""
Paper Figure 6 — Topographic variation.

Frequency-is-universal, location-is-cohort-specific. We show SR1
event-locked topographic ratio maps for the five adult cohorts that
already have rendered 16hz_topography.png panels.  HBN pediatric data
is not currently rendered as a topomap (the per-channel CSV exists but
EGI 129-ch coordinates are not in the repo); a placeholder labelled
panel notes the central-scalp pediatric pattern reported in the
cross-cohort meta and in 16hz_topography.png at the global level.

Sources (cropped SR1 panels from existing PNGs):
  - outputs/schumann/images/coupling/dortmund_composite/16hz_topography.png
  - outputs/schumann/images/coupling/lemon_composite/16hz_topography.png
  - outputs/schumann/images/coupling/lemon_EO_composite/16hz_topography.png
  - outputs/schumann/images/coupling/tdbrain_composite/16hz_topography.png
  - outputs/schumann/images/coupling/tdbrain_EO_composite/16hz_topography.png
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

REPO = Path(__file__).resolve().parent.parent
COUPLING_DIR = REPO / "outputs/schumann/images/coupling"
OUT_PNG = REPO / "papers/schumann_canonical/images/fig6_topographic_variation.png"

# (cohort_dir, label, regime_color, regime_short, location_summary)
PANELS = [
    ("dortmund_composite", "Dortmund (Adult EC, n=124)",
     "#1f78b4", "Regime 1 — Adult EC", "occipital (PO10)"),
    ("lemon_composite", "LEMON EC (Adult EC, n=153)",
     "#1f78b4", "Regime 1 — Adult EC", "posterior (PO9)"),
    ("lemon_EO_composite", "LEMON EO (Adult EO, n=155)",
     "#a6cee3", "Regime 2 — Adult EO", "temporoparietal (TP9)"),
    ("tdbrain_composite", "tdbrain EC (Clinical, n=40)",
     "#e31a1c", "Regime 3 — Clinical", "central (CPz)"),
    ("tdbrain_EO_composite", "tdbrain EO (Clinical, n=66)",
     "#e31a1c", "Regime 3 — Clinical", "central (CPz)"),
]


def crop_sr1_panel(png_path):
    """Crop the leftmost (SR1) panel from a 3-panel topo PNG."""
    img = mpimg.imread(png_path)
    h, w = img.shape[:2]
    # Format: title strip (~6%) + 3 equal topo panels with colorbars stacked
    # horizontally.  The leftmost ~33% of width is the SR1 panel.
    x_lo = int(0.00 * w)
    x_hi = int(0.34 * w)
    return img[:, x_lo:x_hi]


def main():
    n_panels = len(PANELS) + 1  # + 1 schematic for pediatric
    ncols = 3
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.0, 8.0),
                              gridspec_kw={"hspace": 0.32, "wspace": 0.05})
    axes = axes.flatten()

    for ax, (folder, label, col, regime, loc) in zip(axes, PANELS):
        png = COUPLING_DIR / folder / "16hz_topography.png"
        if not png.exists():
            ax.text(0.5, 0.5, f"missing\n{folder}", ha="center", va="center")
            ax.axis("off")
            continue
        sub = crop_sr1_panel(png)
        ax.imshow(sub)
        ax.axis("off")
        # Title + location annotation
        ax.set_title(label, fontsize=10.5, fontweight="bold", color=col)
        ax.text(0.5, -0.03, f"SR1 max @ {loc}\n{regime}",
                 transform=ax.transAxes, ha="center", va="top",
                 fontsize=9, color="#222")

    # Pediatric panel — crop SR1 panel from pooled HBN R1-R11 3-panel topo
    # (same format as adult cohort topos: 3-panel SR1/β16/SR3 PNG, leftmost
    # 33% is SR1).
    ax = axes[len(PANELS)]
    hbn_topo = REPO / "outputs/schumann/images/coupling/hbn_pooled/16hz_topography.png"
    if hbn_topo.exists():
        sub = crop_sr1_panel(hbn_topo)
        ax.imshow(sub)
        ax.axis("off")
    else:
        ax.text(0.5, 0.5, "HBN topo not rendered\n(run /tmp/render_hbn_topo.py)",
                ha="center", va="center")
        ax.axis("off")
    ax.set_title("HBN R1–R11 (Pediatric, n=2,486)",
                  fontsize=10.5, fontweight="bold", color="#ff7f00")
    ax.text(0.5, -0.03, "SR1 max @ central-scalp (E55)\nRegime 4 — Pediatric",
             transform=ax.transAxes, ha="center", va="top",
             fontsize=9, color="#222")

    # Hide any extra axes if grid > number of panels
    for ax in axes[n_panels:]:
        ax.axis("off")

    fig.suptitle("SR1 frequency is universal (7.687 Hz), but cortical topography is cohort-specific",
                  fontsize=12.5, fontweight="bold", y=1.005)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
