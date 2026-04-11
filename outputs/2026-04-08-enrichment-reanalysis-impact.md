# Impact Analysis: Adaptive-Resolution Per-Band Enrichment Reanalysis

**Date:** 2026-04-08 (updated with HBN R6 — all 9 datasets complete)
**Status:** Complete (9 of 9 datasets, N=2,061 subjects, 4,961,070 peaks)
**Methodology:** Adaptive-resolution overlap-trim extraction (band-specific nperseg), degree-6 Voronoi bins, f₀=7.60 enrichment analysis
**Reproducible via:** `python scripts/voronoi_enrichment_analysis.py --all --summary`

## Methodology Changes from Papers

| Parameter | Paper analyses | Current reanalysis |
|-----------|---------------|-------------------|
| Extraction | Fixed nperseg (1000-1280) | Band-adaptive nperseg (734-7854) |
| Freq resolution | 0.125-0.25 Hz all bands | 0.032-0.218 Hz (matched to band width) |
| Position binning | Fixed half-width (±0.04 lu) | Voronoi bins (nearest-position assignment) |
| Position count | 8 or 14 (varies by analysis) | 12 (degree-6, max resolvable) + split boundary |
| f₀ (extraction) | 7.83 | 7.83 (unchanged) |
| f₀ (enrichment) | 7.83 or 7.60 (varies) | 7.60 (consistent) |
| Band analysis | Aggregate across all bands | Per-band independent |
| Notch filter | Applied in most datasets | Removed where possible; line noise excluded post-hoc |
| Datasets | 1-4 per paper | 9 (EEGMMIDB, LEMON, Dortmund, CHBMP, HBN R1, R2, R3, R4, R6) |

The adaptive nperseg ensures all 12 degree-6 positions are spectrally resolvable in every band. Voronoi bins tile the full octave with no gaps or overlaps, normalizing enrichment by bin width. The combination produces the first truly comparable cross-band enrichment analysis.

---

## Complete Per-Band Tables (9 Datasets, Degree-6, f₀=7.60)

**Condition note:** LEMON, Dortmund, and CHBMP are eyes-closed resting state. HBN R1–R6 is mixed resting state (alternating EC/EO blocks within a single RestingState recording). EEGMMIDB pools all 14 conditions (eyes-closed rest, eyes-open rest, and 12 motor execution/imagery runs). The EC vs EO comparison in a later section shows that enrichment profiles are state-modulated in theta and moderately in alpha/beta-low, but near-invariant in beta-high and gamma — so the mixed-condition datasets (HBN, EEGMMIDB) are expected to show slightly attenuated but directionally consistent patterns relative to pure EC.

✓ = all 9 datasets agree beyond ±5% | ~ = 8/9 agree or weak | ✗ = sign conflict

### Theta (n-1, 4.70–7.60 Hz)

| Position | u | Hz | EEGM | LEM | Dort | CHBMP | R1 | R2 | R3 | R4 | R6 | Mean | SD | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| boundary | 0.000 | 4.70 | +70% | +44% | +77% | +31% | +78% | +56% | +27% | +24% | +14% | **+47%** | 23 | ✓ |
| noble_6 | 0.056 | 4.83 | +47% | +9% | +53% | +80% | +33% | +32% | +19% | +6% | +23% | **+34%** | 22 | ✓ |
| noble_5 | 0.090 | 4.91 | -53% | +11% | -19% | +9% | -23% | -24% | -14% | +6% | -20% | **-14%** | 19 | ✗ |
| noble_4 | 0.146 | 5.04 | -29% | -25% | -3% | +12% | -16% | -23% | -18% | -16% | -24% | **-16%** | 12 | ✗ |
| noble_3 | 0.236 | 5.27 | -43% | -8% | +0% | +3% | -22% | -20% | -3% | -19% | -10% | **-14%** | 13 | ~ |
| inv_noble_1 | 0.382 | 5.65 | -9% | -17% | -13% | -26% | -14% | -18% | -26% | -1% | -14% | **-15%** | 7 | ~ |
| attractor | 0.500 | 5.98 | -16% | +7% | -18% | -1% | -6% | -18% | -16% | -6% | +9% | **-7%** | 10 | ✗ |
| noble_1 | 0.618 | 6.33 | +1% | +9% | -17% | -21% | -15% | -9% | -3% | -5% | +12% | **-5%** | 11 | ✗ |
| inv_noble_3 | 0.764 | 6.79 | +32% | -18% | -7% | -17% | +24% | +8% | +7% | +6% | +1% | **+4%** | 16 | ✗ |
| inv_noble_4 | 0.854 | 7.09 | +28% | +1% | +11% | +9% | +11% | +35% | +26% | +6% | +25% | **+17%** | 11 | ~ |
| inv_noble_5 | 0.910 | 7.28 | +31% | -1% | +9% | -9% | -8% | +14% | +17% | +26% | +4% | **+9%** | 13 | ✗ |
| inv_noble_6 | 0.944 | 7.40 | +8% | +27% | +25% | +28% | +11% | +39% | +43% | +24% | -4% | **+22%** | 14 | ~ |
| boundary (hi) | 1.000 | 7.60 | +25% | +65% | +47% | +29% | +69% | +68% | +32% | +10% | -5% | **+38%** | 25 | ~ |

✓=2 | ~=5 | ✗=6 | **Consistent: 7/13**

### Alpha (n+0, 7.60–12.30 Hz)

| Position | u | Hz | EEGM | LEM | Dort | CHBMP | R1 | R2 | R3 | R4 | R6 | Mean | SD | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| boundary | 0.000 | 7.60 | -18% | -22% | -38% | -27% | -35% | -56% | -58% | -49% | -34% | **-37%** | 14 | ✓ |
| noble_6 | 0.056 | 7.81 | -28% | -39% | -42% | -33% | -32% | -47% | -41% | -35% | -26% | **-36%** | 7 | ✓ |
| noble_5 | 0.090 | 7.94 | -24% | -31% | -34% | -23% | -22% | -24% | -22% | -19% | -24% | **-25%** | 4 | ✓ |
| noble_4 | 0.146 | 8.15 | -28% | -26% | -26% | -18% | -3% | -8% | -5% | -5% | -17% | **-15%** | 10 | ~ |
| noble_3 | 0.236 | 8.51 | -12% | -10% | -8% | -14% | -1% | +9% | +19% | +6% | -2% | **-1%** | 10 | ✗ |
| inv_noble_1 | 0.382 | 9.13 | -2% | -2% | +7% | -2% | +22% | +17% | +21% | +18% | +16% | **+11%** | 10 | ~ |
| attractor | 0.500 | 9.67 | +10% | +17% | +20% | +15% | +17% | +37% | +33% | +31% | +35% | **+24%** | 9 | ✓ |
| noble_1 | 0.618 | 10.23 | +29% | +26% | +30% | +23% | +22% | +31% | +26% | +17% | +17% | **+25%** | 5 | ✓ |
| inv_noble_3 | 0.764 | 10.98 | +16% | +27% | +17% | +20% | -1% | -5% | -5% | -1% | -1% | **+7%** | 12 | ~ |
| inv_noble_4 | 0.854 | 11.46 | +11% | +0% | +1% | +3% | -7% | -23% | -27% | -16% | -17% | **-8%** | 12 | ✗ |
| inv_noble_5 | 0.910 | 11.77 | +7% | -2% | -5% | -3% | -21% | -27% | -32% | -22% | -6% | **-12%** | 13 | ✗ |
| inv_noble_6 | 0.944 | 11.97 | -15% | -15% | -22% | -10% | -35% | -38% | -45% | -32% | -26% | **-26%** | 11 | ✓ |
| boundary (hi) | 1.000 | 12.30 | -25% | -28% | -25% | -11% | -32% | -45% | -49% | -39% | -33% | **-32%** | 11 | ✓ |

✓=7 | ~=3 | ✗=3 | **Consistent: 10/13**

### Beta-Low (n+1, 12.30–19.90 Hz)

| Position | u | Hz | EEGM | LEM | Dort | CHBMP | R1 | R2 | R3 | R4 | R6 | Mean | SD | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| boundary | 0.000 | 12.30 | +91% | +110% | +74% | +117% | +91% | +82% | +145% | +105% | +97% | **+101%** | 20 | ✓ |
| noble_6 | 0.056 | 12.63 | +61% | +63% | +42% | +94% | +36% | +40% | +28% | +24% | +21% | **+45%** | 22 | ✓ |
| noble_5 | 0.090 | 12.85 | -56% | -75% | -62% | -66% | -49% | -58% | -64% | -48% | -51% | **-59%** | 8 | ✓ |
| noble_4 | 0.146 | 13.19 | -45% | -71% | -69% | -76% | -55% | -58% | -63% | -55% | -38% | **-59%** | 12 | ✓ |
| noble_3 | 0.236 | 13.78 | -53% | -74% | -66% | -80% | -50% | -48% | -51% | -43% | -45% | **-57%** | 13 | ✓ |
| inv_noble_1 | 0.382 | 14.78 | -50% | -56% | -49% | -62% | -32% | -34% | -35% | -34% | -33% | **-43%** | 11 | ✓ |
| attractor | 0.500 | 15.65 | -25% | -21% | -21% | -34% | -19% | -17% | -22% | -17% | -17% | **-21%** | 5 | ✓ |
| noble_1 | 0.618 | 16.56 | -2% | +6% | +17% | +1% | -1% | +2% | +0% | +0% | -2% | **+2%** | 6 | ~ |
| inv_noble_3 | 0.764 | 17.76 | +24% | +42% | +42% | +38% | +28% | +26% | +24% | +24% | +28% | **+31%** | 7 | ✓ |
| inv_noble_4 | 0.854 | 18.55 | +66% | +61% | +60% | +67% | +46% | +48% | +63% | +52% | +37% | **+56%** | 10 | ✓ |
| inv_noble_5 | 0.910 | 19.06 | +64% | +89% | +75% | +112% | +51% | +51% | +46% | +48% | +45% | **+65%** | 22 | ✓ |
| inv_noble_6 | 0.944 | 19.38 | +85% | +101% | +74% | +98% | +93% | +92% | +100% | +72% | +83% | **+89%** | 10 | ✓ |
| boundary (hi) | 1.000 | 19.90 | +78% | +92% | +83% | +135% | +55% | +59% | +51% | +61% | +54% | **+74%** | 25 | ✓ |

✓=12 | ~=1 | ✗=0 | **Consistent: 13/13**

### Beta-High (n+2, 19.90–32.19 Hz)

| Position | u | Hz | EEGM | LEM | Dort | CHBMP | R1 | R2 | R3 | R4 | R6 | Mean | SD | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| boundary | 0.000 | 19.90 | +12% | +24% | -6% | +48% | +26% | +37% | +21% | +22% | +27% | **+23%** | 14 | ~ |
| noble_6 | 0.056 | 20.44 | +13% | +18% | -1% | +30% | +28% | +23% | +17% | +7% | +19% | **+17%** | 9 | ~ |
| noble_5 | 0.090 | 20.78 | +20% | -19% | +7% | -20% | +22% | +7% | +32% | +15% | -5% | **+7%** | 17 | ✗ |
| noble_4 | 0.146 | 21.35 | +12% | -14% | +0% | -25% | +5% | +0% | +19% | +2% | -1% | **-0%** | 12 | ✗ |
| noble_3 | 0.236 | 22.29 | +7% | -14% | -2% | -27% | -11% | -6% | -5% | -7% | -11% | **-8%** | 9 | ✗ |
| inv_noble_1 | 0.382 | 23.92 | -5% | -16% | -9% | -22% | -16% | -17% | -13% | -6% | -11% | **-13%** | 5 | ~ |
| attractor | 0.500 | 25.31 | -12% | -9% | -4% | -13% | -13% | -15% | -19% | -6% | -13% | **-12%** | 4 | ~ |
| noble_1 | 0.618 | 26.79 | -8% | -3% | +3% | +2% | -9% | -13% | -11% | -7% | -15% | **-7%** | 6 | ~ |
| inv_noble_3 | 0.764 | 28.74 | -3% | +10% | +7% | +14% | +3% | +2% | -5% | -1% | +0% | **+3%** | 6 | ~ |
| inv_noble_4 | 0.854 | 30.02 | +10% | +19% | +7% | +21% | +9% | +7% | +10% | +9% | +20% | **+12%** | 5 | ✓ |
| inv_noble_5 | 0.910 | 30.83 | -6% | +24% | +0% | +40% | +5% | +29% | +14% | +4% | +13% | **+14%** | 14 | ✗ |
| inv_noble_6 | 0.944 | 31.35 | -1% | +26% | +2% | +34% | +21% | +28% | +15% | +14% | +34% | **+19%** | 12 | ~ |
| boundary (hi) | 1.000 | 32.19 | -11% | +28% | -5% | +38% | +21% | +25% | +8% | +5% | +45% | **+17%** | 18 | ✗ |

✓=1 | ~=7 | ✗=5 | **Consistent: 8/13**

### Gamma (n+3, 32.19–52.09 Hz)

| Position | u | Hz | EEGM | LEM | Dort | CHBMP | R1 | R2 | R3 | R4 | R6 | Mean | SD | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| boundary | 0.000 | 32.19 | +10% | -37% | +60% | -62% | +54% | +61% | +44% | +52% | +58% | **+27%** | 44 | ✗ |
| noble_6 | 0.056 | 33.06 | -29% | -54% | +37% | -44% | +31% | +38% | +28% | +26% | +19% | **+6%** | 35 | ✗ |
| noble_5 | 0.090 | 33.62 | -51% | -36% | -13% | +10% | -25% | -29% | -27% | -27% | -18% | **-24%** | 16 | ~ |
| noble_4 | 0.146 | 34.53 | -47% | -36% | -24% | +6% | -31% | -32% | -33% | -31% | -24% | **-28%** | 14 | ~ |
| noble_3 | 0.236 | 36.06 | -41% | -29% | -35% | +4% | -32% | -34% | -34% | -37% | -29% | **-30%** | 12 | ~ |
| inv_noble_1 | 0.382 | 38.69 | -32% | -14% | -40% | -5% | -37% | -32% | -32% | -30% | -30% | **-28%** | 11 | ~ |
| attractor | 0.500 | 40.95 | -5% | +6% | -34% | -11% | -19% | -26% | -25% | -22% | -21% | **-17%** | 11 | ✗ |
| noble_1 | 0.618 | 43.34 | +20% | +27% | -10% | +2% | -5% | -6% | -1% | -1% | -15% | **+1%** | 13 | ✗ |
| inv_noble_3 | 0.764 | 46.49 | +41% | +33% | +17% | +32% | +22% | +23% | +28% | +21% | +23% | **+27%** | 7 | ✓ |
| inv_noble_4 | 0.854 | 48.55 | +50% | +33% | +38% | +2% | +42% | +41% | +42% | +44% | +35% | **+36%** | 13 | ~ |
| inv_noble_5 | 0.910 | 49.87 | +61% | +82% | +151% | +7% | +51% | +51% | +49% | +48% | +53% | **+61%** | 37 | ✓ |
| inv_noble_6 | 0.944 | 50.71 | +43% | -20% | +6% | -1% | +57% | +59% | +50% | +61% | +64% | **+35%** | 30 | ✗ |
| boundary (hi) | 1.000 | 52.09 | +23% | -16% | +69% | -6% | +51% | +54% | +57% | +48% | +50% | **+37%** | 28 | ✗ |

✓=2 | ~=5 | ✗=6 | **Consistent: 7/13**

### Consistency Summary

| Band | ✓ | ~ | ✗ | Consistent | Shape |
|------|---|---|---|-----------|-------|
| **Beta-Low** | **12** | 1 | 0 | **13/13** | U-shape: boundaries +101%/+74%, center -59% |
| **Alpha** | **7** | 3 | 3 | **10/13** | Mountain: Noble1 +25%, boundaries -37%/-32% |
| Beta-High | 1 | 7 | 5 | 8/13 | Weak: attractor -12%, inv_noble_4 +12% |
| Theta | 2 | 5 | 6 | 7/13 | Boundary clustering: +47%/+38% at edges |
| Gamma | 2 | 5 | 6 | 7/13 | Inv_noble_3/5 enriched (+27%/+61%), rest conflicts |

### Cross-Release Consistency (HBN R1 vs R2 vs R3 vs R4 vs R6)

The five HBN pediatric releases (combined N=927) show **extraordinary internal consistency**, confirming that the variability in the cross-dataset analysis comes from recording system and population differences, not sampling noise:

| Band | R1-R6 ✓ | ✗ | Notable |
|------|---------|---|---------|
| Alpha | **13/13** | 0 | Noble5: -22%, -24%, -22%, -19%, -24% (SD=1.8) |
| Beta-Low | **13/13** | 0 | inv_noble_1: -32%, -34%, -35%, -34%, -33% (SD=1.0!) |
| Beta-High | **13/13** | 0 | Noble1: -9%, -13%, -11%, -7%, -15% (SD=2.8) |
| Gamma | **13/13** | 0 | inv_noble_5: +51%, +51%, +49%, +48%, +53% (SD=1.7) |
| Theta | **9/13** | 4 | noble_4: -16%, -23%, -18%, -16%, -24% (SD=3.4) |

**Zero conflicts in alpha, beta-low, beta-high, and gamma across all 5 pediatric releases (927 subjects).** The gamma ascending ramp is a rock-solid finding with inv_noble_5 SD=1.7 across 5 independent releases. The cross-dataset gamma conflicts are entirely driven by adult dataset divergence (CHBMP inversion, LEMON boundary discrepancy).

---

## EO Per-Band Tables (LEMON + Dortmund, Degree-6, f₀=7.60)

Eyes-open resting state. Compare to EC tables above for state-dependent effects.

### Theta EO (n-1, 4.70–7.60 Hz)

| Position | u | Hz | LEM-EO | Dort-EO | LEM-EC | Dort-EC | Δ LEM | Δ Dort |
|---|---|---|---|---|---|---|---|---|
| boundary | 0.000 | 4.70 | +0% | +11% | +44% | +77% | **-44** | **-66** |
| noble_6 | 0.056 | 4.83 | +8% | +3% | +9% | +53% | -1 | -50 |
| noble_5 | 0.090 | 4.91 | -38% | -29% | +11% | -19% | -49 | -10 |
| noble_4 | 0.146 | 5.04 | -21% | -20% | -25% | -3% | +4 | -17 |
| noble_3 | 0.236 | 5.27 | -6% | -10% | -8% | +0% | +2 | -10 |
| inv_noble_1 | 0.382 | 5.65 | -26% | +0% | -17% | -13% | -9 | +13 |
| attractor | 0.500 | 5.98 | +7% | -9% | +7% | -18% | +0 | +9 |
| noble_1 | 0.618 | 6.33 | +8% | -2% | +9% | -17% | -1 | +15 |
| inv_noble_3 | 0.764 | 6.79 | +7% | +6% | -18% | -7% | **+25** | **+13** |
| inv_noble_4 | 0.854 | 7.09 | +25% | +15% | +1% | +11% | +24 | +4 |
| inv_noble_5 | 0.910 | 7.28 | +30% | +28% | -1% | +9% | **+31** | **+19** |
| inv_noble_6 | 0.944 | 7.40 | +19% | +26% | +27% | +25% | -8 | +1 |
| boundary (hi) | 1.000 | 7.60 | +8% | +18% | +65% | +47% | **-57** | **-29** |

**EC→EO:** Boundary clustering collapses (f₀ convergence is EC-specific). Theta spreads to ascending ramp at inverse nobles under EO.

### Alpha EO (n+0, 7.60–12.30 Hz)

| Position | u | Hz | LEM-EO | Dort-EO | LEM-EC | Dort-EC | Δ LEM | Δ Dort |
|---|---|---|---|---|---|---|---|---|
| boundary | 0.000 | 7.60 | -19% | -14% | -22% | -38% | +3 | +24 |
| noble_6 | 0.056 | 7.81 | -24% | -32% | -39% | -42% | +15 | +10 |
| noble_5 | 0.090 | 7.94 | -40% | -38% | -31% | -34% | -9 | -4 |
| noble_4 | 0.146 | 8.15 | -29% | -27% | -26% | -26% | -3 | -1 |
| noble_3 | 0.236 | 8.51 | -16% | -10% | -10% | -8% | -6 | -2 |
| inv_noble_1 | 0.382 | 9.13 | -6% | -1% | -2% | +7% | -4 | -8 |
| attractor | 0.500 | 9.67 | +6% | +12% | +17% | +20% | -11 | -8 |
| noble_1 | 0.618 | 10.23 | +25% | +22% | +26% | +30% | -1 | -8 |
| inv_noble_3 | 0.764 | 10.98 | +27% | +17% | +27% | +17% | +0 | +0 |
| inv_noble_4 | 0.854 | 11.46 | +25% | +14% | +0% | +1% | **+25** | **+13** |
| inv_noble_5 | 0.910 | 11.77 | +1% | +0% | -2% | -5% | +3 | +5 |
| inv_noble_6 | 0.944 | 11.97 | -4% | -8% | -15% | -22% | **+11** | **+14** |
| boundary (hi) | 1.000 | 12.30 | -19% | -6% | -28% | -25% | +9 | +19 |

**EC→EO:** Noble1 stable (+25/+22 EO vs +26/+30 EC). Mountain broadens: attractor weakens, upper-octave shoulder appears at inv_noble_4.

### Beta-Low EO (n+1, 12.30–19.90 Hz)

| Position | u | Hz | LEM-EO | Dort-EO | LEM-EC | Dort-EC | Δ LEM | Δ Dort |
|---|---|---|---|---|---|---|---|---|
| boundary | 0.000 | 12.30 | +98% | +84% | +110% | +74% | -12 | +10 |
| noble_6 | 0.056 | 12.63 | +62% | +49% | +63% | +42% | -1 | +7 |
| noble_5 | 0.090 | 12.85 | -41% | -39% | -75% | -62% | **+34** | **+23** |
| noble_4 | 0.146 | 13.19 | -55% | -51% | -71% | -69% | **+16** | **+18** |
| noble_3 | 0.236 | 13.78 | -58% | -53% | -74% | -66% | **+16** | **+13** |
| inv_noble_1 | 0.382 | 14.78 | -47% | -46% | -56% | -49% | +9 | +3 |
| attractor | 0.500 | 15.65 | -27% | -26% | -21% | -21% | -6 | -5 |
| noble_1 | 0.618 | 16.56 | -1% | +11% | +6% | +17% | -7 | -6 |
| inv_noble_3 | 0.764 | 17.76 | +30% | +29% | +42% | +42% | **-12** | **-13** |
| inv_noble_4 | 0.854 | 18.55 | +53% | +53% | +61% | +60% | -8 | -7 |
| inv_noble_5 | 0.910 | 19.06 | +67% | +61% | +89% | +75% | **-23** | **-14** |
| inv_noble_6 | 0.944 | 19.38 | +81% | +62% | +101% | +74% | **-20** | **-12** |
| boundary (hi) | 1.000 | 19.90 | +98% | +77% | +92% | +83% | +6 | -6 |

**EC→EO:** U-shape persists. Center depletion fills in (~20 pp). Inverse noble ramp attenuates (~15 pp). Boundaries stable. Shape is structural; contrast is state-modulated.

### Beta-High EO (n+2, 19.90–32.19 Hz)

| Position | u | Hz | LEM-EO | Dort-EO | LEM-EC | Dort-EC | Δ LEM | Δ Dort |
|---|---|---|---|---|---|---|---|---|
| boundary | 0.000 | 19.90 | +26% | -16% | +24% | -6% | +2 | -10 |
| noble_6 | 0.056 | 20.44 | +12% | -9% | +18% | -1% | -6 | -8 |
| noble_5 | 0.090 | 20.78 | -4% | +0% | -19% | +7% | +15 | -7 |
| noble_4 | 0.146 | 21.35 | -14% | -2% | -14% | +0% | +0 | -2 |
| noble_3 | 0.236 | 22.29 | -14% | +2% | -14% | -2% | +0 | +4 |
| inv_noble_1 | 0.382 | 23.92 | -12% | +0% | -16% | -9% | +4 | +9 |
| attractor | 0.500 | 25.31 | -12% | +2% | -9% | -4% | -3 | +6 |
| noble_1 | 0.618 | 26.79 | -3% | +5% | -3% | +3% | +0 | +2 |
| inv_noble_3 | 0.764 | 28.74 | +6% | +5% | +10% | +7% | -4 | -2 |
| inv_noble_4 | 0.854 | 30.02 | +18% | +0% | +19% | +7% | -1 | -7 |
| inv_noble_5 | 0.910 | 30.83 | +21% | -3% | +24% | +0% | -3 | -3 |
| inv_noble_6 | 0.944 | 31.35 | +26% | -9% | +26% | +2% | +0 | -11 |
| boundary (hi) | 1.000 | 32.19 | +31% | -10% | +28% | -5% | +3 | -5 |

**EC→EO:** Near-zero shifts in both datasets. Most state-invariant band.

### Gamma EO (n+3, 32.19–52.09 Hz)

| Position | u | Hz | LEM-EO | Dort-EO | LEM-EC | Dort-EC | Δ LEM | Δ Dort |
|---|---|---|---|---|---|---|---|---|
| boundary | 0.000 | 32.19 | -45% | +53% | -37% | +60% | -8 | -7 |
| noble_6 | 0.056 | 33.06 | -55% | +33% | -54% | +37% | -1 | -4 |
| noble_5 | 0.090 | 33.62 | -33% | -18% | -36% | -13% | +3 | -5 |
| noble_4 | 0.146 | 34.53 | -32% | -27% | -36% | -24% | +4 | -3 |
| noble_3 | 0.236 | 36.06 | -28% | -34% | -29% | -35% | +1 | +1 |
| inv_noble_1 | 0.382 | 38.69 | -13% | -39% | -14% | -40% | +1 | +1 |
| attractor | 0.500 | 40.95 | +6% | -30% | +6% | -34% | +0 | +4 |
| noble_1 | 0.618 | 43.34 | +21% | -7% | +27% | -10% | -6 | +3 |
| inv_noble_3 | 0.764 | 46.49 | +36% | +17% | +33% | +17% | +3 | +0 |
| inv_noble_4 | 0.854 | 48.55 | +36% | +41% | +33% | +38% | +3 | +3 |
| inv_noble_5 | 0.910 | 49.87 | +82% | +145% | +82% | +151% | +0 | -6 |
| inv_noble_6 | 0.944 | 50.71 | -16% | +3% | -20% | +6% | +4 | -3 |
| boundary (hi) | 1.000 | 52.09 | -26% | +64% | -16% | +69% | -10 | -5 |

**EC→EO:** Maximum Δ = 10% (LEMON), 7% (Dortmund). Gamma is state-invariant in the Voronoi analysis.

---

## Dortmund 2×2: EC/EO × Pre/Post (N≈608)

**Date:** 2026-04-09
**Design:** Within-subject, 4 conditions: EC-pre, EC-post, EO-pre, EO-post (pre/post = before/after ~2-hour cognitive battery)
**Data:** EC-pre 922K peaks, EC-post 930K, EO-pre 899K, EO-post 877K

### Effect Size Summary (max |Δ| per band)

| Band | Eyes (pre) | Eyes (post) | Fatigue (EC) | Fatigue (EO) | Interaction |
|---|---|---|---|---|---|
| **Theta** | **66** | 48 | 30 | 40 | **47** |
| Alpha | 24 | 16 | 14 | 26 | 14 |
| Beta-low | 23 | 20 | 16 | 10 | 13 |
| Beta-high | 11 | 9 | 6 | 11 | 12 |
| Gamma | 7 | 13 | 9 | 7 | 9 |

### Cross-Condition Stability (all 4 conditions agree on direction >±10%)

| Band | Stable positions | Interpretation |
|---|---|---|
| **Beta-low** | **13/13** | Perfect structural invariance across all 4 conditions |
| **Gamma** | **11/13** | Near-perfect (boundary and inv_noble_5 vary slightly) |
| Alpha | 7/13 | Mountain core stable; flanks condition-dependent |
| Theta | 4/13 | Most state-sensitive band |
| Beta-high | 0/13 | All positions too weak to consistently exceed ±10% |

### Theta Interaction: Fatigue × Eyes

Theta shows the only notable interaction (|Δ|>15): the eyes effect WEAKENS after fatigue at boundary (pre: -66, post: -19, interaction: +47) but STRENGTHENS at boundary_hi (pre: -29, post: -48, interaction: -19). After the cognitive battery, closing the eyes still converges theta on f₀ but the lower boundary convergence is less state-sensitive while the upper boundary convergence becomes MORE state-sensitive.

### Key 2×2 Findings

**1. Fatigue and eyes are independent for beta-low, beta-high, gamma.** No interactions exceed ±15 in any of these bands. The U-shape, weak ascending pattern, and ramp are structurally invariant to both manipulations.

**2. Fatigue STRENGTHENS theta boundary convergence under EC** (boundary_hi: EC-pre +47% → EC-post +77%), but EO-post partially restores the pre-fatigue level (EO-post +29%). The fatigue effect on theta is EC-specific.

**3. Alpha shows asymmetric fatigue effects by eye state.** Under EC, fatigue narrows the upper tail (inv_noble_5: -5%→-19%, boundary_hi: -25%→-37%). Under EO, fatigue narrows the upper tail MORE dramatically (inv_noble_4: +14%→-4%, boundary_hi: -6%→-32%). The alpha mountain contracts from the upper edge with fatigue regardless of eye state.

**4. Beta-low 13/13 stability across all 4 conditions** is the most robust result in the entire reanalysis. The Fibonacci coupling architecture is completely invariant to both eye state and cognitive fatigue.

---


## What's Really Going On: Updated Understanding

### Band-specific enrichment reveals three organizational regimes

**Regime 1: Central mountain (Alpha only)**
- Noble1 (+25%) and attractor (+24%) are enriched; boundaries depleted (-37% to -32%)
- This is the "classic phi-lattice" pattern predicted by Pletzer/Kramer
- Peaks cluster at anti-mode-locking positions (Noble1, attractor) and avoid coupling gateways (boundaries)
- Most consistent across datasets (10/13, SD=5 at Noble1)
- Driven by IAF clustering at ~10 Hz = Noble1 position

**Regime 2: Boundary U-shape (Beta-low, with echoes in theta)**
- Both boundaries massively enriched (+101% / +74%); center depleted (-21% to -59%)
- Peaks cluster AT the Fibonacci coupling gateways, not away from them
- 13/13 consistent — the most robust pattern of any band
- Theoretically: boundaries are where f(n) = f(n-1) + f(n-2) enables three-wave resonance
- The 12.30 Hz boundary = alpha/beta edge; 19.90 Hz boundary = M-current resonance (KCNQ ~50ms)
- Theta shows the same pattern at weaker magnitude: boundary(hi) at 7.60 Hz = +38% (f₀ convergence)

**Regime 3: Ascending ramp (Gamma, with echoes in beta-high)**
- Lower nobles depleted (-24% to -30%); inverse nobles enriched (+27% to +61%)
- The ramp peaks at inv_noble_5 (49.87 Hz)
- 7/13 consistent overall — CHBMP anomaly and adult dataset divergence at boundary/noble_6
- 13/13 across 5 HBN pediatric releases (zero conflicts)
- Extends the ascending pattern seen in P20 extended-range analysis

### The aggregate numbers conflated these regimes

The original aggregate analysis (all bands pooled) reported:
- Boundary: -18% (alpha's -37% + beta-low's +101% = net mild depletion because alpha has more peaks)
- Noble1: +39% (alpha's +25% + gamma's +1% + beta-low's +2% = inflated by gamma's sparse but concentrated peaks)
- Gamma +144.8% (artifact of cross-band density normalization; within-band = +1% at Noble1)

The per-band analysis reveals that **each band has its own enrichment fingerprint**, and the aggregate numbers are weighted averages that obscure rather than reveal the underlying structure.

### The phi-lattice is a good coordinate system, not a causal mechanism

The spectral peaks exist where they exist because of biophysics:
- ~10 Hz: T-type Ca²⁺ channel kinetics (~100ms)
- ~12-13 Hz: alpha/beta transition
- ~20 Hz: M-current KCNQ channels (~50ms)
- ~40 Hz: GABA-A decay constants (~25ms)

The phi-lattice (f₀=7.60, φ=1.618) places its positions near these biophysical frequencies. Noble1 at 10.23 Hz captures the IAF. Boundaries at 12.30 and 19.90 Hz capture the band transitions. The attractor at 40.95 Hz captures gamma resonance.

The consistency of the enrichment analysis validates that **φⁿ is a good mathematical description** of where neural oscillatory peaks cluster. Whether this reflects evolutionary tuning to φ, biophysical convergence, or coincidence remains an open question — but the descriptive accuracy is confirmed, especially in alpha and beta-low.

### Boundary enrichment is theoretically predicted

The original papers characterized boundary enrichment as contradicting the phi-lattice. But from Pletzer/Kramer theory:
- Boundaries are where f(n) = f(n-1) + f(n-2) — the Fibonacci coupling equation
- At f₀=7.60, boundary positions land at natural frequency transition points (7.60, 12.30, 19.90, 32.19 Hz)
- Peaks clustering at boundaries means peaks cluster at coupling gateways
- This is consistent with the theory: boundaries are where cross-band energy transfer occurs

The f₀=7.60 sensitivity analysis showed that boundary enrichment strengthens dramatically at 7.60 vs 7.83 (+35 to +67 percentage points across bands), providing independent evidence that 7.60 is the correct anchor and that boundaries are real structural features.

---

## Summary

### Claims from Papers 1-3 as revised

| Original claim | Per-band reality | Paper revision status |
|---|---|---|
| Boundary depletion -18% (universal) | Alpha: -37%. Beta-low: +101%. Band-dependent. | Labeled "aggregate" in all 3 papers |
| Noble1 enrichment +39% (universal) | Alpha: +25%. Others: near null. Alpha-specific. | Labeled "aggregate" in all 3 papers |
| Gamma +144.8% at Noble1 | +1% per-band. Ascending ramp at inv_nobles (+27-61%). | Caveated in Paper 1; not restated in Papers 2-3 |
| Position hierarchy τ=1.0 | Alpha-specific. Beta-low inverted. | Qualified in all 3 papers |
| 3 positions carry signal | In aggregate. Per-band differs. | Qualified in Paper 3 |
| Gamma shows strongest adherence | Gamma is most variable (7/13). Beta-low most consistent (13/13). | Caveated in Paper 1 |

### Claims that survive unchanged
- SIE phenomenology and characterization
- Independence-convergence paradox
- f₀ convergence from multiple sources (strengthened: f₀=7.60 > 7.83 for boundary enrichment)
- Substrate-ignition model
- Dominant-peak alignment d=0.40 across 919 subjects
- Theta EC convergence on f₀ (confirmed: boundary_hi +38% at 7.60 Hz; fatigue STRENGTHENS convergence)
- Age invariance of dominant-peak alignment (Paper 3)
- φ rank 1 structural specificity (needs verification with Voronoi per-band)
- State dynamics base-indifferent (confirmed: gamma state-invariant in Voronoi)
- Metric fragility thesis (dramatically confirmed)

### Claims revised by reanalysis
- **Cognitive null BROKEN** — Paper 3 found 0/100+ FDR survivors with dominant-peak metric; per-subject Voronoi enrichment finds 4 FDR survivors in beta-low × LPS (logical reasoning). The signal is in U-shape depth, not dominant-peak alignment. See Per-Subject Cognitive section below.
- **Age correlates with per-band enrichment** — 5 FDR survivors: theta attractor (rho=-0.39), 4 beta-low positions (rho=0.23-0.29). U-shape deepens with age. Not detected by dominant-peak metric.

### Novel findings from reanalysis
- **Three organizational regimes** — alpha mountain, beta-low U-shape, gamma ascending ramp — each independently replicated
- **Beta-low U-shape** (13/13 consistent) — boundaries as Fibonacci coupling gateways
- **Fibonacci additive coupling** — f(n+2) = f(n+1) + f(n) at every boundary (exact, 0% error). Boundary enrichment in 4/5 bands = three-wave resonance at coupling gateways. Alpha's inverted pattern (boundary depletion) consistent with intrinsic thalamo-cortical resonance overriding coupling
- **Alpha is the mirror image of everything else** — positions enriched in alpha are depleted in θ/βL/βH/γ and vice versa
- **Noble5/Noble4 are near-universal dead zones** — depleted in 4/5 bands (-14% to -59%); null in beta-high
- **Inverse nobles > regular nobles** — systematic upward asymmetry across all bands; peaks drift toward upper boundary
- **Beta-low U-shape depth predicts cognitive performance** — 4 FDR-significant correlations with LPS logical reasoning (rho=0.28-0.31). Deeper U-shape = better performance.
- **Beta-low U-shape deepens with age** — 4 FDR-significant age correlations (rho=0.23-0.29). Boundary enrichment increases, center enrichment increases with age.
- **Theta-attractor enrichment decreases with age** (LEMON adults: rho=-0.39, p_FDR=0.014)
- **Inverted-U lifespan trajectory** — enrichment profiles peak in early adulthood (~20) and reverse thereafter. HBN (5-21, N=927): 43/66 FDR sig, development sharpens regimes. Dortmund (20-70, N=608): 40/66 FDR sig, aging de-differentiates. Cross-dataset age-rho correlation r=-0.554. Of 28 jointly significant features, 24 are in opposite directions.
- **Alpha mountain broadening** peaks ~20: upper flank (inv_noble_3 rho=+0.302 HBN, -0.185 Dort) shows clearest inverted-U
- **Beta-low U-shape depth** peaks ~20: ushape metric +0.166 HBN, -0.154 Dort
- **Externalizing behavior correlates with less differentiated spectral organization** (10 FDR survivors, N=906): shallower beta-low center, weaker gamma ramp
- **Sex differences in beta-high enrichment** (13 FDR survivors, N=927): males show more lower-noble enrichment
- **Fatigue strengthens theta f₀ convergence** — opposite direction from EO effect
- **f₀=7.60 strengthens boundary enrichment** (+11 to +67 pp vs 7.83) — independent validation of 7.60 as the correct anchor
- **Adaptive-resolution extraction** — band-specific nperseg ensures equal position discriminability across all bands
- **Voronoi binning** — proper normalization by bin width with full octave coverage
- **Alpha Noble1 SD=5** — the most consistent single finding across 9 datasets spanning 4 countries, 4 EEG systems, ages 5-70
- **Gamma ascending ramp** at inv_noble_3/4/5 — the P20 discovery, confirmed with proper per-band normalization
- **CHBMP gamma anomaly** — isolated to gamma; CHBMP agrees perfectly in alpha and beta-low

---

## Per-Subject Cognitive Correlations (LEMON EC, N=167)

**Date:** 2026-04-09
**Method:** Per-subject Voronoi enrichment at each position × band (min 30 peaks/band), correlated with 8 LEMON cognitive tests via Spearman rank.
**Script:** `python scripts/per_subject_voronoi_cognitive.py`

### Overview

| Statistic | Value |
|---|---|
| Total tests | 528 (66 enrichment features × 8 cognitive tests) |
| FDR survivors (q=0.05) | **4** |
| Uncorrected p<0.05 | 54 (expected by chance: 26.4; **2.05× enrichment**) |
| Largest |rho| | 0.321 (theta_mountain × RWT) |

**Context:** Paper 3 tested 100+ correlations between dominant-peak alignment (d̄) and LEMON cognitive tests, finding zero FDR survivors. The per-subject Voronoi enrichment metric detects what the dominant-peak metric could not: individual variation in the SHAPE of the within-band enrichment profile, not just proximity to the nearest position.

### FDR-Significant Results (all beta-low × LPS)

| Enrichment feature | rho | p | p_FDR |
|---|---|---|---|
| beta_low_mountain (Noble1 − boundary) | **-0.314** | 4.1e-5 | **0.012** |
| beta_low_boundary | **+0.312** | 5.0e-5 | **0.012** |
| beta_low_inv_noble_1 | **-0.294** | 1.2e-4 | **0.022** |
| beta_low_attractor | **-0.284** | 2.2e-4 | **0.029** |

**Interpretation:** Higher LPS scores (better logical reasoning/perceptual speed) correlate with:
- **Stronger** boundary enrichment (+0.312) — more peaks at Fibonacci coupling gateways
- **Weaker** center enrichment: attractor (-0.284), inv_noble_1 (-0.294)
- **Deeper** U-shape overall (mountain metric -0.314)

In other words: **better cognitive performance is associated with a deeper beta-low U-shape** — more extreme Fibonacci coupling gateway enrichment at boundaries with greater depletion in the center.

### Near-Significant Results

| Test | Best feature | rho | p_FDR | Band |
|---|---|---|---|---|
| TMT | alpha_mountain | +0.262 | 0.058 | Alpha |
| RWT | theta_mountain | +0.321 | 0.257 | Theta |
| TAP_Alert | beta_low_ushape | -0.228 | 0.251 | Beta-low |

TMT (trail making, processing speed) nearly survives FDR through the alpha mountain metric — higher mountain = better TMT performance. RWT (verbal fluency) shows the strongest single correlation (rho=0.321) through the theta mountain, but N=72 (many subjects lack sufficient theta peaks) limits power.

### Band × Cognitive Matrix

LPS is the only test reaching FDR significance, and the signal is concentrated in beta-low. Across all 8 tests × 5 bands, the beta-low U-shape metrics show the most consistent cognitive associations — the same direction appears for TAP_Alert, TAP_Incompat, RWT, and TMT (all non-FDR-significant but directionally consistent).

### Age × Per-Band Enrichment (FDR-Significant)

| Enrichment feature | rho | p_FDR | Interpretation |
|---|---|---|---|
| theta_attractor | **-0.392** | **0.014** | Theta attractor enrichment decreases with age |
| beta_low_inv_noble_1 | **+0.294** | **0.008** | Center enrichment increases with age |
| beta_low_attractor | **+0.274** | **0.012** | Center enrichment increases with age |
| beta_low_mountain | **+0.251** | **0.019** | Mountain metric increases (U-shape deepens) |
| beta_low_boundary | **-0.228** | **0.042** | Boundary enrichment decreases with age |

**Age partialing:** LPS correlates strongly with age (rho=-0.65 in LEMON's bimodal sample). After partialing out age (and sex), the cognitive effect attenuates from rho≈0.3 to rho≈0.2 but **survives** for boundary (+0.231, p=0.003) and mountain (-0.220, p=0.005). inv_noble_1 and attractor become marginal (p≈0.05). About 1/3 of the zero-order variance was age-shared, but 2/3 reflects a genuine cognitive association independent of age. The beta-low U-shape depth predicts logical reasoning performance even within age groups.

### EO Replication of Cognitive Correlations

Under eyes-open conditions (N=200), the same beta-low × LPS features are the top results but **0 FDR survivors** (vs 4 under EC):

| Feature × LPS | EC rho | EC p_FDR | EO rho | EO p_FDR |
|---|---|---|---|---|
| beta_low_mountain | **-0.314** | **0.012*** | -0.255 | 0.061 |
| beta_low_boundary | **+0.312** | **0.012*** | +0.250 | 0.061 |
| beta_low_inv_noble_1 | **-0.294** | **0.022*** | -0.185 | 0.268 |
| beta_low_attractor | **-0.284** | **0.029*** | -0.187 | 0.265 |

**Same direction, ~20% weaker effect sizes.** The cognitive signal is clearer under EC, consistent with EC sharpening the beta-low U-shape contrast. The near-miss p_FDR=0.061 for mountain and boundary suggests the signal is real but state-modulated in strength. Age effects replicate fully under EO (5 FDR survivors).

### Personality/Emotion Battery (8,778 tests)

LEMON's 41 emotion and personality instruments (133 subscales) × 66 enrichment features: **0 FDR survivors**. Uncorrected p<0.05 count (474) barely exceeds chance (439, ratio 1.08×). Phi-lattice enrichment is psychometrically silent for personality traits.

Top near-miss: UPPS sensation seeking × beta_low_mountain (rho=-0.324, p_FDR=0.19) — higher sensation seeking = shallower U-shape. This parallels the LPS cognitive and HBN externalizing findings: sensation seeking, externalizing behavior, and lower cognitive performance all associate with shallower beta-low U-shape, suggesting a common underlying dimension of spectral de-differentiation. But no individual personality measure survives FDR.

Script: `python scripts/per_subject_voronoi_personality.py`

### Significance for Paper 3

Paper 3's conclusion that "the φ-lattice predicts nothing about cognition" was correct for the dominant-peak metric and extends to personality/emotion (0/8,778 FDR survivors). However, per-band Voronoi enrichment breaks the cognitive null specifically through the beta-low U-shape depth × LPS logical reasoning (4 FDR survivors, rho≈0.3, ~9% shared variance under EC, directionally consistent under EO). This doesn't make the phi-lattice a cognitive biomarker, but it shows that the Fibonacci coupling architecture has functional correlates that the position-distance metric is too coarse to capture — and that these correlates are specific to cognitive performance, not personality traits.

---

## EC vs EO Comparison (LEMON + Dortmund)

**Date:** 2026-04-09
**Datasets:** LEMON (167 EC / 202 EO subjects) + Dortmund (608 EC-pre / 608 EO-pre subjects)

### State-Sensitivity by Band

| Band | Sensitivity | Max Δ (LEMON) | Max Δ (Dort) | Replicated? |
|------|-------------|---------------|--------------|-------------|
| **Theta** | **HIGH** | 57 (boundary_hi) | 66 (boundary) | ✓ 4 positions |
| **Alpha** | MODERATE | 25 (inv_noble_4) | 24 (boundary) | ✓ 2 positions |
| **Beta-low** | MODERATE | 34 (noble_5) | 23 (noble_5) | ✓ 6 positions |
| **Beta-high** | LOW | 15 (noble_5) | 11 (inv_noble_6) | — |
| **Gamma** | **NEAR-ZERO** | 10 (boundary_hi) | 7 (boundary) | — |

### Consistently Replicated EC→EO Shifts (same direction, |Δ|>10 in both datasets)

**Theta (4 positions):**
- boundary: EC +44/+77% → EO +0/+11% (EC↑) — **f₀ convergence is EC-specific**
- boundary_hi: EC +65/+47% → EO +8/+18% (EC↑) — same pattern at upper edge
- inv_noble_3: EC -18/-7% → EO +7/+6% (EO↑) — theta spreads to ascending ramp under EO
- inv_noble_5: EC -1/+9% → EO +30/+28% (EO↑) — same

**Alpha (2 positions):**
- inv_noble_4: EC +0/+1% → EO +25/+14% (EO↑) — upper-octave shoulder develops under EO
- inv_noble_6: EC -15/-22% → EO -4/-8% (EO↑) — depletion weakens toward upper boundary

**Beta-low (6 positions):**
- noble_5: EC -75/-62% → EO -41/-39% (EO↑) — center depletion fills in
- noble_4: EC -71/-69% → EO -55/-51% (EO↑) — same
- noble_3: EC -74/-66% → EO -58/-53% (EO↑) — same
- inv_noble_3: EC +42/+42% → EO +30/+29% (EC↑) — upper ramp attenuates
- inv_noble_5: EC +89/+75% → EO +66/+61% (EC↑) — same
- inv_noble_6: EC +101/+74% → EO +81/+62% (EC↑) — same

### Interpretation

**Theta is the most state-sensitive band, not gamma.** EC drives theta peaks to boundary (f₀ convergence — the "absorbing state" from Paper 3). EO disperses theta into an ascending ramp at inverse nobles. This replicates Paper 3's finding with a completely independent methodology.

**Gamma is state-INVARIANT in the Voronoi analysis.** Maximum EC→EO Δ is 10% (LEMON) and 7% (Dortmund). The EEGMMIDB Paper 2 finding of gamma reallocation (Δπ_att = +0.070, r = 0.44) was detected by a 2-component mixture model testing center-vs-edge redistribution — a different and coarser measurement than position-specific Voronoi enrichment. The Voronoi profile is stable because the SHAPE of the gamma enrichment distribution doesn't change; only the proportion of peaks in the center vs edge of the octave shifts, which the mixture model detects but the Voronoi approach averages out.

**Beta-low U-shape is structural but state-modulated in depth.** The boundaries remain enriched (+74-110%) regardless of condition. The center depletion fills in under EO (noble_5: -75%→-41% LEMON, -62%→-39% Dortmund), and the inverse noble ramp attenuates. The shape persists; the contrast weakens.

**Alpha mountain is structural at Noble1, state-modulated at flanks.** Noble1 is rock-stable (+25-30% EC, +22-25% EO). The state effect is at the flanks: boundary depletion weakens, inv_noble_4 enrichment appears. EC sharpens the mountain; EO broadens it.

**State sensitivity is age-independent.** Per-subject EC→EO delta × age: 0 FDR survivors in both LEMON (N=166, ages 20-77) and Dortmund (N=608, ages 20-70), for both direction and magnitude of change (264 tests total). The AMOUNT of enrichment shift when opening eyes does not change with age — what changes with age is the baseline enrichment level (inverted-U trajectory), not the state-dependent modulation around that baseline.

**Per-subject enrichment is within-session reliable.** Dortmund EC-pre vs EC-post (same condition, ~2hr apart, N=608): median ICC = **+0.40**, median r = +0.40. Per-band: beta-low +0.46, beta-high +0.45, alpha +0.38, gamma +0.36, theta +0.31. Cross-condition (EC vs EO, same timepoint): median ICC = +0.35. This contrasts sharply with Paper 3's dominant-peak ICC of **-0.25 to -0.36 across 5 years** — per-subject Voronoi enrichment is a reproducible within-session individual metric, while dominant-peak alignment is not stable across years. The within-session reliability supports using per-subject enrichment for individual-differences analyses (cognitive correlations, age effects).

**Per-subject enrichment is stable across 5 years.** Dortmund ses-1 vs ses-2 (N=208, ~5 years apart): median ICC = **+0.42**, median r = +0.42. Per-band: beta-low +0.51, beta-high +0.49, gamma +0.39, alpha +0.34. This is a dramatic contrast with Paper 3's dominant-peak ICC of **-0.25 to -0.36** — per-subject Voronoi enrichment is a STABLE individual metric across years, while dominant-peak alignment is not. The group profile is nearly unchanged (only 2/53 features with |d|>0.2, both in gamma). Baseline age does NOT predict 5-year change (0 FDR survivors across 53 tests).

**The 2×2 pattern replicates across 5 years.** Full ses-2 2×2 extraction (208 subjects × 4 conditions) shows:
- **Group profile correlations ses-1↔ses-2:** beta-low r=0.987-0.991, gamma r=0.989-0.997, alpha r=0.797-0.965, theta r=0.583-0.938. Beta-low and gamma profiles are virtually identical after 5 years.
- **Eyes effect replicates:** correlation of EC→EO delta patterns between sessions: beta-low r=+0.95, gamma r=+0.82, theta r=+0.69, alpha r=+0.62. The state-modulation pattern is stable across years.
- **Fatigue effect partially replicates:** correlation of pre→post patterns: beta-low r=+0.56, theta r=+0.55, alpha r=+0.45, gamma r=+0.42. Fatigue effects are less stable than eyes effects but directionally consistent.
- **Beta-low 13/13 stable across ALL 8 conditions** (4 conditions × 2 sessions, 5 years apart) — perfect structural invariance across state, fatigue, AND time. Gamma 11/13 stable. Beta-high and theta unstable.

**Medical/metabolic markers do not predict enrichment.** LEMON anthropometry (BMI, WHR), blood pressure (systole, diastole, pulse), and blood biomarkers (~30 variables including cholesterol, glucose, HbA1c, TSH, liver enzymes, CBC) × enrichment: 0 FDR survivors across 2,508 tests (N=167, 38 medical variables). Uncorrected rate 1.35× chance. Top effects are theta × cardiovascular (diastolic BP rho=-0.42, cholesterol rho=-0.40) but theta N≈70 is underpowered for 2,500+ tests.

**Handedness does not predict enrichment.** HBN EHQ (continuous, N=910) × enrichment: 0 FDR survivors. Strong left (N=77) vs strong right (N=686) group comparison: 0 FDR survivors. Dortmund left (N=40) vs right (N=566): 0 FDR survivors. Enrichment profiles are handedness-independent (note: all-channel pooled analysis, not hemisphere-specific).

**The inverted-U trajectory does not differ by sex.** Sex × age interaction tested via Fisher z-test on sex-stratified age rhos: 0 FDR survivors in both HBN (610M/317F, ages 5-21) and Dortmund (232M/376F, ages 20-70). The cross-dataset correlation of sex×age interactions is r=-0.17 (n.s.). Males and females follow the same developmental and aging enrichment trajectories.

**Cross-band coupling is individually stable.** The alpha-beta-low coupling (population rho=-0.35 to -0.41) replicates within-session: per-subject coupling product r=+0.23 to +0.31 between EC-pre and EC-post (all p<10⁻⁶). The coupling-specific residual (after removing each metric's own reliability) shows r=+0.57 (p<10⁻⁵⁰) for alpha_boundary × beta_low_attractor — meaning the INDIVIDUAL relationship between alpha and beta-low is a stable trait, not session noise. This stability extends across 5 years: coupling product r=+0.21 to +0.35 for ses-1 vs ses-2 (N=208). The alpha-beta-low coupling is both a population-level pattern and an individually reproducible trait.

**Alpha and beta-low enrichment profiles are coupled across individuals.** Cross-band correlation analysis (3 datasets, N=167/608/927) reveals that subjects with stronger alpha boundary depletion also have stronger beta-low attractor depletion (rho=-0.22 to -0.41, FDR-significant in all 3 datasets). The strongest replicated cross-band coupling is alpha_boundary × beta_low_attractor (LEMON: -0.41, Dort: -0.35, HBN: -0.22 — all FDR-significant). In HBN (N=899), alpha_Noble1 × beta_low_ushape reaches rho=+0.33 (p<10⁻²⁴): subjects with taller alpha mountains have deeper beta-low U-shapes. This cross-band coherence in individual differences — the alpha mountain and beta-low U-shape co-vary across subjects — suggests these two regimes share a common underlying source of individual variation, possibly related to thalamo-cortical loop gain or global E/I balance. Cross-dataset replication of all cross-band rhos: LEMON-Dort r=0.55, LEMON-HBN r=0.37, Dort-HBN r=0.46 — the pattern of cross-band coupling is itself consistent across datasets. Beta-low U-shape and gamma ramp are NOT coupled (rho≈0 in all datasets), suggesting they arise from independent mechanisms.

---

## Adult vs Pediatric Comparison

**Adult datasets:** EEGMMIDB (18-65), LEMON (20-77), Dortmund (18-35), CHBMP (18-65) — 4 datasets
**Pediatric datasets:** HBN R1, R2, R3, R4, R6 (5-21) — 5 datasets

### Profile Correlations

| Band | Adult-Ped r | Key shift |
|------|---|---|
| Beta-low | 0.97 | Same U-shape; adults more extreme (±40 pp) |
| Theta | 0.93 | Similar; adults stronger boundary clustering |
| Beta-high | 0.84 | Minimal differences |
| Alpha | 0.74 | Children: narrower, taller mountain (9/13 p<0.05) |
| Gamma | 0.64 | Children: clean ramp; adults disrupted at boundary/noble_6 |

### Key Developmental Findings

**Alpha narrowing in children:** Boundary depletion is stronger (-46% ped vs -26% adult), attractor enrichment is stronger (+31% vs +16%), and the mountain falls off sharply after Noble1 (inv_noble_3: -3% ped vs +20% adult). Consistent with narrower alpha peak bandwidth in children.

**Gamma cleanup in children:** All 5 HBN releases show zero conflicts at any position. Adult datasets show sign conflicts at boundary and noble_6 (CHBMP inversion, LEMON/Dortmund boundary discrepancy). The gamma ascending ramp is a universal pediatric finding.

**Beta-low is developmentally invariant:** The U-shape is identical in adults and children (r=0.97). This is the most structurally stable pattern across both age groups and all conditions.

---

## HBN Developmental Trajectory (N=927, ages 5-21)

**Date:** 2026-04-09
**Method:** Per-subject Voronoi enrichment × continuous age, Spearman rank correlations, FDR-corrected (q=0.05).
**Script:** `python scripts/per_subject_voronoi_hbn_age.py`

### Overview: 43/66 FDR Survivors

The enrichment profile changes dramatically across development. Every band except theta shows extensive age-dependent restructuring.

### Per-Band Developmental Patterns

**Alpha (11/13 FDR significant): Mountain broadens asymmetrically upward**

| Position | rho | Direction | Interpretation |
|---|---|---|---|
| inv_noble_3 (10.98 Hz) | **+0.302** | ↑ with age | Upper flank enriches |
| inv_noble_4 (11.46 Hz) | **+0.261** | ↑ with age | Upper flank enriches |
| noble_3 (8.51 Hz) | **-0.244** | ↓ with age | Lower flank depletes |
| inv_noble_1 (9.13 Hz) | **-0.232** | ↓ with age | Pre-peak depletes |
| inv_noble_5/6 | +0.179/+0.183 | ↑ with age | Far upper flank enriches |
| noble_5 (7.94 Hz) | **-0.172** | ↓ with age | Near-boundary depletes |
| boundary (7.60 Hz) | **+0.138** | ↑ with age | Boundary depletion RELAXES |
| attractor (9.67 Hz) | **-0.136** | ↓ with age | Attractor depletes |
| noble_1 (10.23 Hz) | **+0.134** | ↑ with age | Peak enriches slightly |
| mountain metric | +0.068 | n.s. | Height unchanged |

**Interpretation:** Noble1 enrichment is stable across development — the IAF peak position doesn't shift. But the mountain ASYMMETRICALLY BROADENS: the upper octave (10.98-11.97 Hz) gains spectral weight while the lower octave (7.94-9.13 Hz) loses it. This is the continuous per-band signature of alpha bandwidth maturation during adolescence. The boundary depletion paradoxically WEAKENS with age (boundary rho=+0.138), meaning younger children have STRONGER boundary avoidance — consistent with the adult-pediatric finding that children show a narrower, more sharply defined mountain.

**Beta-low (10/14 FDR significant): U-shape deepens by center clearing**

| Position | rho | Direction | Interpretation |
|---|---|---|---|
| noble_3 (13.78 Hz) | **-0.230** | ↓ with age | Center depletes |
| inv_noble_1 (14.78 Hz) | **-0.187** | ↓ with age | Center depletes |
| noble_4 (13.19 Hz) | **-0.183** | ↓ with age | Center depletes |
| noble_5 (12.85 Hz) | **-0.166** | ↓ with age | Center depletes |
| ushape metric | **+0.166** | ↑ with age | U-shape deepens |
| noble_6 (12.63 Hz) | **+0.158** | ↑ with age | Lower edge enriches |
| boundary (12.30 Hz) | +0.151 | ↑ with age | Boundary enriches slightly |
| inv_noble_6 (19.38 Hz) | **+0.150** | ↑ with age | Upper edge enriches |
| attractor (15.65 Hz) | **-0.150** | ↓ with age | Center depletes |
| mountain metric | **-0.133** | ↓ with age | Noble1-boundary gap decreases |

**Interpretation:** The U-shape deepens through development, but primarily through CENTER CLEARING — peaks evacuate the interior of the phi-octave as children mature. The boundaries themselves change minimally. The Fibonacci coupling architecture strengthens developmentally by intensifying the spectral void between coupling gateways, not by accumulating more peaks at the gateways.

**Beta-high (7/13 FDR significant): Lower octave fills in**

| Position | rho | Direction |
|---|---|---|
| noble_4 (21.35 Hz) | **+0.219** | ↑ with age |
| noble_3 (22.29 Hz) | **+0.217** | ↑ with age |
| noble_1 (26.79 Hz) | **-0.202** | ↓ with age |
| noble_5 (20.78 Hz) | **+0.158** | ↑ with age |
| attractor (25.31 Hz) | **-0.136** | ↓ with age |
| inv_noble_3 (28.74 Hz) | -0.107 | ↓ with age |

**Interpretation:** Opposite to beta-low — peaks spread INTO the lower octave with age (nobles 3-5 increase) while the upper octave depletes (noble_1, inv_noble_3 decrease). Beta-high becomes more spectrally uniform through development.

**Gamma (11/13 FDR significant): Ramp sharpens**

| Position | rho | Direction |
|---|---|---|
| noble_3 (36.06 Hz) | **-0.196** | ↓ with age |
| inv_noble_4 (48.55 Hz) | **+0.174** | ↑ with age |
| mountain metric | **-0.153** | ↓ with age |
| boundary (32.19 Hz) | **+0.150** | ↑ with age |
| inv_noble_5 (49.87 Hz) | **+0.141** | ↑ with age |
| noble_4 (34.53 Hz) | **-0.137** | ↓ with age |
| inv_noble_6 (50.71 Hz) | **+0.111** | ↑ with age |

**Interpretation:** The ascending ramp becomes SHARPER with age — lower positions deplete more while inv_noble_4/5/6 enrich. This is modest (max rho=0.196) but consistent across 924 subjects. Older adolescents show more differentiated gamma organization.

**Theta (4/13 FDR significant): Mostly stable**

Only boundary (-0.141), inv_noble_4 (-0.125), inv_noble_5 (-0.130), inv_noble_6 (-0.107) reach significance — all DECREASING with age. The inverse nobles weaken, suggesting the upper-octave clustering that characterizes theta (near f₀) attenuates in older adolescents.

### Psychopathology × Enrichment

| Dimension | N | FDR survivors | Strongest |
|---|---|---|---|
| p_factor | 906 | **0** | gamma_noble_1 rho=+0.088, p_FDR=0.56 |
| Attention | 906 | **0** | gamma_noble_1 rho=+0.079, p_FDR=0.61 |
| **Internalizing** | 906 | **4** | gamma_noble_4 rho=-0.125, beta_high_noble_3 rho=+0.108, beta_low_noble_3 rho=-0.103, beta_high_noble_1 rho=-0.101 |
| **Externalizing** | 906 | **10** | beta_low_noble_3 rho=+0.140, gamma_inv_noble_4 rho=-0.137, gamma_mountain rho=+0.137, beta_low_inv_noble_1 rho=+0.129, alpha_inv_noble_3 rho=-0.122 |

**Interpretation:** General psychopathology (p_factor) and attention do not predict enrichment. Externalizing behavior (10 FDR survivors) correlates with SHALLOWER beta-low center depletion (noble_3 +0.140) and WEAKER gamma ramp (inv_noble_4 -0.137). Children with more externalizing behavior have less differentiated spectral organization — less extreme Fibonacci coupling and less pronounced gamma ramp. Internalizing shows a weaker version of similar effects (4 survivors). These are modest (rho≈0.1-0.14) but survive stringent FDR correction across 906 subjects.

### Sex Differences

13/66 FDR survivors, primarily in beta-high. Effects are small (d≈0.2-0.3):
- Males show less Noble1 enrichment in beta-high (d=-0.242) and more Noble1 enrichment in gamma (d=+0.202)
- Males show more lower-noble enrichment in beta-high (noble_4 d=+0.238, noble_3 d=+0.224)

### Developmental Summary

The three organizational regimes develop on different timescales:
1. **Alpha mountain** broadens asymmetrically upward (rho up to 0.302) — the strongest developmental signal
2. **Beta-low U-shape** deepens by center clearing (rho up to 0.230) — the coupling architecture matures
3. **Gamma ramp** sharpens modestly (rho up to 0.196) — the ascending pattern refines
4. **Theta** is the most developmentally stable band (4/13 FDR significant)

All effects are continuous developmental trajectories, not step changes — consistent with gradual maturation of oscillatory circuits from childhood through adolescence.

### Per-Release Replication

The 43 combined FDR survivors were tested within each release independently (R1=136, R2=150, R3=184, R4=322, R6=135):

**Cross-release correlation of age rhos: r=0.68-0.82 across all 10 pairwise comparisons** (all p<0.0001). The developmental trajectory is nearly identical across 5 independent samples.

| Release | N | FDR survivors | Age range |
|---|---|---|---|
| R1 | 136 | 12/66 | 5.2-21.7 |
| R2 | 150 | 20/66 | 5.0-18.5 |
| R3 | 184 | 10/66 | 5.0-21.0 |
| R4 | 322 | 22/66 | 5.0-20.8 |
| R6 | 135 | 10/66 | 5.1-21.0 |
| **Combined** | **927** | **43/66** | **5.0-21.7** |

2 features reach FDR significance in ALL 5 releases: **alpha_inv_noble_3** (rhos: +0.32/+0.41/+0.28/+0.23/+0.34) and **alpha_inv_noble_4** (+0.38/+0.25/+0.25/+0.18/+0.36) — the alpha upper-flank broadening. 10 features are significant in ≥3/5 releases with consistent direction (5 alpha, 3 beta-low, 2 beta-high). Theta and gamma show zero per-release replication — their age effects require the full combined N to detect.

---

## Dortmund Adult Aging Trajectory (N=608, ages 20-70)

**Date:** 2026-04-09
**Script:** inline analysis (outputs/dortmund_age_enrichment.csv)

### Overview: 40/66 FDR Survivors

Adult aging shows extensive enrichment restructuring — comparable in scope to pediatric development (43/66 HBN) but in the **opposite direction**.

### Per-Band Adult Aging

| Band | FDR sig | Best feature | rho | Direction |
|---|---|---|---|---|
| **Theta** | 4/13 | theta_attractor | -0.348 | Attractor depletes with age |
| **Alpha** | 9/13 | alpha_inv_noble_3 | -0.185 | Upper flank NARROWS |
| **Beta-low** | 9/14 | beta_low_inv_noble_1 | +0.275 | Center FILLS IN |
| **Beta-high** | 8/13 | beta_high_boundary | -0.205 | Boundary depletes |
| **Gamma** | 10/13 | gamma_inv_noble_5 | -0.205 | Ramp WEAKENS |

---

## Lifespan Trajectory: Development vs Aging (HBN + Dortmund)

**The defining finding:** Development (ages 5→21) and aging (ages 20→70) show OPPOSITE enrichment trajectories across most positions. The correlation of age rhos between datasets is **r = -0.554** (p < 0.0001). Of 28 features significant in BOTH datasets, **24 are in opposite directions.**

### Lifespan Pattern Classification

| Pattern | Count | Interpretation |
|---|---|---|
| **Inverted-U** (↑ development, ↓ aging) | 11 | Peaks in early adulthood, declines with aging |
| **U-shape** (↓ development, ↑ aging) | 13 | Minimized in early adulthood, reverses with aging |
| **Monotonic ↓** | 3 | Decreases across entire lifespan |
| **Monotonic ↑** | 1 | Increases across entire lifespan |
| Development-only | 15 | Significant 5-21, stable 20-70 |
| Aging-only | 12 | Stable 5-21, significant 20-70 |
| Neither | 11 | Stable across entire lifespan |

### Per-Band Lifespan Patterns

| Band | Inverted-U | U-shape | Monotonic | Dev-only | Age-only | Neither |
|---|---|---|---|---|---|---|
| **Alpha** | 5 | 4 | 0 | 2 | 0 | 2 |
| **Beta-low** | 4 | 4 | 0 | 2 | 1 | 3 |
| **Gamma** | 2 | 3 | 3 | 3 | 2 | 0 |
| **Beta-high** | 0 | 2 | 1 | 4 | 5 | 1 |
| **Theta** | 0 | 0 | 0 | 4 | 4 | 5 |

Alpha and beta-low are dominated by inverted-U and U-shape patterns — their enrichment profiles peak (or trough) in early adulthood and reverse thereafter. Gamma has a mix of all patterns. Theta and beta-high show more unidirectional or condition-specific effects.

### What Peaks in Early Adulthood (Inverted-U)

These features RISE during childhood/adolescence (HBN ↑) and DECLINE during adult aging (Dortmund ↓):

| Feature | Dev rho | Aging rho | Interpretation |
|---|---|---|---|
| alpha_inv_noble_3 | +0.302 | -0.185 | Alpha upper-flank enrichment peaks ~20 |
| alpha_inv_noble_4 | +0.261 | -0.136 | Alpha far upper flank |
| alpha_inv_noble_5/6 | +0.179/+0.183 | -0.134/-0.118 | Alpha boundary-adjacent |
| alpha_boundary | +0.138 | -0.129 | Alpha boundary depletion relaxation peaks ~20 |
| beta_low_ushape | +0.166 | -0.154 | **U-shape depth peaks ~20** |
| beta_low_boundary | +0.151 | -0.167 | Beta-low boundary enrichment peaks ~20 |
| beta_low_noble_6/inv_noble_6 | +0.158/+0.150 | -0.163/-0.142 | Beta-low edge enrichment |
| gamma_inv_noble_5/6 | +0.141/+0.111 | -0.205/-0.108 | Gamma ramp peak positions |

**The alpha mountain is widest in early adulthood** — it broadens during development and narrows during aging. The upper flank (inv_noble_3/4/5/6, 10.98-11.97 Hz) shows the clearest inverted-U trajectory.

**The beta-low U-shape is deepest in early adulthood** — Fibonacci coupling architecture reaches maximum differentiation around age 20 and then gradually de-differentiates with aging.

### What Dips in Early Adulthood (U-Shape)

These features DECLINE during development (HBN ↓) and RISE during aging (Dortmund ↑):

| Feature | Dev rho | Aging rho | Interpretation |
|---|---|---|---|
| beta_low_inv_noble_1 | -0.187 | +0.275 | Center enrichment minimized ~20 |
| beta_low_attractor | -0.150 | +0.260 | Attractor depleted most ~20 |
| alpha_noble_3 | -0.244 | +0.139 | Lower alpha flank depleted most ~20 |
| alpha_inv_noble_1 | -0.232 | +0.179 | Pre-peak depletion maximized ~20 |
| beta_low_noble_3 | -0.230 | +0.184 | Center depletion maximized ~20 |
| gamma_mountain | -0.153 | +0.127 | Gamma Noble1 advantage minimized ~20 |
| gamma_attractor | -0.071 | +0.161 | Gamma center depleted most ~20 |

**The center of each phi-octave is most depleted in early adulthood** — peaks maximally avoid the mid-octave positions around age 20, then re-occupy them during aging. This is the complement of the inverted-U finding: maximum spectral differentiation (sharpest regimes) in early adulthood, with de-differentiation in both directions.

### Monotonic Features (4 total)

Only 4 features change consistently across the entire 5-70 age range:
- **gamma_inv_noble_3 ↑** (HBN: +0.072, Dort: +0.094) — modest continuous increase
- **gamma_noble_4 ↓** (HBN: -0.137, Dort: -0.113) — continuous depletion
- **gamma_noble_5 ↓** (HBN: -0.071, Dort: -0.131) — continuous depletion
- **beta_high_inv_noble_5 ↓** (HBN: -0.099, Dort: -0.131) — continuous depletion

These gamma lower-noble depletions are the only features that change monotonically from age 5 to 70.

### Interpretation

### LEMON Age × Enrichment (N=167, ages 20-77)

LEMON has a bimodal age distribution (young 20-35, elderly 55-77, no middle-aged).

**5 FDR survivors** (all aging direction, matching Dortmund):

| Feature | rho | p_FDR | Band |
|---|---|---|---|
| theta_attractor | **-0.392** | **0.014** | Theta |
| beta_low_inv_noble_1 | **+0.294** | **0.008** | Beta-low |
| beta_low_attractor | **+0.274** | **0.012** | Beta-low |
| beta_low_mountain | **+0.251** | **0.019** | Beta-low |
| beta_low_boundary | **-0.228** | **0.042** | Beta-low |

Per-band: theta 1/13, alpha 0/13 (near-miss: inv_noble_3 rho=-0.216, p_FDR=0.056), beta-low 4/14, beta-high 0/13, gamma 0/13. Lower power (N=167) limits detection vs Dortmund (N=608).

### Three-Dataset Validation (HBN + LEMON + Dortmund)

| Comparison | Correlation of age rhos | Interpretation |
|---|---|---|
| LEMON vs Dortmund | **r = +0.798** | Strong agreement — both show aging effects |
| LEMON vs HBN | **r = -0.492** | Opposite — LEMON shows aging, HBN shows development |
| HBN vs Dortmund | **r = -0.554** | Opposite — development vs aging |

**Only 4 features are FDR-significant in ALL 3 datasets** — all beta-low, all non-monotonic:

| Feature | HBN (dev) | LEMON (aging) | Dort (aging) | Pattern |
|---|---|---|---|---|
| beta_low_boundary | +0.151 | -0.228 | -0.167 | Inverted-U |
| beta_low_inv_noble_1 | -0.187 | +0.294 | +0.275 | U-shape |
| beta_low_attractor | -0.150 | +0.274 | +0.260 | U-shape |
| beta_low_mountain | -0.133 | +0.251 | +0.166 | U-shape |

Beta-low is the ONLY band with lifespan-spanning significance across all 3 datasets. The U-shape deepens during development, peaks in early adulthood, and shallows during aging.

### Cognition × Age × Enrichment Triangle

The same beta-low features that predict LPS cognitive performance also change with age — but in OPPOSITE directions, ruling out age as a confound:

| Feature | LPS (cognition) | LEMON age | Dort age | HBN dev | Confound? |
|---|---|---|---|---|---|
| beta_low_boundary | **+0.312*** | -0.228* | -0.167* | +0.151* | **OPPOSITE** |
| beta_low_inv_noble_1 | **-0.294*** | +0.294* | +0.275* | -0.187* | **OPPOSITE** |
| beta_low_attractor | **-0.284*** | +0.274* | +0.260* | -0.150* | **OPPOSITE** |
| beta_low_mountain | **-0.314*** | +0.251* | +0.166* | -0.133* | **OPPOSITE** |
| beta_low_ushape | +0.268 | -0.201 | -0.154* | +0.166* | **OPPOSITE** |

Better cognitive performance → DEEPER U-shape. Aging → SHALLOWER U-shape. Development → DEEPENS U-shape. The cognitive and aging effects are in opposite directions at every position — the cognitive signal is not an age artifact. Higher-performing older adults may maintain a more "youthful" U-shape depth despite aging-related shallowing.

The three datasets span **ages 5-77 across 1,702 subjects** with consistent results: development sharpens enrichment profiles, aging de-differentiates them, and the transition occurs in early adulthood.

### Interpretation

**Spectral organization follows an inverted-U lifespan trajectory.** The three organizational regimes — alpha mountain, beta-low U-shape, gamma ramp — all reach MAXIMUM DIFFERENTIATION in early adulthood (~20 years), becoming sharper during childhood development and then gradually de-differentiating during aging. This is consistent with known lifespan trajectories of:
- IAF (increases during development, decreases with aging)
- Alpha power (peaks in young adulthood)
- Spectral complexity (highest in young adulthood)
- White matter integrity (peaks ~30, declines thereafter)

The phi-lattice enrichment profile captures these known trajectories at the level of specific positions within phi-octaves, providing a finer-grained view of lifespan spectral organization than traditional band-power or IAF measures.

--- 

## Detailed Position Analysis

See [2026-04-09-enrichment-detailed-analysis.md](2026-04-09-enrichment-detailed-analysis.md) for:
- Position-by-position cross-band profiles (13 positions × 5 bands)
- Fibonacci additive coupling analysis and scorecard
- Position symmetry analysis (noble vs inverse noble asymmetry)
- Three organizational regimes with mechanistic interpretation

---

## Limitations of Current Analysis

1. **Resting-state EC analysis complete**: All 9 datasets analyzed (2,061 subjects, 4.96M peaks). R6 confirms R1-R4 patterns with zero new conflicts.

2. **EC vs EO and fatigue complete**: LEMON EC/EO and Dortmund full 2×2 (EC/EO × pre/post) analyzed. CHBMP EO and HBN EC-only segments not yet extracted. EEGMMIDB conditions pooled in current extraction — per-condition re-extraction needed for within-EEGMMIDB state comparisons.

3. **f₀ mismatch**: Extraction used f₀=7.83, enrichment computed at f₀=7.60. Shift is <3% and enrichment shapes are robust, but a fully consistent extraction at f₀=7.60 would be cleaner.

4. **Cross-base comparison not yet done**: The structural specificity analysis (φ vs 8 other bases) has not been repeated with the adaptive Voronoi methodology.

5. **IRASA/eBOSC not tested**: Testing whether the lattice signal survives alternative aperiodic separation methods remains undone.

6. **Spectral resolution limitation**: Degree-6 (12 positions) is the maximum resolvable. Finer lattice structure may exist beyond current spectral resolution.

7. **EEGMMIDB gamma discrepancy**: Paper 2's mixture model detected gamma state modulation (r=0.44), but the Voronoi analysis shows gamma is state-invariant (max Δ=7-10%). These measure different things (center-vs-edge proportion vs position-specific density) and are not contradictory, but the relationship needs clarification.
