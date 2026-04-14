# Spectral Differentiation

Analysis code and manuscript for:

**Spectral Differentiation: Within-Band Oscillatory Organization as a Cognitive and Developmental EEG Biomarker**

Michael Lacy (2026)

## Overview

This repository contains the complete analysis pipeline for a study of EEG frequency band organization using 4,572,489 FOOOF-detected oscillatory peaks from 9 datasets spanning 2,097 subjects aged 5--70. The paper introduces *spectral differentiation* -- the degree to which oscillatory peaks concentrate at specific within-band positions -- as a novel class of EEG biomarker.

Key findings:

- Log-frequency scaling of EEG bands is formally demonstrated (BIC > 46 vs linear)
- The golden ratio is the best fixed-ratio model for inter-band spacing (bootstrap 95% CI: [1.609, 1.623])
- Spectral differentiation predicts reasoning ability, tracks an inverted-U lifespan trajectory, dissociates externalizing from internalizing psychopathology, and is stable over 5 years (ICC = 0.75)
- Results replicate under IRASA, a non-parametric alternative to FOOOF

## Repository Structure

```
papers/              LaTeX manuscript and figures
scripts/             Analysis scripts (see scripts/INDEX.md)
lib/                 Reusable Python modules (see lib/INDEX.md)
outputs/             Generated reports and analysis outputs
data/                Dataset descriptions (raw data on external storage)
wiki/                Research knowledge base (Obsidian vault)
raw/                 Unprocessed source materials
```

## Datasets

| Dataset | N | Ages | Channels | Source |
|---------|---|------|----------|--------|
| EEGMMIDB | 109 | Adult | 64 | [PhysioNet](https://physionet.org/content/eegmmidb/) |
| LEMON | 203 | 20--77 | 59 | [MPI Leipzig](http://fcon_1000.projects.nitrc.org/indi/retro/MPI_LEMON.html) |
| Dortmund | 608 | 18--70 | 64 | [OpenNeuro ds005385](https://doi.org/10.18112/openneuro.ds005385.v1.0.3) |
| CHBMP | 250 | Adult | 62--120 | [CAN-BIND / Zenodo](https://doi.org/10.5281/zenodo.3945385) |
| HBN R1--R6 | 927 | 5--21 | 128 | [Child Mind Institute](http://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/) |

## Reproducing the Results

### Requirements

- Python 3.10+
- NumPy 1.26, SciPy 1.11, pandas 2.1, matplotlib 3.8
- specparam (FOOOF) 1.0

### Step 1: Extract peaks

Download the source datasets, then run extraction per dataset:

```bash
python scripts/run_f0_760_extraction.py --dataset lemon
python scripts/run_f0_760_extraction.py --dataset lemon --method irasa
```

This produces per-subject peak CSVs in `exports_adaptive_v3/` (FOOOF) or `exports_irasa_v4/` (IRASA).

### Step 2: Run analyses

```bash
python scripts/run_all_f0_760_analyses.py --all
```

This runs the full 22-step analysis suite: pooled enrichment, cognitive correlations, developmental trajectories, psychopathology, personality nulls, test-retest reliability, cross-band coupling, and more. Results go to `outputs/f0_760_reanalysis/`.

### Step 3: Additional analyses

```bash
python scripts/log_scaling_test.py              # Log vs linear scaling (Part I)
python scripts/bootstrap_trough_locations.py     # Subject-level bootstrap (Part I)
python scripts/boundary_sweep.py                 # Coordinate system optimization (Part I)
python scripts/within_band_coordinates.py        # Within-band structure tests (Part I)
python scripts/irasa_subsample_test.py           # IRASA subsample power test (Part IV)
```

### Step 4: Generate figures

```bash
python scripts/generate_spectral_diff_figures.py  # Main paper figures
python scripts/generate_trough_figure.py          # Trough analysis figure
```

## Data Availability

Extracted peak CSVs, per-subject enrichment profiles, and all statistical outputs are archived on [Dryad](https://doi.org/10.5061/dryad.1vhhmgr8t).

## Key Scripts

| Script | Description |
|--------|-------------|
| `run_f0_760_extraction.py` | v3 peak extraction pipeline (FOOOF + IRASA) |
| `run_all_f0_760_analyses.py` | Full 22-step analysis suite |
| `boundary_sweep.py` | 36x36 coordinate system optimization |
| `bootstrap_trough_locations.py` | Subject-level bootstrap of trough positions |
| `log_scaling_test.py` | Formal log vs linear scaling comparison |
| `within_band_coordinates.py` | 5 within-band structure tests |
| `assemble_dryad.py` | Dryad data package assembly |

See [scripts/INDEX.md](scripts/INDEX.md) for the complete script inventory.

## Citation

If you use this code or data, please cite:

```
Lacy, M. (2026). Spectral Differentiation: Within-Band Oscillatory Organization
as a Cognitive and Developmental EEG Biomarker. [Manuscript submitted for publication].
```

## License

MIT
