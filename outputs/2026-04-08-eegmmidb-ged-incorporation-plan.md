# Plan: Incorporating GED Analysis into EEGMMIDB Paper

**Date:** 2026-04-08
**Problem:** The submitted Frontiers revision (frontiers_revision.tex, line 1523) cites `lacy2026eegmmidb` as the source for GED results (1,584,561 peaks, 3,261 sessions, tau=1.0). But the EEGMMIDB paper doesn't contain GED analysis — it attributes those results to `lacy2026golden` and lists GED as a future direction. The GED analysis lives in unified_paper.tex (Study 3), which has no publication home.

**Solution:** Incorporate the GED analysis from unified_paper.tex into eegmmidb_paper.tex as a new section, so the Frontiers citation resolves correctly.

## What Needs to Move

From unified_paper.tex Study 3 (lines 1272-1411):

1. **GED methodology** (lines 1278-1310): eigenvalue sweep protocol, regularization, peak detection, lattice coordinate analysis
2. **Dataset description** (lines 1312-1313): PhySF (46 sessions), MPENG (900 sessions), EEGEmotions-27 (2,315 sessions) — brief descriptions needed since these aren't in the EEGMMIDB paper
3. **Results** (lines 1318-1410):
   - 1,584,561 GED peaks
   - 6-position enrichment table (Noble1 +27.5%, Boundary -9.2%, tau=1.0)
   - Extended 12-position table
   - Session consistency (77.8%, Cohen's d=0.24)
   - FOOOF vs GED comparison table
4. **Figure** (line 1341): GED peak distribution (`aggregate_modes_logphi_f0_760.png`)

## Structural Plan for EEGMMIDB Paper

### New Section: "GED Methodological Triangulation" (after current Section 5, before Discussion)

The EEGMMIDB paper currently has this structure:
- Section 4: Methods (EEGMMIDB-specific)
- Section 5: Results (EEGMMIDB-specific)
- Section 6: Discussion

Proposed new structure:
- Section 4: Methods (EEGMMIDB-specific)
- Section 5: Results (EEGMMIDB-specific, as before)
- **Section 6: GED Methodological Triangulation** (NEW)
  - 6.1 Rationale
  - 6.2 Datasets and GED Protocol
  - 6.3 Results
  - 6.4 FOOOF-GED Comparison
- Section 7: Discussion (renumbered from 6)

Alternatively, to minimize restructuring: add GED as a new subsection within the existing Results section (Section 5), after the cross-resolution replication.

**Recommended: Add as Section 5.12 "Methodological Triangulation: GED Spatial Coherence"** — this keeps it in Results and parallels the cross-resolution replication (Section 5.12 currently) as another robustness check.

### Methods Addition (Section 4)

Add a new subsection **4.14 GED Spatial Coherence Analysis** with:

```
To test whether the phi-lattice structure reflects genuine multi-channel network 
organization rather than single-channel spectral artifacts, we applied Generalized 
Eigendecomposition (GED) spatial filtering to three independent datasets with 
sufficient channel counts: PhySF (N=26 subjects, 46 sessions; meditation and 
cognitive flow; Emotiv EPOC X, 14 channels), MPENG (N=36 subjects, 900 sessions; 
gaming engagement; Emotiv EPOC X), and EEGEmotions-27 (N=25 subjects, 2,315 
sessions; emotion induction; Emotiv EPOC X). These non-motor datasets complement 
the EEGMMIDB motor paradigm.

GED solves S w = lambda R w, where S is the covariance matrix of narrowband-filtered 
data (f +/- 0.5 Hz, 4th-order Butterworth) and R is the broadband reference 
covariance (1-50 Hz), regularized as R_reg = (1-alpha)R + alpha*tr(R)/N * I 
(alpha=0.01). A continuous frequency sweep from 4.5-45.0 Hz in 0.1 Hz steps 
yielded eigenvalue profiles lambda(f) from which peaks were detected via 
prominence-based thresholding (minimum prominence = 0.1 x session maximum).

Lattice coordinates and enrichment were computed identically to the FOOOF analysis 
(Section 4.3), enabling direct comparison between single-channel spectral and 
multi-channel spatial coherence criteria.
```

### Results Addition (Section 5)

Add new subsection **5.12 GED Spatial Coherence Triangulation**:

Content from unified_paper.tex:
- Peak count: 1,584,561 across 3,261 sessions
- Position-type enrichment table (6 positions): Noble1 +27.5%, Attractor +14.8%, Noble2 +4.6%, Noble3 +2.6%, Boundary -9.2%, Inv3 -26.2%
- Kendall's tau = 1.0 (perfect ordering preserved)
- Session consistency: 77.8% (Cohen's d=0.24)
- FOOOF vs GED comparison table (3 columns: FOOOF Primary, FOOOF Emotions, GED Combined)
- GED peak distribution figure
- Note on attenuated effect magnitudes (dataset composition, spatial averaging, stricter criterion)

**Critical: label all enrichment values as aggregate**, consistent with the paper's existing caveats.

### Discussion Updates

**Current Section 6.1 (Lattice Replication, line 577):** Currently says the lattice "replicates in an entirely different paradigm." Add GED convergence:

```
The GED analysis extends this replication to a fundamentally different analytical 
criterion: multi-channel spatial coherence rather than single-channel spectral 
prominence. The identical position-type hierarchy (Kendall's tau = 1.0) across 
1,584,561 GED peaks confirms that phi-lattice organization reflects network-level 
architecture, not artifacts of FOOOF spectral parameterization.
```

**Current Section 6.5 (Static Structure, line 605):** The structural specificity results currently use only FOOOF peaks. Note that GED provides independent confirmation:

```
The GED analysis provides independent methodological confirmation: the same 
qualitative ordering emerges from spatial coherence analysis using a completely 
different peak detection criterion, though the aggregate enrichment magnitudes 
are attenuated (Noble1: +27.5% GED vs +39.0% FOOOF primary).
```

### Fixes for Existing Errors

**1. Line 136 — Fix GED attribution:**
Current: "That study confirmed these predictions across 244,955 single-channel peaks and 1,584,561 multi-channel GED peaks"
Change to: "That study confirmed these predictions across 244,955 single-channel peaks. Multi-channel GED analysis (Section~\ref{sec:ged}) provides independent confirmation across 1,584,561 spatially coherent peaks."

This removes the attribution to the companion study and points to the new GED section within this paper.

**2. Line 335 — Fix beta-low +27.5% error:**
Current: "Companion studies using non-motor datasets show noble1 dominance in beta-low at +27.5% enrichment"
Change to: "GED analysis of non-motor datasets shows aggregate noble1 enrichment of +27.5% across all bands (Section~\ref{sec:ged}), while companion studies using non-motor paradigms report noble1 as the dominant position in beta-low"

This correctly attributes +27.5% as aggregate cross-band GED enrichment, not beta-low specific.

**3. Line 658 — Remove GED from Limitations:**
Current: "Multi-channel methods (GED; Cohen, 2017) might yield sharper lattice profiles by emphasizing spatially coherent oscillations."
Change to: "The GED analysis (Section~\ref{sec:ged}) confirms the lattice hierarchy using multi-channel spatial coherence, but was conducted on different datasets with consumer-grade EEG (14 channels). High-density GED on the EEGMMIDB dataset (64 channels) might yield sharper position-specific allocation estimates."

**4. Line 680 — Update Future Direction #2:**
Current: "Second, applying GED-based peak detection to EEGMMIDB would test whether spatially coherent peaks show reduced multimodality"
Change to: "Second, applying GED-based peak detection directly to EEGMMIDB's 64-channel recordings would test whether higher spatial dimensionality yields sharper position-specific allocation than the 14-channel GED analysis reported in Section~\ref{sec:ged}"

**5. Line 628 — Already correct:**
"the companion study (optimized on GED peaks from PhySF/MPENG/Emotions)" — this becomes self-referential once GED is in this paper. Change to: "the GED analysis (optimized on peaks from PhySF/MPENG/Emotions; Section~\ref{sec:ged})"

### Bibliography Update

The EEGMMIDB paper's bibliography (eegmmidb_refs.bib) needs:
- Cohen 2017 GED paper (already referenced as `cohen2017ged`)
- PhySF, MPENG dataset references if they have DOIs
- EEGEmotions-27 reference (Phuong et al., 2025) — check if already present

The `lacy2026golden` reference stays but is no longer cited for GED results — only for the original lattice discovery (244,955 single-channel peaks).

### What NOT to Change

1. **Do not add GED to the structural specificity tests (Tests A-D)** — those test base comparison using FOOOF peaks only. GED is a separate triangulation, not a replacement.
2. **Do not re-run GED on EEGMMIDB** — this is noted as a future direction, not done.
3. **Do not change any FOOOF-based results or statistics.**
4. **Do not add per-band GED analysis** — the GED results are aggregate, consistent with the paper's existing methodology.
5. **Do not change the Frontiers revision** — it already correctly cites `lacy2026eegmmidb` for GED results; once GED is added to the EEGMMIDB paper, the citation resolves.

### Impact on Word Count

Current EEGMMIDB paper: ~11,600 words
Estimated addition: ~800-1,000 words (methods ~300, results ~400, discussion additions ~200)
Post-incorporation: ~12,400-12,600 words

This may exceed Frontiers' 12,000-word limit if the EEGMMIDB paper is submitted to Frontiers. If submitted elsewhere, check target journal limits. Consider whether any existing content can be trimmed (e.g., consolidating some of the extensive Discussion prose).

### Summary of Changes

| Section | Change | Words |
|---------|--------|-------|
| Methods 4.14 | New: GED protocol + dataset descriptions | ~300 |
| Results 5.12 | New: GED results, tables, figure | ~400 |
| Discussion 6.1 | Add GED convergence paragraph | ~60 |
| Discussion 6.5 | Add GED confirmation note | ~50 |
| Intro line 136 | Fix GED attribution | ~0 (rewrite) |
| Results line 335 | Fix +27.5% beta-low error | ~0 (rewrite) |
| Limitations line 658 | Update GED limitation | ~0 (rewrite) |
| Future Direction line 680 | Update GED future direction | ~0 (rewrite) |
| Discussion line 628 | Fix self-reference | ~0 (rewrite) |
| Bibliography | Add dataset references if needed | — |

**Total: ~810 new words, 5 rewrites of existing text.**

### Verification Checklist

After incorporation:
- [ ] Line 1523 of frontiers_revision.tex (`lacy2026eegmmidb` for GED) now resolves to actual content
- [ ] The +27.5% figure is correctly attributed as aggregate cross-band GED (not beta-low specific)
- [ ] GED no longer listed as a limitation/future direction that hasn't been done
- [ ] All GED enrichment values labeled "aggregate" consistent with paper's existing caveats
- [ ] The "companion study" reference at line 136 no longer claims GED results
- [ ] PhySF, MPENG, EEGEmotions-27 datasets adequately described
- [ ] cohen2017ged citation present in bibliography
- [ ] Figure file (aggregate_modes_logphi_f0_760.png) accessible from paper's graphics path
