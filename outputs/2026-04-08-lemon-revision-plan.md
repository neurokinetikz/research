# LEMON Paper Revision Plan

**Date:** 2026-04-08
**Context:** Per-band Voronoi enrichment analysis across 9 datasets reveals that the LEMON paper's central thesis -- aggregate enrichment metrics are fragile and extraction-dependent -- is dramatically confirmed. The LEMON paper is the most vindicated of the three papers by the reanalysis.
**Strategy:** Minimal targeted changes to strengthen existing claims and qualify the few claims that are band-specific. No references to Voronoi reanalysis, companion analyses, or unpublished per-band results. Self-contained caveats using the paper's own data and framework.

## Narrative Arc Across the Three Papers

1. **Frontiers (Paper 1):** Discovers the phi-lattice. Reports aggregate enrichment (boundary -18%, Noble1 +39%). Revision adds "aggregate" caveats and Limitation 5 noting per-band heterogeneity.

2. **EEGMMIDB (Paper 2):** Tests structural specificity. Phi is globally optimal (SS=45.6) but anchor-dependent. Dynamic effects are NOT phi-specific. Adds "aggregate" caveats, incorporates GED analysis, adds scorecard table.

3. **LEMON (Paper 3):** Diagnoses WHY aggregate metrics are fragile. Shows extraction-dependency, IAF projection, frequency-range sensitivity. Identifies dominant-peak alignment (d=0.40) as the robust metric. The "three positions carry signal" claim needs qualification. The paper should frame itself as explaining why Papers 1-2's aggregate results were fragile while validating the underlying architecture through a more robust metric.

## Discrepancy Analysis

### Claims That Are STRENGTHENED by Voronoi Reanalysis

#### S1: "Aggregate enrichment metrics are dominated by extraction parameters"
The paper's Part I diagnostic (SS reversal under frequency range changes, overlap-trim sensitivity, max_n_peaks sensitivity) is dramatically confirmed by per-band analysis. Different bands have fundamentally different enrichment patterns that cancel in the aggregate -- exactly as the LEMON paper warned.

**No change needed.** The paper already makes this case thoroughly.

#### S2: "Per-octave modeling is necessary"
The paper argues (Appendix D) that per-phi-octave fitting creates edge artifacts. The Voronoi reanalysis extends this: not just fitting artifacts but genuine per-band enrichment heterogeneity.

**No change needed.** Already adequately argued.

#### S3: Theta EC convergence on f0
The paper shows theta peaks converge toward f0=7.83 Hz under eyes-closed conditions. Voronoi analysis confirms theta boundary(hi) at 7.60 Hz = +65% (LEMON), consistent with f0 convergence.

**No change needed.**

### Claims That Need Qualification

#### D1: "Three positions carry signal: {boundary, attractor, noble1}" (Section 3.6, lines 492-516)

**Paper claim:** "The effective signal-carrying lattice is {0, 0.5, 0.618}—boundary, attractor, and the upper noble—rather than the full degree-2 set."

**Per-band reality:**
- Alpha: boundary depleted (-22%), attractor enriched (+17%), Noble1 enriched (+26%) -- **consistent with 3-position claim**
- Beta-low: boundary enriched (+110%), noble1 near zero (+6%), inv_noble_3/4/5/6 massively enriched (+42% to +101%) -- **contradicts: different positions carry signal**
- Gamma: Noble1 near zero (-3%), inv_noble_3 (+33%) -- **different position structure**
- Theta: boundary enriched (+44%/+65%), attractor near zero (+7%) -- **boundary-only**

**Issue:** The "3 positions carry signal" claim is alpha-specific. In beta-low, the signal is at boundaries and inverse nobles. In gamma, it's at inverse nobles. The claim should be qualified as reflecting the dominant-peak analysis (which is alpha-dominated) rather than a universal within-band property.

**However:** The paper's dominant-peak alignment uses conventional bands (delta, theta, alpha, gamma) rather than phi-octave bands, and tests distance to nearest position rather than enrichment. The 3-position claim comes from the degree-3 enrichment analysis (Table 5), which pools all peaks. This is an aggregate claim that should be labeled as such.

#### D2: Boundary enrichment direction (throughout)

**Paper claim:** The paper discusses boundary both as depleted (standard enrichment: -6.4%) and enriched (per-position Table 5: +45% in LEMON, +42% in Dortmund). This apparent contradiction is already explained (line 509): "boundary enrichment...reflects theta convergence on f0."

**Per-band reality:** Boundary is depleted in alpha (-22%) and enriched in theta (+44%/+65%) and beta-low (+110%). The paper's observation that boundary enrichment is theta-driven is correct.

**No change needed.** The paper already identifies this pattern.

#### D3: Noble2 (u=0.382) is "consistently null or depleted" (lines 492-493)

**Paper claim:** Noble2 shows -6% to -27% across datasets.

**Per-band reality:** Noble2 = inv_noble_1 in Voronoi terminology. At u=0.382:
- Alpha: -2% (LEMON) -- near zero, consistent with paper's claim
- Beta-low: -56% -- depleted, consistent
- Beta-high: -16% -- depleted, consistent

**No change needed.** The direction is correct across all bands.

#### D4: Per-position enrichment table values (Table 5, lines 472-492)

These are aggregate cross-band values computed from all peaks pooled. They should be labeled as aggregate, consistent with the treatment in Frontiers and EEGMMIDB papers.

#### D5: Position phenotyping (Section 3.14, line 811)

**Paper claim:** 58% boundary, 25% noble1 subjects in theta. This is about dominant-peak positioning, not enrichment, so it's unaffected by per-band normalization.

**No change needed.**

### Claims That Survive Unchanged

- Dominant-peak alignment d=0.40 (different metric from enrichment)
- Age invariance (r=-0.004 Dortmund)
- Cognitive null (0/100+ tests survive FDR)
- Five-year longitudinal ICC results (all ICC < 0)
- Phi rank 1 under OT, rank 7 under standard
- f0 optimization at 7.88 Hz
- Cross-dataset theta convergence to 6.56 Hz
- EC power enhancement at stable positions
- All Part I diagnostic findings (frequency range sensitivity, overlap-trim, max_n_peaks)
- Per-phi-octave edge artifacts (Appendix D)

## Changes by Section

### Abstract (lines 107-116)

**Current (line 110):** "Only three positions carry consistent signal across datasets (noble1, attractor, boundary)"

**Change to:** "Only three positions carry consistent signal across datasets in aggregate analysis (noble1, attractor, boundary); per-band enrichment patterns may differ"

One clause addition.

### Significance Statement (lines 119-121)

No changes needed -- focuses on extraction sensitivity and diagnostic methodology.

### Section 3.6: Per-Position Enrichment (lines 492-516)

**Current (line 492):** "Three positions carry consistent signal across all three datasets: noble1 (u = 0.618, the strongest), attractor (u = 0.500), and boundary (u = 0.000)"

**Add after this sentence:** "This ordering reflects the aggregate cross-band analysis; individual frequency bands may exhibit distinct position preferences, as the band-specific results in Table 7 suggest (e.g., theta boundary enrichment driven by f0 convergence, Table 9)."

**Current (line 509):** "The genuine alignment signal is concentrated at three positions: noble1, attractor, and boundary."

**Change to:** "In the aggregate cross-band analysis, the alignment signal is concentrated at three positions: noble1, attractor, and boundary."

### Table 5 (lines 472-492): Per-Position Enrichment

**Add table note:** "Values are aggregate figures pooling peaks from all frequency bands. Band-specific enrichment patterns may differ from these aggregate values (see Table 7 for per-band dominant-peak positioning)."

### Section 4.2: What Survives (Discussion, lines 831-854)

**Current (line 838):** "Only three of six degree-3 positions carry signal"

**Add qualifier:** "Only three of six degree-3 positions carry signal in the aggregate cross-band analysis"

### Section 4.4: The Cognitive Null (Discussion, lines 856-877)

No changes needed -- cognitive null is unaffected by per-band analysis.

### Section 4.5: Implications for the Phi-Lattice Framework (lines 879-891)

**Current (line 882):** "The genuine φ-lattice signal is subtle, band-specific, and visible only in dominant-peak metrics"

**No change needed.** This already acknowledges band-specificity! The paper is ahead of the other two in this regard.

### Section 4.8: Conclusion (lines 940-948)

**Current (line 942):** "The φ-lattice hypothesis, when applied to aggregate spectral peak distributions, does not survive progressive diagnostic testing"

**No change needed.** Already correctly frames aggregate metrics as the problem.

### Key Statistics Summary Table (Section 5, lines 951-1006)

**Current entry for "Signal-carrying positions":** "Noble1, attractor, boundary"

**Add qualifier:** "Noble1, attractor, boundary (aggregate cross-band; per-band patterns may differ)"

### Appendix A: Relationship to Papers 1 and 2 (lines 1135-1167)

**Current (line 1148):** Lists "Retained claims" and "Revised claims" from Papers 1-2.

**Add to revised claims:** "The aggregate enrichment values from Papers 1 and 2 (boundary -18%, Noble1 +39% in Paper 1; boundary 0.41×, Noble1 1.10× in Paper 2) are now labeled as 'aggregate cross-band' values in both papers' revisions, consistent with this paper's demonstration that aggregate all-peaks metrics are extraction-dependent."

This ties the three papers together: the LEMON paper diagnosed the fragility; the Frontiers and EEGMMIDB revisions now acknowledge it with "aggregate" labels.

---

## Changes NOT to Make

1. **Do not reference Voronoi reanalysis or per-band enrichment tables**
2. **Do not add new per-band analyses or tables**
3. **Do not change dominant-peak alignment results** (different metric, unaffected)
4. **Do not change theta EC convergence** (consistent with per-band findings)
5. **Do not change age invariance or cognitive null** (unaffected)
6. **Do not change longitudinal ICC results** (unaffected)
7. **Do not change f0 optimization** (consistent with per-band findings)
8. **Do not change extraction sensitivity results** (STRENGTHENED)
9. **Do not change figures**
10. **Do not restructure the paper**
11. **Do not change the Part I diagnostic** (these findings are the paper's core contribution)

## Summary of Changes

| Type | Count | Nature |
|------|-------|--------|
| Add "aggregate" qualifier | 4 | Abstract, Section 3.6 (×2), Discussion 4.2 |
| Table note | 1 | Table 5 per-position enrichment |
| Key Statistics qualifier | 1 | Signal-carrying positions entry |
| Appendix A cross-reference | 1 | Tie to Frontiers/EEGMMIDB revisions |

**Total: ~7 targeted text changes.** The LEMON paper requires the fewest changes because its central thesis (aggregate metrics are fragile) is confirmed by the reanalysis. The changes are mostly about labeling its own aggregate values as aggregate, consistent with the treatment in the other two papers.

## What These Changes Accomplish

- The "three positions carry signal" claim is correctly scoped as an aggregate finding
- The paper's Table 5 values are labeled as aggregate, consistent with Frontiers and EEGMMIDB
- The narrative arc is coherent: Paper 1 discovers aggregate patterns → Paper 2 tests structural specificity → Paper 3 diagnoses why aggregate patterns are fragile and identifies what survives
- Appendix A explicitly ties the three papers' revisions together
- A reviewer who later sees per-band analysis will find these caveats appropriate

## Risk Assessment

**Risk of NOT making changes:** Low -- the LEMON paper is already the most cautious of the three. A reviewer might notice that the "3 positions carry signal" claim is unqualified, but the paper already discusses band-specific patterns extensively (Table 7, theta convergence analysis).

**Risk of making changes:** Minimal -- all changes are consistent qualifications.

**Recommendation:** Make all proposed changes for consistency with the Frontiers and EEGMMIDB revisions, but recognize that the LEMON paper is already the strongest and most self-critical of the three.
