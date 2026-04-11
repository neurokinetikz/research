# EEGMMIDB Paper Revision Plan

**Date:** 2026-04-08
**Context:** Per-band adaptive-resolution Voronoi enrichment analysis across 9 datasets reveals that "boundary depletion" and "noble enrichment" are aggregate cross-band claims that mask band-specific patterns. The EEGMMIDB paper's core structural specificity findings (phi rank 1, irrational advantage, state dynamics) are likely robust, but several claims need qualification.
**Strategy:** Minimal targeted changes to qualify aggregate claims using the paper's own data and methodology. No references to companion analyses, Voronoi methodology, or unpublished per-band results. All caveats self-contained within the paper's existing framework.

## Principles

1. **Do not reference the Voronoi reanalysis, companion papers, or external results**
2. **Do not restructure the paper or add new analyses**
3. **Do not change figures**
4. **Do not change the structural specificity results** (Test A-D, bootstrap rankings) -- these are relative comparisons and survive normalization changes
5. **Do not change state dynamics results** -- already acknowledged as non-phi-specific
6. **Label aggregate enrichment values as aggregate where they appear**
7. **Use the paper's own band-stratified analysis (Section 5.9, Test C)** as the anchor for caveats -- it already shows phi is NOT dominant in any single band
8. **Strengthen existing limitations** rather than adding new ones

---

## Discrepancy Analysis

### Claims That Need Qualification

#### D1: "Boundary depletion 0.41x" (appears ~8 times)

**Paper claim:** Boundaries show 0.41x peak density (59% fewer peaks) -- presented as universal property.

**Per-band reality:**
- Alpha: boundaries depleted (-18% to -25%) -- **consistent**
- Beta-low: boundaries massively ENRICHED (+91% / +78%) -- **contradicts**
- Theta: boundaries enriched (+70% / +25%) -- **contradicts**
- Beta-high: mixed (+12% / -11%) -- **ambiguous**
- Gamma: weakly enriched (+10% / +23%) -- **contradicts**

**Issue:** Only alpha shows the predicted depletion. The aggregate 0.41x is driven by alpha having many peaks near the boundary regions with low density, but the pattern reverses in beta-low and theta. This is an aggregate artifact -- the "depletion" framing is misleading.

**However:** The paper's SS formula uses aggregate enrichment across all peaks. The aggregate depletion IS what the aggregate analysis found. The discrepancy is that the paper presents this as a universal lattice property rather than a weighted average across bands with opposing patterns.

#### D2: "Noble enrichment 1.10x" (appears ~5 times)

**Paper claim:** Noble1 shows 1.10x enrichment (10% more peaks) -- presented alongside boundary depletion.

**Per-band reality:**
- Alpha: Noble1 +29% -- **strongly consistent**
- Gamma: Noble1 +20% -- **consistent**
- Others: Noble1 near zero or negative

**Issue:** Less severe than boundary depletion. The alpha signal dominates the aggregate and the direction is correct. But the magnitude (+10% aggregate) understates the alpha-specific finding (+29%) while masking near-zero contributions from other bands.

#### D3: Band-stratified analysis framing (Section 5.9)

**Paper already says:** "φ not consistently dominant in any single band; advantage arises from consistent moderate performance across all bands."

**Per-band reality:** Confirms this. But the paper's specific band scores (gamma SS=146.2 for phi) are computed with aggregate methodology. Per-band Voronoi would produce different numbers for each base. The relative RANKING across bases might or might not survive.

**Issue:** The paper's band-stratified scores use position-specific enrichment computed from the same aggregate methodology. The framing is adequate but could be strengthened by noting that band-specific enrichment patterns may differ from aggregate patterns.

#### D4: Beta-low positive control framing (Section 6.4)

**Paper claim:** "Mode consistently upper octave (0.86-0.99)... Companion studies show noble1 dominance in non-motor datasets."

**Per-band reality:** Beta-low inv_noble_4 (+66%) and inv_noble_6 (+85%) are correctly identified as enriched. But boundaries are even MORE enriched (+91% / +78%) and the paper doesn't discuss this. The beta-low "upper octave" mode actually reflects a U-shape with massive boundary enrichment.

**Issue:** The paper's KDE mode analysis captures the upper-octave clustering but misses the lower-boundary enrichment because it focuses on the single mode, not the full distribution shape. The beta-low pattern is better described as "boundary-dominant U-shape" than "upper-octave noble concentration."

**However:** The paper's mixture model and targeted contrast correctly detect the inv_noble effects. The boundary enrichment would be visible in a different analysis but doesn't contradict the specific tests performed. The paper could note that KDE modes capture only part of the within-octave structure.

#### D5: SS interpretation -- "boundary depletion + noble enrichment = phi advantage"

**Paper claim (Section 6.6):** "φ achieves advantage entirely from nobles" and the SS formula = -E_boundary + E_attractor + E_nobles.

**Per-band reality:** Computing SS per-band shows only alpha yields positive SS for phi. All other bands yield negative SS (boundaries enriched, attractors depleted or weak). The aggregate positive SS is driven by alpha.

**Issue:** This doesn't invalidate the aggregate SS analysis (which correctly measures what it measures), but the interpretation that phi's advantage is "from nobles" needs qualification. Alpha's Noble1 at +29% is the driver. The noble enrichment in other bands is weak or absent.

### Claims That Survive Unchanged

#### S1: Structural specificity ranking (phi rank 1 in 100% bootstraps)
Relative comparison between bases. All bases computed with same aggregate methodology. Ranking likely preserved under per-band correction since all bases are equally affected.

#### S2: Irrational advantage (d=1.22, window f0=7.4-9.4 Hz)
Class-level comparison. Same aggregate methodology applies to all bases equally.

#### S3: State dynamics (gamma reallocation, beta-low motor)
Already acknowledged as non-phi-specific. Within-subject comparisons are unaffected by enrichment normalization.

#### S4: Sigmoid cliff (f0=7.48-7.58 Hz transition)
Relative ranking comparison across anchor values. Unaffected.

#### S5: Non-exponential control (linear grids SS near zero)
Different functional form comparison. Unaffected.

#### S6: Cross-resolution robustness (nperseg=256 replication)
Relative comparison replicated at different resolution. Unaffected.

#### S7: Alpha KDE mode and IAF confound discussion
The paper already notes this concern extensively. Per-band analysis actually validates that the alpha pattern is real (Noble1 +29% across 9 datasets).

---

## Changes by Section

### Abstract (Lines 115-125)

**Current:** "boundary depletion 0.41x ... noble enrichment 1.10x"

**Change to:** "aggregate boundary depletion 0.41x ... aggregate noble enrichment 1.10x"

Two word insertions. No other abstract changes.

### Significance Statement (Lines 127-129)

No changes needed -- focuses on SS ranking, dynamics, and irrational advantage, not specific enrichment values.

### Introduction (Lines 136-142)

**Current (line 142):** "boundary depletion 0.41x, noble enrichment 1.10x"

**Change to:** "aggregate boundary depletion 0.41x, noble enrichment 1.10x"

One word insertion.

### Section 5.2: Overall Lattice Structure (Lines 281-283)

**Current:** "Boundary depletion: 0.41x (59% fewer than expected)" and "Noble1 enrichment: 1.10x"

**Change to:** "Aggregate boundary depletion: 0.41x (59% fewer than expected across all phi-octave bands)" and "Aggregate Noble1 enrichment: 1.10x"

**Add after these values:** "These aggregate values pool peaks from all five phi-octave bands. Band-stratified analysis (Section 5.9) reveals that phi is not consistently dominant in any single band, suggesting that aggregate enrichment values may conflate distinct band-specific patterns."

This uses the paper's own band-stratified finding to motivate the caveat.

### Section 5.9: Band-Stratified Analysis (Lines 468-469)

**Current:** Reports per-band SS for phi and other bases. Notes phi ranks 4th in gamma, 4th in alpha, 7th in beta-low.

**Add after per-band findings:** "The divergence between per-band and aggregate rankings indicates that the aggregate structural score (Table 5) reflects weighted contributions from bands with different enrichment profiles. The aggregate boundary depletion (0.41x) and noble enrichment (1.10x) reported in Section 5.2 should be interpreted accordingly: they describe the population-level peak distribution across all bands pooled, not a universal within-band pattern."

Self-contained -- uses the paper's own band-stratified data.

### Section 6.1: Lattice Replication (Lines 575-577)

**Current:** "Boundary depletion: 0.41x" and "Noble enrichment: 1.10x" presented as replication of companion study.

**Change to:** "Aggregate boundary depletion: 0.41x" and "aggregate noble enrichment: 1.10x" followed by: "These aggregate values replicate the companion study's cross-band analysis methodology. The band-stratified analysis in Section 5.9 indicates that the aggregate pattern reflects primarily alpha-band contributions, with different bands potentially contributing different enrichment profiles."

### Section 6.4: Motor Beta-Low as Positive Control (Lines 595-597)

**Current:** "Mode consistently upper octave (0.86-0.99)" and "Companion studies show noble1 dominance in non-motor datasets"

**Add caveat:** "The KDE mode captures the dominant clustering in the upper octave but does not characterize the full within-octave distribution. The structural score analysis (Section 5.9) shows that beta-low's contribution to the aggregate phi score differs from alpha's, consistent with band-specific enrichment profiles that aggregate analysis may obscure."

### Section 6.5: Dynamic Effects Not Phi-Specific; Static Structure Favors Phi (Lines 599-618)

**Current (line 604):** "φ achieves advantage entirely from nobles" 

**Change to:** "φ achieves its aggregate advantage from noble position enrichment"

One word insertion.

**Current:** "consistent moderate performance across all bands"

**Add qualification:** "consistent moderate performance across all bands in the aggregate analysis, though per-band contributions to the aggregate structural score differ substantially (Section 5.9)"

### Section 6.6: Interpretation of Structural Advantage (Lines 608-612)

**Current:** "Qualitatively correct enrichment: boundary -17.9%, noble +14.0%"

**Change to:** "Qualitatively correct aggregate enrichment: boundary -17.9%, noble +14.0%"

One word insertion.

### Section 7: Limitations

#### Strengthen existing Limitation 10 (Structural score interpretation, line 666)

**Current:** "Negative SS for rationals = anti-correlation... NOT 'active rejection'"

**Extend with new paragraph:** "More broadly, the structural score aggregates enrichment across all phi-octave bands. The band-stratified analysis (Section 5.9) reveals substantial heterogeneity in per-band structural scores, with phi ranking outside the top three in most individual bands. The aggregate SS = 45.6 therefore represents a weighted average across bands with different enrichment profiles, and the boundary depletion / noble enrichment pattern that drives the aggregate score may be primarily alpha-band driven. Future work should examine whether per-band enrichment analysis with band-appropriate normalization preserves the structural advantage observed in the aggregate."

Self-contained -- refers only to the paper's own Section 5.9 data.

#### Add new limitation (after current Limitation 12)

**New Limitation 13:** "The aggregate enrichment values (boundary 0.41x, noble 1.10x) pool peaks from all five phi-octave bands. Because these bands span different frequency ranges with different peak densities and spectral characteristics, the aggregate enrichment may conflate band-specific patterns. Per-band enrichment analysis with band-appropriate spectral resolution and density normalization represents a priority for future validation."

### Section 8: Future Directions

**Add new item:** "Per-band enrichment normalization: The current structural specificity analysis computes enrichment from peaks pooled across all phi-octave bands. Per-band analysis with band-appropriate spectral resolution may reveal whether the aggregate structural advantage of phi reflects consistent enrichment across all bands or is driven by specific band contributions. This extends the band-stratified comparison in Section 5.9 to enrichment-based rather than score-based evaluation."

### Section 9: Conclusions (Lines 680-705)

**Current (line 680):** "φ-lattice replicates across paradigms -- boundary 0.41x, noble 1.10x"

**Change to:** "φ-lattice replicates across paradigms -- aggregate boundary 0.41x, aggregate noble 1.10x"

Two word insertions.

---

## Changes NOT to Make

1. **Do not change any SS values, rankings, or bootstrap results** -- these are what the analysis produced
2. **Do not change the irrational advantage analysis** -- relative comparison, unaffected
3. **Do not change state dynamics results** -- already non-phi-specific
4. **Do not change the sigmoid cliff analysis** -- relative ranking
5. **Do not change Tables 1-5** -- they correctly show what the analyses produced
6. **Do not change Figures 1-7** -- they correctly show the analyses performed
7. **Do not reference Voronoi, adaptive resolution, or companion paper results**
8. **Do not add new per-band tables or figures**
9. **Do not change the non-exponential control** -- unaffected
10. **Do not change the cross-resolution replication** -- unaffected
11. **Do not change the mixture model results** -- within-subject comparisons unaffected
12. **Do not change the alpha IAF confound discussion** -- already thorough and honest

---

## Summary of Changes

| Type | Count | Nature |
|------|-------|--------|
| Add "aggregate" label | 7 | Word insertion (Abstract x2, Intro, 5.2, 6.1, 6.6, Conclusions) |
| Band-stratified cross-reference | 3 | Forward reference to Section 5.9 (5.2, 6.1, 6.5) |
| Strengthen Limitation 10 | 1 | Extend with self-contained band-stratification caveat |
| New Limitation 13 | 1 | Aggregate methodology caveat |
| New Future Direction | 1 | Per-band enrichment normalization |
| Beta-low positive control caveat | 1 | KDE mode vs full distribution note |
| SS interpretation softening | 1 | "aggregate advantage" language |

**Total: ~16 targeted text changes.** All self-contained within the paper's own framework. No new results introduced. No companion papers referenced.

---

## What These Changes Accomplish

- The aggregate enrichment values (0.41x, 1.10x) are correctly labeled as aggregate
- The paper's OWN band-stratified analysis (Section 5.9) is leveraged as the anchor for caveats -- this is honest and internally consistent
- The core structural specificity claim (phi rank 1, SS=45.6, irrational advantage) is PRESERVED because it's a relative comparison
- The state dynamics findings are PRESERVED because they're already non-phi-specific
- A reviewer who later sees per-band analysis will find these caveats appropriate
- The paper acknowledges that aggregate and per-band analyses may tell different stories, without needing to resolve the question

## What These Changes Do NOT Accomplish

- They do not resolve WHETHER phi retains rank 1 under per-band normalization (this requires new analysis)
- They do not explain the beta-low U-shape (this is novel finding for a future paper)
- They do not address the CHBMP gamma anomaly (different dataset)
- They do not reconcile the f0=8.5 vs f0=7.60 optimal anchor discrepancy between papers

## Risk Assessment

**Risk of NOT making changes:** A reviewer with access to per-band analysis tools could reproduce the beta-low boundary enrichment (+91%) and challenge the "boundary depletion 0.41x" framing as misleading. The paper's own band-stratified analysis already hints at this but doesn't connect it to the aggregate enrichment claims.

**Risk of making changes:** Minimal. Adding "aggregate" labels and strengthening caveats with the paper's own data is honest and proportionate. The core findings (structural specificity, irrational advantage, state dynamics) are unaffected.

**Recommendation:** Make all proposed changes before next revision opportunity. The EEGMMIDB paper is more vulnerable than the Frontiers paper because it makes stronger structural claims (SS=45.6, phi rank 1) that implicitly depend on the aggregate enrichment profile being universal across bands.
