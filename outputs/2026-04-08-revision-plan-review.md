# Cross-Check: Revision Plan vs Reviewer Feedback vs Voronoi Findings

## Reviewer Concern → Revision Response → Plan Consistency

### R1-A1: FOOOF dependence and alternative analyses

**Reviewer asked for:** Additional analyses showing effects aren't FOOOF artifacts; alternative approaches.

**Revision response:** Added alternative scaling comparison, GED triangulation, parameter sensitivity discussion, IRASA/eBOSC as future work.

**Plan consistency:** The plan strengthens Limitation 5 with self-contained reasoning about cross-band density effects. This is **consistent** — the revision already acknowledges FOOOF sensitivity, and the plan extends that acknowledgment to the aggregate enrichment figures specifically. No conflict.

**Potential issue:** The revision response says "enrichment scores can reverse sign when the FOOOF frequency range is extended from [1, 45] Hz to [1, 85] Hz" — this is already in the paper. The plan adds that aggregate figures may also mask band-specific patterns. This is additive, not contradictory. **OK.**

### R1-A2: SIE terminology

**Plan impact:** None. SIE terminology is unaffected by enrichment methodology. **OK.**

### R1-A3: Single-meditator discovery phase

**Plan impact:** None. Study 1 is explicitly unchanged. **OK.**

### R1-A4: Separating empirical from speculative

**Reviewer asked for:** Clear separation of empirical findings from speculation.

**Plan impact:** The plan softens the gamma discussion (Section 6.2.1) from definitive interpretation to preliminary. This is **consistent** with the reviewer's request — making the mechanistic interpretation more cautious. **OK.**

**Potential issue:** The plan changes the title from "Why Gamma Shows Strongest Adherence" to "Gamma Band Organization." The revision response to R1-A4 says Section 6.2 was renamed "Theoretical Implications (Speculative)." These are compatible — 6.2 is the parent section (speculative), 6.2.1 is a subsection that can have its own title. **OK.**

### R1-B1: SIE pipeline threshold justification

**Plan impact:** None. **OK.**

### R1-B2: Individual alpha frequency variability

**Revision response:** Notes alpha shows weakest phi-adherence (+4.2%), discusses IAF correction.

**Plan consistency:** The plan doesn't change the alpha +4.2% figure. However, the Voronoi analysis shows alpha Noble1 at +27% per-band, which is much stronger than the aggregate +4.2%. This means the revision's statement "alpha exhibits the weakest enrichment at noble positions (+4.2%)" is misleading — alpha is actually the *strongest* per-band Noble1 enrichment.

**Issue identified:** The revision response to R1-B2 says alpha has the "weakest phi-adherence" — this is the aggregate figure. Per-band, alpha Noble1 is +27%. The plan doesn't address this because it doesn't change the R1-B2 response text. But the revision response is in the reviewer letter, not the paper. The paper itself (Section 6.2.6/6.3) says "+4.2% at noble positions vs +144.8% for gamma." The plan doesn't change this line.

**Recommendation:** Add to the plan: In the IAF discussion (Section 6.2.6 or 6.3), where it says "weak phi-adherence observed in the alpha band (+4.2% at noble positions vs +144.8% for gamma)", consider softening to: "modest aggregate phi-adherence in the alpha band (+4.2% at noble positions in cross-band analysis vs +144.8% for gamma)". This is consistent with labeling aggregate values as aggregate.

### R1-B3: Population-level constraints

**Plan impact:** None. **OK.**

### R1-B4: Schematic figure

**Plan impact:** The plan says don't change figures. The schematic figure shows aggregate values (-18%, +21%, +39%). These are now labeled as aggregate in the text/caption per the plan. **OK** — the figure is correctly labeled.

**Potential issue:** The plan says to add a table note to Table 5 but doesn't mention updating the schematic figure caption (Figure 3). Should the Figure 3 caption also get an "aggregate" label?

**Recommendation:** Yes — add to the plan: Figure 3 caption (phi_lattice_schematic) should note "aggregate cross-band values" where it shows -18%, +21%, +39%.

### R1-B5: Cross-frequency coupling literature

**Plan impact:** None. **OK.**

### R2-1 through R2-6: Formatting/editorial

**Plan impact:** None. **OK.**

---

## Voronoi Findings → Plan Consistency

### Finding: Beta-low U-shape (13/13 consistent, boundary +94%)

**Plan addresses:** No. The plan doesn't mention beta-low at all.

**Issue:** The paper currently doesn't discuss beta-low's boundary enrichment pattern (it's visible in Table 7 at +3.9% Noble1 enrichment, but the U-shape isn't discussed). The Voronoi analysis shows this is the strongest and most consistent signal of any band. But per the plan's principle of not introducing new results, this should not be added. **OK — intentional omission.** Beta-low U-shape is Paper 4 material.

### Finding: Alpha Noble1 +27% (SD=3) across 7 datasets

**Plan addresses:** Indirectly. The plan labels aggregate values as aggregate. The per-band alpha result (+27%) is not introduced.

**Consistency check:** The paper says alpha Noble1 is +4.2% (aggregate). The Voronoi shows +27%. The plan doesn't mention this discrepancy. Is this a problem?

**Answer:** No — the +4.2% is the aggregate value for alpha specifically (from Table 7), which pools peaks within the alpha band but still uses cross-band methodology for the position binning. The +27% is from Voronoi bins. These are different analyses. The plan correctly does not try to reconcile them. **OK.**

### Finding: Gamma Noble1 = +5% per-band vs +144.8% aggregate

**Plan addresses:** Yes, extensively. Multiple locations softened/caveated. **OK.**

### Finding: Gamma inverse noble enrichment (+28% to +64%) vs aggregate depletion

**Plan addresses:** Yes. Section 6.2.3 gamma subsection softened from definitive to preliminary. **OK.**

**Potential issue:** The plan says "The apparent avoidance of inverse noble positions may partially reflect cross-band density effects." This is weaker than what we know — the Voronoi analysis definitively shows enrichment, not avoidance. But since the plan's principle is to not introduce new results, the "may partially reflect" language is appropriately cautious. **OK.**

### Finding: CHBMP gamma anomaly

**Plan addresses:** No. Not relevant to this paper (CHBMP isn't in the Frontiers datasets). **OK.**

### Finding: f₀=7.60 produces stronger boundary enrichment than f₀=7.83

**Plan addresses:** No. The paper already uses f₀=7.60. The Voronoi finding strengthens this choice but no change is needed in the paper. **OK.**

---

## Consistency Between Revision Response and Plan

### Issue 1: Companion paper references

**Revision response** extensively references "Lacy, in preparation" for GED, LEMON, Dortmund, EEGMMIDB results.

**Plan** says "Do not reference companion papers or unpublished analyses."

**Conflict?** No — the revision response is a separate document (reviewer letter). The plan applies to changes being made NOW to the paper text. The companion paper references were already added in the previous revision round. The plan doesn't remove them — it just doesn't add new ones. **OK.**

### Issue 2: GED tau = 1.0 claim

The revision response (R1-A1) says GED "reproduced the identical position-type hierarchy (Kendall's tau = 1.0)." This is the aggregate GED result. The plan's tau caveat applies to the FOOOF aggregate tau, not the GED tau. Should the GED tau also be caveated?

**Recommendation:** The GED result is in Section 6.3 (Independent Replication) which the plan says to leave unchanged. The GED analysis is a different method and may or may not show the same band-specific patterns. Since we haven't done per-band Voronoi on GED data, we can't know. Leave unchanged but note this as a potential future issue. **OK for now.**

### Issue 3: EEGEmotions-27 gamma claim

The revision added EEGEmotions-27 showing "Gamma showed strongest 1deg noble enrichment (+72.3%)." The plan says don't change the EEGEmotions replication. But this +72.3% is the same aggregate methodology that produces +144.8% in the primary dataset — it would also be affected by per-band normalization.

**Recommendation:** Add to the plan: Where EEGEmotions gamma results are mentioned (Section 4.3.5, Figure 7 caption), add "aggregate" label consistent with the treatment of the primary dataset gamma values. This is a small addition that maintains consistency. Currently the plan misses these EEGEmotions-specific gamma locations.

---

## Summary of Issues Found

### Must fix (inconsistencies):

1. **EEGEmotions gamma +72.3%** — needs same "aggregate" treatment as primary +144.8%. Add to plan: Section 4.3.5 and Figure 7 caption.

### Should fix (completeness):

2. **Figure 3 caption** (phi_lattice_schematic) — should label values as aggregate, consistent with Table 5 note. Add to plan.

3. **Alpha +4.2% in IAF discussion** (Section 6.2.6/6.3) — should label as "aggregate" consistent with the plan's approach. Add to plan.

### No action needed (noted for future):

4. **GED tau = 1.0** — may have same aggregate issue but we haven't tested. Leave for now.
5. **Beta-low U-shape** — genuine novel finding but Paper 4 material, not for this revision.
