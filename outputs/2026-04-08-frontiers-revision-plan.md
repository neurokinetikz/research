# Frontiers Revision Update Plan

**Date:** 2026-04-08
**Context:** Per-band adaptive-resolution Voronoi enrichment analysis across 7 datasets reveals band-specific patterns that qualify several aggregate claims in the revision.
**Strategy:** Minimal targeted changes to soften/caveat claims using self-contained language. No references to companion papers, new methods, or unpublished results. Just honest qualification of aggregate methodology.

## Principles

1. **Do not restructure the paper**
2. **Do not reference companion papers or unpublished analyses**
3. **Do not introduce new terminology** (Voronoi, adaptive resolution, etc.)
4. **Label aggregate numbers as aggregate** where they appear
5. **Use the paper's own FOOOF sensitivity caveat** as the anchor for qualification
6. **Do not change figures** — they correctly show the aggregate analysis as performed

## Changes by Section

### Abstract (lines ~120-130)

**Current:** "boundaries showed $-18\%$ depletion, attractors $+21\%$ enrichment, and noble positions ($n + 0.618$) $+39\%$ enrichment"

**Change to:** "boundaries showed $-18\%$ depletion, attractors $+21\%$ enrichment, and noble positions ($n + 0.618$) $+39\%$ enrichment in aggregate cross-band analysis"

**Current:** "Gamma exhibited strongest adherence ($+144.8\%$), consistent with functional requirements for precise phase relationships."

**Change to:** "Gamma exhibited strongest aggregate adherence ($+144.8\%$ at Noble1 in cross-band analysis), consistent with functional requirements for precise phase relationships, though this aggregate figure may partially reflect cross-band density effects (see Section 6.7)."

### Significance Statement (line ~133)

**Current:** "Gamma oscillations show by far the strongest adherence, consistent with their stringent requirements for precise temporal binding."

**Change to:** "Gamma oscillations show the strongest aggregate adherence, consistent with their stringent requirements for precise temporal binding."

One word change — "by far the" → "the". Adds "aggregate". No new claims.

### Section 3.2.5 — Predicted Band-Specific Heterogeneity (lines ~990-997)

**Current:** Predicts gamma >> alpha >> theta gradient, ending with "frequency-dependent magnitude (gamma $\gg$ alpha $>$ theta $>$ delta)".

**Change to:** "frequency-dependent magnitude, with gamma predicted to show strongest expression and lower-frequency bands progressively weaker, though the aggregate analysis in Section 4.2.3 may conflate distinct band-specific patterns (see Limitation 5, Section 6.7)."

### Table 5 (tab:enrichment_results, line ~1060)

**Add table note:** "Values are cross-band aggregates pooling peaks from all $\phisym$-octave bands. Band-specific enrichment patterns may differ from these aggregate values (see Section 4.2.3 and Limitation 5)."

### Section 4.2.3 — Band-Specific Heterogeneity (lines ~1138-1150)

**Current:** "The gamma band showed dramatically stronger $\phisym^n$ adherence ($+144.8\%$ at 1deg noble) than any other band"

**Change to:** "The gamma band showed dramatically stronger aggregate $\phisym^n$ adherence ($+144.8\%$ at 1\degree\ noble) than any other band in this cross-band analysis. As noted in Limitation 5, aggregate enrichment values are sensitive to spectral parameterization methodology and may not fully capture within-band position structure."

### Section 4.2.2 — Position-Type Enrichment (line ~1072)

**Current:** "The core hierarchy (Boundary $<$ 2\degree\ Noble $<$ Attractor $<$ 1\degree\ Noble) confirmed theoretical predictions (Kendall's $\tau = 1.0$, $p = 0.042$)."

**Add after:** "(Note: this ordering reflects the aggregate cross-band analysis; individual frequency bands may show distinct position preferences, as suggested by the band-specific heterogeneity in Table 7.)"

Single caveat here covers all subsequent tau mentions without requiring edits throughout the paper.

### Section 6.2.1 — Why Gamma Shows Strongest Adherence (lines ~1408-1425)

**Current title:** "Why Gamma Shows Strongest Adherence"

**Change title to:** "Gamma Band Organization"

**Current:** "The dramatic gamma enrichment at $\phisym^n$ positions ($+144.8\%$ vs. $<10\%$ for other bands) is consistent with gamma's stringent functional requirements."

**Change to:** "The strong gamma enrichment at $\phisym^n$ positions in the aggregate analysis ($+144.8\%$ at Noble1 vs. $<10\%$ for other bands) is consistent with gamma's stringent functional requirements, though the aggregate methodology may overestimate Noble1-specific enrichment relative to other positions within the gamma band (see Limitation 5)."

### Section 6.2.3 — Band-Specific Position Preferences (lines ~1430-1470)

**Gamma subsection — Current:** "1deg Noble: $+144.0\%$ enrichment...3deg Inverse: $-39.2\%$ depletion...4deg Inverse: $-99.7\%$ depletion (near complete avoidance)"

**Add after the bullet points:** "These values are from the aggregate cross-band analysis. The inverse noble depletion in gamma should be interpreted cautiously, as the aggregate methodology pools peaks across bands with different density distributions. The apparent avoidance of inverse noble positions may partially reflect cross-band density effects rather than genuine within-band depletion at these positions (see Limitation 5)."

**Current:** "The severe depletion at inverse noble positions indicates that gamma oscillators actively avoid these locations, possibly because the sum-structure of inverse nobles introduces subtle resonance pathways incompatible with gamma's precision requirements."

**Change to:** "The apparent depletion at inverse noble positions in the aggregate analysis suggests gamma oscillators may avoid these locations, though this interpretation should be considered preliminary given the sensitivity of aggregate enrichment to spectral parameterization methodology (see Limitation 5)."

Softens from definitive to preliminary without introducing new results.

**Table 9 (tab:band_position_empirical):** Add table note: "Aggregate cross-band values. Within-band enrichment patterns may differ from these aggregate figures (see Limitation 5)."

### Section 6.7 — Limitations, Item 5 (FOOOF dependence, line ~1589)

**Current:** "enrichment scores can reverse sign when the FOOOF frequency range is extended from [1, 45] Hz to [1, 85] Hz, though dominant-peak alignment remains robust."

**Extend to:** "enrichment scores can reverse sign when the FOOOF frequency range is extended from [1, 45] Hz to [1, 85] Hz, though dominant-peak alignment remains robust. More generally, the aggregate cross-band enrichment figures reported in Tables 5 and 7 pool peaks from frequency bands with different spectral density distributions, which may mask band-specific enrichment patterns. The gamma Noble1 enrichment ($+144.8\%$), while the strongest aggregate figure, should be interpreted as reflecting the combined effects of within-band position structure and cross-band density variation. Alternative aperiodic separation methods (IRASA, eBOSC) and per-band enrichment normalization represent priorities for future validation."

Self-contained — uses the paper's own data and methodology concerns to motivate the caveat.

### Section 6.8 — Future Directions

**Add as new item:** "\\textbf{Per-band enrichment normalization:} The aggregate enrichment analysis reported here pools peaks across all frequency bands. Per-band analysis with band-appropriate spectral resolution and density normalization may reveal band-specific position preferences not captured by the aggregate methodology, and represents a natural extension of the present work."

Generic future direction, not previewing specific results.

### Section 7 — Conclusions (line ~1635)

**Current:** "boundaries depleted ($-18\%$), attractors enriched ($+21\%$), noble positions maximally enriched ($+39\%$)"

**Change to:** "boundaries depleted ($-18\%$), attractors enriched ($+21\%$), noble positions maximally enriched ($+39\%$) in aggregate cross-band analysis"

### EEGEmotions-27 gamma mentions (Section 4.3.5, Figure 7 caption)

Wherever EEGEmotions gamma Noble1 +72.3% is mentioned, apply the same "aggregate" treatment as the primary dataset:

**Current (Section 4.3.5/Figure 7):** "Gamma showed strongest 1deg noble enrichment ($+72.3\%$)"

**Change to:** "Gamma showed strongest aggregate 1\degree\ noble enrichment ($+72.3\%$)"

This maintains consistency — both datasets' gamma figures get the same "aggregate" label.

### Figure 3 caption (phi_lattice_schematic)

**Current:** Shows -18%, +21%, +39% without qualification.

**Add to caption:** After the value descriptions, add: "These values are aggregate cross-band figures (see Table 5 note)."

### Alpha +4.2% in IAF discussion (Section 6.2.6 / 6.3)

**Current:** "$+4.2\%$ at noble positions vs.\ $+144.8\%$ for gamma"

**Change to:** "$+4.2\%$ at noble positions in aggregate vs.\ $+144.8\%$ for gamma"

One word insertion, consistent with the "label as aggregate" approach.

### Theta inverse noble values (Section 6.2.3)

**Current:** "4deg Inverse: $+47.2\%$ enrichment" and "3deg Inverse: $+24.0\%$ enrichment"

**No change.** These are the aggregate values and they are what the analysis produced. The theta direction is correct even per-band. Changing specific numbers without explaining why would be inconsistent.

---

## Changes NOT to Make

1. **Do not change Study 1 (SIE)** — entirely unaffected
2. **Do not change the independence-convergence paradox**
3. **Do not change f₀ convergence**
4. **Do not change the substrate-ignition model**
5. **Do not change the theoretical framework** (Section 3, except softening the gradient prediction)
6. **Do not change permutation test results**
7. **Do not change figures**
8. **Do not change the core equation**
9. **Do not change the EEGEmotions-27 replication** — qualitative pattern replicates
10. **Do not add per-band tables or new analysis**
11. **Do not reference companion papers or unpublished work**
12. **Do not introduce new terminology** (Voronoi, adaptive resolution)
13. **Do not change specific numbers** except to label them as "aggregate"

## Summary of Changes

| Type | Count | Nature |
|------|-------|--------|
| Add "aggregate" label | 9 | Word insertion (Abstract, Sig Statement, Table 5, Fig 3, Conclusions, EEGEmotions gamma, alpha IAF discussion) |
| Soften gamma claims | 3 | Reframe as aggregate, add Limitation 5 reference |
| Soften gamma inverse noble claims | 2 | From definitive to preliminary |
| Tau = 1.0 caveat | 1 | Single note at first appearance |
| FOOOF limitation strengthen | 1 | Extend existing caveat with self-contained reasoning |
| Future Direction addition | 1 | Generic per-band normalization item |
| Section title change | 1 | "Why Gamma Shows Strongest Adherence" → "Gamma Band Organization" |
| Band gradient softening | 1 | Add forward reference to Limitation 5 |

**Total: ~19 targeted text changes.** All self-contained within the paper's own framework. No new results introduced. No companion papers referenced. The paper's core contributions are preserved and the caveats are honest and proportionate.

## What These Changes Accomplish

- A reviewer who later sees the per-band analysis will find these caveats appropriate and honest
- No new claims are made that would need defending
- The aggregate analysis is correctly characterized as aggregate
- The gamma narrative shifts from "definitive Noble1 dominance" to "strong aggregate signal, methodology-dependent"
- The FOOOF limitation section becomes the single anchor for all methodological caveats
- The paper remains self-consistent — nothing contradicts anything else
