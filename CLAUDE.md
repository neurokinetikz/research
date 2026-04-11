# EEG Research Knowledge Base

## What This Is
A personal knowledge base about EEG frequency band research, maintained as an Obsidian vault. The AI is the sole maintainer of the wiki -- it reads raw sources, synthesizes findings into wiki articles, and keeps everything consistent and cross-linked.

## Repository Structure

| Directory | Contents | Index |
|-----------|----------|-------|
| raw/ | Unprocessed analysis reports (.md) and papers (.pdf). **Never modify these files.** | -- |
| wiki/ | The organized wiki. AI maintains this entirely. | wiki/INDEX.md |
| scripts/ | Python scripts for EEG analysis (flat, snake_case.py). | scripts/INDEX.md |
| lib/ | Reusable Python modules imported by scripts. | lib/INDEX.md |
| data/ | Descriptions of external datasets stored on /Volumes/T9. | data/INDEX.md |
| papers/ | LaTeX manuscripts (.tex) and bibliographies (.bib). | -- |
| outputs/ | Generated reports, answers, and analyses on request. | -- |

## Wiki Conventions

### Filenames
- One topic per file. Kebab-case: `alpha-oscillations.md`, `cross-frequency-coupling.md`
- Band-specific enrichment maps use the pattern `{band}-band-map.md`

### Article Structure
Every wiki article follows this order:

1. **Title**: `# Topic Name`
2. **Opening paragraph**: A dense, self-contained summary of the topic. Include inline citations `(Author, Year)` and cross-links `[[topic-name]]`. This paragraph should let a reader understand the topic without reading further.
3. **Topical sections**: `## Heading` sections covering distinct subtopics. Use narrative prose, not bullet lists, for substantive content. Tables for structured data (enrichment values, position maps, dataset properties).
4. **Sources section**: `## Sources` at the bottom. Each entry on its own line:
   - For synthesis reports: `- "Report Title" (synthesis report)`
   - For papers: `- Author (Year) -- brief description of contribution`

### Citations
- Inline: `(Author, Year)` -- e.g. `(Lacy, 2026)`, `(Williams et al., 2026)`
- Every citation in the article body must have a corresponding entry in Sources

### Cross-Linking
- Use `[[topic-name]]` for standard links
- Use `[[topic-name|display text]]` when the display text should differ: `[[critical-frequencies|40 Hz]]`
- Link generously on first mention of a related topic within each section
- Prioritize links to: related frequency bands, mechanisms, functions, and the unified framework

### Formatting
- Math and symbols: inline Unicode (f₀, φ, Δf, φⁿ, ~10 Hz, ≈, →, ×)
- Emphasis: **bold** for key terms on first introduction
- Dashes: `--` (double hyphen) for em-dashes in prose
- Data: markdown tables with alignment for numerical columns

### INDEX.md
- Location: `wiki/INDEX.md`
- Organized into sections: Unified Framework, Frequency Bands, Mechanisms, Functions, Individual Differences, Development, Applications
- Each entry: `- [[filename]] - ` followed by a multi-sentence description summarizing the article's key content, findings, and distinctive claims
- Keep INDEX.md entries substantive (2-5 sentences). They serve as an article abstract.
- When adding a new article, place it in the correct section and maintain alphabetical order within sections

### Quality Standards
- Articles should be comprehensive, not stubs. Typical range: 5-30 KB.
- Include specific numbers: frequencies in Hz, effect sizes, sample sizes, correlation coefficients
- Note methodological caveats (e.g., scalp EEG gamma contamination, aperiodic confounds)
- When sources disagree, present both positions with citations rather than picking a winner
- Distinguish established findings from speculative claims

## Workflows

### Ingesting a New Raw Source

When a new file appears in raw/ or the user provides material to ingest:

1. **Read the full source.** Identify: core findings, methods, frequency bands involved, brain regions, relevant existing wiki topics.
2. **Map to existing articles.** For each finding, determine which wiki article(s) it belongs in. Check the current content of those articles to avoid duplication.
3. **Update existing articles.** Integrate new findings into the appropriate sections. Add inline citations. Add the source to `## Sources`. Preserve existing prose -- extend, don't replace.
4. **Create new articles only when** a topic has enough substance for a full article (multiple findings, distinct from existing coverage) and doesn't fit naturally as a section in an existing article. When in doubt, add to an existing article.
5. **Update INDEX.md.** If any new articles were created, add entries in the correct section. If existing articles gained significant new content, update their INDEX descriptions.
6. **Cross-link.** Add `[[links]]` from new content to related articles, and from related articles back to any new articles.
7. **Report what changed.** List which articles were updated or created, and what content was added.

### Answering Questions

When the user asks a research question:

1. **Search the wiki first.** Check relevant articles for existing coverage.
2. **Synthesize from wiki content** with inline citations. If the wiki fully covers the question, answer from it.
3. **If the wiki is insufficient**, draw on general knowledge but flag what comes from the wiki vs. external knowledge.
4. **For substantive answers** (multi-paragraph, novel synthesis), write the output to `outputs/` as a dated markdown file (e.g., `outputs/2026-04-07-theta-gamma-coupling-summary.md`).
5. **If the answer reveals wiki gaps**, note them and offer to update the relevant articles.

### Wiki Maintenance

Perform these checks when explicitly asked or when a significant batch of new content has been added:

- **Splitting**: If an article exceeds ~30 KB or covers clearly separable subtopics, propose splitting it. Move content to a new article, leave a summary + link in the original.
- **Merging**: If two articles substantially overlap or one is too thin to stand alone, merge into the stronger article.
- **Contradictions**: When sources conflict, present both findings with citations and note the disagreement explicitly. Do not silently overwrite one view with another.
- **Stale INDEX.md**: After any article changes, verify the INDEX.md description still accurately reflects the article content.
- **Orphan check**: Ensure every wiki article is listed in INDEX.md and has at least one incoming `[[link]]` from another article.
- **Sources audit**: Ensure every article's `## Sources` section exists and lists all cited works.

## Scripts and Lib

- scripts/ contains standalone analysis scripts (flat directory, snake_case.py)
- lib/ contains reusable modules imported by scripts
- Both have their own INDEX.md documenting each file
- When creating or modifying scripts, update the relevant INDEX.md
- Scripts should import shared functionality from lib/ rather than duplicating code

## Research Interests

1. Frequency architecture of the brain (φ-lattice, canonical bands, f₀)
2. Definition and boundaries of canonical EEG bands
3. Connections between specific frequencies, functions, and brain regions
4. Signatures of consciousness vs unconsciousness
5. Brainwaves as computation (oscillatory coding, inhibitory stencils, ephaptic coupling)
6. Psychedelic states and their oscillatory signatures
7. Meditation and contemplative neuroscience
8. Neurofeedback and brain stimulation
9. Cross-species conservation of frequency organization
10. Developmental trajectories of oscillatory architecture
