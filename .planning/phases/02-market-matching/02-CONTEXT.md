# Phase 2: Market Matching - Context

**Gathered:** 2026-04-02
**Status:** Ready for planning

<domain>
## Phase Boundary

Build a two-stage market matching pipeline that identifies equivalent contracts across Kalshi and Polymarket. Stage 1 uses keyword matching for candidate generation, Stage 2 uses sentence-transformer semantic similarity for scoring. Output is a curated JSON registry of matched pairs with confidence scores and settlement criteria documentation.

</domain>

<decisions>
## Implementation Decisions

### Matching Approach
- **Two-stage pipeline**: keyword-based candidate generation first, then sentence-transformer semantic similarity scoring
- Use `all-MiniLM-L6-v2` from sentence-transformers (recommended in STACK.md research, already installed)
- Score each candidate pair with a combined keyword + semantic similarity score
- Threshold for auto-accept vs. manual review TBD by planner based on score distributions

### Manual Curation
- Build a simple CLI-based review interface (not a web app)
- Present each candidate pair with: Kalshi title, Polymarket title, similarity score, settlement criteria from both
- User accepts, rejects, or flags for further review
- Output: `data/processed/matched_pairs.json`

### Settlement Criteria
- Extract and document settlement/resolution criteria from both platforms for each matched pair
- Flag pairs where settlement criteria diverge significantly
- This documentation feeds into the paper's discussion of settlement divergence risk

### Confidence Scoring
- Combined score from keyword overlap + semantic similarity
- Exact formula: Claude's discretion, but must produce a 0-1 float

### Data Sources
- Kalshi market metadata from Phase 1 parquet files and metadata JSONs in `data/raw/kalshi/`
- Polymarket market metadata from Phase 1 parquet files and metadata JSONs in `data/raw/polymarket/`
- Both adapters store market titles, descriptions, categories, and resolution criteria

### Scope Gate
- After matching completes, report total matched pair count
- If < 30 pairs: flag for scope reduction (drop TFT per roadmap)
- This is a critical project-level decision point

### Claude's Discretion
- Exact keyword matching algorithm (TF-IDF, Jaccard, simple token overlap)
- Similarity score combination formula
- Auto-accept/manual-review threshold
- CLI interface design details
- How to handle one-to-many matches (one Kalshi event matching multiple Polymarket markets)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase 1 Outputs
- `src/data/kalshi.py` — KalshiAdapter.list_markets() returns MarketMetadata objects
- `src/data/polymarket.py` — PolymarketAdapter.list_markets() returns MarketMetadata objects
- `src/data/schemas.py` — MarketMetadata dataclass (platform, ticker, title, description, category, url, resolution_source, close_time)

### Research
- `.planning/research/ARCHITECTURE.md` — Market matching component boundaries
- `.planning/research/PITFALLS.md` — Matching false positives, settlement divergence risk
- `.planning/research/STACK.md` — sentence-transformers verified installed, model recommendations

</canonical_refs>

<specifics>
## Specific Ideas

- MarketMetadata has `title`, `description`, `resolution_source` fields — use these for matching
- Polymarket questions are free-form ("Will BTC hit $80k?"), Kalshi uses structured tickers — matching must handle this asymmetry
- Settlement criteria differ between platforms — this is documented as a project risk and should be surfaced in the matching output
- sentence-transformers `all-MiniLM-L6-v2` is lightweight and fast for pairwise similarity

</specifics>

<deferred>
## Deferred Ideas

- Expanding to additional market categories — deferred to after scope assessment
- Automated re-matching on new data — out of scope (one-time historical matching)
- Web-based curation UI — CLI is sufficient for this project

</deferred>

---

*Phase: 02-market-matching*
*Context gathered: 2026-04-02 via auto-generated from YOLO mode*
