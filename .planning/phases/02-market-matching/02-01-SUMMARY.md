---
phase: 02-market-matching
plan: 01
subsystem: matching
tags: [jaccard, sentence-transformers, cosine-similarity, nlp, all-MiniLM-L6-v2]

# Dependency graph
requires:
  - phase: 01-data-ingestion
    provides: "_metadata.json files with market_id, question, category, platform fields"
provides:
  - "keyword_matcher: Jaccard token overlap with number normalization, synonyms, category filtering"
  - "semantic_matcher: all-MiniLM-L6-v2 cosine similarity scoring"
  - "scorer: combined confidence scoring (30% keyword + 70% semantic) with ranked output"
  - "test fixtures: sample Kalshi and Polymarket market metadata dicts"
affects: [02-market-matching, 03-feature-engineering]

# Tech tracking
tech-stack:
  added: [sentence-transformers, all-MiniLM-L6-v2]
  patterns: [two-stage matching (keyword filter then semantic scoring), weighted confidence scoring, module-scoped test fixtures for model reuse]

key-files:
  created:
    - src/matching/__init__.py
    - src/matching/keyword_matcher.py
    - src/matching/semantic_matcher.py
    - src/matching/scorer.py
    - tests/matching/__init__.py
    - tests/matching/conftest.py
    - tests/matching/test_keyword_matcher.py
    - tests/matching/test_semantic_matcher.py
    - tests/matching/test_scorer.py
  modified: []

key-decisions:
  - "Numpy-based cosine similarity instead of torch tensors for MPS compatibility on Apple Silicon"
  - "Pre-process currency/comma patterns before general punctuation stripping to preserve $80,000 as 80000"
  - "Module-scoped semantic_matcher fixture to avoid reloading transformer model per test"

patterns-established:
  - "Two-stage matching: keyword filter (fast, O(N*M)) then semantic scoring (slow, model inference) on filtered set"
  - "CATEGORY_MAP normalizes platform-specific category names to shared namespace"
  - "score_candidates returns list of dicts with all scoring fields for downstream consumption"

requirements-completed: [MATCH-01, MATCH-02, MATCH-04]

# Metrics
duration: 4min
completed: 2026-04-02
---

# Phase 2 Plan 1: Matching Engine Core Summary

**Two-stage market matching engine with Jaccard keyword overlap, all-MiniLM-L6-v2 semantic similarity, and 30/70 weighted confidence scoring**

## Performance

- **Duration:** 4 min
- **Started:** 2026-04-02T22:59:50Z
- **Completed:** 2026-04-02T23:04:06Z
- **Tasks:** 2
- **Files modified:** 9

## Accomplishments
- Keyword matcher with number normalization (80k->80000, $50,000->50000), synonym mapping (btc->bitcoin, cpi->consumer_price_index), and category-aware candidate generation
- Semantic matcher using all-MiniLM-L6-v2 producing 384-dim embeddings with pairwise cosine similarity
- Combined scorer merging keyword (30%) and semantic (70%) scores into ranked candidate list
- 46 matching tests plus full regression suite (91 total) all passing

## Task Commits

Each task was committed atomically:

1. **Task 1: Test infrastructure and keyword matcher with TDD** - `7b58bb91` (feat)
2. **Task 2: Semantic matcher and combined scorer with TDD** - `59f304b4` (feat)

## Files Created/Modified
- `src/matching/__init__.py` - Package init with docstring
- `src/matching/keyword_matcher.py` - Jaccard overlap with number normalization, synonyms, category filtering
- `src/matching/semantic_matcher.py` - SemanticMatcher class wrapping all-MiniLM-L6-v2 for batch encoding and pairwise cosine similarity
- `src/matching/scorer.py` - Combined confidence scoring (alpha=0.3) and score_and_rank_candidates pipeline
- `tests/matching/__init__.py` - Test package init
- `tests/matching/conftest.py` - Shared fixtures (sample_kalshi_markets, sample_poly_markets, semantic_matcher)
- `tests/matching/test_keyword_matcher.py` - 31 tests for normalization, tokenization, Jaccard similarity, candidate generation
- `tests/matching/test_semantic_matcher.py` - 5 tests for encoding shape, score ranges, similar/dissimilar pairs
- `tests/matching/test_scorer.py` - 10 tests for confidence formula, ranking, field validation, edge cases

## Decisions Made
- Used numpy-based cosine similarity (dot product of normalized vectors) instead of torch tensors to avoid MPS device issues on Apple Silicon. The `convert_to_numpy=True` flag ensures embeddings stay on CPU as numpy arrays.
- Pre-process `$80,000` patterns via regex substitution before general punctuation stripping, so comma-separated numbers aren't split into separate tokens.
- Module-scoped `semantic_matcher` fixture in conftest.py loads the transformer model once per test module rather than per test function, reducing test suite time from ~30s to ~7s.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed $80,000 number normalization in extract_key_tokens**
- **Found during:** Task 1 (GREEN phase)
- **Issue:** General punctuation stripping (`re.sub(r"[^\w\s.]", " ", text)`) converted `$80,000` to `80 000` (two separate tokens) before normalize_number could process it
- **Fix:** Added pre-processing step that collapses `$N,NNN` patterns into plain numbers before the general punctuation strip
- **Files modified:** src/matching/keyword_matcher.py
- **Verification:** test_bitcoin_question passes with "80000" in token set
- **Committed in:** 7b58bb91 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Fix was necessary for correct number normalization. No scope creep.

## Issues Encountered
None beyond the auto-fixed deviation above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Matching engine core ready for Plan 02 (curation pipeline and match output)
- score_and_rank_candidates provides the full pipeline: metadata dicts in, scored/ranked candidate list out
- Test fixtures available for downstream test modules

## Self-Check: PASSED

All 9 files verified present. Both task commits (7b58bb91, 59f304b4) verified in git log.

---
*Phase: 02-market-matching*
*Completed: 2026-04-02*
