# Phase 2: Market Matching - Research

**Researched:** 2026-04-02
**Domain:** NLP-based cross-platform market matching (sentence-transformers, TF-IDF, CLI curation)
**Confidence:** HIGH

## Summary

Phase 2 builds a two-stage matching pipeline that identifies equivalent prediction market contracts across Kalshi and Polymarket. Stage 1 uses keyword/token-based filtering for fast candidate generation (high recall, low precision). Stage 2 uses sentence-transformer semantic similarity scoring for precision ranking. A CLI curation interface allows human review of each candidate pair, and the final output is a JSON registry with confidence scores and settlement criteria documentation.

The technical risk is moderate but the **data risk is critical**: the number of matchable pairs is unknown and determines the viability of all downstream phases. The matching pipeline must be run quickly to inform a scope decision (if <30 pairs, TFT is dropped per ROADMAP).

**Primary recommendation:** Use category-filtered Jaccard token overlap for Stage 1 (fast, eliminates cross-category noise), `all-MiniLM-L6-v2` cosine similarity for Stage 2 scoring, combined into a single 0-1 confidence score. Set auto-accept threshold at 0.85+, manual review for 0.50-0.85, auto-reject below 0.50. Extend MarketMetadata (or fetch separately) to capture settlement criteria fields that Phase 1 adapters currently drop.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **Two-stage pipeline**: keyword-based candidate generation first, then sentence-transformer semantic similarity scoring
- Use `all-MiniLM-L6-v2` from sentence-transformers (already installed at v5.3.0)
- Score each candidate pair with a combined keyword + semantic similarity score
- Build a simple CLI-based review interface (not a web app)
- Present each candidate pair with: Kalshi title, Polymarket title, similarity score, settlement criteria from both
- User accepts, rejects, or flags for further review
- Output: `data/processed/matched_pairs.json`
- Extract and document settlement/resolution criteria from both platforms for each matched pair
- Flag pairs where settlement criteria diverge significantly
- Kalshi market metadata from Phase 1 metadata JSONs in `data/raw/kalshi/`
- Polymarket market metadata from Phase 1 metadata JSONs in `data/raw/polymarket/`
- After matching completes, report total matched pair count
- If < 30 pairs: flag for scope reduction (drop TFT per roadmap)

### Claude's Discretion
- Exact keyword matching algorithm (TF-IDF, Jaccard, simple token overlap)
- Similarity score combination formula
- Auto-accept/manual-review threshold
- CLI interface design details
- How to handle one-to-many matches (one Kalshi event matching multiple Polymarket markets)

### Deferred Ideas (OUT OF SCOPE)
- Expanding to additional market categories -- deferred to after scope assessment
- Automated re-matching on new data -- out of scope (one-time historical matching)
- Web-based curation UI -- CLI is sufficient for this project
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| MATCH-01 | Keyword-based first-pass candidate matching across platforms | Jaccard token overlap with number normalization, category filtering; TF-IDF tested but weak on cross-platform vocabulary divergence |
| MATCH-02 | Sentence-transformer semantic similarity scoring for fuzzy matching | `all-MiniLM-L6-v2` verified working (v5.3.0), `util.cos_sim()` for pairwise scoring, `model.encode()` for batch embedding |
| MATCH-03 | Manual curation interface for reviewing and accepting/rejecting matched pairs | Simple CLI using Python `input()` or argparse; display pair details and prompt accept/reject/flag |
| MATCH-04 | Match confidence scoring per pair | Combined score formula: `alpha * keyword_sim + (1 - alpha) * semantic_sim` with alpha=0.3 recommended |
| MATCH-05 | Settlement criteria comparison documentation for each matched pair | Requires fetching `rules_primary`/`rules_secondary` from Kalshi and `description`/`resolutionSource` from Polymarket -- NOT currently in MarketMetadata |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| sentence-transformers | 5.3.0 | Semantic similarity scoring | Installed, verified working. `all-MiniLM-L6-v2` produces 384-dim embeddings, fast for short text. `util.cos_sim()` handles batch pairwise similarity. |
| scikit-learn | 1.8.0 | TF-IDF vectorization (optional), cosine similarity utilities | `TfidfVectorizer` with `ngram_range=(1,2)` and `cosine_similarity` from `sklearn.metrics.pairwise`. Already installed. |
| pandas | 2.3.3 | Metadata loading, tabular operations | Load `_metadata.json` into DataFrames for filtering and joining. Already installed. |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| json (stdlib) | N/A | Read/write `_metadata.json` and `matched_pairs.json` | All metadata I/O |
| re (stdlib) | N/A | Token extraction, number normalization | Keyword matching stage |
| argparse (stdlib) | N/A | CLI for curation interface and matching pipeline | Entry points |
| dataclasses (stdlib) | N/A | Extend MarketMetadata or create MatchedPair dataclass | Output data structures |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Jaccard token overlap | TF-IDF cosine similarity | TF-IDF tested at 0.19-0.24 for clearly matched pairs due to vocabulary divergence across platforms. Jaccard with number normalization is simpler and more interpretable for Stage 1 candidate filtering. |
| `all-MiniLM-L6-v2` | `all-mpnet-base-v2` | mpnet produces 768-dim embeddings, 5x slower inference. For short contract titles (10-30 tokens), MiniLM quality is sufficient. Upgrade only if matching quality is poor. |
| Simple `input()` CLI | Click library | Click adds dependency complexity. The review interface is a one-time tool for reviewing dozens of pairs. `input()` with formatted output is sufficient. |
| Custom Jaccard | rapidfuzz | rapidfuzz is optimized for fuzzy string matching but not installed and adds a dependency. Simple Jaccard with domain-specific normalization is adequate here. |

**Installation:**
```bash
# No new packages needed -- all dependencies already installed
```

## Architecture Patterns

### Recommended Project Structure
```
src/matching/
    __init__.py
    keyword_matcher.py     # Stage 1: token-overlap candidate generation
    semantic_matcher.py    # Stage 2: sentence-transformer similarity scoring
    scorer.py              # Combined confidence scoring
    curator.py             # CLI review interface
    registry.py            # Read/write matched_pairs.json
    metadata_enricher.py   # Fetch settlement criteria from APIs
```

### Pattern 1: Two-Stage Candidate-then-Score Pipeline
**What:** Stage 1 generates candidates with high recall using cheap keyword matching. Stage 2 scores candidates with expensive semantic similarity. This avoids computing O(N*M) semantic embeddings for all Kalshi x Polymarket pairs.
**When to use:** Always for this pipeline.
**Example:**
```python
# Stage 1: Generate candidates via token overlap
# Filter by compatible category first to reduce search space
CATEGORY_MAP = {
    "Economics": "finance",
    "Financials": "finance",
    "Crypto": "crypto",
}

def generate_candidates(
    kalshi_markets: list[dict],
    poly_markets: list[dict],
    min_keyword_score: float = 0.1,
) -> list[tuple[dict, dict, float]]:
    """Generate candidate pairs using token overlap within compatible categories."""
    candidates = []
    for km in kalshi_markets:
        kalshi_cat = CATEGORY_MAP.get(km["category"])
        for pm in poly_markets:
            if pm["category"] != kalshi_cat:
                continue
            score = jaccard_token_similarity(km["question"], pm["question"])
            if score >= min_keyword_score:
                candidates.append((km, pm, score))
    return candidates
```

### Pattern 2: Batch Encoding with Semantic Similarity Matrix
**What:** Encode all candidate questions in a single batch call, then compute similarity matrix. Avoids per-pair model inference overhead.
**When to use:** Stage 2 scoring.
**Example:**
```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

# Batch encode all unique questions
kalshi_questions = [c[0]["question"] for c in candidates]
poly_questions = [c[1]["question"] for c in candidates]

# Use convert_to_numpy=True to avoid MPS tensor issues
kalshi_embeddings = model.encode(kalshi_questions, convert_to_numpy=True, batch_size=32)
poly_embeddings = model.encode(poly_questions, convert_to_numpy=True, batch_size=32)

# Compute pairwise cosine similarity
# For candidate pairs, use element-wise (not full matrix)
from sentence_transformers.util import pairwise_cos_sim
import torch

similarities = pairwise_cos_sim(
    torch.tensor(kalshi_embeddings),
    torch.tensor(poly_embeddings),
)  # shape: (num_candidates,)
```

### Pattern 3: Registry as Single Source of Truth
**What:** `matched_pairs.json` is the only output. All downstream phases read from it. Only the matching pipeline writes to it.
**When to use:** After curation is complete.
**Example (output schema):**
```json
[
    {
        "pair_id": "btc-80k-dec2025",
        "kalshi_market_id": "KXBTC-25DEC31-T80000",
        "polymarket_market_id": "0xabc123...",
        "kalshi_question": "Will Bitcoin exceed $80,000 by Dec 31?",
        "polymarket_question": "Bitcoin above 80k by end of 2025",
        "category": "crypto",
        "confidence_score": 0.82,
        "keyword_score": 0.33,
        "semantic_score": 0.72,
        "settlement_aligned": true,
        "settlement_notes": "Both resolve on BTC spot price. Kalshi: CoinDesk, Polymarket: Coinbase. 5h timezone offset.",
        "kalshi_settlement": "Resolves based on CoinDesk BTC Index at midnight ET on Dec 31, 2025",
        "polymarket_settlement": "Resolves Yes if Coinbase BTC/USD >= $80,000 at any point before Jan 1, 2026 UTC",
        "kalshi_resolution_date": "2025-12-31T23:59:59Z",
        "polymarket_resolution_date": "2026-01-01T00:00:00Z",
        "status": "accepted",
        "review_notes": ""
    }
]
```

### Anti-Patterns to Avoid
- **Computing full N*M similarity matrix:** If there are 500 Kalshi markets and 300 Polymarket markets, that is 150K pairs to embed and compare. Category filtering + keyword pre-filtering reduces this to hundreds, making semantic scoring tractable.
- **Matching on ticker/ID instead of question text:** Kalshi tickers (e.g., "KXBTC-25DEC31-T50000") encode structured info. Polymarket uses conditionIds ("0xabc123"). There is zero overlap in identifier format. Always match on human-readable question text.
- **Trusting high semantic similarity without settlement verification:** A cosine similarity of 0.90 between "Will BTC exceed $80K by Dec 31?" and "Bitcoin above $80K by year end" does NOT guarantee the contracts are equivalent. Settlement source (CoinDesk vs Coinbase), settlement type (any-time touch vs close-of-day), and exact date/time all matter.
- **Auto-accepting without manual review:** Per testing, semantically similar but non-equivalent pairs (e.g., Bitcoin $80K vs Ethereum $5K) can score 0.66. With category filtering this is partially mitigated, but within the same category, "BTC $80K by Dec" vs "BTC $100K by Dec" would score very high despite being different contracts.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Sentence embedding | Custom word2vec averaging | `SentenceTransformer("all-MiniLM-L6-v2").encode()` | Pre-trained on 1B+ sentence pairs; handles synonyms, paraphrases, abbreviations that custom approaches miss |
| Cosine similarity | Manual dot product / norm | `sentence_transformers.util.cos_sim()` or `sklearn.metrics.pairwise.cosine_similarity()` | Handles batching, numerical stability, edge cases |
| Token extraction / normalization | Regex-only approach | Combine regex for numbers with a curated synonym map for domain terms | Prediction markets use domain-specific abbreviations ("btc"="bitcoin", "80k"="80000", "fed"="federal reserve") |

**Key insight:** The hardest part of this phase is NOT the NLP -- it is the domain-specific normalization (number formats, abbreviations, date formats) and the settlement criteria verification, which no library can automate.

## Common Pitfalls

### Pitfall 1: MarketMetadata Missing Settlement Criteria Fields
**What goes wrong:** The current `MarketMetadata` dataclass has only `market_id`, `question`, `category`, `platform`, `resolution_date`, `result`, `outcomes`. It does NOT have `description`, `resolution_source`, `rules_primary`, or `rules_secondary`. The ingestion scripts' `_metadata.json` files also lack these fields. MATCH-05 requires settlement criteria documentation for each pair.
**Why it happens:** Phase 1 was focused on price data ingestion. Settlement metadata was not a requirement.
**How to avoid:** The matching pipeline must independently fetch settlement criteria. Two approaches:
1. **Re-query APIs during matching** -- Use `ResilientClient` to fetch Kalshi `/markets/{ticker}` for `rules_primary`/`rules_secondary` and Polymarket Gamma API `/markets?id={conditionId}` for `description`/`resolutionSource`. This avoids modifying Phase 1 code.
2. **Extend MarketMetadata and re-run ingestion** -- Add fields and re-run the ingestion scripts. Slower, touches Phase 1.
**Recommendation:** Option 1 (re-query during matching). The matching pipeline only needs settlement info for candidate pairs (dozens, not thousands), so the API cost is minimal.
**Warning signs:** If `matched_pairs.json` entries have empty `kalshi_settlement` or `polymarket_settlement` fields, the enrichment step was skipped.

### Pitfall 2: Number Format Divergence Between Platforms
**What goes wrong:** Kalshi uses exact numbers in titles ("$80,000", "3%", "$50,000"). Polymarket uses informal notation ("80k", "$100K", "3 percent"). Simple token overlap fails because "80000" != "80k".
**Why it happens:** Different platforms have different title formatting conventions.
**How to avoid:** Implement number normalization before keyword matching:
- Convert "80k"/"80K" -> "80000"
- Remove "$", ",", "%" symbols
- Normalize percentage representations ("3%", "3 percent", "3pct" -> "3")
- Normalize date formats ("Dec 31" / "December 31" / "12/31" -> canonical)
**Warning signs:** Keyword stage produces 0 candidates for known matching pairs.

### Pitfall 3: MPS Tensor Compatibility
**What goes wrong:** `sentence_transformers` on Apple Silicon (MPS) produces tensors on the MPS device. Calling `.numpy()` on MPS tensors raises `TypeError: can't convert mps:0 device type tensor to numpy`.
**Why it happens:** The model auto-detects MPS and encodes on GPU. But numpy conversion requires CPU tensors.
**How to avoid:** Always pass `convert_to_numpy=True` to `model.encode()`. Or call `.cpu()` before `.numpy()` on any tensor from `util.cos_sim()`.
**Warning signs:** `TypeError` mentioning "mps:0 device type tensor".

### Pitfall 4: One-to-Many Matches
**What goes wrong:** One Kalshi event (e.g., "Will BTC hit $50K by Dec?") may match multiple Polymarket markets (different Polymarket events asking the same question with slightly different dates or thresholds). Or vice versa. If not handled, duplicate pairs inflate the dataset.
**Why it happens:** Platforms sometimes create multiple versions of similar markets.
**How to avoid:** After scoring all candidates, group by Kalshi market_id and Polymarket market_id. For each group, keep only the highest-confidence match. Flag any market that appears in multiple pairs for manual review.
**Warning signs:** The same `market_id` appearing in multiple entries in `matched_pairs.json`.

### Pitfall 5: Category Mapping Mismatch
**What goes wrong:** Kalshi uses "Economics", "Financials", "Crypto". Polymarket uses "finance", "crypto". If category filtering is strict, "Economics" vs "finance" would be treated as non-matching, eliminating valid candidates.
**Why it happens:** Platform-specific category naming.
**How to avoid:** Implement explicit category mapping:
```python
CATEGORY_MAP = {
    "Economics": "finance",
    "Financials": "finance",
    "Crypto": "crypto",
}
```
Normalize both sides to a common category before filtering.

### Pitfall 6: Empty Metadata Files
**What goes wrong:** The ingestion scripts have not been run yet (Phase 1 data directories are currently empty -- only `.gitkeep` files exist). The matching pipeline depends on `_metadata.json` files that do not yet exist.
**Why it happens:** Phase 1 code is written but not executed.
**How to avoid:** The matching pipeline should validate that metadata files exist and are non-empty before proceeding. Log a clear error if files are missing. The first task should run the ingestion scripts to produce the metadata.

## Code Examples

Verified patterns from testing in the project environment:

### Sentence-Transformer Encoding and Similarity
```python
# Verified working on project environment (2026-04-02)
# sentence-transformers 5.3.0, Python 3.12.12, macOS MPS

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

# CRITICAL: use convert_to_numpy=True on MPS systems
embeddings = model.encode(
    ["Will Bitcoin exceed $80,000 by Dec 31?", "Bitcoin above 80k by end of 2025"],
    convert_to_numpy=True,
    batch_size=32,
)
# embeddings shape: (2, 384), dtype: float32

# Pairwise similarity (for candidate pairs)
sims = util.cos_sim(embeddings[0:1], embeddings[1:2])
# Must call .cpu() before .numpy() if working with tensors
score = sims.cpu().numpy()[0][0]  # 0.715 for this pair

# Full matrix similarity (for all-vs-all)
all_sims = util.cos_sim(embeddings, embeddings)
# shape: (2, 2) tensor
```

### Token Overlap with Number Normalization
```python
import re

# Domain-specific synonyms for prediction markets
SYNONYMS = {
    "btc": "bitcoin",
    "eth": "ethereum",
    "sol": "solana",
    "fed": "federal_reserve",
    "fomc": "federal_reserve",
    "cpi": "consumer_price_index",
    "gdp": "gross_domestic_product",
}

def normalize_number(s: str) -> str:
    """Convert '80k' -> '80000', '1.5m' -> '1500000'."""
    s = s.lower().replace(",", "").replace("$", "")
    match = re.match(r"^(\d+\.?\d*)(k|m|b)?$", s)
    if match:
        num = float(match.group(1))
        suffix = match.group(2)
        if suffix == "k":
            num *= 1000
        elif suffix == "m":
            num *= 1_000_000
        elif suffix == "b":
            num *= 1_000_000_000
        return str(int(num)) if num == int(num) else str(num)
    return s

def extract_key_tokens(text: str) -> set[str]:
    """Extract and normalize meaningful tokens from a market question."""
    text = text.lower()
    # Remove punctuation except numbers/letters
    text = re.sub(r"[^\w\s.]", " ", text)
    # Tokenize
    raw_tokens = text.split()
    # Stop words common in market questions
    stop_words = {
        "will", "the", "be", "by", "in", "at", "for", "of", "a", "an",
        "to", "on", "above", "below", "over", "under", "exceed", "reach",
        "price", "this", "market", "resolves", "yes", "no", "if",
    }
    tokens = set()
    for t in raw_tokens:
        if t in stop_words:
            continue
        # Apply synonym mapping
        t = SYNONYMS.get(t, t)
        # Normalize numbers
        t = normalize_number(t)
        if len(t) >= 2:  # Skip very short tokens
            tokens.add(t)
    return tokens

def jaccard_similarity(text1: str, text2: str) -> float:
    """Compute Jaccard similarity on normalized tokens."""
    tokens1 = extract_key_tokens(text1)
    tokens2 = extract_key_tokens(text2)
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    return len(intersection) / len(union)
```

### Combined Confidence Score
```python
def compute_confidence(
    keyword_score: float,
    semantic_score: float,
    alpha: float = 0.3,
) -> float:
    """Combine keyword and semantic scores into a single 0-1 confidence.

    Alpha controls keyword weight. Default 0.3 means 30% keyword, 70% semantic.
    Semantic is weighted higher because it handles paraphrases and synonyms.
    """
    return alpha * keyword_score + (1 - alpha) * semantic_score
```

### Loading Phase 1 Metadata
```python
import json
from pathlib import Path

def load_platform_metadata(data_dir: Path) -> list[dict]:
    """Load market metadata from Phase 1 _metadata.json files."""
    metadata_path = data_dir / "_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}. "
            "Run Phase 1 ingestion first."
        )
    with open(metadata_path) as f:
        return json.load(f)

kalshi_markets = load_platform_metadata(Path("data/raw/kalshi"))
poly_markets = load_platform_metadata(Path("data/raw/polymarket"))
```

### CLI Review Interface Pattern
```python
import json

def review_candidates(
    candidates: list[dict],
    output_path: Path,
) -> list[dict]:
    """Interactive CLI for reviewing matched pair candidates."""
    results = []
    total = len(candidates)

    for i, candidate in enumerate(candidates, 1):
        print(f"\n{'='*60}")
        print(f"Candidate {i}/{total} (confidence: {candidate['confidence_score']:.3f})")
        print(f"{'='*60}")
        print(f"  Kalshi:      {candidate['kalshi_question']}")
        print(f"  Polymarket:  {candidate['polymarket_question']}")
        print(f"  Category:    {candidate['category']}")
        print(f"  Keyword:     {candidate['keyword_score']:.3f}")
        print(f"  Semantic:    {candidate['semantic_score']:.3f}")
        print(f"  K-Resolve:   {candidate['kalshi_resolution_date']}")
        print(f"  P-Resolve:   {candidate['polymarket_resolution_date']}")
        if candidate.get("kalshi_settlement"):
            print(f"  K-Settlement: {candidate['kalshi_settlement']}")
        if candidate.get("polymarket_settlement"):
            print(f"  P-Settlement: {candidate['polymarket_settlement']}")
        print()

        while True:
            choice = input("  [a]ccept / [r]eject / [f]lag / [s]kip / [q]uit: ").lower().strip()
            if choice in ("a", "r", "f", "s", "q"):
                break
            print("  Invalid choice.")

        if choice == "q":
            break

        candidate["status"] = {
            "a": "accepted",
            "r": "rejected",
            "f": "flagged",
            "s": "skipped",
        }[choice]

        if choice in ("a", "f"):
            notes = input("  Notes (optional, Enter to skip): ").strip()
            candidate["review_notes"] = notes

        results.append(candidate)

        # Auto-save after each decision
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    return results
```

## Empirical Findings from Testing

The following scores were obtained by testing on realistic prediction market question pairs in the project environment:

### Semantic Similarity Score Ranges (all-MiniLM-L6-v2)
| Pair Type | Score Range | Notes |
|-----------|------------|-------|
| Exact semantic match (same question, different wording) | 0.85 - 0.95 | "Will BTC be above 80000 on December 31, 2025?" vs "Will Bitcoin exceed 80000 by December 2025?" |
| Same topic, similar structure | 0.65 - 0.85 | "Will Bitcoin exceed 80000 by December 2025?" vs "Bitcoin above 80k by end of 2025" |
| Same category, different topic | 0.43 - 0.66 | Bitcoin $80K vs Ethereum $5K: 0.66 (dangerously high!) |
| Different category | 0.20 - 0.43 | Bitcoin price vs weather question |
| Completely unrelated | 0.15 - 0.30 | Random text |

### Keyword Jaccard Score Ranges (with normalization)
| Pair Type | Score Range | Notes |
|-----------|------------|-------|
| High overlap (same numbers, same entities) | 0.40 - 0.70 | "CPI above 3% in Feb 2026" |
| Moderate overlap (entity match, different notation) | 0.20 - 0.40 | "Bitcoin 80000" vs "Bitcoin 80k" (need number normalization!) |
| Different entities, same category | 0.00 - 0.10 | Bitcoin vs Ethereum: 0.00 (correctly separated) |

### Recommended Thresholds
| Threshold | Action | Rationale |
|-----------|--------|-----------|
| Combined >= 0.85 | Auto-accept (still shown in review) | Very high confidence, likely exact matches |
| Combined 0.50 - 0.85 | Manual review required | Gray zone where false positives are likely |
| Combined < 0.50 | Auto-reject | Too dissimilar to be the same contract |
| Keyword = 0.0 | Skip semantic scoring | Zero token overlap means fundamentally different questions |

**Critical finding:** Semantic similarity alone cannot distinguish "Bitcoin $80K" from "Ethereum $5K" within the same category (score 0.66). The keyword stage is essential for filtering these out, because Jaccard correctly scores them at 0.00 (no shared key tokens). The two-stage approach is not just an optimization -- it is necessary for correctness.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual string matching | Two-stage keyword + semantic | Standard since 2022+ | Handles paraphrases, abbreviations, cross-platform naming |
| sentence-transformers v2-3 | sentence-transformers 5.3.0 | 2025 | `model.similarity()` method added, `SimilarityFunction` enum, improved batch encoding |
| `model.encode() + manual cosine` | `util.cos_sim()` / `model.similarity()` | sentence-transformers 3.0+ | Built-in similarity functions handle device placement and batching |

**Deprecated/outdated:**
- `sentence_transformers.util.pytorch_cos_sim` -- renamed to `cos_sim` in newer versions (both still work)
- Manual cosine similarity via `numpy.dot` -- use `util.cos_sim()` instead for correctness and device handling

## Data Gap: Settlement Criteria Not in Current Metadata

This is the most important finding for the planner:

**Current state of MarketMetadata:**
```python
@dataclass
class MarketMetadata:
    market_id: str      # Kalshi ticker or Polymarket conditionId
    question: str       # Human-readable question
    category: str       # Platform-specific category
    platform: str       # "kalshi" or "polymarket"
    resolution_date: str  # ISO8601 UTC
    result: str | None  # "yes", "no", or None
    outcomes: list[str] # Default: ["Yes", "No"]
```

**Fields available in APIs but NOT captured:**

| Field | Kalshi Source | Polymarket Source | Needed For |
|-------|-------------|-------------------|------------|
| Settlement rules | `rules_primary`, `rules_secondary` (from `/markets/{ticker}`) | `description` (from Gamma `/markets`) | MATCH-05 |
| Resolution source | Not a direct field; implied in `rules_primary` | `resolutionSource` (from Gamma `/markets`) | MATCH-05 |
| Event-level description | Via `/events/{event_ticker}` | Event-level `description` in Gamma response | Better matching context |

**Recommendation:** The matching pipeline should fetch these fields on-demand for candidate pairs only (not all markets). Create a `metadata_enricher.py` module that:
1. Takes a list of candidate `(kalshi_id, polymarket_id)` pairs
2. Fetches `rules_primary`/`rules_secondary` from Kalshi API for each Kalshi market
3. Fetches `description`/`resolutionSource` from Polymarket Gamma API for each Polymarket market
4. Returns enriched records for the curation interface

This avoids modifying Phase 1 code and keeps the API calls minimal (only for candidates, not all markets).

## Open Questions

1. **How many markets will each platform have?**
   - What we know: Kalshi categories are Economics, Crypto, Financials. Polymarket keywords filter for crypto and finance terms.
   - What's unclear: The actual count of resolved markets in these categories. Phase 1 ingestion has not been run yet.
   - Recommendation: Run ingestion scripts immediately to get market counts before spending time on the matching pipeline. If either platform has <50 markets, the matching pool is very constrained.

2. **Optimal alpha for combined score formula?**
   - What we know: Keyword matching is good at rejection (Bitcoin != Ethereum), semantic is good at matching (paraphrases). Default alpha=0.3 weights semantic higher.
   - What's unclear: The exact distribution of scores on real data.
   - Recommendation: Start with alpha=0.3, log both component scores, adjust based on observed false positive/negative rates during manual review.

3. **Date alignment requirement for matching?**
   - What we know: Matched pairs should have similar resolution dates. A "BTC $80K by Dec 2025" on Kalshi should match "BTC above 80K by end of 2025" on Polymarket, but NOT "BTC above 80K by June 2026".
   - What's unclear: How strict the date matching should be (exact date? same month? same quarter?).
   - Recommendation: Add resolution date proximity as a third scoring component or as a hard filter. Reject pairs where resolution dates differ by more than 7 days.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 9.0.2 |
| Config file | none -- see Wave 0 |
| Quick run command | `.venv/bin/python -m pytest tests/matching/ -x -q` |
| Full suite command | `.venv/bin/python -m pytest tests/ -x -q` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| MATCH-01 | Keyword matching produces candidates from compatible categories with token overlap | unit | `.venv/bin/python -m pytest tests/matching/test_keyword_matcher.py -x` | No -- Wave 0 |
| MATCH-02 | Semantic similarity scoring produces 0-1 scores for candidate pairs | unit | `.venv/bin/python -m pytest tests/matching/test_semantic_matcher.py -x` | No -- Wave 0 |
| MATCH-03 | Curation interface reads candidates, applies user decisions, writes output | unit | `.venv/bin/python -m pytest tests/matching/test_curator.py -x` | No -- Wave 0 |
| MATCH-04 | Combined confidence score computed from keyword + semantic components | unit | `.venv/bin/python -m pytest tests/matching/test_scorer.py -x` | No -- Wave 0 |
| MATCH-05 | Settlement criteria fetched and documented for each accepted pair | integration | `.venv/bin/python -m pytest tests/matching/test_metadata_enricher.py -x` | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `.venv/bin/python -m pytest tests/matching/ -x -q`
- **Per wave merge:** `.venv/bin/python -m pytest tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/matching/__init__.py` -- package init
- [ ] `tests/matching/conftest.py` -- shared fixtures (sample MarketMetadata dicts, mock API responses for settlement enrichment)
- [ ] `tests/matching/test_keyword_matcher.py` -- covers MATCH-01
- [ ] `tests/matching/test_semantic_matcher.py` -- covers MATCH-02
- [ ] `tests/matching/test_curator.py` -- covers MATCH-03
- [ ] `tests/matching/test_scorer.py` -- covers MATCH-04
- [ ] `tests/matching/test_metadata_enricher.py` -- covers MATCH-05
- [ ] `pytest.ini` or `pyproject.toml [tool.pytest.ini_options]` -- test configuration (pytest already installed at 9.0.2)

## Sources

### Primary (HIGH confidence)
- sentence-transformers v5.3.0 -- verified installed and working in project `.venv/`, encode/cos_sim API tested with real prediction market questions
- [sentence-transformers/all-MiniLM-L6-v2 on HuggingFace](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) -- model documentation, 384-dim embeddings, contrastive training
- [Semantic Textual Similarity docs](https://www.sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html) -- `model.similarity()`, `model.encode()` API
- [sentence-transformers util reference](https://sbert.net/docs/package_reference/util.html) -- `cos_sim()`, `semantic_search()`, `pairwise_cos_sim()` function signatures
- scikit-learn 1.8.0 `TfidfVectorizer` -- [official docs](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- Project source code: `src/data/schemas.py`, `src/data/kalshi.py`, `src/data/polymarket.py`, `src/data/ingest_kalshi.py`, `src/data/ingest_polymarket.py` -- verified field availability
- [Kalshi API Get Market endpoint](https://docs.kalshi.com/api-reference/market/get-market) -- confirmed `rules_primary`, `rules_secondary` fields exist

### Secondary (MEDIUM confidence)
- Empirical score ranges from testing in project environment -- representative but not exhaustive; real market titles may differ
- [Kalshi ticker format](https://docs.kalshi.com/websockets/market-ticker) -- structure confirmed (SERIES-DATE-THRESHOLD)
- [sbert.net quickstart](https://sbert.net/docs/quickstart.html) -- encoding patterns

### Tertiary (LOW confidence)
- Optimal threshold values (0.85/0.50) -- based on limited testing with synthetic examples; may need adjustment after seeing real data distribution

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries verified installed, APIs tested in project environment
- Architecture: HIGH -- two-stage pipeline is well-established pattern, code examples tested
- Pitfalls: HIGH -- MPS tensor issue confirmed by reproducing it, settlement gap confirmed by reading source code, number format divergence confirmed by testing
- Thresholds: LOW -- only tested on synthetic examples, not real market data

**Research date:** 2026-04-02
**Valid until:** 2026-04-16 (stable libraries, project-specific findings)
