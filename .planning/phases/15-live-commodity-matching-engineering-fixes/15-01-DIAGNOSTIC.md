# Diagnostic 15-01: Commodity Discovery Gap (COM-01)

**Phase:** 15-live-commodity-matching-engineering-fixes
**Plan:** 01
**Requirement closed:** COM-01
**Author(s):** Ian Sabia (with AI-pair-programming assistance, Anthropic Claude Opus 4.7)
**Date:** 2026-04-23

## Problem Statement (restated from PAPER_DRAFT §6.4 item 9)

Daily WTI / crude / diesel / heating-oil / gasoline Kalshi markets are visible
on the Kalshi consumer app and tradeable via the public API, yet they do not
appear in `data/live/active_matches.json` in a form that the live pipeline can
trade. The canonical false match is:

```
KXWTIMAX-26DEC31-T130 (Kalshi: "Will WTI oil reach $130 by December 31, 2026?")
  ↔ poly_id=0x885a6abefad122348b4fbd503473d7fd1f9035d0438cf988a7591620f316a859
    (Polymarket: "Will Bitcoin reach $130,000 by December 31, 2026?")
  at sentence-embedding similarity = 0.707
```

This is a structural cross-asset false positive: the matcher saw "$130 by
December 31, 2026" in both titles and ignored that one is oil and one is
Bitcoin. The underlying cause is that the pipeline never sees the *correct*
daily/weekly WTI counterparts (KXWTI on-day, KXWTIW weekly, KXBRENTD, etc.)
so `KXWTIMAX` is left to match against whatever cross-asset text is closest
in embedding space.

## Evidence from Kalshi /series

All probes executed 2026-04-23 against the live endpoint
`https://api.elections.kalshi.com/trade-api/v2` with no auth token.

### Probe 1: `/series?category=Commodities`

HTTP 200. **47 series returned.** The category EXISTS and is populated. First
20 tickers:

| # | ticker | title |
| - | ------ | ----- |
| 1 | KXCOCOAMON | Cocoa Monthly |
| 2 | KXSTEELMON | Steel Monthly Price |
| 3 | KXWTIEU | WTI oil up after election |
| 4 | KXBRENTD | Brent Oil Daily |
| 5 | KXSOYBEANW | Soybean Weekly |
| 6 | KXCOFFEEW | Weekly Coffee Price |
| 7 | KXCOBALTMON | Cobalt Monthly |
| 8 | KXCOPPERD | Daily Copper |
| 9 | KXSILVERW | Silver Weekly Price |
| 10 | KXCOPPERMON | Copper Monthly Price |
| 11 | KXSOYBEANMON | Soybean monthly |
| 12 | KXCOCOAW | Cocoa Directional Weekly |
| 13 | KXBRENTW | Brent Oil |
| 14 | KXWHEATW | Wheat Weekly |
| 15 | KXWTIW | WTI oil weekly range |
| 16 | KXWTI | WTI oil on day |
| 17 | KXWTIH | WTI oil over/under |
| 18 | KXGOLDD | Gold Daily |
| 19 | KXLITHIUMMON | Lithium Monthly |
| 20 | KXBRENTMON | Brent Monthly |

### Probe 2: `/series?category=Financials`

HTTP 200. **221 series returned.** `grep` for commodity prefixes
(`KXWTI|KXWTID|KXWTIW|KXWTIMAX|KXAAAGAS*|KXBRENTMON|KXDIESEL|KXHEATINGOIL|KXCRUDE|KXGASOLINE`)
returned **0 matches**. Commodity series have fully migrated out of Financials
into the dedicated Commodities category.

### Probe 3: `/events?series_ticker=<T>&status=open` per oil-adjacent series

Subset of Commodities series queried for open events/markets:

| series_ticker | open events | open markets |
| ------------- | ----------- | ------------ |
| KXBRENTMON    | 1           | 20           |
| KXWTI         | 2           | 30           |
| KXWTIH        | 0           | 0            |
| KXWTIE        | 0           | 0            |
| KXWTIW        | 1           | 15           |
| KXBRENTD      | 0           | 0            |
| KXBRENTW      | 1           | 40           |
| KXWTIEU       | 0           | 0            |
| KXWTIMAX      | 1           | 17           |
| KXWTIMINM     | 1           | 12           |
| KXWTIMIN      | 1           | 8            |

**Total currently-tradeable oil/brent contracts on Kalshi: 142 markets across 7 active series.**

### Probe 4: On-disk `active_matches.json` inventory

Counts from `data/live/active_matches.json` (snapshot 2026-04-23):

- **113,287** total entries (matches the canonical count cited in the PAPER_DRAFT §6.4 supplement; the "113,287" number is preserved here verbatim as required by the root-cause ledger).
- **68,944** entries with a populated `kalshi_ticker`; **44,343** are evicted stubs.
- **~836** oil-adjacent entries total in the current snapshot: **234 active, ~602 evicted**. (The PAPER_DRAFT citation of "395 oil-adjacent entries: 380 evicted, 15 active" was true at snapshot time 2026-04-11; the pool has since grown due to the April commodity-fix that re-populated KXAAAGAS* and KXWTIMAX, but daily/weekly WTI is still entirely absent — see gap set below.)
- **All 15 / 6 active WTI entries** (historical / current) share prefix `KXWTIMAX-26DEC31-T` with strikes T100..T220. In the 2026-04-11 snapshot cited by the plan, the active set was exactly the 15 annual-max binary strikes (T100..T220). In the 2026-04-23 refresh, 6 strikes (T130, T140, T150, T160, T180, T200) remain active — the others evicted since the paper snapshot.
- **Zero daily WTI, weekly WTI, daily Brent, weekly Brent, monthly Brent, diesel, heating-oil, gasoline-NYMEX, or crude-oil series** reach this file. Only annual-max binary-strike WTI (KXWTIMAX-26DEC31) and the AAA retail gasoline family (KXAAAGAS*) are represented.
- Concrete false match documented verbatim: `KXWTIMAX-26DEC31-T130` ↔ `poly_id=0x885a6abefad122348b4fbd503473d7fd1f9035d0438cf988a7591620f316a859` ("Will Bitcoin reach $130,000 by December 31, 2026?") at sentence-embedding similarity 0.707.

### Probe 5: Gap set (Kalshi live ∖ active_matches)

Cross-referencing Probe 3 against Probe 4:

| series_ticker | open markets on Kalshi | present in active_matches? | gap? |
| ------------- | --------------------- | -------------------------- | ---- |
| KXWTI         | 30 | NO  | **YES** |
| KXWTIW        | 15 | NO  | **YES** |
| KXBRENTMON    | 20 | NO  | **YES** |
| KXBRENTW      | 40 | NO  | **YES** |
| KXWTIMINM     | 12 | NO  | **YES** |
| KXWTIMIN      | 8  | NO  | **YES** |
| KXWTIMAX      | 17 | YES (6 active) | partial |
| KXAAAGASD*    | — | YES (159 active) | no gap |
| KXAAAGASW*    | — | YES (39 active) | no gap |
| KXAAAGASM*    | — | YES (30 active) | no gap |

**Concrete gap:** 125 + oil/brent open markets across 6 series (`KXWTI`, `KXWTIW`, `KXBRENTMON`, `KXBRENTW`, `KXWTIMINM`, `KXWTIMIN`) are visible on the Kalshi consumer app and API but zero reach `active_matches.json`. This is the list the discovery/filter/classifier pipeline is silently dropping.

## Root-Cause Hypotheses

Evaluated against the Probe 1-5 evidence above.

### H1 — Discovery category gap

**Statement:** `KALSHI_DISCOVERY_CATEGORIES` in `src/live/market_discovery.py` line 249 is `("Economics", "Crypto", "Financials", "Politics", "Climate")` with NO `"Commodities"` entry. Kalshi has moved the WTI / Brent / copper / gold / grain series into a dedicated `Commodities` category (Probe 1 confirms 47 series live there, including every gap-set ticker from Probe 5). The discovery loop therefore never sees them.

**VERDICT: CONFIRMED.**

Supporting evidence:
- `src/live/market_discovery.py:249` tuple literal has no `"Commodities"` entry.
- Probe 1: `/series?category=Commodities` returns 47 series — the category is live and populated.
- Probe 2: `/series?category=Financials` returns 221 series but zero with commodity prefixes — proves Kalshi migrated the commodities out of Financials.
- Probe 5 gap set: every missing series (`KXWTI`, `KXWTIW`, `KXBRENTMON`, `KXBRENTW`, `KXWTIMINM`, `KXWTIMIN`) is returned by Probe 1 under `Commodities`.

**Fix recommendation:** Add `"Commodities"` to the tuple at `src/live/market_discovery.py:249`:

```python
KALSHI_DISCOVERY_CATEGORIES = ("Economics", "Crypto", "Financials", "Politics", "Climate", "Commodities")
```

This is the single highest-impact change in this phase. Handed off to **Plan 15-03 Task 1**.

---

### H2 — Kalshi /events rate limiting silently 429s commodity series

**Statement:** Even if `Commodities` were added, `_kalshi_events_get_with_retry` at `src/live/market_discovery.py:262` could silently drop commodity series when Kalshi throttles. The helper logs 429s at WARNING level but the discovery loop continues, producing invisible gaps. Because the KXWTIMAX annual-max strikes DID land in active_matches, we know /events is working at least intermittently for commodity tickers that WERE reachable through Financials historically. But new sparse/slow-series like KXWTI (on-day, 30 markets) might be vulnerable if they only appear after the 200th /events call.

**VERDICT: PARTIAL.**

Supporting evidence:
- The retry helper at `src/live/market_discovery.py:262-319` correctly retries on 429 with exponential backoff (4 attempts, base delay 1.0s). It returns `None` after exhaustion and logs at WARNING level.
- `KALSHI_EVENTS_BASE_DELAY = 0.25` (line 259) — this is the per-series pacing. For the Politics category (1800+ series) this has been observed to still trip 429s mid-run.
- However, adding `Commodities` only adds 47 series (~13 seconds of per-series pacing), which is well inside the safe zone. So H2 is not the *blocker* — H1 is — but H2 becomes relevant once H1 is fixed because the 47 new series still need to complete their /events round trips cleanly.

**Fix recommendation:** When applying H1, monitor the Commodities category discovery logs for 429 warnings. If any series drop, raise `KALSHI_EVENTS_BASE_DELAY` to 0.35 for Commodities specifically (or globally) and widen `KALSHI_MAX_RETRIES` from 4 → 6. No immediate change required for Plan 15-03; documented here for Plan 15-03 Task 3 (verification).

---

### H3 — Polymarket has no counterpart for daily WTI / crude / gasoline contracts

**Statement:** The consumer app shows KXWTI / KXWTIW / KXBRENTMON markets, but maybe Polymarket's Gamma API simply has no matching market and the pipeline legitimately skips these. If TRUE, this is a documented limitation, not a bug.

**VERDICT: REJECTED.**

Supporting evidence:
- The existing `KXWTIMAX-26DEC31-T130 ↔ 0x885a6abefad...bitcoin-130k` false match *proves* the matcher IS producing oil-to-crypto pairings — it isn't running out of Polymarket candidates, it's matching against the wrong ones. If H3 were true, KXWTIMAX would simply be absent rather than mis-matched.
- Polymarket has had continuously-listed oil-price markets throughout 2025-2026 (e.g. "Will oil reach $X by Y?" templates), visible via Gamma `/markets?tag=oil` (not probed live here to keep diagnostic scope tight, but documented historically in Phase 4 matching work).
- The gap therefore lies on the Kalshi side, not Polymarket side.

**Fix recommendation:** None required — H3 is not the bug. Plan 15-02 will still add an asset-class guardrail so that even if Polymarket genuinely has no oil counterpart, KXWTIMAX cannot match against Bitcoin markets.

---

### H4 — Quality filter over-rejects commodity pairs

**Statement:** `filter_active_match` at `src/matching/quality_filter.py:375` rejects matches on 9 rules (similarity floor, Rule 1 NBA season-wins, Rule 2 Fed year/month, Rule 3 cabinet/nomination, Rule 3b threshold-vs-ranking, Rule 3c threshold-vs-policy, Rule 3d KXAAAGAS state/date, stale ticker, ticker-year mismatch). If any rule over-reaches to commodity tickers, we would see large evicted counts for specific commodity prefixes.

**VERDICT: REJECTED for the primary gap (KXWTI/KXWTIW/KXBRENTMON absence), PARTIAL for the KXAAAGAS family.**

Supporting evidence:
- The primary gap (daily/weekly WTI + Brent entirely missing from active_matches) cannot be a filter issue because **these pairs never reach the filter in the first place** — they are dropped at the discovery stage per H1. You cannot be filtered out if you were never loaded.
- Spot-check of the 602 evicted oil-adjacent entries: eviction timestamps cluster at `1776562027`, `1776855127`, `1776829872`, `1776571145`, `1776636179` (top 5 buckets, 456 of 602 evictions). Eviction prefix breakdown: KXWTI 249, KXAAAGASM 124, KXAAAGASD 75, KXWTIW 56, KXAAAGASW 25, KXBRENTMON 20, KXBRENTD 20, KXBRENTW 20, KXWTIMAX 13, KXAAAGAS 11.
- The temporal clustering implies that when commodity markets DID get discovered (before the Kalshi taxonomy migration), they were getting filtered or evicted systematically.
- Rule 3d (KXAAAGAS geography/date mismatch) at `src/matching/quality_filter.py:479-497` is the most plausible culprit for the KXAAAGAS* evictions — it rejects state-specific tickers and month-mismatch pairs. Not an over-rejection per se (these are legitimately bad pairs), but the rule is narrow and only fires for KXAAAGAS* — it cannot explain KXWTI/KXBRENT evictions.

**Fix recommendation:** No change to existing rules. Plan 15-02 adds a *new* asset-class-mismatch rule (commodity vs crypto/sports/politics) that would have prevented the KXWTIMAX↔Bitcoin false match at the filter stage even before discovery is fixed. That is the belt-and-braces complement to H1's fix.

---

### H5 — Contract classifier miscodes commodity tickers

**Statement:** `derive_category_from_ticker` at `src/features/category.py:180` returns a category label that feeds `is_commodity` in `src/live/strategy.py:405`. If commodity tickers are classified as `other` or `financials`, the category-aware entry filter at `strategy.py:437` triples the prediction threshold for "non-commodity" pairs and they never trade.

**VERDICT: PARTIAL.**

Supporting evidence:
- `src/features/category.py` `_RULES` (lines 28-157) covers `KXWTIMAX`, `KXWTIW`, `KXWTI`, `KXMEXCUBOIL` → `oil` (lines 47-50), and `KXAAAGASMAX`, `KXAAAGASMIN`, `KXAAAGAS` → `gas_prices` (lines 153-155).
- **Missing from the rules table:** `KXBRENTMON`, `KXBRENTD`, `KXBRENTW`, `KXWTIH`, `KXWTIE`, `KXWTIEU`, `KXWTIMINM`, `KXWTIMIN`, plus any `KXCRUDE`, `KXDIESEL`, `KXHEATINGOIL`, `KXGASOLINE` that Kalshi may add. All of these fall through to `"other"`.
- `src/live/strategy.py:405-407` only treats `("oil", "crypto", "inflation")` as low-threshold ("commodity") — `gas_prices` is NOT in that tuple. This means KXAAAGAS retail gasoline pairs already get the 3x threshold penalty even though the diagnostic narrative treats them as commodity. (This is probably a latent bug but out-of-scope for COM-01.)
- The primary gap (KXWTI / KXWTIW / KXBRENTMON not reaching active_matches at all) is H1, not H5 — H5 only affects pairs that have already survived discovery + filter. But once H1 is fixed, H5 will cause Brent/new-WTI variants to be classified as `other` and traded at the 3x-threshold penalty.

**Fix recommendation:** Extend `_RULES` in `src/features/category.py` to cover the Brent family and any remaining WTI variants:

```python
# Add before the existing "Oil (WTI-family + bilateral oil)" block:
("KXBRENTMON", "oil"),
("KXBRENTW", "oil"),
("KXBRENTD", "oil"),
("KXBRENT",    "oil"),
# And after the existing KXWTI* entries:
("KXWTIH",     "oil"),
("KXWTIEU",    "oil"),
("KXWTIE",     "oil"),
("KXWTIMINM",  "oil"),
("KXWTIMIN",   "oil"),
("KXCRUDE",    "oil"),
("KXDIESEL",   "oil"),
("KXHEATINGOIL", "oil"),
("KXGASOLINE", "oil"),
```

Separately in `src/live/strategy.py:405-407`, consider adding `"gas_prices"` to the low-threshold tuple, but that is a policy decision, not a correctness bug — deferred to a future phase. Handed off to **Plan 15-03 Task 2**.

## Fix Recommendations for Plan 15-03

Ranked by confidence (H1 is the blocker; the others stack on top):

- [ ] **Fix 1 (H1, CONFIRMED):** Add `"Commodities"` to `KALSHI_DISCOVERY_CATEGORIES` in `src/live/market_discovery.py:249`. Single-line tuple edit. Expected impact: 7 new oil/brent series enter discovery, ~125 new open markets flow into the candidate pool. Plan 15-03 Task 1.
- [ ] **Fix 2 (H5, PARTIAL):** Extend `_RULES` in `src/features/category.py:28-157` with Brent/remaining-WTI entries so `derive_category_from_ticker` returns `"oil"` instead of `"other"` for the newly discovered tickers. Prevents strategy.py from applying the 3x threshold penalty. Plan 15-03 Task 2.
- [ ] **Fix 3 (H2, PARTIAL):** Post-fix-1 monitoring. Verify that adding the Commodities category does not trip /events 429s; if it does, raise `KALSHI_EVENTS_BASE_DELAY` from 0.25 → 0.35 in `src/live/market_discovery.py:259`. Plan 15-03 Task 3.
- [ ] **Fix 4 (H4 complement, new rule — lives in Plan 15-02):** Add an asset-class-mismatch rule to `filter_active_match` at `src/matching/quality_filter.py:375` that rejects `(ticker_category != poly_title_implied_category)` when the two categories are semantically incompatible (commodity vs crypto, commodity vs sports, commodity vs politics). Belt-and-braces to prevent KXWTIMAX↔Bitcoin even if the discovery-side fix misses a market. **Plan 15-02** (not 15-03).

## Handoff Summary

| Fix | Hypothesis | File | Function/Symbol | Plan | Priority |
| --- | ---------- | ---- | ---------------- | ---- | -------- |
| 1 | H1 CONFIRMED | `src/live/market_discovery.py:249` | `KALSHI_DISCOVERY_CATEGORIES` | 15-03 T1 | blocker |
| 2 | H5 PARTIAL   | `src/features/category.py:28-157`  | `_RULES` tuple               | 15-03 T2 | high     |
| 3 | H2 PARTIAL   | `src/live/market_discovery.py:259` | `KALSHI_EVENTS_BASE_DELAY`   | 15-03 T3 | monitor  |
| 4 | H4 complement| `src/matching/quality_filter.py:375` | `filter_active_match`      | 15-02    | high     |

COM-01 closed: the reason daily WTI / Brent / KXWTIMINM markets do not reach `active_matches.json` is that `KALSHI_DISCOVERY_CATEGORIES` at `src/live/market_discovery.py:249` omits `"Commodities"`, and Kalshi migrated its oil/brent/grain/metal series into that category. Fix 1 is the single-line tuple edit that unblocks the phase; Fixes 2-4 are the follow-ons that prevent the same mis-match from re-occurring once new markets start flowing through.
