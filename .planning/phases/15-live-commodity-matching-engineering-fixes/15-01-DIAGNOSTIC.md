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
