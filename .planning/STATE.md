---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: Completed 05-04-PLAN.md
last_updated: "2026-04-06T00:55:25.000Z"
last_activity: "2026-04-06 -- Completed 05-04-PLAN.md (Cross-tier harness: GRU RMSE=0.2896, LSTM RMSE=0.2910, 6-model comparison table)"
progress:
  total_phases: 9
  completed_phases: 4
  total_plans: 15
  completed_plans: 13
  percent: 87
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-01)

**Core value:** Empirically answer whether model complexity improves cross-platform prediction market arbitrage detection
**Current focus:** Phase 5 - Time Series Models

## Current Position

Phase: 5 of 9 (Time Series Models)
Plan: 4 of 5 complete (05-01 Sequence Utilities, 05-02 GRU, 05-03 LSTM, 05-04 Cross-Tier Harness done)
Status: Executing
Last activity: 2026-04-06 -- Completed 05-04-PLAN.md (Cross-tier comparison harness: --tier {1,2,both}, GRU RMSE=0.2896, LSTM RMSE=0.2910, 6-model table)

Progress: [████████░░] 87%

## Performance Metrics

**Velocity:**
- Total plans completed: 0
- Average duration: -
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**
- Last 5 plans: -
- Trend: -

*Updated after each plan completion*
| Phase 01 P01 | 5min | 2 tasks | 14 files |
| Phase 01 P02 | 27min | 2 tasks | 3 files |
| Phase 01 P03 | 21min | 2 tasks | 3 files |
| Phase 02 P01 | 4min | 2 tasks | 9 files |
| Phase 03 P01 | 5min | 2 tasks | 11 files |
| Phase 02.1 P01 | 4min | 2 tasks | 6 files |
| Phase 03 P01 (redo) | 6min | 2 tasks | 10 files |
| Phase 04 P01 | 4min | 2 tasks | 11 files |
| Phase 04 P02 | 4min | 2 tasks | 12 files |
| Phase 05 P01 | 3min | 2 tasks | 2 files |
| Phase 05 P02 | 3min | 2 tasks | 2 files |
| Phase 05 P03 | 3min | 2 tasks | 2 files |
| Phase 05 P04 | 40min | 3 tasks | 15 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Roadmap]: EVAL-01/02 assigned to Phase 4 (evaluation framework built with baselines, not deferred to Phase 7)
- [Roadmap]: EVAL-03/04 assigned to Phase 7 (SHAP and bootstrap CIs require all models trained)
- [Roadmap]: Phases 5 and 6 designed for parallel execution (team split: one person per phase)
- [Phase 01]: Used urllib3 Retry with HTTPAdapter for retry logic rather than custom retry loop
- [Phase 01]: File-based parquet caching in get_or_fetch_candlesticks to avoid redundant API calls
- [Phase 01]: Keyword matching on event title+description instead of Gamma tag_id filtering (tags too sparse)
- [Phase 01]: CLOB prices-history with startTs/endTs as primary source, Data API trades as fallback only
- [Phase 01]: Rate limit 18 req/s for Kalshi (buffer below 20 req/s Basic tier limit)
- [Phase 01]: Lexicographic ISO8601 comparison for historical cutoff routing
- [Phase 02]: Numpy-based cosine similarity instead of torch tensors for MPS compatibility on Apple Silicon
- [Phase 02]: Pre-process $N,NNN patterns before punctuation stripping to preserve number tokens
- [Phase 02]: Module-scoped semantic_matcher fixture for efficient model reuse in tests
- [Phase 03]: String None to NaN conversion needed for Kalshi parquet files (object dtype, not NaN)
- [Phase 03]: Forward-fill limited to 6-hour gap to prevent stale price propagation
- [Phase 03]: Only 1/77 matched pairs produces usable features (Kalshi data quality issue)
- [Phase 02.1]: Standalone trade_fetcher and trade_reconstructor modules (no dependency on adapter classes)
- [Phase 02.1]: Polymarket MAX_TRADE_OFFSET=15000 (matching plan spec, higher than existing adapter's 3000)
- [Phase 02.1]: time_since_last_trade deferred to aligner (requires cross-bar context on unified grid)
- [Phase 03-redo]: Per-pair chronological split (not global cutoff) because pairs have different time ranges
- [Phase 03-redo]: NaN fill with 0 for TimeSeriesDataSet compatibility (pytorch_forecasting requires no NaN in feature cols)
- [Phase 03-redo]: group_id mapping fitted on train set and applied to test for consistent encoding
- [Phase 04]: BasePredictor.evaluate returns merged regression+trading metrics dict for one-line model comparison tables
- [Phase 04]: Directional accuracy excludes y_true==0 samples (no direction to predict)
- [Phase 04]: VolumePredictor uses volume_ratio = max_vol/total_vol in [0.5,1.0]; equal volumes give half reversion, dominance approaches full reversion
- [Phase 04]: Sharpe uses 252 annualization; returns 0.0 if num_trades<2 or return std==0
- [Phase 04]: Models use polymarket_volume column (not poly_volume) to match Phase 3 matched-pairs schema
- [Phase 04]: Linear Regression nearly matches XGBoost on RMSE (0.1759 vs 0.1729) — raises bar for Tier 2/3 complexity justification
- [Phase 04]: run_baselines.py computes spread-change target inline (groupby pair_id shift -1) — Phase 3 has no precomputed target column
- [Phase 04]: XGBoost hyperparameters n_estimators=200 max_depth=4 lr=0.05 — shallower trees reduce overfit on 978-row train set
- [Phase 05]: EarlyStopping compares against best_loss (not previous loss) for min_delta threshold
- [Phase 05]: create_sequences preserves first-occurrence group order via OrderedDict
- [Phase 05]: fit_feature_scaler raises ValueError listing zero-variance column names to surface upstream bugs
- [Phase 05]: GRU uses input dropout (nn.Dropout on features) not recurrent dropout, per CONTEXT.md D6
- [Phase 05]: Warm-up stitching prepends cached train rows per group_id for full lookback windows on test data
- [Phase 05]: Padding by repeating first row when total available rows < lookback, logged to _padded_pairs
- [Phase 05]: LSTM implemented independently from GRU (parallel Wave 2 execution); hidden_size=32 per CONTEXT.md D7
- [Phase 05]: Dropped 4 zero-variance columns from NON_FEATURE_COLUMNS (kalshi_order_flow_imbalance + 3 more Kalshi cols all-zero), yielding 31 features not 34
- [Phase 05]: torch.set_num_threads(1) workaround for PyTorch 2.10.0 Apple Silicon segfault in multi-threaded GRU/LSTM forward pass
- [Phase 05]: GRU RMSE=0.2896+/-0.0024 and LSTM RMSE=0.2910+/-0.0004 over 3 seeds; both competitive with but do not beat XGBoost (0.2857)

### Roadmap Evolution

- Phase 2.1 inserted after Phase 2: Trade-Based Data Reconstruction (URGENT) — Kalshi candlestick API returns null prices for 76/77 economics markets. Rebuilding both adapters to use raw trade records with 4-hour VWAP candles, forward-fill alignment, and microstructure features.

### Pending Todos

None yet.

### Blockers/Concerns

- [Critical] TA check-in is April 4 (3 days). Phases 1-4 must complete by then.
- [Risk] Dataset size unknown until Phase 2 matching completes. If <30 pairs, TFT (MOD-07) should be dropped.
- [Risk] Polymarket CLOB and Data API endpoints not yet connectivity-tested.

## Session Continuity

Last session: 2026-04-06T00:55:25Z
Stopped at: Completed 05-04-PLAN.md
Resume file: None
