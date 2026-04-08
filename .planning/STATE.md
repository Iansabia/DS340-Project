---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: Completed 07.3-04-PLAN.md
last_updated: "2026-04-08T22:51:53.000Z"
last_activity: 2026-04-08 -- Completed 07.3-04-PLAN.md (Oracle VM deployment scripts)
progress:
  total_phases: 12
  completed_phases: 9
  total_plans: 34
  completed_plans: 35
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-01)

**Core value:** Empirically answer whether model complexity improves cross-platform prediction market arbitrage detection
**Current focus:** Phase 7.3 -- Adaptive Trading System (Plan 04 of 04 done, Oracle VM deployment)

## Current Position

Phase: 7.3 of 10 (Adaptive Trading System)
Plan: 4 of 4 complete (07.3-04 Oracle VM deployment done)
Status: In Progress
Last activity: 2026-04-08 -- Completed 07.3-04-PLAN.md (Oracle VM deployment scripts)

Progress: [██████████] 100%

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
| Phase 05 P05 | 3min | 2 tasks | 4 files |
| Phase 06 P01 | 2min | 2 tasks | 2 files |
| Phase 06 P02 | 2min | 2 tasks | 2 files |
| Phase 06 P03 | 2min | 2 tasks | 2 files |
| Phase 06 P04 | 3min | 2 tasks | 2 files |
| Phase 06 P05 | 6min | 2 tasks | 5 files |
| Phase 07 P01 | 3min | 1 tasks | 5 files |
| Phase 07 P03 | 3min | 1 tasks | 35 files |
| Phase 07 P02 | 3min | 1 tasks | 11 files |
| Phase 07 P04 | 3min | 2 tasks | 7 files |
| Phase 07 P05 | 2min | 1 tasks | 4 files |
| Phase 07.1 P01 | 6min | 2 tasks | 2 files |
| Phase 07.1 P02 | 7min | 2 tasks | 12 files |
| Phase 07.2 P01 | 8min | 2 tasks | 6 files |
| Phase 07.2 P03 | 3min | 2 tasks | 2 files |
| Phase 07.2 P02 | 4min | 2 tasks | 2 files |
| Phase 07.3 P02 | 5min | 2 tasks | 2 files |
| Phase 07.3 P01 | 8min | 2 tasks | 4 files |
| Phase 07.3 P03 | 12min | 2 tasks | 8 files |
| Phase 07.3 P04 | 3min | 2 tasks | 5 files |

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
- [Phase 05]: MOD-07 (TFT) deferred with documented rationale: param-to-sample ratio ~1.9 (200x above 0.01 threshold), 22-day timeline, GRU/LSTM provide alternative Tier 2 coverage
- [Phase 06]: SpreadTradingEnv reward uses current_position (before update) times spread_change; pairs pre-grouped and pre-scaled at construction for O(1) resets
- [Phase 06]: Episode cycling via modular pair_idx wraps around for multi-epoch SB3 training
- [Phase 06]: AnomalyDetectorAutoencoder is NOT a BasePredictor (signal filter utility); 90/10 chrono val split; threshold_ set automatically at end of fit()
- [Phase 06]: PPO predict() tracks current_position per pair for correct observation augmentation; warm-up stitching reuses GRU pattern
- [Phase 06]: Action-to-prediction mapping {0:0.0, 1:+0.03, 2:-0.03} exposed as class-level dict for direct testing
- [Phase 06]: FilteredTradingEnv as gymnasium.Wrapper (not subclass) for clean separation of reward masking from base env
- [Phase 06]: Predict does NOT use anomaly filter at inference -- filtering only affects training reward signal
- [Phase 06]: PPO-Raw RMSE=0.3189 vs XGBoost RMSE=0.2857: RL adds no value over regression at this dataset scale
- [Phase 06]: PPO-Filtered (anomaly-filtered) performs worse than PPO-Raw: 5% flagging rate too aggressive, reward masking hurts training
- [Phase 06]: Full 8-model cross-tier table confirms complexity-vs-performance thesis: simpler models win at 6.8k samples
- [Phase 07]: Experiment 1 uses horizontal bar chart for RMSE and tier-distinct line styles for equity curves
- [Phase 07]: Short lookback (2 bars/8h) marginally outperforms default (6 bars/24h); longer lookback (18/72h) degrades significantly due to padding and overfitting on small dataset
- [Phase 07]: Single seed=42 for threshold ablation (not 3-seed); PPO discrete predictions {-0.03,0,+0.03} structurally incompatible with thresholds >= 0.05
- [Phase 07]: polymarket_vwap dominates SHAP importance (0.138 vs next 0.016); XGBoost relies heavily on Polymarket price level
- [Phase 07]: At 5-7pp Kalshi fees: Tier 1/2 models remain profitable (XGBoost break-even 15.5pp), naive baselines and PPO-Filtered go negative
- [Phase 07]: Bootstrap 95% CIs confirm XGBoost/GRU/LSTM RMSE CIs overlap -- performance differences not statistically significant at this dataset scale
- [Phase 07]: Reduced training for bootstrap (50 epochs Tier 2, 10k steps Tier 3) since bootstrap resamples predictions not model weights
- [Phase 07.1]: WalkForwardBacktester uses direction-aware P&L (gross = direction * contracts * actual_change) with 3pp entry + 2pp exit fees per trade
- [Phase 07.1]: Backtested Sharpe ratios: LinReg 8.70, XGBoost 8.30, GRU 8.29, LSTM 8.38, PPO-Raw 5.96, PPO-Filtered 1.99; Naive -2.25, Volume -2.26 (correctly negative with fees)
- [Phase 07.1]: PPO uses 50k timesteps for backtest (not 100k) since backtest evaluates prediction quality, not maximal RL convergence
- [Phase 07.2]: Kalshi prices via direct orderbook API (pmxt returns 0 for Kalshi); Polymarket via Gamma API with hex-to-decimal clobTokenId conversion
- [Phase 07.2]: All 144 active pairs are resolved historical markets; --demo flag provides synthetic bars for pipeline testing
- [Phase 07.2]: Retrain pipeline reuses run_baselines.py functions for identical training logic; test set fixed for fair comparison
- [Phase 07.2]: Lazy model training: models trained on first run_cycle(), not in constructor
- [Phase 07.2]: Tier 1 flat features vs Tier 2/3 sequence inference paths for paper trading
- [Phase 07.3]: Exit rules priority: RESOLUTION > STOP_LOSS > TAKE_PROFIT > MOMENTUM > TIME_STOP (safety exits first)
- [Phase 07.3]: SQLite WAL mode + JSONL backup for crash-safe position persistence across cron runs
- [Phase 07.3]: KXPRESNOMD force-API: 40 Dem nominee tickers resolved via Kalshi API close_time; year-only regex relaxed for WC/Liga MX tickers; tier boundaries use strict less-than
- [Phase 07.3]: LR + XGBoost ensemble averaging for entry signal; BasePredictor save/load via pickle for VM deployment; bar interval gating matches cron architecture
- [Phase 07.3]: Oracle VM deployment via idempotent setup_oracle.sh; ~235MB Python footprint (no PyTorch); GitHub Actions as fallback with 30-min Oracle health check

### Roadmap Evolution

- Phase 2.1 inserted after Phase 2: Trade-Based Data Reconstruction (URGENT) — Kalshi candlestick API returns null prices for 76/77 economics markets. Rebuilding both adapters to use raw trade records with 4-hour VWAP candles, forward-fill alignment, and microstructure features.
- Phase 7.1 inserted after Phase 7: Walk-Forward Backtesting (URGENT) — Current Sharpe ratios are inflated (panel data + no transaction costs). Walk-forward portfolio simulator with realistic fees produces the honest, comparable Sharpe and equity curve for the paper.
- Phase 7.2 inserted after Phase 7.1: Live Paper Trading and Data Collection — Deploy models on live prediction market data via pmxt SDK. Paper trade all 8 models, collect 4h bars continuously (growing dataset), retrain weekly to test if more data helps neural/RL models catch up to XGBoost.

### Pending Todos

None yet.

### Blockers/Concerns

- [Critical] TA check-in is April 4 (3 days). Phases 1-4 must complete by then.
- [Risk] Dataset size unknown until Phase 2 matching completes. If <30 pairs, TFT (MOD-07) should be dropped.
- [Risk] Polymarket CLOB and Data API endpoints not yet connectivity-tested.

## Session Continuity

Last session: 2026-04-08T22:51:53.000Z
Stopped at: Completed 07.3-04-PLAN.md
Resume file: None
