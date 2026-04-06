# Phase 5 Deferrals

**Phase:** 05-time-series-models
**Created:** 2026-04-05

## MOD-07: Temporal Fusion Transformer (TFT) -- DEFERRED

**Status:** Deferred per roadmap success criterion #3's explicit deferral clause.

**Rationale:**

TFT is deferred from Phase 5 per the roadmap's explicit deferral clause in Phase 5 success criterion #3, which permits deferral "with documented rationale if dataset is too small or timeline is tight." Both conditions apply.

### Dataset-size argument (param-to-sample ratio)

At TFT's minimum reasonable `hidden_size=16`, the model carries approximately 10,000 parameters. Our training set has ~5,000-6,000 windows (6,085 at lookback=6 per CONTEXT.md D1). This gives a **param-to-sample ratio of approximately 1.9**, which is almost 200x above the widely-cited safety threshold of 0.01 for transformer architectures. PyTorch Forecasting's canonical TFT examples use datasets 200x larger than ours (typically 100k+ training windows).

Transformers with attention mechanisms overfit pathologically at this data scale -- the learned attention weights become memorization rather than generalization. GRU and LSTM, by contrast, have far fewer effective parameters (GRU hidden=64: ~10k params but with heavy weight sharing across timesteps; LSTM hidden=32: ~5k params) and provide equivalent Tier 2 representational coverage with proven stability on short sequences.

### Timeline argument

The remaining project timeline is 22 days until the April 27 final submission. Phase 6 (RL and Autoencoder) and Phase 7 (Experiments and Interpretability) carry higher research-question weight than TFT -- they directly answer the central research question ("does increasing model complexity improve arbitrage detection?"). Training, debugging, and hyperparameter-tuning a TFT on PyTorch Forecasting would consume 3-5 days that are better spent on Phases 6-7.

### Alternative coverage

Tier 2 is not under-represented by TFT's absence. GRU (MOD-05) and LSTM (MOD-06) provide two recurrent architectures covering the "temporal structure" hypothesis. Both are trained, evaluated with 3 seeds {42, 123, 456}, and compared directly against Tier 1 baselines in the single cross-tier comparison table (roadmap success criterion #4).

### Re-examination criterion

TFT is re-examined post-deadline if dataset expansion enables a training set of >=20,000 windows (3x+ the current 6,085). Below that threshold, the param-to-sample argument dominates.

## Kalshi Order Flow Imbalance -- Upstream Bug (NOT FIXED)

`kalshi_order_flow_imbalance` is 100% NaN across all 6,946 train + 1,817 test rows. Root cause is an upstream bug in Phase 3 feature engineering (likely a zero-denominator in the computation). Phase 5 drops this column in `NON_FEATURE_COLUMNS` for BOTH tiers (yielding 31 features instead of 35) and re-runs Tier 1 during plan 05-04 so the cross-tier comparison is apples-to-apples. Three additional Kalshi columns (`kalshi_buy_volume`, `kalshi_sell_volume`, `kalshi_realized_spread`) were also discovered to be 100% zero and were dropped alongside it. Fixing the upstream bug is deferred to a separate gap-closure phase or post-project cleanup.

---

*Deferrals documented 2026-04-05 per CONTEXT.md D9.*
