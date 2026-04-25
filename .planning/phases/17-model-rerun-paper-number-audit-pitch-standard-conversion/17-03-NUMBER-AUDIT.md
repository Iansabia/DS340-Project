# Phase 17-03: PAPER_DRAFT.md Numeric Audit

**Generated:** 2026-04-25T17:59:14.411961+00:00
**Canonical source:** `experiments/results/canonical/headline.json`
**Paper:** `PAPER_DRAFT.md`

## Summary

- MATCH: 18
- MISMATCH: 0
- UNRESOLVABLE: 7

## Tolerances applied

- Dollar amounts: ±0.5% of canonical (or ±$0.50 floor)
- Sharpe (per-trade): ±0.01
- RMSE: ±0.005
- Trade counts: ±1.0% (or ±5 floor)
- bps (alpha/trade): ±0.5

## Mismatches (action required)

| Line | Model | Metric | Paper | Canonical | Context |
|------|-------|--------|-------|-----------|---------|
| — | — | — | — | — | (none) |

## Unresolvable (manual review — typically auxiliary or non-canonical numbers)

| Line | Metric | Paper Value | Context |
|------|--------|-------------|---------|
| 45 | dollar | 1.0 | `**Prediction-market mechanics.** A binary event contract on platform $P$ at time $t$ has a market price $p_P(t) \in [0, 1]$ interpretable as` |
| 45 | dollar | 0.0 | `**Prediction-market mechanics.** A binary event contract on platform $P$ at time $t$ has a market price $p_P(t) \in [0, 1]$ interpretable as` |
| 110 | dollar | 10.73 | `This filter rejected **140 of 615 pairs (22.8%)**. The impact was large: at the first large-scale backtest (April 11, 2026), linear-regressi` |
| 213 | dollar | 100.0 | `Table 2 shows the single-split backtest at 2 pp transaction costs on the full 1,673-row test set. All models use the same feature set (51 nu` |
| 450 | dollar | 1.96 | `**Post-fix live validation (12-hour window, April 24).** After deployment on the BU SCC, a 12-hour observation window (`2026-04-24T01:28Z` t` |
| 706 | dollar | 10.73 | `3. **The alpha lives in the matching pipeline and the asset class.** A 10-rule quality filter added +\$10.73 in P&L with no model changes. O` |
| 710 | sharpe | 3.2 | `5. **The per-pair annualized Sharpe is ≈ 3.2 (robust range 2–4 under realistic slippage assumptions), not 50+** — strong but not other-world` |

## All matches (verification)

| Line | Model | Metric | Value |
|------|-------|--------|-------|
| 219 | naive | total_pnl | 58.12 |
| 220 | volume | total_pnl | 59.81 |
| 221 | linear_regression | total_pnl | 232.67 |
| 222 | xgboost | total_pnl | 232.83 |
| 223 | lstm | total_pnl | 221.84 |
| 224 | gru | total_pnl | 212.5 |
| 225 | tft | total_pnl | 6.57 |
| 226 | ppo_raw | total_pnl | 158.15 |
| 227 | ppo_filtered | total_pnl | 4.61 |
| 231 | linear_regression | alpha_bps_per_trade | 14.9 |
| 646 | ppo_filtered | alpha_bps_per_trade | 0.5 |
| 702 | linear_regression | total_pnl | 232.67 |
| 702 | xgboost | total_pnl | 232.83 |
| 702 | lstm | total_pnl | 221.84 |
| 702 | gru | total_pnl | 212.5 |
| 702 | ppo_filtered | total_pnl | 4.61 |
| 702 | ppo_raw | total_pnl | 158.15 |
| 702 | ppo_filtered | alpha_bps_per_trade | 0.5 |
