# Complexity Is Not an Edge

Cross-platform prediction-market arbitrage on Kalshi and Polymarket, DS340 final project, Boston University, Spring 2026.

**Authors:** Ian Sabia (U33871576), Alvin Jang (U64760665).

## What is this?

An empirical study asking whether increasing model complexity improves arbitrage detection across Kalshi and Polymarket. We train four tiers of models (Linear Regression, XGBoost → GRU, LSTM, TFT → PPO → PPO + autoencoder) on a common evaluation protocol and find that **the simplest models consistently dominate**. The paper (`PAPER_DRAFT.md`) contains the full methodology, results, and discussion.

## Quick start

```bash
git clone https://github.com/iansabia/DS340-Project.git
cd DS340-Project
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$(pwd)
```

Python 3.12+ is required. All experiments run on CPU; no GPU needed. Total runtime for the full reproduction pipeline is ≈ 2–3 hours on a single CPU core.

## Reproducing every paper table and figure

| Paper object | Command | Output |
|---|---|---|
| Table 2 (headline backtest) | `python experiments/verify_headline.py` | `experiments/results/tier1/*.json`, `experiments/results/verify_headline.json` |
| Table 3 + Table 3b + Table 4 + Fig. 1 + Fig. 3 (walk-forward) | `python experiments/run_walk_forward.py --windows 12` | `experiments/results/walk_forward/*.json` + `experiments/figures/walk_forward_pnl.png` + `experiments/figures/walk_forward_sharpe.png` |
| §5.3 per-category breakdown | `python experiments/run_category_breakdown.py` | `experiments/results/category_breakdown.json` |
| Table 5 + Fig. 2 (data-scaling curve) | `python scripts/run_data_scaling.py --bars-per-pair 250` | `experiments/results/data_scaling/*.json` + `experiments/results/data_scaling/pnl_at_2pp_vs_data.png` |
| Table 6 (XGBoost hyperparameter sweep, 48 configs) | Pre-computed artifact at `experiments/results/xgb_hyperparam_sweep.json`; cross-tier aggregation via `python experiments/run_experiment1_comparison.py` | `experiments/results/experiment1/*.json` |
| Table 7 + Fig. 4 (transaction costs) | `python experiments/run_transaction_costs.py` | `experiments/figures/transaction_cost_sensitivity.png` |
| Table 8 (honest Sharpe accounting) | Derived from Table 2 outputs — fields `sharpe_per_trade` and `sharpe_per_pair` in `experiments/results/verify_headline.json` | — |
| §5.9 live vs backtest reconciliation | `python experiments/run_live_reconciliation.py` | `experiments/results/reconciliation/*` |
| Table 9 (feature ablation, LOGO) | `python experiments/run_feature_ablation.py` | `experiments/results/ablation/*.json` |
| Table 10 + Fig. 11 (ensemble variants + weight sweep) | `python experiments/run_ensemble_sweep.py` | `experiments/results/ensemble/summary.json` + `experiments/figures/ensemble_weight_sweep.png` |
| Fig. 5 (SHAP bar plot) | `python experiments/run_shap_analysis.py` | `experiments/figures/shap_bar_plot.png` |
| Fig. 6 (equity curves) | `python experiments/run_backtest.py` | `experiments/figures/backtest_equity_curves.png` |
| Fig. 7 (bootstrap RMSE CI) | `python experiments/run_bootstrap_ci.py` | `experiments/figures/bootstrap_ci_rmse.png` |
| Fig. 8 (lookback sweep — Experiment 2) | `python experiments/run_experiment2_lookback.py` | `experiments/figures/experiment2_lookback_pnl.png` |
| Fig. 9 (threshold heatmap — Experiment 3) | `python experiments/run_experiment3_threshold.py` | `experiments/figures/experiment3_threshold_heatmap.png` |
| Fig. 10 (TFT VSN heatmap) | `python experiments/run_tft.py && python experiments/extract_tft_heatmap.py` | `experiments/figures/tft_variable_importance.png` + `experiments/results/tft/vsn_importance.json` |
| All figures (re-render in IEEE style) | `python scripts/regenerate_figures.py` | `experiments/figures/*.png` + `experiments/results/data_scaling/pnl_at_2pp_vs_data.png` |

## Paper integrity check

```bash
bash scripts/check_paper.sh
```

Exits 0 when the paper passes all POL-04 through POL-10 grep validators (abstract ≤ 250 words, no duplicate table numbers, no dead cross-references, no residual placeholders).

## Live paper-trading system

The autonomous system deployed on BU SCC is documented separately; see `docs/SCC_DEPLOYMENT.md` if present or the §4.4 Live System Architecture section of `PAPER_DRAFT.md`.

## Project structure

```
src/              # Model code, feature pipeline, matching, live system
experiments/      # Runner scripts for each paper table/figure
scripts/          # Ops scripts (data scaling, figure regeneration, paper checks)
tests/            # pytest suite
data/raw/         # Raw API dumps (Kalshi + Polymarket)
data/processed/   # Aligned feature dataframes
experiments/results/  # JSON and CSV outputs from experiment runs
experiments/figures/  # PNG figures (IEEE style, 300 DPI)
.planning/        # GSD phase plans and research documents
```
