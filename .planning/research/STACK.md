# Technology Stack

**Project:** Kalshi vs. Polymarket Price Discrepancies (DS340 Final)
**Researched:** 2026-04-01
**Overall Confidence:** HIGH -- all packages verified installed and working in `.venv/`

## Verified Environment

All versions below are confirmed installed in `.venv/` and import-tested. No version guesswork.

- **Python:** 3.12.12
- **Platform:** macOS (Darwin 25.3.0), Apple Silicon with MPS acceleration
- **CUDA:** Not available (expected on Mac)
- **MPS:** Available and verified working for PyTorch tensors

---

## Recommended Stack

### Core Framework

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| Python | 3.12.12 | Runtime | Already installed. 3.12 has excellent library support and performance improvements (PEP 709 inlined comprehensions, PEP 684 per-interpreter GIL). No reason to change. | HIGH |
| PyTorch | 2.10.0 | Deep learning backbone | Powers GRU, LSTM, autoencoder, and is required by PyTorch Forecasting, SB3, and sentence-transformers. MPS backend works for training acceleration on Apple Silicon. Use MPS for time series models but CPU for SB3/PPO (see notes). | HIGH |
| PyTorch Lightning | 2.6.1 | Training orchestration | Required by pytorch-forecasting for TFT training. Also useful for standardizing GRU/LSTM training loops with built-in checkpointing, early stopping, and logging. Reduces boilerplate significantly. | HIGH |

### Data Pipeline

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| requests | 2.32.5 | API calls | Both Kalshi and Polymarket APIs are simple REST endpoints. No need for async (httpx) since data ingestion is a batch process, not real-time. Rate limiting is easily handled with `time.sleep()`. Verified: both APIs return 200 from this environment. | HIGH |
| pandas | 2.3.3 | Data manipulation | Industry standard for tabular time series data. Arrow-backed string dtype in 2.x improves memory. Used for OHLCV candlestick data, spread calculations, feature engineering, and dataset alignment. | HIGH |
| numpy | 2.4.3 | Numerical ops | Required by virtually everything. 2.x has performance improvements. Used directly for feature normalization, rolling statistics, and array operations that are faster than pandas. | HIGH |

### Market Matching (NLP)

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| sentence-transformers | 5.3.0 | Semantic similarity for contract matching | Generates embeddings for Kalshi/Polymarket contract titles to find matched pairs. Use `all-MiniLM-L6-v2` model -- 384-dim embeddings, fast inference, good quality for short text similarity. Overkill models like `all-mpnet-base-v2` add latency without meaningful accuracy gain for contract title matching. | HIGH |

**Model choice: `all-MiniLM-L6-v2`** -- not `all-mpnet-base-v2`. Rationale:
- Contract titles are short (10-30 tokens). MiniLM captures the semantics.
- 5x faster inference than mpnet. Matching is a one-time batch operation, but faster iteration during development matters.
- 384 vs 768 dimensions -- cosine similarity works equally well at either dimensionality for this use case.
- If matching quality is poor, upgrade to mpnet. But start with MiniLM.

### Regression / Tree-Based Models

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| scikit-learn | 1.8.0 | Linear Regression, evaluation metrics, data splitting | The standard. `LinearRegression`, `train_test_split`, `mean_squared_error`, `mean_absolute_error`, classification metrics for directional accuracy. No alternative needed. | HIGH |
| XGBoost | 3.2.0 | Gradient boosted tree baseline | Best-in-class for tabular data. XGBoost 3.x has significant performance and API improvements. Use `XGBRegressor` for spread prediction, `XGBClassifier` for directional accuracy as secondary metric. Native SHAP integration via `TreeExplainer`. | HIGH |
| LightGBM | 4.6.0 | Optional second tree baseline | Already installed. Faster training than XGBoost on larger datasets, but for this project's likely small dataset (<100K rows), XGBoost suffices. Include only if you want a second tree baseline for comparison. Do not prioritize. | HIGH |

**Use XGBoost, not LightGBM, as the primary tree baseline.** Rationale:
- XGBoost has better SHAP support (native `TreeExplainer` integration is more mature).
- For small datasets, XGBoost and LightGBM perform nearly identically.
- Maintaining two tree models adds complexity without answering the research question.
- If you include LightGBM, frame it as a robustness check, not a primary baseline.

### Time Series Models

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| PyTorch (nn.GRU) | 2.10.0 | GRU model | Implement from scratch using `torch.nn.GRU`. Simpler than LSTM (fewer gates), trains faster, performs comparably on short sequences (6h-7d lookback). This is the preferred recurrent model for this project. | HIGH |
| PyTorch (nn.LSTM) | 2.10.0 | LSTM model | Implement from scratch using `torch.nn.LSTM`. Include for completeness in the complexity comparison. Expect marginal (if any) improvement over GRU given the short lookback windows. | HIGH |
| pytorch-forecasting | 1.6.1 | Temporal Fusion Transformer (TFT) | Provides `TemporalFusionTransformer` and `TimeSeriesDataSet` classes. TFT handles mixed static/temporal features with attention, which is relevant for spread prediction with market metadata. **Requires pytorch_lightning Trainer for training.** Data must be formatted into `TimeSeriesDataSet` -- this is the main integration effort. | HIGH |

**TFT integration notes (verified):**
- `pytorch_forecasting` 1.6.1 depends on `pytorch_lightning` (not the `lightning` package directly). Both are installed and compatible.
- `TimeSeriesDataSet` requires specific column structure: `time_idx`, `group_ids`, `target`, and feature columns. Plan data pipeline output accordingly.
- Available loss functions: `MAE`, `RMSE`, `SMAPE`, `QuantileLoss`. Use `QuantileLoss` for probabilistic forecasting (useful for trading decisions), `MAE`/`RMSE` for point estimates.
- Hyperparameter tuning via optuna integration is available but likely overkill for a class project.

### Reinforcement Learning

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| stable-baselines3 | 2.7.1 | PPO implementation | Battle-tested PPO implementation. Handles all the training infrastructure (rollout buffer, GAE, value function clipping). Avoids reimplementing PPO from scratch, which would be error-prone and time-consuming. | HIGH |
| gymnasium | 1.2.3 | RL environment interface | Standard environment API that SB3 requires. You will write a custom `gymnasium.Env` subclass for the trading environment. Must define `observation_space`, `action_space`, `step()`, `reset()`. | HIGH |

**Critical SB3/PPO notes:**
- **Use CPU, not MPS, for PPO training.** SB3 explicitly warns that PPO with MLP policy runs poorly on GPU. The overhead of CPU-GPU data transfer dominates small network training. Verified: `device='mps'` works but triggers a warning. Use `device='cpu'`.
- **sb3-contrib is NOT installed.** It provides `RecurrentPPO` (PPO with LSTM policy). If you want the RL agent to have memory across timesteps (rather than just receiving a window of features), install it. However, for the project scope, standard PPO with a feature window as observation is sufficient and simpler.
- **Custom environment design is the hard part**, not the PPO implementation. The observation space, action space, reward function, and episode structure will determine PPO's performance far more than hyperparameters.

### Anomaly Detection (Autoencoder)

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| PyTorch (nn.Module) | 2.10.0 | Autoencoder for anomaly detection | Build from scratch as a simple `nn.Module`. Encoder-decoder architecture trained on "normal" spread behavior. Flag high reconstruction error as anomalous (potential arbitrage signal). Feed anomaly scores as additional feature to PPO. No library needed -- autoencoders are 20-50 lines of PyTorch. | HIGH |

**Do not use a library for the autoencoder.** Rationale:
- It is a simple feedforward encoder-decoder. Libraries like PyOD add dependency complexity for a model that is trivially implemented.
- Custom implementation gives full control over the reconstruction error metric and threshold tuning.
- The autoencoder is a signal filter, not the core model. Keep it lightweight.

### Interpretability

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| SHAP | 0.51.0 | Feature importance explanations | `TreeExplainer` for XGBoost (exact Shapley values, fast). `DeepExplainer` or `GradientExplainer` for PyTorch models (approximate). `KernelExplainer` as fallback for any model. Use for the interpretability analysis requirement. | HIGH |

**SHAP strategy by model tier:**
- **XGBoost:** `shap.TreeExplainer(model)` -- exact, fast, preferred. Also `GPUTreeExplainer` available but unnecessary on CPU.
- **GRU/LSTM:** `shap.DeepExplainer(model, background_data)` -- approximate but useful. Alternatively, `shap.GradientExplainer` for gradient-based approximation.
- **PPO:** Use `shap.KernelExplainer` on the policy network, or more practically, analyze the observation features that led to trading decisions via attention/saliency rather than SHAP.
- **TFT:** Has built-in attention weights that serve as interpretability. Use those instead of SHAP for TFT.

### Visualization

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| matplotlib | 3.10.8 | Base plotting | Required by SHAP, seaborn, and useful for custom plots. | HIGH |
| seaborn | 0.13.2 | Statistical plots | Better defaults than matplotlib for heatmaps, distributions, and pairplots. Use for feature correlation analysis and model comparison charts. | HIGH |
| Jupyter | 4.5.6 (JupyterLab) | Interactive exploration | Already installed with full stack (ipykernel, ipywidgets). Use notebooks for EDA and visualization only -- production code goes in `src/`. | HIGH |

### Testing

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| pytest | (to install) | Test runner | **Not currently installed.** AGENTS.md mandates TDD. Install pytest. Use it for data pipeline tests, feature engineering tests, model smoke tests, and environment tests. | HIGH |

---

## What NOT to Use (and Why)

| Technology | Why Not | What to Use Instead |
|------------|---------|---------------------|
| TensorFlow/Keras | PyTorch is already the backbone for SB3, pytorch-forecasting, and sentence-transformers. Mixing frameworks creates dependency hell and cognitive overhead. | PyTorch for everything |
| httpx/aiohttp | Async is unnecessary for batch historical data ingestion. Both APIs are simple REST calls. Async adds complexity without benefit for this use case. | requests with simple rate limiting |
| PyOD | Autoencoder is trivially implemented in ~30 lines of PyTorch. Adding PyOD creates an unnecessary dependency for a simple model. | Custom `nn.Module` autoencoder |
| RLlib (Ray) | Massive dependency, designed for distributed RL at scale. SB3 is simpler, well-documented, and sufficient for single-machine PPO. | stable-baselines3 |
| darts (time series) | Another time series library, but pytorch-forecasting already provides TFT. darts would be redundant and adds a large dependency. | pytorch-forecasting for TFT, raw PyTorch for GRU/LSTM |
| polars | Tempting for speed, but pandas is the standard that every other library expects (SHAP, pytorch-forecasting TimeSeriesDataSet, seaborn). Converting between polars and pandas negates the speed benefit at this dataset scale. | pandas |
| wandb/mlflow | Experiment tracking tools are valuable for large projects, but for a 3-week class project with ~8 model variants, a simple CSV/JSON log is sufficient. Over-engineering experiment tracking is a common time sink. | Manual logging to CSV + matplotlib comparison plots |
| optuna | Available via pytorch-forecasting, but hyperparameter tuning is premature for this project. Get baselines working first. If time permits, use grid search over a small parameter space. | Manual hyperparameter selection or small grid search |
| sb3-contrib (RecurrentPPO) | Adds complexity. Standard PPO with a windowed observation (e.g., last 24 feature vectors flattened) captures temporal structure. RecurrentPPO's LSTM policy is harder to debug and train. Only consider if PPO-with-window performs very poorly. | Standard PPO with windowed observations |

---

## Device Strategy

| Model | Device | Rationale |
|-------|--------|-----------|
| XGBoost | CPU | XGBoost uses its own parallel tree building. No GPU benefit at this dataset scale. |
| Linear Regression | CPU | scikit-learn is CPU-only. Trains in milliseconds. |
| GRU / LSTM | MPS | PyTorch MPS backend accelerates matrix operations. Use `device='mps'` for training, move to CPU for SHAP analysis. |
| TFT | MPS | PyTorch Lightning trainer supports MPS via `accelerator='mps'`. Larger model benefits more from acceleration. |
| Autoencoder | MPS | Small model, but MPS is free to use. Train on MPS, evaluate on CPU. |
| PPO (SB3) | CPU | SB3 explicitly warns against GPU for MLP policies. CPU-GPU transfer overhead dominates. Verified with runtime warning. |
| sentence-transformers | MPS | Embedding generation benefits from MPS acceleration, especially for batch encoding of all contract titles. |

---

## Installation

### Already Installed (verified in `.venv/`)

```bash
# These are already installed and working -- DO NOT reinstall
torch==2.10.0
pytorch-forecasting==1.6.1
pytorch-lightning==2.6.1
stable-baselines3==2.7.1
gymnasium==1.2.3
xgboost==3.2.0
lightgbm==4.6.0
scikit-learn==1.8.0
sentence-transformers==5.3.0
shap==0.51.0
pandas==2.3.3
numpy==2.4.3
matplotlib==3.10.8
seaborn==0.13.2
requests==2.32.5
jupyterlab==4.5.6
```

### Needs Installation

```bash
# Testing framework (required by TDD workflow in AGENTS.md)
.venv/bin/pip install pytest pytest-cov

# Optional: if RecurrentPPO is needed later
# .venv/bin/pip install sb3-contrib
```

### Lock Requirements

```bash
# Generate requirements.txt from current environment
.venv/bin/pip freeze > requirements.txt
```

This should be done early to ensure reproducibility for the team (Ian + Alvin working from the same dependencies).

---

## API Connectivity (Verified)

| API | Endpoint | Status | Notes |
|-----|----------|--------|-------|
| Kalshi | `https://api.elections.kalshi.com/trade-api/v2/exchange/status` | 200 OK | No auth required for public market data |
| Polymarket Gamma | `https://gamma-api.polymarket.com/markets?limit=1` | 200 OK | Metadata API, returns market info and clobTokenIds |
| Polymarket CLOB | `https://clob.polymarket.com/` | Not tested | Prices and orderbook data |
| Polymarket Data | `https://data-api.polymarket.com/` | Not tested | Historical trade records for resolved markets |

---

## Key Integration Points

### 1. Data Pipeline -> Feature Engineering
- **Output format:** pandas DataFrames with DatetimeIndex, one row per hour per matched pair
- **Columns:** kalshi_price, polymarket_price, spread, kalshi_volume, polymarket_volume, bid_ask features
- **TimeSeriesDataSet compatibility:** Include `time_idx` (integer), `group_id` (matched pair ID), and `target` (spread or next-hour spread) columns from the start. This prevents painful reformatting for TFT later.

### 2. Feature Engineering -> Models
- **Tabular models (LR, XGBoost):** Flat feature vectors, one row per observation. Use scikit-learn `Pipeline` for preprocessing.
- **Sequence models (GRU, LSTM):** 3D tensors `(batch, seq_len, features)` via custom PyTorch `Dataset`. Sequence length = lookback window.
- **TFT:** `TimeSeriesDataSet` from pytorch-forecasting. Requires specific column naming and metadata.
- **PPO:** Custom `gymnasium.Env` with `observation_space = spaces.Box(...)`. Observation = flattened feature window or current feature vector + summary stats.

### 3. Models -> Evaluation
- All models must output predictions in a common format for apples-to-apples comparison.
- **Regression output:** Predicted spread value at time t+1 (or t+n).
- **RL output:** Trading actions (buy/sell/hold) which are converted to simulated P&L.
- **Bridge:** Convert regression predictions to trading signals via threshold (predict spread > X -> trade). This makes regression models directly comparable to PPO on trading metrics.

---

## Alternatives Considered (Full Detail)

| Category | Recommended | Alternative | Why Not Alternative |
|----------|-------------|-------------|---------------------|
| Deep Learning | PyTorch | TensorFlow | SB3, pytorch-forecasting, sentence-transformers all require PyTorch. No choice here. |
| RL Library | stable-baselines3 | RLlib (Ray) | RLlib is designed for distributed training. SB3 is simpler, single-machine, well-documented. |
| RL Library | stable-baselines3 | CleanRL | CleanRL is single-file implementations -- good for learning, bad for production. SB3 handles logging, callbacks, saving/loading. |
| Tree Models | XGBoost | CatBoost | CatBoost handles categoricals natively, which isn't relevant for numerical microstructure features. XGBoost has better SHAP integration. |
| Time Series | pytorch-forecasting (TFT) | Nixtla/NeuralForecast | Less mature, smaller community. pytorch-forecasting is the standard for TFT in PyTorch. |
| Embeddings | sentence-transformers | OpenAI embeddings API | Project requires no external API dependencies for model inference. sentence-transformers runs locally. |
| Data | pandas | polars | Every downstream library expects pandas. Conversion overhead negates polars speed benefit at this scale. |
| Viz | matplotlib + seaborn | plotly | Static plots are fine for a paper. Plotly's interactivity is unnecessary overhead. |

---

## Sources

- **All versions verified via direct import** in `.venv/` on 2026-04-01 (not from PyPI or documentation)
- **API connectivity verified** via live HTTP requests to Kalshi and Polymarket endpoints
- **MPS compatibility verified** via `torch.randn(3, 3, device='mps')` test
- **SB3 PPO device warning verified** via actual PPO instantiation with `device='mps'` on CartPole-v1
- **pytorch-forecasting/Lightning compatibility verified** via MRO inspection of TFT class
- **SHAP explainer availability verified** via `dir(shap)` inspection
