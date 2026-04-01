# Architecture Patterns

**Domain:** Cross-platform prediction market arbitrage with ML/RL  
**Researched:** 2026-04-01  
**Confidence:** HIGH (well-established ML pipeline patterns applied to specific domain)

## Recommended Architecture

The system decomposes into six major components arranged in a linear pipeline with a branching model layer. Data flows left-to-right from raw API responses to evaluation metrics. Each component has a clear input/output contract, making them independently testable and buildable in sequence.

```
                         +------------------+
                         |  Kalshi API      |
                         +--------+---------+
                                  |
                                  v
                    +-------------+-------------+
                    |    Data Ingestion Layer    |<--- Polymarket APIs
                    |  (per-platform adapters)   |     (Gamma + CLOB + Data)
                    +-------------+-------------+
                                  |
                         raw JSON / CSVs
                                  |
                                  v
                    +-------------+-------------+
                    |   Market Matching Pipeline |
                    |  (keyword + semantic sim)  |
                    +-------------+-------------+
                                  |
                         matched pairs registry
                                  |
                                  v
                    +-------------+-------------+
                    |   Feature Engineering      |
                    | (time-aligned, per-hour    |
                    |  microstructure vectors)   |
                    +-------------+-------------+
                                  |
                         feature matrices
                                  |
              +-------------------+-------------------+
              |                   |                   |
              v                   v                   v
    +---------+-------+ +--------+--------+ +--------+--------+
    | Tier 1: Regress | | Tier 2: TimeSer | | Tier 3: RL+AE   |
    | LinReg, XGBoost | | GRU, LSTM, TFT  | | PPO, PPO+AutoEnc|
    +---------+-------+ +--------+--------+ +--------+--------+
              |                   |                   |
              +-------------------+-------------------+
                                  |
                         predictions / actions
                                  |
                                  v
                    +-------------+-------------+
                    |   Evaluation & Simulation  |
                    | (metrics, P&L, Sharpe,     |
                    |  SHAP, experiment tracking) |
                    +----------------------------+
```

### Component Boundaries

| Component | Responsibility | Inputs | Outputs | Communicates With |
|-----------|---------------|--------|---------|-------------------|
| **Data Ingestion** | Fetch, cache, and normalize raw market data from both platform APIs | API endpoints, rate limit config | Standardized per-platform DataFrames (parquet/CSV) in `data/raw/` | Market Matching (provides raw data) |
| **Market Matching** | Identify equivalent contracts across Kalshi and Polymarket | Raw market metadata from both platforms | Matched-pairs registry (JSON/CSV mapping Kalshi market IDs to Polymarket token IDs) | Feature Engineering (provides pair mappings) |
| **Feature Engineering** | Time-align price histories, compute microstructure features per matched pair | Raw price/volume data + matched-pairs registry | Per-hour feature matrices with spread, volume, bid-ask, velocity columns in `data/processed/` | All model tiers (provides training/test data) |
| **Model Layer (Tiers 1-3)** | Train models, generate predictions or trading actions | Feature matrices, train/test split config | Predictions (spread forecasts) or action sequences (buy/sell/hold) | Evaluation (provides outputs for scoring) |
| **Autoencoder** | Detect anomalous spread patterns as signal filter for PPO | Feature matrices (spread behavior windows) | Binary anomaly flags + reconstruction error scores | PPO agent (gates when PPO should act) |
| **Evaluation & Simulation** | Score all models on consistent metrics, run profit simulation | Model predictions/actions + ground truth | RMSE, MAE, directional accuracy, P&L, Sharpe, win rate, SHAP plots | None (terminal node; outputs go to paper) |

### Data Flow

**Stage 1: Ingestion (API to disk)**

1. Kalshi adapter calls `/markets` to list resolved markets in economics/finance and crypto categories. For each market, checks `/historical/cutoff` to determine whether to use live or historical endpoints. Fetches hourly candlesticks via `period_interval=60`. Stores raw OHLCV + metadata as parquet files keyed by market ticker.

2. Polymarket adapter calls Gamma API to list resolved markets, extracting `clobTokenIds`. For each token, fetches trade records from the Data API (NOT `/prices-history`, which returns empty for resolved markets). Reconstructs hourly OHLCV from raw trades. Stores as parquet files keyed by token ID.

3. Both adapters write to `data/raw/{platform}/` with a consistent schema: `timestamp`, `open`, `high`, `low`, `close`, `volume`, `market_id`, `market_question`, `category`, `resolution`.

**Stage 2: Matching (raw data to pair registry)**

1. Load market metadata (questions, categories, resolution dates) from both platforms.
2. Keyword pass: exact and fuzzy string matching on market questions to generate candidate pairs.
3. Semantic similarity pass: encode questions with sentence-transformers, compute cosine similarity, threshold at a tunable cutoff (start with 0.85).
4. Manual curation: human reviews candidates, verifies settlement criteria alignment, removes false matches.
5. Output: `data/processed/matched_pairs.json` — a list of `{kalshi_id, polymarket_token_id, question, category, match_confidence}`.

**Stage 3: Feature Engineering (raw + pairs to feature matrices)**

1. For each matched pair, load raw OHLCV from both platforms.
2. Time-align to hourly buckets using outer join (NaN for missing hours, forward-filled or flagged).
3. Compute per-hour feature vector:
   - `spread`: Kalshi mid-price minus Polymarket mid-price
   - `spread_pct`: spread as percentage of average price
   - `kalshi_volume`, `poly_volume`: hourly volume on each platform
   - `volume_ratio`: ratio of volumes (liquidity imbalance signal)
   - `kalshi_bid_ask_spread`, `poly_bid_ask_spread`: per-platform bid-ask (if available)
   - `price_velocity_kalshi`, `price_velocity_poly`: hourly price change
   - `spread_velocity`: hourly change in cross-platform spread
   - `hours_to_resolution`: time remaining before contract settles
   - Rolling window features: `spread_ma_6h`, `spread_std_6h`, `volume_ma_24h`, etc.
4. Output: `data/processed/features.parquet` — one row per (pair, hour) with all feature columns.

**Stage 4: Train/Test Split**

Temporal split only. Train on earlier data, test on more recent data. No random splitting (would leak future information). Split point chosen per matched pair based on resolution date to avoid look-ahead bias.

**Stage 5: Model Training (parallel across tiers)**

- Tier 1 (regression): Flat feature vectors, no sequence dimension. scikit-learn LinearRegression and XGBoost regressor. Target: `spread_t+1` or `spread_change_t+1`.
- Tier 2 (time series): Windowed sequences of feature vectors. GRU/LSTM take `(batch, seq_len, features)` tensors. TFT uses PyTorch Forecasting's `TimeSeriesDataSet` with static (pair identity) and time-varying (all microstructure) features. Target: spread at horizon `h`.
- Tier 3 (RL): Custom Gym environment wrapping the feature data. PPO-only variant receives raw features as state. PPO+AE variant first runs autoencoder to flag anomalous hours, then PPO only acts when anomaly flag is set (or receives anomaly score as additional state feature).

**Stage 6: Evaluation (all models scored identically)**

Every model's outputs are converted to a common format before evaluation:
- Regression predictions -> directional signal (positive = spread will widen, negative = converge)
- RL actions -> trade log with entry/exit timestamps and P&L
- All models scored on: RMSE, MAE, directional accuracy, simulated P&L, win rate, Sharpe ratio
- Naive baselines (spread-always-closes, higher-volume-correct) scored identically for comparison

## Patterns to Follow

### Pattern 1: Platform Adapter Pattern
**What:** Each API gets its own adapter class that normalizes platform-specific quirks into a common schema. All downstream code works with the common schema, never with raw API responses.  
**When:** Always. This is the foundation of the pipeline.  
**Why:** Polymarket's three-API structure (Gamma + CLOB + Data) and Kalshi's live/historical split are messy. Isolating this mess in adapters keeps every other component clean.  
**Example:**
```python
# src/data/base.py
from abc import ABC, abstractmethod
import pandas as pd

class MarketDataAdapter(ABC):
    """Common interface for platform-specific data fetching."""
    
    @abstractmethod
    def list_markets(self, categories: list[str]) -> pd.DataFrame:
        """Return metadata for all markets in given categories."""
        ...
    
    @abstractmethod
    def get_price_history(self, market_id: str) -> pd.DataFrame:
        """Return hourly OHLCV for a single market.
        
        Columns: timestamp, open, high, low, close, volume
        Index: DatetimeIndex at hourly frequency
        """
        ...

# src/data/kalshi.py
class KalshiAdapter(MarketDataAdapter):
    def get_price_history(self, market_id: str) -> pd.DataFrame:
        # Checks cutoff, routes to live or historical endpoint
        # Handles null OHLC fields (no trades in period)
        # Returns standardized DataFrame
        ...

# src/data/polymarket.py
class PolymarketAdapter(MarketDataAdapter):
    def get_price_history(self, market_id: str) -> pd.DataFrame:
        # Fetches token ID from Gamma API
        # Reconstructs OHLCV from raw trades via Data API
        # Returns standardized DataFrame
        ...
```

### Pattern 2: Registry Pattern for Matched Pairs
**What:** The matched-pairs registry is a serialized file (`matched_pairs.json`) that acts as the single source of truth for which markets are paired. All downstream components read from it; only the matching pipeline writes to it.  
**When:** After the matching pipeline produces and curates pairs.  
**Why:** Decouples matching (which involves NLP and human judgment) from feature engineering (which is mechanical). Lets you re-run feature engineering without re-running matching. Lets you manually add/remove pairs without touching code.  
**Example:**
```python
# data/processed/matched_pairs.json
[
    {
        "pair_id": "btc-50k-2025-12",
        "kalshi_ticker": "KXBTC-25DEC31-T50000",
        "polymarket_token_id": "71321095738...",
        "question": "Will Bitcoin exceed $50,000 by Dec 31?",
        "category": "crypto",
        "match_confidence": 0.94,
        "settlement_aligned": true,
        "notes": "Kalshi settles at midnight ET, Polymarket at midnight UTC — 5h offset"
    }
]
```

### Pattern 3: Experiment Configuration as Data
**What:** Each experiment (window length ablation, threshold ablation, architecture comparison) is defined by a config dict/YAML, not by code changes. Models, hyperparameters, data splits, and evaluation metrics are all config-driven.  
**When:** For all three experiments and any ad-hoc runs.  
**Why:** Reproducibility. Two people working in parallel can run different configs without merge conflicts. Makes the paper's experiment tables directly traceable to config files.  
**Example:**
```python
# experiments/configs/window_ablation.yaml
experiment_name: "window_length_ablation"
base_model: "gru"
variants:
  - name: "6h"
    window_length: 6
  - name: "24h"
    window_length: 24
  - name: "72h"
    window_length: 72
  - name: "7d"
    window_length: 168
evaluation:
  metrics: ["rmse", "mae", "directional_accuracy", "sharpe"]
  test_split: "temporal"
```

### Pattern 4: Common Evaluation Interface
**What:** Every model (regardless of tier) must implement a method that returns predictions in a standardized format. The evaluation module consumes this format and produces all metrics.  
**When:** For every model.  
**Why:** The entire point of the project is fair comparison across architectures. If models output different formats, comparison code becomes a mess of special cases. Standardize early.  
**Example:**
```python
# src/models/base.py
@dataclass
class PredictionResult:
    """Standardized output from any model."""
    pair_id: str
    timestamps: list[datetime]
    predicted_spread: np.ndarray        # predicted spread value
    predicted_direction: np.ndarray     # +1 (widen), -1 (converge), 0 (hold)
    confidence: np.ndarray              # model confidence (optional)
    
    # For RL models only
    actions: np.ndarray | None = None   # buy_kalshi, buy_poly, hold, exit
    positions: np.ndarray | None = None # current position state
```

### Pattern 5: Gym Environment Wrapper for RL
**What:** The RL agent interacts with a custom Gym environment that wraps the feature matrix. The environment steps through time, providing observations and computing rewards from realized P&L.  
**When:** Tier 3 (PPO) implementation.  
**Why:** Standard interface that works with Stable Baselines 3 or custom PPO. Separates trading logic (environment) from learning logic (agent). Makes it easy to test the environment independently.  
**Example:**
```python
# src/models/trading_env.py
class SpreadTradingEnv(gym.Env):
    """Gym environment for cross-platform spread trading."""
    
    def __init__(self, features_df, pair_id, anomaly_flags=None):
        # State: current feature vector (+ anomaly score if PPO+AE)
        # Actions: {0: hold, 1: buy_kalshi, 2: buy_polymarket, 3: exit}
        # Reward: realized P&L on position close, small penalty for holding
        self.anomaly_flags = anomaly_flags  # None for PPO-only variant
        ...
    
    def step(self, action):
        # Advance one hour, compute reward, return new observation
        ...
    
    def reset(self):
        # Reset to start of episode (random or sequential start)
        ...
```

### Pattern 6: Autoencoder as Upstream Filter (Not Inline)
**What:** The autoencoder is trained separately on "normal" spread behavior. At inference time, it scores each hour's spread pattern. The PPO agent receives the anomaly score as an additional feature (soft gating) or only activates when score exceeds threshold (hard gating).  
**When:** Tier 3 PPO+AE variant.  
**Why:** Training the autoencoder and PPO jointly is unstable and hard to debug. Training them sequentially (autoencoder first, then PPO with frozen autoencoder signals) is simpler, more interpretable, and lets you evaluate each component's contribution independently. This also directly supports the professor's question about whether the autoencoder adds value.  
**Implementation detail:** Train autoencoder on training set spread windows. Compute reconstruction error for all data. Add `anomaly_score` column to feature matrix. PPO+AE variant includes this column in its observation space; PPO-only variant does not.

## Anti-Patterns to Avoid

### Anti-Pattern 1: Monolithic Data Pipeline Script
**What:** A single `pipeline.py` that fetches data, matches markets, engineers features, and trains models.  
**Why bad:** Impossible to re-run one stage without re-running everything. API calls are slow and rate-limited. If feature engineering logic changes, you do not want to re-fetch all data. Two team members cannot work on different stages simultaneously.  
**Instead:** Each stage writes its output to disk. Next stage reads from disk. Stages are independently runnable scripts or CLI commands. Cache aggressively at every boundary.

### Anti-Pattern 2: Leaky Time Splits
**What:** Using random train/test splits, or including future information in feature windows.  
**Why bad:** Catastrophic for a trading system. Models will appear to perform well but have zero real predictive power. The paper's results would be meaningless.  
**Instead:** Strict temporal splits. Training data ends at time T, test data starts after T. Feature windows look backward only. `hours_to_resolution` is the only forward-looking feature (it is known at each point in time). Validate that no feature uses data from after observation time.

### Anti-Pattern 3: Platform-Specific Logic Leaking Downstream
**What:** Code in feature engineering or model training that checks "if platform == 'kalshi'" to handle data quirks.  
**Why bad:** Fragile, hard to test, and a sign that the ingestion layer did not properly normalize.  
**Instead:** All platform-specific logic lives in the adapter. By the time data reaches feature engineering, it should be platform-agnostic — just two time series of prices/volumes for each matched pair.

### Anti-Pattern 4: Training RL on the Full Dataset
**What:** Giving the PPO agent access to all data including test-period data during training.  
**Why bad:** RL environments are especially prone to information leakage because the environment itself contains the data. If the environment includes test-period timesteps, the agent can memorize profitable sequences.  
**Instead:** Create separate environment instances for train and test periods. The training environment only contains data up to the temporal split point. Evaluation runs the trained agent on a test environment it has never seen.

### Anti-Pattern 5: Comparing Models on Different Feature Sets
**What:** Regression models get flat features, time series models get windowed features, RL gets a different state representation — and then you compare their RMSE as if they saw the same data.  
**Why bad:** You are comparing feature engineering, not model architectures. Undermines the research question.  
**Instead:** All models receive the same underlying features. Regression models get the flattened current-timestep vector. Time series models get a window of those same vectors. RL gets the same vector as its observation. The information content is equivalent; only the model's ability to use temporal structure differs.

### Anti-Pattern 6: Overfitting Autoencoder Anomaly Threshold
**What:** Tuning the reconstruction error threshold on the test set to maximize PPO+AE performance.  
**Why bad:** The threshold becomes a free parameter optimized on test data, inflating PPO+AE results.  
**Instead:** Set the threshold on the validation set (a held-out portion of the training data) using a principled method (e.g., 95th percentile of training set reconstruction errors). Lock it before touching test data.

## Detailed Component Design

### Data Ingestion Layer

```
src/data/
    __init__.py
    base.py              # MarketDataAdapter ABC
    kalshi.py            # KalshiAdapter
    polymarket.py        # PolymarketAdapter
    cache.py             # Disk caching utilities (avoid re-fetching)
    rate_limiter.py      # Per-platform rate limiting
```

**Key design decisions:**
- Each adapter handles its own rate limiting internally. Polymarket limits are per-10-seconds, not per-hour.
- Disk caching at the raw response level. Once a resolved market's data is fetched, it never changes — cache forever.
- Null handling: Kalshi candlestick OHLC fields can be null when no trades occurred. Adapter must forward-fill or mark as NaN with a `has_trades` boolean column, not silently drop rows.
- Polymarket trade reconstruction: Aggregate raw trades into hourly bars. Use volume-weighted average price for OHLCV when multiple trades occur in an hour.

### Market Matching Pipeline

```
src/matching/
    __init__.py
    keyword_matcher.py   # Stage 1: exact/fuzzy string matching
    semantic_matcher.py   # Stage 2: sentence-transformer cosine similarity
    curator.py           # Stage 3: tools for human review
    registry.py          # Read/write matched_pairs.json
```

**Key design decisions:**
- Two-stage pipeline: keyword pass (fast, high recall) then semantic pass (slower, high precision). Keyword pass filters from thousands of markets down to hundreds of candidates. Semantic pass ranks and thresholds.
- sentence-transformers model: Use `all-MiniLM-L6-v2` (fast, good for short texts) rather than a large model. Market questions are short sentences — no need for a 300M parameter encoder.
- The curator module provides a simple CLI or notebook interface for human review. Displays candidate pairs side by side with questions, resolution dates, and settlement criteria for accept/reject.
- Registry is append-only with versioning. When you re-run matching, it merges with existing pairs rather than overwriting.

### Feature Engineering

```
src/features/
    __init__.py
    alignment.py         # Time-align two price series to hourly buckets
    microstructure.py    # Compute spread, volume, bid-ask, velocity features
    windows.py           # Create rolling window features and sequence tensors
    dataset.py           # Final dataset class (PyTorch-compatible)
```

**Key design decisions:**
- Time alignment uses outer join on hourly timestamps. Hours where one platform has no data get NaN, then forward-fill up to a configurable max gap (e.g., 4 hours). Gaps longer than the max are flagged and excluded from training.
- All features are computed from the aligned pair, not from individual platforms. This ensures every feature captures the cross-platform relationship.
- The `dataset.py` module provides both a flat `FeatureDataset` (for Tier 1) and a windowed `SequenceDataset` (for Tier 2) wrapping the same underlying data. RL environments read from the same parquet files.
- Feature normalization: StandardScaler fit on training data only, applied to test data. Stored as part of the pipeline artifact for reproducibility.

### Model Layer

```
src/models/
    __init__.py
    base.py              # PredictionResult dataclass, BaseModel ABC
    linear.py            # LinearRegression wrapper
    xgboost_model.py     # XGBoost wrapper
    gru.py               # GRU model
    lstm.py              # LSTM model
    tft.py               # TFT via PyTorch Forecasting
    autoencoder.py       # Anomaly detection autoencoder
    ppo_agent.py         # PPO implementation or Stable Baselines 3 wrapper
    trading_env.py       # Gym environment for RL
```

**Key design decisions:**
- Every model inherits from a `BaseModel` that enforces `train()`, `predict()`, and `evaluate()` methods with standardized signatures. This is what enables fair comparison.
- GRU over LSTM as the primary recurrent model. GRU is faster to train, has fewer parameters, and performs comparably on short sequences. LSTM is included for completeness but GRU is the recurrent baseline.
- TFT via PyTorch Forecasting rather than from scratch. PyTorch Forecasting handles the `TimeSeriesDataSet` format, variable selection networks, and interpretable attention. Implementing TFT from scratch would consume too much time.
- PPO: Use Stable Baselines 3 (SB3) rather than writing PPO from scratch. SB3's PPO is well-tested and handles the clipped objective, GAE, and value function correctly. Writing custom PPO is error-prone and adds no value to the research question. Wrap SB3's PPO with a thin interface that conforms to `BaseModel`.
- Autoencoder architecture: Simple feedforward (not convolutional or variational). Input is a flattened window of spread features (e.g., 24 hours x N features). Bottleneck dimension is a hyperparameter. Reconstruction loss is MSE. Anomaly score = reconstruction error.

### Evaluation & Simulation

```
src/evaluation/
    __init__.py
    metrics.py           # RMSE, MAE, directional accuracy
    simulation.py        # Simulated trading P&L engine
    sharpe.py            # Sharpe ratio and risk-adjusted metrics
    baselines.py         # Naive baselines (spread-closes, volume-correct)
    shap_analysis.py     # SHAP feature importance
    reporting.py         # Generate comparison tables and plots
```

**Key design decisions:**
- The simulation engine is stateful: it tracks positions, computes P&L per trade, and handles position sizing. It does NOT model transaction costs (out of scope per project constraints) but logs where fees would apply so the paper can discuss them.
- Naive baselines are implemented as model classes that conform to the same `BaseModel` interface. This means they flow through the exact same evaluation pipeline as ML models — zero special-casing.
- SHAP analysis runs on the best-performing Tier 1 and Tier 2 models. RL models are not directly SHAP-interpretable, but the autoencoder's learned features can be analyzed.
- All metrics are computed per-pair and aggregated. This lets the paper discuss whether certain market types (crypto vs. economics) are more predictable.

### Experiment Management

```
experiments/
    configs/                 # YAML config files per experiment
    runners/
        architecture_comparison.py
        window_ablation.py
        threshold_ablation.py
    results/                 # Auto-generated result tables and plots
```

**Key design decisions:**
- Each experiment runner reads a config, instantiates models, trains, evaluates, and writes results to `experiments/results/`. No Jupyter notebooks for final experiments — notebooks are for exploration only.
- Results are saved as both CSV (for the paper) and JSON (for programmatic comparison).
- Simple experiment tracking: no MLflow or Weights & Biases overhead. Just timestamped result directories with config + metrics + model checkpoints. This is a 4-week academic project, not a production ML platform.

## Suggested Build Order

The build order is dictated by data flow dependencies. Each component depends on the output of the previous one. Within the model layer, tiers can be built in parallel once feature engineering is complete.

```
Phase 1: Data Ingestion        (no dependencies)
    |
Phase 2: Market Matching       (depends on: raw data from Phase 1)
    |
Phase 3: Feature Engineering   (depends on: matched pairs from Phase 2)
    |
    +--- Phase 4a: Tier 1 Models (LinReg, XGBoost)  \
    |                                                 |-- all depend on Phase 3
    +--- Phase 4b: Tier 2 Models (GRU, LSTM, TFT)   |   but are independent
    |                                                 |   of each other
    +--- Phase 4c: Tier 3 Models (AE, PPO, PPO+AE)  /
    |
Phase 5: Evaluation & Experiments  (depends on: all model outputs)
    |
Phase 6: Paper & Presentation     (depends on: evaluation results)
```

**Critical path:** Phases 1-3 are strictly sequential and block everything. Getting to a working feature matrix is the single most important milestone. Tier 1 models (Phase 4a) should be built first because they are simplest and validate that the feature matrix works correctly. Tier 2 and Tier 3 can then be built in parallel by two team members.

**TA check-in (April 4) target:** Phases 1-3 complete + Phase 4a (Tier 1 baselines trained and evaluated). This proves the pipeline works end-to-end and provides baseline numbers.

**Parallelization opportunities:**
- Two people can work on Kalshi and Polymarket adapters simultaneously (Phase 1).
- Once Phase 3 is done, one person can build Tier 2 while the other builds Tier 3.
- Evaluation code can be written in parallel with model training (just needs the `PredictionResult` interface defined).

## Scalability Considerations

| Concern | At 10 pairs | At 50 pairs | At 200+ pairs |
|---------|-------------|-------------|---------------|
| **Data ingestion time** | Minutes (one-time) | 30-60 min (rate limits) | Hours; need parallelism and aggressive caching |
| **Feature matrix size** | ~100K rows, fits in RAM | ~500K rows, still fits in RAM | May need chunked processing or parquet partitioning |
| **Model training time** | Seconds (Tier 1), minutes (Tier 2), minutes (Tier 3) | Minutes across the board | Tier 2/3 may need GPU; TFT especially |
| **Matching pipeline** | Manual review feasible | Manual review tedious but feasible | Need higher-precision automated matching, spot-check only |
| **Experiment grid** | 3 experiments x 4 variants each = 12 runs, minutes | Same grid, longer per run | May need to subset pairs or reduce grid |

**Realistic expectation:** Based on the project constraints (economics/finance + crypto categories, resolved markets only), the dataset will likely contain 20-80 matched pairs. This is squarely in the "fits in RAM, train on CPU" range. Do not over-engineer for scale. The bottleneck is data quality and matching accuracy, not compute.

## Sources

- Project proposal PDF (DS340 Project Proposal, Spring 2026)
- CLAUDE.md project instructions and technical context
- PROJECT.md validated requirements and constraints
- Architecture patterns derived from established ML pipeline design (scikit-learn pipelines, PyTorch training loops, Stable Baselines 3 Gym interface, PyTorch Forecasting TimeSeriesDataSet). Confidence: HIGH — these are well-established, stable patterns that have not changed materially.
- RL trading system design patterns from OpenAI Gym / Gymnasium standard interface. Confidence: HIGH.
- Autoencoder anomaly detection as upstream filter is a standard pattern in manufacturing and financial anomaly detection literature. Confidence: HIGH for the pattern; MEDIUM for its effectiveness in this specific domain (cross-platform prediction markets are a novel application).

**Note on web search:** Web search tools were unavailable during this research session. All architecture recommendations are based on established ML engineering patterns from training data. The patterns recommended (adapter pattern, Gym environments, PyTorch Forecasting for TFT, Stable Baselines 3 for PPO) are mature and stable. However, specific version numbers and API details should be verified against current documentation during implementation phases.
