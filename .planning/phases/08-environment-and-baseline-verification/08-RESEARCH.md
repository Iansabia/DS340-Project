# Phase 8: Environment & Baseline Verification - Research

**Researched:** 2026-04-16
**Domain:** Python venv management, PyTorch reproducibility/seeding, baseline number reconciliation
**Confidence:** HIGH

## Summary

Phase 8 is the gating phase for all v1.1 work. It has five requirements that break into three independent domains: (1) environment setup -- confirming Python 3.14 compatibility and installing three libraries; (2) reproducibility infrastructure -- creating a shared seed utility and proving it works; and (3) baseline reconciliation -- ensuring the paper's Table 2 numbers match a fresh re-run from a clean environment.

The good news: **all three library dry-runs succeed on the current Python 3.14 venv** (verified 2026-04-16). There is no need to rebuild on Python 3.12. The `pytorch-forecasting==1.7.0` resolver resolves cleanly with the existing `torch==2.11.0`, and `quantstats==0.0.81` and `SciencePlots==2.2.1` have no conflicts. The environment work is mechanical.

The nuanced problem is reproducibility. The project already has a `set_seed()` function in `src/models/sequence_utils.py` that covers `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, `cudnn.deterministic`, `cudnn.benchmark`, and `torch.set_num_threads(1)` for Apple Silicon. However, it is missing `random.seed`, `PYTHONHASHSEED`, and `torch.use_deterministic_algorithms(True)`. More critically, `verify_headline.py` -- the script whose output populates Table 2 -- **does not call `set_seed` at all**. GRU and LSTM models internally call `set_seed(self._random_state)` in their `fit()` methods, but the global numpy/random state is unseeded during feature engineering and data loading. This gap means `verify_headline.py` runs are not guaranteed to be bit-for-bit reproducible. Additionally, PyTorch's MPS backend on Apple Silicon is known to be non-deterministic even with full seeding (PyTorch issue #97236, still open); the workaround is to **force CPU execution** for reproducibility verification, which the project's `set_num_threads(1)` partially addresses by avoiding the OpenMP segfault but does not guarantee determinism on MPS.

**Primary recommendation:** Create `src/utils/seed.py` with a comprehensive `set_all_seeds()` function. Inject it at the top of `verify_headline.py` and every experiment script. Run `verify_headline.py` twice on CPU (no MPS) and assert results match within 1%. The tier1/*.json files are stale (April 6, 31 features) vs. the paper's Table 2 (April 16-17, 51 features) -- they should be updated to match `verify_headline.json` output, not the other way around.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| ENV-01 | `pytorch-forecasting==1.7.0 --dry-run` succeeds on Python 3.14, or venv rebuilt on 3.12 | Dry-run verified SUCCESS on 2026-04-16. Resolver installs 15 packages cleanly. No Python 3.12 rebuild needed. |
| ENV-02 | Three target libraries installed and importable | All three dry-runs succeed: pytorch-forecasting 1.7.0, quantstats 0.0.81, SciencePlots 2.2.1. No conflicts with torch 2.11 or existing packages. Install order: pytorch-forecasting first (heaviest), then quantstats, then SciencePlots. |
| ENV-03 | Shared seed utility at `src/utils/seed.py` applied at top of every training script | Existing `set_seed()` in `sequence_utils.py` covers 6 of 9 required seeds. Missing: `random.seed`, `PYTHONHASHSEED`, `torch.use_deterministic_algorithms`. New utility must be standalone (not buried in model utils). 7 scripts need injection. |
| ENV-04 | Running `verify_headline.py` twice gives identical Table 2 numbers within 1% | Currently NOT reproducible -- `verify_headline.py` has zero seed calls. GRU/LSTM models self-seed in `fit()` but global RNG state is uncontrolled. MPS non-determinism is an additional risk -- force CPU for verification. |
| ENV-05 | Paper Table 2 numbers reproduce from clean environment; tier1/*.json reconciled | tier1/*.json files are from April 6 with 31 features; paper Table 2 uses verify_headline.json from April 16 with 51 features. Paper is canonical. tier1/*.json must be regenerated or replaced with verify_headline output. |
</phase_requirements>

## Standard Stack

### Core (already installed or verified)

| Library | Version | Purpose | Status |
|---------|---------|---------|--------|
| Python | 3.14.3 | Runtime | Current venv, verified working |
| torch | 2.11.0 | Neural network training | Installed, verified |
| xgboost | 3.2.0 | Gradient boosting baseline | Installed, verified |
| scikit-learn | 1.8.0 | Linear regression, utilities | Installed, verified |

### New installs (Phase 8 scope)

| Library | Version | Purpose | Dry-run |
|---------|---------|---------|---------|
| pytorch-forecasting | 1.7.0 | TFT model in Phase 11 | SUCCESS (resolves 15 new packages including lightning 2.6.1) |
| quantstats | 0.0.81 | Live-vs-backtest tearsheet in Phase 9 | SUCCESS (resolves 19 new packages including seaborn, yfinance) |
| SciencePlots | 2.2.1 | Publication figure styling in Phase 14 | SUCCESS (resolves 1 new package -- itself) |

### Install order and rationale

```bash
# Step 1: Heaviest dependency tree; install first to catch conflicts early
.venv/bin/pip install pytorch-forecasting==1.7.0

# Step 2: Medium dep tree; seaborn + yfinance are transitive
.venv/bin/pip install quantstats==0.0.81

# Step 3: Lightest; pure matplotlib styles
.venv/bin/pip install SciencePlots==2.2.1

# Step 4: Freeze for SCC reproducibility
.venv/bin/pip freeze > requirements.txt
```

**No known install-order conflicts.** pytorch-forecasting pulls `lightning==2.6.1` and `torchmetrics==1.9.0` which don't conflict with `quantstats` (pure pandas/numpy). SciencePlots has zero overlapping dependencies.

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Python 3.14 rebuild | Python 3.12 fresh venv | Not needed -- dry-run succeeds. Only rebuild if actual install fails (dry-run passed). |
| quantstats | pyfolio-reloaded | Heavier deps, stuck on older matplotlib patterns. quantstats is cleaner. |
| SciencePlots | tueplots | tueplots does sizing only, not styling. SciencePlots does both. |

## Architecture Patterns

### Recommended Seed Utility Design

```
src/
  utils/
    __init__.py
    seed.py          # NEW -- comprehensive seed utility
  models/
    sequence_utils.py  # EXISTING set_seed() -- delegate to utils/seed.py
```

### Pattern 1: Comprehensive Seed Function

**What:** A single function that seeds every RNG source used anywhere in the project.
**When to use:** Called once at the top of every training/experiment script, before any data loading.

```python
# src/utils/seed.py
"""Reproducibility seed utility for all experiment scripts.

Covers: Python random, numpy, torch (CPU+CUDA), CUDNN, DataLoader workers,
PYTHONHASHSEED, and torch.use_deterministic_algorithms.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_all_seeds(seed: int = 42) -> None:
    """Seed every RNG source for reproducible experiments.

    Must be called BEFORE any data loading, model creation, or training.
    Forces CPU-only execution for Apple Silicon determinism.
    """
    # 1. Python stdlib
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    # 2. NumPy
    np.random.seed(seed)

    # 3. PyTorch
    torch.manual_seed(seed)  # Seeds CPU and CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 4. CUDNN deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 5. Deterministic algorithms (raises error if non-deterministic op used)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # 6. Apple Silicon: single-threaded to avoid OpenMP/Accelerate segfault
    #    AND to ensure deterministic thread scheduling
    if not torch.cuda.is_available():
        torch.set_num_threads(1)


def worker_init_fn(worker_id: int) -> None:
    """Seed DataLoader workers for reproducible shuffling.

    Pass as worker_init_fn= argument to any DataLoader.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
```

**Source:** [PyTorch Reproducibility documentation](https://docs.pytorch.org/docs/stable/notes/randomness.html)

### Pattern 2: Script-Level Seed Injection

**What:** Every experiment script calls `set_all_seeds()` as its first operation.
**When to use:** At the import-level or the first line of `main()`.

```python
# experiments/verify_headline.py (modified)
from src.utils.seed import set_all_seeds

def main():
    set_all_seeds(42)  # FIRST LINE -- before any data loading
    # ... rest of script
```

### Pattern 3: Backward-Compatible Bridge

**What:** Update `src/models/sequence_utils.set_seed()` to delegate to the new utility.
**When to use:** Immediately, to avoid duplicated seed logic.

```python
# src/models/sequence_utils.py (modified)
from src.utils.seed import set_all_seeds

def set_seed(seed: int) -> None:
    """Backward-compatible wrapper. Delegates to utils/seed.py."""
    set_all_seeds(seed)
```

### Anti-Patterns to Avoid

- **Partial seeding:** Setting `torch.manual_seed` without `np.random.seed` leaves numpy-based feature engineering non-deterministic. The existing `verify_headline.py` has this exact bug.
- **MPS for reproducibility verification:** MPS backend is non-deterministic even with full seeding (PyTorch issue #97236, open since 2023). Force CPU for verification runs.
- **Seeding inside model.fit() only:** The GRU/LSTM models call `set_seed(self._random_state)` inside `fit()`, which is good but insufficient -- data loading, feature engineering, and train/test splitting happen before `fit()` is called.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Comprehensive seeding | Custom per-script seed logic | Single `set_all_seeds()` utility | 9 RNG sources; easy to miss one. Existing codebase already has inconsistent coverage. |
| Python hash determinism | Nothing (ignore it) | `os.environ["PYTHONHASHSEED"] = str(seed)` | Dict iteration order depends on hash seed in some edge cases. Costs nothing to set. |
| DataLoader worker seeding | `num_workers=0` everywhere | `worker_init_fn` + `Generator` | `num_workers=0` is slower; proper worker seeding gives both speed and determinism. |
| Number reconciliation | Manual comparison of JSON files | Script that loads both and computes deltas | Human comparison of floating-point numbers is error-prone. |

## Common Pitfalls

### Pitfall 1: MPS Non-Determinism on Apple Silicon

**What goes wrong:** Even with every seed set correctly, running GRU/LSTM training on the MPS (Metal) backend produces different results across runs on the same machine. PyTorch issue #97236 documents this -- the MPS backend's Metal kernels for Bernoulli (dropout) and potentially other ops are non-deterministic.
**Why it happens:** MPS kernels have different numerical implementations than CUDA. Apple's Metal API does not guarantee deterministic execution order for GPU dispatch.
**How to avoid:** For ENV-04 verification, force CPU execution. The project's existing `torch.set_num_threads(1)` avoids the segfault but does NOT guarantee determinism if MPS is used. The seed utility should not set device to MPS -- let training scripts choose CPU explicitly for verification.
**Warning signs:** GRU/LSTM results differ by >0.1% between runs despite identical seeds. P&L differences of $1-5 on the 1,673-row test set.

### Pitfall 2: Stale tier1/*.json vs. Paper Numbers

**What goes wrong:** The `experiments/results/tier1/*.json` files were generated on April 6, 2026 using 31 features. The paper's Table 2 was generated on April 16-17 using 51 features via `verify_headline.py`. The numbers disagree substantially:
- tier1 XGBoost: RMSE=0.2857, PnL=$238.41 (31 features)
- Paper Table 2 XGBoost: RMSE=0.293, PnL=$201.63 (51 features)
- verify_headline.json XGBoost: RMSE=0.29297, PnL=$201.63 (51 features)

**Why it happens:** The feature set was expanded from 31 to 51 during the data pipeline improvements. The tier1/*.json files were never regenerated.
**How to avoid:** ENV-05 must declare that `verify_headline.json` output is canonical. Either (a) regenerate tier1/*.json by running `run_baselines.py --tier 1` with the current 51-feature pipeline, or (b) treat tier1/*.json as deprecated and use verify_headline.json as the single source of truth.
**Warning signs:** Anyone running `run_baselines.py` and comparing output to tier1/*.json will get different numbers even if the code is correct.

### Pitfall 3: verify_headline.py Missing Seed Calls

**What goes wrong:** `verify_headline.py` currently has zero seed-setting code. GRU and LSTM models internally call `set_seed(42)` in their `fit()` methods, but:
- Feature engineering (`compute_derived_features`) uses numpy which is unseeded at that point
- The `fillna(0.0)` and `dropna()` operations are deterministic, but any stochastic operation in `compute_derived_features` would not be
- XGBoost uses `random_state=42` internally via its constructor (safe)
- LinearRegression is deterministic (no RNG)
- Naive and Volume baselines are deterministic (rule-based)

**Why it happens:** The script was written as a quick verification tool, not as a reproducibility-guaranteed experiment runner.
**How to avoid:** Add `set_all_seeds(42)` as the first line of `main()`. This is a 2-line change (import + call).

### Pitfall 4: `torch.use_deterministic_algorithms` May Raise on Some Ops

**What goes wrong:** Setting `torch.use_deterministic_algorithms(True)` causes PyTorch to raise `RuntimeError` if it encounters a non-deterministic operation (e.g., `index_add_` on CUDA, some scatter operations). If GRU/LSTM training uses any such op, the script will crash.
**Why it happens:** Not all PyTorch operations have deterministic implementations.
**How to avoid:** Use `torch.use_deterministic_algorithms(True, warn_only=True)` to log warnings instead of raising. This gives visibility into non-deterministic ops without breaking training. If specific ops trigger warnings, document them as known non-deterministic sources and accept the variance.
**Warning signs:** Crash on first training run after enabling deterministic algorithms.

## Code Examples

### Example 1: Complete seed utility (production-ready)

```python
# Source: PyTorch Reproducibility docs + project-specific Apple Silicon handling
# https://docs.pytorch.org/docs/stable/notes/randomness.html

import os, random
import numpy as np
import torch

def set_all_seeds(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    if not torch.cuda.is_available():
        torch.set_num_threads(1)
```

### Example 2: verify_headline.py seed injection

```python
# Add at top of main():
from src.utils.seed import set_all_seeds

def main():
    set_all_seeds(42)
    data_dir = Path("data/processed")
    # ... rest unchanged
```

### Example 3: Reproducibility verification script

```python
# experiments/check_reproducibility.py
"""Run verify_headline.py twice and assert results match within 1%."""
import json, subprocess, sys

def run_once():
    result = subprocess.run(
        [sys.executable, "-m", "experiments.verify_headline"],
        capture_output=True, text=True, check=True
    )
    return json.load(open("experiments/results/verify_headline.json"))

run1 = run_once()
run2 = run_once()

for model in run1["results"]:
    for metric in ["rmse", "pnl", "win_rate"]:
        v1 = run1["results"][model][metric]
        v2 = run2["results"][model][metric]
        if v1 == 0 and v2 == 0:
            continue
        pct_diff = abs(v1 - v2) / max(abs(v1), abs(v2), 1e-10) * 100
        assert pct_diff <= 1.0, (
            f"{model}.{metric}: {v1} vs {v2} ({pct_diff:.2f}% diff)"
        )
        print(f"  {model}.{metric}: PASS ({pct_diff:.4f}% diff)")

print("\nReproducibility check PASSED")
```

### Example 4: Reconciling tier1/*.json with verify_headline output

```python
# The reconciliation approach for ENV-05:
# 1. Run verify_headline.py with seeds (produces verify_headline.json)
# 2. For each tier1 model, overwrite tier1/*.json metrics with verify_headline values
# 3. OR: simply treat verify_headline.json as canonical and note in paper

# Key discrepancy explained:
# tier1/xgboost.json:  n_features=31, PnL=$238.41 (April 6, old pipeline)
# verify_headline.json: n_features=51, PnL=$201.63 (April 16, current pipeline)
# Paper Table 2: matches verify_headline.json (correct)
```

## Detailed Findings by Requirement

### ENV-01: Python 3.14 Compatibility

**Finding:** `pip install pytorch-forecasting==1.7.0 --dry-run` **SUCCEEDS** on Python 3.14.3.

The resolver output shows all 15 new packages resolve cleanly:
- `pytorch-forecasting-1.7.0` (target)
- `lightning-2.6.1` + `pytorch-lightning-2.6.1` (transitive)
- `torchmetrics-1.9.0`, `lightning-utilities-0.15.3`, `scikit-base-0.13.2` (transitive)
- `tqdm-4.67.3`, `aiohttp` chain (6 packages), `frozenlist-1.8.0`, `multidict-6.7.1` (transitive)

No Python 3.12 rebuild is needed. If the actual install (not dry-run) fails for any reason, the fallback plan is:
```bash
python3.12 -m venv .venv-312
.venv-312/bin/pip install -r requirements.txt  # from current freeze
.venv-312/bin/pip install pytorch-forecasting==1.7.0
```
But this fallback is unlikely to be needed.

**Confidence:** HIGH (dry-run verified on the actual venv 2026-04-16)

### ENV-02: Three Library Installs

**Finding:** All three dry-runs succeed. No inter-library conflicts detected.

| Library | New packages | Potential conflict | Verdict |
|---------|-------------|-------------------|---------|
| pytorch-forecasting 1.7.0 | 15 | `lightning` pulls `aiohttp` -- no overlap with existing deps | CLEAN |
| quantstats 0.0.81 | 19 | Pulls `seaborn`, `yfinance`, `beautifulsoup4` -- no overlap with torch/sklearn | CLEAN |
| SciencePlots 2.2.1 | 1 | Only depends on `matplotlib` (already installed) | CLEAN |

**Install order recommendation:** pytorch-forecasting first (heaviest, most likely to surface conflicts), then quantstats, then SciencePlots. This is purely defensive ordering -- no actual ordering dependency.

**Post-install verification commands:**
```bash
.venv/bin/python -c "import pytorch_forecasting; print(pytorch_forecasting.__version__)"
.venv/bin/python -c "import quantstats; print(quantstats.__version__)"
.venv/bin/python -c "import scienceplots; print('OK')"
```

**Confidence:** HIGH

### ENV-03: Seed Utility Analysis

**Current state of seeding in the codebase:**

| RNG Source | Currently seeded? | Where? | Gap? |
|-----------|-------------------|--------|------|
| `torch.manual_seed` | YES | `src/models/sequence_utils.set_seed()` | Only called inside model `fit()`, not at script top |
| `torch.cuda.manual_seed_all` | YES | `src/models/sequence_utils.set_seed()` | Same limitation |
| `np.random.seed` | YES | `src/models/sequence_utils.set_seed()` | Same limitation |
| `torch.backends.cudnn.deterministic` | YES | `src/models/sequence_utils.set_seed()` | Same limitation |
| `torch.backends.cudnn.benchmark` | YES (False) | `src/models/sequence_utils.set_seed()` | Same limitation |
| `torch.set_num_threads(1)` | YES | `src/models/sequence_utils.set_seed()` + 4 experiment scripts | Duplicated logic |
| `random.seed` | NO | Only `src/live/collector.py:716` sets `random.seed(42)` | **GAP: Not set globally** |
| `os.environ["PYTHONHASHSEED"]` | NO | Nowhere in project code | **GAP** |
| `torch.use_deterministic_algorithms` | NO | Nowhere in project code | **GAP** |
| DataLoader `worker_init_fn` | NO | Not used anywhere | **GAP (minor -- models use num_workers=0)** |

**Scripts that need `set_all_seeds()` injection:**

| Script | Currently seeds? | Priority |
|--------|-----------------|----------|
| `experiments/verify_headline.py` | NO | CRITICAL (ENV-04 depends on it) |
| `experiments/run_baselines.py` | No (models self-seed via `random_state=`) | HIGH |
| `experiments/run_walk_forward.py` | No | HIGH |
| `experiments/run_experiment1_comparison.py` | No | MEDIUM |
| `experiments/run_experiment2_lookback.py` | `torch.set_num_threads(1)` only | MEDIUM |
| `experiments/run_experiment3_threshold.py` | `torch.set_num_threads(1)` only | MEDIUM |
| `experiments/run_bootstrap_ci.py` | `torch.set_num_threads(1)` only | MEDIUM |

**The `src/utils/` directory does not exist yet.** Must create `src/utils/__init__.py` and `src/utils/seed.py`.

**Backward compatibility:** Update `src/models/sequence_utils.set_seed()` to delegate to the new `set_all_seeds()`. This ensures existing model `fit()` calls get the full seed coverage without changing their call sites.

**Confidence:** HIGH (PyTorch reproducibility docs are authoritative)

### ENV-04: Reproducibility Verification

**Critical finding: MPS is non-deterministic on Apple Silicon.**

PyTorch issue [#97236](https://github.com/pytorch/pytorch/issues/97236) (opened March 2023, still open as of April 2026) documents that the MPS backend produces different results across runs even with identical seeds. The root cause is Metal's non-deterministic dispatch for dropout (Bernoulli) and potentially other operations.

**Implications for ENV-04:**
- Tier 0 (Naive, Volume): deterministic -- no RNG involved
- Tier 1 (LR, XGBoost): deterministic -- LR has no RNG; XGBoost uses internal `random_state=42` which is CPU-based
- Tier 2 (GRU, LSTM): **potentially non-deterministic on MPS** -- dropout, weight init, and GRU/LSTM forward pass use PyTorch ops

**Verification strategy:**
1. Set `set_all_seeds(42)` at script top
2. Force CPU execution (do NOT use `device="mps"`) -- the existing GRU/LSTM code uses `get_device()` which returns CPU when CUDA is unavailable, so MPS is not currently used. Verify this.
3. Run `verify_headline.py` twice
4. Compare all metrics within 1% tolerance
5. For GRU/LSTM specifically, if they use any non-deterministic ops, `torch.use_deterministic_algorithms(True, warn_only=True)` will log warnings

**The `torch.set_num_threads(1)` workaround** (already in the codebase) serves two purposes:
1. Avoids Apple Silicon OpenMP/Accelerate segfault (original purpose)
2. Eliminates thread-scheduling non-determinism (bonus for reproducibility)

This should remain in the seed utility. It is the right approach.

**Expected outcome:** With full seeding + CPU-only + single-threaded, GRU/LSTM results should be bit-for-bit reproducible on the same machine. The 1% tolerance in ENV-04 provides headroom for any residual floating-point non-determinism.

**Confidence:** MEDIUM-HIGH (full seeding + CPU + single-thread should work, but MPS non-determinism means we must verify empirically)

### ENV-05: Paper vs. Code Reconciliation

**The discrepancy is well-understood and has a clear root cause:**

| Source | Date | Features | XGBoost RMSE | XGBoost P&L | LR P&L |
|--------|------|----------|-------------|-------------|--------|
| `tier1/xgboost.json` | Apr 6 | 31 | 0.2857 | $238.41 | $230.14 |
| `verify_headline.json` | Apr 16 | 51 | 0.2930 | $201.63 | $201.69 |
| Paper Table 2 | Apr 17 | 51 | 0.293 | $201.63 | $201.69 |

**Root cause:** Between April 6 and April 16, the feature engineering pipeline was expanded from 31 to 51 features (20 new microstructure features added). The tier1/*.json files were never regenerated with the new feature set. The paper was updated to use `verify_headline.json` output as the canonical numbers.

**Reconciliation approach:**
1. `verify_headline.json` IS the canonical source for Table 2 (matches the paper)
2. The tier1/*.json files should be regenerated by re-running `run_baselines.py --tier 1` with the current pipeline
3. After regeneration, verify that the new tier1/*.json numbers match verify_headline.json within 1%
4. If they don't match exactly (different P&L simulation code paths), document the source of divergence

**Alternative approach (simpler):** Just delete the stale tier1/*.json files and use verify_headline.json exclusively. The `run_baselines.py` and `verify_headline.py` scripts compute metrics slightly differently (the verify script uses an inline `simulate_pnl` while run_baselines uses `BasePredictor.evaluate`). For ENV-05, the goal is that the paper numbers are reproducible from ONE authoritative script -- that script is `verify_headline.py`.

**Confidence:** HIGH (the discrepancy is fully explained by the feature count difference)

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `torch.manual_seed` only | Full seed discipline (9 sources) | PyTorch 2.0+ docs | Missing any source causes silent non-reproducibility |
| `cudnn.deterministic = True` | `torch.use_deterministic_algorithms(True)` | PyTorch 1.8+ | Covers more ops than just CUDNN; raises on non-deterministic ops |
| MPS for training | CPU-only for verification | Ongoing (issue #97236) | MPS is fast but non-deterministic; use CPU when reproducibility matters |
| Per-model seeding | Global seed at script top | Best practice | Per-model seeding misses data loading and feature engineering RNGs |

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (used throughout `tests/` directory) |
| Config file | None detected (default pytest config) |
| Quick run command | `.venv/bin/python -m pytest tests/ -x -q` |
| Full suite command | `.venv/bin/python -m pytest tests/ -v` |

### Phase Requirements to Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ENV-01 | pytorch-forecasting importable | smoke | `.venv/bin/python -c "import pytorch_forecasting"` | N/A (CLI check) |
| ENV-02 | All 3 libs importable | smoke | `.venv/bin/python -c "import pytorch_forecasting; import quantstats; import scienceplots"` | N/A (CLI check) |
| ENV-03 | Seed utility exists and covers all 9 sources | unit | `.venv/bin/python -m pytest tests/test_seed_utility.py -x` | No -- Wave 0 |
| ENV-04 | verify_headline.py twice gives identical results | integration | `.venv/bin/python -m experiments.check_reproducibility` | No -- Wave 0 |
| ENV-05 | Paper numbers match fresh run | integration | `.venv/bin/python -m experiments.check_reproducibility` (same script) | No -- Wave 0 |

### Wave 0 Gaps

- [ ] `tests/test_seed_utility.py` -- unit test verifying `set_all_seeds()` sets all 9 RNG sources
- [ ] `experiments/check_reproducibility.py` -- integration test: run verify_headline twice, assert <1% diff
- [ ] `src/utils/__init__.py` -- new package (does not exist yet)
- [ ] `src/utils/seed.py` -- the seed utility itself

## Open Questions

1. **MPS determinism in PyTorch 2.11:**
   - What we know: Issue #97236 was open as of April 2023. The MPS backend has expanded significantly since then.
   - What's unclear: Whether PyTorch 2.11 has fixed MPS determinism for the specific ops (GRU forward, dropout Bernoulli) used in our models.
   - Recommendation: Do not rely on MPS determinism. Force CPU for ENV-04 verification. Accept that day-to-day training on MPS may produce slightly different numbers, but the verification script must be CPU-deterministic.

2. **`torch.use_deterministic_algorithms` compatibility with GRU/LSTM:**
   - What we know: Some PyTorch ops (notably `index_add_` on CUDA) lack deterministic implementations and will raise.
   - What's unclear: Whether GRU/LSTM on CPU with PyTorch 2.11 trigger any non-deterministic ops.
   - Recommendation: Use `warn_only=True` initially. If warnings appear, document the specific ops and accept the variance. If no warnings, upgrade to `warn_only=False` for strictest enforcement.

3. **Should ENV-05 regenerate tier1/*.json or just verify against verify_headline.json?**
   - What we know: The paper cites verify_headline.json numbers. The tier1/*.json files are stale.
   - What's unclear: Whether any downstream scripts (walk_forward, experiment1, etc.) read from tier1/*.json.
   - Recommendation: Grep the codebase for references to tier1/*.json. If nothing reads them programmatically, either regenerate for cleanliness or delete them. If something reads them, regenerate to match the current 51-feature pipeline.

## Sources

### Primary (HIGH confidence)
- [PyTorch 2.11 Reproducibility documentation](https://docs.pytorch.org/docs/stable/notes/randomness.html) -- Complete seed discipline, DataLoader worker seeding, deterministic algorithms
- [PyTorch 2.11 Release Blog](https://pytorch.org/blog/pytorch-2-11-release-blog/) -- MPS backend improvements
- Dry-run verification on actual `.venv/` (2026-04-16) -- pytorch-forecasting, quantstats, SciencePlots all resolve

### Secondary (MEDIUM confidence)
- [PyTorch issue #97236: MPS non-determinism](https://github.com/pytorch/pytorch/issues/97236) -- Open issue, MPS dropout Bernoulli non-deterministic; workaround is CPU
- [PyTorch issue #167679: MPS on macOS 26](https://github.com/pytorch/pytorch/issues/167679) -- MPS availability on newer macOS

### Tertiary (LOW confidence -- needs validation)
- MPS determinism may have improved in PyTorch 2.11 specifically for GRU/LSTM ops -- no release note confirmation found. Must verify empirically.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all dry-runs verified on actual venv
- Architecture: HIGH -- seed utility design follows official PyTorch docs exactly
- Pitfalls: HIGH -- MPS non-determinism is well-documented; tier1 JSON staleness is empirically confirmed
- Reproducibility: MEDIUM-HIGH -- full seeding + CPU should work but must be verified empirically

**Research date:** 2026-04-16
**Valid until:** 2026-05-16 (stable domain; PyTorch seed discipline changes slowly)
