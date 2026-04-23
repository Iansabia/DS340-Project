# Deferred Items — Phase 13

Out-of-scope issues discovered during phase execution. These predate the
phase-13 changes and are not caused by it; per the GSD scope boundary rule,
they are logged here rather than fixed inline.

## Pre-existing test failures observed during 13-01 execution

Detected by `.venv/bin/python -m pytest tests/ -q`:

### Collection errors (missing dependencies)
- `tests/matching/test_pipeline.py`, `test_scorer.py`, `test_semantic_matcher.py`
- `tests/models/test_trading_env.py`, `test_ppo_filtered.py`, `test_ppo_raw.py`
  - Cause: `ModuleNotFoundError: No module named 'gymnasium'` / matching deps
  - Scope: PPO / matching pipeline — unrelated to ensemble work

### Runtime failures (pre-existing bugs)
- `tests/data/test_polymarket.py::test_trades_offset_limit`
- `tests/experiments/test_retraining_policy.py` (4 tests)
- `tests/features/test_build_features.py::test_output_has_39_columns`
- `tests/models/test_linear_regression.py::test_evaluate_produces_full_metrics_dict`
- `tests/models/test_naive.py::test_evaluate_produces_full_metrics_dict`
- `tests/models/test_volume.py::test_evaluate_produces_full_metrics_dict`
- `tests/models/test_xgboost_model.py::test_evaluate_produces_full_metrics_dict`
- `tests/test_live_collector.py` (4 tests: 39-column schema drift)
  - Cause: feature pipeline now emits >39 columns (51-feature pipeline per
    Phase 8); legacy evaluate tests expect an older metrics dict schema.

**Verification that these are pre-existing, not introduced by 13-01:**
The ensemble module adds only `src/models/ensemble.py` and
`tests/models/test_ensemble.py` — touching none of the failing modules.
The 13 ensemble tests pass cleanly on the GREEN commit.

**Recommendation:** Address in a dedicated tech-debt plan after Phase 13
ensemble work is complete. Not blocking for the paper.
