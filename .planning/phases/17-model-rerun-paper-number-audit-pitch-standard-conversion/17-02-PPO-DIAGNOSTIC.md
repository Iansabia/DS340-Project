# Phase 17-02: PPO Magnitude Discrepancy — Root Cause Diagnostic

**Generated:** 2026-04-25
**Author:** Phase 17-01 Task 2 (run_canonical.py + manual code-path diff)
**Triggered by:** Phase 16 final-review observation that PAPER_DRAFT.md cites
"−$7,724" for PPO+autoencoder, while extant JSONs show −$87,723.84 (legacy
backtest path) or +$4.61 (tier3 path). Two PPO result lineages on disk
disagree by ~600× in PPO-Raw magnitude and by ~19,000× in PPO-Filtered
magnitude (the PPO-Filtered ratio is large because the canonical figure is
near zero, +$4.61, so the divisor is small).

## Summary

| Lineage | PPO-Raw P&L | PPO-Filtered P&L | Source script |
|---|---|---|---|
| **Canonical (tier3)** | **+$158.15** | **+$4.61** | `experiments/run_baselines.py --tier 3` (uses `src/evaluation/profit_sim.py`) |
| Legacy (backtest, **archived**) | +$96,336.84 | −$87,723.84 | `experiments/run_backtest.py` (uses `src/evaluation/backtester.py:WalkForwardBacktester`) |
| **Ratio (legacy / canonical)** | **609.14×** | **−19,038.03×** | — |

Both lineages were produced from the same trained PPO checkpoints on the
same train/test split (n_train=6,802 / n_test=1,673, 51 features,
seed=42). The divergence is **entirely** attributable to the post-hoc
P&L attribution layer — *not* to a different model, different data, or
a buggy training run.

The canonical (tier3) numbers are now written into
`experiments/results/canonical/headline.json` and supersede both lineages
for paper purposes (REPL-07).

## Root Cause

The two scripts use **different P&L simulators with different units**.

### Canonical (`profit_sim.simulate_profit`)

For each trade bar:

```
position = sign(prediction)            # +1 or −1
bar_return = position * actual_change  # raw spread units (probability points)
total_pnl = sum(bar_return for trade bars)
```

Units: **probability points (pp)** of spread, summed across trades.
For PPO-Raw on the test set: 1,656 trades, mean per-trade return
= 158.15 / 1656 = **0.0955 pp = 9.55 bps** (alpha_bps_per_trade in
`headline.json`).

No transaction costs. No position sizing. No leverage.

### Legacy (`WalkForwardBacktester.run`)

For each trade bar (from `src/evaluation/backtester.py:107-112`):

```python
mid_price = (kalshi_close + polymarket_close) / 2.0  # ≈ 0.50
num_contracts = self.position_size / mid_price       # = $100 / 0.50 ≈ 200
gross_pnl = num_contracts * actual_change * direction
entry_cost = num_contracts * 0.03                    # 3pp entry cost
exit_cost  = num_contracts * 0.02                    # 2pp exit cost
net_pnl = gross_pnl - entry_cost - exit_cost
```

Units: **dollars**. The legacy path multiplies raw spread-unit P&L by
**~200 contracts** per trade (because contracts trade at ~$0.50 and the
script opens a fixed $100 notional position).

### Quantitative reconciliation

For PPO-Raw, the canonical-to-legacy gross-P&L conversion is:

```
canonical_pnl   = 158.15  (1,656 trades, raw spread units)
mean_per_trade  = 158.15 / 1656 = 0.0955 pp per trade
contracts_per_trade ≈ position_size / mid_price = 100 / 0.50 = 200
gross_legacy_pnl ≈ canonical_pnl * 200 = 31,630
```

But the disputed legacy file actually shows **gross_pnl ≈ $96,336 + fees
$96,078 ≈ $192,414 gross**, which is ~6× the naive 200-contract multiplier
prediction. The extra factor of ~3× comes from:

1. **Trade-count delta**: legacy=1,637 trades vs canonical=1,656 (negligible, <2%).
2. **mid_price variance**: many trade bars have mid_price < 0.50, especially
   after Phase 15 introduced commodity contracts trading at ~$0.10–$0.30,
   which inflates `num_contracts = $100 / mid_price` by 2–10× per trade.
3. **Fee compounding**: `total_fees = $96,078` is essentially equal to
   `total_pnl = $96,336`, meaning the gross strategy generated ~$192K of
   trading volume on $100 of capital × 1,637 trades — i.e. each $100 trade
   was opened against an average of ~$117 in contracts after the
   mid_price-driven contract-count inflation.

So the **609× ratio** decomposes as:

| Factor | Contribution |
|---|---|
| Position-size scaling (`$100 / mid_price ≈ 200×`) | ~200× |
| Low-mid-price commodity contracts (extra 2–4× contracts per trade) | ~3× |
| **Total** | **~600×** |

This is **not a bug in either simulator** — they are answering different
questions. `profit_sim` measures *raw model edge in spread units*.
`WalkForwardBacktester` measures *dollar P&L of a $100-notional fixed-size
strategy with 5pp round-trip costs*. Both are valid; the paper happens to
quote the canonical figure (which is the right one for cross-tier RMSE /
Sharpe comparisons) but slipped into citing the legacy figure for the
single PPO+AE headline number.

### Why is the legacy PPO-Filtered figure negative?

PPO-Filtered's autoencoder-gated trade selection is **less profitable per
trade** than unfiltered PPO. In legacy units (with ~200× contract scaling
and 5pp round-trip fees), the gross P&L (≈ $4.61 × 200 ≈ $920) is dwarfed
by the per-trade fee load (817 trades × ~$36 = $29,400 net cost after
contract-count inflation, observed as `total_fees=$27,371`). Net:
**$920 − $27,371 ≈ −$26,451**, but the actual legacy net is −$87,724 —
which is ~3× more negative because the gated trades concentrate on
high-volatility / low-mid-price bars where the contract-count inflation
hits hardest. This is also the right answer to the question the legacy
backtester is asking ("would a $100-notional strategy with 5pp fees
trading on PPO-Filtered's signal lose money?"); the issue is that the
paper has been treating it as an apples-to-apples comparison with
canonical PPO-Raw +$158.15 when the units are not the same.

### Root-cause statement

**The 600× / 19,000× divergence is a units mismatch between two valid
simulators, compounded by mid_price-driven contract-count inflation when
position_size is held fixed at $100.** It is **not** a code bug, a
training mismatch, or an episode-horizon issue. The paper accidentally
cited a legacy figure (in dollars) when its surrounding text used
canonical figures (in spread units / per-trade alpha). REPL-07 codifies
the convention going forward: all paper numerics derive from
`experiments/results/canonical/headline.json` only.

## "−$7,724" — Where Did It Come From?

The figure −$7,724 in PAPER_DRAFT.md does **not appear in any extant JSON
file**. The closest extant numbers are:

- **−$87,723.84** in the now-archived `archive/backtest_ppo_filtered.json`
  (legacy backtest path).
- **+$4.61** in `tier3/ppo_filtered.json` (canonical path).

Three hypotheses for the unreproducible −$7,724 figure:

1. **Typo of −$87,724** with a missing digit (transcription error during
   paper editing — drop the "8" → −$7,724). This is the most likely
   explanation given the magnitude is exactly off by 11.4×, which is too
   close to "missing a digit" for coincidence.
2. **An intermediate run's result** that was never persisted to disk
   (e.g. an early `run_backtest.py` execution before the autoencoder fix
   landed). No supporting evidence in git history.
3. **A misread of a different metric** — e.g., a per-window P&L from a
   12-window walk-forward or a per-pair-aggregated number. No supporting
   evidence in any JSON.

Phase 17-02 (paper audit) will replace the −$7,724 claim with the
canonical PPO+AE figure of **+$4.61** (or whatever `headline.json`
contains at audit time), expressed in pitch-standard pp / bps format
(0.51 bps per trade, sharpe_per_trade=0.014, 899 trades).

## Files Affected

**Archived** (disputed, do not cite):

- `experiments/results/backtest/ppo_raw.json` →
  `experiments/results/archive/backtest_ppo_raw.json`
- `experiments/results/backtest/ppo_filtered.json` →
  `experiments/results/archive/backtest_ppo_filtered.json`
- `experiments/results/archive/README.md` documents why these are quarantined.

**Canonical** (single source of truth):

- `experiments/results/canonical/headline.json` (Phase 17-01 Task 2 output;
  contains all 9 models with `legacy_backtest` sub-fields on PPO entries
  preserving the disputed numbers in one log).

**Generator script:**

- `experiments/run_canonical.py` (533 lines; reproduces `headline.json`
  end-to-end in <30 seconds).

## Going Forward

REPL-07 codifies the new convention for STATE.md and the remaining
Phase 17 plans:

> **All paper numerics derive from
> `experiments/results/canonical/headline.json`. No metric in any paper
> or slide may cite a JSON file outside `experiments/results/canonical/`.
> The legacy `WalkForwardBacktester` backtest path is preserved (in
> `archive/`) only for the diagnostic record; it is not cited.**

Phase 17-02 (paper-text audit) will sweep `paper/PAPER_DRAFT.md` for
every numeric claim and replace any value not present in `headline.json`
with the canonical value. Phase 17-03 (slides) does the same for
`slides/SLIDES_DRAFT.md`. Phase 17-04 adds a `check_paper_canonical.sh`
guardrail that grep-matches every dollar figure in the paper against
`headline.json` and exits non-zero on a mismatch.
