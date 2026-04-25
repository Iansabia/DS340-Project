# Phase 18: System Audit — Adversarial Verification — Context

**Gathered:** 2026-04-25
**Status:** Ready for planning
**Source:** Direct user brief (post-Phase 17 / pre-submission)

<domain>
## Phase Boundary

This phase is an **adversarial audit** of every quantitative claim in PAPER_DRAFT.md, the canonical results JSON, the live-trading reconciliation, and the slide deck — performed as if Kevin Gold (the professor) were going to attack each number in office hours. The goal is *not* to "polish" the paper. The goal is to discover any inflated, leaked, or unreproducible result *before* he does, and either (a) confirm the number is defensible with documented assumptions or (b) correct it and update the paper.

The single most important claim under audit is the headline **per-pair-corrected Sharpe ≈ 3.2 backtest** number (and its per-trade ≈ 0.04 / annualized derivation). 3.2 is high enough that a sophisticated reader will assume something is wrong unless we have already shown our work.

Phase 18 must be **kill-or-confirm**: each claim either survives the audit unchanged, survives with documented caveats and possibly a corrected number, or is removed/replaced. If the headline Sharpe drops to (e.g.) 1.8 after honest cost + correlation correction, that becomes the new headline number and the paper is updated accordingly. A weaker but defensible result is strictly better than a strong indefensible one.

**Out of scope for Phase 18:**
- New experiments or new model training runs (Phase 17 closed the model rerun work)
- Live deployment changes (live system stays as-is on SCC)
- Slide redesign (Phase 17-03 closed the slide work; only update slide *numbers* if audit changes them)

**In scope for Phase 18:**
- Recomputation of headline metrics from raw position-level data
- Look-ahead / leakage audit of the feature set and walk-forward protocol
- Fee + slippage realism check
- Survivorship + selection-bias audit of the pair universe
- Cross-correlation / per-pair Sharpe assumption audit
- Paper number-by-number trace to canonical/headline.json
- Confidence intervals on all headline metrics
- Living `AUDIT_REPORT.md` documenting every finding (pass / fail / corrected)
- If any number changes: paper + slide deck updated to match

</domain>

<decisions>
## Implementation Decisions

### Audit Posture
- **Kill-or-confirm, not polish.** If a claim is wrong or inflated, the paper changes — no sugarcoating.
- **Adversarial mindset.** Every claim is presumed guilty until proven defensible. Auditors must actively try to break each number.
- **Document the assumption stack.** For each headline metric, write down the assumptions it depends on (e.g., "Sharpe 3.2 assumes pair returns are i.i.d. and trades occur 2,190 times/year"). If an assumption fails, the metric fails.

### Audit Dimensions (locked — must all be addressed)

**Tier 1 — Sharpe 3.2 verification (highest priority):**
1. Recompute Sharpe from raw per-trade ledger (not from summary stats). Include the script + reproducible command in `experiments/audit/`.
2. Test the cross-sectional independence assumption: are the 144 pairs i.i.d., or do oil pairs / crypto pairs share systematic risk on the same calendar bar? Compute the average pairwise correlation of contemporaneous trade returns and adjust the annualization factor accordingly.
3. Document the annualization formula explicitly in the paper. Currently sqrt(N_trades_per_year) — show the N and justify it.
4. Bootstrap 95% confidence interval on per-trade Sharpe and on annualized Sharpe. Both bounds must be reported in PAPER_DRAFT.md (Table 8 / abstract footnote).
5. Reconcile per-trade Sharpe (≈ 0.04) and per-pair Sharpe (≈ 3.2): show the math, including any compounding step.

**Tier 2 — Leakage / look-ahead bias:**
6. Walk every feature in the 59-feature set and flag any that could carry future information (rolling means with `center=True`, label-aligned z-scores, post-resolution filters applied retroactively, etc.).
7. Verify the walk-forward embargo is large enough that no pair lifecycle bridges a train/test boundary. If pair P opens in train and closes in test, that's leakage.
8. Audit the matching pipeline's 10-rule structural quality filter for retroactive use of post-resolution information. If any rule uses settlement outcomes (or proxies for them), it's survivorship/leakage.

**Tier 3 — Cost realism:**
9. Confirm Kalshi maker/taker fees are charged per trade in `simulate_profit` and in `WalkForwardBacktester`. If they aren't, recompute net Sharpe with fees and report that as the new headline.
10. Confirm Polymarket gas / withdrawal cost assumption is documented (even if unmodeled). Paper §6.4 must explicitly say what is and is not modeled.
11. Position-size sanity check: at the typical orderbook depth on these markets, can $100/trade actually fill at the assumed price without slippage? If not, document the gap; ideally apply a slippage haircut and re-report.

**Tier 4 — Selection / survivorship:**
12. Audit the pair universe construction: was any pair excluded *after* observing its outcome? Cross-reference `data/processed/pairs/` history vs filter applied at training time.
13. Spot-check 10 dropped pairs at random — confirm the drop reason is structural (e.g., insufficient overlap, low liquidity), not retroactive.

**Tier 5 — Number-by-number trace:**
14. Every numeric claim in PAPER_DRAFT.md (every Sharpe, every $, every %, every win rate, every model row in Tables 1–8) must trace to a generation script + canonical file. Build a `paper_numbers.csv` mapping: `{claim_text, paper_section, source_file, source_command}`.
15. Extend `scripts/check_paper.sh` with a regression check that recomputes 5+ headline numbers from canonical files and grep-matches them in PAPER_DRAFT.md.

**Tier 6 — Live-vs-backtest honesty:**
16. The paper currently documents anti-correlation in §5.9 and oil's live underperformance (40% live vs 77% backtest). Audit phase confirms these numbers are current as of audit date and that the paper does NOT bury them.

### Deliverable

A single living document — **`AUDIT_REPORT.md`** at the project root — with one row per audit dimension above, each marked **PASS / CORRECTED / FAILED**, with evidence (commands, file paths, recomputed numbers). If any row is CORRECTED, PAPER_DRAFT.md is updated in the same plan that corrects it.

The audit report itself is published with the paper as a supplementary appendix or linked from §6.4 — Kevin Gold should see we did this work, not just that we passed it.

### Tooling Decisions
- All audit scripts go in `experiments/audit/` — one Python file per Tier above (e.g., `audit_sharpe.py`, `audit_leakage.py`, `audit_costs.py`).
- Each audit script writes a JSON result file to `experiments/results/audit/` (e.g., `sharpe_audit.json`).
- The orchestrating `AUDIT_REPORT.md` is generated from those JSONs (not hand-written, except for prose findings).
- TDD applies: each audit script has at least one test that proves the audit *would* catch the failure mode it's designed for (e.g., `audit_leakage.py` tested against a synthetic look-ahead feature it must flag).

### Anti-goals (locked)
- **No model retraining.** The canonical numbers from Phase 17 are the ground truth being audited; we don't replace them by re-running models.
- **No new figures unless a number changes.** This is an audit, not a paper rewrite.
- **No "soft fail" verdicts.** Every audit dimension is PASS / CORRECTED / FAILED. "Probably fine" is not an option.

### Claude's Discretion
- Exact wave structure (likely Wave 0: tooling/scaffolding, Wave 1: Tiers 1+2 in parallel, Wave 2: Tiers 3+4+5, Wave 3: paper updates if corrections needed)
- Specific bootstrap iteration counts (default to 10,000)
- Specific confidence levels (default to 95%)
- Whether each Tier becomes one PLAN.md or multiple (planner's call based on dependency graph)
- Whether to include a "stretch" Tier 7 examining feature stability across walk-forward windows (only if time permits)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase 17 outputs (the ground truth being audited)
- `experiments/results/canonical/headline.json` — single source of truth for every paper number
- `experiments/run_canonical.py` — script that produced headline.json (audit must reproduce its outputs)
- `.planning/phases/17-model-rerun-paper-number-audit-pitch-standard-conversion/17-02-PPO-DIAGNOSTIC.md` — explains the 600× PPO units mismatch already caught in Phase 17

### Paper + slide artifacts
- `PAPER_DRAFT.md` — every numeric claim must be traced
- `slides_deck.html` — slide numbers must match paper post-audit
- `scripts/check_paper.sh` — existing 19-check validator; audit extends with regression checks

### Live trading data (for live-vs-backtest tier)
- `data/live/position_history.jsonl` — 20,124+ closed live positions
- `src/analysis/reconciliation.py` — existing live/backtest reconciliation
- `PAPER_DRAFT.md §5.9.1` — current live-vs-backtest section (audit confirms numbers, doesn't replace)

### Backtest + evaluation code (must be inspected for leakage)
- `src/evaluation/profit_sim.py` — `simulate_profit()` (per-trade simulator)
- `src/evaluation/walk_forward.py` — `WalkForwardBacktester` (window-by-window simulator)
- `src/features/build_features.py` — feature engineering (audit for `center=True`, future-leaking aggregations)
- `src/matching/quality_filter.py` — 10-rule structural filter (audit for post-resolution info usage)

### Reproducibility infrastructure
- `src/utils/seed.py` — global seed manager (Phase 8 deliverable)
- `requirements.txt` — frozen environment

</canonical_refs>

<specifics>
## Specific Ideas

### Headline Sharpe audit recipe (Tier 1, expand into plan)

Pseudocode for `experiments/audit/audit_sharpe.py`:

```python
# 1. Load every closed test-set trade with per-trade P&L
trades = load_canonical_test_trades()  # dataframe: pair_id, entry_ts, exit_ts, pnl_dollars, pnl_pp

# 2. Per-trade Sharpe (sanity)
per_trade_sharpe = trades.pnl_pp.mean() / trades.pnl_pp.std()
assert abs(per_trade_sharpe - 0.04) < 0.01, "per-trade Sharpe drifted from canonical 0.04"

# 3. Per-pair Sharpe (the 3.2 claim)
pair_returns = trades.groupby('pair_id').pnl_pp.sum()
per_pair_sharpe = pair_returns.mean() / pair_returns.std()

# 4. Bootstrap 95% CI
boot_sharpes = [bootstrap_resample(pair_returns).sharpe() for _ in range(10_000)]
ci_low, ci_high = np.percentile(boot_sharpes, [2.5, 97.5])

# 5. Cross-sectional correlation correction
contemp_returns = pivot_table(trades, index='entry_ts_bucket', columns='pair_id', values='pnl_pp')
avg_pair_corr = contemp_returns.corr().values[np.triu_indices(N, k=1)].mean()
n_eff = N_pairs / (1 + (N_pairs - 1) * avg_pair_corr)  # effective independent observations
sharpe_corrected = per_pair_sharpe * sqrt(n_eff / N_pairs)

# 6. Report all four numbers + assumption stack
write_json({
  'per_trade_sharpe': per_trade_sharpe,
  'per_pair_sharpe_naive': per_pair_sharpe,
  'per_pair_sharpe_ci_95': [ci_low, ci_high],
  'per_pair_sharpe_corr_corrected': sharpe_corrected,
  'avg_pair_corr': avg_pair_corr,
  'n_pairs': N_pairs,
  'n_eff': n_eff,
  'assumptions': [...],
})
```

This is the kind of work the audit produces. Apply the same template to costs, leakage, etc.

### Leakage audit specifics (Tier 2)

For each feature in `src/features/build_features.py`, classify:
- **Safe:** uses only data before the bar's `entry_ts`
- **Suspicious:** uses rolling window — verify `center=False` and window endpoint <= entry_ts
- **Leaking:** uses any post-entry data

For the 59-feature set, expect ~0 leaking features (Phase 8 already verified determinism); but the audit must produce a per-feature classification table, not just a "we checked" sentence.

### Walk-forward embargo audit (Tier 2)

Compute, for each walk-forward window, the set of pair_ids that are open in both train end and test start. If non-empty: that's the embargo violation.

### Test fixture design

Each audit script gets one fixture test that proves the audit catches its target:
- `test_audit_sharpe_catches_inflated_independence`: feeds in synthetic perfectly-correlated pairs, asserts the corrected Sharpe is ~1/sqrt(N) of the naive.
- `test_audit_leakage_catches_synthetic_look_ahead`: injects a feature that uses `df.shift(-1)`, asserts it's flagged.
- `test_audit_costs_catches_zero_fee`: runs `simulate_profit` with `fee=0` vs `fee=0.05`, asserts the audit reports the gap.

</specifics>

<deferred>
## Deferred Ideas

- **Bayesian Sharpe estimation** (Frank Smith / Lo & MacKinlay): nice-to-have but not necessary for audit pass; bootstrap CIs are sufficient for the rubric.
- **Monte Carlo permutation test for strategy edge** (shuffling labels and recomputing P&L): out of scope unless audit reveals that the headline P&L could plausibly be from chance.
- **Benchmark against published prediction-market arbitrage research**: literature comparison is interesting but not part of an internal audit.
- **Slide deck redesign**: only update slide *numbers* if the audit changes any number; no visual changes.
- **Live system code changes**: out of scope. The audit reads live data; it does not modify the live system.

</deferred>

---

*Phase: 18-system-audit-adversarial-verification*
*Context gathered: 2026-04-25 from direct user brief — kill-or-confirm audit before April 27 submission*
