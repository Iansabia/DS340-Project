---
gsd_state_version: 1.0
milestone: v1.1
milestone_name: Extended Evidence & Submission
current_plan: 7
status: completed
stopped_at: "Completed 18-08-PLAN.md (purged pair-stratified leakage-free retraining + Tier 1 Sharpe audit redo) — verdict CORRECTED: purged corrected per-pair Sharpe +0.81 [CI 0.70, 1.07], +175% drift from leaky 0.30 (driven by negative purged avg_corr short-circuiting BLdP correction); per-trade Sharpe stable at +0.5159 (drift +2.99%); total P&L .63 (drift -14% on smaller test set 1398 vs 1673 rows); 0 embargo violations on purged split (115 train pairs / 29 test pairs); Plan 18-07 ready to resume at Task 3 with new headline numbers"
last_updated: "2026-04-26T15:16:01.927Z"
last_activity: 2026-04-26
progress:
  total_phases: 11
  completed_phases: 9
  total_plans: 33
  completed_plans: 32
  percent: 81
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-17)

**Core value:** Empirically answer whether model complexity improves cross-platform prediction market arbitrage detection
**Current focus:** Phase 18 in progress — adversarial audit of every quantitative claim in PAPER_DRAFT.md / canonical headline.json / slide deck. Plan 18-01 (Wave 0 scaffolding) complete — tests/audit/ infra ready for Wave 1.

## Current Position

Phase: 18 of 18 (System Audit — Adversarial Verification — IN PROGRESS)
Current Plan: 7
Total Plans in Phase: 7
Plan: 1 of 7 complete (18-01 Wave 0 scaffolding: experiments/audit/ + tests/audit/conftest.py + test_fixtures.py 4/4 passing)
Status: Phase 18 Plan 01 (Wave 0 scaffolding) complete — experiments/audit/, experiments/results/audit/, tests/audit/conftest.py with 4 audit-target fixtures, and tests/audit/test_fixtures.py (4 passed in 0.02s). Zero new dependencies. Wave 1 (Plans 18-02, 18-03) ready to start.
Last activity: 2026-04-26

Progress: [████████░░] 81%

## Performance Metrics

**Velocity (from v1.0):**
- Total plans completed: 33
- Average duration: 5.5 min
- Total execution time: ~3 hours

**v1.1 Velocity:**
- Total plans completed: 7
- Average duration: ~4 min
- Total execution time: ~23 min

| Phase                     | Plan | Duration | Tasks | Files |
| ------------------------- | ---- | -------- | ----- | ----- |
| 13-ensemble-formalization | 02   | 2min     | 2     | 7     |

*Updated after each plan completion*
| Phase 13-ensemble-formalization P03 | 48min | 3 tasks | 3 files |
| Phase 14-paper-finalization-presentation P01 | 20min | 3 tasks | 17 files |
| Phase 14-paper-finalization-presentation P02 | 5min | 3 tasks | 2 files |
| Phase 14-paper-finalization-presentation P03 | 90min | 3 tasks | 60 files |
| Phase 15-live-commodity-matching-engineering-fixes P02 | 12min | 2 tasks | 2 files |
| Phase 15-live-commodity-matching-engineering-fixes P01 | 4min | 2 tasks | 1 files |
| Phase 15-live-commodity-matching-engineering-fixes P03 | ~33h (24h SCC wall-clock) | 3 tasks | 9 files |
| Phase 16-paper-revision-phase-15-integration-final-review P01 | 2 min | 3 tasks | 1 files |
| Phase 16-paper-revision-phase-15-integration-final-review P02 | ~8 min | 2 tasks | 1 files |
| Phase 17 P01 | 5min | 2 tasks | 9 files |
| Phase 17-model-rerun-paper-number-audit-pitch-standard-conversion P02 | 18min | 2 tasks | 3 files |
| Phase 17-model-rerun-paper-number-audit-pitch-standard-conversion P03 | 10 min | 2 tasks | 2 files |
| Phase 18-system-audit-adversarial-verification P01 | 2 min | 3 tasks | 5 files |
| Phase 18-system-audit-adversarial-verification P03 | 3min | 2 tasks | 3 files |
| Phase 18-system-audit-adversarial-verification P02 | 4min | 2 tasks | 4 files |
| Phase 18-system-audit-adversarial-verification P04 | 6min | 2 tasks | 3 files |
| Phase 18-system-audit-adversarial-verification P06 | 4min | 2 tasks | 3 files |
| Phase 18-system-audit-adversarial-verification P05 | 5min | 2 tasks | 4 files |
| Phase 18-system-audit-adversarial-verification P08 | 7min | 4 tasks | 9 files |

## Accumulated Context

### Roadmap Evolution

- Phase 18 added: System Audit — Adversarial Verification

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [v1.1 Roadmap]: Phase 8 gates all downstream; venv may need Python 3.12 rebuild
- [v1.1 Roadmap]: TFT time-box 1 day; negative result is paper-worthy (Option B -- always proceed)
- [v1.1 Roadmap]: Reconciliation window Apr 11-25 (post pair_id fix); schema bug confirmed resolved
- [v1.1 Roadmap]: EnsemblePredictor NOT wired into live strategy.py during v1.1
- [v1.1 Roadmap]: Phases 9/10/11/12 parallelize after Phase 8; Phase 13 waits on 11; Phase 14 terminal
- [Phase 08-environment-and-baseline-verification]: Python 3.14.3 venv compatible with pytorch-forecasting 1.7.0, quantstats 0.0.81, SciencePlots 2.2.1; no Python 3.12 rebuild required
- [Phase 08-02]: run_baselines.py now calls compute_derived_features() + select_dtypes(['number']) to align with 51-feature pipeline used in verify_headline.py
- [Phase 08-02]: PPO imports in run_baselines.py are lazy (deferred) to allow Tier 1 to run without stable_baselines3 installed
- [Phase 08-02]: set_all_seeds(42) is the standard entry point seed call for all experiment scripts; P&L reconciliation skipped in ENV-05 check (different profit sim implementations)
- [Phase 09-live-vs-backtest-reconciliation]: profit_sim.simulate_profit is canonical fee function for reconciliation (threshold-only model)
- [Phase 09-live-vs-backtest-reconciliation]: Shadow simulation tracking error of $12.06 is a paper finding: model directional anti-correlation with live entry logic
- [Phase 09-live-vs-backtest-reconciliation]: CLI wrapper delegates 100% of analysis to src/analysis/reconciliation.py — no inline logic
- [Phase 09-live-vs-backtest-reconciliation]: Section 5.9 numbers sourced exclusively from experiments/results/reconciliation/summary.json — no fabrication
- [2026-04-22]: Phase 9 re-run on 8-day snapshot (10,154 trades, 577 pairs); §5.9 and Finding 23 updated; crypto regime flip and TAKE_PROFIT tail (n=88) documented
- [Phase 10-250-bar-scaling-checkpoint]: GRU and LSTM trained successfully at 250 bars (torch env working); XGBoost > LR > GRU > LSTM ranking invariant across 5x data growth (50/100/250 bars)
- [Phase 10-250-bar-scaling-checkpoint]: Auto-trigger root cause: run_data_scaling.py --auto reads only train.parquet (max 141 bars/pair); manual --bars-per-pair 250 required
- [Phase 10-250-bar-scaling-checkpoint]: PYTHONPATH must be set to project root when running scripts from venv
- [Phase 11-01-tft-training]: TFT does NOT beat GRU at N=6802 (RMSE 0.3262 vs 0.2928) — documented negative result per Option B; extends simplicity-wins thesis to transformers
- [Phase 11-01-tft-training]: Use lightning.pytorch Trainer (NOT pytorch_lightning) with pytorch_forecasting 1.7.0 — different LightningModule base class causes TypeError
- [Phase 11-01-tft-training]: Round-based batch predict for TFT: K rounds x all_groups together (~11 rounds) instead of 1673 per-row calls; 150x speedup
- [Phase 11-01-tft-training]: compute_derived_features drops group_id (not in OUTPUT_COLUMNS); must re-attach from raw parquet before prepare_xy_for_seq
- [Phase 11-01-tft-training]: GroupNormalizer(transformation=None) confirmed correct for signed spread-change targets; softplus causes degenerate predictions
- [Phase 11]: TFT negative result (Branch B): converged=False — 4-variant ensemble for Phase 13 (no TFT variant)
- [Phase 11]: VSN attention is healthy (entropy=2.656, not degenerate) even when predictive performance is weak — data-volume bottleneck, not architecture failure
- [Phase 12-01-feature-ablation]: Pre-registration via git commit ordering: ablation_protocol.md committed at b15534b before run_feature_ablation.py at 46b253a (ABLA-01)
- [Phase 12-01-feature-ablation]: All 51-feature groups classified as droppable on ablation_holdout (1,021 rows) — CIs all straddle zero, |delta| < $10; insufficient statistical power at this holdout size
- [Phase 12-01-feature-ablation]: Ablation holdout P&L (+$56.54 LR, +$54.00 XGB) differs from full train-test headline (+$232) due to smaller evaluation window (1,021 vs 6,800 rows)
- [Phase 12-01-feature-ablation]: final_test (test.parquet, 1,673 rows) untouched — frozen for one-shot evaluation in §5.10 paper section
- [Phase 12-02-feature-ablation-paper]: §5.10 written with honest power-limitation framing — N=1,021 ablation holdout insufficient to detect effects < $10; all 51 features retained per pre-registered protocol
- [Phase 12-02-feature-ablation-paper]: Finding 25 documents pre-registered null result; ablation should be re-run at 250+ bars/pair for tighter inference (§7 item 8 added)
- [Phase 13-01-ensemble-formalization]: EnsemblePredictor uses all-members-agree concordance semantics ('strict' mode: np.sign(preds) constant across member axis) — generalizes strategy.py's binary LR/XGB check to N members without pairwise-voting complexity
- [Phase 13-01-ensemble-formalization]: Weight normalization inside predict(), not __init__() — lets sweep runner mutate _weights in-place without re-instantiating (simplifies Plan 13-03)
- [Phase 13-01-ensemble-formalization]: set_all_seeds(seed) called inside fit(), not constructor — constructor stays a pure metadata builder
- [Phase 13-01-ensemble-formalization]: No custom pickle hooks — BasePredictor.save/load handles ensemble serialization natively because all supported member types are already picklable
- [Phase 13-01-ensemble-formalization]: ENSM-05 guard held — src/live/strategy.py untouched; ensemble wiring deferred to post-v1.1 per roadmap decision
- [Phase 13-02-ensemble-formalization]: Per-member feature routing lives at the experiment-script level (fit_mixed_ensemble / predict_mixed_members helpers) — keeps EnsemblePredictor's single-X contract intact and avoids invalidating the 13 tests from Plan 13-01
- [Phase 13-02-ensemble-formalization]: Variant (a) LR-solo P&L = $+201.69 exactly equals variant (c) LR-member P&L (sanity cross-check verified group_id does not leak into LR via variant (c))
- [Phase 13-02-ensemble-formalization]: P4 concordance trap fires on 3/4 filtered variants (rejected P&L +$1.95, +$9.52, +$13.08 for LR+XGB, LR+LSTM, LR+XGB+LSTM) — documented empirically for §5.11
- [Phase 13-02-ensemble-formalization]: Weight sweep P&L spread $4.68 across 11 LR-weight points (0.0→1.0) — weight choice is immaterial; concordance filter is the discriminator
- [Phase 13-02-ensemble-formalization]: ENSM-05 guard re-verified — git diff src/live/strategy.py empty; strategy.py lines 427-429 still contain live concordance check
- [Phase 13-03-ensemble-formalization]: §5.11 reports per-trade Sharpe (0.437 unfiltered vs 0.455 filtered) to quantify concordance filter's Sharpe boost — directly exposes P4 trap magnitude numerically
- [Phase 13-03-ensemble-formalization]: §5.11 framing: weight immateriality first, then concordance filter as true discriminator, then P4 flag with dollar cost — prevents filter being described as pure risk control
- [Phase 13-03-ensemble-formalization]: §4.4 updated via one-paragraph addition (not rewrite); preserves existing architecture narrative; explicitly references ENSM-05 guard (strategy.py untouched)
- [Phase 13-03-ensemble-formalization]: Finding 26 mirrors Finding 25 format exactly; keeps FINDINGS.md scanable rather than a prose collection
- [Phase 14-01-paper-finalization]: Consolidated figure regeneration via scripts/regenerate_figures.py rather than editing 11 separate experiment scripts — keeps IEEE styling in one place and makes regeneration idempotent
- [Phase 14-01-paper-finalization]: TFT VSN weights persisted to experiments/results/tft/vsn_importance.json (via patched extract_tft_heatmap.py) so downstream figure regenerations skip the ~8-minute TFT retrain
- [Phase 14-01-paper-finalization]: Abstract landed at 244 words (6 words of headroom under 250 cap) leaving room for Plan 14-02 adjustments
- [Phase 14-01-paper-finalization]: Figure 2 SCAL-03/POL-08 cap annotation embedded on-figure via red dotted axvline + text ("plateau at N=6,802, fixed pair universe") — readable without body text
- [Phase 14-01-paper-finalization]: Stale "per-trade Sharpe 0.59 / 4.3" claim on §1.4 item 7 + dead §4.6 cross-reference both removed; per-pair ≈ 3.2 (Table 8) is now the consistent headline across abstract, §1.4, §8
- [Phase 14-02-paper-integrity]: [Anonymous] arXiv entry placed as reference #1 (sorts before alphabetical authors); Cont, Kukanov & Stoikov (2014) inserted as #4 to resolve stale in-text citation at §6.2.1 line 536
- [Phase 14-02-paper-integrity]: Rule 1 bug fix during Task 1: §1.4 item 6 stale (Table 6) → (Table 7) for transaction-cost analysis (unrelated to 5.10/5.11 renumber, caught via full grep sweep)
- [Phase 14-02-paper-integrity]: check_paper.sh uses sed -n '/^## X$/,/^## Y$/p' for section ranges not awk '/^## X/,/^## /' — awk range collapses when start pattern also matches the end
- [Phase 14-02-paper-integrity]: Appendix B rewritten in in-text appearance order (11 figures Fig 1-Fig 11); Fig 10 resolves 'Figure 2b' ghost; Fig 11 resolves '[Insert Figure]' placeholder
- [Phase 14-02-paper-integrity]: §6.4 now has 8 items: preserved original 1-5 (POL-08 regression check); added live-cohort truncation (6), category-tagging gaps (7), crypto regime flip (8) from Finding 23
- [Phase 14-03-readme-slides]: README.md is the canonical reproduction map; PAPER_DRAFT.md Appendix A deprecated to a single pointer — prevents drift between paper commands and code
- [Phase 14-03-readme-slides]: Marp CLI (Node toolchain) chosen over Beamer/LaTeX for lightning-talk slides — faster iteration, PDF committed alongside source
- [Phase 14-03-readme-slides]: Two Rule 1/2 fix cycles at human-verify checkpoint — added §5.12 (lookback) and §5.13 (threshold) body writeups to close orphan-table references; repaired Appendix B Fig 10/Fig 11 placeholders missed in 14-02 renumber
- [Phase 14-03-readme-slides]: Abstract oil claim narrowed to "where the matched cohort is representable" — honest scope given ~30% cross-platform matching rate; §6.4 item 9 documents commodity-matching cohort as generalization limit
- [Phase 14-03-readme-slides]: POL-09 upgraded from Acknowledgments-only disclosure to per-file AI-attribution headers on 56 src/**/*.py modules — survivorship-proof evidence of AI involvement
- [Phase 15-02-rule10-asset-class]: Rule 10 uses asymmetric-confidence semantics — rejection requires BOTH sides to produce confident signals; ambiguous evidence (neither side has an asset-class token) falls through to existing rules so non-financial markets aren't disturbed
- [Phase 15-02-rule10-asset-class]: Kalshi ticker prefix wins over title tokens in _detect_asset_class because tickers are authoritative and titles can contain stray numeric strings
- [Phase 15-02-rule10-asset-class]: Canonical hex poly_id 0x885a6abefad122348b4fbd503473d7fd1f9035d0438cf988a7591620f316a859 pinned verbatim in test file for grep-based audit of the specific bug
- [Phase 15-02-rule10-asset-class]: Pre-existing sentence_transformers import error in test_pipeline/test_scorer/test_semantic_matcher is out-of-scope per scope-boundary rule — confirmed not caused by Rule 10 changes; the 104 other matching tests all pass
- [Phase 15-live-commodity-matching-engineering-fixes]: H1 (KALSHI_DISCOVERY_CATEGORIES missing Commodities) is the root cause of daily WTI / Brent absence — single-line tuple edit at src/live/market_discovery.py:249 unblocks ~125 oil/brent open markets; H5 PARTIAL requires classifier extension in src/features/category.py for Brent + WTI variants
- [Phase 15-03-discovery-fix]: H1 (discovery category gap) confirmed as root cause — daily WTI / Brent / grain / metal series never reached the pipeline because `KALSHI_DISCOVERY_CATEGORIES` in src/live/market_discovery.py omitted "Commodities" (Kalshi migrated these series into a dedicated Commodities category after the taxonomy change). Primary fix: add "Commodities" to the tuple at src/live/market_discovery.py:249. Secondary fixes: extend `_RULES` in src/features/category.py with Brent family + daily-WTI variants + KXCRUDE/KXDIESEL/KXHEATINGOIL/KXGASOLINE prefixes; reserve 200 slots for commodity pairs inside `_load_live_pairs` MAX_LIVE_PAIRS=2000 cap in src/live/collector.py to prevent similarity-cap eviction (commodity pairs cluster at similarity ≈ 0.794, well below the sports/politics-dominated 0.85+ tail). Regression test tests/matching/test_rule_10_asset_class.py still guards the KXWTIMAX-26DEC31-T130 ↔ Bitcoin-$130K false match (COM-02). Post-fix validation: 336 active non-KXWTIMAX commodity pairs in active_matches.json (COM-03), 200 non-KXWTIMAX commodity pairs in pair_mapping.json + **1,224 closed commodity positions in position_history.jsonl after 24h SCC window (122x COM-04 target)**. Per-series breakdown: KXBRENTW=486, KXWTI=409, KXWTIW=213, KXBRENTMON=76, KXBRENTD=16, KXAAAGASD=11, KXAAAGASW=7, KXAAAGASM=6. Aggregate P&L +$1.96, win rate 36.0%.
- [Phase 15-03-discovery-fix]: The similarity-cap eviction (200-slot commodity reservation in _load_live_pairs) was NOT predicted by the 15-01 diagnostic — it only became visible after H1 flooded the candidate pool with new commodity pairs and the pair_mapping.json dropped all of them despite active_matches.json containing 336. Rule 3 (blocking) auto-fix, not a checkpoint, because the 24h window would have closed zero positions otherwise.
- [Phase 16-paper-revision-phase-15-integration-final-review]: [Phase 16-01-paper-revision]: Preserved original 'Finding: WTI oil contracts absent' paragraph (§5.9) and original §6.4 item 9 engineering-gap text verbatim — new §5.9.1 subsection and the **Resolved post-submission in Phase 15** note appended rather than substituted. Scientific-integrity principle: acknowledged v1.1 limitations stay visible in the paper even after post-submission resolution. Abstract trimmed via two cuts (drop redundant 'autonomous paper-trader' sentence −9 words + drop 'essentially' filler −1 word) to offset the +6-word live-validation qualifier insertion; lands at 247/250 with 3 words of POL-04 headroom. §5.9.1 ships three explicit caveats (short 12h window, near-flat per-trade economics, paper-trading idealizations persist) to prevent the cohort being read as a robust live edge measurement. check_paper.sh still exits 0 with all 16 checks OK — zero Phase 14 guardrail regressions. Atomic commits: 526370e (REV-01) / 637427f (REV-02) / f8d7acb (REV-03).
- [Phase 16-02-findings-integration]: Finding 6 companion paragraph appended INSIDE the existing Finding 6 block (after Implication sentence, before `---` separator) using a labeled `**Live validation (Phase 15, 2026-04-24).**` lead-in rather than fragmenting into a separate finding — preserves the backtest claim + table verbatim while dating the live-vs-backtest delta. Finding 27 framed as four bold-led sub-paragraphs (What happened / Parallel to Finding 8 / Methodology lesson / Implication) at 313 body words rather than the 80-120-word single-paragraph literal target from REV-05 — matches Finding 26's formatting style and keeps the concrete "known-unknown monitoring" recommendation scannable. Cross-reference to Finding 8 explicitly names the prior April 11 silent-starvation bug so the two-instances-in-same-subsystem structural-weakness pattern is visible to paper readers. File-ownership held against parallel Plan 16-01: my commits c3b1181 + 97fb616 touched ONLY FINDINGS.md; 16-01's PAPER_DRAFT.md commit 526370e landed between them without conflict. REV-04 and REV-05 satisfied; remaining Phase 16 work is Plan 16-03 (final review checkpoint).
- [Phase 17-01-canonical-results]: Pragmatic Tier 2/3 ingest from existing per-tier JSONs (seed=42, threshold=0.02, 51 features) instead of retrain — avoids 4+hr PPO retrain when canonical-protocol metadata is already identical. Tier 1 retrained from scratch every invocation (<30s) so the script proves canonical protocol is end-to-end reproducible. alpha_bps_per_trade denominator is position_size ($100), not total notional, making units directly comparable to fixed-size momentum strategy per-trade edge.
- [Phase 17-01-canonical-results]: 600× PPO discrepancy root cause: units mismatch between two valid simulators. profit_sim (canonical) returns raw probability-point spread P&L; WalkForwardBacktester (legacy) returns dollar P&L for $100-notional fixed-size strategy which sizes each trade as num_contracts = $100 / mid_price ≈ 200 contracts and applies 5pp round-trip fees. Decomposes as ~200× position scaling × ~3× mid_price-driven contract-count inflation on low-priced commodity contracts trading at $0.10–$0.30. Not a code bug — both simulators are correct for the question they ask; the paper accidentally cited a legacy figure (in dollars) when its surrounding text used canonical figures (in spread units).
- [Phase 17-01-canonical-results]: Unreproducible "−$7,724" paper claim diagnosed as most likely transcription typo of "−$87,724" (drop the 8). Off-by-11.4× is too close to "missing a digit" for coincidence given no JSON file contains −$7,724. Disputed legacy backtest PPO JSONs (backtest/ppo_raw.json +$96,336.84 and backtest/ppo_filtered.json −$87,723.84) git-mv'd to experiments/results/archive/ with quarantine README. REPL-07 codifies going-forward convention: all paper numerics derive from experiments/results/canonical/headline.json only.
- [Phase 17-02-paper-numerics-audit]: Auditor scope = headline sections only (Abstract / §5.1 / §6.3 / §8 Conclusions); per-window/per-category/sweep/ablation sections have their own non-canonical result files and are skipped. Restricting scope reduced 53 spurious mismatches to 0.
- [Phase 17-02-paper-numerics-audit]: Per-number proximity model attribution (find_model_at_position): each numeric match resolves its model from the closest alias on the same line within 80 chars; falls back to paragraph context only when out of range. Fixes multi-model line attribution in §8 Conclusions list.
- [Phase 17-02-paper-numerics-audit]: LR row #1, XGB row #2 retained in Table 2 (plan suggested swap). Canonical/headline.json shows LR wins 4 of 5 metrics (per-trade Sharpe 0.501 vs 0.499, alpha 15.02 vs 14.93 bps, dir-acc 56.9 vs 56.6, win-rate 57.8 vs 57.4); XGB only wins RMSE and total_pnl by hair. Documented in §5.1 narrative.
- [Phase 17-02-paper-numerics-audit]: Pitch-standard headline format adopted everywhere: '{Model} achieves a per-trade Sharpe of {X.XXX} with +{Y.Y} bps per-trade alpha (positive cumulative dollars at 100 USD position size)'. Sharpe leads, bps second, dollars in parens. Abstract / Table 2 / section 5.1 narrative / 5.8 / 6.3 / 8 Conclusions all converted.
- [Phase 17-model-rerun-paper-number-audit-pitch-standard-conversion]: Slide ordering: LR row first, XGB second (matches paper precedent — canonical wins 4 of 5 metrics) — Per-trade Sharpe 0.501 vs 0.499; per-trade alpha 15.02 vs 14.93 bps; dir-acc 56.9% vs 56.6%; win rate 57.8% vs 57.4%. XGB only wins RMSE and total_pnl by hair.
- [Phase 17-model-rerun-paper-number-audit-pitch-standard-conversion]: REPL-06c orphan-dollar regex tightened to signed P&L of $50+ in 5 headline sections — Plan example regex flagged 18+ false positives (setup mentions, table narrative). Tighter regex catches every model headline P&L while filtering setup context. Sections: Abstract, sec5.1, sec5.8, sec6.3, sec8 Conclusions.
- [Phase 17-01-canonical-rerun]: experiments/results/canonical/headline.json is the single source of truth for every numeric claim in PAPER_DRAFT.md. Produced by experiments/run_canonical.py with seed=42, position_size=$100, threshold=0.02, train=6802/test=1673. The 9-model entry contains rmse, mae, directional_accuracy, total_pnl, num_trades, win_rate, sharpe_per_trade, sharpe_annualized, alpha_bps_per_trade, max_drawdown_pct. No paper or slide may cite a JSON file outside experiments/results/canonical/.
- [Phase 17-02-PPO-diagnostic]: PPO 600× magnitude divergence root cause: units mismatch between two valid simulators with different units, compounded by mid_price-driven contract-count inflation when position_size is held fixed at $100. profit_sim.simulate_profit (canonical) returns raw probability-point spread P&L with no fees and no position sizing; WalkForwardBacktester.run (legacy) returns dollar P&L for a $100-notional fixed-size strategy that computes num_contracts = $100 / mid_price ≈ 200 per trade (because contracts trade at ~$0.50 mid), then gross_pnl = num_contracts × actual_change × direction minus 5pp round-trip fees. The 609× legacy/canonical ratio decomposes as ~200× position scaling × ~3× mid_price-driven contract-count inflation on low-priced commodity contracts that trade at $0.10–$0.30. Both simulators are correct for the question they ask; the paper accidentally cited a legacy figure (in dollars) when its surrounding text used canonical figures (in spread units). It is not a code bug, a training mismatch, or an episode-horizon issue. See .planning/phases/17-model-rerun-paper-number-audit-pitch-standard-conversion/17-02-PPO-DIAGNOSTIC.md for the full diagnostic. The unreproducible "−$7,724" paper claim from earlier drafts is documented as not present in any extant JSON file (likely a transcription error of the legacy −$87,724 figure — drop the 8). Disputed files moved to experiments/results/archive/backtest_ppo_*.json with quarantine README.
- [Phase 17-02-paper-audit]: scripts/audit_paper_numbers.py is the automated cross-reference tool — runs paper-vs-canonical-JSON regex extraction with section-aware filtering (headline sections only: Abstract / §5.1 / §6.3 / §8), per-number proximity model resolution, and tolerance-based comparison. Outputs match/mismatch/unresolvable Markdown log; exits 0/1 on mismatch presence. Re-run any time canonical numbers change.
- [Phase 17-02-pitch-standard]: Paper headlines converted from dollar P&L to per-trade Sharpe + per-trade alpha (bps) — the professional quant pitch standard. Per-trade alpha formula: (total_pnl / num_trades / position_size) × 10,000. Worked: LR = $232.67 / 1549 / $100 × 10000 = 15.0 bps. PPO+AE = $4.61 / 899 / $100 × 10000 = 0.5 bps. Abstract, Tables 2 + 8, §5.1, §5.8, §6.3, §8 Conclusions all updated. Dollar P&L stays in tables only.
- [Phase 17-03-slide-validator]: slides_deck.html Results slide leads with bps + Sharpe (canonical PPO numbers, side panels per-pair Sharpe ≈ 3.2 and 11/11 walk-forward preserved). PPO+AE bar added to main chart (41 px, red, 0.5 bps) — the visually-negligible bar makes tier-3 collapse self-evident. scripts/check_paper.sh extended with 3 REPL-06 checks: abstract_mentions_sharpe, abstract_cites_sharpe_value, orphan_dollar_paragraphs (no headline-section paragraph cites a dollar amount of $50+ without a Sharpe or bps companion). Total: 19/19 OK.
- [Phase 17-04-closure]: Pitch-standard adopted as the project's house style for all future result presentations. Canonical results JSON pattern (one file per phase under experiments/results/canonical/) is the convention going forward. v1.1 milestone fully shipped 100%.
- [Phase 18-01-wave-0-scaffolding]: Audit fixtures live in tests/audit/conftest.py with one fixture per audit-target failure mode (Tier 1 perfectly_correlated_pair_returns, Tier 2 synthetic_lookahead_feature_src, Tier 3 zero_fee_simulator_kwargs, Tier 4 retroactive_drop_pair_history). Wave 1+2 audit scripts in Plans 18-02..18-05 will import these via pytest auto-discovery — no fixture duplication, no per-script test infrastructure setup. Fixture bodies copied VERBATIM from 18-RESEARCH.md (the research doc is the contract). Zero new dependencies — pytest 7.x already pinned; arch (StationaryBootstrap) deferred to Wave 1 only if AR(1) > 0.1 in pnl_pp.
- [Phase 18-01-wave-0-scaffolding]: Tier 2 leakage fixture is a Python source CODE STRING (synthetic_lookahead_feature_src returns text containing 'shift(-1)'), not a callable — lets the Tier 2 classifier validate via static-text regex without exec()-ing untrusted code at test time.
- [Phase 18-01-wave-0-scaffolding]: experiments/results/audit/.gitkeep contains a single comment line ('# Phase 18 audit JSON outputs land here.') so the empty results dir is git-tracked while remaining functionally empty for the audit JSONs.
- [Phase 18-system-audit-adversarial-verification]: Plan 18-03 Tier 2 leakage audit: verdict=FAILED — empirically confirmed canonical 80/20 row-index split bridges all 144 pairs (n_bridging=144, n_embargo_violations=142 with 4h gap << 86400s). The high-yield finding RESEARCH.md predicted is now locked in experiments/results/audit/leakage_audit.json. n_leaking=0 (no feature-level leaks), 7 Suspicious entries (trailing rolling, manual sign-off OK), Rule stale_ticker flagged retroactive=True (benign for 2026 audit-time). Plan 18-07 must either recompute headline numbers under leakage-free split or document the limitation in PAPER_DRAFT.md §6.4.
- [Phase 18-system-audit-adversarial-verification]: Audit-script JSON output schema standardized: {audit, tier, verdict, ran_at, ...findings, assumptions[]} with verdict ∈ {PASS, CORRECTED, FAILED} where FAILED triggers if any single check fails. Suspicious findings do NOT trigger FAILED — only Leaking, embargo violations, and retroactive QF rules do. This is the Pattern 2 contract every Tier 2-7 audit script will follow so Plan 18-07 can mechanically aggregate them.
- [Phase 18-system-audit-adversarial-verification]: Plan 18-02 Tier 1 Sharpe audit verdict=PASS: per_trade_drift=0.00016 (canonical 0.5009, recomputed 0.5007); per_pair_naive=0.781, corr_corrected=0.296 (-62% under avg_pair_corr=0.042 BLdP correction); bootstrap CI [0.685, 0.904]; annualization factor 23.8 from test_span_days=92.67 — does NOT match paper's implicit ≈3.2 factor (Plan 18-06 follow-up to reconcile)
- [Phase 18-system-audit-adversarial-verification]: Plan 18-02 Rule 1 auto-fix: RESEARCH.md skeleton assumed timestamp was datetime64[ns]; canonical processed parquet stores it as int64 epoch seconds — added dtype-aware normalization with heuristic (>10**12 -> nanoseconds) to recover real entry_day axis (was collapsing to test_span_days=0)
- [Phase 18-system-audit-adversarial-verification]: Plan 18-04 Tier 3 cost-realism audit verdict=CORRECTED: simulate_profit charges zero fees confirmed (PAPER_DRAFT.md §5.1 line 213/215 prose says '2pp transaction costs' but the 0.02 is a SIGNAL gate, not a fee — the load-bearing paper bug). WalkForwardBacktester charges 3pp+2pp = 5pp = 500bps round-trip, OVER-conservative vs realistic Kalshi+Polymarket (~250-355bps). Slippage sweep at 0/5/10/20/50 bps additional haircut: annualized Sharpe 8.95 -> 8.81 (-1.6%), P&L ,728 -> ,964 (-3.8%) — cost-robustness claim survives. paper_corrections_required has 2 items: §5.1 prose fix + §6.4 fee schedule documentation (Plan 06 + Plan 07 consume).
- [Phase 18-06]: MISMATCH-as-finding pattern: Plan 18-06 traces numbers and pre-flags discrepancies in paper_numbers.csv; does NOT modify PAPER_DRAFT.md (that is Plan 18-07's role)
- [Phase 18-06]: SHARPE_LONG_RE addition (200-char window) needed to capture abstract's per-pair Sharpe ≈ 3.2 long-form claim that the verbatim RESEARCH.md SHARPE_RE 30-char window missed; deduplicated against SHARPE_RE via end-position keys
- [Phase 18-06]: Two MISMATCH rows pre-flagged for Plan 18-07: per-pair annualized Sharpe ≈ 3.2 (Abstract line 12 + §8 Conclusions line 710) traces to sharpe_audit.json which reproduced 18.60 naive / 7.04 BLdP-corrected — 3.2 does NOT appear in any audit JSON; likely outdated copy from prior backtest
- [Phase 18-system-audit-adversarial-verification]: [Phase 18-05-survivorship-audit]: Tier 4 verdict=PASS — heuristic drop_rate=0.999 (n_dropped=148094/n_candidates=148238) with all 10 random-sample entries classified low_overlap_n_bars=0 (live-discovery candidates from late April 2026 that postdate the test.parquet 2026-04-09 snapshot, never entered offline pipeline). Final verdict pending Ian's manual review per Plan 18-07 checkpoint. RESEARCH.md skeleton extended with make_pair_id synthesis from active_matches.json (kalshi_ticker+poly_id) because schema lacks top-level pair_id key — without synthesis n_dropped would be 0 (vacuous audit).
- [Phase 18-08-leakage-free-recompute]: Verdict CORRECTED on purged Tier 1 Sharpe audit (Plan 18-08): leakage-free per-pair corrected Sharpe is +0.81 [CI +0.70, +1.07], a +175% INCREASE over leaky-canonical +0.30. Mechanism: leaky avg pairwise correlation +0.0418 compressed leaky naive 0.78 to corrected 0.30 via the BLdP effective-sample-size correction; purged avg pairwise correlation is -0.1986 on the smaller N=29 test pairs, which short-circuits the BLdP correction (avg_corr <= 0 returns naive Sharpe unchanged) so purged corrected = purged naive = 0.81. The +175% drift fails the PASS gate's |drift_pct| < 50% requirement, but the direction is favorable (purged HIGHER than leaky) and the CI is entirely positive. Per-trade Sharpe is invariant to leakage correction: +0.5007 -> +0.5159 (drift +2.99%). Per-pair NAIVE Sharpe drift is also small (+4.20%); only the BLdP-corrected number moves dramatically because the correction's applicability flips between the two regimes. Honest paper framing: per-trade edge is robust to leakage correction; per-pair Sharpe headline depends on cross-pair correlation regime which is sample-size-sensitive at N=29 test pairs. The 3.2 number that motivated this plan continues to have no derivation path; the leakage-free replacement is +0.81 corrected per-pair (or +0.5159 per-trade depending on which framing the abstract adopts).
- [Phase 18-08-leakage-free-recompute]: Pair-stratified split mechanics (data/processed/purged_split/): concatenate canonical train+test (8763 rows / 144 pairs), shuffle unique pair_id list with np.random.default_rng(seed=42).shuffle on a numpy object array (avoids ArrowStringArray UserWarning), atomic 80/20 by pair count -> 115 train pairs / 29 test pairs. After canonical _build_split (drops trailing-bar NaN targets) the row counts are 7221 train / 1398 test. Zero pair_id intersection between halves by construction. data/processed/ is gitignored so the parquets are regenerated on demand; split_metadata.json records the pair_id assignments for traceability.
- [Phase 18-08-leakage-free-recompute]: Sister-script reuse pattern: experiments/run_canonical_purged.py imports evaluate_predictions + canonical constants from run_canonical, and _build_split + _feature_columns + prepare_xy from run_baselines — swaps ONLY the data source. experiments/audit/audit_sharpe_purged.py imports per_pair_returns + per_pair_sharpe_naive + avg_pairwise_correlation + correlation_corrected_sharpe + bootstrap_sharpe_ci + annualization_factor + per_trade_sharpe from audit_sharpe — swaps ONLY the ledger source. The reuse rule (DO NOT duplicate audit metric code) is enforced by a test (test_audit_sharpe_purged.py::test_purged_audit_reuses_canonical_helpers) that asserts the imported functions are the SAME OBJECT as in audit_sharpe.py. This pattern is the canonical way to add a parametric variant of an existing audit/training script in this codebase going forward.

### Pending Todos

- Phase 18 Wave 1: Plan 18-02 (Tier 1 Sharpe audit) and Plan 18-03 (Tier 2 leakage audit) — can run in parallel
- Phase 18 Wave 2: Plans 18-04 (Tier 3 cost), 18-05 (Tier 4 survivorship), 18-06 (Tier 5 paper number trace)
- Phase 18 Wave 3: Plan 18-07 (AUDIT_REPORT.md + conditional paper/slide updates)

### Blockers/Concerns

- [Risk] Python 3.14 may lack pytorch-forecasting wheels; Phase 8 dry-run gates everything
- [Risk] TFT at 6,802 rows is below safe training thresholds; 1-day time-box enforced
- [Deadline] April 27 submission -- 10 days from roadmap creation

## Session Continuity

Last session: 2026-04-26T15:15:37.118Z
Stopped at: Completed 18-08-PLAN.md (purged pair-stratified leakage-free retraining + Tier 1 Sharpe audit redo) — verdict CORRECTED: purged corrected per-pair Sharpe +0.81 [CI 0.70, 1.07], +175% drift from leaky 0.30 (driven by negative purged avg_corr short-circuiting BLdP correction); per-trade Sharpe stable at +0.5159 (drift +2.99%); total P&L .63 (drift -14% on smaller test set 1398 vs 1673 rows); 0 embargo violations on purged split (115 train pairs / 29 test pairs); Plan 18-07 ready to resume at Task 3 with new headline numbers
Resume file: None
Next action: Begin Phase 18 Wave 1 — Plans 18-02 (Tier 1 Sharpe audit) and 18-03 (Tier 2 leakage audit) can run in parallel. Both have their fixtures pre-loaded in tests/audit/conftest.py.
