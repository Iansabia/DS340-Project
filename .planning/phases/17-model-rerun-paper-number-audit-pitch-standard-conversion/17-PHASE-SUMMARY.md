# Phase 17: Model Rerun + Paper Number Audit + Pitch-Standard Conversion — PHASE SUMMARY

**Phase ID:** 17-model-rerun-paper-number-audit-pitch-standard-conversion
**Status:** Complete (4/4 plans)
**Duration:** 36 min total (5 + 18 + 10 + 3)
**Started:** 2026-04-25T17:40:18Z
**Completed:** 2026-04-25T18:15:18Z
**Requirements:** REPL-01 through REPL-07 (all 7 Complete)

---

## One-line Per Plan

| Plan | Outcome | Commits |
|---|---|---|
| **17-01** Canonical model rerun + PPO diagnostic | Wrote `experiments/run_canonical.py` (533 lines); produced `experiments/results/canonical/headline.json` (9 models, 13 metric fields each, under documented protocol seed=42 / position=$100 / threshold=0.02 / train=6,802 / test=1,673); diagnosed 600× PPO divergence as units mismatch (`profit_sim` raw spread units vs `WalkForwardBacktester` $100-notional dollars with ~200× contract scaling and 5pp fees) and ~3× mid_price-driven contract-count inflation; archived disputed legacy backtest PPO JSONs to `experiments/results/archive/` with quarantine README | db6fb4a, 5212e14, 9fa60a0 |
| **17-02** Paper numerics audit + pitch-standard conversion | Wrote `scripts/audit_paper_numbers.py` (557 lines, section-aware filtering + per-number proximity model attribution + tolerance-based comparison; exits 0/1 on mismatch presence); audited PAPER_DRAFT.md against `canonical/headline.json` — 18 match / 0 mismatch / 7 unresolvable (auxiliary, correctly out of scope); converted Abstract / Tables 2 + 8 / §5.1 / §5.8 / §6.3 / §8 Conclusions to pitch-standard format (per-trade Sharpe + per-trade alpha in bps lead; cumulative dollars in tables only); replaced disputed `−$7,724` PPO+AE figure with canonical `+$4.61 / +0.5 bps` everywhere; all 16 Phase 14 check_paper.sh guardrails still pass | 232b47b, 62c3368, 8c34f6d |
| **17-03** Slide pitch-standard conversion + check_paper.sh REPL-06 regression checks | Rebuilt `slides_deck.html` Results slide around per-trade alpha (bps) bar chart with Sharpe-leading text labels; added 5th bar for PPO+AE (41 px / 0.5 bps / red) so tier-3 collapse is visually self-evident; added explicit footnote citing 17-02-PPO-DIAGNOSTIC.md to resolve legacy −$87K vs canonical +$4.61 in one glance; preserved per-pair Sharpe ≈ 3.2 and 11/11 walk-forward side panels; extended `scripts/check_paper.sh` from 16 to 19 checks via 3 REPL-06 pitch-standard regression guards (abstract_mentions_sharpe, abstract_cites_sharpe_value, orphan_dollar_paragraphs_in_headline_sections); final 19/19 OK | af04c3c, 18e79ba, c9567fe |
| **17-04** Phase closure (STATE/ROADMAP/REQUIREMENTS) | STATE.md frontmatter advanced to 100% / status complete / completed_phases 10 / completed_plans 25; 6 new Phase 17 decision entries appended (canonical-rerun, PPO-diagnostic with verbatim units-mismatch root cause, paper-audit, pitch-standard, slide-validator, closure); ROADMAP.md Phase 17 entry marked 4/4 Complete with Resolution summary block; v1.1 milestone checklist Phases 15/16/17 entries added; phase-tracking table Phases 16/17 rows added; REQUIREMENTS.md REPL-07 flipped Pending → Complete (now 7/7 REPL Complete); v1.1 milestone fully shipped 100% | a20b40b, d450ff0 |

---

## Final Canonical Numbers (from `experiments/results/canonical/headline.json`)

All 9 models under canonical protocol (seed=42, position=$100, threshold=0.02, train=6,802 / test=1,673):

| Model | Total P&L ($) | # Trades | Per-trade Sharpe | Per-trade alpha (bps) | RMSE | Dir. Acc. | Win Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Linear Regression** | **+$232.67** | 1,549 | **0.501** | **+15.0** | 0.306 | 56.9% | 57.8% |
| **XGBoost** | **+$232.83** | 1,559 | 0.499 | +14.9 | 0.290 | 56.6% | 57.4% |
| **LSTM** | +$221.84 | 1,547 | 0.473 | +14.3 | — | — | — |
| **GRU** | +$212.50 | 1,517 | 0.459 | +14.0 | — | — | — |
| **TFT** (converged=false) | +$6.57 | 120 | 0.155 | +5.5 | — | — | — |
| **PPO-Raw** (canonical) | +$158.15 | 1,656 | 0.31 | +9.6 | — | — | — |
| **PPO+autoencoder** (canonical) | **+$4.61** | 899 | **0.014** | **+0.5** | — | 27.2% | 43.2% |
| **Naive baseline** | +$58.12 | 1,460 | — | +4.0 | — | — | — |
| **Volume baseline** | +$59.81 | 1,440 | — | +4.2 | — | — | — |

**Pitch-standard headline (in paper Abstract):**
> Linear Regression achieves a per-trade Sharpe of 0.501 with +15.0 bps per-trade alpha at $100 position size; XGBoost lands at Sharpe 0.499 with +14.9 bps. PPO+autoencoder (Tier 3) collapses to Sharpe 0.014 / +0.5 bps — essentially zero alpha — answering the central research question: added complexity is not justified at this dataset scale.

**Disputed `−$7,724` paper claim:** diagnosed as a transcription typo of legacy `−$87,723.84` (drop the 8). No JSON file contains `−$7,724`. The legacy `−$87K` figure itself is documented in 17-02-PPO-DIAGNOSTIC.md as a units-mismatch artifact (canonical PPO+AE is `+$4.61`); legacy file archived at `experiments/results/archive/backtest_ppo_filtered.json`.

---

## Validator Status

**`bash scripts/check_paper.sh`:** 19/19 OK (was 16/16 pre-Phase-17; 3 new REPL-06 checks added):

```
== POL-04..POL-10 (Phase 14 baseline) ==
  16 checks all OK

== REPL-06: Pitch-standard headlines (Phase 17) ==
  [OK]   abstract_mentions_sharpe                           (got 1, want >= 1)
  [OK]   abstract_cites_sharpe_value                        (got 1, want >= 1)
  [OK]   orphan_dollar_paragraphs_in_headline_sections      (got 0)

ALL CHECKS PASSED
Exit: 0
```

**`python3 scripts/audit_paper_numbers.py`:** 18 match / 0 mismatch / 7 unresolvable (auxiliary; out of scope) — exit 0.

---

## Key Reference Documents

| Doc | Path | What it contains |
|---|---|---|
| **PPO root-cause diagnostic** | `.planning/phases/17-model-rerun-paper-number-audit-pitch-standard-conversion/17-02-PPO-DIAGNOSTIC.md` | 200-line written explanation of why `backtest/ppo_*.json` differs ~600× in magnitude from `tier3/ppo_*.json` at the same documented position size; identifies the discrepancy as units mismatch + mid_price-driven contract-count inflation; provides quantitative reconciliation; addresses the unreproducible `−$7,724` paper claim |
| **Number audit log** | `.planning/phases/17-model-rerun-paper-number-audit-pitch-standard-conversion/17-03-NUMBER-AUDIT.md` | This run's audit log: 18 match / 0 mismatch / 7 unresolvable; tolerances documented inline (dollars ±0.5%, Sharpe ±0.01, RMSE ±0.005, trades ±1%, bps ±0.5) |
| **Canonical results JSON** | `experiments/results/canonical/headline.json` | Single source of truth for every numeric claim in PAPER_DRAFT.md; 9 models × 13 metric fields each |
| **Canonical regenerator** | `experiments/run_canonical.py` | 533-line single script that reproduces `headline.json` end-to-end in <30 seconds |
| **Paper numeric auditor** | `scripts/audit_paper_numbers.py` | 557-line section-aware paper-vs-canonical numeric auditor; exits 0/1 on mismatch presence |
| **Validator** | `scripts/check_paper.sh` | 113 lines, 19 checks (16 POL-04..POL-10 + 3 REPL-06); exits 0 on all-pass |
| **Quarantine notice** | `experiments/results/archive/README.md` | Explains why disputed legacy backtest PPO JSONs cannot be cited |

---

## Phase 17 in Context

**v1.1 Extended Evidence & Submission milestone:** **100% shipped.** April 27, 2026 paper submission is locked.

Phase 17 closed three issues that surfaced during Phase 16 final-review:
1. **PPO data inconsistency** (paper claimed `−$7,724` but extant JSONs showed `+$4.61` / `−$29.41` / `−$87,724` across different files) — resolved with canonical-results JSON convention
2. **Numeric reproducibility risk** (no single regenerator script existed; numbers in paper could not be re-derived end-to-end) — resolved with `experiments/run_canonical.py` + `scripts/audit_paper_numbers.py`
3. **Headline format mismatch** (dollar P&L was the abstract headline despite the project's quant-pitch context where Sharpe + bps is the standard) — resolved with pitch-standard adoption across paper, slide, and validator

**Going-forward conventions codified in STATE.md:**
- Canonical results JSON pattern: one file per phase under `experiments/results/canonical/`
- Pitch-standard format as house style: per-trade Sharpe + per-trade alpha (bps) lead; cumulative dollars in tables only
- Per-trade alpha formula: `(total_pnl / num_trades / position_size) × 10,000`

---

*Phase 17 closed: 2026-04-25*
*v1.1 milestone shipped: 2026-04-25*
*Paper submission deadline: 2026-04-27*
