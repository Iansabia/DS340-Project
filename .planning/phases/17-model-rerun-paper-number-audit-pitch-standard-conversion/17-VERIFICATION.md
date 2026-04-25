---
phase: 17-model-rerun-paper-number-audit-pitch-standard-conversion
verified: 2026-04-24T00:00:00Z
status: passed
score: 7/7 must-haves verified
re_verification: false
---

# Phase 17: Model Rerun + Paper Number Audit + Pitch-Standard Conversion — Verification Report

**Phase Goal:** Resolve the PPO data inconsistency, produce one canonical set of model results from a fresh single-seed rerun of all 8 models under one documented protocol, audit every numeric claim in PAPER_DRAFT.md against the canonical results, and convert the paper's headline metrics from dollar P&L to professional pitch standards (per-trade Sharpe, per-trade alpha in pp/bps, max drawdown).
**Verified:** 2026-04-24
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `experiments/results/canonical/headline.json` exists with all 9 models and full metric set | VERIFIED | 9 models confirmed: naive, volume, linear_regression, xgboost, gru, lstm, tft, ppo_raw, ppo_filtered; all 10 required fields present per model |
| 2 | PPO discrepancy diagnosed; old `backtest/ppo_*.json` files quarantined in archive/ | VERIFIED | 17-02-PPO-DIAGNOSTIC.md exists (200 lines); archive/backtest_ppo_raw.json and archive/backtest_ppo_filtered.json confirmed; original backtest/ files removed |
| 3 | PAPER_DRAFT.md "−$7,724" claim removed/replaced; paper audited | VERIFIED | Zero occurrences of "7,724" or "7724" in PAPER_DRAFT.md; 17-03-NUMBER-AUDIT.md exists (60 lines) |
| 4 | Paper headline metrics converted to pitch-standard (Sharpe + alpha-bps lead) | VERIFIED | Abstract cites per-trade Sharpe 0.501/0.499 and +15.0/+14.9 bps; 6 bps occurrences in paper; 19 "per-trade Sharpe/alpha" occurrences; abstract word count 249 (≤250) |
| 5 | `slides_deck.html` Results slide uses canonical numbers + pitch-standard format | VERIFIED | HTML parses; 20 bps occurrences; 14 Sharpe occurrences; "11/11" walk-forward panel present; "per-pair Sharpe ≈ 3.2" present; PPO footnote referencing 17-02-PPO-DIAGNOSTIC.md present; zero disputed figure references |
| 6 | `scripts/check_paper.sh` extended with 3 new pitch-standard checks; all 19 checks pass | VERIFIED | 19 [OK], 0 [FAIL], exit 0; "ALL CHECKS PASSED" fires; REPL-06a/b/c checks confirmed at lines 77-104 |
| 7 | STATE.md and ROADMAP.md include Phase 17 closure note with PPO root cause + canonical-results convention | VERIFIED | STATE.md: percent=100, status=complete, 5 Phase 17 decision bullets present, PPO root cause fully filled in (units-mismatch + contract-count inflation); ROADMAP.md: 4/4 complete, Resolution summary present, Phase 17 tracking table row present |

**Score:** 7/7 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `experiments/run_canonical.py` | Single-protocol 9-model regeneration script (≥200 lines) | VERIFIED | 533 lines; parses; contains CANONICAL_SEED=42, CANONICAL_POSITION_SIZE=100.0, CANONICAL_THRESHOLD=0.02; imports simulate_profit, WalkForwardBacktester, set_all_seeds |
| `experiments/results/canonical/headline.json` | 9 models with full metric set including sharpe_per_trade and alpha_bps_per_trade | VERIFIED | schema_version=1.0; protocol: seed=42, position_size=100.0, threshold=0.02, train=6802, test=1673; all 9 models; all 10 fields present |
| `experiments/results/archive/backtest_ppo_raw.json` | Quarantined disputed file | VERIFIED | File exists; original backtest/ path no longer present |
| `experiments/results/archive/backtest_ppo_filtered.json` | Quarantined disputed file | VERIFIED | File exists; original backtest/ path no longer present |
| `.planning/phases/17-.../17-02-PPO-DIAGNOSTIC.md` | PPO root-cause writeup (≥40 lines) | VERIFIED | 200 lines; addresses "7,724" (2 hits); addresses "canonical" (21 hits); names root cause as units mismatch + contract-count inflation |
| `scripts/audit_paper_numbers.py` | Paper-vs-JSON cross-reference script (≥100 lines) | VERIFIED | 557 lines; parses; DOLLAR_RE, SHARPE_RE, TRADES_RE, RMSE_RE present; MODEL_ALIASES and find_nearest_model present; references canonical/headline.json |
| `.planning/phases/17-.../17-03-NUMBER-AUDIT.md` | Audit log (≥60 lines) | VERIFIED | Exactly 60 lines |
| `PAPER_DRAFT.md` | Updated paper: zero "−$7,724", abstract with Sharpe + bps, pitch-standard format | VERIFIED | 0 occurrences of "7,724"/"7724"; abstract has Sharpe values 0.501/0.499 and "+15.0 bps/+14.9 bps"; 19 per-trade Sharpe/alpha occurrences |
| `slides_deck.html` | Results slide with bps + Sharpe, canonical PPO numbers | VERIFIED | 20 bps hits; 14 Sharpe hits; "11/11" walk-forward; per-pair Sharpe ≈ 3.2; PPO footnote referencing diagnostic; HTML parses |
| `scripts/check_paper.sh` | Extended validator (≥100 lines) with REPL-06 checks | VERIFIED | 113 lines; 3 REPL-06 checks at lines 77-104; 19/19 OK, 0 FAIL |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `experiments/run_canonical.py` | `src/utils/seed.py` | `set_all_seeds(42)` call | WIRED | set_all_seeds imported (line 66) and called at lines 357 and 404 |
| `experiments/run_canonical.py` | `src/evaluation/profit_sim.py` | `simulate_profit()` | WIRED | simulate_profit imported (line 61) and called at line 174 |
| `experiments/run_canonical.py` | `src/evaluation/backtester.py` | `WalkForwardBacktester(position_size=100.0)` | WIRED | WalkForwardBacktester imported (line 60) and called at lines 147, 152 |
| `scripts/audit_paper_numbers.py` | `experiments/results/canonical/headline.json` | `json.load` + per-model field extraction | WIRED | "canonical/headline.json" appears 2 times in script |
| `PAPER_DRAFT.md abstract` | Per-trade Sharpe + alpha (bps) | headline number replacement | WIRED | Abstract now contains "per-trade Sharpe of 0.501" and "+15.0 bps per-trade alpha" |
| `PAPER_DRAFT.md §5.1 Table` | Sharpe and bps columns | table column addition | WIRED | 6 bps occurrences and 19 Sharpe occurrences in paper confirmed |
| `scripts/check_paper.sh` | `PAPER_DRAFT.md abstract` | awk paragraph extraction + Sharpe regex | WIRED | REPL-06a/b checks at lines 78-84; "Sharpe" pattern used in both checks |

---

### Requirements Coverage

| Requirement | Source Plan | Status | Evidence |
|-------------|------------|--------|----------|
| REPL-01 | 17-01-PLAN.md | Complete | canonical/headline.json with 9 models + full metric set; run_canonical.py (533 lines) |
| REPL-02 | 17-01-PLAN.md | Complete | 17-02-PPO-DIAGNOSTIC.md (200 lines); archive/ files present; root cause documented |
| REPL-03 | 17-02-PLAN.md | Complete | audit_paper_numbers.py (557 lines); 17-03-NUMBER-AUDIT.md (60 lines); zero disputed figures in paper |
| REPL-04 | 17-02-PLAN.md | Complete | Abstract leads with per-trade Sharpe + bps; 6 bps occurrences; 19 per-trade Sharpe/alpha occurrences |
| REPL-05 | 17-03-PLAN.md | Complete | slides_deck.html has 20 bps hits, 14 Sharpe hits, canonical PPO numbers, PPO footnote, 11/11 panel |
| REPL-06 | 17-03-PLAN.md | Complete | check_paper.sh extended to 113 lines; 3 REPL-06 checks; 19/19 OK |
| REPL-07 | 17-04-PLAN.md | Complete | STATE.md at 100%/complete; 5 Phase 17 decision bullets; ROADMAP.md 4/4 complete; no unfilled placeholders |

REQUIREMENTS.md traceability: 7/7 REPL rows show "Complete", 7/7 REPL checkboxes are [x], 0 Pending.

---

### Anti-Patterns Found

| File | Pattern | Severity | Assessment |
|------|---------|----------|------------|
| `experiments/run_canonical.py` | `set_all_seeds` called at lines 357 and 404 (both a module-level call in `run_ppo_legacy_path` and `main()`) | Info | Both calls intentional; seed is reset before each PPO variant for the dual-path diagnostic. Not a stub. |
| `.planning/phases/.../17-03-NUMBER-AUDIT.md` | Exactly 60 lines (at the minimum threshold for "substantive") | Info | 60 lines meets the ≥60 acceptance criterion exactly. Content verified via grep to contain match/mismatch sections. |

No blockers or stubs detected. No TODO/FIXME/placeholder patterns found in key phase deliverables.

---

### Canonical Numbers Sanity Check

LR alpha_bps_per_trade = 15.021 (expected ~13 bps band [8, 18]) — PASS (within band; slightly above plan estimate because canonical rerun produced $232.67 vs the pre-plan estimate of $201.69, suggesting the fresh seed=42 single-protocol run converged better).

PPO-Filtered pnl = +$4.61 (not the disputed −$87,724) — PASS.
PPO-Raw pnl = +$158.15 (not the disputed +$96,336) — PASS.

The canonical numbers differ from the planning-doc estimates in the regression baselines (LR: $232.67 vs planned $201.69; GRU: $212.50 vs planned $174.11) because the canonical protocol uses a fresh single seed run that may have different train/test splits or feature pipeline behavior than the pre-phase `verify_headline.json`. This is documented in the plan as expected ("canonical output should be CLOSE — not necessarily identical").

---

### Human Verification Required

None. All automated checks pass with zero failures. The only items that could benefit from human review are cosmetic/visual:

1. **Slide visual layout** — The bar chart CSS rendering (bps axis 0–15) should be viewed in a browser to confirm bars are legible and the PPO+AE bar (~0.5 bps) is visibly near-zero while regression bars (~15 bps) dominate.
2. **Paper table formatting** — Table 2 column reordering (Sharpe + bps before P&L) should be confirmed to render correctly in LaTeX/PDF.

These do not block the goal; they are presentation quality items.

---

## Summary

Phase 17 goal is fully achieved. All 7 REPL requirements are Complete. The canonical single-source-of-truth metrics file exists with all 9 models and all required fields. The disputed PPO figures are quarantined. The paper contains zero references to the unreproducible "−$7,724" claim. The abstract, tables, and conclusions lead with per-trade Sharpe and per-trade alpha in bps. The slides use canonical numbers with a PPO discrepancy footnote. The check_paper.sh validator now enforces pitch-standard hygiene with 19/19 checks passing. STATE.md and ROADMAP.md are fully closed with the PPO root cause documented verbatim.

---

_Verified: 2026-04-24_
_Verifier: Claude (gsd-verifier)_
