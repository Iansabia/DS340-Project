# Plan 18-07 — Checkpoint Halt → Option B

**Status:** PAUSED at human-verify checkpoint after Task 1
**Reason:** Ian rejected Option A (replace 3.2 with 7.04) and chose Option B (pair-stratified leakage-free recompute)
**Date:** 2026-04-26

---

## Ian's Decision

> "lets do option B, i think that is the best, more work but the most honest answer"

### Rationale

The audit's BLdP-corrected per-pair annualized Sharpe of 7.04 was computed on the **same leaky data** that Tier 2 (18-03) just flagged. 142 of 144 pairs bridge the canonical 80/20 row-index train/test split, so the same underlying market events appear in both halves. Leakage inflates Sharpe.

Replacing 3.2 with 7.04 in the paper would have swapped one un-validated number (3.2 has no derivation path in the codebase) for another inflated-by-leakage number (7.04). Neither is the honest answer.

The honest answer is to rebuild the train/test split with pair atomicity (each pair entirely in train OR entirely in test, never bridging), retrain the headline models, and report the Sharpe that emerges from a leakage-free protocol.

---

## Tasks 2 + 3 — DEFERRED to 18-08 + 18-07 resumption

| Task | Status | Notes |
|------|--------|-------|
| Task 1 — Tier 6 z-test | COMPLETE (commits `0d28d85` + `6185300`) | verdict=PASS; Cohen's h = −0.842, z = −10.76, p = 5.24e-27 |
| Task 2 — Human checkpoint | **PAUSED — awaiting 18-08 outputs** | 10-pair survivorship classification implicitly approved (all 10 = `structural`, Tier 4 → PASS); paper change scope cancelled |
| Task 3 — AUDIT_REPORT.md + paper/slide updates | **DEFERRED** | Will resume with 18-08's purged headline numbers feeding into the paper updates instead of 7.04 |

---

## 10-Pair Survivorship Sample — Classified

Ian's halt response implicitly approved the heuristic classification (Tier 4 = PASS): all 10 sampled dropped pairs are `26apr*` April-2026-expiry contracts discovered live after the canonical `test.parquet` 2026-04-09 snapshot. Post-snapshot pairs cannot retroactively appear in pre-snapshot training data — the drops are structural by construction.

**Classification: 10/10 structural · 0/10 retroactive · Tier 4 verdict = PASS**

---

## Hand-off to Plan 18-08

Plan 18-08 (to be planned + executed next) must produce:

1. **Pair-stratified split**: 80/20 by `pair_id`, seed=42 (matching canonical convention), no pair appears in both train and test. Output: `data/processed/purged_split/train.parquet` and `test.parquet`.
2. **LR + XGBoost retraining** on the purged split. These are the headline models per Phase 17. Output: `experiments/results/canonical_purged/headline.json` (sibling to the existing canonical/headline.json — both retained for transparency).
3. **Sharpe recomputation** under the same protocol as 18-02: per-trade, per-pair naive, BLdP-corrected, bootstrap 95% CI. Output: `experiments/results/audit/sharpe_audit_purged.json`.
4. **Side-by-side comparison table** documenting leaky canonical vs purged numbers.

After 18-08 completes, Plan 18-07 resumes at Task 3 with the purged Sharpe replacing 3.2 in PAPER_DRAFT.md / slides_deck.html / AUDIT_REPORT.md.

---

## What is NOT Changing (Yet)

- `PAPER_DRAFT.md` — untouched. The 3.2 claim stays in place until 18-08 produces the replacement.
- `slides_deck.html` — untouched.
- `AUDIT_REPORT.md` — not yet generated. Will be written by 18-07 Task 3 with both leaky-canonical and purged numbers in the assumption stack.
- `scripts/check_paper.sh` — the existing 26 audit checks remain green against the current paper text.

---

## Anti-goal Update

Phase 18 CONTEXT.md originally listed "no model retraining" as an anti-goal. Ian explicitly authorized retraining for the leakage-free recompute because the audit found a methodological defect (embargo violation) that cannot be honestly fixed any other way. The anti-goal scope tightens to: "no retraining beyond what is needed to produce a leakage-free headline number." LR + XGBoost only — no GRU/LSTM/TFT/PPO retraining.
