# Phase 16 Final Review Checklist

**Phase:** 16 — Paper Revision: Phase 15 Integration + Final Review
**Created:** 2026-04-24 by Plan 16-03 Task 1 executor
**Reviewers:** Ian Sabia (cover-to-cover / REV-07), Alvin Jang (readability §1 + §8 / REV-06)
**Deadline:** April 27, 2026 (3 days)
**Paper state:** Post-16-01 (abstract, §5.9.1, §6.4 item 9) + post-16-02 (Finding 6 companion, Finding 27)
**Automated guardrails at checklist creation:** `bash scripts/check_paper.sh` → **ALL CHECKS PASSED** (16/16 OK); abstract=247 words; `grep -c "1,224" PAPER_DRAFT.md`=5; `grep -c "^## Finding 27:" FINDINGS.md`=1; `grep -cE "TODO|FIXME|XXX|\[Insert|TBD" PAPER_DRAFT.md`=0.

## Rendering the Paper

Pandoc 3.8.3 is installed at `/opt/homebrew/bin/pandoc`, but **PDF render via pandoc failed** on this paper — the body contains bare `&` inside inline-math expressions like `$+13.08 rejected P&L` and `+0.45 P&L`, which LaTeX interprets as alignment-tab characters (tried both default LaTeX engine and `--pdf-engine=xelatex`; both errored at line ~1625). Render logs saved to `/tmp/paper_render.log` and `/tmp/paper_render_xelatex.log`.

**HTML render succeeded** and is the recommended review medium for this checkpoint:

- Pre-rendered HTML: **`/tmp/paper_check.html`** (121 KB, standalone + TOC, generated 2026-04-24)
- Open with: `open /tmp/paper_check.html` (macOS Preview in browser) or drag into Chrome / Safari

**If you want a PDF anyway**, any of these three options work:

- **Option A (fastest, zero setup):** Open `/tmp/paper_check.html` in Chrome → `Cmd+P` → "Save as PDF" → this gives you a paginated PDF with clickable TOC for Alvin.
- **Option B (VSCode, live while editing):** Open `PAPER_DRAFT.md` in VSCode → `Cmd+Shift+V` for side-by-side live markdown preview. Best if you want to read-and-edit in one window.
- **Option C (GitHub, best for sharing with Alvin):** push this branch and share the `blob/…/PAPER_DRAFT.md` URL — GitHub renders tables, headings, and cross-refs cleanly without any local setup.

**DO NOT attempt** `pandoc PAPER_DRAFT.md -o out.pdf` without first escaping the P&L `&` characters inside math mode — it will fail the same way. Fixing the math escapes is NOT part of this plan's scope (they render fine in HTML and in GitHub markdown).

Re-render command if content changes during Signal B polish:

```bash
pandoc PAPER_DRAFT.md -o /tmp/paper_check.html --standalone --toc \
  --metadata title="DS340 Paper — Review Render"
```

## Alvin's Review (REV-06) — §1 Introduction + §8 Conclusions

**What to send Alvin:** the HTML file `/tmp/paper_check.html` (or a GitHub link to the branch on `blob/…/PAPER_DRAFT.md`). Tell him he only needs to read §1 and §8 — the rest of the paper is stable from v1.1 and already reviewed.

**What to check:**

- [ ] §1.1 Problem Statement reads as a clear onramp (no jargon without definition)
- [ ] §1.2 Motivation distinguishes academic and applied audiences clearly
- [ ] §1.3 Background explanations are correct (prediction-market mechanics, spread definition, Kalshi vs Polymarket regulatory framing)
- [ ] §1.4 Contributions list matches the paper's actual content (does every bullet map to a section?)
- [ ] §8.1–8.3 Conclusions restate the central answer ("simpler models win at this scale") in clean prose
- [ ] §8 flows from §6 Discussion without repeating §5's numeric results verbatim
- [ ] No sentence is hard to parse on a first read
- [ ] No dangling claims that a later section has since softened (e.g. anything about "live oil trading" without the April-24 post-submission qualifier)

**How to respond to Ian:**

- **If no changes:** Slack / text / email Ian: "Reviewed §1 + §8, no changes requested." Ian will capture the note + date in STATE.md via Signal A.
- **If edits:** send a diff, list of specific line-range edits, or marked-up copy. Ian will forward those edits to the executor as Signal B and they'll be applied in a polish commit.

## Ian's Cover-to-Cover Read (REV-07)

**What to look for** (derived from Phase 14 retrospective + Phase 16 integration deltas):

- [ ] **Abstract ≤ 250 words.** Verify: `awk '/^## Abstract/{f=1;next} /^---$/{if(f)exit} f' PAPER_DRAFT.md | wc -w` → currently **247**.
- [ ] **Phase 15 numbers in §5.9.1 match `15-03-SUMMARY.md` verbatim.** Spot-check: `KXBRENTW=486`, `KXWTI=409`, total=`1,224`, aggregate=`+$1.96`, WR=`36.0%`.
- [ ] **§6.4 item 9 preserves the original v1.1 engineering-gap paragraph verbatim** AND appends a one-line **Resolved post-submission in Phase 15** note linking to §5.9. No deletion of the original acknowledged limitation.
- [ ] **Tables 1–10 numbering is sequential** and no prose references an orphan "Table N" that doesn't exist. Verify: `grep -oE "Table [0-9]+" PAPER_DRAFT.md | sort -u`.
- [ ] **Figures 1–11 all referenced somewhere in prose** (Appendix B index sanity). Verify: for each `Fig N` in Appendix B, `grep -c "Figure $N\|Fig $N" PAPER_DRAFT.md` ≥ 2.
- [ ] **No stale Sharpe claims.** Verify: `grep -E "0\.59 annualize|annualizes to 4\.3|per-trade Sharpe 0\.59" PAPER_DRAFT.md` should return nothing (Phase 14-01 removed these).
- [ ] **No TODO / FIXME / XXX / [Insert / TBD markers.** Verify: `grep -cE "TODO|FIXME|XXX|\[Insert|TBD" PAPER_DRAFT.md` == 0 (currently 0).
- [ ] **Finding 6 backtest table preserved** (765 trades, 76.5% WR, +$0.41/trade, +142.7% edge) alongside new **Live validation (Phase 15, 2026-04-24)** companion paragraph (1,224 closures, 36.0% WR, ~$0.0016/trade).
- [ ] **Finding 27 (Silent Category Starvation in Live Systems) present** and cross-references Finding 8 explicitly. Verify: `awk '/^## Finding 27:/,/^---$/' FINDINGS.md | grep -c "Finding 8"` == 1.
- [ ] **§8 Conclusions flows from §6 Discussion** without repeating §5's numeric results verbatim.
- [ ] **Abstract live-validation qualifier is the new text**, not the v1.1 "backtest evidence; live oil-trading remains unobserved" phrasing. Verify: `grep -c "live-validated post-submission" PAPER_DRAFT.md` ≥ 1.
- [ ] **§5.9.1 three honest caveats present** (short 12h window, near-flat per-trade economics, paper-trading idealizations persist). These keep the post-submission cohort from being read as a robust live edge measurement.
- [ ] **Regression guardrail green.** Verify: `bash scripts/check_paper.sh` prints "ALL CHECKS PASSED" with 16/16 OK.

**How to signal completion to the Task 2 checkpoint:**

Report to the executor with exactly ONE of the three structured signals defined in the section below (Signal A / Signal B / Signal C).

## Automated Guardrails (must stay green through any polish edits)

Copy-paste this block after any Signal B polish edits:

```bash
bash scripts/check_paper.sh                                                                # POL-04/05/06/07/08/09/10 regression, exit 0
awk '/^## Abstract/{f=1;next} /^---$/{if(f)exit} f' PAPER_DRAFT.md | wc -w                 # <= 250 (currently 247)
grep -cE "TODO|FIXME|XXX|\[Insert|TBD" PAPER_DRAFT.md                                      # == 0
grep -c "1,224" PAPER_DRAFT.md                                                             # >= 3 (currently 5)
grep -c "^## Finding 27:" FINDINGS.md                                                      # == 1
grep -c "Silent Category Starvation" FINDINGS.md                                           # >= 1
grep -c "live-validated post-submission" PAPER_DRAFT.md                                    # >= 1
grep -cE "0\.59 annualize|annualizes to 4\.3" PAPER_DRAFT.md                               # == 0 (no stale Sharpe)
```

If ANY of these regress, the polish edit introduced a defect — fix before re-presenting the checkpoint.

## Approval Disposition (exactly one of three)

At the Task 2 checkpoint, Ian reports the combined Alvin + self-review outcome. Allowed dispositions:

### Signal A — "Both clean" (happy path, expected)

**Phrasing:** "Alvin reviewed 2026-04-YY: no changes requested. Ian cover-to-cover 2026-04-YY: no changes. Approve Phase 16 closeout."

**Executor will then:**

1. Append a Decisions entry to `.planning/STATE.md`:
   `- [Phase 16-03]: Alvin reviewed 2026-04-YY: no changes requested (§1 + §8 readability pass). Ian cover-to-cover 2026-04-YY: no changes requested. REV-06 + REV-07 closed. Paper ready for April 27 submission.`
2. Flip REV-06 and REV-07 to `[x]` complete in `.planning/REQUIREMENTS.md`.
3. Update ROADMAP.md Phase 16 status to "Complete" and tick any outstanding plan checkboxes.
4. Re-run `bash scripts/check_paper.sh` one final time (must still exit 0).
5. Create `16-03-SUMMARY.md` documenting the Signal A terminal disposition.
6. Atomic commit: `docs(16-03): close Phase 16 — Alvin + Ian review complete, REV-06/REV-07 satisfied`.
7. Phase 16 complete; v1.1 milestone shipped.

### Signal B — "Changes requested"

**Phrasing:** "Alvin requested: <specific edits, line-ranges, or marked-up diff>. Ian found: <specific edits>. Apply these and re-present."

**Executor will then:**

1. Apply the edits as targeted Edit calls to PAPER_DRAFT.md and/or FINDINGS.md (no new sections unless explicitly requested).
2. Re-run the "Automated Guardrails" block above — ALL must still pass.
3. Atomic commit: `docs(16-03): apply final-review polish edits from Alvin + Ian`.
4. Re-present this same checkpoint to Ian with the updated state + commit hash.
5. Loop until Signal A or Signal C. Each polish cycle is its own commit.

### Signal C — "Blocked / Deferred"

**Phrasing (one of):**

- "Alvin unreachable before deadline — ship as-is with solo Ian review. Mark REV-06 deferred."
- "Ian defers cover-to-cover read to post-submission. Mark REV-07 deferred, ship on Alvin-only."
- "Neither review feasible before 2026-04-27 — ship without final human review. Mark REV-06 + REV-07 deferred."

**Executor will then:**

1. Append a Decisions entry to `.planning/STATE.md` documenting WHICH review is deferred and WHY (e.g. "Alvin OOO 2026-04-25 through 2026-04-28, deadline forces ship-as-is").
2. Mark the deferred REV-06 and/or REV-07 in REQUIREMENTS.md with a "deferred: <reason>" suffix instead of flipping to `[x]`.
3. Atomic commit: `docs(16-03): defer REV-06/REV-07 — <reason>` (concrete reason in body).
4. Phase 16 closed as **PARTIAL** — user knowingly ships the paper with incomplete human review.

The checkpoint does not self-close. The executor does NOT time out and does NOT guess a disposition — Ian must explicitly send Signal A, B, or C.

---

*Artifact generated by Plan 16-03 Task 1 on 2026-04-24. Consumed by Plan 16-03 Task 2 checkpoint.*
