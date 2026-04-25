# Slides Guide for Alvin — DS340 Final Presentation

**Presentation date:** 4/28 or 4/30 (pick whichever the schedule lands on)
**Slides due:** night before presentation
**Total time:** 4 minutes — must hit every section to get credit
**Author of slides:** Alvin
**Reference draft already in repo:** `slides.md` (Marp source) + `slides.pdf` (6 pages, ~2:00 spoken at normal pace) — feel free to use as starting point or replace entirely

---

## Rubric (verbatim from `DS340FinalPresentationAndProjectInstrs_S26.pdf`)

You must hit all 6 sections in 4 minutes. The order below is the recommended order. Only one of us needs to present, but we can trade off if you want. Slides due night before so the classroom computer can run them.

> "You must present a section within the time limit to get credit for it, so be sure to rehearse."

| # | Section | Rubric requirement |
|---|---|---|
| 1 | **Team** | Just our names |
| 2 | **Problem statement** | What did you want to know or achieve, and why? |
| 3 | **Methods** | What kind of models or algorithms did you use? Where did the data come from? |
| 4 | **Challenge** | One challenge you overcame, and how you solved it |
| 5 | **Results** | Show your most interesting results |
| 6 | **Conclusions** | What did you learn doing this? |

---

## Time budget (4:00 total)

| Section | Budget | Slide count |
|---|---|---|
| Title + Team | 0:15 | 1 |
| Problem | 0:30 | 1 |
| Methods | 0:45 | 1 |
| Challenge | 0:45 | 1 |
| Results | 1:15 | 1–2 |
| Conclusions | 0:30 | 1 |
| **Total** | **4:00** | **6–7 slides** |

Rehearse with a stopwatch. 4-minute lightning talks always run long the first time.

---

## Slide-by-slide content guide

### Slide 1 — Title + Team (0:15)

**Required by rubric:** Names.

**Concrete content:**
- **Title:** *Complexity Is Not an Edge: An Empirical Study of Machine-Learning Arbitrage on Kalshi and Polymarket*
- **Subtitle/byline:** Ian Sabia, Alvin Jang — DS340, Spring 2026, Boston University
- **Optional visual:** Small Kalshi + Polymarket logo or a single screenshot of cross-platform price divergence

**What to say (~10 seconds):**
> "Hi, I'm [name], this is Ian/Alvin. We studied whether more complex ML models give a real edge in cross-platform prediction-market arbitrage."

---

### Slide 2 — Problem (0:30)

**Required by rubric:** What did we want to know, and why?

**Concrete content (pick 3 bullets):**
- Kalshi (CFTC-regulated event contracts) and Polymarket (on-chain) often have different prices on the **same** event
- **Question:** does adding complexity (RNNs, RL, autoencoders) actually beat plain regression for predicting when those prices converge?
- **Why it matters:** prediction markets processed >$1B in 2024 elections; if simpler models work, traders save training cost and risk
- *(Optional academic framing)* Tests the January 2026 arXiv 2601.07131 thesis that "feature engineering beats deep learning" in this kind of regime

**Suggested visual:**
- One plot showing Kalshi vs Polymarket price for a matched contract diverging then converging at expiry — pick from `experiments/figures/walk_forward_pnl.png` or build a simple side-by-side from `data/live/bars.parquet`
- OR a simple "The Question" box with the central research question from CLAUDE.md

**What to say:** "Kalshi and Polymarket frequently disagree on the same event. We asked: do bigger models predict that disagreement closing better than simple regression — and the answer turned out to be no."

---

### Slide 3 — Methods (0:45)

**Required by rubric:** Models/algorithms + where the data came from.

**Concrete content:**

**Data:**
- Kalshi API + Polymarket Gamma/CLOB/Data APIs
- ~6,802 train / 1,673 test rows across 144 matched pairs
- 59 features including 13 microstructure features (Amihud illiquidity, Kyle's lambda, Roll spread, Corwin-Schultz)

**Models — 4 tiers from simple to complex:**

| Tier | Models |
|---|---|
| 1. Regression | Linear Regression, XGBoost |
| 2. Sequence | GRU, LSTM, TFT |
| 3. RL | PPO on raw features |
| 4. RL + AE | PPO + autoencoder anomaly filter |

Plus 2 naive baselines (spread-always-closes, higher-volume-correct).

**Matching pipeline:** sentence-transformer semantic similarity + 10-rule structural quality filter (the matching is half the work — call this out)

**Suggested visual:**
- Single 4-tier ladder diagram (vertical bar) with model names — emphasizes the "complexity ladder"
- OR `experiments/figures/walk_forward_pnl.png` previewing all 8 models so the audience already sees ranking by results

**What to say:** "Data from both platforms' APIs, ~8.5K matched rows. We trained four tiers: regression baselines, recurrent nets, PPO, and PPO with autoencoder anomaly filtering — all on identical splits."

---

### Slide 4 — Challenge (0:45)

**Required by rubric:** One challenge + how we solved it.

**Recommended challenge (pick ONE — don't list multiple):**

**Option A — Silent Category Starvation (most recent, dramatic, NEW Finding 27)**
- The live system was silently dropping all daily WTI/oil markets. We thought we had no oil edge to validate.
- Discovered post-submission that `KALSHI_DISCOVERY_CATEGORIES` in our code was missing `"Commodities"` — Kalshi had moved daily oil there
- One-line fix; 1,224 commodity positions closed in 24h validation window
- **Lesson:** External-API discovery pipelines need "what's NOT in my data" monitoring, not just "what IS"

**Option B — pair_id schema bug (most technical, biggest near-disaster)**
- Three code paths in our live trading system disagreed on what `live_NNNN` meant — strategy, collector, and pair_mapping each numbered pairs differently
- 25 positions tracked the wrong markets for two days before we caught it
- Fixed with content-addressed pair IDs (`kxwti26apr08t10799-0x43d5953d`) — same scheme `train.parquet` already used
- **Lesson:** ID schemes that depend on iteration order are time bombs

**Option C — TFT didn't converge (most academically honest)**
- We pre-specified TFT hyperparameters for small data (hidden=8, attention head=1) but it still didn't converge at N=6,802 (RMSE 0.326 vs GRU 0.293)
- Reported as a finding, not hidden — showed even minimal TFT can't train at this scale
- **Lesson:** Negative results deserve documentation; don't just drop a tier that didn't work

**Recommended pick:** **A** — most dramatic story, ties to "engineering reality vs paper claims" theme, and we have concrete numbers (1,224 positions in 24h).

**Suggested visual:**
- For A: a single before/after bar — "Pre-fix: 0 oil positions. Post-fix: 1,224 in 24h"
- For B: a small diagram showing the 3 code paths with conflicting `live_0358` arrows pointing at different markets
- For C: GRU vs TFT loss curves side by side

**What to say (option A):** "Our live system claimed there were no oil markets to trade — but Kalshi's app showed plenty of daily WTI contracts. Turned out one tuple in our discovery code was missing the word 'Commodities'. Fix was one line; 24 hours later we had 1,224 commodity positions."

---

### Slide 5 — Results (1:00–1:15) — most important slide

**Required by rubric:** Most interesting results.

**Headline number to lead with:**
> "On 6,802 train / 1,673 test rows: XGBoost +$201.63 ≈ Linear Regression +$201.69. LSTM +$182.72. PPO+autoencoder **−$7,724**. Simpler wins."

**Suggested table for the slide (compact, ~6 rows):**

| Model | P&L | Notes |
|---|---|---|
| Linear Regression | **+$201.69** | Best |
| XGBoost | **+$201.63** | Tied |
| LSTM | +$182.72 | |
| GRU | +$174.11 | |
| PPO (raw) | −$413 | |
| PPO + autoencoder | **−$7,724** | Worst by far |

**Plus one of these as supporting evidence:**
- Per-pair annualized Sharpe: **≈ 3.2** (with per-trade 0.44 in footnote)
- Walk-forward: every window profitable for every ML model
- Live validation: 1,159 commodity positions closed since the discovery fix yesterday — backtest oil edge at least directionally validated

**Suggested visual:**
- `experiments/figures/walk_forward_pnl.png` — shows ranking holds across all 11 windows, very persuasive
- OR `experiments/figures/backtest_equity_curves.png` — clean cumulative-P&L lines, simpler models on top
- OR the table above + Fig 4 transaction cost chart as inset

**What to say:** "Simpler dominates across every metric we measured. Linear regression and XGBoost tied at +$201; PPO-with-autoencoder lost over seven thousand dollars. Walk-forward validation confirms this isn't one lucky window — every window, every ML tier, simple wins."

---

### Slide 6 — Conclusions (0:30)

**Required by rubric:** What did you learn?

**Three takeaways (pick 3):**

1. **At 6,802 rows, complexity is a liability, not an edge.** The added training cost of RNNs, PPO, and autoencoders isn't justified — and PPO+autoencoder is actively destructive.
2. **The alpha is in the matching pipeline, not the models.** Sentence-transformer matching + 10-rule quality filter is what makes any of this work.
3. **Engineering bugs masquerade as model failures.** Silent category starvation (Finding 27), pair_id schema mismatch, transaction-cost JSON drift — all looked like model problems initially. Infrastructure monitoring matters before model tuning.

**Optional fourth:**
4. **Honest negative results are worth shipping.** TFT didn't converge; PPO+autoencoder lost money; we kept those in the paper because they answer the research question empirically.

**Suggested visual:**
- Just 3 numbered bullets, large font — no figure
- OR a small "Future Work" addendum: live validation continues; ensemble formalization complete; oil edge now testable on live data

**What to say:** "At our data scale, complexity is a liability. The matching pipeline and the asset class — not the models — drive performance. And every model failure we initially blamed on the architecture turned out to be an infrastructure bug."

---

## Visuals you can pull straight from the repo

All figures already at 300 DPI IEEE-styled (from Phase 14):

| Figure | Path | Best for |
|---|---|---|
| Walk-forward P&L | `experiments/figures/walk_forward_pnl.png` | Slide 5 — ranking across windows |
| Backtest equity curves | `experiments/figures/backtest_equity_curves.png` | Slide 5 — cumulative P&L visual |
| Data scaling curve | `experiments/results/data_scaling/pnl_at_2pp_vs_data.png` | Slide 5 supporting — shows ranking holds at 250 bars/pair |
| Transaction cost sensitivity | `experiments/figures/transaction_cost_sensitivity.png` | Slide 5 supporting — robustness |
| SHAP feature importance | `experiments/figures/shap_bar_plot.png` | Methods slide — what features matter |
| Threshold heatmap | `experiments/figures/experiment3_threshold_heatmap.png` | Optional — Sharpe rises with threshold |

---

## Style guidance (Marp / PowerPoint / Keynote — your pick)

- **Dark slide background, light text** if presenting from classroom computer (better contrast for projector)
- **No more than 5 bullet points per slide.** Lightning talks fail on slides full of text — most readers can't process more than 5.
- **One graph per slide max.** Two graphs splits attention.
- **Big fonts:** 32pt minimum body, 48pt+ for headers. People in the back are reading from projector glare.
- **No transitions/animations.** Slows you down; you have 4 minutes.
- **Number every figure** ("Fig 1: walk-forward P&L") even on slides — the rubric values clarity over polish.

---

## Format options

The repo already has `slides.md` as a Marp source you can adapt. Three paths:

**Option 1 — Use my existing Marp draft as starting point:**
- Open `slides.md` in VS Code, install "Marp for VS Code" extension
- Edit, then `Cmd+Shift+P` → "Marp: Export slide deck" → PDF
- Quick, version-controlled, theme already styled

**Option 2 — Google Slides / PowerPoint / Keynote:**
- Paste this guide's content section by section
- Add visuals from `experiments/figures/` directory
- Easier collaboration if we want to edit together
- More control over fonts and transitions

**Option 3 — LaTeX Beamer:**
- Overkill for 4 minutes; skip unless you really want academic polish

I'd suggest **Option 2** since you're driving the slides — easier for you to iterate without learning Marp.

---

## Pre-submit checklist (night before presentation)

- [ ] All 6 rubric sections present
- [ ] Slide count: 6–7 (don't try 10)
- [ ] One person can deliver in ≤ 4:00 with stopwatch (rehearse twice)
- [ ] No typos in author names or affiliations
- [ ] Headline result on Results slide is the **per-pair Sharpe ≈ 3.2** OR **simple-beats-complex P&L table** — not the per-trade Sharpe of 0.44 (footnote only)
- [ ] PDF exported and saved as `slides_final.pdf` for upload
- [ ] Test that the PDF opens on a fresh machine (no embedded fonts missing)

---

## Authoritative numbers (ranked by importance)

If you only cite a few, cite these in this order:

1. **+$201.69 (LR) ≈ +$201.63 (XGBoost) vs −$7,724 (PPO+AE)** — the headline complexity comparison
2. **6,802 train / 1,673 test rows, 144 matched pairs** — context for "small data"
3. **Per-pair Sharpe ≈ 3.2** — paper-credible Sharpe number (NOT 0.44 — that's per-trade footnote)
4. **1,224 commodity positions in 24h post-fix** — Phase 15 live validation (newest, freshest result)
5. **76.5% win rate, +$0.41/trade for oil near-expiry** — Finding 6 backtest edge
6. **All 11 walk-forward windows profitable** — robustness evidence
7. **TFT didn't converge at N=6,802** — honest negative result

---

## Open questions for you

- Do you want to present, or should Ian? (rubric allows trade-off — pick whoever rehearses better)
- Pick the challenge story: A (silent category starvation), B (pair_id bug), or C (TFT non-convergence)?
- Single slide for Results or split into two (P&L table + walk-forward visual)?

Send me your draft when you have something — I'll pull numbers from the paper to make sure everything is cited honestly.

— Ian
