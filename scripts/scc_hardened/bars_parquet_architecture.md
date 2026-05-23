# Architecture proposal: stop committing `data/live/bars.parquet` to git

**Status:** Proposal. Not yet implemented. Reviewed and approved by
operator before any code or workflow changes land.

**Context for this document:** The 2026-05-20 corruption incident
([scripts/scc_hardened/README.md](README.md)) traced to silent
quota-truncated writes on the SCC home filesystem. Quota was exhausted
primarily by `.git` (8.2 GB) and `.venv` (2.2 GB) on a 10 GB soft quota.
`.venv` was moved to project space on 2026-05-23 (instant 2.2 GB
recovery). `.git` cannot be trimmed by `git gc` because it contains
hundreds of binary blobs of `data/live/bars.parquet`, each ~11 MB, one
per `*/15 * * * *` SCC cron commit, and binary blobs do not
delta-compress meaningfully.

The root cause of the .git bloat is committing an 11 MB binary every
15 minutes. This proposal addresses that cause without rewriting git
history, which is non-negotiable: `AUDIT_REPORT_OIL.md` Tier 1
reproduction depends on specific commit hashes
(`04625b8`, `6b8f3c5`, `c7ec169`, `fb3fdec` and others), and force-
pushing would invalidate the audit chain that backs the Gold
conversation.

## Inventory of current `data/live/bars.parquet` consumers

Files in the repo that reference this path (from `grep -rl`):

- **Reads in workflows and live system:**
  - `.github/workflows/collect-and-trade.yml` (read-side guard + downstream cycle reads)
  - `src/live/paper_trader.py`
  - `src/analysis/reconciliation.py`
- **Reads in scripts:**
  - `experiments/run_live_reconciliation.py`
  - `scripts/full_retrain.py`
  - `scripts/check_retrain_readiness.py`
  - `scripts/cut_canonical_oil_split.py` (used to BUILD canonical_oil from live)
  - `scripts/train_commodity_specialist.py`
- **Writes:**
  - `src/live/collector.py:append_to_parquet` (the only writer; called from
    `src/live/strategy.py:run_cycle`)
- **SCC cron callers:**
  - `scripts/scc_trading_cycle.sh` (the every-15-min `git add -f` source of the
    .git bloat)
  - `scripts/scc_retrain_batch.sh` (reads, every 6 hours)
- **Documentation and operator runbooks:**
  - `scripts/scc_hardened/README.md`
  - `scripts/scc_hardened/deploy.md`
  - `scripts/scc_hardened/scc_trading_cycle.sh` (the hardened version's Layer 1 guard)

**Notably NOT in the audit chain:**
`AUDIT_REPORT_OIL.md` and `RESULTS_OIL_RETRAIN_DRAFT.md` reference
**`data/processed/canonical_oil/{train,test}.parquet`**, which were
**built from a frozen snapshot** of `bars.parquet` in May 2026 and
checked in as committed artifacts. They are insulated from changes to
the live bars file. The only references to live bars in audit context
are by commit hash (e.g. "last known good 04625b8") which resolve via
git history regardless of where future bars live.

## Storage options compared

| Option | Cost | SCC accessible | GHA accessible | Operator effort | Survives quota events | Audit-chain impact |
|---|---|---|---|---|---|---|
| **A. /projectnb/ds340 (current scratch)** | Free | Yes (network mount) | No | Low (already proven for .venv) | Yes (no per-user quota in the same way) | None |
| **B. S3 / Backblaze B2** | $0.005-0.02/GB/mo | Yes (via boto) | Yes (via boto) | Medium (creds, bucket, lifecycle policy) | Yes | None |
| **C. GHA artifact storage** | Free for ≤500 MB, 90-day retention | No (write-only from GHA) | Yes | Medium (workflow changes) | Yes | None — but does not solve the SCC writer side |
| **D. Git LFS** | Free 1 GB/mo bandwidth, paid above | Yes | Yes | High (LFS install + history migration risk) | Mixed (LFS pointers still in git) | **Possibly broken** if migration touches existing commits |
| **E. Cloudflare R2** | Egress-free, $0.015/GB/mo storage | Yes (via boto-compatible) | Yes | Medium-high (CF account setup) | Yes | None |

**Eliminated:**
- **D (Git LFS)** — operator effort + history-migration risk. LFS often
  requires rewriting existing commits to convert old blobs into LFS
  pointers. Even if we skip the rewrite and only LFS-track new commits,
  the existing 8.2 GB of binary history stays in `.git`, defeating
  the purpose.
- **C (GHA artifacts)** alone — solves the GHA-read side but not the
  SCC-write side. SCC cannot write to GHA artifact storage from a cron.

**Viable:**
- **A (/projectnb/ds340/...)** — simplest. SCC writes locally, GHA does
  not need bars at all (the GHA workflow only reads bars for the read-
  side guard, and that guard's purpose is to detect corruption; if we
  move bars off git, the guard moves too or becomes a no-op).
- **B (S3)** — most portable. Both SCC and GHA can read/write. Adds
  cloud cost (negligible, ~$1/mo for 50 GB of historical bars at S3
  IA pricing).
- **E (R2)** — same as B but with zero egress fees, which matters more
  if GHA is downloading bars on every cycle.

**Recommendation: hybrid A+C.** Bars live primarily on
/projectnb/ds340/projects/iansabia/data/ for SCC writes; GHA only
needs an occasional bootstrap (when running a fresh full reconciliation
or training run) which can come from GHA artifact storage populated by
a once-daily SCC-side push job. Day-to-day GHA cycles do not need to
read bars at all because they perform live trading from API quotes,
not from historical bars.

## What the GHA workflow actually reads bars for

Looking at `collect-and-trade.yml`:

1. **Read-side guard** (the one we added on 2026-05-23): asserts
   bars.parquet exists and is parquet-readable before running the
   cycle. **Purpose: catch corruption in the file as-checked-in.** If
   bars.parquet is no longer committed to the repo, this guard would
   either be a no-op or would check a different path (e.g., does
   `data/live/bars.parquet` exist as a symlink? If yes, follow and
   check; if not present at all, OK).

2. **`src.live.trading_cycle --cycle`** invokes
   `src.live.strategy.run_cycle`, which calls
   `src.live.collector.append_to_parquet`. The collector reads
   bars.parquet to APPEND new rows. If the file is not in the repo and
   not present in the runner's filesystem, the append step would
   create a fresh empty parquet — not what we want. **This is the
   single hardest dependency** to break.

The trading_cycle's collector needs SOME version of bars.parquet on
disk in the GHA runner to append to. Options:
- The cycle no-ops the bars append on GHA (only does it on SCC)
- The cycle's GHA path uses a separate, smaller bars file that lives
  only in GHA artifact storage
- The GHA cycle fetches the latest bars from /projectnb (impossible
  — GHA can't reach SCC's filesystem) or from S3/R2 (possible)

## Recommended migration path (4 phases)

### Phase 0: Inventory and freeze (1 hour, no code changes)

- Confirm the file path inventory above is complete via a more
  thorough grep
- Verify the writeup and audit chain do NOT depend on live bars
  beyond the canonical-oil snapshot (confirmed above)
- Take a one-time snapshot of bars.parquet to a safe archive location
  (/projectnb/ds340/projects/iansabia/data/bars.archive.2026-05-23.parquet)
  as a rollback escape hatch
- Document the freeze: future commits to bars.parquet will continue
  for ~1 week of overlap, then stop

### Phase 1: Add path indirection (2 hours, code change only)

- Introduce a config variable `BARS_PARQUET_PATH` read from environment
  with default `data/live/bars.parquet` (current behavior preserved)
- Update `src.live.collector.append_to_parquet` and the read sites in
  `src.live.paper_trader`, `src.analysis.reconciliation`,
  `scripts/full_retrain.py`, `scripts/check_retrain_readiness.py`,
  `scripts/cut_canonical_oil_split.py`,
  `scripts/train_commodity_specialist.py`,
  `experiments/run_live_reconciliation.py` to use the config
- Same default behavior; no operational change yet
- Commit and verify GHA workflow still passes

### Phase 2: Switch SCC writes to /projectnb (1 hour, deploy only)

- Set `BARS_PARQUET_PATH=/projectnb/ds340/projects/iansabia/data/bars.parquet`
  in the SCC trading_cycle's environment (via the cron script)
- Copy the existing bars.parquet to that new path (one-time bootstrap)
- Remove `git add -f data/live/bars.parquet` from
  `scc_trading_cycle.sh` — but **keep** `git add` for
  `paper_trades*.jsonl` and `positions.db` (they remain in git for now;
  paper_trades are append-only logs with manageable size, positions.db
  is tiny)
- Monitor for 24 hours: bars.parquet writes happen to /projectnb;
  `.git` stops growing; SCC quota drops as old packfiles are eventually
  reclaimed by maintenance
- The path `data/live/bars.parquet` in the SCC working tree becomes a
  **stale snapshot from before the cutover**, gitignored, with a
  symlink pointing to the new location for any read sites that still
  use the old path

### Phase 3: Switch GHA workflow (1-2 hours)

- The GHA workflow no longer reads `data/live/bars.parquet` from the
  repo (the file is gitignored after Phase 2). Two sub-options:
  - **3a (simpler):** the GHA cycle reads bars from S3/R2, populated
    by an SCC-side daily push job
  - **3b (lazier):** the GHA cycle's read-side guard becomes a no-op
    (the cycle creates a fresh bars file in the runner if missing), and
    the GHA cycle's writes go to GHA-side ephemeral storage that
    doesn't need to persist
- Update the read-side guard in `collect-and-trade.yml` to use
  `BARS_PARQUET_PATH` rather than the hard-coded path

### Phase 4: Clean up (30 min)

- Remove the symlinks and the gitignore stub for the legacy path
- Update `scripts/scc_hardened/README.md` and `deploy.md` to reflect
  the new bars location
- Add a brief migration note to the project README pointing future
  readers to this proposal so the "wait, where are the bars?" question
  has an answer

## Audit-reference resolution

Will the existing audit references in `AUDIT_REPORT_OIL.md` still
work after the migration? Yes, in detail:

- The audit references commit hashes
  (`04625b8`, `6b8f3c5`, `c7ec169`, `fb3fdec`, etc.) for Tier 1 and
  Tier 5 traceability. These commits **already contain**
  `data/live/bars.parquet` blobs in their tree. Those blobs do not
  disappear when we stop adding new ones; they remain in `.git/objects`
  forever (or until someone force-pushes, which we are not doing).
  Anyone running `git checkout 04625b8 -- data/live/bars.parquet`
  after the migration gets the same byte content they would have
  gotten today. Reproducibility is preserved.

- The canonical-oil artifacts
  (`data/processed/canonical_oil/{train,test}.parquet`) live in the
  repo permanently as committed artifacts, independent of where live
  bars live. The writeup's "+18.96 bps" and all related numbers
  trace to those frozen artifacts, not to live bars. Insulated.

- `RESULTS_OIL_RETRAIN_DRAFT.md` Appendix A's reproducibility command
  list mentions `data/live/bars.parquet` only via
  `scripts/cut_canonical_oil_split.py` (which CONSUMES bars to BUILD
  the canonical split). If we ever re-run that script post-migration
  (we should not need to, the split is frozen), it would read from the
  new path via `BARS_PARQUET_PATH`.

## Estimated benefit

After Phase 2 deployment:
- **SCC .git stops growing** at its current ~8.2 GB (current commits
  remain, audit chain intact)
- **No new corruption pathway from quota-truncated bars commits**
- **Quota headroom recovered over time** as gpfs deletes old packfiles
  (GPFS has a maintenance cadence; cleanup is not instant but happens
  within days-weeks)
- **GHA workflow simpler** (no need to ship 11 MB on every
  checkout; faster cold cache fills)

## Estimated risk

- **Path indirection bug** in Phase 1: low. Default behavior is
  preserved if `BARS_PARQUET_PATH` is unset. Caught by smoke test on
  GHA before SCC deploy.
- **SCC write-to-/projectnb fails on first cycle** post-Phase 2: low.
  /projectnb is the same filesystem that successfully held the moved
  .venv. We can pre-test by writing a small probe parquet to the new
  location before the cutover.
- **Audit chain reference breaks**: very low. Above analysis confirms
  hash-based references resolve unchanged. Worth a Tier 5 audit re-run
  after Phase 2 just to confirm.
- **Forgotten reader** somewhere in the codebase still hits the old
  path post-Phase 2: medium. Phase 1 mitigates by funneling everything
  through `BARS_PARQUET_PATH`; we'd catch any straggler in CI.

## Recommended next step

After 6-12 clean cron cycles confirm the current hardened scripts +
.venv move is stable, schedule Phase 0+1 (the read-only / config-only
phases) for a single `/goal` session. Total estimated work: 2-3 hours.
Phases 2-4 can follow at operator pace over a week.

---

**Open questions for discussion:**

1. **A vs B vs E (storage backend):** /projectnb is the lowest-friction
   path but ties bars to SCC. Is that acceptable for the foreseeable
   future, or should we plan for portability now via S3/R2?

2. **Continue committing daily snapshots?** Even after the every-15-min
   cron commit stops, would a once-daily commit of bars.parquet
   (~365 commits/year vs 35,000/year today) be useful as a paper-trail
   for the live system, or should we go fully off-git?

3. **What to do with the existing 8.2 GB of .git history?** Three sub-
   options: (a) leave it forever (simplest, audit chain safest);
   (b) re-clone SCC's working tree from a shallow clone in 6 months
   (preserves origin's history, drops SCC's local copy bloat);
   (c) coordinate with operator + audit reviewers to do a one-time
   amend/squash much later when the audit chain is no longer load-
   bearing.
