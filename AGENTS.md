# Agent Instructions — Superpowers Discipline

These rules apply to ALL agents working on this project, including GSD executor agents.
They are derived from the Superpowers framework and are non-negotiable.

## Test-Driven Development

Follow RED-GREEN-REFACTOR for all production code (data pipeline, matching, models, evaluation).

**The Iron Law:** No production code without a failing test first.

1. **RED** — Write one minimal test for the next behavior. Run it. Confirm it fails for the right reason (missing feature, not a typo).
2. **GREEN** — Write the simplest code to pass the test. Nothing more.
3. **REFACTOR** — Clean up while keeping tests green. Don't add behavior.
4. **Repeat.**

**If you wrote code before the test:** Delete it. Start over. No exceptions — don't keep it as "reference," don't "adapt" it. Delete means delete.

**Exceptions** (skip TDD only for these):
- Quick exploration scripts in `notebooks/`
- Configuration files
- One-off plotting/visualization

| Rationalization | Reality |
|--------|---------|
| "Too simple to test" | Simple code breaks. Test takes 30 seconds. |
| "I'll test after" | Tests passing immediately prove nothing. |
| "Need to explore first" | Fine. Throw away exploration, then start with TDD. |
| "TDD will slow me down" | TDD is faster than debugging. |

## Verification Before Completion

**Never claim work is complete without fresh verification evidence.**

Before saying "done," "fixed," "passing," or any completion claim:
1. **IDENTIFY** what command proves the claim (test suite, linter, build)
2. **RUN** the command — fresh, complete, in this session
3. **READ** the full output, check exit code, count failures
4. **VERIFY** output confirms the claim
5. **ONLY THEN** make the claim with evidence

Using "should work," "looks correct," or "seems fine" without running verification = not acceptable.

## Self-Review Before Reporting

Before reporting task completion, review your own work:

**Completeness:**
- Did I implement everything specified?
- Did I miss any requirements or edge cases?

**Quality:**
- Are names clear and accurate?
- Is the code clean and maintainable?
- Does each file have one clear responsibility?

**Discipline:**
- Did I avoid overbuilding (YAGNI)?
- Did I only build what was requested?
- Did I follow existing patterns in the codebase?

**Testing:**
- Do tests verify behavior, not mock behavior?
- Did I follow TDD?
- Are tests comprehensive?

Fix any issues found during self-review before reporting.

## Report Format

When completing a task, report:
- **Status:** DONE | DONE_WITH_CONCERNS | BLOCKED | NEEDS_CONTEXT
- What you implemented
- What you tested and test results (with actual output)
- Files changed
- Self-review findings
- Any concerns

Use DONE_WITH_CONCERNS if you have doubts. Use BLOCKED if you cannot complete. Use NEEDS_CONTEXT if information is missing. Never silently produce uncertain work.

## Code Organization

- Follow the structure defined in plans and CLAUDE.md
- Each file should have one clear responsibility
- If a file is growing beyond plan intent, stop and report DONE_WITH_CONCERNS
- In existing code, follow established patterns — don't restructure outside your task
- No speculative abstractions. Three similar lines > premature abstraction.

## Escalation

It is always OK to stop and escalate. Bad work is worse than no work.

**STOP and escalate when:**
- The task requires architectural decisions with multiple valid approaches
- You need to understand code beyond what was provided
- You feel uncertain about your approach
- The task involves restructuring code the plan didn't anticipate

## Project-Specific Rules

- All models trained from scratch — no pretrained models
- Pure microstructure features only — no external event signals
- Transaction costs should be acknowledged even if not modeled
- Polymarket `/prices-history` returns empty for resolved markets — use trade records from Data API
- Filter out low-liquidity markets (< 10 trades)
- Kalshi candlestick OHLC fields can be null — handle gracefully
