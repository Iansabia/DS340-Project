#!/usr/bin/env bash
# Phase 14 paper-integrity validator. Runs grep-based checks for POL-04/05/06/07/08/09/10.
# Exits 0 if all checks pass, non-zero otherwise. Runs in ~1 second.

set -u
PAPER="PAPER_DRAFT.md"
FAIL=0

check() {
  local name="$1"; local expected="$2"; local actual="$3"
  if [[ "$actual" == "$expected" ]]; then
    printf "  [OK]   %-50s (got %s)\n" "$name" "$actual"
  else
    printf "  [FAIL] %-50s (want %s, got %s)\n" "$name" "$expected" "$actual"
    FAIL=1
  fi
}

check_ge() {
  local name="$1"; local minimum="$2"; local actual="$3"
  if (( actual >= minimum )); then
    printf "  [OK]   %-50s (got %s, want >= %s)\n" "$name" "$actual" "$minimum"
  else
    printf "  [FAIL] %-50s (got %s, want >= %s)\n" "$name" "$actual" "$minimum"
    FAIL=1
  fi
}

echo "== POL-04: Abstract word count =="
WC=$(awk '/^## Abstract/{f=1;next} /^---$/{if(f)exit} f' "$PAPER" | wc -w | tr -d ' ')
if (( WC <= 250 )); then
  printf "  [OK]   abstract_words <= 250                            (got %s)\n" "$WC"
else
  printf "  [FAIL] abstract_words <= 250                            (got %s)\n" "$WC"
  FAIL=1
fi

echo "== POL-05: References alphabetical + Cont entry present =="
REFS=$(sed -n '/^## References$/,/^---$/p' "$PAPER" | grep -cE "^[0-9]+\.")
check_ge "references_count" 14 "$REFS"
CONT=$(sed -n '/^## References$/,/^---$/p' "$PAPER" | grep -c "Cont")
check_ge "cont_kukanov_entry" 1 "$CONT"
# Alphabetical check: skip the [Anonymous] entry (no uppercase first letter), check the rest
DISORDER=$(sed -n '/^## References$/,/^---$/p' "$PAPER" | grep -oE "^[0-9]+\. [A-Z][a-zA-Z-]+" | awk '{print $2}' | sort -c 2>&1 | grep -c "disorder")
check "references_alphabetical" 0 "$DISORDER"

echo "== POL-06: Tables/Figures uniquely numbered =="
check "table_6_count"  1 "$(grep -c '^\*\*Table 6'  "$PAPER")"
check "table_7_count"  1 "$(grep -c '^\*\*Table 7'  "$PAPER")"
check "table_9_count"  1 "$(grep -c '^\*\*Table 9'  "$PAPER")"
check "table_10_count" 1 "$(grep -c '^\*\*Table 10' "$PAPER")"
APPB=$(awk '/^## Appendix B/,0' "$PAPER" | grep -c "^- \*\*Fig\. ")
check_ge "appendix_b_figure_bullets" 11 "$APPB"

echo "== POL-07: Per-pair Sharpe is the headline =="
PP=$(grep -c "per-pair" "$PAPER")
check_ge "per_pair_mentions" 3 "$PP"
STALE=$(grep -cE "0\.59 annualize|annualizes to 4\.3|annualizes to roughly 4\.3" "$PAPER")
check "stale_sharpe_claims" 0 "$STALE"

echo "== POL-08: Limitations + Fig 2 cap annotation =="
SURV=$(awk '/^### 6\.4 Limitations/,/^### 6\.[5-9]|^## /' "$PAPER" | grep -ic "survivorship")
check_ge "survivorship_in_6_4" 1 "$SURV"
LIVECOHORT=$(awk '/^### 6\.4 Limitations/,/^### 6\.[5-9]|^## /' "$PAPER" | grep -ic "live-cohort\|pair_id schema")
check_ge "live_cohort_in_6_4" 1 "$LIVECOHORT"

echo "== POL-09: AI-assistant disclosure =="
AI=$(sed -n '/^## Acknowledgments$/,/^## References$/p' "$PAPER" | grep -ic "claude\|anthropic")
check_ge "ai_disclosure" 1 "$AI"

echo "== POL-10: No residual TODOs/placeholders =="
TODO=$(grep -cE "TODO|FIXME|XXX|\[Insert|TBD" "$PAPER")
check "todo_placeholder_count" 0 "$TODO"
DEAD=$(grep -cE "§4\.6|Figure 2b|Fig\. 2b" "$PAPER")
check "dead_crossrefs" 0 "$DEAD"

echo
if (( FAIL == 0 )); then
  echo "ALL CHECKS PASSED"
  exit 0
else
  echo "SOME CHECKS FAILED"
  exit 1
fi
