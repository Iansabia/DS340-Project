#!/usr/bin/env python3
"""Phase 17 paper-numerics auditor.

Cross-references every numeric claim in PAPER_DRAFT.md against the
single source of truth at experiments/results/canonical/headline.json.

Outputs a Markdown audit log naming each match, mismatch, and
unresolvable claim with line numbers.

Usage:
    python3 scripts/audit_paper_numbers.py [--paper PAPER_DRAFT.md] [--out path]

Exit code: 0 when zero mismatches, 1 when any mismatch remains.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
from pathlib import Path

# Regex patterns. Order matters: more specific first.
DOLLAR_RE = re.compile(r"(?:\+|-|−)?\\?\$([0-9,]+(?:\.[0-9]+)?)")
PCT_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*%")
PP_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*pp\b")
SHARPE_RE = re.compile(r"[Ss]harpe[^0-9]{1,30}([0-9]\.[0-9]+)")
RMSE_RE = re.compile(r"RMSE[^0-9]{1,30}([0-9]\.[0-9]+)")
TRADES_RE = re.compile(r"([0-9]{3,5})\s*(?:trades?|num_trades)")
BPS_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*bps")

MODEL_ALIASES = {
    "linear_regression": ["Linear Regression", "LR ", " LR.", " LR,", " LR)", "LR-solo", "LR solo", "(LR"],
    "xgboost": ["XGBoost", "XGB ", " XGB.", " XGB,", " XGB)", "XGB-solo"],
    "gru": ["GRU"],
    "lstm": ["LSTM"],
    "tft": ["TFT"],
    "ppo_raw": ["PPO-Raw", "PPO Raw", "PPO-raw"],
    "ppo_filtered": [
        "PPO+autoencoder",
        "PPO + Autoencoder",
        "PPO+AE",
        "PPO-Filtered",
        "PPO + autoencoder",
        "PPO+ autoencoder",
        "autoencoder anomaly filter",
    ],
    "naive": ["Naive ", "Naive (", "naive baseline"],
    "volume": ["Volume baseline", "Volume (higher", "volume baseline", " Volume "],
}

DOLLAR_TOLERANCE_PCT = 0.5    # +/- 0.5% for dollar amounts
SHARPE_TOLERANCE = 0.01
RMSE_TOLERANCE = 0.005
PCT_TOLERANCE_PP = 1.0        # +/- 1 percentage point for win rates
TRADES_TOLERANCE_PCT = 1.0    # +/- 1% for trade counts


def find_nearest_model(text_window: str) -> str | None:
    """Find which model the surrounding text is talking about."""
    # Prefer the alias whose match is *closest* to the centre of the window.
    centre = len(text_window) // 2
    best = None
    best_dist = 1_000_000
    for model_key, aliases in MODEL_ALIASES.items():
        for alias in aliases:
            idx = text_window.find(alias)
            if idx == -1:
                continue
            dist = abs(idx - centre)
            if dist < best_dist:
                best_dist = dist
                best = model_key
    return best


def _window(paper_text: str, line_no: int, line: str, radius: int = 200) -> str:
    char_start = sum(len(L) + 1 for L in paper_text.splitlines()[: line_no - 1])
    return paper_text[max(0, char_start - radius) : char_start + len(line) + radius]


def _classify_dollar(line: str, model: str | None, paper_val: float, canonical: dict) -> dict:
    if model is None or model not in canonical["models"]:
        return {
            "metric": "dollar",
            "paper": paper_val,
            "canonical": None,
            "status": "UNRESOLVABLE",
        }
    canon_val = canonical["models"][model].get("total_pnl")
    if canon_val is None:
        return {
            "metric": "total_pnl",
            "paper": paper_val,
            "canonical": None,
            "status": "UNRESOLVABLE",
        }
    tol = max(abs(canon_val) * DOLLAR_TOLERANCE_PCT / 100.0, 0.5)
    status = "MATCH" if abs(paper_val - canon_val) <= tol else "MISMATCH"
    return {
        "metric": "total_pnl",
        "paper": paper_val,
        "canonical": canon_val,
        "status": status,
    }


def _classify_sharpe(model: str | None, paper_val: float, canonical: dict) -> dict:
    if model is None or model not in canonical["models"]:
        return {"metric": "sharpe", "paper": paper_val, "canonical": None, "status": "UNRESOLVABLE"}
    canon_val = canonical["models"][model].get("sharpe_per_trade")
    if canon_val is None:
        return {"metric": "sharpe", "paper": paper_val, "canonical": None, "status": "UNRESOLVABLE"}
    status = "MATCH" if abs(paper_val - canon_val) <= SHARPE_TOLERANCE else "MISMATCH"
    return {
        "metric": "sharpe_per_trade",
        "paper": paper_val,
        "canonical": canon_val,
        "status": status,
    }


def _classify_rmse(model: str | None, paper_val: float, canonical: dict) -> dict:
    if model is None or model not in canonical["models"]:
        return {"metric": "rmse", "paper": paper_val, "canonical": None, "status": "UNRESOLVABLE"}
    canon_val = canonical["models"][model].get("rmse")
    if canon_val is None:
        return {"metric": "rmse", "paper": paper_val, "canonical": None, "status": "UNRESOLVABLE"}
    status = "MATCH" if abs(paper_val - canon_val) <= RMSE_TOLERANCE else "MISMATCH"
    return {"metric": "rmse", "paper": paper_val, "canonical": canon_val, "status": status}


def _classify_trades(model: str | None, paper_val: float, canonical: dict) -> dict:
    if model is None or model not in canonical["models"]:
        return {"metric": "trades", "paper": paper_val, "canonical": None, "status": "UNRESOLVABLE"}
    canon_val = canonical["models"][model].get("num_trades")
    if canon_val is None:
        return {"metric": "num_trades", "paper": paper_val, "canonical": None, "status": "UNRESOLVABLE"}
    tol = max(canon_val * TRADES_TOLERANCE_PCT / 100.0, 5)
    status = "MATCH" if abs(paper_val - canon_val) <= tol else "MISMATCH"
    return {"metric": "num_trades", "paper": paper_val, "canonical": canon_val, "status": status}


def _classify_bps(model: str | None, paper_val: float, canonical: dict) -> dict:
    if model is None or model not in canonical["models"]:
        return {"metric": "bps", "paper": paper_val, "canonical": None, "status": "UNRESOLVABLE"}
    canon_val = canonical["models"][model].get("alpha_bps_per_trade")
    if canon_val is None:
        return {"metric": "alpha_bps_per_trade", "paper": paper_val, "canonical": None, "status": "UNRESOLVABLE"}
    status = "MATCH" if abs(paper_val - canon_val) <= 0.5 else "MISMATCH"
    return {"metric": "alpha_bps_per_trade", "paper": paper_val, "canonical": canon_val, "status": status}


# Sections whose numbers come from non-canonical result files and should NOT be
# cross-checked against headline.json:
#   - §5.2 Walk-Forward Validation (per-window numbers, ablation_walk_forward/*)
#   - §5.3 Per-Category Breakdown (per-category aggregates, not headline-pair)
#   - §5.4 Data-Scaling Curve (per-bar slices, data_scaling/*)
#   - §5.5 XGBoost Hyperparameter Sweep (top-10 configs, sweep_xgb/*)
#   - §5.6 Transaction-Cost Sensitivity (per-fee, fee_sensitivity/*)
#   - §5.7 SHAP Feature Importance (no model-level dollar figures)
#   - §5.8 Honest Sharpe-Ratio Accounting (uses canonical XGBoost per-trade Sharpe)
#   - §5.9 Live vs Backtest Reconciliation (live data, reconciliation/*)
#   - §5.10 Feature Ablation (ablation_features/*)
#   - §5.11 Ensemble Formalization (ensemble/*)
#   - §5.12 Lookback Window Sensitivity (ablation_lookback/*)
#   - §5.13 Minimum Spread Threshold (ablation_threshold/*)
#   - §6.x Discussion (often quotes per-window or projected numbers)
#   - §7  Future Work
#   - §8  Conclusions — AUDITED, this is a headline section
#   - References / Appendix / Acknowledgments
#
# Headline-section markers (numbers HERE must reconcile to headline.json):
#   - Abstract
#   - §5.1 Headline Model Comparison
#   - §6.3 Negative Result on PPO
#   - §8 Conclusions

NON_CANONICAL_SECTIONS = (
    "### 5.2 Walk-Forward Validation",
    "### 5.3 Per-Category Breakdown",
    "### 5.4 Data-Scaling Curve",
    "### 5.5 XGBoost Hyperparameter Sweep",
    "### 5.6 Transaction-Cost Sensitivity",
    "### 5.7 SHAP Feature Importance",
    "### 5.8 Honest Sharpe-Ratio Accounting",
    "### 5.9 Live vs Backtest Reconciliation",
    "### 5.10 Feature Ablation",
    "### 5.11 Ensemble Formalization",
    "### 5.12 Lookback Window Sensitivity",
    "### 5.13 Minimum Spread Threshold",
    "### 6.1 Why Does Simpler Beat More Complex",
    "### 6.2 How Each Model Would Improve",
    "#### 6.2.1",
    "#### 6.2.2",
    "#### 6.2.3",
    "#### 6.2.4",
    "### 6.4 Limitations",
    "## 7. ",
    "## Acknowledgments",
    "## References",
    "## Appendix",
)

HEADLINE_SECTIONS = (
    "## Abstract",
    "### 5.1 Headline Model Comparison",
    "### 6.3 The Negative Result on PPO",
    "## 8. Conclusions",
)


def _section_for_line(paper_text: str, line_no: int) -> str:
    """Return the most recent header marker preceding line_no."""
    current = ""
    for i, line in enumerate(paper_text.splitlines(), start=1):
        if i >= line_no:
            break
        if line.startswith("#"):
            current = line.strip()
    return current


def _is_headline_section(section: str) -> bool:
    if any(section.startswith(m) for m in HEADLINE_SECTIONS):
        return True
    return False


def _is_non_canonical_section(section: str) -> bool:
    if any(section.startswith(m) for m in NON_CANONICAL_SECTIONS):
        return True
    return False


# Inline phrases that should always be skipped regardless of section:
SKIP_LINE_TOKENS = (
    "arXiv",
    "Polymarket processed",
    "$1 billion",
    "API",
    "Polygon gas",
    "$0.0",
)


def audit_line(line_no: int, line: str, paper_text: str, canonical: dict) -> list[dict]:
    """Extract every numeric claim in a single line and check against canonical."""
    if any(tok in line for tok in SKIP_LINE_TOKENS):
        return []
    section = _section_for_line(paper_text, line_no)
    if _is_non_canonical_section(section) and not _is_headline_section(section):
        return []
    # Markdown table header rows
    stripped = line.strip()
    if stripped.startswith("|") and "---" in stripped:
        return []
    if stripped.startswith("|") and ("Tier" in stripped and "Model" in stripped):
        return []

    results: list[dict] = []
    window = _window(paper_text, line_no, line)
    model = find_nearest_model(window)
    context = line.strip()[:140]

    # Dollars
    for m in DOLLAR_RE.finditer(line):
        try:
            paper_val = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        prefix = m.group(0)
        sign = -1.0 if (prefix.startswith("-") or prefix.startswith("−")) else 1.0
        paper_val *= sign
        rec = _classify_dollar(line, model, paper_val, canonical)
        rec.update({"line": line_no, "model": model or "?", "context": context})
        results.append(rec)

    # Sharpe
    for m in SHARPE_RE.finditer(line):
        try:
            paper_val = float(m.group(1))
        except ValueError:
            continue
        rec = _classify_sharpe(model, paper_val, canonical)
        rec.update({"line": line_no, "model": model or "?", "context": context})
        results.append(rec)

    # RMSE
    for m in RMSE_RE.finditer(line):
        try:
            paper_val = float(m.group(1))
        except ValueError:
            continue
        rec = _classify_rmse(model, paper_val, canonical)
        rec.update({"line": line_no, "model": model or "?", "context": context})
        results.append(rec)

    # Trade counts
    for m in TRADES_RE.finditer(line):
        try:
            paper_val = float(m.group(1))
        except ValueError:
            continue
        rec = _classify_trades(model, paper_val, canonical)
        rec.update({"line": line_no, "model": model or "?", "context": context})
        results.append(rec)

    # bps (alpha per trade)
    for m in BPS_RE.finditer(line):
        try:
            paper_val = float(m.group(1))
        except ValueError:
            continue
        rec = _classify_bps(model, paper_val, canonical)
        rec.update({"line": line_no, "model": model or "?", "context": context})
        results.append(rec)

    return results


def render_markdown(results: list[dict], canonical_path: str, paper_path: str) -> str:
    n_match = sum(1 for r in results if r["status"] == "MATCH")
    n_mismatch = sum(1 for r in results if r["status"] == "MISMATCH")
    n_unresolvable = sum(1 for r in results if r["status"] == "UNRESOLVABLE")

    out: list[str] = [
        "# Phase 17-03: PAPER_DRAFT.md Numeric Audit",
        "",
        f"**Generated:** {_dt.datetime.now(_dt.timezone.utc).isoformat()}",
        f"**Canonical source:** `{canonical_path}`",
        f"**Paper:** `{paper_path}`",
        "",
        "## Summary",
        "",
        f"- MATCH: {n_match}",
        f"- MISMATCH: {n_mismatch}",
        f"- UNRESOLVABLE: {n_unresolvable}",
        "",
        "## Tolerances applied",
        "",
        f"- Dollar amounts: ±{DOLLAR_TOLERANCE_PCT}% of canonical (or ±$0.50 floor)",
        f"- Sharpe (per-trade): ±{SHARPE_TOLERANCE}",
        f"- RMSE: ±{RMSE_TOLERANCE}",
        f"- Trade counts: ±{TRADES_TOLERANCE_PCT}% (or ±5 floor)",
        f"- bps (alpha/trade): ±0.5",
        "",
        "## Mismatches (action required)",
        "",
        "| Line | Model | Metric | Paper | Canonical | Context |",
        "|------|-------|--------|-------|-----------|---------|",
    ]
    mismatches = [r for r in results if r["status"] == "MISMATCH"]
    if not mismatches:
        out.append("| — | — | — | — | — | (none) |")
    else:
        for r in mismatches:
            out.append(
                f"| {r['line']} | {r['model']} | {r['metric']} | "
                f"{r['paper']} | {r['canonical']} | `{r['context']}` |"
            )

    out.extend([
        "",
        "## Unresolvable (manual review — typically auxiliary or non-canonical numbers)",
        "",
        "| Line | Metric | Paper Value | Context |",
        "|------|--------|-------------|---------|",
    ])
    unres = [r for r in results if r["status"] == "UNRESOLVABLE"]
    if not unres:
        out.append("| — | — | — | (none) |")
    else:
        for r in unres:
            out.append(
                f"| {r['line']} | {r['metric']} | {r['paper']} | `{r['context']}` |"
            )

    out.extend([
        "",
        "## All matches (verification)",
        "",
        "| Line | Model | Metric | Value |",
        "|------|-------|--------|-------|",
    ])
    matches = [r for r in results if r["status"] == "MATCH"]
    if not matches:
        out.append("| — | — | — | (none) |")
    else:
        for r in matches:
            out.append(f"| {r['line']} | {r['model']} | {r['metric']} | {r['paper']} |")

    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", default="PAPER_DRAFT.md")
    parser.add_argument(
        "--canonical",
        default="experiments/results/canonical/headline.json",
    )
    parser.add_argument(
        "--out",
        default=".planning/phases/17-model-rerun-paper-number-audit-pitch-standard-conversion/17-03-NUMBER-AUDIT.md",
    )
    args = parser.parse_args()

    canonical = json.loads(Path(args.canonical).read_text())
    paper_text = Path(args.paper).read_text()

    all_results: list[dict] = []
    for line_no, line in enumerate(paper_text.splitlines(), start=1):
        all_results.extend(audit_line(line_no, line, paper_text, canonical))

    md = render_markdown(all_results, args.canonical, args.paper)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(md)

    n_match = sum(1 for r in all_results if r["status"] == "MATCH")
    n_mismatch = sum(1 for r in all_results if r["status"] == "MISMATCH")
    n_unresolvable = sum(1 for r in all_results if r["status"] == "UNRESOLVABLE")
    print(
        f"Wrote {args.out}: {n_match} match / {n_mismatch} mismatch / {n_unresolvable} unresolvable"
    )
    return 0 if n_mismatch == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
