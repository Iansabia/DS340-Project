#!/usr/bin/env python3
"""Phase A.E1 realized-PNL report.

Consumes data/live/canonical_realized.jsonl (predictions enriched with
next-bar outcomes via src.live.canonical_inference.resolve_pending_shadow)
and computes:

    - Per-trade alpha (bps) with bootstrap 95% CI (10,000 resamples)
    - Per-trade Sharpe with bootstrap CI
    - Directional accuracy (sign agreement of prediction and realized change)
    - Sample size + time window
    - Per-prefix breakdown (KXWTI / KXWTIW / KXBRENTMON / KXBRENTW /
      KXBRENTD)
    - Comparison row to the writeup's backtest numbers

Trade rule (matches scripts/train_oil_canonical.py per_trade_outcomes()):
    Trade iff |canonical_pred| > 0.001 (PREDICTION_THRESHOLD)
    direction = sign(canonical_pred)
    per_trade_outcome = direction * realized_spread_change

Position-sizing convention: $100 per trade (matches the original paper
and the writeup). One basis-point of alpha = $0.01 per trade.

Usage:
    python scripts/canonical_realized_report.py
    python scripts/canonical_realized_report.py --since 2026-05-26T00:00:00Z
    python scripts/canonical_realized_report.py --json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
REALIZED_LOG = REPO_ROOT / "data" / "live" / "canonical_realized.jsonl"

PREDICTION_THRESHOLD = 0.001
POSITION_SIZE_USD = 100.0
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 42

# Backtest reference numbers from the writeup
# (experiments/results/canonical_oil/headline/xgboost.json, retrained
# on canonical_oil via scripts/train_oil_canonical.py).
BACKTEST_ALPHA_BPS = 18.96
BACKTEST_SHARPE = 0.0644
BACKTEST_TRADE_RATE = 0.87  # 4580 / 5251

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("realized_report")

OIL_PREFIXES = (
    "KXWTIW", "KXWTI", "KXBRENTMON", "KXBRENTW", "KXBRENTD",
    "KXBRENT", "KXCRUDE", "KXDIESEL", "KXHEATINGOIL", "KXGASOLINE",
    "KXMEXCUBOIL",
)


def detailed_prefix(ticker: str) -> str:
    for p in OIL_PREFIXES:
        if ticker.startswith(p):
            return p
    return ticker.split("-")[0] if ticker else "?"


def bootstrap_ci(values: np.ndarray, statistic, n_boot: int = N_BOOTSTRAP,
                 seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    n = len(values)
    if n < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = values[idx]
    boot_stats = np.asarray([statistic(s) for s in samples], dtype=float)
    return (float(np.percentile(boot_stats, 2.5)),
            float(np.percentile(boot_stats, 97.5)))


def per_trade_sharpe(trades: np.ndarray) -> float:
    if len(trades) < 2:
        return 0.0
    std = float(np.std(trades, ddof=1))
    if std <= 0:
        return 0.0
    return float(np.mean(trades) / std)


def alpha_bps(trades: np.ndarray) -> float:
    if len(trades) == 0:
        return 0.0
    return float(np.mean(trades) * 10_000.0)


def load_records(since_iso: str | None) -> list[dict]:
    if not REALIZED_LOG.exists():
        return []
    cutoff = None
    if since_iso:
        try:
            cutoff = datetime.fromisoformat(since_iso.replace("Z", "+00:00"))
        except Exception:
            log.warning("could not parse --since %r; ignoring", since_iso)
    out: list[dict] = []
    with open(REALIZED_LOG) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if cutoff is not None:
                ts_iso = r.get("ts_iso")
                if ts_iso:
                    try:
                        ts = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
                        if ts < cutoff:
                            continue
                    except Exception:
                        pass
            out.append(r)
    return out


def cohort_stats(records: list[dict]) -> dict:
    """Compute the headline stats for one cohort (all or per-prefix)."""
    n_total = len(records)
    if n_total == 0:
        return {"n_total": 0}

    would = [r for r in records if r.get("canonical_would_trade")]
    n_trades = len(would)
    trade_rate = n_trades / n_total if n_total else 0.0

    if n_trades == 0:
        return {
            "n_total": n_total,
            "n_trades": 0,
            "trade_rate": 0.0,
            "alpha_bps": 0.0,
            "alpha_bps_ci": [float("nan"), float("nan")],
            "per_trade_sharpe": 0.0,
            "per_trade_sharpe_ci": [float("nan"), float("nan")],
            "directional_accuracy": float("nan"),
            "total_pnl_usd": 0.0,
        }

    per_trade = np.asarray(
        [r["realized_pnl_per_dollar"] for r in would], dtype=float
    )
    sign_matches = sum(1 for r in would if r.get("realized_sign_match"))

    return {
        "n_total": n_total,
        "n_trades": n_trades,
        "trade_rate": float(trade_rate),
        "alpha_bps": alpha_bps(per_trade),
        "alpha_bps_ci": list(bootstrap_ci(per_trade, alpha_bps)),
        "per_trade_sharpe": per_trade_sharpe(per_trade),
        "per_trade_sharpe_ci": list(bootstrap_ci(per_trade, per_trade_sharpe)),
        "directional_accuracy": float(sign_matches / n_trades) if n_trades else float("nan"),
        "total_pnl_usd": float(per_trade.sum() * POSITION_SIZE_USD),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None, help="ISO timestamp; drop records older than this")
    ap.add_argument("--json", action="store_true", help="Emit JSON to stdout instead of pretty text")
    args = ap.parse_args()

    records = load_records(args.since)
    if not records:
        log.error("No records in %s%s", REALIZED_LOG,
                  f" since {args.since}" if args.since else "")
        return 1

    overall = cohort_stats(records)

    by_prefix: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_prefix[detailed_prefix(r["kalshi_ticker"])].append(r)
    prefix_stats = {k: cohort_stats(v) for k, v in by_prefix.items()}

    timestamps = sorted(r["ts_iso"] for r in records if "ts_iso" in r)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "realized_log_path": str(REALIZED_LOG),
        "since": args.since,
        "window": {
            "first_pred_iso": timestamps[0] if timestamps else None,
            "last_pred_iso": timestamps[-1] if timestamps else None,
            "n_records": len(records),
        },
        "overall": overall,
        "by_prefix": prefix_stats,
        "backtest_reference": {
            "alpha_bps": BACKTEST_ALPHA_BPS,
            "per_trade_sharpe": BACKTEST_SHARPE,
            "trade_rate": BACKTEST_TRADE_RATE,
            "source": "experiments/results/canonical_oil/headline/xgboost.json",
        },
        "config": {
            "prediction_threshold": PREDICTION_THRESHOLD,
            "position_size_usd": POSITION_SIZE_USD,
            "bootstrap_resamples": N_BOOTSTRAP,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    def fmt_ci(stat: float, ci: list[float]) -> str:
        if any(v != v for v in (stat, *ci)):  # NaN
            return "n/a"
        return f"{stat:+.4f}  [{ci[0]:+.4f}, {ci[1]:+.4f}]"

    print("=" * 70)
    print("Phase A.E1 — realized P&L report (canonical-oil shadow vs live)")
    print("=" * 70)
    print(f"Generated:   {report['generated_at']}")
    print(f"Log path:    {REALIZED_LOG}")
    if args.since:
        print(f"Since:       {args.since}")
    print(f"Window:      {report['window']['first_pred_iso']}  →  {report['window']['last_pred_iso']}")
    print(f"Records:     {report['window']['n_records']}")
    print()

    print("--- Overall ---")
    o = overall
    print(f"  n_total:                {o['n_total']}")
    print(f"  n_trades:               {o['n_trades']}  ({o['trade_rate']*100:.1f}% trade rate)")
    print(f"  alpha (bps/trade):      {fmt_ci(o['alpha_bps'], o['alpha_bps_ci'])}")
    print(f"  per-trade Sharpe:       {fmt_ci(o['per_trade_sharpe'], o['per_trade_sharpe_ci'])}")
    print(f"  directional accuracy:   {o['directional_accuracy']*100:.1f}%" if not np.isnan(o.get("directional_accuracy", float("nan"))) else "  directional accuracy:   n/a")
    print(f"  total P&L (USD):        ${o['total_pnl_usd']:+.2f}")
    print()

    print("--- Backtest reference (experiments/results/canonical_oil/headline/xgboost.json) ---")
    print(f"  alpha (bps/trade):      {BACKTEST_ALPHA_BPS:+.2f}")
    print(f"  per-trade Sharpe:       {BACKTEST_SHARPE:+.4f}")
    print(f"  trade rate:             {BACKTEST_TRADE_RATE*100:.1f}%")
    print()

    if o["n_trades"] >= 30:
        alpha_ratio = o["alpha_bps"] / BACKTEST_ALPHA_BPS if BACKTEST_ALPHA_BPS != 0 else 0
        ci_ex_zero = (o["alpha_bps_ci"][0] > 0) or (o["alpha_bps_ci"][1] < 0)
        print("--- Decision gate (per user's three-way framework) ---")
        print(f"  Live/backtest alpha ratio: {alpha_ratio*100:.1f}%")
        print(f"  Alpha 95% CI excludes zero: {ci_ex_zero}")
        if o["alpha_bps"] >= 9.5 and ci_ex_zero:
            print("  → Within 50% of backtest, CI excludes zero. (A) PROPOSE A2 candidate.")
        elif 3.0 <= o["alpha_bps"] < 9.5 and ci_ex_zero:
            print("  → Positive but substantially below backtest. (B) extend shadow.")
        else:
            print("  → Zero, negative, or CI straddles zero. (C) halt, document drift.")
        print()
    else:
        print(f"  (Sample size {o['n_trades']} < 30; decision gate not yet evaluable.)")
        print()

    print("--- By detailed prefix ---")
    print(f"  {'prefix':<14} {'n_tot':>6} {'n_tr':>5} {'trade%':>7} {'alpha_bps':>12} {'sharpe':>9} {'dir_acc':>8}")
    for prefix in sorted(prefix_stats, key=lambda k: -prefix_stats[k]["n_total"]):
        s = prefix_stats[prefix]
        dir_acc = s.get("directional_accuracy")
        dir_str = f"{dir_acc*100:.1f}%" if dir_acc is not None and not np.isnan(dir_acc) else "n/a"
        print(f"  {prefix:<14} {s['n_total']:>6} {s['n_trades']:>5} "
              f"{s['trade_rate']*100:>6.1f}% {s['alpha_bps']:>+11.2f} "
              f"{s['per_trade_sharpe']:>+9.4f} {dir_str:>8}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
