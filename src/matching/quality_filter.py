"""Post-matching quality filters for candidate pairs.

Filters out false positives caused by:
- Temporal mismatch (same question, different year)
- Contract direction mismatch (above vs below vs exactly)
- Threshold number mismatch (4.3% vs 3.4%)

Applied after scoring and deduplication, before curation.
"""
import logging
import re
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Maximum days between resolution dates for a valid match
MAX_RESOLUTION_GAP_DAYS = 30

# Minimum confidence score to keep
MIN_CONFIDENCE = 0.6


def parse_date(date_str: str) -> datetime | None:
    """Parse ISO date string to datetime, handling various formats."""
    if not date_str:
        return None
    try:
        # Handle full ISO format with timezone
        clean = date_str.replace("Z", "+00:00")
        return datetime.fromisoformat(clean)
    except (ValueError, TypeError):
        pass
    try:
        # Handle date-only format
        return datetime.strptime(date_str[:10], "%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def extract_direction(question: str) -> str | None:
    """Extract contract direction from question text.

    Returns 'above', 'below', 'exactly', 'between', or None.
    """
    q = question.lower()

    if any(k in q for k in ["above", "over", "exceed", "more than", "higher than", "at least"]):
        return "above"
    if any(k in q for k in ["below", "under", "less than", "lower than", "drop to"]):
        return "below"
    if any(k in q for k in ["exactly", "be exactly"]):
        return "exactly"
    if "between" in q:
        return "between"
    return None


def extract_threshold(question: str) -> float | None:
    """Extract the primary numeric threshold from a question.

    E.g., "Will unemployment be above 4.3%?" -> 4.3
          "Will BTC be above $80,000?" -> 80000
    """
    q = question.lower()

    # Match patterns like: above 4.3%, above $80,000, above 80000, above 4.3
    patterns = [
        r"(?:above|below|exceed|over|under|exactly|be)\s+\$?([\d,]+\.?\d*)\s*%?",
        r"\$([\d,]+\.?\d*)",
        r"([\d,]+\.?\d*)\s*%",
    ]

    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            num_str = match.group(1).replace(",", "")
            try:
                return float(num_str)
            except ValueError:
                continue
    return None


def directions_compatible(dir1: str | None, dir2: str | None) -> bool:
    """Check if two contract directions are compatible for arbitrage.

    Compatible: above/above, below/below, None/anything
    Incompatible: above/below, above/exactly, below/exactly
    """
    if dir1 is None or dir2 is None:
        return True  # Can't determine, don't filter
    if dir1 == dir2:
        return True
    # above+between or below+between are borderline, allow
    if "between" in (dir1, dir2):
        return True
    return False


def thresholds_compatible(t1: float | None, t2: float | None) -> bool:
    """Check if two thresholds are close enough to be the same contract.

    Allows 5% relative tolerance or 0.5 absolute tolerance (for percentages).
    """
    if t1 is None or t2 is None:
        return True  # Can't determine, don't filter

    # Absolute tolerance for small numbers (percentages)
    if abs(t1 - t2) <= 0.5:
        return True

    # Relative tolerance for large numbers (dollar amounts)
    if t1 != 0 and t2 != 0:
        ratio = max(t1, t2) / min(t1, t2)
        if ratio <= 1.05:  # Within 5%
            return True

    return False


def filter_candidates(candidates: list[dict]) -> list[dict]:
    """Apply all quality filters to scored candidates.

    Filters:
    1. Minimum confidence score
    2. Resolution date proximity (within MAX_RESOLUTION_GAP_DAYS)
    3. Contract direction compatibility
    4. Threshold number compatibility

    Returns filtered list with rejection reasons logged.
    """
    filtered = []
    stats = {
        "total": len(candidates),
        "low_confidence": 0,
        "temporal_mismatch": 0,
        "direction_mismatch": 0,
        "threshold_mismatch": 0,
        "passed": 0,
    }

    for c in candidates:
        # 1. Confidence threshold
        if c.get("confidence_score", 0) < MIN_CONFIDENCE:
            stats["low_confidence"] += 1
            continue

        # 2. Resolution date proximity
        k_date = parse_date(c.get("kalshi_resolution_date", ""))
        p_date = parse_date(c.get("polymarket_resolution_date", ""))

        if k_date and p_date:
            gap = abs((k_date - p_date).days)
            if gap > MAX_RESOLUTION_GAP_DAYS:
                stats["temporal_mismatch"] += 1
                continue

        # 3. Direction compatibility
        k_dir = extract_direction(c.get("kalshi_question", ""))
        p_dir = extract_direction(c.get("polymarket_question", ""))

        if not directions_compatible(k_dir, p_dir):
            stats["direction_mismatch"] += 1
            continue

        # 4. Threshold compatibility
        k_thresh = extract_threshold(c.get("kalshi_question", ""))
        p_thresh = extract_threshold(c.get("polymarket_question", ""))

        if not thresholds_compatible(k_thresh, p_thresh):
            stats["threshold_mismatch"] += 1
            continue

        # Add extracted metadata for curation display
        c["kalshi_direction"] = k_dir
        c["polymarket_direction"] = p_dir
        c["kalshi_threshold"] = k_thresh
        c["polymarket_threshold"] = p_thresh
        c["resolution_gap_days"] = abs((k_date - p_date).days) if k_date and p_date else None

        stats["passed"] += 1
        filtered.append(c)

    logger.info(
        f"Quality filter: {stats['total']} candidates -> {stats['passed']} passed "
        f"(rejected: {stats['low_confidence']} low confidence, "
        f"{stats['temporal_mismatch']} temporal mismatch, "
        f"{stats['direction_mismatch']} direction mismatch, "
        f"{stats['threshold_mismatch']} threshold mismatch)"
    )
    return filtered
