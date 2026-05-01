import json

import structlog
from strands import tool

logger = structlog.get_logger()


def _type_diversity_score(segments: list[dict]) -> float:
    """Score 0-1 based on number of unique duct types detected."""
    if not segments:
        return 0.0
    types = {s.get("type") for s in segments} - {None}
    return len(types) / 3.0  # 3 = supply, return, exhaust


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def confidence_scorer(duct_segments_json: str, measurements_json: str) -> float:
    """
    Compute overall detection quality score (0.0–1.0).

    Formula (calibrated to approve ≥100% label match + avg conf ≥0.75 + ≥2 duct types):
      score = 0.50 * label_coverage
            + 0.30 * avg_confidence
            + 0.20 * type_diversity_score

    label_coverage  = matched_segments / total_segments
    avg_confidence  = mean segment confidence
    type_diversity  = unique_types / 3
    """
    segments: list[dict] = json.loads(duct_segments_json)
    measurements: list[dict] = json.loads(measurements_json)

    if not segments:
        return 0.0

    avg_conf = sum(s.get("confidence", 0.0) for s in segments) / len(segments)

    matched = sum(1 for m in measurements if not m.get("unmatched", True))
    label_cov = matched / len(measurements) if measurements else 0.0

    type_div = _type_diversity_score(segments)

    score = 0.50 * label_cov + 0.30 * avg_conf + 0.20 * type_div
    return round(min(score, 1.0), 4)


@tool
def diff_checker(
    duct_segments_json: str,
    measurements_json: str,
    confidence_threshold: float = 0.85,
) -> str:
    """
    Identify quality issues in detection results.
    Returns a JSON array of human-readable issue strings for use in the reflexion prompt.
    """
    segments: list[dict] = json.loads(duct_segments_json)
    measurements: list[dict] = json.loads(measurements_json)
    issues: list[str] = []

    if not segments:
        issues.append("No duct segments detected — drawing may need higher DPI or a different detection approach")
        return json.dumps(issues)

    # Low average confidence
    avg_conf = sum(s.get("confidence", 0.0) for s in segments) / len(segments)
    if avg_conf < 0.75:
        issues.append(
            f"Average detection confidence is low ({avg_conf:.0%}) — "
            "re-examine duct fills and colors more carefully"
        )

    # Segments below threshold
    low_conf = [s for s in segments if s.get("confidence", 1.0) < confidence_threshold]
    if low_conf:
        issues.append(
            f"{len(low_conf)} segment(s) below confidence threshold ({confidence_threshold:.0%}) "
            f"— IDs: {', '.join(s.get('id', 'unknown') for s in low_conf[:5])}"
        )

    # Unmatched labels
    unmatched = [m for m in measurements if m.get("unmatched", True)]
    if unmatched:
        issues.append(
            f"{len(unmatched)} segment(s) have no matched dimension label — "
            "check for dimension annotations near these regions"
        )

    # Missing CFM
    no_cfm = [m for m in measurements if not m.get("cfm")]
    if no_cfm:
        issues.append(
            f"{len(no_cfm)} segment(s) have no CFM value — "
            "look for airflow labels (e.g. 'F 150', 'A 700', '800 CFM') near duct runs"
        )

    # Type diversity
    types = {s.get("type") for s in segments} - {None}
    if len(types) == 1:
        only = next(iter(types))
        issues.append(
            f"Only '{only}' ducts detected — verify return/exhaust ducts are visible "
            f"(return=gray/dashed, exhaust=orange)"
        )
    elif len(types) == 2 and len(segments) > 5:
        missing = ({"supply", "return", "exhaust"} - types).pop()
        issues.append(
            f"'{missing}' duct type not detected — check for it in the drawing"
        )

    # Low total count (mechanical plans typically have many segments)
    if len(segments) < 5:
        issues.append(
            f"Only {len(segments)} segment(s) detected — expected more for a full mechanical plan; "
            "try splitting the image into smaller regions for better detection"
        )

    logger.info("diff_checker", total_issues=len(issues), segments=len(segments))
    return json.dumps(issues)
