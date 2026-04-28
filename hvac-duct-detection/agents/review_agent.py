import json

import anthropic
import structlog

from config.settings import settings
from tools.review_tools import confidence_scorer, diff_checker

logger = structlog.get_logger()


def run_review(state: dict) -> dict:
    """
    Run review pipeline: score detection quality and generate reflexion feedback.

    - Computes review_score via confidence_scorer
    - Collects quality issues via diff_checker
    - If score < threshold AND retries remain: calls Claude to generate
      targeted feedback for the vision agent's next attempt
    - Sets state["review_score"] and state["reviewer_feedback"]
    """
    segments = state.get("duct_segments", [])
    measurements = state.get("measurements", [])
    retry_count = state.get("retry_count", 0)

    segs_json = json.dumps(segments)
    meas_json = json.dumps(measurements)

    score = confidence_scorer(segs_json, meas_json)
    issues = json.loads(diff_checker(segs_json, meas_json, settings.confidence_threshold))

    needs_retry = score < settings.confidence_threshold and retry_count < settings.max_retries

    feedback = ""
    if needs_retry:
        feedback = _generate_feedback(issues, score, state)

    state["review_score"] = score
    state["reviewer_feedback"] = feedback

    logger.info(
        "review_complete",
        score=score,
        issues=len(issues),
        needs_retry=needs_retry,
        retry_count=retry_count,
        approved=not needs_retry,
    )
    return state


def _generate_feedback(issues: list[str], score: float, state: dict) -> str:
    """
    Ask Claude to turn raw issues into targeted vision guidance.
    Falls back to joining issue strings if the API call fails.
    """
    segments = state.get("duct_segments", [])
    measurements = state.get("measurements", [])

    type_counts: dict[str, int] = {}
    for s in segments:
        t = s.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    matched_count = sum(1 for m in measurements if not m.get("unmatched", True))

    context = (
        f"Review score: {score:.2f} (target ≥ {settings.confidence_threshold:.2f})\n\n"
        f"Issues detected:\n"
        + "\n".join(f"- {i}" for i in issues)
        + f"\n\nDetection summary:\n"
        f"- Total segments: {len(segments)}\n"
        f"- Type breakdown: {type_counts}\n"
        f"- Labels matched: {matched_count}/{len(measurements)}"
    )

    prompt = (
        "You are reviewing an HVAC duct detection run on a mechanical floor plan.\n\n"
        f"{context}\n\n"
        "Provide 2-3 specific, actionable instructions for the vision agent's next attempt. "
        "Focus on: what was likely missed, which colors/patterns to look for, "
        "and which regions to examine more carefully. Under 120 words."
    )

    try:
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        response = client.messages.create(
            model=settings.review_model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        logger.warning("feedback_llm_failed", error=str(e))
        return "\n".join(issues)
