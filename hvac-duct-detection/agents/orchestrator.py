import time

import structlog

from agents.annotation_agent import run_annotation
from agents.ingestion_agent import run_ingestion
from agents.measurement_agent import run_measurement
from agents.review_agent import run_review
from agents.vision_agent import run_vision
from config.settings import settings

logger = structlog.get_logger()


def _init_state(
    pdf_path: str,
    output_dir: str,
    session_id: str = "",
    page_range: str = "",
    scale_ratio_override: float | None = None,
) -> dict:
    return {
        "pdf_path": pdf_path,
        "output_dir": output_dir,
        "session_id": session_id,
        "page_range": page_range,
        "scale_ratio_override": scale_ratio_override,
        "page_images": [],
        "text_blocks": [],
        "scale_ratio": 0.0,
        "duct_segments": [],
        "measurements": [],
        "output_pdf": "",
        "output_pngs": [],
        "review_score": 0.0,
        "retry_count": 0,
        "reviewer_feedback": "",
    }


def _build_summary(state: dict) -> dict:
    pngs = state.get("output_pngs", [])
    return {
        "session_id": state.get("session_id", ""),
        "input_path": state["pdf_path"],
        "output_dir": state["output_dir"],
        "output_pdf": state["output_pdf"],
        "output_pngs": pngs,
        "segments_detected": len(state["duct_segments"]),
        "segments_labelled": sum(
            1 for m in state["measurements"] if not m.get("unmatched", True)
        ),
        "review_score": round(state["review_score"], 4),
        "retries": state["retry_count"],
    }


def _validate_state(state: dict, required_keys: list[str], stage: str) -> None:
    for key in required_keys:
        val = state.get(key)
        if val is None or val == [] or val == "":
            logger.warning("state_validation_empty", stage=stage, key=key)


def run_pipeline(
    pdf_path: str,
    output_dir: str | None = None,
    confidence_threshold: float | None = None,
    max_retries: int | None = None,
    page_range: str = "",
    scale_ratio_override: float | None = None,
    session_id: str = "",
) -> dict:
    """
    Run the full HVAC duct detection pipeline with reflexion loop.

    Flow:
      Ingestion (once) → [Vision → Measurement → Annotation → Review] × retries

    The inner loop retries when review_score < confidence_threshold, up to max_retries.
    Each retry injects reviewer_feedback into the Vision agent and backs off exponentially.

    Returns a summary dict: output_pdf, segments_detected, segments_labelled,
    review_score, retries.
    """
    threshold = confidence_threshold if confidence_threshold is not None else settings.confidence_threshold
    max_r = max_retries if max_retries is not None else settings.max_retries
    out_dir = output_dir or settings.output_dir

    state = _init_state(pdf_path, out_dir, session_id=session_id, page_range=page_range, scale_ratio_override=scale_ratio_override)
    logger.info("pipeline_start", pdf=pdf_path, output_dir=out_dir,
                threshold=threshold, max_retries=max_r,
                page_range=page_range or "all", scale_ratio_override=scale_ratio_override)

    # --- Phase 1: Ingestion (runs once) ---
    state = run_ingestion(state)
    _validate_state(state, ["page_images", "text_blocks"], "ingestion")

    # --- Phases 2–5: Vision → Measurement → Annotation → Review (with reflexion) ---
    while True:
        prev_score = state["review_score"]

        state = run_vision(state)
        _validate_state(state, ["duct_segments"], "vision")

        state = run_measurement(state)
        _validate_state(state, ["measurements"], "measurement")

        state = run_annotation(state)
        _validate_state(state, ["output_pdf"], "annotation")

        state = run_review(state)

        score = state["review_score"]
        approved = score >= threshold or state["retry_count"] >= max_r

        logger.info(
            "pipeline_cycle_complete",
            retry=state["retry_count"],
            score=score,
            prev_score=prev_score,
            approved=approved,
        )

        if approved:
            if score < threshold:
                logger.warning(
                    "pipeline_accepted_below_threshold",
                    score=score,
                    threshold=threshold,
                    retries=state["retry_count"],
                )
            break

        # --- Reflexion: increment counter, back-off, loop ---
        state["retry_count"] += 1
        backoff = 2 ** state["retry_count"]
        logger.info(
            "reflexion_retry",
            retry=state["retry_count"],
            score_delta=round(score - prev_score, 4),
            backoff_s=backoff,
            feedback_preview=state["reviewer_feedback"][:100],
        )
        time.sleep(backoff)

    summary = _build_summary(state)
    logger.info("pipeline_complete", **summary)
    return summary
