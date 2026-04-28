import json

import structlog

from config.prompts import VISION_FOCUSED_PROMPT
from config.settings import settings
from tools.vision_tools import (
    _bbox_from_polygon,
    _extract_json_from_text,
    claude_vision_call,
    crop_region,
    segment_detector,
)

logger = structlog.get_logger()


def run_vision(state: dict) -> dict:
    """
    Run vision pipeline: detect duct segments across all page images.
    Updates state["duct_segments"] with merged, deduplicated DuctSegment dicts.
    """
    page_images: list[str] = state["page_images"]
    text_blocks: list[dict] = state.get("text_blocks", [])
    reviewer_feedback: str = state.get("reviewer_feedback", "")

    all_segments: list[dict] = []

    for page_idx, image_path in enumerate(page_images):
        page_blocks = [b for b in text_blocks if b.get("page", 0) == page_idx]
        page_blocks_json = json.dumps(page_blocks)

        # If this is a retry run, inject reviewer feedback into the detector
        if reviewer_feedback and state.get("retry_count", 0) > 0:
            from config.prompts import VISION_RETRY_PROMPT
            from tools.vision_tools import (
                VISION_MAX_PX,
                _deduplicate_segments,
                _offset_polygon,
                _resize_for_vision,
                _save_temp_png,
                _split_quadrants,
            )
            from pathlib import Path
            from PIL import Image

            retry_prompt = VISION_RETRY_PROMPT.format(feedback=reviewer_feedback)
            img = Image.open(image_path)
            quadrants = _split_quadrants(img)
            retry_segs: list[dict] = []
            seg_counter = 1
            for q_idx, (quad_img, ox, oy) in enumerate(quadrants):
                resized, scale = _resize_for_vision(quad_img)
                inv_scale = 1.0 / scale if scale > 0 else 1.0
                tmp_path = _save_temp_png(resized)
                try:
                    raw = claude_vision_call(tmp_path, retry_prompt)
                    segs = _extract_json_from_text(raw)
                except Exception as e:
                    logger.error("vision_retry_quadrant_failed", q_idx=q_idx, error=str(e))
                    segs = []
                finally:
                    Path(tmp_path).unlink(missing_ok=True)
                for seg in segs:
                    if isinstance(seg.get("polygon"), list) and len(seg["polygon"]) >= 3:
                        seg["polygon"] = _offset_polygon(seg["polygon"], ox, oy, inv_scale)
                        seg["id"] = f"seg_retry_{seg_counter:03d}"
                        seg["page"] = page_idx
                        retry_segs.append(seg)
                        seg_counter += 1
            page_segments = _deduplicate_segments(retry_segs)
        else:
            raw = segment_detector(image_path, page_blocks_json)
            page_segments = json.loads(raw)

        # Low-confidence retry: crop and re-inspect flagged segments
        refined: list[dict] = []
        for seg in page_segments:
            if seg.get("confidence", 1.0) < settings.confidence_threshold:
                bbox = list(_bbox_from_polygon(seg["polygon"]))
                crop_path = crop_region(image_path, json.dumps(bbox))
                try:
                    raw_focused = claude_vision_call(crop_path, VISION_FOCUSED_PROMPT)
                    focused_segs = _extract_json_from_text(raw_focused)
                    if focused_segs:
                        best = max(focused_segs, key=lambda s: s.get("confidence", 0.0))
                        # Keep original polygon (full-page coords) but update type/confidence
                        seg["type"] = best.get("type", seg["type"])
                        seg["confidence"] = max(best.get("confidence", 0.0), seg["confidence"])
                        seg["nearby_labels"] = best.get("nearby_labels", seg.get("nearby_labels", []))
                except Exception as e:
                    logger.warning("focused_retry_failed", seg_id=seg["id"], error=str(e))
            refined.append(seg)

        # Set page index on all segments
        for seg in refined:
            seg["page"] = page_idx

        type_counts = {}
        for seg in refined:
            type_counts[seg.get("type", "unknown")] = type_counts.get(seg.get("type", "unknown"), 0) + 1
        low_conf = sum(1 for s in refined if s.get("confidence", 1.0) < settings.confidence_threshold)

        logger.info(
            "vision_page_complete",
            page=page_idx,
            segments=len(refined),
            types=type_counts,
            low_confidence=low_conf,
        )
        all_segments.extend(refined)

    state["duct_segments"] = all_segments
    logger.info("vision_complete", total_segments=len(all_segments))
    return state
