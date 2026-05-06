import json
from pathlib import Path

import structlog
from PIL import Image

from config.prompts import DUCT_LOCATOR_PROMPT
from tools.vision_tools import (
    _extract_json_object,
    _normalize_type,
    _save_temp_png,
    claude_vision_call,
    label_finder,
)

logger = structlog.get_logger()

_CROP_SIZE = 600  # px — region cropped around each label center


def _locate_duct(image_path: str, label: dict, page_idx: int, counter: int) -> dict | None:
    """
    Crop a region around the label and ask Claude to return the duct's polygon.
    Returns a segment dict in full-page pixel coordinates, or None if not found.
    """
    img = Image.open(image_path)
    img_w, img_h = img.width, img.height

    cx, cy = int(label["x"]), int(label["y"])
    half = _CROP_SIZE // 2
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(img_w, cx + half)
    y2 = min(img_h, cy + half)

    crop = img.crop((x1, y1, x2, y2))
    tmp_path = _save_temp_png(crop)

    try:
        prompt = DUCT_LOCATOR_PROMPT.format(label_text=label["text"])
        raw = claude_vision_call(tmp_path, prompt)
        result = _extract_json_object(raw)
    except Exception as e:
        logger.error("locate_duct_failed", label=label["text"], error=str(e))
        return None
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    polygon = result.get("polygon")
    if not polygon or not isinstance(polygon, list) or len(polygon) < 3:
        logger.warning("locate_duct_no_polygon", label=label["text"])
        return None

    duct_type = _normalize_type(result.get("duct_type")) or "unknown"

    # Offset polygon from crop space back to full-page space
    full_polygon = [
        [
            max(0.0, min(float(p[0]) + x1, img_w)),
            max(0.0, min(float(p[1]) + y1, img_h)),
        ]
        for p in polygon
    ]

    return {
        "id": f"seg_{counter:03d}",
        "type": duct_type,
        "polygon": full_polygon,
        "nearby_labels": [label["text"]],
        "confidence": 0.9,
        "page": page_idx,
    }


def run_vision(state: dict) -> dict:
    """
    Run label-first vision pipeline:
      1. Find all dimension labels on each page (N"Ø, NxM).
      2. For each label, crop around it and locate the duct polygon.
    Updates state["duct_segments"] with the resulting segment dicts.
    """
    page_images: list[str] = state["page_images"]
    all_segments: list[dict] = []
    seg_counter = 1

    for page_idx, image_path in enumerate(page_images):
        raw_labels = label_finder(image_path)
        labels = json.loads(raw_labels)

        logger.info("vision_labels_found", page=page_idx, count=len(labels))

        page_segments: list[dict] = []
        for label in labels:
            seg = _locate_duct(image_path, label, page_idx, seg_counter)
            if seg:
                page_segments.append(seg)
                seg_counter += 1

        type_counts: dict[str, int] = {}
        for s in page_segments:
            type_counts[s["type"]] = type_counts.get(s["type"], 0) + 1

        logger.info(
            "vision_page_complete",
            page=page_idx,
            labels=len(labels),
            segments=len(page_segments),
            types=type_counts,
        )
        all_segments.extend(page_segments)

    state["duct_segments"] = all_segments
    logger.info("vision_complete", total_segments=len(all_segments))
    return state
