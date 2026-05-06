import base64
import json
import re
import tempfile
import time
from pathlib import Path

import anthropic
import structlog
from PIL import Image
from strands import tool

from config.prompts import VISION_DETECTION_PROMPT, VISION_FOCUSED_PROMPT
from config.settings import settings

logger = structlog.get_logger()

VISION_MAX_PX = 2000  # max pixels on longest side per quadrant sent to API
_RATE_LIMIT_MAX_RETRIES = 4  # max retries on 429 / RateLimitError


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _image_to_b64(path: str) -> tuple[str, str]:
    """Return (base64_data, media_type) for a PNG or JPEG image."""
    suffix = Path(path).suffix.lower()
    media_type = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8"), media_type


def _extract_json_object(text: str) -> dict:
    """Robustly extract a JSON object from a model response."""
    code_block = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if code_block:
        try:
            return json.loads(code_block.group(1))
        except json.JSONDecodeError:
            pass
    obj_match = re.search(r"\{[\s\S]*\}", text)
    if obj_match:
        try:
            return json.loads(obj_match.group(0))
        except json.JSONDecodeError:
            pass
    return {}


def _extract_json_from_text(text: str) -> list[dict]:
    """Robustly extract a JSON array from a model response."""
    # Strip markdown code fences
    code_block = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if code_block:
        try:
            return json.loads(code_block.group(1))
        except json.JSONDecodeError:
            pass
    # Find first JSON array in raw text
    array_match = re.search(r"\[[\s\S]*\]", text)
    if array_match:
        try:
            return json.loads(array_match.group(0))
        except json.JSONDecodeError:
            pass
    return []


def _bbox_from_polygon(polygon: list[list[float]]) -> tuple[float, float, float, float]:
    """Compute axis-aligned bounding box (x1, y1, x2, y2) from a polygon."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _iou(poly1: list[list[float]], poly2: list[list[float]]) -> float:
    """Intersection-over-Union of two polygons (approximated via bounding boxes)."""
    a = _bbox_from_polygon(poly1)
    b = _bbox_from_polygon(poly2)
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _deduplicate_segments(segments: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    """Remove duplicate detections using IoU. Keep the higher-confidence segment."""
    segments = sorted(segments, key=lambda s: s.get("confidence", 0.0), reverse=True)
    kept: list[dict] = []
    for seg in segments:
        duplicate = any(
            _iou(seg["polygon"], k["polygon"]) >= iou_threshold for k in kept
        )
        if not duplicate:
            kept.append(seg)
    return kept


def _resize_for_vision(img: Image.Image) -> tuple[Image.Image, float]:
    """Downsample to VISION_MAX_PX on longest side. Returns (image, scale)."""
    max_dim = max(img.width, img.height)
    if max_dim <= VISION_MAX_PX:
        return img, 1.0
    scale = VISION_MAX_PX / max_dim
    new_w = int(img.width * scale)
    new_h = int(img.height * scale)
    return img.resize((new_w, new_h), Image.LANCZOS), scale


def _split_quadrants(img: Image.Image) -> list[tuple[Image.Image, int, int]]:
    """Split into 4 quadrants. Returns [(quad_img, offset_x, offset_y), ...]."""
    w, h = img.width, img.height
    mx, my = w // 2, h // 2
    return [
        (img.crop((0, 0, mx, my)), 0, 0),
        (img.crop((mx, 0, w, my)), mx, 0),
        (img.crop((0, my, mx, h)), 0, my),
        (img.crop((mx, my, w, h)), mx, my),
    ]


def _save_temp_png(img: Image.Image) -> str:
    """Save image to a named temp file and return the path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp.name, format="PNG")
    return tmp.name


def _offset_polygon(polygon: list[list[float]], ox: int, oy: int, inv_scale: float) -> list[list[float]]:
    """Scale polygon up from resized space and add quadrant offset."""
    return [[p[0] * inv_scale + ox, p[1] * inv_scale + oy] for p in polygon]


def _normalize_type(raw: str | None) -> str | None:
    """Map LLM type strings to one of supply / return / exhaust, or None to drop."""
    if not raw:
        return None
    t = raw.lower().replace("_", "").replace("-", "").replace(" ", "")
    if "supply" in t:
        return "supply"
    if "return" in t:
        return "return"
    if "exhaust" in t:
        return "exhaust"
    return None


# ---------------------------------------------------------------------------
# Scale + label helpers (not @tool — called directly by ingestion/vision agents)
# ---------------------------------------------------------------------------

def detect_scale_from_image(image_path: str, dpi: int = 300) -> float:
    """
    Ask Claude to read the drawing scale notation (e.g. 1/4"=1'-0") from a page image.
    Returns pixels_per_foot, or 0.0 if not found.
    """
    from config.prompts import SCALE_READER_PROMPT

    img = Image.open(image_path)
    resized, _ = _resize_for_vision(img)
    tmp_path = _save_temp_png(resized)
    try:
        raw = claude_vision_call(tmp_path, SCALE_READER_PROMPT)
        result = _extract_json_object(raw)
    except Exception as e:
        logger.error("detect_scale_from_image_failed", error=str(e))
        return 0.0
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not result.get("found"):
        logger.warning("detect_scale_from_image", status="not_found")
        return 0.0

    num = result.get("numerator")
    den = result.get("denominator")
    if num and den and den > 0:
        ratio = round(dpi * num / den, 4)
        logger.info("detect_scale_from_image", scale_text=result.get("scale_text"), ratio=ratio)
        return ratio

    return 0.0


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def claude_vision_call(image_path: str, prompt: str) -> str:
    """
    Send an image to Claude's vision API with a prompt and return the raw text response.
    Image is base64-encoded and sent via the Anthropic messages API.
    Automatically retries on rate-limit errors with exponential back-off.
    """
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    img_b64, media_type = _image_to_b64(image_path)

    for attempt in range(_RATE_LIMIT_MAX_RETRIES):
        try:
            response = client.messages.create(
                model=settings.vision_model,
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {"type": "base64", "media_type": media_type, "data": img_b64},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
            return response.content[0].text
        except anthropic.RateLimitError as e:
            if attempt >= _RATE_LIMIT_MAX_RETRIES - 1:
                raise
            backoff = 2 ** (attempt + 1)
            logger.warning("rate_limit_retry", attempt=attempt + 1, backoff_s=backoff)
            time.sleep(backoff)


@tool
def segment_detector(image_path: str, page_text_blocks_json: str) -> str:
    """
    Detect HVAC duct segments in a full page image using quadrant-level vision analysis.
    Splits the page into 4 quadrants, calls vision on each, merges and deduplicates by IoU.
    Returns a JSON array of DuctSegment-compatible dicts with coordinates in full-page pixel space.
    """
    img = Image.open(image_path)
    quadrants = _split_quadrants(img)

    all_segments: list[dict] = []
    seg_counter = 1

    for q_idx, (quad_img, ox, oy) in enumerate(quadrants):
        resized, scale = _resize_for_vision(quad_img)
        inv_scale = 1.0 / scale if scale > 0 else 1.0
        tmp_path = _save_temp_png(resized)

        try:
            raw = claude_vision_call(tmp_path, VISION_DETECTION_PROMPT)
            segs = _extract_json_from_text(raw)
        except Exception as e:
            logger.error("segment_detector_quadrant_failed", quadrant=q_idx, error=str(e))
            segs = []
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        for seg in segs:
            if not isinstance(seg.get("polygon"), list) or len(seg["polygon"]) < 3:
                continue
            seg_type = _normalize_type(seg.get("type"))
            if seg_type is None:
                logger.debug("segment_type_filtered", raw_type=seg.get("type"))
                continue
            seg["type"] = seg_type
            seg["polygon"] = _offset_polygon(seg["polygon"], ox, oy, inv_scale)
            seg["id"] = f"seg_{seg_counter:03d}"
            seg["page"] = 0
            all_segments.append(seg)
            seg_counter += 1

        logger.info("segment_detector_quadrant", q_idx=q_idx, found=len(segs))

    deduped = _deduplicate_segments(all_segments)

    # Clamp polygon coordinates to full-page bounds to prevent out-of-bounds annotations
    img_w, img_h = img.width, img.height
    for seg in deduped:
        seg["polygon"] = [
            [max(0.0, min(float(p[0]), img_w)), max(0.0, min(float(p[1]), img_h))]
            for p in seg["polygon"]
        ]

    logger.info("segment_detector_complete", total=len(all_segments), after_dedup=len(deduped))
    return json.dumps(deduped)


@tool
def crop_region(image_path: str, bbox_json: str, padding: int = 50) -> str:
    """
    Crop a rectangular sub-region from a page image for focused re-inspection.
    bbox_json: JSON-encoded [x1, y1, x2, y2] in full-page pixel coordinates.
    Returns the file path of the saved cropped image.
    """
    bbox = json.loads(bbox_json)
    img = Image.open(image_path)
    x1 = max(0, int(bbox[0]) - padding)
    y1 = max(0, int(bbox[1]) - padding)
    x2 = min(img.width, int(bbox[2]) + padding)
    y2 = min(img.height, int(bbox[3]) + padding)
    cropped = img.crop((x1, y1, x2, y2))
    stem = Path(image_path).stem
    out_path = str(Path(image_path).parent / f"{stem}_crop_{x1}_{y1}_{x2}_{y2}.png")
    cropped.save(out_path)
    logger.info("crop_region", bbox=[x1, y1, x2, y2], out=out_path)
    return out_path


@tool
def label_finder(image_path: str) -> str:
    """
    Find all HVAC duct cross-section dimension labels in a floor plan page image.
    Looks for round labels (N"Ø / NØ) and rectangular labels (NxM / N"xM").
    Returns a JSON array of {text, label_type, x, y} dicts in full-page pixel coordinates.
    """
    from config.prompts import LABEL_FINDER_PROMPT

    img = Image.open(image_path)
    resized, scale = _resize_for_vision(img)
    inv_scale = 1.0 / scale if scale > 0 else 1.0
    tmp_path = _save_temp_png(resized)

    try:
        raw = claude_vision_call(tmp_path, LABEL_FINDER_PROMPT)
        labels = _extract_json_from_text(raw)
    except Exception as e:
        logger.error("label_finder_failed", error=str(e))
        labels = []
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # Scale coordinates back to full-page pixel space
    for lbl in labels:
        lbl["x"] = round(lbl.get("x", 0) * inv_scale, 1)
        lbl["y"] = round(lbl.get("y", 0) * inv_scale, 1)

    logger.info("label_finder", image=image_path, found=len(labels))
    return json.dumps(labels)
