import json
import math
import re

import structlog
from strands import tool

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# Round duct: "8"Ø", "12"Ø", "18"ø", "8Ø" (diameter in inches)
_ROUND_PATTERN = re.compile(
    r'(\d+\.?\d*)\s*["“”]?\s*[ØøoØø∅]',
    re.IGNORECASE,
)

# Rectangular duct: "24x12", "22"x14"", "18×10", "24 x 12"
_RECT_PATTERN = re.compile(
    r'(\d+\.?\d*)\s*["“”]?\s*[xX×]\s*(\d+\.?\d*)',
)

# Standard CFM: "800 CFM", "1,200 cfm", "800cfm", "800 C.F.M."
_CFM_PATTERN = re.compile(r'(\d[\d,]*)\s*(?:CFM|C\.F\.M\.)', re.IGNORECASE)

# Diffuser/grille flow tags: "F 150", "A 700", "SA-150", "EA 300"
_ZONE_FLOW_PATTERN = re.compile(r'^[A-Z]{1,2}[-\s]+(\d{2,4})$')

# Bare airflow numbers next to equipment tags: "(150)", "=150", "~150"
_BARE_FLOW_PATTERN = re.compile(r'^[=(~]?\s*(\d{2,4})\s*[)=]?$')

# Proximity threshold: max pixel distance from label center to segment bbox boundary
_PROXIMITY_MAX_PX = 700


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _parse_round(text: str) -> int | None:
    m = _ROUND_PATTERN.search(text)
    return int(float(m.group(1))) if m else None


def _parse_rect(text: str) -> tuple[int, int] | None:
    m = _RECT_PATTERN.search(text)
    if m:
        return int(float(m.group(1))), int(float(m.group(2)))
    return None


def _parse_cfm(text: str) -> int | None:
    m = _CFM_PATTERN.search(text)
    if m:
        return int(m.group(1).replace(",", ""))
    t = text.strip()
    m2 = _ZONE_FLOW_PATTERN.match(t)
    if m2:
        return int(m2.group(1))
    m3 = _BARE_FLOW_PATTERN.match(t)
    if m3:
        val = int(m3.group(1))
        if 50 <= val <= 9999:  # sanity range: realistic CFM values only
            return val
    return None


def _bbox_from_polygon(polygon: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _dist_point_to_bbox(
    px: float, py: float, bx1: float, by1: float, bx2: float, by2: float
) -> float:
    """Euclidean distance from a point to the nearest edge of a bbox. 0 if inside."""
    dx = max(bx1 - px, 0.0, px - bx2)
    dy = max(by1 - py, 0.0, py - by2)
    return math.sqrt(dx * dx + dy * dy)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def dimension_extractor(text_blocks_json: str) -> str:
    """
    Scan OCR text blocks for HVAC dimension and CFM annotations.
    Detects round ducts (e.g. '8\"Ø'), rectangular ducts (e.g. '24x12'),
    and CFM values (e.g. '800 CFM', 'F 150').
    Returns a JSON array of parsed label dicts with type, values, and pixel coords.
    """
    text_blocks: list[dict] = json.loads(text_blocks_json)
    labels: list[dict] = []

    for block in text_blocks:
        text = block.get("text", "").strip()
        x, y = block.get("x", 0.0), block.get("y", 0.0)
        page = block.get("page", 0)

        diam = _parse_round(text)
        rect = _parse_rect(text)
        cfm = _parse_cfm(text)

        if diam is not None:
            labels.append({"text": text, "label_type": "round", "diameter_in": diam,
                            "width_in": None, "height_in": None, "cfm": None,
                            "x": x, "y": y, "page": page})
        elif rect is not None:
            labels.append({"text": text, "label_type": "rect",
                            "diameter_in": None, "width_in": rect[0], "height_in": rect[1],
                            "cfm": None, "x": x, "y": y, "page": page})

        if cfm is not None:
            labels.append({"text": text, "label_type": "cfm",
                            "diameter_in": None, "width_in": None, "height_in": None,
                            "cfm": cfm, "x": x, "y": y, "page": page})

    logger.info("dimension_extractor", found=len(labels))
    return json.dumps(labels)


@tool
def label_matcher(segment_json: str, dimension_labels_json: str, scale_ratio: float) -> str:
    """
    Associate the best-matching dimension and CFM labels with a duct segment.

    Matching priority:
      1. Vision-identified nearby_labels (parsed directly from segment['nearby_labels'])
      2. Proximity search in dimension_labels from OCR text blocks (same page, within 500px)

    Also computes physical duct length from the segment polygon bounding box and scale_ratio.
    Returns a JSON-encoded MeasurementRecord dict.
    """
    segment: dict = json.loads(segment_json)
    dimension_labels: list[dict] = json.loads(dimension_labels_json)

    seg_id = segment.get("id", "unknown")
    seg_type = segment.get("type", "unknown")
    seg_page = segment.get("page", 0)
    polygon = segment.get("polygon", [])
    nearby_labels: list[str] = segment.get("nearby_labels", [])

    # Compute bounding box and pixel length
    if polygon:
        bx1, by1, bx2, by2 = _bbox_from_polygon(polygon)
    else:
        bx1 = by1 = bx2 = by2 = 0.0
    pixel_length = max(bx2 - bx1, by2 - by1) if polygon else 0.0

    # --- Step 1: parse vision nearby_labels ---
    is_round = False
    diameter_in: int | None = None
    width_in: int | None = None
    height_in: int | None = None
    cfm: int | None = None

    for lbl in nearby_labels:
        if diameter_in is None:
            d = _parse_round(lbl)
            if d is not None:
                is_round = True
                diameter_in = d
        if width_in is None:
            r = _parse_rect(lbl)
            if r is not None:
                width_in, height_in = r
        if cfm is None:
            cfm = _parse_cfm(lbl)

    # --- Step 2: proximity fallback from OCR text blocks (same page) ---
    if (diameter_in is None and width_in is None) or cfm is None:
        same_page = [lb for lb in dimension_labels if lb.get("page", 0) == seg_page]
        scored: list[tuple[float, dict]] = []
        for lb in same_page:
            lx = lb["x"] + (lb.get("w", 0) or 0) / 2
            ly = lb["y"] + (lb.get("h", 0) or 0) / 2
            dist = _dist_point_to_bbox(lx, ly, bx1, by1, bx2, by2)
            if dist <= _PROXIMITY_MAX_PX:
                scored.append((dist, lb))
        scored.sort(key=lambda t: t[0])

        for _, lb in scored:
            if diameter_in is None and width_in is None and lb["label_type"] in ("round", "rect"):
                if lb["label_type"] == "round":
                    is_round = True
                    diameter_in = lb["diameter_in"]
                else:
                    width_in = lb["width_in"]
                    height_in = lb["height_in"]
            if cfm is None and lb["label_type"] == "cfm":
                cfm = lb["cfm"]

    # --- Compute length ---
    length_ft = scale_calculator(scale_ratio, pixel_length) if scale_ratio > 0 else None

    has_dim = (diameter_in is not None) or (width_in is not None)
    unmatched = not has_dim and cfm is None

    record = {
        "segment_id": seg_id,
        "type": seg_type,
        "is_round": is_round,
        "diameter_in": diameter_in,
        "width_in": width_in,
        "height_in": height_in,
        "cfm": cfm,
        "length_ft": length_ft,
        "bbox": [bx1, by1, bx2, by2],
        "polygon": polygon,
        "unmatched": unmatched,
    }
    logger.debug("label_matcher", segment_id=seg_id, matched=not unmatched,
                 is_round=is_round, diameter=diameter_in, dim=f"{width_in}x{height_in}", cfm=cfm)
    return json.dumps(record)


@tool
def scale_calculator(scale_ratio: float, pixel_length: float) -> float:
    """
    Convert a pixel distance to feet using the pixels-per-foot scale ratio.
    Returns 0.0 if scale_ratio is zero or negative.
    """
    if scale_ratio <= 0:
        return 0.0
    return round(pixel_length / scale_ratio, 2)
