import re

import structlog

logger = structlog.get_logger()

_LP = re.compile(r'\b(LP|L\.P\.|LOW[\s\-]?PRES)', re.IGNORECASE)
_MP = re.compile(r'\b(MP|M\.P\.|MED[\s\-]?PRES|MEDIUM[\s\-]?PRES)', re.IGNORECASE)
_HP = re.compile(r'\b(HP|H\.P\.|HIGH[\s\-]?PRES)', re.IGNORECASE)


def classify_from_labels(nearby_labels: list[str]) -> str | None:
    """Return explicit pressure class if an LP / MP / HP marker is present."""
    combined = " ".join(nearby_labels)
    if _HP.search(combined):
        return "High"
    if _MP.search(combined):
        return "Medium"
    if _LP.search(combined):
        return "Low"
    return None


def classify_from_size(
    is_round: bool,
    diameter_in: int | None,
    width_in: int | None,
    height_in: int | None,
    duct_type: str,
) -> str:
    """Rule-based pressure class from duct dimensions (SMACNA construction standards).

    Round ducts  : ≤10" → Low  |  11–18" → Medium  |  >18" → High
    Rectangular  : ≤80 sq-in → Low  |  81–250 sq-in → Medium  |  >250 sq-in → High
    Fallback     : return/exhaust → Low  |  supply → Low
    """
    if is_round and diameter_in:
        if diameter_in <= 10:
            return "Low"
        if diameter_in <= 18:
            return "Medium"
        return "High"

    if width_in and height_in:
        area = width_in * height_in
        if area <= 80:
            return "Low"
        if area <= 250:
            return "Medium"
        return "High"

    if duct_type in ("return", "exhaust"):
        return "Low"
    return "Low"


def classify_pressure(segment: dict, measurement: dict) -> tuple[str, str]:
    """
    Classify pressure class and return (class, reason) tuple.

    Tries explicit drawing labels (LP/MP/HP) first, then falls back to SMACNA size rules.
    """
    nearby = segment.get("nearby_labels", [])
    explicit = classify_from_labels(nearby)
    if explicit:
        label_map = {"Low": "LP", "Medium": "MP", "High": "HP"}
        lbl = label_map.get(explicit, explicit)
        reason = f"Explicit '{lbl}' label found in drawing"
        logger.debug("pressure_explicit", segment_id=segment.get("id"), cls=explicit)
        return explicit, reason

    is_round   = measurement.get("is_round", False)
    diameter   = measurement.get("diameter_in")
    width      = measurement.get("width_in")
    height     = measurement.get("height_in")
    duct_type  = segment.get("type", "")

    cls = classify_from_size(is_round, diameter, width, height, duct_type)

    if is_round and diameter:
        if diameter <= 10:
            reason = f"{diameter}\" Ø ≤ 10\" — Low Pressure (SMACNA)"
        elif diameter <= 18:
            reason = f"{diameter}\" Ø is 11–18\" — Medium Pressure (SMACNA)"
        else:
            reason = f"{diameter}\" Ø > 18\" — High Pressure (SMACNA)"
    elif width and height:
        area = width * height
        if area <= 80:
            reason = f"{width}×{height} = {area} sq-in ≤ 80 — Low Pressure (SMACNA)"
        elif area <= 250:
            reason = f"{width}×{height} = {area} sq-in 81–250 — Medium Pressure (SMACNA)"
        else:
            reason = f"{width}×{height} = {area} sq-in > 250 — High Pressure (SMACNA)"
    else:
        reason = "No dimension found — defaulted to Low Pressure"

    logger.debug("pressure_inferred", segment_id=segment.get("id"), cls=cls)
    return cls, reason
