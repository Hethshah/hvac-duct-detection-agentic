"""
Phase 5 — Vision cross-validation.

For each AnnotatedDuct with confidence < VISION_CONFIDENCE_THRESHOLD
or length_mismatch=True:
  1. Render the PDF page to a full-resolution image.
  2. Crop a VISION_CROP_MARGIN_PT-padded region around the duct bbox.
  3. Ask Claude vision whether the crop contains a genuine duct.
  4. Update AnnotatedDuct.confidence from the vision result.

For input1.pdf (all-vector, no mismatches) Phase 5 produces 0 API calls.
"""

import base64
import io
import json
import re

import anthropic
import fitz
from PIL import Image

from config.settings import (
    VISION_CONFIDENCE_THRESHOLD,
    VISION_CROP_MARGIN_PT,
    VISION_MODEL,
    RASTER_SCALE,
)
from models.annotated_duct import AnnotatedDuct


def _needs_vision(duct: AnnotatedDuct) -> bool:
    return duct.confidence < VISION_CONFIDENCE_THRESHOLD or duct.length_mismatch


def _render_crop(pdf_path: str, duct: AnnotatedDuct, scale: int = RASTER_SCALE) -> bytes:
    """Return PNG bytes of the duct region + VISION_CROP_MARGIN_PT padding."""
    doc = fitz.open(pdf_path)
    page = doc[duct.page]
    rotation = page.rotation
    media_w = int(page.mediabox.width)
    media_h = int(page.mediabox.height)
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
    doc.close()

    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    m = VISION_CROP_MARGIN_PT
    mx0, my0, mx1, my1 = duct.rect

    # Convert padded media bbox → visual pixel bbox (matches raster_duct_extractor transforms)
    if rotation == 270:
        vx0, vx1 = (my0 - m) * scale, (my1 + m) * scale
        vy0, vy1 = (media_w - mx1 - m) * scale, (media_w - mx0 + m) * scale
    elif rotation == 90:
        vx0, vx1 = (media_h - my1 - m) * scale, (media_h - my0 + m) * scale
        vy0, vy1 = (mx0 - m) * scale, (mx1 + m) * scale
    elif rotation == 180:
        vx0, vx1 = (media_w - mx1 - m) * scale, (media_w - mx0 + m) * scale
        vy0, vy1 = (media_h - my1 - m) * scale, (media_h - my0 + m) * scale
    else:
        vx0, vx1 = (mx0 - m) * scale, (mx1 + m) * scale
        vy0, vy1 = (my0 - m) * scale, (my1 + m) * scale

    vx0 = max(0, int(vx0)); vx1 = min(img.width,  int(vx1))
    vy0 = max(0, int(vy0)); vy1 = min(img.height, int(vy1))

    buf = io.BytesIO()
    img.crop((vx0, vy0, vx1, vy1)).save(buf, format="PNG")
    return buf.getvalue()


def _call_vision(image_bytes: bytes, duct: AnnotatedDuct, client: anthropic.Anthropic) -> dict:
    """Query Claude vision on a duct crop. Returns {is_duct, confidence, notes}."""
    b64  = base64.standard_b64encode(image_bytes).decode()
    meas = f"{duct.length_ft_measured:.2f} ft"
    lbl  = f"{duct.length_ft_label:.2f} ft" if duct.length_ft_label else "none"
    ctx  = f"Orientation: {duct.orientation}. Measured length: {meas}. Label: {lbl}."
    if duct.length_mismatch:
        ctx += " Warning: measured and label lengths differ by >15%."

    prompt = (
        "This is a crop from an HVAC mechanical drawing. "
        f"{ctx}\n\n"
        "Is the central shape a rectangular sheet-metal air duct? "
        "Reply with ONLY a JSON object:\n"
        '{"is_duct": true/false, "confidence": 0.0-1.0, "notes": "brief reason"}'
    )

    msg = client.messages.create(
        model=VISION_MODEL,
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png", "data": b64,
                }},
                {"type": "text", "text": prompt},
            ],
        }],
    )

    text = msg.content[0].text
    match = re.search(r'\{[^}]+\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"is_duct": True, "confidence": 0.5, "notes": f"parse_error: {text[:80]}"}


def validate_ducts(
    pdf_path: str,
    annotated: list[AnnotatedDuct],
    api_key: str | None = None,
) -> tuple[list[AnnotatedDuct], list[dict]]:
    """
    Phase 5 pipeline: vision-validate low-confidence or mismatch ducts.

    Parameters
    ----------
    pdf_path  : path to the source PDF
    annotated : Phase 4 AnnotatedDuct list (confidence updated in-place)
    api_key   : Anthropic API key (falls back to ANTHROPIC_API_KEY env var)

    Returns
    -------
    (annotated, vision_log)
      annotated   : same list with confidence updated for reviewed ducts
      vision_log  : list of per-duct result dicts for debugging
    """
    candidates = [d for d in annotated if _needs_vision(d)]
    if not candidates:
        return annotated, []

    client = anthropic.Anthropic(api_key=api_key)
    vision_log: list[dict] = []

    for duct in candidates:
        img_bytes = _render_crop(pdf_path, duct)
        result    = _call_vision(img_bytes, duct, client)

        vision_log.append({
            "segment_id":       duct.segment_id,
            "input_confidence": round(duct.confidence, 3),
            "length_mismatch":  duct.length_mismatch,
            **result,
        })

        if result["is_duct"]:
            duct.confidence = max(duct.confidence, float(result["confidence"]))
        else:
            duct.confidence = round(1.0 - float(result["confidence"]), 3)

    return annotated, vision_log
