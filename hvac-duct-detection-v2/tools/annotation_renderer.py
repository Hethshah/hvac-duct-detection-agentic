"""
Phase 6 — Annotation renderer.

Draws a single blue centerline through the middle of each detected duct.
The line endpoints are derived from the Phase 1 bounding box (rect + orientation)
so the line length precisely matches the duct's measured long_pt.

Color coding:
  blue   — normal / unlabeled (no label available)
  red    — length mismatch flagged (measured vs label >15%)
"""

import fitz
from PIL import Image, ImageDraw, ImageFont

from config.settings import (
    RENDER_DPI, OUTLINE_COLOR_RGB, OUTLINE_WIDTH_PX, LABEL_FONT_SIZE_PX,
)
from models.annotated_duct import AnnotatedDuct

_COLOR_NORMAL   = OUTLINE_COLOR_RGB   # blue (55, 98, 227)
_COLOR_MISMATCH = (198, 40, 40)       # red  — only used when length label disagrees
_COLOR_BLACK    = (0, 0, 0)
_COLOR_WHITE    = (255, 255, 255)


def _load_font(size: int):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            pass
    return ImageFont.load_default()


def _media_rect_to_pixel(
    rect: list[float],
    rotation: int,
    media_w: int,
    media_h: int,
    scale: float,
) -> tuple[int, int, int, int]:
    """Convert media-coord bbox [x0,y0,x1,y1] → visual pixel bbox (px0,py0,px1,py1)."""
    mx0, my0, mx1, my1 = rect
    if rotation == 270:
        px0, px1 = int(my0 * scale), int(my1 * scale)
        py0, py1 = int((media_w - mx1) * scale), int((media_w - mx0) * scale)
    elif rotation == 90:
        px0, px1 = int((media_h - my1) * scale), int((media_h - my0) * scale)
        py0, py1 = int(mx0 * scale), int(mx1 * scale)
    elif rotation == 180:
        px0, px1 = int((media_w - mx1) * scale), int((media_w - mx0) * scale)
        py0, py1 = int((media_h - my1) * scale), int((media_h - my0) * scale)
    else:
        px0, px1 = int(mx0 * scale), int(mx1 * scale)
        py0, py1 = int(my0 * scale), int(my1 * scale)
    return px0, py0, px1, py1


def _outline_color(duct: AnnotatedDuct) -> tuple[int, int, int]:
    return _COLOR_NORMAL


def _centerline_pixels(
    duct: AnnotatedDuct,
    rotation: int,
    media_w: int,
    media_h: int,
    scale: float,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Return (p1, p2) in visual pixel coords for the duct's centerline.

    The centerline is computed from the Phase 1 bbox (rect + orientation):
      H  →  horizontal midline, full x-span
      V  →  vertical midline, full y-span
      D  →  diagonal from corner to corner

    This exactly matches the long_pt measured in Phase 1.
    """
    x0, y0, x1, y1 = duct.rect
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2

    if duct.orientation == "H":
        start_m = (x0, cy)
        end_m   = (x1, cy)
    elif duct.orientation == "V":
        start_m = (cx, y0)
        end_m   = (cx, y1)
    elif duct.centerline and len(duct.centerline) == 2:
        start_m = tuple(duct.centerline[0])
        end_m   = tuple(duct.centerline[1])
    else:  # D fallback — bbox diagonal
        start_m = (x0, y0)
        end_m   = (x1, y1)

    def to_px(mx, my):
        if rotation == 270:
            return int(my * scale), int((media_w - mx) * scale)
        if rotation == 90:
            return int((media_h - my) * scale), int(mx * scale)
        if rotation == 180:
            return int((media_w - mx) * scale), int((media_h - my) * scale)
        return int(mx * scale), int(my * scale)

    p1 = to_px(*start_m)
    p2 = to_px(*end_m)
    return p1, p2


def _draw_label_box(
    draw: ImageDraw.ImageDraw,
    idx: int,
    p1: tuple[int, int],
    p2: tuple[int, int],
    color: tuple,
    font,
    line_w: int,
    img_w: int,
    img_h: int,
) -> None:
    """Small numbered white box with a leader line to the duct midpoint."""
    mx = (p1[0] + p2[0]) // 2
    my = (p1[1] + p2[1]) // 2

    text = str(idx)
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        tw = th = LABEL_FONT_SIZE_PX

    pad  = max(3, line_w * 2)
    bw   = tw + 2 * pad
    bh   = th + 2 * pad
    gap  = 8
    lw   = max(1, line_w // 2)

    # Place box above midpoint; clamp to image bounds
    lx = max(0, min(mx - bw // 2, img_w - bw))
    ly = max(0, min(my - bh - gap, img_h - bh))

    bx0, by0, bx1, by1 = lx, ly, lx + bw, ly + bh

    draw.line([(mx, my), ((bx0 + bx1) // 2, by1)], fill=_COLOR_BLACK, width=lw)
    draw.rectangle([bx0, by0, bx1, by1], fill=_COLOR_WHITE, outline=_COLOR_BLACK, width=lw)
    draw.text((bx0 + pad, by0 + pad), text, fill=color, font=font)


def render_annotated_page(
    pdf_path: str,
    annotated: list[AnnotatedDuct],
    output_path: str,
    page_index: int = 0,
    dpi: int = RENDER_DPI,
) -> None:
    """
    Render the PDF page with a blue centerline through each detected duct.

    The line spans the exact duct length from the Phase 1 bounding box so it
    matches the long_pt measurement precisely.

    Parameters
    ----------
    pdf_path    : source PDF
    annotated   : Phase 4/5 AnnotatedDuct list
    output_path : destination PNG
    page_index  : 0-indexed page
    dpi         : output resolution
    """
    scale = dpi / 72.0

    doc      = fitz.open(pdf_path)
    page     = doc[page_index]
    rotation = page.rotation
    media_w  = int(page.mediabox.width)
    media_h  = int(page.mediabox.height)
    pix      = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
    doc.close()

    img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    draw = ImageDraw.Draw(img)
    font = _load_font(LABEL_FONT_SIZE_PX)

    page_ducts = [d for d in annotated if d.page == page_index]
    line_w     = max(1, OUTLINE_WIDTH_PX)

    for idx, duct in enumerate(page_ducts, start=1):
        color = _outline_color(duct)
        p1, p2 = _centerline_pixels(duct, rotation, media_w, media_h, scale)
        draw.line([p1, p2], fill=color, width=line_w)
        _draw_label_box(draw, idx, p1, p2, color, font, line_w, img.width, img.height)

    img.save(output_path, format="PNG")
