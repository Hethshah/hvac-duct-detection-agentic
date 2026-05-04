import math
from pathlib import Path

import structlog
from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfgen import canvas as rl_canvas

logger = structlog.get_logger()

# duct type → RGB color
COLORS: dict[str, tuple[int, int, int]] = {
    "supply": (21, 101, 192),    # #1565C0 blue
    "return": (198, 40, 40),     # #C62828 red
    "exhaust": (230, 81, 0),     # #E65100 orange
    "unknown": (100, 100, 100),  # gray fallback
}

FILL_ALPHA = 110    # ~43% opacity
OUTLINE_ALPHA = 230
FONT_PATH = "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"
FONT_SIZE = 48      # px at 300 DPI ≈ 11.5 pt printed


def _load_font(size: int = FONT_SIZE) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except (OSError, IOError):
        return ImageFont.load_default()


def _ft_to_feetinches(ft: float) -> str:
    total_in = round(ft * 12)
    feet, inches = divmod(total_in, 12)
    return f"{feet}'-{inches:02d}\""


def _format_label(measurement: dict) -> str:
    """Build compact label string from a MeasurementRecord dict."""
    parts: list[str] = []
    if measurement.get("is_round") and measurement.get("diameter_in"):
        parts.append(f'{measurement["diameter_in"]}"Ø')
    elif measurement.get("width_in") and measurement.get("height_in"):
        parts.append(f'{measurement["width_in"]}×{measurement["height_in"]}')
    if measurement.get("cfm"):
        parts.append(f'{measurement["cfm"]}CFM')
    return "  ".join(parts)


def _polygon_centroid(polygon: list[list[float]]) -> tuple[float, float]:
    cx = sum(p[0] for p in polygon) / len(polygon)
    cy = sum(p[1] for p in polygon) / len(polygon)
    return cx, cy


def _duct_centerline(
    polygon: list[list[float]],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Return (start, end) midpoints of the duct run for a 4-vertex polygon.

    The two shortest edges are the cross-section ends; their midpoints define
    the start and end of the duct run centerline.
    Falls back to bbox long-axis for non-quad polygons.
    """
    if len(polygon) == 4:
        edges = sorted(
            (math.hypot(polygon[(i+1)%4][0]-polygon[i][0],
                        polygon[(i+1)%4][1]-polygon[i][1]), i)
            for i in range(4)
        )
        def mid(idx: int) -> tuple[float, float]:
            a, b = polygon[idx], polygon[(idx + 1) % 4]
            return (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        return mid(edges[0][1]), mid(edges[1][1])

    # Fallback: bbox long-axis midpoints
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    bx1, by1, bx2, by2 = min(xs), min(ys), max(xs), max(ys)
    if (bx2 - bx1) >= (by2 - by1):
        my = (by1 + by2) / 2
        return (bx1, my), (bx2, my)
    mx = (bx1 + bx2) / 2
    return (mx, by1), (mx, by2)


def draw_bbox(
    draw: ImageDraw.ImageDraw,
    polygon: list[list[float]],
    duct_type: str,
) -> None:
    """
    Draw a semi-transparent filled polygon for a duct segment onto a PIL RGBA draw context.
    Color: supply=blue, return=red, exhaust=orange.
    """
    rgb = COLORS.get(duct_type, COLORS["unknown"])
    pts = [tuple(p) for p in polygon]
    draw.polygon(pts, fill=(*rgb, FILL_ALPHA), outline=(*rgb, OUTLINE_ALPHA))


def draw_length_annotation(
    draw: ImageDraw.ImageDraw,
    polygon: list[list[float]],
    length_ft: float,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    img_size: tuple[int, int] = (10800, 7200),
) -> None:
    """Draw a dimension line along the duct run with a feet-inches length label.

    The line runs between the midpoints of the two cross-section (short) edges,
    visually annotating which distance was measured and at what scale.
    """
    cl = _duct_centerline(polygon)
    if cl is None:
        return
    start, end = cl

    dx, dy = end[0] - start[0], end[1] - start[1]
    run = math.hypot(dx, dy)
    if run < 2:
        return

    # Pale yellow dimension line
    line_color = (255, 240, 100, 220)
    draw.line([start, end], fill=line_color, width=3)

    # Perpendicular tick marks at each endpoint
    px_u, py_u = -dy / run, dx / run
    tick = 18
    for ex, ey in [start, end]:
        draw.line(
            [(ex + px_u * tick, ey + py_u * tick), (ex - px_u * tick, ey - py_u * tick)],
            fill=line_color, width=3,
        )

    # Length label at the midpoint of the line
    mx = (start[0] + end[0]) / 2
    my = (start[1] + end[1]) / 2
    text = _ft_to_feetinches(length_ft)

    img_w, img_h = img_size
    try:
        tb = font.getbbox(text)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
    except AttributeError:
        tw, th = len(text) * 7, 11

    pad = 5
    lx = max(tw // 2 + pad, min(int(mx), img_w - tw // 2 - pad))
    ly = max(th // 2 + pad, min(int(my), img_h - th // 2 - pad))

    bg = [lx - tw // 2 - pad, ly - th // 2 - pad, lx + tw // 2 + pad, ly + th // 2 + pad]
    draw.rectangle(bg, fill=(0, 0, 0, 200))
    draw.text((lx, ly), text, fill=(255, 240, 100, 255), font=font, anchor="mm")


def render_label(
    draw: ImageDraw.ImageDraw,
    measurement: dict,
    polygon: list[list[float]],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    img_size: tuple[int, int] = (10800, 7200),
) -> None:
    """
    Render a compact text label at the polygon centroid.
    Label format: '8"Ø  800CFM'. Length is shown separately by draw_length_annotation.
    Draws a dark background box for readability.
    """
    text = _format_label(measurement)
    if not text:
        return

    cx, cy = _polygon_centroid(polygon)
    img_w, img_h = img_size

    try:
        bbox = font.getbbox(text)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except AttributeError:
        text_w, text_h = len(text) * 8, 12  # fallback for default font

    pad = 8
    # Clamp label position so it stays within image
    lx = max(text_w // 2 + pad, min(int(cx), img_w - text_w // 2 - pad))
    ly = max(text_h // 2 + pad, min(int(cy), img_h - text_h // 2 - pad))

    bg = [lx - text_w // 2 - pad, ly - text_h // 2 - pad,
          lx + text_w // 2 + pad, ly + text_h // 2 + pad]
    draw.rectangle(bg, fill=(0, 0, 0, 180))
    draw.text((lx, ly), text, fill=(255, 255, 255, 255), font=font, anchor="mm")


def export_pdf(
    annotated_image_paths: list[str],
    output_path: str,
    dpi: int = 300,
) -> str:
    """
    Composite annotated page PNG images into a single PDF.
    Converts pixel dimensions to points (72 pt/in) at the given DPI.
    Returns output_path.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    c = rl_canvas.Canvas(output_path)

    for img_path in annotated_image_paths:
        img = Image.open(img_path)
        w_px, h_px = img.size
        # Convert pixels → points: 1 px = 72/dpi pt
        w_pt = w_px * 72 / dpi
        h_pt = h_px * 72 / dpi
        c.setPageSize((w_pt, h_pt))
        # reportlab y-origin is bottom-left; drawImage handles the flip
        c.drawImage(img_path, 0, 0, width=w_pt, height=h_pt)
        c.showPage()

    c.save()
    logger.info("export_pdf", pages=len(annotated_image_paths), output=output_path)
    return output_path
