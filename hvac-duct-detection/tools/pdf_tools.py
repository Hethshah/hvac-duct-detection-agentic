import json
import re
from pathlib import Path

import fitz
import structlog
from strands import tool

logger = structlog.get_logger()

_LOW_DPI_THRESHOLD = 150  # warn when effective render DPI is below this

# Common scale notations: 1/4"=1'-0", 1/8"=1'-0", 3/16"=1'-0" etc.
# Handles ASCII and Unicode typographic quotes. Using \uXXXX escapes avoids
# SyntaxError caused by curly-quote literals on Python <=3.11.
_INCH_CHARS = "\"" + "“”″′"  # " plus curly/prime variants
_FOOT_CHARS = "'" + "‘’ʼ`"         # ' plus curly/modifier variants
_SCALE_PATTERN = re.compile(
    r"(\d+)\s*/\s*(\d+)\s*[" + _INCH_CHARS + r"]\s*=\s*1\s*[" + _FOOT_CHARS + r"]",
    re.IGNORECASE,
)


def pdf_coords_to_pixel(
    x: float, y: float, page_height_pts: float, dpi: int = 300
) -> tuple[float, float]:
    """Convert PDF bottom-left origin to pixel top-left origin."""
    scale = dpi / 72.0
    return x * scale, (page_height_pts - y) * scale


def _parse_page_range(page_range: str, total_pages: int) -> list[int]:
    """
    Parse a page range string into a sorted list of 0-based page indices.
    Accepts: "1-5"  -> [0,1,2,3,4]
             "2,4,6" -> [1,3,5]
             "3"    -> [2]
    Page numbers in the string are 1-based; out-of-range values are ignored.
    """
    indices: set[int] = set()
    for part in page_range.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            for n in range(int(lo), int(hi) + 1):
                if 1 <= n <= total_pages:
                    indices.add(n - 1)
        else:
            n = int(part)
            if 1 <= n <= total_pages:
                indices.add(n - 1)
    return sorted(indices)


@tool
def pdf_to_images(
    pdf_path: str,
    dpi: int = 300,
    output_dir: str = "outputs/pages",
    page_range: str = "",
) -> list[str]:
    """
    Convert each page of a PDF to a PNG image at the specified DPI.
    Always renders in RGB (handles CMYK PDFs transparently).
    page_range: optional filter, e.g. "1-3" or "1,3,5" (1-based). Empty = all pages.
    Saves images to output_dir and returns a list of file paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    total = len(doc)
    indices = _parse_page_range(page_range, total) if page_range else list(range(total))

    paths: list[str] = []
    for i in indices:
        page = doc[i]
        # Warn when source page is very small (likely low-res scan)
        w_pts = page.rect.width
        effective_dpi = (w_pts / 8.5) if w_pts > 0 else dpi
        if effective_dpi < _LOW_DPI_THRESHOLD:
            logger.warning(
                "low_resolution_page",
                page=i,
                page_width_pts=round(w_pts, 1),
                effective_dpi=round(effective_dpi, 1),
                tip="Consider using a higher-DPI source PDF for better detection accuracy.",
            )

        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        # Force RGB colorspace -- handles CMYK, grayscale, and spot-color PDFs
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        dest = str(out / f"page_{i:03d}.png")
        pix.save(dest)
        paths.append(dest)

    doc.close()
    logger.info("pdf_to_images", pages=len(paths), dpi=dpi, output_dir=output_dir)
    return paths


@tool
def extract_text_blocks(pdf_path: str, dpi: int = 300) -> list[dict]:
    """
    Extract text blocks from a PDF with pixel coordinates (top-left origin).
    Returns a list of {text, x, y, w, h, page} dicts.
    """
    doc = fitz.open(pdf_path)
    blocks: list[dict] = []
    scale = dpi / 72.0

    for page_num, page in enumerate(doc):
        for b in page.get_text("blocks"):
            x0, y0, x1, y1, text, *_ = b
            text = text.strip()
            if not text:
                continue
            # PyMuPDF already uses top-left origin (y increases downward)
            px = x0 * scale
            py = y0 * scale
            blocks.append(
                {
                    "text": text,
                    "x": round(px, 1),
                    "y": round(py, 1),
                    "w": round((x1 - x0) * scale, 1),
                    "h": round((y1 - y0) * scale, 1),
                    "page": page_num,
                }
            )
    doc.close()

    logger.info("extract_text_blocks", count=len(blocks))
    return blocks


@tool
def detect_scale_bar(text_blocks_json: str, dpi: int = 300) -> float:
    """
    Parse scale notation from a JSON-encoded list of text blocks and compute
    pixels-per-foot ratio. Handles notations like 1/4"=1'-0".
    Returns 0.0 if no scale notation is found (e.g. 'DO NOT SCALE DRAWINGS').
    """
    text_blocks: list[dict] = json.loads(text_blocks_json)

    for block in text_blocks:
        m = _SCALE_PATTERN.search(block["text"])
        if m:
            numerator = int(m.group(1))
            denominator = int(m.group(2))
            ratio = round(dpi * (numerator / denominator), 4)
            logger.info("detect_scale_bar", matched=block["text"][:60], ratio=ratio)
            return ratio

    logger.warning("detect_scale_bar", status="not_found")
    return 0.0
