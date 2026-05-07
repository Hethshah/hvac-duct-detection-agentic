"""
Phase 3 — Raster fallback duct detector.

Renders the PDF page to a binary black mask, finds pairs of parallel
horizontal/vertical line structures (duct walls), converts pixel coords
back to PDF-point media coordinates, and deduplicates against Phase 1
vector segments by IoU.

Coordinate system (input1.pdf, page rotation=270°):
  visual_col / scale  →  y_media_pt
  media_w_pt  –  visual_row / scale  →  x_media_pt

For input1.pdf (pure-vector PDF) this returns 0 new segments — all ducts
are already captured by Phase 1.
"""

import math

import numpy as np
import cv2
import fitz

from config.settings import (
    RASTER_SCALE,
    BLACK_PIXEL_MAX,
    MORPH_H_KERNEL_W,
    MORPH_V_KERNEL_H,
    PARALLEL_GAP_MIN_PX,
    PARALLEL_GAP_MAX_PX,
    PARALLEL_OVERLAP_MIN,
    DEDUP_IOU_THRESHOLD,
    RASTER_DEDUP_MARGIN_PT,
    TITLE_BLOCK_Y_MIN_PT,
    DUCT_MIN_ASPECT,
    DUCT_MIN_SHORT_PT,
    DUCT_MAX_SHORT_PT,
    DUCT_MIN_LONG_PT,
)
from models.duct_segment import DuctSegment

# Blobs longer than this are page-border lines, not duct walls.
_MAX_LINE_LEN_PX = 3000


# ── Rendering ─────────────────────────────────────────────────────────────────

def _render_black_mask(
    pdf_path: str, page_index: int, scale: int
) -> tuple[np.ndarray, dict]:
    """
    Render page and return binary mask (255 = black pixel, 0 = other).
    Also returns page_info dict with geometry metadata.
    """
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    media_w_pt = page.mediabox.width
    media_h_pt = page.mediabox.height
    rotation = page.rotation
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB)
    doc.close()

    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    mask = np.all(img <= BLACK_PIXEL_MAX, axis=2).astype(np.uint8) * 255

    return mask, {
        "scale": scale,
        "img_w": pix.width,
        "img_h": pix.height,
        "media_w_pt": media_w_pt,
        "media_h_pt": media_h_pt,
        "rotation": rotation,
    }


def _apply_title_block_mask(mask: np.ndarray, page_info: dict) -> np.ndarray:
    """Zero out the title block region in the mask (rotation=270 only)."""
    if page_info["rotation"] == 270:
        cutoff_col = int(TITLE_BLOCK_Y_MIN_PT * page_info["scale"])
        mask = mask.copy()
        mask[:, cutoff_col:] = 0
    return mask


# ── Blob detection ────────────────────────────────────────────────────────────

def _find_h_blobs(mask: np.ndarray) -> list[dict]:
    """
    Morphological open with horizontal kernel → find horizontal line blobs.
    Each blob represents one duct wall that runs horizontally in the image.
    """
    kernel = np.ones((1, MORPH_H_KERNEL_W), np.uint8)
    h_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    n, _, stats, centroids = cv2.connectedComponentsWithStats(h_mask, connectivity=8)
    blobs = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        if w < MORPH_H_KERNEL_W or w > _MAX_LINE_LEN_PX:
            continue
        blobs.append({
            "r0": y, "r1": y + h - 1, "c0": x, "c1": x + w - 1,
            "cy": float(centroids[i][1]),
            "cx": float(centroids[i][0]),
        })
    return blobs


def _find_v_blobs(mask: np.ndarray) -> list[dict]:
    """
    Morphological open with vertical kernel → find vertical line blobs.
    Each blob represents one duct wall that runs vertically in the image.
    """
    kernel = np.ones((MORPH_V_KERNEL_H, 1), np.uint8)
    v_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    n, _, stats, centroids = cv2.connectedComponentsWithStats(v_mask, connectivity=8)
    blobs = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        if h < MORPH_V_KERNEL_H or h > _MAX_LINE_LEN_PX:
            continue
        blobs.append({
            "r0": y, "r1": y + h - 1, "c0": x, "c1": x + w - 1,
            "cy": float(centroids[i][1]),
            "cx": float(centroids[i][0]),
        })
    return blobs


# ── Pairing ───────────────────────────────────────────────────────────────────

def _overlap_frac(a_lo: int, a_hi: int, b_lo: int, b_hi: int) -> float:
    """Fraction of the shorter span covered by the overlap with the other."""
    lo = max(a_lo, b_lo)
    hi = min(a_hi, b_hi)
    overlap = max(0, hi - lo + 1)
    shorter = min(a_hi - a_lo + 1, b_hi - b_lo + 1)
    return overlap / shorter if shorter > 0 else 0.0


def _pair_h_blobs(blobs: list[dict]) -> list[tuple[dict, dict]]:
    """
    Pair horizontal-image-line blobs that form the two parallel walls of a duct.
    Blobs are matched by row centroid gap and column overlap.
    Returns list of (top_blob, bottom_blob) pairs.
    """
    sorted_blobs = sorted(blobs, key=lambda b: b["cy"])
    used: set[int] = set()
    pairs: list[tuple[dict, dict]] = []

    for i, a in enumerate(sorted_blobs):
        if i in used:
            continue
        for j in range(i + 1, len(sorted_blobs)):
            if j in used:
                continue
            b = sorted_blobs[j]
            gap = b["cy"] - a["cy"]
            if gap < PARALLEL_GAP_MIN_PX:
                continue
            if gap > PARALLEL_GAP_MAX_PX:
                break
            if _overlap_frac(a["c0"], a["c1"], b["c0"], b["c1"]) < PARALLEL_OVERLAP_MIN:
                continue
            used.add(i)
            used.add(j)
            pairs.append((a, b))
            break

    return pairs


def _pair_v_blobs(blobs: list[dict]) -> list[tuple[dict, dict]]:
    """
    Pair vertical-image-line blobs that form the two parallel walls of a duct.
    Blobs are matched by column centroid gap and row overlap.
    Returns list of (left_blob, right_blob) pairs.
    """
    sorted_blobs = sorted(blobs, key=lambda b: b["cx"])
    used: set[int] = set()
    pairs: list[tuple[dict, dict]] = []

    for i, a in enumerate(sorted_blobs):
        if i in used:
            continue
        for j in range(i + 1, len(sorted_blobs)):
            if j in used:
                continue
            b = sorted_blobs[j]
            gap = b["cx"] - a["cx"]
            if gap < PARALLEL_GAP_MIN_PX:
                continue
            if gap > PARALLEL_GAP_MAX_PX:
                break
            if _overlap_frac(a["r0"], a["r1"], b["r0"], b["r1"]) < PARALLEL_OVERLAP_MIN:
                continue
            used.add(i)
            used.add(j)
            pairs.append((a, b))
            break

    return pairs


# ── Coordinate conversion ─────────────────────────────────────────────────────

def _h_pair_to_segment(
    a: dict, b: dict, scale: int, media_w_pt: float, seg_id: str
) -> "DuctSegment | None":
    """
    Convert a horizontal-image-line pair → V duct in media coords.

    H blobs (horizontal in image) correspond to V ducts in media space:
      image col range → y_media range (duct length)
      image row range → x_media range (duct width, inverted)
    """
    r_top = min(a["r0"], b["r0"])
    r_bot = max(a["r1"], b["r1"])
    c_start = max(a["c0"], b["c0"])  # overlap col range = duct length
    c_end = min(a["c1"], b["c1"])

    if c_start >= c_end:
        return None

    x0 = media_w_pt - r_bot / scale
    x1 = media_w_pt - r_top / scale
    y0 = c_start / scale
    y1 = c_end / scale

    if x0 >= x1 or y0 >= y1:
        return None

    long_pt = y1 - y0    # V duct: long axis is y
    short_pt = x1 - x0   # short axis is x

    if not (DUCT_MIN_SHORT_PT <= short_pt <= DUCT_MAX_SHORT_PT):
        return None
    if long_pt < DUCT_MIN_LONG_PT:
        return None
    if long_pt / short_pt < DUCT_MIN_ASPECT:
        return None

    cx = (x0 + x1) / 2
    return DuctSegment(
        id=seg_id,
        rect=[round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
        orientation="V",
        long_pt=round(long_pt, 2),
        short_pt=round(short_pt, 2),
        aspect=round(long_pt / short_pt, 2),
        centerline=[[round(cx, 2), round(y0, 2)], [round(cx, 2), round(y1, 2)]],
        source="raster",
        confidence=0.8,
    )


def _v_pair_to_segment(
    a: dict, b: dict, scale: int, media_w_pt: float, seg_id: str
) -> "DuctSegment | None":
    """
    Convert a vertical-image-line pair → H duct in media coords.

    V blobs (vertical in image) correspond to H ducts in media space:
      image row overlap range → x_media range (duct length)
      image col range → y_media range (duct width)
    """
    r_top = max(a["r0"], b["r0"])  # overlap row range (intersection)
    r_bot = min(a["r1"], b["r1"])
    c_left = min(a["c0"], b["c0"])  # union col range = duct width extent
    c_right = max(a["c1"], b["c1"])

    if r_top >= r_bot:
        return None

    x0 = media_w_pt - r_bot / scale
    x1 = media_w_pt - r_top / scale
    y0 = c_left / scale
    y1 = c_right / scale

    if x0 >= x1 or y0 >= y1:
        return None

    long_pt = x1 - x0    # H duct: long axis is x
    short_pt = y1 - y0   # short axis is y

    if not (DUCT_MIN_SHORT_PT <= short_pt <= DUCT_MAX_SHORT_PT):
        return None
    if long_pt < DUCT_MIN_LONG_PT:
        return None
    if long_pt / short_pt < DUCT_MIN_ASPECT:
        return None

    cy = (y0 + y1) / 2
    return DuctSegment(
        id=seg_id,
        rect=[round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
        orientation="H",
        long_pt=round(long_pt, 2),
        short_pt=round(short_pt, 2),
        aspect=round(long_pt / short_pt, 2),
        centerline=[[round(x0, 2), round(cy, 2)], [round(x1, 2), round(cy, 2)]],
        source="raster",
        confidence=0.8,
    )


# ── Deduplication ─────────────────────────────────────────────────────────────

def _iou(r1: list, r2: list) -> float:
    """Intersection-over-Union of two axis-aligned [x0,y0,x1,y1] rectangles."""
    ix0 = max(r1[0], r2[0])
    iy0 = max(r1[1], r2[1])
    ix1 = min(r1[2], r2[2])
    iy1 = min(r1[3], r2[3])
    if ix0 >= ix1 or iy0 >= iy1:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    a1 = (r1[2] - r1[0]) * (r1[3] - r1[1])
    a2 = (r2[2] - r2[0]) * (r2[3] - r2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _centroid_near_phase1(cand_rect: list, phase1_rects: list, margin_pt: float) -> bool:
    """
    Returns True if the raster candidate's centroid is within margin_pt of
    any Phase 1 segment's bounding box edge.

    Junction fittings and connectors that raster detects near Phase 1 ducts
    will have centroids close to an existing duct boundary; this check
    suppresses them without affecting genuinely new duct regions far from
    any Phase 1 detection.
    """
    cx = (cand_rect[0] + cand_rect[2]) / 2
    cy = (cand_rect[1] + cand_rect[3]) / 2
    for r in phase1_rects:
        cx_close = max(r[0], min(r[2], cx))
        cy_close = max(r[1], min(r[3], cy))
        if math.hypot(cx - cx_close, cy - cy_close) <= margin_pt:
            return True
    return False


# ── Public API ────────────────────────────────────────────────────────────────

def extract_raster_ducts(
    pdf_path: str,
    phase1_segments: list[DuctSegment],
    page_index: int = 0,
    scale: int = RASTER_SCALE,
) -> list[DuctSegment]:
    """
    Raster fallback: detect duct segments not already captured by Phase 1.

    Returns only NEW DuctSegment objects (source="raster") whose IoU with
    every Phase 1 segment is below DEDUP_IOU_THRESHOLD.

    Only implemented for rotation=270 pages (input1.pdf). Returns [] for
    other rotations until generalisation is needed.
    """
    mask, page_info = _render_black_mask(pdf_path, page_index, scale)

    if page_info["rotation"] != 270:
        return []

    mask = _apply_title_block_mask(mask, page_info)
    media_w_pt = page_info["media_w_pt"]

    h_blobs = _find_h_blobs(mask)
    v_blobs = _find_v_blobs(mask)
    h_pairs = _pair_h_blobs(h_blobs)
    v_pairs = _pair_v_blobs(v_blobs)

    candidates: list[DuctSegment] = []
    counter = 1

    for top, bot in h_pairs:
        seg = _h_pair_to_segment(top, bot, scale, media_w_pt, f"raster_{counter:03d}")
        if seg is not None:
            candidates.append(seg)
            counter += 1

    for left, right in v_pairs:
        seg = _v_pair_to_segment(left, right, scale, media_w_pt, f"raster_{counter:03d}")
        if seg is not None:
            candidates.append(seg)
            counter += 1

    phase1_rects = [s.rect for s in phase1_segments]
    new_segs = [
        c for c in candidates
        if not _centroid_near_phase1(c.rect, phase1_rects, RASTER_DEDUP_MARGIN_PT)
        and all(_iou(c.rect, r) < DEDUP_IOU_THRESHOLD for r in phase1_rects)
    ]

    return new_segs
