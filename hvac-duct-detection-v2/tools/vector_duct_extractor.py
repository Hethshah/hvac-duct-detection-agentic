"""
Phase 1 — Vector Duct Extractor

Extracts duct segments directly from the PDF's native vector quad paths.
Black-stroke 'qu' items only; no rasterization, no LLM.

Key facts (from PDF analysis of input1.pdf):
- Duct walls are stored as 'qu' (quad) path items, NOT 're' (rect)
- Black = RGB all < 0.15; Grey (~0.4) = walls/structure → ignored
- Page may be rotated (input1.pdf = 270°); we work in un-rotated PDF coords
"""

import json
import math
from pathlib import Path

import fitz

from config.settings import (
    BLACK_MAX_CHANNEL,
    CLUSTER_LONG_GAP_MAX_PT,
    CLUSTER_SHORT_OVERLAP_MIN,
    CLUSTER_SHORT_RATIO_MAX,
    DUCT_MAX_SHORT_PT,
    DUCT_MIN_ASPECT,
    DUCT_MIN_LONG_PT,
    DUCT_MIN_SHORT_PT,
    DUCT_MIN_STROKE_PT,
    OUTPUTS_DIR,
    TITLE_BLOCK_Y_MIN_PT,
)
from models.duct_segment import DuctSegment


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _is_black(color: tuple | None) -> bool:
    if color is None:
        return False
    return all(c < BLACK_MAX_CHANNEL for c in color[:3])


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _rect_dims(rect: fitz.Rect | list) -> tuple[float, float, float, float]:
    """Return (x0, y0, x1, y1) normalised so x0<x1, y0<y1."""
    x0, y0, x1, y1 = rect[0], rect[1], rect[2], rect[3]
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def _segment_from_rect(rect: fitz.Rect | list, seg_id: str, page: int) -> DuctSegment | None:
    x0, y0, x1, y1 = _rect_dims(rect)
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return None

    long_pt  = max(w, h)
    short_pt = min(w, h)
    if long_pt < DUCT_MIN_LONG_PT:
        return None
    if short_pt < DUCT_MIN_SHORT_PT or short_pt > DUCT_MAX_SHORT_PT:
        return None

    aspect = long_pt / short_pt
    if aspect < DUCT_MIN_ASPECT:
        return None

    orientation = "H" if w >= h else "V"
    cy = (y0 + y1) / 2
    cx = (x0 + x1) / 2
    if orientation == "H":
        centerline = [[x0, cy], [x1, cy]]
    else:
        centerline = [[cx, y0], [cx, y1]]

    return DuctSegment(
        id=seg_id,
        rect=[x0, y0, x1, y1],
        orientation=orientation,
        long_pt=long_pt,
        short_pt=short_pt,
        aspect=aspect,
        centerline=centerline,
        page=page,
    )


# ---------------------------------------------------------------------------
# Path filtering
# ---------------------------------------------------------------------------

def _path_is_quad_only(path: dict) -> bool:
    """True if ALL items in the path are 'qu' (quad/rectangle) type."""
    items = path.get("items", [])
    if not items:
        return False
    return all(item[0] == "qu" for item in items)


_AXIS_EPS = 1.0  # pt — tolerance for "axis-aligned" segment classification


def _path_is_axis_aligned_rect_via_lines(path: dict) -> bool:
    """
    True if all 'l' items are axis-aligned and collectively form at least 2 sides
    of the path's bounding rectangle.

    Rejects:
    - Any path with a diagonal segment (|dx|>eps AND |dy|>eps)
    - Any path with an interior stroke (not on a bbox edge)
    These two rules reject supply-diffuser grille symbols, equipment outlines,
    and structural hatching boundaries.
    """
    items = path.get("items", [])
    r = path.get("rect")
    if r is None or not items:
        return False
    x0, y0, x1, y1 = min(r[0], r[2]), min(r[1], r[3]), max(r[0], r[2]), max(r[1], r[3])

    sides_hit = {"top": False, "bottom": False, "left": False, "right": False}
    for it in items:
        if it[0] != "l":
            continue
        try:
            p1, p2 = it[1], it[2]
        except (IndexError, AttributeError):
            return False
        dx = abs(p2.x - p1.x)
        dy = abs(p2.y - p1.y)
        # Any diagonal segment → not a clean rectangular outline
        if dx > _AXIS_EPS and dy > _AXIS_EPS:
            return False
        if dy <= _AXIS_EPS:  # horizontal
            y_mid = (p1.y + p2.y) / 2
            if abs(y_mid - y0) <= _AXIS_EPS:
                sides_hit["top"] = True
            elif abs(y_mid - y1) <= _AXIS_EPS:
                sides_hit["bottom"] = True
            else:
                return False  # interior horizontal stroke → not a duct outline
        else:  # vertical
            x_mid = (p1.x + p2.x) / 2
            if abs(x_mid - x0) <= _AXIS_EPS:
                sides_hit["left"] = True
            elif abs(x_mid - x1) <= _AXIS_EPS:
                sides_hit["right"] = True
            else:
                return False  # interior vertical stroke → not a duct outline

    return sum(sides_hit.values()) >= 2


def _path_is_simple_rect_shape(path: dict) -> bool:
    """True if path is a simple rectangular shape suitable for H/V duct detection.

    - Pure 'qu': always accepted (PDF-native rectangle, original behaviour).
    - Pure 'l' (≤6 items): accepted only if axis-aligned rectangular outline
      (rejects diffuser symbols, hatching boundaries, equipment outlines).
    - Mixed 'qu'+'l' (≤3 items): accepted cautiously (the 'qu' guarantees
      rectangularity; the 'l' is typically an additional detail line).
    """
    items = path.get("items", [])
    if not items or len(items) > 6:
        return False
    if all(it[0] == "qu" for it in items):
        return True
    if all(it[0] == "l" for it in items):
        return _path_is_axis_aligned_rect_via_lines(path)
    # Mixed 'qu'+'l' — accept cautiously (≤3 items only)
    if len(items) <= 3 and all(it[0] in ("qu", "l") for it in items):
        return True
    return False


def _extract_candidate_rects(paths: list[dict]) -> list[fitz.Rect]:
    """
    Return one bounding rect per black path with a simple rectangular shape.

    Handles both 'qu'-only paths and paths drawn with 'l' line segments (≤ 6 items).
    Deduplicates by rounded bbox so identical shapes from duplicate path objects
    are only processed once.
    """
    seen: set[tuple] = set()
    rects: list[fitz.Rect] = []
    for p in paths:
        if not _is_black(p.get("color")):
            continue
        if (p.get("width") or 0) < DUCT_MIN_STROKE_PT:
            continue
        if not _path_is_simple_rect_shape(p):
            continue
        r = p.get("rect")
        if r is None:
            continue
        x0, y0, x1, y1 = min(r[0], r[2]), min(r[1], r[3]), max(r[0], r[2]), max(r[1], r[3])
        key = (round(x0), round(y0), round(x1), round(y1))
        if key in seen:
            continue
        seen.add(key)
        rects.append(r)
    return rects


# ---------------------------------------------------------------------------
# Clustering: merge collinear segments split by fittings / labels
# ---------------------------------------------------------------------------

def _short_axis_overlap(a: DuctSegment, b: DuctSegment) -> float:
    """
    Fraction of the shorter segment's short-axis extent that overlaps with the other.
    Used to decide if two segments are on the same duct run.
    """
    if a.orientation != b.orientation:
        return 0.0

    if a.orientation == "H":
        # Compare y extents (short axis)
        a0, a1 = a.rect[1], a.rect[3]
        b0, b1 = b.rect[1], b.rect[3]
    else:
        # Compare x extents (short axis)
        a0, a1 = a.rect[0], a.rect[2]
        b0, b1 = b.rect[0], b.rect[2]

    overlap = max(0.0, min(a1, b1) - max(a0, b0))
    shorter_span = min(a1 - a0, b1 - b0)
    return overlap / shorter_span if shorter_span > 0 else 0.0


def _long_axis_gap(a: DuctSegment, b: DuctSegment) -> float:
    """
    Gap between two segments along their shared long axis.
    Negative means they overlap.
    """
    if a.orientation == "H":
        a_start, a_end = a.rect[0], a.rect[2]
        b_start, b_end = b.rect[0], b.rect[2]
    else:
        a_start, a_end = a.rect[1], a.rect[3]
        b_start, b_end = b.rect[1], b.rect[3]

    # Gap = distance between closest endpoints
    return max(min(b_start, b_end), min(a_start, a_end)) - min(max(b_start, b_end), max(a_start, a_end))


def _merge_two(a: DuctSegment, b: DuctSegment) -> DuctSegment:
    """Merge two collinear segments into one spanning the union bbox."""
    x0 = min(a.rect[0], b.rect[0])
    y0 = min(a.rect[1], b.rect[1])
    x1 = max(a.rect[2], b.rect[2])
    y1 = max(a.rect[3], b.rect[3])
    merged_rect = [x0, y0, x1, y1]

    w, h = x1 - x0, y1 - y0
    long_pt  = max(w, h)
    short_pt = min(w, h)
    aspect   = long_pt / short_pt if short_pt > 0 else 0.0
    orientation = "H" if w >= h else "V"

    cy = (y0 + y1) / 2
    cx = (x0 + x1) / 2
    centerline = [[x0, cy], [x1, cy]] if orientation == "H" else [[cx, y0], [cx, y1]]

    return DuctSegment(
        id=a.id,
        rect=merged_rect,
        orientation=orientation,
        long_pt=long_pt,
        short_pt=short_pt,
        aspect=aspect,
        centerline=centerline,
        page=a.page,
        confidence=min(a.confidence, b.confidence),
    )


def _cluster_segments(segments: list[DuctSegment]) -> list[DuctSegment]:
    """
    Iteratively merge collinear segments that:
    - share orientation
    - short-axis overlap >= CLUSTER_SHORT_OVERLAP_MIN
    - long-axis gap <= CLUSTER_LONG_GAP_MAX_PT
    """
    changed = True
    while changed:
        changed = False
        merged: list[DuctSegment] = []
        used = [False] * len(segments)
        for i, seg_i in enumerate(segments):
            if used[i]:
                continue
            current = seg_i
            for j, seg_j in enumerate(segments):
                if i == j or used[j]:
                    continue
                if current.orientation != seg_j.orientation:
                    continue
                if _short_axis_overlap(current, seg_j) < CLUSTER_SHORT_OVERLAP_MIN:
                    continue
                if _long_axis_gap(current, seg_j) > CLUSTER_LONG_GAP_MAX_PT:
                    continue
                short_ratio = max(current.short_pt, seg_j.short_pt) / max(min(current.short_pt, seg_j.short_pt), 0.1)
                if short_ratio > CLUSTER_SHORT_RATIO_MAX:
                    continue
                current = _merge_two(current, seg_j)
                used[j] = True
                changed = True
            merged.append(current)
            used[i] = True
        segments = merged
    return segments


# ---------------------------------------------------------------------------
# Diagonal duct detection from non-axis-aligned black paths
# ---------------------------------------------------------------------------

def _all_l_items(items: list) -> bool:
    return bool(items) and all(item[0] == "l" for item in items)


def _is_parallelogram_l_path(path: dict) -> bool:
    """
    True if path forms a parallelogram with:
    - Two pairs of parallel, equal-length edges
    - At least one diagonal segment (not axis-aligned)
    - Inner aspect (long edge / short edge) >= 2.0 — ensures it's a duct, not a square

    Deduplicates edges by canonical endpoint pair first to handle PDF paths where
    the same segment appears twice (drawn in both directions as an artifact).
    Rejects X-marks, right-angle triangles, and near-square diagonal hatching.
    """
    items = path.get("items", [])
    if not items or not all(it[0] == "l" for it in items):
        return False

    # Deduplicate by canonical (sorted) endpoint pair
    unique: dict[tuple, tuple] = {}
    for it in items:
        try:
            p1, p2 = it[1], it[2]
        except (IndexError, AttributeError):
            return False
        a = (round(p1.x, 1), round(p1.y, 1))
        b = (round(p2.x, 1), round(p2.y, 1))
        if a == b:
            return False
        key = tuple(sorted((a, b)))
        if key not in unique:
            unique[key] = (p1, p2)

    if len(unique) != 4:
        return False

    edges = []
    has_diagonal = False
    for p1, p2 in unique.values():
        dx, dy = p2.x - p1.x, p2.y - p1.y
        length = math.hypot(dx, dy)
        if length < 1.0:
            return False
        angle = math.atan2(dy, dx) % math.pi  # direction-agnostic (0..π)
        if abs(dx) > _AXIS_EPS and abs(dy) > _AXIS_EPS:
            has_diagonal = True
        edges.append((length, angle))

    if not has_diagonal:
        return False  # axis-aligned rectangle — H/V path, not diagonal duct

    _ANGLE_TOL = 0.15   # rad ≈ 8.6°
    _EDGE_TOL  = 6.0    # pt

    edges_sorted = sorted(edges, key=lambda e: e[1])
    e0, e1, e2, e3 = edges_sorted

    # Pair edges by similar angle
    if abs(e0[1] - e1[1]) > _ANGLE_TOL or abs(e2[1] - e3[1]) > _ANGLE_TOL:
        return False
    # Paired edges must have equal length
    if abs(e0[0] - e1[0]) > _EDGE_TOL or abs(e2[0] - e3[0]) > _EDGE_TOL:
        return False

    long_edge  = max(e0[0], e2[0])
    short_edge = min(e0[0], e2[0])
    if short_edge < 1.0 or long_edge / short_edge < 2.0:
        return False  # square-ish → bracing or hatching, not a duct

    return True


def _is_rotated_qu_path(path: dict) -> bool:
    """True if path has a single 'qu' item whose quad is rotated (non-axis-aligned)."""
    items = path.get("items", [])
    if len(items) != 1 or items[0][0] != "qu":
        return False
    try:
        quad = items[0][1]
        pts = [quad.ul, quad.ur, quad.ll, quad.lr]
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                if abs(pts[i].x - pts[j].x) > 10 and abs(pts[i].y - pts[j].y) > 10:
                    return True
    except Exception:
        pass
    return False


def _extract_diagonal_duct_paths(paths: list[dict]) -> list[tuple[dict, fitz.Rect]]:
    """
    Return (path, rect) pairs for diagonal duct outlines — non-axis-aligned black paths.

    Two shapes indicate a diagonal duct in HVAC plans:
    1. A polygon drawn with 'l' line items that includes at least one diagonal segment.
    2. A rotated 'qu' quad (corners not on bbox edges).

    Noise filters (calibrated on input1.pdf):
    - Bounding-box minimum dimension >= 60 pt — excludes VAV-box X-marks (≤ 36 pt)
      and thin connectors (≤ 50 pt).
    - Bounding-box maximum dimension <= 250 pt — excludes page border paths.
    - Bounding-box aspect < 1.8 — a diagonal duct's bbox is always near-square.
    """
    DIAG_MIN_BBOX = 60.0
    DIAG_MAX_BBOX = 250.0
    DIAG_MAX_ASPECT = 1.8

    seen: set[tuple] = set()
    result: list[tuple[dict, fitz.Rect]] = []

    for p in paths:
        if not _is_black(p.get("color")):
            continue
        r = p.get("rect")
        if r is None:
            continue
        x0, y0, x1, y1 = min(r[0], r[2]), min(r[1], r[3]), max(r[0], r[2]), max(r[1], r[3])
        w, h = x1 - x0, y1 - y0
        if min(w, h) < DIAG_MIN_BBOX or max(w, h) > DIAG_MAX_BBOX:
            continue
        if max(w, h) / max(min(w, h), 0.1) > DIAG_MAX_ASPECT:
            continue

        is_diag = _is_parallelogram_l_path(p) or _is_rotated_qu_path(p)
        if not is_diag:
            continue

        key = (round(x0), round(y0), round(x1), round(y1))
        if key in seen:
            continue
        seen.add(key)
        result.append((p, r))

    return result


def _polygon_from_path(path: dict) -> list[list[float]]:
    """Extract ordered polygon vertices from a diagonal duct path."""
    items = path.get("items", [])

    # Rotated 'qu' path — use quad corners directly (already ordered CW/CCW)
    if len(items) == 1 and items[0][0] == "qu":
        try:
            quad = items[0][1]
            return [
                [quad.ul.x, quad.ul.y],
                [quad.ur.x, quad.ur.y],
                [quad.lr.x, quad.lr.y],
                [quad.ll.x, quad.ll.y],
            ]
        except Exception:
            pass

    # 'l' line items — collect unique endpoints then sort by angle from centroid
    pts: list[list[float]] = []
    seen_keys: set[tuple] = set()
    for item in items:
        if item[0] == "l":
            try:
                for pt in (item[1], item[2]):
                    key = (round(pt.x, 1), round(pt.y, 1))
                    if key not in seen_keys:
                        seen_keys.add(key)
                        pts.append([pt.x, pt.y])
            except Exception:
                pass

    if len(pts) >= 3:
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        pts.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

    return pts


def _centerline_from_polygon(polygon: list[list[float]]) -> list[list[float]] | None:
    """
    Compute centerline by connecting midpoints of the two short sides of the duct polygon.

    For a 4-vertex parallelogram, the two shortest edges are the duct end-caps; their
    midpoints define the true long axis regardless of which diagonal the duct runs along.
    """
    if len(polygon) != 4:
        return None
    edges = []
    for i in range(4):
        p1, p2 = polygon[i], polygon[(i + 1) % 4]
        edges.append((math.hypot(p2[0] - p1[0], p2[1] - p1[1]), i))
    edges.sort()
    midpoints = []
    for _, i in edges[:2]:
        p1, p2 = polygon[i], polygon[(i + 1) % 4]
        midpoints.append([(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2])
    if midpoints[0][0] > midpoints[1][0]:
        midpoints = [midpoints[1], midpoints[0]]
    return midpoints


def _segment_from_diagonal_path(path: dict, rect, seg_id: str, page: int) -> DuctSegment | None:
    """Create a DuctSegment for a diagonal duct with actual polygon vertices."""
    x0, y0, x1, y1 = _rect_dims(rect)
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return None
    long_pt  = max(w, h)
    short_pt = min(w, h)
    aspect   = long_pt / short_pt if short_pt > 0 else 0.0

    polygon  = _polygon_from_path(path)
    centerline = _centerline_from_polygon(polygon) if polygon else [[x0, y0], [x1, y1]]

    return DuctSegment(
        id=seg_id,
        rect=[x0, y0, x1, y1],
        orientation="D",
        long_pt=round(long_pt, 2),
        short_pt=round(short_pt, 2),
        aspect=round(aspect, 2),
        centerline=centerline,
        page=page,
        source="vector",
        confidence=0.95,
        polygon=polygon if polygon else None,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_ducts(pdf_path: str, page_index: int = 0) -> list[DuctSegment]:
    """
    Extract duct segments from a PDF page using native vector quad paths.

    Steps:
    1. Open PDF with PyMuPDF; read page rotation as metadata
    2. Get all drawing paths via page.get_drawings()
    3. Keep only black-stroke ('qu'-only) paths
    4. Apply duct heuristics (aspect, short-side bounds, long-side minimum)
    5. Cluster collinear fragments into single duct runs
    6. Re-assign sequential IDs

    Returns a list of DuctSegment objects in un-rotated PDF point coordinates.
    """
    doc  = fitz.open(pdf_path)
    page = doc[page_index]
    rotation = page.rotation  # store for downstream use; we work in un-rotated coords

    paths = page.get_drawings()

    # Step 1: collect candidate rects from black axis-aligned quad paths
    candidate_rects = _extract_candidate_rects(paths)

    # Step 2: convert each rect to a DuctSegment, applying duct heuristics
    raw_segments: list[DuctSegment] = []
    for i, rect in enumerate(candidate_rects):
        seg = _segment_from_rect(rect, f"raw_{i:04d}", page_index)
        if seg is None:
            continue
        if seg.rect[1] > TITLE_BLOCK_Y_MIN_PT and seg.rect[3] > TITLE_BLOCK_Y_MIN_PT:
            continue
        raw_segments.append(seg)

    # Step 3: cluster collinear fragments into single duct runs
    clustered = _cluster_segments(raw_segments)

    # Step 4: detect diagonal duct outlines from non-axis-aligned black paths
    diag_path_pairs = _extract_diagonal_duct_paths(paths)
    diag_segs: list[DuctSegment] = []
    for i, (path, rect) in enumerate(diag_path_pairs):
        seg = _segment_from_diagonal_path(path, rect, f"diag_raw_{i:03d}", page_index)
        if seg is None:
            continue
        # Exclude title-block region
        if seg.rect[1] > TITLE_BLOCK_Y_MIN_PT and seg.rect[3] > TITLE_BLOCK_Y_MIN_PT:
            continue
        diag_segs.append(seg)

    all_segments = clustered + diag_segs

    # Step 5: re-assign clean sequential IDs
    for idx, seg in enumerate(all_segments):
        seg.id = f"duct_{idx + 1:03d}"

    doc.close()
    return all_segments


def extract_ducts_to_json(pdf_path: str, output_path: str | None = None, page_index: int = 0) -> str:
    """
    Run extract_ducts and write results to JSON.
    Returns the output path.
    """
    segments = extract_ducts(pdf_path, page_index)

    stem = Path(pdf_path).stem
    out_dir = Path(output_path) if output_path else OUTPUTS_DIR / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "phase1_vector_ducts.json"

    data = {
        "pdf": str(pdf_path),
        "page": page_index,
        "segment_count": len(segments),
        "segments": [s.to_dict() for s in segments],
    }
    out_file.write_text(json.dumps(data, indent=2))
    return str(out_file)
