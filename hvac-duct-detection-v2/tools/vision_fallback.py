"""
Vision fallback for missed / diagonal / polygon ducts.

Strategy (user-defined):
  1. Identify candidate regions not covered by vector extraction.
  2. Render a high-res crop of each candidate region.
  3. Ask Claude vision for the duct's pixel-level bounding box inside that crop.
  4. Convert crop pixel coordinates → media (PDF-point) coordinates.
  5. Return DuctSegments that can be merged with the vector-extracted list.

Coordinate pipeline for 270°-rotated pages
───────────────────────────────────────────
  media (x_m, y_m)  ──[rotate 270°]──►  visual pixel (x_v, y_v)
  x_v = y_m  * render_scale
  y_v = (media_w - x_m) * render_scale

  Inverse (visual → media):
  y_m = x_v / render_scale
  x_m = media_w - y_v / render_scale
"""

import base64
import json
import os
import re
import tempfile
from pathlib import Path

import anthropic
import fitz
from PIL import Image

# Load .env from project root (parent of this file's parent)
_env_file = Path(__file__).parent.parent.parent / ".env"
if _env_file.exists() and not os.environ.get("ANTHROPIC_API_KEY"):
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from config.settings import RENDER_DPI, VISION_MODEL
from models.duct_segment import DuctSegment


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _media_to_visual(x_m, y_m, rotation, media_w, media_h, scale):
    if rotation == 270:
        return y_m * scale, (media_w - x_m) * scale
    if rotation == 90:
        return (media_h - y_m) * scale, x_m * scale
    if rotation == 180:
        return (media_w - x_m) * scale, (media_h - y_m) * scale
    return x_m * scale, y_m * scale


def _visual_to_media(x_v, y_v, rotation, media_w, media_h, scale):
    if rotation == 270:
        return media_w - y_v / scale, x_v / scale
    if rotation == 90:
        return y_v / scale, media_h - x_v / scale
    if rotation == 180:
        return media_w - x_v / scale, media_h - y_v / scale
    return x_v / scale, y_v / scale


# ---------------------------------------------------------------------------
# Candidate region discovery
# ---------------------------------------------------------------------------

def _find_diagonal_candidate_regions(pdf_path: str, page_index: int = 0) -> list[dict]:
    """
    Find bounding boxes (in media coords) that contain truly diagonal (non-axis-aligned)
    black line segments.  These mark areas where duct transitions / elbows live that
    the vector-quad extractor cannot see.

    Returns a list of {'x0','y0','x1','y1'} dicts in media coords, each padded 40pt.
    """
    import math

    doc = fitz.open(pdf_path)
    page = doc[page_index]
    paths = page.get_drawings()
    doc.close()

    PAD = 40  # extra margin (media pts) around each cluster

    # Collect all truly diagonal black line endpoints
    diag_points: list[tuple[float, float]] = []
    for p in paths:
        color = p.get("color")
        if not (color and all(c < 0.15 for c in color[:3])):
            continue
        for item in p.get("items", []):
            if item[0] != "l":
                continue
            try:
                p1, p2 = item[1], item[2]
                dx, dy = abs(p2.x - p1.x), abs(p2.y - p1.y)
                if dx > 10 and dy > 10 and math.hypot(dx, dy) >= 40:
                    diag_points.extend([(p1.x, p1.y), (p2.x, p2.y)])
            except Exception:
                pass

    if not diag_points:
        return []

    # Cluster nearby diagonal points (simple radius merge)
    CLUSTER_RADIUS = 80
    clusters: list[list[tuple[float, float]]] = []
    used = [False] * len(diag_points)
    for i, pt in enumerate(diag_points):
        if used[i]:
            continue
        cluster = [pt]
        used[i] = True
        for j, other in enumerate(diag_points):
            if used[j]:
                continue
            if math.hypot(pt[0] - other[0], pt[1] - other[1]) < CLUSTER_RADIUS:
                cluster.append(other)
                used[j] = True
        clusters.append(cluster)

    regions = []
    for cluster in clusters:
        xs = [p[0] for p in cluster]
        ys = [p[1] for p in cluster]
        regions.append({
            "x0": min(xs) - PAD,
            "y0": min(ys) - PAD,
            "x1": max(xs) + PAD,
            "y1": max(ys) + PAD,
        })

    return regions


def _regions_covered_by_vector(vector_ducts: list[DuctSegment]) -> list[dict]:
    """Convert vector duct rects to coverage dicts."""
    return [{"x0": s.rect[0], "y0": s.rect[1], "x1": s.rect[2], "y1": s.rect[3]}
            for s in vector_ducts]


def _is_covered(region: dict, covered: list[dict], threshold: float = 0.6) -> bool:
    """True if region overlaps heavily with any already-detected vector duct."""
    rx0, ry0, rx1, ry1 = region["x0"], region["y0"], region["x1"], region["y1"]
    r_area = max(0, rx1 - rx0) * max(0, ry1 - ry0)
    if r_area == 0:
        return True
    for c in covered:
        ix0, iy0 = max(rx0, c["x0"]), max(ry0, c["y0"])
        ix1, iy1 = min(rx1, c["x1"]), min(ry1, c["y1"])
        inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        if inter / r_area >= threshold:
            return True
    return False


# ---------------------------------------------------------------------------
# Crop + vision
# ---------------------------------------------------------------------------

VISION_PROMPT = """You are analysing a zoomed crop of an HVAC mechanical floor plan.

TASK: Find the SINGLE most prominent duct segment and return its bounding box.

DUCT TYPES to detect:
1. Rectangular axis-aligned duct — thick black border rectangle, wider than tall (horizontal) or taller than wide (vertical).
2. DIAGONAL TRANSITION DUCT — a GREY-FILLED or GREY-HATCHED PARALLELOGRAM / TRAPEZOID running at an angle.
   These appear as a grey shaded area (sometimes with diagonal hatch lines inside) between two duct sections.
   THIS IS THE MOST IMPORTANT type to detect. Return orientation = "diagonal" for these.
3. Diagonal channel — two thick black lines running at an angle forming a channel.

IGNORE completely:
- Thin grey room-outline / structural walls
- Square equipment boxes (VAV, AHU, FCU) — especially any box with an X drawn across it
- Flex duct connectors (corrugated spiral lines)
- Round diffuser circles (circle with X or radial lines)
- Text labels, dimension arrows, leader lines
- Round ducts (circles labeled 10"ø, 12"ø, etc.)

PRIORITY: If you see a GREY FILLED PARALLELOGRAM or GREY SHADED AREA running diagonally — that is a diagonal duct transition. Return its full bounding box and set orientation to "diagonal".

Return ONE JSON object — no markdown, no explanation:
{"found": true, "x0": <left px>, "y0": <top px>, "x1": <right px>, "y1": <bottom px>, "orientation": "horizontal|vertical|diagonal", "confidence": 0.0-1.0}

If no duct is clearly visible:
{"found": false, "confidence": 0.0}

Pixel coordinates must be within this image's bounds.
"""


def _call_vision(image_path: str) -> dict:
    client = anthropic.Anthropic()
    with open(image_path, "rb") as f:
        img_b64 = base64.standard_b64encode(f.read()).decode()

    suffix = Path(image_path).suffix.lower()
    media_type = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"

    response = client.messages.create(
        model=VISION_MODEL,
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": img_b64}},
                {"type": "text", "text": VISION_PROMPT},
            ],
        }],
    )
    raw = response.content[0].text
    # Extract JSON
    obj_match = re.search(r"\{[\s\S]*?\}", raw)
    if obj_match:
        try:
            return json.loads(obj_match.group(0))
        except json.JSONDecodeError:
            pass
    return {"found": False, "confidence": 0.0}


def _render_crop(pdf_path: str, region: dict, page_index: int = 0,
                 render_scale: float = 4.0,
                 save_path: str | None = None) -> tuple[str, float, float, float, int, int, int]:
    """
    Render a PDF media-coord region to a PNG crop at render_scale × 72 DPI.

    Returns:
        (tmp_png_path, crop_vx0, crop_vy0, render_scale, rotation, media_w, media_h)
    """
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    rotation = page.rotation
    media_w = int(page.mediabox.width)
    media_h = int(page.mediabox.height)

    # Render full page
    mat = fitz.Matrix(render_scale, render_scale)
    pix = page.get_pixmap(matrix=mat)
    doc.close()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # Convert region media coords → visual pixel bbox
    corners_vis = [
        _media_to_visual(region["x0"], region["y0"], rotation, media_w, media_h, render_scale),
        _media_to_visual(region["x1"], region["y0"], rotation, media_w, media_h, render_scale),
        _media_to_visual(region["x1"], region["y1"], rotation, media_w, media_h, render_scale),
        _media_to_visual(region["x0"], region["y1"], rotation, media_w, media_h, render_scale),
    ]
    vx0 = max(0, int(min(c[0] for c in corners_vis)))
    vy0 = max(0, int(min(c[1] for c in corners_vis)))
    vx1 = min(img.width,  int(max(c[0] for c in corners_vis)))
    vy1 = min(img.height, int(max(c[1] for c in corners_vis)))

    crop = img.crop((vx0, vy0, vx1, vy1))
    if save_path:
        crop.save(save_path)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    crop.save(tmp.name)
    return tmp.name, vx0, vy0, render_scale, rotation, media_w, media_h


def _vision_result_to_segment(
    result: dict,
    crop_vx0: float, crop_vy0: float,
    render_scale: float,
    rotation: int, media_w: int, media_h: int,
    seg_id: str,
    page: int = 0,
    min_confidence: float = 0.6,
) -> DuctSegment | None:
    """
    Convert a vision bounding-box result (in crop pixel coords) back to a DuctSegment
    in media (PDF-point) coordinates.
    """
    if not result.get("found") or result.get("confidence", 0) < min_confidence:
        return None

    # Crop-pixel → full-image visual pixel
    px0_v = crop_vx0 + result["x0"]
    py0_v = crop_vy0 + result["y0"]
    px1_v = crop_vx0 + result["x1"]
    py1_v = crop_vy0 + result["y1"]

    # Visual pixel → media coords (all four corners)
    corners_m = [
        _visual_to_media(px0_v, py0_v, rotation, media_w, media_h, render_scale),
        _visual_to_media(px1_v, py0_v, rotation, media_w, media_h, render_scale),
        _visual_to_media(px1_v, py1_v, rotation, media_w, media_h, render_scale),
        _visual_to_media(px0_v, py1_v, rotation, media_w, media_h, render_scale),
    ]
    mx0 = min(c[0] for c in corners_m)
    my0 = min(c[1] for c in corners_m)
    mx1 = max(c[0] for c in corners_m)
    my1 = max(c[1] for c in corners_m)

    w_m = mx1 - mx0
    h_m = my1 - my0
    if w_m <= 0 or h_m <= 0:
        return None

    long_pt  = max(w_m, h_m)
    short_pt = min(w_m, h_m)
    aspect   = long_pt / short_pt if short_pt > 0 else 0.0

    raw_orient = result.get("orientation", "")
    if raw_orient == "diagonal":
        orientation = "D"
    else:
        # Always derive H/V from the actual bbox dimensions; never trust vision's H/V label.
        orientation = "H" if w_m >= h_m else "V"

    # If bounding box is near-square (aspect < 1.5) and confidence is high, it is almost
    # certainly a diagonal duct transition (45° diagonals have bounding-box aspect ≈ 1.0).
    # Upgrade H/V labels to D so the diagonal-duct filter keeps them.
    if orientation in ("H", "V") and aspect < 1.5 and result.get("confidence", 0) >= 0.8:
        orientation = "D"

    cx, cy = (mx0 + mx1) / 2, (my0 + my1) / 2
    if orientation == "H":
        centerline = [[mx0, cy], [mx1, cy]]
    elif orientation == "V":
        centerline = [[cx, my0], [cx, my1]]
    else:
        centerline = [[mx0, my0], [mx1, my1]]

    return DuctSegment(
        id=seg_id,
        rect=[mx0, my0, mx1, my1],
        orientation=orientation,
        long_pt=round(long_pt, 2),
        short_pt=round(short_pt, 2),
        aspect=round(aspect, 2),
        centerline=centerline,
        source="vision",
        confidence=result.get("confidence", 0.7),
        page=page,
    )


# ---------------------------------------------------------------------------
# Cluster-based diagonal duct detection (no vision API needed)
# ---------------------------------------------------------------------------

def detect_diagonal_ducts_from_clusters(
    pdf_path: str,
    vector_ducts: list[DuctSegment],
    page_index: int = 0,
    min_hatch_points: int = 6,
    debug: bool = False,
) -> list[DuctSegment]:
    """
    Detect diagonal duct segments directly from diagonal hatch-line clusters.

    Diagonal rectangular ducts are drawn with dense diagonal hatch marks that
    fill their interior.  Each hatch line contributes 2 endpoints; the cluster
    of ALL those endpoints has point_count >> 4 (the 4 endpoints a VAV-box
    X-mark produces).

    Threshold: point_count >= min_hatch_points (default 6) cleanly separates
    the 22"×14" diagonal duct (7 pts) from all VAV-box / elbow noise (≤ 4 pts).

    The RAW (unpadded) cluster bounding box is used directly as the duct rect —
    no vision API call, no false positives from equipment.
    """
    import math

    from config.settings import (
        TITLE_BLOCK_Y_MIN_PT, DUCT_MIN_LONG_PT, DUCT_MIN_SHORT_PT,
    )

    doc = fitz.open(pdf_path)
    page = doc[page_index]
    paths = page.get_drawings()
    doc.close()

    # Collect endpoints of long, truly-diagonal black line items
    diag_points: list[tuple[float, float]] = []
    for p in paths:
        color = p.get("color")
        if not (color and all(c < 0.15 for c in color[:3])):
            continue
        for item in p.get("items", []):
            if item[0] != "l":
                continue
            try:
                p1, p2 = item[1], item[2]
                dx, dy = abs(p2.x - p1.x), abs(p2.y - p1.y)
                if dx > 10 and dy > 10 and math.hypot(dx, dy) >= 40:
                    diag_points.extend([(p1.x, p1.y), (p2.x, p2.y)])
            except Exception:
                pass

    if not diag_points:
        return []

    # Cluster nearby diagonal endpoints
    CLUSTER_RADIUS = 80
    clusters: list[list[tuple[float, float]]] = []
    used = [False] * len(diag_points)
    for i, pt in enumerate(diag_points):
        if used[i]:
            continue
        cluster = [pt]
        used[i] = True
        for j, other in enumerate(diag_points):
            if used[j]:
                continue
            if math.hypot(pt[0] - other[0], pt[1] - other[1]) < CLUSTER_RADIUS:
                cluster.append(other)
                used[j] = True
        clusters.append(cluster)

    if debug:
        print(f"  [diagonal_cluster] total diag pts={len(diag_points)} clusters={len(clusters)}")
        for c in sorted(clusters, key=lambda cl: -len(cl))[:8]:
            xs = [p[0] for p in c]; ys = [p[1] for p in c]
            print(f"    pts={len(c):3d}  bbox=[{min(xs):.0f},{min(ys):.0f},{max(xs):.0f},{max(ys):.0f}]  "
                  f"size={max(xs)-min(xs):.0f}×{max(ys)-min(ys):.0f}")

    segments: list[DuctSegment] = []
    seg_counter = 1

    for cluster in clusters:
        if len(cluster) < min_hatch_points:
            continue

        xs = [p[0] for p in cluster]
        ys = [p[1] for p in cluster]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)

        # Exclude title block
        if y0 > TITLE_BLOCK_Y_MIN_PT:
            if debug:
                print(f"  [diagonal_cluster] pts={len(cluster)} SKIPPED (title block)")
            continue

        w, h = x1 - x0, y1 - y0
        long_pt = max(w, h)
        short_pt = min(w, h)

        if long_pt < DUCT_MIN_LONG_PT or short_pt < DUCT_MIN_SHORT_PT:
            if debug:
                print(f"  [diagonal_cluster] pts={len(cluster)} SKIPPED (too small: {w:.0f}×{h:.0f})")
            continue

        aspect = long_pt / short_pt if short_pt > 0 else 0.0
        seg = DuctSegment(
            id=f"diag_{seg_counter:03d}",
            rect=[x0, y0, x1, y1],
            orientation="D",
            long_pt=round(long_pt, 2),
            short_pt=round(short_pt, 2),
            aspect=round(aspect, 2),
            centerline=[[x0, y0], [x1, y1]],
            source="vector",
            confidence=0.9,
            page=page_index,
        )
        if debug:
            print(f"  [diagonal_cluster] pts={len(cluster)} → {seg.id} rect={[round(v) for v in seg.rect]} "
                  f"long={seg.long_pt:.1f} short={seg.short_pt:.1f}")
        segments.append(seg)
        seg_counter += 1

    return segments


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_vision_fallback_ducts(
    pdf_path: str,
    vector_ducts: list[DuctSegment],
    page_index: int = 0,
    render_scale: float = 4.0,
    min_confidence: float = 0.6,
    debug: bool = False,
    crop_save_dir: str | None = None,
) -> list[DuctSegment]:
    """
    Find duct segments that vector extraction missed (diagonal, polygon-drawn).

    1. Discover candidate regions from diagonal line clusters.
    2. Skip any region already well-covered by a vector duct.
    3. For each remaining candidate: render crop → call vision → parse bbox.
    4. Post-filter: H/V must have aspect >= 2.0; diagonal must have confidence >= 0.8.
    5. Return new DuctSegments in media coords.

    These results should be appended to the vector duct list, then the combined
    list de-duplicated by IoU before Phase 2.
    """
    from config.settings import DUCT_MIN_ASPECT, DUCT_MIN_LONG_PT

    candidates = _find_diagonal_candidate_regions(pdf_path, page_index)
    covered = _regions_covered_by_vector(vector_ducts)

    if debug:
        print(f"  [vision_fallback] diagonal candidate regions found: {len(candidates)}")
        for i, r in enumerate(candidates):
            print(f"    region {i}: x0={r['x0']:.0f} y0={r['y0']:.0f} x1={r['x1']:.0f} y1={r['y1']:.0f}")

    new_segments: list[DuctSegment] = []
    seg_counter = 1

    for i, region in enumerate(candidates):
        if _is_covered(region, covered):
            if debug:
                print(f"  [vision_fallback] region {i} SKIPPED (covered by vector duct)")
            continue

        if debug:
            print(f"  [vision_fallback] region {i} → calling vision...")

        save_path = None
        if crop_save_dir:
            save_path = str(Path(crop_save_dir) / f"vis_crop_region_{i:02d}.png")

        try:
            tmp_path, crop_vx0, crop_vy0, rscale, rotation, media_w, media_h = \
                _render_crop(pdf_path, region, page_index, render_scale, save_path=save_path)
        except Exception as e:
            if debug:
                print(f"  [vision_fallback] region {i} crop FAILED: {e}")
            continue

        try:
            result = _call_vision(tmp_path)
            if debug:
                print(f"  [vision_fallback] region {i} vision result: {result}")
        except Exception as e:
            if debug:
                print(f"  [vision_fallback] region {i} vision call FAILED: {e}")
            result = {"found": False}
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        seg = _vision_result_to_segment(
            result, crop_vx0, crop_vy0, rscale,
            rotation, media_w, media_h,
            seg_id=f"vis_{seg_counter:03d}",
            page=page_index,
            min_confidence=min_confidence,
        )
        if seg is None:
            if debug:
                print(f"  [vision_fallback] region {i} → no segment (low confidence or bad result)")
            continue

        # Post-filter: same quality gates as vector extraction
        if seg.orientation == "D":
            # Diagonal: bounding box is inherently square-ish; require high confidence instead
            if result.get("confidence", 0) < 0.8:
                if debug:
                    print(f"  [vision_fallback] region {i} diagonal SKIPPED (conf={result.get('confidence',0):.2f} < 0.8)")
                continue
            if seg.long_pt < DUCT_MIN_LONG_PT:
                if debug:
                    print(f"  [vision_fallback] region {i} diagonal SKIPPED (long={seg.long_pt:.1f} < {DUCT_MIN_LONG_PT})")
                continue
        else:
            if seg.aspect < DUCT_MIN_ASPECT:
                if debug:
                    print(f"  [vision_fallback] region {i} SKIPPED (aspect={seg.aspect:.2f} < {DUCT_MIN_ASPECT})")
                continue

        if debug:
            print(f"  [vision_fallback] region {i} → segment {seg.id} orient={seg.orientation} "
                  f"aspect={seg.aspect:.1f} rect={[round(v,1) for v in seg.rect]}")
        new_segments.append(seg)
        covered.append({"x0": seg.rect[0], "y0": seg.rect[1],
                        "x1": seg.rect[2], "y1": seg.rect[3]})
        seg_counter += 1

    return new_segments
