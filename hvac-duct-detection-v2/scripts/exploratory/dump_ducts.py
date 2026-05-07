"""
Exploratory script — Phase 1 dump

Usage:
    cd hvac-duct-detection-v2
    python scripts/exploratory/dump_ducts.py [path/to/file.pdf]

Prints a table of detected duct segments and writes phase1_vector_ducts.json
to outputs/<pdf-stem>/.
Also renders a debug PNG with duct rectangles overlaid in red on the page image.
"""

import sys
from pathlib import Path

# Allow running from the project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json

import fitz
from PIL import Image, ImageDraw

from config.settings import SAMPLE_INPUT, RENDER_DPI, TITLE_BLOCK_Y_MIN_PT, OUTPUTS_DIR
from tools.vector_duct_extractor import extract_ducts, extract_ducts_to_json


def _media_to_visual(x_m, y_m, rotation, media_w, media_h, scale):
    """
    Convert un-rotated mediabox coordinates to rendered-pixmap pixel coordinates.

    page.get_drawings() returns coords in the raw mediabox space (un-rotated).
    page.get_pixmap() renders in the visually-rotated space.
    We must apply the rotation transform before multiplying by scale.

    For 270° rotation (most common in landscape HVAC sheets stored portrait):
        visual_x = media_y
        visual_y = media_w - media_x
    """
    if rotation == 270:
        vx = y_m * scale
        vy = (media_w - x_m) * scale
    elif rotation == 90:
        vx = (media_h - y_m) * scale
        vy = x_m * scale
    elif rotation == 180:
        vx = (media_w - x_m) * scale
        vy = (media_h - y_m) * scale
    else:  # 0 — no rotation
        vx = x_m * scale
        vy = y_m * scale
    return vx, vy


def render_debug_image(pdf_path: str, segments, out_dir: Path, dpi: int = RENDER_DPI) -> str:
    """Render the PDF page and overlay detected duct rects in red."""
    doc   = fitz.open(pdf_path)
    page  = doc[0]
    rotation = page.rotation
    media_w  = page.mediabox.width   # un-rotated width
    media_h  = page.mediabox.height  # un-rotated height
    scale = dpi / 72
    mat   = fitz.Matrix(scale, scale)
    pix   = page.get_pixmap(matrix=mat)
    doc.close()

    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    draw = ImageDraw.Draw(img, "RGBA")

    for seg in segments:
        x0_m, y0_m, x1_m, y1_m = seg.rect

        if seg.orientation == "D" and seg.polygon:
            # Draw actual polygon shape for diagonal ducts
            vis_pts = [
                _media_to_visual(p[0], p[1], rotation, media_w, media_h, scale)
                for p in seg.polygon
            ]
            draw.polygon(vis_pts, fill=(220, 50, 50, 60), outline=(220, 50, 50, 255))
            label_x = min(v[0] for v in vis_pts) + 4
            label_y = min(v[1] for v in vis_pts) + 2
            draw.text((label_x, label_y), seg.id, fill=(220, 50, 50, 255))
        else:
            # Convert all four corners, then take the visual bounding box
            corners_vis = [
                _media_to_visual(x0_m, y0_m, rotation, media_w, media_h, scale),
                _media_to_visual(x1_m, y0_m, rotation, media_w, media_h, scale),
                _media_to_visual(x1_m, y1_m, rotation, media_w, media_h, scale),
                _media_to_visual(x0_m, y1_m, rotation, media_w, media_h, scale),
            ]
            vxs = [c[0] for c in corners_vis]
            vys = [c[1] for c in corners_vis]
            px0, py0, px1, py1 = min(vxs), min(vys), max(vxs), max(vys)

            draw.rectangle([px0, py0, px1, py1], fill=(220, 50, 50, 60), outline=(220, 50, 50, 255), width=4)
            draw.text((px0 + 4, py0 + 2), seg.id, fill=(220, 50, 50, 255))

    # Mark title-block boundary — TITLE_BLOCK_Y_MIN_PT is in media_y space;
    # for 270° rotation media_y maps to visual_x.
    if rotation == 270:
        tx = TITLE_BLOCK_Y_MIN_PT * scale
        draw.line([(tx, 0), (tx, img.height)], fill=(255, 165, 0, 200), width=3)
    else:
        ty = TITLE_BLOCK_Y_MIN_PT * scale
        draw.line([(0, ty), (img.width, ty)], fill=(255, 165, 0, 200), width=3)

    out_path = str(out_dir / "phase1_debug.png")
    img.save(out_path)
    return out_path


def main():
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else str(SAMPLE_INPUT)

    print(f"\nAnalysing: {pdf_path}")
    print("=" * 70)

    # Raw PDF path stats
    doc = fitz.open(pdf_path)
    page = doc[0]
    all_paths = page.get_drawings()
    rotation = page.rotation
    black_paths = [p for p in all_paths if p.get("color") and all(c < 0.15 for c in p["color"][:3])]
    quad_only   = [p for p in black_paths if all(item[0] == "qu" for item in p.get("items", []))]
    doc.close()

    print(f"Page rotation   : {rotation}°")
    print(f"Total paths     : {len(all_paths)}")
    print(f"Black-stroke    : {len(black_paths)}")
    print(f"Black quad-only : {len(quad_only)}")
    print()

    # Extract all segments (H/V from axis-aligned quads + D from diagonal paths)
    segments = extract_ducts(pdf_path)

    stem    = Path(pdf_path).stem
    out_dir = OUTPUTS_DIR / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'ID':<12} {'Src':<7} {'Orient':<6} {'Long(pt)':<10} {'Short(pt)':<10} {'Aspect':<8} {'Rect'}")
    print("-" * 90)
    for seg in sorted(segments, key=lambda s: -s.long_pt):
        print(
            f"{seg.id:<12} {seg.source:<7} {seg.orientation:<6} {seg.long_pt:<10.1f} "
            f"{seg.short_pt:<10.1f} {seg.aspect:<8.1f} "
            f"[{seg.rect[0]:.0f},{seg.rect[1]:.0f},{seg.rect[2]:.0f},{seg.rect[3]:.0f}]"
        )

    print()
    print(f"Total segments  : {len(segments)}")
    print(f"  Vector        : {sum(1 for s in segments if s.source == 'vector')}")
    print(f"  Vision        : {sum(1 for s in segments if s.source == 'vision')}")
    print(f"Horizontal      : {sum(1 for s in segments if s.orientation == 'H')}")
    print(f"Vertical        : {sum(1 for s in segments if s.orientation == 'V')}")
    print(f"Diagonal        : {sum(1 for s in segments if s.orientation == 'D')}")
    h_segments = [s for s in segments if s.orientation == "H"]
    v_segments = [s for s in segments if s.orientation == "V"]
    if h_segments:
        print(f"Longest H run   : {max(h_segments, key=lambda s: s.long_pt).long_pt:.1f} pt")
    if v_segments:
        print(f"Longest V run   : {max(v_segments, key=lambda s: s.long_pt).long_pt:.1f} pt")

    # Validation assertions
    print()
    print("Validation")
    print("-" * 40)
    ok = True

    check_count = len(segments) >= 12
    print(f"  [{'OK' if check_count else 'FAIL'}] segment count >= 12  (got {len(segments)})")
    ok = ok and check_count

    in_title_block = [s for s in segments if s.rect[1] > TITLE_BLOCK_Y_MIN_PT and s.rect[3] > TITLE_BLOCK_Y_MIN_PT]
    check_title = len(in_title_block) == 0
    print(f"  [{'OK' if check_title else 'FAIL'}] 0 segments in title block (got {len(in_title_block)})")
    ok = ok and check_title

    # Diagonal duct transitions are inherently near-square (bounding box aspect ≈ 1.0 at 45°).
    # Only apply the aspect constraint to axis-aligned (H/V) segments.
    bad_aspect = [s for s in segments if s.orientation != "D" and s.aspect < 2.0]
    check_aspect = len(bad_aspect) == 0
    print(f"  [{'OK' if check_aspect else 'FAIL'}] all H/V aspects >= 2.0 (violations: {len(bad_aspect)})")
    ok = ok and check_aspect

    # Largest segment should be ~450 pt (C03, 18'-5")
    if segments:
        longest = max(segments, key=lambda s: s.long_pt)
        check_longest = 380 <= longest.long_pt <= 560
        print(f"  [{'OK' if check_longest else 'WARN'}] longest duct ~450 pt  (got {longest.long_pt:.1f} pt, id={longest.id})")

    print()
    print(f"Overall: {'PASS' if ok else 'FAIL'}")

    # Write JSON output
    json_path = extract_ducts_to_json(pdf_path, str(out_dir))
    print(f"\nJSON written : {json_path}")

    # Render debug image
    debug_path = render_debug_image(pdf_path, segments, out_dir)
    print(f"Debug image  : {debug_path}")


if __name__ == "__main__":
    main()
