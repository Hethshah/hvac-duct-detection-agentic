from pathlib import Path

import structlog
from PIL import Image, ImageDraw

from config.settings import settings
from tools.annotation_tools import (
    _load_font,
    draw_bbox,
    draw_length_annotation,
    export_pdf,
    render_label,
)

logger = structlog.get_logger()


def run_annotation(state: dict) -> dict:
    """
    Run annotation pipeline: overlay colored duct polygons and labels onto page images,
    then export to an annotated PDF.
    Updates state["output_pdf"] with the final PDF path.
    """
    page_images: list[str] = state["page_images"]
    duct_segments: list[dict] = state.get("duct_segments", [])
    measurements: list[dict] = state.get("measurements", [])
    output_dir = state.get("output_dir", settings.output_dir)

    # Build lookup: segment_id → segment (for polygon access)
    segments_by_id: dict[str, dict] = {
        s.get("id", f"seg_{i}"): s for i, s in enumerate(duct_segments)
    }

    # Build lookup: segment_id → measurement
    measure_by_seg: dict[str, dict] = {m["segment_id"]: m for m in measurements}

    font = _load_font()
    annotated_paths: list[str] = []
    total_rendered = 0

    for page_idx, page_path in enumerate(page_images):
        img = Image.open(page_path).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Collect segments on this page
        page_segs = [s for s in duct_segments if s.get("page", 0) == page_idx]
        rendered = 0

        for seg in page_segs:
            polygon = seg.get("polygon", [])
            if not polygon:
                continue

            duct_type = seg.get("type", "unknown")
            draw_bbox(draw, polygon, duct_type)

            m = measure_by_seg.get(seg.get("id", ""))
            if m:
                render_label(draw, m, polygon, font, img_size=img.size)
                if m.get("length_ft"):
                    draw_length_annotation(draw, polygon, m["length_ft"], font, img_size=img.size)

            rendered += 1

        # Composite overlay onto page and convert back to RGB for saving
        composited = Image.alpha_composite(img, overlay).convert("RGB")
        out_path = str(Path(output_dir) / f"page_{page_idx:03d}_annotated.png")
        composited.save(out_path)
        annotated_paths.append(out_path)
        total_rendered += rendered

        logger.info(
            "annotation_page_complete",
            page=page_idx,
            segments_rendered=rendered,
            out=out_path,
        )

    # Export all annotated pages to a single PDF
    output_pdf_path = str(Path(output_dir) / "annotated.pdf")
    export_pdf(annotated_paths, output_pdf_path, dpi=settings.dpi)

    state["output_pdf"] = output_pdf_path
    state["output_pngs"] = annotated_paths
    logger.info(
        "annotation_complete",
        pages=len(annotated_paths),
        total_segments=total_rendered,
        output_pdf=output_pdf_path,
    )
    return state
