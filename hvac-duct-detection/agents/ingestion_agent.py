import json
import structlog

from config.settings import settings
from tools.pdf_tools import detect_scale_bar, extract_text_blocks, pdf_to_images

logger = structlog.get_logger()


def run_ingestion(state: dict) -> dict:
    """
    Run ingestion pipeline: PDF → page images, text blocks, scale ratio.

    Reads optional state keys:
      page_range (str)          — e.g. "1-3" to restrict processed pages (1-based)
      scale_ratio_override (float) — skip auto-detection and use this value directly

    Updates and returns the shared pipeline state dict.
    """
    pdf_path = state["pdf_path"]
    output_dir = state.get("output_dir", settings.output_dir)
    page_range: str = state.get("page_range", "")
    scale_ratio_override: float | None = state.get("scale_ratio_override")

    page_images = pdf_to_images(
        pdf_path,
        dpi=settings.dpi,
        output_dir=f"{output_dir}/pages",
        page_range=page_range,
    )

    text_blocks = extract_text_blocks(pdf_path, dpi=settings.dpi)

    if not text_blocks:
        logger.warning(
            "raster_only_pdf",
            pdf=pdf_path,
            tip="No text layer found — PDF may be raster-only. "
                "Dimension labels will rely entirely on vision nearby_labels.",
        )

    if scale_ratio_override is not None and scale_ratio_override > 0:
        scale_ratio = float(scale_ratio_override)
        logger.info("scale_ratio_override_applied", scale_ratio=scale_ratio)
    else:
        scale_ratio = detect_scale_bar(json.dumps(text_blocks), dpi=settings.dpi)
        if scale_ratio == 0.0 and page_images:
            logger.info("scale_bar_not_in_text_trying_vision", page=page_images[0])
            from tools.vision_tools import detect_scale_from_image
            scale_ratio = detect_scale_from_image(page_images[0], dpi=settings.dpi)

    logger.info(
        "ingestion_complete",
        pages=len(page_images),
        text_blocks=len(text_blocks),
        scale_ratio=scale_ratio,
        raster_only=not text_blocks,
    )

    state["page_images"] = page_images
    state["text_blocks"] = text_blocks
    state["scale_ratio"] = scale_ratio
    return state
