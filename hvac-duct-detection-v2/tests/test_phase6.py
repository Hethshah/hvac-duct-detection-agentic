"""
Phase 6 tests — Annotation renderer and end-to-end CLI

Run with:
    cd hvac-duct-detection-v2
    python -m pytest tests/test_phase6.py -v
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from PIL import Image

from config.settings import SAMPLE_INPUT, RENDER_DPI
from models.annotated_duct import AnnotatedDuct
from tools.annotation_renderer import (
    _media_rect_to_pixel,
    _outline_color,
    _centerline_pixels,
    render_annotated_page,
    _COLOR_NORMAL,
)


# ── Helper ─────────────────────────────────────────────────────────────────────

def _make_duct(
    segment_id="duct_001",
    confidence=1.0,
    length_mismatch=False,
    length_ft_measured=10.0,
    length_ft_label=None,
    orientation="H",
    rect=None,
    page=0,
    duct_label_id=None,
    cross_section=None,
    is_round=False,
    unlabeled=True,
):
    return AnnotatedDuct(
        segment_id=segment_id,
        duct_label_id=duct_label_id,
        rect=rect or [100.0, 100.0, 347.0, 120.0],
        orientation=orientation,
        length_ft_measured=length_ft_measured,
        length_ft_label=length_ft_label,
        length_mismatch=length_mismatch,
        cross_section=cross_section,
        is_round=is_round,
        unlabeled=unlabeled,
        confidence=confidence,
        source="vector",
        page=page,
    )


# ── Unit: _media_rect_to_pixel ────────────────────────────────────────────────

class TestMediaRectToPixel:
    def test_rotation_0_scales_directly(self):
        px0, py0, px1, py1 = _media_rect_to_pixel(
            [100, 200, 300, 400], rotation=0, media_w=1000, media_h=2000, scale=2.0
        )
        assert px0 == 200 and py0 == 400
        assert px1 == 600 and py1 == 800

    def test_rotation_270_swaps_axes(self):
        # px = y_media * scale;  py = (media_w - x_media) * scale
        px0, py0, px1, py1 = _media_rect_to_pixel(
            [100, 200, 300, 400], rotation=270, media_w=1000, media_h=2000, scale=2.0
        )
        assert px0 == int(200 * 2)
        assert px1 == int(400 * 2)
        assert py0 == int((1000 - 300) * 2)
        assert py1 == int((1000 - 100) * 2)

    def test_rotation_180_mirrors_both_axes(self):
        px0, py0, px1, py1 = _media_rect_to_pixel(
            [100, 200, 300, 400], rotation=180, media_w=1000, media_h=2000, scale=2.0
        )
        assert px0 == int((1000 - 300) * 2)
        assert px1 == int((1000 - 100) * 2)
        assert py0 == int((2000 - 400) * 2)
        assert py1 == int((2000 - 200) * 2)

    def test_all_rotations_produce_ordered_coords(self):
        for rot in [0, 90, 180, 270]:
            px0, py0, px1, py1 = _media_rect_to_pixel(
                [100, 200, 300, 400], rotation=rot, media_w=1000, media_h=2000, scale=2.0
            )
            assert px0 <= px1, f"px unordered for rotation={rot}"
            assert py0 <= py1, f"py unordered for rotation={rot}"


# ── Unit: _centerline_pixels ──────────────────────────────────────────────────

class TestCenterlinePixels:
    def test_h_duct_centerline_spans_full_x(self):
        # H duct: centerline runs along y-midpoint from x0 to x1
        # At rotation=0, scale=1: px = x, py = y
        d = _make_duct(orientation="H", rect=[0, 90, 200, 110])
        p1, p2 = _centerline_pixels(d, rotation=0, media_w=1000, media_h=2000, scale=1.0)
        assert p1 == (0, 100) and p2 == (200, 100)

    def test_v_duct_centerline_spans_full_y(self):
        d = _make_duct(orientation="V", rect=[90, 0, 110, 300])
        p1, p2 = _centerline_pixels(d, rotation=0, media_w=1000, media_h=2000, scale=1.0)
        assert p1 == (100, 0) and p2 == (100, 300)

    def test_h_duct_rotation270_swaps_axes(self):
        # H duct rect [x0,y0,x1,y1], centerline at y=cy from x0 to x1
        # At rotation=270: px = y*scale, py = (media_w - x)*scale
        d = _make_duct(orientation="H", rect=[0, 90, 200, 110])
        # centerline: (0, 100) → (200, 100) in media
        # In visual: p1=(100*1, (1000-0)*1)=(100,1000), p2=(100*1,(1000-200)*1)=(100,800)
        p1, p2 = _centerline_pixels(d, rotation=270, media_w=1000, media_h=2000, scale=1.0)
        assert p1 == (100, 1000) and p2 == (100, 800)

    def test_centerline_length_matches_long_pt(self):
        import math
        # H duct: long_pt = x1-x0 = 200
        d = _make_duct(orientation="H", rect=[0, 90, 200, 110])
        p1, p2 = _centerline_pixels(d, rotation=0, media_w=1000, media_h=2000, scale=1.0)
        length_px = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        assert length_px == pytest.approx(200.0)


# ── Unit: _outline_color ──────────────────────────────────────────────────────

class TestOutlineColor:
    def test_mismatch_is_blue(self):
        assert _outline_color(_make_duct(length_mismatch=True)) == _COLOR_NORMAL

    def test_normal_is_blue(self):
        d = _make_duct(length_mismatch=False, unlabeled=False)
        assert _outline_color(d) == _COLOR_NORMAL

    def test_unlabeled_is_blue(self):
        d = _make_duct(unlabeled=True, length_mismatch=False)
        assert _outline_color(d) == _COLOR_NORMAL


# ── Integration: render ───────────────────────────────────────────────────────

@pytest.mark.skipif(
    not SAMPLE_INPUT.exists(),
    reason="Sample PDF not available",
)
class TestRenderAnnotatedPage:
    @pytest.fixture(scope="class")
    def annotated(self):
        from tools.vector_duct_extractor import extract_ducts
        from tools.label_extractor import extract_labels_with_scale
        from tools.duct_annotator import annotate_ducts
        segs = extract_ducts(str(SAMPLE_INPUT))
        p2   = extract_labels_with_scale(str(SAMPLE_INPUT), segs)
        return annotate_ducts(segs, p2["labels"], p2["pt_per_ft"])

    @pytest.fixture(scope="class")
    def rendered_png(self, annotated, tmp_path_factory):
        out = tmp_path_factory.mktemp("render") / "annotated.png"
        render_annotated_page(str(SAMPLE_INPUT), annotated, str(out))
        return out

    def test_file_created(self, rendered_png):
        assert rendered_png.exists()

    def test_valid_png(self, rendered_png):
        img = Image.open(rendered_png)
        assert img.format == "PNG"

    def test_dimensions_match_display_at_dpi(self, rendered_png):
        import fitz
        doc  = fitz.open(str(SAMPLE_INPUT))
        page = doc[0]
        scale = RENDER_DPI / 72.0
        ew = int(page.rect.width  * scale)
        eh = int(page.rect.height * scale)
        doc.close()

        img = Image.open(rendered_png)
        assert abs(img.width  - ew) <= 2
        assert abs(img.height - eh) <= 2

    def test_nontrivial_file_size(self, rendered_png):
        assert rendered_png.stat().st_size > 100_000  # > 100 KB for a 300-DPI floor plan


# ── Integration: end-to-end CLI ───────────────────────────────────────────────

@pytest.mark.skipif(
    not SAMPLE_INPUT.exists(),
    reason="Sample PDF not available",
)
class TestCLI:
    @pytest.fixture(scope="class")
    def cli_output(self, tmp_path_factory):
        import subprocess
        out_dir = tmp_path_factory.mktemp("cli")
        script  = Path(__file__).parent.parent / "run.py"
        result  = subprocess.run(
            [sys.executable, str(script), str(SAMPLE_INPUT),
             "--skip-vision", "--output-dir", str(out_dir)],
            capture_output=True, text=True, timeout=120,
        )
        return result, out_dir

    def test_exits_successfully(self, cli_output):
        result, _ = cli_output
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"

    def test_annotated_png_created(self, cli_output):
        _, out_dir = cli_output
        assert (out_dir / f"{SAMPLE_INPUT.stem}_annotated.png").exists()

    def test_summary_json_created(self, cli_output):
        _, out_dir = cli_output
        assert (out_dir / "summary.json").exists()

    def test_phase4_json_created(self, cli_output):
        _, out_dir = cli_output
        assert (out_dir / "phase4_annotated_ducts.json").exists()

    def test_summary_has_required_fields(self, cli_output):
        _, out_dir = cli_output
        summary = json.loads((out_dir / "summary.json").read_text())
        for key in ["input", "pt_per_ft", "segment_counts",
                    "label_counts", "annotation", "annotated_ducts"]:
            assert key in summary, f"Missing key: {key}"

    def test_summary_segment_counts_consistent(self, cli_output):
        _, out_dir = cli_output
        summary = json.loads((out_dir / "summary.json").read_text())
        sc = summary["segment_counts"]
        assert sc["vector"] + sc["raster"] == sc["total"]

    def test_annotated_ducts_count_matches_total(self, cli_output):
        _, out_dir = cli_output
        summary = json.loads((out_dir / "summary.json").read_text())
        assert len(summary["annotated_ducts"]) == summary["segment_counts"]["total"]
