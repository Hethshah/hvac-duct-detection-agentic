"""
Phase 3 tests — Raster Duct Extractor

Run with:
    cd hvac-duct-detection-v2
    python -m pytest tests/test_phase3.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from config.settings import (
    SAMPLE_INPUT,
    RASTER_SCALE,
    PARALLEL_GAP_MIN_PX,
    PARALLEL_GAP_MAX_PX,
    PARALLEL_OVERLAP_MIN,
    DEDUP_IOU_THRESHOLD,
)
from tools.raster_duct_extractor import (
    _render_black_mask,
    _apply_title_block_mask,
    _find_h_blobs,
    _find_v_blobs,
    _overlap_frac,
    _pair_h_blobs,
    _pair_v_blobs,
    _h_pair_to_segment,
    _v_pair_to_segment,
    _iou,
    _centroid_near_phase1,
    extract_raster_ducts,
)
from tools.vector_duct_extractor import extract_ducts
from models.duct_segment import DuctSegment


# ── Unit: overlap fraction ─────────────────────────────────────────────────────

class TestOverlapFrac:
    def test_full_overlap(self):
        assert _overlap_frac(0, 100, 0, 100) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert _overlap_frac(0, 50, 60, 100) == pytest.approx(0.0)

    def test_partial_overlap(self):
        # [0,100] ∩ [50,150] = [50,100] = 51 px; shorter = 101
        result = _overlap_frac(0, 100, 50, 150)
        assert 0.4 < result < 0.6

    def test_contained(self):
        # [20,80] fully inside [0,100]; shorter = 61
        assert _overlap_frac(0, 100, 20, 80) == pytest.approx(1.0)


# ── Unit: IoU ─────────────────────────────────────────────────────────────────

class TestIoU:
    def test_identical_rects(self):
        r = [0.0, 0.0, 10.0, 10.0]
        assert _iou(r, r) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert _iou([0, 0, 5, 5], [10, 10, 20, 20]) == pytest.approx(0.0)

    def test_half_overlap_horizontal(self):
        a = [0.0, 0.0, 10.0, 4.0]
        b = [5.0, 0.0, 15.0, 4.0]
        iou = _iou(a, b)
        # intersection = 5×4=20, union = 10×4 + 10×4 - 20 = 60, iou=1/3
        assert iou == pytest.approx(1 / 3, rel=0.01)

    def test_symmetry(self):
        a, b = [0, 0, 6, 4], [3, 2, 9, 6]
        assert _iou(a, b) == pytest.approx(_iou(b, a))


# ── Unit: H pair → segment ────────────────────────────────────────────────────

class TestHPairToSegment:
    # H blobs at image rows 100-102 and 182-184, cols 0-400 (scale=4, media_w=1728)
    # gap ≈ 82 px = 20.5 pt  → valid short_pt
    # duct col span = 400 px = 100 pt → long_pt

    def _make_blob(self, r0, r1, c0, c1):
        return {"r0": r0, "r1": r1, "c0": c0, "c1": c1,
                "cy": (r0 + r1) / 2, "cx": (c0 + c1) / 2}

    def test_valid_v_duct(self):
        a = self._make_blob(100, 102, 0, 400)
        b = self._make_blob(182, 184, 0, 400)
        seg = _h_pair_to_segment(a, b, scale=4, media_w_pt=1728.0, seg_id="r_001")
        assert seg is not None
        assert seg.orientation == "V"
        assert seg.rect[0] == pytest.approx(1728 - 184 / 4, rel=0.01)
        assert seg.rect[2] == pytest.approx(1728 - 100 / 4, rel=0.01)
        assert seg.rect[1] == pytest.approx(0.0)
        assert seg.rect[3] == pytest.approx(100.0)

    def test_short_pt_too_small(self):
        # gap = 8 px = 2 pt < DUCT_MIN_SHORT_PT=10
        a = self._make_blob(100, 101, 0, 400)
        b = self._make_blob(108, 109, 0, 400)
        seg = _h_pair_to_segment(a, b, scale=4, media_w_pt=1728.0, seg_id="r_001")
        assert seg is None

    def test_long_pt_too_short(self):
        # col overlap = 40 px = 10 pt < DUCT_MIN_LONG_PT=30
        a = self._make_blob(100, 102, 0, 40)
        b = self._make_blob(182, 184, 0, 40)
        seg = _h_pair_to_segment(a, b, scale=4, media_w_pt=1728.0, seg_id="r_001")
        assert seg is None

    def test_no_col_overlap_returns_none(self):
        a = self._make_blob(100, 102, 0, 200)
        b = self._make_blob(182, 184, 250, 400)  # no overlap
        seg = _h_pair_to_segment(a, b, scale=4, media_w_pt=1728.0, seg_id="r_001")
        assert seg is None


# ── Unit: V pair → segment ────────────────────────────────────────────────────

class TestVPairToSegment:
    def _make_blob(self, r0, r1, c0, c1):
        return {"r0": r0, "r1": r1, "c0": c0, "c1": c1,
                "cy": (r0 + r1) / 2, "cx": (c0 + c1) / 2}

    def test_valid_h_duct(self):
        # Blobs at cols 200-202 and 282-284, rows 0-400 (scale=4, media_w=1728)
        # col gap ≈ 82 px = 20.5 pt short_pt; row span = 400 px = 100 pt long_pt
        a = self._make_blob(0, 400, 200, 202)
        b = self._make_blob(0, 400, 282, 284)
        seg = _v_pair_to_segment(a, b, scale=4, media_w_pt=1728.0, seg_id="r_001")
        assert seg is not None
        assert seg.orientation == "H"
        # x range from row overlap [0, 400]: x0 = 1728-400/4=1628, x1=1728-0/4=1728
        assert seg.rect[0] == pytest.approx(1728 - 400 / 4, rel=0.01)
        assert seg.rect[2] == pytest.approx(1728 - 0 / 4, rel=0.01)
        assert seg.rect[1] == pytest.approx(200 / 4, rel=0.01)
        assert seg.rect[3] == pytest.approx(284 / 4, rel=0.01)

    def test_no_row_overlap_returns_none(self):
        a = self._make_blob(0, 200, 200, 202)
        b = self._make_blob(300, 500, 282, 284)  # rows don't overlap
        seg = _v_pair_to_segment(a, b, scale=4, media_w_pt=1728.0, seg_id="r_001")
        assert seg is None


# ── Unit: centroid proximity ──────────────────────────────────────────────────

class TestCentroidNearPhase1:
    def test_centroid_inside_rect_is_near(self):
        assert _centroid_near_phase1([5, 5, 15, 15], [[0, 0, 20, 20]], margin_pt=0)

    def test_centroid_outside_within_margin(self):
        # centroid at (10, 25), rect ends at y=20, distance=5pt → within margin 10
        assert _centroid_near_phase1([5, 22, 15, 28], [[0, 0, 20, 20]], margin_pt=10)

    def test_centroid_outside_beyond_margin(self):
        # centroid at (10, 35), rect ends at y=20, distance=15pt > margin 10
        assert not _centroid_near_phase1([5, 32, 15, 38], [[0, 0, 20, 20]], margin_pt=10)

    def test_multiple_rects_any_match(self):
        # First rect far, second rect close
        assert _centroid_near_phase1(
            [100, 100, 110, 110],
            [[0, 0, 10, 10], [95, 95, 115, 115]],
            margin_pt=5,
        )


# ── Unit: H blob pairing ──────────────────────────────────────────────────────

class TestPairHBlobs:
    def _blob(self, cy, c0, c1):
        thickness = 2
        return {"r0": int(cy), "r1": int(cy) + thickness,
                "c0": c0, "c1": c1, "cy": cy, "cx": (c0 + c1) / 2}

    def test_valid_pair_found(self):
        a = self._blob(100, 0, 400)
        b = self._blob(180, 0, 400)  # gap=80, overlap=1.0
        pairs = _pair_h_blobs([a, b])
        assert len(pairs) == 1

    def test_gap_too_small_not_paired(self):
        a = self._blob(100, 0, 400)
        b = self._blob(100 + PARALLEL_GAP_MIN_PX - 1, 0, 400)
        pairs = _pair_h_blobs([a, b])
        assert len(pairs) == 0

    def test_gap_too_large_not_paired(self):
        a = self._blob(100, 0, 400)
        b = self._blob(100 + PARALLEL_GAP_MAX_PX + 10, 0, 400)
        pairs = _pair_h_blobs([a, b])
        assert len(pairs) == 0

    def test_insufficient_overlap_not_paired(self):
        a = self._blob(100, 0, 200)
        b = self._blob(180, 500, 700)  # no column overlap
        pairs = _pair_h_blobs([a, b])
        assert len(pairs) == 0

    def test_each_blob_used_once(self):
        # Three blobs: a pairs with b, c is left alone
        a = self._blob(100, 0, 400)
        b = self._blob(180, 0, 400)
        c = self._blob(260, 0, 400)
        pairs = _pair_h_blobs([a, b, c])
        assert len(pairs) == 1  # a+b pair; c unpaired


# ── Unit: V blob pairing ──────────────────────────────────────────────────────

class TestPairVBlobs:
    def _blob(self, cx, r0, r1):
        thickness = 2
        return {"r0": r0, "r1": r1,
                "c0": int(cx), "c1": int(cx) + thickness,
                "cy": (r0 + r1) / 2, "cx": cx}

    def test_valid_pair_found(self):
        a = self._blob(100, 0, 400)
        b = self._blob(180, 0, 400)  # gap=80, overlap=1.0
        pairs = _pair_v_blobs([a, b])
        assert len(pairs) == 1

    def test_gap_too_large_not_paired(self):
        a = self._blob(100, 0, 400)
        b = self._blob(100 + PARALLEL_GAP_MAX_PX + 10, 0, 400)
        pairs = _pair_v_blobs([a, b])
        assert len(pairs) == 0


# ── Integration: against actual PDF ──────────────────────────────────────────

@pytest.mark.skipif(
    not SAMPLE_INPUT.exists(),
    reason="Sample PDF not available",
)
class TestRasterExtractIntegration:
    @pytest.fixture(scope="class")
    def page_info(self):
        _, info = _render_black_mask(str(SAMPLE_INPUT), 0, RASTER_SCALE)
        return info

    @pytest.fixture(scope="class")
    def mask_clean(self):
        mask, info = _render_black_mask(str(SAMPLE_INPUT), 0, RASTER_SCALE)
        return _apply_title_block_mask(mask, info)

    @pytest.fixture(scope="class")
    def new_segs(self):
        p1 = extract_ducts(str(SAMPLE_INPUT))
        return extract_raster_ducts(str(SAMPLE_INPUT), p1)

    def test_render_produces_correct_image_size(self, page_info):
        assert page_info["img_w"] == RASTER_SCALE * round(page_info["img_w"] / RASTER_SCALE)
        assert page_info["img_h"] > 0
        assert page_info["img_w"] > 0

    def test_render_has_black_pixels(self, mask_clean):
        assert (mask_clean > 0).sum() > 0, "Mask has no black pixels"

    def test_h_blobs_detected(self, mask_clean):
        blobs = _find_h_blobs(mask_clean)
        assert len(blobs) > 0, "No H blobs detected"

    def test_v_blobs_detected(self, mask_clean):
        blobs = _find_v_blobs(mask_clean)
        assert len(blobs) > 0, "No V blobs detected"

    def test_zero_new_segments_for_vector_pdf(self, new_segs):
        # input1.pdf is pure vector — all ducts already in Phase 1
        assert len(new_segs) == 0, (
            f"Expected 0 new raster segments for vector PDF; got {len(new_segs)}: "
            + str([s.to_dict() for s in new_segs])
        )

    def test_all_new_segs_have_raster_source(self, new_segs):
        for seg in new_segs:
            assert seg.source == "raster"

    def test_all_new_segs_pass_aspect_filter(self, new_segs):
        from config.settings import DUCT_MIN_ASPECT
        for seg in new_segs:
            assert seg.aspect >= DUCT_MIN_ASPECT, (
                f"{seg.id} has aspect {seg.aspect} < {DUCT_MIN_ASPECT}"
            )
