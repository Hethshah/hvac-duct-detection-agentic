"""
Phase 1 tests — Vector Duct Extractor

Run with:
    cd hvac-duct-detection-v2
    python -m pytest tests/test_phase1.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from config.settings import SAMPLE_INPUT, TITLE_BLOCK_Y_MIN_PT, DUCT_MIN_ASPECT
from tools.vector_duct_extractor import (
    extract_ducts,
    _is_black,
    _path_is_quad_only,
    _segment_from_rect,
    _short_axis_overlap,
    _long_axis_gap,
    _cluster_segments,
)
from models.duct_segment import DuctSegment


# ── Unit tests ─────────────────────────────────────────────────────────────────

class TestIsBlack:
    def test_pure_black(self):
        assert _is_black((0.0, 0.0, 0.0))

    def test_near_black(self):
        assert _is_black((0.1, 0.05, 0.12))

    def test_at_boundary(self):
        assert not _is_black((0.15, 0.0, 0.0))

    def test_grey(self):
        assert not _is_black((0.4, 0.4, 0.4))

    def test_none(self):
        assert not _is_black(None)


class TestPathIsQuadOnly:
    def test_quad_only(self):
        path = {"items": [("qu", None), ("qu", None)]}
        assert _path_is_quad_only(path)

    def test_mixed(self):
        path = {"items": [("qu", None), ("l", None, None)]}
        assert not _path_is_quad_only(path)

    def test_empty(self):
        path = {"items": []}
        assert not _path_is_quad_only(path)

    def test_line_only(self):
        path = {"items": [("l", None, None)]}
        assert not _path_is_quad_only(path)


class TestSegmentFromRect:
    def _make_h_rect(self, x0=0, y0=0, length=200, width=20):
        return [x0, y0, x0 + length, y0 + width]

    def _make_v_rect(self, x0=0, y0=0, length=200, width=20):
        return [x0, y0, x0 + width, y0 + length]

    def test_valid_horizontal(self):
        seg = _segment_from_rect(self._make_h_rect(), "t_001", 0)
        assert seg is not None
        assert seg.orientation == "H"
        assert seg.long_pt == 200
        assert seg.short_pt == 20
        assert seg.aspect == 10.0

    def test_valid_vertical(self):
        seg = _segment_from_rect(self._make_v_rect(), "t_002", 0)
        assert seg is not None
        assert seg.orientation == "V"

    def test_too_small_short_side(self):
        seg = _segment_from_rect([0, 0, 200, 5], "t_003", 0)
        assert seg is None

    def test_too_large_short_side(self):
        seg = _segment_from_rect([0, 0, 400, 200], "t_004", 0)
        assert seg is None

    def test_too_short_long_side(self):
        seg = _segment_from_rect([0, 0, 20, 10], "t_005", 0)
        assert seg is None

    def test_aspect_too_low(self):
        # aspect = 60/40 = 1.5 < 2.0
        seg = _segment_from_rect([0, 0, 60, 40], "t_006", 0)
        assert seg is None

    def test_centerline_horizontal(self):
        seg = _segment_from_rect([100, 50, 300, 70], "t_007", 0)
        assert seg is not None
        assert seg.centerline[0][0] == 100   # starts at left edge
        assert seg.centerline[1][0] == 300   # ends at right edge
        assert seg.centerline[0][1] == 60.0  # midpoint of y-range

    def test_centerline_vertical(self):
        seg = _segment_from_rect([50, 100, 70, 300], "t_008", 0)
        assert seg is not None
        assert seg.centerline[0][1] == 100
        assert seg.centerline[1][1] == 300
        assert seg.centerline[0][0] == 60.0


class TestClustering:
    def _make_seg(self, rect, seg_id="s", orientation=None):
        seg = _segment_from_rect(rect, seg_id, 0)
        return seg

    def test_two_collinear_h_segments_merged(self):
        # Two horizontal segments, same y band, gap of 5pt (< CLUSTER_LONG_GAP_MAX_PT=8)
        a = self._make_seg([0, 10, 100, 30], "a")   # length=100, width=20
        b = self._make_seg([105, 10, 200, 30], "b")  # gap of 5pt
        assert a and b
        result = _cluster_segments([a, b])
        assert len(result) == 1
        merged = result[0]
        assert merged.rect[0] == 0
        assert merged.rect[2] == 200

    def test_large_gap_not_merged(self):
        a = self._make_seg([0, 10, 100, 30], "a")
        b = self._make_seg([150, 10, 250, 30], "b")  # gap of 50pt > 8
        assert a and b
        result = _cluster_segments([a, b])
        assert len(result) == 2

    def test_different_orientation_not_merged(self):
        h = self._make_seg([0, 10, 200, 30], "h")   # horizontal
        v = self._make_seg([10, 0, 30, 200], "v")   # vertical
        assert h and v
        result = _cluster_segments([h, v])
        assert len(result) == 2

    def test_short_axis_no_overlap_not_merged(self):
        # Two horizontal ducts at very different y positions
        a = self._make_seg([0, 10, 200, 30], "a")
        b = self._make_seg([0, 100, 200, 120], "b")  # far apart in y
        assert a and b
        result = _cluster_segments([a, b])
        assert len(result) == 2


# ── Integration tests against actual PDF ────────────────────────────────────────

@pytest.mark.skipif(
    not SAMPLE_INPUT.exists(),
    reason="Sample PDF not available",
)
class TestExtractDuctsIntegration:
    def test_minimum_segment_count(self):
        segments = extract_ducts(str(SAMPLE_INPUT))
        assert len(segments) >= 12, f"Expected >= 12 segments, got {len(segments)}"

    def test_no_title_block_segments(self):
        segments = extract_ducts(str(SAMPLE_INPUT))
        in_tb = [s for s in segments if s.rect[1] > TITLE_BLOCK_Y_MIN_PT and s.rect[3] > TITLE_BLOCK_Y_MIN_PT]
        assert len(in_tb) == 0, f"Found {len(in_tb)} segments in title block: {[s.id for s in in_tb]}"

    def test_all_aspects_above_minimum(self):
        segments = extract_ducts(str(SAMPLE_INPUT))
        # Diagonal ducts have square bounding boxes (aspect ≈ 1.0) by geometry; exempt them.
        violations = [s for s in segments if s.orientation != "D" and s.aspect < DUCT_MIN_ASPECT]
        assert len(violations) == 0, f"H/V segments with aspect < {DUCT_MIN_ASPECT}: {violations}"

    def test_longest_duct_range(self):
        segments = extract_ducts(str(SAMPLE_INPUT))
        assert segments, "No segments returned"
        longest = max(segments, key=lambda s: s.long_pt)
        # C03 is 18'-5" ≈ 450 pt; allow ±25% for clustering variations
        assert 300 <= longest.long_pt <= 600, (
            f"Longest duct {longest.long_pt:.1f} pt out of expected range [300, 600]"
        )

    def test_ids_are_unique(self):
        segments = extract_ducts(str(SAMPLE_INPUT))
        ids = [s.id for s in segments]
        assert len(ids) == len(set(ids)), "Duplicate segment IDs detected"

    def test_segments_have_valid_rects(self):
        segments = extract_ducts(str(SAMPLE_INPUT))
        for s in segments:
            x0, y0, x1, y1 = s.rect
            assert x0 < x1, f"{s.id}: x0 >= x1"
            assert y0 < y1, f"{s.id}: y0 >= y1"

    def test_segment_ids_sequential(self):
        segments = extract_ducts(str(SAMPLE_INPUT))
        for i, seg in enumerate(segments):
            assert seg.id == f"duct_{i + 1:03d}", f"Expected duct_{i+1:03d}, got {seg.id}"
