"""
Phase 2 tests — Label Extractor & Scale Calibration

Run with:
    cd hvac-duct-detection-v2
    python -m pytest tests/test_phase2.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from config.settings import (
    SAMPLE_INPUT,
    PT_PER_FT_MIN,
    PT_PER_FT_MAX,
)
from tools.label_extractor import (
    _parse_length,
    _parse_cross_section,
    _RE_DUCT_ID,
    _merge_adjacent_spans,
    _calibrate_scale,
    extract_labels,
    extract_labels_with_scale,
)
from tools.vector_duct_extractor import extract_ducts
from models.duct_segment import DuctSegment


# ── Unit: length parsing ───────────────────────────────────────────────────────

class TestParseLengthLabel:
    def test_standard_format(self):
        assert _parse_length("18' - 5\"") == pytest.approx(18 + 5/12, rel=1e-4)

    def test_zero_inches(self):
        assert _parse_length("10' - 0\"") == pytest.approx(10.0, rel=1e-4)

    def test_eight_six(self):
        assert _parse_length("8' - 6\"") == pytest.approx(8.5, rel=1e-4)

    def test_twelve_six(self):
        assert _parse_length("12' - 6\"") == pytest.approx(12.5, rel=1e-4)

    def test_embedded_in_longer_string(self):
        assert _parse_length("RUN: 10' - 0\" DUCT") == pytest.approx(10.0, rel=1e-4)

    def test_no_match_returns_none(self):
        assert _parse_length("C01") is None
        assert _parse_length("24X18") is None
        assert _parse_length("KITCHEN") is None


# ── Unit: cross-section parsing ───────────────────────────────────────────────

class TestParseCrossSection:
    def test_rect_uppercase(self):
        cs = _parse_cross_section("24X18")
        assert cs == {"cross_type": "rect", "width_in": 24, "height_in": 18}

    def test_rect_lowercase(self):
        cs = _parse_cross_section("22x14")
        assert cs == {"cross_type": "rect", "width_in": 22, "height_in": 14}

    def test_rect_embedded_in_label(self):
        cs = _parse_cross_section("SC-24X18X8.62BOX")
        # First match wins: 24X18, not 18X8
        assert cs == {"cross_type": "rect", "width_in": 24, "height_in": 18}

    def test_round_diameter(self):
        cs = _parse_cross_section("12\"Ø")
        assert cs == {"cross_type": "round", "diameter_in": 12}

    def test_round_no_quotes(self):
        cs = _parse_cross_section("8ø")
        assert cs == {"cross_type": "round", "diameter_in": 8}

    def test_implausible_dimensions_rejected(self):
        # 200 inches > 60 inch cap → rejected
        cs = _parse_cross_section("200X100")
        assert cs is None

    def test_no_match(self):
        assert _parse_cross_section("C03") is None
        assert _parse_cross_section("KITCHEN") is None


# ── Unit: duct ID pattern ─────────────────────────────────────────────────────

class TestDuctIdPattern:
    def test_valid_ids(self):
        for text in ["C01", "C02", "C03", "C05", "SA12", "RA3"]:
            assert _RE_DUCT_ID.match(text), f"Expected match for {text!r}"

    def test_invalid_ids(self):
        for text in ["KITCHEN", "104", "B.1", "10' - 0\"", "SC-24X18"]:
            assert not _RE_DUCT_ID.match(text), f"Expected no match for {text!r}"


# ── Unit: span merging ────────────────────────────────────────────────────────

class TestMergeAdjacentSpans:
    def _span(self, text, x0, y0, x1, y1):
        return {"text": text, "bbox": [x0, y0, x1, y1],
                "cx": (x0+x1)/2, "cy": (y0+y1)/2}

    def test_adjacent_spans_on_same_line_merged(self):
        spans = [
            self._span("18'", 100, 10, 120, 25),
            self._span("- 5\"", 125, 10, 155, 25),
        ]
        result = _merge_adjacent_spans(spans, gap_pt=20)
        assert len(result) == 1
        assert "18'" in result[0]["text"]
        assert "5\"" in result[0]["text"]

    def test_large_gap_not_merged(self):
        spans = [
            self._span("18'", 100, 10, 120, 25),
            self._span("- 5\"", 200, 10, 240, 25),
        ]
        result = _merge_adjacent_spans(spans, gap_pt=20)
        assert len(result) == 2

    def test_different_vertical_position_not_merged(self):
        spans = [
            self._span("10'", 100, 10, 120, 25),
            self._span("- 0\"", 125, 50, 155, 65),
        ]
        result = _merge_adjacent_spans(spans, gap_pt=20)
        assert len(result) == 2

    def test_single_span_unchanged(self):
        spans = [self._span("C01", 100, 10, 130, 25)]
        result = _merge_adjacent_spans(spans)
        assert len(result) == 1
        assert result[0]["text"] == "C01"


# ── Unit: scale calibration ───────────────────────────────────────────────────

class TestCalibrateScale:
    def _make_seg(self, long_pt, rect):
        seg = DuctSegment(
            id="t_001", rect=rect, orientation="H",
            long_pt=long_pt, short_pt=20.0, aspect=long_pt/20,
            centerline=[[rect[0], rect[1]], [rect[2], rect[3]]],
        )
        return seg

    def _make_label(self, feet, cx, cy):
        return {"type": "length", "text": "test", "feet": feet,
                "cx": cx, "cy": cy, "bbox": [cx-5, cy-5, cx+5, cy+5]}

    def test_exact_match_gives_correct_scale(self):
        # 10 ft × 24.7 expected = 247 pt segment, label at (100, 100)
        seg = self._make_seg(247.0, [0, 90, 247, 110])
        lbl = self._make_label(10.0, 123.5, 100)
        result = _calibrate_scale([lbl], [seg], max_dist_pt=200)
        assert result == pytest.approx(24.7, rel=0.01)

    def test_no_match_returns_seed(self):
        seg = self._make_seg(100.0, [0, 0, 100, 20])    # 100pt ÷ 10ft = 10 pt/ft → outside range
        lbl = self._make_label(10.0, 50, 10)
        result = _calibrate_scale([lbl], [seg])
        from config.settings import PT_PER_FT_EXPECTED
        assert result == PT_PER_FT_EXPECTED

    def test_spatial_gate_rejects_distant_match(self):
        # Same length match but label is 500 pt away from segment
        seg = self._make_seg(247.0, [0, 90, 247, 110])
        lbl = self._make_label(10.0, 900, 100)          # 500+ pt away
        result = _calibrate_scale([lbl], [seg], max_dist_pt=200)
        from config.settings import PT_PER_FT_EXPECTED
        assert result == PT_PER_FT_EXPECTED

    def test_median_of_multiple_pairs(self):
        # Two segments matching two labels → median of two pt_per_ft values
        seg1 = self._make_seg(247.0, [0, 90, 247, 110])   # 10 ft → 24.7
        seg2 = self._make_seg(494.0, [0, 190, 494, 210])  # 20 ft → 24.7
        lbl1 = self._make_label(10.0, 123.5, 100)
        lbl2 = self._make_label(20.0, 247.0, 200)
        result = _calibrate_scale([lbl1, lbl2], [seg1, seg2], max_dist_pt=200)
        assert result == pytest.approx(24.7, rel=0.01)


# ── Integration: against actual PDF ──────────────────────────────────────────

@pytest.mark.skipif(
    not SAMPLE_INPUT.exists(),
    reason="Sample PDF not available",
)
class TestExtractLabelsIntegration:
    @pytest.fixture(scope="class")
    def labels(self):
        return extract_labels(str(SAMPLE_INPUT))

    @pytest.fixture(scope="class")
    def phase2_result(self):
        segs = extract_ducts(str(SAMPLE_INPUT))
        return extract_labels_with_scale(str(SAMPLE_INPUT), segs)

    def test_minimum_length_label_count(self, labels):
        length = [l for l in labels if l["type"] == "length"]
        assert len(length) >= 5, f"Expected >= 5 length labels, got {len(length)}"

    def test_all_seven_length_labels_present(self, labels):
        # input1.pdf has 7 unique length label positions
        length = [l for l in labels if l["type"] == "length"]
        assert len(length) >= 7, f"Expected 7 length labels, got {len(length)}"

    def test_cross_section_label_found(self, labels):
        cross = [l for l in labels if l["type"] == "cross_section"]
        assert len(cross) >= 1, "Expected at least 1 cross-section label"

    def test_24x18_cross_section_parsed(self, labels):
        rects = [l for l in labels if l.get("type") == "cross_section"
                 and l.get("width_in") == 24 and l.get("height_in") == 18]
        assert len(rects) >= 1, "Expected to find 24×18 cross-section label"

    def test_duct_ids_c01_c02_c03_present(self, labels):
        ids = {l["text"] for l in labels if l["type"] == "duct_id"}
        for expected in ("C01", "C02", "C03"):
            assert expected in ids, f"{expected} not found in duct IDs: {ids}"

    def test_pt_per_ft_in_valid_range(self, phase2_result):
        pt = phase2_result["pt_per_ft"]
        assert PT_PER_FT_MIN <= pt <= PT_PER_FT_MAX, (
            f"pt_per_ft={pt} outside [{PT_PER_FT_MIN}, {PT_PER_FT_MAX}]"
        )

    def test_18ft5_label_associates_to_duct(self, phase2_result):
        length_labels = [l for l in phase2_result["labels"] if l["type"] == "length"]
        long_label = next(
            (l for l in length_labels if abs(l["feet"] - (18 + 5/12)) < 0.1),
            None,
        )
        assert long_label is not None, "18'-5\" label not found"
        assert long_label["nearest_duct_id"] is not None, (
            f"18'-5\" did not associate to any duct (dist={long_label['nearest_dist_pt']} pt)"
        )

    def test_no_duplicate_labels(self, labels):
        keys = [(l["text"], round(l["cx"]), round(l["cy"])) for l in labels]
        assert len(keys) == len(set(keys)), "Duplicate labels after deduplication"

    def test_all_labels_outside_title_block(self, labels):
        from config.settings import TITLE_BLOCK_Y_MIN_PT
        violations = [l for l in labels if l["cy"] > TITLE_BLOCK_Y_MIN_PT]
        assert len(violations) == 0, f"Labels in title block: {violations}"

    def test_result_schema(self, phase2_result):
        assert "pt_per_ft" in phase2_result
        assert "labels" in phase2_result
        assert "label_count" in phase2_result
        assert phase2_result["scale_text"] == "derived"
        for lbl in phase2_result["labels"]:
            assert "type" in lbl
            assert "nearest_duct_id" in lbl
            assert "nearest_dist_pt" in lbl
