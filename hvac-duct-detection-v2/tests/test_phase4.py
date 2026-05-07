"""
Phase 4 tests — Duct Annotator

Run with:
    cd hvac-duct-detection-v2
    python -m pytest tests/test_phase4.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from config.settings import (
    SAMPLE_INPUT,
    LABEL_MISMATCH_THRESHOLD,
)
from tools.duct_annotator import (
    _perp_along_dist,
    _assign_length_labels,
    _assign_nearest_labels,
    annotate_ducts,
)
from tools.vector_duct_extractor import extract_ducts
from tools.label_extractor import extract_labels_with_scale
from models.duct_segment import DuctSegment
from models.annotated_duct import AnnotatedDuct


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_seg(seg_id, rect, orientation):
    x0, y0, x1, y1 = rect
    long_pt = max(x1 - x0, y1 - y0)
    short_pt = min(x1 - x0, y1 - y0)
    if orientation == "H":
        cl = [[x0, (y0 + y1) / 2], [x1, (y0 + y1) / 2]]
    else:
        cl = [[(x0 + x1) / 2, y0], [(x0 + x1) / 2, y1]]
    return DuctSegment(
        id=seg_id, rect=list(rect), orientation=orientation,
        long_pt=long_pt, short_pt=short_pt, aspect=long_pt / short_pt,
        centerline=cl,
    )


def _make_length_label(feet, cx, cy):
    return {"type": "length", "text": f"{int(feet)}'", "feet": feet,
            "cx": cx, "cy": cy, "bbox": [cx - 5, cy - 5, cx + 5, cy + 5]}


def _make_cs_label(width_in, height_in, cx, cy):
    return {"type": "cross_section", "text": f"{width_in}X{height_in}",
            "cross_type": "rect", "width_in": width_in, "height_in": height_in,
            "cx": cx, "cy": cy, "bbox": [cx - 5, cy - 5, cx + 5, cy + 5]}


def _make_id_label(text, cx, cy):
    return {"type": "duct_id", "text": text,
            "cx": cx, "cy": cy, "bbox": [cx - 5, cy - 5, cx + 5, cy + 5]}


# ── Unit: _perp_along_dist ────────────────────────────────────────────────────

class TestPerpAlongDist:
    def test_h_duct_above_center_no_along(self):
        seg = _make_seg("s1", [0, 90, 200, 110], "H")
        perp, along = _perp_along_dist(100, 50, seg)
        assert perp == pytest.approx(50.0)
        assert along == pytest.approx(0.0)  # label x=100 inside [0,200]

    def test_h_duct_label_past_right_endpoint(self):
        seg = _make_seg("s1", [0, 90, 200, 110], "H")
        perp, along = _perp_along_dist(250, 100, seg)
        assert perp == pytest.approx(0.0)   # label y=100 = centerline y=100
        assert along == pytest.approx(50.0) # 250 - 200 = 50 past endpoint

    def test_v_duct_beside_center_no_along(self):
        seg = _make_seg("s1", [90, 0, 110, 300], "V")
        perp, along = _perp_along_dist(150, 150, seg)
        assert perp == pytest.approx(50.0)  # |150-100|=50
        assert along == pytest.approx(0.0)  # label y=150 inside [0,300]

    def test_v_duct_label_above_top_endpoint(self):
        seg = _make_seg("s1", [90, 100, 110, 400], "V")
        perp, along = _perp_along_dist(100, 50, seg)
        assert perp == pytest.approx(0.0)
        assert along == pytest.approx(50.0)  # 100 - 50 = 50 past top

    def test_diagonal_duct_uses_centroid(self):
        seg = DuctSegment(
            id="d1", rect=[0, 0, 100, 100], orientation="D",
            long_pt=100, short_pt=100, aspect=1,
            centerline=[[0, 0], [100, 100]],
        )
        perp, along = _perp_along_dist(200, 200, seg)
        # centroid = (50, 50); dist = sqrt((200-50)^2+(200-50)^2) = 150√2
        assert perp == pytest.approx(150 * 2 ** 0.5, rel=0.01)
        assert along == pytest.approx(0.0)


# ── Unit: _assign_length_labels ──────────────────────────────────────────────

class TestAssignLengthLabels:
    def test_label_assigned_to_nearest_duct(self):
        # long_pt=200 ≈ 8.1ft at 24.7 pt/ft — passes plausibility for 8ft label
        seg_a = _make_seg("a", [0, 90, 200, 110], "H")   # cy=100
        seg_b = _make_seg("b", [0, 190, 200, 210], "H")  # cy=200
        lbl = _make_length_label(8.0, 100, 80)  # 20pt above seg_a, 120pt above seg_b
        result = _assign_length_labels([seg_a, seg_b], [lbl], pt_per_ft=24.7)
        assert "a" in result
        assert "b" not in result

    def test_plausibility_gate_rejects_wrong_length(self):
        # long_pt=300 ≈ 12.1ft, but label says 8ft → expected=197.6pt → 51.8% error > 30%
        seg = _make_seg("a", [0, 90, 300, 110], "H")
        lbl = _make_length_label(8.0, 150, 80)
        result = _assign_length_labels([seg], [lbl], pt_per_ft=24.7)
        assert result == {}

    def test_label_consumed_once(self):
        seg_a = _make_seg("a", [0, 90, 200, 110], "H")
        seg_b = _make_seg("b", [0, 95, 200, 115], "H")  # very close to a
        lbl = _make_length_label(8.0, 100, 70)  # equidistant-ish from both
        result = _assign_length_labels([seg_a, seg_b], [lbl], pt_per_ft=24.7)
        # Label assigned to exactly one duct
        assert len(result) == 1

    def test_multiple_labels_multiple_ducts(self):
        # seg_a: long_pt=200 ≈ 8.1ft; seg_b: long_pt=296 ≈ 12.0ft
        # Cross-plausibility is rejected: 8ft label won't match 296pt duct (49.7% error)
        seg_a = _make_seg("a", [0, 90, 200, 110], "H")    # cy=100
        seg_b = _make_seg("b", [0, 290, 296, 310], "H")   # cy=300, long_pt=296
        lbl_a = _make_length_label(8.0, 100, 80)    # near seg_a
        lbl_b = _make_length_label(12.0, 148, 280)  # near seg_b
        result = _assign_length_labels([seg_a, seg_b], [lbl_a, lbl_b], pt_per_ft=24.7)
        assert "a" in result and result["a"]["feet"] == pytest.approx(8.0)
        assert "b" in result and result["b"]["feet"] == pytest.approx(12.0)


# ── Unit: annotate_ducts ──────────────────────────────────────────────────────

class TestAnnotateDucts:
    def test_measured_length_correct(self):
        seg = _make_seg("a", [0, 90, 247, 110], "H")  # long_pt=247
        result = annotate_ducts([seg], [], pt_per_ft=24.7)
        assert result[0].length_ft_measured == pytest.approx(10.0, rel=0.01)

    def test_no_label_gives_none_and_unlabeled(self):
        seg = _make_seg("a", [0, 90, 247, 110], "H")
        result = annotate_ducts([seg], [], pt_per_ft=24.7)
        d = result[0]
        assert d.length_ft_label is None
        assert d.length_mismatch is False
        assert d.unlabeled is True

    def test_matched_label_no_mismatch(self):
        seg = _make_seg("a", [0, 90, 247, 110], "H")  # long_pt=247 → 10.0 ft at 24.7
        lbl = _make_length_label(10.0, 123.5, 70)  # 30pt above centerline
        result = annotate_ducts([seg], [lbl], pt_per_ft=24.7)
        d = result[0]
        assert d.length_ft_label == pytest.approx(10.0)
        assert d.length_mismatch is False
        assert d.unlabeled is False

    def test_mismatch_flag_set(self):
        # Duct is 300pt = 12.15ft; label says 10.0ft (expected=247pt).
        # Plausibility: |300-247|/247 = 21.5% ≤ 30% → label is assigned.
        # Mismatch: |12.15-10|/10 = 21.5% > 15% → flagged.
        seg = _make_seg("a", [0, 90, 300, 110], "H")
        lbl = _make_length_label(10.0, 150, 70)
        result = annotate_ducts([seg], [lbl], pt_per_ft=24.7)
        assert result[0].length_mismatch is True

    def test_cross_section_rect_assigned(self):
        seg = _make_seg("a", [0, 90, 247, 110], "H")
        cs = _make_cs_label(24, 18, 123, 100)  # centroid at (123, 100)
        result = annotate_ducts([seg], [cs], pt_per_ft=24.7)
        d = result[0]
        assert d.cross_section == {"width_in": 24, "height_in": 18}
        assert not d.is_round
        assert not d.unlabeled  # has cross_section so not unlabeled

    def test_duct_id_assigned(self):
        seg = _make_seg("a", [0, 90, 247, 110], "H")
        id_lbl = _make_id_label("C03", 123, 100)
        result = annotate_ducts([seg], [id_lbl], pt_per_ft=24.7)
        assert result[0].duct_label_id == "C03"

    def test_to_dict_schema(self):
        seg = _make_seg("a", [0, 90, 247, 110], "H")
        d = annotate_ducts([seg], [], pt_per_ft=24.7)[0]
        dd = d.to_dict()
        assert "duct_idx" in dd
        assert "length_ft_measured" in dd
        assert "length_mismatch" in dd
        assert "cross_section" in dd
        assert "unlabeled" in dd


# ── Integration: against actual PDF ──────────────────────────────────────────

@pytest.mark.skipif(
    not SAMPLE_INPUT.exists(),
    reason="Sample PDF not available",
)
class TestAnnotateDuctsIntegration:
    @pytest.fixture(scope="class")
    def pipeline(self):
        segs = extract_ducts(str(SAMPLE_INPUT))
        p2 = extract_labels_with_scale(str(SAMPLE_INPUT), segs)
        annotated = annotate_ducts(segs, p2["labels"], p2["pt_per_ft"])
        return {"segs": segs, "p2": p2, "annotated": annotated}

    def test_one_annotated_per_segment(self, pipeline):
        assert len(pipeline["annotated"]) == len(pipeline["segs"])

    def test_at_least_one_length_label_assigned(self, pipeline):
        # Plausibility filter means only labels that match a single Phase 1 segment
        # get through; compound-run labels (e.g. "C02 9'-0\"") are left unassigned.
        with_len = [d for d in pipeline["annotated"] if d.length_ft_label is not None]
        assert len(with_len) >= 1, "Expected at least 1 length-labelled duct"

    def test_18ft5_label_assigned_to_longest_duct(self, pipeline):
        # The 18'-5" C03 label is the only label that clearly maps to a single
        # Phase 1 segment (duct_003, 449.8 pt ≈ 18.56 ft).
        longest = max(pipeline["segs"], key=lambda s: s.long_pt)
        annotated_map = {d.segment_id: d for d in pipeline["annotated"]}
        d = annotated_map[longest.id]
        assert d.length_ft_label is not None, (
            f"Longest duct {longest.id} has no length label (expected 18'-5\")"
        )
        assert 17.5 <= d.length_ft_label <= 20.0, (
            f"Unexpected label value on longest duct: {d.length_ft_label}"
        )

    def test_18ft5_no_mismatch(self, pipeline):
        # The 18'-5" label matches duct_003 within 15% — should not be flagged.
        longest = max(pipeline["segs"], key=lambda s: s.long_pt)
        annotated_map = {d.segment_id: d for d in pipeline["annotated"]}
        d = annotated_map[longest.id]
        if d.length_ft_label is not None:
            assert not d.length_mismatch, (
                f"Unexpected mismatch on longest duct: "
                f"measured={d.length_ft_measured:.3f} label={d.length_ft_label:.3f}"
            )

    def test_cross_section_assigned(self, pipeline):
        with_cs = [d for d in pipeline["annotated"] if d.cross_section is not None]
        assert len(with_cs) >= 1, "Expected at least one cross-section label assigned"

    def test_24x18_cross_section_on_one_duct(self, pipeline):
        rects_24x18 = [
            d for d in pipeline["annotated"]
            if d.cross_section
            and d.cross_section.get("width_in") == 24
            and d.cross_section.get("height_in") == 18
        ]
        assert len(rects_24x18) == 1, (
            f"Expected exactly 1 duct with 24×18; got {len(rects_24x18)}"
        )

    def test_unlabeled_count_reasonable(self, pipeline):
        unlabeled = [d for d in pipeline["annotated"] if d.unlabeled]
        total = len(pipeline["annotated"])
        # Plausibility gate leaves only a few clean matches; guarantee ≥2 labeled
        # (at minimum: 1 length label on duct_003 + 1 cross-section on another duct)
        assert len(unlabeled) <= total - 2, (
            f"Too many unlabeled: {len(unlabeled)} out of {total}"
        )

    def test_all_annotated_ducts_have_measured_length(self, pipeline):
        for d in pipeline["annotated"]:
            assert d.length_ft_measured > 0

    def test_no_two_ducts_share_length_label(self, pipeline):
        labels_used = [d.length_ft_label for d in pipeline["annotated"]
                       if d.length_ft_label is not None]
        # Round to 3 decimal to compare (floats from ft/in arithmetic)
        rounded = [round(v, 3) for v in labels_used]
        assert len(rounded) == len(set(rounded)), (
            f"Duplicate length label assignments: {sorted(rounded)}"
        )

    def test_to_dict_has_required_fields(self, pipeline):
        required = {"id", "duct_idx", "rect", "orientation", "length_ft_measured",
                    "length_ft_label", "length_mismatch", "cross_section",
                    "is_round", "unlabeled", "confidence"}
        for d in pipeline["annotated"]:
            dd = d.to_dict()
            missing = required - dd.keys()
            assert not missing, f"{d.segment_id} missing keys: {missing}"
