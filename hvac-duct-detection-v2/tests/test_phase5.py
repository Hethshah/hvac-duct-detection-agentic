"""
Phase 5 tests — Vision validator

Run with:
    cd hvac-duct-detection-v2
    python -m pytest tests/test_phase5.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from config.settings import SAMPLE_INPUT, VISION_CONFIDENCE_THRESHOLD
from models.annotated_duct import AnnotatedDuct
from tools.vision_validator import _needs_vision, _render_crop, validate_ducts


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
):
    return AnnotatedDuct(
        segment_id=segment_id,
        duct_label_id=None,
        rect=rect or [100.0, 100.0, 347.0, 120.0],
        orientation=orientation,
        length_ft_measured=length_ft_measured,
        length_ft_label=length_ft_label,
        length_mismatch=length_mismatch,
        cross_section=None,
        is_round=False,
        unlabeled=True,
        confidence=confidence,
        source="vector",
        page=page,
    )


# ── Unit: _needs_vision ────────────────────────────────────────────────────────

class TestNeedsVision:
    def test_high_confidence_no_mismatch_skipped(self):
        assert _needs_vision(_make_duct(confidence=1.0, length_mismatch=False)) is False

    def test_low_confidence_needs_vision(self):
        assert _needs_vision(_make_duct(confidence=VISION_CONFIDENCE_THRESHOLD - 0.01)) is True

    def test_at_threshold_skipped(self):
        assert _needs_vision(_make_duct(confidence=VISION_CONFIDENCE_THRESHOLD)) is False

    def test_mismatch_triggers_vision_regardless_of_confidence(self):
        assert _needs_vision(_make_duct(confidence=1.0, length_mismatch=True)) is True

    def test_both_triggers(self):
        assert _needs_vision(_make_duct(confidence=0.3, length_mismatch=True)) is True


# ── Unit: validate_ducts (mocked API) ─────────────────────────────────────────

class TestValidateDucts:
    def test_no_candidates_skips_api(self):
        ducts = [_make_duct(confidence=1.0, length_mismatch=False)]
        with patch("tools.vision_validator.anthropic.Anthropic") as mock_cls:
            updated, log = validate_ducts("dummy.pdf", ducts)
        mock_cls.assert_not_called()
        assert log == []
        assert updated[0].confidence == 1.0

    def test_low_confidence_triggers_api_call(self):
        duct = _make_duct(confidence=0.5)
        mock_resp = MagicMock()
        mock_resp.content[0].text = '{"is_duct": true, "confidence": 0.9, "notes": "clear duct"}'
        with patch("tools.vision_validator._render_crop", return_value=b"fake"), \
             patch("tools.vision_validator.anthropic.Anthropic") as mock_cls:
            mock_cls.return_value.messages.create.return_value = mock_resp
            _, log = validate_ducts("dummy.pdf", [duct])
        assert len(log) == 1
        assert log[0]["segment_id"] == "duct_001"

    def test_is_duct_true_raises_confidence(self):
        duct = _make_duct(confidence=0.5)
        mock_resp = MagicMock()
        mock_resp.content[0].text = '{"is_duct": true, "confidence": 0.95, "notes": "ok"}'
        with patch("tools.vision_validator._render_crop", return_value=b"fake"), \
             patch("tools.vision_validator.anthropic.Anthropic") as mock_cls:
            mock_cls.return_value.messages.create.return_value = mock_resp
            updated, _ = validate_ducts("dummy.pdf", [duct])
        assert updated[0].confidence == pytest.approx(0.95)

    def test_is_duct_false_lowers_confidence(self):
        duct = _make_duct(confidence=0.5)
        mock_resp = MagicMock()
        mock_resp.content[0].text = '{"is_duct": false, "confidence": 0.9, "notes": "not a duct"}'
        with patch("tools.vision_validator._render_crop", return_value=b"fake"), \
             patch("tools.vision_validator.anthropic.Anthropic") as mock_cls:
            mock_cls.return_value.messages.create.return_value = mock_resp
            updated, _ = validate_ducts("dummy.pdf", [duct])
        assert updated[0].confidence == pytest.approx(0.1)  # 1.0 - 0.9

    def test_mismatch_duct_is_validated(self):
        duct = _make_duct(confidence=1.0, length_mismatch=True,
                          length_ft_measured=10.0, length_ft_label=8.0)
        mock_resp = MagicMock()
        mock_resp.content[0].text = '{"is_duct": true, "confidence": 0.85, "notes": "confirmed"}'
        with patch("tools.vision_validator._render_crop", return_value=b"fake"), \
             patch("tools.vision_validator.anthropic.Anthropic") as mock_cls:
            mock_cls.return_value.messages.create.return_value = mock_resp
            _, log = validate_ducts("dummy.pdf", [duct])
        assert log[0]["length_mismatch"] is True

    def test_log_has_required_fields(self):
        duct = _make_duct(confidence=0.4)
        mock_resp = MagicMock()
        mock_resp.content[0].text = '{"is_duct": true, "confidence": 0.8, "notes": "ok"}'
        with patch("tools.vision_validator._render_crop", return_value=b"fake"), \
             patch("tools.vision_validator.anthropic.Anthropic") as mock_cls:
            mock_cls.return_value.messages.create.return_value = mock_resp
            _, log = validate_ducts("dummy.pdf", [duct])
        required = {"segment_id", "input_confidence", "length_mismatch",
                    "is_duct", "confidence", "notes"}
        assert required <= set(log[0].keys())

    def test_multiple_candidates_all_validated_skipped_duct_unchanged(self):
        ducts = [
            _make_duct("d1", confidence=0.4),
            _make_duct("d2", confidence=1.0),  # should be skipped
            _make_duct("d3", confidence=0.3),
        ]
        mock_resp = MagicMock()
        mock_resp.content[0].text = '{"is_duct": true, "confidence": 0.9, "notes": "ok"}'
        with patch("tools.vision_validator._render_crop", return_value=b"fake"), \
             patch("tools.vision_validator.anthropic.Anthropic") as mock_cls:
            mock_cls.return_value.messages.create.return_value = mock_resp
            updated, log = validate_ducts("dummy.pdf", ducts)
        assert len(log) == 2
        ids = {e["segment_id"] for e in log}
        assert "d1" in ids and "d3" in ids and "d2" not in ids
        # d2 confidence unchanged
        assert next(d for d in updated if d.segment_id == "d2").confidence == 1.0

    def test_non_duct_confidence_not_below_zero(self):
        duct = _make_duct(confidence=0.3)
        mock_resp = MagicMock()
        mock_resp.content[0].text = '{"is_duct": false, "confidence": 1.0, "notes": "definitely not"}'
        with patch("tools.vision_validator._render_crop", return_value=b"fake"), \
             patch("tools.vision_validator.anthropic.Anthropic") as mock_cls:
            mock_cls.return_value.messages.create.return_value = mock_resp
            updated, _ = validate_ducts("dummy.pdf", [duct])
        assert updated[0].confidence >= 0.0


# ── Integration: against actual PDF ──────────────────────────────────────────

@pytest.mark.skipif(
    not SAMPLE_INPUT.exists(),
    reason="Sample PDF not available",
)
class TestVisionValidationIntegration:
    @pytest.fixture(scope="class")
    def annotated(self):
        from tools.vector_duct_extractor import extract_ducts
        from tools.label_extractor import extract_labels_with_scale
        from tools.duct_annotator import annotate_ducts
        segs = extract_ducts(str(SAMPLE_INPUT))
        p2   = extract_labels_with_scale(str(SAMPLE_INPUT), segs)
        return annotate_ducts(segs, p2["labels"], p2["pt_per_ft"])

    def test_candidates_are_only_mismatches_not_low_confidence(self, annotated):
        # input1.pdf is all-vector (confidence=1.0); any candidates are length mismatches only
        candidates = [d for d in annotated if _needs_vision(d)]
        low_conf = [d for d in candidates if d.confidence < VISION_CONFIDENCE_THRESHOLD]
        assert len(low_conf) == 0, (
            f"Expected no low-confidence candidates for all-vector PDF; got {low_conf}"
        )
        for d in candidates:
            assert d.length_mismatch is True

    def test_render_crop_returns_valid_png(self, annotated):
        png = _render_crop(str(SAMPLE_INPUT), annotated[0])
        assert png[:8] == b'\x89PNG\r\n\x1a\n', "Expected PNG magic bytes"
        assert len(png) > 500

    def test_validate_ducts_calls_api_once_per_candidate(self, annotated):
        candidates = [d for d in annotated if _needs_vision(d)]
        mock_resp = MagicMock()
        mock_resp.content[0].text = '{"is_duct": true, "confidence": 0.9, "notes": "confirmed"}'
        with patch("tools.vision_validator._render_crop", return_value=b"fake_png"), \
             patch("tools.vision_validator.anthropic.Anthropic") as mock_cls:
            mock_cls.return_value.messages.create.return_value = mock_resp
            updated, log = validate_ducts(str(SAMPLE_INPUT), list(annotated))
        assert len(log) == len(candidates)
        assert len(updated) == len(annotated)
