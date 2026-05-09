"""
Phase 2 — Text & Scale Calibration

Extracts dimension labels (length, cross-section, duct IDs) from the PDF
native text layer. Derives calibrated pt_per_ft from label+segment pairs.

Key facts verified on input1.pdf:
- page.get_text("dict") returns bboxes in un-rotated MEDIA coordinates —
  same coordinate system as page.get_drawings(). No transform needed.
- Length labels stored as complete spans: "12' - 6\"" — no multi-span merge needed
  in this PDF, but the merge step is kept for robustness.
- Duplicate span rendering: some labels appear 2–3× at identical positions →
  deduplicated by (text, rounded_cx, rounded_cy).
- Round-duct diameter labels (12"Ø, 8"Ø) are NOT in the text layer of input1.pdf —
  they appear to be rendered as vector paths, not extractable text.
"""

import json
import math
import re
from pathlib import Path

import fitz

from config.settings import (
    OUTPUTS_DIR,
    PT_PER_FT_EXPECTED,
    PT_PER_FT_MAX,
    PT_PER_FT_MIN,
    SAMPLE_INPUT,
    SCALE_MATCH_TOLERANCE,
    TITLE_BLOCK_Y_MIN_PT,
)
from models.duct_segment import DuctSegment


# ── Regex patterns ─────────────────────────────────────────────────────────────

# Length: "12' - 6\"" or "10' - 0\"" (foot-inch format)
_RE_LENGTH = re.compile(r"(\d+)'\s*-\s*(\d+)\"")

# Cross-section rectangular: "24X18" or "22x14"  (digits, optional space, X/x/×, digits)
_RE_CROSS_RECT = re.compile(r"(\d+)\s*[Xx×]\s*(\d+)")

# Cross-section round: "18\"Ø" or "8Ø" or "12ø"  (not present in input1.pdf text layer)
_RE_CROSS_ROUND = re.compile(r'(\d+)["\']?\s*[Øø⌀]', re.UNICODE)

# Duct ID: 1–3 uppercase letters + 1–3 digits (C01, C02, SA12, RA3 …)
_RE_DUCT_ID = re.compile(r"^[A-Z]{1,3}\d{1,3}$")


# ── Text span extraction ──────────────────────────────────────────────────────

def _get_spans(page: fitz.Page) -> list[dict]:
    """Return all non-empty text spans from the drawing area (title block excluded)."""
    spans = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span["text"].strip()
                if not text:
                    continue
                bbox = span["bbox"]
                if bbox[1] > TITLE_BLOCK_Y_MIN_PT or bbox[3] > TITLE_BLOCK_Y_MIN_PT:
                    continue
                spans.append({
                    "text": text,
                    "bbox": list(bbox),
                    "cx": (bbox[0] + bbox[2]) / 2.0,
                    "cy": (bbox[1] + bbox[3]) / 2.0,
                })
    return spans


def _merge_adjacent_spans(spans: list[dict], gap_pt: float = 20.0) -> list[dict]:
    """
    Merge spans on the same text line that are within gap_pt of each other.
    Handles cases where a label like "18' - 5\"" is split across spans in
    some PDF authoring tools.
    """
    if not spans:
        return spans

    sorted_spans = sorted(spans, key=lambda s: (round(s["cy"] / 5) * 5, s["cx"]))
    merged: list[dict] = []
    used = [False] * len(sorted_spans)

    for i, s in enumerate(sorted_spans):
        if used[i]:
            continue
        group = [s]
        used[i] = True
        for j in range(i + 1, len(sorted_spans)):
            if used[j]:
                continue
            t = sorted_spans[j]
            if abs(t["cy"] - s["cy"]) > 8.0:
                break
            gap = t["bbox"][0] - group[-1]["bbox"][2]
            if gap > gap_pt:
                break
            group.append(t)
            used[j] = True

        if len(group) == 1:
            merged.append(s)
        else:
            joined = " ".join(g["text"] for g in group)
            x0 = group[0]["bbox"][0]
            y0 = min(g["bbox"][1] for g in group)
            x1 = group[-1]["bbox"][2]
            y1 = max(g["bbox"][3] for g in group)
            merged.append({
                "text": joined,
                "bbox": [x0, y0, x1, y1],
                "cx": (x0 + x1) / 2.0,
                "cy": (y0 + y1) / 2.0,
            })

    return merged


# ── Label classification ──────────────────────────────────────────────────────

def _parse_length(text: str) -> float | None:
    """Parse "N' - M\"" → total feet as float. Returns None if no match."""
    m = _RE_LENGTH.search(text)
    if not m:
        return None
    return int(m.group(1)) + int(m.group(2)) / 12.0


def _parse_cross_section(text: str) -> dict | None:
    """
    Parse cross-section dimension from text.
    Returns {"cross_type": "round", "diameter_in": N}
         or {"cross_type": "rect",  "width_in": W, "height_in": H}
         or None.

    Uses "cross_type" (not "type") so the caller can safely spread via **cs
    without colliding with the outer label's "type" key.

    For strings like "SC-24X18X8.62BOX", takes the FIRST regex match ("24X18")
    since duct face dimensions precede box depth in standard HVAC notation.
    Sanity-checks dimensions in the range 2–60 inches.
    """
    mr = _RE_CROSS_ROUND.search(text)
    if mr:
        return {"cross_type": "round", "diameter_in": int(mr.group(1))}

    mx = _RE_CROSS_RECT.search(text)
    if mx:
        w, h = int(mx.group(1)), int(mx.group(2))
        if 2 <= w <= 60 and 2 <= h <= 60:
            return {"cross_type": "rect", "width_in": w, "height_in": h}

    return None


def _classify_span(span: dict) -> dict | None:
    """Return a typed label dict if span matches a known pattern, else None."""
    text = span["text"]

    feet = _parse_length(text)
    if feet is not None:
        return {
            "text": text, "type": "length",
            "feet": round(feet, 4),
            "bbox": span["bbox"], "cx": span["cx"], "cy": span["cy"],
        }

    cs = _parse_cross_section(text)
    if cs is not None:
        return {
            "text": text, "type": "cross_section",
            **cs,
            "bbox": span["bbox"], "cx": span["cx"], "cy": span["cy"],
        }

    if _RE_DUCT_ID.match(text):
        return {
            "text": text, "type": "duct_id",
            "bbox": span["bbox"], "cx": span["cx"], "cy": span["cy"],
        }

    return None


# ── Scale calibration ─────────────────────────────────────────────────────────

def _calibrate_scale(
    length_labels: list[dict],
    segments: list[DuctSegment],
    max_dist_pt: float = 200.0,
) -> float:
    """
    Derive pt_per_ft from length-label / duct-segment pairs.

    For each length label:
    1. Find all segments whose long_pt is within SCALE_MATCH_TOLERANCE of
       (label_feet * PT_PER_FT_EXPECTED) — a length-plausibility gate.
    2. Keep only those within max_dist_pt of the label (spatial gate).
    3. Compute pt_per_ft = segment.long_pt / label_feet for each surviving pair.

    Returns the median of all valid pt_per_ft values.
    Falls back to PT_PER_FT_EXPECTED if no pairs survive both gates.
    """
    seed = PT_PER_FT_EXPECTED
    tol = SCALE_MATCH_TOLERANCE
    pairs: list[float] = []

    for lbl in length_labels:
        feet = lbl["feet"]
        if feet <= 0:
            continue
        expected_pt = feet * seed
        lx, ly = lbl["cx"], lbl["cy"]

        for seg in segments:
            if abs(seg.long_pt - expected_pt) / expected_pt > tol:
                continue
            cx = (seg.rect[0] + seg.rect[2]) / 2.0
            cy = (seg.rect[1] + seg.rect[3]) / 2.0
            if math.hypot(lx - cx, ly - cy) > max_dist_pt:
                continue
            pt_per_ft = seg.long_pt / feet
            if PT_PER_FT_MIN <= pt_per_ft <= PT_PER_FT_MAX:
                pairs.append(pt_per_ft)

    if not pairs:
        return PT_PER_FT_EXPECTED

    pairs.sort()
    mid = len(pairs) // 2
    return pairs[mid] if len(pairs) % 2 else (pairs[mid - 1] + pairs[mid]) / 2.0


# ── Label-duct association ────────────────────────────────────────────────────

def _associate_labels(
    labels: list[dict],
    segments: list[DuctSegment],
    max_dist_pt: float = 150.0,
) -> list[dict]:
    """
    Attach nearest_duct_id to each label.
    Sets nearest_duct_id = None when the closest duct is beyond max_dist_pt.
    """
    centroids = [
        (seg.id,
         (seg.rect[0] + seg.rect[2]) / 2.0,
         (seg.rect[1] + seg.rect[3]) / 2.0)
        for seg in segments
    ]

    result: list[dict] = []
    for lbl in labels:
        lx, ly = lbl["cx"], lbl["cy"]
        best_id, best_dist = None, float("inf")
        for sid, sx, sy in centroids:
            d = math.hypot(lx - sx, ly - sy)
            if d < best_dist:
                best_dist, best_id = d, sid
        result.append({
            **lbl,
            "nearest_duct_id": best_id if best_dist <= max_dist_pt else None,
            "nearest_dist_pt": round(best_dist, 1),
        })
    return result


# ── Public API ────────────────────────────────────────────────────────────────

def extract_labels(pdf_path: str, page_index: int = 0) -> list[dict]:
    """
    Extract all parsed labels from PDF text.
    Deduplicates identical labels rendered at the same position.
    Returns a list of label dicts (type = "length" | "cross_section" | "duct_id").

    Two-pass strategy:
    - Pass 1 (raw spans): captures duct IDs (C01, C02 …) before they are
      absorbed into adjacent length-label spans during merging.
    - Pass 2 (merged spans): captures length labels and cross-section labels.
    """
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    raw_spans = _get_spans(page)
    doc.close()

    seen: set[tuple] = set()
    labels: list[dict] = []

    # Pass 1 — raw spans: duct IDs only
    for span in raw_spans:
        parsed = _classify_span(span)
        if parsed is None or parsed["type"] != "duct_id":
            continue
        key = (parsed["text"], round(parsed["cx"]), round(parsed["cy"]))
        if key in seen:
            continue
        seen.add(key)
        labels.append(parsed)

    # Pass 2 — merged spans: length + cross_section
    merged = _merge_adjacent_spans(raw_spans)
    for span in merged:
        parsed = _classify_span(span)
        if parsed is None or parsed["type"] == "duct_id":
            continue
        key = (parsed["text"], round(parsed["cx"]), round(parsed["cy"]))
        if key in seen:
            continue
        seen.add(key)
        labels.append(parsed)

    return labels


def extract_labels_with_scale(
    pdf_path: str,
    segments: list[DuctSegment],
    page_index: int = 0,
    output_path: str | None = None,
) -> dict:
    """
    Full Phase 2 pipeline: extract labels, calibrate scale, associate to ducts.

    Returns a dict with:
      pt_per_ft        — calibrated points-per-foot
      scale_text       — "derived" (no explicit scale bar in input1.pdf)
      label_count      — total distinct labels parsed
      labels           — list of label dicts, each with nearest_duct_id attached

    Optionally writes JSON to output_path.
    """
    labels = extract_labels(pdf_path, page_index)
    length_labels = [l for l in labels if l["type"] == "length"]

    pt_per_ft = _calibrate_scale(length_labels, segments)
    associated = _associate_labels(labels, segments)

    result = {
        "pdf": str(pdf_path),
        "page": page_index,
        "pt_per_ft": round(pt_per_ft, 4),
        "scale_text": "derived",
        "label_count": len(labels),
        "labels": associated,
    }

    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))

    return result
