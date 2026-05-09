"""
Phase 4 — Duct-Label Association & Measurement.

Binds each DuctSegment to its length label, cross-section label, and duct ID.
Computes real-world physical dimensions from calibrated pt_per_ft.

Association strategy:
  Length labels  — projection-based: perpendicular distance from label centre to
                   the duct's centreline axis must be < LABEL_PERP_MAX_PT.
  Cross-section  — centroid-to-centroid ≤ LABEL_CENTROID_MAX_PT, nearest wins.
  Duct IDs       — centroid-to-centroid ≤ LABEL_CENTROID_MAX_PT, nearest wins.

All three use greedy matching: candidates are sorted by distance and each label
is consumed at most once; each duct receives at most one label of each type.
"""

import json
import math
from pathlib import Path

from config.settings import (
    LABEL_LENGTH_MAX_PT,
    LABEL_LENGTH_PLAUSIBILITY,
    LABEL_CENTROID_MAX_PT,
    LABEL_MISMATCH_THRESHOLD,
)
from models.duct_segment import DuctSegment
from models.annotated_duct import AnnotatedDuct


# ── Projection helpers ────────────────────────────────────────────────────────

def _perp_along_dist(lx: float, ly: float, seg: DuctSegment) -> tuple[float, float]:
    """
    Return (perp_dist, along_dist) from label centre (lx, ly) to the duct axis.

    perp_dist  — distance perpendicular to the duct's long axis
    along_dist — distance along the axis beyond the segment endpoints (0 if inside)

    For diagonal ducts, falls back to centroid-to-centroid distance.
    """
    if seg.orientation == "H":
        x0, x1 = seg.rect[0], seg.rect[2]
        cy = (seg.rect[1] + seg.rect[3]) / 2
        perp = abs(ly - cy)
        along = max(0.0, x0 - lx, lx - x1)
    elif seg.orientation == "V":
        y0, y1 = seg.rect[1], seg.rect[3]
        cx = (seg.rect[0] + seg.rect[2]) / 2
        perp = abs(lx - cx)
        along = max(0.0, y0 - ly, ly - y1)
    else:  # D — diagonal
        rcx = (seg.rect[0] + seg.rect[2]) / 2
        rcy = (seg.rect[1] + seg.rect[3]) / 2
        perp = math.hypot(lx - rcx, ly - rcy)
        along = 0.0
    return perp, along


def _centroid_dist(lx: float, ly: float, seg: DuctSegment) -> float:
    rcx = (seg.rect[0] + seg.rect[2]) / 2
    rcy = (seg.rect[1] + seg.rect[3]) / 2
    return math.hypot(lx - rcx, ly - rcy)


# ── Greedy bipartite matching helpers ─────────────────────────────────────────

def _label_key(lbl: dict) -> tuple:
    return (lbl["text"], round(lbl["cx"]), round(lbl["cy"]))


def _greedy_assign(candidates: list[tuple]) -> dict[str, dict]:
    """
    Greedy bipartite match from a sorted list of (score, seg_id, label_dict).
    Each segment gets at most one label; each label is used at most once.
    Returns {seg_id: label_dict}.
    """
    assigned_segs: set[str] = set()
    assigned_labels: set[tuple] = set()
    result: dict[str, dict] = {}

    for _, seg_id, lbl in candidates:
        key = _label_key(lbl)
        if seg_id in assigned_segs or key in assigned_labels:
            continue
        result[seg_id] = lbl
        assigned_segs.add(seg_id)
        assigned_labels.add(key)

    return result


# ── Per-type assignment ────────────────────────────────────────────────────────

def _assign_length_labels(
    segments: list[DuctSegment],
    length_labels: list[dict],
    pt_per_ft: float,
    max_dist_pt: float = LABEL_LENGTH_MAX_PT,
    plausibility: float = LABEL_LENGTH_PLAUSIBILITY,
) -> dict[str, dict]:
    """
    Match length labels to segments using centroid distance + length plausibility.

    Length plausibility gate: |seg.long_pt - label_feet * pt_per_ft| / expected_pt
    must be <= plausibility (default 30%).  This prevents nearby but wrong-sized
    ducts from stealing labels that refer to composed duct runs.

    Returns {seg_id: label_dict}.
    """
    candidates: list[tuple] = []
    for lbl in length_labels:
        if lbl["feet"] <= 0:
            continue
        expected_pt = lbl["feet"] * pt_per_ft
        lx, ly = lbl["cx"], lbl["cy"]
        for seg in segments:
            if abs(seg.long_pt - expected_pt) / expected_pt > plausibility:
                continue
            d = _centroid_dist(lx, ly, seg)
            if d <= max_dist_pt:
                candidates.append((d, seg.id, lbl))

    candidates.sort(key=lambda t: t[0])
    return _greedy_assign(candidates)


def _assign_nearest_labels(
    segments: list[DuctSegment],
    label_list: list[dict],
    max_dist_pt: float = LABEL_CENTROID_MAX_PT,
) -> dict[str, dict]:
    """
    Match labels to segments by nearest centroid-to-centroid distance.
    Returns {seg_id: label_dict}.
    """
    candidates: list[tuple] = []
    for lbl in label_list:
        lx, ly = lbl["cx"], lbl["cy"]
        for seg in segments:
            d = _centroid_dist(lx, ly, seg)
            if d <= max_dist_pt:
                candidates.append((d, seg.id, lbl))

    candidates.sort(key=lambda t: t[0])
    return _greedy_assign(candidates)


# ── Public API ────────────────────────────────────────────────────────────────

def annotate_ducts(
    segments: list[DuctSegment],
    labels: list[dict],
    pt_per_ft: float,
    output_path: str | None = None,
) -> list[AnnotatedDuct]:
    """
    Phase 4 pipeline: associate labels → segments, compute physical dimensions.

    Parameters
    ----------
    segments   : Phase 1 + Phase 3 duct segments
    labels     : label list from extract_labels_with_scale (type=length/cross_section/duct_id)
    pt_per_ft  : calibrated scale from Phase 2
    output_path: if given, write phase4_annotated_ducts.json here

    Returns
    -------
    list of AnnotatedDuct (one per input segment, preserving order)
    """
    length_labels = [l for l in labels if l["type"] == "length"]
    cs_labels     = [l for l in labels if l["type"] == "cross_section"]
    id_labels     = [l for l in labels if l["type"] == "duct_id"]

    len_map = _assign_length_labels(segments, length_labels, pt_per_ft)
    cs_map  = _assign_nearest_labels(segments, cs_labels)
    id_map  = _assign_nearest_labels(segments, id_labels)

    result: list[AnnotatedDuct] = []

    for seg in segments:
        len_lbl = len_map.get(seg.id)
        cs_lbl  = cs_map.get(seg.id)
        id_lbl  = id_map.get(seg.id)

        length_ft_measured = seg.long_pt / pt_per_ft

        length_ft_label: float | None = len_lbl["feet"] if len_lbl else None
        if length_ft_label is not None and length_ft_label > 0:
            mismatch = (
                abs(length_ft_measured - length_ft_label) / length_ft_label
                > LABEL_MISMATCH_THRESHOLD
            )
        else:
            mismatch = False

        cross_section: dict | None = None
        is_round = False
        if cs_lbl:
            ct = cs_lbl.get("cross_type")
            if ct == "round":
                cross_section = {"diameter_in": cs_lbl["diameter_in"]}
                is_round = True
            elif ct == "rect":
                cross_section = {
                    "width_in": cs_lbl["width_in"],
                    "height_in": cs_lbl["height_in"],
                }

        unlabeled = len_lbl is None and cs_lbl is None

        result.append(AnnotatedDuct(
            segment_id=seg.id,
            duct_label_id=id_lbl["text"] if id_lbl else None,
            rect=seg.rect,
            orientation=seg.orientation,
            length_ft_measured=round(length_ft_measured, 3),
            length_ft_label=round(length_ft_label, 4) if length_ft_label is not None else None,
            length_mismatch=mismatch,
            cross_section=cross_section,
            is_round=is_round,
            unlabeled=unlabeled,
            confidence=seg.confidence,
            source=seg.source,
            page=seg.page,
            centerline=seg.centerline,
        ))

    if output_path:
        payload = {
            "pt_per_ft": round(pt_per_ft, 4),
            "duct_count": len(result),
            "annotated_ducts": [d.to_dict() for d in result],
        }
        Path(output_path).write_text(json.dumps(payload, indent=2))

    return result
