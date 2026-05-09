"""HVAC Duct Detection — Agentic Pipeline (v2.1)"""
import base64
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import anthropic
import fitz

V2_DIR = Path(__file__).parent.parent / "hvac-duct-detection-v2"
sys.path.insert(0, str(V2_DIR))
sys.path.insert(0, str(Path(__file__).parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

from config.settings import OUTPUTS_DIR, RENDER_DPI, VISION_MODEL, VISION_CROP_MARGIN_PT
from models.duct_segment import DuctSegment
from models.annotated_duct import AnnotatedDuct
from tools.vector_duct_extractor import extract_ducts
from tools.label_extractor import extract_labels_with_scale
from tools.raster_duct_extractor import extract_raster_ducts
from tools.duct_annotator import annotate_ducts
from tools.annotation_renderer import render_annotated_page


@dataclass
class PipelineState:
    pdf_path: str
    vector_segs: list = field(default_factory=list)
    raster_segs: list = field(default_factory=list)
    labels: list = field(default_factory=list)
    pt_per_ft: float = 0.0
    annotated: list = field(default_factory=list)
    output_dir: Path | None = None
    outputs: dict = field(default_factory=dict)


# ── Tool implementations ──────────────────────────────────────────────────────

def _tool_extract_vector_ducts(state: PipelineState, inp: dict) -> dict:
    segs = extract_ducts(state.pdf_path)
    state.vector_segs = segs
    orientations = {"H": 0, "V": 0, "D": 0}
    for s in segs:
        orientations[s.orientation] = orientations.get(s.orientation, 0) + 1
    return {"segment_count": len(segs), "orientations": orientations}


def _tool_extract_labels_and_scale(state: PipelineState, inp: dict) -> dict:
    result = extract_labels_with_scale(state.pdf_path, state.vector_segs)
    state.labels = result["labels"]
    state.pt_per_ft = result["pt_per_ft"]
    labels = result["labels"]
    return {
        "pt_per_ft": round(result["pt_per_ft"], 4),
        "length_labels": sum(1 for l in labels if l["type"] == "length"),
        "cross_section_labels": sum(1 for l in labels if l["type"] == "cross_section"),
        "duct_id_labels": sum(1 for l in labels if l["type"] == "duct_id"),
    }


def _tool_detect_raster_ducts(state: PipelineState, inp: dict) -> dict:
    segs = extract_raster_ducts(state.pdf_path, state.vector_segs)
    state.raster_segs = segs
    return {
        "new_segments": len(segs),
        "total_segments": len(state.vector_segs) + len(segs),
    }


def _tool_annotate_segments(state: PipelineState, inp: dict) -> dict:
    all_segs = state.vector_segs + state.raster_segs
    annotated = annotate_ducts(all_segs, state.labels, state.pt_per_ft)
    state.annotated = annotated
    with_length = sum(1 for d in annotated if d.length_ft_label is not None)
    with_cs = sum(1 for d in annotated if d.cross_section is not None)
    mismatches = [d.segment_id for d in annotated if d.length_mismatch]
    unlabeled = [d.segment_id for d in annotated if d.unlabeled]
    return {
        "total_ducts": len(annotated),
        "with_length_label": with_length,
        "with_cross_section": with_cs,
        "length_mismatches": len(mismatches),
        "unlabeled": len(unlabeled),
        "mismatch_ids": mismatches,
        "unlabeled_ids": unlabeled,
    }


def _tool_inspect_duct(state: PipelineState, inp: dict) -> dict:
    segment_id = inp["segment_id"]
    margin = float(inp.get("margin_pt", VISION_CROP_MARGIN_PT))

    duct = next((d for d in state.annotated if d.segment_id == segment_id), None)
    if duct is None:
        return {"error": f"Segment not found: {segment_id}"}

    x0, y0, x1, y1 = duct.rect
    doc = fitz.open(state.pdf_path)
    page = doc[duct.page]
    clip = fitz.Rect(x0 - margin, y0 - margin, x1 + margin, y1 + margin)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip)
    doc.close()

    b64 = base64.b64encode(pix.tobytes("png")).decode()

    return {
        "segment_id": duct.segment_id,
        "duct_label_id": duct.duct_label_id,
        "orientation": duct.orientation,
        "length_ft_measured": round(duct.length_ft_measured, 3),
        "length_ft_label": round(duct.length_ft_label, 4) if duct.length_ft_label is not None else None,
        "length_mismatch": duct.length_mismatch,
        "cross_section": duct.cross_section,
        "unlabeled": duct.unlabeled,
        "image_base64": b64,
    }


def _tool_update_duct_assessment(state: PipelineState, inp: dict) -> dict:
    segment_id = inp["segment_id"]
    duct = next((d for d in state.annotated if d.segment_id == segment_id), None)
    if duct is None:
        return {"error": f"Segment not found: {segment_id}"}

    is_valid = inp.get("is_valid_duct")
    if is_valid is False:
        duct.confidence = 0.0
    elif "confidence" in inp:
        duct.confidence = float(inp["confidence"])

    return {
        "segment_id": duct.segment_id,
        "confidence": round(duct.confidence, 3),
        "updated": True,
    }


def _tool_render_output(state: PipelineState, inp: dict) -> dict:
    stem = Path(state.pdf_path).stem
    out_dir = state.output_dir if state.output_dir is not None else OUTPUTS_DIR / stem
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    png_path = str(out_dir / f"{stem}_annotated.png")
    render_annotated_page(state.pdf_path, state.annotated, png_path)

    summary = {
        "input": state.pdf_path,
        "pt_per_ft": round(state.pt_per_ft, 4),
        "segment_counts": {
            "vector": len(state.vector_segs),
            "raster": len(state.raster_segs),
        },
        "annotated_ducts": [d.to_dict() for d in state.annotated],
        "outputs": state.outputs,
    }
    summary_path = str(out_dir / "summary.json")
    Path(summary_path).write_text(json.dumps(summary, indent=2))

    state.outputs = {"png_path": png_path, "summary_path": summary_path}
    return {
        "png_path": png_path,
        "summary_path": summary_path,
        "duct_count": len(state.annotated),
    }


# ── Tool registry ─────────────────────────────────────────────────────────────

TOOL_MAP = {
    "extract_vector_ducts": _tool_extract_vector_ducts,
    "extract_labels_and_scale": _tool_extract_labels_and_scale,
    "detect_raster_ducts": _tool_detect_raster_ducts,
    "annotate_segments": _tool_annotate_segments,
    "inspect_duct": _tool_inspect_duct,
    "update_duct_assessment": _tool_update_duct_assessment,
    "render_output": _tool_render_output,
}

TOOL_DEFINITIONS = [
    {
        "name": "extract_vector_ducts",
        "description": "Extract duct geometry from PDF vector paths. Returns segment count and orientation breakdown.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "extract_labels_and_scale",
        "description": "Parse dimension text labels and calibrate the pt/ft scale from the drawing.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "detect_raster_ducts",
        "description": "OpenCV raster fallback to detect ducts not captured by vector extraction.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "annotate_segments",
        "description": "Associate labels with duct segments, compute physical lengths and cross-sections.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "inspect_duct",
        "description": "Crop and return a visual image of a duct segment for inspection.",
        "input_schema": {
            "type": "object",
            "properties": {
                "segment_id": {
                    "type": "string",
                    "description": "The segment_id of the duct to inspect (e.g. 'duct_003').",
                },
                "margin_pt": {
                    "type": "number",
                    "description": "Padding in PDF points around the duct rect. Default 80.",
                },
            },
            "required": ["segment_id"],
        },
    },
    {
        "name": "update_duct_assessment",
        "description": "Update the confidence score of a duct after visual inspection.",
        "input_schema": {
            "type": "object",
            "properties": {
                "segment_id": {
                    "type": "string",
                    "description": "The segment_id of the duct to update.",
                },
                "confidence": {
                    "type": "number",
                    "description": "New confidence value in range 0.0–1.0.",
                },
                "is_valid_duct": {
                    "type": "boolean",
                    "description": "If false, sets confidence to 0.0 (marks as non-duct).",
                },
                "notes": {
                    "type": "string",
                    "description": "Optional notes about the assessment.",
                },
            },
            "required": ["segment_id"],
        },
    },
    {
        "name": "render_output",
        "description": "Render the annotated PNG and save summary JSON. Must be called last.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]

SYSTEM_PROMPT = """You are an HVAC duct detection agent. Analyze a mechanical floor plan PDF and produce a complete inventory of all duct segments with physical dimensions.

All measurements come from deterministic tools. Never guess coordinates or dimensions.

Workflow:
1. extract_vector_ducts — extract duct geometry from PDF vector paths
2. extract_labels_and_scale — parse dimension text, calibrate pt/ft scale
3. detect_raster_ducts — OpenCV fallback for ducts outside the vector layer
4. annotate_segments — associate labels with ducts, compute physical lengths
5. For each segment in mismatch_ids or unlabeled_ids: call inspect_duct to visually examine it
6. Call update_duct_assessment if inspection changes your confidence in a duct
7. Call render_output to finalize

When inspecting: a mismatch (>15% measured vs label) may be caused by a wrong label match, a segment split, or a CAD annotation error. Set confidence 0.85–1.0 if the duct looks valid, 0.1–0.3 if it looks like a fitting or non-duct geometry.

Always call render_output as the final step."""


def dispatch_tool(name: str, inp: dict, state: PipelineState) -> dict:
    fn = TOOL_MAP.get(name)
    if fn is None:
        return {"error": f"Unknown tool: {name}"}
    try:
        return fn(state, inp)
    except Exception as e:
        return {"error": str(e)}


def run_agent(
    pdf_path: str,
    output_dir: Path | None = None,
    on_event: Callable[[str, dict], None] | None = None,
) -> tuple[list, dict]:
    """
    Returns (annotated_ducts, summary_dict).
    on_event(event_type, data) is called for each agent event.
    Event types: "tool_call", "tool_result", "agent_text", "error"
    """
    client = anthropic.Anthropic()
    state = PipelineState(pdf_path=str(pdf_path), output_dir=output_dir)

    def emit(t, d):
        if on_event:
            on_event(t, d)

    messages = [{"role": "user", "content": f"Analyze the HVAC duct drawing: {pdf_path}"}]

    while True:
        response = client.messages.create(
            model=VISION_MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOL_DEFINITIONS,
            messages=messages,
            temperature=0,
        )

        for block in response.content:
            if hasattr(block, "type") and block.type == "text" and block.text.strip():
                emit("agent_text", {"text": block.text})

        if response.stop_reason in ("end_turn", "stop_sequence"):
            break
        if response.stop_reason != "tool_use":
            break

        tool_results = []
        for block in response.content:
            if not hasattr(block, "type") or block.type != "tool_use":
                continue

            emit("tool_call", {"tool": block.name, "input": block.input})
            result = dispatch_tool(block.name, block.input, state)

            safe_result = {k: v for k, v in result.items() if k != "image_base64"}
            emit("tool_result", {"tool": block.name, "result": safe_result})

            if "image_base64" in result:
                b64 = result.pop("image_base64")
                content = [
                    {"type": "text", "text": json.dumps(result)},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                ]
            else:
                content = json.dumps(result)

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": content,
            })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    summary = {
        "input": str(pdf_path),
        "pt_per_ft": round(state.pt_per_ft, 4),
        "segment_counts": {"vector": len(state.vector_segs), "raster": len(state.raster_segs)},
        "annotated_ducts": [d.to_dict() for d in state.annotated],
        "outputs": state.outputs,
    }
    return state.annotated, summary
