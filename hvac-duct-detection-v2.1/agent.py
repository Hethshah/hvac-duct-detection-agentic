"""HVAC Duct Detection — Agentic Pipeline (v2.1)"""
import base64
import datetime
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import anthropic
import fitz

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
    event_log: list = field(default_factory=list)  # captured for HTML report


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
        "scale_source": result.get("scale_source", "derived"),
        "scale_text": result.get("scale_text", "derived"),
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


def _infer_duct_type(duct) -> str:
    if duct.duct_type:
        return duct.duct_type.capitalize()
    lid = (duct.duct_label_id or "").upper()
    if any(lid.startswith(p) for p in ("SA", "SUP", "S-", "SB")):
        return "Supply"
    if any(lid.startswith(p) for p in ("RA", "RET", "R-", "RB")):
        return "Return"
    if any(lid.startswith(p) for p in ("EA", "EXH", "EF", "E-", "OA")):
        return "Exhaust"
    return "Unknown"


def _pressure_class(duct) -> str:
    if duct.pressure_class:
        return duct.pressure_class.capitalize()
    cs = duct.cross_section
    if not cs:
        return "Unknown"
    dim = cs.get("diameter_in", 0) if duct.is_round else max(cs.get("width_in", 0), cs.get("height_in", 0))
    if dim >= 24:
        return "Low"
    if dim >= 12:
        return "Medium"
    return "High"


def _format_cs(duct) -> str:
    cs = duct.cross_section
    if not cs:
        return "—"
    if duct.is_round:
        return f'Ø {cs["diameter_in"]}"'
    return f'{cs["width_in"]}" × {cs["height_in"]}"'


def _tool_set_duct_classification(state: PipelineState, inp: dict) -> dict:
    """Batch-capable: accepts segment_ids (list) or segment_id (single string)."""
    duct_type = inp["duct_type"].lower()
    pressure = inp["pressure_class"].lower()

    ids = inp.get("segment_ids") or ([inp["segment_id"]] if "segment_id" in inp else [])
    if not ids:
        return {"error": "Provide segment_id or segment_ids"}

    updated = []
    for sid in ids:
        duct = next((d for d in state.annotated if d.segment_id == sid), None)
        if duct:
            duct.duct_type = duct_type
            duct.pressure_class = pressure
            updated.append(sid)

    return {
        "updated": len(updated),
        "duct_type": duct_type,
        "pressure_class": pressure,
        "segment_ids": updated,
    }


def _generate_html_report(state: PipelineState, out_dir: Path, stem: str, png_path: str) -> str:
    ducts = state.annotated
    generated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    # Embed annotated PNG as base64 so the file is self-contained
    try:
        img_b64 = base64.b64encode(Path(png_path).read_bytes()).decode()
        img_tag = f'<img src="data:image/png;base64,{img_b64}" style="max-width:100%;border-radius:6px;border:1px solid #e2e8f0">'
    except Exception:
        img_tag = "<p><em>Annotated image not available.</em></p>"

    # Stats
    total    = len(ducts)
    labeled  = sum(1 for d in ducts if not d.unlabeled)
    mismatch = sum(1 for d in ducts if d.length_mismatch)
    unlabeled = sum(1 for d in ducts if d.unlabeled)
    vector_n = len(state.vector_segs)
    raster_n = len(state.raster_segs)

    stat_cards = f"""
    <div class="stats">
      <div class="card"><span class="num">{total}</span><span class="lbl">Total Ducts</span></div>
      <div class="card"><span class="num">{labeled}</span><span class="lbl">Labeled</span></div>
      <div class="card warn"><span class="num">{mismatch}</span><span class="lbl">Mismatches</span></div>
      <div class="card muted"><span class="num">{unlabeled}</span><span class="lbl">Unlabeled</span></div>
      <div class="card"><span class="num">{vector_n}</span><span class="lbl">Vector Segments</span></div>
      <div class="card"><span class="num">{raster_n}</span><span class="lbl">Raster Segments</span></div>
      <div class="card"><span class="num">{state.pt_per_ft:.2f}</span><span class="lbl">pt / ft Scale</span></div>
    </div>"""

    # Duct table rows
    rows = []
    for i, d in enumerate(ducts, 1):
        mismatch_cls = ' class="mismatch-row"' if d.length_mismatch else ""
        label_len = f"{d.length_ft_label:.2f}" if d.length_ft_label is not None else "—"
        match_cell = '<span class="badge warn">Mismatch</span>' if d.length_mismatch else '<span class="badge ok">OK</span>'
        unlabeled_badge = ' <span class="badge muted">Unlabeled</span>' if d.unlabeled else ""
        rows.append(f"""<tr{mismatch_cls}>
          <td>{i}</td>
          <td>{d.duct_label_id or "—"}{unlabeled_badge}</td>
          <td>{_infer_duct_type(d)}</td>
          <td>{d.orientation}</td>
          <td>{_format_cs(d)}</td>
          <td>{d.length_ft_measured:.2f} ft</td>
          <td>{label_len}</td>
          <td>{match_cell}</td>
          <td>{_pressure_class(d)}</td>
          <td>{d.source}</td>
          <td>{d.confidence:.2f}</td>
        </tr>""")
    table_body = "\n".join(rows)

    # Agent log
    log_items = []
    tool_labels = {
        "extract_vector_ducts":    "Extract vector ducts",
        "extract_labels_and_scale": "Parse labels & calibrate scale",
        "detect_raster_ducts":     "Raster fallback detection",
        "annotate_segments":       "Annotate segments",
        "inspect_duct":            "Visual inspection",
        "update_duct_assessment":  "Update assessment",
        "render_output":           "Render output",
    }
    i = 0
    log_events = state.event_log
    while i < len(log_events):
        ev = log_events[i]
        if ev["type"] == "tool_call":
            tool = ev["data"].get("tool", "")
            label = tool_labels.get(tool, tool)
            extra = ""
            if tool == "inspect_duct":
                extra = f' — {ev["data"].get("input", {}).get("segment_id", "")}'
            elif tool == "update_duct_assessment":
                inp = ev["data"].get("input", {})
                extra = f' — {inp.get("segment_id", "")} → confidence {inp.get("confidence", "?")}'
            # Look ahead for result
            result_summary = ""
            if i + 1 < len(log_events) and log_events[i + 1]["type"] == "tool_result":
                r = log_events[i + 1]["data"].get("result", {})
                parts = []
                for k in ("segment_count", "total_ducts", "new_segments", "pt_per_ft", "duct_count"):
                    if k in r:
                        parts.append(f"{k.replace('_', ' ')}: <strong>{r[k]}</strong>")
                result_summary = f'<div class="result-line">{" · ".join(parts)}</div>' if parts else ""
                i += 1
            log_items.append(f'<div class="log-tool"><span class="tool-icon">⚙</span> <strong>{label}</strong>{extra}{result_summary}</div>')
        elif ev["type"] == "agent_text":
            text = ev["data"].get("text", "").strip()
            if text:
                log_items.append(f'<div class="log-agent"><span class="tool-icon">🤖</span> {text}</div>')
        i += 1
    agent_log_html = "\n".join(log_items) if log_items else "<p><em>No agent log captured.</em></p>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HVAC Duct Detection Report — {stem}</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: system-ui, -apple-system, sans-serif; background: #f8fafc; color: #1e293b; line-height: 1.6; }}
  .page {{ max-width: 1200px; margin: 0 auto; padding: 2rem; }}
  h1 {{ font-size: 1.75rem; font-weight: 700; color: #0f172a; }}
  h2 {{ font-size: 1.2rem; font-weight: 600; color: #1e293b; margin: 2rem 0 1rem; border-bottom: 2px solid #e2e8f0; padding-bottom: .4rem; }}
  .meta {{ color: #64748b; font-size: .875rem; margin-top: .25rem; }}
  .stats {{ display: flex; flex-wrap: wrap; gap: .75rem; margin: 1.5rem 0; }}
  .card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; padding: .75rem 1.25rem; display: flex; flex-direction: column; align-items: center; min-width: 110px; }}
  .card.warn {{ border-color: #fbbf24; background: #fffbeb; }}
  .card.muted {{ border-color: #cbd5e1; background: #f8fafc; }}
  .num {{ font-size: 1.6rem; font-weight: 700; color: #3762E3; }}
  .card.warn .num {{ color: #d97706; }}
  .card.muted .num {{ color: #94a3b8; }}
  .lbl {{ font-size: .75rem; color: #64748b; text-transform: uppercase; letter-spacing: .04em; }}
  .drawing {{ margin: 1rem 0; }}
  table {{ width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px; overflow: hidden; border: 1px solid #e2e8f0; font-size: .875rem; }}
  th {{ background: #f1f5f9; text-align: left; padding: .6rem .75rem; font-weight: 600; color: #475569; border-bottom: 1px solid #e2e8f0; }}
  td {{ padding: .55rem .75rem; border-bottom: 1px solid #f1f5f9; vertical-align: middle; }}
  tr:last-child td {{ border-bottom: none; }}
  tr.mismatch-row {{ background: #fffbeb; }}
  .badge {{ display: inline-block; font-size: .7rem; font-weight: 600; padding: .15em .5em; border-radius: 4px; }}
  .badge.ok {{ background: #dcfce7; color: #15803d; }}
  .badge.warn {{ background: #fef9c3; color: #a16207; }}
  .badge.muted {{ background: #f1f5f9; color: #64748b; }}
  .log-tool {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 6px; padding: .6rem .9rem; margin: .4rem 0; }}
  .log-agent {{ background: #f0f9ff; border-left: 3px solid #3762E3; padding: .6rem .9rem; margin: .4rem 0; font-style: italic; color: #334155; border-radius: 0 6px 6px 0; }}
  .result-line {{ font-size: .8rem; color: #64748b; margin-top: .2rem; }}
  .tool-icon {{ margin-right: .4rem; }}
  footer {{ margin-top: 3rem; font-size: .8rem; color: #94a3b8; text-align: center; }}
</style>
</head>
<body>
<div class="page">
  <h1>HVAC Duct Detection Report</h1>
  <p class="meta">Input: <strong>{Path(state.pdf_path).name}</strong> &nbsp;|&nbsp; Generated: {generated_at} &nbsp;|&nbsp; Scale: {state.pt_per_ft:.2f} pt/ft</p>

  {stat_cards}

  <h2>Annotated Floor Plan</h2>
  <div class="drawing">{img_tag}</div>

  <h2>Duct Inventory ({total} ducts)</h2>
  <table>
    <thead>
      <tr>
        <th>#</th><th>Duct ID</th><th>Type</th><th>Dir</th>
        <th>Dimensions</th><th>Measured</th><th>Label</th>
        <th>Match</th><th>Pressure</th><th>Source</th><th>Conf.</th>
      </tr>
    </thead>
    <tbody>
{table_body}
    </tbody>
  </table>

  <h2>Agent Analysis Log</h2>
  <div class="agent-log">
{agent_log_html}
  </div>

  <footer>Generated by HVAC Duct Detection v2.1 (Agentic)</footer>
</div>
</body>
</html>"""

    report_path = str(out_dir / f"{stem}_report.html")
    Path(report_path).write_text(html, encoding="utf-8")
    return report_path


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

    report_path = _generate_html_report(state, out_dir, stem, png_path)

    state.outputs = {
        "png_path": png_path,
        "summary_path": summary_path,
        "report_path": report_path,
    }
    return {
        "png_path": png_path,
        "summary_path": summary_path,
        "report_path": report_path,
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
    "set_duct_classification": _tool_set_duct_classification,
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
        "name": "set_duct_classification",
        "description": (
            "Record duct type (supply/return/exhaust) and pressure class (low/medium/high) "
            "for one or many duct segments. Use segment_ids list to batch-classify ducts of the same type."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "segment_id": {
                    "type": "string",
                    "description": "Single segment ID to classify.",
                },
                "segment_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Multiple segment IDs to classify with the same type and pressure.",
                },
                "duct_type": {
                    "type": "string",
                    "enum": ["supply", "return", "exhaust"],
                    "description": "Duct function: supply (delivers conditioned air), return (recirculates), exhaust (vents out).",
                },
                "pressure_class": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "Low: largest dim ≥24\", Medium: 12–24\", High: <12\". Use cross_section from annotate_segments.",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of why this classification was chosen.",
                },
            },
            "required": ["duct_type", "pressure_class"],
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

SYSTEM_PROMPT = """You are an HVAC duct detection agent. Analyze a mechanical floor plan PDF and produce a complete inventory of all duct segments with physical dimensions, types, and pressure classes.

All measurements come from deterministic tools. Never guess coordinates or dimensions.

Workflow:
1. extract_vector_ducts — extract duct geometry from PDF vector paths
2. extract_labels_and_scale — parse dimension text, calibrate pt/ft scale
3. detect_raster_ducts — OpenCV fallback for ducts outside the vector layer
4. annotate_segments — associate labels with ducts, compute physical lengths
5. Classify every duct using set_duct_classification:
   - Group ducts by label pattern and classify in batches using segment_ids
   - Label conventions: SA/SUP/SB = supply, RA/RET/RB = return, EA/EXH/EF/OA = exhaust
   - Numeric or ambiguous labels (C01, 001, etc.): call inspect_duct first, then classify
     based on airflow arrows, text annotations, proximity to AHU/VAV boxes, or routing context
   - Pressure class from cross_section: largest dim ≥24" = low, 12–24" = medium, <12" = high
   - When cross_section is missing, use inspect_duct to read dimension text from the crop
   - Every duct must receive a classification — do not skip any
6. For mismatched or unlabeled ducts not yet inspected: call inspect_duct and update_duct_assessment
7. Call render_output last

Classification tips:
- Supply ducts are typically the main trunk and branch ducts distributing conditioned air
- Return ducts are usually wider, lower-pressure paths back to the air handler
- Exhaust ducts connect to fans, toilet rooms, or exterior louvers
- On unlabeled segments, check if the duct is connected to a labeled trunk of known type

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
        state.event_log.append({"type": t, "data": d})
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
