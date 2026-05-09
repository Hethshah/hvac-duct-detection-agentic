#!/usr/bin/env python3
"""
HVAC Duct Detection — Agentic pipeline CLI (v2.1)

Usage:
    cd hvac-duct-detection-v2.1
    python run.py path/to/drawing.pdf [--output-dir DIR]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

from agent import run_agent


def _summarise_result(tool: str, result: dict) -> str:
    if tool == "extract_vector_ducts":
        o = result.get("orientations", {})
        return f"{result.get('segment_count', '?')} segments (H={o.get('H',0)} V={o.get('V',0)} D={o.get('D',0)})"
    if tool == "extract_labels_and_scale":
        return (
            f"pt_per_ft={result.get('pt_per_ft','?')}, "
            f"{result.get('length_labels','?')} length labels, "
            f"{result.get('cross_section_labels','?')} cross-section labels"
        )
    if tool == "detect_raster_ducts":
        return f"{result.get('new_segments','?')} new segments, total={result.get('total_segments','?')}"
    if tool == "annotate_segments":
        return (
            f"{result.get('total_ducts','?')} ducts, "
            f"{result.get('length_mismatches','?')} mismatches, "
            f"{result.get('unlabeled','?')} unlabeled"
        )
    if tool == "inspect_duct":
        return f"examining {result.get('segment_id','?')}..."
    if tool == "update_duct_assessment":
        return f"segment={result.get('segment_id','?')} confidence={result.get('confidence','?')}"
    if tool == "render_output":
        return f"outputs/ (duct_count={result.get('duct_count','?')})"
    return str(result)


def main() -> int:
    parser = argparse.ArgumentParser(description="HVAC duct detection agentic pipeline")
    parser.add_argument("pdf", help="Path to the input PDF")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: outputs/<pdf-stem>/)")
    args = parser.parse_args()

    pdf_path = args.pdf
    out_dir = Path(args.output_dir) if args.output_dir else None

    t0 = time.perf_counter()

    def on_event(event_type: str, data: dict) -> None:
        if event_type == "tool_call":
            tool = data.get("tool", "?")
            inp = data.get("input", {})
            if tool == "inspect_duct":
                print(f"[tool] {tool} {inp.get('segment_id', '')} → ", end="", flush=True)
            else:
                print(f"[tool] {tool} → ", end="", flush=True)
        elif event_type == "tool_result":
            tool = data.get("tool", "?")
            result = data.get("result", {})
            print(_summarise_result(tool, result))
        elif event_type == "agent_text":
            text = data.get("text", "").strip()
            if text:
                print(f"[agent] {text}")
        elif event_type == "error":
            print(f"[error] {data.get('error', '?')}", file=sys.stderr)

    annotated, summary = run_agent(pdf_path, output_dir=out_dir, on_event=on_event)

    elapsed = time.perf_counter() - t0
    duct_count = len(annotated)
    print(f"\nDone in {elapsed:.1f}s — {duct_count} ducts annotated")

    outputs = summary.get("outputs", {})
    if outputs:
        print("Outputs:")
        for k, v in outputs.items():
            print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
