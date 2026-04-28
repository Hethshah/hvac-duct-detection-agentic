#!/usr/bin/env python3
"""
CLI entry point for the HVAC duct detection pipeline.

Usage:
    python scripts/run_pipeline.py --pdf path/to/plan.pdf
    python scripts/run_pipeline.py --pdf plan.pdf --confidence 0.85 --max-retries 3

Outputs always land in:
    hvac-duct-detection/outputs/<session_id>/

Every run is recorded in:
    hvac-duct-detection/runs/registry.csv
"""
import argparse
import csv
import sys
import uuid
from datetime import datetime
from pathlib import Path

# Ensure the package root (hvac-duct-detection/) is on sys.path regardless of
# where the script is invoked from.
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PACKAGE_ROOT))

import structlog

from agents.orchestrator import run_pipeline

logger = structlog.get_logger()

_OUTPUTS_ROOT = _PACKAGE_ROOT / "outputs"
_RUNS_DIR = _PACKAGE_ROOT / "runs"
_REGISTRY_CSV = _RUNS_DIR / "registry.csv"

_REGISTRY_HEADERS = [
    "session_id",
    "timestamp",
    "input_path",
    "output_dir",
    "output_pdf",
    "output_png",
    "segments_detected",
    "segments_labelled",
    "review_score",
    "retries",
]


def _make_session_id() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:6]
    return f"{ts}_{short}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and annotate HVAC ducts in a mechanical floor plan PDF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pdf", required=True, metavar="PATH",
        help="Path to the input PDF file.",
    )
    parser.add_argument(
        "--confidence", type=float, default=None, metavar="FLOAT",
        help="Minimum review score to accept (0.0-1.0). Overrides .env setting.",
    )
    parser.add_argument(
        "--max-retries", type=int, default=None, metavar="N",
        help="Maximum reflexion retries before accepting best-effort output. Overrides .env setting.",
    )
    parser.add_argument(
        "--scale-ratio", type=float, default=None, metavar="FLOAT",
        help=(
            "Manual pixels-per-foot scale override. Use when the drawing has no scale bar "
            "(e.g. 'DO NOT SCALE'). At 300 DPI, 1/4\"=1'-0\" is ~75.0."
        ),
    )
    parser.add_argument(
        "--pages", default="", metavar="RANGE",
        help=(
            "Page range to process (1-based). Examples: '1-3', '2,4,6', '5'. "
            "Default is all pages."
        ),
    )
    return parser.parse_args()


def _write_registry(summary: dict) -> None:
    """Append one row to runs/registry.csv, creating the file with headers if needed."""
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not _REGISTRY_CSV.exists() or _REGISTRY_CSV.stat().st_size == 0

    pngs = summary.get("output_pngs", [])
    output_png = ";".join(pngs) if pngs else ""

    row = {
        "session_id": summary.get("session_id", ""),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_path": summary.get("input_path", ""),
        "output_dir": summary.get("output_dir", ""),
        "output_pdf": summary.get("output_pdf", ""),
        "output_png": output_png,
        "segments_detected": summary.get("segments_detected", 0),
        "segments_labelled": summary.get("segments_labelled", 0),
        "review_score": summary.get("review_score", 0.0),
        "retries": summary.get("retries", 0),
    }

    with open(_REGISTRY_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_REGISTRY_HEADERS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _print_summary(summary: dict) -> None:
    pngs = summary.get("output_pngs", [])
    print()
    print("=" * 56)
    print("  HVAC Duct Detection — Pipeline Complete")
    print("=" * 56)
    print(f"  session_id        : {summary.get('session_id', '')}")
    print(f"  input_path        : {summary.get('input_path', '')}")
    print(f"  output_dir        : {summary.get('output_dir', '')}")
    print(f"  output_pdf        : {summary['output_pdf']}")
    for png in pngs:
        print(f"  output_png        : {png}")
    print(f"  segments_detected : {summary['segments_detected']}")
    print(f"  segments_labelled : {summary['segments_labelled']}")
    print(f"  review_score      : {summary['review_score']:.4f}")
    print(f"  retries           : {summary['retries']}")
    print("=" * 56)
    print(f"  Run log           : {_REGISTRY_CSV}")
    print("=" * 56)
    print()


def main() -> int:
    args = _parse_args()

    pdf_path = str(Path(args.pdf).resolve())
    if not Path(pdf_path).is_file():
        print(f"Error: PDF not found: {pdf_path}", file=sys.stderr)
        return 1

    session_id = _make_session_id()
    output_dir = str(_OUTPUTS_ROOT / session_id)

    try:
        summary = run_pipeline(
            pdf_path=pdf_path,
            output_dir=output_dir,
            confidence_threshold=args.confidence,
            max_retries=args.max_retries,
            page_range=args.pages,
            scale_ratio_override=args.scale_ratio,
            session_id=session_id,
        )
    except Exception as exc:
        logger.error("pipeline_failed", error=str(exc), exc_info=True)
        print(f"Error: pipeline failed — {exc}", file=sys.stderr)
        return 1

    _write_registry(summary)
    _print_summary(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
