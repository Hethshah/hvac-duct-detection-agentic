#!/usr/bin/env python3
"""
HVAC Duct Detection — end-to-end pipeline

Usage:
    cd hvac-duct-detection-v2
    python run.py path/to/drawing.pdf [--skip-vision] [--output-dir DIR]

Phases:
  1. Vector duct extraction   (PDF paths → DuctSegments)
  2. Label + scale extraction (OCR text → calibrated pt_per_ft)
  3. Raster fallback          (OpenCV morphology → additional DuctSegments)
  4. Duct annotation          (labels → AnnotatedDucts, physical dimensions)
  5. Vision cross-validation  (Claude vision → confidence update for mismatches)
  6. Render + summary         (annotated PNG + summary.json)

"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Load .env from repo root or current dir (sets ANTHROPIC_API_KEY etc.)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

from config.settings import OUTPUTS_DIR
from tools.vector_duct_extractor import extract_ducts
from tools.label_extractor import extract_labels_with_scale
from tools.raster_duct_extractor import extract_raster_ducts
from tools.duct_annotator import annotate_ducts
from tools.vision_validator import validate_ducts, _needs_vision
from tools.annotation_renderer import render_annotated_page


def main() -> int:
    parser = argparse.ArgumentParser(description="HVAC duct detection pipeline")
    parser.add_argument("pdf", help="Path to the input PDF")
    parser.add_argument(
        "--skip-vision", action="store_true",
        help="Skip Phase 5 vision cross-validation (no API calls)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: outputs/<pdf-stem>/)",
    )
    args = parser.parse_args()

    pdf_path = args.pdf
    stem     = Path(pdf_path).stem
    out_dir  = Path(args.output_dir) if args.output_dir else OUTPUTS_DIR / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    print("Phase 1  vector extraction …", end=" ", flush=True)
    vector_segs = extract_ducts(pdf_path)
    print(f"{len(vector_segs)} segments")

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    print("Phase 2  label + scale …", end=" ", flush=True)
    p2        = extract_labels_with_scale(pdf_path, vector_segs)
    pt_per_ft = p2["pt_per_ft"]
    labels    = p2["labels"]
    print(f"{len(labels)} labels  pt_per_ft={pt_per_ft:.4f}")

    # ── Phase 3 ──────────────────────────────────────────────────────────────
    print("Phase 3  raster fallback …", end=" ", flush=True)
    raster_segs = extract_raster_ducts(pdf_path, vector_segs)
    all_segs    = vector_segs + raster_segs
    print(f"{len(raster_segs)} new  total={len(all_segs)}")

    # ── Phase 4 ──────────────────────────────────────────────────────────────
    print("Phase 4  annotation …", end=" ", flush=True)
    p4_path   = str(out_dir / "phase4_annotated_ducts.json")
    annotated = annotate_ducts(all_segs, labels, pt_per_ft, output_path=p4_path)
    with_len   = sum(1 for d in annotated if d.length_ft_label is not None)
    with_cs    = sum(1 for d in annotated if d.cross_section is not None)
    mismatches = sum(1 for d in annotated if d.length_mismatch)
    unlabeled  = sum(1 for d in annotated if d.unlabeled)
    print(
        f"{len(annotated)} ducts  "
        f"len={with_len}  cs={with_cs}  mismatch={mismatches}  unlabeled={unlabeled}"
    )

    # ── Phase 5 ──────────────────────────────────────────────────────────────
    vision_log: list[dict] = []
    if args.skip_vision:
        print("Phase 5  vision — skipped (--skip-vision)")
    else:
        n_cand = sum(1 for d in annotated if _needs_vision(d))
        if n_cand == 0:
            print("Phase 5  vision — 0 candidates, skipped")
        else:
            print(f"Phase 5  vision — reviewing {n_cand} duct(s) …", end=" ", flush=True)
            annotated, vision_log = validate_ducts(pdf_path, annotated)
            print("done")

    # ── Phase 6 ──────────────────────────────────────────────────────────────
    print("Phase 6  rendering …", end=" ", flush=True)
    png_path = str(out_dir / f"{stem}_annotated.png")
    render_annotated_page(pdf_path, annotated, png_path)
    print(f"→ {png_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = {
        "input": pdf_path,
        "pt_per_ft": round(pt_per_ft, 4),
        "segment_counts": {
            "vector": len(vector_segs),
            "raster": len(raster_segs),
            "total":  len(all_segs),
        },
        "label_counts": {
            "length":        sum(1 for l in labels if l["type"] == "length"),
            "cross_section": sum(1 for l in labels if l["type"] == "cross_section"),
            "duct_id":       sum(1 for l in labels if l["type"] == "duct_id"),
        },
        "annotation": {
            "with_length_label":  with_len,
            "with_cross_section": with_cs,
            "length_mismatches":  mismatches,
            "unlabeled":          unlabeled,
        },
        "vision_reviews": len(vision_log),
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "annotated_ducts": [d.to_dict() for d in annotated],
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\nOutputs written to {out_dir}/")
    print(f"  {stem}_annotated.png")
    print(f"  summary.json")
    print(f"  phase4_annotated_ducts.json")
    print(f"\nCompleted in {summary['elapsed_s']:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
