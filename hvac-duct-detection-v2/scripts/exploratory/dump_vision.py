"""
Exploratory script — Phase 5 vision validation dump

Usage:
    cd hvac-duct-detection-v2
    python scripts/exploratory/dump_vision.py [path/to/file.pdf]

Runs Phase 1 → Phase 2 → Phase 4 → Phase 5 and prints a table of which ducts
were sent for vision review and what the API returned.
Writes phase5_vision_log.json to outputs/<pdf-stem>/.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import OUTPUTS_DIR, SAMPLE_INPUT
from tools.vector_duct_extractor import extract_ducts
from tools.label_extractor import extract_labels_with_scale
from tools.duct_annotator import annotate_ducts
from tools.vision_validator import validate_ducts, _needs_vision


def main():
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else str(SAMPLE_INPUT)
    print(f"\nAnalysing: {pdf_path}")
    print("=" * 80)

    segs = extract_ducts(pdf_path)
    p2   = extract_labels_with_scale(pdf_path, segs)
    ann  = annotate_ducts(segs, p2["labels"], p2["pt_per_ft"])

    candidates = [d for d in ann if _needs_vision(d)]
    print(f"Phase 1 segments  : {len(segs)}")
    print(f"Phase 5 candidates: {len(candidates)}")

    if not candidates:
        print("\nNo vision review needed — all ducts have high confidence and no mismatches.")
        return

    print(f"\nRunning vision on {len(candidates)} duct(s)…")
    updated, log = validate_ducts(pdf_path)

    stem    = Path(pdf_path).stem
    out_dir = OUTPUTS_DIR / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "phase5_vision_log.json"
    log_path.write_text(json.dumps(log, indent=2))

    print(f"\n{'─'*70}")
    print(f"{'Seg':12}  {'Conf in':8}  {'Mismatch':8}  {'is_duct':7}  {'Conf out':8}  Notes")
    print(f"{'─'*70}")
    for entry in log:
        seg_id    = entry["segment_id"]
        after_d   = next(d for d in updated if d.segment_id == seg_id)
        print(
            f"{seg_id:12}  {entry['input_confidence']:8.3f}  "
            f"{'YES' if entry['length_mismatch'] else 'no':8}  "
            f"{'YES' if entry['is_duct'] else 'NO':7}  "
            f"{after_d.confidence:8.3f}  {entry.get('notes','')[:40]}"
        )

    print(f"\nVision log written: {log_path}")


if __name__ == "__main__":
    main()
