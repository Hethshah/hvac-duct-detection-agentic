"""
Exploratory script — Phase 4 annotation dump

Usage:
    cd hvac-duct-detection-v2
    python scripts/exploratory/dump_annotations.py [path/to/file.pdf]

Runs Phase 1 → Phase 2 → Phase 4 and prints a table of annotated ducts,
including association quality and mismatch flags.
Writes phase4_annotated_ducts.json to outputs/<pdf-stem>/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import OUTPUTS_DIR, SAMPLE_INPUT
from tools.vector_duct_extractor import extract_ducts
from tools.label_extractor import extract_labels_with_scale
from tools.duct_annotator import annotate_ducts


def main():
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else str(SAMPLE_INPUT)
    print(f"\nAnalysing: {pdf_path}")
    print("=" * 80)

    # Phase 1
    segments = extract_ducts(pdf_path)
    print(f"Phase 1 segments : {len(segments)}")

    # Phase 2
    stem = Path(pdf_path).stem
    out_dir = OUTPUTS_DIR / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    p2 = extract_labels_with_scale(pdf_path, segments)
    pt_per_ft = p2["pt_per_ft"]
    labels = p2["labels"]
    print(f"Phase 2 pt_per_ft: {pt_per_ft:.4f}")
    print(f"Phase 2 labels   : {len(labels)}")

    # Phase 4
    json_path = str(out_dir / "phase4_annotated_ducts.json")
    annotated = annotate_ducts(segments, labels, pt_per_ft, output_path=json_path)

    # ── Summary counts ─────────────────────────────────────────────────────────
    with_len   = [d for d in annotated if d.length_ft_label is not None]
    with_cs    = [d for d in annotated if d.cross_section is not None]
    with_id    = [d for d in annotated if d.duct_label_id is not None]
    mismatches = [d for d in annotated if d.length_mismatch]
    unlabeled  = [d for d in annotated if d.unlabeled]

    print(f"\nAssociation summary:")
    print(f"  Ducts with length label   : {len(with_len)} / {len(annotated)}")
    print(f"  Ducts with cross-section  : {len(with_cs)} / {len(annotated)}")
    print(f"  Ducts with ID label       : {len(with_id)} / {len(annotated)}")
    print(f"  Length mismatches         : {len(mismatches)}")
    print(f"  Fully unlabeled           : {len(unlabeled)}")

    # ── Per-duct table ─────────────────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print(f"{'Seg':10}  {'ID':6}  {'Or':3}  {'Meas ft':8}  {'Lbl ft':8}  "
          f"{'Match':5}  {'CrossSec':14}  Flags")
    print(f"{'─'*80}")

    for d in annotated:
        lbl_str = f"{d.length_ft_label:.3f}" if d.length_ft_label else "  —   "
        cs_str  = "—"
        if d.cross_section:
            if d.is_round:
                cs_str = f"{d.cross_section['diameter_in']}\"ø"
            else:
                cs_str = f"{d.cross_section['width_in']}×{d.cross_section['height_in']}\""
        flags = []
        if d.length_mismatch: flags.append("MISMATCH")
        if d.unlabeled:       flags.append("UNLABELED")
        match_str = "FAIL" if d.length_mismatch else ("ok" if d.length_ft_label else "—")
        print(
            f"{d.segment_id:10}  {(d.duct_label_id or '—'):6}  {d.orientation:3}  "
            f"{d.length_ft_measured:8.3f}  {lbl_str:8}  "
            f"{match_str:5}  {cs_str:14}  {' '.join(flags)}"
        )

    # ── Mismatch detail ────────────────────────────────────────────────────────
    if mismatches:
        print(f"\n{'─'*80}")
        print("Mismatch details")
        print(f"{'─'*80}")
        for d in mismatches:
            delta = abs(d.length_ft_measured - d.length_ft_label) / d.length_ft_label
            print(f"  {d.segment_id}  measured={d.length_ft_measured:.3f}ft  "
                  f"label={d.length_ft_label:.3f}ft  delta={delta:.1%}")

    print(f"\nOverall: {'PASS' if not mismatches else 'FAIL — mismatches found'}")
    print(f"JSON written: {json_path}")


if __name__ == "__main__":
    main()
