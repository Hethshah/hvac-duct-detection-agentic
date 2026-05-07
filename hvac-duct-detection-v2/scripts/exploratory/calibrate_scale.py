"""
Exploratory script — Phase 2 dump

Usage:
    cd hvac-duct-detection-v2
    python scripts/exploratory/calibrate_scale.py [path/to/file.pdf]

Prints all parsed labels, the derived pt_per_ft, and which labels associate
to Phase 1 duct segments. Writes phase2_labels.json to outputs/<pdf-stem>/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import OUTPUTS_DIR, PT_PER_FT_MIN, PT_PER_FT_MAX, SAMPLE_INPUT
from tools.label_extractor import extract_labels_with_scale, extract_labels
from tools.vector_duct_extractor import extract_ducts


def main():
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else str(SAMPLE_INPUT)
    print(f"\nAnalysing: {pdf_path}")
    print("=" * 70)

    segments = extract_ducts(pdf_path)
    print(f"Phase 1 segments loaded: {len(segments)}")

    stem = Path(pdf_path).stem
    out_dir = OUTPUTS_DIR / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = str(out_dir / "phase2_labels.json")

    result = extract_labels_with_scale(pdf_path, segments, output_path=json_path)
    labels = result["labels"]

    # ── Print labels grouped by type ───────────────────────────────────────────
    length_labels = [l for l in labels if l["type"] == "length"]
    cross_labels  = [l for l in labels if l["type"] == "cross_section"]
    id_labels     = [l for l in labels if l["type"] == "duct_id"]

    print(f"\n{'─'*70}")
    print(f"Length labels  ({len(length_labels)})")
    print(f"{'─'*70}")
    for l in sorted(length_labels, key=lambda x: x["feet"]):
        assoc = l["nearest_duct_id"] or "—"
        dist  = l["nearest_dist_pt"]
        print(f"  {l['text']:<18}  {l['feet']:.3f} ft  "
              f"cx={l['cx']:.0f} cy={l['cy']:.0f}  "
              f"→ {assoc}  ({dist:.0f} pt)")

    print(f"\n{'─'*70}")
    print(f"Cross-section labels ({len(cross_labels)})")
    print(f"{'─'*70}")
    for l in cross_labels:
        assoc = l["nearest_duct_id"] or "—"
        dist  = l["nearest_dist_pt"]
        if l.get("diameter_in"):
            dim = f"{l['diameter_in']}\"ø"
        elif l.get("width_in"):
            dim = f"{l['width_in']}×{l['height_in']}\""
        else:
            dim = "?"
        print(f"  {l['text']:<26}  {dim:<10}  → {assoc}  ({dist:.0f} pt)")

    print(f"\n{'─'*70}")
    print(f"Duct ID labels ({len(id_labels)})")
    print(f"{'─'*70}")
    for l in sorted(id_labels, key=lambda x: x["text"]):
        assoc = l["nearest_duct_id"] or "—"
        dist  = l["nearest_dist_pt"]
        print(f"  {l['text']:<8}  cx={l['cx']:.0f} cy={l['cy']:.0f}  → {assoc}  ({dist:.0f} pt)")

    # ── Scale calibration result ───────────────────────────────────────────────
    pt_per_ft = result["pt_per_ft"]
    print(f"\n{'─'*70}")
    print(f"Scale calibration")
    print(f"{'─'*70}")
    print(f"  pt_per_ft       : {pt_per_ft:.4f}")
    print(f"  valid range     : [{PT_PER_FT_MIN}, {PT_PER_FT_MAX}]")
    print(f"  1 ft in drawing : {pt_per_ft:.1f} pt  ({pt_per_ft * 72 / 300:.3f} mm at 300 DPI)")

    # ── Validation ─────────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("Validation")
    print(f"{'─'*70}")
    ok = True

    check_scale = PT_PER_FT_MIN <= pt_per_ft <= PT_PER_FT_MAX
    print(f"  [{'OK' if check_scale else 'FAIL'}] pt_per_ft in [{PT_PER_FT_MIN},{PT_PER_FT_MAX}]  (got {pt_per_ft:.4f})")
    ok = ok and check_scale

    check_len = len(length_labels) >= 5
    print(f"  [{'OK' if check_len else 'FAIL'}] >= 5 length labels  (got {len(length_labels)})")
    ok = ok and check_len

    check_cs = len(cross_labels) >= 1
    print(f"  [{'OK' if check_cs else 'FAIL'}] >= 1 cross-section label  (got {len(cross_labels)})")
    ok = ok and check_cs

    known_ids = {"C01", "C02", "C03"}
    found_ids = {l["text"] for l in id_labels}
    check_ids = known_ids.issubset(found_ids)
    print(f"  [{'OK' if check_ids else 'FAIL'}] C01/C02/C03 present  (found: {sorted(found_ids)})")
    ok = ok and check_ids

    # 18'-5" label must associate to a duct (the longest dining run)
    long_label = next((l for l in length_labels if abs(l["feet"] - 18.417) < 0.1), None)
    check_long = long_label is not None and long_label["nearest_duct_id"] is not None
    got_str = f"{long_label['nearest_duct_id']} @ {long_label['nearest_dist_pt']:.0f}pt" if long_label else "missing"
    print(f"  [{'OK' if check_long else 'FAIL'}] 18'-5\" associates to a duct  ({got_str})")
    ok = ok and check_long

    print()
    print(f"Overall: {'PASS' if ok else 'FAIL'}")
    print(f"\nJSON written: {json_path}")


if __name__ == "__main__":
    main()
