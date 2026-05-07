"""
Exploratory script — Phase 3 raster dump

Usage:
    cd hvac-duct-detection-v2
    python scripts/exploratory/dump_raster.py [path/to/file.pdf]

Prints blob stats, paired candidates, and any new segments after dedup
against Phase 1. Expected result for input1.pdf: 0 new segments.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import SAMPLE_INPUT, RASTER_SCALE, TITLE_BLOCK_Y_MIN_PT
from tools.vector_duct_extractor import extract_ducts
from tools.raster_duct_extractor import (
    _render_black_mask,
    _apply_title_block_mask,
    _find_h_blobs,
    _find_v_blobs,
    _pair_h_blobs,
    _pair_v_blobs,
    _h_pair_to_segment,
    _v_pair_to_segment,
    extract_raster_ducts,
)


def main():
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else str(SAMPLE_INPUT)
    print(f"\nAnalysing: {pdf_path}")
    print("=" * 70)

    p1_segs = extract_ducts(pdf_path)
    print(f"Phase 1 segments: {len(p1_segs)}")

    mask, page_info = _render_black_mask(pdf_path, 0, RASTER_SCALE)
    print(f"\nPage info:")
    print(f"  rotation     : {page_info['rotation']}°")
    print(f"  image size   : {page_info['img_w']} × {page_info['img_h']} px")
    print(f"  media_w_pt   : {page_info['media_w_pt']:.1f}")
    print(f"  title block  : col > {int(TITLE_BLOCK_Y_MIN_PT * RASTER_SCALE)} px")
    print(f"  black pixels : {(mask > 0).sum():,} / {mask.size:,}")

    mask = _apply_title_block_mask(mask, page_info)
    print(f"  after masking: {(mask > 0).sum():,} black pixels")

    h_blobs = _find_h_blobs(mask)
    v_blobs = _find_v_blobs(mask)
    print(f"\nBlobs found:")
    print(f"  H blobs (→ V ducts in media) : {len(h_blobs)}")
    print(f"  V blobs (→ H ducts in media) : {len(v_blobs)}")

    h_pairs = _pair_h_blobs(h_blobs)
    v_pairs = _pair_v_blobs(v_blobs)
    print(f"\nPairs found:")
    print(f"  H pairs : {len(h_pairs)}")
    print(f"  V pairs : {len(v_pairs)}")

    scale = page_info["scale"]
    media_w_pt = page_info["media_w_pt"]

    candidates = []
    counter = 1
    for top, bot in h_pairs:
        seg = _h_pair_to_segment(top, bot, scale, media_w_pt, f"raster_{counter:03d}")
        candidates.append((seg, "H-pair→V", top, bot))
        if seg:
            counter += 1
    for left, right in v_pairs:
        seg = _v_pair_to_segment(left, right, scale, media_w_pt, f"raster_{counter:03d}")
        candidates.append((seg, "V-pair→H", left, right))
        if seg:
            counter += 1

    print(f"\n{'─'*70}")
    print(f"Candidate segments (before dedup)")
    print(f"{'─'*70}")
    valid = [c for c in candidates if c[0] is not None]
    rejected = [c for c in candidates if c[0] is None]
    print(f"  valid    : {len(valid)}")
    print(f"  rejected : {len(rejected)}  (failed size/aspect filter)")
    for seg, kind, a, b in valid:
        gap_px = abs(b["cy"] - a["cy"]) if kind.startswith("H") else abs(b["cx"] - a["cx"])
        print(f"  [{seg.id}] {seg.orientation}  rect={[round(v,1) for v in seg.rect]}"
              f"  long={seg.long_pt:.1f}pt  short={seg.short_pt:.1f}pt"
              f"  gap={gap_px:.0f}px")

    new_segs = extract_raster_ducts(pdf_path, p1_segs)
    print(f"\n{'─'*70}")
    print(f"New segments after dedup against Phase 1: {len(new_segs)}")
    print(f"{'─'*70}")
    for seg in new_segs:
        print(f"  [{seg.id}] {seg.orientation}  rect={[round(v,1) for v in seg.rect]}")

    print(f"\nOverall: {'PASS (0 new, all in Phase 1)' if len(new_segs) == 0 else f'WARNING: {len(new_segs)} raster-only segments found'}")


if __name__ == "__main__":
    main()
