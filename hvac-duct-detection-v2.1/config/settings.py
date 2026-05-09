from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
SAMPLE_INPUT = PROJECT_ROOT.parent / "sample input" / "input1.pdf"
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"

# ── Phase 1: Vector extraction thresholds (all in PDF points) ─────────────────
# Black-stroke colour gate: all RGB channels must be below this value
BLACK_MAX_CHANNEL = 0.15

# Duct bounding-box filters
DUCT_MIN_ASPECT    = 2.0    # must be at least this elongated (long/short)
DUCT_MIN_SHORT_PT  = 10.0   # duct width lower bound — excludes wall hairlines (≤9pt), catches round ducts (≥11.9pt)
DUCT_MAX_SHORT_PT  = 50.0   # duct width upper bound  (~24 inch physical max at 25 pt/ft)
DUCT_MIN_STROKE_PT = 0.5    # minimum path stroke weight — excludes annotation/dimension hairlines (0.18pt)
                             # 81pt+ indicates an equipment casing (AHU, VAV box), not a duct wall
DUCT_MIN_LONG_PT   = 30.0   # minimum duct run length (~1.2 ft at 25 pt/ft)

# Clustering: merge collinear quads into a single run
CLUSTER_SHORT_OVERLAP_MIN = 0.80  # fraction of shorter segment's short-side that must overlap
CLUSTER_LONG_GAP_MAX_PT   = 8.0   # max gap along long axis to still merge (fittings/labels gap)
CLUSTER_SHORT_RATIO_MAX   = 2.5   # don't merge if short sides differ by more than this factor
                                   # prevents inner duct (27pt) merging with outer casing (81pt)

# Title-block exclusion for input1.pdf (y > this in PDF points = title block region)
TITLE_BLOCK_Y_MIN_PT = 2200.0

# ── Phase 2: Scale calibration ────────────────────────────────────────────────
PT_PER_FT_EXPECTED   = 24.7   # empirical from input1.pdf; used as seed
PT_PER_FT_MIN        = 22.0
PT_PER_FT_MAX        = 27.0
SCALE_MATCH_TOLERANCE = 0.20  # ±20% when pairing label length to duct length

# ── Phase 3: Raster fallback ──────────────────────────────────────────────────
RASTER_SCALE         = 4       # render scale factor (4× ≈ 288 DPI)
BLACK_PIXEL_MAX      = 60      # each BGR channel must be <= this to be "black"
MORPH_H_KERNEL_W     = 60      # horizontal line morphology kernel width (px)
MORPH_V_KERNEL_H     = 60      # vertical line morphology kernel height (px)
PARALLEL_GAP_MIN_PX  = 16      # min gap between parallel lines (px @ 4× scale)
PARALLEL_GAP_MAX_PX  = 240     # max gap (= 60 pt = ~2.4 inch physical)
PARALLEL_OVERLAP_MIN = 0.60    # min fractional x/y overlap to pair lines
DEDUP_IOU_THRESHOLD  = 0.50    # IoU above which raster duct is dropped (covered by vector)
RASTER_DEDUP_MARGIN_PT = 50.0  # also drop raster cand if its centroid is within this pt of any P1 bbox edge

# ── Phase 4: Duct-label association ──────────────────────────────────────────
LABEL_LENGTH_MAX_PT      = 200.0 # max centroid distance for length label → duct match
LABEL_LENGTH_PLAUSIBILITY = 0.30 # length label rejected if |seg_pt - expected_pt|/expected > this
LABEL_CENTROID_MAX_PT   = 150.0 # max centroid distance for cross-section / ID label → duct
LABEL_MISMATCH_THRESHOLD = 0.15 # flag mismatch if |measured-label|/label > this

# ── Phase 5: Vision validation ────────────────────────────────────────────────
VISION_CONFIDENCE_THRESHOLD = 0.80   # ducts below this are sent for vision check
VISION_CROP_MARGIN_PT       = 100.0  # padding around duct rect for vision crop
VISION_MODEL                = "claude-opus-4-7"

# ── Phase 6: Annotation rendering ────────────────────────────────────────────
RENDER_DPI           = 72              # output PNG DPI (72 = screen resolution, ~2592×1728 px)
OUTLINE_COLOR_RGB    = (55, 98, 227)   # duct wall stroke colour (matches expected output)
OUTLINE_WIDTH_PX     = 3              # stroke width in pixels at RENDER_DPI
LABEL_FONT_SIZE_PX   = 13             # label box number font size
LABEL_OFFSET_PT      = 20.0           # perpendicular offset from duct to label anchor
