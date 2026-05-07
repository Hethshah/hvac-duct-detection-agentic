# HVAC Duct Detection v2 — Technical Approach

## Why v1 Failed

All v1 attempts used Claude vision (LLM) to *estimate* pixel coordinates for duct bounding boxes. This is fundamentally wrong because:

1. LLM coordinate guesses are always approximate → misaligned overlays
2. No way to enforce pixel-accuracy from a language model
3. Aspect ratio filter (4:1) eliminated valid ducts
4. OpenCV parallel-line pairing on raster produced massive false positives from the page border

## Core Insight

HVAC mechanical drawings are exported from CAD software as **100% vector PDFs**. Every duct rectangle is a **quad path (`'qu'`)** with exact point coordinates. We extract these directly — no vision-based coordinate guessing at all. Vision is only used in Phase 5 for semantic disambiguation (duct vs. fitting).

## PDF Analysis Results (input1.pdf)

| Property | Value |
|---|---|
| Type | 100% vector (CAD export) |
| Total drawing paths | 20,886 |
| Black-stroke paths (ducts + structure) | 9,376 |
| Grey-stroke paths (walls, ignore) | ~6,910 |
| Black rect primitive type | `'qu'` (quad) — NOT `'re'` (critical!) |
| Duct candidates via vector extraction | ~30 clean candidates |
| Page rotation | 270° |
| Scale notation | No explicit label — derived empirically |
| Calibrated scale | ~24–25 pt/ft (~1/2"=1'-0") |
| Length labels in PDF text | 7 (e.g. `10'-0"`, `18'-5"`, `12'-6"`) |
| Duct IDs in PDF text | C01, C02, C03, C05 etc. |

## Drawing Conventions

- **Black lines/rectangles = duct walls** (RGB < 0.15 each channel)
- **Grey lines = walls, structural elements → ignore entirely** (RGB ~0.4)
- Round duct label format: `N"Ø` or `NØ` (e.g. `18"Ø`, `12"Ø`)
- Rectangular duct label format: `N"xM"` or `NxM` (e.g. `22"x14"`, `16"x10"`)
- Length labels: foot-inch format (e.g. `10'-0"`, `18'-5"`)
- Duct IDs: 1-3 letters + 1-3 digits (e.g. `C01`, `SA12`, `RA3`)

## Coordinate System

PyMuPDF (`fitz`) is the authoritative coordinate source.

```
pixel_x = pdf_x_pt * render_scale
pixel_y = pdf_y_pt * render_scale
render_scale = DPI / 72  (e.g. 300 DPI → scale = 4.167)
```

- PyMuPDF uses **top-left origin**, pixmap matches directly — no y-flip needed
- Always read `page.rotation` first; work in un-rotated PDF coords throughout
- **Do NOT use pdfplumber for geometry** — it applies the 270° rotation and reverses text span order, causing coordinate bugs

---

## Phase 1 — Vector Duct Extraction ✅ COMPLETE

### Goal
Extract duct segments from PDF using native vector quad paths. No rasterization, no LLM.

### Input
- PDF file path
- Page index (default: 0)

### Algorithm

#### Step 1 — Collect axis-aligned candidate rects (`_extract_candidate_rects`)
1. Open with PyMuPDF: `doc = fitz.open(pdf_path)`
2. Get all drawing paths: `paths = page.get_drawings()`
3. Filter to black-stroke only: all RGB channels < `BLACK_MAX_CHANNEL = 0.15`
4. Filter out annotation hairlines: `path['width'] < DUCT_MIN_STROKE_PT = 0.5` → discard
   - Real duct walls: 1.08–1.44 pt stroke. Annotation/dimension lines: 0.18 pt → excluded.
5. Keep paths whose shape is a simple rectangular outline (`_path_is_simple_rect_shape`):
   - **Pure `'qu'`** (quad) items — the primary duct wall encoding in this PDF
   - **Pure `'l'`** (≤6 items) that are axis-aligned and form ≥2 sides of the bbox — catches ducts drawn as separate line strokes
   - **Mixed `'qu'`+`'l'`** (≤3 items) — the `'qu'` guarantees rectangularity; `'l'` is a detail line
   - Paths with diagonal `'l'` segments or Bezier `'c'` items are **excluded** from axis-aligned detection (they distort the clustering of adjacent correct segments)
6. Deduplicate by rounded bbox

#### Step 2 — Apply duct heuristics (`_segment_from_rect`)
For each candidate rect compute `w`, `h`, `long_pt = max(w,h)`, `short_pt = min(w,h)`:
- `long_pt >= DUCT_MIN_LONG_PT = 30.0`
- `DUCT_MIN_SHORT_PT = 10.0 ≤ short_pt ≤ DUCT_MAX_SHORT_PT = 50.0`
  - Lower bound 10.0 excludes wall hairlines (9.0 pt) while keeping 8"ø round ducts (11.9 pt)
  - Upper bound 50.0 excludes AHU/VAV equipment casings (81 pt+)
- `aspect = long_pt / short_pt >= DUCT_MIN_ASPECT = 2.0`
- Exclude title-block region: `y > TITLE_BLOCK_Y_MIN_PT = 2200.0 pt`
- Set orientation: `"H"` if `w >= h`, else `"V"`

#### Step 3 — Cluster collinear fragments (`_cluster_segments`)
Iteratively merge pairs that satisfy all three:
- Same orientation (H with H, V with V)
- Short-axis overlap `>= CLUSTER_SHORT_OVERLAP_MIN = 0.80` of the shorter segment's span
- Long-axis gap `<= CLUSTER_LONG_GAP_MAX_PT = 8.0 pt` (covers fitting/label gaps)
- Short-side ratio `<= CLUSTER_SHORT_RATIO_MAX = 2.5` (prevents inner duct merging with outer casing)

#### Step 4 — Detect diagonal ducts (`_extract_diagonal_duct_paths`)
Separate pipeline for non-axis-aligned duct geometry:
- **Parallelogram `'l'` paths**: 4 unique edges (after deduplication — some paths draw one edge twice), at least one diagonal, two pairs of parallel equal-length edges, inner aspect ≥ 2.0
  - Edge deduplication is essential: some PDF paths contain 5 `'l'` items where one edge appears twice in both directions
- **Rotated `'qu'` quads**: single `'qu'` item where corners are not axis-aligned
- Bbox filters (calibrated to exclude VAV X-marks and page borders):
  - Min bbox dimension ≥ 60 pt, max ≤ 250 pt, bbox aspect < 1.8
- Emit with `orientation = "D"` and actual polygon vertices for accurate overlay rendering

### Calibrated Constants (input1.pdf)

| Constant | Value | Reason |
|---|---|---|
| `BLACK_MAX_CHANNEL` | 0.15 | Separates black duct lines from grey walls (~0.4) |
| `DUCT_MIN_SHORT_PT` | 10.0 | Gap between wall hairline (9.0 pt) and 8"ø duct (11.9 pt) |
| `DUCT_MAX_SHORT_PT` | 50.0 | Max duct ~24" physical at 25 pt/ft |
| `DUCT_MIN_LONG_PT` | 30.0 | ~1.2 ft minimum run |
| `DUCT_MIN_ASPECT` | 2.0 | Ducts are always elongated |
| `DUCT_MIN_STROKE_PT` | 0.5 | Excludes 0.18 pt annotation hairlines |
| `CLUSTER_LONG_GAP_MAX_PT` | 8.0 | Max fitting/label gap along a duct run |
| `CLUSTER_SHORT_OVERLAP_MIN` | 0.80 | Fraction of short-side that must align |
| `CLUSTER_SHORT_RATIO_MAX` | 2.5 | Prevents inner duct merging with outer casing |
| `TITLE_BLOCK_Y_MIN_PT` | 2200.0 | Title block starts here in input1.pdf |

### What Was Tried and Reverted

- **Bezier arc `'c'` path acceptance**: single-arc paths (elbow fittings) pass geometric filters but, when added as candidate rects, they feed into the clustering step and distort existing correct segment bounding boxes. Reverted.
- **`DUCT_MIN_ASPECT = 1.8`**: needed to catch one mixed `'qu'+'l'` transition at aspect 1.85, but same clustering side-effect problem — caused an adjacent H segment's short-side to grow from 20.9 pt to 34.4 pt. Reverted.
- Curved elbow/transition fittings (3 Bezier arcs + 1 mixed path) are detectable geometry but cannot be added at Phase 1 without corrupting axis-aligned segment clustering. Deferred to a downstream phase.

### Output
```json
{
  "pdf": "...",
  "page": 0,
  "segment_count": 18,
  "segments": [
    {
      "id": "duct_001",
      "rect": [x0, y0, x1, y1],
      "orientation": "H",
      "long_pt": 252.4,
      "short_pt": 18.0,
      "aspect": 14.0,
      "centerline": [[x0, cy], [x1, cy]],
      "source": "vector",
      "confidence": 1.0,
      "page": 0
    }
  ]
}
```
File: `outputs/{pdf_stem}/phase1_vector_ducts.json`

### Actual Results (input1.pdf)
- **18 segments total**: 10 vertical, 6 horizontal, 2 diagonal
- **Longest**: duct_003 V 449.8 pt (~18.2 ft, C03 label)
- **Shortest**: duct_015 H 36.9 pt (~1.5 ft, small fitting stub)
- All 28 unit + integration tests passing

### Validation Criteria
- Total segments = 18 for `input1.pdf`
- Zero segments in title-block region (`y > 2200 pt`)
- All H/V segments: `aspect >= 2.0`
- Longest segment: 380–560 pt range
- All segment IDs unique and sequential (`duct_001` … `duct_018`)
- All rects: `x0 < x1`, `y0 < y1`

### Test Cases
| Test | Expected |
|---|---|
| `input1.pdf` segment count | >= 12 segments |
| Largest long_pt segment | ~450 pt (C03, 18'-5") |
| Smallest valid duct | long >= 30 pt, short >= 10 pt |
| Title block region (y>2200) | 0 segments |
| Grey-stroke paths kept | 0 |
| Aspect ratio of all H/V results | all >= 2.0 |
| Segment IDs | sequential, unique |

### Key Files
- `tools/vector_duct_extractor.py` — extraction + clustering + diagonal detection
- `models/duct_segment.py` — `DuctSegment` dataclass
- `config/settings.py` — all calibrated constants
- `tests/test_phase1.py` — 28 tests (unit + integration)
- `scripts/exploratory/dump_ducts.py` — debug PNG + console table

---

## Phase 2 — Text & Scale Calibration

### Goal
Extract dimension labels, length labels, and duct IDs from PDF native text. Derive calibrated `pt_per_ft`.

### Input
- PDF file path
- Phase 1 duct segments JSON

### Algorithm
1. Extract all text spans: `page.get_text("dict")['blocks']` via PyMuPDF
2. For each span collect `(text, bbox, center_x, center_y)`
3. Regex passes:
   - **Length labels**: `r"(\d+)\s*['']\s*-?\s*(\d*)\s*[\""]*"` → `(feet, inches)`
   - **Cross-section rect**: `r"(\d+)\s*[xX×]\s*(\d+)"` → `(width_in, height_in)`
   - **Cross-section round**: `r"(\d+)[\""]?\s*[Øø⌀]"` → `diameter_in`
   - **Duct IDs**: `r"^[A-Z]{1,3}\d{1,3}$"` (C01, SA12, RA3)
4. Handle multi-span label fragments (merge adjacent spans within 20 pt gap)
5. Scale calibration:
   - For each length label, find nearest Phase 1 duct whose `long_pt` is within ±20% of `label_feet * 25`
   - Compute `pt_per_ft = duct.long_pt / label_feet` for each match
   - Take **median** of top-5 pairs as `pt_per_ft`
6. Associate each label to nearest duct (centroid within 150 pt)

### Output
```json
{
  "pt_per_ft": 24.7,
  "scale_text": "derived",
  "labels": [
    {
      "text": "10'-0\"",
      "type": "length",
      "feet": 10.0,
      "cx": 840.0,
      "cy": 312.0,
      "nearest_duct_id": "duct_003"
    },
    {
      "text": "24X18",
      "type": "cross_section",
      "width_in": 24,
      "height_in": 18,
      "cx": 1240.0,
      "cy": 824.0,
      "nearest_duct_id": "duct_007"
    }
  ]
}
```
File: `outputs/{pdf_stem}/phase2_labels.json`

### Validation Criteria
- `pt_per_ft` in range [22, 27] for `input1.pdf`
- 7/7 length labels parsed for `input1.pdf`
- At least 1 cross-section label parsed (`24X18` from `SC-24X18X8.62BOX`)
- Per-pair calibration residuals < 15% (i.e., no single pair wildly off)
- Each length label associates to exactly 1 duct within 150 pt

### Test Cases
| Test | Expected |
|---|---|
| `pt_per_ft` value | 24–25 (±1) |
| Length labels found | 7 |
| `C03` label → duct long_pt | ~450 pt (18'-5" × 24.4 ≈ 450) |
| `C01` label → duct long_pt | ~252 pt (10'-0" × 25.2 ≈ 252) |
| Cross-section labels | >= 1 |
| Labels with no duct within 150pt | 0 (flag as warning if any) |

### Key File
`tools/label_extractor.py` + `scripts/exploratory/calibrate_scale.py`

---

## Phase 3 — Raster Fallback (Parallel-Line Ducts)

### Goal
Catch ducts drawn as separate parallel lines (not quads) for robustness on other PDFs. Deduplicate against Phase 1.

### Input
- PDF file path
- Phase 1 duct segments JSON (for deduplication)

### Algorithm
1. Render page at 4× scale (`fitz.Matrix(4, 4)` → 288 DPI)
2. Convert to BGR; build black mask: `cv2.inRange(img, (0,0,0), (60,60,60))`
3. **Mask title block**: detect the outermost long horizontal line; crop mask to interior
4. Horizontal line detection: `cv2.morphologyEx(mask, MORPH_OPEN, (60,1) kernel)`
5. Vertical line detection: `cv2.morphologyEx(mask, MORPH_OPEN, (1,60) kernel)`
6. Find contours; extract bounding rects for lines `width > 100 px` (H) or `height > 100 px` (V)
7. Parallel-pair matching:
   - Gap between lines: 16–240 px (= 4–60 pt physical duct width)
   - x/y-overlap: >= 60% of the shorter line
   - Endpoint alignment: both start/end within 30 px
8. Convert px → PDF pt by dividing by 4
9. Deduplicate: discard any raster duct with IoU > 0.5 against a Phase 1 duct

### Output
Same schema as Phase 1, with `"source": "raster"` field added.
File: `outputs/{pdf_stem}/phase3_raster_ducts.json` (merged with Phase 1 result)

### Validation Criteria
- On `input1.pdf`: 0–3 new candidates added after dedup (pure-quad PDF, Phase 1 covers all)
- On a synthetic line-only test PDF: all line-drawn ducts found
- No title-block false positives (y > boundary → excluded)
- All returned pairs have gap 16–240 px and overlap >= 60%

### Test Cases
| Test | Expected |
|---|---|
| `input1.pdf` new segments added | 0–3 |
| Page border lines as duct | 0 (masked out) |
| Title-block horizontal lines as ducts | 0 (masked out) |
| Parallel pair gap range | 16–240 px |

### Key File
`tools/raster_duct_extractor.py`

---

## Phase 4 — Duct-Label Association & Measurement

### Goal
Bind each duct to its ID, length label, cross-section label. Compute real-world physical dimensions.

### Input
- Merged duct segments (Phase 1 + Phase 3)
- Phase 2 labels JSON
- Calibrated `pt_per_ft`

### Algorithm
1. For each duct segment:
   a. **Length label association**: project each length label's center onto the duct's centerline; keep label with perpendicular distance < 80 pt and minimum along-axis distance
   b. **Cross-section label association**: nearest cross-section label within 150 pt of duct centroid
   c. **Duct ID association**: nearest duct-ID text within 150 pt that co-occurs near the same length label
2. Compute physical length: `length_ft = segment.long_pt / pt_per_ft`
3. Flag `length_mismatch = True` if `|measured - label| / label > 0.15`
4. For ducts without any label: mark `unlabeled = True`, still emit measured length
5. Build `AnnotatedDuct` per segment

### Output
```json
[
  {
    "id": "C03",
    "duct_idx": "duct_005",
    "rect": [x0, y0, x1, y1],
    "orientation": "H",
    "length_ft_measured": 18.4,
    "length_ft_label": 18.417,
    "length_mismatch": false,
    "cross_section": {"width_in": 24, "height_in": 18},
    "is_round": false,
    "unlabeled": false,
    "confidence": 0.95
  }
]
```
File: `outputs/{pdf_stem}/phase4_annotated_ducts.json`

### Validation Criteria
- 7/7 known label-duct pairs associated, all with `length_mismatch = False`
- `24×18` cross-section bound to correct duct
- No duct assigned two conflicting length labels
- Measured vs. label lengths within 15% for all matched ducts
- `unlabeled` count <= total_ducts - 7 for `input1.pdf`

### Test Cases
| Test | Expected |
|---|---|
| C03 length_ft_measured | 18.3–18.5 ft |
| C01 length_ft_measured | 9.9–10.1 ft |
| length_mismatch count | 0 for all 7 labelled ducts |
| Cross-section `24x18` assigned | to exactly 1 duct |
| Ducts with perpendicular label dist > 80pt | flagged, not associated |

### Key File
`tools/duct_annotator.py` + `outputs/{pdf_stem}/phase4_annotated_ducts.json`

---

## Phase 5 — Vision Cross-Validation

### Goal
Use Claude vision ONLY to validate ambiguous detections (aspect ~1, junctions). Never for coordinate estimation.

### Input
- Phase 4 annotated ducts JSON
- PDF file path (for crop rendering)

### Algorithm
1. Filter to ducts with `confidence < 0.8` OR `length_mismatch = True`
2. For each flagged duct: render tight crop (duct rect ±100 pt) at 4× via `fitz.Matrix(4,4)`
3. Send crop to Claude vision with updated prompt:
   - "Black rectangles formed by two parallel black lines = duct runs. Grey lines = structural walls, ignore entirely."
   - "Is this crop a duct run or a fitting/equipment (elbow, tee, VAV box, diffuser)?"
   - "If duct: confirm orientation and approximate endpoints."
4. Parse response: update `confidence` and set `fitting_type` if not a duct
5. Remove non-duct entries (fittings, diffusers) from the annotated list

### Output
Updated Phase 4 JSON with `confidence` and optional `fitting_type` field.
File: `outputs/{pdf_stem}/phase5_validated_ducts.json`

### Validation Criteria
- All 6 square (aspect ~1) candidates correctly classified as fittings/diffusers and removed
- All true duct runs retain `confidence >= 0.9` after validation
- Vision never overrides a Phase 1 vector coordinate (only updates `confidence` + `fitting_type`)
- API call count: only flagged ducts, not all ducts

### Test Cases
| Test | Expected |
|---|---|
| VAV box candidates removed | All aspect~1 shapes removed |
| True duct confidence after vision | >= 0.9 |
| Vision modifying rect coordinates | Never (read-only validation) |
| API calls for `input1.pdf` | <= 10 (only low-confidence ducts) |

### Key File
Updated `agents/vision_agent.py`, updated `config/prompts.py`

---

## Phase 6 — Annotation Renderer & End-to-End Pipeline

### Goal
Render annotated PDF/PNG matching expected output. Wire into CLI.

### Input
- Phase 5 validated ducts JSON
- PDF file path
- Calibrated `pt_per_ft`

### Annotation Style (matching expected output)
- Blue outline (`#1565C0`, RGB 21,101,192) traced exactly on each duct rectangle
- Outline width: 3–4 pt in PDF space (≈12 px at 300 DPI)
- Label badge per duct with dark background: `{ID} | {W}×{H} | {length_ft}'`
- Label placement: along duct centerline, offset perpendicular by 20 pt to avoid covering the duct
- For unlabeled ducts: show measured length only

### Algorithm
1. Render base page at 300 DPI: `fitz.Matrix(300/72, 300/72)`
2. For each validated duct:
   a. Draw filled semi-transparent blue rectangle overlay (alpha ~60)
   b. Draw solid blue outline (OUTLINE_COLOR, width 12 px)
   c. Build label string: `f"{id} {w}×{h}  {length_ft:.1f}'"`
   d. Place label badge at centerline midpoint, perpendicular offset
3. Save as PNG: `outputs/{pdf_stem}/annotated.png`
4. Also write annotated PDF using PyMuPDF `page.draw_rect` + `page.insert_text`
5. Write `summary.json` with all duct data + scale + confidence

### Output
```
outputs/{pdf_stem}/
├── phase1_vector_ducts.json
├── phase2_labels.json
├── phase3_raster_ducts.json
├── phase4_annotated_ducts.json
├── phase5_validated_ducts.json
├── annotated.png
├── annotated.pdf
└── summary.json
```

### Validation Criteria
- All 7 known ducts visually annotated with correct length (±0.3 ft)
- No annotations in title-block region
- No annotation on grey-line elements
- Visual match to `sample output/expected output.png`
- `summary.json` contains: duct count, `pt_per_ft`, per-duct `{id, rect, length_ft, cross_section, confidence}`

### Test Cases
| Test | Expected |
|---|---|
| C03 annotation label | `18.4 ft` or `18'-05"` |
| C01 annotation label | `10.0 ft` or `10'-00"` |
| Annotations in title block | 0 |
| `annotated.png` matches expected output | Blue outlines on all duct runs |
| `summary.json` duct count | >= 12 |

### Key Files
`pipelines/run_extract.py`, updated `tools/annotation_tools.py`, `outputs/input1/`

---

## Library Stack

| Library | Version | Purpose |
|---|---|---|
| `pymupdf` | >= 1.27 | PDF vector extraction, text extraction, rendering, annotation writing |
| `opencv-python` | >= 4.13 | Phase 3 raster fallback morphology |
| `numpy` | any | Phase 3 image arrays |
| `anthropic` | existing | Phase 5 vision validation only |
| `Pillow` | existing | PNG export |
| `reportlab` | existing | PDF export fallback |

**Do NOT use pdfplumber for geometry** — rotation (270°) reverses coordinate order and text span sequence.

---

## Critical Gotchas

1. **Page rotation = 270°** — always read `page.rotation` first. Use PyMuPDF coords throughout; never pdfplumber for geometry.
2. **Black rects are `'qu'` not `'re'`** — code filtering on `item[0] == 're'` finds nothing in this PDF.
3. **No explicit scale label** — `pt_per_ft` must be derived empirically from label+duct pairs. Expected value: 24–25 for `input1.pdf`.
4. **Grey paths (RGB ~0.4) must be excluded** — they outnumber black paths and represent walls/structure.
5. **Title block must be masked** before any morphological analysis — the page border is the longest "line" in the image.
6. **Vision is read-only in Phase 5** — it never sets or adjusts coordinates, only updates confidence and fitting_type flags.
7. **Duct label text may be split across multiple spans** — merge adjacent spans within 20 pt gap before regex matching.
