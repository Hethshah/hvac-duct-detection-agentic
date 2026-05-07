# HVAC Duct Detection v2 — End-to-End Process

This document explains every step the pipeline takes from a raw PDF mechanical drawing to the final annotated image, interactive web output, and structured data files.

---

## Overview

```
PDF Drawing
    │
    ▼
Phase 1 — Vector Duct Extraction
    │  Reads PDF paths directly from CAD geometry
    │  Outputs: 10–20 duct segment rectangles with exact coordinates
    ▼
Phase 2 — Scale Calibration & Label Extraction
    │  Reads dimension text from the PDF
    │  Outputs: pt_per_ft scale + length/cross-section/ID labels
    ▼
Phase 3 — Raster Fallback
    │  OpenCV morphology on rendered page (catches non-quad ducts)
    │  Outputs: additional segments (usually 0 for clean CAD exports)
    ▼
Phase 4 — Duct-Label Association
    │  Matches labels to segments, computes physical lengths
    │  Outputs: AnnotatedDuct list with real-world measurements
    ▼
Phase 5 — Vision Cross-Validation (optional)
    │  Claude Opus reviews low-confidence or mismatched ducts
    │  Outputs: updated confidence scores
    ▼
Phase 6 — Rendering
    │  Draws blue centerlines on a PDF page render
    │  Outputs: annotated PNG + annotated PDF
    ▼
Final Output
  ├── input1_annotated.png       Annotated floor plan image
  ├── input1_annotated.pdf       Annotated PDF with vector blue lines
  └── artifacts/
        ├── phase2_labels.json
        ├── phase4_annotated_ducts.json
        └── summary.json
```

---

## Input

A standard mechanical floor plan PDF exported from CAD software (AutoCAD, Revit, etc.).

Requirements:
- PDF format (single or multi-page; pipeline processes page 0 by default)
- Vector-drawn ducts (lines drawn as actual vector paths, not scanned images)
- Black duct lines (RGB < 0.15) — grey lines are structural walls and are ignored
- Dimension labels in standard HVAC notation (e.g. `10'-0"`, `24x18`, `12"Ø`)

---

## Phase 1 — Vector Duct Extraction

**What it does:** Reads the raw PDF drawing paths directly using PyMuPDF and finds the duct rectangles from the CAD geometry — no image recognition, no LLM coordinate guessing.

**Why this works:** HVAC mechanical drawings are CAD exports. Every duct wall is stored as an exact vector path with precise coordinates. We read those coordinates directly rather than trying to infer them from a rendered image.

**Steps:**

1. Open the PDF with PyMuPDF (`fitz.open`) and read all drawing paths via `page.get_drawings()`
2. Filter to black-stroke paths only — duct walls are black (RGB < 0.15); grey lines (~0.4) are structural walls and are discarded
3. Discard annotation hairlines (stroke weight < 0.5 pt) — duct walls are 1.0–1.4 pt; dimension lines are 0.18 pt
4. Keep only paths that form rectangular outlines (straight-line quad shapes)
5. Apply duct size filters to the bounding box:
   - Short side (duct width): 10–50 PDF points (~4"–24" physical)
   - Long side (duct length): minimum 30 points (~1.2 ft)
   - Aspect ratio: at least 2:1 (ducts are always elongated)
6. Exclude the title block region (bottom of the drawing)
7. Cluster collinear fragments — ducts drawn with gaps at fittings/labels are merged into a single segment
8. Detect diagonal ducts separately using parallelogram geometry; their true centerline is computed from the midpoints of the two short sides

**Output:** A list of `DuctSegment` objects, each with exact PDF-coordinate bounding box, orientation (H/V/D), and measured long-axis length in PDF points.

**Actual result for input1.pdf:** 18 segments — 6 horizontal, 10 vertical, 2 diagonal.

---

## Phase 2 — Scale Calibration & Label Extraction

**What it does:** Reads all text from the PDF and extracts dimension labels, cross-section labels, and duct IDs. Uses matched label+duct pairs to derive the drawing scale.

**Steps:**

1. Extract all text spans from the PDF via `page.get_text("dict")`
2. Parse each text span with regex patterns:
   - **Length labels**: `10'-0"`, `18'-5"` → feet and inches
   - **Cross-section rectangular**: `24x18`, `24X18` → width × height in inches
   - **Cross-section round**: `12"Ø`, `18Ø` → diameter in inches
   - **Duct IDs**: `C01`, `SA12`, `RA3` → alphanumeric identifiers
3. Multi-span fragments (labels split across PDF text runs) are merged if within 20 pt
4. **Scale derivation:** For each length label, find the Phase 1 duct segment whose physical length most closely matches. Compute `pt_per_ft = segment.long_pt / label_feet` for each matching pair. Take the median of the best-matching pairs.
5. Associate each label to its nearest duct segment by centroid distance

**Why scale derivation matters:** Most HVAC drawings do not have an explicit numeric scale. The scale is derived empirically from the agreement between label text and measured segment lengths.

**Output:** Calibrated `pt_per_ft` value (~24.2 for input1.pdf, equivalent to approximately ½"=1'-0") plus a list of parsed labels with their coordinates and nearest duct.

---

## Phase 3 — Raster Fallback

**What it does:** Catches any ducts that were not encoded as quad paths in the PDF — for example, ducts drawn as two separate parallel line strokes rather than a filled rectangle.

**Steps:**

1. Render the PDF page at 4× scale (288 DPI) using PyMuPDF
2. Build a black-pixel mask (all pixels with RGB < 60 in each channel)
3. Apply horizontal morphological opening (60-pixel-wide kernel) to find horizontal black lines
4. Apply vertical morphological opening (60-pixel-tall kernel) to find vertical black lines
5. Find contours; filter to lines longer than 100 px
6. Pair parallel lines that are 4–60 pt apart with ≥60% overlap along their shared axis
7. Deduplicate against Phase 1: discard any raster candidate whose bounding box overlaps a Phase 1 segment by more than 50% IoU

**Output:** Additional `DuctSegment` objects tagged with `source: "raster"`. For clean CAD exports like input1.pdf, this typically adds 0 new segments.

---

## Phase 4 — Duct-Label Association & Measurement

**What it does:** Binds each duct segment to its corresponding labels and computes real-world physical dimensions.

**Steps:**

1. **Length label matching:** For each length label, find ducts whose measured `long_pt` is within ±30% of `label_feet × pt_per_ft`. Among those within 200 pt centroid distance, use greedy bipartite matching (closest first, each label and each duct consumed at most once). This length plausibility gate prevents compound room labels (e.g. a `9'-0"` annotation referring to a combined duct run) from incorrectly matching individual shorter segments.

2. **Cross-section matching:** Match each cross-section label to the nearest duct centroid within 150 pt.

3. **Duct ID matching:** Match each ID label to the nearest duct centroid within 150 pt.

4. **Measurement computation:**
   - `length_ft_measured = segment.long_pt / pt_per_ft`
   - `length_mismatch = True` if `|measured − label| / label > 15%`
   - `unlabeled = True` if neither length nor cross-section label was found

5. Build `AnnotatedDuct` for each segment containing all associated metadata.

**Output:** `phase4_annotated_ducts.json` — one record per duct segment with measured and label-derived lengths, cross-section, ID, mismatch flag, and confidence.

**Actual result for input1.pdf:**
- 2 ducts with matched length labels
- 1 duct with matched cross-section (24×18)
- 1 duct with a length mismatch flag
- 15 unlabeled ducts (measured length only)

---

## Phase 5 — Vision Cross-Validation

**What it does:** Uses Claude Opus to verify ambiguous duct detections — ducts with low geometric confidence or a length label mismatch. Vision is used only for semantic validation, never for coordinate estimation.

**Trigger conditions:** A duct is sent for vision review if:
- Confidence < 0.80 (set during Phase 1/3 extraction for borderline geometry), OR
- `length_mismatch = True` (measured length disagrees with label by more than 15%)

**Steps:**

1. For each flagged duct, render a cropped image: the duct's bounding rect plus 100 pt padding on all sides, at 4× scale
2. Send the crop to Claude Opus (`claude-opus-4-7`) with a prompt describing:
   - What a duct looks like (two parallel black lines forming a rectangle)
   - That grey lines are structural walls to ignore
   - What to confirm (is this a duct run? is it oriented correctly?)
3. Parse the JSON response: `{is_duct, confidence, notes}`
4. Update the duct's confidence:
   - If `is_duct: true` → set confidence to the vision confidence
   - If `is_duct: false` → set confidence to `1.0 - vision_confidence`

**Output:** Same AnnotatedDuct list with updated confidence scores and vision review log.

**For input1.pdf:** Only ducts with a length mismatch are reviewed (all Phase 1 vector ducts have confidence = 1.0 by default; there are no raster-only low-confidence candidates).

Use `--skip-vision` in the CLI to bypass this phase and avoid API calls.

---

## Phase 6 — Rendering

**What it does:** Produces the final visual outputs — an annotated PNG and an annotated PDF.

**Annotated PNG:**

1. Render the PDF page at 72 DPI using PyMuPDF (1 pixel = 1 PDF point)
2. For each duct in the validated list:
   - Compute the centerline endpoints in pixel coordinates, applying the page's rotation transform (input1.pdf has 270° rotation)
   - For H ducts: horizontal line at y-midpoint, spanning x0→x1
   - For V ducts: vertical line at x-midpoint, spanning y0→y1
   - For D (diagonal) ducts: line from the midpoint of one short side to the midpoint of the opposite short side (computed from the actual polygon vertices)
3. Draw blue centerline (`#3762E3`, 3 px wide) using Pillow
4. Draw numbered label box above each duct midpoint with a thin leader line
5. Save as `{stem}_annotated.png`

**Annotated PDF:**

1. Reopen the original PDF
2. For each duct, draw the same centerline directly on the PDF page using PyMuPDF's `page.draw_line()` — this produces native vector graphics, not a rasterized overlay
3. Save as `{stem}_annotated.pdf`

The PDF version preserves full vector quality. The PNG version is suitable for display and sharing.

**Coordinate transform for 270° rotation:**

The PDF page is physically rotated 270°. PyMuPDF reports paths in media coordinates (pre-rotation). To place annotations correctly on the rendered image:

```
pixel_x = media_y × scale
pixel_y = (media_width − media_x) × scale
```

---

## Output Files

```
outputs/{stem}/
├── {stem}_annotated.png        Final annotated floor plan (PNG, 72 DPI)
├── {stem}_annotated.pdf        Annotated PDF with vector blue centerlines
└── artifacts/
      ├── phase2_labels.json    All parsed text labels with coordinates
      ├── phase4_annotated_ducts.json  One record per duct (measurements + labels)
      └── summary.json          Run summary (scale, segment counts, all ducts)
```

### `phase4_annotated_ducts.json` schema

```json
{
  "pt_per_ft": 24.2389,
  "duct_count": 18,
  "annotated_ducts": [
    {
      "id": "C03",
      "duct_idx": "duct_003",
      "rect": [1259.2, 491.5, 1709.0, 561.5],
      "orientation": "H",
      "length_ft_measured": 18.556,
      "length_ft_label": 18.4167,
      "length_mismatch": false,
      "cross_section": null,
      "is_round": false,
      "unlabeled": false,
      "confidence": 1.0,
      "source": "vector",
      "page": 0
    }
  ]
}
```

### `summary.json` schema

```json
{
  "input": "input1.pdf",
  "pt_per_ft": 24.2389,
  "segment_counts": {
    "vector": 18,
    "raster": 0,
    "total": 18
  },
  "annotation": {
    "with_length_label": 2,
    "with_cross_section": 1,
    "length_mismatches": 1,
    "unlabeled": 15
  },
  "annotated_ducts": [ ... ]
}
```

---

## Web Application

The interactive web app runs the same 6-phase pipeline and presents the results as a hoverable floor plan.

**What the web app adds:**

- **Duct type inference** — inferred from the duct label prefix: `SA`/`S-` → Supply, `RA`/`R-` → Return, `EA`/`EF` → Exhaust. Defaults to Supply when no typed label is found.
- **Pressure class inference** — inferred from cross-section dimensions: ≥24" max dimension → Low, 12–24" → Medium, <12" → High.
- **Interactive SVG overlay** — transparent hit-area lines overlaid on the PNG. Hovering a duct shows a tooltip card.
- **Tooltip content:** Type of duct, Pressure class, Section type (Round/Rectangular), Dimension, Length in ft
- **Duct list panel** — scrollable list of all detected ducts, filterable by type/pressure. Clicking a card pans the plan to that duct.
- **Download buttons** — direct download of the annotated PNG and PDF.

---

## Key Design Decisions

**Why not use LLM vision for coordinate detection?**
LLMs cannot produce pixel-accurate coordinates. Even with careful prompting, the coordinates returned by a vision model are approximations. HVAC drawings are CAD exports — the exact geometry already exists as vector data in the PDF. We read it directly.

**Why 72 DPI for rendering?**
At 72 DPI, 1 pixel = 1 PDF point. This makes the coordinate transform trivial and matches the scale at which the duct rects were measured. The output image is 2592×1728 pixels for input1.pdf — high enough for clear display.

**Why is Phase 3 (raster fallback) needed?**
Not all HVAC PDFs are clean CAD exports. Some are scanned or have ducts drawn as separate line strokes rather than quads. Phase 3 handles those cases using OpenCV morphology, then deduplicates against Phase 1 to avoid double-counting.

**Why use greedy matching with a plausibility gate in Phase 4?**
Many length annotations in HVAC drawings refer to composed duct runs (the total length of a branch), not individual extracted segments. A 30% plausibility gate filters out these "wrong size" matches before the greedy distance match runs, preventing incorrect label-duct associations.

---

## Running the Pipeline

### CLI

```bash
cd hvac-duct-detection-v2
python3 run.py "../sample input/input1.pdf" --skip-vision
```

With vision validation:
```bash
python3 run.py "../sample input/input1.pdf"
```

Custom output directory:
```bash
python3 run.py "../sample input/input1.pdf" --skip-vision --output-dir /path/to/outputs
```

### Web Application

```bash
cd hvac-duct-detection-v2
python3 web_app.py
```

Open `http://localhost:5000` in your browser. Upload a PDF, click **Analyze Drawing**, then hover the blue annotation lines to inspect each duct.

### Tests

```bash
cd hvac-duct-detection-v2
python3 -m pytest tests/ -v
```

157 tests across 6 phases (unit + integration).

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `pymupdf` | ≥ 1.27 | PDF vector extraction, text reading, rendering, annotation writing |
| `opencv-python` | ≥ 4.13 | Phase 3 raster morphology |
| `numpy` | any | Phase 3 image arrays |
| `anthropic` | any | Phase 5 Claude Opus vision API |
| `Pillow` | any | PNG rendering and centerline drawing |
| `flask` | any | Web application server |

Install:
```bash
pip install -r requirements.txt
pip install flask
```
