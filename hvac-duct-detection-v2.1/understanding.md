# Understanding HVAC Duct Detection v2.1 — Complete Code Walkthrough

This document explains every file, every library, every method, and the full flow of how the system works — end to end. By the time you finish reading this, you should be able to explain and modify any part of the codebase.

---

## Table of Contents

1. [What does this system do?](#1-what-does-this-system-do)
2. [The core idea: Agentic + Deterministic](#2-the-core-idea-agentic--deterministic)
3. [Directory structure](#3-directory-structure)
4. [Libraries used](#4-libraries-used)
5. [Data models](#5-data-models)
6. [The 8 pipeline tools (deterministic)](#6-the-8-pipeline-tools-deterministic)
   - [Tool 1 — Vector duct extraction](#tool-1--vector-duct-extraction-toolsvector_duct_extractorpy)
   - [Tool 2 — Label & scale extraction](#tool-2--label--scale-extraction-toolslabel_extractorpy)
   - [Tool 3 — Raster fallback](#tool-3--raster-fallback-toolsraster_duct_extractorpy)
   - [Tool 4 — Annotation](#tool-4--annotation-toolsduct_annotatorpy)
   - [Tool 5 — Inspect duct](#tool-5--inspect-duct-agentpy)
   - [Tool 6 — Update assessment](#tool-6--update-assessment-agentpy)
   - [Tool 7 — Set classification](#tool-7--set-classification-agentpy)
   - [Tool 8 — Render output](#tool-8--render-output-agentpy)
7. [The agent loop](#7-the-agent-loop-agentpy)
8. [The CLI runner](#8-the-cli-runner-runpy)
9. [The web application](#9-the-web-application-web_apppy)
10. [The HTML report generator](#10-the-html-report-generator)
11. [The web UI](#11-the-web-ui-templatesindexhtml)
12. [Configuration](#12-configuration-configsettingspy)
13. [Full end-to-end flow](#13-full-end-to-end-flow)
14. [Key concepts glossary](#14-key-concepts-glossary)

---

## 1. What does this system do?

You feed it a PDF of a mechanical floor plan (an HVAC drawing exported from CAD software). It automatically:

1. Finds every duct segment drawn on the page
2. Reads the dimension labels (lengths, cross-sections, duct IDs)
3. Measures the physical length of each duct in feet
4. Classifies each duct as Supply / Return / Exhaust and Low / Medium / High pressure
5. Visually inspects ambiguous ducts using Claude AI
6. Produces an annotated floor plan image (PNG + PDF) and a readable HTML report

The output is a complete duct inventory — the kind that would normally require a human HVAC engineer to sit down with the drawing for hours.

---

## 2. The core idea: Agentic + Deterministic

This is the most important concept in the whole codebase.

### Why not just use AI to read the drawing?

The previous version (v1) asked Claude to look at an image of the floor plan and output pixel coordinates. This failed because **LLMs cannot reliably produce accurate pixel coordinates from images**. The coordinates were off by tens or hundreds of pixels, making annotations misaligned.

### The v2 insight: CAD drawings are already data

HVAC drawings are exported from CAD software (Revit, AutoCAD). Every duct rectangle already exists as an **exact vector path** inside the PDF file with precise coordinates. We don't need AI to find them — we just read the data directly.

### The v2.1 architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Claude Opus (Agent)                    │
│                                                          │
│   "I see 23 segments. 2 have length mismatches.         │
│    Let me inspect duct_003 visually..."                  │
│                                                          │
│   Decides WHAT to do and WHEN                           │
└──────────────┬──────────────────────────────────────────┘
               │ calls tools
               ▼
┌─────────────────────────────────────────────────────────┐
│              Deterministic Tool Functions                │
│                                                          │
│   extract_ducts()    → exact PDF coordinates            │
│   extract_labels()   → exact OCR text positions         │
│   annotate_ducts()   → rule-based label matching        │
│   render_annotated_page() → pixel-accurate drawing      │
│                                                          │
│   All numbers come from these functions, never from AI  │
└─────────────────────────────────────────────────────────┘
```

**Claude is the brain. The tools are the hands. The tools always do the measuring.**

This means:
- Same PDF → same geometric measurements every single time (deterministic)
- Claude adds intelligence: visual inspection, classification, confidence assessment
- No hallucinated coordinates or measurements

---

## 3. Directory structure

```
hvac-duct-detection-v2.1/
│
├── agent.py              ← THE CORE FILE. Defines tools + agent loop.
├── run.py                ← CLI entry point. Run from terminal.
├── web_app.py            ← Flask web server with streaming.
├── requirements.txt      ← Python package dependencies.
│
├── config/
│   └── settings.py       ← All constants and thresholds. Single source of truth.
│
├── models/
│   ├── duct_segment.py   ← DuctSegment dataclass (raw detected geometry)
│   └── annotated_duct.py ← AnnotatedDuct dataclass (with labels + measurements)
│
├── tools/
│   ├── vector_duct_extractor.py  ← Phase 1: reads PDF vector paths
│   ├── label_extractor.py        ← Phase 2: reads PDF text, calibrates scale
│   ├── raster_duct_extractor.py  ← Phase 3: OpenCV image processing fallback
│   ├── duct_annotator.py         ← Phase 4: matches labels to ducts
│   └── annotation_renderer.py   ← Phase 6: draws annotated PNG
│
├── templates/
│   └── index.html        ← Single-page web UI (upload → live log → result)
│
└── outputs/              ← Created at runtime. Contains all output files.
    └── {stem}/
        ├── {stem}_annotated.png
        ├── {stem}_report.html
        └── summary.json
```

---

## 4. Libraries used

### PyMuPDF (`fitz`)
- **What it is:** Python bindings for the MuPDF library, the fastest open-source PDF engine
- **Installed as:** `pymupdf` in requirements.txt, imported as `import fitz`
- **What we use it for:**
  - `fitz.open(pdf_path)` — opens a PDF file
  - `page.get_drawings()` — returns every vector path on the page (lines, rectangles, quads) with their exact coordinates and stroke color
  - `page.get_text("dict")` — returns all text with exact bounding boxes
  - `page.get_pixmap(matrix=fitz.Matrix(scale, scale))` — renders the page to a pixel image
  - `page.draw_line()` — draws a vector line directly on the PDF (used for annotated PDF output)
  - `fitz.Rect(x0, y0, x1, y1)` — a rectangle object used for cropping
  - `fitz.Matrix(scale, scale)` — a scaling transform for rendering

### OpenCV (`cv2`)
- **What it is:** Open Source Computer Vision Library — the industry standard for image processing
- **Installed as:** `opencv-python`
- **What we use it for:** Phase 3 raster fallback — morphological operations to detect duct walls as parallel lines in a rendered image
- **Key methods:**
  - `cv2.getStructuringElement()` — creates a kernel for morphological ops
  - `cv2.morphologyEx()` — applies erosion/dilation to isolate horizontal or vertical lines
  - `cv2.findContours()` — finds the outlines of detected shapes

### NumPy (`numpy`)
- **What it is:** Numerical Python — array operations
- **Used in:** Phase 3 raster processing — image data is a NumPy array of pixel values

### Pillow (`PIL`)
- **What it is:** Python Imaging Library fork — image creation and drawing
- **What we use it for:**
  - `Image.frombytes()` — converts PyMuPDF pixmap data to a PIL Image
  - `ImageDraw.Draw()` — drawing context to draw lines and boxes on images
  - `ImageFont.truetype()` — loads system fonts for label numbers
  - `img.save(output_path, format="PNG")` — writes the final annotated PNG

### Anthropic SDK (`anthropic`)
- **What it is:** Official Python SDK for the Claude API
- **What we use it for:** The agent loop — sending messages to Claude with tool definitions and receiving tool_use responses
- **Key class:** `anthropic.Anthropic()` — creates the API client
- **Key method:** `client.messages.create()` — sends a conversation turn and gets Claude's response

### Flask
- **What it is:** Lightweight Python web framework
- **What we use it for:** The web application server
- **Key concepts:**
  - `@app.route("/path")` — decorator that registers a URL handler
  - `request.files["pdf"]` — accesses the uploaded file
  - `jsonify(dict)` — converts a dict to a JSON HTTP response
  - `Response(generator, mimetype="text/event-stream")` — SSE streaming response
  - `send_file(path)` — serves a file to the browser

### python-dotenv (`dotenv`)
- **What it is:** Loads environment variables from a `.env` file
- **What we use it for:** Loading `ANTHROPIC_API_KEY` from the `.env` file at the repo root so you don't have to `export` it every time

### `re` (Python standard library)
- **What it is:** Regular expressions
- **What we use it for:** Parsing dimension text like `"12' - 6\""` and `"24X18"` from PDF text labels

### `math` (Python standard library)
- **What we use it for:** `math.hypot(dx, dy)` for distance calculations, `math.atan2()` for angle calculations in diagonal duct detection

### `dataclasses` (Python standard library)
- **What we use it for:** `@dataclass` decorator creates classes with auto-generated `__init__`, `__repr__` etc. `field(default_factory=list)` creates mutable defaults safely.

### `threading` and `queue` (Python standard library)
- **What we use it for:** In `web_app.py` — the agent runs in a background thread while the main Flask thread streams events to the browser via SSE

### `base64` (Python standard library)
- **What we use it for:** Encoding PNG images as base64 strings to embed in HTML reports and to send as vision content to Claude

---

## 5. Data models

### `DuctSegment` — `models/duct_segment.py`

Created in Phase 1 (vector extraction) and Phase 3 (raster fallback). Represents raw detected geometry.

```python
@dataclass
class DuctSegment:
    id: str                    # "duct_001" — sequential ID assigned after clustering
    rect: list[float]          # [x0, y0, x1, y1] in PDF points (un-rotated coordinates)
    orientation: str           # "H" (horizontal), "V" (vertical), "D" (diagonal)
    long_pt: float             # length along the long axis in PDF points
    short_pt: float            # width of the duct in PDF points
    aspect: float              # long_pt / short_pt — must be >= 2.0 to be a duct
    centerline: list[list[float]]  # [[x_start, y_start], [x_end, y_end]]
    source: str                # "vector" or "raster"
    confidence: float          # 1.0 for vector (exact), 0.9 for diagonal, varies for raster
    page: int                  # 0-indexed page number
    polygon: list[list[float]] | None  # for diagonal ducts: the 4 corner vertices
```

**Key point about coordinates:** All coordinates are in "PDF points" in the un-rotated media coordinate system. 1 PDF point = 1/72 inch. The drawing may be rotated (e.g. 270°) — we always work in media coordinates, not visual coordinates. The coordinate transform to pixels happens only in Phase 6 (rendering).

### `AnnotatedDuct` — `models/annotated_duct.py`

Created in Phase 4. Extends a `DuctSegment` with matched labels and computed dimensions.

```python
@dataclass
class AnnotatedDuct:
    segment_id: str            # same as DuctSegment.id
    duct_label_id: str | None  # the drawing label "C01" etc. — None if no ID label found
    rect: list[float]          # same as DuctSegment.rect
    orientation: str
    length_ft_measured: float  # long_pt / pt_per_ft — the actual geometric measurement
    length_ft_label: float | None  # what the dimension label says — None if no label
    length_mismatch: bool      # True if |measured - label| / label > 15%
    cross_section: dict | None # {"width_in": 24, "height_in": 18} or {"diameter_in": 12}
    is_round: bool
    unlabeled: bool            # True when NEITHER length nor cross-section label was found
    confidence: float          # how certain we are this is a real duct
    source: str
    page: int
    centerline: list[list[float]] | None
    duct_type: str | None      # "supply" / "return" / "exhaust" — set by agent
    pressure_class: str | None # "low" / "medium" / "high" — set by agent
```

The `to_dict()` method on both dataclasses serializes them to JSON-safe dicts for writing to files and sending to the browser.

### `PipelineState` — `agent.py`

This is the memory of the agent run. It holds all intermediate results so each tool can access what previous tools produced.

```python
@dataclass
class PipelineState:
    pdf_path: str           # absolute path to the input PDF
    vector_segs: list       # list[DuctSegment] — from Tool 1
    raster_segs: list       # list[DuctSegment] — from Tool 3
    labels: list            # list[dict] — from Tool 2
    pt_per_ft: float        # calibrated scale — from Tool 2
    annotated: list         # list[AnnotatedDuct] — from Tool 4
    output_dir: Path | None # where to write outputs (set by caller)
    outputs: dict           # output file paths — populated by Tool 8
    event_log: list         # every emit() event — used to build the HTML report
```

---

## 6. The 8 pipeline tools (deterministic)

These are the functions that do the actual work. Claude calls them via tool_use; they always return exact data from deterministic algorithms.

---

### Tool 1 — Vector duct extraction (`tools/vector_duct_extractor.py`)

**Function called by agent:** `_tool_extract_vector_ducts()` in `agent.py`
**Internal implementation:** `extract_ducts(pdf_path)` in `tools/vector_duct_extractor.py`

**What it does:** Reads the PDF's native vector drawing data and extracts duct rectangles.

**Step by step:**

```
PDF file
  │
  ▼
fitz.open(pdf_path) → opens PDF
page.get_drawings()  → returns list of ALL vector paths on the page
  │
  ▼ Filter: keep only paths that are:
  ├── Black stroke color (RGB all < 0.15) — _is_black()
  ├── Stroke weight >= 0.5 pt — excludes annotation hairlines
  └── Simple rectangular shape — _path_is_simple_rect_shape()
      ├── 'qu' only paths: PDF-native quads (always rectangular)
      └── 'l' line paths: must be axis-aligned and form a rectangle outline
  │
  ▼ _extract_candidate_rects() — deduplicated list of bounding boxes
  │
  ▼ _segment_from_rect() — apply duct heuristics to each rectangle:
  ├── long_pt >= 30 pt (duct must be at least ~1.2 ft long)
  ├── short_pt between 10–50 pt (duct width in 0.4"–2" physical range)
  └── aspect ratio >= 2.0 (must be elongated, not square)
  │
  ▼ _cluster_segments() — merge collinear fragments
  │   Ducts split by fittings or label boxes in the PDF are stored as
  │   separate paths. This step merges them back into one segment if:
  │   ├── Same orientation (both H or both V)
  │   ├── Short-axis overlap >= 80% (they're on the same run)
  │   ├── Long-axis gap <= 8 pt (small gap, like a fitting)
  │   └── Short sides differ by less than 2.5× (not casing vs duct)
  │
  ▼ _extract_diagonal_duct_paths() — diagonal duct detection
  │   HVAC drawings sometimes have ducts at 45° angles. These appear as
  │   parallelogram shapes drawn with 'l' line items or rotated 'qu' quads.
  │   Detected by _is_parallelogram_l_path() which checks:
  │   ├── Exactly 4 unique edges after deduplication
  │   ├── At least one diagonal edge (dx>1pt AND dy>1pt)
  │   ├── Two pairs of parallel equal-length edges (parallelogram shape)
  │   └── Long edge / short edge >= 2.0 (elongated, not square)
  │
  ▼ _centerline_from_polygon() — for diagonal ducts:
  │   Finds the 2 shortest edges (the duct "end caps"), computes their
  │   midpoints, and connects them. This gives the true centerline
  │   direction regardless of which diagonal the duct runs along.
  │
  ▼ Re-assign clean sequential IDs: duct_001, duct_002, ...
  │
  ▼ Return list[DuctSegment]
```

**Key insight:** Duct walls in CAD-exported PDFs are stored as `'qu'` (quad) path items — not `'re'` (rect). Grey paths (RGB ~0.4) are building walls and structure — we ignore them. Only black paths are HVAC ducts.

---

### Tool 2 — Label & scale extraction (`tools/label_extractor.py`)

**Function called by agent:** `_tool_extract_labels_and_scale()` in `agent.py`
**Internal implementation:** `extract_labels_with_scale(pdf_path, vector_segs)` in `tools/label_extractor.py`

**What it does:** Reads all text from the PDF, classifies it as dimension labels, and derives the drawing scale.

**Step by step:**

```
PDF file
  │
  ▼
page.get_text("dict") — returns all text as structured dict:
  blocks → lines → spans
  Each span has: text, bbox [x0,y0,x1,y1], size, color
  │
  ▼ _get_spans() — filter to drawing area (exclude title block)
  │
  ▼ Pass 1 (raw spans) — capture duct IDs before text merging:
  │   _RE_DUCT_ID = r"^[A-Z]{1,3}\d{1,3}$"  matches: C01, SA3, RA12
  │
  ▼ Pass 2 (_merge_adjacent_spans) — join spans on the same line:
  │   "12'" and "- 6\"" → "12' - 6\""
  │   (some CAD software splits labels across separate text objects)
  │
  ▼ _classify_span() — for each span, try regex patterns:
  │   Length:       _RE_LENGTH = r"(\d+)'\s*-\s*(\d+)\""
  │                 "12' - 6\"" → 12.5 feet
  │   Cross-rect:   _RE_CROSS_RECT = r"(\d+)\s*[Xx×]\s*(\d+)"
  │                 "24X18" → width=24", height=18"
  │   Cross-round:  _RE_CROSS_ROUND = r'(\d+)["\']?\s*[Øø⌀]'
  │                 "12Ø" → diameter=12"
  │   Duct ID:      _RE_DUCT_ID = r"^[A-Z]{1,3}\d{1,3}$"
  │                 "C01" → duct identifier
  │
  ▼ Deduplicate: some labels render 2-3x at identical positions
  │
  ▼ _calibrate_scale() — derive pt_per_ft from label+segment pairs:
  │   For each length label "N feet":
  │   1. Find segments whose long_pt ≈ N × PT_PER_FT_EXPECTED (±20%)
  │   2. Keep those within 200pt of the label centroid
  │   3. Compute: pt_per_ft = seg.long_pt / N_feet
  │   Take the median of all valid pt_per_ft values.
  │   Result: ~24.7 pt/ft for input1.pdf (≈ 1/4" = 1ft scale)
  │
  ▼ Return: {pt_per_ft, labels: [...]}
```

**Why median and not average?** A single bad label (e.g. a room dimension accidentally near a duct) would skew the average. The median is robust to outliers.

---

### Tool 3 — Raster fallback (`tools/raster_duct_extractor.py`)

**Function called by agent:** `_tool_detect_raster_ducts()` in `agent.py`
**Internal implementation:** `extract_raster_ducts(pdf_path, vector_segs)` in `tools/raster_duct_extractor.py`

**What it does:** Renders the PDF to an image and uses OpenCV morphology to find duct walls that weren't captured by vector extraction. For pure CAD-exported PDFs (like input1.pdf), this typically finds 0 new segments.

**Step by step:**

```
PDF file
  │
  ▼
page.get_pixmap(matrix=fitz.Matrix(4, 4)) — render at 4× scale (288 DPI)
  │
  ▼ Convert to binary mask: pixels where all BGR channels < 60 → black (duct walls)
  │
  ▼ OpenCV morphological line detection:
  │   Horizontal: cv2.morphologyEx with a wide flat kernel (60px × 1px)
  │               Keeps only long horizontal black lines
  │   Vertical:   cv2.morphologyEx with a tall thin kernel (1px × 60px)
  │               Keeps only long vertical black lines
  │
  ▼ cv2.findContours() — find bounding boxes of detected line structures
  │
  ▼ Pair parallel lines that face each other:
  │   Two horizontal lines at similar y positions = top & bottom of a duct
  │   Gap between them 16-240px (0.4"–6" physical) = duct width
  │   Overlap along x-axis >= 60% = they're facing each other
  │
  ▼ Convert pixel coordinates → PDF media coordinates
  │   (inverse of the render transform, accounting for page rotation)
  │
  ▼ IoU deduplication: discard raster candidates that overlap > 50%
  │   with any vector segment already found in Phase 1
  │
  ▼ Return list[DuctSegment] with source="raster"
```

**IoU** = Intersection over Union. If a raster-detected rectangle has more than 50% overlap with a vector-detected rectangle, it's the same duct — discard the raster one.

---

### Tool 4 — Annotation (`tools/duct_annotator.py`)

**Function called by agent:** `_tool_annotate_segments()` in `agent.py`
**Internal implementation:** `annotate_ducts(all_segs, labels, pt_per_ft)` in `tools/duct_annotator.py`

**What it does:** Matches dimension labels to duct segments (greedy bipartite matching) and computes physical measurements.

**Step by step:**

```
all_segs (vector + raster) + labels (length / cross-section / duct_id) + pt_per_ft
  │
  ▼ Separate labels by type:
  │   length_labels, cs_labels, id_labels
  │
  ▼ _assign_length_labels() — match length labels to segments:
  │   For each length label (N feet):
  │   1. expected_pt = N × pt_per_ft
  │   2. Keep segments where |seg.long_pt - expected_pt| / expected_pt <= 30%
  │      (plausibility gate: prevents compound room dims from matching single segments)
  │   3. Of those, keep segments within 200pt of label centroid
  │   4. Sort all (distance, seg_id, label) pairs ascending by distance
  │   5. Greedy assign: take closest pair, mark both as used, repeat
  │   Returns {seg_id: label_dict}
  │
  ▼ _assign_nearest_labels() — match cross-section and ID labels:
  │   Simpler: just nearest centroid-to-centroid within 150pt
  │   Same greedy assign loop
  │
  ▼ For each DuctSegment, build AnnotatedDuct:
  │   length_ft_measured = seg.long_pt / pt_per_ft
  │   length_ft_label    = matched label's feet value (or None)
  │   length_mismatch    = |measured - label| / label > 15%
  │   cross_section      = from matched cross-section label
  │   unlabeled          = no length label AND no cross-section label
  │
  ▼ Return list[AnnotatedDuct]
```

**Greedy matching:** We sort all possible (label → duct) pairings by distance and consume them greedily. Each label can only be assigned once; each duct can only receive one label of each type. This prevents two ducts from claiming the same label.

---

### Tool 5 — Inspect duct (`agent.py`)

**Function:** `_tool_inspect_duct(state, inp)` in `agent.py`

This tool is special: it returns an **image** so Claude can actually see the duct region.

```python
def _tool_inspect_duct(state, inp):
    segment_id = inp["segment_id"]
    margin = inp.get("margin_pt", 100.0)  # padding around the duct

    # Find the AnnotatedDuct object in state
    duct = next(d for d in state.annotated if d.segment_id == segment_id)

    x0, y0, x1, y1 = duct.rect
    clip = fitz.Rect(x0 - margin, y0 - margin, x1 + margin, y1 + margin)

    # Open PDF, crop to the duct region, render at 2× scale for clarity
    doc = fitz.open(state.pdf_path)
    pix = doc[duct.page].get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip)
    doc.close()

    b64 = base64.b64encode(pix.tobytes("png")).decode()
    return {...metadata..., "image_base64": b64}
```

When the agent loop processes this tool result, it detects the `image_base64` key and formats the tool_result as a **multi-part content block** with both text metadata AND an image:

```python
if "image_base64" in result:
    b64 = result.pop("image_base64")
    content = [
        {"type": "text", "text": json.dumps(result)},       # metadata
        {"type": "image", "source": {                         # image for Claude to see
            "type": "base64",
            "media_type": "image/png",
            "data": b64
        }},
    ]
```

Claude literally sees a cropped image of that duct and uses its vision capability to reason about it.

---

### Tool 6 — Update assessment (`agent.py`)

**Function:** `_tool_update_duct_assessment(state, inp)`

Simple: finds the AnnotatedDuct by `segment_id` and updates its `confidence` field. If Claude says `is_valid_duct: false`, it sets confidence to 0.0 (marks it as not a real duct). This affects which ducts appear in the final output.

---

### Tool 7 — Set classification (`agent.py`)

**Function:** `_tool_set_duct_classification(state, inp)`

```python
def _tool_set_duct_classification(state, inp):
    duct_type = inp["duct_type"]      # "supply" / "return" / "exhaust"
    pressure  = inp["pressure_class"] # "low" / "medium" / "high"

    # Accepts either a single segment_id OR a list segment_ids (batch)
    ids = inp.get("segment_ids") or [inp.get("segment_id")]

    for sid in ids:
        duct = next(d for d in state.annotated if d.segment_id == sid)
        duct.duct_type = duct_type
        duct.pressure_class = pressure
```

**Why batch?** If there are 15 supply ducts all labeled C01–C15, Claude can classify them all in one tool call instead of 15 separate calls. This makes the agent efficient.

These values are stored directly on the `AnnotatedDuct` dataclass (the new `duct_type` and `pressure_class` fields). Everywhere that needs to display type/pressure checks these fields first, then falls back to heuristics.

---

### Tool 8 — Render output (`agent.py` + `tools/annotation_renderer.py`)

**Function:** `_tool_render_output(state, inp)` → calls `render_annotated_page()` + `_generate_html_report()`

**`render_annotated_page()` in `tools/annotation_renderer.py`:**

```
state.annotated (list of AnnotatedDuct)
  │
  ▼
fitz.open → page.get_pixmap(scale=72/72=1.0) → PIL Image
  │
  ▼ For each AnnotatedDuct:
  │   _centerline_pixels() — convert media coordinates to visual pixels:
  │   │   Accounts for page rotation (270° for input1.pdf):
  │   │   rotation=270: visual_x = media_y * scale
  │   │                 visual_y = (media_width - media_x) * scale
  │   │
  │   draw.line([p1, p2], fill=(55,98,227), width=3) — blue centerline
  │   _draw_label_box() — small white numbered box with leader line
  │
  ▼ img.save(output_path, format="PNG")
```

**Why blue `(55, 98, 227)`?** It's a specific CAD-drawing blue that's clearly visible on white drawing backgrounds without being aggressive.

**Coordinate transform detail:** Input1.pdf is rotated 270°. This means the drawing is stored "sideways" in the PDF. Media coordinates (what PyMuPDF returns from `get_drawings()` and `get_text()`) are in the un-rotated space. To draw on the rendered image (which is visually correct, i.e. "right-side up"), we must apply the rotation transform. For 270° rotation:
- Visual pixel X = media_y × scale
- Visual pixel Y = (media_width_pt - media_x) × scale

The `_media_rect_to_pixel()` and `_centerline_pixels()` functions implement all four rotation cases (0°, 90°, 180°, 270°).

---

## 7. The agent loop (`agent.py`)

This is the heart of the agentic system. Here's exactly what happens:

### Tool definitions (TOOL_DEFINITIONS)

The Anthropic API requires tools to be described as JSON schemas so Claude knows what parameters each tool accepts:

```python
TOOL_DEFINITIONS = [
    {
        "name": "extract_vector_ducts",
        "description": "Extract duct geometry from PDF vector paths...",
        "input_schema": {
            "type": "object",
            "properties": {},    # no parameters needed
            "required": [],
        },
    },
    {
        "name": "inspect_duct",
        "description": "...",
        "input_schema": {
            "type": "object",
            "properties": {
                "segment_id": {"type": "string", "description": "..."},
                "margin_pt": {"type": "number", "description": "..."},
            },
            "required": ["segment_id"],
        },
    },
    # ... 6 more tools
]
```

Claude reads these schemas and knows what parameters to pass when calling each tool.

### System prompt (SYSTEM_PROMPT)

The system prompt is a set of instructions that persists across all conversation turns. It tells Claude:
1. What its job is
2. The 7-step workflow to follow
3. How to classify duct types
4. HVAC domain knowledge (what supply/return/exhaust means)
5. When to use inspect_duct vs when to batch classify

### The run_agent() function — step by step

```python
def run_agent(pdf_path, output_dir=None, on_event=None):
    client = anthropic.Anthropic()
    state = PipelineState(pdf_path=str(pdf_path), output_dir=output_dir)

    def emit(t, d):
        state.event_log.append({"type": t, "data": d})  # captured for HTML report
        if on_event:
            on_event(t, d)  # sent to SSE stream / CLI printer

    messages = [{"role": "user", "content": f"Analyze the HVAC duct drawing: {pdf_path}"}]
```

First turn — Claude responds with tool_use blocks (it decides to call extract_vector_ducts first):

```
Turn 1:
  User: "Analyze the HVAC duct drawing..."
  Claude: [tool_use: extract_vector_ducts, id="toolu_abc"]
          stop_reason: "tool_use"
```

We call the tool, get results, send them back:

```python
tool_results.append({
    "type": "tool_result",
    "tool_use_id": "toolu_abc",
    "content": json.dumps({"segment_count": 23, "orientations": {...}})
})

messages.append({"role": "assistant", "content": response.content})  # Claude's tool_use blocks
messages.append({"role": "user", "content": tool_results})           # Our tool results
```

This continues in a loop until `stop_reason == "end_turn"` (Claude is done):

```
Turn 1: User → Claude calls extract_vector_ducts
Turn 2: Tool result → Claude calls extract_labels_and_scale
Turn 3: Tool result → Claude calls detect_raster_ducts
Turn 4: Tool result → Claude calls annotate_segments
Turn 5: Tool result → Claude says "I see 2 mismatches: duct_003, duct_007"
                      Claude calls inspect_duct(segment_id="duct_003")
Turn 6: Tool result (with image) → Claude looks at image, says "this looks valid"
                                    Claude calls update_duct_assessment(confidence=0.9)
Turn 7: Tool result → Claude calls inspect_duct(segment_id="duct_007")
...
Turn N: Tool result → Claude calls set_duct_classification(segment_ids=[...], ...)
Turn N+1: Tool result → Claude calls render_output
Turn N+2: Tool result → Claude says "Done. Found 23 ducts..." (end_turn)
```

**The full message history grows with each turn.** This is how the Anthropic API maintains "conversation context" — all previous messages are sent with every new request. Claude can see all the tool results it received earlier.

---

## 8. The CLI runner (`run.py`)

Simple entry point for running from the terminal:

```bash
python run.py "../sample input/input1.pdf"
python run.py "../sample input/input1.pdf" --output-dir /tmp/test/
```

```python
def on_event(event_type, data):
    if event_type == "tool_call":
        print(f"[tool] {data['tool']} → ", end="", flush=True)
    elif event_type == "tool_result":
        print(_summarise_result(data["tool"], data["result"]))
    elif event_type == "agent_text":
        print(f"[agent] {data['text']}")

annotated, summary = run_agent(pdf_path, output_dir=out_dir, on_event=on_event)
```

The `on_event` callback is called every time the agent emits an event. Here it just prints to stdout. In the web app, it puts events on a queue for SSE streaming.

---

## 9. The web application (`web_app.py`)

Flask app on port 5001. Enables uploading a PDF through the browser and watching the agent work in real time.

### SSE — Server-Sent Events

SSE is a browser API for one-way streaming from server to client. Unlike WebSockets (bidirectional), SSE is simpler: the server sends events in a specific text format, the browser receives them via `EventSource`.

Format:
```
event: tool_call
data: {"tool": "extract_vector_ducts", "input": {}}

event: tool_result
data: {"tool": "extract_vector_ducts", "result": {"segment_count": 23}}

event: done
data: {"ducts": [...], "image_url": "/session/abc/image"}
```

### Request flow

```
Browser uploads PDF
        │
        ▼
POST /analyze
  1. Save PDF to uploads/{sid}/input.pdf
  2. Create queue.Queue()
  3. Store session: analysis_sessions[sid] = {"queue": q, ...}
  4. Start background thread: run_agent_thread(sid, pdf_path, q)
  5. Return {"session_id": sid, "stream_url": "/session/sid/stream"}
        │
        ▼ (immediately, browser opens SSE connection)
GET /session/{sid}/stream
  Generator function runs:
    while True:
        event = q.get(timeout=60)   ← blocks until agent puts something on queue
        yield f"event: {event['type']}\ndata: {json.dumps(event['data'])}\n\n"
        if event['type'] in ('done', 'error'):
            break
        │
        ▼ Browser receives event, updates the live log
```

### Background thread (`run_agent_thread`)

```python
def run_agent_thread(sid, pdf_path, q):
    def on_event(event_type, data):
        q.put({"type": event_type, "data": data})  # → SSE stream

    annotated, summary = run_agent(pdf_path, output_dir=..., on_event=on_event)

    # After agent finishes:
    render_annotated_page(pdf_path, annotated, session_dir/"annotated.png")
    create_annotated_pdf(pdf_path, annotated, session_dir/"annotated.pdf")

    # Compute pixel coordinates for SVG overlay
    for i, duct in enumerate(annotated):
        p1, p2 = _centerline_pixels(duct, rotation, media_w, media_h, scale)
        ducts_out.append({...p1, p2, duct_type, pressure_class, dimension...})

    q.put({"type": "done", "data": {"ducts": ducts_out, "image_url": ...}})
```

The thread pushes every agent event onto the queue as it runs. The SSE generator on the main Flask thread pops from the queue and sends to the browser.

### _centerline_pixels() — why it's used in web_app.py

The SVG overlay in the browser needs to draw lines on top of the annotated PNG image. The browser works in pixel coordinates (top-left = 0,0). But our ducts are stored in PDF media coordinates. `_centerline_pixels()` converts:

`(media_x, media_y) → (visual_pixel_x, visual_pixel_y)`

accounting for page rotation and the 72 DPI render scale.

---

## 10. The HTML report generator

**Function:** `_generate_html_report(state, out_dir, stem, png_path)` in `agent.py`

Called automatically by `_tool_render_output` as the last step. Produces `{stem}_report.html` — a single self-contained HTML file that opens in any browser.

**What goes in it:**
1. **Stats cards** — computed from `state.annotated`: total ducts, labeled count, mismatches, unlabeled, vector/raster segment counts, pt_per_ft scale
2. **Annotated floor plan** — the PNG is read and base64-encoded, embedded as `<img src="data:image/png;base64,...">`. No external files needed.
3. **Duct inventory table** — one row per AnnotatedDuct: ID, type, orientation, dimensions, measured length, label length, match status, pressure class, source, confidence. Mismatch rows highlighted amber.
4. **Agent analysis log** — built from `state.event_log` (every event that was emitted during the run). Shows tool calls with one-line result summaries, and Claude's text reasoning in italics.

The `event_log` is populated in `run_agent()` inside the `emit()` closure:
```python
def emit(t, d):
    state.event_log.append({"type": t, "data": d})  # ← captured here
    if on_event:
        on_event(t, d)
```

So by the time `render_output` is called, every tool call, tool result, and agent text block is in `state.event_log`.

---

## 11. The web UI (`templates/index.html`)

Single-page application — no page reloads. Three views managed by JavaScript `style.display`:

### View 1: Upload
- Drag-and-drop area or file picker
- `FormData` API to POST the file to `/analyze`
- On success → switches to Analyzing view, opens SSE connection

### View 2: Analyzing (live agent log)
Split layout:
- **Left panel (40%):** Agent log. Each SSE event adds an item:
  - `tool_call` event → spinner item with tool name
  - `tool_result` event → updates spinner to ✓ + shows result stats
  - `agent_text` event → italic text block with robot icon
- **Right panel (60%):** Loading placeholder → replaced by annotated image + SVG overlay when `done` event arrives

### View 3: Result
- Annotated floor plan with SVG overlay for hover interactions
- Right panel: searchable duct list
- Hover any blue line → tooltip showing Type, Pressure, Section, Dimensions, Length
- Download buttons for PNG and PDF

### SVG overlay system

```javascript
// SVG is positioned exactly over the image
const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
svg.style.position = "absolute";
svg.style.top = "0"; svg.style.left = "0";
svg.style.width = "100%"; svg.style.height = "100%";

// For each duct, draw two overlapping lines:
// 1. Invisible 24px-wide hit-area line (catches mouse events)
const hitLine = document.createElementNS(..., "line");
hitLine.setAttribute("stroke-width", "24");
hitLine.setAttribute("stroke-opacity", "0");  // invisible

// 2. Visible glow line (shown on hover)
const glowLine = document.createElementNS(..., "line");
glowLine.setAttribute("stroke", colorForDuctType);
glowLine.setAttribute("stroke-opacity", "0");  // hidden by default

hitLine.addEventListener("mouseenter", () => {
    glowLine.setAttribute("stroke-opacity", "0.65");
    showTooltip(duct);
});
```

**Zoom and pan:**
```javascript
let zoom = 1.0, panX = 0, panY = 0;

// Mouse wheel → zoom
container.addEventListener("wheel", e => {
    zoom *= e.deltaY < 0 ? 1.1 : 0.9;
    updateTransform();
});

// Mouse drag → pan
container.addEventListener("mousedown", e => { isDragging = true; ... });

function updateTransform() {
    imageWrapper.style.transform = `translate(${panX}px, ${panY}px) scale(${zoom})`;
}
```

The image and SVG overlay transform together since they share the same wrapper div, keeping the hover lines aligned with the image at any zoom level.

---

## 12. Configuration (`config/settings.py`)

All numeric constants in one place. When tuning for a new drawing type, change values here — never inside the tool code.

```python
# Phase 1 thresholds
BLACK_MAX_CHANNEL = 0.15   # RGB < 0.15 = black (duct) vs grey (wall)
DUCT_MIN_ASPECT   = 2.0    # duct must be at least 2× longer than wide
DUCT_MIN_SHORT_PT = 10.0   # min duct width (10pt ≈ 0.4" physical)
DUCT_MAX_SHORT_PT = 50.0   # max duct width (50pt ≈ 2" physical at 25pt/ft)
DUCT_MIN_LONG_PT  = 30.0   # min duct length (30pt ≈ 1.2ft physical)
CLUSTER_LONG_GAP_MAX_PT = 8.0  # max gap between fragments to merge (fittings)

# Phase 2 thresholds
PT_PER_FT_EXPECTED = 24.7  # empirical seed value for input1.pdf
PT_PER_FT_MIN = 22.0       # reject scale calibration outside this range
PT_PER_FT_MAX = 27.0

# Phase 4 thresholds
LABEL_MISMATCH_THRESHOLD = 0.15  # flag mismatch if >15% difference

# Phase 5 (vision)
VISION_MODEL = "claude-opus-4-7"
VISION_CROP_MARGIN_PT = 100.0   # padding around duct rect for inspection crop

# Phase 6 rendering
RENDER_DPI = 72              # output PNG resolution
OUTLINE_COLOR_RGB = (55, 98, 227)  # blue annotation color
```

**Why `TITLE_BLOCK_Y_MIN_PT = 2200.0`?** The title block (project name, drawing number, revision history) is at the bottom of the drawing in media coordinates. Labels and geometry below this y-value are excluded from analysis to prevent "18'-0"" room dimensions in the title block from polluting the scale calibration.

---

## 13. Full end-to-end flow

```
User uploads "mechanical_floor.pdf" via browser
│
▼ POST /analyze (Flask, web_app.py)
│   Save to uploads/{sid}/input.pdf
│   Start background thread
│   Return {session_id: "abc123"}
│
▼ Browser opens EventSource("/session/abc123/stream")
│
▼ Background thread calls run_agent("...input.pdf", on_event=→queue)
│
▼ AGENT LOOP BEGINS (agent.py, run_agent())
│   ┌─────────────────────────────────────────────────────────┐
│   │ Turn 1: Claude → extract_vector_ducts                   │
│   │   _tool_extract_vector_ducts() calls extract_ducts()    │
│   │   → fitz reads PDF paths, filters black quads           │
│   │   → clustering merges split segments                     │
│   │   → diagonal detection finds angled ducts               │
│   │   → returns: 23 segments (H=15, V=7, D=1)              │
│   │   emit("tool_call", ...) emit("tool_result", ...)       │
│   │   → SSE event → browser shows "✓ Extract vector ducts" │
│   │                                                          │
│   │ Turn 2: Claude → extract_labels_and_scale               │
│   │   → fitz reads all PDF text spans                       │
│   │   → regex matches "12' - 6\"", "24X18", "C01" etc      │
│   │   → scale calibration: median pt_per_ft = 24.71        │
│   │   → returns: 7 length labels, 6 cross-section labels    │
│   │   → SSE event → browser shows "✓ Parse labels"         │
│   │                                                          │
│   │ Turn 3: Claude → detect_raster_ducts                    │
│   │   → OpenCV morphology on 4× rendered image              │
│   │   → 0 new segments (pure vector PDF)                    │
│   │                                                          │
│   │ Turn 4: Claude → annotate_segments                      │
│   │   → greedy label matching: 7 length labels matched      │
│   │   → 2 mismatches: duct_003, duct_007                   │
│   │   → 1 unlabeled: duct_018                              │
│   │                                                          │
│   │ Turn 5: Claude → inspect_duct("duct_003")               │
│   │   → fitz crops PDF region around duct_003               │
│   │   → base64 PNG returned as vision content               │
│   │   → Claude sees image, reasons about it                 │
│   │   Claude (text): "I can see this segment is a valid     │
│   │    supply duct with label C03..."                       │
│   │   → SSE agent_text event → browser shows Claude's text │
│   │                                                          │
│   │ Turn 6: Claude → update_duct_assessment("duct_003",     │
│   │                    confidence=0.92)                      │
│   │                                                          │
│   │ ... (inspect duct_007 and duct_018 similarly)           │
│   │                                                          │
│   │ Turn N: Claude → set_duct_classification(               │
│   │   segment_ids=["duct_001",...,"duct_015"],               │
│   │   duct_type="supply", pressure_class="medium")          │
│   │                                                          │
│   │ Turn N+1: Claude → set_duct_classification(             │
│   │   segment_ids=["duct_016", "duct_017"],                  │
│   │   duct_type="return", pressure_class="low")             │
│   │                                                          │
│   │ Turn N+2: Claude → render_output                        │
│   │   → render_annotated_page() draws blue centerlines      │
│   │   → _generate_html_report() writes report.html          │
│   │   → summary.json written                                │
│   └─────────────────────────────────────────────────────────┘
│
▼ Thread emits "done" event to queue
│   {ducts: [{p1, p2, duct_type, pressure_class, ...}, ...],
│    image_url: "/session/abc123/image"}
│
▼ SSE generator sends "done" event to browser
│
▼ Browser:
│   Shows annotated floor plan
│   Draws SVG overlay lines (colored by duct type)
│   Populates duct list panel
│   "New Analysis" button available
│
▼ User hovers a blue line → tooltip shows:
│   Type: Supply
│   Pressure: Medium
│   Section: Rectangular, 24" × 18"
│   Length: 12.50 ft
│
▼ User downloads report.html → opens in browser:
│   Full duct inventory table
│   Embedded annotated floor plan
│   Agent's complete reasoning log
```

---

## 14. Key concepts glossary

**PDF points:** The unit of measurement in PDF files. 1 point = 1/72 inch. A drawing at 1/4"=1ft scale means 1 ft of physical duct = 0.25" on paper = 0.25 × 72 = 18 PDF points. (input1.pdf scale is ~24.7 pt/ft, so 1/4" = 1ft at 72 DPI gives ~18pt; the actual drawing may use a slightly different scale.)

**Media coordinates vs visual coordinates:** A rotated PDF stores all path data in the un-rotated "media" space. When you render the page to an image, PyMuPDF automatically rotates it so it looks right-side-up. The coordinate transform between these two spaces depends on the rotation angle. We always work in media coordinates until the final rendering step.

**'qu' vs 're' paths:** PDF has two rectangle primitives. `'re'` is a true rectangle aligned to the page axes. `'qu'` is a quadrilateral (four corners, possibly non-rectangular, possibly rotated). HVAC CAD software exports duct rectangles as `'qu'` items. We filter for these specifically.

**Greedy bipartite matching:** An algorithm for pairing items from two sets (labels and ducts) where each item can only be paired once. "Greedy" means we take the best (closest) pair first, then remove both from consideration and repeat. Not globally optimal, but fast and good enough for this use case.

**Tool use / function calling:** The Anthropic API feature where Claude can request the execution of external functions. Claude sends a `tool_use` content block with `name` and `input`; you execute the function and send back a `tool_result`. This is how the agent loop works.

**Server-Sent Events (SSE):** A web standard for one-way streaming from server to browser. The browser opens a persistent HTTP connection; the server sends text events in a specific format whenever it has data. Unlike WebSockets, SSE is unidirectional and works over standard HTTP.

**Morphological operations (OpenCV):** Image processing operations that erode or dilate shapes based on a kernel. A long thin horizontal kernel used in erosion finds only very long horizontal black lines (duct walls). Combining erosion and dilation (morphologyEx with MORPH_OPEN) removes short noise while preserving long structures.

**IoU (Intersection over Union):** A measure of overlap between two rectangles. IoU = area(A ∩ B) / area(A ∪ B). Value of 1.0 means identical; 0.0 means no overlap. Used to deduplicate raster-detected ducts against vector-detected ducts.

**pt_per_ft (scale calibration):** The conversion factor from PDF point coordinates to physical feet. Derived empirically from matching dimension labels ("12' - 6\"") to the measured length of nearby duct segments in PDF points. The median of multiple calibration pairs gives a robust estimate. For input1.pdf: ~24.7 pt/ft (meaning 1 foot of duct = 24.7 PDF points on the page).
