# Automated HVAC Duct Detection and Annotation Tool

An automated pipeline that ingests mechanical floor plan PDFs, detects HVAC duct segments directly from CAD vector geometry, associates dimension labels, measures physical lengths, and produces annotated drawings with structured data exports.

---

## How It Works

```
PDF Drawing (CAD export)
        │
        ▼
Phase 1 — Vector duct extraction     Read exact coordinates from PDF paths
        ▼
Phase 2 — Scale calibration          Parse dimension text, derive pt/ft scale
        ▼
Phase 3 — Raster fallback            OpenCV morphology for non-quad ducts
        ▼
Phase 4 — Label association          Match labels to ducts, compute lengths
        ▼
Phase 5 — Vision cross-validation    Claude Opus reviews low-confidence ducts
        ▼
Phase 6 — Rendering                  Annotated PNG + PDF with blue centerlines
```

The pipeline reads duct geometry **directly from PDF vector paths** — no LLM coordinate guessing. Claude vision is used only in Phase 5 for semantic validation of ambiguous detections.

---

## Quick Start

### Prerequisites

- Python 3.10+
- An Anthropic API key (only required for Phase 5 vision validation)

### Setup

```bash
git clone <repo-url>
cd "Automated HVAC Duct Detection and Annotation Tool - Agentic"

cd hvac-duct-detection-v2
pip install -r requirements.txt
pip install flask
```

Set your API key (only needed if not using `--skip-vision`):
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## Usage

### Web Application

The easiest way to use the tool. Upload a PDF, get an interactive annotated floor plan you can hover to inspect each duct.

```bash
cd hvac-duct-detection-v2
python3 web_app.py
```

Open **http://localhost:5000** in your browser.

**What the web app does:**
1. Upload a PDF via drag-and-drop or file picker
2. Pipeline runs Phases 1–4 and 6 automatically (~3 seconds)
3. Interactive floor plan appears — hover any blue annotation line to see:
   - Type of duct (Supply / Return / Exhaust)
   - Pressure class (Low / Medium / High)
   - Section type and dimension (e.g. `24" × 18"` or `Ø 12"`)
   - Measured length in feet
4. Right panel lists all detected ducts — click any to pan the plan to it
5. Download buttons for annotated PNG and PDF

### CLI Pipeline

```bash
cd hvac-duct-detection-v2
python3 run.py "../sample input/input1.pdf" --skip-vision
```

With vision validation (requires API key):
```bash
python3 run.py "../sample input/input1.pdf"
```

| Option | Description |
|---|---|
| `--skip-vision` | Skip Phase 5 (no API calls, faster) |
| `--output-dir DIR` | Custom output directory (default: `outputs/{stem}/`) |

---

## Output

Every run writes to `hvac-duct-detection-v2/outputs/{stem}/`:

```
outputs/input1/
├── input1_annotated.png            Annotated floor plan image (72 DPI)
├── input1_annotated.pdf            Annotated PDF with vector blue centerlines
└── artifacts/
      ├── phase2_labels.json        All parsed dimension labels with coordinates
      ├── phase4_annotated_ducts.json  One record per duct (measurements + labels)
      └── summary.json              Run summary with all duct data
```

### Annotation Style

Each detected duct is annotated with:
- A **blue centerline** (`#3762E3`) running the exact measured length of the duct
- A **numbered label box** with a leader line to the duct midpoint

Color coding in the web app:

| Type | Color |
|---|---|
| Supply Air | Blue `#3762E3` |
| Return Air | Amber `#D97706` |
| Exhaust Air | Green `#059669` |

| Pressure | Color | Criteria |
|---|---|---|
| Low | Green | Largest dimension ≥ 24" |
| Medium | Orange | 12–24" |
| High | Red | < 12" |

---

## Duct Data Schema

`phase4_annotated_ducts.json` — one record per detected segment:

```json
{
  "id": "C03",
  "duct_idx": "duct_003",
  "rect": [1259.2, 491.5, 1709.0, 561.5],
  "orientation": "H",
  "length_ft_measured": 18.556,
  "length_ft_label": 18.4167,
  "length_mismatch": false,
  "cross_section": {"width_in": 24, "height_in": 18},
  "is_round": false,
  "unlabeled": false,
  "confidence": 1.0,
  "source": "vector",
  "page": 0
}
```

| Field | Description |
|---|---|
| `id` | Duct identifier from drawing text (C01, SA3, etc.) or `null` |
| `rect` | Bounding box `[x0, y0, x1, y1]` in PDF points |
| `orientation` | `H` horizontal, `V` vertical, `D` diagonal |
| `length_ft_measured` | Physical length from vector geometry ÷ calibrated scale |
| `length_ft_label` | Length from matched dimension label, or `null` |
| `length_mismatch` | `true` if measured vs. label differ by more than 15% |
| `cross_section` | `{"width_in":24,"height_in":18}` or `{"diameter_in":12}` |
| `unlabeled` | `true` when no dimension or cross-section label was found |
| `source` | `"vector"` (Phase 1) or `"raster"` (Phase 3 fallback) |

---

## Technical Documentation

See [`hvac-duct-detection-v2/doc.md`](hvac-duct-detection-v2/doc.md) for a full walkthrough of every phase — what it does, why, the algorithm steps, and the design decisions behind the approach.

---

## Running Tests

```bash
cd hvac-duct-detection-v2
python3 -m pytest tests/ -v
```

157 tests across 6 phases (unit + integration). Integration tests require the sample PDF at `sample input/input1.pdf`.

| Test file | Coverage |
|---|---|
| `test_phase1.py` | Vector extraction, clustering, diagonal detection |
| `test_phase2.py` | Scale calibration, label parsing, multi-span merge |
| `test_phase3.py` | Raster morphology, parallel-pair matching, IoU dedup |
| `test_phase4.py` | Label-duct association, plausibility gate, mismatch flag |
| `test_phase5.py` | Vision trigger logic, mocked API calls, crop rendering |
| `test_phase6.py` | Pixel coordinate transform, centerline geometry, CLI end-to-end |

---

## Project Structure

```
hvac-duct-detection-v2/
├── run.py                      CLI entry point (Phases 1–6)
├── web_app.py                  Flask web application
├── doc.md                      End-to-end process documentation
├── requirements.txt
├── config/
│   ├── settings.py             All calibrated constants and thresholds
│   └── prompts.py              Phase 5 Claude vision prompt
├── models/
│   ├── duct_segment.py         DuctSegment dataclass (Phases 1–3)
│   └── annotated_duct.py       AnnotatedDuct dataclass (Phases 4–6)
├── tools/
│   ├── vector_duct_extractor.py  Phase 1 — PDF path extraction
│   ├── label_extractor.py        Phase 2 — text parsing + scale calibration
│   ├── raster_duct_extractor.py  Phase 3 — OpenCV morphology fallback
│   ├── duct_annotator.py         Phase 4 — label association + measurement
│   ├── vision_validator.py       Phase 5 — Claude Opus cross-validation
│   └── annotation_renderer.py   Phase 6 — PNG/PDF rendering
├── templates/
│   └── index.html              Web application UI
├── tests/
│   ├── test_phase1.py
│   ├── test_phase2.py
│   ├── test_phase3.py
│   ├── test_phase4.py
│   ├── test_phase5.py
│   └── test_phase6.py
└── outputs/                    Per-run outputs (git-ignored)
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `pymupdf` | PDF vector extraction, text reading, rendering, annotation writing |
| `opencv-python` | Phase 3 raster morphology |
| `numpy` | Phase 3 image arrays |
| `anthropic` | Phase 5 Claude Opus vision validation |
| `Pillow` | PNG rendering and centerline drawing |
| `flask` | Web application server |

---

## Why Vector-First?

The previous version of this tool (v1) used Claude vision to estimate pixel coordinates for duct bounding boxes. This produced systematically misaligned annotations because LLMs cannot return pixel-accurate coordinates from rendered images.

HVAC mechanical drawings are exported directly from CAD software. Every duct rectangle exists as an exact vector path with precise coordinates already stored in the PDF. Reading those coordinates directly — bypassing image recognition entirely — gives exact results with no approximation error.

Vision (Claude Opus) is still used in Phase 5, but only to answer semantic questions ("is this a duct or a fitting?") rather than geometric ones ("where exactly is the boundary?").
