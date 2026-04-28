# Automated HVAC Duct Detection and Annotation Tool

Agentic AI pipeline that ingests mechanical floor plan PDFs (M-plans), detects and classifies HVAC duct segments (supply / return / exhaust), associates dimension annotations and CFM callouts with each segment, then outputs an annotated PDF with colored overlays and structured data exports.

---

## Architecture

```
Ingestion → Vision → Measurement → Annotation → Review
               ↑_________feedback (Reflexion loop)________↑
```

| Agent | Model | Role |
|---|---|---|
| Ingestion | claude-haiku-4-5 | PDF → images, text blocks, scale detection |
| Vision | claude-opus-4-7 | Quadrant-level duct detection with bounding polygons |
| Measurement | claude-sonnet-4-6 | Dimension and CFM label extraction and matching |
| Annotation | — (pure Python) | Colored overlay rendering, PDF export |
| Review | claude-sonnet-4-6 | Quality scoring, reflexion feedback generation |

---

## Requirements

- Python 3.10+
- An Anthropic API key

---

## Setup

**1. Clone and enter the repo**

```bash
cd "Automated HVAC Duct Detection and Annotation Tool - Agentic"
```

**2. Create virtual environment and install dependencies**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**3. Configure your API key**

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=sk-ant-api03-...

# Models (optional — defaults shown)
VISION_MODEL=claude-opus-4-7
ORCHESTRATOR_MODEL=claude-sonnet-4-6
INGESTION_MODEL=claude-haiku-4-5-20251001
MEASUREMENT_MODEL=claude-sonnet-4-6
REVIEW_MODEL=claude-sonnet-4-6

# Pipeline settings (optional — defaults shown)
CONFIDENCE_THRESHOLD=0.85
MAX_RETRIES=3
DPI=300
```

---

## Running the Pipeline

All commands are run from the **project root** with the venv active.

### Basic usage

```bash
python hvac-duct-detection/scripts/run_pipeline.py --pdf "path/to/plan.pdf"
```

### Full example

```bash
python hvac-duct-detection/scripts/run_pipeline.py \
  --pdf "sample input/input.pdf" \
  --confidence 0.85 \
  --max-retries 3 \
  --pages "1-3"
```

---

## CLI Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `--pdf` | path | *(required)* | Path to the input PDF mechanical floor plan |
| `--confidence` | float | `0.85` | Minimum review score (0.0–1.0) required to accept the output. Below this, the reflexion loop retries. Overrides `.env` setting. |
| `--max-retries` | int | `3` | Maximum number of reflexion retries before accepting best-effort output. Overrides `.env` setting. |
| `--scale-ratio` | float | auto-detect | Manual pixels-per-foot override. Use when the drawing says "DO NOT SCALE". At 300 DPI, `1/4"=1'-0"` ≈ `75.0`. |
| `--pages` | string | all pages | Page range to process (1-based). Examples: `1-3`, `2,4,6`, `5`. |

---

## Output

Each run creates a **unique session directory** inside `hvac-duct-detection/outputs/`:

```
hvac-duct-detection/outputs/<session_id>/
├── pages/
│   └── page_000.png              Raw page render at 300 DPI
├── page_000_annotated.png        Annotated page image
├── annotated.pdf                 Final annotated PDF (all pages)
├── measurements.json             Structured duct data (one record per segment)
└── measurements.csv              Same data as a spreadsheet
```

### Session ID format

`YYYYMMDD_HHMMSS_<6-char hex>` — e.g. `20260428_125629_1ecd8a`

### Terminal summary

```
========================================================
  HVAC Duct Detection — Pipeline Complete
========================================================
  session_id        : 20260428_125629_1ecd8a
  input_path        : /home/.../sample input/input.pdf
  output_dir        : .../outputs/20260428_125629_1ecd8a
  output_pdf        : .../outputs/20260428_125629_1ecd8a/annotated.pdf
  output_png        : .../outputs/20260428_125629_1ecd8a/page_000_annotated.png
  segments_detected : 15
  segments_labelled : 15
  review_score      : 1.0000
  retries           : 0
  Run log           : .../runs/registry.csv
========================================================
```

---

## Run Registry

Every pipeline run is appended to a persistent CSV log:

```
hvac-duct-detection/runs/registry.csv
```

| Column | Description |
|---|---|
| `session_id` | Unique run identifier |
| `timestamp` | Date and time of the run |
| `input_path` | Absolute path to the input PDF |
| `output_dir` | Output directory for this run |
| `output_pdf` | Path to the annotated PDF |
| `output_png` | Semicolon-separated paths to annotated PNGs (one per page) |
| `segments_detected` | Total duct segments found |
| `segments_labelled` | Segments successfully matched to a dimension/CFM label |
| `review_score` | Final quality score (0.0–1.0) |
| `retries` | Number of reflexion retries used |

---

## Duct Type Color Coding

| Type | Color | Hex |
|---|---|---|
| Supply | Blue | `#1565C0` |
| Return | Red | `#C62828` |
| Exhaust | Orange | `#E65100` |

---

## Measurement Data Schema

`measurements.json` contains one record per detected segment:

```json
{
  "segment_id": "seg_003",
  "type": "return",
  "is_round": true,
  "diameter_in": 10,
  "width_in": null,
  "height_in": null,
  "cfm": 400,
  "length_ft": null,
  "bbox": [2362.5, 2241.0, 2902.5, 2484.0],
  "unmatched": false
}
```

| Field | Description |
|---|---|
| `type` | `supply`, `return`, or `exhaust` |
| `is_round` | `true` for circular ducts (e.g. `8"Ø`), `false` for rectangular |
| `diameter_in` | Diameter in inches for round ducts |
| `width_in` / `height_in` | Dimensions in inches for rectangular ducts |
| `cfm` | Airflow in CFM if found in nearby labels |
| `length_ft` | Duct run length in feet (requires scale bar; `null` for "DO NOT SCALE" drawings) |
| `bbox` | Bounding box in pixel coordinates `[x1, y1, x2, y2]` at 300 DPI |
| `unmatched` | `true` if no dimension label could be associated with this segment |

---

## Reflexion Loop

The pipeline automatically retries when the review score falls below `--confidence`:

```
Attempt 1 → score 0.62 (below 0.85) → Claude generates feedback → retry
Attempt 2 → score 0.89 (above 0.85) → accepted ✓
```

Back-off between retries: `2^retry` seconds (2s, 4s, 8s, …).

At `--max-retries` the best-effort result is accepted with a warning in the log.

---

## Running Tests

```bash
cd hvac-duct-detection
source ../.venv/bin/activate

# Unit tests only (no API key needed)
python -m pytest -q

# Include integration tests (requires ANTHROPIC_API_KEY)
python -m pytest -q -m integration
```

---

## Project Structure

```
hvac-duct-detection/
├── agents/
│   ├── orchestrator.py        Pipeline coordinator + reflexion loop
│   ├── ingestion_agent.py     PDF ingestion
│   ├── vision_agent.py        Duct detection via Claude vision
│   ├── measurement_agent.py   Dimension + CFM extraction
│   ├── annotation_agent.py    Overlay rendering
│   └── review_agent.py        Quality scoring + feedback
├── tools/
│   ├── pdf_tools.py           PDF → image, text extraction, scale detection
│   ├── vision_tools.py        Vision API calls, quadrant split, IoU dedup
│   ├── measurement_tools.py   Regex parsers, label matching
│   ├── annotation_tools.py    Drawing, label rendering, PDF export
│   └── review_tools.py        Confidence scorer, diff checker
├── config/
│   ├── settings.py            Pydantic-settings with .env auto-discovery
│   └── prompts.py             All LLM prompt templates
├── models/
│   └── duct_segment.py        Pydantic data models
├── scripts/
│   └── run_pipeline.py        CLI entry point
├── outputs/                   Per-session output directories
├── runs/
│   └── registry.csv           Persistent run log
└── tests/                     Unit + integration test suite
```
