# Automated HVAC Duct Detection and Annotation Tool

An agentic AI pipeline that ingests mechanical floor plan PDFs (M-plans), detects and classifies HVAC duct segments (supply / return / exhaust), associates dimension annotations and CFM callouts with each segment, and outputs an annotated PDF with colored overlays and structured data exports (JSON + CSV).

---

## How It Works

```
PDF → Ingestion → Vision → Measurement → Annotation → Review
                     ↑_________ reflexion feedback _________↑
```

The pipeline runs five specialized agents in sequence. If the quality score from the Review agent falls below the confidence threshold, it automatically retries the Vision → Measurement → Annotation → Review cycle with targeted feedback — up to `--max-retries` times.

You can run the pipeline from the **CLI** or through the **web UI** — both produce the same annotated PDF and structured data outputs.

| Agent | Model | Role |
|---|---|---|
| Ingestion | claude-haiku-4-5 | Renders PDF pages to 300 DPI images, extracts text blocks with coordinates, detects the drawing scale |
| Vision | claude-opus-4-7 | Splits each page into quadrants, detects duct segments with bounding polygons, deduplicates overlaps |
| Measurement | claude-sonnet-4-6 | Matches each detected segment to its dimension label (e.g. `12"Ø`), CFM callout, and pressure class |
| Annotation | pure Python | Draws colored overlays on the page images and exports an annotated PDF |
| Review | claude-sonnet-4-6 | Scores output quality (0–1) and generates targeted feedback for failed attempts |

---

## Prerequisites

- **Python 3.10 or higher**
- **An Anthropic API key** — get one at [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys)

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Hethshah/hvac-duct-detection-agentic.git
cd hvac-duct-detection-agentic
```

### 2. Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows
pip install -r hvac-duct-detection/requirements.txt
```

### 3. Configure your API key

Copy the example environment file and fill in your key:

```bash
cp .env.example .env
```

Open `.env` and set your key:

```env
ANTHROPIC_API_KEY=sk-ant-api03-...
```

All other settings are optional — the defaults shown in `.env.example` work out of the box.

---

## Running the Web UI

The easiest way to use the tool is through the browser-based interface. Start the server from the `hvac-duct-detection/` directory:

```bash
cd hvac-duct-detection
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

**What the web UI does:**
1. Upload a PDF mechanical floor plan via drag-and-drop or file picker
2. The pipeline runs in the background — a progress indicator shows while processing
3. Results appear as a side-by-side view: annotated image on the left, measurement table on the right
4. **Click any highlighted duct region** on the image to see a popup with its dimensions, CFM, and pressure class
5. The full measurement table shows all segments with color-coded pressure class badges (Low / Medium / High)

---

## Running the Pipeline (CLI)

All commands are run from the **repository root** with the virtual environment active.

### Basic usage

```bash
python hvac-duct-detection/scripts/run_pipeline.py --pdf "path/to/plan.pdf"
```

### Full example with all options

```bash
python hvac-duct-detection/scripts/run_pipeline.py \
  --pdf "sample input/input.pdf" \
  --confidence 0.85 \
  --max-retries 3 \
  --pages "1-3"
```

### CLI Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `--pdf` | path | *(required)* | Path to the input PDF mechanical floor plan |
| `--confidence` | float | `0.85` | Minimum quality score (0.0–1.0) to accept output without retrying. Overrides `.env`. |
| `--max-retries` | int | `3` | Maximum reflexion retries before accepting best-effort output. Overrides `.env`. |
| `--scale-ratio` | float | auto-detect | Manual pixels-per-foot override. Useful when the drawing says "DO NOT SCALE". At 300 DPI, `1/4"=1'-0"` ≈ `75.0`. |
| `--pages` | string | all pages | 1-based page range to process. Supports ranges, lists, and single pages — e.g. `1-3`, `2,4,6`, `5`. |

---

## Output

Every run gets a **unique session directory** created automatically inside `hvac-duct-detection/outputs/`. You never need to specify an output path.

```
hvac-duct-detection/outputs/<session_id>/
├── pages/
│   ├── page_000.png                     Raw page render at 300 DPI
│   └── page_000_crop_<x1>_<y1>_...png  Focused crop regions (low-confidence re-checks)
├── page_000_annotated.png               Annotated page with colored duct overlays
├── annotated.pdf                        Final annotated PDF (all processed pages)
├── measurements.json                    Structured duct data — one record per segment
└── measurements.csv                     Same data in spreadsheet format
```

### Session ID

Each session ID follows the format `YYYYMMDD_HHMMSS_<6-char hex>` — for example `20260428_125629_1ecd8a`. This guarantees runs never overwrite each other.

### Terminal output

When the pipeline completes, a summary is printed:

```
========================================================
  HVAC Duct Detection — Pipeline Complete
========================================================
  session_id        : 20260428_125629_1ecd8a
  input_path        : /path/to/sample input/input.pdf
  output_dir        : hvac-duct-detection/outputs/20260428_125629_1ecd8a
  output_pdf        : .../20260428_125629_1ecd8a/annotated.pdf
  output_png        : .../20260428_125629_1ecd8a/page_000_annotated.png
  segments_detected : 15
  segments_labelled : 15
  review_score      : 1.0000
  retries           : 0
  Run log           : hvac-duct-detection/runs/registry.csv
========================================================
```

---

## Run Registry

Every pipeline run is automatically appended to a persistent CSV log at:

```
hvac-duct-detection/runs/registry.csv
```

This file is created on first run and acts as a history of all sessions.

| Column | Description |
|---|---|
| `session_id` | Unique run identifier |
| `timestamp` | Date and time the run completed |
| `input_path` | Absolute path to the input PDF |
| `output_dir` | Session output directory |
| `output_pdf` | Path to the annotated PDF |
| `output_png` | Semicolon-separated paths to annotated PNGs (one per page) |
| `segments_detected` | Total duct segments found |
| `segments_labelled` | Segments successfully matched to a dimension/CFM label |
| `review_score` | Final quality score (0.0–1.0) |
| `retries` | Number of reflexion retries used |

---

## Duct Type Color Coding

Duct overlays are color-coded by type so supply, return, and exhaust paths are immediately distinguishable:

| Type | Color | Hex |
|---|---|---|
| Supply | Blue | `#1565C0` |
| Return | Red | `#C62828` |
| Exhaust | Orange | `#E65100` |

### Pressure Class Badges (Web UI)

In the web UI, each segment row also shows a pressure class badge based on SMACNA size rules:

| Pressure Class | Color | Criteria |
|---|---|---|
| Low | Green | Round ≤10" or rectangular ≤80 sq-in |
| Medium | Orange | Round 11–18" or rectangular 81–250 sq-in |
| High | Red | Round >18" or rectangular >250 sq-in |

Explicit labels in the drawing (`LP`, `MP`, `HP`) always take precedence over the size-rule fallback.

---

## Measurement Data Schema

`measurements.json` contains one record per detected duct segment:

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
  "pressure_class": "Low",
  "pressure_reason": "10\" Ø ≤ 10\" — Low Pressure (SMACNA)",
  "bbox": [2362.5, 2241.0, 2902.5, 2484.0],
  "polygon": [[2362, 2241], [2902, 2241], [2902, 2484], [2362, 2484]],
  "unmatched": false
}
```

| Field | Description |
|---|---|
| `type` | `supply`, `return`, or `exhaust` |
| `is_round` | `true` for circular ducts (e.g. `10"Ø`), `false` for rectangular |
| `diameter_in` | Diameter in inches — round ducts only |
| `width_in` / `height_in` | Dimensions in inches — rectangular ducts only |
| `cfm` | Airflow in CFM if a nearby callout was found, otherwise `null` |
| `length_ft` | Duct run length in feet derived from the drawing scale; `null` when the drawing says "DO NOT SCALE" |
| `pressure_class` | `Low`, `Medium`, or `High` per SMACNA rules (explicit drawing label takes precedence) |
| `pressure_reason` | Human-readable explanation of how the pressure class was determined (shown in the web UI popup) |
| `bbox` | Bounding box in pixel coordinates `[x1, y1, x2, y2]` at 300 DPI |
| `polygon` | Full polygon vertex list in pixel coordinates — used by the web UI for click-to-inspect |
| `unmatched` | `true` if no dimension label could be associated with this segment |

---

## Reflexion Loop

If the review score is below `--confidence`, Claude generates targeted feedback and the pipeline retries automatically:

```
Attempt 1 → score 0.62  (below 0.85) → feedback generated → retry
Attempt 2 → score 0.89  (above 0.85) → accepted ✓
```

Retries use exponential back-off: `2 ^ retry_count` seconds — so 2 s, 4 s, 8 s, … between attempts.

Once `--max-retries` is exhausted, the best-effort result is accepted and a warning is logged.

---

## Running Tests

Tests live in `hvac-duct-detection/tests/` and are run from the `hvac-duct-detection/` directory.

```bash
cd hvac-duct-detection
source ../.venv/bin/activate   # if not already active

# Unit tests — no API key required
python -m pytest -q

# Unit + integration tests — requires ANTHROPIC_API_KEY in .env
python -m pytest -q -m integration
```

Test coverage includes:

| Test file | What it covers |
|---|---|
| `test_orchestrator.py` | State initialisation, summary building, reflexion loop, retry back-off |
| `test_pipeline.py` | End-to-end mocked pipeline + 3 live integration tests |
| `test_hardening.py` | Page range parsing, RGB mode, raster-only PDFs, polygon clamping, rate-limit retries, IoU dedup |
| `test_cli.py` | Session ID generation, registry writing, CLI argument handling |
| `test_ingestion.py` | Text extraction, scale detection |
| `test_vision.py` | Quadrant splitting, segment deduplication |
| `test_measurement.py` | Dimension regex patterns, label matching |
| `test_annotation.py` | Overlay drawing, PDF export |
| `test_review.py` | Confidence scoring formula, diff checker |

---

## Project Structure

```
hvac-duct-detection-agentic/          ← repository root
├── .env.example                      Template environment file
├── .gitignore
├── README.md
├── script.md                         Video walkthrough script
└── hvac-duct-detection/              Package root (add to sys.path)
    ├── app.py                        FastAPI web server + session management
    ├── requirements.txt
    ├── pytest.ini
    ├── agents/
    │   ├── orchestrator.py           Pipeline coordinator + reflexion loop
    │   ├── ingestion_agent.py        PDF ingestion
    │   ├── vision_agent.py           Duct detection via Claude vision
    │   ├── measurement_agent.py      Dimension + CFM + pressure class extraction
    │   ├── annotation_agent.py       Overlay rendering
    │   └── review_agent.py           Quality scoring + feedback generation
    ├── tools/
    │   ├── pdf_tools.py              PDF → image conversion, text extraction, scale detection
    │   ├── vision_tools.py           Vision API calls, quadrant split, IoU deduplication
    │   ├── measurement_tools.py      Regex dimension parsers, label matching
    │   ├── pressure_tools.py         Pressure class classification (SMACNA rules)
    │   ├── annotation_tools.py       Drawing primitives, label rendering, PDF export
    │   └── review_tools.py           Confidence scorer, diff checker
    ├── config/
    │   ├── settings.py               Pydantic-settings with .env auto-discovery
    │   └── prompts.py                All LLM prompt templates
    ├── models/
    │   └── duct_segment.py           Pydantic data models
    ├── scripts/
    │   └── run_pipeline.py           CLI entry point
    ├── static/
    │   ├── index.html                Single-page web UI (upload → results)
    │   ├── style.css                 Dark header, card layout, pressure class badges
    │   └── app.js                    SVG overlay, click-to-inspect, polling logic
    ├── uploads/                      Temporary upload storage (auto-cleaned)
    ├── outputs/                      Per-session output directories (git-ignored)
    ├── runs/
    │   └── registry.csv              Persistent run log (git-ignored)
    └── tests/
        ├── conftest.py               Shared fixtures
        ├── fixtures/                 JSON test data
        ├── test_orchestrator.py
        ├── test_pipeline.py
        ├── test_hardening.py
        ├── test_cli.py
        ├── test_ingestion.py
        ├── test_vision.py
        ├── test_measurement.py
        ├── test_annotation.py
        └── test_review.py
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `anthropic` | Claude API client (vision, text generation) |
| `strands-agents` | Agent SDK used to build each pipeline agent |
| `pymupdf` | PDF rendering and text extraction |
| `Pillow` | Image processing and overlay composition |
| `reportlab` | Annotated PDF assembly |
| `fastapi` | Web server framework for the browser-based UI |
| `uvicorn` | ASGI server for running the FastAPI app |
| `python-multipart` | Multipart form data parsing (file uploads) |
| `pydantic` / `pydantic-settings` | Data validation and `.env` config loading |
| `structlog` | Structured JSON logging |
| `python-dotenv` | `.env` file loading |
| `pytest` / `pytest-mock` | Test runner and mocking utilities |
