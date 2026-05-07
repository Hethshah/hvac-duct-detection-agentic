#!/usr/bin/env python3
"""
HVAC Duct Detection — Interactive web application.

Usage:
    cd hvac-duct-detection-v2
    python3 web_app.py
    Open http://localhost:5000
"""

import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fitz
from flask import Flask, jsonify, render_template, request, send_file

from config.settings import RENDER_DPI, OUTPUTS_DIR
from tools.annotation_renderer import _centerline_pixels, render_annotated_page
from tools.duct_annotator import annotate_ducts
from tools.label_extractor import extract_labels_with_scale
from tools.raster_duct_extractor import extract_raster_ducts
from tools.vector_duct_extractor import extract_ducts

BASE_DIR    = Path(__file__).resolve().parent
UPLOAD_DIR  = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB


# ── Duct metadata helpers ─────────────────────────────────────────────────────

def _infer_duct_type(duct) -> str:
    lid = (duct.duct_label_id or "").upper()
    if any(lid.startswith(p) for p in ("SA", "SUP", "S-", "SB")):
        return "supply"
    if any(lid.startswith(p) for p in ("RA", "RET", "R-", "RB")):
        return "return"
    if any(lid.startswith(p) for p in ("EA", "EXH", "EF", "E-", "OA")):
        return "exhaust"
    return "supply"


def _infer_pressure_class(duct) -> str:
    cs = duct.cross_section
    if not cs:
        return "low"
    dim = (
        cs.get("diameter_in", 0)
        if duct.is_round
        else max(cs.get("width_in", 0), cs.get("height_in", 0))
    )
    if dim >= 24:
        return "low"
    if dim >= 12:
        return "medium"
    return "high"


def _format_dimension(duct) -> str | None:
    cs = duct.cross_section
    if not cs:
        return None
    if duct.is_round:
        return f'Ø {cs["diameter_in"]}"'
    return f'{cs["width_in"]}" × {cs["height_in"]}"'


# ── Annotated PDF creation ────────────────────────────────────────────────────

def create_annotated_pdf(pdf_path: str, annotated: list, output_path: str, page_index: int = 0) -> None:
    """
    Draw blue centerlines directly on the PDF as vector graphics, then save.
    Uses PyMuPDF drawing API so lines remain vector quality (no rasterization).
    """
    BLUE  = (55 / 255, 98 / 255, 227 / 255)
    LINE_W = 2.0  # points

    doc  = fitz.open(pdf_path)
    page = doc[page_index]

    for duct in annotated:
        x0, y0, x1, y1 = duct.rect
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

        if duct.orientation == "H":
            p1, p2 = fitz.Point(x0, cy), fitz.Point(x1, cy)
        elif duct.orientation == "V":
            p1, p2 = fitz.Point(cx, y0), fitz.Point(cx, y1)
        elif duct.centerline and len(duct.centerline) == 2:
            p1 = fitz.Point(*duct.centerline[0])
            p2 = fitz.Point(*duct.centerline[1])
        else:
            p1, p2 = fitz.Point(x0, y0), fitz.Point(x1, y1)

        page.draw_line(p1, p2, color=BLUE, width=LINE_W)

    doc.save(str(output_path))
    doc.close()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "pdf" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["pdf"]
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Please upload a PDF file"}), 400

    sid  = str(uuid.uuid4())
    stem = Path(file.filename).stem

    # Session dir (for web serving)
    session_dir = UPLOAD_DIR / sid
    session_dir.mkdir(parents=True)
    pdf_path = session_dir / "input.pdf"
    file.save(str(pdf_path))

    # Persistent outputs dir
    out_dir       = OUTPUTS_DIR / stem
    artifacts_dir = out_dir / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(exist_ok=True)

    # ── Phases 1-4 ───────────────────────────────────────────────────────────
    vector_segs = extract_ducts(str(pdf_path))
    p2          = extract_labels_with_scale(str(pdf_path), vector_segs)
    raster_segs = extract_raster_ducts(str(pdf_path), vector_segs)
    all_segs    = vector_segs + raster_segs
    annotated   = annotate_ducts(all_segs, p2["labels"], p2["pt_per_ft"])

    # ── Phase 6: render annotated PNG ────────────────────────────────────────
    png_session = session_dir / "annotated.png"
    png_output  = out_dir / f"{stem}_annotated.png"
    render_annotated_page(str(pdf_path), annotated, str(png_session))

    # Copy PNG to persistent output
    import shutil
    shutil.copy2(str(png_session), str(png_output))

    # ── Annotated PDF ────────────────────────────────────────────────────────
    pdf_output = out_dir / f"{stem}_annotated.pdf"
    create_annotated_pdf(str(pdf_path), annotated, str(pdf_output))
    # Copy to session dir for serving
    shutil.copy2(str(pdf_output), str(session_dir / "annotated.pdf"))

    # ── Save artifacts ───────────────────────────────────────────────────────
    phase4_path = str(artifacts_dir / "phase4_annotated_ducts.json")
    annotate_ducts(all_segs, p2["labels"], p2["pt_per_ft"], output_path=phase4_path)

    summary = {
        "input": file.filename,
        "pt_per_ft": round(p2["pt_per_ft"], 4),
        "segment_counts": {
            "vector": len(vector_segs),
            "raster": len(raster_segs),
            "total": len(all_segs),
        },
        "annotated_ducts": [d.to_dict() for d in annotated],
    }
    (artifacts_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    labels_payload = {
        "pdf": file.filename,
        "pt_per_ft": round(p2["pt_per_ft"], 4),
        "label_count": len(p2["labels"]),
        "labels": p2["labels"],
    }
    (artifacts_dir / "phase2_labels.json").write_text(json.dumps(labels_payload, indent=2))

    # ── Compute pixel hotspot coordinates ────────────────────────────────────
    doc     = fitz.open(str(pdf_path))
    page    = doc[0]
    rotation = page.rotation
    media_w  = int(page.mediabox.width)
    media_h  = int(page.mediabox.height)
    scale    = RENDER_DPI / 72.0
    doc.close()

    ducts_out = []
    for i, duct in enumerate(annotated, start=1):
        p1, p2_px = _centerline_pixels(duct, rotation, media_w, media_h, scale)
        ducts_out.append({
            "idx":              i,
            "p1":               list(p1),
            "p2":               list(p2_px),
            "duct_type":        _infer_duct_type(duct),
            "pressure_class":   _infer_pressure_class(duct),
            "measurement_type": "round" if duct.is_round else "rectangular",
            "dimension":        _format_dimension(duct),
            "length_ft":        round(duct.length_ft_measured, 2),
            "segment_id":       duct.segment_id,
            "label_id":         duct.duct_label_id,
            "mismatch":         duct.length_mismatch,
        })

    return jsonify({
        "session_id": sid,
        "stem":       stem,
        "image_url":  f"/session/{sid}/image",
        "pdf_url":    f"/session/{sid}/pdf",
        "ducts":      ducts_out,
        "pt_per_ft":  round(p2["pt_per_ft"], 4),
        "outputs": {
            "png": str(png_output),
            "pdf": str(pdf_output),
            "artifacts": str(artifacts_dir),
        },
    })


@app.route("/session/<sid>/image")
def get_image(sid):
    path = UPLOAD_DIR / sid / "annotated.png"
    if not path.exists():
        return "Not found", 404
    return send_file(str(path.resolve()), mimetype="image/png")


@app.route("/session/<sid>/pdf")
def get_pdf(sid):
    path = UPLOAD_DIR / sid / "annotated.pdf"
    if not path.exists():
        return "Not found", 404
    return send_file(str(path.resolve()), mimetype="application/pdf",
                     as_attachment=True, download_name="annotated.pdf")


if __name__ == "__main__":
    print("HVAC Duct Inspector → http://localhost:5000")
    app.run(debug=False, port=5000, host="0.0.0.0")
