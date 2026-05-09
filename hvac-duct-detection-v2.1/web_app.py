#!/usr/bin/env python3
"""
HVAC Duct Detection — Agentic web application (v2.1)

Usage:
    cd hvac-duct-detection-v2.1
    python3 web_app.py
    Open http://localhost:5001

pkill -f "python3 web_app.py"
To kill the application
"""

import json
import sys
import threading
import queue
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

V2_DIR = Path(__file__).resolve().parent.parent / "hvac-duct-detection-v2"
sys.path.insert(0, str(V2_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

import fitz
from flask import Flask, Response, jsonify, render_template, request, send_file

from agent import run_agent
from config.settings import RENDER_DPI, OUTPUTS_DIR
from tools.annotation_renderer import _centerline_pixels

BASE_DIR   = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB

analysis_sessions: dict[str, dict] = {}


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
    """Draw blue centerlines directly on the PDF as vector graphics, then save."""
    BLUE  = (55 / 255, 98 / 255, 227 / 255)
    LINE_W = 2.0

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


# ── Background agent thread ───────────────────────────────────────────────────

def run_agent_thread(sid: str, pdf_path: str, q: queue.Queue) -> None:
    try:
        stem = Path(pdf_path).stem
        out_dir = OUTPUTS_DIR / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        session_dir = UPLOAD_DIR / sid

        def on_event(event_type: str, data: dict) -> None:
            q.put({"type": event_type, "data": data})

        annotated, summary = run_agent(
            pdf_path,
            output_dir=out_dir,
            on_event=on_event,
        )

        # Render session PNG
        png_session = session_dir / "annotated.png"
        from tools.annotation_renderer import render_annotated_page
        render_annotated_page(pdf_path, annotated, str(png_session))

        # Annotated PDF for session serving
        pdf_session = session_dir / "annotated.pdf"
        create_annotated_pdf(pdf_path, annotated, str(pdf_session))

        # Compute pixel hotspot coordinates
        doc = fitz.open(pdf_path)
        page = doc[0]
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

        analysis_sessions[sid]["annotated"] = annotated

        q.put({
            "type": "done",
            "data": {
                "ducts":      ducts_out,
                "image_url":  f"/session/{sid}/image",
                "pdf_url":    f"/session/{sid}/pdf",
                "pt_per_ft":  round(summary.get("pt_per_ft", 0.0), 4),
                "session_id": sid,
            },
        })
    except Exception as e:
        q.put({"type": "error", "data": {"error": str(e)}})


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

    sid = str(uuid.uuid4())
    session_dir = UPLOAD_DIR / sid
    session_dir.mkdir(parents=True)
    pdf_path = str(session_dir / "input.pdf")
    file.save(pdf_path)

    q = queue.Queue()
    analysis_sessions[sid] = {
        "queue":    q,
        "annotated": None,
        "pdf_path": pdf_path,
    }

    t = threading.Thread(target=run_agent_thread, args=(sid, pdf_path, q), daemon=True)
    t.start()

    return jsonify({"session_id": sid, "stream_url": f"/session/{sid}/stream"})


@app.route("/session/<sid>/stream")
def stream(sid):
    if sid not in analysis_sessions:
        return "Session not found", 404

    q = analysis_sessions[sid]["queue"]

    def generate():
        while True:
            try:
                event = q.get(timeout=60)
            except queue.Empty:
                yield "event: heartbeat\ndata: {}\n\n"
                continue

            yield f"event: {event['type']}\ndata: {json.dumps(event['data'])}\n\n"

            if event["type"] in ("done", "error"):
                break

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


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
    return send_file(
        str(path.resolve()),
        mimetype="application/pdf",
        as_attachment=True,
        download_name="annotated.pdf",
    )


if __name__ == "__main__":
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    print("HVAC Duct Inspector v2.1 (Agentic) → http://localhost:5001")
    app.run(debug=False, port=5001, host="0.0.0.0")
