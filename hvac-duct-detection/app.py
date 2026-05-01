"""
FastAPI web server for the HVAC Duct Detection pipeline.

Run from the hvac-duct-detection/ directory:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""
import json
import traceback
import uuid
from pathlib import Path

import structlog
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from agents.orchestrator import run_pipeline

logger = structlog.get_logger()

app = FastAPI(title="HVAC Duct Detection")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_sessions: dict[str, dict] = {}

_UPLOADS = Path(__file__).parent / "uploads"
_UPLOADS.mkdir(exist_ok=True)
_STATIC = Path(__file__).parent / "static"


# ── Background pipeline task ─────────────────────────────────────────────────

def _run(session_id: str, upload_path: str) -> None:
    try:
        summary = run_pipeline(upload_path)
        out_dir = Path(summary["output_dir"])

        meas_path = out_dir / "measurements.json"
        measurements = json.loads(meas_path.read_text()) if meas_path.exists() else []

        pngs = summary.get("output_pngs", [])
        img_w = img_h = 0
        if pngs and Path(pngs[0]).exists():
            with Image.open(pngs[0]) as img:
                img_w, img_h = img.size

        _sessions[session_id].update({
            "status": "complete",
            "summary": summary,
            "measurements": measurements,
            "img_width": img_w,
            "img_height": img_h,
            "png_path": pngs[0] if pngs else None,
        })
        logger.info("session_complete", session_id=session_id,
                    segments=len(measurements))
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("session_error", session_id=session_id, error=str(exc), traceback=tb)
        _sessions[session_id].update({"status": "error", "error": str(exc), "traceback": tb})
    finally:
        Path(upload_path).unlink(missing_ok=True)


# ── API routes ────────────────────────────────────────────────────────────────

@app.post("/api/process")
async def process_drawing(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Accept a PDF or image upload and start the detection pipeline."""
    session_id = f"web_{uuid.uuid4().hex[:10]}"
    upload_path = _UPLOADS / f"{session_id}_{file.filename}"

    content = await file.read()
    upload_path.write_bytes(content)

    _sessions[session_id] = {"status": "processing"}
    background_tasks.add_task(_run, session_id, str(upload_path))

    logger.info("session_started", session_id=session_id, filename=file.filename)
    return {"session_id": session_id, "status": "processing"}


@app.get("/api/session/{session_id}/status")
async def get_status(session_id: str):
    """Poll processing status: processing | complete | error."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    s = _sessions[session_id]
    return {"status": s["status"], "error": s.get("error")}


@app.get("/api/session/{session_id}/result")
async def get_result(session_id: str):
    """Return measurements + image dimensions once processing is complete."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    s = _sessions[session_id]
    if s["status"] != "complete":
        raise HTTPException(status_code=400, detail="Processing not complete")
    return {
        "measurements": s["measurements"],
        "img_width": s["img_width"],
        "img_height": s["img_height"],
        "summary": s["summary"],
    }


@app.get("/api/session/{session_id}/image")
async def get_image(session_id: str):
    """Stream the annotated PNG for the given session."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    png = _sessions[session_id].get("png_path")
    if not png or not Path(png).exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(png, media_type="image/png")


# ── Static frontend (must be mounted last) ────────────────────────────────────
app.mount("/", StaticFiles(directory=str(_STATIC), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
