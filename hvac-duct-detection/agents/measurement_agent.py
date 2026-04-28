import csv
import json
from pathlib import Path

import structlog

from config.settings import settings
from tools.measurement_tools import dimension_extractor, label_matcher

logger = structlog.get_logger()


def run_measurement(state: dict) -> dict:
    """
    Run measurement pipeline: associate dimension/CFM labels with each duct segment.
    Populates state["measurements"] and exports measurements.json + measurements.csv.
    """
    duct_segments: list[dict] = state.get("duct_segments", [])
    text_blocks: list[dict] = state.get("text_blocks", [])
    scale_ratio: float = state.get("scale_ratio", 0.0)
    output_dir = state.get("output_dir", settings.output_dir)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Extract all dimension/CFM labels from OCR text blocks (used as fallback)
    all_dim_labels_json = dimension_extractor(json.dumps(text_blocks))

    measurements: list[dict] = []
    matched_count = 0
    unmatched_count = 0

    for seg in duct_segments:
        record_json = label_matcher(
            json.dumps(seg),
            all_dim_labels_json,
            scale_ratio,
        )
        record = json.loads(record_json)
        measurements.append(record)

        if record["unmatched"]:
            unmatched_count += 1
        else:
            matched_count += 1

    state["measurements"] = measurements

    _export_json(measurements, output_dir)
    _export_csv(measurements, output_dir)

    confs = [seg.get("confidence", 0.0) for seg in duct_segments]
    avg_conf = round(sum(confs) / len(confs), 3) if confs else 0.0

    logger.info(
        "measurement_complete",
        total=len(measurements),
        matched=matched_count,
        unmatched=unmatched_count,
        avg_confidence=avg_conf,
    )
    return state


def _export_json(measurements: list[dict], output_dir: str) -> None:
    path = Path(output_dir) / "measurements.json"
    with open(path, "w") as f:
        json.dump(measurements, f, indent=2)
    logger.info("exported_json", path=str(path))


def _export_csv(measurements: list[dict], output_dir: str) -> None:
    path = Path(output_dir) / "measurements.csv"
    fieldnames = [
        "segment_id", "type", "is_round", "diameter_in",
        "width_in", "height_in", "cfm", "length_ft",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "unmatched",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in measurements:
            bbox = m.get("bbox", [0, 0, 0, 0])
            writer.writerow({
                "segment_id": m["segment_id"],
                "type": m["type"],
                "is_round": m["is_round"],
                "diameter_in": m.get("diameter_in", ""),
                "width_in": m.get("width_in", ""),
                "height_in": m.get("height_in", ""),
                "cfm": m.get("cfm", ""),
                "length_ft": m.get("length_ft", ""),
                "bbox_x1": round(bbox[0]) if len(bbox) > 0 else "",
                "bbox_y1": round(bbox[1]) if len(bbox) > 1 else "",
                "bbox_x2": round(bbox[2]) if len(bbox) > 2 else "",
                "bbox_y2": round(bbox[3]) if len(bbox) > 3 else "",
                "unmatched": m["unmatched"],
            })
    logger.info("exported_csv", path=str(path))
