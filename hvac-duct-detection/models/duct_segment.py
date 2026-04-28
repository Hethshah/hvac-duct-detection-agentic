from typing import Literal, Optional
from pydantic import BaseModel


class DuctSegment(BaseModel):
    id: str
    type: Literal["supply", "return", "exhaust"]
    polygon: list[list[float]]
    nearby_labels: list[str] = []
    confidence: float
    page: int = 0


class MeasurementRecord(BaseModel):
    segment_id: str
    type: str
    is_round: bool = False         # True for circular ducts (e.g. 8"Ø)
    diameter_in: Optional[int] = None   # round duct diameter
    width_in: Optional[int] = None      # rectangular duct width
    height_in: Optional[int] = None     # rectangular duct height
    cfm: Optional[int] = None
    length_ft: Optional[float] = None
    bbox: list[float] = []
    unmatched: bool = False


class ReviewResult(BaseModel):
    score: float
    issues: list[str]
    approved: bool


class PipelineState(BaseModel):
    pdf_path: str
    output_dir: str = "outputs"
    page_images: list[str] = []
    text_blocks: list[dict] = []
    scale_ratio: float = 0.0
    duct_segments: list[dict] = []
    measurements: list[dict] = []
    output_pdf: str = ""
    review_score: float = 0.0
    retry_count: int = 0
    reviewer_feedback: str = ""
