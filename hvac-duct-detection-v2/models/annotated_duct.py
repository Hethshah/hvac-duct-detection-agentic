from dataclasses import dataclass, field


@dataclass
class AnnotatedDuct:
    segment_id: str                   # "duct_001" — the Phase 1/3 segment ID
    duct_label_id: str | None         # "C03", "C01" etc. — from ID label text; None if unknown
    rect: list[float]                 # [x0, y0, x1, y1] PDF points
    orientation: str                  # "H", "V", "D"
    length_ft_measured: float         # long_pt / pt_per_ft
    length_ft_label: float | None     # from matched length label; None if no label
    length_mismatch: bool             # |measured - label| / label > MISMATCH_THRESHOLD
    cross_section: dict | None        # {"width_in":24,"height_in":18} or {"diameter_in":12}
    is_round: bool
    unlabeled: bool                   # True when neither length nor cross-section found
    confidence: float
    source: str                       # "vector" | "raster"
    page: int
    centerline: list[list[float]] | None = field(default=None)  # [[x0,y0],[x1,y1]] media coords

    def to_dict(self) -> dict:
        return {
            "id": self.duct_label_id,
            "duct_idx": self.segment_id,
            "rect": [round(v, 2) for v in self.rect],
            "orientation": self.orientation,
            "length_ft_measured": round(self.length_ft_measured, 3),
            "length_ft_label": round(self.length_ft_label, 4) if self.length_ft_label is not None else None,
            "length_mismatch": self.length_mismatch,
            "cross_section": self.cross_section,
            "is_round": self.is_round,
            "unlabeled": self.unlabeled,
            "confidence": round(self.confidence, 3),
            "source": self.source,
            "page": self.page,
        }
