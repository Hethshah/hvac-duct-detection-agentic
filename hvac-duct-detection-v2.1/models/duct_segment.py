from dataclasses import dataclass, field


@dataclass
class DuctSegment:
    id: str
    rect: list[float]        # [x0, y0, x1, y1] in PDF points (axis-aligned bbox)
    orientation: str         # "H", "V", or "D" (diagonal)
    long_pt: float           # length of the long axis in PDF points
    short_pt: float          # length of the short axis (duct width) in PDF points
    aspect: float
    centerline: list[list[float]]  # [[x_start, y_start], [x_end, y_end]]
    source: str = "vector"   # "vector" | "raster"
    confidence: float = 1.0
    page: int = 0
    # For diagonal ducts: actual polygon vertices in media PDF-point coords
    # Each entry is [x, y]. Forms a closed polygon (first != last).
    polygon: list[list[float]] | None = None

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "rect": [round(v, 2) for v in self.rect],
            "orientation": self.orientation,
            "long_pt": round(self.long_pt, 2),
            "short_pt": round(self.short_pt, 2),
            "aspect": round(self.aspect, 2),
            "centerline": [[round(v, 2) for v in pt] for pt in self.centerline],
            "source": self.source,
            "confidence": round(self.confidence, 3),
            "page": self.page,
        }
        if self.polygon is not None:
            d["polygon"] = [[round(v, 2) for v in pt] for pt in self.polygon]
        return d
