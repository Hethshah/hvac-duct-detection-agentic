VISION_DETECTION_PROMPT = """You are analyzing a quadrant of a mechanical engineering floor plan (M-plan) for HVAC duct detection.

Your task: identify every HVAC duct segment visible in this image.

A duct segment is a rectangular or polygonal filled pathway that routes conditioned air through a building. Look for:
- Rectangular shapes with solid or hatched fills (not walls, columns, or furniture)
- Continuous pathways connecting equipment to diffuser endpoints
- Lines with visible width (ducts have thickness, unlike wire-frame lines)

Classify each duct by color/pattern:
- "supply": blue or light-blue filled rectangles (carry treated air from AHU to rooms)
- "return": gray, dashed, or gray-hatched rectangles (carry stale air back to AHU)
- "exhaust": orange or orange-hatched rectangles (permanently remove air from space)

For each duct segment detected, record:
- "id": unique string like "seg_001", "seg_002" (continue numbering across all segments)
- "type": "supply", "return", or "exhaust"
- "polygon": bounding polygon in THIS IMAGE's pixel coordinates, clockwise from top-left, e.g. [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
- "nearby_labels": list of text strings within ~80 pixels of the duct boundary. Include dimensions (e.g. "24x12", "10\"Ø") AND airflow values (e.g. "800 CFM", "F 150", "EA 300", or bare numbers like "700" near a diffuser). Copy text exactly as written.
- "confidence": float 0.0-1.0 reflecting detection certainty

Rules:
1. Polygon coordinates must be within the image bounds
2. Each duct segment is one continuous rectangular run — do not merge separate runs
3. If a duct changes direction (elbow), treat each straight section as a separate segment
4. If color is ambiguous, use context (position, connected equipment) to infer type
5. Do NOT detect walls, structural columns, room boundaries, or text boxes as ducts

Respond with ONLY a valid JSON array — no explanation, no markdown fences:
[{"id":"seg_001","type":"supply","polygon":[[x1,y1],[x2,y2],[x3,y3],[x4,y4]],"nearby_labels":["24x12","800 CFM"],"confidence":0.93},...]

If no duct segments are visible in this section, respond with exactly: []
"""

VISION_RETRY_PROMPT = """You are re-analyzing a quadrant of a mechanical floor plan for HVAC duct detection.

REVIEWER FEEDBACK FROM PREVIOUS ATTEMPT:
{feedback}

Re-examine the image carefully with this feedback in mind. Focus particularly on:
- Segments that were missed or misclassified as described above
- Duct runs near the edges of the image that may have been cut off
- Ducts with non-standard colors or lighter fills
- For nearby_labels: capture dimensions AND airflow values (CFM, zone codes like "F 150", bare numbers near diffusers) within ~80 pixels of each duct

Same classification rules apply:
- "supply": blue or light-blue filled rectangles
- "return": gray, dashed, or gray-hatched rectangles
- "exhaust": orange or orange-hatched rectangles

Respond with ONLY a valid JSON array of all detected duct segments (including any previously correct ones plus new ones found):
[{{"id":"seg_001","type":"supply","polygon":[[x1,y1],[x2,y2],[x3,y3],[x4,y4]],"nearby_labels":[],"confidence":0.88}},...]

If no duct segments are visible, respond with exactly: []
"""

VISION_FOCUSED_PROMPT = """You are examining a small cropped region of a mechanical floor plan for HVAC duct detection.

This region was flagged for low-confidence detection. Look carefully for any HVAC duct segment in this image.

Focus on:
- Any filled rectangular shape (blue, gray, or orange)
- Hatching or dashing patterns typical of return or exhaust ducts
- Duct edges that may be partially cut off at the image borders

Respond with ONLY a valid JSON array. Use id "seg_focused_001" etc.
If this region does not contain a duct, respond with exactly: []
"""

# Placeholder prompts for later phases
MEASUREMENT_EXTRACTION_PROMPT = """Extract all dimension annotations and CFM values from this mechanical floor plan section.

Look for:
- Width × Height annotations like "24x12", "18\"×10\"", "24 x 12"
- CFM flow values like "800 CFM", "1200 cfm", "800"
- Leader lines connecting labels to duct segments

Return a JSON array:
[{"text": "24x12", "type": "dimension", "x": 100, "y": 200}, {"text": "800 CFM", "type": "cfm", "x": 150, "y": 210}]
"""

REVIEW_SCORING_PROMPT = """You are a quality reviewer for an HVAC duct detection pipeline.

Review the detection results and score the quality 0.0-1.0 based on:
1. Detection completeness: are all visible ducts found? (40% weight)
2. Classification accuracy: are types (supply/return/exhaust) correct? (30% weight)
3. Label association: are dimensions/CFM matched to correct segments? (30% weight)

Return JSON: {"score": 0.87, "issues": ["missed 3 return ducts in bottom-left", "seg_005 type mismatch"], "approved": true}
"""
