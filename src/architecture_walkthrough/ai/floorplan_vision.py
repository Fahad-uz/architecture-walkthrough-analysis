from __future__ import annotations

import json
import logging
import mimetypes
import os
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, field_validator

from architecture_walkthrough.config import AISettings, AppConfig
from architecture_walkthrough.geometry.models import (
    DoorOpening,
    FurniturePlacement,
    Point2D,
    RoomPolygon,
    WallSegment,
    WindowOpening,
)

LOGGER = logging.getLogger(__name__)


class GeminiFloorPlanVisionError(RuntimeError):
    pass


class NormalizedPoint(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)


class AIWallHint(BaseModel):
    start: NormalizedPoint
    end: NormalizedPoint
    confidence: float = Field(ge=0.0, le=1.0)
    external: bool = False


class AIRoomHint(BaseModel):
    name: str | None = None
    points: list[NormalizedPoint]
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("points")
    @classmethod
    def require_polygon(cls, value: list[NormalizedPoint]) -> list[NormalizedPoint]:
        if len(value) < 3:
            raise ValueError("room hint requires at least three polygon points")
        return value


class AIOpeningHint(BaseModel):
    kind: str
    center: NormalizedPoint
    confidence: float = Field(ge=0.0, le=1.0)


class AIFurnitureHint(BaseModel):
    category: str
    center: NormalizedPoint
    width: float = Field(gt=0.0, le=1.0)
    depth: float = Field(gt=0.0, le=1.0)
    rotation_deg: float = 0.0
    confidence: float = Field(ge=0.0, le=1.0)


class AIDimensionText(BaseModel):
    text: str
    center: NormalizedPoint
    confidence: float = Field(ge=0.0, le=1.0)


class FloorPlanVisionHints(BaseModel):
    walls: list[AIWallHint] = Field(default_factory=list)
    rooms: list[AIRoomHint] = Field(default_factory=list)
    openings: list[AIOpeningHint] = Field(default_factory=list)
    furniture: list[AIFurnitureHint] = Field(default_factory=list)
    dimension_texts: list[AIDimensionText] = Field(default_factory=list)
    notes: str = ""


class FloorPlanVisionAnalysis(BaseModel):
    attempted: bool = False
    succeeded: bool = False
    hints: FloorPlanVisionHints | None = None
    error: str | None = None


def _strict_schema() -> dict:
    point_schema = {
        "type": "object",
        "required": ["x", "y"],
        "properties": {
            "x": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "y": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
    }
    return {
        "type": "object",
        "required": ["walls", "rooms", "openings", "furniture", "notes"],
        "properties": {
            "walls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["start", "end", "confidence", "external"],
                    "properties": {
                        "start": point_schema,
                        "end": point_schema,
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "external": {"type": "boolean"},
                    },
                },
            },
            "rooms": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["name", "points", "confidence"],
                    "properties": {
                        "name": {"type": "string"},
                        "points": {"type": "array", "minItems": 3, "items": point_schema},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    },
                },
            },
            "openings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["kind", "center", "confidence"],
                    "properties": {
                        "kind": {"type": "string", "enum": ["door", "window"]},
                        "center": point_schema,
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    },
                },
            },
            "furniture": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["category", "center", "width", "depth", "rotation_deg", "confidence"],
                    "properties": {
                        "category": {"type": "string"},
                        "center": point_schema,
                        "width": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "depth": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "rotation_deg": {"type": "number"},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    },
                },
            },
            "dimension_texts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["text", "center", "confidence"],
                    "properties": {
                        "text": {"type": "string"},
                        "center": point_schema,
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    },
                },
            },
            "notes": {"type": "string"},
        },
    }


class GeminiFloorPlanVisionAnalyzer:
    def __init__(self, settings: AISettings) -> None:
        self.settings = settings

    def is_available(self) -> bool:
        return self.settings.gemini_enabled and bool(os.getenv("GEMINI_API_KEY"))

    def analyze_with_diagnostics(self, image_path: Path, require_success: bool = False) -> FloorPlanVisionAnalysis:
        if not self.settings.gemini_enabled:
            error = "Gemini vision is disabled in configuration"
            if require_success:
                raise GeminiFloorPlanVisionError(error)
            return FloorPlanVisionAnalysis(error=error)
        if not os.getenv("GEMINI_API_KEY"):
            error = "GEMINI_API_KEY is not set for the running server process"
            if require_success:
                raise GeminiFloorPlanVisionError(error)
            return FloorPlanVisionAnalysis(attempted=True, error=error)
        try:
            hints = self._request_hints(image_path)
            return FloorPlanVisionAnalysis(attempted=True, succeeded=True, hints=hints)
        except GeminiFloorPlanVisionError as exc:
            LOGGER.warning("%s", exc)
            if require_success:
                raise
            return FloorPlanVisionAnalysis(attempted=True, error=str(exc))
        except (json.JSONDecodeError, ValidationError, Exception) as exc:
            error = f"Gemini floor-plan analysis failed: {exc}"
            LOGGER.warning("%s", error)
            if require_success:
                raise GeminiFloorPlanVisionError(error) from exc
            return FloorPlanVisionAnalysis(attempted=True, error=error)

    def analyze(self, image_path: Path) -> FloorPlanVisionHints | None:
        return self.analyze_with_diagnostics(image_path).hints

    def _request_hints(self, image_path: Path) -> FloorPlanVisionHints:
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise GeminiFloorPlanVisionError("google-genai package is not installed")

        mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
        image_bytes = image_path.read_bytes()
        prompt = (
            "Analyze this architectural floor-plan image. Return only JSON. "
            "Use normalized image coordinates from 0 to 1, origin at top-left. "
            "Identify straight wall centerlines as one segment per wall, not both wall edges. "
            "Identify room polygons, door/window openings, and furniture footprints separately. "
            "Furniture includes beds, sofas, chairs, tables, kitchen counters, wardrobes, fixtures, and plants. "
            "Also transcribe any printed dimension annotations (e.g. '3.2m x 4.0m', '300X420') "
            "into dimension_texts with the text exactly as written and its center position. "
            "Do not classify furniture outlines, labels, tiles, stairs, or shadows as walls. "
            "Prefer fewer high-confidence segments over noisy guesses. "
            "Do not invent hidden geometry. "
            "Return empty arrays when uncertain, but include all required keys."
        )
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        response = client.models.generate_content(
            model=self.settings.gemini_model,
            contents=[
                prompt,
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_strict_schema(),
            ),
        )
        content = getattr(response, "text", None)
        if not content:
            raise GeminiFloorPlanVisionError("Gemini response did not contain text")
        return FloorPlanVisionHints.model_validate(json.loads(content))

def hints_to_floorplan_geometry(
    hints: FloorPlanVisionHints,
    image_width_px: int,
    image_height_px: int,
    pixels_per_metre: float,
    config: AppConfig,
) -> tuple[
    list[WallSegment],
    list[RoomPolygon],
    list[DoorOpening],
    list[WindowOpening],
    list[FurniturePlacement],
]:
    def to_point(point: NormalizedPoint) -> Point2D:
        return Point2D(
            x=(point.x * image_width_px) / pixels_per_metre,
            y=((1.0 - point.y) * image_height_px) / pixels_per_metre,
        )

    walls = [
        WallSegment(
            start=to_point(hint.start),
            end=to_point(hint.end),
            thickness_m=config.defaults.external_wall_thickness_m if hint.external else config.defaults.internal_wall_thickness_m,
            height_m=config.defaults.wall_height_m,
            external=hint.external,
        )
        for hint in hints.walls
        if hint.confidence >= config.ai.gemini_min_confidence
    ]
    rooms = [
        RoomPolygon(name=hint.name, points=[to_point(point) for point in hint.points])
        for hint in hints.rooms
        if hint.confidence >= config.ai.gemini_min_confidence
    ]
    doors: list[DoorOpening] = []
    windows: list[WindowOpening] = []
    for hint in hints.openings:
        if hint.confidence < config.ai.gemini_min_confidence:
            continue
        if hint.kind.lower() == "door":
            doors.append(DoorOpening(center=to_point(hint.center), width_m=config.defaults.door_width_m))
        elif hint.kind.lower() == "window":
            windows.append(
                WindowOpening(
                    center=to_point(hint.center),
                    width_m=config.defaults.window_width_m,
                    sill_height_m=config.defaults.sill_height_m,
                )
            )
    furniture = [
        FurniturePlacement(
            category=hint.category,
            center=to_point(hint.center),
            width_m=(hint.width * image_width_px) / pixels_per_metre,
            depth_m=(hint.depth * image_height_px) / pixels_per_metre,
            rotation_deg=hint.rotation_deg,
        )
        for hint in hints.furniture
        if hint.confidence >= config.ai.gemini_min_confidence
    ]
    return walls, rooms, doors, windows, furniture
