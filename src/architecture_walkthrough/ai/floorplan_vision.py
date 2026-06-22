from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, field_validator

from architecture_walkthrough.config import AISettings, AppConfig
from architecture_walkthrough.geometry.models import DoorOpening, Point2D, RoomPolygon, WallSegment, WindowOpening

LOGGER = logging.getLogger(__name__)


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


class FloorPlanVisionHints(BaseModel):
    walls: list[AIWallHint] = Field(default_factory=list)
    rooms: list[AIRoomHint] = Field(default_factory=list)
    openings: list[AIOpeningHint] = Field(default_factory=list)
    notes: str = ""


class OpenAIFloorPlanVisionAnalyzer:
    def __init__(self, settings: AISettings) -> None:
        self.settings = settings

    def is_available(self) -> bool:
        return self.settings.openai_enabled and bool(os.getenv("OPENAI_API_KEY"))

    def analyze(self, image_path: Path) -> FloorPlanVisionHints | None:
        if not self.is_available():
            return None
        try:
            from openai import OpenAI
        except ImportError:
            LOGGER.warning("OpenAI package is not installed; AI floor-plan hints disabled")
            return None

        mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
        image_data = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        prompt = (
            "Analyze this architectural floor-plan image. Return only JSON. "
            "Use normalized image coordinates from 0 to 1, origin at top-left. "
            "Identify straight wall centerlines, room polygons, and door/window openings only when visible. "
            "Prefer fewer high-confidence segments over noisy guesses. "
            "Do not invent hidden geometry."
        )
        schema = FloorPlanVisionHints.model_json_schema()
        client = OpenAI()
        try:
            response = client.chat.completions.create(
                model=self.settings.openai_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
                            },
                        ],
                    }
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "floorplan_vision_hints",
                        "strict": True,
                        "schema": schema,
                    },
                },
            )
            content = response.choices[0].message.content or "{}"
            return FloorPlanVisionHints.model_validate(json.loads(content))
        except (json.JSONDecodeError, ValidationError, Exception) as exc:
            LOGGER.warning("OpenAI floor-plan analysis failed; falling back to rule-based detection: %s", exc)
            return None


def hints_to_floorplan_geometry(
    hints: FloorPlanVisionHints,
    image_width_px: int,
    image_height_px: int,
    pixels_per_metre: float,
    config: AppConfig,
) -> tuple[list[WallSegment], list[RoomPolygon], list[DoorOpening], list[WindowOpening]]:
    def to_point(point: NormalizedPoint) -> Point2D:
        return Point2D(
            x=(point.x * image_width_px) / pixels_per_metre,
            y=(point.y * image_height_px) / pixels_per_metre,
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
        if hint.confidence >= config.ai.openai_min_confidence
    ]
    rooms = [
        RoomPolygon(name=hint.name, points=[to_point(point) for point in hint.points])
        for hint in hints.rooms
        if hint.confidence >= config.ai.openai_min_confidence
    ]
    doors: list[DoorOpening] = []
    windows: list[WindowOpening] = []
    for hint in hints.openings:
        if hint.confidence < config.ai.openai_min_confidence:
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
    return walls, rooms, doors, windows
