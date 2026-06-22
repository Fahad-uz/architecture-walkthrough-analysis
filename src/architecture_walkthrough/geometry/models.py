from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, field_validator


class CoordinateSystem(str, Enum):
    PIXELS = "pixels"
    METRES = "metres"


class Point2D(BaseModel):
    x: float
    y: float

    def distance_to(self, other: "Point2D") -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


class WallSegment(BaseModel):
    start: Point2D
    end: Point2D
    thickness_m: PositiveFloat = 0.12
    height_m: PositiveFloat = 3.0
    external: bool = False


class DoorOpening(BaseModel):
    center: Point2D
    width_m: PositiveFloat = 0.90
    wall_id: str | None = None


class WindowOpening(BaseModel):
    center: Point2D
    width_m: PositiveFloat = 1.20
    sill_height_m: PositiveFloat = 0.90
    wall_id: str | None = None


class RoomPolygon(BaseModel):
    name: str | None = None
    points: list[Point2D]

    @field_validator("points")
    @classmethod
    def require_polygon(cls, value: list[Point2D]) -> list[Point2D]:
        if len(value) < 3:
            raise ValueError("room polygon requires at least three points")
        return value


class FurniturePlacement(BaseModel):
    category: str
    center: Point2D
    width_m: PositiveFloat
    depth_m: PositiveFloat
    rotation_deg: float = 0.0


class CameraWaypoint(BaseModel):
    position: Point2D
    look_at: Point2D | None = None
    pause_seconds: float = Field(default=0.0, ge=0.0)


class FloorPlanModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    coordinate_system: CoordinateSystem = CoordinateSystem.METRES
    pixels_per_metre: PositiveFloat | None = None
    walls: list[WallSegment] = Field(default_factory=list)
    doors: list[DoorOpening] = Field(default_factory=list)
    windows: list[WindowOpening] = Field(default_factory=list)
    rooms: list[RoomPolygon] = Field(default_factory=list)
    furniture: list[FurniturePlacement] = Field(default_factory=list)
    entrance: Point2D | None = None
    camera_waypoints: list[CameraWaypoint] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: Path) -> "FloorPlanModel":
        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))
