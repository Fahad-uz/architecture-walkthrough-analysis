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
    wall_type: str = "internal"
    material_preset: str | None = None
    room_side: str | None = None


class DoorOpening(BaseModel):
    center: Point2D
    width_m: PositiveFloat = 0.90
    height_m: PositiveFloat = 2.10
    wall_id: str | None = None
    offset_m: float | None = None
    opening_type: str = "single_leaf"
    asset_preset: str | None = None


class WindowOpening(BaseModel):
    center: Point2D
    width_m: PositiveFloat = 1.20
    height_m: PositiveFloat = 1.20
    sill_height_m: PositiveFloat = 0.90
    wall_id: str | None = None
    offset_m: float | None = None
    opening_type: str = "fixed"
    asset_preset: str | None = None


class RoomPolygon(BaseModel):
    name: str | None = None
    points: list[Point2D]

    @field_validator("points")
    @classmethod
    def require_polygon(cls, value: list[Point2D]) -> list[Point2D]:
        if len(value) < 3:
            raise ValueError("room polygon requires at least three points")
        return value


class BalconyPolygon(RoomPolygon):
    pass


class SlabPolygon(RoomPolygon):
    thickness_m: PositiveFloat = 0.10
    material_preset: str | None = None


class MaterialAssignment(BaseModel):
    target: str
    preset: str


class AssetPlacement(BaseModel):
    category: str
    center: Point2D
    width_m: PositiveFloat
    depth_m: PositiveFloat
    height_m: PositiveFloat | None = None
    rotation_deg: float = 0.0
    asset_preset: str | None = None


class CeilingSettings(BaseModel):
    enabled: bool = False
    height_m: PositiveFloat = 3.0
    thickness_m: PositiveFloat = 0.08
    material_preset: str = "painted_wall"


class SceneStyleSettings(BaseModel):
    preset: str = "neutral"
    wall_material: str = "painted_wall"
    floor_material: str = "marble"
    door_material: str = "wood"
    window_frame_material: str = "metal"
    furniture_family: str = "procedural"


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
    balconies: list[BalconyPolygon] = Field(default_factory=list)
    slabs: list[SlabPolygon] = Field(default_factory=list)
    furniture: list[FurniturePlacement] = Field(default_factory=list)
    asset_placements: list[AssetPlacement] = Field(default_factory=list)
    material_assignments: list[MaterialAssignment] = Field(default_factory=list)
    ceiling: CeilingSettings = Field(default_factory=CeilingSettings)
    style: SceneStyleSettings = Field(default_factory=SceneStyleSettings)
    entrance: Point2D | None = None
    camera_waypoints: list[CameraWaypoint] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: Path) -> "FloorPlanModel":
        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))
