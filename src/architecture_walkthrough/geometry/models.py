from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Literal

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
    id: str | None = None
    start: Point2D
    end: Point2D
    thickness_m: PositiveFloat = 0.12
    height_m: PositiveFloat = 3.0
    external: bool = False
    wall_type: str = "internal"
    material_preset: str | None = None
    room_side: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence_source: str = "unknown"
    source_band_id: str | None = None


class DoorOpening(BaseModel):
    id: str | None = None
    center: Point2D
    width_m: PositiveFloat = 0.90
    height_m: PositiveFloat = 2.10
    wall_id: str | None = None
    offset_m: float | None = None
    opening_type: str = "single_leaf"
    asset_preset: str | None = None
    opening_direction: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence_source: str = "unknown"


class WindowOpening(BaseModel):
    id: str | None = None
    center: Point2D
    width_m: PositiveFloat = 1.20
    height_m: PositiveFloat = 1.20
    sill_height_m: PositiveFloat = 0.90
    wall_id: str | None = None
    offset_m: float | None = None
    opening_type: str = "fixed"
    asset_preset: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence_source: str = "unknown"


class RoomPolygon(BaseModel):
    id: str | None = None
    name: str | None = None
    points: list[Point2D]
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence_source: str = "unknown"
    dimension_m: tuple[float, float] | None = None

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


class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class PlanROI(BaseModel):
    rect: BoundingBox
    confidence: float = Field(ge=0.0, le=1.0)
    source: str = "auto"
    padding_px: int = 0
    full_image_width_px: int | None = None
    full_image_height_px: int | None = None


class ScaleConstraintRecord(BaseModel):
    id: str
    source: str
    label: str | None = None
    measured_px: tuple[float, float]
    expected_m: tuple[float, float]
    pixels_per_metre: float
    residual: float = 0.0
    weight: float = 1.0
    accepted: bool = True
    reason: str | None = None


class ValidationIssue(BaseModel):
    code: str
    severity: Literal["info", "warning", "error", "severe"] = "warning"
    message: str
    element_id: str | None = None


class ReconstructionMetadata(BaseModel):
    schema_version: str = "2.0"
    quality_state: Literal["high", "review_required", "failed"] = "review_required"
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    stages: list[dict[str, Any]] = Field(default_factory=list)
    ai_geometry_originated: bool = False
    semantic_hints_used: list[str] = Field(default_factory=list)
    semantic_hints_rejected: list[str] = Field(default_factory=list)


class ArchitecturalElement(BaseModel):
    id: str
    kind: str
    polygon: list[Point2D] = Field(default_factory=list)
    center: Point2D | None = None
    width_m: PositiveFloat | None = None
    depth_m: PositiveFloat | None = None
    rotation_deg: float = 0.0
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence_source: str = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)


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
    schema_version: str = "2.0"
    pixels_per_metre: PositiveFloat | None = None
    plan_roi: PlanROI | None = None
    walls: list[WallSegment] = Field(default_factory=list)
    doors: list[DoorOpening] = Field(default_factory=list)
    windows: list[WindowOpening] = Field(default_factory=list)
    rooms: list[RoomPolygon] = Field(default_factory=list)
    balconies: list[BalconyPolygon] = Field(default_factory=list)
    slabs: list[SlabPolygon] = Field(default_factory=list)
    special_elements: list[ArchitecturalElement] = Field(default_factory=list)
    furniture: list[FurniturePlacement] = Field(default_factory=list)
    asset_placements: list[AssetPlacement] = Field(default_factory=list)
    material_assignments: list[MaterialAssignment] = Field(default_factory=list)
    ceiling: CeilingSettings = Field(default_factory=CeilingSettings)
    style: SceneStyleSettings = Field(default_factory=SceneStyleSettings)
    entrance: Point2D | None = None
    camera_waypoints: list[CameraWaypoint] = Field(default_factory=list)
    scale_constraints: list[ScaleConstraintRecord] = Field(default_factory=list)
    validation_issues: list[ValidationIssue] = Field(default_factory=list)
    reconstruction: ReconstructionMetadata = Field(default_factory=ReconstructionMetadata)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: Path) -> "FloorPlanModel":
        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))
