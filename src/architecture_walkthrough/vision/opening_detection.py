from __future__ import annotations

from dataclasses import dataclass

from architecture_walkthrough.ai.floorplan_vision import FloorPlanVisionHints
from architecture_walkthrough.geometry.constraints import point_to_wall_distance, project_point_to_wall
from architecture_walkthrough.geometry.models import DoorOpening, Point2D, WallSegment, WindowOpening


@dataclass(frozen=True)
class OpeningAttachmentResult:
    doors: list[DoorOpening]
    windows: list[WindowOpening]
    rejected: list[str]


def _hint_point(point_x: float, point_y: float, image_width_px: int, image_height_px: int, pixels_per_metre: float) -> Point2D:
    return Point2D(
        x=(point_x * image_width_px) / pixels_per_metre,
        y=((1.0 - point_y) * image_height_px) / pixels_per_metre,
    )


def _nearest_wall(walls: list[WallSegment], point: Point2D, tolerance_m: float) -> tuple[WallSegment, float] | None:
    best: tuple[WallSegment, float, float] | None = None
    for wall in walls:
        distance, offset = point_to_wall_distance(wall, point)
        if best is None or distance < best[1]:
            best = (wall, distance, offset)
    if best is None or best[1] > tolerance_m:
        return None
    return best[0], best[2]


def attach_openings_to_walls(
    walls: list[WallSegment],
    doors: list[DoorOpening],
    windows: list[WindowOpening],
    tolerance_m: float = 0.35,
) -> OpeningAttachmentResult:
    attached_doors: list[DoorOpening] = []
    attached_windows: list[WindowOpening] = []
    rejected: list[str] = []
    for index, door in enumerate(doors):
        match = _nearest_wall(walls, door.center, tolerance_m)
        if match is None:
            rejected.append(f"door {index} is not close enough to any wall")
            continue
        wall, offset = match
        attached_doors.append(
            door.model_copy(
                update={
                    "id": door.id or f"door_{len(attached_doors):03d}",
                    "wall_id": wall.id,
                    "offset_m": offset,
                    "start_offset_m": offset - door.width_m / 2,
                    "end_offset_m": offset + door.width_m / 2,
                    "center": project_point_to_wall(wall, door.center),
                    "evidence_source": door.evidence_source,
                    "confidence": min(door.confidence, wall.confidence),
                }
            )
        )
    for index, window in enumerate(windows):
        match = _nearest_wall(walls, window.center, tolerance_m)
        if match is None:
            rejected.append(f"window {index} is not close enough to any wall")
            continue
        wall, offset = match
        attached_windows.append(
            window.model_copy(
                update={
                    "id": window.id or f"window_{len(attached_windows):03d}",
                    "wall_id": wall.id,
                    "offset_m": offset,
                    "start_offset_m": offset - window.width_m / 2,
                    "end_offset_m": offset + window.width_m / 2,
                    "center": project_point_to_wall(wall, window.center),
                    "evidence_source": window.evidence_source,
                    "confidence": min(window.confidence, wall.confidence),
                }
            )
        )
    return OpeningAttachmentResult(doors=attached_doors, windows=attached_windows, rejected=rejected)


def openings_from_semantic_hints(
    hints: FloorPlanVisionHints | None,
    image_width_px: int,
    image_height_px: int,
    pixels_per_metre: float,
    door_width_m: float,
    window_width_m: float,
    min_confidence: float,
) -> tuple[list[DoorOpening], list[WindowOpening], list[str]]:
    if hints is None:
        return [], [], []
    doors: list[DoorOpening] = []
    windows: list[WindowOpening] = []
    rejected: list[str] = []
    for index, hint in enumerate(hints.openings):
        if hint.confidence < min_confidence:
            rejected.append(f"opening hint {index} rejected below confidence threshold")
            continue
        center = _hint_point(hint.center.x, hint.center.y, image_width_px, image_height_px, pixels_per_metre)
        if hint.kind.lower() == "door":
            doors.append(
                DoorOpening(
                    id=f"door_hint_{index:03d}",
                    center=center,
                    width_m=door_width_m,
                    evidence_source="gemini_semantic_hint_projected",
                    confidence=hint.confidence,
                )
            )
        elif hint.kind.lower() == "window":
            windows.append(
                WindowOpening(
                    id=f"window_hint_{index:03d}",
                    center=center,
                    width_m=window_width_m,
                    evidence_source="gemini_semantic_hint_projected",
                    confidence=hint.confidence,
                )
            )
    return doors, windows, rejected


def detect_openings_placeholder() -> tuple[list[DoorOpening], list[WindowOpening]]:
    return [], []
