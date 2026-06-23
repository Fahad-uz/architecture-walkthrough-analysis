from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import trimesh

from architecture_walkthrough.geometry.models import DoorOpening, Point2D, WallSegment, WindowOpening


@dataclass(frozen=True)
class WallOpening:
    offset_m: float
    width_m: float
    bottom_m: float
    height_m: float
    kind: str

    @property
    def start_m(self) -> float:
        return self.offset_m - self.width_m / 2

    @property
    def end_m(self) -> float:
        return self.offset_m + self.width_m / 2

    @property
    def top_m(self) -> float:
        return self.bottom_m + self.height_m


@dataclass(frozen=True)
class WallSection:
    start_m: float
    end_m: float
    bottom_m: float
    top_m: float


def wall_length(wall: WallSegment) -> float:
    return wall.start.distance_to(wall.end)


def point_offset_on_wall(wall: WallSegment, point: Point2D) -> float:
    length = wall_length(wall)
    if length <= 0:
        return 0.0
    dx = (wall.end.x - wall.start.x) / length
    dy = (wall.end.y - wall.start.y) / length
    return (point.x - wall.start.x) * dx + (point.y - wall.start.y) * dy


def opening_from_door(wall: WallSegment, door: DoorOpening) -> WallOpening:
    return WallOpening(
        offset_m=door.offset_m if door.offset_m is not None else point_offset_on_wall(wall, door.center),
        width_m=door.width_m,
        bottom_m=0.0,
        height_m=door.height_m,
        kind="door",
    )


def opening_from_window(wall: WallSegment, window: WindowOpening) -> WallOpening:
    return WallOpening(
        offset_m=window.offset_m if window.offset_m is not None else point_offset_on_wall(wall, window.center),
        width_m=window.width_m,
        bottom_m=window.sill_height_m,
        height_m=window.height_m,
        kind="window",
    )


def validate_wall_openings(wall: WallSegment, openings: list[WallOpening]) -> list[str]:
    issues: list[str] = []
    length = wall_length(wall)
    ordered = sorted(openings, key=lambda item: item.start_m)
    previous_end = -math.inf
    for index, opening in enumerate(ordered):
        if opening.start_m < 0 or opening.end_m > length:
            issues.append(f"opening {index} extends outside wall length")
        if opening.top_m > wall.height_m:
            issues.append(f"opening {index} exceeds wall height")
        if opening.start_m < previous_end:
            issues.append(f"opening {index} overlaps previous opening")
        previous_end = max(previous_end, opening.end_m)
    return issues


def split_wall_sections(wall: WallSegment, openings: list[WallOpening]) -> list[WallSection]:
    issues = validate_wall_openings(wall, openings)
    if issues:
        raise ValueError("; ".join(issues))
    length = wall_length(wall)
    sections: list[WallSection] = []
    spans = sorted(openings, key=lambda item: item.start_m)
    cursor = 0.0
    for opening in spans:
        if opening.start_m > cursor:
            sections.append(WallSection(cursor, opening.start_m, 0.0, wall.height_m))
        if opening.bottom_m > 0:
            sections.append(WallSection(opening.start_m, opening.end_m, 0.0, opening.bottom_m))
        if opening.top_m < wall.height_m:
            sections.append(WallSection(opening.start_m, opening.end_m, opening.top_m, wall.height_m))
        cursor = opening.end_m
    if cursor < length:
        sections.append(WallSection(cursor, length, 0.0, wall.height_m))
    return [section for section in sections if section.end_m > section.start_m and section.top_m > section.bottom_m]


def wall_section_mesh(wall: WallSegment, section: WallSection, color: tuple[int, int, int, int]) -> trimesh.Trimesh:
    length = section.end_m - section.start_m
    height = section.top_m - section.bottom_m
    center_along = (section.start_m + section.end_m) / 2
    angle = math.atan2(wall.end.y - wall.start.y, wall.end.x - wall.start.x)
    center_x = wall.start.x + center_along * math.cos(angle)
    center_y = wall.start.y + center_along * math.sin(angle)
    transform = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
    transform[:3, 3] = [center_x, center_y, section.bottom_m + height / 2]
    mesh = trimesh.creation.box(extents=[length, wall.thickness_m, height], transform=transform)
    mesh.visual.vertex_colors = np.tile(np.array(color, dtype=np.uint8), (len(mesh.vertices), 1))
    return mesh


def split_wall_meshes(
    wall: WallSegment,
    openings: list[WallOpening],
    color: tuple[int, int, int, int],
) -> list[trimesh.Trimesh]:
    return [wall_section_mesh(wall, section, color) for section in split_wall_sections(wall, openings)]
