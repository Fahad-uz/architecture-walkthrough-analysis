from __future__ import annotations

from architecture_walkthrough.geometry.models import DoorOpening, WallSegment, WindowOpening
from architecture_walkthrough.scene.wall_builder import WallOpening, opening_from_door, opening_from_window


def _point_wall_distance(wall: WallSegment, point) -> float:
    length = wall.start.distance_to(wall.end)
    if length <= 0:
        return float("inf")
    dx = wall.end.x - wall.start.x
    dy = wall.end.y - wall.start.y
    t = max(0.0, min(1.0, ((point.x - wall.start.x) * dx + (point.y - wall.start.y) * dy) / (length * length)))
    nearest_x = wall.start.x + t * dx
    nearest_y = wall.start.y + t * dy
    return ((point.x - nearest_x) ** 2 + (point.y - nearest_y) ** 2) ** 0.5


def nearest_wall_index(walls: list[WallSegment], point) -> int | None:
    if not walls:
        return None
    distances = [(_point_wall_distance(wall, point), index) for index, wall in enumerate(walls)]
    distance, index = min(distances)
    return index if distance <= max(walls[index].thickness_m * 3.0, 0.35) else None


def openings_for_wall(
    wall: WallSegment,
    doors: list[DoorOpening],
    windows: list[WindowOpening],
    wall_index: int,
    all_walls: list[WallSegment] | None = None,
) -> list[WallOpening]:
    wall_id = str(wall_index)
    stable_wall_id = wall.id
    wall_openings: list[WallOpening] = []
    for door in doors:
        selected = nearest_wall_index(all_walls, door.center) if door.wall_id is None and all_walls else None
        if (door.wall_id is not None and door.wall_id in {wall_id, stable_wall_id}) or selected == wall_index:
            wall_openings.append(opening_from_door(wall, door))
    for window in windows:
        selected = nearest_wall_index(all_walls, window.center) if window.wall_id is None and all_walls else None
        if (window.wall_id is not None and window.wall_id in {wall_id, stable_wall_id}) or selected == wall_index:
            wall_openings.append(opening_from_window(wall, window))
    return wall_openings
