from __future__ import annotations

from .models import Point2D, WallSegment


def snap_value(value: float, grid: float) -> float:
    if grid <= 0:
        raise ValueError("grid must be positive")
    return round(value / grid) * grid


def snap_point(point: Point2D, grid: float = 0.05) -> Point2D:
    return Point2D(x=snap_value(point.x, grid), y=snap_value(point.y, grid))


def normalize_axis_aligned(wall: WallSegment, tolerance_deg: float = 5.0) -> WallSegment:
    dx = wall.end.x - wall.start.x
    dy = wall.end.y - wall.start.y
    if abs(dx) < abs(dy) * tolerance_deg / 45.0:
        avg_x = (wall.start.x + wall.end.x) / 2
        return wall.model_copy(update={"start": Point2D(x=avg_x, y=wall.start.y), "end": Point2D(x=avg_x, y=wall.end.y)})
    if abs(dy) < abs(dx) * tolerance_deg / 45.0:
        avg_y = (wall.start.y + wall.end.y) / 2
        return wall.model_copy(update={"start": Point2D(x=wall.start.x, y=avg_y), "end": Point2D(x=wall.end.x, y=avg_y)})
    return wall


def cleanup_walls(walls: list[WallSegment], min_length_m: float = 0.2, snap_grid_m: float = 0.05) -> list[WallSegment]:
    cleaned: list[WallSegment] = []
    seen: set[tuple[float, float, float, float]] = set()
    for wall in walls:
        normalized = normalize_axis_aligned(wall)
        start = snap_point(normalized.start, snap_grid_m)
        end = snap_point(normalized.end, snap_grid_m)
        if start.distance_to(end) < min_length_m:
            continue
        key = (start.x, start.y, end.x, end.y)
        reverse_key = (end.x, end.y, start.x, start.y)
        if key in seen or reverse_key in seen:
            continue
        seen.add(key)
        cleaned.append(normalized.model_copy(update={"start": start, "end": end}))
    return cleaned
