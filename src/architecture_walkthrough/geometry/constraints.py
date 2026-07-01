from __future__ import annotations

import math

from architecture_walkthrough.geometry.models import Point2D, WallSegment


def wall_length(wall: WallSegment) -> float:
    return wall.start.distance_to(wall.end)


def is_horizontal(wall: WallSegment) -> bool:
    return abs(wall.end.x - wall.start.x) >= abs(wall.end.y - wall.start.y)


def snap_wall_axis(wall: WallSegment, angle_tolerance_deg: float = 7.0) -> WallSegment:
    dx = wall.end.x - wall.start.x
    dy = wall.end.y - wall.start.y
    angle = abs(math.degrees(math.atan2(dy, dx))) % 180
    if min(angle, abs(180 - angle)) <= angle_tolerance_deg:
        y = (wall.start.y + wall.end.y) / 2
        return wall.model_copy(update={"start": Point2D(x=wall.start.x, y=y), "end": Point2D(x=wall.end.x, y=y)})
    if abs(angle - 90) <= angle_tolerance_deg:
        x = (wall.start.x + wall.end.x) / 2
        return wall.model_copy(update={"start": Point2D(x=x, y=wall.start.y), "end": Point2D(x=x, y=wall.end.y)})
    return wall


def sorted_wall(wall: WallSegment) -> WallSegment:
    if is_horizontal(wall):
        if wall.start.x <= wall.end.x:
            return wall
    elif wall.start.y <= wall.end.y:
        return wall
    return wall.model_copy(update={"start": wall.end, "end": wall.start})


def point_to_wall_distance(wall: WallSegment, point: Point2D) -> tuple[float, float]:
    length = wall_length(wall)
    if length <= 0:
        return point.distance_to(wall.start), 0.0
    ux = (wall.end.x - wall.start.x) / length
    uy = (wall.end.y - wall.start.y) / length
    offset = (point.x - wall.start.x) * ux + (point.y - wall.start.y) * uy
    clamped = max(0.0, min(length, offset))
    px = wall.start.x + ux * clamped
    py = wall.start.y + uy * clamped
    return Point2D(x=px, y=py).distance_to(point), clamped


def project_point_to_wall(wall: WallSegment, point: Point2D) -> Point2D:
    _, offset = point_to_wall_distance(wall, point)
    length = wall_length(wall)
    if length <= 0:
        return wall.start
    ux = (wall.end.x - wall.start.x) / length
    uy = (wall.end.y - wall.start.y) / length
    return Point2D(x=wall.start.x + ux * offset, y=wall.start.y + uy * offset)
