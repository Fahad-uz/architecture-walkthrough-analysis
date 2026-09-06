from __future__ import annotations

from shapely.geometry import LineString, Point

from architecture_walkthrough.geometry.models import Point2D, WallSegment


def wall_geometries(walls: list[WallSegment]) -> list[LineString]:
    return [LineString([(wall.start.x, wall.start.y), (wall.end.x, wall.end.y)]) for wall in walls]


def point_has_clearance(point: Point2D, walls: list[WallSegment], radius_m: float) -> bool:
    p = Point(point.x, point.y)
    return all(p.distance(wall) >= radius_m for wall in wall_geometries(walls))


def path_collides(path: list[Point2D], walls: list[WallSegment], radius_m: float = 0.25) -> bool:
    if len(path) < 2:
        return False
    buffered_walls = [wall.buffer(radius_m) for wall in wall_geometries(walls)]
    for a, b in zip(path, path[1:]):
        segment = LineString([(a.x, a.y), (b.x, b.y)])
        if any(segment.intersects(obstacle) for obstacle in buffered_walls):
            return True
    return False
