from __future__ import annotations

from architecture_walkthrough.geometry.models import CameraWaypoint, Point2D


def simplify_path(path: list[Point2D]) -> list[Point2D]:
    if len(path) <= 2:
        return path
    simplified = [path[0]]
    for previous, current, nxt in zip(path, path[1:], path[2:]):
        dx1, dy1 = current.x - previous.x, current.y - previous.y
        dx2, dy2 = nxt.x - current.x, nxt.y - current.y
        if abs(dx1 * dy2 - dy1 * dx2) > 1e-6:
            simplified.append(current)
    simplified.append(path[-1])
    return simplified


def waypoints_from_points(path: list[Point2D]) -> list[CameraWaypoint]:
    return [CameraWaypoint(position=point) for point in simplify_path(path)]
