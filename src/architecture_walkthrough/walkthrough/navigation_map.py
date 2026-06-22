from __future__ import annotations

from dataclasses import dataclass

from architecture_walkthrough.geometry.models import FloorPlanModel, Point2D


@dataclass(frozen=True)
class NavigationMap:
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    grid_size_m: float


def build_navigation_map(model: FloorPlanModel, grid_size_m: float = 0.25) -> NavigationMap:
    points: list[Point2D] = []
    for wall in model.walls:
        points.extend([wall.start, wall.end])
    if not points:
        return NavigationMap(-2, -2, 2, 2, grid_size_m)
    return NavigationMap(
        min(point.x for point in points) - 1,
        min(point.y for point in points) - 1,
        max(point.x for point in points) + 1,
        max(point.y for point in points) + 1,
        grid_size_m,
    )
