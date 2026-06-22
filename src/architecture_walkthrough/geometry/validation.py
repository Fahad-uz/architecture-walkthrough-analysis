from __future__ import annotations

from shapely.geometry import Polygon

from .models import FloorPlanModel


def validate_floorplan(model: FloorPlanModel) -> list[str]:
    issues: list[str] = []
    if not model.walls:
        issues.append("floorplan contains no walls")
    for idx, room in enumerate(model.rooms):
        polygon = Polygon([(point.x, point.y) for point in room.points])
        if not polygon.is_valid:
            issues.append(f"room {idx} polygon is invalid")
        if polygon.area <= 0:
            issues.append(f"room {idx} polygon has no area")
    return issues
