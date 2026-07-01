from __future__ import annotations

from shapely.geometry import Polygon

from .models import FloorPlanModel, ValidationIssue


def validate_floorplan(model: FloorPlanModel) -> list[str]:
    return [issue.message for issue in validate_reconstruction(model)]


def validate_reconstruction(model: FloorPlanModel) -> list[ValidationIssue]:
    if not model.walls:
        return [
            ValidationIssue(
                code="no_structural_walls",
                severity="severe",
                message="floorplan contains no structural wall geometry",
            )
        ]
    structured: list[ValidationIssue] = []
    for idx, room in enumerate(model.rooms):
        polygon = Polygon([(point.x, point.y) for point in room.points])
        if not polygon.is_valid:
            structured.append(ValidationIssue(code="invalid_room_polygon", severity="error", message=f"room {idx} polygon is invalid", element_id=room.id))
        if polygon.area <= 0:
            structured.append(ValidationIssue(code="empty_room_polygon", severity="error", message=f"room {idx} polygon has no area", element_id=room.id))
    wall_ids = {wall.id for wall in model.walls if wall.id}
    for door in model.doors:
        if door.wall_id not in wall_ids:
            structured.append(ValidationIssue(code="door_without_valid_wall", severity="error", message="door does not reference a valid wall", element_id=door.id))
    for window in model.windows:
        if window.wall_id not in wall_ids:
            structured.append(ValidationIssue(code="window_without_valid_wall", severity="error", message="window does not reference a valid wall", element_id=window.id))
    if not model.rooms:
        structured.append(ValidationIssue(code="no_closed_rooms", severity="warning", message="no closed room polygons were extracted"))
    if model.pixels_per_metre is None:
        structured.append(ValidationIssue(code="missing_scale", severity="severe", message="pixels-per-metre scale was not established"))
    return structured


def score_quality(model: FloorPlanModel) -> tuple[float, str]:
    issues = validate_reconstruction(model)
    severe = sum(1 for issue in issues if issue.severity == "severe")
    errors = sum(1 for issue in issues if issue.severity == "error")
    warnings = sum(1 for issue in issues if issue.severity == "warning")
    wall_conf = sum(wall.confidence for wall in model.walls) / len(model.walls) if model.walls else 0.0
    room_score = min(1.0, len(model.rooms) / 4.0)
    opening_score = 1.0
    if model.doors or model.windows:
        valid = sum(1 for door in model.doors if door.wall_id) + sum(1 for window in model.windows if window.wall_id)
        opening_score = valid / max(1, len(model.doors) + len(model.windows))
    scale_conf = float(model.metadata.get("scale_confidence", 0.0 if model.pixels_per_metre is None else 0.5))
    score = (
        wall_conf * 0.35
        + room_score * 0.20
        + opening_score * 0.15
        + scale_conf * 0.20
        + max(0.0, 1.0 - (warnings * 0.08 + errors * 0.18 + severe * 0.35)) * 0.10
    )
    score = max(0.0, min(1.0, score))
    if severe or score < 0.25:
        return score, "failed"
    if errors or warnings or score < 0.70:
        return score, "review_required"
    return score, "high"
