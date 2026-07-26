from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import LineString, Polygon

from architecture_walkthrough.geometry.wall_graph import enumerate_faces

from .models import DoorOpening, FloorPlanModel, ValidationIssue, WindowOpening

# Validation is a gate, not a cosmetic score. Every component is scored from
# actual evidence and consistency, and MISSING evidence lowers confidence --
# zero detected doors in a multi-room plan reads as "unknown/low", never as a
# perfect score.

UNKNOWN_COMPONENT_SCORE = 0.4  # what "we could not verify this" is worth


@dataclass(frozen=True)
class SourceEvidence:
    """Pixel evidence from the analyzed image for cross-checking geometry."""

    dark_mask: np.ndarray
    pixels_per_metre: float
    image_height_px: int
    # Long straight structural runs (wall-band union). Preferred recall target:
    # raw ink also contains fixtures, stairs, and arcs that are not walls.
    band_mask: np.ndarray | None = None


def load_source_evidence(job_dir: Path, model: FloorPlanModel) -> SourceEvidence | None:
    mask_path = job_dir / "debug" / "05_dark_structural_stroke.png"
    if not mask_path.exists() or not model.pixels_per_metre:
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    band_mask = None
    horizontal = cv2.imread(str(job_dir / "debug" / "08_horizontal_wall_band.png"), cv2.IMREAD_GRAYSCALE)
    vertical = cv2.imread(str(job_dir / "debug" / "09_vertical_wall_band.png"), cv2.IMREAD_GRAYSCALE)
    if horizontal is not None and vertical is not None and horizontal.shape == vertical.shape:
        band_mask = cv2.bitwise_or(horizontal, vertical)
    return SourceEvidence(
        dark_mask=mask,
        pixels_per_metre=model.pixels_per_metre,
        image_height_px=mask.shape[0],
        band_mask=band_mask,
    )


@dataclass
class QualityReport:
    score: float
    state: str
    components: dict[str, float] = field(default_factory=dict)
    issues: list[ValidationIssue] = field(default_factory=list)


def _wall_px(model: FloorPlanModel, evidence: SourceEvidence) -> np.ndarray:
    canvas = np.zeros_like(evidence.dark_mask)
    ppm = evidence.pixels_per_metre
    for wall in model.walls:
        p0 = (int(wall.start.x * ppm), int(evidence.image_height_px - wall.start.y * ppm))
        p1 = (int(wall.end.x * ppm), int(evidence.image_height_px - wall.end.y * ppm))
        thickness = max(1, int(wall.thickness_m * ppm))
        cv2.line(canvas, p0, p1, 255, thickness)
    return canvas


def wall_mask_overlap_score(model: FloorPlanModel, evidence: SourceEvidence) -> float:
    """How well do the vectorized walls agree with structural ink in the image?

    Precision is measured against (slightly dilated) ink; recall against the
    wall-band evidence when available — raw ink also contains fixtures and
    furniture linework that walls are not expected to explain.
    """
    if not model.walls:
        return 0.0
    rendered = _wall_px(model, evidence)
    rendered_on = int((rendered > 0).sum())
    if rendered_on == 0:
        return 0.0
    ink = cv2.dilate(evidence.dark_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    precision = int(((rendered > 0) & (ink > 0)).sum()) / rendered_on
    recall_target = evidence.band_mask if evidence.band_mask is not None else evidence.dark_mask
    target_on = int((recall_target > 0).sum())
    if target_on == 0:
        return 0.0
    recall = int(((rendered > 0) & (recall_target > 0)).sum()) / target_on
    return max(0.0, min(1.0, 0.65 * precision + 0.35 * min(1.0, recall / 0.6)))


def junction_support_score(model: FloorPlanModel, evidence: SourceEvidence) -> float:
    """Are the graph's junctions backed by ink in the source image?"""
    endpoints: dict[tuple[int, int], int] = {}
    ppm = evidence.pixels_per_metre
    for wall in model.walls:
        for point in (wall.start, wall.end):
            key = (round(point.x * 20), round(point.y * 20))  # 5 cm buckets
            endpoints[key] = endpoints.get(key, 0) + 1
    junctions = [key for key, count in endpoints.items() if count >= 2]
    if not junctions:
        return UNKNOWN_COMPONENT_SCORE
    height, width = evidence.dark_mask.shape[:2]
    supported = 0
    for key in junctions:
        x_px = int(key[0] / 20 * ppm)
        y_px = int(evidence.image_height_px - key[1] / 20 * ppm)
        x0, x1 = max(0, x_px - 5), min(width, x_px + 6)
        y0, y1 = max(0, y_px - 5), min(height, y_px + 6)
        if x1 > x0 and y1 > y0 and (evidence.dark_mask[y0:y1, x0:x1] > 0).any():
            supported += 1
    return supported / len(junctions)


def wall_crossings_without_junction(model: FloorPlanModel) -> list[ValidationIssue]:
    """Two walls crossing mid-span means the graph missed a junction split."""
    issues: list[ValidationIssue] = []
    lines = [
        (wall.id, LineString([(wall.start.x, wall.start.y), (wall.end.x, wall.end.y)]))
        for wall in model.walls
        if wall.start.distance_to(wall.end) > 0
    ]
    for index, (id_a, line_a) in enumerate(lines):
        for id_b, line_b in lines[index + 1 :]:
            if line_a.crosses(line_b):
                issues.append(
                    ValidationIssue(
                        code="wall_crossing_without_junction",
                        severity="warning",
                        message=f"walls {id_a} and {id_b} cross without a shared junction",
                        element_id=id_a,
                    )
                )
    return issues


def room_geometry_issues(model: FloorPlanModel) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    polygons: list[tuple[str | None, Polygon]] = []
    for idx, room in enumerate(model.rooms):
        polygon = Polygon([(point.x, point.y) for point in room.points])
        if not polygon.is_valid:
            issues.append(ValidationIssue(code="invalid_room_polygon", severity="error", message=f"room {idx} polygon is invalid", element_id=room.id))
            continue
        if polygon.area <= 0:
            issues.append(ValidationIssue(code="empty_room_polygon", severity="error", message=f"room {idx} polygon has no area", element_id=room.id))
            continue
        polygons.append((room.id, polygon))
    for index, (id_a, poly_a) in enumerate(polygons):
        for id_b, poly_b in polygons[index + 1 :]:
            overlap = poly_a.intersection(poly_b).area
            if overlap > 0.05 * min(poly_a.area, poly_b.area):
                issues.append(
                    ValidationIssue(
                        code="room_polygons_overlap",
                        severity="error",
                        message=f"rooms {id_a} and {id_b} overlap by {overlap:.2f} m²",
                        element_id=id_a,
                    )
                )
    balcony_polygons: list[tuple[str | None, Polygon]] = []
    for balcony in model.balconies:
        polygon = Polygon([(point.x, point.y) for point in balcony.points])
        if polygon.is_valid and polygon.area > 0:
            balcony_polygons.append((balcony.id, polygon))
    for room_id, room_polygon in polygons:
        for balcony_id, balcony_polygon in balcony_polygons:
            overlap = room_polygon.intersection(balcony_polygon).area
            tolerance = max(
                0.01,
                0.01 * min(room_polygon.area, balcony_polygon.area),
            )
            if overlap > tolerance:
                issues.append(
                    ValidationIssue(
                        code="room_balcony_overlap",
                        severity="warning",
                        message=(
                            f"room {room_id} and balcony {balcony_id} overlap "
                            f"by {overlap:.2f} m²"
                        ),
                        element_id=room_id,
                    )
                )
    return issues


def opening_interval_issues(model: FloorPlanModel) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    walls = {wall.id: wall for wall in model.walls if wall.id}
    by_wall: dict[str, list[tuple[float, float, str | None]]] = {}
    openings: list[DoorOpening | WindowOpening] = [*model.doors, *model.windows]
    for opening in openings:
        if opening.wall_id not in walls:
            issues.append(
                ValidationIssue(
                    code="opening_without_valid_wall",
                    severity="error",
                    message=f"opening {opening.id} does not reference a valid wall",
                    element_id=opening.id,
                )
            )
            continue
        wall = walls[opening.wall_id]
        interval = opening.interval(opening.width_m)
        if interval is None:
            continue
        start, end = interval
        length = wall.start.distance_to(wall.end)
        if start < -0.01 or end > length + 0.01:
            issues.append(
                ValidationIssue(
                    code="opening_outside_wall_span",
                    severity="error",
                    message=f"opening {opening.id} interval {start:.2f}-{end:.2f} m exceeds wall {wall.id} length {length:.2f} m",
                    element_id=opening.id,
                )
            )
        by_wall.setdefault(opening.wall_id, []).append((start, end, opening.id))
    for wall_id, intervals in by_wall.items():
        ordered = sorted(intervals)
        for (start_a, end_a, id_a), (start_b, _end_b, id_b) in zip(ordered, ordered[1:]):
            if start_b < end_a - 0.01:
                issues.append(
                    ValidationIssue(
                        code="openings_overlap",
                        severity="error",
                        message=f"openings {id_a} and {id_b} overlap on wall {wall_id}",
                        element_id=id_b,
                    )
                )
    return issues


def exterior_closure_score(model: FloorPlanModel) -> float:
    """Do the enumerated faces account for the building footprint?"""
    if not model.walls:
        return 0.0
    faces = enumerate_faces(model.walls, model.doors, model.windows).faces
    if not faces:
        return 0.0
    xs = [p.x for wall in model.walls for p in (wall.start, wall.end)]
    ys = [p.y for wall in model.walls for p in (wall.start, wall.end)]
    bbox_area = max((max(xs) - min(xs)) * (max(ys) - min(ys)), 1e-6)
    coverage = sum(face.polygon.area for face in faces) / bbox_area
    return max(0.0, min(1.0, coverage / 0.70))


def opening_presence_score(model: FloorPlanModel) -> float:
    """Zero openings in a multi-room plan is unknown/low - never perfect."""
    total = len(model.doors) + len(model.windows)
    if total == 0:
        return 0.2 if len(model.rooms) >= 2 else UNKNOWN_COMPONENT_SCORE
    valid_wall_ids = {wall.id for wall in model.walls if wall.id}
    openings: list[DoorOpening | WindowOpening] = [*model.doors, *model.windows]
    attached = sum(1 for opening in openings if opening.wall_id in valid_wall_ids)
    low_conf = sum(1 for opening in openings if opening.confidence < 0.45)
    attachment = attached / total
    certainty = 1.0 - 0.5 * (low_conf / total)
    return max(0.0, min(1.0, attachment * certainty))


def scale_confidence_score(model: FloorPlanModel) -> float:
    confidence = model.metadata.get("scale_confidence")
    if confidence is None:
        return 0.0 if model.pixels_per_metre is None else UNKNOWN_COMPONENT_SCORE
    return max(0.0, min(1.0, float(confidence)))


def validate_reconstruction(model: FloorPlanModel, evidence: SourceEvidence | None = None) -> list[ValidationIssue]:
    if not model.walls:
        return [
            ValidationIssue(
                code="no_structural_walls",
                severity="severe",
                message="floorplan contains no structural wall geometry",
            )
        ]
    issues: list[ValidationIssue] = []
    for wall in model.walls:
        if wall.start.distance_to(wall.end) <= 1e-6:
            issues.append(
                ValidationIssue(
                    code="zero_length_wall",
                    severity="error",
                    message=f"wall {wall.id or '<unnamed>'} has zero length and cannot be rendered",
                )
            )
    issues.extend(room_geometry_issues(model))
    issues.extend(opening_interval_issues(model))
    issues.extend(wall_crossings_without_junction(model))
    if not model.rooms:
        issues.append(ValidationIssue(code="no_closed_rooms", severity="warning", message="no closed room polygons were extracted"))
    if model.pixels_per_metre is None:
        issues.append(ValidationIssue(code="missing_scale", severity="severe", message="pixels-per-metre scale was not established"))
    scale_source = model.metadata.get("scale_source")
    if scale_source == "wall_thickness":
        issues.append(
            ValidationIssue(
                code="scale_from_assumed_wall_thickness",
                severity="warning",
                message="scale was calibrated from assumed wall thickness only; set a manual reference before trusting dimensions",
            )
        )
    if not model.doors and not model.windows and len(model.rooms) >= 2:
        issues.append(
            ValidationIssue(
                code="no_openings_detected",
                severity="warning",
                message="no doors or windows were detected in a multi-room plan; openings are unknown, not absent",
            )
        )
    if evidence is None:
        issues.append(
            ValidationIssue(
                code="no_source_evidence",
                severity="info",
                message="source image evidence unavailable; wall/junction support could not be verified",
            )
        )
    return issues


def evaluate_quality(model: FloorPlanModel, evidence: SourceEvidence | None = None) -> QualityReport:
    issues = list(model.validation_issues) or validate_reconstruction(model, evidence)
    severe = sum(1 for issue in issues if issue.severity == "severe")
    errors = sum(1 for issue in issues if issue.severity == "error")
    warnings = sum(1 for issue in issues if issue.severity == "warning")

    components: dict[str, float] = {}
    if evidence is not None and model.walls:
        components["wall_mask_overlap"] = wall_mask_overlap_score(model, evidence)
        components["junction_support"] = junction_support_score(model, evidence)
    else:
        components["wall_mask_overlap"] = UNKNOWN_COMPONENT_SCORE
        components["junction_support"] = UNKNOWN_COMPONENT_SCORE
    components["rooms"] = 0.0 if not model.rooms else max(0.2, 1.0 - 0.3 * sum(1 for i in issues if i.code in {"invalid_room_polygon", "room_polygons_overlap", "empty_room_polygon"}))
    components["openings"] = opening_presence_score(model)
    components["scale"] = scale_confidence_score(model)
    components["exterior_closure"] = exterior_closure_score(model)

    score = (
        components["wall_mask_overlap"] * 0.20
        + components["junction_support"] * 0.10
        + components["rooms"] * 0.20
        + components["openings"] * 0.20
        + components["scale"] * 0.20
        + components["exterior_closure"] * 0.10
    )
    # Warnings are review prompts, not defects; cap their combined penalty so
    # a thorough sanity check can't sink an otherwise sound reconstruction.
    score = max(0.0, score - errors * 0.08 - min(0.15, warnings * 0.02) - severe * 0.40)
    score = max(0.0, min(1.0, score))

    if severe or score < 0.35:
        state = "failed"
    elif errors or score < 0.70:
        state = "review_required"
    else:
        state = "review_required" if warnings > 3 else "high"
    return QualityReport(score=score, state=state, components=components, issues=issues)


def score_quality(model: FloorPlanModel, evidence: SourceEvidence | None = None) -> tuple[float, str]:
    report = evaluate_quality(model, evidence)
    return report.score, report.state


def validate_floorplan(model: FloorPlanModel) -> list[str]:
    return [issue.message for issue in validate_reconstruction(model)]
