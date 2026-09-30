"""Conservative, reviewable floor boundaries at wide open entrances.

These proposals affect floor polygons only. They are never wall segments or
confirmed door openings, and callers must retain the returned review warning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

from shapely import unary_union
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize

from architecture_walkthrough.geometry.models import (
    DoorOpening,
    RoomPolygon,
    ValidationIssue,
    WallSegment,
    WindowOpening,
)
from architecture_walkthrough.geometry.room_extraction import _label_point, _to_room_polygon
from architecture_walkthrough.geometry.wall_graph import (
    DEFAULT_JUNCTION_SNAP_M,
    GRAPH_NODING_GRID_M,
    _face_id_for,
    collinear_gaps,
    enumerate_faces,
    snap_endpoints_to_walls,
)
from architecture_walkthrough.vision.ocr import OCRText, parse_dimension_pair


@dataclass(frozen=True)
class OpenFloorRecoveryResult:
    rooms: list[RoomPolygon]
    issues: list[ValidationIssue] = field(default_factory=list)
    candidates: list[dict[str, object]] = field(default_factory=list)


def _confident_room_name(label: OCRText) -> bool:
    if (
        label.semantic_type != "room_label"
        or label.confidence < 0.75
        or len(label.polygon) < 3
        or not label.normalized_text.strip()
    ):
        return False
    try:
        return parse_dimension_pair(label.normalized_text) is None
    except ValueError:
        return False


def recover_open_labeled_floors(
    walls: list[WallSegment],
    rooms: list[RoomPolygon],
    labels: list[OCRText],
    pixels_per_metre: float,
    image_height_px: int,
    *,
    doors: list[DoorOpening] | None = None,
    windows: list[WindowOpening] | None = None,
    max_gap_m: float = 5.0,
    junction_snap_m: float = DEFAULT_JUNCTION_SNAP_M,
) -> OpenFloorRecoveryResult:
    """Recover a uniquely bounded labeled floor through one broad virtual gap.

    Only measured, nearly collinear wall endpoints are candidates. A bridge
    must be wider than the existing doorway policy, at most five metres and
    at most half the plan's largest extent. It must independently close a
    previously missing confident room label without consuming existing room
    area. Competing floor shapes for the same label are left unresolved.
    Physical walls, opening lists, and previously extracted rooms never change.
    """
    if not walls or not isfinite(pixels_per_metre) or pixels_per_metre <= 0:
        return OpenFloorRecoveryResult(rooms=list(rooms))
    existing = [Polygon([(point.x, point.y) for point in room.points]) for room in rooms]
    existing = [polygon for polygon in existing if polygon.is_valid and not polygon.is_empty]
    names = [label for label in labels if _confident_room_name(label)]
    missing = [
        label for label in names
        if not any(polygon.covers(_label_point(label, pixels_per_metre, image_height_px)) for polygon in existing)
    ]
    if not missing:
        return OpenFloorRecoveryResult(rooms=list(rooms))

    snapped = snap_endpoints_to_walls(walls, junction_snap_m)
    lines = [
        LineString([(wall.start.x, wall.start.y), (wall.end.x, wall.end.y)])
        for wall in snapped if wall.start.distance_to(wall.end) > 0
    ]
    if not lines:
        return OpenFloorRecoveryResult(rooms=list(rooms))
    linework = unary_union(lines)
    x0, y0, x1, y1 = linework.bounds
    gap_limit = min(5.0, max_gap_m, max(x1 - x0, y1 - y0) * 0.5)
    topology = enumerate_faces(
        walls, doors, windows,
        junction_snap_m=junction_snap_m,
        min_room_area_m2=0.45,
        unconfirmed_opening_range_m=(0.55, 1.40),
    )
    occupied = unary_union(existing)
    coord_tolerance = max(0.04, min(wall.thickness_m for wall in snapped) * 1.5)
    proposals: list[tuple[Polygon, dict[str, object], list[int]]] = []
    for gap in collinear_gaps(snapped, coord_tolerance):
        length = float(gap["length_m"])  # type: ignore[arg-type]
        if not 1.40 < length <= gap_limit:
            continue
        bridge: LineString = gap["line"]  # type: ignore[assignment]
        # Avoid diagonal floor cuts from unrelated parallel fragments. Minor
        # centreline offsets remain permissible, as in ordinary wall noding.
        (ax, ay), (bx, by) = bridge.coords
        if min(abs(ax - bx), abs(ay - by)) > min(0.10, coord_tolerance):
            continue
        graph = unary_union([*lines, *topology.closures, bridge], grid_size=GRAPH_NODING_GRID_M)
        for polygon in polygonize(graph):
            if not polygon.is_valid or polygon.interiors or polygon.area < 1.0:
                continue
            if polygon.intersection(occupied).area > 0.01:
                continue
            if polygon.boundary.intersection(bridge.buffer(2e-6)).length < 0.5:
                continue
            contained = [
                index for index, label in enumerate(missing)
                if polygon.buffer(-0.10).covers(_label_point(label, pixels_per_metre, image_height_px))
            ]
            if not contained:
                continue
            # Two distinct room names in one new face need a person to decide
            # their division. Duplicate OCR detections of one name are benign.
            contained_names = {missing[index].normalized_text.casefold() for index in contained}
            if len(contained_names) != 1:
                continue
            details: dict[str, object] = {
                "wall_a": gap["wall_a"], "wall_b": gap["wall_b"],
                "gap_length_m": length,
                "boundary_start": [ax, ay], "boundary_end": [bx, by],
                "physical_wall_added": False,
            }
            proposals.append((polygon, details, contained))

    recovered = list(rooms)
    issues: list[ValidationIssue] = []
    records: list[dict[str, object]] = []
    used_ids = {room.id for room in rooms}
    accepted_shapes: list[Polygon] = []
    for polygon, details, indices in proposals:
        competitors = [
            other for other, _details, other_indices in proposals
            if set(indices).intersection(other_indices)
            and polygon.symmetric_difference(other).area > 0.01
        ]
        if competitors or any(polygon.symmetric_difference(other).area <= 0.01 for other in accepted_shapes):
            continue
        if any(polygon.intersection(other).area > 0.01 for other in accepted_shapes):
            continue
        room = _to_room_polygon(len(recovered), polygon, labels, pixels_per_metre, image_height_px)
        # A nearby low-confidence OCR name must not replace the confident name
        # that justified recovering this face. Keep dimensions from the normal
        # room parser, while selecting the name from the accepted evidence.
        room_name = max((missing[index] for index in indices), key=lambda item: item.confidence)
        room_id = f"open_floor_{len(records):03d}"
        while room_id in used_ids:
            room_id += "_"
        used_ids.add(room_id)
        room = room.model_copy(update={
            "id": room_id,
            "name": room_name.normalized_text,
            "face_id": _face_id_for(polygon),
            "confidence": min(room.confidence, 0.65),
            "evidence_source": "inferred_open_floor_boundary",
        })
        recovered.append(room)
        accepted_shapes.append(polygon)
        records.append({**details, "room_id": room_id, "room_name": room.name, "area_m2": polygon.area})
        issues.append(ValidationIssue(
            code="inferred_open_floor_boundary",
            severity="warning",
            element_id=room_id,
            message=(f'Floor boundary for "{room.name}" crosses an open entrance inferred from '
                     "aligned wall ends. No physical wall or door was added. Review this floor edge before export."),
        ))
    return OpenFloorRecoveryResult(rooms=recovered, issues=issues, candidates=records)
