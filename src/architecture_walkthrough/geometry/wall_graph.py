from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import polygonize, unary_union

from architecture_walkthrough.geometry.models import (
    DoorOpening,
    Point2D,
    RoomPolygon,
    WallSegment,
    WindowOpening,
)

# Endpoint sloppiness we silently repair. Anything larger must be explained by
# a confirmed opening or fixed in the correction editor — never auto-closed.
DEFAULT_JUNCTION_SNAP_M = 0.20
# Widest gap a confirmed door/window is allowed to close during face enumeration.
DEFAULT_OPENING_CLOSURE_MAX_M = 2.0
DEFAULT_MIN_ROOM_AREA_M2 = 0.8
# Faces thinner than this are wall-thickness slivers, not rooms.
MIN_FACE_THINNESS_RATIO = 0.05


@dataclass(frozen=True)
class GraphFace:
    face_id: str
    polygon: Polygon


@dataclass(frozen=True)
class FaceEnumerationResult:
    faces: list[GraphFace]
    closures: list[LineString] = field(default_factory=list)
    unclosed_gaps: list[dict[str, object]] = field(default_factory=list)


def _wall_lines(walls: list[WallSegment]) -> list[tuple[WallSegment, LineString]]:
    lines: list[tuple[WallSegment, LineString]] = []
    for wall in walls:
        if wall.start.distance_to(wall.end) <= 0:
            continue
        lines.append((wall, LineString([(wall.start.x, wall.start.y), (wall.end.x, wall.end.y)])))
    return lines


def snap_endpoints_to_walls(walls: list[WallSegment], tolerance_m: float = DEFAULT_JUNCTION_SNAP_M) -> list[WallSegment]:
    """Snap wall endpoints onto nearby walls so T/L junctions actually touch.

    Only endpoints within `tolerance_m` of another wall's centerline move; this
    repairs detector sloppiness without inventing geometry.
    """
    lines = _wall_lines(walls)
    snapped: list[WallSegment] = []
    for wall, _ in lines:
        updates: dict[str, Point2D] = {}
        for attr in ("start", "end"):
            endpoint: Point2D = getattr(wall, attr)
            point = Point(endpoint.x, endpoint.y)
            best: tuple[float, Point] | None = None
            for other, other_line in lines:
                if other is wall:
                    continue
                distance = point.distance(other_line)
                if distance <= tolerance_m and (best is None or distance < best[0]):
                    projected = other_line.interpolate(other_line.project(point))
                    best = (distance, projected)
            if best is not None and best[0] > 1e-9:
                updates[attr] = Point2D(x=best[1].x, y=best[1].y)
        snapped.append(wall.model_copy(update=updates) if updates else wall)
    return snapped


def _collinear_gaps(walls: list[WallSegment], coord_tol: float) -> list[dict[str, object]]:
    """Find gaps between successive collinear wall endpoints on the same line."""
    gaps: list[dict[str, object]] = []
    horizontal = [w for w in walls if abs(w.end.x - w.start.x) >= abs(w.end.y - w.start.y)]
    vertical = [w for w in walls if abs(w.end.x - w.start.x) < abs(w.end.y - w.start.y)]
    for oriented, is_horizontal in ((horizontal, True), (vertical, False)):
        def line_coord(wall: WallSegment) -> float:
            return (wall.start.y + wall.end.y) / 2 if is_horizontal else (wall.start.x + wall.end.x) / 2

        def span(wall: WallSegment) -> tuple[float, float]:
            values = (wall.start.x, wall.end.x) if is_horizontal else (wall.start.y, wall.end.y)
            return min(values), max(values)

        ordered = sorted(oriented, key=lambda wall: (round(line_coord(wall) / max(coord_tol, 1e-6)), span(wall)[0]))
        for first, second in zip(ordered, ordered[1:]):
            if abs(line_coord(first) - line_coord(second)) > coord_tol:
                continue
            gap_start = span(first)[1]
            gap_end = span(second)[0]
            if gap_end - gap_start <= 1e-6:
                continue
            coord = (line_coord(first) + line_coord(second)) / 2
            if is_horizontal:
                a, b = (gap_start, coord), (gap_end, coord)
            else:
                a, b = (coord, gap_start), (coord, gap_end)
            gaps.append(
                {
                    "wall_a": first.id,
                    "wall_b": second.id,
                    "length_m": gap_end - gap_start,
                    "line": LineString([a, b]),
                }
            )
    return gaps


def _opening_spans_gap(
    gap: dict[str, object],
    walls_by_id: dict[str, WallSegment],
    openings: list[DoorOpening | WindowOpening],
) -> bool:
    """True when a confirmed opening interval reaches into the gap span."""
    gap_line: LineString = gap["line"]  # type: ignore[assignment]
    for opening in openings:
        if opening.wall_id not in (gap["wall_a"], gap["wall_b"]):
            continue
        wall = walls_by_id.get(opening.wall_id or "")
        if wall is None:
            continue
        interval = opening.interval(opening.width_m)
        if interval is None:
            continue
        length = wall.start.distance_to(wall.end)
        if length <= 0:
            continue
        # Opening interval endpoints in plan coordinates.
        ux = (wall.end.x - wall.start.x) / length
        uy = (wall.end.y - wall.start.y) / length
        for offset in interval:
            px = wall.start.x + ux * offset
            py = wall.start.y + uy * offset
            if gap_line.distance(Point(px, py)) <= max(opening.width_m, 0.3):
                return True
    return False


def _face_id_for(polygon: Polygon) -> str:
    centroid = polygon.centroid
    digest = hashlib.sha1(
        f"{centroid.x:.2f}|{centroid.y:.2f}|{polygon.area:.2f}".encode()
    ).hexdigest()[:10]
    return f"face_{digest}"


def enumerate_faces(
    walls: list[WallSegment],
    doors: list[DoorOpening] | None = None,
    windows: list[WindowOpening] | None = None,
    *,
    junction_snap_m: float = DEFAULT_JUNCTION_SNAP_M,
    opening_closure_max_m: float = DEFAULT_OPENING_CLOSURE_MAX_M,
    min_room_area_m2: float = DEFAULT_MIN_ROOM_AREA_M2,
) -> FaceEnumerationResult:
    """Enumerate room faces of the planar wall graph.

    Gap policy: endpoint sloppiness up to `junction_snap_m` is repaired; larger
    collinear gaps close for enumeration only when a confirmed opening spans
    them. Everything else is reported in `unclosed_gaps`, never auto-closed.
    """
    snapped = snap_endpoints_to_walls(walls, junction_snap_m)
    lines = [line for _, line in _wall_lines(snapped)]
    if not lines:
        return FaceEnumerationResult(faces=[])

    walls_by_id = {wall.id: wall for wall in snapped if wall.id}
    openings: list[DoorOpening | WindowOpening] = [*(doors or []), *(windows or [])]
    thickness = min((wall.thickness_m for wall in snapped), default=0.12)
    coord_tol = max(0.04, thickness * 0.75)

    closures: list[LineString] = []
    unclosed: list[dict[str, object]] = []
    for gap in _collinear_gaps(snapped, coord_tol):
        length_m = float(gap["length_m"])  # type: ignore[arg-type]
        if length_m <= junction_snap_m:
            closures.append(gap["line"])  # type: ignore[arg-type]
        elif length_m <= opening_closure_max_m and _opening_spans_gap(gap, walls_by_id, openings):
            closures.append(gap["line"])  # type: ignore[arg-type]
        else:
            unclosed.append({"wall_a": gap["wall_a"], "wall_b": gap["wall_b"], "length_m": length_m})

    merged = unary_union([*lines, *closures])
    faces: list[GraphFace] = []
    for polygon in polygonize(merged):
        if polygon.area < min_room_area_m2:
            continue
        if polygon.length > 0 and polygon.area / polygon.length < MIN_FACE_THINNESS_RATIO:
            continue
        faces.append(GraphFace(face_id=_face_id_for(polygon), polygon=polygon))
    faces.sort(key=lambda face: face.polygon.area, reverse=True)
    return FaceEnumerationResult(faces=faces, closures=closures, unclosed_gaps=unclosed)


def _polygon_points(polygon: Polygon) -> list[Point2D]:
    return [Point2D(x=float(x), y=float(y)) for x, y in list(polygon.exterior.coords)[:-1]]


def match_faces_to_rooms(
    faces: list[GraphFace],
    previous_rooms: list[RoomPolygon],
    min_overlap_ratio: float = 0.5,
) -> list[RoomPolygon]:
    """Carry room semantics (face_id, name, confidence) onto regenerated faces.

    A previous room transfers to the new face it overlaps most (IoU of the
    smaller area >= `min_overlap_ratio`). Each previous room maps at most once.
    """
    previous_polygons: list[tuple[RoomPolygon, Polygon]] = []
    for room in previous_rooms:
        polygon = Polygon([(point.x, point.y) for point in room.points])
        if polygon.is_valid and polygon.area > 0:
            previous_polygons.append((room, polygon))

    consumed: set[int] = set()
    rooms: list[RoomPolygon] = []
    for index, face in enumerate(faces):
        best: tuple[float, int] | None = None
        for prev_index, (_, prev_polygon) in enumerate(previous_polygons):
            if prev_index in consumed:
                continue
            intersection = face.polygon.intersection(prev_polygon).area
            smaller = min(face.polygon.area, prev_polygon.area)
            if smaller <= 0:
                continue
            ratio = intersection / smaller
            if ratio >= min_overlap_ratio and (best is None or ratio > best[0]):
                best = (ratio, prev_index)
        if best is not None:
            consumed.add(best[1])
            previous = previous_polygons[best[1]][0]
            rooms.append(
                previous.model_copy(
                    update={
                        "id": previous.id or f"room_{index:03d}",
                        "face_id": previous.face_id or face.face_id,
                        "points": _polygon_points(face.polygon),
                        "evidence_source": "wall_graph_face",
                    }
                )
            )
        else:
            rooms.append(
                RoomPolygon(
                    id=f"room_{index:03d}",
                    face_id=face.face_id,
                    points=_polygon_points(face.polygon),
                    confidence=0.65,
                    evidence_source="wall_graph_face",
                )
            )
    return rooms
