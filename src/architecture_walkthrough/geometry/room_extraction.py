from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import polygonize, unary_union

from architecture_walkthrough.geometry.models import Point2D, RoomPolygon, WallSegment
from architecture_walkthrough.vision.ocr import OCRText, parse_dimension_pair


@dataclass(frozen=True)
class RoomExtractionResult:
    rooms: list[RoomPolygon]
    rejected: list[str]


def _label_point(label: OCRText, pixels_per_metre: float, image_height_px: int | None) -> Point:
    lx = sum(point[0] for point in label.polygon) / len(label.polygon)
    ly = sum(point[1] for point in label.polygon) / len(label.polygon)
    if image_height_px is None:
        return Point(lx / pixels_per_metre, ly / pixels_per_metre)
    return Point(lx / pixels_per_metre, (image_height_px - ly) / pixels_per_metre)


def _to_room_polygon(
    index: int,
    polygon: Polygon,
    labels: list[OCRText],
    pixels_per_metre: float,
    image_height_px: int | None,
) -> RoomPolygon:
    coords = list(polygon.exterior.coords)[:-1]
    points = [Point2D(x=x, y=y) for x, y in coords]
    centroid = polygon.centroid
    best_label: OCRText | None = None
    best_distance = float("inf")
    for label in labels:
        label_point = _label_point(label, pixels_per_metre, image_height_px)
        if polygon.contains(label_point):
            best_label = label
            best_distance = 0.0
            break
        distance = centroid.distance(label_point)
        if distance < best_distance:
            best_distance = distance
            best_label = label
    name = None
    dimension = None
    confidence = 0.65
    if best_label is not None and best_distance < max(1.5, polygon.length * 0.12):
        name = best_label.normalized_text
        confidence = max(confidence, best_label.confidence)
        try:
            parsed = parse_dimension_pair(best_label.normalized_text)
        except ValueError:
            parsed = None
        if parsed:
            dimension = (parsed.width_m, parsed.height_m)
    return RoomPolygon(
        id=f"room_{index:03d}",
        name=name,
        points=points,
        confidence=confidence,
        evidence_source="wall_topology",
        dimension_m=dimension,
    )


def _close_collinear_gaps(lines: list[LineString], max_gap: float, coord_tol: float) -> list[LineString]:
    endpoints: list[tuple[float, float, str]] = []
    for line in lines:
        coords = list(line.coords)
        if len(coords) < 2:
            continue
        (x1, y1), (x2, y2) = coords[0], coords[-1]
        if abs(y1 - y2) <= coord_tol:
            y = (y1 + y2) / 2
            endpoints.append((x1, y, "h"))
            endpoints.append((x2, y, "h"))
        elif abs(x1 - x2) <= coord_tol:
            x = (x1 + x2) / 2
            endpoints.append((x, y1, "v"))
            endpoints.append((x, y2, "v"))
    closures: list[LineString] = []
    for orientation in ("h", "v"):
        oriented = [item for item in endpoints if item[2] == orientation]
        if orientation == "h":
            oriented.sort(key=lambda item: (round(item[1] / coord_tol), item[0]))
        else:
            oriented.sort(key=lambda item: (round(item[0] / coord_tol), item[1]))
        for first, second in zip(oriented, oriented[1:]):
            if orientation == "h":
                same_line = abs(first[1] - second[1]) <= coord_tol
                gap = second[0] - first[0]
                if same_line and 0.03 < gap <= max_gap:
                    closures.append(LineString([(first[0], first[1]), (second[0], first[1])]))
            else:
                same_line = abs(first[0] - second[0]) <= coord_tol
                gap = second[1] - first[1]
                if same_line and 0.03 < gap <= max_gap:
                    closures.append(LineString([(first[0], first[1]), (first[0], second[1])]))
    return closures


def extract_rooms_from_walls(
    walls: list[WallSegment],
    labels: list[OCRText],
    pixels_per_metre: float,
    min_area_m2: float = 0.45,
    close_gap_m: float = 2.4,
    image_height_px: int | None = None,
) -> RoomExtractionResult:
    lines = [LineString([(wall.start.x, wall.start.y), (wall.end.x, wall.end.y)]) for wall in walls if wall.start.distance_to(wall.end) > 0]
    if not lines:
        return RoomExtractionResult(rooms=[], rejected=["no wall lines available for room extraction"])
    coord_tol = max(0.04, min((wall.thickness_m for wall in walls), default=0.12) * 0.75)
    merged = unary_union([*lines, *_close_collinear_gaps(lines, close_gap_m, coord_tol)])
    polygons = list(polygonize(merged))
    if not polygons:
        return RoomExtractionResult(rooms=[], rejected=["wall topology did not close any room polygons"])
    max_area = max(polygon.area for polygon in polygons)
    rooms: list[RoomPolygon] = []
    rejected: list[str] = []
    for polygon in sorted(polygons, key=lambda item: item.area, reverse=True):
        if polygon.area < min_area_m2:
            rejected.append("polygon area below room threshold")
            continue
        if polygon.area == max_area and len(polygons) > 1 and not any(
            polygon.contains(_label_point(label, pixels_per_metre, image_height_px))
            for label in labels
        ):
            rejected.append("discarded likely exterior polygon")
            continue
        rooms.append(_to_room_polygon(len(rooms), polygon, labels, pixels_per_metre, image_height_px))
    return RoomExtractionResult(rooms=rooms, rejected=rejected)
