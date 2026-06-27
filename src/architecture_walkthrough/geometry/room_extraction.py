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


def _to_room_polygon(index: int, polygon: Polygon, labels: list[OCRText], pixels_per_metre: float) -> RoomPolygon:
    coords = list(polygon.exterior.coords)[:-1]
    points = [Point2D(x=x, y=y) for x, y in coords]
    centroid = polygon.centroid
    best_label: OCRText | None = None
    best_distance = float("inf")
    for label in labels:
        lx = sum(point[0] for point in label.polygon) / len(label.polygon)
        ly = sum(point[1] for point in label.polygon) / len(label.polygon)
        label_point = Point(lx / pixels_per_metre, ly / pixels_per_metre)
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


def extract_rooms_from_walls(
    walls: list[WallSegment],
    labels: list[OCRText],
    pixels_per_metre: float,
    min_area_m2: float = 0.45,
) -> RoomExtractionResult:
    lines = [LineString([(wall.start.x, wall.start.y), (wall.end.x, wall.end.y)]) for wall in walls if wall.start.distance_to(wall.end) > 0]
    if not lines:
        return RoomExtractionResult(rooms=[], rejected=["no wall lines available for room extraction"])
    merged = unary_union(lines)
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
            polygon.contains(Point(sum(p[0] for p in label.polygon) / len(label.polygon) / pixels_per_metre, sum(p[1] for p in label.polygon) / len(label.polygon) / pixels_per_metre))
            for label in labels
        ):
            rejected.append("discarded likely exterior polygon")
            continue
        rooms.append(_to_room_polygon(len(rooms), polygon, labels, pixels_per_metre))
    return RoomExtractionResult(rooms=rooms, rejected=rejected)
