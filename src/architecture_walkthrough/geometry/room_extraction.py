from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Point, Polygon

from architecture_walkthrough.geometry.models import DoorOpening, Point2D, RoomPolygon, WallSegment, WindowOpening
from architecture_walkthrough.geometry.wall_graph import DEFAULT_JUNCTION_SNAP_M, enumerate_faces
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
    # A room commonly contains two independent pieces of text (for example
    # ``BEDROOM`` and ``300X290``).  Treating the first label encountered as
    # both the name and the dimension made the result depend on list order and
    # silently threw away the dimension that the scale solver needs.
    nearby: list[tuple[OCRText, bool, float]] = []
    max_distance = max(1.5, polygon.length * 0.12)
    for label in labels:
        label_point = _label_point(label, pixels_per_metre, image_height_px)
        inside = polygon.covers(label_point)
        distance = centroid.distance(label_point)
        if inside or distance <= max_distance:
            nearby.append((label, inside, distance))

    def label_rank(candidate: tuple[OCRText, bool, float]) -> tuple[bool, float, float]:
        label, inside, distance = candidate
        return (not inside, distance, -label.confidence)

    nearby.sort(key=label_rank)
    name = None
    dimension = None
    confidence = 0.65

    # Dimension parsing is deliberately independent of semantic_type.  OCR
    # and vision models occasionally classify a bare ``300X290`` string as a
    # room label; it is still valid dimension evidence.
    for label, inside, _distance in nearby:
        if not inside:
            continue
        try:
            parsed = parse_dimension_pair(label.normalized_text)
        except ValueError:
            parsed = None
        if parsed and min(parsed.width_m, parsed.height_m) >= 1.2:
            dimension = (parsed.width_m, parsed.height_m)
            confidence = max(confidence, label.confidence)
            break

    for label, inside, _distance in nearby:
        if not inside:
            continue
        try:
            parsed = parse_dimension_pair(label.normalized_text)
        except ValueError:
            parsed = None
        if parsed is not None or label.semantic_type == "dimension":
            continue
        if label.semantic_type not in {
            "room_label",
            "balcony_label",
            "lift_label",
            "stair_label",
        }:
            continue
        name = label.normalized_text
        confidence = max(confidence, label.confidence)
        break
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
    image_height_px: int | None = None,
    doors: list[DoorOpening] | None = None,
    windows: list[WindowOpening] | None = None,
    junction_snap_m: float = DEFAULT_JUNCTION_SNAP_M,
    bridge_ambiguous_openings: bool = False,
) -> RoomExtractionResult:
    """Rooms are faces of the planar wall graph.

    Doorway gaps close during face enumeration only when a confirmed opening
    spans them; unexplained gaps are reported, never auto-closed.
    """
    if not any(wall.start.distance_to(wall.end) > 0 for wall in walls):
        return RoomExtractionResult(rooms=[], rejected=["no wall lines available for room extraction"])
    result = enumerate_faces(
        walls,
        doors,
        windows,
        junction_snap_m=junction_snap_m,
        min_room_area_m2=min_area_m2,
        unconfirmed_opening_range_m=(0.55, 1.40) if bridge_ambiguous_openings else None,
    )
    if not result.faces:
        return RoomExtractionResult(rooms=[], rejected=["wall topology did not close any room polygons"])
    rooms: list[RoomPolygon] = []
    for face in result.faces:
        room = _to_room_polygon(len(rooms), face.polygon, labels, pixels_per_metre, image_height_px)
        rooms.append(room.model_copy(update={"face_id": face.face_id}))
    rejected = [
        f"unclosed gap of {gap['length_m']:.2f} m between walls {gap['wall_a']} and {gap['wall_b']}"
        for gap in result.unclosed_gaps
    ]
    return RoomExtractionResult(rooms=rooms, rejected=rejected)


def extract_rooms_from_geometry_mask(
    mask_path: Path,
    labels: list[OCRText],
    pixels_per_metre: float,
    image_height_px: int,
    close_gap_px: int = 90,
    min_area_m2: float = 0.45,
) -> RoomExtractionResult:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return RoomExtractionResult(rooms=[], rejected=[f"failed to read geometry mask: {mask_path}"])
    _, wall_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    kernel_size = max(5, close_gap_px | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed_walls = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    closed_walls = cv2.dilate(closed_walls, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
    free = cv2.bitwise_not(closed_walls)
    flood = free.copy()
    flood_mask = np.zeros((free.shape[0] + 2, free.shape[1] + 2), dtype=np.uint8)
    for seed in ((0, 0), (free.shape[1] - 1, 0), (0, free.shape[0] - 1), (free.shape[1] - 1, free.shape[0] - 1)):
        if flood[seed[1], seed[0]]:
            cv2.floodFill(flood, flood_mask, seed, 0)
    interior = flood
    contours, _ = cv2.findContours(interior, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rooms: list[RoomPolygon] = []
    rejected: list[str] = []
    min_area_px = min_area_m2 * pixels_per_metre * pixels_per_metre
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area_px = cv2.contourArea(contour)
        if area_px < min_area_px:
            rejected.append("mask room contour area below threshold")
            continue
        epsilon = max(2.0, cv2.arcLength(contour, True) * 0.01)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            rejected.append("mask room contour simplified below polygon threshold")
            continue
        points = [
            Point2D(
                x=float(point[0][0]) / pixels_per_metre,
                y=float(image_height_px - point[0][1]) / pixels_per_metre,
            )
            for point in approx
        ]
        polygon = Polygon([(point.x, point.y) for point in points])
        if not polygon.is_valid or polygon.area < min_area_m2:
            rejected.append("mask room polygon invalid or too small")
            continue
        room = _to_room_polygon(len(rooms), polygon, labels, pixels_per_metre, image_height_px)
        rooms.append(room.model_copy(update={"points": points, "evidence_source": "closed_wall_mask"}))
    return RoomExtractionResult(rooms=rooms, rejected=rejected)
