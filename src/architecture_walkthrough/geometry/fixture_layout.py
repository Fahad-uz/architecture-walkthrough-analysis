"""Place built-in fixtures using detected outlines and conservative room context."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from shapely.geometry import LineString, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from architecture_walkthrough.geometry.models import (
    FurniturePlacement,
    Point2D,
    RoomPolygon,
    WallSegment,
)
from architecture_walkthrough.vision.wall_detection import FixtureDetailRegion


def _polygons(geometry: BaseGeometry) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [] if geometry.is_empty else [geometry]
    return [part for child in getattr(geometry, "geoms", ()) for part in _polygons(child)]


def _rectangles(geometry: BaseGeometry) -> list[Polygon]:
    """Partition an orthogonal footprint without filling concavities or holes.

    Intersecting with an angled room/wall can leave a nonorthogonal edge. In
    that case only fully covered rectangles survive; no furniture is moved
    or expanded beyond the measured outline to compensate.
    """
    polygons = _polygons(geometry)
    coordinates = [
        (x, y)
        for polygon in polygons
        for ring in (polygon.exterior, *polygon.interiors)
        for x, y in ring.coords
    ]
    xs = sorted({x for x, _y in coordinates})
    ys = sorted({y for _x, y in coordinates})
    rectangles: list[Polygon] = []
    active: dict[tuple[float, float], tuple[float, float]] = {}
    # GEOS intersections may differ at a shared edge by roundoff only.
    covered = geometry.buffer(1e-9, join_style=2)
    for low_y, high_y in zip(ys, ys[1:]):
        runs: list[tuple[float, float]] = []
        run_start: float | None = None
        for low_x, high_x in zip(xs, xs[1:]):
            if covered.covers(box(low_x, low_y, high_x, high_y)):
                if run_start is None:
                    run_start = low_x
            elif run_start is not None:
                runs.append((run_start, low_x))
                run_start = None
        if run_start is not None:
            runs.append((run_start, xs[-1]))
        current = {
            span: (active[span][0] if span in active else low_y, high_y)
            for span in runs
        }
        rectangles.extend(
            box(span[0], bottom, span[1], top)
            for span, (bottom, top) in active.items()
            if span not in current
        )
        active = current
    rectangles.extend(
        box(span[0], bottom, span[1], top)
        for span, (bottom, top) in active.items()
    )
    return rectangles


def _fixture_kind(room: RoomPolygon) -> tuple[str, float] | None:
    name = (room.name or "").lower()
    if re.search(r"\bkitchen\b", name):
        return "kitchen_counter", 0.90
    if re.search(r"\b(?:bedroom|bed room)\b", name):
        return "wardrobe", 2.40
    return None


def fixture_furniture_from_regions(
    regions: Sequence[FixtureDetailRegion],
    rooms: Sequence[RoomPolygon],
    pixels_per_metre: float,
    image_height_px: float,
    *,
    walls: Sequence[WallSegment] = (),
    clearance_m: float = 0.07,
    max_depth_m: float = 1.20,
) -> list[FurniturePlacement]:
    """Convert image-space built-ins to world-space counter/wardrobe pieces.

    Geometry comes from the detected fixture outline. Category and height are
    room-context assumptions, which the caller should flag for review in its
    reconstruction metadata. Unnamed/unrecognized rooms are left unchanged.
    """
    if not math.isfinite(pixels_per_metre) or pixels_per_metre <= 0:
        raise ValueError("pixels_per_metre must be finite and positive")
    if not math.isfinite(image_height_px) or image_height_px <= 0:
        raise ValueError("image_height_px must be finite and positive")
    if not math.isfinite(clearance_m) or clearance_m < 0:
        raise ValueError("clearance_m must be finite and nonnegative")
    if not math.isfinite(max_depth_m) or max_depth_m <= 0:
        raise ValueError("max_depth_m must be finite and positive")

    room_polygons = []
    for room in rooms:
        polygon = Polygon([(point.x, point.y) for point in room.points])
        if polygon.is_valid and polygon.area > 0.05:
            room_polygons.append((room, polygon))
    wall_footprints = unary_union([
        LineString([(wall.start.x, wall.start.y), (wall.end.x, wall.end.y)]).buffer(
            wall.thickness_m / 2 + clearance_m, cap_style=2, join_style=2,
        )
        for wall in walls
        if wall.start.distance_to(wall.end) > 1e-6
    ])

    placements: list[FurniturePlacement] = []
    occupied: BaseGeometry = Polygon()
    for region in regions:
        if len(region.polygon) < 3:
            continue
        footprint = Polygon([
            (x / pixels_per_metre, (image_height_px - y) / pixels_per_metre)
            for x, y in region.polygon
        ])
        if not footprint.is_valid or footprint.area <= 0.05:
            continue
        candidates = [
            (polygon.intersection(footprint).area, room, polygon)
            for room, polygon in room_polygons
        ]
        if not candidates:
            continue
        overlap, room, room_polygon = max(candidates, key=lambda item: item[0])
        kind = _fixture_kind(room)
        if overlap < footprint.area * 0.70 or kind is None:
            continue
        available = (
            footprint.buffer(-clearance_m, join_style=2)
            .intersection(room_polygon.buffer(-clearance_m, join_style=2))
            .difference(wall_footprints)
            .difference(occupied)
        )
        for rectangle in _rectangles(available):
            min_x, min_y, max_x, max_y = rectangle.bounds
            width, depth = max_x - min_x, max_y - min_y
            # A broad filled region is not evidence for a shallow built-in.
            # Reject it rather than shrinking or relocating invented furniture.
            if min(width, depth) < 0.20 or min(width, depth) > max_depth_m:
                continue
            if rectangle.area < 0.04:
                continue
            center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
            # Cabinet fronts face the room; their long axis runs along the
            # measured wall, rather than stretching a narrow end into doors.
            if depth > width:
                width, depth = depth, width
                rotation = -90.0 if room_polygon.centroid.x < center_x else 90.0
            else:
                rotation = 0.0 if room_polygon.centroid.y < center_y else 180.0
            placements.append(FurniturePlacement(
                category=kind[0],
                center=Point2D(x=center_x, y=center_y),
                width_m=width,
                depth_m=depth,
                rotation_deg=rotation,
                height_m=kind[1],
            ))
            occupied = occupied.union(rectangle)
    return placements
