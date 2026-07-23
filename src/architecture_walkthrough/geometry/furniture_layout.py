from __future__ import annotations

from dataclasses import dataclass

from shapely.affinity import rotate, translate
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon, box
from shapely.ops import polylabel

from architecture_walkthrough.geometry.models import FurniturePlacement, Point2D, RoomPolygon


@dataclass(frozen=True)
class RoomBounds:
    room: RoomPolygon
    polygon: Polygon
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def depth(self) -> float:
        return self.max_y - self.min_y

    @property
    def center(self) -> Point2D:
        return Point2D(x=(self.min_x + self.max_x) / 2, y=(self.min_y + self.max_y) / 2)


def _bounds(room: RoomPolygon) -> RoomBounds:
    xs = [point.x for point in room.points]
    ys = [point.y for point in room.points]
    polygon = Polygon([(point.x, point.y) for point in room.points])
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    return RoomBounds(
        room=room,
        polygon=polygon,
        min_x=min(xs),
        min_y=min(ys),
        max_x=max(xs),
        max_y=max(ys),
    )


def _contains(bounds: RoomBounds, point: Point2D) -> bool:
    return bounds.polygon.covers(ShapelyPoint(point.x, point.y))


def _nearest_room(
    item: FurniturePlacement,
    rooms: list[RoomBounds],
    max_relocation_m: float,
) -> RoomBounds | None:
    if not rooms:
        return None
    containing = [room for room in rooms if _contains(room, item.center)]
    if containing:
        return min(containing, key=lambda room: room.width * room.depth)

    # A detected footprint just outside a room edge can be harmless detector
    # jitter, but moving an object across the plan hides a bad semantic hint
    # and creates the characteristic pile of furniture in the nearest room.
    # Only repair small edge errors; otherwise reject the placement.
    def distance_to_bounds(room: RoomBounds) -> float:
        return room.polygon.distance(ShapelyPoint(item.center.x, item.center.y))

    nearest = min(rooms, key=distance_to_bounds)
    return nearest if distance_to_bounds(nearest) <= max_relocation_m else None


def _footprint(item: FurniturePlacement, center: Point2D, width: float, depth: float) -> Polygon:
    footprint = box(-width / 2, -depth / 2, width / 2, depth / 2)
    footprint = rotate(footprint, item.rotation_deg, origin=(0, 0), use_radians=False)
    return translate(footprint, center.x, center.y)


def _fit_item(
    item: FurniturePlacement,
    room: RoomBounds,
    margin_m: float,
) -> FurniturePlacement | None:
    available = room.polygon.buffer(-margin_m)
    if available.is_empty:
        return None
    if available.geom_type == "MultiPolygon":
        available = max(available.geoms, key=lambda polygon: polygon.area)
    if not isinstance(available, Polygon) or available.area <= 0.02:
        return None

    width = min(item.width_m, max(0.20, room.width - margin_m * 2), max(0.20, room.width * 0.90))
    depth = min(item.depth_m, max(0.20, room.depth - margin_m * 2), max(0.20, room.depth * 0.90))
    safe = polylabel(available, tolerance=0.03)
    # Pull an edge-straddling footprint inward gradually. This preserves the
    # detector's location when possible and never jumps it to another room.
    for movement in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        center = Point2D(
            x=item.center.x + (float(safe.x) - item.center.x) * movement,
            y=item.center.y + (float(safe.y) - item.center.y) * movement,
        )
        for scale in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
            fitted_width = max(0.20, width * scale)
            fitted_depth = max(0.20, depth * scale)
            if available.covers(_footprint(item, center, fitted_width, fitted_depth)):
                return item.model_copy(
                    update={
                        "center": center,
                        "width_m": fitted_width,
                        "depth_m": fitted_depth,
                    }
                )
    return None


def fit_furniture_to_rooms(
    furniture: list[FurniturePlacement],
    rooms: list[RoomPolygon],
    margin_m: float = 0.08,
    max_relocation_m: float = 0.35,
    preserve_unassigned: bool = False,
) -> list[FurniturePlacement]:
    room_bounds = [
        bounds
        for room in rooms
        if len(room.points) >= 3
        and not (bounds := _bounds(room)).polygon.is_empty
        and bounds.polygon.area > 0.02
    ]
    if not room_bounds:
        return deduplicate_furniture(furniture)

    fitted: list[FurniturePlacement] = []
    for item in furniture:
        room = _nearest_room(item, room_bounds, max_relocation_m)
        if room is None:
            if preserve_unassigned:
                fitted.append(item)
            continue
        if room.width <= margin_m * 2 or room.depth <= margin_m * 2:
            continue
        fitted_item = _fit_item(item, room, margin_m)
        if fitted_item is not None:
            fitted.append(fitted_item)
    return deduplicate_furniture(fitted)


def deduplicate_furniture(items: list[FurniturePlacement]) -> list[FurniturePlacement]:
    kept: list[FurniturePlacement] = []
    for item in sorted(items, key=lambda value: value.width_m * value.depth_m, reverse=True):
        if any(
            item.category == existing.category
            and item.center.distance_to(existing.center) < max(0.25, min(item.width_m, item.depth_m) * 0.55)
            for existing in kept
        ):
            continue
        kept.append(item)
    return kept[:100]
