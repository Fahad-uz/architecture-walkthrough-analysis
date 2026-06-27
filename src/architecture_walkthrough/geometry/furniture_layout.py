from __future__ import annotations

from dataclasses import dataclass

from architecture_walkthrough.geometry.models import FurniturePlacement, Point2D, RoomPolygon


@dataclass(frozen=True)
class RoomBounds:
    room: RoomPolygon
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
    return RoomBounds(room=room, min_x=min(xs), min_y=min(ys), max_x=max(xs), max_y=max(ys))


def _contains(bounds: RoomBounds, point: Point2D) -> bool:
    return bounds.min_x <= point.x <= bounds.max_x and bounds.min_y <= point.y <= bounds.max_y


def _nearest_room(item: FurniturePlacement, rooms: list[RoomBounds]) -> RoomBounds | None:
    if not rooms:
        return None
    containing = [room for room in rooms if _contains(room, item.center)]
    if containing:
        return min(containing, key=lambda room: room.width * room.depth)
    return min(rooms, key=lambda room: item.center.distance_to(room.center))


def fit_furniture_to_rooms(
    furniture: list[FurniturePlacement],
    rooms: list[RoomPolygon],
    margin_m: float = 0.08,
) -> list[FurniturePlacement]:
    room_bounds = [_bounds(room) for room in rooms if len(room.points) >= 3]
    if not room_bounds:
        return deduplicate_furniture(furniture)

    fitted: list[FurniturePlacement] = []
    for item in furniture:
        room = _nearest_room(item, room_bounds)
        if room is None or room.width <= margin_m * 2 or room.depth <= margin_m * 2:
            continue
        width = min(item.width_m, max(0.20, room.width - margin_m * 2), max(0.20, room.width * 0.90))
        depth = min(item.depth_m, max(0.20, room.depth - margin_m * 2), max(0.20, room.depth * 0.90))
        half_w = width / 2
        half_d = depth / 2
        min_x = room.min_x + margin_m + half_w
        max_x = room.max_x - margin_m - half_w
        min_y = room.min_y + margin_m + half_d
        max_y = room.max_y - margin_m - half_d
        center = Point2D(
            x=min(max(item.center.x, min_x), max_x) if min_x <= max_x else room.center.x,
            y=min(max(item.center.y, min_y), max_y) if min_y <= max_y else room.center.y,
        )
        fitted.append(item.model_copy(update={"center": center, "width_m": width, "depth_m": depth}))
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
