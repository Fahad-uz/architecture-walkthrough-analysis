from __future__ import annotations

import heapq
import math
import re
from collections.abc import Iterable

from shapely.affinity import rotate, translate
from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import polylabel, unary_union
from shapely.prepared import prep

from architecture_walkthrough.geometry.models import (
    AssetPlacement,
    CameraWaypoint,
    FloorPlanModel,
    FurniturePlacement,
    Point2D,
    RoomPolygon,
)
from architecture_walkthrough.walkthrough.navigation_map import NavigationMap, build_navigation_map


GridPoint = tuple[int, int]

DEFAULT_CAMERA_RADIUS_M = 0.30
ROUTE_SAFETY_MARGIN_M = 0.03
ROUTE_GRID_SIZE_M = 0.10
DOOR_LEAF_OPEN_DEG = 90.0
_DOOR_LEAF_THICKNESS_M = 0.045
_MAX_WAYPOINT_SNAP_M = 2.0


def _to_grid(point: Point2D, nav: NavigationMap) -> GridPoint:
    return (
        round((point.x - nav.min_x) / nav.grid_size_m),
        round((point.y - nav.min_y) / nav.grid_size_m),
    )


def _from_grid(point: GridPoint, nav: NavigationMap) -> Point2D:
    return Point2D(
        x=nav.min_x + point[0] * nav.grid_size_m,
        y=nav.min_y + point[1] * nav.grid_size_m,
    )


def _neighbors(point: GridPoint) -> Iterable[GridPoint]:
    x, y = point
    yield x + 1, y
    yield x - 1, y
    yield x, y + 1
    yield x, y - 1


def _walkable_wall_parts(model: FloorPlanModel) -> list[tuple[LineString, float]]:
    """Return wall centreline pieces with confirmed doorway intervals removed."""

    doors_by_wall: dict[str, list[tuple[float, float]]] = {}
    for door in model.doors:
        if not door.wall_id:
            continue
        interval = door.interval(door.width_m)
        if interval is not None:
            doors_by_wall.setdefault(door.wall_id, []).append(interval)

    obstacles: list[tuple[LineString, float]] = []
    for wall in model.walls:
        length = wall.start.distance_to(wall.end)
        if length <= 1e-6:
            continue
        ux = (wall.end.x - wall.start.x) / length
        uy = (wall.end.y - wall.start.y) / length
        cursor = 0.0
        intervals = sorted(
            (max(0.0, start), min(length, end))
            for start, end in doors_by_wall.get(wall.id or "", [])
            if end > 0 and start < length
        )
        for start, end in intervals:
            if start > cursor + 1e-6:
                obstacles.append(
                    (
                        LineString(
                            [
                                (wall.start.x + ux * cursor, wall.start.y + uy * cursor),
                                (wall.start.x + ux * start, wall.start.y + uy * start),
                            ]
                        ),
                        wall.thickness_m,
                    )
                )
            cursor = max(cursor, end)
        if cursor < length - 1e-6:
            obstacles.append(
                (
                    LineString(
                        [
                            (wall.start.x + ux * cursor, wall.start.y + uy * cursor),
                            (wall.end.x, wall.end.y),
                        ]
                    ),
                    wall.thickness_m,
                )
            )
    return obstacles


def _walkable_wall_lines(model: FloorPlanModel) -> list[LineString]:
    """Compatibility helper returning doorway-split wall centrelines."""

    return [line for line, _thickness in _walkable_wall_parts(model)]


def _is_walkable_placement(category: str) -> bool:
    normalized = category.lower().replace("-", "_").replace(" ", "_")
    tokens = set(re.split(r"_+", normalized))
    return (
        "floor_patch" in normalized
        or "rug" in tokens
        or "carpet" in tokens
        or "door" in tokens
        or "window" in tokens
    )


def _placement_footprint(
    center: Point2D,
    width_m: float,
    depth_m: float,
    rotation_deg: float,
) -> BaseGeometry:
    footprint = box(-width_m / 2, -depth_m / 2, width_m / 2, depth_m / 2)
    footprint = rotate(footprint, rotation_deg, origin=(0, 0), use_radians=False)
    return translate(footprint, center.x, center.y)


def _placement_obstacle(
    placement: FurniturePlacement | AssetPlacement,
    clearance_m: float,
) -> BaseGeometry | None:
    if _is_walkable_placement(placement.category):
        return None
    footprint = _placement_footprint(
        placement.center,
        placement.width_m,
        placement.depth_m,
        placement.rotation_deg,
    )
    return footprint.buffer(clearance_m)


def _door_leaf_obstacles(
    model: FloorPlanModel,
    clearance_m: float,
) -> list[BaseGeometry]:
    """Model procedural door leaves at the renderer's fully-open angle."""

    walls_by_id = {wall.id: wall for wall in model.walls if wall.id}
    obstacles: list[BaseGeometry] = []
    for door in model.doors:
        if not door.wall_id:
            continue
        wall = walls_by_id.get(door.wall_id)
        if wall is None:
            continue
        wall_length = wall.start.distance_to(wall.end)
        if wall_length <= 1e-6:
            continue
        interval = door.interval(door.width_m)
        if interval is None:
            continue
        start_m = max(0.0, min(interval))
        end_m = min(wall_length, max(interval))
        width_m = end_m - start_m
        if width_m < 0.18:
            continue

        wall_angle = math.atan2(
            wall.end.y - wall.start.y,
            wall.end.x - wall.start.x,
        )
        hinge_at_start = (door.hinge_side or "start") == "start"
        hinge_offset = start_m if hinge_at_start else end_m
        leaf_direction = 1.0 if hinge_at_start else -1.0
        swing = 1.0 if (door.swing_side or "left") == "left" else -1.0
        leaf_angle = wall_angle + leaf_direction * swing * math.radians(DOOR_LEAF_OPEN_DEG)
        wall_ux = math.cos(wall_angle)
        wall_uy = math.sin(wall_angle)
        hinge_x = wall.start.x + wall_ux * hinge_offset
        hinge_y = wall.start.y + wall_uy * hinge_offset
        leaf_length = max(0.14, width_m - 0.04)
        center = Point2D(
            x=hinge_x + math.cos(leaf_angle) * leaf_direction * width_m / 2,
            y=hinge_y + math.sin(leaf_angle) * leaf_direction * width_m / 2,
        )
        footprint = _placement_footprint(
            center,
            leaf_length,
            _DOOR_LEAF_THICKNESS_M,
            math.degrees(leaf_angle),
        )
        obstacles.append(footprint.buffer(clearance_m))
    return obstacles


def _valid_room_polygons(model: FloorPlanModel) -> list[Polygon]:
    polygons: list[Polygon] = []
    for room in model.rooms:
        polygon = Polygon([(point.x, point.y) for point in room.points])
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if isinstance(polygon, Polygon) and polygon.area > 0.05:
            polygons.append(polygon)
    return polygons


def _polygon_parts(geometry: BaseGeometry) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry]
    if hasattr(geometry, "geoms"):
        return [
            polygon
            for member in geometry.geoms
            for polygon in _polygon_parts(member)
            if polygon.area > 0.01
        ]
    return []


def _build_free_space(
    model: FloorPlanModel,
    clearance_m: float,
    grid_size_m: float,
) -> tuple[BaseGeometry, NavigationMap]:
    rooms = _valid_room_polygons(model)
    if rooms:
        room_union = unary_union(rooms)
        domain = room_union.buffer(-clearance_m)
    else:
        fallback_nav = build_navigation_map(model, grid_size_m=grid_size_m)
        domain = box(
            fallback_nav.min_x,
            fallback_nav.min_y,
            fallback_nav.max_x,
            fallback_nav.max_y,
        )

    obstacles: list[BaseGeometry] = []
    for wall_line, thickness_m in _walkable_wall_parts(model):
        # Model the wall's rectangular footprint first so its doorway jamb ends
        # are expanded only by player clearance, not by wall half-thickness too.
        wall_footprint = wall_line.buffer(thickness_m / 2, cap_style="flat")
        obstacles.append(wall_footprint.buffer(clearance_m))

    obstacles.extend(_door_leaf_obstacles(model, clearance_m))
    for placement in model.furniture:
        if (obstacle := _placement_obstacle(placement, clearance_m)) is not None:
            obstacles.append(obstacle)
    for asset_placement in model.asset_placements:
        if (obstacle := _placement_obstacle(asset_placement, clearance_m)) is not None:
            obstacles.append(obstacle)

    free_space = domain
    if obstacles:
        free_space = domain.difference(unary_union(obstacles))
    if free_space.is_empty:
        raise ValueError("floor plan has no collision-safe walkable space")

    min_x, min_y, max_x, max_y = free_space.bounds
    nav = NavigationMap(
        min_x=min_x,
        min_y=min_y,
        max_x=max_x,
        max_y=max_y,
        grid_size_m=grid_size_m,
    )
    return free_space, nav


class _RoutePlanner:
    def __init__(
        self,
        model: FloorPlanModel,
        camera_radius_m: float,
        grid_size_m: float = ROUTE_GRID_SIZE_M,
    ) -> None:
        clearance_m = camera_radius_m + ROUTE_SAFETY_MARGIN_M
        self.free_space, self.nav = _build_free_space(model, clearance_m, grid_size_m)
        self.prepared_free_space = prep(self.free_space)
        self.max_x = math.ceil((self.nav.max_x - self.nav.min_x) / self.nav.grid_size_m)
        self.max_y = math.ceil((self.nav.max_y - self.nav.min_y) / self.nav.grid_size_m)
        self._walkable_cache: dict[GridPoint, bool] = {}
        self._edge_cache: dict[tuple[GridPoint, GridPoint], bool] = {}

    def contains(self, point: Point2D) -> bool:
        return self.prepared_free_space.covers(Point(point.x, point.y))

    def segment_is_safe(self, start: Point2D, end: Point2D) -> bool:
        return self.prepared_free_space.covers(
            LineString([(start.x, start.y), (end.x, end.y)])
        )

    def _grid_is_walkable(self, point: GridPoint) -> bool:
        cached = self._walkable_cache.get(point)
        if cached is not None:
            return cached
        walkable = (
            0 <= point[0] <= self.max_x
            and 0 <= point[1] <= self.max_y
            and self.contains(_from_grid(point, self.nav))
        )
        self._walkable_cache[point] = walkable
        return walkable

    def _edge_is_walkable(self, start: GridPoint, end: GridPoint) -> bool:
        key = (start, end) if start <= end else (end, start)
        cached = self._edge_cache.get(key)
        if cached is not None:
            return cached
        walkable = self.segment_is_safe(_from_grid(start, self.nav), _from_grid(end, self.nav))
        self._edge_cache[key] = walkable
        return walkable

    def _nearest_walkable_grid(self, requested: Point2D) -> GridPoint:
        requested_grid = _to_grid(requested, self.nav)
        queue = [requested_grid]
        visited = {requested_grid}
        max_steps = math.ceil(_MAX_WAYPOINT_SNAP_M / self.nav.grid_size_m)
        while queue:
            current = queue.pop(0)
            if self._grid_is_walkable(current):
                return current
            for neighbor in _neighbors(current):
                if neighbor in visited:
                    continue
                if (
                    abs(neighbor[0] - requested_grid[0])
                    + abs(neighbor[1] - requested_grid[1])
                    > max_steps
                ):
                    continue
                visited.add(neighbor)
                queue.append(neighbor)
        raise ValueError("no collision-safe point near requested waypoint")

    def snap(self, requested: Point2D) -> Point2D:
        if self.contains(requested):
            return requested
        return _from_grid(self._nearest_walkable_grid(requested), self.nav)

    def path(self, start: Point2D, goal: Point2D) -> list[Point2D]:
        start_grid = self._nearest_walkable_grid(start)
        goal_grid = self._nearest_walkable_grid(goal)
        queue: list[tuple[float, float, GridPoint]] = [(0.0, 0.0, start_grid)]
        came_from: dict[GridPoint, GridPoint | None] = {start_grid: None}
        cost: dict[GridPoint, float] = {start_grid: 0.0}
        while queue:
            _priority, current_cost, current = heapq.heappop(queue)
            if current_cost > cost.get(current, math.inf):
                continue
            if current == goal_grid:
                break
            for nxt in _neighbors(current):
                if not self._grid_is_walkable(nxt) or not self._edge_is_walkable(current, nxt):
                    continue
                new_cost = current_cost + 1
                if new_cost >= cost.get(nxt, math.inf):
                    continue
                cost[nxt] = new_cost
                priority = new_cost + abs(goal_grid[0] - nxt[0]) + abs(goal_grid[1] - nxt[1])
                heapq.heappush(queue, (priority, new_cost, nxt))
                came_from[nxt] = current
        if goal_grid not in came_from:
            raise ValueError("no collision-safe path found")

        grid_path: list[GridPoint] = []
        cursor: GridPoint | None = goal_grid
        while cursor is not None:
            grid_path.append(cursor)
            cursor = came_from[cursor]
        points = [_from_grid(point, self.nav) for point in reversed(grid_path)]

        # Keep authored coordinates when they are already safe and can connect
        # to the discrete route without clipping a rounded obstacle corner.
        safe_start = (
            start
            if self.contains(start) and self.segment_is_safe(start, points[0])
            else points[0]
        )
        safe_goal = (
            goal
            if self.contains(goal) and self.segment_is_safe(points[-1], goal)
            else points[-1]
        )
        if len(points) == 1:
            if safe_start.distance_to(safe_goal) <= 1e-6:
                return [safe_start]
            if not self.segment_is_safe(safe_start, safe_goal):
                raise ValueError("no collision-safe path found")
            return [safe_start, safe_goal]
        points[0] = safe_start
        points[-1] = safe_goal
        return self._string_pull(points)

    def _string_pull(self, points: list[Point2D]) -> list[Point2D]:
        """Remove grid stair-steps while retaining continuous free-space sightlines."""

        if len(points) <= 2:
            return points
        simplified = [points[0]]
        anchor = 0
        while anchor < len(points) - 1:
            candidate = len(points) - 1
            while candidate > anchor + 1:
                if self.segment_is_safe(points[anchor], points[candidate]):
                    break
                candidate -= 1
            simplified.append(points[candidate])
            anchor = candidate
        return simplified


def plan_path(
    model: FloorPlanModel,
    start: Point2D,
    goal: Point2D,
    camera_radius_m: float = DEFAULT_CAMERA_RADIUS_M,
) -> list[Point2D]:
    """Plan a continuous path clear of walls, furniture, and room boundaries."""

    return _RoutePlanner(model, camera_radius_m).path(start, goal)


def _safe_room_target(room: RoomPolygon, free_space: BaseGeometry) -> Point2D | None:
    room_polygon = Polygon([(point.x, point.y) for point in room.points])
    if not room_polygon.is_valid:
        room_polygon = room_polygon.buffer(0)
    if room_polygon.is_empty or room_polygon.area <= 0.05:
        return None
    candidates = _polygon_parts(room_polygon.intersection(free_space))
    if not candidates:
        return None
    labelled = [(polylabel(candidate, tolerance=0.03), candidate) for candidate in candidates]
    target, _polygon = max(
        labelled,
        key=lambda item: (item[0].distance(item[1].boundary), item[1].area),
    )
    return Point2D(x=float(target.x), y=float(target.y))


def _append_segment(
    waypoints: list[CameraWaypoint],
    segment: list[Point2D],
    terminal: CameraWaypoint | None = None,
) -> None:
    for point in segment[1:-1]:
        waypoints.append(CameraWaypoint(position=point))
    endpoint = segment[-1]
    if endpoint.distance_to(waypoints[-1].position) <= 1e-6:
        if terminal is not None:
            waypoints[-1] = terminal.model_copy(update={"position": endpoint})
        return
    if terminal is None:
        waypoints.append(CameraWaypoint(position=endpoint))
    else:
        waypoints.append(terminal.model_copy(update={"position": endpoint}))


def _route_manual_waypoints(
    model: FloorPlanModel,
    planner: _RoutePlanner,
) -> list[CameraWaypoint]:
    first = model.camera_waypoints[0]
    first_position = planner.snap(first.position)
    routed = [first.model_copy(update={"position": first_position})]
    for requested in model.camera_waypoints[1:]:
        try:
            segment = planner.path(routed[-1].position, requested.position)
        except ValueError:
            # A disconnected authored stop must never reintroduce a straight,
            # collision-unsafe leg. Later stops may still be reachable.
            continue
        _append_segment(routed, segment, terminal=requested)
    return routed


def _route_auto_waypoints(
    model: FloorPlanModel,
    planner: _RoutePlanner,
) -> list[CameraWaypoint]:
    targets = [
        target
        for room in model.rooms
        if (target := _safe_room_target(room, planner.free_space)) is not None
    ]
    if not targets:
        raise ValueError("camera waypoints or at least one collision-safe room are required")

    if model.entrance is not None:
        start = planner.snap(model.entrance)
        remaining = list(targets)
    else:
        start = targets[0]
        remaining = targets[1:]
    routed = [CameraWaypoint(position=start)]
    while remaining:
        remaining.sort(key=lambda target: routed[-1].position.distance_to(target))
        connected = False
        for index, target in enumerate(remaining):
            try:
                segment = planner.path(routed[-1].position, target)
            except ValueError:
                continue
            _append_segment(routed, segment)
            remaining.pop(index)
            connected = True
            break
        if not connected:
            break

    # A single room still gets a short orientation-changing tour, but only
    # along a segment proven to stay inside continuous free space.
    if len(routed) == 1:
        for distance in (0.75, 0.50, 0.30):
            candidates = [
                Point2D(x=start.x + dx, y=start.y + dy)
                for dx, dy in (
                    (distance, 0.0),
                    (-distance, 0.0),
                    (0.0, distance),
                    (0.0, -distance),
                )
            ]
            candidate = next(
                (
                    point
                    for point in candidates
                    if planner.contains(point) and planner.segment_is_safe(start, point)
                ),
                None,
            )
            if candidate is not None:
                routed.append(CameraWaypoint(position=candidate))
                break
    if len(routed) < 2:
        raise ValueError("no collision-safe walkthrough route could be generated")
    return routed


def camera_waypoints_for_model(
    model: FloorPlanModel,
    camera_radius_m: float = DEFAULT_CAMERA_RADIUS_M,
) -> list[CameraWaypoint]:
    """Return safe waypoints, preserving metadata on authored tour stops."""

    planner = _RoutePlanner(model, camera_radius_m)
    if model.camera_waypoints:
        return _route_manual_waypoints(model, planner)
    return _route_auto_waypoints(model, planner)


def manual_or_auto_waypoints(model: FloorPlanModel) -> list[Point2D]:
    """Backward-compatible point-only view of the collision-safe route."""

    return [waypoint.position for waypoint in camera_waypoints_for_model(model)]
