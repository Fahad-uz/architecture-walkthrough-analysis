from __future__ import annotations

import heapq
from collections.abc import Iterable

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import polylabel

from architecture_walkthrough.geometry.models import FloorPlanModel, Point2D
from architecture_walkthrough.walkthrough.navigation_map import NavigationMap, build_navigation_map


GridPoint = tuple[int, int]


def _to_grid(point: Point2D, nav: NavigationMap) -> GridPoint:
    return (round((point.x - nav.min_x) / nav.grid_size_m), round((point.y - nav.min_y) / nav.grid_size_m))


def _from_grid(point: GridPoint, nav: NavigationMap) -> Point2D:
    return Point2D(x=nav.min_x + point[0] * nav.grid_size_m, y=nav.min_y + point[1] * nav.grid_size_m)


def _neighbors(point: GridPoint) -> Iterable[GridPoint]:
    x, y = point
    yield x + 1, y
    yield x - 1, y
    yield x, y + 1
    yield x, y - 1


def _walkable_wall_lines(model: FloorPlanModel) -> list[LineString]:
    """Wall obstacles with confirmed doorway intervals removed."""

    doors_by_wall: dict[str, list[tuple[float, float]]] = {}
    for door in model.doors:
        if not door.wall_id:
            continue
        interval = door.interval(door.width_m)
        if interval is not None:
            doors_by_wall.setdefault(door.wall_id, []).append(interval)

    obstacles: list[LineString] = []
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
                    LineString(
                        [
                            (wall.start.x + ux * cursor, wall.start.y + uy * cursor),
                            (wall.start.x + ux * start, wall.start.y + uy * start),
                        ]
                    )
                )
            cursor = max(cursor, end)
        if cursor < length - 1e-6:
            obstacles.append(
                LineString(
                    [
                        (wall.start.x + ux * cursor, wall.start.y + uy * cursor),
                        (wall.end.x, wall.end.y),
                    ]
                )
            )
    return obstacles


def _has_clearance(point: Point2D, obstacles: list[LineString], radius_m: float) -> bool:
    candidate = Point(point.x, point.y)
    return all(candidate.distance(obstacle) >= radius_m for obstacle in obstacles)


def _nearest_clear_grid(
    requested: GridPoint,
    nav: NavigationMap,
    obstacles: list[LineString],
    radius_m: float,
    max_x: int,
    max_y: int,
) -> GridPoint:
    queue = [requested]
    visited = {requested}
    while queue:
        current = queue.pop(0)
        if (
            0 <= current[0] <= max_x
            and 0 <= current[1] <= max_y
            and _has_clearance(_from_grid(current, nav), obstacles, radius_m)
        ):
            return current
        for neighbor in _neighbors(current):
            if neighbor not in visited and abs(neighbor[0] - requested[0]) + abs(neighbor[1] - requested[1]) <= 12:
                visited.add(neighbor)
                queue.append(neighbor)
    raise ValueError("no collision-safe point near requested waypoint")


def plan_path(model: FloorPlanModel, start: Point2D, goal: Point2D, camera_radius_m: float = 0.25) -> list[Point2D]:
    nav = build_navigation_map(model)
    max_x = round((nav.max_x - nav.min_x) / nav.grid_size_m)
    max_y = round((nav.max_y - nav.min_y) / nav.grid_size_m)
    obstacles = _walkable_wall_lines(model)
    start_g = _nearest_clear_grid(_to_grid(start, nav), nav, obstacles, camera_radius_m, max_x, max_y)
    goal_g = _nearest_clear_grid(_to_grid(goal, nav), nav, obstacles, camera_radius_m, max_x, max_y)
    queue: list[tuple[float, GridPoint]] = [(0, start_g)]
    came_from: dict[GridPoint, GridPoint | None] = {start_g: None}
    cost: dict[GridPoint, float] = {start_g: 0}
    while queue:
        _, current = heapq.heappop(queue)
        if current == goal_g:
            break
        for nxt in _neighbors(current):
            if nxt[0] < 0 or nxt[1] < 0 or nxt[0] > max_x or nxt[1] > max_y:
                continue
            point = _from_grid(nxt, nav)
            if not _has_clearance(point, obstacles, camera_radius_m):
                continue
            new_cost = cost[current] + 1
            if nxt not in cost or new_cost < cost[nxt]:
                cost[nxt] = new_cost
                priority = new_cost + abs(goal_g[0] - nxt[0]) + abs(goal_g[1] - nxt[1])
                heapq.heappush(queue, (priority, nxt))
                came_from[nxt] = current
    if goal_g not in came_from:
        raise ValueError("no collision-safe path found")
    path: list[Point2D] = []
    cursor: GridPoint | None = goal_g
    while cursor is not None:
        path.append(_from_grid(cursor, nav))
        cursor = came_from[cursor]
    return list(reversed(path))


def _room_target(points: list[Point2D]) -> Point2D | None:
    polygon = Polygon([(point.x, point.y) for point in points])
    if not polygon.is_valid or polygon.area <= 0.05:
        return None
    target = polylabel(polygon, tolerance=0.05)
    return Point2D(x=float(target.x), y=float(target.y))


def manual_or_auto_waypoints(model: FloorPlanModel) -> list[Point2D]:
    if model.camera_waypoints:
        return [waypoint.position for waypoint in model.camera_waypoints]
    targets = [target for room in model.rooms if (target := _room_target(room.points)) is not None]
    if not targets:
        raise ValueError("camera waypoints or at least one valid room polygon are required")

    start = model.entrance or targets.pop(0)
    route = [start]
    remaining = list(targets)
    while remaining:
        remaining.sort(key=lambda target: route[-1].distance_to(target))
        connected = False
        for index, target in enumerate(remaining):
            try:
                segment = plan_path(model, route[-1], target)
            except ValueError:
                continue
            route.extend(segment[1:])
            remaining.pop(index)
            connected = True
            break
        if not connected:
            break

    # A single room still gets a short, safe orientation-changing tour.
    if len(route) == 1:
        nav = build_navigation_map(model)
        obstacles = _walkable_wall_lines(model)
        candidates = [
            Point2D(x=route[0].x + dx, y=route[0].y + dy)
            for dx, dy in ((0.75, 0.0), (-0.75, 0.0), (0.0, 0.75), (0.0, -0.75))
        ]
        for candidate in candidates:
            if (
                nav.min_x <= candidate.x <= nav.max_x
                and nav.min_y <= candidate.y <= nav.max_y
                and _has_clearance(candidate, obstacles, 0.25)
            ):
                route.append(candidate)
                break
    if len(route) < 2:
        raise ValueError("no collision-safe walkthrough route could be generated")
    return route
