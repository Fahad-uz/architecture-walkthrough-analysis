from __future__ import annotations

import heapq
from collections.abc import Iterable

from architecture_walkthrough.geometry.models import FloorPlanModel, Point2D
from architecture_walkthrough.walkthrough.collision_detection import point_has_clearance
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


def plan_path(model: FloorPlanModel, start: Point2D, goal: Point2D, camera_radius_m: float = 0.25) -> list[Point2D]:
    nav = build_navigation_map(model)
    start_g = _to_grid(start, nav)
    goal_g = _to_grid(goal, nav)
    max_x = round((nav.max_x - nav.min_x) / nav.grid_size_m)
    max_y = round((nav.max_y - nav.min_y) / nav.grid_size_m)
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
            if not point_has_clearance(point, model.walls, camera_radius_m):
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
    current: GridPoint | None = goal_g
    while current is not None:
        path.append(_from_grid(current, nav))
        current = came_from[current]
    return list(reversed(path))


def manual_or_auto_waypoints(model: FloorPlanModel) -> list[Point2D]:
    if model.camera_waypoints:
        return [waypoint.position for waypoint in model.camera_waypoints]
    if model.entrance and model.rooms:
        target = model.rooms[0].points[0]
        return plan_path(model, model.entrance, target)
    raise ValueError("camera waypoints or entrance plus room targets are required")
