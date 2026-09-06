from __future__ import annotations

from shapely.affinity import rotate, translate
from shapely.geometry import LineString, Polygon, box

from architecture_walkthrough.geometry.models import (
    CameraWaypoint,
    DoorOpening,
    FloorPlanModel,
    FurniturePlacement,
    Point2D,
    RoomPolygon,
    WallSegment,
)
from architecture_walkthrough.walkthrough.collision_detection import path_collides
from architecture_walkthrough.walkthrough.path_planner import (
    DOOR_LEAF_OPEN_DEG,
    camera_waypoints_for_model,
    plan_path,
)


def _room(width: float = 6, depth: float = 4) -> RoomPolygon:
    return RoomPolygon(
        points=[
            Point2D(x=0, y=0),
            Point2D(x=width, y=0),
            Point2D(x=width, y=depth),
            Point2D(x=0, y=depth),
        ]
    )


def _route_line(points: list[Point2D]) -> LineString:
    return LineString([(point.x, point.y) for point in points])


def _furniture_footprint(item: FurniturePlacement) -> Polygon:
    footprint = box(-item.width_m / 2, -item.depth_m / 2, item.width_m / 2, item.depth_m / 2)
    footprint = rotate(footprint, item.rotation_deg, origin=(0, 0), use_radians=False)
    return translate(footprint, item.center.x, item.center.y)


def test_collision_detection_flags_wall_crossing() -> None:
    wall = WallSegment(start=Point2D(x=1, y=-1), end=Point2D(x=1, y=1))
    assert path_collides([Point2D(x=0, y=0), Point2D(x=2, y=0)], [wall], radius_m=0.1)
    assert not path_collides([Point2D(x=0, y=2), Point2D(x=2, y=2)], [wall], radius_m=0.1)


def test_path_planner_finds_path_without_obstacles() -> None:
    model = FloorPlanModel()
    path = plan_path(model, Point2D(x=0, y=0), Point2D(x=1, y=0), camera_radius_m=0.1)
    assert path[0].distance_to(Point2D(x=0, y=0)) < 0.3
    assert path[-1].distance_to(Point2D(x=1, y=0)) < 0.3


def test_path_planner_uses_confirmed_door_but_not_solid_wall() -> None:
    wall = WallSegment(
        id="divider",
        start=Point2D(x=3, y=0),
        end=Point2D(x=3, y=4),
        thickness_m=0.2,
    )
    start = Point2D(x=1, y=2)
    goal = Point2D(x=5, y=2)
    blocked = FloorPlanModel(rooms=[_room()], walls=[wall])
    try:
        plan_path(blocked, start, goal)
    except ValueError as error:
        assert str(error) == "no collision-safe path found"
    else:
        raise AssertionError("a room-spanning solid wall must split the walkable space")

    door = DoorOpening(
        wall_id="divider",
        center=Point2D(x=3, y=2),
        width_m=1,
        start_offset_m=1.5,
        end_offset_m=2.5,
    )
    opened = blocked.model_copy(update={"doors": [door]})
    path = plan_path(opened, start, goal)

    assert path[0] == start
    assert path[-1] == goal
    assert all(1.5 < point.y < 2.5 for point in path if 2.9 <= point.x <= 3.1)


def test_path_planner_avoids_fully_open_door_leaf() -> None:
    wall = WallSegment(
        id="divider",
        start=Point2D(x=3, y=0),
        end=Point2D(x=3, y=4),
        thickness_m=0.2,
    )
    door = DoorOpening(
        wall_id="divider",
        center=Point2D(x=3, y=2),
        width_m=1,
        start_offset_m=1.5,
        end_offset_m=2.5,
        hinge_side="start",
        swing_side="left",
    )
    model = FloorPlanModel(rooms=[_room()], walls=[wall], doors=[door])
    start = Point2D(x=1.2, y=1.5)
    goal = Point2D(x=2.8, y=1.9)

    path = plan_path(model, start, goal)
    leaf = box(2.02, 1.5 - 0.045 / 2, 2.98, 1.5 + 0.045 / 2)

    assert DOOR_LEAF_OPEN_DEG == 90.0
    assert len(path) >= 3
    assert _route_line(path).distance(leaf) >= 0.329


def test_path_planner_avoids_rotated_furniture_and_room_edges() -> None:
    furniture = FurniturePlacement(
        category="dining_table",
        center=Point2D(x=3, y=2),
        width_m=1.8,
        depth_m=0.9,
        rotation_deg=32,
    )
    model = FloorPlanModel(rooms=[_room()], furniture=[furniture])
    path = plan_path(model, Point2D(x=1, y=2), Point2D(x=5, y=2))
    route = _route_line(path)

    assert len(path) >= 3
    assert route.distance(_furniture_footprint(furniture)) >= 0.329
    assert route.distance(Polygon([(0, 0), (6, 0), (6, 4), (0, 4)]).boundary) >= 0.329


def test_safe_manual_route_preserves_stop_metadata() -> None:
    furniture = FurniturePlacement(
        category="sofa",
        center=Point2D(x=3, y=2),
        width_m=1.4,
        depth_m=0.8,
        rotation_deg=20,
    )
    first = CameraWaypoint(
        position=Point2D(x=3, y=2),
        look_at=Point2D(x=3, y=3),
        pause_seconds=0.75,
    )
    last = CameraWaypoint(
        position=Point2D(x=5, y=2),
        look_at=Point2D(x=4, y=3),
        pause_seconds=1.25,
    )
    model = FloorPlanModel(
        rooms=[_room()],
        furniture=[furniture],
        camera_waypoints=[first, last],
    )

    routed = camera_waypoints_for_model(model)
    points = [waypoint.position for waypoint in routed]

    assert routed[0].position != first.position
    assert routed[0].look_at == first.look_at
    assert routed[0].pause_seconds == first.pause_seconds
    assert routed[-1].position == last.position
    assert routed[-1].look_at == last.look_at
    assert routed[-1].pause_seconds == last.pause_seconds
    assert all(waypoint.look_at is None for waypoint in routed[1:-1])
    assert all(waypoint.pause_seconds == 0 for waypoint in routed[1:-1])
    assert _route_line(points).distance(_furniture_footprint(furniture)) >= 0.329


def test_auto_route_chooses_free_room_target_instead_of_furniture_center() -> None:
    furniture = FurniturePlacement(
        category="bed",
        center=Point2D(x=3, y=2),
        width_m=1.8,
        depth_m=1.4,
    )
    model = FloorPlanModel(rooms=[_room()], furniture=[furniture])

    waypoints = camera_waypoints_for_model(model)
    route = _route_line([waypoint.position for waypoint in waypoints])

    assert len(waypoints) >= 2
    assert route.distance(_furniture_footprint(furniture)) >= 0.329


def test_only_floor_level_placement_categories_are_walkable() -> None:
    lamp = FurniturePlacement(
        category="lamp",
        center=Point2D(x=3, y=2),
        width_m=1.4,
        depth_m=0.8,
    )
    rug = lamp.model_copy(update={"category": "area_rug"})
    start = Point2D(x=1, y=2)
    goal = Point2D(x=5, y=2)

    lamp_path = plan_path(FloorPlanModel(rooms=[_room()], furniture=[lamp]), start, goal)
    rug_path = plan_path(FloorPlanModel(rooms=[_room()], furniture=[rug]), start, goal)

    assert _route_line(lamp_path).distance(_furniture_footprint(lamp)) >= 0.329
    assert rug_path == [start, goal]
