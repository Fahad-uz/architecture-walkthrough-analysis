from __future__ import annotations

from architecture_walkthrough.geometry.models import FloorPlanModel, Point2D, WallSegment
from architecture_walkthrough.walkthrough.collision_detection import path_collides
from architecture_walkthrough.walkthrough.path_planner import plan_path


def test_collision_detection_flags_wall_crossing() -> None:
    wall = WallSegment(start=Point2D(x=1, y=-1), end=Point2D(x=1, y=1))
    assert path_collides([Point2D(x=0, y=0), Point2D(x=2, y=0)], [wall], radius_m=0.1)
    assert not path_collides([Point2D(x=0, y=2), Point2D(x=2, y=2)], [wall], radius_m=0.1)


def test_path_planner_finds_path_without_obstacles() -> None:
    model = FloorPlanModel()
    path = plan_path(model, Point2D(x=0, y=0), Point2D(x=1, y=0), camera_radius_m=0.1)
    assert path[0].distance_to(Point2D(x=0, y=0)) < 0.3
    assert path[-1].distance_to(Point2D(x=1, y=0)) < 0.3
