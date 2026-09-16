from __future__ import annotations

import pytest
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from architecture_walkthrough.geometry.fixture_layout import fixture_furniture_from_regions
from architecture_walkthrough.geometry.models import (
    FurniturePlacement,
    Point2D,
    RoomPolygon,
    WallSegment,
)
from architecture_walkthrough.vision.wall_detection import FixtureDetailRegion


def _room(name: str | None, width: float = 4, depth: float = 3) -> RoomPolygon:
    return RoomPolygon(name=name, points=[
        Point2D(x=0, y=0), Point2D(x=width, y=0),
        Point2D(x=width, y=depth), Point2D(x=0, y=depth),
    ])


def _footprint(item: FurniturePlacement) -> Polygon:
    return translate(rotate(box(-item.width_m / 2, -item.depth_m / 2,
        item.width_m / 2, item.depth_m / 2), item.rotation_deg, origin=(0, 0)),
        item.center.x, item.center.y)


def _region(points) -> FixtureDetailRegion:
    return FixtureDetailRegion(polygon=tuple(points), front_band_ids=("front",), max_depth_px=60)


def test_l_shaped_counter_preserves_open_floor_and_does_not_overlap_itself() -> None:
    region = _region([(0, 0), (400, 0), (400, 300), (340, 300), (340, 60), (0, 60)])
    placements = fixture_furniture_from_regions([region], [_room("Kitchen")], 100, 300)

    assert len(placements) == 2
    assert all(item.category == "kitchen_counter" and item.height_m == 0.9 for item in placements)
    rectangles = [_footprint(item) for item in placements]
    combined = unary_union(rectangles)
    expected = Polygon([(x / 100, (300 - y) / 100) for x, y in region.polygon]).buffer(
        -0.07, join_style=2,
    )
    assert combined.symmetric_difference(expected).area < 1e-8
    assert rectangles[0].intersection(rectangles[1]).area == pytest.approx(0)
    assert combined.intersection(box(0.5, 0.5, 3, 2)).is_empty


def test_bedroom_fixture_converts_pixels_and_y_direction_without_relocation() -> None:
    region = _region([(300, 50), (350, 50), (350, 200), (300, 200)])
    placements = fixture_furniture_from_regions([region], [_room("Master Bedroom")], 100, 300)

    assert len(placements) == 1
    wardrobe = placements[0]
    assert wardrobe.category == "wardrobe"
    assert wardrobe.height_m == 2.4
    assert (wardrobe.center.x, wardrobe.center.y) == pytest.approx((3.25, 1.75))
    assert (wardrobe.width_m, wardrobe.depth_m) == pytest.approx((1.36, 0.36))
    assert wardrobe.rotation_deg == -90


def test_fixture_clips_out_structural_wall_and_duplicate_regions() -> None:
    region = _region([(0, 0), (400, 0), (400, 60), (0, 60)])
    wall = WallSegment(
        start=Point2D(x=2, y=2), end=Point2D(x=2, y=3), thickness_m=0.2,
    )
    placements = fixture_furniture_from_regions(
        [region, region], [_room("Kitchen")], 100, 300, walls=[wall],
    )

    assert len(placements) == 2
    rectangles = [_footprint(item) for item in placements]
    assert rectangles[0].intersection(rectangles[1]).area == pytest.approx(0)
    assert all(rect.intersection(box(1.83, 2, 2.17, 3)).area < 1e-8 for rect in rectangles)
    assert unary_union(rectangles).bounds == pytest.approx((0.07, 2.47, 3.93, 2.93))


@pytest.mark.parametrize("name", [None, "Room 1", "Study room", "Living and dining"])
def test_unknown_fixture_semantics_are_not_invented(name) -> None:
    region = _region([(0, 0), (400, 0), (400, 60), (0, 60)])

    assert fixture_furniture_from_regions([region], [_room(name)], 100, 300) == []


def test_broad_or_outside_regions_are_not_moved_into_the_room() -> None:
    broad = _region([(0, 0), (400, 0), (400, 300), (0, 300)])
    outside = _region([(600, 0), (1000, 0), (1000, 60), (600, 60)])

    assert fixture_furniture_from_regions([broad, outside], [_room("Kitchen")], 100, 300) == []


@pytest.mark.parametrize("scale", [0, -1, float("inf"), float("nan")])
def test_invalid_scale_is_rejected(scale) -> None:
    with pytest.raises(ValueError, match="pixels_per_metre"):
        fixture_furniture_from_regions([], [], scale, 300)
