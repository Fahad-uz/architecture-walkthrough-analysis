from __future__ import annotations

import pytest

from architecture_walkthrough.geometry.cleanup import cleanup_walls, snap_value
from architecture_walkthrough.geometry.models import FloorPlanModel, Point2D, RoomPolygon, WallSegment
from architecture_walkthrough.geometry.scale import ScaleConverter, parse_dimension_pair
from architecture_walkthrough.geometry.validation import validate_floorplan


def test_scale_converter_requires_explicit_positive_scale() -> None:
    converter = ScaleConverter(pixels_per_metre=100)
    assert converter.px_to_m(250) == 2.5
    assert converter.m_to_px(2.5) == 250
    with pytest.raises(ValueError):
        ScaleConverter(pixels_per_metre=0)


def test_parse_dimension_pair_defaults_cm_and_rejects_unreasonable() -> None:
    assert parse_dimension_pair("599x340") == (5.99, 3.4, "m")
    assert parse_dimension_pair("3.2 x 4.1 m") == (3.2, 4.1, "m")
    with pytest.raises(ValueError):
        parse_dimension_pair("99999x2m")


def test_cleanup_walls_snaps_removes_short_and_deduplicates() -> None:
    walls = [
        WallSegment(start=Point2D(x=0.01, y=0), end=Point2D(x=1.02, y=0.01)),
        WallSegment(start=Point2D(x=1.0, y=0), end=Point2D(x=0, y=0)),
        WallSegment(start=Point2D(x=0, y=0), end=Point2D(x=0.05, y=0.05)),
    ]
    cleaned = cleanup_walls(walls, min_length_m=0.2, snap_grid_m=0.05)
    assert len(cleaned) == 1
    assert cleaned[0].start == Point2D(x=0, y=0)
    assert cleaned[0].end == Point2D(x=1.0, y=0)
    assert snap_value(0.12, 0.05) == 0.1


def test_geometry_validation_reports_invalid_room() -> None:
    model = FloorPlanModel(
        walls=[WallSegment(start=Point2D(x=0, y=0), end=Point2D(x=1, y=0))],
        rooms=[RoomPolygon(points=[Point2D(x=0, y=0), Point2D(x=1, y=1), Point2D(x=1, y=0), Point2D(x=0, y=1)])],
    )
    assert validate_floorplan(model)
