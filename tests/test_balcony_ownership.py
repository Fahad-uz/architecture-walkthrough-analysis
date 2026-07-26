from __future__ import annotations

from shapely.geometry import Polygon

from architecture_walkthrough.geometry.balcony_ownership import (
    reconcile_room_balcony_ownership,
)
from architecture_walkthrough.geometry.models import (
    BalconyPolygon,
    Point2D,
    RoomPolygon,
)


def _points(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> list[Point2D]:
    return [
        Point2D(x=x0, y=y0),
        Point2D(x=x1, y=y0),
        Point2D(x=x1, y=y1),
        Point2D(x=x0, y=y1),
    ]


def _polygon(region: RoomPolygon) -> Polygon:
    return Polygon([(point.x, point.y) for point in region.points])


def test_topology_balcony_face_replaces_oversized_patterned_geometry() -> None:
    balcony_face = RoomPolygon(
        id="room_balcony",
        face_id="face_balcony",
        name="BALCONY",
        points=_points(0, 4, 3, 5),
        evidence_source="wall_graph_face",
    )
    bedroom = RoomPolygon(
        id="bedroom",
        name="BEDROOM",
        points=_points(3, 3, 6, 5),
    )
    patterned = BalconyPolygon(
        id="balcony_000",
        name="BALCONY",
        points=_points(-0.2, 3.9, 3.4, 5.1),
        evidence_source="patterned_region+ocr_balcony_label",
    )

    result = reconcile_room_balcony_ownership(
        [balcony_face, bedroom],
        [patterned],
    )

    assert [room.id for room in result.rooms] == ["bedroom"]
    assert len(result.balconies) == 1
    assert result.balconies[0].id == "balcony_000"
    assert result.balconies[0].face_id == "face_balcony"
    assert result.balconies[0].points == balcony_face.points
    assert "wall_graph_face" in result.balconies[0].evidence_source
    assert result.matched_topology_faces == 1
    assert _polygon(result.balconies[0]).intersection(
        _polygon(bedroom)
    ).area == 0


def test_boundary_balcony_is_subtracted_from_one_dominant_room() -> None:
    living = RoomPolygon(
        id="living",
        name="LIVING",
        points=_points(0, 0, 6, 4),
        evidence_source="wall_graph_face",
    )
    balcony = BalconyPolygon(
        id="balcony",
        name="BALCONY",
        points=_points(2, -0.1, 5, 1),
    )

    result = reconcile_room_balcony_ownership([living], [balcony])

    assert len(result.rooms) == 1
    assert result.subtracted_room_count == 1
    assert _polygon(result.rooms[0]).area == 21
    assert _polygon(result.rooms[0]).intersection(
        _polygon(result.balconies[0])
    ).area == 0
    assert "balcony_subtracted" in result.rooms[0].evidence_source


def test_interior_balcony_does_not_cut_an_invented_hole() -> None:
    living = RoomPolygon(
        id="living",
        name="LIVING",
        points=_points(0, 0, 6, 4),
    )
    interior_balcony = BalconyPolygon(
        id="balcony",
        name="BALCONY",
        points=_points(2, 1, 4, 2),
    )

    result = reconcile_room_balcony_ownership(
        [living],
        [interior_balcony],
    )

    assert result.rooms[0].points == living.points
    assert result.subtracted_room_count == 0


def test_unmatched_named_terrace_face_is_promoted_without_changing_open_plan() -> None:
    living = RoomPolygon(
        id="living",
        name="LIVING",
        points=_points(0, 0, 6, 4),
    )
    terrace = RoomPolygon(
        id="terrace",
        face_id="face_terrace",
        name="Roof Terrace",
        points=_points(0, 4, 6, 5),
    )

    result = reconcile_room_balcony_ownership(
        [living, terrace],
        [],
    )

    assert result.rooms == [living]
    assert len(result.balconies) == 1
    assert result.balconies[0].name == "Roof Terrace"
    assert result.balconies[0].face_id == "face_terrace"
    assert result.matched_topology_faces == 0
