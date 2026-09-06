from __future__ import annotations

import pytest

from architecture_walkthrough.geometry.models import DoorOpening, Point2D, RoomPolygon, WallSegment
from architecture_walkthrough.geometry.wall_graph import (
    collinear_gaps,
    enumerate_faces,
    match_faces_to_rooms,
    perpendicular_endpoint_gaps,
    snap_endpoints_to_walls,
)


def _wall(wall_id: str, x1: float, y1: float, x2: float, y2: float, thickness: float = 0.12) -> WallSegment:
    return WallSegment(id=wall_id, start=Point2D(x=x1, y=y1), end=Point2D(x=x2, y=y2), thickness_m=thickness)


def _square(size: float = 4.0) -> list[WallSegment]:
    return [
        _wall("w0", 0, 0, size, 0),
        _wall("w1", size, 0, size, size),
        _wall("w2", size, size, 0, size),
        _wall("w3", 0, size, 0, 0),
    ]


def test_square_produces_single_face() -> None:
    result = enumerate_faces(_square())
    assert len(result.faces) == 1
    assert result.faces[0].polygon.area == pytest.approx(16.0)


def test_divider_wall_splits_into_two_faces() -> None:
    walls = [*_square(), _wall("w4", 2, 0, 2, 4)]
    result = enumerate_faces(walls)
    assert len(result.faces) == 2
    assert sum(face.polygon.area for face in result.faces) == pytest.approx(16.0)


def test_t_junction_endpoint_sloppiness_is_snapped() -> None:
    # Divider stops 15 cm short of the top wall; junction snapping repairs it.
    walls = [*_square(), _wall("w4", 2, 0, 2, 3.85)]
    result = enumerate_faces(walls, junction_snap_m=0.2)
    assert len(result.faces) == 2


def test_reciprocal_endpoint_snaps_node_on_a_stable_precision_grid() -> None:
    # Scale conversion and projection can leave two mutually snapped endpoints
    # a few floating-point ulps apart. GEOS then sees an open loop even though
    # both points describe the same measured junction.
    bottom_y = 4.765941533279317
    right_x = 5.174885080427878
    walls = [
        _wall("bottom", 0, bottom_y, 5.302585029690035, bottom_y),
        _wall("left", 0, bottom_y, 0, 8),
        _wall("top", 0, 8, right_x, 8),
        _wall("right", right_x, 4.765942, right_x, 8),
    ]

    result = enumerate_faces(
        walls,
        junction_snap_m=0.2,
        min_room_area_m2=0.1,
    )

    assert len(result.faces) == 1
    assert result.faces[0].polygon.area == pytest.approx(
        right_x * (8 - bottom_y),
        abs=1e-5,
    )


def test_doorway_gap_stays_open_without_confirmed_opening() -> None:
    # Divider has a 0.9 m doorway gap; without an opening the rooms must merge
    # (single face) rather than being closed by an invented wall.
    walls = [
        *_square(),
        _wall("w4", 2, 0, 2, 1.5),
        _wall("w5", 2, 2.4, 2, 4),
    ]
    result = enumerate_faces(walls)
    assert len(result.faces) == 1
    assert len(result.unclosed_gaps) == 1
    assert result.unclosed_gaps[0]["length_m"] == pytest.approx(0.9)


def test_unconfirmed_door_sized_gap_can_bridge_topology_without_inventing_opening() -> None:
    walls = [
        *_square(),
        _wall("w4", 2, 0, 2, 1.5),
        _wall("w5", 2, 2.4, 2, 4),
    ]

    result = enumerate_faces(walls, unconfirmed_opening_range_m=(0.55, 1.40))

    assert len(result.faces) == 2
    assert len(result.closures) == 1
    assert result.unclosed_gaps[0]["topology_bridge"] is True


def test_perpendicular_door_sized_corner_gap_can_bridge_topology() -> None:
    walls = [
        _wall("bottom", 0, 0, 4, 0),
        _wall("left", 0, 0, 0, 4),
        _wall("top", 0, 4, 4, 4),
        _wall("right", 4, 0, 4, 3.1),
    ]

    gaps = perpendicular_endpoint_gaps(walls, 0.55, 1.40)
    result = enumerate_faces(walls, unconfirmed_opening_range_m=(0.55, 1.40))

    assert len(gaps) == 1
    assert gaps[0]["length_m"] == pytest.approx(0.9)
    assert len(result.faces) == 1
    assert result.unclosed_gaps[0]["corner_bridge"] is True


def test_collinear_gap_grouping_sorts_spans_after_coordinate_matching() -> None:
    # Slightly offset detector centerlines used to sort by x first, leaving the
    # upper segment before the lower segment and silently missing their gap.
    walls = [
        _wall("upper", 0.0, 5.0, 0.0, 10.0),
        _wall("lower", 0.17, 0.0, 0.17, 4.0),
    ]

    gaps = collinear_gaps(walls, coord_tol=0.18)

    assert len(gaps) == 1
    assert gaps[0]["length_m"] == pytest.approx(1.0)


def test_doorway_gap_closes_when_opening_confirms_it() -> None:
    walls = [
        *_square(),
        _wall("w4", 2, 0, 2, 1.5),
        _wall("w5", 2, 2.4, 2, 4),
    ]
    door = DoorOpening(
        id="door_0",
        center=Point2D(x=2, y=1.95),
        wall_id="w4",
        start_offset_m=1.5,
        end_offset_m=2.4,
        width_m=0.9,
    )
    result = enumerate_faces(walls, doors=[door])
    assert len(result.faces) == 2
    assert not result.unclosed_gaps


def test_multi_metre_gap_never_auto_closes() -> None:
    walls = [
        *_square(8.0),
        _wall("w4", 4, 0, 4, 1.0),
        _wall("w5", 4, 4.0, 4, 8.0),
    ]
    result = enumerate_faces(walls)
    assert len(result.faces) == 1
    assert result.unclosed_gaps and result.unclosed_gaps[0]["length_m"] == pytest.approx(3.0)


def test_faces_have_stable_ids_and_slivers_are_dropped() -> None:
    walls = [*_square(), _wall("w4", 2, 0, 2, 4), _wall("w5", 2.05, 0, 2.05, 4)]
    result = enumerate_faces(walls)
    # The 5 cm sliver between the doubled divider must not become a room.
    assert len(result.faces) == 2
    assert len({face.face_id for face in result.faces}) == 2


def test_snap_endpoints_moves_only_near_misses() -> None:
    walls = [_wall("w0", 0, 0, 4, 0), _wall("w1", 2, 0.1, 2, 3)]
    snapped = snap_endpoints_to_walls(walls, tolerance_m=0.2)
    assert snapped[1].start.y == pytest.approx(0.0)
    far = [_wall("w0", 0, 0, 4, 0), _wall("w1", 2, 1.0, 2, 3)]
    unchanged = snap_endpoints_to_walls(far, tolerance_m=0.2)
    assert unchanged[1].start.y == pytest.approx(1.0)


def test_room_semantics_survive_regeneration() -> None:
    walls = [*_square(), _wall("w4", 2, 0, 2, 4)]
    faces = enumerate_faces(walls).faces
    previous = [
        RoomPolygon(
            id="room_000",
            face_id="face_keep_me",
            name="Kitchen",
            points=[Point2D(x=0.1, y=0.1), Point2D(x=1.9, y=0.1), Point2D(x=1.9, y=3.9), Point2D(x=0.1, y=3.9)],
            confidence=0.9,
        )
    ]
    rooms = match_faces_to_rooms(faces, previous)
    assert len(rooms) == 2
    kitchen = next(room for room in rooms if room.name == "Kitchen")
    assert kitchen.face_id == "face_keep_me"
    assert kitchen.evidence_source == "wall_graph_face"
    other = next(room for room in rooms if room.name != "Kitchen")
    assert other.face_id and other.face_id != "face_keep_me"


def test_manual_room_without_matching_face_is_dropped() -> None:
    faces = enumerate_faces(_square()).faces
    previous = [
        RoomPolygon(
            id="ghost",
            name="Ghost Room",
            points=[Point2D(x=10, y=10), Point2D(x=12, y=10), Point2D(x=12, y=12), Point2D(x=10, y=12)],
        )
    ]
    rooms = match_faces_to_rooms(faces, previous)
    assert len(rooms) == 1
    assert rooms[0].name is None


def test_new_face_ids_do_not_collide_with_later_preserved_room_ids() -> None:
    # Faces are area-sorted: the larger right face is visited first and is new,
    # while the smaller left face preserves room_000.  The old implementation
    # assigned room_000 to the first face before discovering the collision.
    walls = [*_square(), _wall("w4", 1.5, 0, 1.5, 4)]
    faces = enumerate_faces(walls).faces
    previous = [
        RoomPolygon(
            id="room_000",
            face_id="left_stable_face",
            name="Bedroom",
            points=[Point2D(x=0.1, y=0.1), Point2D(x=1.4, y=0.1), Point2D(x=1.4, y=3.9), Point2D(x=0.1, y=3.9)],
        )
    ]

    rooms = match_faces_to_rooms(faces, previous)

    assert len({room.id for room in rooms}) == 2
    bedroom = next(room for room in rooms if room.name == "Bedroom")
    other = next(room for room in rooms if room.name != "Bedroom")
    assert bedroom.id == "room_000"
    assert other.id == "room_001"

    regenerated = match_faces_to_rooms(faces, rooms)
    assert {room.id for room in regenerated} == {"room_000", "room_001"}


def test_legacy_duplicate_room_ids_are_repaired_on_match() -> None:
    walls = [*_square(), _wall("w4", 2, 0, 2, 4)]
    faces = enumerate_faces(walls).faces
    previous = [
        RoomPolygon(
            id="room_000",
            name="Left",
            points=[Point2D(x=0.1, y=0.1), Point2D(x=1.9, y=0.1), Point2D(x=1.9, y=3.9), Point2D(x=0.1, y=3.9)],
        ),
        RoomPolygon(
            id="room_000",
            name="Right",
            points=[Point2D(x=2.1, y=0.1), Point2D(x=3.9, y=0.1), Point2D(x=3.9, y=3.9), Point2D(x=2.1, y=3.9)],
        ),
    ]

    rooms = match_faces_to_rooms(faces, previous)

    assert len(rooms) == 2
    assert len({room.id for room in rooms}) == 2
    assert {room.name for room in rooms} == {"Left", "Right"}
