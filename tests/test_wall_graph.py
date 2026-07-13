from __future__ import annotations

import pytest

from architecture_walkthrough.geometry.models import DoorOpening, Point2D, RoomPolygon, WallSegment
from architecture_walkthrough.geometry.wall_graph import (
    enumerate_faces,
    match_faces_to_rooms,
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
