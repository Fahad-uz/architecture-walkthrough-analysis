from __future__ import annotations

import pytest

from architecture_walkthrough.geometry.models import ArchitecturalElement, Point2D, RoomPolygon
from architecture_walkthrough.geometry.stair_voids import infer_stair_ceiling_voids


def _points(x: float, y: float, width: float, height: float) -> list[Point2D]:
    return [
        Point2D(x=x, y=y), Point2D(x=x + width, y=y),
        Point2D(x=x + width, y=y + height), Point2D(x=x, y=y + height),
    ]


def _room(name: str | None = None) -> RoomPolygon:
    return RoomPolygon(id="stair_room", name=name, points=_points(0, 0, 4.4, 2.4))


def _stair() -> ArchitecturalElement:
    return ArchitecturalElement(
        id="stairs", kind="staircase", polygon=_points(0.9, 0.1, 2.45, 2.2),
        confidence=0.88, evidence_source="repetitive_parallel_treads",
    )


@pytest.mark.parametrize("name", [None, "Stair hall", "Staircase"])
def test_measured_stair_opens_only_its_enclosing_room_ceiling(name: str | None) -> None:
    room = _room(name)
    living = RoomPolygon(id="living", name="Living", points=_points(5, 0, 8, 6))
    result = infer_stair_ceiling_voids([room, living], [_stair()])

    assert result.room_ids == ["stair_room"]
    assert result.inferences[0]["stair_containment"] == pytest.approx(1)
    assert result.inferences[0]["room_coverage"] == pytest.approx(5.39 / 10.56)
    assert "unknown" in result.inferences[0]["assumption"]
    assert result.issues[0].code == "inferred_stair_ceiling_void"
    assert result.issues[0].severity == "warning"


def test_stair_in_large_unnamed_room_does_not_remove_room_ceiling() -> None:
    large_room = _room().model_copy(update={"points": _points(0, 0, 8, 6)})
    assert infer_stair_ceiling_voids([large_room], [_stair()]).room_ids == []


def test_stair_cannot_remove_a_named_habitable_room_ceiling() -> None:
    assert infer_stair_ceiling_voids([_room("Living and Dining")], [_stair()]).room_ids == []


def test_stair_crossing_room_boundary_is_not_a_contained_stairwell() -> None:
    stair = _stair().model_copy(update={"polygon": _points(-0.3, 0.1, 3, 2.2)})
    assert infer_stair_ceiling_voids([_room()], [stair]).room_ids == []


@pytest.mark.parametrize("update", [
    {"evidence_source": "gemini"},
    {"confidence": 0.79},
    {"kind": "lift"},
    {"polygon": []},
    {"polygon": [Point2D(x=0, y=0), Point2D(x=2, y=2), Point2D(x=0, y=2), Point2D(x=2, y=0)]},
])
def test_unmeasured_or_invalid_stair_does_not_open_ceiling(update: dict) -> None:
    assert infer_stair_ceiling_voids([_room()], [_stair().model_copy(update=update)]).room_ids == []


def test_duplicate_overlapping_room_faces_do_not_claim_the_stair() -> None:
    room = _room()
    duplicate = room.model_copy(update={"id": "duplicate"})
    assert infer_stair_ceiling_voids([room, duplicate], [_stair()]).room_ids == []


def test_inference_does_not_change_rooms_or_stair_geometry() -> None:
    room, stair = _room(), _stair()
    before = room.model_dump(), stair.model_dump()
    infer_stair_ceiling_voids([room], [stair])
    assert (room.model_dump(), stair.model_dump()) == before
