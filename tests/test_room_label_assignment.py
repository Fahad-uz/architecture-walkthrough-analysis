from __future__ import annotations

from architecture_walkthrough.geometry.models import Point2D, WallSegment
from architecture_walkthrough.geometry.room_extraction import extract_rooms_from_walls
from architecture_walkthrough.vision.ocr import OCRText


def _label(text: str, semantic_type: str, x: float, y: float) -> OCRText:
    return OCRText(
        text=text,
        normalized_text=text,
        semantic_type=semantic_type,
        confidence=0.95,
        polygon=[(x - 1, y - 1), (x + 1, y - 1), (x + 1, y + 1), (x - 1, y + 1)],
    )


def test_room_keeps_name_and_dimension_from_separate_labels() -> None:
    walls = [
        WallSegment(start=Point2D(x=0, y=0), end=Point2D(x=100, y=0)),
        WallSegment(start=Point2D(x=100, y=0), end=Point2D(x=100, y=80)),
        WallSegment(start=Point2D(x=100, y=80), end=Point2D(x=0, y=80)),
        WallSegment(start=Point2D(x=0, y=80), end=Point2D(x=0, y=0)),
    ]
    # Name intentionally comes first: the old implementation stopped there
    # and never retained the scale-bearing dimension label.
    labels = [
        _label("BEDROOM", "room_label", 50, 35),
        _label("300X400", "dimension", 50, 45),
    ]

    result = extract_rooms_from_walls(
        walls,
        labels,
        pixels_per_metre=1.0,
        junction_snap_m=0.0,
    )

    assert len(result.rooms) == 1
    assert result.rooms[0].name == "BEDROOM"
    assert result.rooms[0].dimension_m == (3.0, 4.0)
