from __future__ import annotations

from architecture_walkthrough.geometry.models import Point2D, WallSegment
from architecture_walkthrough.pipeline import (
    _scale_constraints_from_dimension_annotations,
    _scale_constraints_from_door_gaps,
)
from architecture_walkthrough.vision.ocr import OCRText


def _pair(y: float, gap: float) -> list[WallSegment]:
    return [
        WallSegment(
            id=f"a_{y}",
            start=Point2D(x=0, y=y),
            end=Point2D(x=100, y=y),
        ),
        WallSegment(
            id=f"b_{y}",
            start=Point2D(x=100 + gap, y=y),
            end=Point2D(x=220 + gap, y=y),
        ),
    ]


def test_door_scale_uses_repeated_small_gap_cluster_not_every_wall_break() -> None:
    walls = [
        WallSegment(
            id="extent",
            start=Point2D(x=0, y=200),
            end=Point2D(x=750, y=200),
        ),
        *_pair(0, 64),
        *_pair(20, 72),
        *_pair(40, 96),
        *_pair(60, 101),
        *_pair(80, 236),
        *_pair(100, 377),
        *_pair(120, 11),
    ]

    constraints = _scale_constraints_from_door_gaps(walls)

    assert [constraint.measured_px[0] for constraint in constraints] == [64.0, 72.0]


def test_door_scale_rejects_a_single_unconfirmed_gap() -> None:
    walls = [
        WallSegment(id="extent", start=Point2D(x=0, y=100), end=Point2D(x=500, y=100)),
        *_pair(0, 70),
    ]

    assert _scale_constraints_from_door_gaps(walls) == []


def test_dimension_annotation_uses_four_crossing_wall_boundaries() -> None:
    walls = [
        WallSegment(id="left", start=Point2D(x=20, y=0), end=Point2D(x=20, y=160)),
        WallSegment(id="right", start=Point2D(x=180, y=0), end=Point2D(x=180, y=160)),
        WallSegment(id="top", start=Point2D(x=0, y=10), end=Point2D(x=200, y=10)),
        WallSegment(id="bottom", start=Point2D(x=0, y=150), end=Point2D(x=200, y=150)),
        # Closer in x, but it does not span the annotation's y coordinate.
        WallSegment(id="short", start=Point2D(x=60, y=0), end=Point2D(x=60, y=40)),
    ]
    label = OCRText(
        text="200X175",
        normalized_text="200X175",
        semantic_type="dimension",
        confidence=0.93,
        polygon=[(80, 70), (120, 70), (120, 90), (80, 90)],
    )

    constraints = _scale_constraints_from_dimension_annotations([label], walls)

    assert len(constraints) == 1
    assert constraints[0].measured_px == (160.0, 140.0)
    assert constraints[0].expected_m == (2.0, 1.75)
    assert constraints[0].tier == "dimension_annotation"


def test_open_plan_dimension_without_four_boundaries_is_not_scale_evidence() -> None:
    label = OCRText(
        text="599X340",
        normalized_text="599X340",
        semantic_type="dimension",
        confidence=0.9,
        polygon=[(80, 70), (120, 70), (120, 90), (80, 90)],
    )
    walls = [
        WallSegment(id="left", start=Point2D(x=20, y=0), end=Point2D(x=20, y=160)),
        WallSegment(id="top", start=Point2D(x=0, y=10), end=Point2D(x=200, y=10)),
        WallSegment(id="bottom", start=Point2D(x=0, y=150), end=Point2D(x=200, y=150)),
    ]

    assert _scale_constraints_from_dimension_annotations([label], walls) == []
