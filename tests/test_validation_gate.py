from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from architecture_walkthrough.config import AppConfig
from architecture_walkthrough.geometry.models import (
    DoorOpening,
    FloorPlanModel,
    Point2D,
    RoomPolygon,
    WallSegment,
)
from architecture_walkthrough.geometry.validation import (
    SourceEvidence,
    evaluate_quality,
    opening_interval_issues,
    validate_reconstruction,
    wall_crossings_without_junction,
    wall_mask_overlap_score,
)
from architecture_walkthrough.pipeline import build_model

FIXTURES = Path(__file__).parent / "fixtures"


def _square_model(**overrides) -> FloorPlanModel:
    walls = [
        WallSegment(id="w0", start=Point2D(x=0, y=0), end=Point2D(x=4, y=0)),
        WallSegment(id="w1", start=Point2D(x=4, y=0), end=Point2D(x=4, y=3)),
        WallSegment(id="w2", start=Point2D(x=4, y=3), end=Point2D(x=0, y=3)),
        WallSegment(id="w3", start=Point2D(x=0, y=3), end=Point2D(x=0, y=0)),
    ]
    rooms = [
        RoomPolygon(
            id="room_0",
            face_id="face_0",
            points=[Point2D(x=0, y=0), Point2D(x=4, y=0), Point2D(x=4, y=3), Point2D(x=0, y=3)],
        )
    ]
    payload = {
        "pixels_per_metre": 100.0,
        "walls": walls,
        "rooms": rooms,
        "metadata": {"scale_source": "room_dimension", "scale_confidence": 0.85},
    }
    payload.update(overrides)
    return FloorPlanModel(**payload)


def test_zero_openings_in_multiroom_plan_lowers_score() -> None:
    two_rooms = _square_model(
        walls=[
            WallSegment(id="w0", start=Point2D(x=0, y=0), end=Point2D(x=4, y=0)),
            WallSegment(id="w1", start=Point2D(x=4, y=0), end=Point2D(x=4, y=3)),
            WallSegment(id="w2", start=Point2D(x=4, y=3), end=Point2D(x=0, y=3)),
            WallSegment(id="w3", start=Point2D(x=0, y=3), end=Point2D(x=0, y=0)),
            WallSegment(id="w4", start=Point2D(x=2, y=0), end=Point2D(x=2, y=3)),
        ],
        rooms=[
            RoomPolygon(id="a", points=[Point2D(x=0, y=0), Point2D(x=2, y=0), Point2D(x=2, y=3), Point2D(x=0, y=3)]),
            RoomPolygon(id="b", points=[Point2D(x=2, y=0), Point2D(x=4, y=0), Point2D(x=4, y=3), Point2D(x=2, y=3)]),
        ],
    )
    report = evaluate_quality(two_rooms)
    assert report.components["openings"] == pytest.approx(0.2)
    issues = validate_reconstruction(two_rooms)
    assert any(issue.code == "no_openings_detected" for issue in issues)
    # The old scorer gave 0.9+ here; unknown openings must keep it out of "high".
    assert report.state != "high" or report.score < 0.9


def test_missing_evidence_scores_unknown_not_perfect() -> None:
    report = evaluate_quality(_square_model())
    assert report.components["wall_mask_overlap"] == pytest.approx(0.4)
    assert report.components["junction_support"] == pytest.approx(0.4)


def test_wall_mask_overlap_rewards_agreement() -> None:
    model = _square_model()
    ppm = 100.0
    mask = np.zeros((400, 500), dtype=np.uint8)
    import cv2

    for wall in model.walls:
        p0 = (int(wall.start.x * ppm), int(400 - wall.start.y * ppm))
        p1 = (int(wall.end.x * ppm), int(400 - wall.end.y * ppm))
        cv2.line(mask, p0, p1, 255, 12)
    good = wall_mask_overlap_score(model, SourceEvidence(mask, ppm, 400))
    empty = wall_mask_overlap_score(model, SourceEvidence(np.zeros_like(mask), ppm, 400))
    assert good > 0.8
    assert empty == 0.0


def test_wall_thickness_scale_is_flagged() -> None:
    model = _square_model(metadata={"scale_source": "wall_thickness", "scale_confidence": 0.2})
    issues = validate_reconstruction(model)
    assert any(issue.code == "scale_from_assumed_wall_thickness" for issue in issues)


def test_opening_outside_wall_span_is_error() -> None:
    model = _square_model(
        doors=[
            DoorOpening(
                id="door_bad",
                center=Point2D(x=3.9, y=0),
                wall_id="w0",
                start_offset_m=3.6,
                end_offset_m=4.5,
                width_m=0.9,
            )
        ]
    )
    issues = opening_interval_issues(model)
    assert any(issue.code == "opening_outside_wall_span" for issue in issues)


def test_overlapping_openings_are_errors() -> None:
    model = _square_model(
        doors=[
            DoorOpening(id="d0", center=Point2D(x=1, y=0), wall_id="w0", start_offset_m=0.6, end_offset_m=1.5),
            DoorOpening(id="d1", center=Point2D(x=1.3, y=0), wall_id="w0", start_offset_m=1.0, end_offset_m=1.9),
        ]
    )
    issues = opening_interval_issues(model)
    assert any(issue.code == "openings_overlap" for issue in issues)


def test_crossing_walls_without_junction_reported() -> None:
    model = _square_model(
        walls=[
            WallSegment(id="wa", start=Point2D(x=0, y=1), end=Point2D(x=4, y=1)),
            WallSegment(id="wb", start=Point2D(x=2, y=-1), end=Point2D(x=2, y=2)),
        ],
        rooms=[],
    )
    issues = wall_crossings_without_junction(model)
    assert issues and issues[0].code == "wall_crossing_without_junction"


def test_export_gate_blocks_low_quality_unless_forced(tmp_path: Path) -> None:
    model = _square_model()
    model = model.model_copy(
        update={"reconstruction": model.reconstruction.model_copy(update={"quality_score": 0.2, "quality_state": "failed"})}
    )
    plan = tmp_path / "floorplan.json"
    model.save_json(plan)
    config = AppConfig()
    with pytest.raises(ValueError, match="force=True"):
        build_model(plan, tmp_path / "out.glb", config)
    output = build_model(plan, tmp_path / "out.glb", config, force=True)
    assert output.exists()


def test_export_gate_allows_good_quality(tmp_path: Path) -> None:
    model = _square_model()
    model = model.model_copy(
        update={"reconstruction": model.reconstruction.model_copy(update={"quality_score": 0.8, "quality_state": "high"})}
    )
    plan = tmp_path / "floorplan.json"
    model.save_json(plan)
    output = build_model(plan, tmp_path / "out.glb", AppConfig())
    assert output.exists()


def test_fixture_two_bedroom_flat_passes_gate(tmp_path: Path) -> None:
    model = FloorPlanModel.load_json(FIXTURES / "sample_two_bedroom_flat.json")
    model = model.model_copy(update={"metadata": {"scale_source": "manual", "scale_confidence": 1.0}})
    report = evaluate_quality(model)
    assert report.score >= 0.45, report.components
