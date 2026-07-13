from __future__ import annotations

import pytest

from architecture_walkthrough.geometry.scale_solver import ScaleConstraint, solve_scale


def _room(ppm: float, index: int = 0) -> ScaleConstraint:
    return ScaleConstraint(
        id=f"room_{index}",
        source="dimension_label_in_room_polygon",
        measured_px=(6 * ppm, 3 * ppm),
        expected_m=(6, 3),
        tier="room_dimension",
    )


def _door(ppm: float, index: int = 0) -> ScaleConstraint:
    return ScaleConstraint(
        id=f"door_{index}",
        source="gap",
        measured_px=(0.85 * ppm, 0.85 * ppm),
        expected_m=(0.85, 0.85),
        tier="door_width",
    )


def _thickness(ppm: float, index: int = 0) -> ScaleConstraint:
    return ScaleConstraint(
        id=f"thick_{index}",
        source="assumed_band_thickness",
        measured_px=(0.12 * ppm, 0.12 * ppm),
        expected_m=(0.12, 0.12),
        tier="wall_thickness",
    )


def test_manual_scale_wins_over_everything() -> None:
    result = solve_scale([_room(100)], manual_pixels_per_metre=42.0)
    assert result.pixels_per_metre == 42.0
    assert result.source == "manual"
    assert result.confidence == 1.0


def test_room_dimension_tier_beats_many_thickness_constraints() -> None:
    # Five agreeing thickness constraints say 200 ppm; one real room says 100.
    constraints = [_room(100), *[_thickness(200, i) for i in range(5)]]
    result = solve_scale(constraints)
    assert result.pixels_per_metre == pytest.approx(100)
    assert result.source == "room_dimension"
    superseded = [r for r in result.rejected_constraints if r.reason and "superseded" in r.reason]
    assert len(superseded) == 5


def test_tiers_are_never_averaged_together() -> None:
    # Doors disagree with the room tier; the answer must be exactly the room ppm.
    constraints = [_room(100), _door(140, 0), _door(140, 1)]
    result = solve_scale(constraints)
    assert result.pixels_per_metre == pytest.approx(100)


def test_door_width_fallback_when_no_dimensions() -> None:
    result = solve_scale([_door(90, 0), _door(90, 1), _thickness(50)])
    assert result.source == "door_width"
    assert result.pixels_per_metre == pytest.approx(0.85 * 90 / 0.85)
    assert result.confidence < 0.5


def test_wall_thickness_is_last_resort_and_low_confidence() -> None:
    result = solve_scale([_thickness(80, 0), _thickness(80, 1)])
    assert result.source == "wall_thickness"
    assert result.confidence <= 0.25


def test_multiple_consistent_rooms_raise_confidence() -> None:
    single = solve_scale([_room(100)])
    multiple = solve_scale([_room(100, 0), _room(101, 1), _room(99, 2)])
    assert multiple.confidence > single.confidence


def test_tier_with_only_outliers_falls_through() -> None:
    # A room constraint outside the plausible ppm range falls through to doors.
    bad_room = ScaleConstraint(
        id="room_bad",
        source="dimension_label_in_room_polygon",
        measured_px=(6 * 5000, 3 * 5000),
        expected_m=(6, 3),
        tier="room_dimension",
    )
    result = solve_scale([bad_room, _door(90)])
    assert result.source == "door_width"
    assert any(r.reason == "pixels_per_metre_out_of_range" for r in result.rejected_constraints)


def test_no_constraints_raises() -> None:
    with pytest.raises(ValueError):
        solve_scale([])
