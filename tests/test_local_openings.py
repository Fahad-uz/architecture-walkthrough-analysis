from __future__ import annotations

import cv2
import numpy as np
import pytest

from architecture_walkthrough.config import GeometryDefaults, OpeningDetectionSettings
from architecture_walkthrough.geometry.models import Point2D, WallSegment
from architecture_walkthrough.geometry.wall_graph import enumerate_faces
from architecture_walkthrough.vision.local_openings import build_thin_line_mask, detect_local_openings

PPM = 50.0
HEIGHT = 300
WIDTH = 520
SETTINGS = OpeningDetectionSettings()
DEFAULTS = GeometryDefaults()


def _wall(wall_id: str, x1: float, y1: float, x2: float, y2: float) -> WallSegment:
    return WallSegment(id=wall_id, start=Point2D(x=x1, y=y1), end=Point2D(x=x2, y=y2), thickness_m=0.16)


def _blank() -> np.ndarray:
    return np.zeros((HEIGHT, WIDTH), dtype=np.uint8)


def _dark_wall_pair_with_gap() -> np.ndarray:
    # Horizontal wall at metre y=3.0 -> pixel row 150; gap 0.9 m between x px 200..245.
    dark = _blank()
    cv2.rectangle(dark, (50, 146), (200, 154), 255, -1)
    cv2.rectangle(dark, (245, 146), (450, 154), 255, -1)
    return dark


def _walls_pair() -> list[WallSegment]:
    return [_wall("w0", 1.0, 3.0, 4.0, 3.0), _wall("w1", 4.9, 3.0, 9.0, 3.0)]


def test_gap_with_swing_arc_is_a_door_with_hinge_and_merged_wall() -> None:
    dark = _dark_wall_pair_with_gap()
    thin = _blank()
    # Quarter arc hinged at the gap's right end, sweeping up in image space.
    cv2.ellipse(thin, (245, 150), (45, 45), 0, 180, 270, 255, 2)
    result = detect_local_openings(_walls_pair(), dark, thin, PPM, HEIGHT, SETTINGS, DEFAULTS)

    assert len(result.walls) == 1
    merged = result.walls[0]
    assert merged.id == "w0"
    assert merged.start.x == pytest.approx(1.0)
    assert merged.end.x == pytest.approx(9.0)

    assert len(result.doors) == 1 and not result.windows
    door = result.doors[0]
    assert door.wall_id == "w0"
    assert door.start_offset_m == pytest.approx(3.0, abs=0.05)
    assert door.end_offset_m == pytest.approx(3.9, abs=0.05)
    assert "swing_arc" in door.evidence_source
    assert door.hinge_side == "end"
    assert door.confidence > 0.5
    assert not result.ambiguous


def test_gap_with_parallel_lines_is_a_window() -> None:
    dark = _dark_wall_pair_with_gap()
    thin = _blank()
    for row in (147, 150, 153):
        cv2.line(thin, (200, row), (245, row), 255, 1)
    result = detect_local_openings(_walls_pair(), dark, thin, PPM, HEIGHT, SETTINGS, DEFAULTS)
    assert len(result.windows) == 1 and not result.doors
    window = result.windows[0]
    assert "parallel_lines" in window.evidence_source
    assert window.start_offset_m == pytest.approx(3.0, abs=0.05)
    assert window.sill_height_m == DEFAULTS.sill_height_m


def test_bare_gap_falls_back_to_width_heuristic_and_is_flagged() -> None:
    dark = _dark_wall_pair_with_gap()
    result = detect_local_openings(_walls_pair(), dark, _blank(), PPM, HEIGHT, SETTINGS, DEFAULTS)
    assert len(result.doors) == 1  # 0.9 m is door-sized
    assert result.doors[0].confidence <= 0.4
    assert len(result.ambiguous) == 1
    assert "width_heuristic" in result.ambiguous[0].evidence


def test_interior_mask_break_on_single_wall_is_detected() -> None:
    dark = _dark_wall_pair_with_gap()
    thin = _blank()
    cv2.ellipse(thin, (245, 150), (45, 45), 0, 180, 270, 255, 2)
    walls = [_wall("w0", 1.0, 3.0, 9.0, 3.0)]
    result = detect_local_openings(walls, dark, thin, PPM, HEIGHT, SETTINGS, DEFAULTS)
    assert len(result.doors) == 1
    door = result.doors[0]
    assert door.start_offset_m == pytest.approx(3.0, abs=0.1)
    assert door.end_offset_m == pytest.approx(3.9, abs=0.1)


def test_confirmed_opening_lets_rooms_close_across_the_gap() -> None:
    dark = _blank()
    thin = _blank()
    # Square room 1..9 / 1..5 metres with a divider at x=5 broken by a door gap.
    walls = [
        _wall("w0", 1, 1, 9, 1),
        _wall("w1", 9, 1, 9, 5),
        _wall("w2", 9, 5, 1, 5),
        _wall("w3", 1, 5, 1, 1),
        _wall("w4", 5, 1, 5, 2.5),
        _wall("w5", 5, 3.4, 5, 5),
    ]
    for wall in walls:
        p0 = (int(wall.start.x * PPM), int(HEIGHT - wall.start.y * PPM))
        p1 = (int(wall.end.x * PPM), int(HEIGHT - wall.end.y * PPM))
        cv2.line(dark, p0, p1, 255, 8)
    # Erase the doorway span from the divider and draw its swing arc.
    cv2.rectangle(dark, (int(5 * PPM) - 6, int(HEIGHT - 3.4 * PPM)), (int(5 * PPM) + 6, int(HEIGHT - 2.5 * PPM)), 0, -1)
    cv2.ellipse(thin, (int(5 * PPM), int(HEIGHT - 2.5 * PPM)), (45, 45), 0, 180, 270, 255, 2)

    result = detect_local_openings(walls, dark, thin, PPM, HEIGHT, SETTINGS, DEFAULTS)
    assert len(result.doors) == 1
    faces = enumerate_faces(result.walls, result.doors, result.windows)
    assert len(faces.faces) == 2


def test_no_walls_yields_empty_result() -> None:
    result = detect_local_openings([], _blank(), _blank(), PPM, HEIGHT, SETTINGS, DEFAULTS)
    assert result.walls == [] and result.doors == [] and result.windows == []


def test_thin_line_mask_excludes_bold_structure() -> None:
    adaptive = _blank()
    dark = _blank()
    cv2.rectangle(adaptive, (50, 146), (450, 154), 255, -1)  # everything drawn
    cv2.rectangle(dark, (50, 146), (450, 154), 255, -1)  # bold walls
    cv2.line(adaptive, (100, 200), (200, 200), 255, 1)  # a thin annotation
    thin = build_thin_line_mask(adaptive, dark)
    assert thin[200, 150] > 0
    assert thin[150, 250] == 0
