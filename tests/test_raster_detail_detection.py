from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from architecture_walkthrough.pipeline import _classify_special_elements
from architecture_walkthrough.vision.wall_detection import (
    WallBand,
    _components_to_bands,
    _suppress_repetitive_detail_bands,
    detect_wall_bands,
)


@pytest.mark.parametrize("transpose", [False, True])
def test_skewed_thin_detail_keeps_thick_perimeter_and_partitions(
    tmp_path: Path, transpose: bool,
) -> None:
    horizontal = np.zeros((700, 1000), dtype=np.uint8)
    vertical = np.zeros_like(horizontal)
    for y in range(80, 165, 12):
        cv2.line(horizontal, (100, y), (650, y + 4), 255, 3)
    # Substantial repeated partitions have similar endpoints and pitch. Their
    # real cross-sections, unlike skew-inflated hairlines, must remain walls.
    for y in range(300, 385, 20):
        cv2.line(horizontal, (100, y), (650, y + 4), 255, 13)
    cv2.rectangle(horizontal, (80, 45), (670, 58), 255, -1)
    cv2.rectangle(horizontal, (80, 190), (670, 203), 255, -1)
    cv2.rectangle(vertical, (80, 45), (92, 203), 255, -1)
    cv2.rectangle(vertical, (658, 45), (670, 203), 255, -1)
    if transpose:
        horizontal, vertical = vertical.T.copy(), horizontal.T.copy()
    paths = [tmp_path / "horizontal.png", tmp_path / "vertical.png"]
    for path, mask in zip(paths, (horizontal, vertical), strict=True):
        assert cv2.imwrite(str(path), mask)

    result = detect_wall_bands(*paths)

    assert len(result.rejected_detail_bands) == 8
    assert len(result.repetitive_detail_regions) == 1
    assert _classify_special_elements(result.repetitive_detail_regions, 50, 700) == []
    for band in result.rejected_detail_bands:
        assert band.thickness_px > 6  # The bounding box retains the raster drift.
        assert band.ink_thickness_px == 5
    orientation = "v" if transpose else "h"
    partitions = [
        band for band in result.bands
        if band.orientation == orientation
        and 295 <= (band.centerline[0] if transpose else band.centerline[1]) <= 390
    ]
    assert len(partitions) == 5
    # All four enclosure sides survive; no inferred line closes the interior.
    enclosure = [band for band in result.bands if band not in partitions]
    assert len(enclosure) == 4


def _horizontal_band(identifier: str, y: float, thickness: float = 4) -> WallBand:
    return WallBand(
        identifier, "h", (100, y - thickness / 2, 550, thickness),
        (100, y, 650, y), thickness, 1,
    )


def test_separate_regular_runs_are_detected_without_spanning_the_interruption() -> None:
    first = [_horizontal_band(f"first_{y}", y) for y in range(80, 129, 12)]
    second = [_horizontal_band(f"second_{y}", y) for y in range(182, 231, 12)]
    isolated = _horizontal_band("isolated", 151)

    kept, rejected, regions = _suppress_repetitive_detail_bands(
        [*first, isolated, *second], (700, 1000),
    )

    assert kept == [isolated]
    assert {band.id for band in rejected} == {band.id for band in [*first, *second]}
    assert len(regions) == 2
    assert all(not (region.rect[1] <= 151 <= region.rect[1] + region.rect[3]) for region in regions)


def test_sparse_parallel_outline_faces_remain_structural() -> None:
    faces = [_horizontal_band(f"face_{y}", y) for y in (80, 100, 240, 260, 400, 420)]

    kept, rejected, regions = _suppress_repetitive_detail_bands(faces, (700, 1000))

    assert kept == faces
    assert rejected == []
    assert regions == []


def test_measured_cross_section_does_not_change_component_geometry_or_thickness() -> None:
    mask = np.zeros((200, 700), dtype=np.uint8)
    cv2.line(mask, (50, 80), (600, 84), 255, 3)

    [band] = _components_to_bands(mask, "h", 40, 3, 30, "raw_")

    assert band.rect == (48, 78, 555, 9)
    assert band.centerline == (48, 82.5, 603, 82.5)
    assert band.thickness_px == 9
    assert band.ink_thickness_px == 5
