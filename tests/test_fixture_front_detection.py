from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from shapely.geometry import Polygon

from architecture_walkthrough.vision.preprocessing import preprocess_array
from architecture_walkthrough.vision.wall_detection import (
    WallBand,
    _suppress_enclosed_fixture_fronts,
    detect_wall_bands,
)


def _band(name: str, orientation: str, coordinate: float, start: float, end: float,
          thickness: float = 4.0) -> WallBand:
    line = ((start, coordinate, end, coordinate) if orientation == "h"
            else (coordinate, start, coordinate, end))
    rect = ((start, coordinate - thickness / 2, end - start, thickness)
            if orientation == "h" else (coordinate - thickness / 2, start, thickness, end - start))
    return WallBand(name, orientation, rect, line, thickness, 1.0)


def _outline_evidence() -> list[WallBand]:
    return [
        _band("paired_top", "h", 100, 100, 800, 22),
        _band("paired_bottom", "h", 850, 100, 800, 22),
        _band("paired_left", "v", 100, 100, 850, 22),
        _band("paired_right", "v", 800, 100, 850, 22),
    ]


def _cupboard(depth: float = 50) -> list[WallBand]:
    return [
        _band("front", "v", 500 - depth, 100, 400),
        _band("back", "v", 500, 100, 400, 14),
        _band("end_panel", "h", 400, 500 - depth, 500, 14),
    ]


def test_closed_thin_cupboard_front_becomes_fixture_without_removing_its_back() -> None:
    bands = [*_outline_evidence(), *_cupboard()]
    kept, rejected, regions = _suppress_enclosed_fixture_fronts(bands, (1000, 1000))
    assert [band.id for band in rejected] == ["front"]
    assert {band.id for band in kept} == {band.id for band in bands} - {"front"}
    assert len(regions) == 1
    assert Polygon(regions[0].polygon).area == pytest.approx(50 * 300)
    assert regions[0].front_band_ids == ("front",)


def test_closed_l_counter_keeps_concave_footprint_and_structural_walls() -> None:
    bands = [*_outline_evidence(),
             _band("back_vertical", "v", 600, 300, 700, 14),
             _band("back_horizontal", "h", 700, 300, 600, 14),
             _band("cap_top", "h", 300, 550, 600, 14),
             _band("cap_left", "v", 300, 650, 700, 14),
             _band("front_vertical", "v", 550, 300, 650),
             _band("front_horizontal", "h", 650, 300, 550)]
    kept, rejected, regions = _suppress_enclosed_fixture_fronts(bands, (1000, 1000))
    assert {band.id for band in rejected} == {"front_horizontal", "front_vertical"}
    assert all(band in kept for band in bands if band.thickness_px >= 14)
    assert len(regions) == 1
    shape = Polygon(regions[0].polygon)
    assert shape.area == pytest.approx(50 * 400 + 50 * 250)
    assert shape.area < shape.convex_hull.area


@pytest.mark.parametrize("scenario", ["single_line", "open_end", "wide_room", "paired_front", "partial_front"])
def test_ambiguous_partition_evidence_is_preserved(scenario: str) -> None:
    bands = [*_outline_evidence(), *_cupboard()]
    if scenario == "single_line":
        bands = [band for band in bands if not band.id.startswith("paired_")]
        bands.append(_band("single_top", "h", 100, 100, 800))
    elif scenario == "open_end":
        bands = [band for band in bands if band.id != "end_panel"]
    elif scenario == "wide_room":
        bands = [*_outline_evidence(), *_cupboard(depth=180)]
    elif scenario == "paired_front":
        bands = [band for band in bands if band.id != "front"]
        bands.append(_band("paired_front", "v", 450, 100, 400, 14))
    elif scenario == "partial_front":
        bands = [band for band in bands if band.id != "front"]
        bands.append(_band("front", "v", 450, 100, 650))
    kept, rejected, regions = _suppress_enclosed_fixture_fronts(bands, (1000, 1000))
    assert kept == bands
    assert not rejected and not regions


@pytest.mark.parametrize("rotation", [0, 1])
def test_raster_outlined_cupboard_is_not_an_extra_room(tmp_path: Path, rotation: int) -> None:
    image = np.full((800, 900, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (100, 100), (800, 700), (0, 0, 0), 2)
    cv2.rectangle(image, (120, 120), (780, 680), (0, 0, 0), 2)
    # Structural partition is outlined, its cupboard front is a single stroke.
    cv2.rectangle(image, (550, 120), (560, 400), (0, 0, 0), 2)
    cv2.rectangle(image, (500, 390), (560, 400), (0, 0, 0), 2)
    cv2.line(image, (500, 120), (500, 390), (0, 0, 0), 2)
    if rotation:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    layers = preprocess_array(image, tmp_path, max_side=900).layers
    result = detect_wall_bands(layers["horizontal_wall_band"], layers["vertical_wall_band"])
    assert len(result.fixture_detail_regions) == 1
    assert len(result.rejected_fixture_bands) == 1
    assert len(result.walls) == 6  # Four perimeter runs, partition and cupboard end.
    assert result.rejected_fixture_bands[0].orientation == ("h" if rotation else "v")
