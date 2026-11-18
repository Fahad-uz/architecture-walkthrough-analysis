from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

import architecture_walkthrough.pipeline as pipeline
from architecture_walkthrough.config import AISettings, AppConfig, OCRSettings
from architecture_walkthrough.geometry.models import FloorPlanModel
from architecture_walkthrough.vision.image_preparation import (
    preparation_metadata,
    rotate_image,
)
from architecture_walkthrough.vision.plan_roi import detect_plan_roi
from architecture_walkthrough.vision.preprocessing import resize_preserving_aspect


def project(matrix, points):
    values = np.c_[points, np.ones(len(points))] @ matrix.T
    return values[:, :2] / values[:, 2:]


def oriented_matrix(width, height, turns, degrees):
    quarter = np.asarray([
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[0, -1, height], [1, 0, 0], [0, 0, 1]],
        [[-1, 0, width], [0, -1, height], [0, 0, 1]],
        [[0, 1, 0], [-1, 0, width], [0, 0, 1]],
    ][turns], dtype=float)
    qw, qh = (height, width) if turns % 2 else (width, height)
    cosine, sine = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    ow = math.ceil(abs(qw * cosine) + abs(qh * sine) - 1e-9)
    oh = math.ceil(abs(qw * sine) + abs(qh * cosine) - 1e-9)
    fine = np.asarray([
        [cosine, -sine, ow / 2 - cosine * qw / 2 + sine * qh / 2],
        [sine, cosine, oh / 2 - sine * qw / 2 - cosine * qh / 2],
        [0, 0, 1],
    ])
    return fine @ quarter


@pytest.fixture
def config():
    result = AppConfig(ai=AISettings(gemini_enabled=False), ocr=OCRSettings(enabled=False))
    result.preprocessing.max_side_px = 137
    return result


@pytest.fixture
def photographed_plan(tmp_path):
    # The fixture is generated locally from a known page and an independently
    # chosen projective map. No model, dataset or external image is required.
    page = np.full((181, 241, 3), 255, dtype=np.uint8)
    cv2.rectangle(page, (30, 25), (210, 155), (0, 0, 0), 8)
    cv2.line(page, (110, 25), (110, 105), (0, 0, 0), 8)
    source_from_page = np.asarray([
        [1.1, .1, 22], [.04, 1.25, 16], [.0005, .0007, 1],
    ])
    to_centres = np.asarray([[1, 0, .5], [0, 1, .5], [0, 0, 1]])
    to_indices = np.asarray([[1, 0, -.5], [0, 1, -.5], [0, 0, 1]])
    photo = cv2.warpPerspective(
        page, to_indices @ source_from_page @ to_centres, (320, 280),
        flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255),
    )
    source = tmp_path / "photo.png"
    assert cv2.imwrite(str(source), photo)
    return source, photo, source_from_page


def perspective_spec(source_from_page, turns=0, degrees=0, source_width=320, source_height=280):
    orientation = oriented_matrix(source_width, source_height, turns, degrees)
    return {
        "corners": project(orientation @ source_from_page,
                           [[0, 0], [241, 0], [241, 181], [0, 181]]).tolist(),
        "aspect_ratio": 241 / 181,
        "output_width_px": 241,
    }


def scale_reference():
    return {"points": [[40, 40], [195, 135]], "distance_m": 3}


def test_projective_pipeline_uses_real_roi_resize_reference_and_saves_audit(
    photographed_plan, config, tmp_path, monkeypatch,
):
    source, original_image, source_from_page = photographed_plan
    original_bytes = source.read_bytes()
    reference = scale_reference()
    supplied = deepcopy(reference)
    calls = []
    warp = pipeline.warp_perspective

    def warp_once(image, frame):
        np.testing.assert_array_equal(image, original_image)
        calls.append(frame)
        return warp(image, frame)

    monkeypatch.setattr(pipeline, "warp_perspective", warp_once)
    monkeypatch.setattr(pipeline, "rotate_image", lambda *args: pytest.fail("second resampling"))
    output = tmp_path / "analysis"
    model = pipeline.analyze_image(
        source, output, config, crop_rect=(19, 13, 203, 149),
        perspective=perspective_spec(source_from_page), prepared_scale_reference=reference,
    )
    assert len(calls) == 1
    assert source.read_bytes() == original_bytes
    assert reference == supplied
    record = model.metadata["image_preparation"]
    assert record["version"] == 3
    assert (record["source_width_px"], record["source_height_px"]) == (320, 280)
    assert (record["prepared_width_px"], record["prepared_height_px"]) == (241, 181)
    assert record["prepared_roi"] == {"x": 14, "y": 8, "width": 213, "height": 159}
    assert (record["analysis_width_px"], record["analysis_height_px"]) == (137, 102)
    points = np.asarray([[26 * 137 / 213, 32 * 102 / 159],
                         [181 * 137 / 213, 127 * 102 / 159]])
    measured = math.dist(*points)
    ppm = measured / 3
    nominal_ppm = math.hypot(155, 95) / 3 * 137 / 213
    assert abs(ppm - nominal_ppm) > .01  # Rounded height changes a diagonal reference.
    assert model.pixels_per_metre == pytest.approx(ppm, abs=1e-12)
    assert model.metadata["scale_source"] == "manual"
    assert model.metadata["manual_source_metres_per_pixel"] is None
    assert model.metadata["geometry_originated_only_from_ai"] is False
    assert model.reconstruction.ai_geometry_originated is False
    assert model.walls and model.rooms
    audit = record["prepared_scale_reference"]
    assert audit["points"] == reference["points"]
    assert audit["distance_m"] == 3
    assert audit["measured_px"] == pytest.approx(math.hypot(155, 95))
    assert audit["pixels_per_metre"] == pytest.approx(math.hypot(155, 95) / 3)
    np.testing.assert_allclose(audit["analysis_points"], points)
    assert audit["analysis_measured_px"] == pytest.approx(measured)
    assert audit["analysis_pixels_per_metre"] == pytest.approx(ppm)
    # Independently map measured page landmarks through the known photograph
    # and compare the saved full projective source/analysis transforms.
    landmarks = [[40, 40], [195, 135], [95, 65]]
    source_points = project(source_from_page, landmarks)
    expected = (np.asarray(landmarks) - [14, 8]) * [137 / 213, 102 / 159]
    np.testing.assert_allclose(project(np.asarray(record["source_to_analysis"]), source_points),
                               expected, atol=1e-11)
    np.testing.assert_allclose(project(np.asarray(record["analysis_to_source"]), expected),
                               source_points, atol=1e-11)
    for name in ("floorplan.raw.json", "floorplan.optimized.json", "floorplan.json"):
        saved = FloorPlanModel.load_json(output / name)
        assert saved.metadata["image_preparation"] == record
        assert saved.pixels_per_metre == model.pixels_per_metre
    assert json.loads((output / "validation_report.json").read_text())["metadata"][
        "image_preparation"
    ] == record
    reference["points"][0][0] = 0
    assert audit["points"] == supplied["points"]  # Provenance is detached from caller input.


@pytest.mark.parametrize("turns,degrees", [(0, 0), (1, -8.4), (2, 7.5), (3, 0)])
def test_projective_orientation_is_composed_before_one_warp(
    photographed_plan, config, tmp_path, turns, degrees,
):
    source, photo, source_from_page = photographed_plan
    first = pipeline.analyze_image(
        source, tmp_path / "baseline", config, crop_rect=(19, 13, 203, 149),
        perspective=perspective_spec(source_from_page), prepared_scale_reference=scale_reference(),
    )
    rotated = np.rot90(photo, turns).copy()
    rotated_source = tmp_path / "rotated.png"
    assert cv2.imwrite(str(rotated_source), rotated)
    raw_height, raw_width = rotated.shape[:2]
    raw_from_original = np.linalg.inv(oriented_matrix(raw_width, raw_height, turns, 0))
    actual = pipeline.analyze_image(
        rotated_source, tmp_path / "oriented", config, crop_rect=(19, 13, 203, 149),
        rotation_quarter_turns=turns, straighten_degrees=degrees,
        perspective=perspective_spec(raw_from_original @ source_from_page, turns, degrees,
                                     raw_width, raw_height),
        prepared_scale_reference=scale_reference(),
    )
    assert Path(first.metadata["roi_image"]).read_bytes() == Path(actual.metadata["roi_image"]).read_bytes()
    for field in ("walls", "doors", "windows", "rooms", "balconies"):
        assert getattr(actual, field) == getattr(first, field)
    assert actual.pixels_per_metre == first.pixels_per_metre
    record = actual.metadata["image_preparation"]
    assert record["rotation_quarter_turns"] == turns
    assert record["straighten_degrees"] == degrees
    source_landmarks = project(source_from_page, [[40, 40], [195, 135], [95, 65]])
    actual_points = project(np.asarray(record["source_to_analysis"]),
                            project(raw_from_original, source_landmarks))
    expected_points = project(np.asarray(first.metadata["image_preparation"]["source_to_analysis"]),
                              source_landmarks)
    np.testing.assert_allclose(actual_points, expected_points, atol=1e-11)


@pytest.mark.parametrize("args,message", [
    ({"perspective": {}}, "fresh prepared scale reference"),
    ({"prepared_scale_reference": {}}, "requires perspective"),
    ({"perspective": {}, "prepared_scale_reference": {}, "manual_scale": .01},
     "cannot reuse source-image manual scale"),
])
def test_perspective_contract_rejects_inherited_or_missing_scale_before_loading(
    config, tmp_path, monkeypatch, args, message,
):
    monkeypatch.setattr(pipeline, "load_image", lambda *args: pytest.fail("image loaded"))
    with pytest.raises(ValueError, match=message):
        pipeline.analyze_image(tmp_path / "missing.png", tmp_path / "output", config, **args)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("reference", [
    {}, {"points": [[0, 0], [1, 1]], "distance_m": 1, "extra": 1},
    {"points": [[0, 0], [242, 100]], "distance_m": 1},
    {"points": [[0, 0], [10, 182]], "distance_m": 1},
    {"points": [[10, 10], [10.5, 10.5]], "distance_m": 1},
    {"points": [[0, 0], [10, 10]], "distance_m": 0},
    {"points": [[0, 0], [10, 10]], "distance_m": float("nan")},
    {"points": [[0, 0], [10, 10]], "distance_m": True},
])
def test_invalid_prepared_reference_rejected_before_warp_or_analysis(
    photographed_plan, config, tmp_path, monkeypatch, reference,
):
    source, _, transform = photographed_plan
    monkeypatch.setattr(pipeline, "warp_perspective", lambda *args: pytest.fail("image warped"))
    monkeypatch.setattr(pipeline, "detect_plan_roi", lambda *args, **kwargs: pytest.fail("ROI run"))
    with pytest.raises(ValueError, match="prepared scale"):
        pipeline.analyze_image(source, tmp_path / "output", config,
                               perspective=perspective_spec(transform),
                               prepared_scale_reference=reference)


@pytest.mark.parametrize("invalid", ["perspective", "crop"])
def test_invalid_rectification_or_crop_is_rejected_before_resampling(
    photographed_plan, config, tmp_path, monkeypatch, invalid,
):
    source, _, transform = photographed_plan
    spec = perspective_spec(transform)
    crop = (0, 0, 241, 181)
    if invalid == "perspective":
        spec["unexpected"] = "not part of the core JSON contract"
    else:
        # Valid in the original 320 x 280 image, outside the rectified page.
        crop = (0, 0, 242, 181)
    monkeypatch.setattr(pipeline, "warp_perspective", lambda *args: pytest.fail("image warped"))
    with pytest.raises(ValueError, match="perspective|crop"):
        pipeline.analyze_image(source, tmp_path / "output", config, crop_rect=crop,
                               perspective=spec, prepared_scale_reference=scale_reference())


@pytest.mark.parametrize("limit,value,message", [
    ("max_image_width", 240, "output_width_px"),
    ("max_image_height", 180, "output height"),
])
def test_direct_pipeline_respects_configured_rectified_dimension_limits(
    photographed_plan, config, tmp_path, monkeypatch, limit, value, message,
):
    source, _, transform = photographed_plan
    setattr(config.limits, limit, value)
    monkeypatch.setattr(pipeline, "warp_perspective", lambda *args: pytest.fail("image warped"))
    with pytest.raises(ValueError, match=message):
        pipeline.analyze_image(source, tmp_path / "output", config,
                               perspective=perspective_spec(transform),
                               prepared_scale_reference=scale_reference())


@pytest.mark.parametrize("distance_m", [1e-9, 1e9])
def test_prepared_reference_outside_solver_range_is_rejected_without_clamping(
    photographed_plan, config, tmp_path, monkeypatch, distance_m,
):
    source, _, transform = photographed_plan
    monkeypatch.setattr(pipeline, "_floorplan_vision_analysis",
                        lambda *args, **kwargs: pytest.fail("semantic analysis run"))
    monkeypatch.setattr(pipeline, "solve_scale", lambda *args, **kwargs: pytest.fail("solver run"))
    reference = scale_reference()
    reference["distance_m"] = distance_m
    with pytest.raises(ValueError, match="outside the configured range"):
        pipeline.analyze_image(source, tmp_path / "output", config,
                               perspective=perspective_spec(transform),
                               prepared_scale_reference=reference)


@pytest.mark.parametrize("turns,degrees", [(0, 0), (1, 0), (2, 7.5)])
def test_legacy_pipeline_keeps_existing_raster_and_metadata_path(
    tmp_path, config, monkeypatch, turns, degrees,
):
    source = tmp_path / "legacy.png"
    image = np.full((180, 240, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (220, 160), (0, 0, 0), 8)
    assert cv2.imwrite(str(source), image)
    original = source.read_bytes()
    prepared = rotate_image(image, turns, degrees)
    height, width = prepared.shape[:2]
    roi = detect_plan_roi(prepared, crop_rect=(0, 0, width, height), padding_ratio=.02)
    expected = resize_preserving_aspect(roi.image, config.preprocessing.max_side_px)
    ah, aw = expected.shape[:2]
    metadata = preparation_metadata(240, 180, turns, roi.roi, aw, ah, degrees)
    monkeypatch.setattr(pipeline, "warp_perspective", lambda *args: pytest.fail("perspective used"))
    model = pipeline.analyze_image(
        source, tmp_path / "output", config, manual_scale=.01,
        crop_rect=(0, 0, width, height), rotation_quarter_turns=turns,
        straighten_degrees=degrees,
    )
    np.testing.assert_array_equal(cv2.imread(model.metadata["roi_image"]), expected)
    assert model.metadata["image_preparation"] == metadata
    assert model.pixels_per_metre == pytest.approx(100 * max(aw, ah) / max(width, height))
    assert model.metadata["manual_source_metres_per_pixel"] == .01
    assert "prepared_scale_reference" not in model.metadata["image_preparation"]
    assert source.read_bytes() == original
