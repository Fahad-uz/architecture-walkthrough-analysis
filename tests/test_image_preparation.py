from __future__ import annotations

from io import BytesIO
import math
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

import architecture_walkthrough.api.app as api_app
from architecture_walkthrough.api.app import JobRecord
from architecture_walkthrough.config import AISettings, AppConfig, OCRSettings, PathSettings
from architecture_walkthrough.geometry.models import FloorPlanModel
from architecture_walkthrough.pipeline import analyze_image
from architecture_walkthrough.vision.image_preparation import (
    preparation_frame,
    preparation_metadata,
    prepared_dimensions,
    rotate_image,
)
from architecture_walkthrough.vision.plan_roi import detect_plan_roi
from architecture_walkthrough.vision.preprocessing import resize_preserving_aspect


@pytest.mark.parametrize("turns", range(4))
def test_rotation_pixels_and_edge_mapping_round_trip(turns):
    image = np.arange(5 * 9 * 3, dtype=np.uint8).reshape(5, 9, 3)
    prepared = rotate_image(image, turns)
    np.testing.assert_array_equal(prepared, np.rot90(image, -turns))
    height, width = prepared.shape[:2]
    roi = detect_plan_roi(prepared, crop_rect=(0, 0, width, height), padding_ratio=0)
    record = preparation_metadata(9, 5, turns, roi.roi, width, height)
    forward = np.asarray(record["source_to_analysis"])
    inverse = np.asarray(record["analysis_to_source"])
    corners = np.asarray([[0, 0, 1], [9, 0, 1], [0, 5, 1], [9, 5, 1], [2.25, 3.75, 1]])
    np.testing.assert_allclose((inverse @ forward @ corners.T).T, corners, atol=1e-12)
    for y in range(5):
        for x in range(9):
            point = forward @ [x + .5, y + .5, 1]
            np.testing.assert_array_equal(prepared[int(point[1]), int(point[0])], image[y, x])


def test_mapping_uses_actual_padded_crop_and_per_axis_resize():
    source = np.zeros((91, 123, 3), dtype=np.uint8)
    prepared = rotate_image(source, 1)
    roi = detect_plan_roi(prepared, crop_rect=(12, 20, 66, 81), padding_ratio=.02)
    resized = resize_preserving_aspect(roi.image, max_side=40)
    ah, aw = resized.shape[:2]
    metadata = preparation_metadata(123, 91, 1, roi.roi, aw, ah)
    # Source (40, 61) -> prepared (30, 40) -> padded ROI origin (10,18).
    expected = [20 * aw / roi.roi.rect.width, 22 * ah / roi.roi.rect.height, 1]
    forward = np.asarray(metadata["source_to_analysis"])
    inverse = np.asarray(metadata["analysis_to_source"])
    np.testing.assert_allclose(forward @ [40, 61, 1], expected)
    ppm = 25
    metre_point = np.asarray([expected[0] / ppm, (ah - expected[1]) / ppm, 1])
    metre_to_analysis = np.asarray([[ppm, 0, 0], [0, -ppm, ah], [0, 0, 1]])
    np.testing.assert_allclose(inverse @ metre_to_analysis @ metre_point, [40, 61, 1])


@pytest.mark.parametrize("turns", range(4))
@pytest.mark.parametrize("degrees", [-15, -4.3, .1, 7.5, 15])
@pytest.mark.parametrize("size", [(31, 19), (30, 18)])
def test_fine_rotation_expands_edges_preserves_lengths_and_matches_raster(turns, degrees, size):
    width, height = size
    output_width, output_height, matrix = preparation_frame(width, height, turns, degrees)
    corners = matrix @ np.asarray([[0, width, width, 0], [0, 0, height, height], [1, 1, 1, 1]])
    assert corners[0].min() >= -1e-9
    assert corners[1].min() >= -1e-9
    assert corners[0].max() <= output_width + 1e-9
    assert corners[1].max() <= output_height + 1e-9
    np.testing.assert_allclose(matrix @ [width / 2, height / 2, 1],
                               [output_width / 2, output_height / 2, 1])
    np.testing.assert_allclose(matrix[:2, :2].T @ matrix[:2, :2], np.eye(2), atol=1e-15)
    angle = math.radians(turns * 90 + degrees)
    np.testing.assert_allclose(matrix[:2, 0], [math.cos(angle), math.sin(angle)], atol=1e-15)

    # Two linear colour ramps encode source pixel centres. Bilinear interpolation
    # must match the inverse edge map, apart from OpenCV's 1/32 lookup rounding.
    yy, xx = np.mgrid[:height, :width]
    image = np.stack([xx + .5, yy + .5, np.full_like(xx, 100)], axis=-1).astype(np.float32)
    actual = rotate_image(image, turns, degrees)
    assert actual.shape[:2] == (output_height, output_width)
    yy, xx = np.mgrid[:output_height, :output_width]
    destination = np.stack([xx + .5, yy + .5, np.ones_like(xx)], axis=-1)
    expected = destination @ np.linalg.inv(matrix).T
    inside = ((expected[..., 0] >= 1.5) & (expected[..., 0] <= width - 1.5)
              & (expected[..., 1] >= 1.5) & (expected[..., 1] <= height - 1.5))
    assert inside.any()
    np.testing.assert_allclose(actual[inside, :2], expected[inside, :2], atol=.017)
    if abs(degrees) >= 7.5:
        np.testing.assert_array_equal(actual[0, 0], [255, 255, 255])


@pytest.mark.parametrize("turns", range(4))
def test_zero_fine_angle_keeps_exact_legacy_pixels_and_metadata(turns):
    image = np.arange(19 * 31 * 3, dtype=np.uint8).reshape(19, 31, 3)
    prepared = rotate_image(image, turns)
    np.testing.assert_array_equal(rotate_image(image, turns, 0), prepared)
    if turns == 0:
        assert rotate_image(image, turns, 0) is image
    height, width = prepared.shape[:2]
    roi = detect_plan_roi(prepared, crop_rect=(0, 0, width, height), padding_ratio=0)
    metadata = preparation_metadata(31, 19, turns, roi.roi, width, height)
    assert preparation_metadata(31, 19, turns, roi.roi, width, height, 0) == metadata
    assert metadata["version"] == 1
    assert "straighten_degrees" not in metadata


def test_fine_mapping_composes_actual_padded_crop_and_rounded_resize():
    image = np.zeros((91, 123, 3), dtype=np.uint8)
    prepared = rotate_image(image, 1, -8.4)
    roi = detect_plan_roi(prepared, crop_rect=(12, 20, 66, 81), padding_ratio=.02)
    resized = resize_preserving_aspect(roi.image, max_side=40)
    ah, aw = resized.shape[:2]
    record = preparation_metadata(123, 91, 1, roi.roi, aw, ah, -8.4)
    width, height, source_to_prepared = preparation_frame(123, 91, 1, -8.4)
    assert record["version"] == 2
    assert record["straighten_degrees"] == -8.4
    assert (record["prepared_width_px"], record["prepared_height_px"]) == (width, height)
    point = source_to_prepared @ [40, 61, 1]
    expected = [(point[0] - roi.roi.rect.x) * aw / roi.roi.rect.width,
                (point[1] - roi.roi.rect.y) * ah / roi.roi.rect.height, 1]
    forward = np.asarray(record["source_to_analysis"])
    inverse = np.asarray(record["analysis_to_source"])
    np.testing.assert_allclose(forward @ [40, 61, 1], expected)
    np.testing.assert_allclose(inverse @ expected, [40, 61, 1])


@pytest.mark.parametrize("degrees", [-15.01, 15.01, True, "5", None,
                                         float("nan"), float("inf"), -float("inf")])
def test_direct_preparation_rejects_invalid_fine_angles(degrees):
    with pytest.raises(ValueError, match="straighten_degrees"):
        rotate_image(np.zeros((2, 3, 3), dtype=np.uint8), 0, degrees)
    with pytest.raises(ValueError, match="straighten_degrees"):
        prepared_dimensions(3, 2, 0, degrees)


@pytest.mark.parametrize("turns", [-1, 4, .5, True, float("nan")])
def test_direct_preparation_rejects_invalid_turns(turns):
    with pytest.raises(ValueError, match="rotation_quarter_turns"):
        rotate_image(np.zeros((2, 3, 3), dtype=np.uint8), turns)


@pytest.fixture
def upload_app(tmp_path, monkeypatch):
    monkeypatch.setattr(api_app, "load_config", lambda: AppConfig(paths=PathSettings(work_root=tmp_path)))
    app = api_app.create_app()
    calls = []
    monkeypatch.setattr(app.state.runner, "start_analysis", lambda *args: calls.append(args))
    data = BytesIO()
    Image.new("RGB", (120, 90), "white").save(data, format="PNG")
    return TestClient(app), calls, data.getvalue()


@pytest.mark.parametrize("value", ["", "-1", "4", "1.5", "NaN", "true"])
def test_api_rejects_invalid_turns_before_analysis(upload_app, tmp_path, value):
    client, calls, data = upload_app
    response = client.post("/jobs", data={"rotation_quarter_turns": value},
                           files={"file": ("plan.png", data, "image/png")})
    assert response.status_code == 400
    assert "rotation_quarter_turns" in response.json()["detail"]
    assert not calls
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("value", ["", " ", "-15.01", "15.01", "NaN", "Inf", "-Inf", "true", "bad"])
def test_api_rejects_invalid_fine_angles_before_creating_job(upload_app, tmp_path, value):
    client, calls, data = upload_app
    response = client.post("/jobs", data={"straighten_degrees": value},
                           files={"file": ("plan.png", data, "image/png")})
    assert response.status_code == 400
    assert "straighten_degrees" in response.json()["detail"]
    assert not calls
    assert not list(tmp_path.iterdir())


def test_api_rejects_duplicate_fine_angles(upload_app, tmp_path):
    client, calls, data = upload_app
    response = client.post("/jobs", files=[
        ("file", ("plan.png", data, "image/png")),
        ("straighten_degrees", (None, "2")), ("straighten_degrees", (None, "3")),
    ])
    assert response.status_code == 400
    assert "duplicate" in response.json()["detail"]
    assert not calls
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("turns", range(4))
@pytest.mark.parametrize("degrees", [-15, 0, 15])
def test_api_fine_rotation_crop_bounds_and_worker_arguments(upload_app, turns, degrees):
    client, calls, data = upload_app
    width, height = prepared_dimensions(120, 90, turns, degrees)
    crop = (width - 10, height - 20, 10, 20)
    fields = dict(zip(("crop_x", "crop_y", "crop_width", "crop_height"), map(str, crop)))
    fields.update(rotation_quarter_turns=str(turns), straighten_degrees=str(degrees))
    response = client.post("/jobs", data=fields, files={"file": ("plan.png", data, "image/png")})
    assert response.status_code == 200, response.text
    assert calls[0][-3:] == (crop, turns, degrees)
    assert calls[0][1].read_bytes() == data
    fields["crop_width"] = "11"
    response = client.post("/jobs", data=fields, files={"file": ("plan.png", data, "image/png")})
    assert response.status_code == 400
    assert len(calls) == 1


@pytest.mark.parametrize("turns", range(4))
def test_api_crop_uses_prepared_frame_and_passes_rotation(upload_app, turns):
    client, calls, data = upload_app
    width, height = (90, 120) if turns % 2 else (120, 90)
    crop = (width - 10, height - 20, 10, 20)
    fields = dict(zip(("crop_x", "crop_y", "crop_width", "crop_height"), map(str, crop)))
    response = client.post("/jobs", data={**fields, "rotation_quarter_turns": str(turns)},
                           files={"file": ("plan.png", data, "image/png")})
    assert response.status_code == 200, response.text
    assert calls[0][-3:] == (crop, turns, 0)
    assert calls[0][1].read_bytes() == data
    fields["crop_width"] = "11"
    rejected = client.post("/jobs", data={**fields, "rotation_quarter_turns": str(turns)},
                          files={"file": ("plan.png", data, "image/png")})
    assert rejected.status_code == 400
    assert len(calls) == 1


def test_pipeline_rotation_preserves_scale_and_mapping(tmp_path: Path):
    source = tmp_path / "source.png"
    image = Image.new("RGB", (240, 180), "white")
    ImageDraw.Draw(image).rectangle((20, 20, 220, 160), outline="black", width=8)
    image.save(source)
    original = source.read_bytes()
    config = AppConfig(ai=AISettings(gemini_enabled=False), ocr=OCRSettings(enabled=False))
    config.preprocessing.max_side_px = 160
    model = analyze_image(source, tmp_path / "analysis", config, manual_scale=.01,
                          crop_rect=(0, 0, 180, 240), rotation_quarter_turns=1)
    metadata = model.metadata["image_preparation"]
    assert metadata["rotation_quarter_turns"] == 1
    assert (metadata["source_width_px"], metadata["source_height_px"]) == (240, 180)
    assert (metadata["prepared_width_px"], metadata["prepared_height_px"]) == (180, 240)
    assert (metadata["analysis_width_px"], metadata["analysis_height_px"]) == (120, 160)
    assert model.pixels_per_metre == pytest.approx(100 * 160 / 240)
    assert source.read_bytes() == original
    raw = FloorPlanModel.load_json(tmp_path / "analysis" / "floorplan.raw.json")
    saved = FloorPlanModel.load_json(tmp_path / "analysis" / "floorplan.optimized.json")
    assert raw.metadata["image_preparation"] == saved.metadata["image_preparation"] == metadata


@pytest.mark.parametrize("degrees", [-7.5, 7.5])
def test_pipeline_fine_rotation_keeps_source_scale_and_saved_provenance(tmp_path: Path, degrees):
    source = tmp_path / "source.png"
    image = Image.new("RGB", (240, 180), "white")
    ImageDraw.Draw(image).rectangle((20, 20, 220, 160), outline="black", width=8)
    image.save(source)
    original = source.read_bytes()
    config = AppConfig(ai=AISettings(gemini_enabled=False), ocr=OCRSettings(enabled=False))
    config.preprocessing.max_side_px = 160
    width, height = prepared_dimensions(240, 180, 1, degrees)
    model = analyze_image(source, tmp_path / "analysis", config, manual_scale=.01,
                          crop_rect=(0, 0, width, height), rotation_quarter_turns=1,
                          straighten_degrees=degrees)
    metadata = model.metadata["image_preparation"]
    assert metadata["version"] == 2
    assert metadata["straighten_degrees"] == degrees
    assert model.pixels_per_metre == pytest.approx(100 * 160 / max(width, height))
    assert model.metadata["manual_source_metres_per_pixel"] == .01
    assert source.read_bytes() == original
    raw = FloorPlanModel.load_json(tmp_path / "analysis" / "floorplan.raw.json")
    saved = FloorPlanModel.load_json(tmp_path / "analysis" / "floorplan.optimized.json")
    assert raw.metadata["image_preparation"] == saved.metadata["image_preparation"] == metadata


def test_rotated_upload_with_corrective_turn_reproduces_analysis_and_geometry(tmp_path: Path):
    image = Image.new("RGB", (240, 180), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 220, 160), outline="black", width=8)
    draw.line((100, 20, 100, 110), fill="black", width=8)
    source, rotated = tmp_path / "upright.png", tmp_path / "rotated.png"
    image.save(source)
    image.transpose(Image.Transpose.ROTATE_270).save(rotated)
    original_bytes = rotated.read_bytes()
    config = AppConfig(ai=AISettings(gemini_enabled=False), ocr=OCRSettings(enabled=False))
    config.preprocessing.max_side_px = 160
    first = analyze_image(source, tmp_path / "first", config, manual_scale=.01,
                          crop_rect=(0, 0, 240, 180))
    second = analyze_image(rotated, tmp_path / "second", config, manual_scale=.01,
                           crop_rect=(0, 0, 240, 180), rotation_quarter_turns=3)
    with Image.open(first.metadata["roi_image"]) as a, Image.open(second.metadata["roi_image"]) as b:
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert first.pixels_per_metre == second.pixels_per_metre
    a, b = first.model_dump(), second.model_dump()
    for key in ("walls", "doors", "windows", "rooms", "balconies"):
        assert a[key] == b[key]
    assert rotated.read_bytes() == original_bytes


@pytest.mark.parametrize("degrees", [0, 7.5])
def test_correction_cannot_replace_preparation_provenance_and_original_endpoint(
    tmp_path, monkeypatch, degrees,
):
    monkeypatch.setattr(api_app, "load_config", lambda: AppConfig(paths=PathSettings(work_root=tmp_path)))
    job_id = "rotated-job"
    job_dir = tmp_path / job_id
    (job_dir / "debug").mkdir(parents=True)
    original_path = job_dir / "original.webp"
    Image.new("RGB", (240, 320), "red").save(original_path)
    analysis_path = job_dir / "debug" / "01_original_roi.png"
    width, height = prepared_dimensions(240, 320, 1, degrees)
    Image.new("RGB", (width, height), "white").save(analysis_path)
    (job_dir / "job.json").write_text(JobRecord(job_id=job_id, status="needs_review").model_dump_json())
    model = FloorPlanModel.load_json(Path("tests/fixtures/sample_floorplan.json"))
    roi = detect_plan_roi(np.zeros((height, width, 3), dtype=np.uint8),
                          crop_rect=(0, 0, width, height), padding_ratio=0)
    metadata = preparation_metadata(240, 320, 1, roi.roi, width, height, degrees)
    model.plan_roi = roi.roi
    model.metadata.update(image_preparation=metadata, source_image=str(original_path), roi_image=str(analysis_path))
    model.save_json(job_dir / "floorplan.optimized.json")
    payload = model.model_dump(mode="json")
    payload["metadata"] = {"image_preparation": {"version": 999}, "source_image": "/outside.png"}
    payload["plan_roi"] = None
    with TestClient(api_app.create_app()) as client:
        response = client.post(f"/jobs/{job_id}/corrections", json=payload)
        assert response.status_code == 200, response.text
        corrected = response.json()["model"]
        assert corrected["metadata"]["image_preparation"] == metadata
        assert corrected["metadata"]["source_image"] == str(original_path)
        assert corrected["plan_roi"] == model.plan_roi.model_dump()
        original = client.get(f"/jobs/{job_id}/original-image")
        assert original.status_code == 200
        assert original.headers["content-type"] == "image/webp"
        assert original.content == original_path.read_bytes()
        assert client.get(f"/jobs/{job_id}/source-image").content == analysis_path.read_bytes()
