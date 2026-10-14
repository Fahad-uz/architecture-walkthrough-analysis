from __future__ import annotations

from io import BytesIO
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageOps

import architecture_walkthrough.api.app as api_app
from architecture_walkthrough.config import AppConfig, LimitSettings, PathSettings
from architecture_walkthrough.security.file_validation import validate_image_file
from architecture_walkthrough.vision.plan_roi import detect_plan_roi
from architecture_walkthrough.vision.preprocessing import load_image


CROP_FIELDS = ("crop_x", "crop_y", "crop_width", "crop_height")
VALID_CROP = dict(zip(CROP_FIELDS, ("20", "10", "100", "80"), strict=True))


@pytest.fixture
def upload_client(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        api_app, "load_config", lambda: AppConfig(paths=PathSettings(work_root=tmp_path))
    )
    app = api_app.create_app()
    calls = []

    def capture_analysis(record, image_path, config, manual_scale, crop_rect):
        calls.append((image_path, image_path.read_bytes(), crop_rect))

    monkeypatch.setattr(app.state.runner, "start_analysis", capture_analysis)
    image = BytesIO()
    Image.new("RGB", (120, 90), "white").save(image, format="PNG")
    with TestClient(app) as client:
        yield client, calls, image.getvalue()


@pytest.mark.parametrize(
    "fields", [fields for count in range(1, 4) for fields in combinations(CROP_FIELDS, count)]
)
def test_upload_rejects_partial_crop_before_creating_job(upload_client, tmp_path: Path, fields):
    client, calls, image_bytes = upload_client
    response = client.post(
        "/jobs", data={field: VALID_CROP[field] for field in fields},
        files={"file": ("plan.png", image_bytes, "image/png")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "crop requires x, y, width and height"
    assert calls == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("value", ["", "1.5", "NaN", "inf", "true", "1e2"])
def test_upload_rejects_noninteger_crop_before_creating_job(upload_client, tmp_path: Path, value):
    client, calls, image_bytes = upload_client
    response = client.post(
        "/jobs", data={**VALID_CROP, "crop_width": value},
        files={"file": ("plan.png", image_bytes, "image/png")},
    )
    assert response.status_code == 400
    assert "crop" in response.json()["detail"]
    assert calls == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("crop", [
    (-1, 10, 100, 80), (20, -1, 100, 80), (20, 10, 0, 80), (20, 10, 100, 0),
    (20, 10, -1, 80), (20, 10, 100, -1), (120, 10, 1, 80), (20, 90, 100, 1),
    (20, 10, 101, 80), (20, 10, 100, 81), (1000, 1000, 40, 40),
])
def test_upload_rejects_invalid_crop_without_starting_analysis(upload_client, tmp_path: Path, crop):
    client, calls, image_bytes = upload_client
    response = client.post(
        "/jobs", data=dict(zip(CROP_FIELDS, map(str, crop), strict=True)),
        files={"file": ("plan.png", image_bytes, "image/png")},
    )
    assert response.status_code == 400
    assert "crop" in response.json()["detail"]
    assert calls == []
    assert not list(tmp_path.glob("*/*.png"))


@pytest.mark.parametrize("crop", [None, (0, 0, 120, 90), (20, 10, 100, 80), (119, 89, 1, 1)])
def test_upload_preserves_original_and_valid_boundary_crop(upload_client, crop):
    client, calls, image_bytes = upload_client
    response = client.post(
        "/jobs", data=dict(zip(CROP_FIELDS, map(str, crop), strict=True)) if crop else {},
        files={"file": ("plan.png", image_bytes, "image/png")},
    )
    assert response.status_code == 200
    assert len(calls) == 1
    saved_path, saved_bytes, actual_crop = calls[0]
    assert saved_path.exists()
    assert saved_bytes == image_bytes
    assert actual_crop == crop


@pytest.mark.parametrize("crop", [
    (-1, 0, 40, 40), (0, -1, 40, 40), (0, 0, 0, 40), (0, 0, 40, 0),
    (0, 0, -1, 40), (0, 0, 40, -1), (120, 0, 1, 40), (0, 90, 40, 1),
    (0, 0, 121, 90), (0, 0, 120, 91), (0.5, 0, 40, 40), (True, 0, 40, 40),
    (float("nan"), 0, 40, 40), (float("inf"), 0, 40, 40), (0, 0, 40),
])
def test_direct_roi_rejects_invalid_crop_before_slicing(crop):
    image = np.zeros((90, 120, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="crop"):
        detect_plan_roi(image, padding_ratio=0.02, crop_rect=crop)


@pytest.mark.parametrize("crop, expected_rect", [
    ((0, 0, 120, 90), (0, 0, 120, 90)),
    ((20, 10, 100, 80), (18, 8, 102, 82)),
    ((119, 89, 1, 1), (117, 87, 3, 3)),
])
def test_valid_roi_retains_padding_and_exact_source_pixels(crop, expected_rect):
    image = np.arange(90 * 120 * 3, dtype=np.uint8).reshape((90, 120, 3))
    result = detect_plan_roi(image, padding_ratio=0.02, crop_rect=crop)
    rect = result.roi.rect
    assert (rect.x, rect.y, rect.width, rect.height) == expected_rect
    x, y, width, height = expected_rect
    assert result.transform == (x, y)
    assert result.roi.source == "manual"
    assert result.roi.padding_px == 2
    np.testing.assert_array_equal(result.image, image[y:y + height, x:x + width])


@pytest.fixture(params=["JPEG", "PNG", "WEBP"])
def oriented_image(request, tmp_path: Path):
    image = Image.new("RGB", (80, 48), "white")
    for color, rect in (
        ((230, 20, 20), (0, 0, 40, 24)), ((20, 200, 20), (40, 0, 80, 24)),
        ((20, 20, 230), (0, 24, 40, 48)), ((230, 200, 20), (40, 24, 80, 48)),
    ):
        image.paste(color, rect)

    def save(orientation):
        exif = Image.Exif()
        exif[274] = orientation
        suffix, mime = {
            "JPEG": ("jpg", "image/jpeg"), "PNG": ("png", "image/png"),
            "WEBP": ("webp", "image/webp"),
        }[request.param]
        path = tmp_path / f"orientation-{orientation}.{suffix}"
        image.save(path, format=request.param, exif=exif, lossless=True)
        return path, mime

    return save


@pytest.mark.parametrize("orientation", range(1, 9))
def test_validation_dimensions_match_display_orientation(oriented_image, orientation):
    path, mime = oriented_image(orientation)
    original = path.read_bytes()
    validated = validate_image_file(path, LimitSettings(), mime)
    with Image.open(path) as image:
        displayed = ImageOps.exif_transpose(image)
        assert (validated.width, validated.height) == displayed.size
    assert path.read_bytes() == original


@pytest.mark.parametrize("orientation", range(1, 9))
def test_analysis_pixels_match_display_orientation(oriented_image, orientation):
    path, _mime = oriented_image(orientation)
    with Image.open(path) as image:
        expected = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))[:, :, ::-1]
    actual = load_image(path)
    # Allow decoder rounding while requiring the same pixel axes and landmarks.
    np.testing.assert_allclose(actual, expected, atol=3)


@pytest.mark.parametrize("orientation", [5, 6, 7, 8])
def test_upload_uses_oriented_bounds_and_preserves_original(upload_client, oriented_image, orientation):
    client, calls, _image_bytes = upload_client
    path, mime = oriented_image(orientation)
    image_bytes = path.read_bytes()
    accepted = client.post(
        "/jobs", data=dict(zip(CROP_FIELDS, ("0", "40", "48", "40"), strict=True)),
        files={"file": (path.name, image_bytes, mime)},
    )
    assert accepted.status_code == 200
    assert len(calls) == 1
    saved_path, saved_bytes, actual_crop = calls[0]
    assert saved_bytes == image_bytes
    assert actual_crop == (0, 40, 48, 40)
    result = detect_plan_roi(load_image(saved_path), crop_rect=actual_crop, padding_ratio=0)
    assert result.image.shape[:2] == (40, 48)

    rejected = client.post(
        "/jobs", data=dict(zip(CROP_FIELDS, ("40", "0", "40", "48"), strict=True)),
        files={"file": (path.name, image_bytes, mime)},
    )
    assert rejected.status_code == 400
    assert rejected.json()["detail"] == "crop must lie entirely within the image"
    assert len(calls) == 1


def test_image_dimension_limits_use_display_orientation(oriented_image):
    path, mime = oriented_image(6)
    validated = validate_image_file(
        path, LimitSettings(max_image_width=48, max_image_height=80), mime
    )
    assert (validated.width, validated.height) == (48, 80)
    for limits in (
        LimitSettings(max_image_width=47, max_image_height=80),
        LimitSettings(max_image_width=48, max_image_height=79),
    ):
        with pytest.raises(ValueError, match="image dimensions exceed configured limits"):
            validate_image_file(path, limits, mime)
