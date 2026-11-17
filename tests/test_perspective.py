from __future__ import annotations

from copy import deepcopy
import json
import math

import cv2
import numpy as np
import pytest

import architecture_walkthrough.vision.perspective_rectification as perspective_module
from architecture_walkthrough.vision.perspective_rectification import (
    PerspectiveLimits,
    parse_perspective_spec,
    perspective_frame,
    validate_prepared_scale_reference,
    warp_perspective,
)


def rectangle_spec(width=120, height=90, output_width=None):
    return {"corners": [[0, 0], [width, 0], [width, height], [0, height]],
            "aspect_ratio": width / height,
            "output_width_px": width if output_width is None else output_width}


def project(matrix, points):
    values = np.c_[points, np.ones(len(points))] @ matrix.T
    return values[:, :2] / values[:, 2:]


def known_oriented_inverse(width, height):
    # Independent map from a unit rectangle to a convex photographed patch.
    # x=(.60u+.06v+.12)/(.1u+.2v+1),
    # y=(.04u+.60v+.12)/(.1u+.2v+1), then image dimensions.
    return np.diag([width, height, 1.0]) @ np.asarray([
        [.60, .06, .12], [.04, .60, .12], [.1, .2, 1.0],
    ])


def known_spec(width, height):
    inverse = known_oriented_inverse(width, height)
    corners = project(inverse, [[0, 0], [1, 0], [1, 1], [0, 1]])
    return {"corners": corners.tolist(), "aspect_ratio": 1.5, "output_width_px": 120}


def independent_orientation(width, height, turns, degrees):
    quarter = [
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[0, -1, height], [1, 0, 0], [0, 0, 1]],
        [[-1, 0, width], [0, -1, height], [0, 0, 1]],
        [[0, 1, 0], [-1, 0, width], [0, 0, 1]],
    ][turns]
    qw, qh = (height, width) if turns % 2 else (width, height)
    theta = math.radians(degrees)
    cosine, sine = math.cos(theta), math.sin(theta)
    ow = math.ceil(abs(qw * cosine) + abs(qh * sine) - 1e-9)
    oh = math.ceil(abs(qw * sine) + abs(qh * cosine) - 1e-9)
    fine = np.asarray([
        [cosine, -sine, ow / 2 - cosine * qw / 2 + sine * qh / 2],
        [sine, cosine, oh / 2 - sine * qw / 2 - cosine * qh / 2], [0, 0, 1],
    ])
    return ow, oh, fine @ quarter


@pytest.mark.parametrize("turns", range(4))
def test_full_rectangle_quarter_turn_matches_lossless_pixels_and_preserves_original(turns):
    image = np.arange(61 * 83 * 3, dtype=np.uint8).reshape(61, 83, 3)
    original = image.copy()
    width, height = (61, 83) if turns % 2 else (83, 61)
    frame = perspective_frame(83, 61, rectangle_spec(width, height), rotation_quarter_turns=turns)
    actual = warp_perspective(image, frame)
    np.testing.assert_array_equal(actual, np.rot90(original, -turns))
    np.testing.assert_array_equal(image, original)
    assert not np.shares_memory(actual, image)
    assert actual.shape == (height, width, 3)
    np.testing.assert_allclose(frame.prepared_to_source @ frame.source_to_prepared,
                               np.eye(3), atol=1e-12)


def test_known_projective_landmarks_including_pixel_edges_and_interiors():
    frame = perspective_frame(200, 160, known_spec(200, 160))
    target_points = np.asarray([[0, 0], [120, 0], [120, 80], [0, 80],
                                [.5, .5], [17.25, 61.75], [110.5, 31.5]])
    expected_inverse = known_oriented_inverse(200, 160) @ np.diag([1 / 120, 1 / 80, 1])
    expected_source = project(expected_inverse, target_points)
    np.testing.assert_allclose(project(frame.prepared_to_source, target_points),
                               expected_source, atol=1e-11)
    np.testing.assert_allclose(project(frame.source_to_prepared, expected_source),
                               target_points, atol=1e-11)
    # A real projective transform must not quietly fall back to an affine one.
    assert np.linalg.norm(frame.source_to_prepared[2, :2]) > 1e-4


def test_valid_retained_patch_can_have_a_projective_pole_at_the_external_source_origin():
    # The inverse map's upper-left 2x2 determinant is zero, so the forward
    # homography has h33=0. The pole lies outside this valid convex patch.
    independent_inverse = np.diag([1000, 1000, 1.0]) @ np.asarray([
        [.4, .2, .7], [.2, .1, .3], [.1, .2, 1],
    ])
    corners = project(independent_inverse, [[0, 0], [1, 0], [1, 1], [0, 1]])
    frame = perspective_frame(1000, 1000, {"corners": corners.tolist(),
                                          "aspect_ratio": 1.5, "output_width_px": 120})
    targets = np.asarray([[0, 0], [120, 0], [120, 80], [0, 80], [60, 40]])
    expected = project(independent_inverse @ np.diag([1 / 120, 1 / 80, 1]), targets)
    np.testing.assert_allclose(project(frame.prepared_to_source, targets), expected, atol=1e-10)
    assert abs(frame.source_to_prepared[2, 2]) < 1e-12


@pytest.mark.parametrize("turns", range(4))
@pytest.mark.parametrize("degrees", [-7.5, 0, 7.5])
def test_orientation_composition_and_half_pixel_ramps_are_independently_correct(turns, degrees):
    width, height = 203, 159
    ow, oh, orientation = independent_orientation(width, height, turns, degrees)
    frame = perspective_frame(width, height, known_spec(ow, oh),
                              rotation_quarter_turns=turns, straighten_degrees=degrees)
    expected_inverse = (np.linalg.inv(orientation) @ known_oriented_inverse(ow, oh)
                        @ np.diag([1 / 120, 1 / 80, 1]))
    assert (frame.oriented_width_px, frame.oriented_height_px) == (ow, oh)
    yy, xx = np.mgrid[:height, :width]
    image = np.stack([xx + .5, yy + .5, np.full_like(xx, 100)], axis=-1).astype(np.float32)
    before = image.copy()
    actual = warp_perspective(image, frame)
    dy, dx = np.mgrid[:frame.height, :frame.width]
    centres = np.stack([dx.ravel() + .5, dy.ravel() + .5], axis=1)
    expected = project(expected_inverse, centres)
    recorded = project(frame.prepared_to_source, centres)
    np.testing.assert_allclose(recorded, expected, atol=1e-10)
    inside = ((expected[:, 0] >= 1.5) & (expected[:, 0] <= width - 1.5)
              & (expected[:, 1] >= 1.5) & (expected[:, 1] <= height - 1.5))
    assert inside.sum() > 1000
    # OpenCV quantizes linear interpolation weights to 1/32 pixel.
    np.testing.assert_allclose(actual.reshape(-1, 3)[inside, :2], expected[inside], atol=.017)
    np.testing.assert_array_equal(image, before)


def test_warp_samples_original_once_and_uses_white_outside_original(monkeypatch):
    image = np.zeros((61, 83, 3), dtype=np.uint8)
    ow, oh, _orientation = independent_orientation(83, 61, 1, 15)
    frame = perspective_frame(83, 61, rectangle_spec(ow, oh),
                              rotation_quarter_turns=1, straighten_degrees=15)
    calls = []
    original_warp = cv2.warpPerspective

    def record(source, *args, **kwargs):
        assert source is image
        calls.append(args)
        return original_warp(source, *args, **kwargs)

    def no_intermediate_resampling(*args, **kwargs):
        pytest.fail("the composed perspective path must sample the original once")

    monkeypatch.setattr(perspective_module.cv2, "warpPerspective", record)
    monkeypatch.setattr(perspective_module.cv2, "warpAffine", no_intermediate_resampling)
    monkeypatch.setattr(perspective_module.cv2, "rotate", no_intermediate_resampling)
    actual = warp_perspective(image, frame)
    assert len(calls) == 1
    np.testing.assert_array_equal(actual[0, 0], [255, 255, 255])
    np.testing.assert_array_equal(image, 0)


def test_metadata_is_detached_json_and_matrices_are_read_only():
    raw = known_spec(200, 160)
    before = deepcopy(raw)
    frame = perspective_frame(200, 160, raw)
    assert raw == before
    metadata = json.loads(json.dumps(frame.rectification_metadata(), allow_nan=False))
    assert metadata["corner_coordinate_frame"] == "oriented_image_edges"
    assert metadata["aspect_ratio_source"] == "user_known_rectangle"
    assert metadata["aspect_ratio"] == 1.5
    assert (metadata["output_width_px"], metadata["output_height_px"]) == (120, 80)
    assert "version" not in metadata
    metadata["corners"][0][0] = -100
    metadata["oriented_to_prepared"][0][0] = -100
    assert frame.spec.corners[0][0] == before["corners"][0][0]
    assert frame.oriented_to_prepared[0, 0] != -100
    for matrix in (frame.source_to_prepared, frame.prepared_to_source, frame.oriented_to_prepared):
        assert matrix.dtype == np.float64
        with pytest.raises(ValueError, match="read-only"):
            matrix[0, 0] = 0


@pytest.mark.parametrize("scale", [1, 10_000])
def test_conditioning_is_independent_of_source_pixel_resolution(scale):
    width, height = 200 * scale, 160 * scale
    frame = perspective_frame(width, height, known_spec(width, height))
    expected = [[0, 0], [120, 0], [120, 80], [0, 80]]
    np.testing.assert_allclose(project(frame.source_to_prepared, frame.spec.corners),
                               expected, atol=1e-10)


@pytest.mark.parametrize("value", [None, [], "{}", True, {},
                                    {"corners": [], "aspect_ratio": 1},
                                    {**rectangle_spec(), "unexpected": 1},
                                    {**rectangle_spec(), "manual_scale": .01}])
def test_spec_requires_exact_json_object(value):
    with pytest.raises(ValueError, match="perspective"):
        perspective_frame(120, 90, value)


@pytest.mark.parametrize("corners", [
    None, "corners", [], [[0, 0]] * 3,
    [[0, 0], [120, 0], [120, 90], [0, 90], [1, 1]],
    ((0, 0), (120, 0), (120, 90), (0, 90)),
    [[0, 0], [120], [120, 90], [0, 90]],
    [[0, 0], [120, 0, 1], [120, 90], [0, 90]],
    [[0, 0], ["120", 0], [120, 90], [0, 90]],
    [[0, 0], [True, 0], [120, 90], [0, 90]],
    [[0, 0], [float("nan"), 0], [120, 90], [0, 90]],
    [[0, 0], [float("inf"), 0], [120, 90], [0, 90]],
    [[0, 0], [10 ** 1000, 0], [120, 90], [0, 90]],
    [[-0.001, 0], [120, 0], [120, 90], [0, 90]],
    [[0, 0], [120.001, 0], [120, 90], [0, 90]],
    [[0, 0], [120, 0], [120, 90.001], [0, 90]],
    # Duplicated, crossed, reversed, cyclically mislabelled, concave, collinear.
    [[0, 0], [120, 0], [120, 0], [0, 90]],
    [[0, 0], [120, 90], [120, 0], [0, 90]],
    [[0, 0], [0, 90], [120, 90], [120, 0]],
    [[120, 0], [120, 90], [0, 90], [0, 0]],
    [[0, 0], [120, 0], [40, 20], [0, 90]],
    [[0, 0], [60, 0], [120, 0], [0, 90]],
    # Numerically unstable corner, area too small, subpixel edge.
    [[0, 0], [120, 0], [120, 1e-9], [0, 90]],
    [[0, 0], [.01, 0], [.01, .01], [0, .01]],
    [[0, 0], [.04, 0], [.04, .03], [0, .03]],
    [[0, 0], [.5, 0], [120, 90], [0, 90]],
])
def test_rejects_bad_corner_shapes_coordinates_order_and_degeneracy(corners):
    with pytest.raises(ValueError, match="perspective"):
        perspective_frame(120, 90, {**rectangle_spec(), "corners": corners})


@pytest.mark.parametrize("field,value", [
    ("aspect_ratio", True), ("aspect_ratio", "1.5"), ("aspect_ratio", None),
    ("aspect_ratio", 0), ("aspect_ratio", -1), ("aspect_ratio", float("nan")),
    ("aspect_ratio", float("inf")), ("aspect_ratio", 10 ** 1000),
    ("aspect_ratio", 5e-324), ("aspect_ratio", 1e300),
    ("output_width_px", True), ("output_width_px", 120.0), ("output_width_px", "120"),
    ("output_width_px", 0), ("output_width_px", 31), ("output_width_px", 6001),
])
def test_rejects_bad_known_ratio_and_target_width(field, value):
    with pytest.raises(ValueError, match="perspective"):
        parse_perspective_spec({**rectangle_spec(), field: value}, 120, 90)


def test_output_height_rounds_halves_up_and_uses_exact_output_pixel_limits():
    spec = parse_perspective_spec({**rectangle_spec(), "aspect_ratio": 100 / 32.5,
                                  "output_width_px": 100}, 120, 90)
    assert spec.output_height_px == 33
    limits = PerspectiveLimits(max_width_px=100, max_height_px=100, max_pixels=10_000)
    assert perspective_frame(100, 100, rectangle_spec(100, 100), limits=limits).height == 100
    with pytest.raises(ValueError, match="pixel count"):
        perspective_frame(100, 100, rectangle_spec(100, 100),
                          limits=PerspectiveLimits(max_pixels=9999))
    with pytest.raises(ValueError, match="height"):
        perspective_frame(120, 90, {**rectangle_spec(), "aspect_ratio": 120 / 6001})
    with pytest.raises(ValueError, match="height"):
        perspective_frame(120, 90, {**rectangle_spec(), "aspect_ratio": 120 / 31})


@pytest.mark.parametrize("kwargs", [
    {"min_side_px": 0}, {"max_pixels": True}, {"max_width_px": 31},
    {"max_height_px": 6000.0}, {"max_pixels": -1},
])
def test_limits_are_validated(kwargs):
    with pytest.raises(ValueError, match="perspective"):
        PerspectiveLimits(**kwargs)


@pytest.mark.parametrize("width,height", [(0, 90), (120, -1), (True, 90), (120.0, 90)])
def test_source_dimensions_require_positive_integers(width, height):
    with pytest.raises(ValueError, match="dimensions"):
        perspective_frame(width, height, rectangle_spec())


@pytest.mark.parametrize("kwargs", [
    {"rotation_quarter_turns": True}, {"rotation_quarter_turns": 4},
    {"straighten_degrees": 15.1}, {"straighten_degrees": float("nan")},
])
def test_existing_orientation_validation_remains_in_force(kwargs):
    with pytest.raises(ValueError):
        perspective_frame(120, 90, rectangle_spec(), **kwargs)


def test_pole_guard_rejects_crossing_or_near_zero_denominators():
    corners = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]])
    for row in ([2, 0, -1], [1, 0, 1e-10], [0, 0, 0]):
        matrix = np.asarray([[1, 0, 0], [0, 1, 0], row], dtype=np.float64)
        with pytest.raises(ValueError, match="pole|denominator"):
            perspective_module._check_denominators(matrix, corners)
    perspective_module._check_denominators(np.eye(3), corners)
    perspective_module._check_denominators(-np.eye(3), corners)


def test_normalized_solver_rejects_ill_conditioned_corner_system():
    # Exercise the solver's numerical guard separately from the earlier
    # geometric-area guard that rejects such a spec before a frame is built.
    corners = np.asarray([[0, 0], [1, 0], [1, 1e-10], [0, 1e-10]])
    with pytest.raises(ValueError, match="ill-conditioned"):
        perspective_module._unit_homography(corners)


@pytest.mark.parametrize("image", [
    None, np.zeros((90, 120, 1), dtype=np.uint8), np.zeros((90, 120, 5), dtype=np.uint8),
    np.zeros((91, 120, 3), dtype=np.uint8), np.zeros((90, 120), dtype=np.int64),
])
def test_warp_rejects_mismatched_or_unsupported_rasters(image):
    frame = perspective_frame(120, 90, rectangle_spec())
    with pytest.raises(ValueError, match="perspective image"):
        warp_perspective(image, frame)


@pytest.mark.parametrize("channels", [None, 3, 4])
def test_supported_image_channel_layouts(channels):
    shape = (90, 120) if channels is None else (90, 120, channels)
    image = np.full(shape, 42, dtype=np.uint8)
    frame = perspective_frame(120, 90, rectangle_spec())
    actual = warp_perspective(image, frame)
    np.testing.assert_array_equal(actual, image)


def test_prepared_scale_reference_is_fresh_frame_measurement_without_mutation():
    raw = {"points": [[0, 0], [120, 50]], "distance_m": 6.5}
    before = deepcopy(raw)
    reference = validate_prepared_scale_reference(raw, 120, 90)
    assert reference.measured_px == 130
    assert reference.pixels_per_metre == 20
    assert reference.distance_m == 6.5
    assert reference.points == ((0, 0), (120, 50))
    assert raw == before


@pytest.mark.parametrize("raw", [
    None, {}, {"points": [[0, 0], [20, 0]]},
    {"points": [[0, 0], [20, 0]], "distance_m": 1, "manual_scale": .01},
    {"points": [[0, 0], [20, 0]], "distance_m": 1, "unexpected": None},
    {"points": [[0, 0]], "distance_m": 1},
    {"points": [[0, 0], [0, 0]], "distance_m": 1},
    {"points": [[0, 0], [.99, 0]], "distance_m": 1},
    {"points": [[-1, 0], [20, 0]], "distance_m": 1},
    {"points": [[0, 0], [121, 0]], "distance_m": 1},
    {"points": [[0, 0], [20, 91]], "distance_m": 1},
    {"points": [[0, 0], [True, 0]], "distance_m": 1},
    {"points": [[0, 0], [float("nan"), 0]], "distance_m": 1},
    *[{"points": [[0, 0], [20, 0]], "distance_m": value} for value in
      [True, "1", None, 0, -1, float("nan"), float("inf"), 5e-324]],
])
def test_rejects_invalid_prepared_scale_references_and_inherited_scale(raw):
    with pytest.raises(ValueError, match="prepared scale"):
        validate_prepared_scale_reference(raw, 120, 90)


def test_scale_reference_handles_exactly_one_pixel_and_frame_edges():
    reference = validate_prepared_scale_reference(
        {"points": [[119, 90], [120, 90]], "distance_m": .01}, 120, 90,
    )
    assert reference.pixels_per_metre == 100
    with pytest.raises(ValueError, match="dimensions"):
        validate_prepared_scale_reference({"points": [[0, 0], [1, 0]], "distance_m": 1}, 0, 90)
