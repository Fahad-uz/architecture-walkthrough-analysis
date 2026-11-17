"""Manual rectangle rectification primitives, not yet exposed by the pipeline.

Corners use continuous image-edge coordinates in the frame after quarter turns
and fine straightening, before any crop. The aspect ratio must describe a known
physical rectangle; photographed edge lengths cannot supply that information.
Output height rounds to the nearest pixel (halves up), so metadata records the
actual raster dimensions as well as the requested aspect ratio.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np

from architecture_walkthrough.vision.image_preparation import preparation_frame

Point = tuple[float, float]
MIN_NORMALIZED_AREA = 1e-6
MIN_NORMALIZED_CROSS = 1e-8
MAX_NORMALIZED_CONDITION = 1e8
MIN_NORMALIZED_DENOMINATOR = 1e-8


@dataclass(frozen=True)
class PerspectiveLimits:
    min_side_px: int = 32
    max_width_px: int = 6000
    max_height_px: int = 6000
    max_pixels: int = 36_000_000

    def __post_init__(self) -> None:
        values = (self.min_side_px, self.max_width_px, self.max_height_px, self.max_pixels)
        if any(type(value) is not int or value <= 0 for value in values):
            raise ValueError("perspective limits must be positive integers")
        if self.min_side_px > min(self.max_width_px, self.max_height_px):
            raise ValueError("perspective minimum side exceeds the maximum dimensions")


DEFAULT_LIMITS = PerspectiveLimits()


@dataclass(frozen=True)
class PerspectiveSpec:
    corners: tuple[Point, Point, Point, Point]
    aspect_ratio: float
    output_width_px: int
    output_height_px: int


@dataclass(frozen=True)
class PerspectiveFrame:
    source_width_px: int
    source_height_px: int
    oriented_width_px: int
    oriented_height_px: int
    width: int
    height: int
    rotation_quarter_turns: int
    straighten_degrees: float
    spec: PerspectiveSpec
    oriented_to_prepared: np.ndarray
    source_to_prepared: np.ndarray
    prepared_to_source: np.ndarray

    def rectification_metadata(self) -> dict[str, object]:
        """Return JSON-ready facts, without claiming a supported provenance version."""
        return {
            "corner_coordinate_frame": "oriented_image_edges",
            "corners": [list(point) for point in self.spec.corners],
            "aspect_ratio": self.spec.aspect_ratio,
            "aspect_ratio_source": "user_known_rectangle",
            "output_width_px": self.width,
            "output_height_px": self.height,
            "oriented_width_px": self.oriented_width_px,
            "oriented_height_px": self.oriented_height_px,
            "oriented_to_prepared": self.oriented_to_prepared.tolist(),
        }


@dataclass(frozen=True)
class PreparedScaleReference:
    points: tuple[Point, Point]
    distance_m: float
    measured_px: float
    pixels_per_metre: float


def _dimensions(width: int, height: int) -> None:
    if any(type(value) is not int or value <= 0 for value in (width, height)):
        raise ValueError("perspective frame dimensions must be positive integers")


def _number(value: object, label: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(cast(int | float, value))
    except OverflowError as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _point(value: object, width: int, height: int, label: str) -> Point:
    if type(value) is not list or len(value) != 2:
        raise ValueError(f"{label} must be a JSON array of two numbers")
    x, y = (_number(coordinate, label) for coordinate in value)
    if not (0 <= x <= width and 0 <= y <= height):
        raise ValueError(f"{label} must lie within the image edges")
    return x, y


def _validate_corners(points: tuple[Point, ...], width: int, height: int) -> None:
    normalized = np.asarray(points, dtype=np.float64) / [width, height]
    edges = np.roll(normalized, -1, axis=0) - normalized
    following = np.roll(edges, -1, axis=0)
    crosses = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    # Positive winding is clockwise in image coordinates, where Y points down.
    if np.any(crosses <= MIN_NORMALIZED_CROSS):
        raise ValueError("perspective corners must be strictly convex in TL/TR/BR/BL order")
    area = .5 * np.sum(normalized[:, 0] * np.roll(normalized[:, 1], -1)
                       - normalized[:, 1] * np.roll(normalized[:, 0], -1))
    if area < MIN_NORMALIZED_AREA:
        raise ValueError("perspective corner area is too small for stable rectification")
    lengths = np.linalg.norm(np.roll(np.asarray(points), -1, axis=0) - points, axis=1)
    if np.any(lengths < 1):
        raise ValueError("perspective corner edges must be at least one pixel long")
    tl, tr, br, bl = normalized
    if tl[0] + bl[0] >= tr[0] + br[0] or tl[1] + tr[1] >= bl[1] + br[1]:
        raise ValueError("perspective corners must identify top/left in TL/TR/BR/BL order")


def parse_perspective_spec(
    value: object, oriented_width: int, oriented_height: int,
    *, limits: PerspectiveLimits = DEFAULT_LIMITS,
) -> PerspectiveSpec:
    """Validate exactly {corners, aspect_ratio, output_width_px}, decoded from JSON.

    Lists and builtin JSON numbers are required; strings, booleans, extra fields
    and implicit coordinate reordering are rejected. Bounds include image edges.
    This validates a user's known-aspect assertion, not its physical correctness.
    """
    _dimensions(oriented_width, oriented_height)
    if type(value) is not dict or set(value) != {"corners", "aspect_ratio", "output_width_px"}:
        raise ValueError("perspective requires exactly corners, aspect_ratio and output_width_px")
    corners = value["corners"]
    if type(corners) is not list or len(corners) != 4:
        raise ValueError("perspective corners must be four JSON points in TL/TR/BR/BL order")
    points = tuple(_point(point, oriented_width, oriented_height, "perspective corner")
                   for point in corners)
    _validate_corners(points, oriented_width, oriented_height)
    ratio = _number(value["aspect_ratio"], "perspective aspect_ratio")
    if ratio <= 0:
        raise ValueError("perspective aspect_ratio must be positive and known")
    width = value["output_width_px"]
    if type(width) is not int or not limits.min_side_px <= width <= limits.max_width_px:
        raise ValueError("perspective output_width_px exceeds the allowed dimensions")
    height_float = width / ratio
    if not math.isfinite(height_float) or height_float > limits.max_height_px + .5:
        raise ValueError("perspective output height exceeds the allowed dimensions")
    height = math.floor(height_float + .5)
    if not limits.min_side_px <= height <= limits.max_height_px:
        raise ValueError("perspective output height exceeds the allowed dimensions")
    if width * height > limits.max_pixels:
        raise ValueError("perspective output exceeds the allowed pixel count")
    return PerspectiveSpec((points[0], points[1], points[2], points[3]), ratio, width, height)


def _check_denominators(matrix: np.ndarray, points: np.ndarray) -> None:
    row_length = float(np.linalg.norm(matrix[2]))
    if not math.isfinite(row_length) or row_length == 0:
        raise ValueError("perspective transform has an invalid projective denominator")
    values = np.c_[points, np.ones(len(points))] @ (matrix[2] / row_length)
    # A linear denominator with one strict sign at all convex-domain vertices
    # has that sign throughout the domain, so no retained edge crosses a pole.
    if not (np.all(values > MIN_NORMALIZED_DENOMINATOR)
            or np.all(values < -MIN_NORMALIZED_DENOMINATOR)):
        raise ValueError("perspective transform has a pole in or too near the retained frame")


def _unit_homography(corners: np.ndarray) -> np.ndarray:
    target = np.asarray(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.float64)
    rows: list[tuple[float, ...]] = []
    for (x, y), (u, v) in zip(corners, target, strict=True):
        rows.extend(((-x, -y, -1, 0, 0, 0, u * x, u * y, u),
                     (0, 0, 0, -x, -y, -1, v * x, v * y, v)))
    # Work in unit image coordinates to avoid resolution-dependent thresholds.
    _u, singular, vt = np.linalg.svd(np.asarray(rows, dtype=np.float64), full_matrices=True)
    if singular[-1] <= 0 or singular[0] / singular[-1] > MAX_NORMALIZED_CONDITION:
        raise ValueError("perspective corner system is ill-conditioned")
    matrix = vt[-1].reshape(3, 3)
    if not np.isfinite(matrix).all() or np.linalg.cond(matrix) > MAX_NORMALIZED_CONDITION:
        raise ValueError("perspective homography is ill-conditioned")
    inverse = np.linalg.inv(matrix)
    _check_denominators(matrix, corners)
    _check_denominators(inverse, target)
    projected = np.c_[corners, np.ones(4)] @ matrix.T
    if not np.allclose(projected[:, :2] / projected[:, 2:], target, rtol=0, atol=1e-8):
        raise ValueError("perspective homography does not reproduce its corner landmarks")
    return matrix


def perspective_frame(
    source_width: int, source_height: int, perspective: object,
    *, rotation_quarter_turns: int = 0, straighten_degrees: float = 0,
    limits: PerspectiveLimits = DEFAULT_LIMITS,
) -> PerspectiveFrame:
    """Compose source-edge orientation and a known-aspect rectangle homography.

    The returned matrices cover retained image points, not arbitrary points
    beyond the selected quadrilateral. They are float64 and read-only.
    """
    _dimensions(source_width, source_height)
    oriented_width, oriented_height, orientation = preparation_frame(
        source_width, source_height, rotation_quarter_turns, straighten_degrees,
    )
    spec = parse_perspective_spec(perspective, oriented_width, oriented_height, limits=limits)
    normalized_corners = np.asarray(spec.corners, dtype=np.float64) / [
        oriented_width, oriented_height,
    ]
    try:
        unit = _unit_homography(normalized_corners)
        oriented_to_prepared = (
            np.diag([spec.output_width_px, spec.output_height_px, 1.0]) @ unit
            @ np.diag([1 / oriented_width, 1 / oriented_height, 1.0])
        )
        # Homographies are defined up to a scalar. Prefer h33=1 when safe, but
        # do not reject a valid retained patch whose external pole is the origin.
        if abs(unit[2, 2]) > 1e-12:
            oriented_to_prepared /= unit[2, 2]
        source_to_prepared = oriented_to_prepared @ orientation
        prepared_to_source = np.linalg.inv(source_to_prepared)
    except np.linalg.LinAlgError as exc:
        raise ValueError("perspective transform is singular or could not be solved") from exc
    for matrix in (oriented_to_prepared, source_to_prepared, prepared_to_source):
        if not np.isfinite(matrix).all():
            raise ValueError("perspective transform must have finite forward and inverse matrices")
        matrix.setflags(write=False)
    return PerspectiveFrame(
        source_width, source_height, oriented_width, oriented_height,
        spec.output_width_px, spec.output_height_px,
        int(rotation_quarter_turns), float(straighten_degrees), spec,
        oriented_to_prepared, source_to_prepared, prepared_to_source,
    )


def warp_perspective(image: np.ndarray, frame: PerspectiveFrame) -> np.ndarray:
    """Sample the original raster once using a frame returned by perspective_frame.

    Supports grayscale or three/four-channel OpenCV images. The white border
    value follows the existing 0..255 preparation convention, including float
    colour ramps used to check interpolation. The input array is never modified.
    """
    if (not isinstance(image, np.ndarray) or image.ndim not in (2, 3)
            or (image.ndim == 3 and image.shape[2] not in (3, 4))):
        raise ValueError("perspective image must be grayscale or have three/four channels")
    if image.shape[:2] != (frame.source_height_px, frame.source_width_px):
        raise ValueError("perspective image dimensions do not match the source frame")
    if image.dtype not in (np.uint8, np.uint16, np.int16, np.float32, np.float64):
        raise ValueError("perspective image has an unsupported OpenCV sample type")
    to_edges = np.asarray(((1, 0, .5), (0, 1, .5), (0, 0, 1)), dtype=np.float64)
    to_indices = np.asarray(((1, 0, -.5), (0, 1, -.5), (0, 0, 1)), dtype=np.float64)
    index_matrix = to_indices @ frame.source_to_prepared @ to_edges
    return cv2.warpPerspective(
        image, index_matrix, (frame.width, frame.height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255, 255),
    )


def validate_prepared_scale_reference(
    value: object, prepared_width: int, prepared_height: int,
) -> PreparedScaleReference:
    """Validate exactly {points: [[x,y],[x,y]], distance_m} after rectification.

    A reference must span at least one prepared pixel. Callers must invalidate
    this reference whenever preparation changes; this pure numeric validator
    cannot establish that a user's measurement is fresh or physically correct.
    No inherited source-image scale is accepted by this contract.
    """
    _dimensions(prepared_width, prepared_height)
    if type(value) is not dict or set(value) != {"points", "distance_m"}:
        raise ValueError("prepared scale reference requires exactly points and distance_m")
    points = value["points"]
    if type(points) is not list or len(points) != 2:
        raise ValueError("prepared scale reference requires two JSON points")
    first, second = (_point(point, prepared_width, prepared_height, "prepared scale point")
                     for point in points)
    distance_m = _number(value["distance_m"], "prepared scale distance_m")
    if distance_m <= 0:
        raise ValueError("prepared scale distance_m must be positive")
    measured_px = math.hypot(second[0] - first[0], second[1] - first[1])
    if measured_px < 1:
        raise ValueError("prepared scale reference must span at least one pixel")
    ppm = measured_px / distance_m
    if not math.isfinite(ppm) or ppm <= 0:
        raise ValueError("prepared scale reference must produce finite positive pixels per metre")
    return PreparedScaleReference((first, second), distance_m, measured_px, ppm)
